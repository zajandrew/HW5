import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# --- Hook into parent directory ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

import cr_config as cr

# ==============================================================================
# GLOBAL STATE
# ==============================================================================
MODEL_SNAPSHOT = {} 
GLOBAL_FUNDING_ANCHOR = 0.0 
DEFAULT_VOL_PROXY = 0.05 

def load_midnight_model(yymm_str):
    """Loads the EOD parquet file (The Ruler)."""
    global MODEL_SNAPSHOT, GLOBAL_FUNDING_ANCHOR
    
    try:
        path = Path(cr.PATH_ENH) / f"{yymm_str}_enh.parquet"
        
        if not path.exists():
            print(f"[WARN] Model file not found: {path}")
            return False

        df = pd.read_parquet(path)
        if df.empty:
            print(f"[WARN] Model file empty: {path}")
            return False

        last_ts = df['ts'].max()
        snapshot = df[df['ts'] == last_ts].copy()
        
        if not snapshot.empty:
            min_tenor_idx = snapshot['tenor_yrs'].idxmin()
            GLOBAL_FUNDING_ANCHOR = float(snapshot.loc[min_tenor_idx, 'rate'])
        
        has_scale = 'scale' in snapshot.columns
        has_vol = 'vol' in snapshot.columns
        
        MODEL_SNAPSHOT = snapshot.set_index('tenor_yrs').to_dict('index')
        
        for t, row in MODEL_SNAPSHOT.items():
            if has_scale:
                row['vol_proxy'] = float(row['scale'])
            elif has_vol:
                row['vol_proxy'] = float(row['vol'])
            else:
                row['vol_proxy'] = DEFAULT_VOL_PROXY
        
        print(f"[SYSTEM] Loaded Midnight Model: {last_ts} ({len(MODEL_SNAPSHOT)} tenors). Funding Anchor: {GLOBAL_FUNDING_ANCHOR:.4f}%")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

def get_live_z_scores(live_rates_map):
    """Projects Live Rates onto the Frozen Model."""
    if not MODEL_SNAPSHOT:
        return {}

    z_map = {}
    
    for ticker, live_rate in live_rates_map.items():
        tenor = cr.TENOR_YEARS.get(ticker)
        
        if tenor is None or tenor not in MODEL_SNAPSHOT:
            continue
            
        model_row = MODEL_SNAPSHOT[tenor]
        model_rate = float(model_row.get('rate', live_rate))
        model_z = float(model_row.get('z_comb', 0.0))
        vol_proxy = float(model_row.get('vol_proxy', DEFAULT_VOL_PROXY))

        if vol_proxy == 0: vol_proxy = DEFAULT_VOL_PROXY

        rate_diff = live_rate - model_rate
        z_live = model_z + (rate_diff / vol_proxy)
        
        z_map[ticker] = {
            'z_live': z_live, 
            'tenor': tenor, 
            'model_z': model_z,
            'model_rate': model_rate,
            'vol_proxy': vol_proxy
        }
    return z_map

def calc_live_drift(tenor, direction, live_map):
    """
    Calculates Daily Drift (Carry + Roll) in BPS.
    Matches portfolio_test.py logic.
    direction: -1.0 (Pay/Short), +1.0 (Rec/Long)
    """
    # 1. Get Candidate Fixed Rate
    rate_fixed = 0.0
    
    # Reverse lookup ticker for live map check
    ticker = None
    for k, v in cr.TENOR_YEARS.items():
        if abs(v - tenor) < 0.001:
            ticker = k
            break
            
    if ticker and ticker in live_map:
        rate_fixed = live_map[ticker]
    elif tenor in MODEL_SNAPSHOT:
        rate_fixed = float(MODEL_SNAPSHOT[tenor].get('rate', 0.0))
    else:
        return -999.0

    # 2. Get Funding Rate (Shortest Live < 1Y or Anchor)
    funding_rate = GLOBAL_FUNDING_ANCHOR if GLOBAL_FUNDING_ANCHOR > 0 else 4.0
    if live_map:
        valid_keys = [k for k in live_map.keys() if k in cr.TENOR_YEARS]
        if valid_keys:
            shortest_ticker = min(valid_keys, key=lambda k: cr.TENOR_YEARS[k])
            if cr.TENOR_YEARS[shortest_ticker] <= 1.0:
                funding_rate = live_map[shortest_ticker]

    # 3. Calculate Rolldown
    # Build sorted composite curve for interpolation
    curve_data = {t: r.get('rate', 0.0) for t, r in MODEL_SNAPSHOT.items()}
    for k, v in live_map.items():
        t_live = cr.TENOR_YEARS.get(k)
        if t_live: curve_data[t_live] = v
    
    xp = sorted(curve_data.keys())
    fp = [curve_data[t] for t in xp]

    if not xp: return 0.0

    # Roll 1 day (Act/360)
    day_fraction = 1.0 / 360.0
    t_roll = max(0.0, tenor - day_fraction)
    rate_rolled = np.interp(t_roll, xp, fp)

    # 4. Total Drift (BPS)
    # Carry = (Fixed - Float) * 100 * (1/360)
    raw_carry = (rate_fixed - funding_rate) * 100.0 * day_fraction
    
    # Roll = (Curr - Rolled) * 100
    raw_roll = (rate_fixed - rate_rolled) * 100.0
    
    total_drift = (raw_carry + raw_roll) * direction
    return total_drift

def get_valid_partners(active_ticker, direction, live_z_map, live_map):
    """
    Finds candidates based on COMPOSITE SCORE (Z + Drift).
    """
    active_tenor = cr.TENOR_YEARS.get(active_ticker)
    if active_ticker not in live_z_map:
        return []

    active_z = live_z_map[active_ticker]['z_live']
    candidates = []
    
    # Configs
    min_alt_tenor = getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0)
    min_sep = getattr(cr, "MIN_SEP_YEARS", 0.5)
    max_span = getattr(cr, "MAX_SPAN_YEARS", 10.0)
    
    DRIFT_GATE = float(getattr(cr, "DRIFT_GATE_BPS", -0.5))
    DRIFT_WEIGHT = float(getattr(cr, "DRIFT_WEIGHT", 0.2))
    
    # Direction Scalar:
    # If input is 'PAY', user pays Candidate. Candidate is Short (-1.0).
    # If input is 'REC', user recs Candidate. Candidate is Long (+1.0).
    dir_scalar = -1.0 if direction == 'PAY' else 1.0

    for ticker, info in live_z_map.items():
        tenor = info['tenor']
        z_live = info['z_live']
        
        # Constraints
        if tenor == active_tenor: continue
        if tenor < min_alt_tenor: continue
        dist = abs(tenor - active_tenor)
        if dist < min_sep: continue
        if dist > max_span: continue
        
        # 1. Z-Spread (Candidate Z - Active Z)
        # We assume higher Z is always "Cheaper" (better to buy/pay)
        z_spread = z_live - active_z
        
        # 2. Drift (BPS)
        drift_bps = calc_live_drift(tenor, dir_scalar, live_map)
        
        # 3. Composite Score
        composite = z_spread + (DRIFT_WEIGHT * drift_bps)
        
        # 4. Gate
        is_gated = (drift_bps < DRIFT_GATE)

        candidates.append({
            'ticker': ticker,
            'tenor': tenor,
            'z_live': z_live,
            'spread_imp': z_spread,
            'drift_bps': drift_bps,
            'composite': composite,
            'is_gated': is_gated
        })
        
    # Sort by COMPOSITE Score
    candidates.sort(key=lambda x: x['composite'], reverse=True)
    return candidates

def calculate_live_pnl(row, live_map, now_dt):
    """
    Calculates Price, Carry, and Rolldown PnL.
    Uses sorted composite curve for robust interpolation.
    """
    entry_dt = pd.to_datetime(row['open_ts'])
    dt_days = max(0.0, (now_dt - entry_dt).days)
    
    # Rates
    entry_pay = row['entry_rate_pay']
    entry_rec = row['entry_rate_rec']
    curr_pay = live_map.get(row['ticker_pay'], entry_pay)
    curr_rec = live_map.get(row['ticker_rec'], entry_rec)
    
    # Funding
    funding_rate = GLOBAL_FUNDING_ANCHOR if GLOBAL_FUNDING_ANCHOR > 0 else 4.0
    if live_map:
        valid_keys = [k for k in live_map.keys() if k in cr.TENOR_YEARS]
        if valid_keys:
            shortest_ticker = min(valid_keys, key=lambda k: cr.TENOR_YEARS[k])
            if cr.TENOR_YEARS[shortest_ticker] <= 1.0:
                funding_rate = live_map[shortest_ticker]

    dv01 = row['dv01'] 
    
    # Price PnL
    prc_pay_bp = (curr_pay - entry_pay) * 100.0
    prc_rec_bp = (entry_rec - curr_rec) * 100.0
    
    # Carry PnL
    cry_pay_bp = (entry_pay - funding_rate) * 100.0 * (-1) * (dt_days / 360.0)
    cry_rec_bp = (entry_rec - funding_rate) * 100.0 * (+1) * (dt_days / 360.0)
    
    # Rolldown PnL (Sorted Interpolation)
    rol_pay_bp, rol_rec_bp = 0.0, 0.0
    
    curve_data = {t: r.get('rate', 0.0) for t, r in MODEL_SNAPSHOT.items()}
    for k, v in live_map.items():
        t = cr.TENOR_YEARS.get(k)
        if t: curve_data[t] = v
            
    xp = sorted(curve_data.keys())
    fp = [curve_data[t] for t in xp]
    
    if xp:
        # Pay Leg (Short)
        t_roll_pay = max(0.0, row['tenor_pay'] - (dt_days/360.0))
        y_roll_pay = np.interp(t_roll_pay, xp, fp)
        rol_pay_bp = (curr_pay - y_roll_pay) * 100.0 * (-1)
        
        # Rec Leg (Long)
        t_roll_rec = max(0.0, row['tenor_rec'] - (dt_days/360.0))
        y_roll_rec = np.interp(t_roll_rec, xp, fp)
        rol_rec_bp = (curr_rec - y_roll_rec) * 100.0 * (+1)

    # Totals
    total_price_bp = prc_pay_bp + prc_rec_bp
    total_carry_bp = cry_pay_bp + cry_rec_bp
    total_roll_bp  = rol_pay_bp + rol_rec_bp
    total_bp       = total_price_bp + total_carry_bp + total_roll_bp
    
    cash_tuple = (
        total_bp * dv01,
        total_price_bp * dv01,
        total_carry_bp * dv01,
        total_roll_bp * dv01
    )
    
    bp_tuple = (
        total_bp,
        total_price_bp,
        total_carry_bp,
        total_roll_bp
    )
    
    return cash_tuple, bp_tuple
