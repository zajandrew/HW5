import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# --- Hook into parent directory to import research files ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

import cr_config as cr
import feature_creation as fc
import portfolio_test as pt

# Global Cache for the "Frozen Model"
MODEL_SNAPSHOT = None

def load_midnight_model(yymm_str):
    """
    Loads the EOD parquet file. 
    This is the 'Ruler' we measure live prices against.
    """
    global MODEL_SNAPSHOT
    try:
        path = Path(cr.PATH_ENH) / f"{yymm_str}_enh.parquet"
        
        if not path.exists():
            print(f"[WARN] Model file not found: {path}")
            return False

        df = pd.read_parquet(path)
        
        # Get the very last timestamp available in the file (EOD Yesterday)
        last_ts = df['ts'].max()
        snapshot = df[df['ts'] == last_ts].copy()
        
        # Index by Ticker for fast O(1) lookup
        MODEL_SNAPSHOT = snapshot.set_index('tenor_yrs').to_dict('index')
        print(f"[SYSTEM] Loaded Midnight Model: {last_ts} ({len(MODEL_SNAPSHOT)} tenors)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

def get_live_z_scores(live_rates_map):
    """
    Projects Live Rates onto the Frozen Model.
    """
    if not MODEL_SNAPSHOT:
        return {}

    z_map = {}
    
    # VOL PROXY ADJUSTMENT
    # Inputs are percentages (e.g., 4.25).
    # 1 bp = 0.01 change in rate.
    # We estimate 1 Sigma daily vol is roughly 5 bps.
    VOL_PROXY = 0.05 

    for ticker, live_rate in live_rates_map.items():
        tenor = cr.TENOR_YEARS.get(ticker)
        
        # Check if Tenor exists in Model
        if tenor is None or tenor not in MODEL_SNAPSHOT:
            continue
            
        model_row = MODEL_SNAPSHOT[tenor]
        model_rate = float(model_row.get('rate', live_rate)) # EOD Rate
        model_z = float(model_row.get('z_comb', 0.0))        # EOD Z-Score
        
        # The Projection
        # Yield Up = Price Down = Cheap = High Z-Score (Standard Convention)
        rate_diff = live_rate - model_rate
        z_live = model_z + (rate_diff / VOL_PROXY)
        
        z_map[ticker] = {
            'z_live': z_live, 
            'tenor': tenor, 
            'model_z': model_z,
            'model_rate': model_rate
        }
    return z_map

def get_valid_partners(active_ticker, direction, live_z_map):
    """
    Finds candidates based on Z-Spread improvement.
    Respects Tenor Limits (ALT_LEG_TENOR_YEARS) and Separation Constraints.
    """
    active_tenor = cr.TENOR_YEARS.get(active_ticker)
    
    # Safety check: if active ticker not in map
    if active_ticker not in live_z_map:
        return []

    active_z = live_z_map[active_ticker]['z_live']
    candidates = []
    
    # LOAD CONFIG LIMITS
    min_alt_tenor = getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0)
    min_sep = getattr(cr, "MIN_SEP_YEARS", 0.5)
    max_span = getattr(cr, "MAX_SPAN_YEARS", 10.0)
    
    for ticker, info in live_z_map.items():
        tenor = info['tenor']
        z_score = info['z_live']
        
        # 1. Skip self
        if tenor == active_tenor: 
            continue
            
        # 2. Enforce Minimum Tenor Length
        if tenor < min_alt_tenor:
            continue
            
        # 3. Enforce Separation Constraints
        dist = abs(tenor - active_tenor)
        if dist < min_sep: continue
        if dist > max_span: continue
        
        # Calculate Spread Improvement
        if direction == 'PAY':
            # Desk wants to PAY Original (Short). Hedge is REC Original.
            # Strategy: PAY Alt (Cheap) / REC Original (Rich).
            # We want Alt Z > Original Z.
            spread_imp = z_score - active_z
            
        else: # direction == 'REC'
            # Desk wants to REC Original (Long). Hedge is PAY Original.
            # Strategy: REC Alt (Cheap) / PAY Original (Rich).
            # We want Alt Z > Original Z.
            spread_imp = z_score - active_z

        candidates.append({
            'ticker': ticker,
            'tenor': tenor,
            'z_live': z_score,
            'spread_imp': spread_imp
        })
        
    # Sort by spread improvement (descending)
    candidates.sort(key=lambda x: x['spread_imp'], reverse=True)
    return candidates

def calculate_live_pnl(row, live_map, now_dt):
    """
    Calculates Price, Carry, and Rolldown PnL for a single trade row.
    Returns nested tuple: ( (Tot, Prc, Cry, Rol)_CASH,  (Tot, Prc, Cry, Rol)_BP )
    """
    # 1. Setup Data
    entry_dt = pd.to_datetime(row['open_ts'])
    dt_days = max(0.0, (now_dt - entry_dt).days)
    
    # Rates
    entry_pay = row['entry_rate_pay']
    entry_rec = row['entry_rate_rec']
    curr_pay = live_map.get(row['ticker_pay'], entry_pay)
    curr_rec = live_map.get(row['ticker_rec'], entry_rec)
    
    # 2. Funding Rate Proxy (Shortest Tenor in live_map)
    # This finds the shortest tenor currently alive in the feed (e.g., 1M or 3M)
    # to serve as the floating rate proxy for Carry calculation.
    funding_rate = 4.0 # Fallback
    try:
        if live_map:
            # Sort keys by tenor length
            valid_keys = [k for k in live_map.keys() if k in cr.TENOR_YEARS]
            if valid_keys:
                shortest_ticker = min(valid_keys, key=lambda k: cr.TENOR_YEARS[k])
                funding_rate = live_map[shortest_ticker]
    except Exception:
        pass

    # 3. PnL Components
    # 'dv01' from DB is the trade size scalar (Magnitude).
    # Logic:
    # PAY Leg (Short): Profit if Rate Rises. Sign = -1 relative to rate drop.
    # REC Leg (Long): Profit if Rate Falls. Sign = +1 relative to rate drop.
    
    dv01 = row['dv01'] 
    
    # --- A. PRICE PNL ---
    # Pay Leg: (Curr - Entry) * DV01. (If rates rise, we win).
    prc_pay_bp = (curr_pay - entry_pay) * 100.0
    # Rec Leg: (Entry - Curr) * DV01. (If rates fall, we win).
    prc_rec_bp = (entry_rec - curr_rec) * 100.0
    
    # --- B. CARRY PNL ---
    # Carry = (Fixed - Float) * Time * Direction
    # Pay Leg (Short): Pays Fixed, Receives Float. Net = Float - Fixed.
    # Math: (Fixed - Float) * -1 = Float - Fixed.
    cry_pay_bp = (entry_pay - funding_rate) * 100.0 * (-1) * (dt_days / 360.0)
    # Rec Leg (Long): Receives Fixed, Pays Float. Net = Fixed - Float.
    # Math: (Fixed - Float) * +1.
    cry_rec_bp = (entry_rec - funding_rate) * 100.0 * (+1) * (dt_days / 360.0)
    
    # --- C. ROLLDOWN PNL ---
    rol_pay_bp, rol_rec_bp = 0.0, 0.0
    
    # Build current curve for interpolation
    curve_points = []
    for k, v in cr.TENOR_YEARS.items():
        if k in live_map: curve_points.append((v, live_map[k]))
    curve_points.sort(key=lambda x: x[0])
    
    if curve_points:
        xp = [x[0] for x in curve_points]
        fp = [x[1] for x in curve_points]
        t_pay, t_rec = row['tenor_pay'], row['tenor_rec']
        
        # Pay Leg (Short)
        # Roll Down: As tenor shortens, yield falls (on normal curve). Price rises.
        # Short position LOSES money on roll down.
        t_roll_pay = max(0.0, t_pay - (dt_days/360.0))
        y_roll_pay = np.interp(t_roll_pay, xp, fp)
        # (Curr - Rolled) is positive. Multiplied by -1 (Short). Result Negative. Correct.
        rol_pay_bp = (curr_pay - y_roll_pay) * 100.0 * (-1)
        
        # Rec Leg (Long)
        # Long position GAINS money on roll down.
        t_roll_rec = max(0.0, t_rec - (dt_days/360.0))
        y_roll_rec = np.interp(t_roll_rec, xp, fp)
        # (Curr - Rolled) is positive. Multiplied by +1 (Long). Result Positive. Correct.
        rol_rec_bp = (curr_rec - y_roll_rec) * 100.0 * (+1)

    # --- SUMS (Basis Points) ---
    total_price_bp = prc_pay_bp + prc_rec_bp
    total_carry_bp = cry_pay_bp + cry_rec_bp
    total_roll_bp  = rol_pay_bp + rol_rec_bp
    total_bp       = total_price_bp + total_carry_bp + total_roll_bp
    
    # --- SUMS (Cash) ---
    # Cash = BP * DV01 Scalar
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
