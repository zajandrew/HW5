import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Hook into parent directory to import your research files
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))
import cr_config as cr
import feature_creation as fc
import portfolio_test as pt

# Global Cache for the "Frozen Model"
MODEL_SNAPSHOT = None

def load_midnight_model(yymm_str):
    global MODEL_SNAPSHOT
    try:
        path = Path(cr.PATH_ENH) / f"{yymm_str}_enh.parquet"
        if not path.exists():
            print(f"[WARN] Model file not found: {path}")
            return False

        df = pd.read_parquet(path)
        last_ts = df['ts'].max()
        snapshot = df[df['ts'] == last_ts].copy()
        
        # Index by Ticker for fast lookup
        MODEL_SNAPSHOT = snapshot.set_index('tenor_yrs').to_dict('index')
        print(f"[SYSTEM] Loaded Midnight Model: {last_ts} ({len(MODEL_SNAPSHOT)} tenors)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

def get_live_z_scores(live_rates_map):
    if not MODEL_SNAPSHOT:
        return {}

    z_map = {}
    VOL_PROXY = 0.05 

    for ticker, live_rate in live_rates_map.items():
        tenor = cr.TENOR_YEARS.get(ticker)
        if tenor is None or tenor not in MODEL_SNAPSHOT:
            continue
            
        model_row = MODEL_SNAPSHOT[tenor]
        model_rate = float(model_row.get('rate', live_rate))
        model_z = float(model_row.get('z_comb', 0.0))
        
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
    active_tenor = cr.TENOR_YEARS.get(active_ticker)
    if active_ticker not in live_z_map:
        return []

    active_z = live_z_map[active_ticker]['z_live']
    candidates = []
    
    # Load Constraints
    min_alt_tenor = getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0)
    min_sep = getattr(cr, "MIN_SEP_YEARS", 0.5)
    max_span = getattr(cr, "MAX_SPAN_YEARS", 10.0)
    
    for ticker, info in live_z_map.items():
        tenor = info['tenor']
        z_score = info['z_live']
        
        if tenor == active_tenor: continue
        if tenor < min_alt_tenor: continue
        
        dist = abs(tenor - active_tenor)
        if dist < min_sep: continue
        if dist > max_span: continue
        
        # Spread Improvement Logic
        if direction == 'PAY':
            # Desk Pays Original. We Pay Alt / Rec Orig.
            # We want Alt to be Cheap (High Z).
            spread_imp = z_score - active_z
        else: 
            # Desk Rec Original. We Rec Alt / Pay Orig.
            # We want Alt to be Cheap (High Z).
            spread_imp = z_score - active_z

        candidates.append({
            'ticker': ticker,
            'tenor': tenor,
            'z_live': z_score,
            'spread_imp': spread_imp
        })
        
    candidates.sort(key=lambda x: x['spread_imp'], reverse=True)
    return candidates

def calculate_live_pnl(row, live_map, now_dt):
    """
    Calculates Price, Carry, and Rolldown PnL using the shortest tenor as Carry Proxy.
    Returns: (Total, Price, Carry, Roll)
    """
    # 1. Setup
    entry_dt = pd.to_datetime(row['open_ts'])
    dt_days = max(0.0, (now_dt - entry_dt).days)
    
    # Rates
    entry_pay = row['entry_rate_pay']
    entry_rec = row['entry_rate_rec']
    curr_pay = live_map.get(row['ticker_pay'], entry_pay)
    curr_rec = live_map.get(row['ticker_rec'], entry_rec)
    
    # 2. Funding Rate Proxy (Shortest Tenor in live_map)
    funding_rate = 4.0 # Safe fallback
    try:
        if live_map:
            # Sort live map by tenor logic to find shortest
            valid_keys = [k for k in live_map.keys() if k in cr.TENOR_YEARS]
            if valid_keys:
                shortest_ticker = min(valid_keys, key=lambda k: cr.TENOR_YEARS[k])
                funding_rate = live_map[shortest_ticker]
    except Exception:
        pass

    # 3. PnL Components
    # We assume 'dv01' is the trade size scalar.
    # Pay Leg Sign = -1. Rec Leg Sign = +1.
    dv01 = row['dv01']
    
    # A. Price PnL: (Curr - Entry) * DV01 (signed)
    # Pay Leg (-): (Curr - Entry) * -1 = (Entry - Curr). Wait.
    # Short Bond: If Rates Rise, Price Falls, Short Wins.
    # (Curr - Entry) is positive. PnL should be positive.
    # So Pay PnL = (Curr - Entry) * DV01.
    pnl_price_pay = (curr_pay - entry_pay) * 100.0 * dv01
    pnl_price_rec = (entry_rec - curr_rec) * 100.0 * dv01
    
    # B. Carry PnL: (Fixed - Float) * DV01 (signed) * Time
    # Pay Leg (-): (Fixed - Float) * -1 = (Float - Fixed). Negative Carry if Fixed > Float. Correct.
    carry_pay = (entry_pay - funding_rate) * 100.0 * (-dv01) * (dt_days / 360.0)
    carry_rec = (entry_rec - funding_rate) * 100.0 * (+dv01) * (dt_days / 360.0)
    
    # C. Rolldown PnL: (Curr - Rolled) * DV01 (signed)
    # Build Curve for Interp
    curve_points = []
    for k, v in cr.TENOR_YEARS.items():
        if k in live_map: curve_points.append((v, live_map[k]))
    curve_points.sort(key=lambda x: x[0])
    
    roll_pay, roll_rec = 0.0, 0.0
    if curve_points:
        xp = [x[0] for x in curve_points]
        fp = [x[1] for x in curve_points]
        
        t_pay, t_rec = row['tenor_pay'], row['tenor_rec']
        
        # Pay Leg
        t_roll_pay = max(0.0, t_pay - (dt_days/360.0))
        y_roll_pay = np.interp(t_roll_pay, xp, fp)
        # Gain if yield drops (roll down). Short loses if yield drops.
        roll_pay = (curr_pay - y_roll_pay) * 100.0 * (-dv01)
        
        # Rec Leg
        t_roll_rec = max(0.0, t_rec - (dt_days/360.0))
        y_roll_rec = np.interp(t_roll_rec, xp, fp)
        roll_rec = (curr_rec - y_roll_rec) * 100.0 * (+dv01)

    total_price = pnl_price_pay + pnl_price_rec
    total_carry = carry_pay + carry_rec
    total_roll  = roll_pay + roll_rec
    total_pnl   = total_price + total_carry + total_roll
    
    return total_pnl, total_price, total_carry, total_roll
