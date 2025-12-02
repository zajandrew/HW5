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
# Note: portfolio_test is imported only if needed for shared logic, 
# but here we reimplement the vector logic for scalar live ticks.

# ==============================================================================
# GLOBAL STATE
# ==============================================================================
# The "Frozen Model" Snapshot
MODEL_SNAPSHOT = {} 
# The Funding Anchor (Shortest tenor rate from the model) to prevent 0.0 or 4.0 fallbacks
GLOBAL_FUNDING_ANCHOR = 0.0 
# Default Volatility Proxy (Daily Std Dev of Rates in %) if not found in file
DEFAULT_VOL_PROXY = 0.05 

def load_midnight_model(yymm_str):
    """
    Loads the EOD parquet file. 
    This is the 'Ruler' we measure live prices against.
    
    Fixes:
    1. Extracts 'scale' (vol) if available for dynamic Z-scores.
    2. Identifies the shortest tenor rate as the Funding Anchor.
    """
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

        # Get the very last timestamp available in the file (EOD Yesterday)
        last_ts = df['ts'].max()
        snapshot = df[df['ts'] == last_ts].copy()
        
        # --- Fix 1: Establish Funding Anchor from Model (Safety Net) ---
        # Find the row with the smallest tenor_yrs
        if not snapshot.empty:
            min_tenor_idx = snapshot['tenor_yrs'].idxmin()
            GLOBAL_FUNDING_ANCHOR = float(snapshot.loc[min_tenor_idx, 'rate'])
        
        # --- Fix 2: Index by Ticker for fast O(1) lookup ---
        # We also check if a 'scale' or 'vol' column exists for Z-score accuracy
        has_scale = 'scale' in snapshot.columns
        has_vol = 'vol' in snapshot.columns
        
        # Convert to dict format: {tenor_yrs: {rate, z_comb, scale...}}
        # Note: We index by tenor_yrs because cr.TENOR_YEARS maps Ticker -> Tenor
        MODEL_SNAPSHOT = snapshot.set_index('tenor_yrs').to_dict('index')
        
        # Post-processing to ensure keys are floats and defaults exist
        for t, row in MODEL_SNAPSHOT.items():
            # Determine Vol Proxy for this tenor
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
    """
    Projects Live Rates onto the Frozen Model.
    
    Fix: Uses per-tenor 'vol_proxy' if available, else falls back to default.
    """
    if not MODEL_SNAPSHOT:
        return {}

    z_map = {}
    
    for ticker, live_rate in live_rates_map.items():
        tenor = cr.TENOR_YEARS.get(ticker)
        
        # Check if Tenor exists in Model
        if tenor is None or tenor not in MODEL_SNAPSHOT:
            continue
            
        model_row = MODEL_SNAPSHOT[tenor]
        model_rate = float(model_row.get('rate', live_rate)) # EOD Rate
        model_z = float(model_row.get('z_comb', 0.0))        # EOD Z-Score
        vol_proxy = float(model_row.get('vol_proxy', DEFAULT_VOL_PROXY)) # Vol Fix

        # Prevent div by zero
        if vol_proxy == 0: vol_proxy = DEFAULT_VOL_PROXY

        # The Projection
        # Yield Up = Price Down = Cheap = High Z-Score (Standard Convention)
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
    
    Fixes:
    1. Robust Funding: Uses Live Shortest if avail, else Model Shortest.
    2. Composite Curve: Builds curve using Model + Live Overwrite for robust interpolation.
    """
    # 1. Setup Data
    entry_dt = pd.to_datetime(row['open_ts'])
    dt_days = max(0.0, (now_dt - entry_dt).days)
    
    # Rates
    entry_pay = row['entry_rate_pay']
    entry_rec = row['entry_rate_rec']
    curr_pay = live_map.get(row['ticker_pay'], entry_pay)
    curr_rec = live_map.get(row['ticker_rec'], entry_rec)
    
    # 2. Funding Rate Proxy
    # Priority: 
    #   A) Shortest Tenor currently ticking in Live Map
    #   B) Global Funding Anchor (Shortest Tenor from Midnight Model)
    #   C) Hard fallback (4.0)
    funding_rate = GLOBAL_FUNDING_ANCHOR if GLOBAL_FUNDING_ANCHOR > 0 else 4.0
    
    if live_map:
        # Try to find a live ticker shorter than 0.25y (3M)
        valid_keys = [k for k in live_map.keys() if k in cr.TENOR_YEARS]
        if valid_keys:
            # Find the ticker with minimum tenor
            shortest_ticker = min(valid_keys, key=lambda k: cr.TENOR_YEARS[k])
            shortest_tenor = cr.TENOR_YEARS[shortest_ticker]
            
            # Only use live if it is actually a short rate (e.g. < 1Y)
            # Otherwise we risk using the 2Y as funding if 1M/3M aren't ticking.
            if shortest_tenor <= 1.0:
                funding_rate = live_map[shortest_ticker]

    # 3. PnL Components
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
    
    # Build Composite Curve for Interpolation (Fixes Interpolation Drops)
    # Start with Frozen Model Points
    curve_data = {}
    if MODEL_SNAPSHOT:
        for t, r in MODEL_SNAPSHOT.items():
            curve_data[t] = r.get('rate', 0.0)
    
    # Overwrite with Live Points
    for k, v in live_map.items():
        t = cr.TENOR_YEARS.get(k)
        if t is not None:
            curve_data[t] = v
            
    # Sort for interpolation
    sorted_tenors = sorted(curve_data.keys())
    
    if sorted_tenors:
        xp = sorted_tenors
        fp = [curve_data[t] for t in sorted_tenors]
        
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
