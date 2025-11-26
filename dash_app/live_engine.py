import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Hook into parent directory to import your research files
sys.path.append(str(Path(__file__).parent.parent))
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
        # Construct path based on cr_config or hardcoded logic
        # Assuming format {YYMM}_enh.parquet in PATH_ENH
        path = Path(cr.PATH_ENH) / f"{yymm_str}_enh.parquet"
        
        if not path.exists():
            print(f"[WARN] Model file not found: {path}")
            return False

        df = pd.read_parquet(path)
        
        # Get the very last timestamp available in the file (EOD Yesterday)
        last_ts = df['ts'].max()
        snapshot = df[df['ts'] == last_ts].copy()
        
        # Simplify for lookup: Index by Ticker (using Tenor Map reverse lookup if needed)
        # Here we assume the parquet has 'tenor_yrs' as a float.
        # We index by tenor_yrs for fast O(1) lookup.
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
        # 1. Get Tenor
        tenor = cr.TENOR_YEARS.get(ticker)
        
        # 2. Check if Tenor exists in Model
        if tenor is None or tenor not in MODEL_SNAPSHOT:
            continue
            
        # 3. Extract Frozen Model Data
        model_row = MODEL_SNAPSHOT[tenor]
        model_rate = float(model_row.get('rate', live_rate)) # EOD Rate
        model_z = float(model_row.get('z_comb', 0.0))        # EOD Z-Score
        
        # 4. The Projection
        # Rate Diff: Live (4.25) - Model (4.20) = +0.05
        rate_diff = live_rate - model_rate
        
        # Z Live: Model Z + (Diff / Vol)
        # Yield Up = Price Down = Cheap = High Z-Score (Standard Convention)
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
    # Ensure we don't pick an alternative that is too short (e.g. < 1Y)
    min_alt_tenor = getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0)
    min_sep = getattr(cr, "MIN_SEP_YEARS", 0.5)
    max_span = getattr(cr, "MAX_SPAN_YEARS", 10.0)
    
    for ticker, info in live_z_map.items():
        tenor = info['tenor']
        z_score = info['z_live']
        
        # 1. Skip self
        if tenor == active_tenor: 
            continue
            
        # 2. Enforce Minimum Tenor Length for the Alternative
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
