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
        # Here we assume the parquet has 'tenor_yrs' and we map back to tickers via cr.TENOR_YEARS
        # For simplicity in prototype: create a map of Tenor -> Row
        MODEL_SNAPSHOT = snapshot.set_index('tenor_yrs').to_dict('index')
        print(f"[SYSTEM] Loaded Midnight Model: {last_ts} ({len(MODEL_SNAPSHOT)} tenors)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

def get_live_z_scores(live_rates_map):
    """
    Project Live Rates onto Frozen Model.
    Z_live = Z_model + (Rate_live - Rate_model) / Vol_Proxy
    """
    if not MODEL_SNAPSHOT:
        return {}

    z_map = {}
    
    # Vol Proxy: 10bps = 1 Sigma (Simplification if sigma not in parquet)
    # Ideally, pull 'sigma' from feature_creation if saved.
    VOL_PROXY = 0.0010 

    for ticker, live_rate in live_rates_map.items():
        tenor = cr.TENOR_YEARS.get(ticker)
        if tenor is None or tenor not in MODEL_SNAPSHOT:
            continue
            
        model_row = MODEL_SNAPSHOT[tenor]
        model_rate = model_row.get('rate', live_rate)
        model_z = model_row.get('z_comb', 0.0)
        
        # The Projection
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
    Finds candidates.
    If Direction = 'PAY' (We are paying fixed on active_ticker), 
    we want to REC fixed on a cheap partner.
    Active Ticker is likely RICH (High Z) if we are Paying?
    Actually, let's follow standard RV:
    - If user clicks PAY 5Y -> They think 5Y is CHEAP (Low Z) or they have client flow.
    - If flow is Pay 5Y, we need to Rec X (Hedge).
    - We want to Rec the CHEAPEST possible tenor (Highest Z).
    
    Returns list of dicts: [{'ticker':..., 'tenor':..., 'z_score':..., 'rank':...}]
    """
    active_tenor = cr.TENOR_YEARS.get(active_ticker)
    candidates = []
    
    # We want to HEDGE the input. 
    # Input: Pay Fixed. Hedge: Rec Fixed.
    # We look for High Z-scores to Rec.
    
    for ticker, info in live_z_map.items():
        tenor = info['tenor']
        z_score = info['z_live']
        
        # Skip self
        if tenor == active_tenor: 
            continue
            
        # Basic Constraints (from cr_config logic)
        if abs(tenor - active_tenor) < cr.MIN_SEP_YEARS: continue
        if abs(tenor - active_tenor) > cr.MAX_SPAN_YEARS: continue
        
        # Calculate Spread Improvement
        # If paying active (Z_a) and receiving candidate (Z_c)
        # We capture (Z_c - Z_a). We want to maximize this.
        spread_imp = z_score - live_z_map[active_ticker]['z_live']
        
        # In Rec mode (User Recs Active), Hedge is Pay Candidate.
        # We want to Pay the RICHEST possible tenor (Lowest Z).
        if direction == 'REC':
            spread_imp = live_z_map[active_ticker]['z_live'] - z_score
            
        candidates.append({
            'ticker': ticker,
            'tenor': tenor,
            'z_live': z_score,
            'spread_imp': spread_imp
        })
        
    # Sort by spread improvement (descending)
    candidates.sort(key=lambda x: x['spread_imp'], reverse=True)
    return candidates
