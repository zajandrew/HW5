"""
synthetic_data_gen.py

Generates 'Mutant' variants of the historical trade tape for robustness testing.
1. Loads the real trade tape (trades.pkl).
2. Creates N variants (default 10).
3. Applies random Time Jitter (Business Days) and Size Jitter (DV01).
4. Calls 'stitch_curve_trades' to attach the actual historical curve data 
   corresponding to the NEW (jittered) timestamps.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from pandas.tseries.offsets import BusinessDay
import time

# Project Config
import cr_config as cr

# Import the Stitcher Module
try:
    import stitch_curve_trades as stitcher
except ImportError:
    print("[ERR] Could not import 'stitch_curve_trades'.")
    print("      Please ensure your stitching script is in this directory.")
    raise

# ==============================================================================
# CONFIG
# ==============================================================================
N_VARIANTS = 10
JITTER_DAYS_MIN = -5
JITTER_DAYS_MAX = 5
JITTER_SIZE_PCT = 0.10  # +/- 10% size variance
SOURCE_FILE = "trades.pkl" 

def load_original_trades() -> pd.DataFrame:
    path = Path(SOURCE_FILE)
    if not path.exists():
        # Fallback to config if explicit file missing
        path_alt = Path(f"{cr.TRADE_TYPES}.pkl")
        if path_alt.exists():
            print(f"[WARN] {SOURCE_FILE} not found, falling back to {path_alt}")
            path = path_alt
        else:
            raise FileNotFoundError(f"Original trade tape not found at: {path}")
    
    print(f"[LOAD] Reading original tape: {path}")
    df = pd.read_pickle(path)
    
    # Standardize Timestamp
    if "tradetimeUTC" in df.columns:
        df["tradetimeUTC"] = pd.to_datetime(df["tradetimeUTC"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    
    return df

def generate_mutant_tape(original_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Creates a single mutated copy of the trade tape.
    """
    df = original_df.copy()
    rng = np.random.default_rng(seed)
    n = len(df)
    
    # 1. Time Jitter (Business Days)
    #    We use BusinessDay to avoid moving trades to weekends where curve data is missing.
    days_shift = rng.integers(low=JITTER_DAYS_MIN, high=JITTER_DAYS_MAX + 1, size=n)
    
    # Optimization: Create a cache: shift -> BDay object
    unique_shifts = np.unique(days_shift)
    offset_map = {s: BusinessDay(s) for s in unique_shifts}
    
    # Apply Jitter
    timestamps = df["tradetimeUTC"].tolist()
    new_dates = []
    
    for ts, shift in zip(timestamps, days_shift):
        new_ts = ts + offset_map[shift]
        new_dates.append(new_ts)
        
    df["tradetimeUTC"] = new_dates
    
    # 2. Size Jitter
    #    Vary EqVolDelta / dv01 by +/- JITTER_SIZE_PCT
    tgt_col = "EqVolDelta" if "EqVolDelta" in df.columns else "dv01"
    
    if tgt_col in df.columns:
        noise = rng.uniform(1.0 - JITTER_SIZE_PCT, 1.0 + JITTER_SIZE_PCT, size=n)
        df[tgt_col] = df[tgt_col] * noise
    
    # 3. Cleanup
    #    Sort by new times so stitching works efficiently
    df = df.sort_values("tradetimeUTC").reset_index(drop=True)
    
    return df

def run_generation():
    trades = load_original_trades()
    print(f"[INFO] Original Trades: {len(trades)} rows.")
    
    out_dir = Path(".") 
    
    for i in range(N_VARIANTS):
        print(f"\n--- Generating Variant {i+1}/{N_VARIANTS} ---")
        
        # 1. Mutate
        mutant_df = generate_mutant_tape(trades, seed=42 + i)
        
        # 2. Stitch
        print(f"   ... Stitching curve data (might take a moment)...")
        
        try:
            # Use the imported stitcher module
            stitched_df = stitcher.attach_curve_to_trades(
                trades_path=f"temp_mutant_{i}.pkl", # Dummy path logic if func requires path
                out_path=None 
            )
            # WAIT: attach_curve_to_trades expects a PATH string as input, 
            # but we have a DataFrame in memory.
            # We need to adapt slightly. The user's provided stitcher file 
            # reads from disk. Let's save temp file to be safe.
            
            temp_path = Path(f"temp_mutant_input_{i}.pkl")
            mutant_df.to_pickle(temp_path)
            
            stitched_df = stitcher.attach_curve_to_trades(
                trades_path=temp_path,
                out_path=None
            )
            
            # Clean up temp
            if temp_path.exists():
                temp_path.unlink()
            
            # 3. Save Final
            fname = f"synth_trades_{i}.pkl"
            save_path = out_dir / fname
            stitched_df.to_pickle(save_path)
            
            print(f"   [SUCCESS] Saved {fname} ({len(stitched_df)} rows)")
            
            kept_pct = len(stitched_df) / len(trades) * 100
            print(f"   Retention: {kept_pct:.1f}%")
            
        except Exception as e:
            print(f"   [FAIL] Variant {i} failed stitching: {e}")
            # Clean up temp if failed
            if Path(f"temp_mutant_input_{i}.pkl").exists():
                Path(f"temp_mutant_input_{i}.pkl").unlink()

if __name__ == "__main__":
    t0 = time.time()
    run_generation()
    print(f"\n[DONE] Total time: {time.time() - t0:.1f}s")
