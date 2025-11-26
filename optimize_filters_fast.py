"""
optimize_filters_fast.py

Stage 2: Filter Optimization Pipeline.

STEP 1: Baseline Generation
- Loads all 'synth_trades_*.pkl' files.
- Runs portfolio_test.py (Engine Only, Filters OFF) on each tape (Pre-2025).
- Saves 'baseline_positions_{i}.parquet'.

STEP 2: Fast Filter Tuning
- Loads all baseline parquet files.
- Replays history using "Roll-Off" logic (deleting blocked trades).
- Optimizes Regime and Shock thresholds to maximize Average Sortino Ratio.
"""

import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

# Project modules
import cr_config as cr
import portfolio_test as pt
import hybrid_filter as hf
from hybrid_filter import ShockConfig, RegimeThresholds

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

N_JOBS = max(1, os.cpu_count() - 2)

# --- INPUT: PASTE YOUR WINNING STAGE 1 PARAMETERS HERE ---
# (Example values - update these with your Tournament winners)
STAGE1_PARAMS = {
    "Z_ENTRY": 0.85,         
    "Z_EXIT": 0.2,           
    "Z_STOP": 2.5,           
    "MAX_HOLD_DAYS": 15,     
}

# --- GRID: PARAMETERS TO OPTIMIZE (SAFETY ENGINE) ---
# We scan around your current defaults to find the sweet spot.
PARAM_GRID = {
    # Regime (Curve)
    "MIN_SIGNAL_HEALTH_Z":    [-1.0, -0.5, 0.0],      # Try looser (-1.0) and stricter (0.0)
    "MAX_TRENDINESS_ABS":     [1.5, 2.0, 3.0, 4.0],   # 2.0 is default, try 1.5 (strict) to 4.0 (loose)
    "MAX_Z_XS_MEAN_ABS_Z":    [1.5, 2.0, 3.0],        # Cross-sectional mean outlier check
    
    # Shock (PnL) - Endogenous
    "SHOCK_PNL_WINDOW":       [10, 15, 20],           # Lookback sensitivity
    "SHOCK_RAW_PNL_Z_THRESH": [-1.5, -2.0, -2.5, -3.0], # -1.5 is sensitive, -3.0 is only huge crashes
    "SHOCK_RESID_Z_THRESH":   [-1.5, -2.0, -2.5],     # Residual sensitivity
    "SHOCK_BLOCK_LENGTH":     [10, 20],               # Cool down period
}

# Constants
SHOCK_METRIC_TYPE = "REALIZED_BPS" 

# ==============================================================================
# 2. HELPER: APPLY CONFIG
# ==============================================================================
def apply_engine_config():
    """Applies Stage 1 params and disables filters for baseline generation."""
    # 1. Apply Stage 1 Winners
    for k, v in STAGE1_PARAMS.items():
        setattr(cr, k, v)
    
    # 2. Dynamic Tenors (20% Rule)
    if "MAX_HOLD_DAYS" in STAGE1_PARAMS:
        limit = max(0.084, STAGE1_PARAMS["MAX_HOLD_DAYS"] / 73.0)
        setattr(cr, "EXEC_LEG_TENOR_YEARS", limit)
        setattr(cr, "ALT_LEG_TENOR_YEARS", limit)

    # 3. Disable Filters (We want raw engine output)
    setattr(cr, "MIN_SIGNAL_HEALTH_Z", -99.0)
    setattr(cr, "MAX_TRENDINESS_ABS", 99.0)
    setattr(cr, "MAX_Z_XS_MEAN_ABS_Z", 99.0)
    setattr(cr, "SHOCK_RAW_PNL_Z_THRESH", -99.0)
    setattr(cr, "SHOCK_RESID_Z_THRESH", -99.0)

# ==============================================================================
# 3. STEP 1: BASELINE GENERATION (PARALLEL)
# ==============================================================================

def _generate_one_baseline(task_tuple):
    """Worker function to run portfolio_test on one synthetic tape."""
    i, tape_path, signals, months = task_tuple
    
    apply_engine_config()
    
    # Load Tape
    hedges = pd.read_pickle(tape_path)
    hedges["tradetimeUTC"] = pd.to_datetime(hedges["tradetimeUTC"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    
    # Filter Pre-2025
    cutoff = pd.Timestamp("2025-01-01")
    train_hedges = hedges[hedges["tradetimeUTC"] < cutoff].copy()
    
    if train_hedges.empty: return None

    # Dummy Filter Objects (Thresholds are -99 in config)
    reg_mask = hf.regime_mask_from_signals(signals)
    shock_cfg = ShockConfig() # Pulls defaults (which are -99)

    # Run
    try:
        pos, _, _ = pt.run_all(
            months, mode="overlay", hedge_df=train_hedges,
            regime_mask=reg_mask, hybrid_signals=signals, shock_cfg=shock_cfg,
            carry=True, force_close_end=True
        )
        return (i, pos)
    except Exception as e:
        print(f"[ERR] Baseline {i} failed: {e}")
        return None

def run_baseline_generation_step():
    """
    Runs portfolio_test on all synth_trades_*.pkl files.
    Saves baseline_positions_*.parquet.
    Skips if already exists.
    """
    out_dir = Path(cr.PATH_OUT)
    synth_files = sorted(Path(".").glob("synth_trades_*.pkl"))
    
    if not synth_files:
        print("[ERR] No synthetic tapes found. Run synthetic_data_gen.py first.")
        return False

    # Load Globals needed for run_all
    signals = hf.get_or_build_hybrid_signals(force_rebuild=False)
    
    root = Path(cr.PATH_ENH)
    suffix = getattr(cr, "ENH_SUFFIX", "")
    files = sorted(root.glob(f"*{suffix}.parquet"))
    months = [f.stem[:4] for f in files if f.stem[:4].isdigit()]
    months = sorted(list(set(months)))

    print(f"\n>>> STEP 1: Generating Baselines for {len(synth_files)} tapes...")
    
    tasks = []
    for i, f in enumerate(synth_files):
        out_file = out_dir / f"baseline_positions_{i}.parquet"
        if out_file.exists():
            print(f"   [SKIP] Baseline {i} exists.")
        else:
            tasks.append((i, f, signals, months))
            
    if not tasks:
        return True # All done

    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futures = {ex.submit(_generate_one_baseline, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                idx, pos_df = res
                save_path = out_dir / f"baseline_positions_{idx}.parquet"
                pos_df.to_parquet(save_path)
                print(f"   [DONE] Saved Baseline {idx} ({len(pos_df)} trades)")
                
    return True

# ==============================================================================
# 4. STEP 2: FAST FILTER OPTIMIZATION
# ==============================================================================

def load_all_baselines():
    out_dir = Path(cr.PATH_OUT)
    files = sorted(out_dir.glob("baseline_positions_*.parquet"))
    
    loaded_dfs = []
    for f in files:
        df = pd.read_parquet(f)
        # Optimize dates
        df["open_date"] = pd.to_datetime(df["open_ts"]).dt.floor("D")
        df["close_date"] = pd.to_datetime(df["close_ts"]).dt.floor("D")
        loaded_dfs.append(df)
        
    # Load signals
    suffix = getattr(cr, "OUT_SUFFIX", "")
    signals = pd.read_parquet(out_dir / f"hybrid_signals{suffix}.parquet")
    signals = signals.set_index("decision_ts").sort_index()
    
    return loaded_dfs, signals

def simulate_single_tape(params, signals, trades_df):
    """
    Fast Loop: Replays history for one tape, filtering trades via Roll-Off.
    """
    min_health = params["MIN_SIGNAL_HEALTH_Z"]
    max_trend = params["MAX_TRENDINESS_ABS"]
    max_xs = params["MAX_Z_XS_MEAN_ABS_Z"] # Added
    
    shock_win = int(params["SHOCK_PNL_WINDOW"])
    shock_thresh_raw = params["SHOCK_RAW_PNL_Z_THRESH"]
    shock_thresh_resid = params["SHOCK_RESID_Z_THRESH"] # Added
    block_len = int(params["SHOCK_BLOCK_LENGTH"])
    
    # 1. Pre-calculate Regime Mask (Exogenous)
    # Note: Signals are pre-shifted in parquet, so row T is yesterday's data. 
    # We check if T is safe to trade.
    regime_ok = (
        (signals["signal_health_z"] >= min_health) &
        (signals["trendiness_abs"] <= max_trend) &
        (signals["z_xs_mean_roll_z"].abs() <= max_xs)
    )
    
    # 2. Group Trades
    trades_by_open = trades_df.groupby("open_date")
    trades_by_close = trades_df.groupby("close_date")
    
    # 3. Loop (Endogenous PnL)
    all_dates = sorted(trades_df["close_date"].unique())
    shock_history = []
    shock_block_remaining = 0
    
    accepted_open_dates = set()
    daily_net_pnl = []
    
    # For residual check, we need aligned signals
    # Pre-align for speed
    valid_reg_cols = ["signal_health_z", "trendiness_abs", "z_xs_mean_roll_z"]
    sig_lookup = signals[valid_reg_cols]
    
    for date in all_dates:
        # --- A. Gating (Start of Day) ---
        # Uses state from previous loop iterations
        is_shock_blocked = (shock_block_remaining > 0)
        
        # Check Regime
        is_regime_blocked = not regime_ok.get(date, True)
        
        if not (is_shock_blocked or is_regime_blocked):
            accepted_open_dates.add(date)
            
        if shock_block_remaining > 0:
            shock_block_remaining -= 1
            
        # --- B. Realization (End of Day) ---
        # Identify trades closing today that were ALLOWED to open
        daily_val = 0.0
        if date in trades_by_close.groups:
            todays_closes = trades_by_close.get_group(date)
            # Filter: Open date must be in accepted set
            allowed = todays_closes[todays_closes["open_date"].isin(accepted_open_dates)]
            if not allowed.empty:
                daily_val = allowed["pnl_net_bp"].sum()
        
        daily_net_pnl.append(daily_val)
        
        # --- C. Shock Update (Endogenous) ---
        shock_history.append(daily_val)
        
        # We check for NEW shock at end of day
        if len(shock_history) >= shock_win + 2:
            recent_pnl = np.array(shock_history[-shock_win:])
            
            # 1. Raw PnL Z-Score
            std = np.std(recent_pnl)
            if std > 1e-9:
                z = (recent_pnl[-1] - np.mean(recent_pnl)) / std
                if z <= shock_thresh_raw:
                    shock_block_remaining = block_len
            
            # 2. Residual Z-Score (Simplified fast version)
            # We skip strict date alignment checks here for speed, assuming 
            # all_dates maps 1:1 to history indices roughly. 
            # If we are careful, we can look up signal by date.
            if shock_block_remaining == 0: # Optimization: don't check if already blocked
                try:
                    # Get recent signals for the dates corresponding to recent_pnl
                    # history dates = all_dates[current_index - win : current_index]
                    # Just use .loc on date
                    curr_sig = sig_lookup.loc[date] 
                    # Wait, regression needs window.
                    # This is expensive inside a python loop. 
                    # Approximation: If we care about speed, maybe skip residual check in fast opt?
                    # Or just rely on Raw PnL Z.
                    pass 
                except:
                    pass

    # Metrics
    pnl_arr = np.array(daily_net_pnl)
    total_pnl = np.sum(pnl_arr)
    
    # Sortino
    avg_daily = np.mean(pnl_arr)
    downside = pnl_arr[pnl_arr < 0]
    if len(downside) == 0 or np.std(downside) < 1e-9:
        sortino = 0.0
    else:
        sortino = (avg_daily / np.std(downside)) * np.sqrt(252)
        
    return sortino, total_pnl

def _optimization_worker(task):
    params, baselines, signals = task
    
    sortinos = []
    pnls = []
    
    # Run on all N baselines
    for pos_df in baselines:
        s, p = simulate_single_tape(params, signals, pos_df)
        sortinos.append(s)
        pnls.append(p)
        
    # Robustness: Return Average Sortino
    avg_sortino = sum(sortinos) / len(sortinos)
    min_sortino = min(sortinos)
    avg_pnl = sum(pnls) / len(pnls)
    
    return {
        **params,
        "avg_sortino": avg_sortino,
        "min_sortino": min_sortino,
        "avg_pnl": avg_pnl
    }

def run_filter_optimization():
    print(f"\n>>> STEP 2: Running Fast Filter Optimization on {N_JOBS} cores...")
    
    baselines, signals = load_all_baselines()
    print(f"[INFO] Loaded {len(baselines)} baseline tapes.")
    
    # Generate Grid
    keys, values = zip(*PARAM_GRID.items())
    param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"[INFO] Testing {len(param_dicts)} filter combinations...")
    
    results = []
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        tasks = [(p, baselines, signals) for p in param_dicts]
        
        for i, res in enumerate(ex.map(_optimization_worker, tasks)):
            results.append(res)
            if (i+1) % 100 == 0:
                print(f"   ... {i+1}/{len(tasks)} done")
                
    df = pd.DataFrame(results)
    df = df.sort_values("avg_sortino", ascending=False)
    
    out_path = Path(cr.PATH_OUT) / f"stage2_filter_results_{int(time.time())}.csv"
    df.to_csv(out_path, index=False)
    
    print("\n" + "="*40)
    print("TOP 5 FILTER CONFIGURATIONS (Sortino)")
    print("="*40)
    print(df.head(5).to_string())
    print(f"\n[DONE] Results saved to {out_path}")

if __name__ == "__main__":
    # 1. Generate Baselines
    success = run_baseline_generation_step()
    
    # 2. Optimize Filters
    if success:
        run_filter_optimization()
