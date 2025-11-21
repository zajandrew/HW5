"""
overlay_tscv.py

Randomized Hyperparameter Optimization for Overlay Strategy.

- Optimizes 13 parameters (Core Strategy + Regime Filter + Shock Filter).
- Training Data: Trades opened prior to 2025-01-01.
- Execution: Runs simulation slightly past Jan 1, 2025 to allow carry-over trades to close.
- Metric: Maximizes Total Net PnL (bps).
"""

import os
import sys
import time
import random
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# Project modules
import cr_config as cr
import portfolio_test as pt  # Ensure this matches your actual filename
import hybrid_filter as hf
from hybrid_filter import ShockConfig, RegimeThresholds

# ==============================================================================
# 1. CONFIGURATION & SEARCH SPACE
# ==============================================================================

# How many random combinations to try?
# 13 params is huge. 
# 100 iters = rough idea. 
# 1000 iters = decent optimization.
# 5000+ iters = rigorous.
N_ITER = 500 

# Parallel jobs (set to os.cpu_count() or similar)
N_JOBS = max(1, os.cpu_count() - 2)

# The Search Space
# Keys must match cr_config attribute names exactly.
PARAM_SPACE = {
    # --- Core Strategy ---
    "Z_ENTRY":              [0.50, 0.75, 1.00, 1.25],
    "Z_EXIT":               [0.00, 0.25, 0.40, 0.50],
    "Z_STOP":               [1.5, 2.0, 2.5, 3.0],
    "MAX_HOLD_DAYS":        [5, 10, 20],
    
    # --- Regime Filter (Curve) ---
    "MIN_SIGNAL_HEALTH_Z":  [-1.0, -0.5, 0.0],
    "MAX_TRENDINESS_ABS":   [1.5, 2.0, 3.0, 4.0],
    "MAX_Z_XS_MEAN_ABS_Z":  [1.5, 2.0, 3.0],
    
    # --- Shock Filter (PnL) ---
    "SHOCK_PNL_WINDOW":     [5, 10, 20],
    "SHOCK_RAW_PNL_Z_THRESH": [-1.5, -2.0, -2.5],
    "SHOCK_RESID_Z_THRESH":   [-1.5, -2.0, -2.5],
    "SHOCK_BLOCK_LENGTH":     [5, 10, 20],
    "SHOCK_METRIC_TYPE":      ["BPS", "CASH"],
    "SHOCK_MODE":             ["ROLL_OFF", "EXIT_ALL"]
}

# ==============================================================================
# 2. DATA PREPARATION (GLOBAL)
# ==============================================================================
# We load data once in the main process. Workers inherit via copy-on-write (Linux)
# or via global initialization (Windows).

_GLOBAL_TRADES = None
_GLOBAL_SIGNALS = None
_GLOBAL_MONTHS = None

def load_global_data():
    """
    Load trade tape and hybrid signals (curve features) once.
    """
    print("[INIT] Loading global data for optimization...")
    
    # 1. Load Trades
    trades_path = Path(f"{cr.TRADE_TYPES}.pkl")
    if not trades_path.exists():
        raise FileNotFoundError(f"Trade tape {trades_path} not found.")
    
    trades = pd.read_pickle(trades_path)
    # Standardize timestamps
    trades["tradetimeUTC"] = pd.to_datetime(trades["tradetimeUTC"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    
    # 2. Load Signals (Feature Engineering is static, we don't re-run PCA)
    signals = hf.get_or_build_hybrid_signals(force_rebuild=False)
    
    # 3. Define Month List (Train on everything available)
    #    We grab all months from enhanced files.
    root = Path(cr.PATH_ENH)
    suffix = getattr(cr, "ENH_SUFFIX", "")
    files = sorted(root.glob(f"*{suffix}.parquet"))
    months = []
    for f in files:
        # rudimentary parsing "2304_enh..." -> "2304"
        stem = f.stem
        if stem[:4].isdigit():
            months.append(stem[:4])
            
    return trades, signals, sorted(list(set(months)))

# ==============================================================================
# 3. WORKER LOGIC
# ==============================================================================

def _apply_config(params: Dict[str, Any]):
    """Apply params to the local process's cr_config module."""
    for k, v in params.items():
        setattr(cr, k, v)

def _worker_task(params: Dict[str, Any]):
    """
    Executes one backtest with the given parameters.
    """
    # 1. Configure Local Environment
    _apply_config(params)
    
    # Access global data
    trades = _GLOBAL_TRADES
    signals = _GLOBAL_SIGNALS
    all_months = _GLOBAL_MONTHS
    
    # 2. Filter Trade Tape for TRAINING PERIOD only
    #    We strictly want to enter trades BEFORE 2025.
    #    However, we allow the simulation to run INTO 2025 to let them close.
    cutoff_dt = pd.Timestamp("2025-01-01")
    
    # Filter: Entries must be before cutoff
    train_hedges = trades[trades["tradetimeUTC"] < cutoff_dt].copy()
    
    if train_hedges.empty:
        return {**params, "pnl_net_bp": -99999.0, "n_trades": 0}

    # 3. Build Regime Mask (Dynamic based on optimized params)
    #    We must rebuild this because MIN_SIGNAL_HEALTH_Z etc changed.
    regime_mask = hf.regime_mask_from_signals(
        signals, 
        thresholds=RegimeThresholds(
            min_signal_health_z=params["MIN_SIGNAL_HEALTH_Z"],
            max_trendiness_abs=params["MAX_TRENDINESS_ABS"],
            max_z_xs_mean_abs_z=params["MAX_Z_XS_MEAN_ABS_Z"]
        )
    )

    # 4. Configure Shock Filter (Dynamic)
    shock_config = ShockConfig(
        pnl_window=int(params["SHOCK_PNL_WINDOW"]),
        use_raw_pnl=True,
        use_residuals=True,
        raw_pnl_z_thresh=float(params["SHOCK_RAW_PNL_Z_THRESH"]),
        resid_z_thresh=float(params["SHOCK_RESID_Z_THRESH"]),
        regression_cols=list(getattr(cr, "SHOCK_REGRESSION_COLS", [])),
        block_length=int(params["SHOCK_BLOCK_LENGTH"]),
        metric_type=params["SHOCK_METRIC_TYPE"]
    )

    # 5. Run Backtest
    #    We run on ALL months to allow natural rolloff, but `train_hedges` 
    #    ensures we don't open NEW positions in 2025.
    try:
        pos, _, _ = pt.run_all(
            all_months,
            mode="overlay",
            hedge_df=train_hedges,
            regime_mask=regime_mask,
            hybrid_signals=signals,
            shock_cfg=shock_config,
            carry=True,
            force_close_end=True # Force close at very end of dataset if still open
        )
        
        # 6. Calculate Metric: Sum of Net PnL Bps
        #    Note: pos['pnl_net_bp'] accounts for t-costs.
        #    run_month handles the DV01 decay logic internally for pnl_bp.
        if pos.empty:
            score = 0.0
            count = 0
        else:
            score = pos["pnl_net_bp"].sum()
            count = len(pos)
            
    except Exception as e:
        print(f"[ERR] Worker failed: {e}")
        score = -999999.0
        count = 0

    # Return result dict
    result = params.copy()
    result["pnl_net_bp"] = score
    result["n_trades"] = count
    return result

# ==============================================================================
# 4. OPTIMIZER DRIVER
# ==============================================================================

def generate_random_params(n=100) -> List[Dict[str, Any]]:
    """Generates N random combinations from PARAM_SPACE."""
    combos = []
    keys = list(PARAM_SPACE.keys())
    for _ in range(n):
        item = {}
        for k in keys:
            item[k] = random.choice(PARAM_SPACE[k])
        combos.append(item)
    return combos

def init_pool_globals(trades, sigs, months):
    """Initializer for worker processes to set globals."""
    global _GLOBAL_TRADES, _GLOBAL_SIGNALS, _GLOBAL_MONTHS
    _GLOBAL_TRADES = trades
    _GLOBAL_SIGNALS = sigs
    _GLOBAL_MONTHS = months

def run_optimization():
    # 1. Prepare Data
    trades, signals, months = load_global_data()
    print(f"[INFO] Optimization Scope: {len(months)} months available.")
    print(f"[INFO] Training Cutoff: 2025-01-01")
    print(f"[INFO] Parameter Space: {len(PARAM_SPACE)} dimensions.")
    print(f"[INFO] Workers: {N_JOBS} | Iterations: {N_ITER}")

    # 2. Generate Jobs
    jobs = generate_random_params(N_ITER)
    
    # 3. Run Parallel
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=init_pool_globals, initargs=(trades, signals, months)) as executor:
        futures = {executor.submit(_worker_task, job): job for job in jobs}
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                print(f"[PROGRESS] {completed}/{N_ITER} done ({rate:.2f} iters/s)")

    # 4. Analyze Results
    df = pd.DataFrame(results)
    df = df.sort_values("pnl_net_bp", ascending=False)
    
    # Save
    out_dir = Path(cr.PATH_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"optim_results_{int(time.time())}.parquet"
    df.to_parquet(out_path)
    
    print("\n" + "="*40)
    print("TOP 5 CONFIGURATIONS")
    print("="*40)
    print(df.head(5).to_string())
    print(f"\n[DONE] Results saved to {out_path}")

if __name__ == "__main__":
    run_optimization()
