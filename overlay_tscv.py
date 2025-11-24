"""
overlay_tscv.py

Randomized Hyperparameter Optimization for Overlay Strategy.
Stage 1: Tournament Optimization for "Alpha Engine" (Core Params).
Metric: Winsorized Daily t-Statistic.
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
import portfolio_test as pt 
import hybrid_filter as hf
from hybrid_filter import ShockConfig, RegimeThresholds

# ==============================================================================
# 1. CONFIGURATION (STAGE 1: CORE ENGINE)
# ==============================================================================

N_ITER = 2000 
N_JOBS = max(1, os.cpu_count() - 2)

# LOCKED FILTERS (Turned OFF for Stage 1)
LOCKED_FILTERS = {
    "MIN_SIGNAL_HEALTH_Z": -99.0,
    "MAX_TRENDINESS_ABS": 99.0,
    "MAX_Z_XS_MEAN_ABS_Z": 99.0,
    "SHOCK_RAW_PNL_Z_THRESH": -99.0,
    "SHOCK_RESID_Z_THRESH": -99.0,
    "SHOCK_PNL_WINDOW": 10, 
    "SHOCK_BLOCK_LENGTH": 0,
}

# PARAMETER SPACE (Core Strategy Only)
# Tenors are removed here because they are dynamically derived below
PARAM_SPACE = {
    "Z_ENTRY":              [0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2],
    "Z_EXIT":               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "Z_STOP":               [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    "MAX_HOLD_DAYS":        [5, 10, 15, 20, 30],
}

# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================

_GLOBAL_TRADES = None
_GLOBAL_SIGNALS = None
_GLOBAL_MONTHS = None

def load_global_data():
    print("[INIT] Loading global data for optimization...")
    
    # 1. Load Signals (Static)
    signals = hf.get_or_build_hybrid_signals(force_rebuild=False)
    
    # 2. Define Month List
    root = Path(cr.PATH_ENH)
    suffix = getattr(cr, "ENH_SUFFIX", "")
    files = sorted(root.glob(f"*{suffix}.parquet"))
    months = []
    for f in files:
        stem = f.stem
        if stem[:4].isdigit():
            months.append(stem[:4])
            
    return None, signals, sorted(list(set(months)))

# ==============================================================================
# 3. WORKER LOGIC
# ==============================================================================

def _apply_config(params: Dict[str, Any]):
    """
    Apply params to config with DYNAMIC CONSTRAINTS.
    """
    # 1. Apply Variable Params
    for k, v in params.items():
        setattr(cr, k, v)
        
    # 2. Apply Locked Filters
    for k, v in LOCKED_FILTERS.items():
        setattr(cr, k, v)
        
    # 3. DYNAMIC TENOR LIMITS (The 20% Rule)
    #    Constraint: Max Hold Days cannot represent more than 20% of Tenor.
    #    Math: Min Tenor >= Hold_Days / (365 * 0.20) = Hold_Days / 73
    
    if "MAX_HOLD_DAYS" in params:
        hold_days = params["MAX_HOLD_DAYS"]
        
        # Calculate dynamic floor (Years)
        dynamic_min = hold_days / 73.0
        
        # Enforce hard floor (e.g. ~1 month) just in case hold days is tiny
        limit = max(0.084, dynamic_min)
        
        # Apply derived config
        setattr(cr, "EXEC_LEG_TENOR_YEARS", limit)
        setattr(cr, "ALT_LEG_TENOR_YEARS", limit)

def _worker_task(task_tuple):
    """
    Executes backtest and calculates WINSORIZED DAILY STATISTICS.
    """
    params, tape_source = task_tuple
    _apply_config(params)
    
    # Globals
    signals = _GLOBAL_SIGNALS
    all_months = _GLOBAL_MONTHS
    
    if isinstance(tape_source, (str, Path)):
        hedges = pd.read_pickle(tape_source)
        hedges["tradetimeUTC"] = pd.to_datetime(hedges["tradetimeUTC"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        hedges = tape_source

    # Filter Pre-2025
    cutoff_dt = pd.Timestamp("2025-01-01")
    train_hedges = hedges[hedges["tradetimeUTC"] < cutoff_dt].copy()
    
    if train_hedges.empty:
        return {**params, "daily_t_stat": -99.0, "n_trades": 0, "avg_daily_bp": 0.0}

    # Dummy Filters
    regime_mask = hf.regime_mask_from_signals(
        signals, thresholds=hf.RegimeThresholds(min_signal_health_z=-99.0)
    )
    shock_config = ShockConfig(raw_pnl_z_thresh=-99.0, resid_z_thresh=-99.0)

    try:
        pos, _, _ = pt.run_all(
            all_months,
            mode="overlay",
            hedge_df=train_hedges,
            regime_mask=regime_mask,
            hybrid_signals=signals,
            shock_cfg=shock_config,
            carry=True,
            force_close_end=True
        )
        
        if pos.empty:
            daily_t_stat = -10.0
            count = 0
            avg_daily = 0.0
        else:
            pos["day"] = pos["close_ts"].dt.floor("D")
            daily_series = pos.groupby("day")["pnl_net_bp"].sum()
            daily_vals = daily_series.values
            
            if len(daily_vals) < 10:
                daily_t_stat = -10.0
                avg_daily = daily_vals.mean() if len(daily_vals) > 0 else 0.0
            else:
                # Winsorize Left Tail (5%)
                floor_val = np.percentile(daily_vals, 5)
                clipped_vals = np.clip(daily_vals, a_min=floor_val, a_max=None)
                
                mu = clipped_vals.mean()
                sigma = clipped_vals.std(ddof=1)
                n_days = len(clipped_vals)
                
                if sigma < 1e-9:
                    daily_t_stat = 0.0
                else:
                    daily_t_stat = (mu / sigma) * np.sqrt(n_days)
                
                avg_daily = mu
                count = len(pos)

    except Exception as e:
        daily_t_stat = -99.0
        count = 0
        avg_daily = 0.0

    result = params.copy()
    result["n_trades"] = count
    result["daily_t_stat"] = daily_t_stat 
    result["avg_daily_bp"] = avg_daily
    return result

# ==============================================================================
# 4. TOURNAMENT DRIVER
# ==============================================================================

def generate_random_params(n=100) -> List[Dict[str, Any]]:
    combos = []
    keys = list(PARAM_SPACE.keys())
    for _ in range(n):
        item = {}
        for k in keys:
            item[k] = random.choice(PARAM_SPACE[k])
        combos.append(item)
    return combos

def init_pool_globals(sigs, months):
    global _GLOBAL_SIGNALS, _GLOBAL_MONTHS
    _GLOBAL_SIGNALS = sigs
    _GLOBAL_MONTHS = months

def log_progress(start_time, current_idx, total_items, round_name):
    if current_idx == 0: return
    elapsed = time.time() - start_time
    rate = current_idx / elapsed
    remaining = total_items - current_idx
    eta_seconds = remaining / rate if rate > 0 else 0
    
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
    
    print(f"   [{round_name}] {current_idx}/{total_items} done | Rate: {rate:.2f} it/s | Elapsed: {elapsed_str} | ETA: {eta_str}")

def run_tournament():
    _, signals, months = load_global_data()
    
    synthetic_files = sorted(Path(".").glob("synth_trades_*.pkl"))
    if not synthetic_files:
        print("[ERR] No synthetic trade files found. Run synthetic_data_gen.py first.")
        return

    print(f"[INFO] Found {len(synthetic_files)} synthetic tapes. Starting Tournament.")
    candidates = generate_random_params(N_ITER)
    
    # --- ROUND 1 ---
    print(f"\n>>> ROUND 1: Testing {len(candidates)} params on Tape 0...")
    r1_results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=init_pool_globals, initargs=(signals, months)) as ex:
        futures = {ex.submit(_worker_task, (c, synthetic_files[0])): c for c in candidates}
        total = len(candidates)
        for i, f in enumerate(as_completed(futures)):
            r1_results.append(f.result())
            if (i + 1) % max(10, total // 20) == 0:
                log_progress(t0, i + 1, total, "R1")
            
    df_r1 = pd.DataFrame(r1_results).sort_values("daily_t_stat", ascending=False)
    top_200_params = df_r1.head(200)[list(PARAM_SPACE.keys())].to_dict('records')
    best = df_r1.iloc[0]
    print(f"   [R1 BEST] t-Stat: {best['daily_t_stat']:.2f} | Trades: {best['n_trades']} | Avg Day: {best['avg_daily_bp']:.1f}bp")

    # --- ROUND 2 ---
    print(f"\n>>> ROUND 2: Testing Top 200 params on Tapes 1-4...")
    tasks_r2 = []
    for cand in top_200_params:
        for tape in synthetic_files[1:5]:
            tasks_r2.append((cand, tape))
    
    results_r2_raw = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=init_pool_globals, initargs=(signals, months)) as ex:
        futures = {ex.submit(_worker_task, t): t for t in tasks_r2}
        total = len(tasks_r2)
        for i, f in enumerate(as_completed(futures)):
            results_r2_raw.append(f.result())
            if (i + 1) % max(10, total // 10) == 0:
                log_progress(t0, i + 1, total, "R2")
            
    df_r2_raw = pd.DataFrame(results_r2_raw)
    param_cols = list(PARAM_SPACE.keys())
    grouped = df_r2_raw.groupby(param_cols)[["daily_t_stat"]].mean().reset_index()
    grouped = grouped.sort_values("daily_t_stat", ascending=False)
    top_50_params = grouped.head(50)[param_cols].to_dict('records')
    print(f"   [R2 BEST] Avg t-Stat: {grouped.iloc[0]['daily_t_stat']:.2f}")

    # --- ROUND 3 ---
    print(f"\n>>> ROUND 3: Testing Top 50 params on ALL 10 Tapes...")
    tasks_r3 = []
    for cand in top_50_params:
        for tape in synthetic_files:
            tasks_r3.append((cand, tape))
    
    results_r3_raw = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=init_pool_globals, initargs=(signals, months)) as ex:
        futures = {ex.submit(_worker_task, t): t for t in tasks_r3}
        total = len(tasks_r3)
        for i, f in enumerate(as_completed(futures)):
            results_r3_raw.append(f.result())
            if (i + 1) % max(10, total // 10) == 0:
                log_progress(t0, i + 1, total, "R3")
            
    df_r3 = pd.DataFrame(results_r3_raw)
    final_stats = df_r3.groupby(param_cols).agg(
        avg_t_stat=("daily_t_stat", "mean"),
        min_t_stat=("daily_t_stat", "min"),
        avg_daily_pnl=("avg_daily_bp", "mean"),
        avg_trades=("n_trades", "mean")
    ).reset_index()
    
    final_stats = final_stats.sort_values("avg_t_stat", ascending=False)
    out_path = Path(cr.PATH_OUT) / f"tournament_results_{int(time.time())}.parquet"
    final_stats.to_parquet(out_path)
    
    print("\n" + "="*40)
    print("TOP 5 ROBUST CONFIGURATIONS (Winsorized t-Stat)")
    print("="*40)
    print(final_stats.head(5).to_string())
    print(f"\n[DONE] Results saved to {out_path}")

if __name__ == "__main__":
    run_tournament()
