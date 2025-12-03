"""
overlay_tscv.py

Deterministic Grid Search Optimization for Overlay Strategy.
Features:
1. Exhaustive Grid Search of PARAM_SPACE (constrained by logic).
2. Multi-Stage Tournament (R1 -> R2 -> R3).
3. Checkpointing: Skips computation if Round Parquet already exists.
4. "Significance Gating": Penalizes low trade counts.
"""

import os
import sys
import time
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
# 1. CONFIGURATION
# ==============================================================================

# With Grid Search, N_ITER is ignored (we run everything valid).
N_JOBS = max(1, os.cpu_count() - 2)

# Tournament Survival Rates (Percentages)
R1_TOP_PCT = 0.15         
R2_TOP_PCT = 0.25         

# --- STATISTICAL SIGNIFICANCE GATE ---
# Require a minimum number of trades to consider the result valid.
MIN_TRADES_REQUIRED = 500  

OUT_DIR = Path(cr.PATH_OUT)
R1_FILE = OUT_DIR / "mc_checkpoint_r1.parquet"
R2_FILE = OUT_DIR / "mc_checkpoint_r2.parquet"
R3_FILE = OUT_DIR / "mc_checkpoint_r3.parquet"

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

# UPDATED PARAMETER SPACE (From your Image)
PARAM_SPACE = {
    "Z_ENTRY": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
    
    "Z_EXIT":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    
    "Z_STOP":  [0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0, 3.5, 4.0],
    
    "MAX_HOLD_DAYS": [3, 5, 7, 10, 12, 15, 20, 30],
}

# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================

_GLOBAL_TRADES = None
_GLOBAL_SIGNALS = None
_GLOBAL_MONTHS = None

def load_global_data():
    print("[INIT] Loading global data for optimization...")
    signals = hf.get_or_build_hybrid_signals(force_rebuild=False)
    
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
    for k, v in params.items():
        setattr(cr, k, v)
    for k, v in LOCKED_FILTERS.items():
        setattr(cr, k, v)
        
    if "MAX_HOLD_DAYS" in params:
        hold_days = params["MAX_HOLD_DAYS"]
        safety_factor = getattr(cr, "MIN_TENOR_SAFETY_FACTOR", 73.0)
        dynamic_min = hold_days / safety_factor
        limit_exec = max(0.085, dynamic_min)
        limit_alt = max(0.083, dynamic_min)
        setattr(cr, "EXEC_LEG_TENOR_YEARS", limit_exec)
        setattr(cr, "ALT_LEG_TENOR_YEARS", limit_alt)

def _worker_task(task_tuple):
    """Executes backtest and calculates robust t-Stat."""
    params, tape_source = task_tuple
    _apply_config(params)
    
    signals = _GLOBAL_SIGNALS
    all_months = _GLOBAL_MONTHS
    
    if isinstance(tape_source, (str, Path)):
        hedges = pd.read_pickle(tape_source)
        hedges["tradetimeUTC"] = pd.to_datetime(hedges["tradetimeUTC"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        hedges = tape_source

    cutoff_dt = pd.Timestamp("2025-01-01")
    train_hedges = hedges[hedges["tradetimeUTC"] < cutoff_dt].copy()
    
    if train_hedges.empty:
        return {**params, "daily_t_stat": -99.0, "n_trades": 0, "avg_daily_bp": 0.0}

    regime_mask = hf.regime_mask_from_signals(
        signals, thresholds=RegimeThresholds(min_signal_health_z=-99.0)
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
            count = 0
            daily_t_stat = -99.0
            avg_daily = 0.0
        else:
            count = len(pos)
            
            # --- THE GATE: MINIMUM TRADES ---
            if count < MIN_TRADES_REQUIRED:
                daily_t_stat = -99.0 # Disqualified
                
                # Log stats anyway for debugging
                pos["day"] = pos["close_ts"].dt.floor("D")
                avg_daily = pos.groupby("day")["pnl_net_bp"].sum().mean()
            else:
                pos["day"] = pos["close_ts"].dt.floor("D")
                daily_series = pos.groupby("day")["pnl_net_bp"].sum()
                daily_vals = daily_series.values
                
                if len(daily_vals) < 10:
                    daily_t_stat = -99.0
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

    except Exception:
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

def generate_grid_params() -> List[Dict[str, Any]]:
    """
    Generates EVERY valid combination (Deterministic Grid).
    Skips combinations that violate logic constraints to save time.
    """
    keys = list(PARAM_SPACE.keys())
    values = list(PARAM_SPACE.values())
    
    valid_dicts = []
    
    # Cartesian Product of all lists
    for combo in itertools.product(*values):
        item = dict(zip(keys, combo))
        
        # --- LOGIC CONSTRAINTS (The Filter) ---
        
        # 1. Profit Width: Entry must be > Exit by at least 0.5 sigma
        if item["Z_ENTRY"] - item["Z_EXIT"] < 0.5:
            continue
            
        # 2. Stop Distance: Stop must be > Entry by at least 0.5 sigma
        if item["Z_STOP"] - item["Z_ENTRY"] < 0.5:
            continue
            
        valid_dicts.append(item)
        
    print(f"[GRID] Generated {len(valid_dicts)} valid combinations (from {np.prod([len(v) for v in values])} raw).")
    return valid_dicts

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
    if not OUT_DIR.exists(): OUT_DIR.mkdir(parents=True)
    _, signals, months = load_global_data()
    
    synthetic_files = sorted(Path(".").glob("synth_trades_*.pkl"))
    if not synthetic_files:
        print("[ERR] No synthetic trade files found. Run synthetic_data_gen.py first.")
        return

    print(f"[INFO] Found {len(synthetic_files)} synthetic tapes.")
    
    # ==========================================================================
    # ROUND 1: Broad Sweep (Tape 0)
    # ==========================================================================
    if R1_FILE.exists():
        print(f"\n>>> [ROUND 1] SKIPPING: Found checkpoint {R1_FILE.name}")
        df_r1 = pd.read_parquet(R1_FILE)
        print(f"    Loaded {len(df_r1)} results.")
    else:
        print(f"\n>>> [ROUND 1] GENERATING: Full Grid Search on Tape 0...")
        
        # SWITCHED TO GRID GENERATOR
        candidates = generate_grid_params() 
        
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
        df_r1.to_parquet(R1_FILE)
        print(f"    Saved Round 1 results to {R1_FILE.name}")

    # Filter R1 Survivors
    r1_count = len(df_r1)
    r1_survivors = int(r1_count * R1_TOP_PCT)
    top_r1_df = df_r1.head(r1_survivors)
    top_r1_params = top_r1_df[list(PARAM_SPACE.keys())].to_dict('records')
    
    best_r1 = df_r1.iloc[0]
    print(f"   [R1 STATS] Survivors: {r1_survivors} | Best t-Stat: {best_r1['daily_t_stat']:.2f}")

    # ==========================================================================
    # ROUND 2: Validation (Tapes 1-4)
    # ==========================================================================
    if R2_FILE.exists():
        print(f"\n>>> [ROUND 2] SKIPPING: Found checkpoint {R2_FILE.name}")
        df_r2_agg = pd.read_parquet(R2_FILE)
        print(f"    Loaded {len(df_r2_agg)} results.")
    else:
        print(f"\n>>> [ROUND 2] RUNNING: Testing {len(top_r1_params)} params on Tapes 1-4...")
        tasks_r2 = []
        for cand in top_r1_params:
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
        df_r2_agg = df_r2_raw.groupby(param_cols)[["daily_t_stat", "n_trades"]].mean().reset_index()
        df_r2_agg = df_r2_agg.sort_values("daily_t_stat", ascending=False)
        df_r2_agg.to_parquet(R2_FILE)
        print(f"    Saved Round 2 results to {R2_FILE.name}")

    # Filter R2 Survivors
    r2_count = len(df_r2_agg)
    r2_survivors = int(r2_count * R2_TOP_PCT)
    top_r2_df = df_r2_agg.head(r2_survivors)
    top_r2_params = top_r2_df[list(PARAM_SPACE.keys())].to_dict('records')
    print(f"   [R2 STATS] Survivors: {r2_survivors} | Best Avg t-Stat: {df_r2_agg.iloc[0]['daily_t_stat']:.2f}")

    # ==========================================================================
    # ROUND 3: Final Selection (All Tapes)
    # ==========================================================================
    if R3_FILE.exists():
        print(f"\n>>> [ROUND 3] SKIPPING: Found checkpoint {R3_FILE.name}")
        final_stats = pd.read_parquet(R3_FILE)
    else:
        print(f"\n>>> [ROUND 3] RUNNING: Testing {len(top_r2_params)} params on ALL Tapes...")
        tasks_r3 = []
        for cand in top_r2_params:
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
        param_cols = list(PARAM_SPACE.keys())
        
        final_stats = df_r3.groupby(param_cols).agg(
            avg_t_stat=("daily_t_stat", "mean"),
            min_t_stat=("daily_t_stat", "min"),
            std_t_stat=("daily_t_stat", "std"),
            avg_daily_pnl=("avg_daily_bp", "mean"),
            avg_trades=("n_trades", "mean")
        ).reset_index()
        
        final_stats = final_stats.sort_values("avg_t_stat", ascending=False)
        final_stats.to_parquet(R3_FILE)
        print(f"    Saved Round 3 results to {R3_FILE.name}")

    # DISPLAY
    timestamp = int(time.time())
    final_export_path = OUT_DIR / f"tournament_final_results_{timestamp}.csv"
    final_stats.to_csv(final_export_path)
    
    print("\n" + "="*50)
    print(f"TOURNAMENT COMPLETE (Saved to {final_export_path.name})")
    print("="*50)
    print("TOP 5 ROBUST CONFIGURATIONS:")
    print(final_stats.head(5).to_string())

if __name__ == "__main__":
    run_tournament()
