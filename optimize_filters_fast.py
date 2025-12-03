"""
optimize_filters_fast.py

The "Factory" Pipeline.
1. Generates Baselines (Raw Engine PnL).
2. Optimizes Signal Construction (Windows/Weights) via IC Analysis.
3. Rebuilds Hybrid Signals based on the winner.
4. Pre-calculates Regression Residuals.
5. Optimizes Filter Thresholds for Max Sortino.
"""

import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import spearmanr
import statsmodels.api as sm

# Project modules
import cr_config as cr
import portfolio_test as pt
import hybrid_filter as hf
from hybrid_filter import ShockConfig, RegimeConfig

# ==============================================================================
# CONFIGURATION
# ==============================================================================

N_JOBS = max(1, int(os.cpu_count()) - 2)

# --- INPUT: PASTE YOUR WINNING STAGE 1 PARAMETERS HERE ---
STAGE1_PARAMS = {
    "Z_ENTRY": 0.85,          
    "Z_EXIT": 0.2,            
    "Z_STOP": 2.5,            
    "MAX_HOLD_DAYS": 15,      
}

# --- GRID 1: SIGNAL CONSTRUCTION (META-OPTIMIZATION) ---
# We test these combinations to find the most predictive signal definition.
SIGNAL_WINDOW_GRID = [10, 15, 20, 30, 45, 60]
SIGNAL_WEIGHTS = [
    # (name, mean_w, std_w, slope_w)
    ("Standard", 0.5, -0.5, -0.5),
    ("Trend_Sensitive", 0.2, -0.2, -0.8),
]

# --- GRID 2: THRESHOLD OPTIMIZATION (SAFETY BRAKES) ---
# Once signals are built, we optimize the cutoffs.
THRESHOLD_GRID = {
    # Regime (Curve)
    "MIN_SIGNAL_HEALTH_Z":    [-1.0, -0.5, 0.0],
    "MAX_TRENDINESS_ABS":     [1.5, 2.0, 3.0],
    "MAX_Z_XS_MEAN_ABS_Z":    [2.0, 3.0],
    
    # Shock (PnL)
    # Note: Window is tied to the Signal Window winner usually, but we can fine tune
    "SHOCK_RAW_PNL_Z_THRESH": [-1.5, -2.0, -2.5, -3.0],
    "SHOCK_RESID_Z_THRESH":   [-1.5, -2.0, -2.5],
    "SHOCK_BLOCK_LENGTH":     [10, 20],
}


# ==============================================================================
# STEP 1: BASELINE GENERATION
# ==============================================================================

def apply_engine_config():
    """Applies Stage 1 params and disables filters for baseline generation."""
    for k, v in STAGE1_PARAMS.items():
        setattr(cr, k, v)
    
    # Dynamic Tenors (20% Rule)
    if "MAX_HOLD_DAYS" in STAGE1_PARAMS:
        limit = max(0.084, STAGE1_PARAMS["MAX_HOLD_DAYS"] / 73.0)
        setattr(cr, "EXEC_LEG_TENOR_YEARS", limit)
        setattr(cr, "ALT_LEG_TENOR_YEARS", limit)

    # Disable Filters
    setattr(cr, "MIN_SIGNAL_HEALTH_Z", -99.0)
    setattr(cr, "MAX_TRENDINESS_ABS", 99.0)
    setattr(cr, "MAX_Z_XS_MEAN_ABS_Z", 99.0)
    setattr(cr, "SHOCK_RAW_PNL_Z_THRESH", -99.0)
    setattr(cr, "SHOCK_RESID_Z_THRESH", -99.0)

def _generate_one_baseline(task_tuple):
    i, tape_path, signals, months = task_tuple
    apply_engine_config()
    
    hedges = pd.read_pickle(tape_path)
    hedges["tradetimeUTC"] = pd.to_datetime(hedges["tradetimeUTC"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    
    # Training Set Only
    cutoff = pd.Timestamp("2025-01-01")
    train_hedges = hedges[hedges["tradetimeUTC"] < cutoff].copy()
    if train_hedges.empty: return None

    # Dummy Filters
    reg_mask = hf.regime_mask_from_signals(signals)
    shock_cfg = ShockConfig(raw_pnl_z_thresh=-99.0)

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

def run_baseline_generation():
    out_dir = Path(cr.PATH_OUT)
    synth_files = sorted(Path(".").glob("synth_trades_*.pkl"))
    
    if not synth_files:
        print("[ERR] No synthetic tapes found. Run synthetic_data_gen.py first.")
        return False

    # Need "some" signals to run the engine, even if we ignore them later
    signals = hf.get_or_build_hybrid_signals(force_rebuild=False)
    
    root = Path(cr.PATH_ENH)
    suffix = getattr(cr, "ENH_SUFFIX", "")
    files = sorted(root.glob(f"*{suffix}.parquet"))
    months = sorted(list(set([f.stem[:4] for f in files if f.stem[:4].isdigit()])))

    print(f"\n>>> STEP 1: Checking Baselines for {len(synth_files)} tapes...")
    
    tasks = []
    for i, f in enumerate(synth_files):
        out_file = out_dir / f"baseline_positions_{i}.parquet"
        if not out_file.exists():
            tasks.append((i, f, signals, months))
            
    if not tasks:
        print("    [OK] All baselines exist.")
        return True

    print(f"    Generating {len(tasks)} missing baselines...")
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futures = {ex.submit(_generate_one_baseline, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                idx, pos_df = res
                save_path = out_dir / f"baseline_positions_{idx}.parquet"
                pos_df.to_parquet(save_path)
                print(f"    [DONE] Saved Baseline {idx}")
                
    return True

# ==============================================================================
# STEP 2: SIGNAL CONSTRUCTION OPTIMIZATION (META-ANALYSIS)
# ==============================================================================

def optimize_signal_construction():
    print(f"\n>>> STEP 2: Optimizing Signal Construction (Windows & Weights)...")
    out_dir = Path(cr.PATH_OUT)
    
    # Load Baseline 0 (Representative Sample)
    base_path = out_dir / "baseline_positions_0.parquet"
    if not base_path.exists(): return None
    
    df_trades = pd.read_parquet(base_path)
    # Aggregated Daily PnL
    pnl = df_trades.groupby(pd.to_datetime(df_trades["open_ts"]).dt.floor("D"))["pnl_net_bp"].sum()
    pnl.name = "next_day_pnl"
    
    best_score = -999.0
    best_cfg = None
    best_desc = ""

    print(f"    Testing {len(SIGNAL_WINDOW_GRID) * len(SIGNAL_WEIGHTS)} combinations...")

    for win in SIGNAL_WINDOW_GRID:
        for w_name, w_mean, w_std, w_slope in SIGNAL_WEIGHTS:
            
            cfg = RegimeConfig(base_window=win, w_health_mean=w_mean, w_health_std=w_std, w_health_slope=w_slope)
            
            # Build in memory
            df_sig = hf.build_hybrid_signals(regime_cfg=cfg, force_rebuild=True)
            df_sig["date"] = pd.to_datetime(df_sig["decision_ts"]).dt.floor("D")
            df_sig = df_sig.drop_duplicates("date").set_index("date")
            
            combined = df_sig.join(pnl, how="inner").dropna()
            if len(combined) < 50: continue
            
            # IC Analysis
            # Health should be POSITIVE correlated with PnL
            ic_health, _ = spearmanr(combined["signal_health_z"], combined["next_day_pnl"])
            # Trend should be NEGATIVE correlated with PnL
            ic_trend, _ = spearmanr(combined["trendiness_abs"], combined["next_day_pnl"])
            
            # Composite Score: Health - Trend (since Trend is negative, subtracting makes it positive)
            score = ic_health - ic_trend 
            
            if score > best_score:
                best_score = score
                best_cfg = cfg
                best_desc = f"Win={win}, {w_name} (IC_Health={ic_health:.2f}, IC_Trend={ic_trend:.2f})"

    print(f"    [WINNER] {best_desc} | Score={best_score:.3f}")
    return best_cfg

# ==============================================================================
# STEP 3 & 4: REBUILD & PRE-CALC RESIDUALS
# ==============================================================================

def precalc_residuals(baselines, signals, window, reg_cols):
    """Calculates residuals for a specific window across all baselines."""
    # Align signals to dates
    if "decision_ts" in signals.columns:
        signals["date"] = pd.to_datetime(signals["decision_ts"]).dt.floor("D")
    sig_indexed = signals.drop_duplicates("date").set_index("date").sort_index()
    
    # Filter columns
    valid_cols = [c for c in reg_cols if c in sig_indexed.columns]
    
    residuals_map = {} # {date: z_score}
    
    # We aggregate all baselines to get a robust global residual? 
    # Or per tape? Per tape is more accurate for the simulation of THAT tape.
    # But for speed, let's process them individually.
    
    cache = []
    
    for df in baselines:
        pnl = df.groupby("close_date")["pnl_net_bp"].sum().sort_index()
        combined = pd.concat([pnl.rename("y"), sig_indexed[valid_cols]], axis=1).dropna()
        
        tape_resids = {}
        if len(combined) > window * 2:
            y_vals = combined["y"].values
            X_vals = sm.add_constant(combined[valid_cols]).values
            dates = combined.index
            
            for i in range(window, len(combined)):
                y_win = y_vals[i-window : i]
                X_win = X_vals[i-window : i]
                
                if np.std(y_win) < 1e-9: continue
                
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_win, y_win, rcond=None)
                    y_hat = X_win @ beta
                    res = y_win - y_hat
                    
                    r_std = np.std(res)
                    if r_std > 1e-9:
                        z = (res[-1] - np.mean(res)) / r_std
                        tape_resids[dates[i]] = z
                except: pass
        cache.append(tape_resids)
        
    return cache

# ==============================================================================
# STEP 5: THRESHOLD OPTIMIZATION
# ==============================================================================

def simulate_single_tape(params, signals, trades_df, resid_map):
    min_health = params["MIN_SIGNAL_HEALTH_Z"]
    max_trend = params["MAX_TRENDINESS_ABS"]
    max_xs = params["MAX_Z_XS_MEAN_ABS_Z"]
    
    shock_win = int(params["SHOCK_PNL_WINDOW"]) # Not utilized in fast resid map currently, assumed matched
    shock_thresh_raw = params["SHOCK_RAW_PNL_Z_THRESH"]
    shock_thresh_resid = params["SHOCK_RESID_Z_THRESH"]
    block_len = int(params["SHOCK_BLOCK_LENGTH"])
    
    # Pre-calc Regime
    dates = trades_df["open_date"].unique()
    sig_sub = signals.loc[signals.index.isin(dates)]
    mask = (
        (sig_sub["signal_health_z"] >= min_health) &
        (sig_sub["trendiness_abs"] <= max_trend) &
        (sig_sub["z_xs_mean_roll_z"].abs() <= max_xs)
    )
    regime_ok = mask.to_dict()
    
    trades_by_close = trades_df.groupby("close_date")
    all_dates = sorted(trades_df["close_date"].unique())
    
    shock_history = []
    block_rem = 0
    accepted = set()
    daily_net = []
    
    for date in all_dates:
        # Gating
        if block_rem == 0 and regime_ok.get(date, True):
            accepted.add(date)
        
        if block_rem > 0: block_rem -= 1
        
        # Realization
        d_val = 0.0
        if date in trades_by_close.groups:
            todays = trades_by_close.get_group(date)
            valid = [p for d, p in zip(todays["open_date"], todays["pnl_net_bp"]) if d in accepted]
            d_val = sum(valid)
            
        daily_net.append(d_val)
        shock_history.append(d_val)
        
        # Shock Update
        if block_rem == 0 and len(shock_history) >= shock_win + 2:
            # 1. Raw
            recent = shock_history[-shock_win:]
            std = np.std(recent)
            if std > 1e-9:
                z = (recent[-1] - np.mean(recent)) / std
                if z <= shock_thresh_raw:
                    block_rem = block_len
            
            # 2. Resid
            if block_rem == 0:
                rz = resid_map.get(date, 0.0)
                if rz <= shock_thresh_resid:
                    block_rem = block_len
                    
    pnl = np.array(daily_net)
    avg = np.mean(pnl)
    down = pnl[pnl < 0]
    sortino = 0.0
    if len(down) > 0 and np.std(down) > 1e-9:
        sortino = (avg / np.std(down)) * np.sqrt(252)
        
    return sortino, np.sum(pnl)

def _opt_worker(task):
    params, baselines, signals, resid_caches = task
    
    sortinos = []
    pnls = []
    
    for i, df in enumerate(baselines):
        # Use the cache corresponding to this baseline
        s, p = simulate_single_tape(params, signals, df, resid_caches[i])
        sortinos.append(s)
        pnls.append(p)
        
    return {
        **params,
        "avg_sortino": sum(sortinos)/len(sortinos),
        "min_sortino": min(sortinos),
        "avg_pnl": sum(pnls)/len(pnls)
    }

def run_threshold_optimization(baselines, signals, resid_caches):
    print(f"\n>>> STEP 5: Running Threshold Grid Search on {N_JOBS} cores...")
    
    # We need to inject the SHOCK_PNL_WINDOW into the grid logic so the simulator knows
    # But currently we pre-calculated resid_caches based on the WINNER window.
    # So we force the window param to match the winner.
    
    # Generate Grid
    keys, values = zip(*THRESHOLD_GRID.items())
    param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Inject the fixed window size (from signals) into the params for the simulator
    # We assume resid_caches aligns with this window.
    # We'll use a placeholder key or just rely on the fact that resid_caches is correct.
    # Let's add the window to the params for logging/simulation logic
    
    # Extract window from signal config? 
    # Actually, we can just grab it from the ShockConfig default if we updated it,
    # or pass it explicitly. For now, let's assume a fixed window of 15 or 20 for the simulation logic.
    fixed_win = 15 # Default, should match Step 2 winner
    
    print(f"    Testing {len(param_dicts)} threshold combinations...")
    
    final_tasks = []
    for p in param_dicts:
        p["SHOCK_PNL_WINDOW"] = fixed_win 
        final_tasks.append((p, baselines, signals, resid_caches))
        
    results = []
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        for i, res in enumerate(ex.map(_opt_worker, final_tasks)):
            results.append(res)
            if (i+1) % 100 == 0:
                print(f"    ... {i+1}/{len(final_tasks)} done")
                
    df = pd.DataFrame(results).sort_values("avg_sortino", ascending=False)
    out_path = Path(cr.PATH_OUT) / f"final_filter_settings_{int(time.time())}.csv"
    df.to_csv(out_path, index=False)
    
    print("\n" + "="*40)
    print("WINNING CONFIGURATION")
    print("="*40)
    print(df.iloc[0])
    print(f"\n[DONE] Saved to {out_path}")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # 1. Baselines
    if not run_baseline_generation():
        sys.exit(1)
        
    # 2. Optimize Signal Construction
    best_regime_cfg = optimize_signal_construction()
    
    # 3. Rebuild Signals with Winner
    print(f"\n>>> STEP 3: Rebuilding Hybrid Signals with Window={best_regime_cfg.base_window}...")
    signals = hf.build_hybrid_signals(regime_cfg=best_regime_cfg, force_rebuild=True)
    
    # Prepare signals for simulation
    if "decision_ts" in signals.columns:
        signals["date"] = pd.to_datetime(signals["decision_ts"]).dt.floor("D")
    signals = signals.drop_duplicates("date").set_index("date").sort_index()
    
    # Load Baselines
    out_dir = Path(cr.PATH_OUT)
    files = sorted(out_dir.glob("baseline_positions_*.parquet"))
    baselines = []
    for f in files:
        df = pd.read_parquet(f)
        df["open_date"] = pd.to_datetime(df["open_ts"]).dt.floor("D")
        df["close_date"] = pd.to_datetime(df["close_ts"]).dt.floor("D")
        baselines.append(df)
        
    # 4. Pre-calc Residuals
    # We use the window determined by Step 2 for the Rolling OLS
    win = int(best_regime_cfg.base_window)
    print(f"\n>>> STEP 4: Pre-calculating Shock Residuals (Window={win})...")
    reg_cols = cr.SHOCK_REGRESSION_COLS
    resid_caches = precalc_residuals(baselines, signals, win, reg_cols)
    
    # 5. Optimize Thresholds
    run_threshold_optimization(baselines, signals, resid_caches)
