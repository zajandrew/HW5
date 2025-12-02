"""
tune_signal_construction.py

Meta-Analysis to select the best construction parameters for hybrid_signals.
Does NOT optimize thresholds. Optimizes DEFINITIONS.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

import cr_config as cr
import hybrid_filter as hf
from hybrid_filter import RegimeConfig

# ==============================================================================
# CONFIG GRID TO TEST
# ==============================================================================
# We will test which Rolling Window creates the most predictive signals
WINDOW_GRID = [5, 10, 15, 20, 30, 45, 60]

# We can also test weights if you want, but Window is usually the critical one.
# For now, we stick to standard weights but vary the lookback.
WEIGHT_CONFIGS = [
    # (name, mean_w, std_w, slope_w)
    ("Standard", 0.5, -0.5, -0.5),
    ("Vol_Sensitive", 0.2, -0.8, -0.2), # Heavily penalized by volatility
    ("Trend_Sensitive", 0.2, -0.2, -0.8), # Heavily penalized by slope
]

def analyze_construction():
    # 1. Load Baseline PnL (The "Truth")
    # We use Baseline 0 as the representative sample
    out_dir = Path(cr.PATH_OUT)
    base_path = out_dir / "baseline_positions_0.parquet"
    
    if not base_path.exists():
        print("[ERR] Baseline 0 not found. Run Step 1 of optimize_filters_fast.py first.")
        return

    print("[LOAD] Loading Baseline PnL...")
    df_trades = pd.read_parquet(base_path)
    # Daily Net PnL
    pnl = df_trades.groupby(pd.to_datetime(df_trades["close_ts"]).dt.floor("D"))["pnl_net_bp"].sum()
    pnl.name = "next_day_pnl"
    
    results = []

    print(f"\n[TEST] Running Grid Search on Signal Construction ({len(WINDOW_GRID)*len(WEIGHT_CONFIGS)} combos)...")
    
    for win in WINDOW_GRID:
        for w_name, w_mean, w_std, w_slope in WEIGHT_CONFIGS:
            
            # 2. Build Signals on the Fly
            cfg = RegimeConfig(
                base_window=win,
                w_health_mean=w_mean,
                w_health_std=w_std,
                w_health_slope=w_slope
            )
            
            # Force rebuild with new params (in memory)
            # We use the internal builder to get the DF directly
            df_sig = hf.build_hybrid_signals(regime_cfg=cfg, force_rebuild=True)
            
            # 3. Align Data
            # Note: hybrid_filter.py ALREADY shifts signals by 1. 
            # So "decision_ts" T contains data known at T-1.
            # We match decision_ts T with PnL at T.
            df_sig["date"] = pd.to_datetime(df_sig["decision_ts"]).dt.floor("D")
            df_sig = df_sig.drop_duplicates("date").set_index("date")
            
            # Join PnL
            combined = df_sig.join(pnl, how="inner").dropna()
            
            if len(combined) < 50:
                continue

            # 4. Calculate Predictive Power (IC)
            # We want Health to correlate POSITIVELY with PnL
            ic_health, _ = spearmanr(combined["signal_health_z"], combined["next_day_pnl"])
            
            # We want Trendiness to correlate NEGATIVELY with PnL (High Trend = Crash)
            # So we look for a strong NEGATIVE correlation.
            ic_trend, _ = spearmanr(combined["trendiness_abs"], combined["next_day_pnl"])
            
            # We want Cross-Sectional Mean Z (Outlier) to correlate NEGATIVELY (Extreme Val = Reversion Risk? or Momentum?)
            # Usually Extreme Z = Reversion = Profit, BUT Extreme Mean Z = Systemic Shift = Risk.
            ic_xs, _ = spearmanr(combined["z_xs_mean_roll_z"].abs(), combined["next_day_pnl"])
            
            results.append({
                "Window": win,
                "Config": w_name,
                "IC_Health": ic_health,
                "IC_Trend": ic_trend,
                "IC_XS_Mean": ic_xs,
                "Obs": len(combined)
            })
            
            print(f"   Win={win:<2} | {w_name:<15} -> IC_Health: {ic_health:6.3f} | IC_Trend: {ic_trend:6.3f}")

    # 5. Display Best Results
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("WINNER: BEST PREDICTIVE WINDOW")
    print("="*60)
    # We want max IC_Health
    print(res_df.sort_values("IC_Health", ascending=False).head(5))
    
    print("\n" + "="*60)
    print("WINNER: MOST DANGEROUS TREND DETECTOR")
    print("="*60)
    # We want MINIMUM (Most Negative) IC_Trend
    print(res_df.sort_values("IC_Trend", ascending=True).head(5))

if __name__ == "__main__":
    analyze_construction()
