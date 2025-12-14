"""
analysis_regime_robust.py

The "Wind Tunnel" for your Strategy Engine.
Diagnoses which Regime Factors (Trend, Vol, Health) cause the strategy to crash.

Methodology:
1. Runs the 'Sacrificial' Backtest across ALL synthetic tapes (10+ histories).
2. Aggregates thousands of trades into a Super-Ledger.
3. Bins trades by Regime Factor Quartiles (Q1=Low, Q4=High).
4. Analyzes PRICE PNL primarily to identify "Falling Knife" regimes without Carry bias.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import cr_config as cr
import portfolio_test as pt
import hybrid_filter as hf

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Factors to test from hybrid_signals
REGIME_FACTORS = [
    "scale_mean",           # Noise Level / Volatility
    "scale_roll_z",         # Volatility Spike (Shock)
    "signal_health_z",      # Mean Reversion Quality (High = Good)
    "trendiness_abs",       # Directional Trend Strength (High = Bad?)
    "z_xs_std",             # Cross-Sectional Dispersion (Opportunity Set)
    "z_xs_slope_roll_z"     # Curve Slope Momentum
]

QUARTILE_LABELS = ["Q1_Low", "Q2_MidLow", "Q3_MidHigh", "Q4_High"]

# ==============================================================================
# CORE LOGIC
# ==============================================================================

def run_robust_analysis():
    print(f"\n" + "="*80)
    print(f"{'ROBUST REGIME DIAGNOSTIC (PRICE-FOCUSED)':^80}")
    print(f"="*80)
    print(f"[INFO] Using Engine Settings from 'cr_config.py'")
    print(f"[NOTE] Ensure Z_ENTRY/Z_STOP are LOOSE to catch 'Falling Knives' for analysis.\n")

    # 1. Load Regime Signals
    signals = hf.get_or_build_hybrid_signals()
    # Ensure correct timestamp format for merging (Naive)
    signals["decision_ts"] = pd.to_datetime(signals["decision_ts"], utc=True).dt.tz_convert(None)
    
    # 2. Find All Synthetic Tapes
    tape_files = sorted(Path(".").glob("synth_trades_*.pkl"))
    if not tape_files:
        print("[ERROR] No 'synth_trades_*.pkl' files found. Run synthetic generator first.")
        return
    
    print(f"[INIT] Found {len(tape_files)} synthetic tapes. Generating Super-Ledger...")

    # 3. Generate Trades for Every Tape (The Super-Ledger)
    all_trades = []
    enh_path = Path(cr.PATH_ENH)
    suffix = getattr(cr, "ENH_SUFFIX", "")
    months = [f.stem[:4] for f in enh_path.glob(f"*{suffix}.parquet") if f.stem[:4].isdigit()]
    months = sorted(list(set(months)))

    for i, tape_path in enumerate(tape_files):
        print(f"  > Processing Tape {i+1:02d}/{len(tape_files)}: {tape_path.name:<25} ... ", end="", flush=True)
        
        try:
            # Load Tape
            hedges = pd.read_pickle(tape_path)
            # Fix Timezone
            hedges["tradetimeUTC"] = pd.to_datetime(hedges["tradetimeUTC"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
            
            # Run Engine (Phase 1 Settings)
            # Pass signals so dynamic logic *could* run if enabled, but usually disabled for diagnostics.
            pos, _, _ = pt.run_all(
                months, 
                decision_freq="D", 
                carry=True, 
                force_close_end=True, 
                hedge_df=hedges, 
                regime_signals=signals
            )
            
            if not pos.empty:
                pos["source_tape"] = tape_path.name
                all_trades.append(pos)
                print(f"OK ({len(pos)} trades)")
            else:
                print("No Trades.")
                
        except Exception as e:
            print(f"FAILED: {e}")

    if not all_trades:
        print("[ERROR] No trades generated across any tape.")
        return

    # 4. Aggregate
    super_ledger = pd.concat(all_trades, ignore_index=True)
    
    # Filter for Overlay only
    if "mode" in super_ledger.columns:
        super_ledger = super_ledger[super_ledger["mode"] == "overlay"].copy()
        
    print(f"\n[LOAD] Super-Ledger created with {len(super_ledger)} total trades.")

    # 5. Merge with Signals
    # Map Trade Open Time -> Decision Time
    super_ledger["open_ts"] = pd.to_datetime(super_ledger["open_ts"], utc=True).dt.tz_convert(None)
    
    if cr.DECISION_FREQ == "D":
        super_ledger["decision_ts"] = super_ledger["open_ts"].dt.floor("D")
    else:
        super_ledger["decision_ts"] = super_ledger["open_ts"].dt.floor("H")

    df = pd.merge(super_ledger, signals, on="decision_ts", how="inner")
    
    print(f"[MERGE] {len(df)} trades successfully matched with regime signals.")

    # 6. Factor Analysis
    valid_factors = [f for f in REGIME_FACTORS if f in df.columns]
    
    summary_data = []

    for factor in valid_factors:
        print(f"\n" + "-"*80)
        print(f"FACTOR ANALYSIS: {factor}")
        print("-"*80)
        
        try:
            # Binning across the entire Super-Ledger
            df["bin"] = pd.qcut(df[factor], 4, labels=QUARTILE_LABELS, duplicates='drop')
        except ValueError:
            print(f"  [SKIP] Not enough unique values to bin {factor}.")
            continue
            
        # Detailed Aggregation
        # We focus on PRICE PnL to isolate the Mean Reversion signal from the Carry noise.
        stats = df.groupby("bin", observed=False).agg(
            Count=("pnl_net_bp", "count"),
            
            # --- The Truth (Price) ---
            Price_Mean=("pnl_price_bp", "mean"),
            Price_Med=("pnl_price_bp", "median"),
            Price_WinRate=("pnl_price_bp", lambda x: (x > 0).mean()),
            
            # --- The Buffer (Carry/Roll) ---
            Carry_Mean=("pnl_carry_bp", "mean"),
            Roll_Mean=("pnl_roll_bp", "mean"),
            
            # --- The Bottom Line (Total) ---
            Total_Mean=("pnl_net_bp", "mean"),
            Total_WinRate=("pnl_net_bp", lambda x: (x > 0).mean()),
            
            # --- Robustness ---
            # Consistency: What % of tapes agreed with the Price Direction of this bin?
            Tape_Agree=("pnl_price_bp", lambda x: _check_consistency(df.loc[x.index]))
        )
        
        # Calculate Spread (Q4 - Q1) for Price PnL
        if len(stats) == 4:
            q1_price = stats.loc["Q1_Low", "Price_Mean"]
            q4_price = stats.loc["Q4_High", "Price_Mean"]
            price_spread = q4_price - q1_price
            
            summary_data.append({
                "Factor": factor, 
                "Price_Spread_Q4-Q1": price_spread,
                "Q1_Price_PnL": q1_price,
                "Q4_Price_PnL": q4_price,
                "Q4_Price_WinRate": stats.loc["Q4_High", "Price_WinRate"],
                "Total_PnL_Spread": stats.loc["Q4_High", "Total_Mean"] - stats.loc["Q1_Low", "Total_Mean"]
            })
        
        # Format for display
        pd.options.display.float_format = '{:,.2f}'.format
        print(stats.to_string())

    # 7. Impact Summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data).sort_values("Price_Spread_Q4-Q1", ascending=True)
        
        print(f"\n" + "="*80)
        print(f"{'IMPACT SUMMARY (Ranked by PRICE PnL Damage)':^80}")
        print(f"="*80)
        print("INTERPRETATION:")
        print(" * LARGE NEGATIVE SPREAD: Q4 (High Value) Kills Price PnL. -> Defensive Weight (SENS > 0).")
        print(" * LARGE POSITIVE SPREAD: Q4 (High Value) Boosts Price PnL. -> Offensive Weight (SENS < 0).")
        print("-" * 80)
        
        cols = ["Factor", "Price_Spread_Q4-Q1", "Q1_Price_PnL", "Q4_Price_PnL", "Q4_Price_WinRate"]
        print(summary_df[cols].to_string(index=False))

def _check_consistency(sub_df):
    """
    Returns the fraction of unique tapes that had >0 Price PnL in this subset.
    This prevents 'One Lucky Tape' from skewing the stats.
    """
    if sub_df.empty: return 0.0
    # Group by Source Tape and calc mean Price PnL for this bin
    tape_performance = sub_df.groupby("source_tape")["pnl_price_bp"].mean()
    positive_tapes = (tape_performance > 0).sum()
    total_tapes = len(tape_performance)
    return positive_tapes / total_tapes if total_tapes > 0 else 0.0

if __name__ == "__main__":
    run_robust_analysis()
