"""
analysis_regime_robust.py

The "Wind Tunnel" for your Strategy Engine.
Diagnoses which Regime Factors (Trend, Vol, Health) cause the strategy to crash.

Methodology:
1. Runs the 'Sacrificial' Backtest across ALL synthetic tapes (10+ histories).
2. Aggregates thousands of trades into a Super-Ledger.
3. Bins trades by Regime Factor Quartiles (Q1=Low, Q4=High).
4. Analyzes PRICE PNL primarily to identify "Falling Knife" regimes without Carry bias.
5. Outputs detailed Ranges, PnL, and Consistency for EVERY Quartile.
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
    print(f"\n" + "="*120)
    print(f"{'ROBUST REGIME DIAGNOSTIC (PRICE-FOCUSED)':^120}")
    print(f"="*120)
    print(f"[INFO] Using Engine Settings from 'cr_config.py'")
    print(f"[NOTE] Ensure Z_ENTRY/Z_STOP are LOOSE to catch 'Falling Knives' for analysis.\n")

    # 1. Load Regime Signals
    signals = hf.get_or_build_hybrid_signals()
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
            hedges["tradetimeUTC"] = pd.to_datetime(hedges["tradetimeUTC"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
            
            # Run Engine
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
    
    if "mode" in super_ledger.columns:
        super_ledger = super_ledger[super_ledger["mode"].isin(["overlay", "fly"])].copy()
        
    print(f"\n[LOAD] Super-Ledger created with {len(super_ledger)} total trades.")

    # 5. Merge with Signals
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
        print(f"\n" + "-"*120)
        print(f"FACTOR ANALYSIS: {factor}")
        print("-"*120)
        
        try:
            df["bin"] = pd.qcut(df[factor], 4, labels=QUARTILE_LABELS, duplicates='drop')
        except ValueError:
            print(f"  [SKIP] Not enough unique values to bin {factor}.")
            continue
            
        stats = df.groupby("bin", observed=False).agg(
            Min_Val=(factor, "min"),
            Max_Val=(factor, "max"),
            Count=("pnl_net_bp", "count"),
            Price_Mean=("pnl_price_bp", "mean"),
            Price_WinRate=("pnl_price_bp", lambda x: (x > 0).mean()),
            Total_Mean=("pnl_net_bp", "mean"),
            Tape_Agree=("pnl_price_bp", lambda x: _check_consistency(df.loc[x.index]))
        )
        
        # Calculate Metrics for Summary Table
        if len(stats) >= 2: 
            try:
                # Calc Spread (High - Low)
                q1_idx = stats.index[0]
                q4_idx = stats.index[-1]
                q1_price = stats.loc[q1_idx, "Price_Mean"]
                q4_price = stats.loc[q4_idx, "Price_Mean"]
                
                entry = {"Factor": factor, "Spread_Q4-Q1": q4_price - q1_price}

                # Extract Per-Quartile Metrics
                for q_label in QUARTILE_LABELS:
                    if q_label in stats.index:
                        mn = stats.loc[q_label, "Min_Val"]
                        mx = stats.loc[q_label, "Max_Val"]
                        pnl = stats.loc[q_label, "Price_Mean"]
                        agree = stats.loc[q_label, "Tape_Agree"]
                        
                        # Short labels for column width
                        short_q = q_label.split("_")[0] # "Q1", "Q2"
                        entry[f"{short_q}_Range"] = f"{mn:.2f}->{mx:.2f}"
                        entry[f"{short_q}_PnL"] = f"{pnl:+.1f}"
                        entry[f"{short_q}_Agr"] = f"{agree:.0%}"
                    else:
                        short_q = q_label.split("_")[0]
                        entry[f"{short_q}_Range"] = "-"
                        entry[f"{short_q}_PnL"] = "-"
                        entry[f"{short_q}_Agr"] = "-"

                summary_data.append(entry)
            except Exception as e: 
                print(f"[WARN] Could not summarize {factor}: {e}")
        
        # Display Intermediate Table
        pd.options.display.float_format = '{:,.2f}'.format
        cols = ["Min_Val", "Max_Val", "Count", "Price_Mean", "Price_WinRate", "Tape_Agree"]
        print(stats[cols].to_string())

    # 7. Impact Summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data).sort_values("Spread_Q4-Q1", ascending=True)
        
        print(f"\n" + "="*160)
        print(f"{'FULL IMPACT SUMMARY (Price PnL & Consistency)':^160}")
        print(f"="*160)
        
        # Build Column List dynamically
        cols = ["Factor", "Spread_Q4-Q1"]
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            cols.extend([f"{q}_Range", f"{q}_PnL", f"{q}_Agr"])
            
        # Rename for compact printing
        rename_map = {c: c.replace("_Range", " Range").replace("_PnL", " PnL").replace("_Agr", " %") for c in cols}
        summary_df = summary_df.rename(columns=rename_map)
        final_cols = list(rename_map.values())
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')

        print(summary_df[final_cols].to_string(index=False))
        print("-" * 160)

def _check_consistency(sub_df):
    """
    Returns the fraction of unique tapes that had >0 Price PnL in this subset.
    """
    if sub_df.empty: return 0.0
    tape_performance = sub_df.groupby("source_tape")["pnl_price_bp"].mean()
    positive_tapes = (tape_performance > 0).sum()
    total_tapes = len(tape_performance)
    return positive_tapes / total_tapes if total_tapes > 0 else 0.0

if __name__ == "__main__":
    run_robust_analysis()
