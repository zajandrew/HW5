import pandas as pd
import numpy as np
from pathlib import Path
import cr_config as cr
import portfolio_test as pt
import hybrid_filter as hf

# ==============================================================================
# CONFIGURATION
# ==============================================================================
REGIME_FACTORS = [
    "scale_mean",           # Noise Level
    "scale_roll_z",         # Volatility Spike?
    "signal_health_z",      # Mean Reversion Quality (High = Good)
    "trendiness_abs",       # Directional Trend Strength (High = Bad)
    "z_xs_std",             # Cross-Sectional Dispersion
    "z_xs_slope_roll_z"     # Curve Slope Momentum
]

QUARTILE_LABELS = ["Q1_Low", "Q2_MidLow", "Q3_MidHigh", "Q4_High"]

def run_robust_analysis():
    print(f"\n[INIT] Starting ROBUST Regime Meta-Analysis...")
    print(f"[INFO] Using Engine Settings from 'cr_config.py' (Make sure Phase 1 winners are set!)")

    # 1. Load Regime Signals
    signals = hf.get_or_build_hybrid_signals()
    # Ensure correct timestamp format for merging
    signals["decision_ts"] = pd.to_datetime(signals["decision_ts"], utc=True).dt.tz_convert(None)
    
    # 2. Find All Synthetic Tapes
    tape_files = sorted(Path(".").glob("synth_trades_*.pkl"))
    if not tape_files:
        print("[ERROR] No 'synth_trades_*.pkl' files found. Run synthetic generator first.")
        return
    
    print(f"[INFO] Found {len(tape_files)} synthetic tapes. Generating Super-Ledger...")

    # 3. Generate Trades for Every Tape (The Super-Ledger)
    all_trades = []
    
    # We need the list of months from the enhanced files to run the backtest
    enh_path = Path(cr.PATH_ENH)
    suffix = getattr(cr, "ENH_SUFFIX", "")
    months = [f.stem[:4] for f in enh_path.glob(f"*{suffix}.parquet") if f.stem[:4].isdigit()]
    months = sorted(list(set(months)))

    for i, tape_path in enumerate(tape_files):
        print(f"  > Processing Tape {i+1}/{len(tape_files)}: {tape_path.name} ... ", end="", flush=True)
        
        try:
            # Load Tape
            hedges = pd.read_pickle(tape_path)
            # Fix Timezone
            hedges["tradetimeUTC"] = pd.to_datetime(hedges["tradetimeUTC"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
            
            # Run Engine (Phase 1 Settings)
            # We pass the signals so the engine CAN use them if Dynamic is ON, 
            # but usually we run this with DYN_THRESH_ENABLE = False to find the baseline flaws.
            pos, _, _ = pt.run_all(
                months, 
                decision_freq="D", 
                carry=True, 
                force_close_end=True, 
                hedge_df=hedges, 
                regime_signals=signals
            )
            
            if not pos.empty:
                # Tag the source tape for reference
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
    print(f"\n[LOAD] Super-Ledger created with {len(super_ledger)} total trades.")

    # Filter for Overlay only
    if "mode" in super_ledger.columns:
        super_ledger = super_ledger[super_ledger["mode"] == "overlay"].copy()

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
    
    print("\n" + "="*70)
    print(f"{'ROBUST FACTOR ANALYSIS (Aggregated Across Tapes)':^70}")
    print("="*70)
    
    summary_data = []

    for factor in valid_factors:
        print(f"\n>>> ANALYZING: {factor}")
        
        try:
            # Binning across the entire Super-Ledger
            df["bin"] = pd.qcut(df[factor], 4, labels=QUARTILE_LABELS, duplicates='drop')
        except ValueError:
            print(f"  [SKIP] Not enough unique values to bin {factor}.")
            continue
            
            stats = df.groupby("bin", observed=False).agg(
            Count=("pnl_net_bp", "count"),
            
            # The Bottom Line
            Win_Rate=("pnl_net_bp", lambda x: (x > 0).mean()),
            Avg_Total_BP=("pnl_net_bp", "mean"),
            
            # The Diagnostic Split (CRITICAL)
            Avg_Price_BP=("pnl_price_bp", "mean"),  # <-- The Mean Reversion Score
            Avg_Carry_BP=("pnl_carry_bp", "mean"),  # <-- The Buffer
            
            Tape_Consistency=("pnl_net_bp", lambda x: _check_consistency(df.loc[x.index]))
        )

        
        # Calculate Spread (Q4 - Q1)
        if len(stats) == 4:
            q1_perf = stats.loc["Q1_Low", "Avg_PnL_BP"]
            q4_perf = stats.loc["Q4_High", "Avg_PnL_BP"]
            spread = q4_perf - q1_perf
            summary_data.append({
                "Factor": factor, 
                "Q4-Q1_Spread_BP": spread,
                "Q1_WinRate": stats.loc["Q1_Low", "Win_Rate"],
                "Q4_WinRate": stats.loc["Q4_High", "Win_Rate"]
            })
        
        print(stats.round(4).to_string())
        print("-" * 70)

    # 7. Summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data).sort_values("Q4-Q1_Spread_BP", ascending=False)
        print("\n" + "="*70)
        print(f"{'IMPACT SUMMARY':^70}")
        print("="*70)
        print(summary_df.to_string(index=False))
        print("\n[GUIDE]")
        print(" * Positive Spread: Q4 is Better -> Offensive Weight (SENS < 0).")
        print(" * Negative Spread: Q4 is Worse -> Defensive Weight (SENS > 0).")
        print(" * Tape_Consistency: Fraction of tapes (0.0-1.0) that agreed with the bin's profitability.")

def _check_consistency(sub_df):
    """Returns the fraction of unique tapes that had >0 PnL in this subset."""
    if sub_df.empty: return 0.0
    tape_performance = sub_df.groupby("source_tape")["pnl_net_bp"].mean()
    positive_tapes = (tape_performance > 0).sum()
    total_tapes = len(tape_performance)
    return positive_tapes / total_tapes if total_tapes > 0 else 0.0

if __name__ == "__main__":
    run_robust_analysis()
