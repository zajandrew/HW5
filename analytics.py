import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# Import config for paths
import cr_config as cr

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================
def assign_bucket_local(tenor):
    # Replicating bucket logic locally for analysis
    buckets = getattr(cr, "BUCKETS", {
        "short": (0.0, 0.99),
        "front": (1.0, 3.0),
        "belly": (3.1, 9.0),
        "long": (10.0, 40.0),
    })
    for name, (lo, hi) in buckets.items():
        if (tenor >= lo) and (tenor <= hi):
            return name
    return "other"

def get_daily_pnl_series(marks_df, positions_df):
    """
    Constructs a clean daily PnL series (Bps) accounting for MTM variance
    and Transaction Costs on the days they occur.
    """
    # 1. Aggregate MTM PnL from Marks (Gross)
    # marks_df contains cumulative PnL per position. 
    # We take the sum of all open positions per day, then diff to get daily change.
    daily_gross = marks_df.groupby("decision_ts")["pnl_bp"].sum().diff().fillna(0.0)
    
    # 2. Aggregate Transaction Costs from Positions (Realized)
    # T-costs happen on 'close_ts'.
    tcosts = positions_df.set_index("close_ts")["tcost_bp"].groupby(level=0).sum()
    # Align to decision timestamps (daily/hourly)
    tcosts = tcosts.reindex(daily_gross.index, fill_value=0.0)
    
    # 3. Net Daily PnL
    daily_net = daily_gross - tcosts
    return daily_net

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
out_dir = Path(cr.PATH_OUT)
suffix = getattr(cr, "OUT_SUFFIX", "")

pos_path = out_dir / f"positions_ledger{suffix}.parquet"
mark_path = out_dir / f"marks_ledger{suffix}.parquet"

print(f"[LOAD] Reading from {out_dir}...")

if not pos_path.exists() or not mark_path.exists():
    print("[ERROR] Parquet files not found. Run portfolio_test.py first.")
else:
    # --- Load Positions ---
    pos_df = pd.read_parquet(pos_path)
    if "mode" in pos_df.columns:
        pos_df = pos_df[pos_df["mode"] == "overlay"].copy()
    pos_df["close_ts"] = pd.to_datetime(pos_df["close_ts"])
    pos_df["open_ts"] = pd.to_datetime(pos_df["open_ts"])
    
    # --- Load Marks ---
    mark_df = pd.read_parquet(mark_path)
    if "mode" in mark_df.columns:
        mark_df = mark_df[mark_df["mode"] == "overlay"].copy()
    mark_df["decision_ts"] = pd.to_datetime(mark_df["decision_ts"])

    # ==============================================================================
    # 3. CALCULATE METRICS
    # ==============================================================================
    
    # --- A. Trade-Level Statistics ---
    pos_df["hold_days"] = (pos_df["close_ts"] - pos_df["open_ts"]).dt.total_seconds() / 86400
    
    n_trades = len(pos_df)
    win_rate = (pos_df["pnl_net_bp"] > 0).mean()
    avg_pnl_bp = pos_df["pnl_net_bp"].mean()
    total_pnl_bp = pos_df["pnl_net_bp"].sum()
    avg_hold = pos_df["hold_days"].mean()
    
    # Profit Factor
    gross_wins = pos_df.loc[pos_df["pnl_net_bp"] > 0, "pnl_net_bp"].sum()
    gross_loss = abs(pos_df.loc[pos_df["pnl_net_bp"] < 0, "pnl_net_bp"].sum())
    profit_factor = gross_wins / gross_loss if gross_loss != 0 else np.inf

    # --- B. Time-Series Statistics (Daily) ---
    daily_pnl = get_daily_pnl_series(mark_df, pos_df)
    cum_pnl = daily_pnl.cumsum()
    
    # Resample to strictly daily if data is hourly, to get annualization right
    daily_pnl_d = daily_pnl.resample('D').sum()
    daily_pnl_d = daily_pnl_d[daily_pnl_d != 0] # Remove empty weekend fillers if any

    # Risk Metrics
    ann_factor = 252
    mean_ret = daily_pnl_d.mean()
    std_ret = daily_pnl_d.std()
    
    sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor) if std_ret != 0 else 0.0
    
    # Sortino (Downside Vol)
    downside = daily_pnl_d[daily_pnl_d < 0]
    std_down = downside.std()
    sortino = (mean_ret / std_down) * np.sqrt(ann_factor) if std_down != 0 else 0.0
    
    # Drawdown
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()
    
    # Calmar (Annualized Return / Max DD) - Approximated total return
    total_days = (daily_pnl.index.max() - daily_pnl.index.min()).days
    if total_days > 0 and max_dd != 0:
        ann_ret_bp = total_pnl_bp * (365.0 / total_days)
        calmar = abs(ann_ret_bp / max_dd)
    else:
        calmar = 0.0

    # --- Print Summary Report ---
    print("\n" + "="*30 + " STRATEGY PERFORMANCE " + "="*30)
    print(f"{'Metric':<25} {'Value':>15}")
    print("-" * 60)
    print(f"{'Total Net PnL (bps)':<25} {total_pnl_bp:,.1f}")
    print(f"{'Sharpe Ratio':<25} {sharpe:.2f}")
    print(f"{'Sortino Ratio':<25} {sortino:.2f}")
    print(f"{'Calmar Ratio':<25} {calmar:.2f}")
    print(f"{'Max Drawdown (bps)':<25} {max_dd:,.1f}")
    print(f"{'Win Rate':<25} {win_rate:.1%}")
    print(f"{'Profit Factor':<25} {profit_factor:.2f}")
    print(f"{'Avg Trade PnL (bps)':<25} {avg_pnl_bp:.2f}")
    print(f"{'Avg Hold Time (days)':<25} {avg_hold:.2f}")
    print(f"{'Total Trades':<25} {n_trades}")
    print("=" * 60 + "\n")

    # ==============================================================================
    # 4. VISUALIZATIONS
    # ==============================================================================
    
    # --- Plot 1: Equity Curve & Drawdown ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(cum_pnl.index, cum_pnl, color='#1f77b4', lw=1.5, label='Cumulative Net PnL (bps)')
    ax1.fill_between(cum_pnl.index, cum_pnl, 0, alpha=0.1, color='#1f77b4')
    ax1.set_title('Strategy Equity Curve (bps)', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_ylabel('Bps')
    
    ax2.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.3, label='Drawdown')
    ax2.plot(drawdown.index, drawdown, color='#d62728', lw=1)
    ax2.set_title('Drawdown Profile', fontsize=14)
    ax2.set_ylabel('Bps')
    ax2.legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Risk Exposure by Bucket (Stacked Area) ---
    # We need to assign buckets to the marks ledger
    print("[PLOT] Generating Risk Buckets...")
    # Explode marks to individual legs to count DV01 contribution
    # Note: PairPos stores w_i, w_j. In Overlay, scale_dv01 is the cash DV01.
    # Total DV01 usage = abs(w_i * scale) + abs(w_j * scale).
    # Since w_i = 1, w_j = -1, Total = 2 * scale. 
    # Let's bucket by the "Trade Tenor" (the alt leg usually drives the view, 
    # but for risk mgmt we care about where we have exposure).
    # Simplification: Just sum the Absolute Cash DV01 per bucket.
    
    # Map buckets
    mark_df['bucket_i'] = mark_df['tenor_i'].apply(assign_bucket_local)
    mark_df['bucket_j'] = mark_df['tenor_j'].apply(assign_bucket_local)
    
    # We split the dv01 cash roughly 50/50 between legs for visualization (approx)
    # or just sum them. Let's do sum of DV01 exposed in that bucket.
    # Since w is usually 1 and -1, both legs carry 'scale_dv01' amount of risk.
    # However, marks_ledger doesn't store scale_dv01 for every row usually, 
    # wait, run_month saves `scale_dv01` in ledger! 
    
    if "scale_dv01" in mark_df.columns:
        # Construct a long-form frame for bucket aggregation
        leg_i = mark_df[['decision_ts', 'bucket_i', 'scale_dv01']].rename(columns={'bucket_i': 'bucket', 'scale_dv01': 'dv01'})
        leg_j = mark_df[['decision_ts', 'bucket_j', 'scale_dv01']].rename(columns={'bucket_j': 'bucket', 'scale_dv01': 'dv01'})
        risk_long = pd.concat([leg_i, leg_j])
        
        # Pivot
        risk_pivot = risk_long.groupby(['decision_ts', 'bucket'])['dv01'].sum().unstack(fill_value=0)
        
        # Sort columns order
        desired_order = ['short', 'front', 'belly', 'long', 'other']
        cols = [c for c in desired_order if c in risk_pivot.columns]
        risk_pivot = risk_pivot[cols]
        
        # Smooth slightly for chart readability
        risk_pivot_smooth = risk_pivot.rolling(window=5, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(14, 6))
        risk_pivot_smooth.plot.area(ax=ax, alpha=0.7, cmap='viridis')
        ax.set_title('Gross DV01 Risk Exposure by Curve Bucket ($)', fontsize=16)
        ax.set_ylabel('Total DV01 ($)')
        plt.tight_layout()
        plt.show()

    # --- Plot 3: Distributions ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(pos_df["pnl_net_bp"], kde=True, ax=ax1, color='teal', bins=30)
    ax1.axvline(0, color='k', linestyle='--')
    ax1.set_title('Distribution of Trade PnL (Net Bps)')
    ax1.set_xlabel('Bps')
    
    sns.histplot(pos_df["hold_days"], kde=True, ax=ax2, color='orange', bins=20)
    ax2.set_title('Distribution of Hold Times')
    ax2.set_xlabel('Days')
    
    plt.tight_layout()
    plt.show()

    # --- Plot 4: Monthly Heatmap ---
    # Group daily PnL by Year/Month
    daily_res = daily_pnl_d.to_frame(name='pnl')
    daily_res['year'] = daily_res.index.year
    daily_res['month'] = daily_res.index.month
    
    monthly_pnl = daily_res.groupby(['year', 'month'])['pnl'].sum().unstack()
    
    plt.figure(figsize=(10, len(monthly_pnl)/2 + 2))
    sns.heatmap(monthly_pnl, annot=True, fmt=".0f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Bps'})
    plt.title('Monthly Strategy PnL (Net Bps)', fontsize=16)
    plt.ylabel('Year')
    plt.xlabel('Month')
    plt.show()
