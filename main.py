import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from math import sqrt
import sys

# Import config
import cr_config as cr

# ==============================================================================
# 0. MODE SELECTION & CONFIGURATION
# ==============================================================================
# Detect mode from command line argument
REPORT_MODE = "BPS"  # Default
if len(sys.argv) > 1 and sys.argv[1].lower() == "cash":
    REPORT_MODE = "CASH"

print(f"[INIT] Running Report in {REPORT_MODE} Mode...")

# Configuration Mapping
if REPORT_MODE == "CASH":
    cfg = {
        'col_net': 'pnl_net_cash',
        'col_price': 'pnl_price_cash',
        'col_carry': 'pnl_carry_cash',
        'col_roll': 'pnl_roll_cash',
        'col_tcost': 'tcost_cash',
        'unit': '$',
        'label': 'USD',
        'fmt': '${:,.0f}'  # e.g., $1,500
    }
else:
    cfg = {
        'col_net': 'pnl_net_bp',
        'col_price': 'pnl_price_bp',
        'col_carry': 'pnl_carry_bp',
        'col_roll': 'pnl_roll_bp',
        'col_tcost': 'tcost_bp',
        'unit': 'bps',
        'label': 'Basis Points',
        'fmt': '{:,.1f}'   # e.g., 12.5
    }

# ==============================================================================
# STYLE & HELPERS
# ==============================================================================
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
colors = {'pnl': '#2ecc71', 'dd': '#e74c3c', 'price': '#3498db', 'carry': '#f1c40f', 'roll': '#9b59b6'}

def safe_divide(n, d, default=0.0):
    return n / d if d != 0 else default

def fmt_val(val):
    return cfg['fmt'].format(val)

def human_format(x, pos):
    """Formats 1500 as 1.5k, 1000000 as 1M"""
    if x == 0: return '0'
    abs_x = abs(x)
    if abs_x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif abs_x >= 1e3:
        return f'{x*1e-3:.0f}k'
    else:
        return f'{x:.0f}'

def assign_bucket_simple(tenor):
    """Re-derives buckets for analysis if not present"""
    if pd.isna(tenor): return "Unknown"
    if tenor < 2.0: return "Short (<2Y)"
    if tenor < 5.0: return "Front (2-5Y)"
    if tenor < 10.0: return "Belly (5-10Y)"
    return "Long (10Y+)"

# ==============================================================================
# 1. DATA LOADING & PREP
# ==============================================================================
out_dir = Path(cr.PATH_OUT)
suffix = getattr(cr, "OUT_SUFFIX", "")
pos_path = out_dir / f"positions_ledger{suffix}.parquet"

if not pos_path.exists():
    raise FileNotFoundError(f"[ERROR] {pos_path} not found.")

print(f"[LOAD] Reading {pos_path}...")
df = pd.read_parquet(pos_path)

# Filter Overlay only
if "mode" in df.columns:
    df = df[df["mode"] == "overlay"].copy()

# Sort by Close Time (Critical for Equity Curve)
df["close_ts"] = pd.to_datetime(df["close_ts"])
df = df.sort_values("close_ts").reset_index(drop=True)

# Derive Bucket if missing (Assuming 'tenor_i' exists)
if "bucket" not in df.columns and "tenor_i" in df.columns:
    df["bucket"] = df["tenor_i"].apply(assign_bucket_simple)

# --- Create Equity Curves using the Selected Column ---
df["equity_curve"] = df[cfg['col_net']].cumsum()

# ==============================================================================
# 2. METRICS ENGINE
# ==============================================================================

# --- A. Trade Statistics ---
total_trades = len(df)
win_trades = df[df[cfg['col_net']] > 0]
loss_trades = df[df[cfg['col_net']] <= 0]

win_rate = safe_divide(len(win_trades), total_trades)
gross_win = win_trades[cfg['col_net']].sum()
gross_loss = abs(loss_trades[cfg['col_net']].sum())
profit_factor = safe_divide(gross_win, gross_loss)

avg_trade = df[cfg['col_net']].mean()
avg_win = win_trades[cfg['col_net']].mean()
avg_loss = loss_trades[cfg['col_net']].mean()

# Hold Times
if "days_held_equiv" in df.columns:
    avg_hold = df["days_held_equiv"].mean()
else:
    avg_hold = (df["close_ts"] - pd.to_datetime(df["open_ts"])).dt.days.mean()

# --- B. Time-Series Statistics (Sharpe/Sortino) ---
# Resample to Daily for correct Sharpe math
daily_idx = pd.date_range(start=df["close_ts"].min(), end=df["close_ts"].max(), freq='D')
daily_pnl = df.set_index("close_ts")[cfg['col_net']].resample('D').sum().reindex(daily_idx, fill_value=0.0)

ANN_FACTOR = 252
mean_daily = daily_pnl.mean()
std_daily = daily_pnl.std()
downside_daily = daily_pnl[daily_pnl < 0].std()

sharpe = safe_divide(mean_daily * ANN_FACTOR, std_daily * sqrt(ANN_FACTOR))
sortino = safe_divide(mean_daily * ANN_FACTOR, downside_daily * sqrt(ANN_FACTOR))

cum_equity_daily = daily_pnl.cumsum()
running_max = cum_equity_daily.cummax()
dd_series = cum_equity_daily - running_max
max_dd = dd_series.min()
calmar = safe_divide(cum_equity_daily.iloc[-1], abs(max_dd))

# --- C. Attribution ---
total_net = df[cfg['col_net']].sum()
total_price = df[cfg['col_price']].sum()
total_carry = df[cfg['col_carry']].sum()
total_roll = df[cfg['col_roll']].sum()

# ==============================================================================
# 3. PRINT PROFESSIONAL TABLE
# ==============================================================================
print("\n" + "="*60)
print(f"{f'SYSTEMATIC OVERLAY REPORT ({REPORT_MODE})':^60}")
print("="*60)

stats = [
    (f"Total Net PnL ({cfg['unit']})", fmt_val(total_net)),
    ("Total Trades", f"{total_trades}"),
    ("Win Rate", f"{win_rate:.1%}"),
    ("-" * 20, "-" * 20),
    ("Profit Factor", f"{profit_factor:.2f}"),
    (f"Avg Trade ({cfg['unit']})", fmt_val(avg_trade)),
    ("Avg Win / Avg Loss", f"{abs(avg_win/avg_loss):.2f}"),
    ("Avg Hold (Days)", f"{avg_hold:.1f}"),
    ("-" * 20, "-" * 20),
    ("Sharpe Ratio (Ann.)", f"{sharpe:.2f}"),
    ("Sortino Ratio (Ann.)", f"{sortino:.2f}"),
    (f"Max Drawdown ({cfg['unit']})", fmt_val(max_dd)),
    ("Return / DD (Calmar)", f"{abs(calmar):.2f}"),
]

col_width = 35
for label, val in stats:
    if "---" in label:
        print(f"{label}   {val}")
    else:
        print(f"{label:<{col_width}} {val:>15}")
print("="*60)
print(f"ATTRIBUTION:\n Price: {fmt_val(total_price)} | Carry: {fmt_val(total_carry)} | Roll: {fmt_val(total_roll)}")
print("="*60 + "\n")

# ==============================================================================
# 4. PLOTTING SUITE
# ==============================================================================

# --- FIGURE 1: EXECUTIVE DASHBOARD ---
fig = plt.figure(figsize=(16, 12), constrained_layout=True)
gs = fig.add_gridspec(3, 2)

# 1. Equity Curve
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum_equity_daily.index, cum_equity_daily.values, color=colors['pnl'], lw=2)
ax1.fill_between(cum_equity_daily.index, cum_equity_daily.values, 0, color=colors['pnl'], alpha=0.1)
ax1.set_title(f"Realized Equity Curve (Net {cfg['label']})", fontweight='bold')
ax1.set_ylabel(f"Cumulative {cfg['label']}")
ax1.margins(x=0)
if REPORT_MODE == "CASH":
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(human_format))

# 2. Drawdown
ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
ax2.fill_between(dd_series.index, dd_series.values, 0, color=colors['dd'], alpha=0.3)
ax2.plot(dd_series.index, dd_series.values, color=colors['dd'], lw=1)
ax2.set_title("Drawdown Profile", fontsize=11)
ax2.set_ylabel(f"Drawdown ({cfg['unit']})")
if REPORT_MODE == "CASH":
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(human_format))

# 3. Monthly Returns (Bar)
ax3 = fig.add_subplot(gs[2, 0])
monthly = df.set_index("close_ts")[cfg['col_net']].resample('M').sum()
clrs = ['#e74c3c' if x < 0 else '#2ecc71' for x in monthly.values]
# Only show last 12 months if too many
plot_monthly = monthly.iloc[-24:] if len(monthly) > 24 else monthly
plot_monthly.index = plot_monthly.index.strftime('%Y-%m')
plot_monthly.plot(kind='bar', ax=ax3, color=clrs[-len(plot_monthly):], width=0.8)
ax3.set_title(f"Monthly Net PnL ({cfg['unit']})")
ax3.set_xlabel("")
ax3.tick_params(axis='x', rotation=45, labelsize=9)

# 4. PnL Distribution (Histogram) - FIXED LABELS
ax4 = fig.add_subplot(gs[2, 1])
sns.histplot(df[cfg['col_net']], kde=True, ax=ax4, color='#34495e', bins=30)
ax4.axvline(0, color='black', linestyle='--')
ax4.axvline(avg_trade, color=colors['pnl'], linestyle='-', label=f'Mean: {human_format(avg_trade,0)}')
ax4.set_title(f"Trade Distribution ({cfg['unit']})")
ax4.set_xlabel(f"PnL Net {cfg['label']}") # Explicit Label
# Use k formatter
ax4.xaxis.set_major_formatter(mtick.FuncFormatter(human_format))
ax4.legend()

plt.show()

# --- FIGURE 2: ADVANCED ANALYTICS (New Requests) ---
fig2 = plt.figure(figsize=(16, 10), constrained_layout=True)
gs2 = fig2.add_gridspec(2, 2)

# 1. PnL Attribution (Drift vs Price)
ax_attr = fig2.add_subplot(gs2[0, 0])
cum_price = df[cfg['col_price']].cumsum()
cum_drift = (df[cfg['col_carry']] + df[cfg['col_roll']]).cumsum()
ax_attr.plot(df["close_ts"], cum_price, label="Price (Timing)", color=colors['price'], alpha=0.8)
ax_attr.plot(df["close_ts"], cum_drift, label="Drift (Carry+Roll)", color=colors['carry'], lw=2)
ax_attr.plot(df["close_ts"], df["equity_curve"], label="Total Net", color='black', linestyle='--')
ax_attr.set_title("PnL Source Attribution", fontweight='bold')
ax_attr.legend()
if REPORT_MODE == "CASH":
    ax_attr.yaxis.set_major_formatter(mtick.FuncFormatter(human_format))

# 2. PnL by Curve Sector (Tenor Bucket)
ax_buck = fig2.add_subplot(gs2[0, 1])
if "bucket" in df.columns:
    bucket_perf = df.groupby("bucket")[cfg['col_net']].sum().sort_values()
    bucket_perf.plot(kind='barh', ax=ax_buck, color='#2980b9')
    ax_buck.set_title(f"Total PnL by Curve Sector ({cfg['unit']})")
    ax_buck.set_xlabel(cfg['unit'])
    if REPORT_MODE == "CASH":
        ax_buck.xaxis.set_major_formatter(mtick.FuncFormatter(human_format))

# 3. Monthly Heatmap (The 'Calendar' View)
ax_heat = fig2.add_subplot(gs2[1, 0])
heat_data = daily_pnl.resample('M').sum()
heat_df = pd.DataFrame({'year': heat_data.index.year, 'month': heat_data.index.month, 'pnl': heat_data.values})
heat_piv = heat_df.pivot(index='year', columns='month', values='pnl')

# --- FIX START: Custom k-formatting for Cash ---
if REPORT_MODE == "CASH":
    # Create a DataFrame of strings like "15k", "-2k"
    annot_labels = heat_piv.applymap(lambda x: f'{x/1000:.0f}k' if pd.notnull(x) else '')
    fmt_param = ""  # Formatting is already done in the strings
else:
    # Standard behavior for Bps
    annot_labels = True 
    fmt_param = ".0f"
# -----------------------------------------------

sns.heatmap(heat_piv, ax=ax_heat, cmap="RdYlGn", center=0, annot=annot_labels, fmt=fmt_param, cbar=False)
ax_heat.set_title(f"Monthly Returns Heatmap ({cfg['unit']})")


# 4. Rolling Sharpe Ratio (6-Month Lookback)
ax_roll = fig2.add_subplot(gs2[1, 1])
roll_window = 126 # ~6 months
rolling_sharpe = daily_pnl.rolling(roll_window).mean() / daily_pnl.rolling(roll_window).std() * sqrt(252)
ax_roll.plot(rolling_sharpe.index, rolling_sharpe, color='#8e44ad')
ax_roll.axhline(0, color='black', lw=0.5)
ax_roll.axhline(1, color='gray', linestyle='--', alpha=0.5, label="Target > 1.0")
ax_roll.set_title(f"Rolling {roll_window}D Sharpe Ratio")
ax_roll.legend()

plt.show()


