from pathlib import Path

PATH_OUT = Path(cr.PATH_OUT)

# Example: use a different OUT_SUFFIX for overlay vs pure strategy if you like
suffix_overlay = cr.OUT_SUFFIX  # or "_overlay_d" if you set it that way

pos_overlay = pd.read_parquet(PATH_OUT / f"positions_ledger{suffix_overlay}.parquet")
diag_overlay = pd.read_parquet(PATH_OUT / f"overlay_diag{suffix_overlay}.parquet")

summarize_overlay_backtest(pos_overlay, diag_overlay, label="overlay D")



import pandas as pd
import overlay_diag as od

hedge_df = pd.read_parquet("my_trade_tape.parquet")
months = ["2304", "2305"]

diag = od.analyze_overlay(months, hedge_df)

# See why things are failing:
diag["reason"].value_counts()

# Look at failed ones by category:
diag[diag["reason"] == "no_exec_tenor"].head()
diag[diag["reason"] == "no_zdisp_ge_entry"].head()
diag[diag["reason"] == "fly_block"].head()
diag[diag["reason"] == "caps_block"].head()

# Check that any would have opened:
diag[diag["reason"] == "opened"].head()






ledger["dv01_leg_i"] = np.abs(ledger["w_i"]) * ledger["tenor_i"] / (1 + 0.01*ledger["rate_i"])
ledger["dv01_leg_j"] = np.abs(ledger["w_j"]) * ledger["tenor_j"] / (1 + 0.01*ledger["rate_j"])
ledger["pair_dv01"]  = ledger["dv01_leg_i"] + ledger["dv01_leg_j"]

# average and peak concurrent exposure
avg_dv01 = ledger.groupby("decision_ts")["pair_dv01"].sum().mean()
max_dv01 = ledger.groupby("decision_ts")["pair_dv01"].sum().max()
print(f"Average concurrent DV01 units: {avg_dv01:.2f}, peak: {max_dv01:.2f}")

# === RV analytics (drop-in) ==============================================
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import cr_config as cr

PATH_OUT = Path(cr.PATH_OUT)
SFX = (getattr(cr, "ENH_SUFFIX", "") or "").lower()

files = {
    "positions": PATH_OUT / f"positions_ledger{SFX}.parquet",
    "marks":     PATH_OUT / f"marks_ledger{SFX}.parquet",
    "pnl":       PATH_OUT / f"pnl_by_bucket{SFX}.parquet",
}

dfs = {}
for k, p in files.items():
    if p.exists():
        dfs[k] = pd.read_parquet(p)
        print(f"[OK] {k}: {len(dfs[k]):,} rows ({p.name})")
    else:
        print(f"[MISS] {k}: {p.name}")

def _assign_bucket(tenor):
    for name, (lo, hi) in cr.BUCKETS.items():
        if (tenor >= lo) and (tenor <= hi): return name
    return "other"

def _safe_div(a, b):
    try:
        return float(a) / float(b) if (b not in (0, None) and pd.notna(b) and float(b)!=0) else np.nan
    except Exception:
        return np.nan

# -------------------- Position-level stats --------------------
pos = dfs.get("positions", pd.DataFrame()).copy()
marks = dfs.get("marks", pd.DataFrame()).copy()
by   = dfs.get("pnl", pd.DataFrame()).copy()

if not pos.empty:
    # Buckets for each leg
    pos["bucket_i"] = pos["tenor_i"].astype(float).map(_assign_bucket)
    pos["bucket_j"] = pos["tenor_j"].astype(float).map(_assign_bucket)
    pos["win"] = pos["pnl"] > 0
    pos["days_held_equiv"] = pd.to_numeric(pos["days_held_equiv"], errors="coerce")

    # Exit reason distribution
    exit_ct = pos["exit_reason"].value_counts().rename_axis("exit_reason").to_frame("count")

    # Core stats
    n_trades = len(pos)
    win_rate = _safe_div(pos["win"].sum(), n_trades)
    avg_pnl  = pos["pnl"].mean()
    med_pnl  = pos["pnl"].median()
    p90_pnl  = pos["pnl"].quantile(0.90)
    hold_mean = pos["days_held_equiv"].mean()
    hold_p90  = pos["days_held_equiv"].quantile(0.90)

    # By bucket contribution (using leg i as proxy; you can do both legs if preferred)
    by_bucket = pos.groupby("bucket_i")["pnl"].sum().rename("cum_pnl").sort_values(ascending=False).to_frame()

# -------------------- Time-series PnL & risk --------------------
daily = pd.DataFrame()
if not by.empty:
    # Normalize to calendar day for a clean equity curve (works for D and H)
    daily = by.copy()
    daily["date"] = pd.to_datetime(daily["bucket"]).dt.floor("D")
    daily = (daily.groupby("date")["pnl"].sum()
                   .rename("daily_pnl").to_frame().reset_index())
    daily = daily.sort_values("date")
    daily["cum_pnl"] = daily["daily_pnl"].cumsum()

    # Max drawdown on the daily curve
    eq = daily["cum_pnl"].astype(float)
    roll_max = eq.cummax()
    dd = roll_max - eq
    max_dd = float(dd.max())
    dd_end_idx = int(dd.idxmax()) if len(dd) else 0
    dd_end_date = daily.iloc[dd_end_idx]["date"] if len(daily) else None

    # Simple Sharpe-like on (bucket) PnL
    pnl_mean = by["pnl"].astype(float).mean()
    pnl_std  = by["pnl"].astype(float).std(ddof=1)
    sharpe_like = _safe_div(pnl_mean, pnl_std)

# -------------------- Open vs Closed sanity (position-level) --------------------
open_now_ct = np.nan
if not marks.empty:
    key = ["open_ts", "tenor_i", "tenor_j"]
    last_mark = (marks.loc[marks["event"]=="mark"]
                      .sort_values(key + ["decision_ts"])
                      .groupby(key, as_index=False)
                      .tail(1))
    open_now_ct = int((last_mark["closed"]==False).sum())

# -------------------- Display --------------------
display(HTML("<h3>Summary</h3>"))
summary_rows = []

if not pos.empty:
    summary_rows += [
        ("Trades (closed)", n_trades),
        ("Win rate", f"{100.0*win_rate:.1f}%"),
        ("Avg PnL / trade", f"{avg_pnl:,.2f}"),
        ("Median PnL / trade", f"{med_pnl:,.2f}"),
        ("90th pct PnL / trade", f"{p90_pnl:,.2f}"),
        ("Mean hold (days eq.)", f"{hold_mean:.2f}"),
        ("P90 hold (days eq.)", f"{hold_p90:.2f}"),
    ]
if not by.empty:
    summary_rows += [
        ("Cum PnL", f"{float(daily['cum_pnl'].iloc[-1]):,.2f}" if len(daily) else "—"),
        ("Max drawdown", f"{max_dd:,.2f}"),
        ("Sharpe-like (mean/std, bucket PnL)", f"{sharpe_like:.3f}"),
    ]
if not marks.empty:
    summary_rows += [("Open positions at end", open_now_ct)]

display(pd.DataFrame(summary_rows, columns=["metric", "value"]))

if not pos.empty:
    display(HTML("<h4>Exit reason counts</h4>"))
    display(exit_ct)
    display(HTML("<h4>PnL by bucket (leg i proxy)</h4>"))
    display(by_bucket)

# -------------------- Plots --------------------
if not daily.empty:
    plt.figure()
    plt.plot(daily["date"], daily["cum_pnl"])
    plt.title("Cumulative PnL")
    plt.xlabel("Date"); plt.ylabel("Cum PnL")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure()
    plt.plot(daily["date"], daily["daily_pnl"])
    plt.title("Daily PnL")
    plt.xlabel("Date"); plt.ylabel("PnL")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Drawdown curve (optional)
    plt.figure()
    plt.plot(daily["date"], (daily["cum_pnl"].cummax() - daily["cum_pnl"]))
    plt.title("Drawdown")
    plt.xlabel("Date"); plt.ylabel("DD")
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("No time-series PnL loaded; skipping plots.")
# =======================================================================

# Jupyter one-off: add "_D" suffix to enhanced files and sweeper outputs

from pathlib import Path
import re

import cr_config as cr  # uses your configured PATH_ENH / PATH_OUT

# -------- Settings --------
SUFFIX = "_D"
DRY_RUN = True   # set to False to actually rename

# Sweeper artifacts to rename (prefixes without suffix)
# Add/remove prefixes if you use different names
SWEEPER_PREFIXES = ["sweep_results"]   # will catch parquet/csv + *_best.json

# --------------------------

enh_dir = Path(getattr(cr, "PATH_ENH", "."))
out_dir = Path(getattr(cr, "PATH_OUT", "."))

def add_suffix_before_ext(p: Path, suffix: str) -> Path:
    """Return a new path with suffix inserted before the final extension."""
    return p.with_name(p.stem + suffix + p.suffix)

def has_any_suffix(name: str, suffixes=("_D", "_H")) -> bool:
    """Detect if a filename stem already ends with a known suffix."""
    return any(name.endswith(s) for s in suffixes)

def resolve_collision(target: Path) -> Path:
    """If target exists, append a numeric counter _D2, _D3, ..."""
    if not target.exists():
        return target
    base, ext = target.stem, target.suffix
    m = re.match(r"^(.*?)(_D|_H)(\d+)?$", base)
    if m:
        root, sfx, num = m.groups()
        n = int(num) + 1 if num else 2
        return target.with_name(f"{root}{sfx}{n}{ext}")
    # generic fallback
    n = 2
    while True:
        cand = target.with_name(f"{base}{n}{ext}")
        if not cand.exists():
            return cand
        n += 1

plan = []

# 1) Enhanced parquet files: {yymm}_enh.parquet  ->  {yymm}_enh_D.parquet
# Only pick files that look like *_enh.parquet AND do not already carry _D/_H
for p in sorted(enh_dir.glob("*_enh.parquet")):
    stem = p.stem  # e.g., "2304_enh"
    if has_any_suffix(stem):  # already suffixed like *_enh_D or *_enh_H
        continue
    target = add_suffix_before_ext(p, SUFFIX)
    target = resolve_collision(target)
    plan.append((p, target))

# 2) Sweeper outputs:
#    sweep_results.parquet / sweep_results.csv / sweep_results_best.json  ->  *_D.*
for prefix in SWEEPER_PREFIXES:
    # core tables
    for ext in (".parquet", ".csv"):
        p = out_dir / f"{prefix}{ext}"
        if p.exists():
            stem = p.stem
            if not has_any_suffix(stem):
                target = add_suffix_before_ext(p, SUFFIX)
                target = resolve_collision(target)
                plan.append((p, target))
    # best-variant JSON
    pbest = out_dir / f"{prefix}_best.json"
    if pbest.exists():
        stem = pbest.stem  # e.g., "sweep_results_best"
        if not has_any_suffix(stem):
            # insert suffix before extension (after entire stem)
            target = add_suffix_before_ext(pbest, SUFFIX)
            target = resolve_collision(target)
            plan.append((pbest, target))

# ---- Show the plan ----
if not plan:
    print("[INFO] Nothing to rename (files already suffixed or not found).")
else:
    print(f"[INFO] Planned renames ({'DRY RUN' if DRY_RUN else 'COMMIT'}):")
    for src, dst in plan:
        print(f"  {src}  ->  {dst}")

    if not DRY_RUN:
        moved = 0
        for src, dst in plan:
            try:
                src.rename(dst)
                moved += 1
            except Exception as e:
                print(f"[ERROR] Failed to rename {src} -> {dst}: {e}")
        print(f"[DONE] Renamed {moved}/{len(plan)} files.")
    else:
        print("Set DRY_RUN=False and re-run to perform the renames.")


