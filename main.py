import matplotlib.pyplot as plt

# Assuming these exist:
# signals_df, regime_df

fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# --- 1) Signal Health ---
ax[0].plot(signals_df.index, signals_df["health"], label="Signal Health", linewidth=1.8)
ax[0].axhline(0, color='black', linewidth=0.8)
ax[0].set_title("Signal Health Over Time")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

# --- 2) Raw shocks (day-over-day) ---
ax[1].plot(signals_df.index, signals_df["shock_raw"].astype(int), 
           drawstyle="steps-post", label="Raw Shock Flags")
ax[1].set_title("Shock Flags (Before Moving Window)")
ax[1].grid(True, alpha=0.3)
ax[1].legend()

# --- 3) Final regime ---
ax[2].plot(regime_df.index, regime_df["regime"].astype(int),
           drawstyle="steps-post", label="Final Regime")
ax[2].set_title("Final Regime State (0 = OK, 1 = Blocked)")
ax[2].grid(True, alpha=0.3)
ax[2].legend()

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(14, 5))

# Signal health line
ax.plot(signals_df.index, signals_df["health"], label="Signal Health", linewidth=1.8)

# Add shaded regions where regime == 1 (blocked)
for ts, val in regime_df["regime"].iteritems():
    if val == 1:
        ax.axvspan(ts, ts, color="red", alpha=0.15)

ax.set_title("Signal Health with Regime Blocking")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# If you have z_comb series for comparison:
z_comb = signals_df["z_comb"] if "z_comb" in signals_df else None

if z_comb is not None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(z_comb.index, z_comb, label="z_comb", linewidth=1.5)

    # Shade blocks
    for ts, val in regime_df["regime"].items():
        if val == 1:
            ax.axvspan(ts, ts, color="red", alpha=0.15)

    ax.set_title("z_comb with Regime Filter")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()
    
fig, ax = plt.subplots(figsize=(15, 5))

# main signal — pick whichever you rely on most.
ax.plot(signals_df.index, signals_df["health"], label="Health", linewidth=1.8)

# Regime shading
reg = regime_df["regime"]
starts = reg[(reg == 1) & (reg.shift(1, fill_value=0) == 0)].index
ends   = reg[(reg == 0) & (reg.shift(1, fill_value=0) == 1)].index

# ensure ends align
if len(ends) < len(starts):
    ends = ends.append(pd.Index([reg.index[-1]]))

for s, e in zip(starts, ends):
    ax.axvspan(s, e, color='red', alpha=0.15)

ax.set_title("Regime Filter Applied to Signal Health")
ax.legend()
ax.grid(True)
plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import cr_config as cr  # assumes cr_config.py is importable from this notebook

plt.rcParams["figure.figsize"] = (12, 4)

# ---------- FILE ITERATION ----------

def iter_enhanced_files():
    """
    Yield enhanced parquet files one by one from PATH_ENH matching ENH_SUFFIX,
    e.g. *_d.parquet or *_h.parquet.
    """
    enh_path = Path(cr.PATH_ENH)
    pattern = f"*{cr.ENH_SUFFIX}.parquet"
    files = sorted(enh_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No enhanced parquet files matching {pattern} in {enh_path}")
    for f in files:
        yield f


# ---------- PER-FILE STATS (SMALL) ----------

def compute_file_stats(parquet_path: Path) -> pd.DataFrame:
    """
    Load ONE enhanced parquet file and compute per-bucket summary stats.
    Returns a small DataFrame (one row per decision bucket in that file).
    """
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return pd.DataFrame()

    if "ts" not in df.columns or "tenor_yrs" not in df.columns:
        raise ValueError(f"{parquet_path} missing 'ts' or 'tenor_yrs'.")

    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "tenor_yrs"])

    freq = str(cr.DECISION_FREQ).upper()
    if freq == "D":
        df["bucket"] = df["ts"].dt.floor("D")
    elif freq == "H":
        df["bucket"] = df["ts"].dt.floor("H")
    else:
        raise ValueError("cr.DECISION_FREQ must be 'D' or 'H'.")

    if "z_comb" not in df.columns:
        raise ValueError(f"{parquet_path} has no 'z_comb' column.")

    have_z_pca    = ("z_pca" in df.columns) and df["z_pca"].notna().any()
    have_z_spline = ("z_spline" in df.columns) and df["z_spline"].notna().any()

    if have_z_pca and have_z_spline:
        df["abs_resid"] = (df["z_spline"] - df["z_pca"]).abs()
    else:
        df["abs_resid"] = np.nan

    g = df.groupby("bucket", sort=True)

    stats = g.apply(
        lambda g_: pd.Series({
            "n_tenors": g_["tenor_yrs"].nunique(),
            "z_std": g_["z_comb"].std(skipna=True),
            "z_iqr": g_["z_comb"].quantile(0.75) - g_["z_comb"].quantile(0.25),
            "z_max_abs": g_["z_comb"].abs().max(),
            "median_abs_resid": g_["abs_resid"].median(skipna=True),
            "p95_abs_resid": (
                g_["abs_resid"].quantile(0.95)
                if g_["abs_resid"].notna().any()
                else np.nan
            ),
            "mean_z": g_["z_comb"].mean(skipna=True),
        })
    ).reset_index()

    return stats


# ---------- STREAM ALL FILES & BUILD GLOBAL STATS ----------

def build_global_stats_streaming() -> pd.DataFrame:
    """
    Iterate over all enhanced files, compute small per-bucket stats per file,
    and concatenate them. This keeps memory usage low.
    """
    all_stats = []
    for f in iter_enhanced_files():
        print(f"[INFO] Processing {f.name} ...")
        s = compute_file_stats(f)
        if not s.empty:
            all_stats.append(s)

    if not all_stats:
        raise ValueError("No non-empty per-bucket stats produced from enhanced files.")

    stats = pd.concat(all_stats, ignore_index=True)

    # Ensure sorted by time
    stats = stats.sort_values("bucket").reset_index(drop=True)
    return stats


# ---------- STRESS / HEALTH METRICS (NO RAW PANEL NEEDED) ----------

def add_signal_health(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Given per-bucket stats with columns:
      bucket, z_std, median_abs_resid, mean_z
    compute:
      - z_stress
      - resid_stress
      - trend_stress (from Δ mean_z autocorr)
      - signal_health (PCA+z)
      - signal_health_combined (incl. trend)
    """
    out = stats.copy()

    # Robust scale for z_std
    z_std_med = out["z_std"].median(skipna=True)
    z_std_hi  = out["z_std"].quantile(0.9) if out["z_std"].notna().any() else np.nan

    resid_med = out["median_abs_resid"].median(skipna=True)
    resid_hi  = out["median_abs_resid"].quantile(0.9) if out["median_abs_resid"].notna().any() else np.nan

    def _normalize(x, mid, hi):
        if not np.isfinite(mid) or not np.isfinite(hi) or hi <= mid:
            return np.zeros_like(x)
        return np.clip((x - mid) / (hi - mid), 0.0, 1.0)

    out["z_stress"] = _normalize(out["z_std"].values, z_std_med, z_std_hi)
    out["resid_stress"] = _normalize(out["median_abs_resid"].values, resid_med, resid_hi)

    out["stress_score"] = 0.5 * out["z_stress"] + 0.5 * out["resid_stress"]
    out["signal_health"] = 1.0 - out["stress_score"]

    # --- trend / persistence based only on mean_z series ---
    out = out.sort_values("bucket").reset_index(drop=True)
    out["d_mean_z"] = out["mean_z"].diff()

    window = 20  # you can tune this
    ac_vals = []
    d = out["d_mean_z"]
    for i in range(len(d)):
        if i < window:
            ac_vals.append(np.nan)
            continue
        x = d.iloc[i-window+1:i+1]
        if x.isna().any():
            ac_vals.append(np.nan)
            continue
        ac_vals.append(x.autocorr(lag=1))

    out["d_mean_z_ac1"] = ac_vals

    ac_series = out["d_mean_z_ac1"]
    ac_med = ac_series.median(skipna=True)
    ac_hi  = ac_series.quantile(0.9) if ac_series.notna().any() else np.nan

    if np.isfinite(ac_med) and np.isfinite(ac_hi) and ac_hi > ac_med:
        trend_stress = np.clip((ac_series - ac_med) / (ac_hi - ac_med), 0.0, 1.0)
    else:
        trend_stress = pd.Series(np.zeros(len(ac_series)), index=ac_series.index)

    out["trend_stress"] = trend_stress
    out["trend_health"] = 1.0 - out["trend_stress"]

    # Combined health
    out["stress_score_combined"] = (
        0.4 * out["z_stress"] +
        0.4 * out["resid_stress"] +
        0.2 * out["trend_stress"].fillna(0.0)
    )
    out["signal_health_combined"] = 1.0 - out["stress_score_combined"]

    return out


# ---------- RUN & PLOT ----------

stats = build_global_stats_streaming()
stats = add_signal_health(stats)

display(stats.head())

# Signal health
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["signal_health"], label="PCA / z signal_health")
ax.plot(stats["bucket"], stats["signal_health_combined"], label="Combined (incl. trend)", alpha=0.7)
ax.set_title("Signal Health Over Time")
ax.set_ylabel("Health (1 = good, 0 = bad)")
ax.legend()
plt.tight_layout()
plt.show()

# PCA residuals
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["median_abs_resid"], label="median |z_spline - z_pca|")
ax.set_title("PCA Residuals Over Time")
ax.legend()
plt.tight_layout()
plt.show()

# Trend metric
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["d_mean_z_ac1"], label="AC1 of Δ mean_z")
ax.set_title("Trendiness of z_comb (higher = more trending / less mean-reverting)")
ax.legend()
plt.tight_layout()
plt.show()

# --- imports & config ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import cr_config as cr  # assumes cr_config.py is importable from this notebook

plt.rcParams["figure.figsize"] = (12, 4)

# --- helper: load all enhanced parquet files into one panel ---

def load_enhanced_panel():
    """
    Load all enhanced files from PATH_ENH that match ENH_SUFFIX.
    Returns a single DataFrame with ts, tenor_yrs, z_spline, z_pca, z_comb, etc.
    """
    enh_path = Path(cr.PATH_ENH)
    pattern = f"*{cr.ENH_SUFFIX}.parquet"  # e.g. *_d.parquet or *_h.parquet
    files = sorted(enh_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No enhanced parquet files matching {pattern} in {enh_path}")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        df["src_file"] = f.name
        dfs.append(df)

    if not dfs:
        raise ValueError("All enhanced parquet files are empty.")

    panel = pd.concat(dfs, ignore_index=True)
    if "ts" not in panel.columns or "tenor_yrs" not in panel.columns:
        raise ValueError("Enhanced files must contain at least 'ts' and 'tenor_yrs' columns.")

    panel["ts"] = pd.to_datetime(panel["ts"], utc=False, errors="coerce")
    panel = panel.dropna(subset=["ts", "tenor_yrs"])

    # decision bucket consistent with backtest
    freq = str(cr.DECISION_FREQ).upper()
    if freq == "D":
        panel["bucket"] = panel["ts"].dt.floor("D")
    elif freq == "H":
        panel["bucket"] = panel["ts"].dt.floor("H")
    else:
        raise ValueError("cr.DECISION_FREQ must be 'D' or 'H'.")

    return panel


# --- helper: compute diagnostics per decision bucket ---

def compute_signal_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each decision bucket, compute:
      - cross-sectional dispersion of z_comb
      - PCA residual stats: |z_spline - z_pca|
      - basic counts
      And build a simple signal_health index in [0, 1].
    """
    have_z_pca = ("z_pca" in panel.columns) and panel["z_pca"].notna().any()
    have_z_spline = ("z_spline" in panel.columns) and panel["z_spline"].notna().any()

    if "z_comb" not in panel.columns:
        raise ValueError("Expected 'z_comb' in enhanced panel.")

    if not have_z_pca or not have_z_spline:
        print("[WARN] z_pca and/or z_spline not present/usable. PCA residual diagnostics will be skipped.")

    df = panel.copy()

    if have_z_pca and have_z_spline:
        df["resid"] = df["z_spline"] - df["z_pca"]
        df["abs_resid"] = df["resid"].abs()
    else:
        df["abs_resid"] = np.nan

    # group by decision bucket
    grouped = df.groupby("bucket", sort=True)

    stats = grouped.apply(
        lambda g: pd.Series({
            "n_tenors": g["tenor_yrs"].nunique(),
            "z_std": g["z_comb"].std(skipna=True),
            "z_iqr": (g["z_comb"].quantile(0.75) - g["z_comb"].quantile(0.25)),
            "z_max_abs": g["z_comb"].abs().max(),
            "median_abs_resid": g["abs_resid"].median(skipna=True),
            "p95_abs_resid": g["abs_resid"].quantile(0.95) if g["abs_resid"].notna().any() else np.nan,
        })
    ).reset_index()

    # --- build a very simple "signal_health" index ---
    # Higher dispersion and larger residuals => worse health.

    # robust scales (avoid division by zero)
    z_std_med = stats["z_std"].median(skipna=True)
    z_std_hi = stats["z_std"].quantile(0.9) if stats["z_std"].notna().any() else np.nan

    resid_med = stats["median_abs_resid"].median(skipna=True)
    resid_hi = stats["median_abs_resid"].quantile(0.9) if stats["median_abs_resid"].notna().any() else np.nan

    def _normalize(x, mid, hi):
        if not np.isfinite(mid) or not np.isfinite(hi) or hi <= mid:
            return np.zeros_like(x)
        # 0 near mid, 1 near hi, clipped
        return np.clip((x - mid) / (hi - mid), 0.0, 1.0)

    stats["z_stress"] = _normalize(stats["z_std"].values, z_std_med, z_std_hi)  # shape change / stress
    stats["resid_stress"] = _normalize(stats["median_abs_resid"].values, resid_med, resid_hi)

    # composite stress and health
    stats["stress_score"] = 0.5 * stats["z_stress"] + 0.5 * stats["resid_stress"]
    stats["signal_health"] = 1.0 - stats["stress_score"]

    return stats.sort_values("bucket").reset_index(drop=True)


# --- helper: rough "trending vs reversion" metric using z_comb changes ---

def add_trend_metric(stats: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a simple trending metric based on cross-sectional mean z_comb changes.
    This is very coarse but gives a sense of whether z moves are persistent.
    """
    df = panel.copy()
    # cross-sectional mean z per bucket
    mean_z = df.groupby("bucket")["z_comb"].mean().rename("mean_z").to_frame().reset_index()
    mean_z["d_mean_z"] = mean_z["mean_z"].diff()

    # rolling window for autocorr / trendiness
    window = 20  # ~1 month if daily, or adjust
    d = mean_z["d_mean_z"]

    ac_vals = []
    for i in range(len(d)):
        if i < window:
            ac_vals.append(np.nan)
            continue
        x = d.iloc[i-window+1:i+1]
        if x.isna().any():
            ac_vals.append(np.nan)
            continue
        # lag-1 autocorr approx
        ac = x.autocorr(lag=1)
        ac_vals.append(ac)

    mean_z["d_mean_z_ac1"] = ac_vals

    # normalize trendiness into [0,1] "trend_stress"
    ac_series = mean_z["d_mean_z_ac1"]
    ac_med = ac_series.median(skipna=True)
    ac_hi = ac_series.quantile(0.9) if ac_series.notna().any() else np.nan

    if np.isfinite(ac_med) and np.isfinite(ac_hi) and ac_hi > ac_med:
        trend_stress = np.clip((ac_series - ac_med) / (ac_hi - ac_med), 0.0, 1.0)
    else:
        trend_stress = pd.Series(np.zeros(len(ac_series)), index=ac_series.index)

    mean_z["trend_stress"] = trend_stress
    mean_z["trend_health"] = 1.0 - mean_z["trend_stress"]

    # merge into stats
    out = stats.merge(mean_z[["bucket", "mean_z", "d_mean_z", "d_mean_z_ac1", "trend_stress", "trend_health"]],
                      on="bucket", how="left")
    # update overall signal_health to include trend
    out["stress_score_combined"] = (
        0.4 * out["z_stress"] +
        0.4 * out["resid_stress"] +
        0.2 * out["trend_stress"].fillna(0.0)
    )
    out["signal_health_combined"] = 1.0 - out["stress_score_combined"]

    return out


# --- run everything and plot ---

panel = load_enhanced_panel()
stats = compute_signal_diagnostics(panel)
stats = add_trend_metric(stats, panel)

display(stats.head())

# Plot basic signal health over time
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["signal_health"], label="PCA/residual signal_health")
ax.plot(stats["bucket"], stats["signal_health_combined"], label="Combined (incl. trend)", alpha=0.7)
ax.set_title("Signal Health Over Time")
ax.set_ylabel("Health (1 = good, 0 = bad)")
ax.legend()
plt.tight_layout()
plt.show()

# Optionally look at residual stress directly
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["median_abs_resid"], label="median |z_spline - z_pca|")
ax.set_title("PCA Residuals Over Time")
ax.legend()
plt.tight_layout()
plt.show()

# Optionally inspect trending metric
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["d_mean_z_ac1"], label="AC1 of Δ mean z_comb")
ax.set_title("Trendiness of z_comb (higher = more trending / less mean-reverting)")
ax.legend()
plt.tight_layout()
plt.show()





import cr_analytics as cra

cra.overlay_full_report(pos_overlay, diag_overlay, label="overlay_D")


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


