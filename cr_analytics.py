# cr_analytics.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


# --------------------
# Small helpers
# --------------------

def _safe_ratio(num, den):
    try:
        return float(num) / float(den) if den else np.nan
    except Exception:
        return np.nan


def _ensure_ts(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a datetime64[ns] Series from df[col] with NaNs dropped."""
    s = pd.to_datetime(df[col], errors="coerce")
    return s


# --------------------
# Cumulative PnL series (closes only)
# --------------------

def cum_pnl_series_overlay(
    pos_overlay: pd.DataFrame,
    *,
    pnl_col: str = "pnl_net_bp",
    time_col: str = "close_ts",
) -> pd.DataFrame:
    """
    Build a time series of *realized* cumulative PnL from overlay positions.

    - Uses one row per closed pair (i.e. positions_ledger overlay),
      so we are NOT summing mark events.
    - Returns DataFrame with: [time_col, 'pnl', 'cum_pnl'].
    """
    if pos_overlay is None or pos_overlay.empty:
        return pd.DataFrame(columns=[time_col, "pnl", "cum_pnl"])

    if pnl_col not in pos_overlay.columns:
        raise ValueError(f"{pnl_col} not found in pos_overlay.columns")

    df = pos_overlay.copy()
    df[time_col] = _ensure_ts(df, time_col)
    df = df[df[time_col].notna()].copy()

    # Sort closes in time; if you want, we could also sort by open_ts second
    sort_cols = [time_col]
    if "open_ts" in df.columns:
        sort_cols.append("open_ts")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    out = df[[time_col, pnl_col]].rename(columns={pnl_col: "pnl"})
    out["cum_pnl"] = out["pnl"].cumsum()
    return out


def plot_cum_pnl_overlay(
    pos_overlay: pd.DataFrame,
    *,
    pnl_col: str = "pnl_net_bp",
    time_col: str = "close_ts",
    label: str = "",
    ax=None,
):
    """
    Plot cumulative realized overlay PnL (bp × unit DV01), based on closes only.
    """
    series = cum_pnl_series_overlay(pos_overlay, pnl_col=pnl_col, time_col=time_col)
    if series.empty:
        print("[PLOT] No closed overlay positions to plot.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(series[time_col], series["cum_pnl"])
    ttl = "Overlay cumulative realized PnL (bp × unit DV01)"
    if label:
        ttl += f" – {label}"
    ax.set_title(ttl)
    ax.set_xlabel("Close time")
    ax.set_ylabel("Cum PnL (bp × unit DV01)")
    ax.grid(True)
    plt.tight_layout()
    return ax


def plot_cum_pnl_overlay_cash(
    pos_overlay: pd.DataFrame,
    *,
    pnl_col: str = "pnl_net_cash",
    time_col: str = "close_ts",
    label: str = "",
    ax=None,
):
    """
    Same as plot_cum_pnl_overlay but using pnl_net_cash for dollar terms.
    """
    if pos_overlay is None or pos_overlay.empty:
        print("[PLOT] No closed overlay positions to plot.")
        return

    if pnl_col not in pos_overlay.columns:
        print(f"[PLOT] {pnl_col} not in pos_overlay; skipping cash PnL plot.")
        return

    df = pos_overlay.copy()
    df[time_col] = _ensure_ts(df, time_col)
    df = df[df[time_col].notna()].copy()
    df = df.sort_values([time_col]).reset_index(drop=True)

    df["cum_pnl"] = df[pnl_col].cumsum()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df[time_col], df["cum_pnl"])
    ttl = "Overlay cumulative realized PnL (cash)"
    if label:
        ttl += f" – {label}"
    ax.set_title(ttl)
    ax.set_xlabel("Close time")
    ax.set_ylabel("Cum PnL (cash units)")
    ax.grid(True)
    plt.tight_layout()
    return ax


# --------------------
# Overlay positions analytics (pos_overlay)
# --------------------

def summarize_pos_overlay(pos_overlay: pd.DataFrame, label: str = "overlay"):
    """
    High-level analytics for overlay positions (pairs actually traded).

    Expected columns in pos_overlay:
      'open_ts','close_ts','exit_reason','tenor_i','tenor_j','w_i','w_j',
      'entry_zspread','pnl_gross_bp','pnl_gross_cash','tcost_bp','tcost_cash',
      'pnl_net_bp','pnl_net_cash','days_held_equiv','conv_proxy','mode',
      'scale_dv01','yymm'
    """
    print(f"\n=== Overlay positions summary: {label} ===")

    if pos_overlay is None or pos_overlay.empty:
        print("No overlay positions (positions_ledger is empty).")
        return

    df = pos_overlay.copy()

    # 1) Core metrics (net PnL)
    n_pairs = len(df)
    exit_counts = df["exit_reason"].value_counts(dropna=False)

    pnl_net_bp = df["pnl_net_bp"].sum()
    pnl_net_cash = df["pnl_net_cash"].sum()
    avg_pnl_bp = df["pnl_net_bp"].mean()
    std_pnl_bp = df["pnl_net_bp"].std(ddof=1) if n_pairs > 1 else np.nan
    sharpe_like = avg_pnl_bp / std_pnl_bp if (std_pnl_bp not in (0, None) and not np.isnan(std_pnl_bp)) else np.nan

    avg_days_held = df["days_held_equiv"].mean() if "days_held_equiv" in df.columns else np.nan

    core = pd.DataFrame({
        "n_pairs": [n_pairs],
        "cum_pnl_net_bp": [pnl_net_bp],
        "cum_pnl_net_cash": [pnl_net_cash],
        "avg_pnl_net_bp": [avg_pnl_bp],
        "std_pnl_net_bp": [std_pnl_bp],
        "sharpe_like": [sharpe_like],
        "avg_days_held": [avg_days_held],
    }).T.rename(columns={0: "value"})

    print("\n-- Core overlay pair metrics (net PnL) --")
    display(core)

    # 2) Exit breakdown & ratios
    print("\n-- Exit breakdown (count) --")
    display(exit_counts.to_frame("count"))

    rev = exit_counts.get("reversion", 0)
    stop = exit_counts.get("stop", 0)
    maxh = exit_counts.get("max_hold", 0)

    ratios = pd.Series({
        "reversion_ratio": _safe_ratio(rev, n_pairs),
        "stop_ratio": _safe_ratio(stop, n_pairs),
        "max_hold_ratio": _safe_ratio(maxh, n_pairs),
    })
    print("\n-- Exit ratios --")
    display(ratios.to_frame("value"))

    # 3) By month (yymm)
    if "yymm" in df.columns:
        by_m = (
            df.groupby("yymm", as_index=False)
              .agg(
                  n_pairs=("exit_reason", "count"),
                  cum_pnl_net_bp=("pnl_net_bp", "sum"),
                  cum_pnl_net_cash=("pnl_net_cash", "sum"),
                  avg_pnl_net_bp=("pnl_net_bp", "mean"),
              )
              .sort_values("yymm")
        )
        print("\n-- Overlay PnL by month (yymm) --")
        display(by_m)

    # 4) Holding-period distribution
    if "days_held_equiv" in df.columns:
        print("\n-- Holding period (days_held_equiv) describe --")
        display(df["days_held_equiv"].describe().to_frame("days_held_equiv"))


# --------------------
# Overlay diagnostics analytics (diag_overlay)
# --------------------

def summarize_diag_overlay(diag_overlay: pd.DataFrame, label: str = "overlay"):
    """
    Analytics for overlay diagnostics (per hedge).

    Expected columns in diag_overlay:
      'yymm','trade_id','trade_ts','decision_ts','side','dv01','trade_tenor',
      'exec_tenor','exec_z','best_alt_tenor','best_alt_z','best_zdisp','reason',
      'hit_any_alt','hit_z_threshold','hit_fly_block','hit_caps_block'
    """
    print(f"\n=== Overlay diagnostics summary: {label} ===")

    if diag_overlay is None or diag_overlay.empty:
        print("diag_overlay is empty; no hedge-level diagnostics.")
        return

    df = diag_overlay.copy()

    n_hedges = len(df)
    reason_counts = df["reason"].value_counts(dropna=False)
    opened = int((df["reason"] == "opened").sum())
    hit_ratio = _safe_ratio(opened, n_hedges)

    core = pd.DataFrame({
        "n_hedges": [n_hedges],
        "n_opened_pairs": [opened],
        "hit_ratio": [hit_ratio],
    }).T.rename(columns={0: "value"})

    print("\n-- Hedge-level core metrics --")
    display(core)

    print("\n-- Reason breakdown (per hedge) --")
    display(reason_counts.to_frame("count"))

    # By side (CPAY/CREC)
    if "side" in df.columns:
        by_side = (
            df.groupby("side")["reason"]
              .value_counts()
              .rename("count")
              .reset_index()
              .pivot(index="side", columns="reason", values="count")
              .fillna(0)
        )
        print("\n-- Reason breakdown by side --")
        display(by_side)

        # Hit ratio by side
        opened_by_side = df[df["reason"] == "opened"].groupby("side")["trade_id"].count()
        total_by_side = df.groupby("side")["trade_id"].count()
        hit_by_side = (opened_by_side / total_by_side).rename("hit_ratio")
        print("\n-- Hit ratio by side --")
        display(hit_by_side.to_frame())

    # By month
    if "yymm" in df.columns:
        by_m = (
            df.groupby("yymm")
              .agg(
                  n_hedges=("trade_id", "count"),
                  n_opened_pairs=("reason", lambda s: (s == "opened").sum()),
                  avg_best_zdisp_opened=("best_zdisp", lambda x: x[df.loc[x.index, "reason"] == "opened"].mean()),
              )
              .reset_index()
        )
        by_m["hit_ratio"] = by_m["n_opened_pairs"] / by_m["n_hedges"]
        print("\n-- Hedge diagnostics by month (yymm) --")
        display(by_m.sort_values("yymm"))

    # Distribution of best_zdisp where overlay actually opened
    opened_df = df[df["reason"] == "opened"].copy()
    if not opened_df.empty and "best_zdisp" in opened_df.columns:
        print("\n-- best_zdisp for opened hedges (describe) --")
        display(opened_df["best_zdisp"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame("best_zdisp"))

        # Quick histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(opened_df["best_zdisp"].dropna(), bins=30)
        ax.set_title("Overlay: best_zdisp distribution (opened hedges)")
        ax.set_xlabel("best_zdisp")
        ax.set_ylabel("count")
        ax.grid(True)
        plt.tight_layout()

    # Which blockers dominate among non-opened?
    non_open = df[df["reason"] != "opened"].copy()
    if not non_open.empty:
        blockers = {
            "hit_any_alt": "any_alt",
            "hit_z_threshold": "z_threshold",
            "hit_fly_block": "fly_block",
            "hit_caps_block": "caps_block",
        }
        blk_counts = {}
        for col, name in blockers.items():
            if col in non_open.columns:
                blk_counts[name] = int(non_open[col].sum())
        if blk_counts:
            print("\n-- Non-opened hedges: blocker flags (count) --")
            display(pd.Series(blk_counts).to_frame("count"))


# --------------------
# One-shot full report
# --------------------

def overlay_full_report(
    pos_overlay: pd.DataFrame,
    diag_overlay: pd.DataFrame,
    *,
    label: str = "overlay",
):
    """
    Convenience function: run both position-level and hedge-level analytics
    and draw cumulative PnL plots (bp and cash).
    """
    summarize_pos_overlay(pos_overlay, label=label)
    summarize_diag_overlay(diag_overlay, label=label)

    # PnL in bp × unit DV01
    plot_cum_pnl_overlay(pos_overlay, label=label)

    # PnL in cash terms, if available
    plot_cum_pnl_overlay_cash(pos_overlay, label=label)
