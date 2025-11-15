import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# --- core helpers ---

def compute_cum_pnl_from_positions(pos: pd.DataFrame,
                                   pnl_col_candidates=("pnl_after_cost", "pnl"),
                                   time_col="close_ts") -> pd.DataFrame:
    """
    Build a time series of cumulative *realized* PnL from closed trades.

    - Uses the first existing column from pnl_col_candidates.
    - Sorts by close_ts (and open_ts as secondary if present).
    - Returns DataFrame with [time_col, 'pnl', 'cum_pnl'].
    """
    if pos is None or pos.empty:
        return pd.DataFrame(columns=[time_col, "pnl", "cum_pnl"])

    pnl_col = None
    for c in pnl_col_candidates:
        if c in pos.columns:
            pnl_col = c
            break
    if pnl_col is None:
        raise ValueError(f"No PnL column found in positions: tried {pnl_col_candidates}")

    df = pos.copy()
    # ensure timestamp
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].copy()

    # sort chronologically
    sort_cols = [time_col]
    if "open_ts" in df.columns:
        sort_cols.append("open_ts")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    out = df[[time_col, pnl_col]].rename(columns={pnl_col: "pnl"})
    out["cum_pnl"] = out["pnl"].cumsum()
    return out


def plot_cum_pnl_from_positions(pos: pd.DataFrame,
                                label: str = "",
                                *,
                                pnl_col_candidates=("pnl_after_cost", "pnl"),
                                time_col="close_ts",
                                ax=None):
    """
    Plot cumulative realized PnL from closes only.
    """
    series = compute_cum_pnl_from_positions(pos, pnl_col_candidates, time_col)
    if series.empty:
        print("[PLOT] No closed trades to plot.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(series[time_col], series["cum_pnl"])
    ax.set_title(f"Cumulative realized PnL (closes only) {label}")
    ax.set_xlabel("Close time")
    ax.set_ylabel("Cum PnL (bp × unit DV01)")
    ax.grid(True)
    plt.tight_layout()
    return ax


def add_cash_pnl_column(pos: pd.DataFrame,
                        pnl_col_candidates=("pnl_after_cost", "pnl"),
                        dv01_col="pair_dv01") -> pd.DataFrame:
    """
    Add a 'pnl_cash' column = pnl * pair_dv01 (if dv01_col exists), else NaN.
    Interprets:
      - pnl: bp × (unit DV01)
      - pair_dv01: actual cash DV01 of the pair (e.g. 1.5m = $1.5m/bp)
    """
    if pos is None or pos.empty:
        return pos

    pnl_col = None
    for c in pnl_col_candidates:
        if c in pos.columns:
            pnl_col = c
            break
    if pnl_col is None:
        return pos

    df = pos.copy()
    if dv01_col in df.columns:
        df["pnl_cash"] = df[pnl_col] * df[dv01_col]
    else:
        df["pnl_cash"] = np.nan
    return df


def summarize_strategy_backtest(pos: pd.DataFrame,
                                led: pd.DataFrame | None = None,
                                label: str = "strategy"):
    """
    High-level analytics for Mode A (strategy mode).
    Assumes 'pos' = positions_ledger; 'led' = marks_ledger (optional).
    """
    print(f"\n=== Strategy summary: {label} ===")

    if pos is None or pos.empty:
        print("No closed positions.")
        return

    pos = add_cash_pnl_column(pos)

    # Basic counts
    n_trades = len(pos)
    exit_counts = pos["exit_reason"].value_counts(dropna=False)
    cum_pnl = pos["pnl"].sum()
    cum_pnl_cash = pos["pnl_cash"].sum() if "pnl_cash" in pos.columns else np.nan
    avg_pnl = pos["pnl"].mean()
    std_pnl = pos["pnl"].std(ddof=1) if n_trades > 1 else np.nan
    sharpe_like = avg_pnl / std_pnl if (std_pnl not in (0, None) and not np.isnan(std_pnl)) else np.nan

    holding = pos.get("days_held_equiv", np.nan)
    avg_hold = holding.mean() if hasattr(holding, "mean") else np.nan

    summary = pd.DataFrame({
        "n_trades": [n_trades],
        "cum_pnl_bp_unit": [cum_pnl],
        "cum_pnl_cash": [cum_pnl_cash],
        "avg_pnl_bp_unit": [avg_pnl],
        "std_pnl_bp_unit": [std_pnl],
        "sharpe_like": [sharpe_like],
        "avg_days_held": [avg_hold],
    }).T.rename(columns={0: "value"})

    print("\n-- Core metrics --")
    display(summary)

    print("\n-- Exit breakdown --")
    display(exit_counts.to_frame("count"))

    # Reversion / stop / max_hold ratios
    def _safe_ratio(num, den):
        try:
            return float(num) / float(den) if den else np.nan
        except Exception:
            return np.nan

    rev = exit_counts.get("reversion", 0)
    stop = exit_counts.get("stop", 0)
    maxh = exit_counts.get("max_hold", 0)

    exit_ratios = pd.Series({
        "reversion_ratio": _safe_ratio(rev, n_trades),
        "stop_ratio": _safe_ratio(stop, n_trades),
        "max_hold_ratio": _safe_ratio(maxh, n_trades),
    })
    print("\n-- Exit ratios --")
    display(exit_ratios.to_frame("value"))

    # Cumulative realized PnL (closes-only)
    plot_cum_pnl_from_positions(pos, label=label)

    # Optional: distribution by bucket/tenor if your pos has bucket_i/bucket_j
    for col in ["bucket_i", "bucket_j"]:
        if col in pos.columns:
            print(f"\n-- {col} counts --")
            display(pos[col].value_counts().to_frame("count"))

def summarize_overlay_backtest(pos_overlay: pd.DataFrame,
                               diag_overlay: pd.DataFrame | None = None,
                               label: str = "overlay"):
    """
    Analytics for Mode B (hedge overlay) using:
      - pos_overlay: positions_ledger from overlay run
      - diag_overlay: overlay_diag DataFrame (per-hedge diagnostics)
    """
    print(f"\n=== Overlay summary: {label} ===")

    # --------- Position-level (pairs actually traded) ---------
    if pos_overlay is None or pos_overlay.empty:
        print("No overlay positions (no pairs opened).")
    else:
        pos_overlay = add_cash_pnl_column(pos_overlay)

        n_trades = len(pos_overlay)
        exit_counts = pos_overlay["exit_reason"].value_counts(dropna=False)

        pnl_col = "pnl_after_cost" if "pnl_after_cost" in pos_overlay.columns else "pnl"
        cum_pnl = pos_overlay[pnl_col].sum()
        cum_pnl_cash = pos_overlay["pnl_cash"].sum() if "pnl_cash" in pos_overlay.columns else np.nan
        avg_pnl = pos_overlay[pnl_col].mean()
        std_pnl = pos_overlay[pnl_col].std(ddof=1) if n_trades > 1 else np.nan
        sharpe_like = avg_pnl / std_pnl if (std_pnl not in (0, None) and not np.isnan(std_pnl)) else np.nan

        holding = pos_overlay.get("days_held_equiv", np.nan)
        avg_hold = holding.mean() if hasattr(holding, "mean") else np.nan

        summary_pos = pd.DataFrame({
            "n_pairs": [n_trades],
            "cum_pnl_bp_unit": [cum_pnl],
            "cum_pnl_cash": [cum_pnl_cash],
            "avg_pnl_bp_unit": [avg_pnl],
            "std_pnl_bp_unit": [std_pnl],
            "sharpe_like": [sharpe_like],
            "avg_days_held": [avg_hold],
        }).T.rename(columns={0: "value"})

        print("\n-- Overlay pair metrics (pairs that actually traded) --")
        display(summary_pos)

        print("\n-- Overlay pair exit breakdown --")
        display(exit_counts.to_frame("count"))

        # realized cum PnL from closes-only
        plot_cum_pnl_from_positions(pos_overlay,
                                    label=f"{label} (pairs)",
                                    pnl_col_candidates=(pnl_col,))

    # --------- Hedge-level (from overlay_diag) ---------
    if diag_overlay is not None and not diag_overlay.empty:
        print("\n-- Overlay diagnostic summary (per hedge) --")

        n_hedges = len(diag_overlay)
        reason_counts = diag_overlay["reason"].value_counts(dropna=False)
        opened = int((diag_overlay["reason"] == "opened").sum())

        # Hit ratio = fraction of hedges where overlay actually put on a pair
        hit_ratio = opened / n_hedges if n_hedges else np.nan

        diag_summary = pd.DataFrame({
            "n_hedges": [n_hedges],
            "n_opened_pairs": [opened],
            "hit_ratio": [hit_ratio],
        }).T.rename(columns={0: "value"})

        display(diag_summary)

        print("\nReason breakdown (per hedge):")
        display(reason_counts.to_frame("count"))

        # Quality of candidates (zdisp) where overlay would open
        opened_diag = diag_overlay[diag_overlay["reason"] == "opened"].copy()
        if not opened_diag.empty:
            print("\n-- Distribution of best_zdisp for opened hedges --")
            display(opened_diag["best_zdisp"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame("best_zdisp"))

            # Quick scatter: best_zdisp vs trade_tenor
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(opened_diag["trade_tenor"], opened_diag["best_zdisp"], s=10)
            ax.set_xlabel("Trade tenor (yrs)")
            ax.set_ylabel("Best z-dispersion vs executed tenor")
            ax.set_title("Overlay: best_zdisp vs trade_tenor")
            ax.grid(True)
            plt.tight_layout()

    else:
        print("\n[WARN] No overlay_diag provided or empty; hedge-level diagnostics unavailable.")
