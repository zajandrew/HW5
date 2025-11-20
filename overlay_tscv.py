"""
overlay_tscv.py

Time-series cross-validation for overlay mode using portfolio_test_new.

- Uses monthly enhanced files to infer available yymm months.
- Builds rolling train/test splits at the month level.
- For each split and parameter tuple (z_entry, z_exit, z_stop, max_hold_days),
  runs portfolio_test_new.run_all in overlay mode on TRAIN+TEST months
  but **evaluates PnL only on the TRAIN window**, in **bps**:

    train_pnl_bp = sum(pnl_gross_bp) for trades whose close_ts in [train_start, train_end]

- Outputs a parquet with one row per (split, param) combination:
    split_id, train_yymms, test_yymms,
    train_start, train_end, test_start, test_end,
    z_entry, z_exit, z_stop, max_hold_days,
    train_pnl_bp, n_trades_train
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import cr_config as cr
import portfolio_test_new as pt


# ======================================================================
# 0) Global hedge tape for worker processes
# ======================================================================

_GLOBAL_HEDGE_DF: Optional[pd.DataFrame] = None


def _init_worker(hedge_df: pd.DataFrame) -> None:
    """
    Initializer for worker processes: store hedge_df in a module-level
    global so we don't pickle it for every task.
    """
    global _GLOBAL_HEDGE_DF
    _GLOBAL_HEDGE_DF = hedge_df


# ======================================================================
# 1) Discover months (yymm) from enhanced files
# ======================================================================

def _iter_enhanced_paths_overlay() -> List[Path]:
    """
    Discover enhanced parquet files in PATH_ENH, mirroring portfolio_test_new
    naming convention as much as possible.
    """
    root = Path(getattr(cr, "PATH_ENH", "."))
    suffix = getattr(cr, "ENH_SUFFIX", "")
    if suffix:
        return sorted(root.glob(f"*{suffix}.parquet"))
    else:
        # reasonably permissive fallback
        return sorted(root.glob("*.parquet"))


def _parse_yymm_from_stem(stem: str) -> Optional[str]:
    """
    Try to extract a 'yymm' prefix from a file stem, e.g.:

      '2304_enh_v2' -> '2304'
      '2304_enh'    -> '2304'

    Returns None if we can't parse 4 leading digits into a valid month.
    """
    if len(stem) < 4:
        return None
    yymm = stem[:4]
    if not yymm.isdigit():
        return None
    yy = int(yymm[:2])
    mm = int(yymm[2:])
    if mm < 1 or mm > 12:
        return None
    return yymm


def discover_yymms_from_enhanced() -> List[str]:
    """
    Scan enhanced files and return sorted unique yymm strings.
    """
    paths = _iter_enhanced_paths_overlay()
    yymms: List[str] = []
    for p in paths:
        yymm = _parse_yymm_from_stem(p.stem)
        if yymm is not None:
            yymms.append(yymm)
    yymms = sorted(set(yymms))
    if not yymms:
        raise FileNotFoundError(f"No yymm months could be inferred from {cr.PATH_ENH}")
    return yymms


def _yymm_to_month_bounds(yymm: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Given 'yymm' (e.g. '2304'), return (month_start, month_end) as Timestamps
    in the local calendar (no timezone).
    """
    yy = int(yymm[:2])
    mm = int(yymm[2:])
    year = 2000 + yy
    month_start = pd.Timestamp(year=year, month=mm, day=1)
    # month_end = last day of the month
    month_end = (month_start + pd.offsets.MonthEnd(1))
    return month_start, month_end


# ======================================================================
# 2) Split spec and split builder
# ======================================================================

@dataclass
class OverlaySplitSpec:
    split_id: int
    train_yymms: List[str]
    test_yymms: List[str]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_overlay_splits_from_yymms(
    yymms: List[str],
    *,
    train_months: int = 6,
    test_months: int = 3,
    step_months: int = 3,
    min_test_months: int = 1,
) -> List[OverlaySplitSpec]:
    """
    Build rolling month-based splits from an ordered yymm list.

    Example (train_months=6, test_months=3, step_months=3):
      - Split 1: train = first 6 months, test = next 3
      - Split 2: train = months 4-9,      test = months 10-12
      - etc.

    We require at least `min_test_months` test months; otherwise we stop.
    """
    yymms = sorted(yymms)
    n = len(yymms)
    splits: List[OverlaySplitSpec] = []
    split_id = 1

    idx = 0
    while True:
        train_yymms = yymms[idx : idx + train_months]
        if len(train_yymms) < train_months:
            break

        test_start_idx = idx + train_months
        test_end_idx = test_start_idx + test_months
        test_yymms = yymms[test_start_idx:test_end_idx]

        if len(test_yymms) < min_test_months:
            break

        # Bounds for train window
        train_start, _ = _yymm_to_month_bounds(train_yymms[0])
        _, train_end = _yymm_to_month_bounds(train_yymms[-1])

        # Bounds for test window
        test_start, _ = _yymm_to_month_bounds(test_yymms[0])
        _, test_end = _yymm_to_month_bounds(test_yymms[-1])

        splits.append(
            OverlaySplitSpec(
                split_id=split_id,
                train_yymms=train_yymms,
                test_yymms=test_yymms,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        split_id += 1
        idx += step_months
        if idx >= n:
            break

    return splits


def describe_overlay_splits(
    *,
    train_months: int = 6,
    test_months: int = 3,
    step_months: int = 3,
    min_test_months: int = 1,
) -> List[OverlaySplitSpec]:
    """
    Convenience helper: discover yymms and print the overlay splits.
    """
    yymms = discover_yymms_from_enhanced()
    splits = build_overlay_splits_from_yymms(
        yymms,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        min_test_months=min_test_months,
    )

    print(f"Available months (yymm): {yymms}")
    print(f"train_months={train_months}, test_months={test_months}, step_months={step_months}")
    print(f"Number of splits: {len(splits)}\n")
    for s in splits:
        print(
            f"Split {s.split_id}: "
            f"train {s.train_yymms} ({s.train_start.date()} → {s.train_end.date()}) | "
            f"test {s.test_yymms} ({s.test_start.date()} → {s.test_end.date()})"
        )
    return splits


# ======================================================================
# 3) Config snapshot / restore and train-only metrics
# ======================================================================

_PARAM_NAMES = ["Z_ENTRY", "Z_EXIT", "Z_STOP", "MAX_HOLD_DAYS"]


def _snapshot_cr_params() -> Dict[str, object]:
    """
    Snapshot the Z-entry/exit/stop/max-hold params so we can restore after each run.
    """
    snap: Dict[str, object] = {}
    for name in _PARAM_NAMES:
        snap[name] = getattr(cr, name, None)
    return snap


def _apply_param_tuple_to_cr(
    z_entry: float,
    z_exit: float,
    z_stop: float,
    max_hold_days: int,
) -> None:
    """
    Apply one parameter tuple into cr_config.
    """
    setattr(cr, "Z_ENTRY", float(z_entry))
    setattr(cr, "Z_EXIT", float(z_exit))
    setattr(cr, "Z_STOP", float(z_stop))
    setattr(cr, "MAX_HOLD_DAYS", int(max_hold_days))


def _restore_cr_params(snap: Dict[str, object]) -> None:
    """
    Restore params in cr_config from a snapshot.
    """
    for name, val in snap.items():
        if val is not None:
            setattr(cr, name, val)


def _compute_train_metrics_bp(
    pos_df: pd.DataFrame,
    split: OverlaySplitSpec,
) -> Dict[str, float]:
    """
    Compute TRAIN PnL in *bps* and train trade count from positions_ledger
    for a given split.

    - PnL is sum of pnl_gross_bp for trades whose close_ts falls inside
      the TRAIN window [train_start, train_end].
    - n_trades_train is the number of such trades.
    """
    if (
        pos_df is None
        or pos_df.empty
        or "close_ts" not in pos_df.columns
        or "pnl_gross_bp" not in pos_df.columns
    ):
        return {
            "train_pnl_bp": 0.0,
            "n_trades_train": 0,
        }

    close_ts = pd.to_datetime(pos_df["close_ts"], utc=False, errors="coerce")
    mask_train = (close_ts >= split.train_start) & (close_ts <= split.train_end)

    train_pnl_bp = float(pos_df.loc[mask_train, "pnl_gross_bp"].sum())
    n_trades_train = int(mask_train.sum())

    return {
        "train_pnl_bp": train_pnl_bp,
        "n_trades_train": n_trades_train,
    }


# ======================================================================
# 4) Core worker: evaluate one (split, params) combo
# ======================================================================

def _eval_param_combo_for_split(
    split: OverlaySplitSpec,
    z_entry: float,
    z_exit: float,
    z_stop: float,
    max_hold_days: int,
    decision_freq: str,
    hedge_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Evaluate a single param tuple on a single split.
    Uses the *passed per-split trimmed hedge_df* if provided.
    Otherwise falls back to _GLOBAL_HEDGE_DF.
    """
    global _GLOBAL_HEDGE_DF

    decision_freq = decision_freq.upper()

    # --- NEW: prefer the passed hedge_df (trimmed) ---
    if hedge_df is not None:
        hedge_df_local = hedge_df
    else:
        hedge_df_local = _GLOBAL_HEDGE_DF

    if hedge_df_local is None or hedge_df_local.empty:
        raise ValueError("_eval_param_combo_for_split requires a non-empty hedge_df.")

    snap = _snapshot_cr_params()
    try:
        _apply_param_tuple_to_cr(z_entry, z_exit, z_stop, max_hold_days)

        # Run portfolio_test_new on train+test months
        months_span = split.train_yymms + split.test_yymms

        pos_df, led_df, pnl_by = pt.run_all(
            months_span,
            decision_freq=decision_freq,
            carry=True,
            force_close_end=False,
            mode="overlay",
            hedge_df=hedge_df_local,      # <-- NEW: pass trimmed hedges
            overlay_use_caps=None,
        )

        metrics = _compute_train_metrics_bp(pos_df, split)

        return {
            "split_id": split.split_id,
            "train_yymms": ",".join(split.train_yymms),
            "test_yymms": ",".join(split.test_yymms),
            "train_start": split.train_start,
            "train_end": split.train_end,
            "test_start": split.test_start,
            "test_end": split.test_end,
            "z_entry": float(z_entry),
            "z_exit": float(z_exit),
            "z_stop": float(z_stop),
            "max_hold_days": int(max_hold_days),
            "train_pnl_bp": metrics["train_pnl_bp"],
            "n_trades_train": metrics["n_trades_train"],
        }

    finally:
        _restore_cr_params(snap)


# ======================================================================
# 5) Top-level grid search driver (with optional parallelism)
# ======================================================================

def run_overlay_grid_search_from_portfolio(
    z_entry_grid,
    z_exit_grid,
    z_stop_grid,
    max_hold_grid,
    *,
    decision_freq: str | None = None,
    train_months: int = 6,
    test_months: int = 3,
    step_months: int = 3,
    min_test_months: int = 1,
    hedge_df: pd.DataFrame | None = None,
    n_jobs: int = 1,
    out_name: str | None = None,
) -> pd.DataFrame:

    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()

    if hedge_df is None or hedge_df.empty:
        raise ValueError("Need non-empty hedge_df.")

    # normalize tradetimeUTC
    h = hedge_df.copy()
    h["tradetimeUTC"] = (
        pd.to_datetime(h["tradetimeUTC"], utc=True, errors="coerce")
          .dt.tz_convert("UTC")
          .dt.tz_localize(None)
    )

    # discover months
    yymms = discover_yymms_from_enhanced()

    # build month-based split specs
    splits = build_overlay_splits_from_yymms(
        yymms,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        min_test_months=min_test_months,
    )

    tasks = []
    results_rows = []

    # --------- NEW: TRIM HEDGE TAPE FOR EACH SPLIT ---------
    for split in splits:
        print(
            f"[SPLIT {split.split_id}] "
            f"train {split.train_yymms} ({split.train_start.date()} → {split.train_end.date()}) | "
            f"test {split.test_yymms} ({split.test_start.date()} → {split.test_end.date()})"
        )

        hedge_sub = h[
            (h["tradetimeUTC"] >= split.train_start)
            & (h["tradetimeUTC"] <= split.test_end)
        ].copy()

        if hedge_sub.empty:
            print(f"  [WARN] No trades for split {split.split_id}; skipping.")
            continue

        for ze in z_entry_grid:
            for zx in z_exit_grid:
                for zs in z_stop_grid:
                    for mh in max_hold_grid:
                        tasks.append(
                            (
                                split,
                                float(ze),
                                float(zx),
                                float(zs),
                                int(mh),
                                decision_freq,
                                hedge_sub,   # <-- pass trimmed hedge tape
                            )
                        )

    if not tasks:
        raise RuntimeError("No tasks to run.")

    # --------- Parallel execution ----------
    if n_jobs > 1:
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            initargs=(None,),     # unused now, because we pass hedge_sub explicitly
        ) as ex:
            fut_to_meta = {
                ex.submit(
                    _eval_param_combo_for_split,
                    split, ze, zx, zs, mh,
                    decision_freq,
                    hedge_sub         # <-- pass trimmed hedge tape
                ): (split, ze, zx, zs, mh)
                for (split, ze, zx, zs, mh, decision_freq, hedge_sub) in tasks
            }

            for fut in as_completed(fut_to_meta):
                split, ze, zx, zs, mh = fut_to_meta[fut]
                results_rows.append(fut.result())

    else:
        # ---------- Serial version ----------
        for (split, ze, zx, zs, mh, dfreq, hedge_sub) in tasks:
            row = _eval_param_combo_for_split(
                split, ze, zx, zs, mh, dfreq, hedge_sub
            )
            results_rows.append(row)

    # -------- results dataframe --------
    res_df = pd.DataFrame(results_rows)
    if not res_df.empty:
        res_df = res_df.sort_values(
            ["split_id", "train_pnl_bp"],
            ascending=[True, False]
        )

    if out_name:
        out_dir = Path(getattr(cr, "PATH_OUT", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = getattr(cr, "OUT_SUFFIX", "")
        fname = (
            f"{out_name}{suffix}.parquet" if suffix else f"{out_name}.parquet"
        )
        out_path = out_dir / fname
        res_df.to_parquet(out_path, index=False)
        print(f"[WRITE] Overlay TSCV optimization results -> {out_path}")

    return res_df


# ======================================================================
# 6) CLI example
# ======================================================================

if __name__ == "__main__":
    # Example usage:
    #   python overlay_tscv.py  (expects cr.TRADE_TYPES.pkl to exist)
    trades_path = Path(f"{cr.TRADE_TYPES}.pkl")
    if not trades_path.exists():
        raise FileNotFoundError(f"Trade tape pickle not found: {trades_path}")
    hedge_df = pd.read_pickle(trades_path)

    # Small example grid – you’ll likely tighten/expand this in practice
    z_entry_grid = [0.75, 1.0]
    z_exit_grid = [0.25, 0.40]
    z_stop_grid = [1.5, 2.0]
    max_hold_grid = [5, 10]

    df_res = run_overlay_grid_search_from_portfolio(
        hedge_df=hedge_df,
        z_entry_grid=z_entry_grid,
        z_exit_grid=z_exit_grid,
        z_stop_grid=z_stop_grid,
        max_hold_grid=max_hold_grid,
        decision_freq="D",
        train_months=6,
        test_months=3,
        step_months=3,
        min_test_months=1,
        out_name="overlay_grid_synth",
        n_jobs=4,  # <-- bump this up/down as you like
    )

    # Best combo per split by TRAIN PnL in bps
    if not df_res.empty:
        best_per_split = df_res.groupby("split_id").head(1)
        print("\nBest params per split (by train_pnl_bp):")
        print(best_per_split)