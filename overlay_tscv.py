"""
overlay_tscv_from_portfolio.py

Time-series grid search for overlay mode using portfolio_test_new.

- Discover months (yymm) from enhanced files.
- Build rolling month-based train/test splits.
- For each split and parameter tuple (Z_ENTRY, Z_EXIT, Z_STOP, MAX_HOLD_DAYS),
  run portfolio_test_new.run_all in overlay mode on train+test months.
- Compute train/test PnL (cash) + trade counts from the resulting outputs.
- Return a DataFrame of results and optionally write to parquet.

Intended usage:
  - Optimize on synthetic hedge tape (hedge_df_synth).
  - Inspect best params (especially by test_pnl_cash) per split.
  - Optionally re-run on real hedge tape to assess OOS behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from concurrent.futures import ProcessPoolExecutor, as_completed

import cr_config as cr
import portfolio_test_new as pt


# ======================================================================
# 1) Discover months (yymm) from enhanced files
# ======================================================================

def discover_yymms_from_enh() -> List[str]:
    """
    Discover available months (yymm) from enhanced files, mirroring the
    naming convention used in portfolio_test_new._enhanced_in_path.

    We assume enhanced files live under PATH_ENH and look like:
      {yymm}_enh{ENH_SUFFIX}.parquet  (or similar).
    """
    root = Path(getattr(cr, "PATH_ENH", "."))
    suffix = getattr(cr, "ENH_SUFFIX", "")

    if suffix:
        paths = sorted(root.glob(f"*{suffix}.parquet"))
    else:
        paths = sorted(root.glob("*_enh.parquet"))

    yymms: List[str] = []
    for p in paths:
        stem = p.stem  # e.g. "2304_enh_v1"
        # Take the first 4 characters if they are digits
        cand = stem[:4]
        if len(cand) == 4 and cand.isdigit():
            yymms.append(cand)

    # Remove duplicates, keep sorted in ascending chronological order
    yymms = sorted(set(yymms))
    return yymms


# ======================================================================
# 2) Month-based split construction
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


def _yymm_to_month_start_end(yymm: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Map 'yymm' (e.g. '2304') → (month_start, month_end).
    """
    year = 2000 + int(yymm[:2])
    month = int(yymm[2:])
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + MonthEnd(1)
    return start, end


def build_overlay_splits_from_yymms(
    yymms: List[str],
    *,
    train_months: int = 6,
    test_months: int = 3,
    step_months: int = 3,
    min_test_months: int = 2,
) -> List[OverlaySplitSpec]:
    """
    Build rolling month-based train/test splits over a list of yymms.

    Logic:
      - We roll in steps of 'step_months'.
      - For each base index, we try to allocate up to 'train_months' for train,
        then up to 'test_months' for test.
      - We always enforce at least 'min_test_months' in the test window.
      - For the *last* split, the train window may be shorter than train_months
        if needed to make room for at least 'min_test_months' test months.

    Returns a list of OverlaySplitSpec.
    """
    n = len(yymms)
    if n == 0:
        return []

    splits: List[OverlaySplitSpec] = []
    split_id = 1

    i = 0
    while i < n:
        # Remaining months from i
        remaining = n - i

        # If we can't even fit the minimum test window, break
        if remaining < min_test_months + 1:
            break

        # Max train length we can use at this starting point
        max_train_len = min(train_months, remaining - min_test_months)
        if max_train_len <= 0:
            break

        # Actual train_len = max_train_len
        train_len = max_train_len
        test_len = min(test_months, remaining - train_len)
        if test_len < min_test_months:
            # If we can't get min_test_months, stop
            break

        train_yymms = yymms[i : i + train_len]
        test_yymms = yymms[i + train_len : i + train_len + test_len]

        train_start, _ = _yymm_to_month_start_end(train_yymms[0])
        _, train_end = _yymm_to_month_start_end(train_yymms[-1])
        test_start, _ = _yymm_to_month_start_end(test_yymms[0])
        _, test_end = _yymm_to_month_start_end(test_yymms[-1])

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

        # Move forward by step_months
        i += step_months

    return splits


def describe_overlay_splits(
    yymms: Optional[List[str]] = None,
    *,
    train_months: int = 6,
    test_months: int = 3,
    step_months: int = 3,
    min_test_months: int = 2,
) -> List[OverlaySplitSpec]:
    """
    Convenience helper to print and inspect overlay splits.
    """
    if yymms is None:
        yymms = discover_yymms_from_enh()

    splits = build_overlay_splits_from_yymms(
        yymms,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        min_test_months=min_test_months,
    )

    print(f"Available yymms: {yymms}")
    print(f"train_months={train_months}, test_months={test_months}, step_months={step_months}, min_test_months={min_test_months}")
    print(f"Number of splits: {len(splits)}\n")

    for s in splits:
        print(
            f"Split {s.split_id}: "
            f"TRAIN {s.train_yymms} ({s.train_start.date()} → {s.train_end.date()}) | "
            f"TEST {s.test_yymms} ({s.test_start.date()} → {s.test_end.date()})"
        )

    return splits


# ======================================================================
# 3) Helpers to temporarily set params in cr_config
# ======================================================================

def _snapshot_cr_params() -> Dict[str, object]:
    """
    Capture the relevant strategy parameters from cr_config so we can restore later.
    """
    keys = ["Z_ENTRY", "Z_EXIT", "Z_STOP", "MAX_HOLD_DAYS"]
    snap = {}
    for k in keys:
        if hasattr(cr, k):
            snap[k] = getattr(cr, k)
    return snap


def _apply_param_tuple_to_cr(
    z_entry: float,
    z_exit: float,
    z_stop: float,
    max_hold_days: int,
):
    """
    Apply parameters to cr_config in-place.
    """
    cr.Z_ENTRY = float(z_entry)
    cr.Z_EXIT = float(z_exit)
    cr.Z_STOP = float(z_stop)
    cr.MAX_HOLD_DAYS = int(max_hold_days)


def _restore_cr_params(snap: Dict[str, object]):
    """
    Restore previously captured cr_config parameters.
    """
    for k, v in snap.items():
        setattr(cr, k, v)


# ======================================================================
# 4) Metrics from run_all outputs
# ======================================================================

def _compute_train_test_metrics(
    pos_df: pd.DataFrame,
    pnl_by: pd.DataFrame,
    split: OverlaySplitSpec,
) -> Dict[str, float]:
    """
    Compute train/test PnL (cash) and trade counts from run_all outputs
    for a given split.

    - PnL is computed from pnl_by (pnl_cash by bucket).
    - n_trades counts closed positions whose close_ts falls in train/test.
    """
    # PnL by bucket
    if pnl_by is None or pnl_by.empty:
        train_pnl_cash = 0.0
        test_pnl_cash = 0.0
    else:
        buckets = pd.to_datetime(pnl_by["bucket"], utc=False, errors="coerce")
        pnl_vals = pnl_by["pnl_cash"].astype(float)

        train_mask = (buckets >= split.train_start) & (buckets <= split.train_end)
        test_mask = (buckets >= split.test_start) & (buckets <= split.test_end)

        train_pnl_cash = float(pnl_vals[train_mask].sum())
        test_pnl_cash = float(pnl_vals[test_mask].sum())

    # Trade counts by close_ts
    if pos_df is None or pos_df.empty or "close_ts" not in pos_df.columns:
        n_trades_train = 0
        n_trades_test = 0
    else:
        close_ts = pd.to_datetime(pos_df["close_ts"], utc=False, errors="coerce")

        train_trades_mask = (close_ts >= split.train_start) & (close_ts <= split.train_end)
        test_trades_mask = (close_ts >= split.test_start) & (close_ts <= split.test_end)

        n_trades_train = int(train_trades_mask.sum())
        n_trades_test = int(test_trades_mask.sum())

    return {
        "train_pnl_cash": train_pnl_cash,
        "test_pnl_cash": test_pnl_cash,
        "n_trades_train": n_trades_train,
        "n_trades_test": n_trades_test,
    }


# ======================================================================
# 5) Worker function for parallel evaluation
# ======================================================================

def _eval_param_combo_for_split(
    split: OverlaySplitSpec,
    hedge_df: pd.DataFrame,
    z_entry: float,
    z_exit: float,
    z_stop: float,
    max_hold_days: int,
    decision_freq: str,
) -> Dict[str, object]:
    """
    Evaluate a single param tuple on a single split:

      - Set params in cr_config.
      - Run run_all on train+test months in overlay mode.
      - Compute train/test metrics.
      - Return a result row dict.
    """
    # Local snapshot/restore within worker for safety
    snap = _snapshot_cr_params()
    try:
        _apply_param_tuple_to_cr(z_entry, z_exit, z_stop, max_hold_days)

        months_span = split.train_yymms + split.test_yymms

        pos_df, led_df, pnl_by = pt.run_all(
            months_span,
            decision_freq=decision_freq,
            carry=True,
            force_close_end=False,
            mode="overlay",
            hedge_df=hedge_df,
            overlay_use_caps=None,  # uses cr_config caps if any
        )

        metrics = _compute_train_test_metrics(pos_df, pnl_by, split)

        row = {
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
            "train_pnl_cash": metrics["train_pnl_cash"],
            "test_pnl_cash": metrics["test_pnl_cash"],
            "n_trades_train": metrics["n_trades_train"],
            "n_trades_test": metrics["n_trades_test"],
        }
        return row
    finally:
        _restore_cr_params(snap)


# ======================================================================
# 6) Main driver: overlay grid search
# ======================================================================

def run_overlay_grid_search_from_portfolio(
    hedge_df: pd.DataFrame,
    z_entry_grid: Iterable[float],
    z_exit_grid: Iterable[float],
    z_stop_grid: Iterable[float],
    max_hold_grid: Iterable[int],
    *,
    yymms: Optional[List[str]] = None,
    decision_freq: str | None = None,
    train_months: int = 6,
    test_months: int = 3,
    step_months: int = 3,
    min_test_months: int = 2,
    n_jobs: int = 1,
    out_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main grid-search driver using portfolio_test_new in overlay mode.

    Arguments:
      hedge_df       : trade tape (synthetic or real) with columns expected by prepare_hedge_tape
                       (side, instrument, EqVolDelta, tradetimeUTC, plus any *_mid curve cols
                        already attached).
      z_entry_grid   : iterable of candidate Z_ENTRY thresholds.
      z_exit_grid    : iterable of candidate Z_EXIT thresholds.
      z_stop_grid    : iterable of candidate Z_STOP thresholds.
      max_hold_grid  : iterable of candidate MAX_HOLD_DAYS.
      yymms          : optional explicit list of months; if None, discovered from PATH_ENH.
      decision_freq  : 'D' or 'H'; typically 'D' here.
      train_months   : number of months in each train window (max; last split may be shorter).
      test_months    : number of months in each test window (desired).
      step_months    : how many months to shift between splits.
      min_test_months: minimum test months required for a split (default 2).
      n_jobs         : number of processes for parallel evaluation (1 = sequential).
      out_name       : if provided, write results to PATH_OUT/{out_name}{OUT_SUFFIX}.parquet.

    Returns:
      DataFrame with one row per (split, param) combination.
    """
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    if yymms is None:
        yymms = discover_yymms_from_enh()

    splits = build_overlay_splits_from_yymms(
        yymms,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        min_test_months=min_test_months,
    )
    if not splits:
        raise RuntimeError("No valid overlay splits constructed; check train/test/step parameters.")

    # Snapshot current cr_config params once at top-level (for main process)
    base_snap = _snapshot_cr_params()

    results: List[Dict[str, object]] = []

    try:
        # Build all tasks
        tasks = []
        for s in splits:
            for ze in z_entry_grid:
                for zx in z_exit_grid:
                    for zs in z_stop_grid:
                        for mh in max_hold_grid:
                            tasks.append((s, float(ze), float(zx), float(zs), int(mh)))

        if n_jobs == 1:
            # Sequential
            for (s, ze, zx, zs, mh) in tasks:
                print(
                    f"[SPLIT {s.split_id}] "
                    f"params: z_entry={ze}, z_exit={zx}, z_stop={zs}, max_hold_days={mh}"
                )
                row = _eval_param_combo_for_split(
                    s,
                    hedge_df=hedge_df,
                    z_entry=ze,
                    z_exit=zx,
                    z_stop=zs,
                    max_hold_days=mh,
                    decision_freq=decision_freq,
                )
                results.append(row)
        else:
            # Parallel with ProcessPoolExecutor
            # Note: cr_config is imported in each worker process; we still snapshot/restore there.
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                future_to_task = {
                    ex.submit(
                        _eval_param_combo_for_split,
                        s,
                        hedge_df,
                        ze,
                        zx,
                        zs,
                        mh,
                        decision_freq,
                    ): (s.split_id, ze, zx, zs, mh)
                    for (s, ze, zx, zs, mh) in tasks
                }

                for fut in as_completed(future_to_task):
                    split_id, ze, zx, zs, mh = future_to_task[fut]
                    try:
                        row = fut.result()
                        results.append(row)
                        print(
                            f"[DONE] split={split_id}, "
                            f"z_entry={ze}, z_exit={zx}, z_stop={zs}, max_hold_days={mh}"
                        )
                    except Exception as e:
                        print(
                            f"[ERROR] split={split_id}, params=({ze},{zx},{zs},{mh}): {e}"
                        )

    finally:
        # Ensure we restore base cr_config params in the main process
        _restore_cr_params(base_snap)

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(
            ["split_id", "test_pnl_cash", "train_pnl_cash"],
            ascending=[True, False, False],
        )

    if out_name:
        out_dir = Path(getattr(cr, "PATH_OUT", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = getattr(cr, "OUT_SUFFIX", "")
        fname = f"{out_name}{suffix}.parquet" if suffix else f"{out_name}.parquet"
        out_path = out_dir / fname
        res_df.to_parquet(out_path, index=False)
        print(f"[WRITE] overlay grid-search results -> {out_path}")

    return res_df


# ======================================================================
# 7) CLI example
# ======================================================================

if __name__ == "__main__":
    # Example usage with synthetic tape:
    #   - Assumes you have a synthetic trades file like 'trades_synth.pkl'
    #     in the working directory (adjust as needed).
    synth_path = Path("trades_synth.pkl")
    if not synth_path.exists():
        raise FileNotFoundError(
            f"Synthetic trades file not found: {synth_path} "
            "(adjust path in overlay_tscv_from_portfolio.py __main__)."
        )

    hedge_df_synth = pd.read_pickle(synth_path)

    # Small example grid – replace with your real search space
    z_entry_grid = [0.75, 1.0, 1.25]
    z_exit_grid = [0.30, 0.40]
    z_stop_grid = [1.5, 2.0]
    max_hold_grid = [5, 9]

    # Discover months and build splits (6m train, 3m test, rolling every 3m)
    yymms = discover_yymms_from_enh()
    describe_overlay_splits(
        yymms,
        train_months=6,
        test_months=3,
        step_months=3,
        min_test_months=2,
    )

    df_res = run_overlay_grid_search_from_portfolio(
        hedge_df=hedge_df_synth,
        z_entry_grid=z_entry_grid,
        z_exit_grid=z_exit_grid,
        z_stop_grid=z_stop_grid,
        max_hold_grid=max_hold_grid,
        yymms=yymms,
        decision_freq="D",
        train_months=6,
        test_months=3,
        step_months=3,
        min_test_months=2,
        n_jobs=1,  # bump this if you want parallel evaluation
        out_name="overlay_grid_synth",
    )

    # Show best params per split by test_pnl_cash
    best_per_split = df_res.groupby("split_id").head(1)
    print("\nBest params per split (by test_pnl_cash):")
    print(best_per_split)