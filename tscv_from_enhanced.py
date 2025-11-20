"""
tscv_from_enhanced.py

Time-series cross-validation on the enhanced curve panel.

- Build decision calendar from enhanced files.
- Define rolling train/test splits.
- Run a simplified cheap–rich backtest (no caps) on the TRAIN window only,
  governed solely by (z_entry, z_exit, z_stop, max_hold_days, max_concurrent).
- Optimize these parameters per split, outputting a parquet table of
  (split_id, date ranges, params, train_pnl_bp).

This is meant as a *training driver* from enhanced only (no hedge tape).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
from numba import njit

import cr_config as cr


# ======================================================================
# 1) Enhanced discovery + decision calendar
# ======================================================================

def _iter_enhanced_paths_tscv() -> List[Path]:
    """
    Discover enhanced parquet files, mirroring portfolio_test_new logic.
    """
    root = Path(getattr(cr, "PATH_ENH", "."))
    suffix = getattr(cr, "ENH_SUFFIX", "")
    if suffix:
        return sorted(root.glob(f"*{suffix}.parquet"))
    else:
        # reasonably permissive fallback
        return sorted(root.glob("*.parquet"))


def build_decision_calendar(decision_freq: str | None = None) -> pd.DatetimeIndex:
    """
    Scan all enhanced files, reconstruct decision buckets, and return the
    full sorted decision_ts index (no duplicates).

    This is the canonical calendar for TSCV.
    """
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    paths = _iter_enhanced_paths_tscv()
    if not paths:
        raise FileNotFoundError(f"No enhanced files found under {cr.PATH_ENH}")

    all_decisions: list[pd.Series] = []

    for pth in paths:
        df = pd.read_parquet(pth)
        if df.empty or "ts" not in df.columns:
            continue
        ts = pd.to_datetime(df["ts"], utc=False, errors="coerce")
        if decision_freq == "D":
            dec = ts.dt.floor("d")
        elif decision_freq == "H":
            dec = ts.dt.floor("h")
        else:
            raise ValueError("DECISION_FREQ must be 'D' or 'H'.")
        all_decisions.append(dec.dropna())

    if not all_decisions:
        raise RuntimeError("No decision_ts could be built from enhanced files.")

    decisions = pd.concat(all_decisions).dropna().sort_values().unique()
    decisions = pd.DatetimeIndex(decisions)
    return decisions


@dataclass
class SplitSpec:
    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_time_splits_from_calendar(
    decisions: pd.DatetimeIndex,
    *,
    train_days: int = 365,
    test_days: int = 90,
    step_days: int = 90,
) -> List[SplitSpec]:
    """
    Given a full decision_ts calendar, build rolling time-based splits.

    - train_days: length of the in-sample window in calendar days
    - test_days : length of the OOS window in calendar days
    - step_days : how far to roll the window between splits

    Returns a list of SplitSpec objects.
    """
    if decisions.empty:
        return []

    start_date = decisions.min().normalize()
    end_date = decisions.max().normalize()

    splits: List[SplitSpec] = []
    cur_train_start = start_date
    split_id = 1

    while True:
        train_end = cur_train_start + pd.Timedelta(days=train_days - 1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)

        # Stop if test window would start beyond data
        if test_start > end_date:
            break

        # Clip test_end to available data
        if test_end > end_date:
            test_end = end_date

        splits.append(
            SplitSpec(
                split_id=split_id,
                train_start=cur_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        split_id += 1

        # Move forward
        cur_train_start = cur_train_start + pd.Timedelta(days=step_days)
        # If moving forward leaves no room for a full train window, stop
        if cur_train_start + pd.Timedelta(days=train_days) > end_date:
            break

    return splits


def describe_splits(
    decision_freq: str | None = None,
    train_days: int = 365,
    test_days: int = 90,
    step_days: int = 90,
) -> List[SplitSpec]:
    """
    Helper to quickly see how many splits you get and their date ranges.
    """
    decisions = build_decision_calendar(decision_freq=decision_freq)
    splits = build_time_splits_from_calendar(
        decisions,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )

    print(f"Total decision buckets: {len(decisions)}")
    print(f"From {decisions.min().date()} to {decisions.max().date()}")
    print(f"train_days={train_days}, test_days={test_days}, step_days={step_days}")
    print(f"Number of splits: {len(splits)}\n")

    for s in splits:
        print(
            f"Split {s.split_id}: "
            f"train {s.train_start.date()} → {s.train_end.date()} | "
            f"test {s.test_start.date()} → {s.test_end.date()}"
        )

    return splits


# ======================================================================
# 2) Load enhanced once and build per-bucket cross-section snapshots
# ======================================================================

def _to_float(x, default=np.nan) -> float:
    try:
        if isinstance(x, (pd.Series, pd.Index)):
            if len(x) == 0:
                return default
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return default


def load_enhanced_panel_all(
    decision_freq: str | None = None,
) -> Tuple[pd.DatetimeIndex, Dict[pd.Timestamp, pd.DataFrame]]:
    """
    Load all enhanced parquet files once, compute decision_ts for each row,
    collapse to last tick per (decision_ts, tenor_yrs), and return:

      decisions : sorted unique decision_ts index
      snapshots : dict[decision_ts -> DataFrame with columns
                       ['tenor_yrs', 'rate', 'z_comb'] (and possibly others)]

    This mirrors the way portfolio_test_new builds snap_last.
    """
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    paths = _iter_enhanced_paths_tscv()
    if not paths:
        raise FileNotFoundError(f"No enhanced files found under {cr.PATH_ENH}")

    frames = []
    for pth in paths:
        df = pd.read_parquet(pth)
        if df.empty:
            continue
        if not {"ts", "tenor_yrs", "rate", "z_comb"}.issubset(df.columns):
            continue
        df = df[["ts", "tenor_yrs", "rate", "z_comb"]].copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")

        if decision_freq == "D":
            df["decision_ts"] = df["ts"].dt.floor("d")
        elif decision_freq == "H":
            df["decision_ts"] = df["ts"].dt.floor("h")
        else:
            raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

        frames.append(df)

    if not frames:
        raise RuntimeError("No suitable enhanced rows found for TSCV.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["decision_ts", "tenor_yrs", "rate", "z_comb"])

    # Last tick per tenor per decision bucket
    all_df = (
        all_df.sort_values("ts")
              .groupby(["decision_ts", "tenor_yrs"], as_index=False)
              .tail(1)
    )

    snapshots: Dict[pd.Timestamp, pd.DataFrame] = {}
    for dts, snap in all_df.groupby("decision_ts", sort=True):
        snapshots[dts] = snap.reset_index(drop=True)

    decisions = pd.DatetimeIndex(sorted(snapshots.keys()))
    return decisions, snapshots


# ======================================================================
# 3) Numba-accelerated panel simulator (multiple concurrent pairs)
# ======================================================================

@njit
def simulate_params_panel(
    z_panel: np.ndarray,
    rate_panel: np.ndarray,
    tenors: np.ndarray,
    z_entry: float,
    z_exit: float,
    z_stop: float,
    max_hold_days: int,
    decisions_per_day: int,
    min_leg_tenor: float,
    min_sep_years: float,
    max_span_years: float,
    max_concurrent: int,
) -> float:
    """
    Numba core:

    Inputs:
      - z_panel:   shape (T, N), z_comb per bucket × tenor
      - rate_panel:shape (T, N), rate (%) per bucket × tenor
      - tenors:    shape (N,), tenor_yrs per column
      - z_entry, z_exit, z_stop, max_hold_days: strategy params
      - decisions_per_day: currently not used (assume daily buckets)
      - min_leg_tenor, min_sep_years, max_span_years: tenor constraints
      - max_concurrent: max number of concurrent pairs

    Logic:
      - Greedy open up to max_concurrent pairs per bucket based on zspread,
        respecting tenor separation and tenor reuse within that bucket.
      - Long cheap (higher z), short rich (lower z).
      - PnL in bps per unit notional (no DV01 scaling).
    """
    T, N = z_panel.shape
    max_pos = max_concurrent

    open_flag = np.zeros(max_pos, dtype=np.bool_)
    tenor_i_idx = np.zeros(max_pos, dtype=np.int64)
    tenor_j_idx = np.zeros(max_pos, dtype=np.int64)
    entry_rate_i = np.zeros(max_pos, dtype=np.float64)
    entry_rate_j = np.zeros(max_pos, dtype=np.float64)
    entry_zspread = np.zeros(max_pos, dtype=np.float64)
    age = np.zeros(max_pos, dtype=np.int64)

    total_pnl_bp = 0.0

    for t in range(T):
        # 1) Mark existing positions and decide exits
        for k in range(max_pos):
            if not open_flag[k]:
                continue

            i = tenor_i_idx[k]
            j = tenor_j_idx[k]

            zi = z_panel[t, i]
            zj = z_panel[t, j]
            ri = rate_panel[t, i]
            rj = rate_panel[t, j]

            # If we don't have a valid mark, just age and possibly max-hold exit
            if not (np.isfinite(zi) and np.isfinite(zj) and np.isfinite(ri) and np.isfinite(rj)):
                age[k] += 1
                if age[k] >= max_hold_days:
                    open_flag[k] = False
                continue

            # PnL in bps relative to entry
            d_i = (entry_rate_i[k] - ri) * 100.0
            d_j = (entry_rate_j[k] - rj) * 100.0
            pnl_bp = d_i - d_j  # long cheap (i), short rich (j)

            # Exit logic (directional z)
            z_entry_val = entry_zspread[k]
            exit_flag = False

            if np.isfinite(z_entry_val):
                # directional sign from entry z
                if z_entry_val > 0.0:
                    dir_sign = 1.0
                elif z_entry_val < 0.0:
                    dir_sign = -1.0
                else:
                    dir_sign = 1.0

                entry_dir = dir_sign * z_entry_val
                curr_dir = dir_sign * (zi - zj)

                same_side = (entry_dir != 0.0) and (entry_dir * curr_dir > 0.0)
                moved_towards_zero = abs(curr_dir) <= abs(entry_dir)
                within_exit_band = abs(curr_dir) <= z_exit
                dz_dir = curr_dir - entry_dir
                moved_away = (
                    same_side
                    and (abs(curr_dir) >= abs(entry_dir))
                    and (abs(dz_dir) >= z_stop)
                )

                if same_side and moved_towards_zero and within_exit_band:
                    exit_flag = True
                elif moved_away:
                    exit_flag = True

            # Max-hold
            age[k] += 1
            if (not exit_flag) and (age[k] >= max_hold_days):
                exit_flag = True

            if exit_flag:
                open_flag[k] = False
                total_pnl_bp += pnl_bp

        # 2) Compute open_count and tenor usage
        open_count = 0
        tenor_used = np.zeros(N, dtype=np.bool_)
        for k in range(max_pos):
            if open_flag[k]:
                open_count += 1
                i = tenor_i_idx[k]
                j = tenor_j_idx[k]
                if 0 <= i < N:
                    tenor_used[i] = True
                if 0 <= j < N:
                    tenor_used[j] = True

        capacity = max_concurrent - open_count
        if capacity <= 0:
            continue

        # 3) New entries: greedy fill capacity pairs
        for _ in range(capacity):
            best_i = -1
            best_j = -1
            best_zdisp = z_entry

            for i in range(N):
                if tenor_used[i]:
                    continue
                Ti = tenors[i]
                if not np.isfinite(Ti) or (Ti < min_leg_tenor):
                    continue

                zi = z_panel[t, i]
                ri = rate_panel[t, i]
                if not (np.isfinite(zi) and np.isfinite(ri)):
                    continue

                for j in range(N):
                    if j == i or tenor_used[j]:
                        continue
                    Tj = tenors[j]
                    if not np.isfinite(Tj) or (Tj < min_leg_tenor):
                        continue

                    diff = abs(Ti - Tj)
                    if diff < min_sep_years or diff > max_span_years:
                        continue

                    zj = z_panel[t, j]
                    rj = rate_panel[t, j]
                    if not (np.isfinite(zj) and np.isfinite(rj)):
                        continue

                    # cheap = higher z, rich = lower z
                    zdisp = zi - zj
                    if zdisp < z_entry:
                        continue

                    if zdisp > best_zdisp:
                        best_zdisp = zdisp
                        best_i = i
                        best_j = j

            if best_i == -1:
                break  # no more candidates

            # Open new position in first free slot
            for k in range(max_pos):
                if not open_flag[k]:
                    open_flag[k] = True
                    tenor_i_idx[k] = best_i
                    tenor_j_idx[k] = best_j
                    entry_rate_i[k] = rate_panel[t, best_i]
                    entry_rate_j[k] = rate_panel[t, best_j]
                    entry_zspread[k] = z_panel[t, best_i] - z_panel[t, best_j]
                    age[k] = 0
                    break

            tenor_used[best_i] = True
            tenor_used[best_j] = True

    return total_pnl_bp


# ======================================================================
# 4) Wrapper: evaluate params on a train window
# ======================================================================

def evaluate_params_on_window(
    decisions: pd.DatetimeIndex,
    snapshots: Dict[pd.Timestamp, pd.DataFrame],
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    z_entry: float,
    z_exit: float,
    z_stop: float,
    max_hold_days: int,
    min_sep_years: float = 0.5,
    min_leg_tenor: float = 0.0,
    max_concurrent: int = 3,
) -> float:
    """
    Evaluate a simple cheap–rich strategy on [train_start, train_end]
    using given parameters, with up to max_concurrent concurrent pairs.

    - Uses a numba-accelerated panel simulator (simulate_params_panel).
    - Returns total PnL in bps (per unit, no DV01 scaling).
    """
    # restrict to train range
    mask = (decisions >= train_start) & (decisions <= train_end)
    dts_list = decisions[mask]
    if len(dts_list) == 0:
        return 0.0

    # Build tenor universe: intersection of tenors across the window
    tenor_sets = []
    for dts in dts_list:
        snap = snapshots.get(dts)
        if snap is None or snap.empty:
            continue
        tenor_sets.append(set(snap["tenor_yrs"].astype(float)))

    if not tenor_sets:
        return 0.0

    common = tenor_sets[0]
    for s in tenor_sets[1:]:
        common = common & s
    if not common:
        return 0.0

    tenors = np.array(sorted(common), dtype=np.float64)
    T = len(dts_list)
    N = len(tenors)

    z_panel = np.full((T, N), np.nan, dtype=np.float64)
    rate_panel = np.full((T, N), np.nan, dtype=np.float64)

    # Fill panels
    for ti, dts in enumerate(dts_list):
        snap = snapshots.get(dts)
        if snap is None or snap.empty:
            continue
        s = snap[["tenor_yrs", "rate", "z_comb"]].dropna().copy()
        if s.empty:
            continue

        # vectorized-ish: map tenor -> row
        for j in range(N):
            Tj = tenors[j]
            rows = s.loc[s["tenor_yrs"] == Tj]
            if rows.empty:
                continue
            r = rows.iloc[0]
            rate_panel[ti, j] = float(r["rate"])
            z_panel[ti, j] = float(r["z_comb"])

    decisions_per_day = 1  # we are using daily buckets for TSCV

    pnl_bp = simulate_params_panel(
        z_panel,
        rate_panel,
        tenors,
        float(z_entry),
        float(z_exit),
        float(z_stop),
        int(max_hold_days),
        int(decisions_per_day),
        float(min_leg_tenor),
        float(min_sep_years),
        float(getattr(cr, "MAX_SPAN_YEARS", 10.0)),
        int(max_concurrent),
    )

    return float(pnl_bp)


# ======================================================================
# 5) Grid optimizer over splits
# ======================================================================

def run_tscv_optimization_from_enhanced(
    z_entry_grid: Iterable[float],
    z_exit_grid: Iterable[float],
    z_stop_grid: Iterable[float],
    max_hold_grid: Iterable[int],
    *,
    decision_freq: str | None = None,
    train_days: int = 365,
    test_days: int = 90,
    step_days: int = 90,
    min_sep_years: float | None = None,
    min_leg_tenor: float | None = None,
    max_concurrent_train: int = 3,
    out_name: str | None = None,
) -> pd.DataFrame:
    """
    Main TSCV driver:

    - Build decision calendar and splits.
    - Load enhanced + snapshots once.
    - For each split and each parameter tuple, evaluate train PnL in bps.
    - Return a DataFrame with one row per (split, param) combo and optional
      parquet dump to PATH_OUT/out_name.parquet.

    Arguments:
      z_entry_grid      : candidate z_entry thresholds
      z_exit_grid       : candidate z_exit thresholds
      z_stop_grid       : candidate z_stop thresholds
      max_hold_grid     : candidate max_hold_days
      decision_freq     : 'D' or 'H' (use 'D' for now)
      train_days        : length of train window in calendar days
      test_days         : length of test window
      step_days         : roll step between splits
      min_sep_years     : override MIN_SEP_YEARS from cr_config if not None
      min_leg_tenor     : override MIN_LEG_TENOR_YEARS from cr_config if not None
      max_concurrent_train : max concurrent pairs during training sim
      out_name          : base filename (without suffix) for parquet; if None, no write.
    """
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    decisions_all, snapshots = load_enhanced_panel_all(decision_freq=decision_freq)

    # Use calendar based on what's actually in snapshots
    decisions = decisions_all.sort_values().unique()
    decisions = pd.DatetimeIndex(decisions)

    splits = build_time_splits_from_calendar(
        decisions,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    if not splits:
        raise RuntimeError("No valid splits constructed; check train/test/step days.")

    # Knobs (fall back to cr_config if not provided)
    if min_sep_years is None:
        min_sep_years = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    if min_leg_tenor is None:
        min_leg_tenor = float(getattr(cr, "MIN_LEG_TENOR_YEARS", 0.0))

    results: List[Dict] = []

    for s in splits:
        print(
            f"[SPLIT {s.split_id}] "
            f"train {s.train_start.date()} → {s.train_end.date()} | "
            f"test {s.test_start.date()} → {s.test_end.date()}"
        )

        for ze in z_entry_grid:
            for zx in z_exit_grid:
                for zs in z_stop_grid:
                    for mh in max_hold_grid:
                        pnl_bp = evaluate_params_on_window(
                            decisions,
                            snapshots,
                            train_start=s.train_start,
                            train_end=s.train_end,
                            z_entry=float(ze),
                            z_exit=float(zx),
                            z_stop=float(zs),
                            max_hold_days=int(mh),
                            min_sep_years=min_sep_years,
                            min_leg_tenor=min_leg_tenor,
                            max_concurrent=int(max_concurrent_train),
                        )

                        results.append(
                            {
                                "split_id": s.split_id,
                                "train_start": s.train_start,
                                "train_end": s.train_end,
                                "test_start": s.test_start,
                                "test_end": s.test_end,
                                "z_entry": float(ze),
                                "z_exit": float(zx),
                                "z_stop": float(zs),
                                "max_hold_days": int(mh),
                                "max_concurrent": int(max_concurrent_train),
                                "train_pnl_bp": float(pnl_bp),
                            }
                        )

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(["split_id", "train_pnl_bp"], ascending=[True, False])

    if out_name:
        out_dir = Path(getattr(cr, "PATH_OUT", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = getattr(cr, "OUT_SUFFIX", "")
        fname = f"{out_name}{suffix}.parquet" if suffix else f"{out_name}.parquet"
        out_path = out_dir / fname
        res_df.to_parquet(out_path, index=False)
        print(f"[WRITE] TSCV optimization results -> {out_path}")

    return res_df


# ======================================================================
# 6) CLI / quick usage example
# ======================================================================

if __name__ == "__main__":
    # Example: small grid, daily decisions, 1y train, 3m test, roll 3m
    z_entry_grid = [0.75, 1.0, 1.25]
    z_exit_grid = [0.25, 0.40]
    z_stop_grid = [1.5, 2.0]
    max_hold_grid = [5, 10]

    df_res = run_tscv_optimization_from_enhanced(
        z_entry_grid,
        z_exit_grid,
        z_stop_grid,
        max_hold_grid,
        decision_freq="D",
        train_days=365,
        test_days=90,
        step_days=90,
        max_concurrent_train=3,
        out_name="tscv_params_enh",
    )

    # Show best combo per split
    best_per_split = df_res.groupby("split_id").head(1)
    print("\nBest params per split (by train_pnl_bp):")
    print(best_per_split)