"""
tscv_from_enhanced.py

Time-series cross-validation on the enhanced curve panel.

- Build decision calendar from enhanced files.
- Define rolling train/test splits.
- Run a simplified cheap–rich backtest (no caps) on the TRAIN window only,
  governed solely by (z_entry, z_exit, z_stop, max_hold_days).
- Optimize these parameters per split, outputing a parquet table of
  (split_id, date ranges, params, train_pnl_bp, n_trades).

This is meant as a *training driver* from enhanced only (no hedge tape).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

import numpy as np
import pandas as pd

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
                train_start=train_days and train_end and cur_train_start,
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
# 3) Simplified cheap–rich strategy from enhanced only
# ======================================================================

@dataclass
class SimplePos:
    tenor_i: float
    tenor_j: float
    entry_rate_i: float
    entry_rate_j: float
    entry_zspread: float
    open_ts: pd.Timestamp
    age: int = 0  # in decision buckets


def _choose_best_pair_for_snapshot(
    snap: pd.DataFrame,
    z_entry: float,
    *,
    min_sep_years: float = 0.5,
    min_leg_tenor: float = 0.0,
) -> SimplePos | None:
    """
    Given one cross-sectional snapshot (tenor_yrs, rate, z_comb),
    find the single best cheap–rich pair with zspread >= z_entry.

    - We ignore caps and all other desk knobs.
    - Return None if no pair satisfies z_entry.
    """
    if snap.empty:
        return None

    s = snap[["tenor_yrs", "rate", "z_comb"]].dropna().copy()
    if s.empty:
        return None

    ten = s["tenor_yrs"].to_numpy(float)
    rate = s["rate"].to_numpy(float)
    z = s["z_comb"].to_numpy(float)
    n = len(s)

    best_zdisp = -np.inf
    best_pair = None

    for i in range(n):
        Ti, ri, zi = ten[i], rate[i], z[i]
        if Ti < min_leg_tenor:
            continue
        for j in range(n):
            if j == i:
                continue
            Tj, rj, zj = ten[j], rate[j], z[j]
            if Tj < min_leg_tenor:
                continue
            if abs(Ti - Tj) < min_sep_years:
                continue

            # zspread = cheap - rich
            zdisp = zi - zj
            if zdisp < z_entry:
                continue
            if zdisp > best_zdisp:
                best_zdisp = zdisp
                best_pair = (Ti, Tj, ri, rj, zdisp)

    if best_pair is None:
        return None

    Ti, Tj, ri, rj, zsp = best_pair
    pos = SimplePos(
        tenor_i=float(Ti),
        tenor_j=float(Tj),
        entry_rate_i=float(ri),
        entry_rate_j=float(rj),
        entry_zspread=float(zsp),
        open_ts=pd.NaT,   # will be set by caller
        age=0,
    )
    return pos


def _mark_and_maybe_exit(
    pos: SimplePos,
    dts: pd.Timestamp,
    snap: pd.DataFrame,
    *,
    z_exit: float,
    z_stop: float,
    max_hold_days: int,
) -> Tuple[SimplePos | None, float]:
    """
    Mark position at decision bucket dts using snapshot snap, and decide if it exits.

    - PnL is measured in bps (per unit notional).
    - Uses simplified directional z logic similar to portfolio_test_new:
        * reversion: moves toward zero and inside |z_exit|
        * stop: moves away from entry by at least z_stop
        * max_hold: age >= max_hold_days (in decisions; daily => days)
    """
    # get current rates + zspread
    ti, tj = pos.tenor_i, pos.tenor_j

    def _lookup(s: pd.DataFrame, tenor: float) -> Tuple[float, float]:
        r = s.loc[s["tenor_yrs"] == tenor]
        if r.empty:
            # fall back to nearest tenor if exact not present
            s2 = s.assign(_dist=(s["tenor_yrs"] - tenor).abs())
            row = s2.loc[s2["_dist"].idxmin()]
        else:
            row = r.iloc[0]
        return float(row["rate"]), float(row["z_comb"])

    ri, zi_cur = _lookup(snap, ti)
    rj, zj_cur = _lookup(snap, tj)
    zsp_cur = zi_cur - zj_cur

    # PnL in bp relative to entry
    d_i = (pos.entry_rate_i - ri) * 100.0
    d_j = (pos.entry_rate_j - rj) * 100.0
    pnl_bp = d_i - d_j  # long cheap (i), short rich (j)

    # Exit logic
    entry_z = pos.entry_zspread
    exit_flag = False

    if np.isfinite(zsp_cur) and np.isfinite(entry_z):
        # directional z
        dir_sign = np.sign(entry_z) if entry_z != 0 else 1.0
        entry_dir = dir_sign * entry_z
        curr_dir = dir_sign * zsp_cur

        same_side = (np.sign(entry_dir) != 0) and (np.sign(entry_dir) == np.sign(curr_dir))
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
    pos.age += 1
    if not exit_flag and pos.age >= max_hold_days:
        exit_flag = True

    if exit_flag:
        # position closes here
        return None, pnl_bp
    else:
        return pos, pnl_bp


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
) -> Tuple[float, int]:
    """
    Evaluate a simple cheap–rich strategy on [train_start, train_end]
    using given parameters. Returns (total_pnl_bp, n_trades).

    - Only one position at a time (for simplicity).
    - No caps, no fly gating, no overlay complexity.
    - Uses snapshots per decision_ts.
    """
    # restrict to train range
    mask = (decisions >= train_start) & (decisions <= train_end)
    dts_list = decisions[mask]
    if len(dts_list) == 0:
        return 0.0, 0

    open_pos: SimplePos | None = None
    total_pnl_bp = 0.0
    n_trades = 0

    for dts in dts_list:
        snap = snapshots.get(dts)
        if snap is None or snap.empty:
            continue

        # Mark existing position
        if open_pos is not None:
            open_pos, pnl_bp = _mark_and_maybe_exit(
                open_pos,
                dts,
                snap,
                z_exit=z_exit,
                z_stop=z_stop,
                max_hold_days=max_hold_days,
            )
            total_pnl_bp += pnl_bp
            if open_pos is None:
                n_trades += 1  # we just closed one

        # If flat after marking, consider new entry
        if open_pos is None:
            new_pos = _choose_best_pair_for_snapshot(
                snap,
                z_entry=z_entry,
                min_sep_years=min_sep_years,
                min_leg_tenor=min_leg_tenor,
            )
            if new_pos is not None:
                new_pos.open_ts = dts
                open_pos = new_pos

    # If we end the train window with an open position, we *mark* it at last day
    # (already done in the loop). We do NOT force an extra closure beyond train_end
    # to keep it strictly in-sample.
    return float(total_pnl_bp), int(n_trades)


# ======================================================================
# 4) Grid optimizer over splits
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
      z_entry_grid : candidate z_entry thresholds
      z_exit_grid  : candidate z_exit thresholds
      z_stop_grid  : candidate z_stop thresholds
      max_hold_grid: candidate max_hold_days
      decision_freq: 'D' or 'H' (use 'D' for now)
      train_days   : length of train window in calendar days
      test_days    : length of test window
      step_days    : roll step between splits
      min_sep_years: override MIN_SEP_YEARS from cr_config if not None
      min_leg_tenor: override MIN_LEG_TENOR_YEARS from cr_config if not None
      out_name     : base filename (without suffix) for parquet; if None, no write.
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
                        pnl_bp, n_trades = evaluate_params_on_window(
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
                                "train_pnl_bp": float(pnl_bp),
                                "n_trades": int(n_trades),
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
# 5) CLI / quick usage example
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
        out_name="tscv_params_enh",
    )

    # Show best combo per split
    best_per_split = df_res.groupby("split_id").head(1)
    print("\nBest params per split (by train_pnl_bp):")
    print(best_per_split)