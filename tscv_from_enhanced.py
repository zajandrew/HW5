"""
tscv_from_enhanced.py

Time-series CV from ENHANCED curve data.

Pipeline
--------
1) Load ALL enhanced daily panels from PATH_ENH.
2) Build a pair panel over tenors:
       z_panel[i, t]      = z_dir for pair i at day t
       spread_panel[i, t] = directional spread in bp per unit DV01 for pair i at day t
   where pair i = (tenor_a, tenor_b), a < b, subject to tenor constraints.
3) Generate rolling time-series splits: train and test segments.
4) For each train segment, run the Numba grid search from tscv.run_param_grid
   and record the best parameters.
5) Save a parquet of split-wise best params under PATH_OUT.

This file DOES NOT touch your real strategy engine; it's purely for
parameter training on a simplified, vectorized reversion model.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import cr_config as cr
from tscv import run_param_grid, ReversionParams


# ----------------------------------------------------------------------
# 1) Load all enhanced daily data
# ----------------------------------------------------------------------

def _iter_enhanced_paths() -> List[Path]:
    root = Path(cr.PATH_ENH)
    suffix = getattr(cr, "ENH_SUFFIX", "")
    pattern = f"*{suffix}.parquet" if suffix else "*.parquet"
    return sorted(root.glob(pattern))


def load_all_enhanced_daily() -> pd.DataFrame:
    """
    Load all enhanced parquet files and collapse to *daily* last-tick
    per tenor_yrs per decision_ts.

    Returns a DataFrame with at least:
        ['decision_ts', 'tenor_yrs', 'rate', 'z_comb']
    """
    paths = _iter_enhanced_paths()
    if not paths:
        raise FileNotFoundError(f"No enhanced files found under {cr.PATH_ENH}")

    all_rows = []
    for pth in paths:
        df = pd.read_parquet(pth)
        if df.empty:
            continue

        # Basic checks
        need = {"ts", "tenor_yrs", "rate", "z_comb"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{pth} missing columns: {missing}")

        df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
        # Daily decision buckets (we are training on daily)
        df["decision_ts"] = df["ts"].dt.floor("D")

        # Last tick per tenor per decision_ts
        df_last = (
            df.sort_values("ts")
              .groupby(["decision_ts", "tenor_yrs"], as_index=False)
              .tail(1)
        )[["decision_ts", "tenor_yrs", "rate", "z_comb"]]

        all_rows.append(df_last)

    if not all_rows:
        raise RuntimeError("No enhanced rows accumulated; check PATH_ENH and contents.")

    out = pd.concat(all_rows, ignore_index=True)
    out = out.dropna(subset=["decision_ts", "tenor_yrs", "rate", "z_comb"])
    out["decision_ts"] = pd.to_datetime(out["decision_ts"], utc=False)
    out = out.sort_values(["decision_ts", "tenor_yrs"]).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------
# 2) Build pair panel (z_panel, spread_panel) from enhanced
# ----------------------------------------------------------------------

def build_pair_panel_from_enhanced(
    df: pd.DataFrame,
    *,
    min_sep_years: Optional[float] = None,
    max_span_years: Optional[float] = None,
    min_leg_tenor: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]], np.ndarray]:
    """
    df: output of load_all_enhanced_daily.

    Returns
    -------
    z_panel : np.ndarray, shape (N_pairs, T)
    spread_panel : np.ndarray, shape (N_pairs, T)
    pairs : list of (tenor_a, tenor_b)
    dts : np.ndarray of decision_ts (length T)

    Conventions
    -----------
    For each pair (a, b) with a < b:
        z_dir(t)   = z_a(t) - z_b(t)
        spread(t)  = (rate_a(t) - rate_b(t)) * 100.0   # bp per unit DV01

    This is consistent with a “cheap-rich” orientation where positive z_dir
    means leg a is "cheap" vs leg b and we are effectively short the spread.
    """
    df = df.copy()
    df["tenor_yrs"] = df["tenor_yrs"].astype(float)

    # Tenor constraints: use cr values if not provided
    if min_sep_years is None:
        min_sep_years = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    if max_span_years is None:
        max_span_years = float(getattr(cr, "MAX_SPAN_YEARS", 10.0))
    if min_leg_tenor is None:
        min_leg_tenor = float(getattr(cr, "MIN_LEG_TENOR_YEARS", 0.0))

    # Full tenor universe
    tenors = np.sort(df["tenor_yrs"].unique())
    # Filter by MIN_LEG_TENOR
    tenors = tenors[tenors >= min_leg_tenor]
    if tenors.size < 2:
        raise RuntimeError("Not enough tenors after MIN_LEG_TENOR filter to form pairs.")

    # All calendar of decision_ts
    dts = np.sort(df["decision_ts"].unique())
    T = dts.size

    # Build tenor x time matrices for rate and z
    tenor_to_idx: Dict[float, int] = {float(t): i for i, t in enumerate(tenors)}
    n_tenors = tenors.size

    rate_mat = np.full((n_tenors, T), np.nan, dtype=float)
    z_mat = np.full((n_tenors, T), np.nan, dtype=float)

    # Efficiently fill matrices
    # Map decision_ts -> column index
    dt_to_col: Dict[pd.Timestamp, int] = {dts[i]: i for i in range(T)}

    for _, row in df.iterrows():
        t = float(row["tenor_yrs"])
        d = row["decision_ts"]
        i = tenor_to_idx.get(t, None)
        j = dt_to_col.get(d, None)
        if i is None or j is None:
            continue
        rate_mat[i, j] = float(row["rate"])
        z_mat[i, j] = float(row["z_comb"])

    # Build pair list subject to tenor constraints
    pairs: List[Tuple[float, float]] = []
    for idx_a in range(n_tenors):
        for idx_b in range(idx_a + 1, n_tenors):
            ta = tenors[idx_a]
            tb = tenors[idx_b]
            diff = abs(ta - tb)
            if diff < min_sep_years or diff > max_span_years:
                continue
            pairs.append((ta, tb))

    if not pairs:
        raise RuntimeError("No tenor pairs passed separation/span constraints.")

    n_pairs = len(pairs)
    z_panel = np.full((n_pairs, T), np.nan, dtype=float)
    spread_panel = np.full((n_pairs, T), np.nan, dtype=float)

    # Fill panels
    for k, (ta, tb) in enumerate(pairs):
        ia = tenor_to_idx[ta]
        ib = tenor_to_idx[tb]

        z_a = z_mat[ia, :]
        z_b = z_mat[ib, :]
        r_a = rate_mat[ia, :]
        r_b = rate_mat[ib, :]

        # directional z and spread
        z_dir = z_a - z_b
        spread = (r_a - r_b) * 100.0  # bp per unit DV01

        z_panel[k, :] = z_dir
        spread_panel[k, :] = spread

    return z_panel, spread_panel, pairs, dts


# ----------------------------------------------------------------------
# 3) Time-series splits
# ----------------------------------------------------------------------

def make_time_splits(
    dts: np.ndarray,
    *,
    train_len: int = 252,  # trading days
    test_len: int = 63,
    step: int = 63,
) -> List[Dict]:
    """
    Build rolling time-series splits on a DAILY calendar.

    dts : sorted np.ndarray of Timestamp

    Returns a list of dicts, each with:
        {
            "split_id": int,
            "train_start": Timestamp,
            "train_end": Timestamp,
            "test_start": Timestamp,
            "test_end": Timestamp,
            "train_idx": np.ndarray[bool],  # mask over dts
            "test_idx": np.ndarray[bool],
        }

    For now we only use train_idx to fit parameters, but test segments
    are defined for later evaluation.
    """
    n = dts.size
    splits: List[Dict] = []
    split_id = 0

    start_idx = 0
    while True:
        train_start_idx = start_idx
        train_end_idx = train_start_idx + train_len - 1
        if train_end_idx >= n - 1:
            break

        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_len - 1
        if test_end_idx >= n:
            break

        train_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        train_mask[train_start_idx:train_end_idx + 1] = True
        test_mask[test_start_idx:test_end_idx + 1] = True

        splits.append({
            "split_id": split_id,
            "train_start": dts[train_start_idx],
            "train_end": dts[train_end_idx],
            "test_start": dts[test_start_idx],
            "test_end": dts[test_end_idx],
            "train_idx": train_mask,
            "test_idx": test_mask,
        })

        split_id += 1
        start_idx += step

    return splits


# ----------------------------------------------------------------------
# 4) Parameter grid
# ----------------------------------------------------------------------

def default_param_grid() -> np.ndarray:
    """
    Build a reasonably small but expressive parameter grid for:
        z_entry, z_exit, z_stop, max_hold_days

    You can tweak this later. For now:

        z_entry in [0.5, 0.75, 1.0, 1.25]
        z_exit  in [0.20, 0.30, 0.40]
        z_stop  in [0.5, 1.0, 1.5, 2.0]
        max_hold_days in [3, 5, 7, 10]

    => 4 * 3 * 4 * 4 = 192 combinations
    """
    z_entry_vals = np.array([0.5, 0.75, 1.0, 1.25], dtype=float)
    z_exit_vals = np.array([0.20, 0.30, 0.40], dtype=float)
    z_stop_vals = np.array([0.5, 1.0, 1.5, 2.0], dtype=float)
    max_hold_vals = np.array([3, 5, 7, 10], dtype=float)

    grid = []
    for ze in z_entry_vals:
        for zx in z_exit_vals:
            for zs in z_stop_vals:
                for mh in max_hold_vals:
                    grid.append([ze, zx, zs, mh])
    return np.asarray(grid, dtype=float)


# ----------------------------------------------------------------------
# 5) High-level runner
# ----------------------------------------------------------------------

def run_tscv_from_enhanced(
    *,
    train_len: int = 252,
    test_len: int = 63,
    step: int = 63,
    out_suffix: str = "tscv_true",
    param_grid: Optional[np.ndarray] = None,
    min_sep_years: Optional[float] = None,
    max_span_years: Optional[float] = None,
    min_leg_tenor: Optional[float] = None,
) -> pd.DataFrame:
    """
    End-to-end driver:

      1) Load enhanced daily panel.
      2) Build pair panel (z_panel, spread_panel).
      3) Construct time-series splits.
      4) For each split, run run_param_grid on the TRAIN segment only.
      5) Collect best params and metrics, save a parquet.

    Returns the summary DataFrame.
    """
    print("[TSCV] Loading enhanced daily panel...")
    df_enh = load_all_enhanced_daily()
    print(f"[TSCV] Enhanced rows: {len(df_enh)}")

    print("[TSCV] Building pair panel...")
    z_panel, spread_panel, pairs, dts = build_pair_panel_from_enhanced(
        df_enh,
        min_sep_years=min_sep_years,
        max_span_years=max_span_years,
        min_leg_tenor=min_leg_tenor,
    )
    print(f"[TSCV] Panel shape: z_panel={z_panel.shape}, spread_panel={spread_panel.shape}")
    print(f"[TSCV] Tenor pairs: {len(pairs)}")

    print("[TSCV] Constructing time-series splits...")
    splits = make_time_splits(
        dts,
        train_len=train_len,
        test_len=test_len,
        step=step,
    )
    if not splits:
        raise RuntimeError("No valid time splits produced. Check train_len/test_len/step vs data length.")
    print(f"[TSCV] Number of splits: {len(splits)}")

    if param_grid is None:
        param_grid = default_param_grid()
    print(f"[TSCV] Param grid size: {param_grid.shape[0]}")

    rows = []
    for sp in splits:
        sid = sp["split_id"]
        train_idx = sp["train_idx"]

        z_train = z_panel[:, train_idx]
        spread_train = spread_panel[:, train_idx]

        print(f"[TSCV] Split {sid}: "
              f"train {sp['train_start'].date()} -> {sp['train_end'].date()} "
              f"(T={z_train.shape[1]})")

        res = run_param_grid(z_train, spread_train, param_grid)
        best_params: ReversionParams = res["best_params"]

        row = {
            "split_id": sid,
            "train_start": sp["train_start"],
            "train_end": sp["train_end"],
            "test_start": sp["test_start"],
            "test_end": sp["test_end"],
            "best_z_entry": best_params.z_entry,
            "best_z_exit": best_params.z_exit,
            "best_z_stop": best_params.z_stop,
            "best_max_hold_days": best_params.max_hold_days,
            "best_pnl_unit": res["best_pnl"],
            "best_trades": res["best_trades"],
            "best_win_rate": res["best_win_rate"],
            "grid_size": param_grid.shape[0],
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("split_id").reset_index(drop=True)

    # Save to PATH_OUT
    out_dir = Path(cr.PATH_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = getattr(cr, "OUT_SUFFIX", "")
    fname = f"tscv_params_{out_suffix}{suffix}.parquet"
    out_path = out_dir / fname
    summary.to_parquet(out_path, index=False)
    print(f"[TSCV] Saved split params to {out_path}")

    return summary


# ----------------------------------------------------------------------
# CLI helper
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple CLI: python tscv_from_enhanced.py
    summary_df = run_tscv_from_enhanced()
    print(summary_df)