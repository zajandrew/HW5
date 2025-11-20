"""
tscv.py

Time-series CV utilities for optimizing reversion parameters:

    z_entry, z_exit, z_stop, max_hold_days

This module is deliberately *simplified* and decoupled from the full
portfolio_test_new engine: it assumes a pre-built panel of "unit trades"
or "pairs" with:

    - z_dir_panel:   shape (N_series, T), directional z values
    - spread_panel:  shape (N_series, T), directional spread in bp per unit DV01

Conventions
-----------
- z_dir_panel[i, t] > 0  means "signal in the direction we want" (e.g. cheap-rich).
- We ENTER when z_dir >= z_entry and no position is open.
- While open, we EXIT when ANY of:

    1) Reversion:  abs(z_curr) <= abs(z_entry) AND abs(z_curr) <= z_exit
    2) Stop:       same sign as entry, AND abs(z_curr) >= abs(z_entry) + z_stop
    3) Max-hold:   (t - entry_idx) >= max_hold

- PnL per unit DV01 is:

    pnl_unit = spread_entry - spread_exit

  i.e. we assume a positive z_dir means we are short the spread and
  profit when the spread narrows.

We aggregate PnL across all series for a given parameter set, and can
evaluate a whole parameter grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ----------------------------------------------------------------------
# Optional numba acceleration
# ----------------------------------------------------------------------
try:
    from numba import njit
except ImportError:  # graceful fallback if numba isn't installed
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


# ----------------------------------------------------------------------
# Parameter container (for clarity)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ReversionParams:
    z_entry: float
    z_exit: float
    z_stop: float
    max_hold_days: int  # for daily panel; for hourly this is "buckets"


# ----------------------------------------------------------------------
# Core 1D path simulator (single series)
# ----------------------------------------------------------------------
@njit
def _simulate_path_1d(
    z_dir: np.ndarray,
    spread: np.ndarray,
    z_entry: float,
    z_exit: float,
    z_stop: float,
    max_hold: int,
) -> Tuple[float, int, int]:
    """
    Simulate a single 1D series with a simple reversion strategy.

    Parameters
    ----------
    z_dir : 1D array (T,)
        Directional z per bucket. z_dir[t] > 0 means signal in our direction.
    spread : 1D array (T,)
        Directional spread in bp per unit DV01 (same orientation as z_dir).
    z_entry, z_exit, z_stop : floats
        Thresholds as described in module docstring.
    max_hold : int
        Max number of buckets to hold a trade.

    Returns
    -------
    total_pnl : float
        Sum of PnL (per unit DV01) across all completed trades.
    n_trades : int
        Number of completed trades.
    n_wins : int
        Number of trades with positive PnL.
    """
    T = z_dir.shape[0]

    open_pos = False
    entry_idx = -1
    entry_z = 0.0
    entry_spread = 0.0

    total_pnl = 0.0
    n_trades = 0
    n_wins = 0

    for t in range(T):
        z = z_dir[t]
        s = spread[t]

        # If either is NaN, skip decision but time still passes
        if np.isnan(z) or np.isnan(s):
            # max-hold is still governed by time, so we do count these steps
            if open_pos and (t - entry_idx) >= max_hold:
                # force close at last valid spread we saw
                # if spread is NaN here, we approximate using entry spread (flat)
                exit_spread = entry_spread if np.isnan(s) else s
                pnl = entry_spread - exit_spread
                total_pnl += pnl
                n_trades += 1
                if pnl > 0.0:
                    n_wins += 1
                open_pos = False
                entry_idx = -1
            continue

        if not open_pos:
            # ENTRY condition
            if z >= z_entry:
                open_pos = True
                entry_idx = t
                entry_z = z
                entry_spread = s
        else:
            # Already open: check exit conditions
            dz_curr = z
            same_sign = (entry_z * dz_curr) > 0.0

            # 1) Reversion: closer to zero and inside exit band
            moved_towards_zero = np.abs(dz_curr) <= np.abs(entry_z)
            within_exit_band = np.abs(dz_curr) <= z_exit

            reversion_exit = same_sign and moved_towards_zero and within_exit_band

            # 2) Stop: same sign and at least z_stop worse than entry
            moved_away = same_sign and (np.abs(dz_curr) >= np.abs(entry_z) + z_stop)

            # 3) Max-hold
            age = t - entry_idx
            max_hold_exit = age >= max_hold

            if reversion_exit or moved_away or max_hold_exit:
                exit_spread = s
                pnl = entry_spread - exit_spread
                total_pnl += pnl
                n_trades += 1
                if pnl > 0.0:
                    n_wins += 1

                open_pos = False
                entry_idx = -1

    # Note: we leave any still-open trade at the end as *unrealized* and ignore it.
    return total_pnl, n_trades, n_wins


# ----------------------------------------------------------------------
# Multi-series aggregator (panel) for a single parameter set
# ----------------------------------------------------------------------
@njit
def _evaluate_param_on_panel(
    z_panel: np.ndarray,
    spread_panel: np.ndarray,
    z_entry: float,
    z_exit: float,
    z_stop: float,
    max_hold: int,
) -> Tuple[float, int, int]:
    """
    Aggregate over all series in the panel for a single param set.

    Parameters
    ----------
    z_panel : 2D array (N_series, T)
    spread_panel : 2D array (N_series, T)
    z_entry, z_exit, z_stop, max_hold : see above.

    Returns
    -------
    total_pnl : float
    total_trades : int
    total_wins : int
    """
    n_series = z_panel.shape[0]

    total_pnl = 0.0
    total_trades = 0
    total_wins = 0

    for i in range(n_series):
        pnl_i, trades_i, wins_i = _simulate_path_1d(
            z_panel[i],
            spread_panel[i],
            z_entry,
            z_exit,
            z_stop,
            max_hold,
        )
        total_pnl += pnl_i
        total_trades += trades_i
        total_wins += wins_i

    return total_pnl, total_trades, total_wins


# ----------------------------------------------------------------------
# Param grid evaluator (Numba)
# ----------------------------------------------------------------------
@njit
def _evaluate_param_grid_numba(
    z_panel: np.ndarray,
    spread_panel: np.ndarray,
    param_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a parameter grid on the panel.

    Parameters
    ----------
    z_panel : 2D array (N_series, T)
    spread_panel : 2D array (N_series, T)
    param_grid : 2D array (N_param, 4)
        param_grid[k] = [z_entry, z_exit, z_stop, max_hold_days]

    Returns
    -------
    pnl_vec : 1D array (N_param,)
        Total pnl per param set (sum over all series).
    trades_vec : 1D array (N_param,)
        Total trade count per param set.
    wins_vec : 1D array (N_param,)
        Total winning trades per param set.
    """
    n_param = param_grid.shape[0]

    pnl_vec = np.zeros(n_param, dtype=np.float64)
    trades_vec = np.zeros(n_param, dtype=np.int64)
    wins_vec = np.zeros(n_param, dtype=np.int64)

    for k in range(n_param):
        z_entry = param_grid[k, 0]
        z_exit  = param_grid[k, 1]
        z_stop  = param_grid[k, 2]
        max_hold = int(param_grid[k, 3])

        pnl, n_tr, n_win = _evaluate_param_on_panel(
            z_panel,
            spread_panel,
            z_entry,
            z_exit,
            z_stop,
            max_hold,
        )
        pnl_vec[k] = pnl
        trades_vec[k] = n_tr
        wins_vec[k] = n_win

    return pnl_vec, trades_vec, wins_vec


# ----------------------------------------------------------------------
# Python-friendly wrapper
# ----------------------------------------------------------------------
def run_param_grid(
    z_panel: np.ndarray,
    spread_panel: np.ndarray,
    param_grid: np.ndarray,
) -> dict:
    """
    Python wrapper around the numba-accelerated grid evaluator.

    Parameters
    ----------
    z_panel : array-like, shape (N_series, T)
    spread_panel : array-like, shape (N_series, T)
    param_grid : array-like, shape (N_param, 4)
        Columns: [z_entry, z_exit, z_stop, max_hold_days]

    Returns
    -------
    result : dict
        {
            "param_grid": param_grid (as np.ndarray),
            "pnl": pnl_vec,
            "trades": trades_vec,
            "wins": wins_vec,
            "win_rate": wins_vec / np.maximum(trades_vec, 1),
            "best_idx": idx_of_best_pnl,
            "best_params": ReversionParams(...),
            "best_pnl": best_pnl,
            "best_trades": best_trades,
            "best_win_rate": best_win_rate,
        }
    """
    z_panel = np.asarray(z_panel, dtype=np.float64)
    spread_panel = np.asarray(spread_panel, dtype=np.float64)
    param_grid = np.asarray(param_grid, dtype=np.float64)

    if z_panel.shape != spread_panel.shape:
        raise ValueError("z_panel and spread_panel must have the same shape.")

    if param_grid.ndim != 2 or param_grid.shape[1] != 4:
        raise ValueError("param_grid must have shape (N_param, 4).")

    pnl_vec, trades_vec, wins_vec = _evaluate_param_grid_numba(
        z_panel,
        spread_panel,
        param_grid,
    )

    # protect from div-by-zero
    win_rate = wins_vec / np.maximum(trades_vec, 1)

    # For now, choose best by total pnl; you can change this to Sharpe-like metric later
    best_idx = int(np.argmax(pnl_vec))
    best_row = param_grid[best_idx]
    best_params = ReversionParams(
        z_entry=float(best_row[0]),
        z_exit=float(best_row[1]),
        z_stop=float(best_row[2]),
        max_hold_days=int(best_row[3]),
    )

    result = {
        "param_grid": param_grid,
        "pnl": pnl_vec,
        "trades": trades_vec,
        "wins": wins_vec,
        "win_rate": win_rate,
        "best_idx": best_idx,
        "best_params": best_params,
        "best_pnl": float(pnl_vec[best_idx]),
        "best_trades": int(trades_vec[best_idx]),
        "best_win_rate": float(win_rate[best_idx]),
    }
    return result