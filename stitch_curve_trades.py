# stitch_curve_trades.py

from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

import cr_config as cr


def _raw_curve_path_for_month(yymm: str) -> Path:
    """
    Helper: map yymm -> raw tick parquet.

    If you already have a function with this name in your file, KEEP YOUR VERSION
    and delete this stub. Pattern below is just a placeholder.
    """
    # >>> ADJUST IF NEEDED <<<
    # e.g. Path(cr.PATH_DATA) / f"sofr_ticks_{yymm}.parquet"
    return Path(cr.PATH_DATA) / f"raw_{yymm}.parquet"


def _month_key_from_ts(ts: pd.Series) -> pd.Series:
    """Map timestamps to 'yymm' key (e.g. 2023-04-15 -> '2304')."""
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    return ts.dt.strftime("%y%m")


def attach_curve_to_trades(
    trades_path: str | Path,
    out_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Attach FULL curve snapshots (all *_mid columns) at each trade's tradetimeUTC.

    Logic:
      1) Load trades.pkl (must have 'tradetimeUTC').
      2) For each yymm:
           - collect unique trade timestamps for that month,
           - load raw tick data for that month from PATH_DATA,
           - reindex curve to those timestamps,
           - drop timestamps with any NaNs (missing curve),
           - merge trimmed curve snapshot onto trades.
      3) Concatenate all monthly results, drop trades whose timestamps
         never matched the curve, and save to out_path (if given).

    Any trade whose tradetimeUTC does NOT exist in the raw curve index
    is dropped (with a warning).
    """
    trades_path = Path(trades_path)
    if out_path is not None:
        out_path = Path(out_path)

    # ------------------------------------------------------------------
    # 1) Load trades and basic sanity
    # ------------------------------------------------------------------
    trades = pd.read_pickle(trades_path)
    if "tradetimeUTC" not in trades.columns:
        raise ValueError("trades must have a 'tradetimeUTC' column.")

    trades = trades.copy()
    trades["tradetimeUTC"] = pd.to_datetime(
        trades["tradetimeUTC"],
        utc=True,
        errors="coerce",
    )

    # Drop blatantly bad timestamps up front
    bad_ts = trades["tradetimeUTC"].isna().sum()
    if bad_ts > 0:
        print(f"[STITCH] Dropping {bad_ts} trades with NaN tradetimeUTC.")
        trades = trades[trades["tradetimeUTC"].notna()].copy()

    if trades.empty:
        raise RuntimeError("No trades left after cleaning tradetimeUTC.")

    # Month key per trade
    trades["yymm"] = _month_key_from_ts(trades["tradetimeUTC"])

    # ------------------------------------------------------------------
    # 2) Month-by-month stitching
    # ------------------------------------------------------------------
    stitched: List[pd.DataFrame] = []
    total_dropped = 0

    for yymm, trades_m in trades.groupby("yymm"):
        trades_m = trades_m.sort_values("tradetimeUTC").copy()
        if trades_m.empty:
            continue

        raw_path = _raw_curve_path_for_month(yymm)
        if not raw_path.exists():
            print(f"[STITCH] Missing raw curve file for {yymm}: {raw_path}. "
                  f"Dropping {len(trades_m)} trades in this month.")
            total_dropped += len(trades_m)
            continue

        curve = pd.read_parquet(raw_path)

        # Ensure curve index is a DateTimeIndex
        if not isinstance(curve.index, pd.DatetimeIndex):
            # Try to promote a time column to index if necessary
            if "ts" in curve.columns:
                curve["ts"] = pd.to_datetime(curve["ts"], utc=True, errors="coerce")
                curve = curve.set_index("ts").sort_index()
            else:
                raise ValueError(
                    f"Raw curve file {raw_path} does not have a DatetimeIndex "
                    "or a 'ts' column to use as index."
                )
        else:
            # Make sure index is timezone-aware UTC to match trades
            if curve.index.tz is None:
                curve.index = curve.index.tz_localize("UTC")
            else:
                curve.index = curve.index.tz_convert("UTC")

        # ------------------------------------------------------------------
        # Unique trade timestamps for this month
        # ------------------------------------------------------------------
        unique_ts = np.sort(trades_m["tradetimeUTC"].unique())

        # Reindex curve to EXACT trade timestamps
        curve_trim = curve.reindex(unique_ts)

        # Detect missing timestamps (any NaN across columns)
        bad_rows = curve_trim.isna().any(axis=1)
        if bad_rows.any():
            bad_count = int(bad_rows.sum())
            good_count = len(curve_trim) - bad_count
            print(
                f"[STITCH] {yymm}: {bad_count} trade timestamps "
                f"have no exact curve row; keeping {good_count}."
            )
            curve_trim = curve_trim[~bad_rows].copy()

        if curve_trim.empty:
            print(f"[STITCH] {yymm}: no trade timestamps matched curve index; "
                  f"dropping {len(trades_m)} trades in this month.")
            total_dropped += len(trades_m)
            continue

        # Keep only trades whose tradetimeUTC is in curve_trim index
        good_ts = curve_trim.index
        trades_good = trades_m[trades_m["tradetimeUTC"].isin(good_ts)].copy()
        dropped_here = len(trades_m) - len(trades_good)
        if dropped_here > 0:
            print(f"[STITCH] {yymm}: dropping {dropped_here} trades "
                  "with non-matching timestamps.")
            total_dropped += dropped_here

        if trades_good.empty:
            continue

        # Prepare curve_trim for merge: bring index back as 'tradetimeUTC'
        idx_name = curve_trim.index.name or "index"
        curve_trim_reset = curve_trim.reset_index().rename(
            columns={idx_name: "tradetimeUTC"}
        )

        # Merge full curve snapshot onto trades
        merged = trades_good.merge(
            curve_trim_reset,
            on="tradetimeUTC",
            how="left",
            validate="many_to_one",
        )

        stitched.append(merged)

    if not stitched:
        raise RuntimeError(
            "No trades could be stitched with the curve. "
            "Check timestamps and raw curve files."
        )

    out = (
        pd.concat(stitched, ignore_index=True)
          .sort_values("tradetimeUTC")
          .reset_index(drop=True)
    )

    if total_dropped > 0:
        print(f"[STITCH] Total trades dropped due to missing curve rows: {total_dropped}")

    # Clean up helper column
    if "yymm" in out.columns:
        out = out.drop(columns=["yymm"])

    # ------------------------------------------------------------------
    # 3) Save & return
    # ------------------------------------------------------------------
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_pickle(out_path)
        print(f"[STITCH] Wrote stitched trades+curve to: {out_path}")

    return out
