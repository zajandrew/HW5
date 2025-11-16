from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

import cr_config as cr
import portfolio_test_new as pt
from portfolio_test_new import (
    _enhanced_in_path,
    _get_z_at_tenor,
    fly_alignment_ok,
    pv01_proxy,
    assign_bucket,
)

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def analyze_overlay(
    yymms: List[str],
    hedge_df: pd.DataFrame,
    *,
    decision_freq: str | None = None,
    overlay_use_caps: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Diagnostics for overlay mode (Mode B).

    For each hedge in hedge_df and each month in yymms, this:
      - Aligns hedges to decision buckets (same logic as portfolio_test),
      - Loads the enhanced curve snapshots,
      - Runs the *entry* logic used in overlay mode,
      - Records whether a pair would be opened and, if not, why.

    Returns a DataFrame with one row per (hedge, decision_ts), including:
      trade_id, side, trade_ts, decision_ts, trade_tenor, dv01,
      exec_tenor, exec_z, best_alt_tenor, best_alt_z, best_zdisp,
      reason (opened / no_snapshot / no_exec_tenor / no_alt_tenors / ...),
      and some flags for which filters blocked candidates.
    """
    import math

    if hedge_df is None or hedge_df.empty:
        raise ValueError("hedge_df is empty; nothing to analyze.")

    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    # Default: apply overlay caps (per-trade + per-timestamp) in diagnostics
    overlay_use_caps = True if overlay_use_caps is None else bool(overlay_use_caps)

    # Global floors / spans
    MIN_LEG_TENOR = float(getattr(cr, "MIN_LEG_TENOR_YEARS", 0.0))
    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    MAX_SPAN_YEARS = float(getattr(cr, "MAX_SPAN_YEARS", 10.0))
    SHORT_END_EXTRA_Z = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))

    # Overlay DV01 cap config
    OVERLAY_DV01_CAP_PER_TRADE = float(getattr(cr, "OVERLAY_DV01_CAP_PER_TRADE", float("inf")))
    OVERLAY_DV01_CAP_PER_TRADE_BUCKET = dict(
        getattr(cr, "OVERLAY_DV01_CAP_PER_TRADE_BUCKET", {})
    )
    OVERLAY_DV01_TS_CAP = float(getattr(cr, "OVERLAY_DV01_TS_CAP", float("inf")))

    # DV01-scaled Z-entry config
    BASE_Z_ENTRY = float(getattr(cr, "Z_ENTRY", 0.75))
    Z_REF = float(getattr(cr, "OVERLAY_Z_ENTRY_DV01_REF", 5_000.0))
    Z_K = float(getattr(cr, "OVERLAY_Z_ENTRY_DV01_K", 0.0))

    def _overlay_effective_z_entry(dv01_cash: float) -> float:
        dv = abs(float(dv01_cash))
        if dv <= 0 or Z_REF <= 0 or Z_K == 0.0:
            return BASE_Z_ENTRY
        return BASE_Z_ENTRY + Z_K * math.log(dv / Z_REF)

    def _per_trade_dv01_cap_for_bucket(bucket: str) -> float:
        # Bucket-specific cap overrides global; fall back to "other" then global.
        bucket_caps = OVERLAY_DV01_CAP_PER_TRADE_BUCKET
        if bucket in bucket_caps:
            return float(bucket_caps[bucket])
        if "other" in bucket_caps:
            return float(bucket_caps["other"])
        return OVERLAY_DV01_CAP_PER_TRADE

    # 1) Reuse the same hedge-tape preparation logic as overlay mode
    clean_hedges = pt.prepare_hedge_tape(hedge_df, decision_freq)
    if clean_hedges.empty:
        print("[DIAG] After prepare_hedge_tape: no valid hedges (check side, tenor mapping, dv01).")
        return pd.DataFrame()

    print(f"[DIAG] Total raw hedges: {len(hedge_df)}")
    print(f"[DIAG] After clean-up (side, tenor mapping, dv01): {len(clean_hedges)}")

    records: List[Dict] = []

    for yymm in yymms:
        # 2) Load enhanced month file
        enh_path = _enhanced_in_path(yymm)
        if not enh_path.exists():
            print(f"[DIAG] Missing enhanced file for {yymm}: {enh_path}")
            continue

        df = pd.read_parquet(enh_path)
        if df.empty:
            print(f"[DIAG] Enhanced file for {yymm} is empty.")
            continue

        df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
        if decision_freq == "D":
            df["decision_ts"] = df["ts"].dt.floor("D")
        elif decision_freq == "H":
            df["decision_ts"] = df["ts"].dt.floor("H")
        else:
            raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

        valid_decisions = df["decision_ts"].dropna().unique()
        hedges_month = _slice_month(clean_hedges, yymm)
        if hedges_month.empty:
            print(f"[DIAG] No hedges in month {yymm} after time slicing.")
            continue

        # Keep hedges whose decision_ts actually exist in the enhanced file
        hedges_month = hedges_month[hedges_month["decision_ts"].isin(valid_decisions)].copy()
        if hedges_month.empty:
            print(f"[DIAG] Hedges in {yymm} exist, but none map to enhanced decision_ts.")
            continue

        print(f"[DIAG] {yymm}: hedges after decision_ts intersection: {len(hedges_month)}")

        # Pre-split enhanced file by decision_ts to avoid repeated filtering
        grouped = dict(tuple(df.groupby("decision_ts", sort=True)))

        # Pre-compute per-timestamp DV01 sums (for OVERLAY_DV01_TS_CAP)
        if overlay_use_caps and OVERLAY_DV01_TS_CAP < float("inf"):
            ts_dv01_sum = (
                hedges_month.groupby("decision_ts")["dv01"]
                .apply(lambda s: s.abs().sum())
                .to_dict()
            )
        else:
            ts_dv01_sum = {}

        for _, h in hedges_month.iterrows():
            trade_id = h.get("trade_id", None)
            trade_ts = h["trade_ts"]
            dts = h["decision_ts"]
            side = str(h["side"]).upper()
            dv01_cash = float(h["dv01"])
            trade_tenor = float(h["tenor_yrs"])

            rec = {
                "yymm": yymm,
                "trade_id": trade_id,
                "trade_ts": trade_ts,
                "decision_ts": dts,
                "side": side,
                "dv01": dv01_cash,
                "trade_tenor": trade_tenor,
                "exec_tenor": np.nan,
                "exec_z": np.nan,
                "best_alt_tenor": np.nan,
                "best_alt_z": np.nan,
                "best_zdisp": np.nan,
                "reason": None,
                "hit_any_alt": False,
                "hit_z_threshold": False,
                "hit_fly_block": False,
                "hit_caps_block": False,
            }

            # Tenor floor on executed hedge
            if trade_tenor < MIN_LEG_TENOR:
                rec["reason"] = "too_short_exec_tenor"
                records.append(rec)
                continue

            # Per-timestamp DV01 gate
            if overlay_use_caps and ts_dv01_sum:
                ts_sum = float(ts_dv01_sum.get(dts, 0.0))
                if ts_sum > OVERLAY_DV01_TS_CAP:
                    rec["reason"] = "caps_block"
                    rec["hit_caps_block"] = True
                    records.append(rec)
                    continue

            # 2a) Snapshot at this decision_ts?
            snap = grouped.get(dts)
            if snap is None or snap.empty:
                rec["reason"] = "no_snapshot"
                records.append(rec)
                continue

            snap_last = (
                snap.sort_values("ts")
                    .groupby("tenor_yrs", as_index=False)
                    .tail(1)
                    .reset_index(drop=True)
            )
            if snap_last.empty:
                rec["reason"] = "no_snapshot"
                records.append(rec)
                continue

            # 2b) Executed tenor on curve?
            exec_z = _get_z_at_tenor(snap_last, trade_tenor)
            if exec_z is None or not np.isfinite(exec_z):
                rec["reason"] = "no_exec_tenor"
                records.append(rec)
                continue

            nearest_idx = (snap_last["tenor_yrs"] - trade_tenor).abs().idxmin()
            exec_row = snap_last.iloc[nearest_idx]
            exec_tenor = float(exec_row["tenor_yrs"])
            rec["exec_tenor"] = exec_tenor
            rec["exec_z"] = exec_z

            # Side sign (for directional "better tenor" logic)
            if side == "CRCV":
                side_sign = +1.0
            elif side == "CPAY":
                side_sign = -1.0
            else:
                rec["reason"] = "bad_side"
                records.append(rec)
                continue

            # Per-trade DV01 caps
            if overlay_use_caps:
                dv_abs = abs(dv01_cash)
                if dv_abs <= 0.0:
                    rec["reason"] = "caps_block"
                    rec["hit_caps_block"] = True
                    records.append(rec)
                    continue

                bucket = assign_bucket(trade_tenor)
                per_trade_cap = _per_trade_dv01_cap_for_bucket(bucket)
                if dv_abs > OVERLAY_DV01_CAP_PER_TRADE or dv_abs > per_trade_cap:
                    rec["reason"] = "caps_block"
                    rec["hit_caps_block"] = True
                    records.append(rec)
                    continue

            # DV01-scaled entry threshold for this hedge
            z_entry_eff = _overlay_effective_z_entry(dv01_cash)

            # 3) Scan alt tenors and track reasons why they are killed
            best_candidate = None
            best_zdisp = 0.0
            reasons_seen = set()

            for _, alt_row in snap_last.sort_values("tenor_yrs").iterrows():
                alt_tenor = float(alt_row["tenor_yrs"])
                if alt_tenor == exec_tenor:
                    continue

                # Tenor floor on both legs
                if (alt_tenor < MIN_LEG_TENOR) or (exec_tenor < MIN_LEG_TENOR):
                    reasons_seen.add("too_short_leg")
                    continue

                diff = abs(alt_tenor - exec_tenor)
                if diff < MIN_SEP_YEARS or diff > MAX_SPAN_YEARS:
                    reasons_seen.add("span")
                    continue

                z_alt = _safe_float(alt_row["z_comb"])
                if not np.isfinite(z_alt):
                    reasons_seen.add("z_nan")
                    continue

                # Direction-dependent better-tenor logic
                if side == "CRCV":
                    if z_alt <= exec_z:
                        reasons_seen.add("z_dir")
                        continue
                    zdisp = z_alt - exec_z
                else:  # CPAY
                    if z_alt >= exec_z:
                        reasons_seen.add("z_dir")
                        continue
                    zdisp = exec_z - z_alt

                if zdisp < z_entry_eff:
                    reasons_seen.add("z_threshold")
                    continue

                # Cheap/rich by z only
                if z_alt > exec_z:
                    cheap_tenor, rich_tenor = alt_tenor, exec_tenor
                else:
                    cheap_tenor, rich_tenor = exec_tenor, alt_tenor

                # Short-end extra hurdle: if either leg in short bucket, require extra zdisp
                b_alt = assign_bucket(alt_tenor)
                b_exec = assign_bucket(exec_tenor)
                if (b_alt == "short" or b_exec == "short") and (zdisp < (z_entry_eff + SHORT_END_EXTRA_Z)):
                    reasons_seen.add("z_threshold")
                    continue

                ok_i = fly_alignment_ok(cheap_tenor, +1.0, snap_last, zdisp_for_pair=zdisp)
                ok_j = fly_alignment_ok(rich_tenor, -1.0, snap_last, zdisp_for_pair=zdisp)
                if not (ok_i and ok_j):
                    reasons_seen.add("fly")
                    continue

                # If we got here, this alt is a valid candidate
                rec["hit_any_alt"] = True
                if zdisp > best_zdisp:
                    best_zdisp = zdisp
                    best_candidate = alt_row

            # Decide final reason
            if best_candidate is None:
                if "too_short_leg" in reasons_seen:
                    final_reason = "too_short_leg"
                elif "span" in reasons_seen and len(reasons_seen) == 1:
                    final_reason = "no_alt_tenors"
                elif ("z_threshold" in reasons_seen or "z_dir" in reasons_seen):
                    final_reason = "no_zdisp_ge_entry"
                    rec["hit_z_threshold"] = True
                elif "fly" in reasons_seen:
                    final_reason = "fly_block"
                    rec["hit_fly_block"] = True
                elif "caps" in reasons_seen:
                    final_reason = "caps_block"
                    rec["hit_caps_block"] = True
                else:
                    final_reason = "no_candidate"
                rec["reason"] = final_reason
                records.append(rec)
                continue

            # We would open a pair here in overlay mode
            alt_row = best_candidate
            alt_tenor = float(alt_row["tenor_yrs"])
            best_alt_z = _safe_float(alt_row["z_comb"])

            rec["best_alt_tenor"] = alt_tenor
            rec["best_alt_z"] = best_alt_z
            rec["best_zdisp"] = best_zdisp
            rec["reason"] = "opened"

            records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        print("[DIAG] No diagnostic records produced (check that yymms and hedge_df overlap).")
        return out

    # Quick summary
    print("\n[DIAG] Overlay entry summary (by reason):")
    print(out["reason"].value_counts(dropna=False))

    opened = int((out["reason"] == "opened").sum())
    total = len(out)
    pct = (opened / total * 100.0) if total > 0 else 0.0
    print(f"[DIAG] opened: {opened} / {total} ({pct:.2f}%)")

    return out

def _slice_month(hedges: pd.DataFrame, yymm: str) -> pd.DataFrame:
    """Slice hedges to a given yymm using decision_ts (UTC-naive)."""
    if hedges is None or hedges.empty:
        return pd.DataFrame()
    year = 2000 + int(yymm[:2])
    month = int(yymm[2:])
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + MonthEnd(1)) + pd.Timedelta(days=1)
    return hedges[(hedges["decision_ts"] >= start) & (hedges["decision_ts"] < end)].copy()


if __name__ == "__main__":
    # Simple CLI wire-up: python overlay_diag.py 2304 2305 trade_tape.parquet
    if len(sys.argv) < 3:
        print("Usage: python overlay_diag.py 2304 [2305 ...] trade_tape.parquet")
        raise SystemExit(1)

    *months, tape_path = sys.argv[1:]
    tape = pd.read_parquet(tape_path)
    diag = analyze_overlay(months, tape)
    out_dir = Path(getattr(cr, "PATH_OUT", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = getattr(cr, "OUT_SUFFIX", "")
    out_path = out_dir / f"overlay_diag{suffix}.parquet"
    diag.to_parquet(out_path, index=False)
    print(f"[DIAG] Saved overlay diagnostics to {out_path}")
