from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

import cr_config as cr
import portfolio_test as pt
from portfolio_test import (
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
    if hedge_df is None or hedge_df.empty:
        raise ValueError("hedge_df is empty; nothing to analyze.")

    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    overlay_use_caps = cr.OVERLAY_USE_CAPS if overlay_use_caps is None else bool(overlay_use_caps)

    # NEW: global min-leg floor, same as portfolio_test overlay logic
    MIN_LEG_TENOR = float(getattr(cr, "MIN_LEG_TENOR_YEARS", 0.0))

    # 1) Reuse the same hedge-tape preparation logic as overlay mode
    clean_hedges = pt.prepare_hedge_tape(hedge_df, decision_freq)
    if clean_hedges.empty:
        print("[DIAG] After prepare_hedge_tape: no valid hedges (check side, tenor mapping, dv01).")
        return pd.DataFrame()

    print(f"[DIAG] Total raw hedges: {len(hedge_df)}")
    print(f"[DIAG] After clean-up (side, tenor mapping, dv01): {len(clean_hedges)}")

    records: List[Dict] = []

    # Caps config if overlay caps are "on"
    PER_BUCKET_DV01_CAP  = float(getattr(cr, "PER_BUCKET_DV01_CAP", 1.0))
    TOTAL_DV01_CAP       = float(getattr(cr, "TOTAL_DV01_CAP", 3.0))
    FRONT_END_DV01_CAP   = float(getattr(cr, "FRONT_END_DV01_CAP", 1.0))
    Z_ENTRY              = float(getattr(cr, "Z_ENTRY", 0.75))
    SHORT_END_EXTRA_Z    = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))
    MIN_SEP_YEARS        = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    MAX_SPAN_YEARS       = float(getattr(cr, "MAX_SPAN_YEARS", 10.0))

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

            # NEW: if executed hedge tenor itself is too short, bail early
            if trade_tenor < MIN_LEG_TENOR:
                rec["reason"] = "too_short_exec_tenor"
                records.append(rec)
                continue

            # 2a) Is there a snapshot at this decision_ts?
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

            # Actual row of executed tenor (nearest in tenor space)
            nearest_idx = (snap_last["tenor_yrs"] - trade_tenor).abs().idxmin()
            exec_row = snap_last.iloc[nearest_idx]
            exec_tenor = float(exec_row["tenor_yrs"])
            rec["exec_tenor"] = exec_tenor
            rec["exec_z"] = exec_z

            # Side sign (as in overlay code)
            if side == "CRCV":
                side_sign = +1.0
            elif side == "CPAY":
                side_sign = -1.0
            else:
                rec["reason"] = "bad_side"
                records.append(rec)
                continue

            # 3) Scan alt tenors and track reasons why they are killed
            best_candidate = None
            best_zdisp = 0.0
            reasons_seen = set()

            for _, alt_row in snap_last.sort_values("tenor_yrs").iterrows():
                alt_tenor = float(alt_row["tenor_yrs"])
                if alt_tenor == exec_tenor:
                    continue

                # NEW: enforce tenor floor on *both* legs of the potential pair
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

                if zdisp < Z_ENTRY:
                    reasons_seen.add("z_threshold")
                    continue

                # Fly gate orientation by z only (cheap vs rich)
                if z_alt > exec_z:
                    cheap_tenor, rich_tenor = alt_tenor, exec_tenor
                else:
                    cheap_tenor, rich_tenor = exec_tenor, alt_tenor

                ok_i = fly_alignment_ok(cheap_tenor, +1.0, snap_last, zdisp_for_pair=zdisp)
                ok_j = fly_alignment_ok(rich_tenor, -1.0, snap_last, zdisp_for_pair=zdisp)
                if not (ok_i and ok_j):
                    reasons_seen.add("fly")
                    continue

                # Caps (only if we are “using caps” in overlay)
                if overlay_use_caps:
                    b_alt = assign_bucket(alt_tenor)
                    b_exec = assign_bucket(exec_tenor)
                    pv_alt = pv01_proxy(alt_tenor, float(alt_row["rate"]))
                    pv_exec = pv01_proxy(exec_tenor, float(exec_row["rate"]))
                    dv_alt = abs(pv_alt)
                    dv_exec = abs(pv_exec)

                    # quick one-pair caps, like in diag_selector (not accumulative across hedges)
                    per_bucket_ok = True
                    if b_alt in cr.BUCKETS and dv_alt > PER_BUCKET_DV01_CAP:
                        per_bucket_ok = False
                    if b_exec in cr.BUCKETS and dv_exec > PER_BUCKET_DV01_CAP:
                        per_bucket_ok = False

                    short_used = (dv_alt if b_alt == "short" else 0.0) + (dv_exec if b_exec == "short" else 0.0)
                    fe_ok = short_used <= FRONT_END_DV01_CAP
                    total_ok = (dv_alt + dv_exec) <= TOTAL_DV01_CAP

                    # Short-end extra hurdle
                    if (b_alt == "short" or b_exec == "short") and (zdisp < (Z_ENTRY + SHORT_END_EXTRA_Z)):
                        per_bucket_ok = False

                    if not (per_bucket_ok and fe_ok and total_ok):
                        reasons_seen.add("caps")
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
    total  = len(out)
    print(f"[DIAG] opened: {opened} / {total} ({opened/total*100:.2f}% if total>0 else 0.0)")

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
