# diag_selector.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# All config access via module namespace
import cr_config as cr


# ---------- helpers ----------
def _to_float(x, default=np.nan):
    try:
        if isinstance(x, (pd.Series, pd.Index)):
            return float(x.iloc[0]) if len(x) else default
        return float(x)
    except Exception:
        return default

def pv01_proxy(tenor_yrs, rate_pct):
    return tenor_yrs / max(1e-6, 1.0 + 0.01 * rate_pct)

def assign_bucket(tenor):
    buckets = getattr(cr, "BUCKETS", {})
    for name, (lo, hi) in buckets.items():
        if (tenor >= lo) and (tenor <= hi):
            return name
    return "other"

def _get_z_at_tenor(snap_last: pd.DataFrame, tenor: float, tol: float | None = None) -> float | None:
    """Return z near requested tenor within tolerance; None if no close tenor."""
    if tol is None:
        tol = float(getattr(cr, "FLY_TENOR_TOL_YEARS", 0.02))
    s = snap_last[["tenor_yrs", "z_comb"]].dropna()
    if s.empty:
        return None
    s = s.assign(_dist=(s["tenor_yrs"] - float(tenor)).abs())
    row = s.loc[s["_dist"].idxmin()]
    return float(row["z_comb"]) if row["_dist"] <= tol else None

def compute_fly_z(snap_last: pd.DataFrame, a: float, b: float, c: float) -> float | None:
    """Standardized fly: [0.5*(z_a+z_c) - z_b] / cross-sec std(z_comb)."""
    try:
        z_a = _get_z_at_tenor(snap_last, a)
        z_b = _get_z_at_tenor(snap_last, b)
        z_c = _get_z_at_tenor(snap_last, c)
        if any(v is None for v in (z_a, z_b, z_c)):
            return None
        xs = snap_last["z_comb"].astype(float).to_numpy()
        sd = np.nanstd(xs, ddof=1) if xs.size > 1 else 1.0
        sd = sd if (np.isfinite(sd) and sd > 0) else 1.0
        fly = 0.5 * (z_a + z_c) - z_b
        return fly / sd
    except Exception:
        return None

def fly_alignment_ok(
    leg_tenor: float,
    leg_sign_z: float,                 # +1 for “cheap should go down”, -1 for “rich should go up”
    snap_last: pd.DataFrame,
    *,
    zdisp_for_pair: float | None = None
) -> bool:
    """
    Tolerant fly gate with:
      - neighborhood window around leg tenor (optional),
      - strong-fly threshold (|fly_z| >= FLY_Z_MIN),
      - short-end skip,
      - big-dispersion waiver,
      - mode logic: off | loose | tolerant | strict.
    """
    FLY_ENABLE          = bool(getattr(cr, "FLY_ENABLE", True))
    FLY_MODE            = str(getattr(cr, "FLY_MODE", "tolerant")).lower()
    FLY_DEFS            = list(getattr(cr, "FLY_DEFS", []))
    FLY_Z_MIN           = float(getattr(cr, "FLY_Z_MIN", 0.8))
    FLY_NEIGHBOR_ONLY   = bool(getattr(cr, "FLY_NEIGHBOR_ONLY", True))
    FLY_WINDOW_YEARS    = float(getattr(cr, "FLY_WINDOW_YEARS", 3.0))
    FLY_REQUIRE_COUNT   = int(getattr(cr, "FLY_REQUIRE_COUNT", 2))
    FLY_SKIP_SHORT_UNDER= getattr(cr, "FLY_SKIP_SHORT_UNDER", None)
    FLY_ALLOW_BIG_ZDISP = bool(getattr(cr, "FLY_ALLOW_BIG_ZDISP", True))
    FLY_BIG_ZDISP_MARGIN= float(getattr(cr, "FLY_BIG_ZDISP_MARGIN", 0.20))
    Z_ENTRY             = float(getattr(cr, "Z_ENTRY", 0.75))

    if not FLY_ENABLE or FLY_MODE == "off":
        return True

    # Short-end skip
    if (FLY_SKIP_SHORT_UNDER is not None) and (leg_tenor < float(FLY_SKIP_SHORT_UNDER)):
        return True

    # Waive if dispersion already large
    if (FLY_ALLOW_BIG_ZDISP and (zdisp_for_pair is not None)
        and (float(zdisp_for_pair) >= (Z_ENTRY + FLY_BIG_ZDISP_MARGIN))):
        return True

    triplets = FLY_DEFS
    if FLY_NEIGHBOR_ONLY:
        W = float(FLY_WINDOW_YEARS)
        triplets = [(a,b,c) for (a,b,c) in FLY_DEFS if abs(float(b) - float(leg_tenor)) <= W]
        if not triplets:
            return True

    contradictions = 0
    for (a, b, c) in triplets:
        zz = compute_fly_z(snap_last, a, b, c)
        if zz is None or not np.isfinite(zz) or abs(zz) < FLY_Z_MIN:
            continue
        # CONTRADICTION when sign(fly)*sign(leg) < 0
        if np.sign(zz) * np.sign(leg_sign_z) < 0:
            contradictions += 1

    if FLY_MODE == "strict":
        return contradictions == 0
    if FLY_MODE == "loose":
        # allow up to 1 contradiction
        return contradictions <= 1
    if FLY_MODE == "tolerant":
        return contradictions <= int(FLY_REQUIRE_COUNT)
    return True


# ---------- filename helpers ----------
def _enhanced_in_path(yymm: str) -> Path:
    """Use cr.enh_fname if present; else {yymm}_enh{ENH_SUFFIX}.parquet; fallback {yymm}_enh.parquet."""
    if hasattr(cr, "enh_fname") and callable(cr.enh_fname):
        name = cr.enh_fname(yymm)
    else:
        suffix = getattr(cr, "ENH_SUFFIX", "")
        name = f"{yymm}_enh{suffix}.parquet" if suffix else f"{yymm}_enh.parquet"
    return Path(getattr(cr, "PATH_ENH", ".")) / name

def _diag_csv_out(yymm: str) -> Path:
    if hasattr(cr, "diag_selector_csv") and callable(cr.diag_selector_csv):
        name = cr.diag_selector_csv(yymm)
    else:
        suffix = getattr(cr, "OUT_SUFFIX", "")
        name = f"diag_selector_{yymm}{suffix}.csv" if suffix else f"diag_selector_{yymm}.csv"
    return Path(getattr(cr, "PATH_OUT", ".")) / name

def _diag_txt_out(yymm: str) -> Path:
    if hasattr(cr, "diag_selector_txt") and callable(cr.diag_selector_txt):
        name = cr.diag_selector_txt(yymm)
    else:
        suffix = getattr(cr, "OUT_SUFFIX", "")
        name = f"diag_selector_{yymm}{suffix}.txt" if suffix else f"diag_selector_{yymm}.txt"
    return Path(getattr(cr, "PATH_OUT", ".")) / name


# ---------- core diagnostics ----------
def analyze_month(yymm: str) -> pd.DataFrame:
    p = _enhanced_in_path(yymm)
    if not p.exists():
        raise FileNotFoundError(f"Missing enhanced file: {p}")

    df = pd.read_parquet(p)
    if df.empty:
        print(f"[{yymm}] empty enhanced file.")
        return pd.DataFrame()

    need = {"ts","tenor_yrs","rate","z_comb"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{p} missing columns: {miss}")

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    decision_freq = str(getattr(cr, "DECISION_FREQ", "D")).upper()
    if decision_freq == "D":
        df["decision_ts"] = df["ts"].dt.floor("D")
    elif decision_freq == "H":
        df["decision_ts"] = df["ts"].dt.floor("H")
    else:
        raise ValueError("DECISION_FREQ must be 'D' or 'H'")

    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    Z_ENTRY       = float(getattr(cr, "Z_ENTRY", 0.75))
    PER_BUCKET_DV01_CAP = float(getattr(cr, "PER_BUCKET_DV01_CAP", 1.0))
    TOTAL_DV01_CAP      = float(getattr(cr, "TOTAL_DV01_CAP", 3.0))
    FRONT_END_DV01_CAP  = float(getattr(cr, "FRONT_END_DV01_CAP", 1.0))
    SHORT_END_EXTRA_Z   = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))

    rows = []
    for dts, snap in df.groupby("decision_ts", sort=True):
        snap_last = (snap.sort_values("ts")
                         .groupby("tenor_yrs", as_index=False)
                         .tail(1)
                         .reset_index(drop=True))
        rec = {
            "decision_ts": dts,
            "n_tenors": int(snap_last.shape[0]),
            "z_min": np.nan, "z_p50": np.nan, "z_p90": np.nan, "z_max": np.nan,
            "z_range": np.nan,
            "stage": "init",
            "cand_pairs": 0, "cand_after_fly": 0,
            "caps_feasible": np.nan,
            "why": ""
        }

        if snap_last.empty:
            rec["stage"] = "no_data"
            rows.append(rec); continue

        z = snap_last["z_comb"].astype(float).to_numpy()
        rec["z_min"] = np.nanmin(z) if z.size else np.nan
        rec["z_p50"] = np.nanpercentile(z, 50) if z.size else np.nan
        rec["z_p90"] = np.nanpercentile(z, 90) if z.size else np.nan
        rec["z_max"] = np.nanmax(z) if z.size else np.nan
        rec["z_range"] = (rec["z_max"] - rec["z_min"]) if (np.isfinite(rec["z_max"]) and np.isfinite(rec["z_min"])) else np.nan

        ten = snap_last["tenor_yrs"].astype(float).sort_values().to_numpy()
        if len(ten) < 2:
            rec["stage"] = "too_few_tenors"; rows.append(rec); continue
        sep_ok = (np.max(np.diff(ten)) >= MIN_SEP_YEARS) or (np.max(ten) - np.min(ten) >= MIN_SEP_YEARS)
        if not sep_ok:
            rec["stage"] = "sep_blocker"; rows.append(rec); continue

        if not (np.isfinite(rec["z_range"]) and rec["z_range"] >= Z_ENTRY):
            rec["stage"] = "z_bar_not_met"
            rows.append(rec); continue

        sig = snap_last[["tenor_yrs","rate","z_comb"]].dropna().sort_values("z_comb")
        L = len(sig)
        cand = 0; cand_fly = 0
        best_caps_feasible = False

        for k_low in range(min(6, L)):
            rich = sig.iloc[k_low]
            for k_hi in range(1, min(10, L)+1):
                cheap = sig.iloc[-k_hi]
                t_i = _to_float(cheap["tenor_yrs"]); t_j = _to_float(rich["tenor_yrs"])
                if abs(t_i - t_j) < MIN_SEP_YEARS:
                    continue
                zdisp = _to_float(cheap["z_comb"]) - _to_float(rich["z_comb"])
                if not np.isfinite(zdisp) or (zdisp < Z_ENTRY):
                    continue
                cand += 1

                ok_i = fly_alignment_ok(t_i, +1, snap_last, zdisp_for_pair=zdisp)
                ok_j = fly_alignment_ok(t_j, -1, snap_last, zdisp_for_pair=zdisp)
                if not (ok_i and ok_j):
                    continue
                cand_fly += 1

                # quick caps feasibility for THIS pair:
                r_i = _to_float(cheap["rate"]); r_j = _to_float(rich["rate"])
                pv_i = pv01_proxy(t_i, r_i)
                pv_j = pv01_proxy(t_j, r_j)
                w_i = 1.0
                w_j = - w_i * pv_i / max(1e-9, pv_j)

                b_i = assign_bucket(t_i); b_j = assign_bucket(t_j)
                dv_i = abs(w_i) * pv_i; dv_j = abs(w_j) * pv_j

                per_bucket_ok = True
                if b_i in getattr(cr, "BUCKETS", {}) and (dv_i > PER_BUCKET_DV01_CAP): per_bucket_ok = False
                if b_j in getattr(cr, "BUCKETS", {}) and (dv_j > PER_BUCKET_DV01_CAP): per_bucket_ok = False

                front_end_used = (dv_i if b_i=="short" else 0.0) + (dv_j if b_j=="short" else 0.0)
                fe_ok = front_end_used <= FRONT_END_DV01_CAP
                total_ok = (dv_i + dv_j) <= TOTAL_DV01_CAP

                pair_ok = per_bucket_ok and fe_ok and total_ok

                if (b_i=="short" or b_j=="short") and (zdisp < (Z_ENTRY + SHORT_END_EXTRA_Z)):
                    pair_ok = False

                best_caps_feasible = best_caps_feasible or pair_ok

        rec["cand_pairs"] = cand
        rec["cand_after_fly"] = cand_fly
        rec["caps_feasible"] = bool(best_caps_feasible)

        if cand == 0:
            rec["stage"] = "no_pair_meets_threshold"
        elif cand_fly == 0:
            rec["stage"] = "fly_blocker"
        elif not best_caps_feasible:
            rec["stage"] = "caps_blocker"
        else:
            rec["stage"] = "ok"

        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("decision_ts")
    csv_path = _diag_csv_out(yymm)
    out.to_csv(csv_path, index=False)

    # Text summary
    zr = out["z_range"].describe(percentiles=[.5,.9]).to_dict()
    ok = int((out["stage"]=="ok").sum())
    summary = out["stage"].value_counts(dropna=False).to_dict()

    msg = []
    msg.append(f"[DIAG] {yymm}")
    msg.append(f"path: {p}")
    msg.append(f"decisions: {out.shape[0]}")
    msg.append(f"z_range min/p50/p90/max: "
               f"{zr.get('min',np.nan):.4f}/"
               f"{zr.get('50%',np.nan):.4f}/"
               f"{zr.get('90%',np.nan):.4f}/"
               f"{zr.get('max',np.nan):.4f}")
    msg.append(f"stages: {summary}")
    msg.append(f"ok_decisions: {ok}")

    txt_path = _diag_txt_out(yymm)
    with open(txt_path, "w") as f:
        f.write("\n".join(msg) + "\n")

    print("\n".join(msg))
    print(f"[SAVE] {csv_path}")
    print(f"[SAVE] {txt_path}")
    return out


# -------------- CLI --------------
if __name__ == "__main__":
    yymms = sys.argv[1:]
    if not yymms:
        print("Usage: python diag_selector.py 2304 [2305 2306 ...]")
        sys.exit(0)
    for y in yymms:
        try:
            analyze_month(y)
        except Exception as e:
            print(f"[ERROR] {y}: {e}")
