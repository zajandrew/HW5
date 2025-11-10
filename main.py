# cr_config.py
from zoneinfo import ZoneInfo
from pathlib import Path

# ========= Paths =========
PATH_DATA   = r"C:/Users/zajohns/OneDrive - US Bank/Desktop/AI Hedging/downloads/Features_CPU"   # raw monthly wide parquet
PATH_ENH    = r"C:/Users/zajohns/OneDrive - US Bank/Desktop/AI Hedging/RV Strategy/CR_Sandbox/enhanced"
PATH_MODELS = r"C:/Users/zajohns/OneDrive - US Bank/Desktop/AI Hedging/RV Strategy/CR_Sandbox/models"
PATH_OUT    = r"C:/Users/zajohns/OneDrive - US Bank/Desktop/AI Hedging/RV Strategy/CR_Sandbox/out"

Path(PATH_ENH).mkdir(parents=True, exist_ok=True)
Path(PATH_MODELS).mkdir(parents=True, exist_ok=True)
Path(PATH_OUT).mkdir(parents=True, exist_ok=True)

# ========= Instruments (explicit) =========
# Provide your exact mapping (instrument root WITHOUT "_mid" -> tenor years).
TENOR_YEARS = {
    "USOSFRA BGN Curncy": 1/12,   "USOSFRB BGN Curncy": 2/12,   "USOSFRC BGN Curncy": 3/12,
    "USOSFRD BGN Curncy": 4/12,   "USOSFRE BGN Curncy": 5/12,   "USOSFRF BGN Curncy": 6/12,
    "USOSFRG BGN Curncy": 7/12,   "USOSFRH BGN Curncy": 8/12,   "USOSFRI BGN Curncy": 9/12,
    "USOSFRJ BGN Curncy": 10/12,  "USOSFRK BGN Curncy": 11/12,  "USOSFR1 BGN Curncy": 1,
    "USOSFR1F BGN Curncy": 18/12, "USOSFR2 BGN Curncy": 2,      "USOSFR3 BGN Curncy": 3,
    "USOSFR4 BGN Curncy": 4,      "USOSFR5 BGN Curncy": 5,      "USOSFR6 BGN Curncy": 6,
    "USOSFR7 BGN Curncy": 7,      "USOSFR8 BGN Curncy": 8,      "USOSFR9 BGN Curncy": 9,
    "USOSFR10 BGN Curncy": 10,    "USOSFR12 BGN Curncy": 12,    "USOSFR15 BGN Curncy": 15,
    "USOSFR20 BGN Curncy": 20,    "USOSFR25 BGN Curncy": 25,    "USOSFR30 BGN Curncy": 30,
    "USOSFR40 BGN Curncy": 40,
}

# ========= Calendar filtering =========
USE_QL_CALENDAR = True
QL_US_MARKET    = "FederalReserve"
CAL_TZ          = "America/New_York"
TRADING_HOURS = ("07:00", "17:30")

# ========= Feature (builder) settings =========
# PCA trained on a rolling panel built at the chosen decision frequency (D/H).
PCA_COMPONENTS     = 3         # number of components to keep
PCA_LOOKBACK_DAYS  = 126       # original setting (≈ 6 months of trading days)

# Decision frequency for BOTH feature buckets and the backtest layer
DECISION_FREQ      = 'D'       # 'D' (daily) or 'H' (hourly)

# Enable/disable PCA; when disabled we still compute spline residuals and z_comb will fallback.
PCA_ENABLE         = True

# Parallelism for feature creation (0 = auto ~ half the cores up to a small cap)
N_JOBS             = 0

# ---- Derived: convert day lookback to "bucket" lookback used by the builder
# If daily, 126 days means 126 buckets.
# If hourly, we expand to hours but cap to avoid huge fits (2 weeks cap by default).
PCA_LOOKBACK_CAP_HOURS = 24 * 14
if DECISION_FREQ == 'D':
    PCA_LOOKBACK = int(PCA_LOOKBACK_DAYS)                    # 126 --> 126 daily buckets
else:  # 'H'
    PCA_LOOKBACK = max(1, min(PCA_LOOKBACK_DAYS * 24, PCA_LOOKBACK_CAP_HOURS))

# ========= Backtest decision layer =========
Z_ENTRY       = 0.75     # enter when cheap-rich z-spread >= Z_ENTRY
Z_EXIT        = 0.40     # take profit when |z-spread| <= Z_EXIT
Z_STOP        = 3.00     # stop if divergence since entry >= Z_STOP
MAX_HOLD_DAYS = 10       # max holding period for a pair (days when DECISION_FREQ='D')

# ========= Risk & selection =========
BUCKETS = {
    "short": (0.4, 1.9),   # ~6M–<2Y
    "front": (2.0, 3.0),
    "belly": (3.1, 9.0),
    "long" : (10.0, 40.0),
}
MIN_SEP_YEARS        = 0.5
MAX_CONCURRENT_PAIRS = 3
PER_BUCKET_DV01_CAP  = 1.0
TOTAL_DV01_CAP       = 3.0
FRONT_END_DV01_CAP   = 1.0

# Fly-alignment gating (optional micro-sanity gate, no calendar dependency)
FLY_GATE_ENABLE = True
FLY_DEFS        = [(1.0, 3.0, 5.0), (2.0, 5.0, 10.0)]
FLY_Z_MIN       = 0.3
FLY_ALIGN_MODE  = "loose"   # "loose" (reject opposite sign) or "strict" (must agree)
SHORT_END_EXTRA_Z = 0.3     # extra entry threshold if any leg is in 'short'


#diag_backtest.py
import os, sys, json, numpy as np, pandas as pd
from pathlib import Path

from cr_config import (
    PATH_ENH, PATH_OUT,
    DECISION_FREQ, MIN_SEP_YEARS, Z_ENTRY
)

os.makedirs(PATH_OUT, exist_ok=True)

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _dispersion_ok(snap_last: pd.DataFrame, z_entry: float, min_sep_years: float) -> (bool, str, dict):
    """
    Decide if this bucket has at least one tradable pair under very simple rules:
      - at least 2 tenors with separation >= MIN_SEP_YEARS
      - z-dispersion (max-min) >= Z_ENTRY
    Returns: (ok, reason, extras)
    """
    extras = {}
    if snap_last.empty:
        return False, "no_data", extras

    ten = snap_last["tenor_yrs"].to_numpy()
    if len(ten) < 2:
        return False, "too_few_tenors", extras

    ten_sorted = np.sort(ten)
    if np.max(np.diff(ten_sorted)) < min_sep_years and (ten_sorted[-1]-ten_sorted[0] < min_sep_years):
        # extremely conservative “sep blocker”: every pair too close
        return False, "sep_blocker", extras

    z = snap_last["z_comb"].astype(float).to_numpy()
    if z.size == 0 or np.all(~np.isfinite(z)):
        return False, "no_data", extras

    z_rng = np.nanmax(z) - np.nanmin(z)
    extras["z_range"] = _safe_float(z_rng)
    if z_rng >= float(z_entry):
        return True, "ok", extras
    return False, "no_pair_meets_threshold", extras

def _bucket_ts(df: pd.DataFrame, decision_freq: str) -> pd.Series:
    if decision_freq.upper() == "D":
        return df["ts"].dt.floor("D")
    elif decision_freq.upper() == "H":
        return df["ts"].dt.floor("H")
    else:
        raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

def check_month(yymm: str) -> dict:
    """Return a plain dict of numeric/string fields (safe for Parquet), also write a TXT report."""
    path = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    out_txt = Path(PATH_OUT) / f"diag_{yymm}.txt"

    if not path.exists():
        msg = f"[WARN] missing enhanced file: {path}"
        print(msg)
        out_txt.write_text(msg)
        return {
            "yymm": yymm, "rows": 0, "dec": 0, "medTenors": np.nan,
            "zmin": np.nan, "zp50": np.nan, "zp90": np.nan, "zmax": np.nan,
            "ok_samples": 0,
            "n_no_data": 0, "n_too_few_tenors": 0, "n_no_pair_meets_threshold": 0, "n_sep_blocker": 0, "n_ok": 0
        }

    df = pd.read_parquet(path)
    if df.empty:
        out_txt.write_text(f"[WARN] empty enhanced file: {path}")
        return {
            "yymm": yymm, "rows": 0, "dec": 0, "medTenors": np.nan,
            "zmin": np.nan, "zp50": np.nan, "zp90": np.nan, "zmax": np.nan,
            "ok_samples": 0,
            "n_no_data": 0, "n_too_few_tenors": 0, "n_no_pair_meets_threshold": 0, "n_sep_blocker": 0, "n_ok": 0
        }

    # Type hygiene
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "tenor_yrs", "z_comb"])
    if df.empty:
        out_txt.write_text(f"[WARN] all NaN after basic hygiene: {path}")
        return {
            "yymm": yymm, "rows": 0, "dec": 0, "medTenors": np.nan,
            "zmin": np.nan, "zp50": np.nan, "zp90": np.nan, "zmax": np.nan,
            "ok_samples": 0,
            "n_no_data": 0, "n_too_few_tenors": 0, "n_no_pair_meets_threshold": 0, "n_sep_blocker": 0, "n_ok": 0
        }

    df["bucket"] = _bucket_ts(df, DECISION_FREQ)

    # Per-bucket last tick per tenor
    reasons = {"no_data":0, "too_few_tenors":0, "no_pair_meets_threshold":0, "sep_blocker":0, "ok":0}
    z_ranges = []
    ok_samples = 0
    decisions = 0
    tenors_per_bucket = []

    for bkt, snap in df.groupby("bucket", sort=True):
        decisions += 1
        snap_last = (snap.sort_values("ts")
                          .groupby("tenor_yrs", as_index=False)
                          .tail(1)[["tenor_yrs","rate","z_comb"]]
                          .dropna(subset=["tenor_yrs","z_comb"]))
        tenors_per_bucket.append(snap_last["tenor_yrs"].nunique())

        ok, reason, extras = _dispersion_ok(snap_last, Z_ENTRY, MIN_SEP_YEARS)
        reasons[reason] = reasons.get(reason, 0) + 1
        if "z_range" in extras and np.isfinite(extras["z_range"]):
            z_ranges.append(extras["z_range"])
        if ok:
            ok_samples += 1

    # summarize z stats
    z_ranges = np.array(z_ranges, dtype=float)
    zmin = _safe_float(np.nanmin(z_ranges)) if z_ranges.size else np.nan
    zp50 = _safe_float(np.nanpercentile(z_ranges, 50)) if z_ranges.size else np.nan
    zp90 = _safe_float(np.nanpercentile(z_ranges, 90)) if z_ranges.size else np.nan
    zmax = _safe_float(np.nanmax(z_ranges)) if z_ranges.size else np.nan

    medTen = _safe_float(np.nanmedian(np.array(tenors_per_bucket))) if tenors_per_bucket else np.nan

    # Write a small human-readable TXT
    lines = []
    lines.append(f"[DIAG] {yymm}")
    lines.append(f"path: {path}")
    lines.append(f"rows: {len(df)}")
    lines.append(f"decisions: {decisions} | median tenors: {medTen}")
    lines.append(f"z_range min/p50/p90/max: {zmin:.4f}/{zp50:.4f}/{zp90:.4f}/{zmax:.4f}" if np.isfinite(zp50) else "z_range: (no valid buckets)")
    lines.append(f"reasons: {json.dumps(reasons)}")
    lines.append(f"ok_samples: {ok_samples}")
    out_txt.write_text("\n".join(lines))

    # Return a parquet-safe dict (no tuples/dicts)
    return {
        "yymm": str(yymm),
        "rows": int(len(df)),
        "dec": int(decisions),
        "medTenors": _safe_float(medTen),
        "zmin": _safe_float(zmin),
        "zp50": _safe_float(zp50),
        "zp90": _safe_float(zp90),
        "zmax": _safe_float(zmax),
        "ok_samples": int(ok_samples),
        "n_no_data": int(reasons.get("no_data",0)),
        "n_too_few_tenors": int(reasons.get("too_few_tenors",0)),
        "n_no_pair_meets_threshold": int(reasons.get("no_pair_meets_threshold",0)),
        "n_sep_blocker": int(reasons.get("sep_blocker",0)),
        "n_ok": int(reasons.get("ok",0)),
    }

def main():
    yymms = sys.argv[1:] if len(sys.argv) > 1 else []
    if not yymms:
        print("Usage: python diag_backtest.py 2304 [2305 2306 ...]")
        sys.exit(0)

    rows = []
    print(f"[INFO] months: {len(yymms)} -> {yymms}")
    for y in yymms:
        print(f"[RUN] {y}")
        rec = check_month(y)
        rows.append(rec)
        print(f"[DONE] {y} | ok_samples={rec['ok_samples']}")

    df_out = pd.DataFrame(rows)

    # Ensure pyarrow-safe dtypes (no objects except strings)
    for c in df_out.columns:
        if df_out[c].dtype == "object":
            df_out[c] = df_out[c].astype(str)

    out_parq = Path(PATH_OUT) / "diag_summary.parquet"
    df_out.to_parquet(out_parq, index=False)
    print(f"[SAVE] {out_parq}")

if __name__ == "__main__":
    main()


# feature_creation.py
import os, sys, time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cr_config import (
    PATH_DATA, PATH_ENH,
    CAL_TZ, USE_QL_CALENDAR, QL_US_MARKET,
    TRADING_HOURS,                    # ("07:00","17:30") in America/New_York
    PCA_ENABLE, PCA_LOOKBACK_DAYS, PCA_COMPONENTS,
    DECISION_FREQ,                    # 'D' or 'H'
    TENOR_YEARS,                      # {"USOSFR1 BGN Curncy": 1, ...}
)

# Optional; if missing we treat as 1
try:
    from cr_config import N_JOBS  # 0 → auto
except Exception:
    N_JOBS = 1


# -----------------------
# Small utilities
# -----------------------
def _now():
    return time.strftime("%H:%M:%S")

def _to_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a UTC-naive DatetimeIndex column 'ts' exists and is sorted."""
    if "ts" not in df.columns:
        # user said index name may be "sec" in older files; handle both
        if df.index.name in ("ts", "sec"):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "ts"})
        else:
            raise KeyError("No 'ts' column or index found.")
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    # convert to tz-naive UTC (consistent downstream)
    df["ts"] = df["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("ts")


# -----------------------
# Calendar & hours
# -----------------------
def _get_ql_calendar():
    if not USE_QL_CALENDAR:
        return None
    try:
        import QuantLib as ql
    except Exception:
        print("[CAL] QuantLib not available; falling back to weekday filter.")
        return None

    try:
        # Try direct enum (most builds): ql.UnitedStates.FederalReserve
        direct = getattr(ql.UnitedStates, str(QL_US_MARKET), None)
        if direct is not None:
            return ql.UnitedStates(direct)

        # Fallback: nested Market enum
        market_enum_cls = getattr(ql.UnitedStates, "Market", None)
        if market_enum_cls is not None:
            mkt = getattr(market_enum_cls, str(QL_US_MARKET), None)
            if mkt is not None:
                return ql.UnitedStates(mkt)

        # Last resort defaults
        if hasattr(ql.UnitedStates, "FederalReserve"):
            return ql.UnitedStates(ql.UnitedStates.FederalReserve)
        if hasattr(ql.UnitedStates, "NYSE"):
            return ql.UnitedStates(ql.UnitedStates.NYSE)
        return ql.UnitedStates()
    except Exception as e:
        print(f"[CAL] Failed to init QuantLib calendar: {e}; using weekday filter.")
        return None


def _apply_calendar_and_hours(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Filter to business days (QuantLib if available) + local trading hours."""
    if df_wide.empty:
        return df_wide

    # 1) Business-day filter (QuantLib if available, else Mon–Fri)
    cal = _get_ql_calendar()
    ts = pd.to_datetime(df_wide["ts"])
    if cal is not None:
        try:
            import QuantLib as ql
            ymd = np.array([(t.year, t.month, t.day) for t in ts], dtype=int)
            days = pd.to_datetime(
                {"year": ymd[:, 0], "month": ymd[:, 1], "day": ymd[:, 2]}
            ).drop_duplicates().sort_values()
            bd_mask = []
            for d in days:
                qd = ql.Date(int(d.day), int(d.month), int(d.year))
                bd_mask.append(cal.isBusinessDay(qd))
            day_ok = pd.Series(bd_mask, index=days)
            ok = day_ok.reindex(ts.dt.floor("D"), fill_value=False).values
            df_wide = df_wide.loc[ok]
            print(f"[CAL] QuantLib calendar active ({QL_US_MARKET}); days={day_ok.sum()}")
        except Exception as e:
            print(f"[CAL] QuantLib daily check failed: {e}; using weekday filter.")
            df_wide = df_wide[ts.dt.weekday < 5]
            print(f"[CAL] Simple Mon–Fri filter active; "
                  f"days={df_wide['ts'].dt.floor('D').nunique()}")
    else:
        df_wide = df_wide[ts.dt.weekday < 5]
        print(f"[CAL] Simple Mon–Fri filter active; "
              f"days={df_wide['ts'].dt.floor('D').nunique()}")

    # 2) Session hours in local tz (e.g., America/New_York)
    tz_local = CAL_TZ
    start_str, end_str = TRADING_HOURS
    pre = len(df_wide)

    # Make a tz-aware local-time index specifically for slicing,
    # then convert back to UTC-naive.
    df_wide = df_wide.copy()
    df_wide["ts_local"] = df_wide["ts"].dt.tz_localize("UTC").dt.tz_convert(tz_local)

    tmp = df_wide.set_index("ts_local").sort_index()
    tmp = tmp.between_time(start_str, end_str)

    # Recreate UTC-naive 'ts' from the local index; then drop the index safely.
    tmp = tmp.copy()
    tmp["ts"] = tmp.index.tz_convert("UTC").tz_localize(None)
    tmp = tmp.reset_index(drop=True)  # removes 'ts_local' index without needing .drop(columns=...)

    kept = len(tmp)
    if pre > 0:
        print(f"[CAL] kept {kept:,}/{pre:,} rows ({(kept/pre*100):.2f}%) after calendar+hours")

    # Ensure only expected columns remain
    if "ts_local" in tmp.columns:
        tmp = tmp.drop(columns=["ts_local"], errors="ignore")

    return tmp.reset_index(drop=True)


# -----------------------
# Cleaning & reshaping
# -----------------------
def _zeros_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric, turn literal 0.0 to NaN (but not timestamps)."""
    num = df.drop(columns=["ts"])
    num = num.apply(pd.to_numeric, errors="coerce")
    num = num.mask(num == 0)
    out = df[["ts"]].join(num)
    return out


def _melt_long(df_wide: pd.DataFrame, tenormap: Dict[str, float]) -> pd.DataFrame:
    """Wide → long using tolerant column matching to TENOR_YEARS."""
    def norm(s: str) -> str:
        s = str(s).strip()
        if s.endswith("_mid"):
            s = s[:-4]
        return " ".join(s.split())

    tenormap_norm = {norm(k): v for k, v in tenormap.items()}

    cand = []
    for c in df_wide.columns:
        if c == "ts":
            continue
        nc = norm(c)
        if nc in tenormap_norm:
            cand.append((c, tenormap_norm[nc]))

    if not cand:
        sample_cols = [c for c in df_wide.columns if c != "ts"][:20]
        raise ValueError(
            "No overlapping tickers between input file and TENOR_YEARS.\n"
            f"Example file columns: {sample_cols}\n"
            f"Example tenor keys: {list(tenormap.keys())[:20]}"
        )

    use_cols = ["ts"] + [c for c, _ in cand]
    df_sel = df_wide[use_cols].copy()

    col_to_tenor = {c: t for c, t in cand}
    long = df_sel.melt(id_vars="ts", var_name="instrument", value_name="rate")
    long["tenor_yrs"] = long["instrument"].map(col_to_tenor).astype(float)

    long["rate"] = pd.to_numeric(long["rate"], errors="coerce")
    long = long.dropna(subset=["ts", "tenor_yrs", "rate"])
    return long[["ts", "tenor_yrs", "rate"]]


def _decision_key(ts: pd.Series, freq: str) -> pd.Series:
    if freq == "D":
        return ts.dt.floor("D")
    elif freq == "H":
        return ts.dt.floor("H")
    else:
        raise ValueError("DECISION_FREQ must be 'D' or 'H'.")


# -----------------------
# Spline “shape” z (robust)
# -----------------------
def _spline_fit_safe(snap_long: pd.DataFrame) -> pd.Series:
    """
    Per-bucket cross-section: fit low-order polynomial (cubic) to rate(tenor),
    z-score residuals using robust scale.
    """
    s = snap_long[["tenor_yrs", "rate"]].dropna()
    out = pd.Series(index=snap_long.index, dtype=float)

    if s.shape[0] < 5:
        return out  # leave NaN

    x = s["tenor_yrs"].values.astype(float)
    y = s["rate"].values.astype(float)

    # degree 3 if possible, else lower
    deg = 3 if len(x) >= 4 else min(2, len(x)-1)
    try:
        coef = np.polyfit(x, y, deg=deg)
        fit = np.polyval(coef, x)
        resid = y - fit

        # robust scale: MAD or fallback to std
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        scale = (1.4826 * mad) if mad > 0 else resid.std(ddof=1)
        if not np.isfinite(scale) or scale == 0:
            return out

        z = (resid - resid.mean()) / scale
        m = {ten: val for ten, val in zip(x, z)}
        out.loc[s.index] = s["tenor_yrs"].map(m).values
        return out
    except Exception:
        return out


# -----------------------
# PCA helpers (fit on lookback restricted to current columns)
# -----------------------
def _pca_fit_panel(panel_long: pd.DataFrame, cols_ordered: List[float], n_comps: int):
    """Return dict{'cols','mean','components','evr'} or None."""
    if panel_long.empty:
        return None
    W = (panel_long.pivot(index="ts", columns="tenor_yrs", values="rate")
                  .sort_index())
    W = W.reindex(columns=cols_ordered).ffill().dropna(how="any")
    if W.shape[0] < (n_comps + 5) or W.shape[1] < n_comps:
        return None

    X = W.values.astype(float)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    comps = VT[:n_comps, :]           # (k, n_features)
    evr = (S**2) / (S**2).sum()
    return {"cols": list(W.columns), "mean": mu.ravel(), "components": comps, "evr": evr[:n_comps]}


def _pca_apply_block(df_block: pd.DataFrame, pca_model: dict) -> pd.Series:
    """
    Robust application of a previously-fitted PCA model to one decision bucket.
    - Chooses the LAST tick per tenor inside the bucket
    - Aligns strictly to model['cols'] (both set & order)
    - Returns a Series of standardized reconstructed scores mapped to df_block rows
    """
    out = pd.Series(index=df_block.index, dtype=float)
    if not pca_model or df_block.empty:
        return out

    cols  = list(pca_model["cols"])
    mu    = np.asarray(pca_model["mean"], dtype=float)
    comps = np.asarray(pca_model["components"], dtype=float)

    # Last observation per tenor in this bucket
    # (guards against multiple ticks per tenor within the bucket)
    last = (df_block.sort_values("ts")
                     .groupby("tenor_yrs", as_index=False)
                     .tail(1)
                     .set_index("tenor_yrs")["rate"])
    # Align to model cols (exact set + order)
    if any(c not in last.index for c in cols):
        # Missing a required tenor → skip PCA for this bucket
        return out

    x = last.reindex(cols).values.astype(float)   # shape (n_features,)
    xc = x - mu                                   # center
    score = comps @ xc                            # (k,)
    recon = comps.T @ score                       # (n_features,)

    # Standardize reconstruction across features for a z-like shape
    sd = recon.std()
    if not np.isfinite(sd) or sd == 0:
        return out
    z_std = (recon - recon.mean()) / sd

    # Map back to tenor_yrs, then to the original df_block rows
    z_map = dict(zip(cols, z_std))
    return df_block["tenor_yrs"].map(z_map)

# -----------------------
# Per-bucket processor
# -----------------------
def _process_bucket(dts, df_bucket, df_all, lookback_days, pca_enable, pca_n_comps, yymm):
    out = df_bucket[["ts","tenor_yrs","rate"]].copy().reset_index(drop=True)

    # 1) spline z
    out["z_spline"] = _spline_fit_safe(out)

    # 2) PCA z (fit on lookback restricted to current columns)
    out["z_pca"] = np.nan
    if pca_enable:
        cols_now = sorted(df_bucket["tenor_yrs"].unique().tolist())
        if len(cols_now) >= pca_n_comps:
            t_end   = df_bucket["ts"].min()
            t_start = t_end - pd.Timedelta(days=float(lookback_days))
            panel = df_all[(df_all["ts"]>=t_start) & (df_all["ts"]<t_end) &
                           (df_all["tenor_yrs"].isin(cols_now))]
            model = _pca_fit_panel(panel, cols_now, pca_n_comps)
            if model:
                evr = list(np.round(model["evr"], 3))
                print(f"[PCA] {yymm} {str(dts)[:16]} obs={len(panel):,} cols={len(cols_now)} EVR={evr}")
                out["z_pca"] = _pca_apply_block(out, model)
            else:
                print(f"[WARN] PCA skipped {yymm} {dts}: insufficient history/features "
                      f"(cols={len(cols_now)}, lookback_days={lookback_days})")

    # 3) Combine
    if out["z_pca"].notna().any():
        out["z_comb"] = 0.5*out["z_spline"] + 0.5*out["z_pca"]
    else:
        out["z_comb"] = out["z_spline"]

    return out


# -----------------------
# Month builder (public)
# -----------------------
def build_month(yymm: str) -> None:
    in_path  = Path(PATH_DATA) / f"{yymm}.parquet"
    out_path = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    Path(PATH_ENH).mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing raw month file: {in_path}")

    # Load & normalize
    df_wide = pd.read_parquet(in_path)
    df_wide = _to_ts_index(df_wide)
    df_wide = df_wide[~df_wide["ts"].duplicated(keep="last")]
    # Calendar + hours
    df_wide = _apply_calendar_and_hours(df_wide)
    # Clean
    df_wide = _zeros_to_nan(df_wide)

    # Long
    df_long = _melt_long(df_wide, TENOR_YEARS)
    df_long["decision_ts"] = _decision_key(df_long["ts"], DECISION_FREQ)

    buckets = (df_long["decision_ts"].dropna().unique().tolist())
    buckets.sort()

    # jobs
    if isinstance(N_JOBS, int):
        if N_JOBS == 0:
            import multiprocessing as mp
            jobs = max(1, min((mp.cpu_count() // 2), 8))
        else:
            jobs = int(N_JOBS)
    else:
        jobs = 1

    print(f"[{_now()}] [MONTH] {yymm} buckets={len(buckets)} freq={DECISION_FREQ} | jobs={jobs} "
          f"| PCA={'on' if PCA_ENABLE else 'off'} | lookback={PCA_LOOKBACK_DAYS} | comps={PCA_COMPONENTS}")

    if not buckets:
        pd.DataFrame(columns=['ts','tenor_yrs','rate','z_spline','z_pca','z_comb']).to_parquet(out_path, index=False)
        print(f"[SAVE] {out_path}")
        return

    def _one(dts):
        snap = df_long[df_long["decision_ts"] == dts]
        t0 = time.time()
        out = _process_bucket(
            dts=dts,
            df_bucket=snap,
            df_all=df_long[['ts','tenor_yrs','rate','decision_ts']],
            lookback_days=PCA_LOOKBACK_DAYS,
            pca_enable=PCA_ENABLE,
            pca_n_comps=int(PCA_COMPONENTS),
            yymm=yymm
        )
        dt = time.time() - t0
        print(f"[BUCKET] {yymm} {dts} rows:{len(snap):,} tenors:{snap['tenor_yrs'].nunique()} "
              f"PCA:{'yes' if out['z_pca'].notna().any() else 'no '} t={dt:.2f}s")
        return out

    parts = Parallel(n_jobs=jobs, backend="loky")(delayed(_one)(d) for d in buckets)
    out = pd.concat(parts, ignore_index=True).sort_values(['ts','tenor_yrs']).reset_index(drop=True)

    zr = pd.to_numeric(out['z_comb'], errors='coerce')
    z_valid_pct = float(np.isfinite(zr).mean() * 100.0) if not out.empty else 0.0
    med_ten = out.groupby('ts')['tenor_yrs'].nunique().median() if not out.empty else 0
    print(f"[DONE] {yymm} rows:{len(out):,} tenors_med:{med_ten:.1f} z_valid%:{z_valid_pct:.2f}")

    out.to_parquet(out_path, index=False)
    print(f"[SAVE] {out_path}")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_creation.py 2304 [2305 2306 ...]")
        sys.exit(1)
    for m in sys.argv[1:]:
        build_month(m)

# portfolio_test.py
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cr_config import (
    PATH_ENH, PATH_OUT,
    DECISION_FREQ,
    Z_ENTRY, Z_EXIT, Z_STOP, MAX_HOLD_DAYS,
    BUCKETS, MIN_SEP_YEARS,
    MAX_CONCURRENT_PAIRS, PER_BUCKET_DV01_CAP, TOTAL_DV01_CAP, FRONT_END_DV01_CAP,
    FLY_GATE_ENABLE, FLY_DEFS, FLY_Z_MIN, FLY_ALIGN_MODE, SHORT_END_EXTRA_Z,
)

# ------------------------
# Utilities / conventions
# ------------------------
os.makedirs(PATH_OUT, exist_ok=True)

def _to_float(x, default=np.nan):
    """Safe scalar float extraction (works for scalar, 1-elem Series, or np types)."""
    try:
        if isinstance(x, (pd.Series, pd.Index)):
            if len(x) == 0:
                return default
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return default

def pv01_proxy(tenor_yrs, rate_pct):
    """Simple PV01 proxy so pair is roughly DV01-neutral."""
    return tenor_yrs / max(1e-6, 1.0 + 0.01 * rate_pct)

def assign_bucket(tenor):
    for name, (lo, hi) in BUCKETS.items():
        if (tenor >= lo) and (tenor <= hi):
            return name
    return "other"

# ------------------------
# Fly alignment (optional)
# ------------------------
def _row_for_tenor(snap_last: pd.DataFrame, tenor: float) -> pd.Series | None:
    r = snap_last.loc[snap_last["tenor_yrs"] == tenor]
    if r.empty:
        return None
    return r.iloc[0]

def compute_fly_z(snap_last: pd.DataFrame, a: float, b: float, c: float) -> float | None:
    """
    Compute 0.5*(a+c)-b on the *z_comb* cross-section, then standardize by xs std (same cross-section).
    """
    ra = _row_for_tenor(snap_last, a)
    rb = _row_for_tenor(snap_last, b)
    rc = _row_for_tenor(snap_last, c)
    if (ra is None) or (rb is None) or (rc is None):
        return None
    za = _to_float(ra["z_comb"])
    zb = _to_float(rb["z_comb"])
    zc = _to_float(rc["z_comb"])
    if any(np.isnan([za, zb, zc])):
        return None
    fly_shape = 0.5 * (za + zc) - zb
    xs = snap_last["z_comb"].astype(float).to_numpy()
    sd = np.nanstd(xs, ddof=1) if xs.size > 1 else 1.0
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return fly_shape / sd

def fly_alignment_ok(leg_tenor: float, leg_sign_z: int, snap_last: pd.DataFrame) -> bool:
    """
    Gate entries using fly info. If disabled, always True.
    leg_sign_z: +1 if we expect z to fall (cheap->rich), -1 if z to rise (rich->cheap).
    """
    if not FLY_GATE_ENABLE:
        return True
    for (a, b, c) in FLY_DEFS:
        fz = compute_fly_z(snap_last, a, b, c)
        if fz is None:
            continue
        if abs(fz) < FLY_Z_MIN:
            continue
        if FLY_ALIGN_MODE == "strict" and np.sign(fz) != np.sign(leg_sign_z):
            return False
        if FLY_ALIGN_MODE == "loose" and (np.sign(fz) * np.sign(leg_sign_z) < 0):
            return False
    return True

# ------------------------
# Pair object
# ------------------------
class PairPos:
    def __init__(self, open_ts, cheap_row, rich_row, w_i, w_j, decisions_per_day: int):
        self.open_ts = open_ts

        self.tenor_i = _to_float(cheap_row["tenor_yrs"])
        self.rate_i  = _to_float(cheap_row["rate"])
        self.tenor_j = _to_float(rich_row["tenor_yrs"])
        self.rate_j  = _to_float(rich_row["rate"])

        self.w_i = float(w_i); self.w_j = float(w_j)

        zi = _to_float(cheap_row["z_comb"])
        zj = _to_float(rich_row["z_comb"])
        self.entry_zspread = zi - zj

        self.closed = False
        self.close_ts = None
        self.exit_reason = None
        self.pnl = 0.0

        # bookkeeping / aging
        self.decisions_per_day = decisions_per_day
        self.age_decisions = 0  # increments by 1 each mark

        self.bucket_i = assign_bucket(self.tenor_i)
        self.bucket_j = assign_bucket(self.tenor_j)

        # attribution proxy
        self.last_zspread = self.entry_zspread
        self.conv_pnl_proxy = 0.0

    def mark(self, snap_last: pd.DataFrame):
        """Mark-to-market at decision time using last rate per tenor; update convergence proxy."""
        ri = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "rate"])
        rj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "rate"])

        # Pair PnL: (old - new) * weight * 100  (rates in % → *100 to bps)
        d_i = (self.rate_i - ri) * self.w_i * 100.0
        d_j = (self.rate_j - rj) * self.w_j * 100.0
        self.pnl = d_i + d_j

        zi = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "z_comb"])
        zj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "z_comb"])
        zsp = zi - zj

        if np.isfinite(zsp) and np.isfinite(self.last_zspread):
            self.conv_pnl_proxy += (self.last_zspread - zsp) * 10.0
        self.last_zspread = zsp

        # age by one decision step
        self.age_decisions += 1
        return zsp

# ------------------------
# Greedy selector with caps
# ------------------------
def choose_pairs_under_caps(
    snap_last: pd.DataFrame,
    max_pairs: int,
    per_bucket_cap: float,
    total_cap: float,
    front_end_cap: float,
    extra_z_entry: float,
) -> list[tuple[pd.Series, pd.Series, float, float]]:
    """
    Returns a list of (cheap_row, rich_row, w_i, w_j).
    Greedy across extremes of z_comb, with tenor uniqueness and DV01 caps.
    """
    sig = snap_last[["tenor_yrs", "rate", "z_comb"]].dropna().copy()
    if sig.empty:
        return []

    sig = sig.sort_values("z_comb")
    used_tenors: set[float] = set()
    candidates: list[tuple[float, pd.Series, pd.Series]] = []

    # Build candidate pairs from extremes
    L = len(sig)
    for k_low in range(min(6, L)):          # pick a few richest
        rich = sig.iloc[k_low]
        for k_hi in range(1, min(10, L) + 1):  # and several cheapest
            cheap = sig.iloc[-k_hi]
            t_i = _to_float(cheap["tenor_yrs"]); t_j = _to_float(rich["tenor_yrs"])
            if (t_i in used_tenors) or (t_j in used_tenors):
                continue
            if abs(t_i - t_j) < MIN_SEP_YEARS:
                continue

            # Fly gate
            if not fly_alignment_ok(t_i, +1, snap_last):  # cheap leg: expect z to fall
                continue
            if not fly_alignment_ok(t_j, -1, snap_last):  # rich leg: expect z to rise
                continue

            zdisp = _to_float(cheap["z_comb"]) - _to_float(rich["z_comb"])
            if not np.isfinite(zdisp) or (zdisp < (Z_ENTRY + extra_z_entry)):
                continue

            candidates.append((zdisp, cheap, rich))

    # pack under caps
    selected: list[tuple[pd.Series, pd.Series, float, float]] = []
    bucket_dv01 = {b: 0.0 for b in BUCKETS.keys()}
    total_dv = 0.0

    for zdisp, cheap, rich in sorted(candidates, key=lambda x: x[0], reverse=True):
        if len(selected) >= max_pairs:
            break

        t_i = _to_float(cheap["tenor_yrs"]); r_i = _to_float(cheap["rate"])
        t_j = _to_float(rich["tenor_yrs"]);  r_j = _to_float(rich["rate"])
        if (t_i in used_tenors) or (t_j in used_tenors):
            continue

        pv_i = pv01_proxy(t_i, r_i)
        pv_j = pv01_proxy(t_j, r_j)

        w_i = 1.0
        w_j = - w_i * pv_i / pv_j

        b_i = assign_bucket(t_i); b_j = assign_bucket(t_j)
        dv_i = abs(w_i) * pv_i; dv_j = abs(w_j) * pv_j

        # bucket caps
        if b_i in bucket_dv01 and (bucket_dv01[b_i] + dv_i > per_bucket_cap):
            continue
        if b_j in bucket_dv01 and (bucket_dv01[b_j] + dv_j > per_bucket_cap):
            continue

        # front-end aggregate cap
        short_add = (dv_i if b_i == "short" else 0.0) + (dv_j if b_j == "short" else 0.0)
        short_tot = bucket_dv01.get("short", 0.0)
        if short_tot + short_add > front_end_cap:
            continue

        # total cap
        if total_dv + dv_i + dv_j > total_cap:
            continue

        # accept
        used_tenors.add(t_i); used_tenors.add(t_j)
        bucket_dv01[b_i] = bucket_dv01.get(b_i, 0.0) + dv_i
        bucket_dv01[b_j] = bucket_dv01.get(b_j, 0.0) + dv_j
        total_dv += dv_i + dv_j
        selected.append((cheap, rich, w_i, w_j))

    return selected

# ------------------------
# Month runner
# ------------------------
def run_month(yymm: str, decision_freq: str = DECISION_FREQ):
    """
    Load enhanced features for a month and run the cheap-rich portfolio selection.
    """
    enh_path = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    if not enh_path.exists():
        raise FileNotFoundError(f"Missing enhanced file {enh_path}. Run feature_creation.py first.")

    df = pd.read_parquet(enh_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Required columns
    need = {"ts", "tenor_yrs", "rate", "z_spline", "z_pca", "z_comb"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{enh_path} missing columns: {missing}")

    # Time / decision bins
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    if decision_freq.upper() == "D":
        df["decision_ts"] = df["ts"].dt.floor("D")
        decisions_per_day = 1
    elif decision_freq.upper() == "H":
        df["decision_ts"] = df["ts"].dt.floor("H")
        decisions_per_day = 24  # used only for max-hold conversion logic
    else:
        raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

    # Outputs
    open_positions: list[PairPos] = []
    ledger_rows: list[dict] = []
    closed_rows: list[dict] = []

    # How many decisions before MAX_HOLD_DAYS is reached?
    max_hold_decisions = MAX_HOLD_DAYS * (1 if decision_freq.upper() == "D" else 24)

    # Iterate decisions in time order
    for dts, snap in df.groupby("decision_ts", sort=True):
        # latest tick per tenor within the decision bucket
        snap_last = (snap.sort_values("ts")
                          .groupby("tenor_yrs", as_index=False)
                          .tail(1)
                          .reset_index(drop=True))

        if snap_last.empty:
            continue

        # 1) Mark & exit evaluation
        still_open = []
        for pos in open_positions:
            zsp = pos.mark(snap_last)

            # exit rules
            if np.isfinite(zsp) and abs(zsp) <= Z_EXIT:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "reversion"
            elif np.isfinite(zsp) and np.isfinite(pos.entry_zspread) and abs(zsp - pos.entry_zspread) >= Z_STOP:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "stop"
            elif pos.age_decisions >= max_hold_decisions:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "max_hold"

            # ledger mark
            ledger_rows.append({
                "decision_ts": dts, "event": "mark",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "w_i": pos.w_i, "w_j": pos.w_j,
                "pnl": pos.pnl, "z_spread": zsp,
                "conv_proxy": pos.conv_pnl_proxy,
                "open_ts": pos.open_ts, "closed": pos.closed,
                "exit_reason": pos.exit_reason
            })

            if pos.closed:
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": pos.close_ts, "exit_reason": pos.exit_reason,
                    "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i, "w_j": pos.w_j, "entry_zspread": pos.entry_zspread,
                    "pnl": pos.pnl, "days_held_equiv": pos.age_decisions / decisions_per_day,
                    "conv_proxy": pos.conv_pnl_proxy
                })
            else:
                still_open.append(pos)

        open_positions = still_open

        # 2) Entries (under caps)
        # extra entry hurdle if any leg is in short bucket
        selected = choose_pairs_under_caps(
            snap_last=snap_last,
            max_pairs=max(0, MAX_CONCURRENT_PAIRS - len(open_positions)),
            per_bucket_cap=PER_BUCKET_DV01_CAP,
            total_cap=TOTAL_DV01_CAP,
            front_end_cap=FRONT_END_DV01_CAP,
            extra_z_entry=0.0  # we enforce SHORT_END_EXTRA_Z per-pair below
        )

        for (cheap, rich, w_i, w_j) in selected:
            t_i = _to_float(cheap["tenor_yrs"]); t_j = _to_float(rich["tenor_yrs"])
            if (assign_bucket(t_i) == "short") or (assign_bucket(t_j) == "short"):
                zdisp = _to_float(cheap["z_comb"]) - _to_float(rich["z_comb"])
                if not np.isfinite(zdisp) or (zdisp < (Z_ENTRY + SHORT_END_EXTRA_Z)):
                    continue

            pos = PairPos(open_ts=dts, cheap_row=cheap, rich_row=rich,
                          w_i=w_i, w_j=w_j, decisions_per_day=decisions_per_day)
            open_positions.append(pos)
            ledger_rows.append({
                "decision_ts": dts, "event": "open",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "w_i": pos.w_i, "w_j": pos.w_j,
                "entry_zspread": pos.entry_zspread
            })

    # Finalize outputs
    pos_df = pd.DataFrame(closed_rows)
    ledger = pd.DataFrame(ledger_rows)

    # PnL by day/hour from marks
    if not ledger.empty:
        marks = ledger[ledger["event"] == "mark"].copy()
        if DECISION_FREQ.upper() == "D":
            idx = marks["decision_ts"].dt.floor("D")
        else:
            idx = marks["decision_ts"].dt.floor("H")
        pnl_by = marks.groupby(idx)["pnl"].sum().rename("pnl").to_frame().reset_index()
        pnl_by = pnl_by.rename(columns={"decision_ts": "bucket"})
    else:
        pnl_by = pd.DataFrame(columns=["bucket", "pnl"])

    # Optional monthly PnL curve (daily aggregation for readability)
    if not pnl_by.empty:
        daily = pnl_by.copy()
        daily["date"] = pd.to_datetime(daily["bucket"]).dt.floor("D")
        daily = daily.groupby("date")["pnl"].sum().rename("daily_pnl").to_frame().reset_index()
        daily = daily.sort_values("date")
        daily["cum"] = daily["daily_pnl"].cumsum()
        plt.figure()
        plt.plot(daily["date"], daily["cum"])
        plt.title(f"Cumulative PnL proxy {yymm}")
        plt.xlabel("Date"); plt.ylabel("Cum PnL")
        out_png = Path(PATH_OUT) / f"pnl_curve_{yymm}.png"
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close()

    return pos_df, ledger, pnl_by

# ------------------------
# Multi-month runner
# ------------------------
def run_all(yymms: list[str]):
    all_pos, all_ledger, all_by = [], [], []
    print(f"[INFO] months: {len(yymms)} -> {yymms}")
    for yymm in yymms:
        print(f"[RUN] month {yymm}")
        p, l, b = run_month(yymm, decision_freq=DECISION_FREQ)
        if not p.empty: all_pos.append(p.assign(yymm=yymm))
        if not l.empty: all_ledger.append(l.assign(yymm=yymm))
        if not b.empty: all_by.append(b.assign(yymm=yymm))
        print(f"[DONE] {yymm} | pos={0 if p is None else len(p)} "
              f"ledger={0 if l is None else len(l)} by={0 if b is None else len(b)}")
    pos = pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame()
    led = pd.concat(all_ledger, ignore_index=True) if all_ledger else pd.DataFrame()
    by  = pd.concat(all_by, ignore_index=True) if all_by else pd.DataFrame()
    return pos, led, by

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python portfolio_test.py 2304 [2305 2306 ...]")
        sys.exit(1)
    months = sys.argv[1:]
    pos, led, by = run_all(months)
    if not pos.empty: pos.to_parquet(Path(PATH_OUT) / "positions_ledger.parquet")
    if not led.empty: led.to_parquet(Path(PATH_OUT) / "marks_ledger.parquet")
    if not by.empty:  by.to_parquet(Path(PATH_OUT) / "pnl_by_bucket.parquet")
    print("[DONE]")


