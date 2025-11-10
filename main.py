
# feature_creation.py
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

# ==== import your config (exact names as in cr_config.py) ====
from cr_config import (
    PATH_DATA, PATH_ENH,
    TENOR_YEARS,
    TRADING_HOURS,           # e.g. ("10:00","22:10")
    DECISION_FREQ,           # 'D' or 'H'
    PCA_COMPONENTS,          # int
    PCA_LOOKBACK_DAYS,       # int
)

# Optional knobs (provide sensible defaults if not present in cr_config)
try:
    PCA_ENABLE = PCA_ENABLE           # type: ignore[name-defined]
except NameError:
    PCA_ENABLE = True

try:
    N_JOBS = N_JOBS                    # type: ignore[name-defined]
except NameError:
    N_JOBS = 0                         # 0 = auto (conservative)

# For hourly mode, convert days→hours but cap to keep fits tractable
# (You can lift this cap if your box is beefy.)
PCA_LOOKBACK_CAP_HOURS = 24 * 14  # 2 weeks

def _effective_pca_lookback() -> int:
    if DECISION_FREQ == 'D':
        return max(1, int(PCA_LOOKBACK_DAYS))
    # hourly
    hours = int(PCA_LOOKBACK_DAYS) * 24
    return max(1, min(hours, PCA_LOOKBACK_CAP_HOURS))

# ===================== utils =====================

def _now() -> str:
    return time.strftime("%H:%M:%S")

def _decision_key(ts: pd.Series, freq: str) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    if freq == 'D':
        return ts.dt.floor("D")
    if freq == 'H':
        return ts.dt.floor("H")
    raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

def _to_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a 'ts' column (coerced to datetime)."""
    out = df.copy()
    if 'ts' in out.columns:
        out['ts'] = pd.to_datetime(out['ts'], errors='coerce')
        return out
    idxn = getattr(out.index, "name", None)
    if idxn in ('ts', 'sec'):
        out = out.reset_index().rename(columns={idxn: 'ts'})
        out['ts'] = pd.to_datetime(out['ts'], errors='coerce')
        return out
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={'index':'ts'})
        out['ts'] = pd.to_datetime(out['ts'], errors='coerce')
        return out
    # last resort
    out = out.reset_index().rename(columns={'index':'ts'})
    out['ts'] = pd.to_datetime(out['ts'], errors='coerce')
    return out

def _apply_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """If TRADING_HOURS is set, slice intraday window. Assumes naive UTC timestamps."""
    if not TRADING_HOURS or len(TRADING_HOURS) != 2:
        return df
    try:
        return (df.set_index('ts')
                  .between_time(TRADING_HOURS[0], TRADING_HOURS[1])
                  .reset_index())
    except Exception:
        # If index isn't time-like, just return df
        return df

def _zeros_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Convert placeholder zeros to NaN only when the column has non-zero values elsewhere."""
    out = df.copy()
    for c in out.columns:
        if c == 'ts':
            continue
        s = pd.to_numeric(out[c], errors='coerce')
        if (s == 0).any() and (s != 0).any():
            s = s.mask(s == 0, np.nan)
        out[c] = s
    return out

def _melt_long(df_wide: pd.DataFrame, tenormap: Dict[str, float]) -> pd.DataFrame:
    """Select columns present in TENOR_YEARS, melt to long: ['ts','tenor_yrs','rate']."""
    cols = [c for c in df_wide.columns if c in tenormap]
    if not cols:
        raise ValueError("No overlapping tickers between input file and TENOR_YEARS.")
    long = df_wide[['ts'] + cols].melt('ts', var_name='instrument', value_name='rate')
    long['tenor_yrs'] = long['instrument'].map(tenormap).astype(float)
    long = long.dropna(subset=['ts','tenor_yrs'])
    # Ensure numeric rates
    long['rate'] = pd.to_numeric(long['rate'], errors='coerce')
    long = long.dropna(subset=['rate'])
    return long[['ts','tenor_yrs','rate']]

# ---------------- curve spline (per snapshot) ----------------

def _fit_spline_guarded(x: np.ndarray, y: np.ndarray, k: int = 3):
    """Return callable f(t) producing fitted curve; robust fallbacks if spline fails."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        if x.size < 2:
            return lambda t: np.full_like(np.asarray(t, float), np.nan, dtype=float)
        a, b = np.polyfit(x, y, 1)
        return lambda t: a * np.asarray(t, float) + b
    try:
        from scipy.interpolate import UnivariateSpline
        s = max(1e-8, 0.5 * x.size * np.nanvar(y))
        spl = UnivariateSpline(x, y, k=min(k, max(1, x.size-1)), s=s, ext=3)
        return lambda t: spl(np.asarray(t, float))
    except Exception:
        deg = min(2, max(1, x.size - 1))
        coefs = np.polyfit(x, y, deg)
        poly = np.poly1d(coefs)
        return lambda t: poly(np.asarray(t, float))

def _zscore_xsec(vals: pd.Series) -> pd.Series:
    v = pd.to_numeric(vals, errors="coerce")
    mu = v.mean(skipna=True)
    sd = v.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=v.index)
    return (v - mu) / sd

# ---------------- PCA over lookback window ----------------

def _pca_z_for_bucket(bucket_df: pd.DataFrame,
                      last_snap: pd.DataFrame,
                      n_comps: int,
                      verbose_note: str = "") -> pd.Series:
    """
    Fit PCA on time×tenor matrix (lookback window).
    Return z-score of the standardized residual vector for the last row.
    """
    mat = (bucket_df
           .pivot_table(index='ts', columns='tenor_yrs', values='rate', aggfunc='last')
           .sort_index())
    # require some depth and breadth
    if mat.shape[0] < 12 or mat.shape[1] < 3:
        return pd.Series(np.nan, index=last_snap.index)

    # keep tenors with good coverage
    keep = mat.columns[mat.notna().mean() > 0.9]
    mat = mat[keep]
    if mat.shape[1] < 3:
        return pd.Series(np.nan, index=last_snap.index)

    mat = mat.fillna(method='ffill').dropna(axis=0, how='any')
    if mat.shape[0] < 8:
        return pd.Series(np.nan, index=last_snap.index)

    # standardize by tenor across time
    mat_std = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0)
    mat_std = mat_std.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    if mat_std.shape[0] < 8:
        return pd.Series(np.nan, index=last_snap.index)

    try:
        k = min(int(n_comps), mat_std.shape[1])
        model = PCA(n_components=k, svd_solver='auto', random_state=0)
        scores = model.fit_transform(mat_std.values)
        evr = getattr(model, "explained_variance_ratio_", None)
        if evr is not None:
            print(f"[{_now()}] [PCA] {verbose_note} rows={mat_std.shape[0]} cols={mat_std.shape[1]} "
                  f"EVR={np.round(evr[:k],3)}")

        last_row = mat_std.iloc[-1].values
        comps = model.components_
        proj = comps @ (comps.T @ last_row)
        resid_std = last_row - proj
        resid_std_s = pd.Series(resid_std, index=mat_std.columns)

        rs = resid_std_s.reindex(last_snap['tenor_yrs'].values)
        return _zscore_xsec(rs)
    except Exception as e:
        print(f"[{_now()}] [WARN] PCA failed ({verbose_note}): {e}")
        return pd.Series(np.nan, index=last_snap.index)

# ---------------- combiner ----------------

def _combine_z_safe(df_block: pd.DataFrame) -> pd.DataFrame:
    out = df_block.copy()
    if "z_pca" not in out.columns:    out["z_pca"] = np.nan
    if "z_spline" not in out.columns: out["z_spline"] = np.nan
    zp = pd.to_numeric(out["z_pca"], errors="coerce")
    zs = pd.to_numeric(out["z_spline"], errors="coerce")
    out["z_comb"] = np.where(np.isfinite(zp), zp, zs)
    return out

# ===================== per-bucket process =====================

def _process_bucket(dts: pd.Timestamp,
                    df_bucket: pd.DataFrame,
                    df_all: pd.DataFrame,
                    lookback_buckets: int,
                    pca_enable: bool,
                    pca_n_comps: int,
                    yymm: str) -> pd.DataFrame:
    # last snapshot per tenor
    snap_last = (df_bucket.sort_values('ts')
                 .groupby('tenor_yrs', as_index=False)
                 .tail(1)
                 .dropna(subset=['tenor_yrs','rate']))
    if snap_last.empty:
        return pd.DataFrame(columns=['ts','tenor_yrs','rate','z_spline','z_pca','z_comb'])

    # spline residual z
    f = _fit_spline_guarded(snap_last['tenor_yrs'].values, snap_last['rate'].values)
    fit_vals = f(snap_last['tenor_yrs'].values)
    spline_resid = pd.Series(snap_last['rate'].values - fit_vals, index=snap_last.index)
    z_spline = _zscore_xsec(spline_resid)

    # PCA residual z (time-series over lookback buckets)
    z_pca = pd.Series(np.nan, index=snap_last.index)
    if pca_enable:
        uniq = sorted(df_all['decision_ts'].dropna().unique())
        try:
            pos = uniq.index(dts)
        except ValueError:
            pos = None
        if pos is not None:
            lo = max(0, pos - (lookback_buckets - 1))
            win = uniq[lo:pos+1]
            mask = df_all['decision_ts'].isin(win)
            z_pca = _pca_z_for_bucket(
                bucket_df=df_all.loc[mask, ['ts','tenor_yrs','rate']],
                last_snap=snap_last[['tenor_yrs','rate']],
                n_comps=pca_n_comps,
                verbose_note=f"{yymm} {str(dts)}"
            )

    out = snap_last[['ts','tenor_yrs','rate']].copy()
    out['z_spline'] = z_spline.values
    out['z_pca'] = z_pca.values
    out = _combine_z_safe(out)
    out['ts'] = dts  # collapse to decision timestamp
    return out

# ===================== month runner =====================

def build_month(yymm: str) -> None:
    in_path  = Path(PATH_DATA) / f"{yymm}.parquet"
    out_path = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    Path(PATH_ENH).mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing raw month file: {in_path}")

    # Load wide
    df_wide = pd.read_parquet(in_path)
    df_wide = _to_ts_index(df_wide)
    df_wide = df_wide.sort_values('ts')
    # drop duplicated timestamps (keep last)
    df_wide = df_wide[~df_wide['ts'].duplicated(keep='last')]

    # Optional intraday slice
    df_wide = _apply_trading_hours(df_wide)

    # zeros->NaN & numeric coercion
    df_wide = _zeros_to_nan(df_wide)

    # Wide -> long with tenor mapping
    df_long = _melt_long(df_wide, TENOR_YEARS)

    # Decision buckets
    df_long['decision_ts'] = _decision_key(df_long['ts'], DECISION_FREQ)

    buckets = list(df_long['decision_ts'].dropna().unique())
    buckets.sort()

    # jobs
    if N_JOBS == 0:
        import multiprocessing as mp
        jobs = max(1, min((mp.cpu_count() // 2), 8))
    else:
        jobs = int(N_JOBS)

    lookback_buckets = _effective_pca_lookback()

    print(f"[{_now()}] [MONTH] {yymm} | buckets={len(buckets)} freq={DECISION_FREQ} "
          f"| jobs={jobs} | PCA={'on' if PCA_ENABLE else 'off'} "
          f"| lookback={lookback_buckets} | comps={PCA_COMPONENTS}")

    if len(buckets) == 0:
        print(f"[WARN] {yymm}: no buckets found.")
        pd.DataFrame(columns=['ts','tenor_yrs','rate','z_spline','z_pca','z_comb']).to_parquet(out_path, index=False)
        print(f"[SAVE] {out_path}")
        return

    def _one_bucket(dts):
        snap = df_long[df_long['decision_ts'] == dts]
        t0 = time.time()
        out = _process_bucket(
            dts=dts,
            df_bucket=snap,
            df_all=df_long[['ts','tenor_yrs','rate','decision_ts']],
            lookback_buckets=lookback_buckets,
            pca_enable=PCA_ENABLE,
            pca_n_comps=int(PCA_COMPONENTS),
            yymm=yymm
        )
        dt = time.time() - t0
        print(f"[BUCKET] {yymm} {dts} rows:{len(snap):,} tenors:{snap['tenor_yrs'].nunique()} "
              f"PCA:{'yes' if out['z_pca'].notna().any() else 'no '} t={dt:.2f}s")
        return out

    parts = Parallel(n_jobs=jobs, backend="loky")(delayed(_one_bucket)(dts) for dts in buckets)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['ts','tenor_yrs','rate','z_spline','z_pca','z_comb'])
    out = out.sort_values(['ts','tenor_yrs']).reset_index(drop=True)

    # quick month diagnostics
    zr = pd.to_numeric(out['z_comb'], errors='coerce')
    z_valid_pct = float(np.isfinite(zr).mean() * 100.0)
    med_ten = out.groupby('ts')['tenor_yrs'].nunique().median() if not out.empty else 0
    print(f"[DONE] {yymm} rows:{len(out):,} tenors_med:{med_ten:.1f} z_valid%:{z_valid_pct:.2f}")

    out.to_parquet(out_path, index=False)
    print(f"[SAVE] {out_path}")

# ===================== CLI =====================

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_creation.py 2304 [2305 2306 ...]")
        raise SystemExit(1)
    for yymm in sys.argv[1:]:
        build_month(yymm)

# cr_config.py
from zoneinfo import ZoneInfo
from pathlib import Path

Path(PATH_ENH).mkdir(parents=True, exist_ok=True)
Path(PATH_MODELS).mkdir(parents=True, exist_ok=True)
Path(PATH_OUT).mkdir(parents=True, exist_ok=True)

# ========= Session =========
TRADING_HOURS = ("10:00", "22:10")   # optional slice intraday when building features (UTC-naive index)
CHI = ZoneInfo("America/Chicago")

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

# ========= (Optional) Calendar filtering =========
# Not used by the current feature builder/backtest; keep for later.
USE_QL_CALENDAR = True
QL_US_MARKET    = "FederalReserve"
CAL_TZ          = "America/New_York"

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



# feature_creation.py
from __future__ import annotations
import os, time, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.decomposition import PCA

from cr_config import (
    PATH_RAW, PATH_ENH,
    TENOR_YEARS,
    DECISION_FREQ,    # 'D' or 'H'
    PCA_ENABLE,       # bool
    PCA_LOOKBACK,     # int (number of buckets to use for fit)
    PCA_N_COMPS,      # int
    N_JOBS            # 0 -> auto, or int
)

# ----------------------------
# Utilities & robust helpers
# ----------------------------

def _now() -> str:
    return time.strftime("%H:%M:%S")

def _decision_key(ts: pd.Series, freq: str) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce", utc=False)
    if freq == 'D':
        return ts.dt.floor("D")
    if freq == 'H':
        return ts.dt.floor("H")
    raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

def _to_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts index named 'sec' or 'ts' (or DatetimeIndex). Returns frame with a 'ts' column."""
    out = df.copy()
    if 'ts' in out.columns:
        out['ts'] = pd.to_datetime(out['ts'], errors='coerce')
        return out
    # maybe index
    idx_name = getattr(out.index, "name", None)
    if idx_name in ('sec', 'ts'):
        out = out.reset_index().rename(columns={idx_name: 'ts'})
        out['ts'] = pd.to_datetime(out['ts'], errors='coerce')
        return out
    # if DatetimeIndex without name
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={'index':'ts'})
        out['ts'] = pd.to_datetime(out['ts'], errors='coerce')
        return out
    # last resort: try to coerce index
    out = out.reset_index().rename(columns={'index':'ts'})
    out['ts'] = pd.to_datetime(out['ts'], errors='coerce')
    return out

def _zeros_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Convert obvious placeholder zeros to NaN (but only for float-like columns)."""
    out = df.copy()
    for c in out.columns:
        if c == 'ts':
            continue
        s = pd.to_numeric(out[c], errors='coerce')
        # treat exact 0 as missing if the column otherwise has non-zero values
        if (s == 0).any() and (s != 0).any():
            s = s.mask(s == 0, np.nan)
        out[c] = s
    return out

def _melt_long(df_wide: pd.DataFrame, tenormap: Dict[str, float]) -> pd.DataFrame:
    """Wide -> long with tenor mapping: columns=tickers; index time; value=rate (percent units)."""
    cols = [c for c in df_wide.columns if c in tenormap]
    if not cols:
        raise ValueError("No overlapping tickers between input file and TENOR_YEARS.")
    long = df_wide[['ts'] + cols].melt('ts', var_name='instrument', value_name='rate')
    long['tenor_yrs'] = long['instrument'].map(tenormap).astype(float)
    long = long.dropna(subset=['ts','tenor_yrs'])
    return long[['ts','tenor_yrs','rate']]

# -------- curve (tenor) spline residual at a single snapshot --------

def _fit_spline_guarded(x: np.ndarray, y: np.ndarray, k: int = 3):
    """Return callable f(t) producing fitted curve; robust fallbacks if spline fails."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        # linear fallback
        if x.size < 2:
            return lambda t: np.full_like(np.asarray(t, float), np.nan, dtype=float)
        a, b = np.polyfit(x, y, 1)
        return lambda t: a * np.asarray(t, float) + b
    try:
        from scipy.interpolate import UnivariateSpline
        s = max(1e-8, 0.5 * x.size * np.nanvar(y))
        spl = UnivariateSpline(x, y, k=min(k, max(1, x.size-1)), s=s, ext=3)
        return lambda t: spl(np.asarray(t, float))
    except Exception:
        deg = min(2, max(1, x.size - 1))
        coefs = np.polyfit(x, y, deg)
        poly = np.poly1d(coefs)
        return lambda t: poly(np.asarray(t, float))

def _zscore_xsec(vals: pd.Series) -> pd.Series:
    """Cross-sectional z-score (per snapshot across tenors)."""
    v = pd.to_numeric(vals, errors='coerce')
    mu = v.mean(skipna=True)
    sd = v.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=v.index)
    return (v - mu) / sd

# -------- PCA per bucket window (time-series variance) --------

def _pca_z_for_bucket(bucket_df: pd.DataFrame,
                      last_snap: pd.DataFrame,
                      n_comps: int,
                      verbose_note: str = "") -> pd.Series:
    """
    Fit PCA on the lookback window's time-series matrix (rows: time, cols: tenor).
    Return cross-sectional z-score of residual vector for the *last* snapshot in the bucket.
    """
    # build time x tenor matrix (align by timestamp; inner join for robustness)
    mat = (bucket_df
           .pivot_table(index='ts', columns='tenor_yrs', values='rate', aggfunc='last')
           .sort_index())
    # Require at least 12 rows and >= 3 tenors
    if mat.shape[0] < 12 or mat.shape[1] < 3:
        return pd.Series(np.nan, index=last_snap.index)

    # Drop tenors with too many NaNs; then forward-fill small gaps
    keep = mat.columns[mat.notna().mean() > 0.9]  # at least 90% present in window
    mat = mat[keep]
    if mat.shape[1] < 3:
        return pd.Series(np.nan, index=last_snap.index)
    mat = mat.fillna(method='ffill').dropna(axis=0, how='any')
    if mat.empty or mat.shape[0] < 8:
        return pd.Series(np.nan, index=last_snap.index)

    # Standardize by tenor across *time*
    mat_std = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0)
    mat_std = mat_std.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    if mat_std.shape[0] < 8:
        return pd.Series(np.nan, index=last_snap.index)

    # Fit PCA
    try:
        k = min(n_comps, mat_std.shape[1])
        model = PCA(n_components=k, svd_solver='auto', random_state=0)
        scores = model.fit_transform(mat_std.values)
        evr = getattr(model, "explained_variance_ratio_", None)
        if evr is not None:
            print(f"[{_now()}] [PCA] {verbose_note} rows={mat_std.shape[0]} cols={mat_std.shape[1]} "
                  f"EVR={np.round(evr[:k],3)}")
        # reconstruct the *last* row and take residual vector
        last_row = mat_std.iloc[-1].values
        comps = model.components_               # shape (k, n_cols)
        # projection -> reconstruction in standardized space
        proj = comps @ (comps.T @ last_row)
        resid_std = last_row - proj             # standardized residual vector
        # map residuals back to tenor index order used in last_snap
        # Build a Series on mat_std columns first:
        resid_std_s = pd.Series(resid_std, index=mat_std.columns)
        # now reindex to last_snap tenors (may drop some if absent)
        rs = resid_std_s.reindex(last_snap['tenor_yrs'].values)
        # cross-sectional z-score of residuals (already standardized, but normalize again for stability)
        return _zscore_xsec(rs)
    except Exception as e:
        print(f"[{_now()}] [WARN] PCA failed ({verbose_note}): {e}")
        return pd.Series(np.nan, index=last_snap.index)

# -------- combiner --------

def _combine_z_safe(df_block: pd.DataFrame) -> pd.DataFrame:
    out = df_block.copy()
    if "z_pca" not in out.columns:    out["z_pca"] = np.nan
    if "z_spline" not in out.columns: out["z_spline"] = np.nan
    zp = pd.to_numeric(out["z_pca"], errors="coerce")
    zs = pd.to_numeric(out["z_spline"], errors="coerce")
    out["z_comb"] = np.where(np.isfinite(zp), zp, zs)
    return out

# ----------------------------
# Core per-bucket processing
# ----------------------------

def _process_bucket(dts: pd.Timestamp,
                    df_bucket: pd.DataFrame,
                    df_all: pd.DataFrame,
                    lookback: int,
                    pca_enable: bool,
                    pca_n_comps: int,
                    freq: str,
                    yymm: str) -> pd.DataFrame:
    """
    Returns frame with one row per tenor at decision timestamp:
      ['ts','tenor_yrs','rate','z_spline','z_pca','z_comb']
    """
    # last snapshot within bucket
    snap_last = (df_bucket.sort_values('ts')
                 .groupby('tenor_yrs', as_index=False)
                 .tail(1)
                 .dropna(subset=['tenor_yrs','rate']))
    if snap_last.empty:
        return pd.DataFrame(columns=['ts','tenor_yrs','rate','z_spline','z_pca','z_comb'])

    # ---- spline z (cross-sectional) at the snapshot
    f = _fit_spline_guarded(snap_last['tenor_yrs'].values, snap_last['rate'].values)
    fit_vals = f(snap_last['tenor_yrs'].values)
    spline_resid = pd.Series(snap_last['rate'].values - fit_vals, index=snap_last.index)
    z_spline = _zscore_xsec(spline_resid)

    # ---- PCA z (time-series over lookback buckets)
    z_pca = pd.Series(np.nan, index=snap_last.index)
    if pca_enable:
        # collect lookback window buckets
        # we rely on df_all having 'decision_ts' column
        uniq = sorted(df_all['decision_ts'].dropna().unique())
        # find index of current dts
        try:
            pos = uniq.index(dts)
        except ValueError:
            pos = None
        if pos is not None:
            # window includes current bucket and previous (lookback-1)
            lo = max(0, pos - (lookback - 1))
            win = uniq[lo:pos+1]
            mask = df_all['decision_ts'].isin(win)
            z_pca = _pca_z_for_bucket(
                bucket_df=df_all.loc[mask, ['ts','tenor_yrs','rate']],
                last_snap=snap_last[['tenor_yrs','rate']],
                n_comps=pca_n_comps,
                verbose_note=f"{yymm} {str(dts)}"
            )

    out = snap_last[['ts','tenor_yrs','rate']].copy()
    out['z_spline'] = z_spline.values
    out['z_pca'] = z_pca.values
    out = _combine_z_safe(out)
    # tag bucket ts as the decision time (collapse to D/H key for clarity)
    out['ts'] = dts
    return out

# ----------------------------
# Month runner
# ----------------------------

def build_month(yymm: str) -> None:
    in_path  = Path(PATH_RAW) / f"{yymm}.parquet"
    out_path = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    Path(PATH_ENH).mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing raw month file: {in_path}")

    # ---------- load & clean wide
    df_wide = pd.read_parquet(in_path)
    df_wide = _to_ts_index(df_wide)
    df_wide = df_wide.sort_values('ts')
    # de-duplicate timestamps (keep last)
    df_wide = df_wide[~df_wide['ts'].duplicated(keep='last')]
    # zeros->NaN and numeric coercion
    df_wide = _zeros_to_nan(df_wide)

    # ---------- long format with tenor mapping
    df_long = _melt_long(df_wide, TENOR_YEARS)
    # sanity numeric
    df_long['rate'] = pd.to_numeric(df_long['rate'], errors='coerce')
    # drop rows with no rate
    df_long = df_long.dropna(subset=['rate'])

    # ---------- assign decision buckets
    df_long['decision_ts'] = _decision_key(df_long['ts'], DECISION_FREQ)

    # ---------- parallel over buckets
    buckets = list(df_long['decision_ts'].dropna().unique())
    buckets.sort()
    if N_JOBS == 0:
        # "auto": use sensible default without oversubscribing
        import multiprocessing as mp
        jobs = max(1, min( (mp.cpu_count()//2), 8 ))
    else:
        jobs = int(N_JOBS)

    print(f"[{_now()}] [MONTH] {yymm} | buckets={len(buckets)} freq={DECISION_FREQ} "
          f"| jobs={jobs} | PCA={'on' if PCA_ENABLE else 'off'} "
          f"| lookback={PCA_LOOKBACK} | comps={PCA_N_COMPS}")

    def _one_bucket(dts):
        snap = df_long[df_long['decision_ts'] == dts]
        t0 = time.time()
        out = _process_bucket(
            dts=dts,
            df_bucket=snap,
            df_all=df_long[['ts','tenor_yrs','rate','decision_ts']],
            lookback=PCA_LOOKBACK,
            pca_enable=PCA_ENABLE,
            pca_n_comps=PCA_N_COMPS,
            freq=DECISION_FREQ,
            yymm=yymm
        )
        dt = time.time() - t0
        print(f"[DAY ] {yymm} {dts} rows:{len(snap):,} tenors:{snap['tenor_yrs'].nunique()} "
              f"PCA:{'yes' if out['z_pca'].notna().any() else 'no '} t={dt:.2f}s")
        return out

    if len(buckets) == 0:
        print(f"[WARN] {yymm}: no buckets found.")
        pd.DataFrame(columns=['ts','tenor_yrs','rate','z_spline','z_pca','z_comb']).to_parquet(out_path, index=False)
        print(f"[SAVE] {out_path}")
        return

    parts = Parallel(n_jobs=jobs, backend="loky")(
        delayed(_one_bucket)(dts) for dts in buckets
    )
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['ts','tenor_yrs','rate','z_spline','z_pca','z_comb'])

    # final hygiene
    out = out.sort_values(['ts','tenor_yrs']).reset_index(drop=True)

    # quick month diagnostics
    zr = pd.to_numeric(out['z_comb'], errors='coerce')
    z_valid_pct = float(np.isfinite(zr).mean() * 100.0)
    print(f"[DONE] {yymm} rows:{len(out):,} tenors_med:{out.groupby('ts')['tenor_yrs'].nunique().median():.1f} "
          f"z_valid%:{z_valid_pct:.2f}")

    out.to_parquet(out_path, index=False)
    print(f"[SAVE] {out_path}")

# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_creation.py 2304 [2305 2306 ...]")
        raise SystemExit(1)
    months = sys.argv[1:]
    for yymm in months:
        build_month(yymm)

# check_features.py
import sys, pandas as pd
from pathlib import Path
from cr_config import PATH_ENH

def check_month(yymm):
    p = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    if not p.exists():
        print(f"[MISS] {p}")
        return
    df = pd.read_parquet(p)
    print(f"[{yymm}] rows={len(df):,}")
    cols = [c for c in df.columns if "z" in c.lower() or "pca" in c.lower()]
    print("Columns with z/PCA:", cols)
    print(df[cols].describe(include='all'))
    # quick validity ratio
    if "z_comb" in df.columns:
        z = pd.to_numeric(df["z_comb"], errors="coerce")
        print(f"Valid z_comb % = {(z.notna() & (z!=0)).mean()*100:.2f}%")
        print("z_comb unique nonzero values:", z[z!=0].unique()[:5])
    else:
        print("No z_comb found.")
    print("-"*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_features.py 2304 [2305 ...]")
        sys.exit(1)
    for yymm in sys.argv[1:]:
        check_month(yymm)

# diag_backtest.py
import sys, os, math, traceback
from pathlib import Path
import numpy as np
import pandas as pd

# Use your existing config (paths, thresholds)
from cr_config import (
    PATH_ENH, PATH_OUT,
    DECISION_FREQ, Z_ENTRY, MIN_SEP_YEARS
)

# ---------- helpers ----------

def _ok_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    return len(missing) == 0, missing

def _is_monotonic(dt):
    return (dt.values[1:] >= dt.values[:-1]).all()

def _tenor_bucket(tenor):
    # purely for reporting; doesn’t need exact buckets
    if tenor < 2: return "short"
    if tenor < 7: return "belly"
    return "long"

def _pairable_any(snap_last, z_entry, min_sep_yrs):
    """
    Fast dry-run: ignore caps/flies; just check if any pair satisfies
    z_diff >= z_entry and tenor separation >= min_sep_yrs.
    Returns (ok:boolean, max_dispersion, (t_lo, z_lo, t_hi, z_hi))
    """
    sl = snap_last[["tenor_yrs","z_comb"]].dropna()
    if sl.empty or len(sl) < 2:
        return False, np.nan, None
    sl = sl.sort_values("z_comb")
    # best candidates are extremes; try K extremes to be robust
    K = min(10, len(sl))
    best = (-1.0, None)
    lo_side = sl.iloc[:K].copy()
    hi_side = sl.iloc[-K:].copy()
    for _, lo in lo_side.iterrows():
        for __, hi in hi_side.iterrows():
            if abs(hi["tenor_yrs"] - lo["tenor_yrs"]) < min_sep_yrs:
                continue
            disp = float(hi["z_comb"] - lo["z_comb"])
            if disp > best[0]:
                best = (disp, (float(lo["tenor_yrs"]), float(lo["z_comb"]),
                               float(hi["tenor_yrs"]), float(hi["z_comb"])))
    ok = best[0] >= z_entry if best[0] is not None else False
    return ok, best[0], best[1]

def _day_summary(df, decision_freq):
    # Assign decision buckets (D/H) like the backtest
    ts = pd.to_datetime(df["ts"], errors="coerce")
    if decision_freq == 'D':
        key = ts.dt.floor("D")
    elif decision_freq == 'H':
        key = ts.dt.floor("H")
    else:
        raise ValueError("DECISION_FREQ must be 'D' or 'H'")
    # Per bucket, take last tick per tenor (no look-ahead)
    out = []
    for dts, snap in df.groupby(key, sort=True):
        snap_last = snap.sort_values("ts").groupby("tenor_yrs", as_index=False).tail(1)
        if snap_last.empty:
            out.append((dts, 0, np.nan, np.nan, np.nan))
        else:
            z = pd.to_numeric(snap_last["z_comb"], errors="coerce")
            out.append((
                dts,
                snap_last["tenor_yrs"].nunique(),
                float(np.nanmin(z)),
                float(np.nanmax(z)),
                float(np.nanmax(z) - np.nanmin(z))
            ))
    return pd.DataFrame(out, columns=["decision_ts","n_tenors","z_min","z_max","z_range"])

def _nice_num(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.4f}"

# ---------- main per-month diag ----------

def diagnose_month(yymm: str, decision_freq: str = DECISION_FREQ) -> dict:
    p = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    res = {"yymm": yymm, "path": str(p), "exists": p.exists(), "errors": []}
    if not p.exists():
        res["errors"].append("missing_file")
        print(f"[MISS] {yymm}: {p}")
        return res

    try:
        df = pd.read_parquet(p)
    except Exception as e:
        res["errors"].append(f"read_error:{e}")
        print(f"[ERR ] {yymm}: read parquet failed: {e}")
        return res

    # ---- schema checks
    needed = ["ts","tenor_yrs","rate","z_comb"]
    ok, missing = _ok_cols(df, needed)
    if not ok:
        res["errors"].append(f"missing_cols:{missing}")
        print(f"[ERR ] {yymm}: missing columns {missing}")
        return res

    # types and NaNs/zeros
    n_rows = len(df)
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    # convert to numeric but keep NaN (don’t crash)
    df["tenor_yrs"] = pd.to_numeric(df["tenor_yrs"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df["z_comb"] = pd.to_numeric(df["z_comb"], errors="coerce")

    n_ts_nan = df["ts"].isna().sum()
    n_ten_nan = df["tenor_yrs"].isna().sum()
    n_rate_nan = df["rate"].isna().sum()
    n_z_nan = df["z_comb"].isna().sum()
    n_zero_rate = (df["rate"] == 0).sum()
    n_zero_z = (df["z_comb"] == 0).sum()

    # monotonic time within month
    # allow per-day gaps, but global monotonic should generally hold for tick feeds
    ts_monotonic = _is_monotonic(df["ts"].sort_values())

    # decision bucket stats
    stats = _day_summary(df, decision_freq)
    if stats.empty:
        res["errors"].append("empty_stats")
        print(f"[ERR ] {yymm}: no decision buckets.")
        return res

    # z-range percentiles (across decision buckets)
    zrng = stats["z_range"].dropna()
    z_min = float(zrng.min()) if not zrng.empty else np.nan
    z_p50 = float(zrng.quantile(0.50)) if not zrng.empty else np.nan
    z_p90 = float(zrng.quantile(0.90)) if not zrng.empty else np.nan
    z_max = float(zrng.max()) if not zrng.empty else np.nan

    # feasibility sampling: try 12 buckets spread through the month
    idxs = np.linspace(0, max(0, len(stats)-1), num=min(12, len(stats)), dtype=int)
    sample_times = stats.iloc[idxs]["decision_ts"].tolist()

    reasons = {"no_data":0, "too_few_tenors":0, "no_pair_meets_threshold":0, "sep_blocker":0, "ok":0}
    examples = []

    # Map times back to a quick snap_last per bucket and dry-run
    df["decision_ts"] = (df["ts"].dt.floor("D") if decision_freq=='D' else df["ts"].dt.floor("H"))
    for dts in sample_times:
        snap = df[df["decision_ts"]==dts]
        if snap.empty:
            reasons["no_data"] += 1
            continue
        snap_last = snap.sort_values("ts").groupby("tenor_yrs", as_index=False).tail(1)
        if snap_last.empty or snap_last["tenor_yrs"].nunique() < 2:
            reasons["too_few_tenors"] += 1
            continue

        ok, best_disp, best_tuple = _pairable_any(snap_last, Z_ENTRY, MIN_SEP_YEARS)
        if ok:
            reasons["ok"] += 1
            examples.append((dts, best_disp, best_tuple))
        else:
            # Was it dispersion or separation?
            # Check dispersion without separation
            sl = snap_last[["tenor_yrs","z_comb"]].dropna().sort_values("z_comb")
            if sl.empty or len(sl) < 2:
                reasons["too_few_tenors"] += 1
                continue
            zdisp_any = float(sl["z_comb"].iloc[-1] - sl["z_comb"].iloc[0])
            if zdisp_any < Z_ENTRY:
                reasons["no_pair_meets_threshold"] += 1
            else:
                # dispersion was there but all candidate pairs violated separation
                reasons["sep_blocker"] += 1

    # Assemble report dictionary
    res.update({
        "n_rows": int(n_rows),
        "n_ts_nan": int(n_ts_nan),
        "n_tenor_nan": int(n_ten_nan),
        "n_rate_nan": int(n_rate_nan),
        "n_z_nan": int(n_z_nan),
        "n_zero_rate": int(n_zero_rate),
        "n_zero_z": int(n_zero_z),
        "ts_monotonic": bool(ts_monotonic),
        "n_decisions": int(len(stats)),
        "median_n_tenors": float(stats["n_tenors"].median()),
        "z_range_min": z_min, "z_range_p50": z_p50, "z_range_p90": z_p90, "z_range_max": z_max,
        "reasons": reasons,
        "examples": examples[:3],  # at most 3 illustrative successes
    })

    # Optional: write a small txt report
    Path(PATH_OUT).mkdir(parents=True, exist_ok=True)
    rpt = Path(PATH_OUT)/f"diag_{yymm}.txt"
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"[DIAG] {yymm}\n")
        f.write(f"path: {p}\n")
        f.write(f"rows: {n_rows}\n")
        f.write(f"NaNs ts/tenor/rate/z: {n_ts_nan}/{n_ten_nan}/{n_rate_nan}/{n_z_nan}\n")
        f.write(f"Zeros rate/z: {n_zero_rate}/{n_zero_z}\n")
        f.write(f"ts_monotonic: {ts_monotonic}\n")
        f.write(f"decisions: {len(stats)} | median tenors: {stats['n_tenors'].median():.1f}\n")
        f.write(f"z_range min/p50/p90/max: {_nice_num(z_min)}/{_nice_num(z_p50)}/{_nice_num(z_p90)}/{_nice_num(z_max)}\n")
        f.write(f"reasons: {reasons}\n")
        if examples:
            f.write("examples_ok (dts, best_disp, (t_lo,z_lo,t_hi,z_hi)):\n")
            for e in examples:
                f.write(f"  {e}\n")
    print(f"[REPORT] {rpt}")
    return res

# ---------- CLI ----------

def main(argv):
    if len(argv) < 2:
        print("Usage: python diag_backtest.py 2304 [2305 2306 ...]")
        sys.exit(1)

    yymms = argv[1:]
    print(f"[INFO] months: {len(yymms)} -> {yymms}")
    all_res = []
    for yymm in yymms:
        try:
            r = diagnose_month(yymm, DECISION_FREQ)
            all_res.append(r)
            # quick console line:
            print(f"[DONE] {yymm} | rows={r.get('n_rows','?')} "
                  f"dec={r.get('n_decisions','?')} medTenors={_nice_num(r.get('median_n_tenors'))} "
                  f"zP50={_nice_num(r.get('z_range_p50'))} zP90={_nice_num(r.get('z_range_p90'))} "
                  f"ok_samples={r.get('reasons',{}).get('ok',0)}")
        except Exception as e:
            print(f"[FATAL] {yymm}: {e}")
            traceback.print_exc()

    # Summarize across months
    df = pd.DataFrame(all_res)
    if not df.empty:
        out = Path(PATH_OUT)/"diag_summary.parquet"
        df.to_parquet(out)
        print(f"[SAVE ] {out}")
        # Quick top-line reason counts
        if "reasons" in df.columns:
            agg = {"ok":0,"no_data":0,"too_few_tenors":0,"no_pair_meets_threshold":0,"sep_blocker":0}
            for d in df["reasons"]:
                for k in agg:
                    agg[k] += int(d.get(k,0))
            print(f"[REASONS] aggregate samples: {agg}")

if __name__ == "__main__":
    main(sys.argv)


def run_month(
    yymm: str,
    decision_freq: str = DECISION_FREQ,   # 'D' or 'H'
    *,
    plot: bool = False,                   # disable heavy plotting by default
    heatmap: bool = False,                # disable big heatmap by default
    verbose: bool = True
):
    from pathlib import Path
    import pandas as pd
    import numpy as np

    p = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run feature creation first.")

    df = pd.read_parquet(p)
    if df.empty:
        if verbose: print(f"[WARN] {yymm}: enhanced parquet is empty.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # time keys
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    if decision_freq == 'D':
        df["decision_ts"] = df["ts"].dt.floor("D")
    elif decision_freq == 'H':
        df["decision_ts"] = df["ts"].dt.floor("H")
    else:
        raise ValueError("decision_freq must be 'D' or 'H'.")

    # optional (big) heatmap — gated
    if heatmap:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _heat = (df.pivot_table(index="ts", columns="tenor_yrs",
                                    values="z_comb", aggfunc="last")
                       .sort_index())
            if not _heat.empty:
                plt.figure()
                plt.imshow(_heat.T, aspect="auto", origin="lower")
                plt.title(f"z_comb heatmap {yymm}")
                plt.xlabel("time idx"); plt.ylabel("tenor (yrs)")
                plt.colorbar()
                plt.savefig(Path(PATH_OUT) / f"z_heatmap_{yymm}.png",
                            dpi=120, bbox_inches="tight")
                plt.close()
        except Exception as e:
            if verbose: print(f"[WARN] heatmap skipped for {yymm}: {e}")

    open_positions: list[PairPos] = []
    ledger_rows: list[dict] = []
    closed_rows: list[dict] = []

    # main loop
    for dts, snap in df.groupby("decision_ts", sort=True):
        # last observed tick per tenor up to this decision timestamp (no look-ahead)
        snap_last = snap.sort_values("ts").groupby("tenor_yrs").tail(1)
        if snap_last.empty:
            continue

        # ---- mark & exit existing positions ----
        still_open = []
        for pos in open_positions:
            zsp = pos.mark(snap_last)
            if abs(zsp) <= Z_EXIT:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "reversion"
            elif abs(zsp - pos.entry_zspread) >= Z_STOP:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "stop"
            elif pos.days_held >= MAX_HOLD_DAYS:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "max_hold"
            else:
                # light aging by decision frequency
                if decision_freq == 'D':
                    pos.days_held += 1

            # always append a 'mark' to the ledger
            ledger_rows.append({
                "decision_ts": dts,
                "event": "mark",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "w_i": pos.w_i, "w_j": pos.w_j,
                "pnl": pos.pnl, "z_spread": zsp,
                "conv_proxy": pos.conv_pnl_proxy,
                "open_ts": pos.open_ts, "closed": pos.closed,
                "exit_reason": pos.exit_reason,
                "entry_zspread": pos.entry_zspread
            })

            if pos.closed:
                closed_rows.append({
                    "open_ts": pos.open_ts,
                    "close_ts": pos.close_ts,
                    "exit_reason": pos.exit_reason,
                    "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i, "w_j": pos.w_j,
                    "entry_zspread": pos.entry_zspread,
                    "pnl": pos.pnl, "days_held": pos.days_held,
                    "conv_proxy": pos.conv_pnl_proxy
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # ---- entries (greedy under caps) ----
        selected = choose_pairs_under_caps(
            snap_last=snap_last,
            max_pairs=max(0, MAX_CONCURRENT_PAIRS - len(open_positions)),
            per_bucket_cap=PER_BUCKET_DV01_CAP,
            total_cap=TOTAL_DV01_CAP,
            front_end_cap=FRONT_END_DV01_CAP,
            extra_z_entry=0.0
        )

        for (cheap, rich, w_i, w_j) in selected:
            # short-end requires extra dispersion if configured
            if assign_bucket(cheap["tenor_yrs"]) == "short" or assign_bucket(rich["tenor_yrs"]) == "short":
                if float(cheap["z_comb"] - rich["z_comb"]) < (Z_ENTRY + SHORT_END_EXTRA_Z):
                    continue

            pos = PairPos(open_ts=dts, cheap=cheap, rich=rich, w_i=w_i, w_j=w_j)
            open_positions.append(pos)

            ledger_rows.append({
                "decision_ts": dts,
                "event": "open",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "w_i": pos.w_i, "w_j": pos.w_j,
                "entry_zspread": pos.entry_zspread
            })

    # -------- outputs (schema-safe) --------
    pos_df = pd.DataFrame(closed_rows)

    LEDGER_COLS = [
        "decision_ts","event","tenor_i","tenor_j","w_i","w_j",
        "pnl","z_spread","conv_proxy","open_ts","closed","exit_reason","entry_zspread"
    ]
    ledger = pd.DataFrame(ledger_rows)
    # ensure all expected columns exist and order them
    for c in LEDGER_COLS:
        if c not in ledger.columns:
            ledger[c] = pd.NA
    ledger = ledger[LEDGER_COLS]

    # daily PnL (guarded)
    if not ledger.empty and "pnl" in ledger.columns:
        marks = ledger.loc[ledger["event"] == "mark"].copy()
        if not marks.empty:
            gkey = pd.to_datetime(marks["decision_ts"], errors="coerce").dt.floor("D")
            pnl_by_day = (marks.groupby(gkey, dropna=True)["pnl"]
                          .sum()
                          .rename("daily_pnl").to_frame().reset_index())
            pnl_by_day = pnl_by_day.rename(columns={"decision_ts": "date", "index": "date"})
            pnl_by_day["date"] = pd.to_datetime(pnl_by_day["date"])
        else:
            pnl_by_day = pd.DataFrame(columns=["date","daily_pnl"])
    else:
        pnl_by_day = pd.DataFrame(columns=["date","daily_pnl"])

    # optional monthly PnL plot
    if plot and not pnl_by_day.empty:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            pnl_by_day = pnl_by_day.sort_values("date")
            pnl_by_day["cum"] = pnl_by_day["daily_pnl"].cumsum()
            plt.figure()
            plt.plot(pnl_by_day["date"], pnl_by_day["cum"])
            plt.title(f"Cumulative PnL proxy {yymm}")
            plt.xlabel("Date"); plt.ylabel("Cum PnL")
            out_png = Path(PATH_OUT) / f"pnl_curve_{yymm}.png"
            plt.savefig(out_png, dpi=120, bbox_inches="tight")
            plt.close()
        except Exception as e:
            if verbose: print(f"[WARN] plot skipped for {yymm}: {e}")

    return pos_df, ledger, pnl_by_day
    
def run_all(yymms, *, decision_freq: str = DECISION_FREQ, plot=False, heatmap=False, verbose=True):
    import pandas as pd

    all_pos, all_ledger, all_byday = [], [], []
    if verbose: print(f"[INFO] months: {len(yymms)} -> {yymms}", flush=True)

    for yymm in yymms:
        if verbose: print(f"[RUN] month {yymm}", flush=True)
        p, l, b = run_month(yymm, decision_freq=decision_freq, plot=plot, heatmap=heatmap, verbose=verbose)
        if p is not None and not p.empty: all_pos.append(p)
        if l is not None and not l.empty: all_ledger.append(l)
        if b is not None and not b.empty: all_byday.append(b.assign(yymm=yymm))
        if verbose:
            print(f"[DONE] {yymm} | pos={0 if p is None else len(p)} "
                  f"ledger={0 if l is None else len(l)} "
                  f"byday={0 if b is None else len(b)}", flush=True)

    pos   = pd.concat(all_pos,   ignore_index=True) if all_pos   else pd.DataFrame()
    led   = pd.concat(all_ledger,ignore_index=True) if all_ledger else pd.DataFrame()
    byday = pd.concat(all_byday, ignore_index=True) if all_byday else pd.DataFrame()
    return pos, led, byday
    
            # optional integrity check: did we produce one or more rows for each day?
    ts_ser = pd.to_datetime(enh["ts"], errors="coerce")
    got_days = ts_ser.dt.normalize().nunique()

    if got_days != len(dates):
        print(f"[WARN] {yymm}: output has {got_days}/{len(dates)} distinct days.")
        out_days_set = set(ts_ser.dt.normalize().dt.date.unique())
        dates_set = set(pd.to_datetime(dates).date)  # DatetimeIndex -> ndarray of dates
        missing = sorted(dates_set - out_days_set)
        if missing:
            print(f"[WARN] missing days: {missing}")

def build_month_guarded(
    yymm: str,
    lam=EWMA_LAMBDA,
    pca_presence_frac: float = 0.6,
    n_jobs: int | None = None,
) -> None:
    """
    Writes PATH_ENH/<yymm>_enh.parquet with guarded spline & PCA z's per tick.
    - Trains/loads a PCA model per decision day (serial, once per day)
    - Processes each day in parallel with joblib (safe queue + retry on failure)
    - Prints progress & timing
    """
    import time
    t0 = time.perf_counter()

    # ---- load & reshape month ----
    df_wide = _load_month(yymm)
    if df_wide.empty:
        print(f"[{yymm}] empty month after trading hours slice.")
        return

    df_long = _wide_to_long(df_wide)
    if df_long.empty:
        print(f"[{yymm}] no recognizable *_mid columns / tenor map.")
        return

    # ---- daily close panel for PCA training ----
    df_long["date"] = pd.to_datetime(df_long["ts"]).dt.floor("D")
    daily_close = (
        df_long.sort_values("ts")
               .groupby(["date", "tenor_yrs"])
               .tail(1)
               .pivot(index="date", columns="tenor_yrs", values="rate")
               .sort_index()
    )
    daily_close.columns = [f"{c:.3f}" for c in daily_close.columns]

    panel = _load_daily_panel()
    panel = pd.concat([panel, daily_close], axis=0) if not panel.empty else daily_close.copy()
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    _save_daily_panel(panel)

    dates = sorted(daily_close.index.unique())
    if n_jobs is None:
        n_jobs = max(1, min(os.cpu_count() - 1, len(dates)))
    print(f"[{yymm}] days: {len(dates)} | jobs: {n_jobs} | rows(long): {df_long.shape[0]:,}")

    # ---- ensure PCA model exists for each day (serial) ----
    for d in dates:
        date_key = pd.Timestamp(d).strftime("%Y-%m-%d")
        if _pca_read(date_key) is None:
            model = _train_pca_daily(
                panel,
                asof_date=pd.Timestamp(d),
                lookback_days=PCA_LOOKBACK_DAYS,
                K=PCA_COMPONENTS,
            )
            if model is not None:
                _pca_write(date_key, model)

    # ---- prepare day slices ----
    day_slices = [(d, df_long[df_long["date"] == d].copy()) for d in dates if not df_long[df_long["date"] == d].empty]

    # ---- parallel process days (safe queue + retry) ----
    from joblib import Parallel, delayed

    def _work(item):
        d, day_df = item
        date_key = pd.Timestamp(d).strftime("%Y-%m-%d")
        model = _pca_read(date_key)  # may be None if insufficient history
        return _process_one_day(day_df, model, lam, pca_presence_frac)

    def _work_safe(item):
        d, day_df = item
        try:
            res = _work(item)
            return (d, res, None)
        except Exception:
            import traceback
            return (d, None, traceback.format_exc())

    triples = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        prefer="processes",
        pre_dispatch=n_jobs,  # keep queue small to reduce memory pressure on Windows
        batch_size=1,         # one day per batch
        verbose=0,
    )(delayed(_work_safe)(item) for item in day_slices)

    ok = [(d, df) for (d, df, err) in triples if err is None]
    bad = [(d, err) for (d, df, err) in triples if err is not None]

    if bad:
        failed_days = [pd.Timestamp(d).date().isoformat() for d, _ in bad]
        print(f"[WARN] {len(bad)} day(s) failed in parallel; retrying serially: {failed_days}")
        for d, err in bad:
            print(f"\n[TRACEBACK] day={pd.Timestamp(d).date()}\n{err}")
        # retry failed days serially
        for d, _ in bad:
            slice_df = df_long[df_long["date"] == d].copy()
            try:
                res = _work((d, slice_df))
                ok.append((d, res))
            except Exception:
                import traceback
                print(f"[ERROR] day {pd.Timestamp(d).date()} failed again:\n{traceback.format_exc()}")

    # collect results in chronological order
    results = [res for d, res in sorted(ok, key=lambda t: t[0])]
    enh = pd.concat(results, axis=0, ignore_index=True) if results else pd.DataFrame()
    if enh.empty:
        print(f"[{yymm}] produced no enhanced rows (insufficient PCA history or empty).")
        return

    # ---- finalize & write ----
    enh = enh.sort_values(["ts", "tenor_yrs"])[[
        "ts", "tenor_yrs", "rate",
        "model_spline", "eps_spline", "eps_spline_ewstd", "z_spline",
        "model_pca", "eps_pca", "eps_pca_ewstd", "z_pca",
        "z_comb",
    ]]
    out_path = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    enh.to_parquet(out_path, engine="pyarrow")

    # optional integrity check: did we produce one or more rows for each day?
    got_days = pd.to_datetime(enh["ts"]).normalize().nunique()
    if got_days != len(dates):
        print(f"[WARN] {yymm}: output has {got_days}/{len(dates)} distinct days.")
        missing = sorted(set(pd.to_datetime(dates).date()) -
                         set(pd.to_datetime(enh["ts"]).dt.normalize().dt.date.unique()))
        if missing:
            print(f"[WARN] missing days: {missing}")

    dt = time.perf_counter() - t0
    print(f"[{yymm}] wrote enhanced -> {out_path} | rows: {enh.shape[0]:,} | {dt:0.2f}s")
    
            # ---- parallel process days (safe with retry) ----
    if n_jobs is None:
        n_jobs = max(1, min(os.cpu_count() - 1, len(dates)))

    def _work(item):
        d, day_df = item
        date_key = pd.Timestamp(d).strftime("%Y-%m-%d")
        model = _pca_read(date_key)  # may be None if insufficient history
        return _process_one_day(day_df, model, lam, pca_presence_frac)

    def _work_safe(item):
        d, day_df = item
        try:
            res = _work(item)
            return (d, res, None)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return (d, None, tb)

    # keep the dispatch queue small on Windows to reduce memory pressure
    triples = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        prefer="processes",
        pre_dispatch=n_jobs,   # don't queue up too many jobs
        batch_size=1,          # one day per batch
        verbose=0,
    )(delayed(_work_safe)(item) for item in day_slices)

    # separate successes and failures
    ok = [(d, df) for (d, df, err) in triples if err is None]
    bad = [(d, err) for (d, df, err) in triples if err is not None]

    if bad:
        failed_days = [pd.Timestamp(d).date().isoformat() for d, _ in bad]
        print(f"[WARN] {len(bad)} day(s) failed in parallel; retrying serially: {failed_days}")
        for d, err in bad:
            # optional: show the traceback now, so you keep a record
            print(f"\n[TRACEBACK] day={pd.Timestamp(d).date()}\n{err}")

        # retry failed days serially with fresh slices (keeps memory low)
        for d, _ in bad:
            slice_df = df_long[df_long["date"] == d].copy()
            try:
                res = _work((d, slice_df))
                ok.append((d, res))
            except Exception as e:
                import traceback
                print(f"[ERROR] day {pd.Timestamp(d).date()} failed again:\n{traceback.format_exc()}")

    # collect results in original date order
    results = [res for d, res in sorted(ok, key=lambda t: t[0])]


# ---- QuantLib calendar helpers ----
def _get_us_calendar(market_name: str):
    try:
        import QuantLib as ql
    except ImportError:
        raise ImportError("QuantLib not installed. `pip install QuantLib` (or quantlib-python)")

    m = market_name.lower()
    # Map string -> QuantLib enum
    if   m in ("federalreserve","fed","frb"): market = ql.UnitedStates(ql.UnitedStates.FederalReserve)
    elif m in ("settlement","usd","us_settlement"): market = ql.UnitedStates(ql.UnitedStates.Settlement)
    elif m in ("nyse","exchange"): market = ql.UnitedStates(ql.UnitedStates.NYSE)
    elif m in ("governmentbond","govt","govbond"): market = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    elif m in ("nerc",): market = ql.UnitedStates(ql.UnitedStates.NERC)
    else: raise ValueError(f"Unknown US market '{market_name}'")
    return market

def _filter_by_calendar(df: pd.DataFrame, *, market_name: str, tz_local: str) -> pd.DataFrame:
    """
    Keep only rows whose local DATE (in tz_local) is a QuantLib business day.
    Works on a tz-naive UTC DatetimeIndex (what we use everywhere).
    """
    if df.empty: return df
    import numpy as np
    import pytz
    cal = _get_us_calendar(market_name)
    # Convert UTC index -> local date (e.g., America/New_York)
    idx_utc = pd.DatetimeIndex(df.index, tz="UTC")
    idx_local = idx_utc.tz_convert(tz_local)
    local_dates = idx_local.normalize()  # midnight-local dates

    # Vectorized business-day mask using QuantLib
    # (QuantLib expects ql.Date(year, month, day))
    try:
        import QuantLib as ql
        y = local_dates.year.to_numpy()
        m = local_dates.month.to_numpy()
        d = local_dates.day.to_numpy()
        mask = np.fromiter(
            (cal.isBusinessDay(ql.Date(int(yy), int(mm), int(dd))) for yy, mm, dd in zip(y, m, d)),
            dtype=bool,
            count=len(local_dates)
        )
    except Exception:
        # Fallback: simple Mon–Fri if QuantLib not available at runtime
        mask = local_dates.dayofweek.to_series().isin([0,1,2,3,4]).to_numpy()

    # Apply mask on original df (UTC-naive index preserved)
    return df.iloc[mask]
    
def _load_month(yymm: str) -> pd.DataFrame:
    p = Path(PATH_DATA) / f"{yymm}_features.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_parquet(p, engine="pyarrow")

    # 1) ensure UTC-naive DateTimeIndex + dedupe + sort
    df = _ensure_dt_index(df)

    # 2) (optional) calendar filter BEFORE trading-hours slice to remove whole days
    if USE_QL_CALENDAR:
        try:
            df = _filter_by_calendar(df, market_name=QL_US_MARKET, tz_local=CAL_TZ)
        except Exception as e:
            print(f"[CAL] Warning: calendar filter failed ({e}); falling back to raw data.")

    # 3) intraday trading hours (still in UTC)
    if TRADING_HOURS:
        df = df.between_time(*TRADING_HOURS)

    # 4) zero→NaN cleanup
    if ZERO_AS_NAN:
        # to_numeric warning fix: use errors='coerce' explicitly
        df = df.apply(pd.to_numeric, errors="ignore")
        df = df.mask(df == 0.0)

    return df

def _process_one_day(day_df: pd.DataFrame, pca_model: dict | None,
                     lam: float, pca_presence_frac: float) -> pd.DataFrame:
    """
    day_df: long frame for ONE date with columns ['ts','tenor_yrs','rate', ...]
    returns: enhanced day frame with spline/pca/z columns for that date

    Notes:
      - Robust per-tick spline via _spline_fit_safe
      - EWMA std via numba (_ewma_std_nb)
      - PCA residuals without pivot(); handles duplicate tenors and partial coverage
      - RangeIndex used throughout to avoid label-alignment issues
      - Prints per-day progress and elapsed time
    """
    import time
    day_key = pd.to_datetime(day_df["ts"].iloc[0]).floor("D")
    n_ts    = int(day_df["ts"].nunique())
    n_tenor = int(day_df["tenor_yrs"].nunique())
    print(f"[DAY  ] {day_key.date()} | ticks: {n_ts:,} | tenors: {n_tenor} | PCA:{'yes' if pca_model else 'no'} ...", flush=True)
    t0 = time.perf_counter()

    # ---------- A) Spline per timestamp ----------
    frames = []
    for _, snap in day_df.groupby("ts", sort=True):
        frames.append(_spline_fit_safe(snap))

    day_splined = (
        pd.concat(frames, axis=0, ignore_index=True)
          .sort_values(["ts","tenor_yrs"])
          .reset_index(drop=True)
    )

    # EWMA per-tenor (Numba) + warmup-aware z
    day_splined["eps_spline_ewstd"] = (
        day_splined.groupby("tenor_yrs")["eps_spline"]
                   .transform(lambda s: _ewma_std_nb(s.fillna(0.0).to_numpy(), lam))
    ).clip(lower=SIGMA_FLOOR)

    cnt = day_splined.groupby("tenor_yrs").cumcount().to_numpy()
    z = (day_splined["eps_spline"].to_numpy() /
         day_splined["eps_spline_ewstd"].to_numpy())
    z[cnt < WARMUP_MIN_OBS] = np.nan
    day_splined["z_spline"] = z

    # ---------- B) PCA residuals per timestamp (if model available) ----------
    if pca_model is not None:
        # PCA model expects tenor keys like "0.500", "2.000", ...
        def _k(t) -> str:
            return f"{float(t):.3f}"

        def _pca_apply_block(df_block: pd.DataFrame) -> pd.DataFrame:
            # One timestamp slice: columns ['ts','tenor_yrs','rate',...]
            # 1) collapse duplicate tenors (keep last), build a 1-row levels vector
            dfb = (df_block[["tenor_yrs","rate"]]
                   .dropna()
                   .sort_values("tenor_yrs")
                   .drop_duplicates("tenor_yrs", keep="last"))
            out = df_block.copy()

            if dfb.empty or not pca_model.get("cols"):
                out["model_pca"] = np.nan
                out["eps_pca"]   = np.nan
                return out

            s = pd.Series(dfb["rate"].to_numpy(),
                          index=[_k(t) for t in dfb["tenor_yrs"].to_numpy()])

            model_cols = pca_model["cols"]
            present = [c for c in s.index if c in model_cols]
            coverage = len(present) / float(len(model_cols))
            if coverage < pca_presence_frac:
                # not enough instruments available to apply the model robustly
                out["model_pca"] = np.nan
                out["eps_pca"]   = np.nan
                return out

            # 2) Build a complete levels row on model grid:
            #    start from mu (ensures no NaNs) and overwrite with observed levels
            base = pd.Series(pca_model["mu"], index=model_cols)
            base.update(s)  # overwrite only where observed

            # 3) Fair reconstruction on model grid, then map back to each row’s tenor
            xhat = _pca_fair_row(base, pca_model)  # Series indexed by model_cols

            out["model_pca"] = out["tenor_yrs"].apply(lambda t: xhat.get(_k(t), np.nan))
            out["eps_pca"]   = out["rate"] - out["model_pca"]
            return out

        blocks = []
        for _, snap in day_splined.groupby("ts", sort=True):
            blocks.append(_pca_apply_block(snap))

        day_pca = (
            pd.concat(blocks, axis=0, ignore_index=True)
              .sort_values(["ts","tenor_yrs"])
              .reset_index(drop=True)
        )

        day_pca["eps_pca_ewstd"] = (
            day_pca.groupby("tenor_yrs")["eps_pca"]
                   .transform(lambda s: _ewma_std_nb(s.fillna(0.0).to_numpy(), lam))
        ).clip(lower=SIGMA_FLOOR)

        cnt_p = day_pca.groupby("tenor_yrs").cumcount().to_numpy()
        z_p = (day_pca["eps_pca"].to_numpy() /
               day_pca["eps_pca_ewstd"].to_numpy())
        z_p[cnt_p < WARMUP_MIN_OBS] = np.nan
        day_pca["z_pca"] = z_p
    else:
        day_pca = day_splined.copy()
        day_pca["model_pca"] = np.nan
        day_pca["eps_pca"]   = np.nan
        day_pca["eps_pca_ewstd"] = np.nan
        day_pca["z_pca"]     = np.nan

    # Blended z
    day_pca["z_comb"] = 0.5 * day_pca["z_spline"].astype(float) + 0.5 * day_pca["z_pca"].astype(float)

    dt = time.perf_counter() - t0
    print(f"[DONE ] {day_key.date()} | rows: {day_pca.shape[0]:,} | {dt:0.2f}s", flush=True)
    return day_pca

# feature_creation.py
# Guarded, parallelized feature builder with Numba EWMA
# - Handles 0.0 → NaN
# - Robust spline (per-tick) with positional assignment
# - Per-day PCA model (read-or-train once), applied intraday
# - Parallel per-day processing
# - RangeIndex everywhere to avoid pandas reindex issues

import os, sys, re, numpy as np, pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed
import numba as nb

from cr_config import (
    PATH_DATA, PATH_ENH, PATH_MOD, TRADING_HOURS,
    TENOR_YEARS, PCA_LOOKBACK_DAYS, PCA_COMPONENTS,
    EWMA_LAMBDA, SIGMA_FLOOR, WARMUP_MIN_OBS
)

os.makedirs(PATH_ENH, exist_ok=True)
os.makedirs(PATH_MOD, exist_ok=True)

# -----------------------
# toggles / constants
# -----------------------
ZERO_AS_NAN = True            # treat numeric 0.0 as NaN for rate columns
MID_SUFFIX  = "_mid"
RE_USOSFR   = re.compile(r"USOSFR(\d+(\.\d+)?)[A-Z]*\sBGN\sCurncy", re.IGNORECASE)

DAILY_LEVELS_PATH = Path(PATH_MOD) / "daily_levels.parquet"

# -----------------------
# utilities & guards
# -----------------------
def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Make the index a tz-naive UTC DatetimeIndex; keep ordering; drop dups."""
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            df.index = idx.tz_convert("UTC").tz_localize(None)
    else:
        # coerce; some inputs might be epoch seconds in a numeric index named 'sec'
        df.index = pd.to_datetime(idx, utc=True, errors="coerce").tz_convert("UTC").tz_localize(None)
    df = df[~df.index.isna()]
    return df[~df.index.duplicated(keep="last")].sort_index()

def _detect_mid_cols(df: pd.DataFrame) -> list[str]:
    explicit = [f"{k}{MID_SUFFIX}" for k in TENOR_YEARS.keys() if f"{k}{MID_SUFFIX}" in df.columns]
    return explicit if explicit else [c for c in df.columns if c.endswith(MID_SUFFIX)]

def _instrument_root(col: str) -> str:
    return col[:-len(MID_SUFFIX)] if col.endswith(MID_SUFFIX) else col

def _tenor_from_name(name: str) -> float | None:
    if name in TENOR_YEARS:  # explicit mapping takes priority
        return float(TENOR_YEARS[name])
    m = RE_USOSFR.match(name)  # best-effort fallback
    return float(m.group(1)) if m else None

def _load_month(yymm: str) -> pd.DataFrame:
    p = Path(PATH_DATA) / f"{yymm}_features.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_parquet(p, engine="pyarrow")
    df = _ensure_dt_index(df)
    if TRADING_HOURS:
        df = df.between_time(*TRADING_HOURS)
    # Optional: force numeric and zero→NaN
    if ZERO_AS_NAN:
        df = df.apply(pd.to_numeric, errors="ignore").mask(df == 0.0)
    return df

def _reset_index_to_ts(frame: pd.DataFrame) -> pd.DataFrame:
    """Reset index and ensure there's a column named 'ts' regardless of index name."""
    idx_name = frame.index.name or "index"
    out = frame.reset_index()
    if "ts" not in out.columns:
        out = out.rename(columns={idx_name: "ts", "index": "ts"})
    return out

def _wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    cols = _detect_mid_cols(df_wide)
    rows = []
    for col in cols:
        root  = _instrument_root(col)
        tenor = _tenor_from_name(root)
        if tenor is None:
            continue
        tmp = _reset_index_to_ts(df_wide[[col]].rename(columns={col: "rate"}))
        if ZERO_AS_NAN:
            tmp.loc[tmp["rate"] == 0.0, "rate"] = np.nan
        tmp = tmp.dropna(subset=["rate"])
        if tmp.empty:
            continue
        tmp["tenor_yrs"] = float(tenor)
        rows.append(tmp[["ts","tenor_yrs","rate"]])
    if not rows:
        return pd.DataFrame(columns=["ts","tenor_yrs","rate"])
    out = pd.concat(rows, axis=0, ignore_index=True).sort_values(["ts","tenor_yrs"])
    out["ts"] = pd.to_datetime(out["ts"], utc=False, errors="coerce")
    return out.dropna(subset=["ts"])

# -----------------------
# Numba EWMA std (fast)
# -----------------------
@nb.njit(cache=True, fastmath=True)
def _ewma_std_nb(x: np.ndarray, lam: float) -> np.ndarray:
    v = 0.0
    out = np.empty(x.shape[0], dtype=np.float64)
    one_minus = 1.0 - lam
    for i in range(x.shape[0]):
        xi = x[i]
        v = lam * v + one_minus * xi * xi
        if v < 1e-12:
            v = 1e-12
        out[i] = np.sqrt(v)
    return out

# -----------------------
# Robust spline fitter (final)
# -----------------------
def _spline_fit_safe(df_snap: pd.DataFrame) -> pd.DataFrame:
    """
    df_snap: rows for one timestamp; columns: ts, tenor_yrs, rate.
    Robust to duplicate timestamp labels & duplicate tenors.
    Positional assignment to avoid index alignment.
    """
    out = df_snap.reset_index(drop=True).copy()
    out["model_spline"] = np.nan
    out["eps_spline"]   = np.nan

    sub = out.dropna(subset=["rate"]).copy()
    if sub.shape[0] < 2:
        return out

    sub_uni = (
        sub.groupby("tenor_yrs", as_index=False, sort=True)["rate"]
           .mean().sort_values("tenor_yrs").reset_index(drop=True)
    )
    x = sub_uni["tenor_yrs"].to_numpy()
    y = sub_uni["rate"].to_numpy()
    if x.size < 2:
        return out

    if x.size == 2:
        if np.isclose(x[1] - x[0], 0.0):
            ybar = float(np.mean(y))
            def f_eval(z): 
                z = np.asarray(z, float); return np.full_like(z, ybar)
        else:
            m = (y[1] - y[0]) / (x[1] - x[0]); b = y[0] - m * x[0]
            def f_eval(z): 
                z = np.asarray(z, float); return m * z + b
    else:
        try:
            cs = CubicSpline(x, y, bc_type="natural")
            def f_eval(z): return cs(np.asarray(z, float))
        except Exception:
            coef = np.polyfit(x, y, 1)
            def f_eval(z): 
                z = np.asarray(z, float); return np.polyval(coef, z)

    tenors_all = sub["tenor_yrs"].to_numpy()
    m_vals  = f_eval(tenors_all)
    eps_vals = sub["rate"].to_numpy() - m_vals

    pos = sub.index.to_numpy()
    out.iloc[pos, out.columns.get_loc("model_spline")] = m_vals
    out.iloc[pos, out.columns.get_loc("eps_spline")]   = eps_vals
    return out

# -----------------------
# PCA helpers
# -----------------------
def _load_daily_panel() -> pd.DataFrame:
    return pd.read_parquet(DAILY_LEVELS_PATH, engine="pyarrow") if DAILY_LEVELS_PATH.exists() else pd.DataFrame()

def _save_daily_panel(panel: pd.DataFrame):
    panel.to_parquet(DAILY_LEVELS_PATH, engine="pyarrow")

def _train_pca_daily(dfx_levels: pd.DataFrame, asof_date: pd.Timestamp,
                     lookback_days=126, K=3) -> dict | None:
    hist = dfx_levels[dfx_levels.index < asof_date].iloc[-lookback_days:]
    if hist.shape[0] < max(20, lookback_days // 4):
        return None
    dX = hist.diff().dropna()
    if dX.shape[0] < 5:
        return None
    C = np.cov(dX.values.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order[:K]]
    mu = hist.mean().values
    cols = list(hist.columns)
    return {"mu": mu, "V": V, "cols": cols}

def _pca_write(date_key: str, model: dict):
    npz = Path(PATH_MOD) / f"pca_{date_key}.npz"
    np.savez_compressed(npz, mu=model["mu"], V=model["V"], cols=np.array(model["cols"], dtype="U"))

def _pca_read(date_key: str) -> dict | None:
    npz = Path(PATH_MOD) / f"pca_{date_key}.npz"
    if not npz.exists():
        return None
    z = np.load(npz, allow_pickle=False)
    return {"mu": z["mu"], "V": z["V"], "cols": list(z["cols"])}

def _pca_fair_row(levels_row: pd.Series, model: dict) -> pd.Series:
    x = levels_row.reindex(model["cols"]).values
    mu, V = model["mu"], model["V"]
    coeffs = (x - mu) @ V
    xhat = mu + coeffs @ V.T
    return pd.Series(xhat, index=model["cols"])

# -----------------------
# Per-day worker (parallel)
# -----------------------
def _process_one_day(day_df: pd.DataFrame, pca_model: dict | None,
                     lam: float, pca_presence_frac: float) -> pd.DataFrame:
    """
    day_df: long frame for ONE date [ts, tenor_yrs, rate]
    returns: enhanced day frame with spline/pca/z columns
    """
    # --- A) spline per timestamp ---
    frames = []
    for ts, snap in day_df.groupby("ts", sort=True):
        frames.append(_spline_fit_safe(snap))

    day_splined = (
        pd.concat(frames, axis=0, ignore_index=True)
          .sort_values(["ts","tenor_yrs"])
          .reset_index(drop=True)
    )

    # EWMA via Numba transform
    day_splined["eps_spline_ewstd"] = (
        day_splined.groupby("tenor_yrs")["eps_spline"]
                   .transform(lambda s: _ewma_std_nb(s.fillna(0.0).to_numpy(), lam))
    ).clip(lower=SIGMA_FLOOR)

    cnt = day_splined.groupby("tenor_yrs").cumcount().to_numpy()
    z = (day_splined["eps_spline"].to_numpy() /
         day_splined["eps_spline_ewstd"].to_numpy())
    z[cnt < WARMUP_MIN_OBS] = np.nan
    day_splined["z_spline"] = z

    # --- B) PCA residuals (if model available) ---
    if pca_model is not None:
        def _pca_apply_block(df_block: pd.DataFrame) -> pd.DataFrame:
            lv = df_block.pivot(index=None, columns="tenor_yrs", values="rate")
            lv.columns = [f"{c:.3f}" for c in lv.columns]
            present = lv.columns.intersection(pca_model["cols"])
            if len(present) / max(1, len(pca_model["cols"])) < pca_presence_frac:
                out = df_block.copy()
                out["model_pca"] = np.nan
                out["eps_pca"]   = np.nan
                return out
            lv_row = lv.iloc[0]
            xhat = _pca_fair_row(lv_row, pca_model)
            out = df_block.copy()
            out["model_pca"] = out["tenor_yrs"].apply(lambda t: xhat.get(f"{t:.3f}", np.nan))
            out["eps_pca"]   = out["rate"] - out["model_pca"]
            return out

        blocks = []
        for ts, snap in day_splined.groupby("ts", sort=True):
            blocks.append(_pca_apply_block(snap))

        day_pca = (
            pd.concat(blocks, axis=0, ignore_index=True)
              .sort_values(["ts","tenor_yrs"])
              .reset_index(drop=True)
        )

        day_pca["eps_pca_ewstd"] = (
            day_pca.groupby("tenor_yrs")["eps_pca"]
                   .transform(lambda s: _ewma_std_nb(s.fillna(0.0).to_numpy(), lam))
        ).clip(lower=SIGMA_FLOOR)

        cnt_p = day_pca.groupby("tenor_yrs").cumcount().to_numpy()
        z_p = (day_pca["eps_pca"].to_numpy() /
               day_pca["eps_pca_ewstd"].to_numpy())
        z_p[cnt_p < WARMUP_MIN_OBS] = np.nan
        day_pca["z_pca"] = z_p
    else:
        day_pca = day_splined.copy()
        day_pca["model_pca"] = np.nan
        day_pca["eps_pca"]   = np.nan
        day_pca["eps_pca_ewstd"] = np.nan
        day_pca["z_pca"]     = np.nan

    # blended z
    day_pca["z_comb"] = 0.5 * day_pca["z_spline"].astype(float) + 0.5 * day_pca["z_pca"].astype(float)
    return day_pca

# -----------------------
# Main builder (parallel by day)
# -----------------------
def build_month_guarded(yymm: str, lam=EWMA_LAMBDA, pca_presence_frac=0.6, n_jobs: int | None = None) -> None:
    """
    Writes PATH_ENH/<yymm>_enh.parquet with guarded spline & PCA z's per tick.
    - Trains/loads a PCA model per decision day (serial, once per day)
    - Processes each day in parallel (joblib) with Numba EWMA
    """
    df_wide = _load_month(yymm)
    if df_wide.empty:
        print(f"[{yymm}] empty month after trading hours slice.")
        return

    df_long = _wide_to_long(df_wide)
    if df_long.empty:
        print(f"[{yymm}] no recognizable *_mid columns / tenor map.")
        return

    # ---- daily close panel for PCA training ----
    df_long["date"] = pd.to_datetime(df_long["ts"]).dt.floor("D")
    daily_close = (
        df_long.sort_values("ts")
               .groupby(["date","tenor_yrs"])
               .tail(1)
               .pivot(index="date", columns="tenor_yrs", values="rate")
               .sort_index()
    )
    daily_close.columns = [f"{c:.3f}" for c in daily_close.columns]

    panel = _load_daily_panel()
    panel = pd.concat([panel, daily_close], axis=0) if not panel.empty else daily_close.copy()
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    _save_daily_panel(panel)

    dates = sorted(daily_close.index.unique())

    # ---- ensure PCA model file exists for each day (serial) ----
    for d in dates:
        date_key = pd.Timestamp(d).strftime("%Y-%m-%d")
        if _pca_read(date_key) is None:
            model = _train_pca_daily(panel, asof_date=pd.Timestamp(d),
                                     lookback_days=PCA_LOOKBACK_DAYS, K=PCA_COMPONENTS)
            if model is not None:
                _pca_write(date_key, model)

    # ---- prepare day slices ----
    day_slices = [(d, df_long[df_long["date"] == d].copy()) for d in dates if not df_long[df_long["date"] == d].empty]

    # ---- parallel process days ----
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)

    def _work(item):
        d, day_df = item
        date_key = pd.Timestamp(d).strftime("%Y-%m-%d")
        model = _pca_read(date_key)  # might be None if insufficient history
        return _process_one_day(day_df, model, lam, pca_presence_frac)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_work)(item) for item in day_slices
    )

    enh = pd.concat(results, axis=0, ignore_index=True) if results else pd.DataFrame()
    if enh.empty:
        print(f"[{yymm}] produced no enhanced rows (insufficient PCA history or empty).")
        return

    enh = enh.sort_values(["ts","tenor_yrs"])[[
        "ts","tenor_yrs","rate",
        "model_spline","eps_spline","eps_spline_ewstd","z_spline",
        "model_pca","eps_pca","eps_pca_ewstd","z_pca",
        "z_comb"
    ]]
    out_path = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    enh.to_parquet(out_path, engine="pyarrow")
    print(f"[{yymm}] wrote enhanced -> {out_path}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_creation.py 2501 [2502 2503 ...] [--jobs N]")
        sys.exit(1)

    # optional: --jobs N at the end
    args = [a for a in sys.argv[1:] if not a.startswith("--jobs")]
    jobs = None
    for a in sys.argv[1:]:
        if a.startswith("--jobs"):
            try:
                jobs = int(a.split()[-1])  # handle "--jobs 6"
            except Exception:
                try:
                    jobs = int(a.split("=")[1])  # handle "--jobs=6"
                except Exception:
                    pass

    for yymm in args:
        build_month_guarded(yymm, n_jobs=jobs)
        
def _wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    cols = _detect_mid_cols(df_wide)

    # quick audit of zeros → NaN
    zero_cells = int((df_wide[cols] == 0.0).sum().sum()) if len(cols) else 0

    # map instruments → tenor years and show summary
    mapping = []
    for col in cols:
        root  = _instrument_root(col)
        tenor = _tenor_from_name(root)
        if tenor is not None:
            mapping.append((root, tenor))
    mapping = sorted(mapping, key=lambda x: x[1])

    if mapping:
        roots_preview = ", ".join([f"{r}→{t:g}y" for r, t in mapping[:10]])
        extra = f" … (+{len(mapping)-10} more)" if len(mapping) > 10 else ""
        print(f"[FEATURE] Detected {len(mapping)} instruments | zeros→NaN: {zero_cells:,}")
        print(f"[FEATURE] Tenor map: {roots_preview}{extra}")
    else:
        print("[FEATURE] No recognizable *_mid columns / tenor map.")

    # build long frame
    rows = []
    for col in cols:
        root  = _instrument_root(col)
        tenor = _tenor_from_name(root)
        if tenor is None:
            continue
        tmp = _reset_index_to_ts(df_wide[[col]].rename(columns={col: "rate"}))
        # Treat 0.0 as NaN if present
        tmp.loc[tmp["rate"] == 0.0, "rate"] = np.nan
        tmp = tmp.dropna(subset=["rate"])
        if tmp.empty:
            continue
        tmp["tenor_yrs"] = float(tenor)
        rows.append(tmp[["ts","tenor_yrs","rate"]])

    if not rows:
        return pd.DataFrame(columns=["ts","tenor_yrs","rate"])

    out = pd.concat(rows, axis=0, ignore_index=True).sort_values(["ts","tenor_yrs"])
    out["ts"] = pd.to_datetime(out["ts"], utc=False, errors="coerce")
    out = out.dropna(subset=["ts"])
    print(f"[FEATURE] Long frame rows: {out.shape[0]:,} (ts×tenor rows)")
    return out
    
def _process_one_day(day_df: pd.DataFrame, pca_model: dict | None,
                     lam: float, pca_presence_frac: float) -> pd.DataFrame:
    """
    day_df: long frame for ONE date [ts, tenor_yrs, rate]
    returns: enhanced day frame with spline/pca/z columns
    """
    import time
    day_key = pd.to_datetime(day_df["ts"].iloc[0]).floor("D")
    n_ts    = int(day_df["ts"].nunique())
    n_tenor = int(day_df["tenor_yrs"].nunique())
    print(f"[DAY  ] {day_key.date()} | ticks: {n_ts:,} | tenors: {n_tenor} | PCA:{'yes' if pca_model else 'no'} ...", flush=True)
    t0 = time.perf_counter()

    # --- A) spline per timestamp ---
    frames = []
    for _, snap in day_df.groupby("ts", sort=True):
        frames.append(_spline_fit_safe(snap))

    day_splined = (
        pd.concat(frames, axis=0, ignore_index=True)
          .sort_values(["ts","tenor_yrs"])
          .reset_index(drop=True)
    )

    # EWMA via Numba transform
    day_splined["eps_spline_ewstd"] = (
        day_splined.groupby("tenor_yrs")["eps_spline"]
                   .transform(lambda s: _ewma_std_nb(s.fillna(0.0).to_numpy(), lam))
    ).clip(lower=SIGMA_FLOOR)

    cnt = day_splined.groupby("tenor_yrs").cumcount().to_numpy()
    z = (day_splined["eps_spline"].to_numpy() /
         day_splined["eps_spline_ewstd"].to_numpy())
    z[cnt < WARMUP_MIN_OBS] = np.nan
    day_splined["z_spline"] = z

    # --- B) PCA residuals (if model available) ---
    if pca_model is not None:
        def _pca_apply_block(df_block: pd.DataFrame) -> pd.DataFrame:
            lv = df_block.pivot(index=None, columns="tenor_yrs", values="rate")
            lv.columns = [f"{c:.3f}" for c in lv.columns]
            present = lv.columns.intersection(pca_model["cols"])
            if len(present) / max(1, len(pca_model["cols"])) < pca_presence_frac:
                out = df_block.copy()
                out["model_pca"] = np.nan
                out["eps_pca"]   = np.nan
                return out
            lv_row = lv.iloc[0]
            xhat = _pca_fair_row(lv_row, pca_model)
            out = df_block.copy()
            out["model_pca"] = out["tenor_yrs"].apply(lambda t: xhat.get(f"{t:.3f}", np.nan))
            out["eps_pca"]   = out["rate"] - out["model_pca"]
            return out

        blocks = []
        for _, snap in day_splined.groupby("ts", sort=True):
            blocks.append(_pca_apply_block(snap))

        day_pca = (
            pd.concat(blocks, axis=0, ignore_index=True)
              .sort_values(["ts","tenor_yrs"])
              .reset_index(drop=True)
        )

        day_pca["eps_pca_ewstd"] = (
            day_pca.groupby("tenor_yrs")["eps_pca"]
                   .transform(lambda s: _ewma_std_nb(s.fillna(0.0).to_numpy(), lam))
        ).clip(lower=SIGMA_FLOOR)

        cnt_p = day_pca.groupby("tenor_yrs").cumcount().to_numpy()
        z_p = (day_pca["eps_pca"].to_numpy() /
               day_pca["eps_pca_ewstd"].to_numpy())
        z_p[cnt_p < WARMUP_MIN_OBS] = np.nan
        day_pca["z_pca"] = z_p
    else:
        day_pca = day_splined.copy()
        day_pca["model_pca"] = np.nan
        day_pca["eps_pca"]   = np.nan
        day_pca["eps_pca_ewstd"] = np.nan
        day_pca["z_pca"]     = np.nan

    # blended z
    day_pca["z_comb"] = 0.5 * day_pca["z_spline"].astype(float) + 0.5 * day_pca["z_pca"].astype(float)

    dt = time.perf_counter() - t0
    print(f"[DONE ] {day_key.date()} | rows: {day_pca.shape[0]:,} | {dt:0.2f}s", flush=True)
    return day_pca
    
def build_month_guarded(yymm: str, lam=EWMA_LAMBDA, pca_presence_frac=0.6, n_jobs: int | None = None) -> None:
    """
    Writes PATH_ENH/<yymm>_enh.parquet with guarded spline & PCA z's per tick.
    - Trains/loads a PCA model per decision day (serial, once per day)
    - Processes each day in parallel (joblib) with Numba EWMA
    - Prints progress: instruments, dates, per-day timing, overall elapsed
    """
    import time
    t0 = time.perf_counter()

    df_wide = _load_month(yymm)
    if df_wide.empty:
        print(f"[{yymm}] empty month after trading hours slice.")
        return

    df_long = _wide_to_long(df_wide)
    if df_long.empty:
        print(f"[{yymm}] no recognizable *_mid columns / tenor map.")
        return

    # ---- daily close panel for PCA training ----
    df_long["date"] = pd.to_datetime(df_long["ts"]).dt.floor("D")
    daily_close = (
        df_long.sort_values("ts")
               .groupby(["date","tenor_yrs"])
               .tail(1)
               .pivot(index="date", columns="tenor_yrs", values="rate")
               .sort_index()
    )
    daily_close.columns = [f"{c:.3f}" for c in daily_close.columns]

    panel = _load_daily_panel()
    panel = pd.concat([panel, daily_close], axis=0) if not panel.empty else daily_close.copy()
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    _save_daily_panel(panel)

    dates = sorted(daily_close.index.unique())
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)
    print(f"[{yymm}] days: {len(dates)} | jobs: {n_jobs} | rows(long): {df_long.shape[0]:,}")

    # ---- ensure PCA model file exists for each day (serial) ----
    for d in dates:
        date_key = pd.Timestamp(d).strftime("%Y-%m-%d")
        if _pca_read(date_key) is None:
            model = _train_pca_daily(panel, asof_date=pd.Timestamp(d),
                                     lookback_days=PCA_LOOKBACK_DAYS, K=PCA_COMPONENTS)
            if model is not None:
                _pca_write(date_key, model)

    # ---- prepare day slices ----
    day_slices = [(d, df_long[df_long["date"] == d].copy()) for d in dates if not df_long[df_long["date"] == d].empty]

    # ---- parallel process days ----
    from joblib import Parallel, delayed
    def _work(item):
        d, day_df = item
        date_key = pd.Timestamp(d).strftime("%Y-%m-%d")
        model = _pca_read(date_key)  # might be None if insufficient history
        return _process_one_day(day_df, model, lam, pca_presence_frac)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_work)(item) for item in day_slices
    )

    enh = pd.concat(results, axis=0, ignore_index=True) if results else pd.DataFrame()
    if enh.empty:
        print(f"[{yymm}] produced no enhanced rows (insufficient PCA history or empty).")
        return

    enh = enh.sort_values(["ts","tenor_yrs"])[[
        "ts","tenor_yrs","rate",
        "model_spline","eps_spline","eps_spline_ewstd","z_spline",
        "model_pca","eps_pca","eps_pca_ewstd","z_pca",
        "z_comb"
    ]]
    out_path = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    enh.to_parquet(out_path, engine="pyarrow")

    dt = time.perf_counter() - t0
    print(f"[{yymm}] wrote enhanced -> {out_path} | rows: {enh.shape[0]:,} | {dt:0.2f}s")

def _spline_fit_safe(df_snap: pd.DataFrame) -> pd.DataFrame:
    """
    df_snap: rows for one timestamp; columns: ts, tenor_yrs, rate.
    Robust to:
      - duplicate timestamp index labels
      - duplicate tenors at the same timestamp
      - only 2 unique tenors (linear fallback)
      - identical 2x tenor (flat fallback)
    Uses positional assignment only (no label alignment).
    """
    # fresh integer index → no dup-label alignment issues
    out = df_snap.reset_index(drop=True).copy()
    out["model_spline"] = np.nan
    out["eps_spline"]   = np.nan

    sub = out.dropna(subset=["rate"]).copy()
    if sub.shape[0] < 2:
        return out  # not enough points to fit

    # aggregate duplicate tenors
    sub_uni = (
        sub.groupby("tenor_yrs", as_index=False, sort=True)["rate"]
           .mean()
           .sort_values("tenor_yrs")
           .reset_index(drop=True)
    )
    x = sub_uni["tenor_yrs"].to_numpy()
    y = sub_uni["rate"].to_numpy()
    if x.size < 2:
        return out

    # build evaluator
    if x.size == 2:
        if np.isclose(x[1] - x[0], 0.0):
            ybar = float(np.mean(y))
            def f_eval(z): 
                z = np.asarray(z, float); return np.full_like(z, ybar)
        else:
            m = (y[1] - y[0]) / (x[1] - x[0]); b = y[0] - m * x[0]
            def f_eval(z): 
                z = np.asarray(z, float); return m * z + b
    else:
        try:
            cs = CubicSpline(x, y, bc_type="natural")
            def f_eval(z): return cs(np.asarray(z, float))
        except Exception:
            coef = np.polyfit(x, y, 1)
            def f_eval(z): 
                z = np.asarray(z, float); return np.polyval(coef, z)

    # evaluate at each original-row tenor (positional)
    tenors_all = sub["tenor_yrs"].to_numpy()
    m_vals   = f_eval(tenors_all)
    eps_vals = sub["rate"].to_numpy() - m_vals

    # positional assignment (no alignment)
    pos = sub.index.to_numpy()
    out.iloc[pos, out.columns.get_loc("model_spline")] = m_vals
    out.iloc[pos, out.columns.get_loc("eps_spline")]   = eps_vals
    return out

def build_month_guarded(yymm: str, lam=EWMA_LAMBDA, pca_presence_frac=0.6) -> None:
    """
    Writes PATH_ENH/<yymm>_enh.parquet with guarded spline & PCA z's per tick.
    PCA model: one per decision day, applied to all ticks that day.
    Notes:
      - Uses RangeIndex everywhere to avoid duplicate-label reindex issues
      - Group EWMA via .transform; z computed positionally with cumcount warmup
    """
    # ---- load month (with trading-hours slice & zero→NaN masking done upstream) ----
    df_wide = _load_month(yymm)
    if df_wide.empty:
        print(f"[{yymm}] empty month after trading hours slice.")
        return

    # ---- wide→long (ts, tenor_yrs, rate), dropping NaNs/zeros ----
    df_long = _wide_to_long(df_wide)
    if df_long.empty:
        print(f"[{yymm}] no recognizable *_mid columns / tenor map.")
        return

    # ---- daily close panel (levels in %) for PCA training ----
    df_long["date"] = pd.to_datetime(df_long["ts"]).dt.floor("D")
    daily_close = (
        df_long.sort_values("ts")
               .groupby(["date","tenor_yrs"])
               .tail(1)
               .pivot(index="date", columns="tenor_yrs", values="rate")
               .sort_index()
    )
    daily_close.columns = [f"{c:.3f}" for c in daily_close.columns]

    panel = _load_daily_panel()
    panel = pd.concat([panel, daily_close], axis=0) if not panel.empty else daily_close.copy()
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    _save_daily_panel(panel)

    out_days = []
    dates = sorted(daily_close.index.unique())

    for d in dates:
        date_key = pd.Timestamp(d).strftime("%Y-%m-%d")

        # ---- PCA model for the day (read-or-train) ----
        pca_model = _pca_read(date_key)
        if pca_model is None:
            pca_model = _train_pca_daily(panel, asof_date=pd.Timestamp(d),
                                         lookback_days=PCA_LOOKBACK_DAYS, K=PCA_COMPONENTS)
            if pca_model is not None:
                _pca_write(date_key, pca_model)

        # ---- day slice ----
        day_slice = df_long[df_long["date"] == d].copy()
        if day_slice.empty:
            continue

        # ==========================================================
        # A) Spline per timestamp (guards inside _spline_fit_safe)
        #    -> concat with ignore_index, keep RangeIndex
        # ==========================================================
        frames = []
        for ts, snap in day_slice.groupby("ts", sort=True):
            frames.append(_spline_fit_safe(snap))

        day_splined = (
            pd.concat(frames, axis=0, ignore_index=True)
              .sort_values(["ts","tenor_yrs"])
              .reset_index(drop=True)     # RangeIndex
        )

        # ---- EWMA per-tenor (intraday only) via transform; sigma floor ----
        day_splined["eps_spline_ewstd"] = (
            day_splined.groupby("tenor_yrs")["eps_spline"]
                       .transform(lambda s: _ewma_std(s.fillna(0.0).to_numpy(), lam=lam))
        ).clip(lower=SIGMA_FLOOR)

        # ---- warmup-aware z (positional; no label alignment) ----
        cnt = day_splined.groupby("tenor_yrs").cumcount().to_numpy()
        z = (day_splined["eps_spline"].to_numpy() /
             day_splined["eps_spline_ewstd"].to_numpy())
        z[cnt < WARMUP_MIN_OBS] = np.nan
        day_splined["z_spline"] = z

        # ==========================================================
        # B) PCA residuals per timestamp (if model available)
        #    -> concat with ignore_index, RangeIndex; EWMA via transform
        # ==========================================================
        if pca_model is not None:
            def _pca_apply_block(df_block: pd.DataFrame) -> pd.DataFrame:
                lv = df_block.pivot(index=None, columns="tenor_yrs", values="rate")
                lv.columns = [f"{c:.3f}" for c in lv.columns]
                present = lv.columns.intersection(pca_model["cols"])
                if len(present) / max(1, len(pca_model["cols"])) < pca_presence_frac:
                    out = df_block.copy()
                    out["model_pca"] = np.nan
                    out["eps_pca"]   = np.nan
                    return out
                lv_row = lv.iloc[0]
                xhat = _pca_fair_row(lv_row, pca_model)
                out = df_block.copy()
                out["model_pca"] = out["tenor_yrs"].apply(lambda t: xhat.get(f"{t:.3f}", np.nan))
                out["eps_pca"]   = out["rate"] - out["model_pca"]
                return out

            blocks = []
            for ts, snap in day_splined.groupby("ts", sort=True):
                blocks.append(_pca_apply_block(snap))

            day_pca = (
                pd.concat(blocks, axis=0, ignore_index=True)
                  .sort_values(["ts","tenor_yrs"])
                  .reset_index(drop=True)   # RangeIndex
            )

            day_pca["eps_pca_ewstd"] = (
                day_pca.groupby("tenor_yrs")["eps_pca"]
                       .transform(lambda s: _ewma_std(s.fillna(0.0).to_numpy(), lam=lam))
            ).clip(lower=SIGMA_FLOOR)

            cnt_p = day_pca.groupby("tenor_yrs").cumcount().to_numpy()
            z_p = (day_pca["eps_pca"].to_numpy() /
                   day_pca["eps_pca_ewstd"].to_numpy())
            z_p[cnt_p < WARMUP_MIN_OBS] = np.nan
            day_pca["z_pca"] = z_p
        else:
            day_pca = day_splined.copy()
            day_pca["model_pca"] = np.nan
            day_pca["eps_pca"]   = np.nan
            day_pca["eps_pca_ewstd"] = np.nan
            day_pca["z_pca"]     = np.nan

        # ---- blended z (simple average for now) ----
        day_pca["z_comb"] = 0.5 * day_pca["z_spline"].astype(float) + 0.5 * day_pca["z_pca"].astype(float)

        out_days.append(day_pca)

    # ---- concat all days and write ----
    enh = pd.concat(out_days, axis=0, ignore_index=True) if out_days else pd.DataFrame()
    if enh.empty:
        print(f"[{yymm}] produced no enhanced rows (insufficient PCA history or empty).")
        return

    enh = enh.sort_values(["ts","tenor_yrs"])[[
        "ts","tenor_yrs","rate",
        "model_spline","eps_spline","eps_spline_ewstd","z_spline",
        "model_pca","eps_pca","eps_pca_ewstd","z_pca",
        "z_comb"
    ]]
    out_path = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    enh.to_parquet(out_path)
    print(f"[{yymm}] wrote enhanced -> {out_path}")

def _spline_fit_safe(df_snap: pd.DataFrame) -> pd.DataFrame:
    """
    df_snap: rows for one timestamp; columns: ts, tenor_yrs, rate.
    Robust to:
      - duplicate timestamp index labels
      - duplicate tenors at the same timestamp
      - only 2 unique tenors (linear fallback)
      - identical 2x tenor (flat fallback)
    Uses positional assignment only (no label alignment).
    """
    # Work on a copy with a fresh integer index to avoid any label alignment entirely
    out = df_snap.reset_index(drop=True).copy()
    out["model_spline"] = np.nan
    out["eps_spline"]   = np.nan

    # Subset with valid rates
    sub = out.dropna(subset=["rate"]).copy()
    if sub.shape[0] < 2:
        return out  # not enough points to fit

    # Aggregate duplicates by tenor before fitting
    sub_uni = (
        sub.groupby("tenor_yrs", as_index=False, sort=True)["rate"]
           .mean()
           .sort_values("tenor_yrs")
           .reset_index(drop=True)
    )
    x = sub_uni["tenor_yrs"].to_numpy()
    y = sub_uni["rate"].to_numpy()
    if x.size < 2:
        return out

    # Build evaluator
    if x.size == 2:
        if np.isclose(x[1] - x[0], 0.0):
            ybar = float(np.mean(y))
            def f_eval(z): 
                z = np.asarray(z, float); return np.full_like(z, ybar)
        else:
            m = (y[1] - y[0]) / (x[1] - x[0]); b = y[0] - m * x[0]
            def f_eval(z): 
                z = np.asarray(z, float); return m * z + b
    else:
        try:
            cs = CubicSpline(x, y, bc_type="natural")
            def f_eval(z): return cs(np.asarray(z, float))
        except Exception:
            coef = np.polyfit(x, y, 1)
            def f_eval(z): 
                z = np.asarray(z, float); return np.polyval(coef, z)

    # Evaluate at each original row’s tenor (positional)
    tenors_all = sub["tenor_yrs"].to_numpy()
    m_vals  = f_eval(tenors_all)
    eps_vals = sub["rate"].to_numpy() - m_vals

    # Pure positional assignment via iloc (no alignment, no reindexing)
    pos = sub.index.to_numpy()                      # integer positions inside 'out'
    out.iloc[pos, out.columns.get_loc("model_spline")] = m_vals
    out.iloc[pos, out.columns.get_loc("eps_spline")]   = eps_vals
    return out

def _spline_fit_safe(df_snap: pd.DataFrame) -> pd.DataFrame:
    """
    df_snap: rows for one timestamp; columns: ts, tenor_yrs, rate
    Handles duplicate tenors safely and writes spline results
    using pure positional (iloc) assignment to avoid alignment errors.
    """
    out = df_snap.copy()
    out["model_spline"] = np.nan
    out["eps_spline"]   = np.nan

    sub = df_snap.dropna(subset=["rate"]).copy()
    if sub.shape[0] < 2:
        return out  # cannot fit anything

    # ----- deduplicate by tenor -----
    sub_uni = (
        sub.groupby("tenor_yrs", as_index=False, sort=True)["rate"]
           .mean()
           .sort_values("tenor_yrs")
           .reset_index(drop=True)
    )
    x = sub_uni["tenor_yrs"].to_numpy()
    y = sub_uni["rate"].to_numpy()
    if x.size < 2:
        return out

    # ----- fit curve -----
    if x.size == 2:
        if np.isclose(x[1] - x[0], 0.0):
            ybar = float(np.mean(y))
            def f_eval(z): return np.full_like(z, ybar, dtype=float)
        else:
            m = (y[1] - y[0]) / (x[1] - x[0])
            b = y[0] - m * x[0]
            def f_eval(z): return m * np.asarray(z, float) + b
    else:
        try:
            cs = CubicSpline(x, y, bc_type="natural")
            def f_eval(z): return cs(np.asarray(z, float))
        except Exception:
            coef = np.polyfit(x, y, 1)
            def f_eval(z): return np.polyval(coef, np.asarray(z, float))

    # ----- evaluate at each original tenor -----
    tenors_all = sub["tenor_yrs"].to_numpy()
    m_vals  = f_eval(tenors_all)
    eps_vals = sub["rate"].to_numpy() - m_vals

    # ----- pure positional assignment (no alignment) -----
    idx_pos = out.index.get_indexer(sub.index)
    col_model = out.columns.get_loc("model_spline")
    col_eps   = out.columns.get_loc("eps_spline")
    out.iloc[idx_pos, col_model] = m_vals
    out.iloc[idx_pos, col_eps]   = eps_vals
    return out

def _spline_fit_safe(df_snap: pd.DataFrame) -> pd.DataFrame:
    """
    df_snap: rows for one timestamp; columns: ts, tenor_yrs, rate
    Handles duplicate tenors by aggregating before fitting; writes results
    back aligned to the original (possibly duplicated) rows.
    """
    out = df_snap.copy()
    out["model_spline"] = np.nan
    out["eps_spline"]   = np.nan

    sub = df_snap.dropna(subset=["rate"]).copy()
    if sub.shape[0] < 2:
        return out  # cannot fit anything

    # ---- deduplicate by tenor (aggregate mean rate) ----
    sub_uni = (
        sub.groupby("tenor_yrs", as_index=False, sort=True)["rate"]
           .mean()
           .sort_values("tenor_yrs")
           .reset_index(drop=True)
    )
    x = sub_uni["tenor_yrs"].to_numpy()
    y = sub_uni["rate"].to_numpy()

    if x.size < 2:
        return out  # still not enough unique points

    # ---- fit curve on unique points ----
    if x.size == 2:
        # linear fallback; guard for identical x’s
        if np.isclose(x[1] - x[0], 0.0):
            ybar = float(np.mean(y))
            def f_eval(z): 
                z = np.asarray(z, dtype=float)
                return np.full_like(z, ybar)
        else:
            m = (y[1] - y[0]) / (x[1] - x[0])
            b = y[0] - m * x[0]
            def f_eval(z):
                z = np.asarray(z, dtype=float)
                return m * z + b
    else:
        try:
            cs = CubicSpline(x, y, bc_type="natural")
            def f_eval(z):
                return cs(np.asarray(z, dtype=float))
        except Exception:
            coef = np.polyfit(x, y, 1)
            def f_eval(z):
                z = np.asarray(z, dtype=float)
                return np.polyval(coef, z)

    # ---- evaluate at every original row’s tenor and assign (ndarray RHS!) ----
    tenors_all = sub["tenor_yrs"].to_numpy()
    m_vals  = f_eval(tenors_all)                       # ndarray
    eps_vals = sub["rate"].to_numpy() - m_vals        # ndarray

    # Using ndarray RHS prevents pandas from trying to align by (possibly duplicated) index labels.
    out.loc[sub.index, "model_spline"] = m_vals
    out.loc[sub.index, "eps_spline"]   = eps_vals
    return out

def _spline_fit_safe(df_snap: pd.DataFrame) -> pd.DataFrame:
    """
    df_snap: rows for one timestamp; columns: ts, tenor_yrs, rate
    Handles duplicate tenors by aggregating before fitting; writes results
    back aligned to the original (possibly duplicated) rows.
    """
    out = df_snap.copy()
    out["model_spline"] = np.nan
    out["eps_spline"]   = np.nan

    sub = df_snap.dropna(subset=["rate"]).copy()
    if sub.shape[0] < 2:
        return out  # cannot fit anything

    # ---- deduplicate by tenor (aggregate mean rate) ----
    # keep a small view for fitting
    sub_uni = (
        sub.groupby("tenor_yrs", as_index=False, sort=True)["rate"]
           .mean()
           .sort_values("tenor_yrs")
           .reset_index(drop=True)
    )
    x = sub_uni["tenor_yrs"].values
    y = sub_uni["rate"].values

    if len(x) < 2:
        return out  # still not enough unique points

    # ---- fit curve on unique points ----
    if len(x) == 2:
        # linear fallback; guard for identical x’s
        if np.isclose(x[1] - x[0], 0.0):
            # identical tenor twice → just use the average y as flat curve
            def f_eval(z): 
                return np.full_like(z, fill_value=np.mean(y), dtype=float)
        else:
            m = (y[1] - y[0]) / (x[1] - x[0])
            b = y[0] - m * x[0]
            def f_eval(z): 
                return m * np.asarray(z, dtype=float) + b
    else:
        try:
            cs = CubicSpline(x, y, bc_type="natural")
            def f_eval(z):
                return cs(np.asarray(z, dtype=float))
        except Exception:
            # robust linear fallback on unique points
            coef = np.polyfit(x, y, 1)
            def f_eval(z):
                return np.polyval(coef, np.asarray(z, dtype=float))

    # ---- evaluate at every original row’s tenor and assign ----
    m_all = pd.Series(f_eval(sub["tenor_yrs"].values), index=sub.index)
    eps_all = sub["rate"] - m_all

    out.loc[sub.index, "model_spline"] = m_all
    out.loc[sub.index, "eps_spline"]   = eps_all
    return out

 from zoneinfo import ZoneInfo

# ========= Session =========
TRADING_HOURS = ("10:00", "22:10")  # slice intraday when building features (UTC-naive index)
CHI = ZoneInfo("America/Chicago")

# ========= Instruments (explicit) =========
# Provide your exact mapping (instrument root WITHOUT "_mid" -> tenor years).
TENOR_YEARS = {
    # Example — replace with your real set (incl. sub-2y if you want them):
    "USOSFR1 BGN Curncy": 1.0,
    "USOSFR2 BGN Curncy": 2.0,
    "USOSFR3 BGN Curncy": 3.0,
    "USOSFR4 BGN Curncy": 4.0,
    "USOSFR5 BGN Curncy": 5.0,
    "USOSFR7 BGN Curncy": 7.0,
    "USOSFR10 BGN Curncy": 10.0,
    "USOSFR12 BGN Curncy": 12.0,
    "USOSFR15 BGN Curncy": 15.0,
    "USOSFR20 BGN Curncy": 20.0,
    "USOSFR25 BGN Curncy": 25.0,
    "USOSFR30 BGN Curncy": 30.0,
    "USOSFR40 BGN Curncy": 40.0,
    # Include short end if present, e.g.:
    # "USOSFR0.5Y BGN Curncy": 0.5,
    # "USOSFR1.5 BGN Curncy": 1.5,
}

# ========= Feature (builder) settings =========
# PCA trained off rolling DAILY CLOSE panel; applied intraday (per day).
PCA_LOOKBACK_DAYS = 126
PCA_COMPONENTS    = 3

# Intraday EWMA for residual std (per day, resets daily)
EWMA_LAMBDA       = 0.97
SIGMA_FLOOR       = 1e-4      # in % units (~0.01bp) to avoid huge z at open
WARMUP_MIN_OBS    = 5         # ignore first N ticks per tenor for z (NaN z during warmup)

# ========= Backtest decision layer =========
DECISION_FREQ = 'D'           # 'D' (daily) or 'H' (hourly) — used ONLY by the backtest

# Cheap–rich thresholds (restored)
Z_ENTRY = 1.25                # enter when cheap-rich z-spread >= Z_ENTRY
Z_EXIT  = 0.35                # take profit when |z-spread| <= Z_EXIT
Z_STOP  = 2.00                # stop if divergence since entry >= Z_STOP
MAX_HOLD_DAYS = 10            # max holding period for a pair

# ========= Risk & selection (unchanged from upgraded backtest) =========
# Buckets across the curve. Adjust if you add sub-2y.
BUCKETS = {
    "short":  (0.4, 1.9),     # 6M–<2Y
    "front":  (2.0, 3.0),
    "belly":  (3.1, 9.0),
    "long":   (10.0, 40.0),
}
MIN_SEP_YEARS = 1.0           # min tenor separation between pair legs

MAX_CONCURRENT_PAIRS = 3      # max pairs open at any decision time
PER_BUCKET_DV01_CAP  = 1.0    # proxy units per bucket
TOTAL_DV01_CAP       = 3.0    # proxy units total across curve
FRONT_END_DV01_CAP   = 1.0    # aggregate cap for 'short' bucket

# Fly-alignment gating (kept; no calendar). Turn off if you prefer.
FLY_GATE_ENABLE = True
FLY_DEFS        = [(1.0, 3.0, 5.0), (2.0, 5.0, 10.0)]
FLY_Z_MIN       = 0.3
FLY_ALIGN_MODE  = "loose"     # "loose" (reject opposite sign) | "strict" (must agree)
SHORT_END_EXTRA_Z = 0.3       # extra Z_ENTRY if a leg is in 'short' bucket
        
import os, sys, re, numpy as np, pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline

from cr_config import (
    PATH_DATA, PATH_ENH, PATH_MOD, TRADING_HOURS,
    TENOR_YEARS, PCA_LOOKBACK_DAYS, PCA_COMPONENTS,
    EWMA_LAMBDA, SIGMA_FLOOR, WARMUP_MIN_OBS
)

os.makedirs(PATH_ENH, exist_ok=True)
os.makedirs(PATH_MOD, exist_ok=True)

# ------- toggles -------
ZERO_AS_NAN = True   # treat literal 0.0 in rate columns as missing
MID_SUFFIX  = "_mid"

# Regex fallback for tenor years if not explicitly in TENOR_YEARS
RE_USOSFR = re.compile(r"USOSFR(\d+(\.\d+)?)[A-Z]*\sBGN\sCurncy", re.IGNORECASE)

DAILY_LEVELS_PATH = Path(PATH_MOD) / "daily_levels.parquet"

# -----------------------
# Utilities & guards
# -----------------------
def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Make the index a tz-naive UTC DatetimeIndex; keep ordering; drop dups."""
    idx = df.index
    # If it's already datetime, normalize tz
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            df.index = idx.tz_convert("UTC").tz_localize(None)
        else:
            df.index = idx
    else:
        # Attempt to coerce generically
        df.index = pd.to_datetime(idx, utc=True, errors="coerce").tz_convert("UTC").tz_localize(None)
    # Drop any rows that failed coercion
    df = df[~df.index.isna()]
    return df[~df.index.duplicated(keep="last")].sort_index()

def _detect_mid_cols(df: pd.DataFrame) -> list[str]:
    explicit = [f"{k}{MID_SUFFIX}" for k in TENOR_YEARS.keys() if f"{k}{MID_SUFFIX}" in df.columns]
    return explicit if explicit else [c for c in df.columns if c.endswith(MID_SUFFIX)]

def _instrument_root(col: str) -> str:
    return col[:-len(MID_SUFFIX)] if col.endswith(MID_SUFFIX) else col

def _tenor_from_name(name: str) -> float | None:
    if name in TENOR_YEARS:  # explicit mapping takes priority
        return float(TENOR_YEARS[name])
    m = RE_USOSFR.match(name)  # best-effort fallback
    return float(m.group(1)) if m else None

def _load_month(yymm: str) -> pd.DataFrame:
    p = Path(PATH_DATA) / f"{yymm}_features.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_parquet(p)
    df = _ensure_dt_index(df)
    if TRADING_HOURS:
        df = df.between_time(*TRADING_HOURS)
    # optional: treat 0.0 rates as NaN (safe for OIS)
    if ZERO_AS_NAN:
        df = df.mask(df == 0.0)
    return df

def _reset_index_to_ts(frame: pd.DataFrame) -> pd.DataFrame:
    """Reset index and ensure there's a column named 'ts' regardless of index name."""
    idx_name = frame.index.name or "index"
    out = frame.reset_index()
    if "ts" not in out.columns:
        # rename the index column (whatever it was) to 'ts'
        out = out.rename(columns={idx_name: "ts", "index": "ts"})
    return out

def _wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    cols = _detect_mid_cols(df_wide)
    rows = []
    for col in cols:
        root  = _instrument_root(col)
        tenor = _tenor_from_name(root)
        if tenor is None:
            continue  # skip unknown instruments safely
        tmp = _reset_index_to_ts(df_wide[[col]].rename(columns={col: "rate"}))
        # Treat zeros as NaN if requested (extra safety in case upstream masking missed)
        if ZERO_AS_NAN:
            tmp.loc[tmp["rate"] == 0.0, "rate"] = np.nan
        tmp = tmp.dropna(subset=["rate"])
        if tmp.empty:
            continue
        tmp["tenor_yrs"] = float(tenor)
        rows.append(tmp[["ts", "tenor_yrs", "rate"]])
    if not rows:
        return pd.DataFrame(columns=["ts","tenor_yrs","rate"])
    out = pd.concat(rows, axis=0).sort_values(["ts","tenor_yrs"])
    # ensure datetime type for 'ts'
    out["ts"] = pd.to_datetime(out["ts"], utc=False, errors="coerce")
    return out.dropna(subset=["ts"])

def _ewma_std(x: np.ndarray, lam=0.97) -> np.ndarray:
    v = 0.0; out = np.empty_like(x, dtype=float)
    for i, xi in enumerate(x):
        v = lam * v + (1 - lam) * xi * xi
        out[i] = np.sqrt(max(v, 1e-12))
    return out

def _spline_fit_safe(df_snap: pd.DataFrame) -> pd.DataFrame:
    sub = df_snap.dropna(subset=["rate"]).copy()
    out = df_snap.copy()
    out["model_spline"] = np.nan
    out["eps_spline"]   = np.nan
    if sub.shape[0] < 2:
        return out
    x = sub["tenor_yrs"].values
    y = sub["rate"].values
    idx = np.argsort(x)
    if sub.shape[0] == 2:
        x0, x1 = x[idx][0], x[idx][1]
        y0, y1 = y[idx][0], y[idx][1]
        m = y0 + (y1 - y0) * (sub["tenor_yrs"].values - x0) / (x1 - x0)
    else:
        try:
            cs = CubicSpline(x[idx], y[idx], bc_type="natural")
            m = cs(sub["tenor_yrs"].values)
        except Exception:
            coef = np.polyfit(x[idx], y[idx], 1)
            m = np.polyval(coef, sub["tenor_yrs"].values)
    eps = sub["rate"].values - m
    out.loc[sub.index, "model_spline"] = m
    out.loc[sub.index, "eps_spline"]   = eps
    return out

def _load_daily_panel() -> pd.DataFrame:
    return pd.read_parquet(DAILY_LEVELS_PATH) if DAILY_LEVELS_PATH.exists() else pd.DataFrame()

def _save_daily_panel(panel: pd.DataFrame):
    panel.to_parquet(DAILY_LEVELS_PATH)

def _train_pca_daily(dfx_levels: pd.DataFrame, asof_date: pd.Timestamp,
                     lookback_days=126, K=3) -> dict | None:
    hist = dfx_levels[dfx_levels.index < asof_date].iloc[-lookback_days:]
    if hist.shape[0] < max(20, lookback_days // 4):
        return None
    dX = hist.diff().dropna()
    if dX.shape[0] < 5:
        return None
    C = np.cov(dX.values.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order[:K]]
    mu = hist.mean().values
    cols = list(hist.columns)
    return {"mu": mu, "V": V, "cols": cols}

def _pca_write(date_key: str, model: dict):
    npz = Path(PATH_MOD) / f"pca_{date_key}.npz"
    np.savez_compressed(npz, mu=model["mu"], V=model["V"], cols=np.array(model["cols"], dtype="U"))

def _pca_read(date_key: str) -> dict | None:
    npz = Path(PATH_MOD) / f"pca_{date_key}.npz"
    if not npz.exists():
        return None
    z = np.load(npz, allow_pickle=False)
    return {"mu": z["mu"], "V": z["V"], "cols": list(z["cols"])}

def _pca_fair_row(levels_row: pd.Series, model: dict) -> pd.Series:
    x = levels_row.reindex(model["cols"]).values
    mu, V = model["mu"], model["V"]
    coeffs = (x - mu) @ V
    xhat = mu + coeffs @ V.T
    return pd.Series(xhat, index=model["cols"])

# -----------------------
# Main builder
# -----------------------
def build_month_guarded(yymm: str, lam=EWMA_LAMBDA, pca_presence_frac=0.6) -> None:
    """
    Writes PATH_ENH/<yymm>_enh.parquet with guarded spline & PCA z's per tick.
    PCA model: one per decision day, applied to all ticks that day.
    """
    df_wide = _load_month(yymm)
    if df_wide.empty:
        print(f"[{yymm}] empty month after trading hours slice.")
        return

    df_long = _wide_to_long(df_wide)
    if df_long.empty:
        print(f"[{yymm}] no recognizable *_mid columns / tenor map.")
        return

    # Daily close panel (levels in %)
    df_long["date"] = pd.to_datetime(df_long["ts"]).dt.floor("D")
    daily_close = (
        df_long.sort_values("ts")
               .groupby(["date","tenor_yrs"])
               .tail(1)
               .pivot(index="date", columns="tenor_yrs", values="rate")
               .sort_index()
    )
    daily_close.columns = [f"{c:.3f}" for c in daily_close.columns]

    panel = _load_daily_panel()
    panel = pd.concat([panel, daily_close], axis=0) if not panel.empty else daily_close.copy()
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    _save_daily_panel(panel)

    out_days = []
    dates = sorted(daily_close.index.unique())

    for d in dates:
        date_key = pd.Timestamp(d).strftime("%Y-%m-%d")
        # PCA model for the day
        pca_model = _pca_read(date_key)
        if pca_model is None:
            pca_model = _train_pca_daily(panel, asof_date=pd.Timestamp(d),
                                         lookback_days=PCA_LOOKBACK_DAYS, K=PCA_COMPONENTS)
            if pca_model is not None:
                _pca_write(date_key, pca_model)

        day_slice = df_long[df_long["date"] == d].copy()
        if day_slice.empty:
            continue

        # ----- Spline per timestamp with guards -----
        frames = []
        for ts, snap in day_slice.groupby("ts", sort=True):
            frames.append(_spline_fit_safe(snap))
        day_splined = pd.concat(frames, axis=0).sort_values(["ts","tenor_yrs"])

        # EWMA per-tenor (intraday), sigma floor + warmup
        day_splined["eps_spline_ewstd"] = (
            day_splined.groupby("tenor_yrs", group_keys=False)["eps_spline"]
                       .apply(lambda s: pd.Series(_ewma_std(s.fillna(0.0).values, lam=lam), index=s.index))
        ).clip(lower=SIGMA_FLOOR)

        def _z_with_warmup(g: pd.DataFrame, col_eps: str, col_std: str) -> pd.Series:
            idx = np.arange(len(g))
            z = g[col_eps] / g[col_std]
            z[idx < WARMUP_MIN_OBS] = np.nan
            return z

        day_splined["z_spline"] = (
            day_splined.groupby("tenor_yrs", group_keys=False)
                       .apply(lambda g: _z_with_warmup(g, "eps_spline", "eps_spline_ewstd"))
        )

        # ----- PCA residuals (if model available) -----
        if pca_model is not None:
            def _pca_apply_block(df_block: pd.DataFrame) -> pd.DataFrame:
                lv = df_block.pivot(index=None, columns="tenor_yrs", values="rate")
                lv.columns = [f"{c:.3f}" for c in lv.columns]
                present = lv.columns.intersection(pca_model["cols"])
                if len(present) / max(1, len(pca_model["cols"])) < pca_presence_frac:
                    out = df_block.copy()
                    out["model_pca"] = np.nan
                    out["eps_pca"]   = np.nan
                    return out
                lv_row = lv.iloc[0]
                xhat = _pca_fair_row(lv_row, pca_model)
                out = df_block.copy()
                out["model_pca"] = out["tenor_yrs"].apply(lambda t: xhat.get(f"{t:.3f}", np.nan))
                out["eps_pca"]   = out["rate"] - out["model_pca"]
                return out

            blocks = []
            for ts, snap in day_splined.groupby("ts", sort=True):
                blocks.append(_pca_apply_block(snap))
            day_pca = pd.concat(blocks, axis=0).sort_values(["ts","tenor_yrs"])

            day_pca["eps_pca_ewstd"] = (
                day_pca.groupby("tenor_yrs", group_keys=False)["eps_pca"]
                       .apply(lambda s: pd.Series(_ewma_std(s.fillna(0.0).values, lam=lam), index=s.index))
            ).clip(lower=SIGMA_FLOOR)

            day_pca["z_pca"] = (
                day_pca.groupby("tenor_yrs", group_keys=False)
                       .apply(lambda g: _z_with_warmup(g, "eps_pca", "eps_pca_ewstd"))
            )
        else:
            day_pca = day_splined.copy()
            day_pca["model_pca"] = np.nan
            day_pca["eps_pca"]   = np.nan
            day_pca["eps_pca_ewstd"] = np.nan
            day_pca["z_pca"]     = np.nan

        # ----- blended z -----
        day_pca["z_comb"] = 0.5 * day_pca["z_spline"].astype(float) + 0.5 * day_pca["z_pca"].astype(float)
        out_days.append(day_pca)

    enh = pd.concat(out_days, axis=0) if out_days else pd.DataFrame()
    if enh.empty:
        print(f"[{yymm}] produced no enhanced rows (insufficient PCA history or empty).")
        return

    enh = enh.sort_values(["ts","tenor_yrs"])[[
        "ts","tenor_yrs","rate",
        "model_spline","eps_spline","eps_spline_ewstd","z_spline",
        "model_pca","eps_pca","eps_pca_ewstd","z_pca",
        "z_comb"
    ]]
    out_path = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    enh.to_parquet(out_path)
    print(f"[{yymm}] wrote enhanced -> {out_path}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_features_ois_guarded.py 2501 [2502 2503 ...]")
        sys.exit(1)
    for yymm in sys.argv[1:]:
        build_month_guarded(yymm)
        
import os, sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from cr_config import (
    PATH_ENH, PATH_OUT, DECISION_FREQ, MIN_SEP_YEARS,
    Z_ENTRY, Z_EXIT, Z_STOP, MAX_HOLD_DAYS,
    BUCKETS, TENOR_YEARS,
    MAX_CONCURRENT_PAIRS, PER_BUCKET_DV01_CAP, TOTAL_DV01_CAP, FRONT_END_DV01_CAP,
    FLY_GATE_ENABLE, FLY_DEFS, FLY_Z_MIN, FLY_ALIGN_MODE,
    SHORT_END_EXTRA_Z
)

os.makedirs(PATH_OUT, exist_ok=True)

def pv01_proxy(tenor_yrs, rate_pct):
    return tenor_yrs / max(1e-6, 1.0 + 0.01*rate_pct)

def assign_bucket(tenor):
    for name, (lo, hi) in BUCKETS.items():
        if (tenor >= lo) and (tenor <= hi):
            return name
    return "other"

# ---- fly z computation (from z_comb cross-section) ----
def compute_fly_z(snap_last, a, b, c):
    # 0.5*(a+c) - b  on z_comb (shape, not on levels)
    vals = {}
    for T in [a,b,c]:
        row = snap_last[snap_last["tenor_yrs"]==T]
        if row.empty: return None
        vals[T] = float(row["z_comb"])
    fly = 0.5*(vals[a] + vals[c]) - vals[b]
    # crude standardization: use cross-sec std at that ts
    xs = snap_last["z_comb"].astype(float)
    sd = xs.std(ddof=1) if xs.size>1 else 1.0
    return fly / (sd if sd>0 else 1.0)

def fly_alignment_ok(leg_tenor, leg_sign_z, snap_last):
    """Return True if flies do not contradict the leg direction."""
    if not FLY_GATE_ENABLE: return True
    for (a,b,c) in FLY_DEFS:
        zz = compute_fly_z(snap_last, a,b,c)
        if zz is None: 
            continue
        if abs(zz) < FLY_Z_MIN:
            continue
        # If strict: require same sign; if loose: just reject opposite sign
        if FLY_ALIGN_MODE == "strict" and np.sign(zz) != np.sign(leg_sign_z):
            return False
        if FLY_ALIGN_MODE == "loose" and (np.sign(zz) * np.sign(leg_sign_z) < 0):
            return False
    return True

# ---- position object (unchanged-ish) ----
class PairPos:
    def __init__(self, open_ts, cheap, rich, w_i, w_j):
        self.open_ts = open_ts
        self.tenor_i = cheap["tenor_yrs"]; self.rate_i = cheap["rate"]
        self.tenor_j = rich["tenor_yrs"];  self.rate_j = rich["rate"]
        self.w_i = w_i; self.w_j = w_j
        self.entry_zspread = cheap["z_comb"] - rich["z_comb"]
        self.closed = False; self.close_ts = None; self.exit_reason = None
        self.pnl = 0.0; self.days_held = 0
        self.bucket_i = assign_bucket(self.tenor_i); self.bucket_j = assign_bucket(self.tenor_j)
        # attribution proxies
        self.last_zspread = self.entry_zspread
        self.conv_pnl_proxy = 0.0  # cum (Δz_spread) scaled

    def mark(self, snap_last):
        ri = float(snap_last[snap_last["tenor_yrs"]==self.tenor_i]["rate"])
        rj = float(snap_last[snap_last["tenor_yrs"]==self.tenor_j]["rate"])
        d_i = (self.rate_i - ri) * self.w_i * 100.0
        d_j = (self.rate_j - rj) * self.w_j * 100.0
        self.pnl = d_i + d_j

        # convergence proxy: move in z_spread (scale by 10 for readability)
        zi = float(snap_last[snap_last["tenor_yrs"]==self.tenor_i]["z_comb"])
        zj = float(snap_last[snap_last["tenor_yrs"]==self.tenor_j]["z_comb"])
        zsp = zi - zj
        self.conv_pnl_proxy += (self.last_zspread - zsp) * 10.0
        self.last_zspread = zsp
        return zsp

# ---- greedy selector for multiple pairs under caps ----
def choose_pairs_under_caps(snap_last, max_pairs,
                            per_bucket_cap, total_cap, front_end_cap, extra_z_entry):
    """
    Returns list of (cheap_row, rich_row, w_i, w_j)
    Greedy: rank by dispersion, enforce tenor uniqueness & DV01 caps by bucket and total.
    """
    sig = snap_last[["tenor_yrs","rate","z_comb"]].dropna().copy()
    if sig.empty: return []

    # Sort by z for extremes
    sig = sig.sort_values("z_comb")
    candidates = []
    used_tenors = set()

    # Build extreme pairs pool
    for k_low in range(min(5, len(sig))):
        rich = sig.iloc[k_low]
        for k_hi in range(1, min(8, len(sig))+1):
            cheap = sig.iloc[-k_hi]
            if cheap["tenor_yrs"] in used_tenors or rich["tenor_yrs"] in used_tenors:
                continue
            # separation and bucket diversification preferred
            if abs(cheap["tenor_yrs"] - rich["tenor_yrs"]) < MIN_SEP_YEARS:
                continue
            # fly alignment gates
            if not fly_alignment_ok(cheap["tenor_yrs"], +1, snap_last):  # cheap leg should revert down
                continue
            if not fly_alignment_ok(rich["tenor_yrs"], -1, snap_last):   # rich leg should revert up
                continue
            zdisp = float(cheap["z_comb"] - rich["z_comb"])
            if zdisp < (Z_ENTRY + extra_z_entry):
                continue
            candidates.append((zdisp, cheap, rich))

    # Greedy pack by zdisp descending under caps
    bucket_dv01 = {b: 0.0 for b in BUCKETS.keys()}
    total_dv01 = 0.0
    selected = []

    for zdisp, cheap, rich in sorted(candidates, key=lambda x: x[0], reverse=True):
        if len(selected) >= max_pairs:
            break
        if cheap["tenor_yrs"] in used_tenors or rich["tenor_yrs"] in used_tenors:
            continue

        # PV01 proxy weights (neutrality within pair)
        pv_i = pv01_proxy(cheap["tenor_yrs"], cheap["rate"])
        pv_j = pv01_proxy(rich["tenor_yrs"],  rich["rate"])
        w_i =  1.0
        w_j = - w_i * pv_i / pv_j

        # DV01 usage by buckets (proxy, abs weights)
        b_i = assign_bucket(cheap["tenor_yrs"]); b_j = assign_bucket(rich["tenor_yrs"])
        dv_i = abs(w_i) * pv_i; dv_j = abs(w_j) * pv_j
        # Check caps
        def _would_violate(bmap, b, add, cap):
            return (bmap[b] + add) > cap

        # per-bucket
        if b_i in bucket_dv01 and _would_violate(bucket_dv01, b_i, dv_i, PER_BUCKET_DV01_CAP):
            continue
        if b_j in bucket_dv01 and _would_violate(bucket_dv01, b_j, dv_j, PER_BUCKET_DV01_CAP):
            continue
        # front-end aggregate cap
        short_add = (dv_i if b_i=="short" else 0.0) + (dv_j if b_j=="short" else 0.0)
        short_tot = sum(v for k,v in bucket_dv01.items() if k=="short")
        if short_tot + short_add > front_end_cap:
            continue
        # total cap
        if total_dv01 + dv_i + dv_j > total_cap:
            continue

        # accept
        used_tenors.add(cheap["tenor_yrs"]); used_tenors.add(rich["tenor_yrs"])
        bucket_dv01[b_i] += dv_i; bucket_dv01[b_j] += dv_j
        total_dv01 += dv_i + dv_j
        selected.append((cheap, rich, w_i, w_j))

    return selected

def run_month(yymm: str, decision_freq='D'):
    p = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run build_features_ois.py first.")
    df = pd.read_parquet(p)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"])
    if decision_freq == 'D':
        df["decision_ts"] = df["ts"].dt.floor("D")
    elif decision_freq == 'H':
        df["decision_ts"] = df["ts"].dt.floor("H")
    else:
        raise ValueError("decision_freq must be 'D' or 'H'.")

    # z-heatmap (per month)
    _heat = (df.pivot_table(index="ts", columns="tenor_yrs", values="z_comb", aggfunc="last")
               .sort_index())
    heat_path = Path(PATH_OUT) / f"z_heatmap_{yymm}.png"
    if not _heat.empty:
        plt.figure()
        plt.imshow(_heat.T, aspect="auto", origin="lower")
        plt.title(f"z_comb heatmap {yymm}")
        plt.xlabel("time idx"); plt.ylabel("tenor (yrs)")
        plt.colorbar()
        plt.savefig(heat_path, dpi=120, bbox_inches="tight")
        plt.close()

    open_positions: list[PairPos] = []
    ledger_rows = []
    closed_rows = []

    for dts, snap in df.groupby("decision_ts", sort=True):
        snap_last = snap.sort_values("ts").groupby("tenor_yrs").tail(1)
        if snap_last.empty:
            continue

        # 1) Mark all open positions, decide exits
        still_open = []
        for pos in open_positions:
            zsp = pos.mark(snap_last)
            # exit rules
            if abs(zsp) <= Z_EXIT:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "reversion"
            elif abs(zsp - pos.entry_zspread) >= Z_STOP:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "stop"
            elif pos.days_held >= MAX_HOLD_DAYS:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "max_hold"
            else:
                pos.days_held += (1 if DECISION_FREQ=='D' else 0)  # light aging
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
                    "pnl": pos.pnl, "days_held": pos.days_held,
                    "conv_proxy": pos.conv_pnl_proxy
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2) Entries (greedy under caps), extra z threshold if front-end involved
        # Compute extra entry cost if any leg lands in short bucket
        extra_z = 0.0
        # try preview: if best dispersion uses any 'short' we’ll re-check in selector with added threshold.
        selected = choose_pairs_under_caps(
            snap_last=snap_last,
            max_pairs=max(0, MAX_CONCURRENT_PAIRS - len(open_positions)),
            per_bucket_cap=PER_BUCKET_DV01_CAP,
            total_cap=TOTAL_DV01_CAP,
            front_end_cap=FRONT_END_DV01_CAP,
            extra_z_entry=extra_z
        )

        for (cheap, rich, w_i, w_j) in selected:
            # “short” penalty: require larger dispersion if short bucket used
            if assign_bucket(cheap["tenor_yrs"])=="short" or assign_bucket(rich["tenor_yrs"])=="short":
                if (float(cheap["z_comb"] - rich["z_comb"]) < (Z_ENTRY + SHORT_END_EXTRA_Z)):
                    continue
            pos = PairPos(open_ts=dts, cheap=cheap, rich=rich, w_i=w_i, w_j=w_j)
            open_positions.append(pos)
            ledger_rows.append({
                "decision_ts": dts, "event": "open",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "w_i": pos.w_i, "w_j": pos.w_j,
                "entry_zspread": pos.entry_zspread
            })

    # Outputs
    pos_df = pd.DataFrame(closed_rows)
    ledger = pd.DataFrame(ledger_rows)
    # PnL by day
    pnl_by_day = (ledger[ledger["event"]=="mark"]
                  .groupby(ledger["decision_ts"].dt.floor("D"))["pnl"].sum()
                  .rename("daily_pnl").to_frame().reset_index()
                  .rename(columns={"decision_ts":"date"}))
    # Save month-level plots (cumulative PnL)
    if not pnl_by_day.empty:
        pnl_by_day = pnl_by_day.sort_values("date")
        pnl_by_day["cum"] = pnl_by_day["daily_pnl"].cumsum()
        plt.figure()
        plt.plot(pnl_by_day["date"], pnl_by_day["cum"])
        plt.title(f"Cumulative PnL proxy {yymm}")
        plt.xlabel("Date"); plt.ylabel("Cum PnL")
        out_png = Path(PATH_OUT) / f"pnl_curve_{yymm}.png"
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close()

    return pos_df, ledger, pnl_by_day

def run_all(yymms):
    all_pos = []; all_ledger = []; all_byday = []
    for yymm in yymms:
        print(f"[RUN] month {yymm}")
        p, l, b = run_month(yymm, decision_freq=DECISION_FREQ)
        if not p.empty: all_pos.append(p)
        if not l.empty: all_ledger.append(l)
        if not b.empty: all_byday.append(b.assign(yymm=yymm))
    pos = pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame()
    led = pd.concat(all_ledger, ignore_index=True) if all_ledger else pd.DataFrame()
    byday = pd.concat(all_byday, ignore_index=True) if all_byday else pd.DataFrame()
    return pos, led, byday

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cheap_rich_backtest.py 2501 [2502 2503 ...]")
        sys.exit(1)
    pos, led, byday = run_all(sys.argv[1:])
    if not pos.empty: pos.to_parquet(Path(PATH_OUT) / "positions_ledger.parquet")
    if not led.empty: led.to_parquet(Path(PATH_OUT) / "marks_ledger.parquet")
    if not byday.empty: byday.to_parquet(Path(PATH_OUT) / "pnl_by_day.parquet")
    print("[DONE]")