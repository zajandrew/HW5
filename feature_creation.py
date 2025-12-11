import os, sys, time
import datetime
import math
from pathlib import Path
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Use module-level config access everywhere
import cr_config as cr

# -----------------------
# Small utilities
# -----------------------
def _now():
    return time.strftime("%H:%M:%S")

def _to_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" not in df.columns:
        if df.index.name in ("ts", "sec"):
            df = df.reset_index().rename(columns={df.columns[0]: "ts"})
        else:
            raise KeyError("No 'ts' column found.")
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["ts"] = df["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("ts")

# -----------------------
# Calendar & hours
# -----------------------
def _get_ql_calendar():
    if not getattr(cr, "USE_QL_CALENDAR", False): return None
    try:
        import QuantLib as ql
        market = str(getattr(cr, "QL_US_MARKET", "FederalReserve"))
        direct = getattr(ql.UnitedStates, market, None)
        if direct: return ql.UnitedStates(direct)
        return ql.UnitedStates()
    except: return None

def _apply_calendar_and_hours(df_wide: pd.DataFrame) -> pd.DataFrame:
    if df_wide.empty: return df_wide
    
    # 1. Standard Weekday Filter first (fastest)
    ts = pd.to_datetime(df_wide["ts"])
    df_wide = df_wide[ts.dt.weekday < 5].copy()
    if df_wide.empty: return df_wide

    # 2. QuantLib Holiday Filter (The Missing Piece)
    cal = _get_ql_calendar()
    if cal:
        import QuantLib as ql
        # Get unique dates to minimize expensive QL calls
        unique_dates = df_wide["ts"].dt.date.unique()
        
        # Identify valid business days
        # Note: ql.Date takes (Day, Month, Year)
        valid_dates = set()
        for d in unique_dates:
            ql_date = ql.Date(d.day, d.month, d.year)
            if cal.isBusinessDay(ql_date):
                valid_dates.add(d)
        
        # Filter dataframe
        df_wide = df_wide[df_wide["ts"].dt.date.isin(valid_dates)]

    # 3. Time of Day Filter
    tz_local = getattr(cr, "CAL_TZ", "America/New_York")
    start_str, end_str = getattr(cr, "TRADING_HOURS", ("07:00", "17:00"))

    df_wide["ts_local"] = df_wide["ts"].dt.tz_localize("UTC").dt.tz_convert(tz_local)
    tmp = df_wide.set_index("ts_local").sort_index().between_time(start_str, end_str)
    tmp["ts"] = tmp.index.tz_convert("UTC").tz_localize(None)
    
    return tmp.reset_index(drop=True).drop(columns=["ts_local"], errors="ignore")


# -----------------------
# Cleaning & reshaping
# -----------------------
def _zeros_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    num = df.drop(columns=["ts"])
    num = num.apply(pd.to_numeric, errors="coerce").mask(num == 0)
    return df[["ts"]].join(num)

def _melt_long(df_wide: pd.DataFrame, tenormap: Dict[str, float]) -> pd.DataFrame:
    def norm(s): return " ".join(str(s).strip().replace("_mid","").split())
    tenormap_norm = {norm(k): v for k, v in tenormap.items()}
    cand = [c for c in df_wide.columns if c != "ts" and norm(c) in tenormap_norm]
    if not cand: return pd.DataFrame()

    use_cols = ["ts"] + cand
    df_sel = df_wide[use_cols].copy()
    col_to_tenor = {c: tenormap_norm[norm(c)] for c in cand}
    
    long = df_sel.melt(id_vars="ts", var_name="instrument", value_name="rate")
    long["tenor_yrs"] = long["instrument"].map(col_to_tenor).astype(float)
    long["rate"] = pd.to_numeric(long["rate"], errors="coerce")
    return long.dropna(subset=["ts", "tenor_yrs", "rate"])

def _decision_key(ts: pd.Series, freq: str) -> pd.Series:
    return ts.dt.floor("D") if freq == "D" else ts.dt.floor("H")

# -----------------------
# Spline & PCA Math
# -----------------------
def _spline_fit_safe(snap_long: pd.DataFrame) -> Tuple[pd.Series, float]:
    # Initialize output with NaNs
    out = pd.Series(np.nan, index=snap_long.index, dtype=float)
    DEFAULT_SCALE = 0.05 
    
    # 1. Filter: Only fit the Swap Curve (>= 2Y)
    # Exclude Money Markets (< 2Y) which distort the Cubic fit and blow up Scale
    s_all = snap_long[["tenor_yrs", "rate"]].dropna()
    s_fit = s_all[s_all["tenor_yrs"] >= 2.0].copy()

    # Need at least 5 points to fit a cubic + calculate meaningful scale stats
    if s_fit.shape[0] < 5: return out, DEFAULT_SCALE

    # 2. Sort by tenor (Critical for consistent arrays)
    s_fit = s_fit.sort_values("tenor_yrs")
    
    x = s_fit["tenor_yrs"].values.astype(float)
    y = s_fit["rate"].values.astype(float)
    
    # deg=3 is standard for yield curve fitting
    deg = 3 if len(x) >= 4 else min(2, len(x)-1)
    
    try:
        # Fit Polynomial
        coef = np.polyfit(x, y, deg=deg)
        fit = np.polyval(coef, x)
        resid = y - fit
        
        # Calculate Scale from the Swap Curve residuals only
        # Use MAD (Robust) to ignore outliers
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        scale = (1.4826 * mad) if mad > 0 else resid.std(ddof=1)
        
        # Safety check on scale
        if not np.isfinite(scale) or scale <= 1e-6: scale = DEFAULT_SCALE
        
        # Calculate Z-scores ONLY for the fitted range (>= 2Y)
        z = (resid - resid.mean()) / scale
        
        # Map back to original index using the tenor as key
        # (This handles the case where input index might not be sorted)
        m = {ten: val for ten, val in zip(x, z)}
        
        # Only rows >= 2Y get a z_spline value. Others stay NaN.
        out.loc[s_fit.index] = s_fit["tenor_yrs"].map(m).values
        
        return out, scale

    except Exception as e: 
        # On math failure, return NaNs and default scale
        return out, DEFAULT_SCALE

def _pca_fit_panel(panel_long: pd.DataFrame, cols_ordered: List[float], n_comps: int):
    if panel_long.empty: return None
    W = (panel_long.pivot(index="ts", columns="tenor_yrs", values="rate").sort_index())
    W = W.reindex(columns=cols_ordered).ffill().dropna(how="any")
    if W.shape[0] < (n_comps + 5) or W.shape[1] < n_comps: return None

    X = W.values.astype(float)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    comps = VT[:n_comps, :]
    evr = (S**2) / (S**2).sum()
    return {"cols": list(W.columns), "mean": mu.ravel(), "components": comps, "evr": evr[:n_comps]}

def _pca_apply_block(df_block: pd.DataFrame, pca_model: dict) -> pd.Series:
    out = pd.Series(index=df_block.index, dtype=float)
    if not pca_model or df_block.empty: return out

    cols, mu, comps = pca_model["cols"], np.asarray(pca_model["mean"]), np.asarray(pca_model["components"])
    
    # Block is already downsampled to HEAD(1), so no extra grouping needed strictly,
    # but we keep groupby/head(1) to be safe against mis-shaped inputs.
    snap = df_block.sort_values("ts").groupby("tenor_yrs", as_index=False).head(1).set_index("tenor_yrs")["rate"]
    if any(c not in snap.index for c in cols): return out

    x = snap.reindex(cols).values.astype(float)
    recon = comps.T @ (comps @ (x - mu))
    sd = recon.std()
    
    if not np.isfinite(sd) or sd == 0: return out
    z_std = (recon - recon.mean()) / sd
    return df_block["tenor_yrs"].map(dict(zip(cols, z_std)))

# -----------------------
# Per-bucket processor
# -----------------------
def _process_bucket(dts, df_bucket, df_history, lookback_days, pca_enable, pca_n_comps, yymm):
    # Bucket is already downsampled to HEAD(1), so this is efficient
    out = df_bucket.sort_values("ts").groupby("tenor_yrs", as_index=False).head(1).reset_index(drop=True)

    # 1) Spline Z + SCALE 
    z_spline, scale_val = _spline_fit_safe(out)
    out["z_spline"] = z_spline
    out["scale"] = scale_val 

    # 2) PCA Z
    out["z_pca"] = np.nan
    if pca_enable:
        cols_now = sorted(out["tenor_yrs"].unique().tolist())
        if len(cols_now) >= pca_n_comps:
            t_end = out["ts"].min() 
            t_start = t_end - pd.Timedelta(days=float(lookback_days))
            
            # History is Open prices. Target is Open price. Consistency achieved.
            panel = df_history[(df_history["ts"]>=t_start) & (df_history["ts"]<t_end) & 
                               (df_history["tenor_yrs"].isin(cols_now))]
            
            model = _pca_fit_panel(panel, cols_now, pca_n_comps)
            if model:
                out["z_pca"] = _pca_apply_block(out, model)

    # 3) Combine Robustly
    # If z_spline is NaN (<2Y), this falls back to z_pca.
    # If z_pca is NaN (rare), this falls back to z_spline.
    # If both exist, it averages them (0.5 / 0.5).
    out["z_comb"] = out[["z_spline", "z_pca"]].mean(axis=1)
    
    # If everything is NaN (rare), fill with 0
    out["z_comb"] = out["z_comb"].fillna(0.0)
    
    return out

# -----------------------
# SMART LOADER (OPEN PRICES)
# -----------------------
def load_history_downsampled(target_yymm: str) -> pd.DataFrame:
    """
    Loads previous N months. 
    Uses HEAD(1) (Opens) for history to match target execution logic.
    """
    path_data = Path(getattr(cr, "PATH_DATA", "."))
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    target_dt = datetime.datetime.strptime(target_yymm, "%y%m")
    
    lookback_days = float(getattr(cr, "PCA_LOOKBACK_DAYS", 126))
    history_months = math.ceil(lookback_days / 20.0) + 1
    freq = str(getattr(cr, "DECISION_FREQ", "D")).upper()
    tenormap = getattr(cr, "TENOR_YEARS", {})
    
    history_dfs = []
    print(f"[{_now()}] [HIST] Loading {history_months} months history (Cache={freq}_OPEN)...")
    
    for i in range(history_months, 0, -1):
        curr_dt = target_dt - relativedelta(months=i)
        curr_str = curr_dt.strftime("%y%m")
        
        raw_path = path_data / f"{curr_str}_features.parquet"
        cache_name = f"{curr_str}_summary_{freq}.parquet"
        cache_path = path_enh / cache_name
        
        # 1. Try Cache
        if cache_path.exists():
            try:
                df_long = pd.read_parquet(cache_path)
                history_dfs.append(df_long)
                continue 
            except Exception as e:
                print(f"[WARN] Cache corrupt {cache_name}, regenerating. Error: {e}")

        # 2. Cache Miss: Generate
        if raw_path.exists():
            try:
                print(f"   -> Cache Miss. Generating OPEN summary for {curr_str}...")
                
                df = pd.read_parquet(raw_path)
                df = _to_ts_index(df)
                df = _apply_calendar_and_hours(df) 
                df = _zeros_to_nan(df)
                
                if not df.empty:
                    bucket_key = _decision_key(df["ts"], freq)
                    df_sampled = df.sort_values("ts").groupby(bucket_key).head(1)
                    
                    df_long = _melt_long(df_sampled, tenormap)
                    
                    df_long.to_parquet(cache_path, index=False)
                    history_dfs.append(df_long)
                del df
            except Exception as e:
                print(f"[WARN] Failed history {curr_str}: {e}")
                
    if not history_dfs: return pd.DataFrame()
    return pd.concat(history_dfs).sort_values("ts").reset_index(drop=True)

# -----------------------
# Month builder (SINGLE PASS OPTIMIZED)
# -----------------------
def build_month(yymm: str) -> None:
    path_data = Path(getattr(cr, "PATH_DATA", "."))
    path_enh  = Path(getattr(cr, "PATH_ENH", "."))
    path_enh.mkdir(parents=True, exist_ok=True)

    # 1. Load Target (Raw)
    in_path = path_data / f"{yymm}_features.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing target file: {in_path}")

    print(f"[{_now()}] [TARGET] Loading {yymm}...")
    df_wide = pd.read_parquet(in_path)
    df_wide = _to_ts_index(df_wide)
    df_wide = _apply_calendar_and_hours(df_wide)
    df_wide = _zeros_to_nan(df_wide)

    tenormap = getattr(cr, "TENOR_YEARS", {})
    df_long = _melt_long(df_wide, tenormap)
    
    decision_freq = str(getattr(cr, "DECISION_FREQ", "D")).upper()
    df_long['decision_ts'] = _decision_key(df_long['ts'], decision_freq)
    
    cache_name = f"{yymm}_summary_{decision_freq}.parquet"
    cache_path = path_enh / cache_name
    bucket_key = _decision_key(df_wide['ts'], decision_freq)
    df_sampled = df_wide.sort_values("ts").groupby(bucket_key).head(1)
    df_cache = _melt_long(df_sampled, tenormap)
    df_cache.to_parquet(cache_path, index=False)
    print(f"[{_now()}] [CACHE] Saved self-summary: {cache_name}")

    buckets = (df_long["decision_ts"].dropna().unique().tolist())
    buckets.sort()

    # 2. Load History & Combine
    df_history = load_history_downsampled(yymm)

    # 3. Create PCA Context Panel
    df_target_daily = df_long.sort_values("ts").groupby(["decision_ts", "tenor_yrs"], as_index=False).head(1)
    df_context = pd.concat([df_history, df_target_daily]).sort_values("ts").reset_index(drop=True)

    # 4. Process
    N_JOBS = int(getattr(cr, "N_JOBS", 1))
    if isinstance(N_JOBS, int):
        if N_JOBS == 0:
            import multiprocessing as mp
            jobs = max(1, min((mp.cpu_count() // 2), 8))
        else:
            jobs = int(N_JOBS)
    else:
        jobs = 1
    
    pca_enable = bool(getattr(cr, "PCA_ENABLE", True))
    lookback_days = float(getattr(cr, "PCA_LOOKBACK_DAYS", 126))
    pca_components = int(getattr(cr, "PCA_COMPONENTS", 3))

    print(f"[{_now()}] [PROCESS] Processing {len(buckets)} buckets (Mode=MORNING_OPEN)...")

    def _one(dts):
        snap = df_long[df_long["decision_ts"] == dts]
        return _process_bucket(
            dts=dts,
            df_bucket=snap,
            df_history=df_context,
            lookback_days=lookback_days,
            pca_enable=pca_enable,
            pca_n_comps=pca_components,
            yymm=yymm
        )

    parts = Parallel(n_jobs=jobs, backend="loky")(delayed(_one)(d) for d in buckets)
    
    if not parts:
        out = pd.DataFrame(columns=['ts','tenor_yrs','rate','z_spline','z_pca','z_comb','scale'])
    else:
        out = pd.concat(parts, ignore_index=True).sort_values(['ts','tenor_yrs']).reset_index(drop=True)

    out_name = f"{yymm}_enh{getattr(cr, 'ENH_SUFFIX', '')}.parquet"
    out_path = path_enh / out_name
    out.to_parquet(out_path, index=False)
    
    print(f"[DONE] {yymm} rows:{len(out):,} -> {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_creation.py 2304")
        sys.exit(1)
    for m in sys.argv[1:]:
        build_month(m)
