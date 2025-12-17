import os, sys, time
import datetime
import math
from pathlib import Path
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import UnivariateSpline

# Use module-level config access
import cr_config_new as cr

# -----------------------
# Utilities
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
# Calendar & Hours
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
    
    # 1. Standard Weekday Filter
    ts = pd.to_datetime(df_wide["ts"])
    df_wide = df_wide[ts.dt.weekday < 5].copy()
    if df_wide.empty: return df_wide
        
    cal = _get_ql_calendar()
    if cal:
        unique_dates = df_wide["ts"].dt.date.unique()
        valid_dates = set()
        for d in unique_dates:
            ql_date = ql.Date(d.day, d.month, d.year)
            if cal.isBusinessDay(ql_date):
                valid_dates.add(d)
        df_wide = df_wide[df_wide["ts"].dt.date.isin(valid_dates)]

    tz_local = getattr(cr, "CAL_TZ", "America/New_York")
    start_str, end_str = getattr(cr, "TRADING_HOURS", ("08:00", "17:00"))

    df_wide["ts_local"] = df_wide["ts"].dt.tz_localize("UTC").dt.tz_convert(tz_local)
    tmp = df_wide.set_index("ts_local").sort_index().between_time(start_str, end_str)
    tmp["ts"] = tmp.index.tz_convert("UTC").tz_localize(None)
    
    return tmp.reset_index(drop=True).drop(columns=["ts_local"], errors="ignore")

# -----------------------
# Cleaning & Reshaping
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

# -----------------------
# Sparse Bucketing Logic
# -----------------------
def _make_decision_buckets(df_long: pd.DataFrame, freq: str, mode: str = 'head') -> pd.DataFrame:
    """
    Groups irregular ticks into clean decision buckets.
    mode='head': Open of the bucket (for execution)
    mode='tail': Close of the bucket (for history)
    """
    df = df_long.copy()
    
    # 1. Floor the timestamp to the bucket resolution
    if freq.upper() == 'H':
        df['decision_ts'] = df['ts'].dt.floor('h')
    elif freq.upper() == 'D':
        df['decision_ts'] = df['ts'].dt.floor('d')
    else:
        df['decision_ts'] = df['ts'].dt.floor('d') # Default
    
    # 2. Sort to ensure we get true open/close
    df = df.sort_values(['decision_ts', 'tenor_yrs', 'ts'])
    
    # 3. Group by Bucket + Tenor
    # This ensures that if 2y trades at :05 and 10y at :20, both land in the bucket
    g = df.groupby(['decision_ts', 'tenor_yrs'], as_index=False)
    
    if mode == 'tail':
        df_bucketed = g.tail(1)
    else:
        df_bucketed = g.head(1)
    
    # Clean up
    return df_bucketed.drop(columns=['ts']).rename(columns={'decision_ts': 'ts'})

# -----------------------
# Math: Spline & PCA
# -----------------------
def _spline_fit_safe(snap_long: pd.DataFrame) -> Tuple[pd.Series, float]:
    out = pd.Series(np.nan, index=snap_long.index, dtype=float)
    DEFAULT_SCALE = 0.05 
    
    s_fit = snap_long[snap_long["tenor_yrs"] >= 2.0].dropna().sort_values("tenor_yrs")
    if s_fit.shape[0] < 5: return out, DEFAULT_SCALE

    x = s_fit["tenor_yrs"].values.astype(float)
    y = s_fit["rate"].values.astype(float)
    
    try:
        # s=1e-2 is approx 3bps smoothing noise for Percent units (4.25)
        spl = UnivariateSpline(x, y, k=3, s=1e-2)
        fit = spl(x)
        resid = y - fit
        
        # Robust MAD Scale
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        scale = (1.4826 * mad) if mad > 0 else resid.std(ddof=1)
        if scale < 1e-4: scale = 0.01
        
        z = (resid - resid.mean()) / scale
        m = {ten: val for ten, val in zip(x, z)}
        out.loc[s_fit.index] = s_fit["tenor_yrs"].map(m).values
        return out, scale
    except:
        return out, DEFAULT_SCALE

def _pca_fit_panel_robust(panel_long: pd.DataFrame, cols_ordered: List[float], n_comps: int):
    """
    Fits PCA on Daily History with SIGN FLIP CORRECTION.
    """
    if panel_long.empty: return None
    
    # Pivot to Matrix (Days x Tenors)
    W = (panel_long.pivot(index="ts", columns="tenor_yrs", values="rate").sort_index())
    W = W.reindex(columns=cols_ordered).ffill().dropna(how="any")
    if W.shape[0] < (n_comps + 5) or W.shape[1] < n_comps: return None

    X = W.values.astype(float)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    
    # SVD
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    comps = VT[:n_comps, :] # Shape (n_comps, n_tenors)
    
    # --- SIGN FLIP FIX ---
    for i in range(n_comps):
        # PC1 (Level): Ensure sum is Positive
        if i == 0:
            if np.sum(comps[i]) < 0:
                comps[i] = -comps[i]
        
        # PC2 (Slope): Ensure Long End (Last) > Short End (First)
        # Assumes cols_ordered is sorted (2y -> 30y)
        elif i == 1:
            if comps[i][-1] < comps[i][0]: 
                comps[i] = -comps[i]
        
        # PC3 (Curve/Fly): Ensure Belly (Middle) is Positive (Rich Belly = High Rate)
        elif i == 2:
            mid_idx = len(comps[i]) // 2
            if comps[i][mid_idx] < 0:
                comps[i] = -comps[i]

    evr = (S**2) / (S**2).sum()
    
    # Calculate Residual Scale (from History) for Z-scoring
    # We want to know the typical residual noise of the history to scale the live z-score
    # Reconstruct History
    weights = (Xc @ comps.T) # (Dates x Comps)
    recon = weights @ comps # (Dates x Tenors)
    resid_hist = Xc - recon
    
    # Global Residual Scale (Aggregated across tensor or per tenor? Usually per tenor or global)
    # Let's use a single global robust scale for simplicity, or we can return the matrix.
    # For now, we compute it during application to keep this function pure.
    
    return {
        "cols": list(W.columns), 
        "mean": mu.ravel(), 
        "components": comps, 
        "evr": evr[:n_comps]
    }

def _pca_apply_hybrid(df_hourly: pd.DataFrame, pca_model: dict) -> Tuple[pd.Series, float]:
    """
    Applies DAILY model to HOURLY data.
    """
    out = pd.Series(index=df_hourly.index, dtype=float)
    scale = np.nan
    
    if not pca_model or df_hourly.empty: return out, scale

    cols, mu, comps = pca_model["cols"], np.asarray(pca_model["mean"]), np.asarray(pca_model["components"])
    
    # Align Data
    snap = df_hourly.set_index("tenor_yrs")["rate"].reindex(cols)
    if snap.isnull().any(): return out, scale # Can't project if tenors missing

    x = snap.values.astype(float)
    
    # Project: (Rate - DailyMean) dot Components
    # weights represents the "Level", "Slope" value of this specific hour
    weights = comps @ (x - mu) 
    
    # Reconstruct
    recon = (comps.T @ weights) + mu
    
    # Residual (The Alpha)
    resid = x - recon
    
    # Calculate Scale (Robust MAD of this specific snapshot's residuals across tenors)
    # This tells us "How disjointed is the curve right now?"
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    scale = (1.4826 * mad) if mad > 0 else resid.std(ddof=1)
    
    # Normalize
    if scale < 1e-4: scale = 0.01
    z_scores = (resid - resid.mean()) / scale
    
    return df_hourly["tenor_yrs"].map(dict(zip(cols, z_scores))), scale

# -----------------------
# History Loading (Daily Close)
# -----------------------
def load_history_daily(target_yymm: str) -> pd.DataFrame:
    """
    Loads strictly DAILY CLOSES from history.
    Ensures 'ts' is a column, not an index.
    """
    path_data = Path(getattr(cr, "PATH_DATA", "."))
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    target_dt = datetime.datetime.strptime(target_yymm, "%y%m")
    
    lookback_days = float(getattr(cr, "PCA_LOOKBACK_DAYS", 126))
    history_months = math.ceil(lookback_days / 20.0) + 1
    tenormap = getattr(cr, "TENOR_YEARS", {})
    
    history_dfs = []
    print(f"[{_now()}] [HIST] Loading {history_months} months DAILY history...")
    
    for i in range(history_months, 0, -1):
        curr_dt = target_dt - relativedelta(months=i)
        curr_str = curr_dt.strftime("%y%m")
        
        # We look for the SUMMARY_D file (Daily Closes)
        cache_name = f"{curr_str}_summary_D.parquet"
        cache_path = path_enh / cache_name
        
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            history_dfs.append(df)
        else:
            # If Summary doesn't exist, build it from raw
            raw_path = path_data / f"{curr_str}_features.parquet"
            if raw_path.exists():
                try:
                    df = pd.read_parquet(raw_path)
                    
                    # --- SAFETY: Handle User's Custom Index Logic ---
                    df = _to_ts_index(df)
                    # If _to_ts_index put ts in the index, move it back to column
                    if "ts" not in df.columns and (df.index.name == "ts" or "ts" in df.index.names):
                        df = df.reset_index()
                    
                    df = _apply_calendar_and_hours(df) 
                    df = _zeros_to_nan(df)
                    df_long = _melt_long(df, tenormap)
                    
                    # Create Daily Close Summary
                    df_daily = _make_decision_buckets(df_long, 'D', mode='tail')
                    
                    # Save and Append
                    df_daily.to_parquet(cache_path, index=False)
                    history_dfs.append(df_daily)
                except Exception as e:
                    print(f"[WARN] Failed generating history {curr_str}: {e}")

        # Define the expected schema
    expected_cols = ["ts", "tenor_yrs", "rate"]
    
    if not history_dfs:
        # Return empty DF but WITH columns so filters don't crash
        print(f"[{_now()}] [WARN] No history found. PCA will be skipped for this month.")
        return pd.DataFrame(columns=expected_cols)
    
    df_final = pd.concat(history_dfs).sort_values("ts").reset_index(drop=True)
    
    # Safety: Ensure columns exist
    if "ts" not in df_final.columns and df_final.index.name == "ts":
        df_final = df_final.reset_index()
        
    return df_final

    
# -----------------------
# Main Processor
# -----------------------
def _process_hybrid_bucket(dts, df_bucket, df_history_daily, pca_config):
    # --- FIX: Ensure History has 'ts' column ---
    # This prevents the KeyError inside the worker
    if "ts" not in df_history_daily.columns and df_history_daily.index.name == "ts":
        df_history_daily = df_history_daily.reset_index()
    # -------------------------------------------

    # 1. Setup Container
    out = df_bucket.copy()
    
    # 2. Get History (Strictly < Bucket Date)
    t_date = pd.to_datetime(dts).normalize()
    
    # Now this line is safe because we guaranteed 'ts' is a column above
    hist_window = df_history_daily[df_history_daily["ts"] < t_date]
    
    pca_z = np.nan
    pca_scale = np.nan
    
    # 3. Fit & Apply PCA
    if pca_config['enable']:
        cols = sorted(out["tenor_yrs"].unique().tolist())
        # Pass hist_window (which definitely has 'ts' column now)
        model = _pca_fit_panel_robust(hist_window, cols, pca_config['n_comps'])
        
        if model:
            pca_z, pca_scale = _pca_apply_hybrid(out, model)
            out["z_pca"] = pca_z

    # 4. Spline (Intraday)
    z_spline, spline_scale = _spline_fit_safe(out)
    out["z_spline"] = z_spline
    
    # 5. Combine Logic
    raw_scale = 0.01
    if np.isfinite(pca_scale) and pca_scale > 1e-6:
        raw_scale = pca_scale
    elif np.isfinite(spline_scale):
        raw_scale = spline_scale
        
    out["scale"] = raw_scale
    out["z_comb"] = out[["z_pca", "z_spline"]].mean(axis=1).fillna(0.0)
    
    return out

def build_month(yymm: str) -> None:
    path_data = Path(getattr(cr, "PATH_DATA", "."))
    path_enh  = Path(getattr(cr, "PATH_ENH", "."))
    path_enh.mkdir(parents=True, exist_ok=True)

    # 1. Load Raw Data
    in_path = path_data / f"{yymm}_features.parquet"
    if not in_path.exists(): raise FileNotFoundError(f"Missing {in_path}")

    print(f"[{_now()}] [TARGET] Loading {yymm}...")
    df_wide = pd.read_parquet(in_path)
    
    # --- SAFETY: V3 Time Logic ---
    df_wide = _to_ts_index(df_wide)
    # Ensure ts is a column before applying calendar
    if "ts" not in df_wide.columns and df_wide.index.name == "ts":
        df_wide = df_wide.reset_index()

    df_wide = _apply_calendar_and_hours(df_wide)
    df_wide = _zeros_to_nan(df_wide)
    
    # Check if empty after filters
    if df_wide.empty:
        print(f"[{_now()}] [WARN] {yymm}: No data remaining after Calendar/Hours filter.")
        return

    df_long = _melt_long(df_wide, getattr(cr, "TENOR_YEARS", {}))
    if df_long.empty:
        print(f"[{_now()}] [WARN] {yymm}: Data empty after melt (check TENOR_YEARS config).")
        return
    
    # 2. Generate Files 1 & 2 (Hourly Rates & Daily Summary)
    
    # File 1: Hourly Rates (The Target)
    print(f"[{_now()}] [PREP] Bucketing Hourly Rates...")
    df_hourly = _make_decision_buckets(df_long, 'H', mode='head')
    path_hourly = path_enh / f"{yymm}_rates_H.parquet"
    df_hourly.to_parquet(path_hourly, index=False)
    
    # File 2: Daily Summary (The History for Next Month)
    print(f"[{_now()}] [PREP] Generating Daily Summary (Close)...")
    df_daily = _make_decision_buckets(df_long, 'D', mode='tail')
    path_daily = path_enh / f"{yymm}_summary_D.parquet"
    df_daily.to_parquet(path_daily, index=False)

    # 3. Load History & COMBINE (The Fix for Warm Start)
    # Load previous months
    df_past_history = load_history_daily(yymm)
    
    # COMBINE: Past Months + Current Month (So Far)
    # This ensures that on Day 15, we have Days 1-14 available for PCA
    if not df_past_history.empty:
        df_full_context = pd.concat([df_past_history, df_daily], ignore_index=True)
    else:
        df_full_context = df_daily.copy() # Cold start: Context is just this month

    # Ensure context is sorted and has no duplicates (just in case)
    df_full_context = df_full_context.drop_duplicates(subset=['ts', 'tenor_yrs']).sort_values('ts')

    # 4. Processing Context
    buckets = np.sort(df_hourly["ts"].unique())
    
    pca_cfg = {
        'enable': bool(getattr(cr, "PCA_ENABLE", True)),
        'n_comps': int(getattr(cr, "PCA_COMPONENTS", 3))
    }
    
    # 5. Parallel Execution
    N_JOBS = int(getattr(cr, "N_JOBS", 1))
    jobs = max(1, min((os.cpu_count() // 2), 8)) if N_JOBS == 0 else N_JOBS

    print(f"[{_now()}] [PROCESS] Hybrid PCA on {len(buckets)} buckets...")

    def _one(dts):
        snap = df_hourly[df_hourly["ts"] == dts]
        # Pass the FULL context (Past + Current Month)
        return _process_hybrid_bucket(dts, snap, df_full_context, pca_cfg)

    parts = Parallel(n_jobs=jobs, backend="loky")(delayed(_one)(d) for d in buckets)
    
    if parts:
        out = pd.concat(parts, ignore_index=True).sort_values(['ts','tenor_yrs'])
    else:
        out = pd.DataFrame()

    # File 3: Enhanced Output
    out_name = f"{yymm}_enh{getattr(cr, 'ENH_SUFFIX', '')}.parquet"
    out_path = path_enh / out_name
    
    if not out.empty and 'z_comb' in out.columns:
        out.to_parquet(out_path, index=False)
        zr = pd.to_numeric(out['z_comb'], errors='coerce')
        valid_pct = float(np.isfinite(zr).mean() * 100)
        print(f"[DONE] {yymm} -> {out_path} (Valid Z: {valid_pct:.1f}%)")
    else:
        print(f"[WARN] {yymm} produced EMPTY output.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_creation.py 2304")
        sys.exit(1)
    for m in sys.argv[1:]:
        build_month(m)
