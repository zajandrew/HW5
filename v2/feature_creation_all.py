"""
feature_creation.py

The Universal Feature Factory.
- Preserves original PCA/Spline/Hurst logic EXACTLY.
- Adds Physics (Carry/Roll) via math_core (Exact match to trading engine).
- Adds Regime (Rolling Stats) via vectorized post-processing.
- Adds Time-Aware Drift Accrual and Cumulative Sums for instant scanning.
"""

import os, sys, time
import datetime
import math
from pathlib import Path
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import UnivariateSpline
import QuantLib as ql

# --- IMPORTS ---
import config as cr
import math_core as mc  # <--- CRITICAL: Uses your Math Core

# ==============================================================================
# 1. UTILITIES & CLEANING
# ==============================================================================
def _now():
    return time.strftime("%H:%M:%S")

def _to_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" not in df.columns:
        if df.index.name in ("ts", "sec"):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "ts"})
        else:
            raise KeyError("No 'ts' column found.")
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["ts"] = df["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("ts")

def _apply_calendar_and_hours(df_wide: pd.DataFrame) -> pd.DataFrame:
    if df_wide.empty: return df_wide
    
    ts = pd.to_datetime(df_wide["ts"])
    df_wide = df_wide[ts.dt.weekday < 5].copy()
    if df_wide.empty: return df_wide
        
    if getattr(cr, "USE_QL_CALENDAR", False):
        try:
            market = str(getattr(cr, "QL_US_MARKET", "FederalReserve"))
            direct = getattr(ql.UnitedStates, market, None)
            cal = ql.UnitedStates(direct) if direct else ql.UnitedStates()
            
            unique_dates = df_wide["ts"].dt.date.unique()
            valid_dates = set()
            for d in unique_dates:
                ql_date = ql.Date(d.day, d.month, d.year)
                if cal.isBusinessDay(ql_date):
                    valid_dates.add(d)
            df_wide = df_wide[df_wide["ts"].dt.date.isin(valid_dates)]
        except: pass

    tz_local = getattr(cr, "CAL_TZ", "America/New_York")
    start_str, end_str = getattr(cr, "TRADING_HOURS", ("08:00", "17:00"))

    df_wide["ts_local"] = df_wide["ts"].dt.tz_localize("UTC").dt.tz_convert(tz_local)
    tmp = df_wide.set_index("ts_local").sort_index()
    tmp = tmp.between_time(start_str, end_str)
    
    tmp["ts"] = tmp.index.tz_convert("UTC").tz_localize(None)
    tmp = tmp.reset_index(drop=True)
    if "ts_local" in tmp.columns: tmp = tmp.drop(columns=["ts_local"])
        
    return tmp

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

def _make_decision_buckets(df_long: pd.DataFrame, freq: str, mode: str = 'head') -> pd.DataFrame:
    df = df_long.copy()
    if freq.upper() == 'H':
        df['decision_ts'] = df['ts'].dt.floor('h')
    elif freq.upper() == 'D':
        df['decision_ts'] = df['ts'].dt.floor('d')
    else:
        df['decision_ts'] = df['ts'].dt.floor('d')
    
    df = df.sort_values(['decision_ts', 'tenor_yrs', 'ts'])
    g = df.groupby(['decision_ts', 'tenor_yrs'], as_index=False)
    df_bucketed = g.tail(1) if mode == 'tail' else g.head(1)
    
    return df_bucketed.drop(columns=['ts']).rename(columns={'decision_ts': 'ts'})

# ==============================================================================
# 2. MATH LOGIC
# ==============================================================================

# [ORIGINAL SIGNAL LOGIC]
def _spline_fit_safe(snap_long: pd.DataFrame) -> Tuple[pd.Series, float]:
    """SIGNAL ENGINE: Fits a SMOOTHING spline (UnivariateSpline) to find 'Rich/Cheap' residuals."""
    out = pd.Series(np.nan, index=snap_long.index, dtype=float)
    DEFAULT_SCALE = 0.05 
    
    s_fit = snap_long[snap_long["tenor_yrs"] >= 0.25].dropna().sort_values("tenor_yrs")
    if s_fit.shape[0] < 4: return out, DEFAULT_SCALE

    x = s_fit["tenor_yrs"].values.astype(float)
    y = s_fit["rate"].values.astype(float)
    
    try:
        spl = UnivariateSpline(x, y, k=3, s=1e-2)
        fit = spl(x)
        resid = y - fit
        
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

# [NEW PHYSICS LOGIC - MATCHES MATH CORE]
def _calc_physics_features(snap_df: pd.DataFrame) -> pd.DataFrame:
    """PHYSICS ENGINE: Uses math_core.SplineCurve (Interpolation) for exact Carry/Roll."""
    if snap_df.empty: return pd.DataFrame()

    valid = snap_df[snap_df["tenor_yrs"] >= 0.25].dropna()
    if valid.shape[0] < 3: return pd.DataFrame()
    
    tenors = valid["tenor_yrs"].values
    rates = valid["rate"].values
    
    try:
        # 1. Instantiate Core Curve (Handles 4.25% correctly)
        curve = mc.SplineCurve(tenors, rates)
    except:
        return pd.DataFrame()
        
    results = []
    dt = 1.0 / 360.0
    funding = curve.get_funding_rate() # e.g. 4.00
    
    for t in tenors:
        r_t = curve.get_rate(t)          # e.g. 4.25
        r_rolled = curve.get_rate(t - dt) # e.g. 4.24
        
        # 2. Calculate Bps (Matches calc_signal_drift)
        # (Percent - Percent) * 100 = Basis Points
        carry = (r_t - funding) * 100.0 * dt 
        roll = (r_t - r_rolled) * 100.0      
        
        # 3. Get Risk Factor
        dv01 = curve.get_dv01(t) 
        
        results.append({
            "tenor_yrs": t,
            "carry_bps_day": carry,
            "roll_bps_day": roll,
            "total_drift_day": carry + roll,
            "dv01": dv01
        })
        
    return pd.DataFrame(results)

# ==============================================================================
# 3. PCA & CONTEXT LOGIC (ORIGINAL PRESERVED)
# ==============================================================================
def _pca_fit_panel_robust(panel_long: pd.DataFrame, cols_ordered: List[float], n_comps: int) -> Optional[Dict[str, Any]]:
    if panel_long.empty: return None
    W = (panel_long.pivot(index="ts", columns="tenor_yrs", values="rate").sort_index())
    W = W.reindex(columns=cols_ordered).ffill().dropna(how="any")
    
    X_levels = W.values.astype(float)
    if X_levels.shape[0] < (n_comps + 10): return None

    X_diffs = np.diff(X_levels, axis=0)
    mu = np.median(X_diffs, axis=0)
    q75, q25 = np.percentile(X_diffs, [75, 25], axis=0)
    sigma = 0.7413 * (q75 - q25)
    sigma[sigma < 1e-6] = 1.0 
    
    Z = (X_diffs - mu) / sigma
    U, S, VT = np.linalg.svd(Z, full_matrices=False)
    comps = VT[:n_comps, :] 
    
    for i in range(n_comps):
        if i == 0 and np.sum(comps[i]) < 0: comps[i] = -comps[i]
        elif i == 1 and comps[i][-1] < comps[i][0]: comps[i] = -comps[i]
        elif i == 2:
            mid = len(comps[i]) // 2
            if comps[i][mid] < 0: comps[i] = -comps[i]

    evr = (S**2) / (S**2).sum()
    factors = comps @ Z.T 
    recon_z = (comps.T @ factors).T
    resid_z = Z - recon_z 
    
    return {
        "cols": list(W.columns), 
        "mean_diff": mu, "sigma_diff": sigma, "components": comps, 
        "last_level": X_levels[-1, :], "hist_resid_z": resid_z 
    }

def _pca_apply_hybrid(df_hourly: pd.DataFrame, pca_model: dict) -> Tuple[pd.Series, float]:
    out = pd.Series(index=df_hourly.index, dtype=float)
    scale = np.nan
    if not pca_model or df_hourly.empty: return out, scale

    cols = pca_model["cols"]
    snap = df_hourly.set_index("tenor_yrs")["rate"].reindex(cols)
    if snap.isnull().any(): return out, scale 

    current_level = snap.values.astype(float)
    live_diff = current_level - pca_model["last_level"]
    live_z_input = (live_diff - pca_model["mean_diff"]) / pca_model["sigma_diff"]
    
    factors = pca_model["components"] @ live_z_input
    recon_z = pca_model["components"].T @ factors
    resid_z = live_z_input - recon_z
    
    med = np.median(resid_z)
    mad = np.median(np.abs(resid_z - med))
    scale = (1.4826 * mad) if mad > 0 else np.std(resid_z, ddof=1)
    if scale < 1e-4: scale = 0.01
    
    final_z = (resid_z - resid_z.mean()) / scale
    return df_hourly["tenor_yrs"].map(dict(zip(cols, final_z))), scale

def _calc_hurst_rs(series: np.ndarray, min_chunk: int = 8) -> float:
    series = np.array(series)
    N = len(series)
    if N < 100: return 0.5
    chunks = np.unique(np.linspace(min_chunk, N // 2, num=10).astype(int))
    rs_values = []
    for n in chunks:
        num = N // n
        tmp = series[:num * n].reshape(num, n)
        r = np.max(np.cumsum(tmp - np.mean(tmp, axis=1, keepdims=True), axis=1), axis=1) - \
            np.min(np.cumsum(tmp - np.mean(tmp, axis=1, keepdims=True), axis=1), axis=1)
        s = np.std(tmp, axis=1, ddof=1); s[s==0] = 1e-9
        rs_values.append(np.mean(r/s))
    try:
        H, _ = np.polyfit(np.log(chunks), np.log(rs_values), 1)
        return float(H)
    except: return 0.5

def _calc_ou_halflife(series: np.ndarray) -> Tuple[float, float]:
    if len(series) < 10: return (np.nan, 0.0)
    y, x = series[1:], series[:-1]
    num = np.sum((x - x.mean()) * (y - y.mean()))
    den = np.sum((x - x.mean())**2)
    if den == 0: return (np.nan, 0.0)
    beta = num / den
    if beta >= 0.999: return (999.0, 0.0)
    if beta <= 0.0: return (0.1, 0.0)
    return (-np.log(2) / np.log(beta), 0.0)

# ==============================================================================
# 4. ORCHESTRATORS
# ==============================================================================

def _process_instantaneous_bucket(dts, df_bucket, df_hist_daily, pca_config):
    """PHASE 1: Compute Instantaneous features (Snapshot only)."""
    out = df_bucket.copy()
    
    # 1. PCA
    pca_z, pca_scale = np.nan, np.nan
    hurst_map, hl_map = {}, {}
    if pca_config['enable']:
        cols = sorted(out["tenor_yrs"].unique().tolist())
        t_date = pd.to_datetime(dts).normalize()
        daily_window = df_hist_daily[df_hist_daily["ts"] < t_date]
        model = _pca_fit_panel_robust(daily_window, cols, pca_config['n_comps'])
        if model:
            pca_z, pca_scale = _pca_apply_hybrid(out, model)
            resid = model["hist_resid_z"]
            for i, ten in enumerate(model["cols"]):
                if i < resid.shape[1]:
                    hurst_map[ten] = _calc_hurst_rs(resid[:, i])
                    hl_map[ten] = _calc_ou_halflife(resid[:, i])[0]
    out["z_pca"] = pca_z
    
    # 2. Spline (Signal)
    z_spl, spl_sc = _spline_fit_safe(out)
    out["z_spline"] = z_spl
    
    # 3. Physics (Drift)
    df_phys = _calc_physics_features(out)
    if not df_phys.empty:
        out = out.merge(df_phys, on='tenor_yrs', how='left')
    else:
        out['total_drift_day'] = 0.0
        out['carry_bps_day'] = 0.0
        out['roll_bps_day'] = 0.0
        out['dv01'] = 0.0

    # 4. Combine
    if hurst_map: out["hurst"] = out["tenor_yrs"].map(hurst_map)
    else: out["hurst"] = 0.5
    if hl_map: out["halflife"] = out["tenor_yrs"].map(hl_map)
    else: out["halflife"] = 999.0
    
    sc = pca_scale if (np.isfinite(pca_scale) and pca_scale > 1e-6) else 0.01
    out["scale"] = sc
    out["z_comb"] = out[["z_pca", "z_spline"]].mean(axis=1).fillna(0.0)
    
    return out

def build_month(yymm: str) -> None:
    path_data = Path(getattr(cr, "PATH_DATA", "."))
    path_enh  = Path(getattr(cr, "PATH_ENH", "."))
    path_enh.mkdir(parents=True, exist_ok=True)
    
    # --- A. LOAD RAW ---
    in_path = path_data / f"{yymm}.parquet"
    if not in_path.exists(): raise FileNotFoundError(f"Missing {in_path}")
    print(f"[{_now()}] [TARGET] Loading {yymm}...")
    
    df_wide = pd.read_parquet(in_path)
    df_wide = _to_ts_index(df_wide)
    if "ts" not in df_wide.columns: df_wide = df_wide.reset_index()
    df_wide = _apply_calendar_and_hours(df_wide)
    df_wide = _zeros_to_nan(df_wide)
    df_long = _melt_long(df_wide, getattr(cr, "TENOR_YEARS", {}))
    
    # --- B. BUCKET ---
    df_hourly = _make_decision_buckets(df_long, 'H', mode='head')
    df_daily = _make_decision_buckets(df_long, 'D', mode='tail')
    
    # --- C. HISTORY ---
    df_hist_daily = pd.concat([pd.read_parquet(f) for f in path_enh.glob("*_summary_D.parquet")] or [df_daily], ignore_index=True)
    df_hist_daily = df_hist_daily.sort_values("ts").drop_duplicates(["ts", "tenor_yrs"])
    
    prev_dt = datetime.datetime.strptime(yymm, "%y%m") - relativedelta(months=1)
    prev_yymm = prev_dt.strftime("%y%m")
    path_prev_enh = path_enh / f"{prev_yymm}_enh{getattr(cr, 'ENH_SUFFIX', '')}.parquet"
    df_hist_hourly = pd.read_parquet(path_prev_enh) if path_prev_enh.exists() else pd.DataFrame()

    # --- D. PHASE 1: INSTANTANEOUS (Parallel) ---
    buckets = np.sort(df_hourly["ts"].unique())
    pca_cfg = {'enable': True, 'n_comps': 3}
    N_JOBS = int(getattr(cr, "N_JOBS", 1))
    jobs = max(1, min((os.cpu_count() // 2), 8)) if N_JOBS == 0 else N_JOBS
    
    print(f"[{_now()}] [PHASE 1] Calculating Physics/Signal on {len(buckets)} buckets...")
    def _one(dts):
        snap = df_hourly[df_hourly["ts"] == dts]
        return _process_instantaneous_bucket(dts, snap, df_hist_daily, pca_cfg)
    parts = Parallel(n_jobs=jobs, backend="loky")(delayed(_one)(d) for d in buckets)
    df_instant = pd.concat(parts, ignore_index=True).sort_values(['ts', 'tenor_yrs'])
    
    # --- E. PHASE 2: REGIME & CUMSUM DRIFT (Vectorized) ---
    print(f"[{_now()}] [PHASE 2] Calculating Rolling Regime Stats & Drift Accumulation...")
    
    # 1. Prep for Rolling
    target_cols = ['z_comb', 'z_pca', 'z_spline', 'carry_bps_day', 'roll_bps_day', 'total_drift_day', 'hurst', 'halflife']
    keep_cols = ['ts', 'tenor_yrs'] + [c for c in target_cols if c in df_instant.columns]
    
    hist_subset = pd.DataFrame()
    if not df_hist_hourly.empty:
        common = [c for c in keep_cols if c in df_hist_hourly.columns]
        hist_subset = df_hist_hourly[common]
        
    df_full = pd.concat([hist_subset, df_instant[keep_cols]], ignore_index=True).sort_values(['tenor_yrs', 'ts'])
    
    # 2. Vectorized Rolling (Slope/Accel/Std)
    windows = [12, 24]
    for col in target_cols:
        if col not in df_full.columns: continue
        for w in windows:
            shift_w = df_full.groupby('tenor_yrs')[col].shift(w)
            df_full[f"{col}_slope_{w}h"] = (df_full[col] - shift_w) / w
            
            shift_w_half = df_full.groupby('tenor_yrs')[f"{col}_slope_{w}h"].shift(w // 2)
            df_full[f"{col}_accel_{w}h"] = df_full[f"{col}_slope_{w}h"] - shift_w_half
            
            df_full[f"{col}_std_{w}h"] = df_full.groupby('tenor_yrs')[col].transform(lambda x: x.rolling(w).std())

    # --- F. TIME-AWARE DRIFT ACCRUAL ---
    # 1. Calc Time Elapsed (Handle Weekends/Overnight)
    df_full['hours_elapsed'] = df_full.groupby('tenor_yrs')['ts'].diff().dt.total_seconds() / 3600.0
    df_full['hours_elapsed'] = df_full['hours_elapsed'].fillna(1.0)
    
    # 2. Accrue & CumSum (For O(1) Scanning)
    for d_col in ['carry_bps_day', 'roll_bps_day', 'total_drift_day']:
        if d_col in df_full.columns:
            # Accrued = (Daily_Rate / 24) * Hours_Elapsed
            accrued_col = d_col.replace('_day', '_accrued')
            df_full[accrued_col] = (df_full[d_col] / 24.0) * df_full['hours_elapsed']
            
            # CumSum = Running Total (To scan via subtraction)
            cumsum_col = d_col.replace('_day', '_cumsum')
            df_full[cumsum_col] = df_full.groupby('tenor_yrs')[accrued_col].cumsum().fillna(0.0)

    # 3. Trim back to current month
    min_ts = df_instant['ts'].min()
    df_regime_only = df_full[df_full['ts'] >= min_ts].copy()
    
    # 4. Merge
    new_cols = [c for c in df_regime_only.columns if c not in df_instant.columns and c != 'tenor_yrs' and c != 'ts']
    if new_cols:
        df_final = df_instant.merge(df_regime_only[['ts', 'tenor_yrs'] + new_cols], on=['ts', 'tenor_yrs'], how='left')
    else:
        df_final = df_instant

    # --- G. SAVE ---
    out_name = f"{yymm}_enh{getattr(cr, 'ENH_SUFFIX', '')}.parquet"
    out_path = path_enh / out_name
    df_final.to_parquet(out_path, index=False)
    
    df_daily.to_parquet(path_enh / f"{yymm}_summary_D.parquet", index=False)
    print(f"[DONE] {yymm} -> {out_path} (Shape: {df_final.shape})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_creation.py 2304")
    for m in sys.argv[1:]:
        build_month(m)
