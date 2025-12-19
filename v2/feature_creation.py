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
# Use module-level config access
import config as cr

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
        market = str(getattr(cr, "QL_US_MARKET", "FederalReserve"))
        direct = getattr(ql.UnitedStates, market, None)
        if direct is not None: return ql.UnitedStates(direct)
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
    tmp = df_wide.set_index("ts_local").sort_index()
    tmp = tmp.between_time(start_str, end_str)
    tmp["ts"] = tmp.index.tz_convert("UTC").tz_localize(None)
    tmp = tmp.reset_index(drop=True)
    if "ts_local" in tmp.columns: tmp = tmp.drop(columns=["ts_local"])                                             
    return tmp

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
    g = df.groupby(['decision_ts', 'tenor_yrs'], as_index=False)
    
    if mode == 'tail':
        df_bucketed = g.tail(1)
    else:
        df_bucketed = g.head(1)
    
    # Clean up
    return df_bucketed.drop(columns=['ts']).rename(columns={'decision_ts': 'ts'})

# -----------------------
# Math: Hurst, Spline & PCA
# -----------------------

def _calc_hurst_rs(series: np.ndarray, min_chunk: int = 8) -> float:
    """
    Calculates Hurst Exponent using Rescaled Range (R/S) analysis.
    H < 0.5: Mean Reverting (Butterfly)
    H > 0.5: Trending (Momentum)
    """
    series = np.array(series)
    N = len(series)
    if N < 100: return 0.5 # Not enough history for structural signal
    
    max_chunk = N // 2
    if max_chunk < min_chunk: return 0.5
    
    # Log-spaced chunks
    chunks = np.unique(np.linspace(min_chunk, max_chunk, num=10).astype(int))
    rs_values = []
    
    for n in chunks:
        num_splits = N // n
        # Crop to perfectly divisible
        tmp = series[:num_splits * n].reshape(num_splits, n)
        
        # R/S Calculation
        means = np.mean(tmp, axis=1, keepdims=True)
        y = tmp - means
        z = np.cumsum(y, axis=1)
        r = np.max(z, axis=1) - np.min(z, axis=1)
        s = np.std(tmp, axis=1, ddof=1)
        s[s == 0] = 1e-9 # Protect div/0
        
        rs = np.mean(r / s)
        rs_values.append(rs)
        
    # Regression: log(R/S) ~ H * log(n)
    try:
        y_reg = np.log(rs_values)
        x_reg = np.log(chunks)
        H, _ = np.polyfit(x_reg, y_reg, 1)
        return float(H)
    except:
        return 0.5

def _calc_ou_halflife(series: np.ndarray) -> Tuple[float, float]:
    """
    Fits an Ornstein-Uhlenbeck process to the residual series.
    dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    
    Discrete form: x_t = alpha + beta * x_{t-1} + epsilon
    Half-Life = -ln(2) / ln(beta)
    
    Returns: (HalfLife_Days, R_Squared)
    """
    if len(series) < 10: return (np.nan, 0.0)
    
    # Lag the series: y = x_t, x = x_{t-1}
    y = series[1:]
    x = series[:-1]
    
    # Linear Regression
    # We want slope (beta)
    # center data for stability
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    
    if denominator == 0: return (np.nan, 0.0)
    
    beta = numerator / denominator
    alpha = y_mean - beta * x_mean
    
    # Calculate R2
    y_pred = alpha + beta * x
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y_mean)**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Calculate Half Life
    # If beta >= 1.0, it's non-stationary (Random Walk or Trending), Infinite HL
    if beta >= 0.999: 
        return (999.0, r2)
    if beta <= 0.0: # Oscillation
        return (0.1, r2)
        
    # HL = -ln(2) / ln(beta) * dt (assuming dt=1 day)
    hl = -np.log(2) / np.log(beta)
    return (float(hl), float(r2))

def _spline_fit_safe(snap_long: pd.DataFrame) -> Tuple[pd.Series, float]:
    out = pd.Series(np.nan, index=snap_long.index, dtype=float)
    DEFAULT_SCALE = 0.05 
    
    s_fit = snap_long[snap_long["tenor_yrs"] >= 2.0].dropna().sort_values("tenor_yrs")
    if s_fit.shape[0] < 5: return out, DEFAULT_SCALE

    x = s_fit["tenor_yrs"].values.astype(float)
    y = s_fit["rate"].values.astype(float)
    
    try:
        # s=1e-2 preserves local microstructure ("wiggles") for RV trading
        spl = UnivariateSpline(x, y, k=3, s=1e-2)
        fit = spl(x)
        resid = y - fit
        
        # Robust MAD Scale
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        scale = (1.4826 * mad) if mad > 0 else resid.std(ddof=1)
        if scale < 1e-4: scale = 0.01
        
        # Z-score
        z = (resid - resid.mean()) / scale
        m = {ten: val for ten, val in zip(x, z)}
        out.loc[s_fit.index] = s_fit["tenor_yrs"].map(m).values
        return out, scale
    except:
        return out, DEFAULT_SCALE

def _pca_fit_panel_robust(panel_long: pd.DataFrame, cols_ordered: List[float], n_comps: int) -> Optional[Dict[str, Any]]:
    """
    Fits PCA on DIFFERENCED data (Yield Changes), consistent with RV literature.
    Returns model + Historical Residuals for Hurst calculation.
    """
    if panel_long.empty: return None
    
    # 1. Pivot to Matrix (Days x Tenors)
    W = (panel_long.pivot(index="ts", columns="tenor_yrs", values="rate").sort_index())
    W = W.reindex(columns=cols_ordered).ffill().dropna(how="any")
    
    X_levels = W.values.astype(float)
    if X_levels.shape[0] < (n_comps + 10): return None

    # 2. Compute Changes (Diffs) -> Stationarity
    # We lose the first row
    X_diffs = np.diff(X_levels, axis=0)
    
    # 3. Robust Scaling (Median/IQR) -> Correlation-like Matrix
    mu = np.median(X_diffs, axis=0)
    q75, q25 = np.percentile(X_diffs, [75, 25], axis=0)
    sigma = 0.7413 * (q75 - q25)
    sigma[sigma < 1e-6] = 1.0 # Protect div/0
    
    # Z-scored changes
    Z = (X_diffs - mu) / sigma
    
    # 4. SVD
    U, S, VT = np.linalg.svd(Z, full_matrices=False)
    comps = VT[:n_comps, :] # (n_comps, n_tenors)
    
    # --- SIGN FLIP FIX ---
    for i in range(n_comps):
        if i == 0: # PC1 Level: Sum Positive
            if np.sum(comps[i]) < 0: comps[i] = -comps[i]
        elif i == 1: # PC2 Slope: Long > Short
            if comps[i][-1] < comps[i][0]: comps[i] = -comps[i]
        elif i == 2: # PC3 Curve: Belly Positive
            mid_idx = len(comps[i]) // 2
            if comps[i][mid_idx] < 0: comps[i] = -comps[i]

    evr = (S**2) / (S**2).sum()
    
    # 5. Calculate Historical Residuals (for Hurst)
    # Project Z onto Comps
    factors = comps @ Z.T # (comps x days)
    recon_z = (comps.T @ factors).T # (days x tenors)
    resid_z = Z - recon_z # These are the historical Z-score errors
    
    return {
        "cols": list(W.columns), 
        "mean_diff": mu,
        "sigma_diff": sigma,
        "components": comps, 
        "evr": evr[:n_comps],
        "last_level": X_levels[-1, :], # Needed to calc live shock
        "hist_resid_z": resid_z # Needed for Hurst
    }

def _pca_apply_hybrid(df_hourly: pd.DataFrame, pca_model: dict) -> Tuple[pd.Series, float]:
    """
    Applies PCA by calculating the LIVE SHOCK (Current - Prev Close).
    """
    out = pd.Series(index=df_hourly.index, dtype=float)
    scale = np.nan
    
    if not pca_model or df_hourly.empty: return out, scale

    cols = pca_model["cols"]
    mu_diff = pca_model["mean_diff"]
    sigma_diff = pca_model["sigma_diff"]
    comps = pca_model["components"]
    last_level = pca_model["last_level"] # Vector of yesterday's close rates
    
    # Align Data
    snap = df_hourly.set_index("tenor_yrs")["rate"].reindex(cols)
    if snap.isnull().any(): return out, scale # Can't project if tenors missing

    current_level = snap.values.astype(float)
    
    # 1. Calculate Live Shock (The "Move" of the day)
    live_diff = current_level - last_level
    
    # 2. Standardize using Historical Vol
    live_z_input = (live_diff - mu_diff) / sigma_diff
    
    # 3. Project & Reconstruct
    factors = comps @ live_z_input
    recon_z_move = comps.T @ factors
    
    # 4. Residual (Alpha)
    resid_z = live_z_input - recon_z_move
    
    # 5. Scale Logic (Robust MAD of the residual vector)
    # We convert back to bps space for intuitive scaling, or keep in Z space.
    # Let's keep Z space consistency with Spline.
    # How noisy is this specific curve snapshot?
    med = np.median(resid_z)
    mad = np.median(np.abs(resid_z - med))
    scale = (1.4826 * mad) if mad > 0 else np.std(resid_z, ddof=1)
    
    if scale < 1e-4: scale = 0.01
    
    # Final Z-Score (Residual normalized by cross-sectional noise)
    final_z = (resid_z - resid_z.mean()) / scale
    
    return df_hourly["tenor_yrs"].map(dict(zip(cols, final_z))), scale

# -----------------------
# History Loading (Daily Close)
# -----------------------
def load_history_daily(target_yymm: str) -> pd.DataFrame:
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
        
        cache_name = f"{curr_str}_summary_D.parquet"
        cache_path = path_enh / cache_name
        
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            history_dfs.append(df)
        else:
            raw_path = path_data / f"{curr_str}.parquet"
            if raw_path.exists():
                try:
                    df = pd.read_parquet(raw_path)
                    df = _to_ts_index(df)
                    if "ts" not in df.columns and (df.index.name == "ts" or "ts" in df.index.names):
                        df = df.reset_index()
                    
                    df = _apply_calendar_and_hours(df) 
                    df = _zeros_to_nan(df)
                    df_long = _melt_long(df, tenormap)
                    df_daily = _make_decision_buckets(df_long, 'D', mode='tail')
                    
                    df_daily.to_parquet(cache_path, index=False)
                    history_dfs.append(df_daily)
                except Exception as e:
                    print(f"[WARN] Failed generating history {curr_str}: {e}")

    expected_cols = ["ts", "tenor_yrs", "rate"]
    
    if not history_dfs:
        print(f"[{_now()}] [WARN] No history found. PCA will be skipped.")
        return pd.DataFrame(columns=expected_cols)
    
    df_final = pd.concat(history_dfs).sort_values("ts").reset_index(drop=True)
    if "ts" not in df_final.columns and df_final.index.name == "ts":
        df_final = df_final.reset_index()
        
    return df_final

    
# -----------------------
# Main Processor
# -----------------------
def _process_hybrid_bucket(dts, df_bucket, df_history_daily, pca_config):
    # Ensure History has 'ts' column
    if "ts" not in df_history_daily.columns and df_history_daily.index.name == "ts":
        df_history_daily = df_history_daily.reset_index()

    out = df_bucket.copy()
    
    # 1. Get History (Strictly < Bucket Date)
    t_date = pd.to_datetime(dts).normalize()
    hist_window = df_history_daily[df_history_daily["ts"] < t_date]
    
    pca_z = np.nan
    pca_scale = np.nan
    hurst_map = {}
    halflife_map = {}
    
    # 2. Fit PCA & Calc Hurst
    out["z_pca"] = pca_z
    if pca_config['enable']:
        cols = sorted(out["tenor_yrs"].unique().tolist())
        # This now returns the model AND the historical residuals
        model = _pca_fit_panel_robust(hist_window, cols, pca_config['n_comps'])
        
        if model:
            # A. Live PCA Signal
            pca_z, pca_scale = _pca_apply_hybrid(out, model)
            out["z_pca"] = pca_z
            
            # B. Hurst Regime Calculation
            # We have historical residuals in model["hist_resid_z"]
            # Shape: (Days, Tenors)
            resid_hist = model["hist_resid_z"]
            
            for i, tenor in enumerate(model["cols"]):
                if i < resid_hist.shape[1]:
                    # Calculate H on the residuals of this specific tenor
                    h_val = _calc_hurst_rs(resid_hist[:, i])
                    hurst_map[tenor] = h_val
                    hl, r2 = _calc_ou_halflife(resid_hist[:, i])
                    halflife_map[tenor] = hl

    # 3. Spline (Intraday)
    z_spline, spline_scale = _spline_fit_safe(out)
    out["z_spline"] = z_spline
    
    # 4. Map Hurst
    if hurst_map:
        out["hurst"] = out["tenor_yrs"].map(hurst_map)
    else:
        out["hurst"] = 0.5

    if halflife_map:
        out["halflife"] = out["tenor_yrs"].map(halflife_map)
    else:
        out["halflife"] = 999.0 # Default to "Zombie" if calc fails
    
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
    in_path = path_data / f"{yymm}.parquet"
    if not in_path.exists(): raise FileNotFoundError(f"Missing {in_path}")

    print(f"[{_now()}] [TARGET] Loading {yymm}...")
    df_wide = pd.read_parquet(in_path)
    
    df_wide = _to_ts_index(df_wide)
    if "ts" not in df_wide.columns and df_wide.index.name == "ts":
        df_wide = df_wide.reset_index()

    df_wide = _apply_calendar_and_hours(df_wide)
    df_wide = _zeros_to_nan(df_wide)
    
    if df_wide.empty:
        print(f"[{_now()}] [WARN] {yymm}: No data after filters.")
        return

    df_long = _melt_long(df_wide, getattr(cr, "TENOR_YEARS", {}))
    if df_long.empty:
        print(f"[{_now()}] [WARN] {yymm}: Data empty after melt.")
        return
    
    # 2. Generate Files
    print(f"[{_now()}] [PREP] Bucketing Hourly Rates...")
    df_hourly = _make_decision_buckets(df_long, 'H', mode='head')
    df_hourly.to_parquet(path_enh / f"{yymm}_rates_H.parquet", index=False)
    
    print(f"[{_now()}] [PREP] Generating Daily Summary...")
    df_daily = _make_decision_buckets(df_long, 'D', mode='tail')
    df_daily.to_parquet(path_enh / f"{yymm}_summary_D.parquet", index=False)

    # 3. Load Context
    df_past_history = load_history_daily(yymm)
    if not df_past_history.empty:
        df_full_context = pd.concat([df_past_history, df_daily], ignore_index=True)
    else:
        df_full_context = df_daily.copy()

    df_full_context = df_full_context.drop_duplicates(subset=['ts', 'tenor_yrs']).sort_values('ts')
    buckets = np.sort(df_hourly["ts"].unique())
    
    pca_cfg = {
        'enable': bool(getattr(cr, "PCA_ENABLE", True)),
        'n_comps': int(getattr(cr, "PCA_COMPONENTS", 3))
    }
    
    # 4. Parallel Execution
    N_JOBS = int(getattr(cr, "N_JOBS", 1))
    jobs = max(1, min((os.cpu_count() // 2), 8)) if N_JOBS == 0 else N_JOBS

    print(f"[{_now()}] [PROCESS] Hybrid PCA + Hurst on {len(buckets)} buckets...")

    def _one(dts):
        snap = df_hourly[df_hourly["ts"] == dts]
        return _process_hybrid_bucket(dts, snap, df_full_context, pca_cfg)

    parts = Parallel(n_jobs=jobs, backend="loky")(delayed(_one)(d) for d in buckets)
    
    if parts:
        out = pd.concat(parts, ignore_index=True).sort_values(['ts','tenor_yrs'])
    else:
        out = pd.DataFrame()

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
