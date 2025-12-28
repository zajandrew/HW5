"""
feature_creation.py

The XGBoost Alpha Engine: Feature Factory (v2.0)

Role:
1. Ingests Raw Hourly Rates.
2. Calculates Physics (Drift/DV01) via math_core.
3. Calculates Signals (PCA/Spline/Hurst).
4. Ingests Exogenous Context (Daily Volatility, Auction Schedules).
5. Applies "Kitchen Sink" Rolling Statistics (Slope, Accel, Z-Local, Ranges) to all key metrics.
6. Solves the "Weekend Problem" via Time-Aware Drift Accrual.

Outputs: _enh.parquet files ready for the Strategy Matrix.
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

# --- CUSTOM IMPORTS ---
import config as cr
import math_core as mc  # Physics Core

# ==============================================================================
# 1. UTILITIES & CLEANING
# ==============================================================================

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

def _get_ql_calendar():
    if not getattr(cr, "USE_QL_CALENDAR", False): 
        return None
    try:
        market = str(getattr(cr, "QL_US_MARKET", "FederalReserve"))
        direct = getattr(ql.UnitedStates, market, None)
        if direct is not None: return ql.UnitedStates(direct)
        return ql.UnitedStates()
    except: 
        return None

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
    """
    freq='H' -> Hourly Buckets (for Execution/Feature Creation)
    freq='D' -> Daily Buckets (for Historical Context/PCA Training)
    """
    df = df_long.copy()
    if freq.upper() == 'H':
        df['decision_ts'] = df['ts'].dt.floor('h')
    elif freq.upper() == 'D':
        df['decision_ts'] = df['ts'].dt.floor('d')
    else:
        df['decision_ts'] = df['ts'].dt.floor('d')
    
    df = df.sort_values(['decision_ts', 'tenor_yrs', 'ts'])
    g = df.groupby(['decision_ts', 'tenor_yrs'], as_index=False)
    
    if mode == 'tail':
        df_bucketed = g.tail(1)
    else:
        df_bucketed = g.head(1)
    
    return df_bucketed.drop(columns=['ts']).rename(columns={'decision_ts': 'ts'})

# ==============================================================================
# 2. THE KITCHEN SINK ENGINE (Shared Statistics)
# ==============================================================================

def _calc_kitchen_sink_stats(df: pd.DataFrame, col: str, windows: List[int], group_col: str = 'tenor_yrs') -> pd.DataFrame:
    """
    Applies the "Kitchen Sink" rolling stats to any feature column.
    Generates: Slope, Accel, Mean, Std, Max, Min, MaxAbs, ZLocal, RangePos, Quantiles.
    """
    # Create a holder for results to avoid fragmentation
    res = pd.DataFrame(index=df.index)
    
    # Pre-calculate GroupBy object for speed
    grp = df.groupby(group_col)[col]
    
    for w in windows:
        # 'd' suffix for Daily Vol, 'b' suffix for Hourly Buckets
        suffix = f"{w}d" if "exog" in col else f"{w}b" 
        
        # --- A. KINETICS ---
        shift_w = grp.shift(w)
        
        # Slope: (Current - Old) / Window
        col_slope = f"{col}_slope_{suffix}"
        res[col_slope] = (df[col] - shift_w) / w
        
        # Accel: Slope - Slope_Old (Half window shift)
        slope_series = (df[col] - shift_w) / w
        shift_w_half = slope_series.groupby(df[group_col]).shift(w // 2)
        res[f"{col}_accel_{suffix}"] = slope_series - shift_w_half

        # --- B. DISTRIBUTION ---
        roll = grp.rolling(w)
        
        mean_val = roll.mean()
        std_val  = roll.std()
        max_val  = roll.max()
        min_val  = roll.min()
        
        res[f"{col}_mean_{suffix}"] = mean_val.values
        res[f"{col}_std_{suffix}"]  = std_val.values
        res[f"{col}_max_{suffix}"]  = max_val.values
        res[f"{col}_min_{suffix}"]  = min_val.values
        
        # --- C. DERIVED METRICS ---
        
        # 1. Max Abs (Stress Detector: How far from 0 did we get?)
        res[f"{col}_max_abs_{suffix}"] = np.maximum(np.abs(max_val), np.abs(min_val))
        
        # 2. Local Z-Score (Regime Normalization)
        # (Current - RollingMean) / RollingStd
        res[f"{col}_zlocal_{suffix}"] = (df[col] - mean_val) / (std_val + 1e-8)
        
        # 3. Range Position (Stochastic Oscillator 0-1)
        rng = max_val - min_val
        res[f"{col}_rng_pos_{suffix}"] = (df[col] - min_val) / (rng + 1e-8)

        # --- D. QUANTILES ---
        # Critical for defining regimes (e.g., Top Quartile Volatility)
        res[f"{col}_q25_{suffix}"] = roll.quantile(0.25).values
        res[f"{col}_q75_{suffix}"] = roll.quantile(0.75).values
        
    return res

# ==============================================================================
# 3. EXOGENOUS PROCESSORS (Vol & Auctions)
# ==============================================================================

def _process_daily_vol_stats(vol_path: Path, tenor_map_rev: Dict[str, float]) -> pd.DataFrame:
    """
    Loads Daily Vol CSV, Maps Tickers, Calcs Kitchen Sink Stats.
    """
    if not vol_path.exists():
        print(f"[{_now()}] [WARN] Vol file not found: {vol_path}. Skipping Vol features.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(vol_path)
    except Exception as e:
        print(f"[{_now()}] [ERR] Failed reading Vol CSV: {e}")
        return pd.DataFrame()
    
    # 1. Parse Timestamp (file_modified is usually 6pm NY time in UTC)
    if 'file_modified' not in df.columns:
        # Fallback if manual file
        if 'Date' in df.columns: df['ts_vol'] = pd.to_datetime(df['Date'], utc=True)
        else: return pd.DataFrame()
    else:
        df['ts_vol'] = pd.to_datetime(df['file_modified'], utc=True)
    
    # 2. Map Tickers to Floats
    df['tenor_yrs'] = df['ticker'].map(tenor_map_rev)
    df = df.dropna(subset=['tenor_yrs']).sort_values(['tenor_yrs', 'ts_vol'])
    
    # 3. Rename columns
    # Assumes CSV has 'Implied Vol' and 'Skew'
    if 'Implied Vol' in df.columns: df = df.rename(columns={'Implied Vol': 'vol_implied'})
    if 'Skew' in df.columns: df = df.rename(columns={'Skew': 'vol_skew'})
    
    # 4. Calculate Kitchen Sink Stats (Daily Windows)
    # 5d (Week), 20d (Month), 60d (Quarter)
    vol_windows = [5, 20, 60] 
    
    features = []
    # Pass raw levels through
    base_cols = ['ts_vol', 'tenor_yrs', 'vol_implied', 'vol_skew']
    features.append(df[[c for c in base_cols if c in df.columns]])
    
    for col in ['vol_implied', 'vol_skew']:
        if col in df.columns:
            stats = _calc_kitchen_sink_stats(df, col, vol_windows, group_col='tenor_yrs')
            features.append(stats)
        
    df_processed = pd.concat(features, axis=1)
    
    # 5. Add Prefix to all feature columns (except keys)
    keys = ['ts_vol', 'tenor_yrs']
    new_cols = {c: f"exog_{c}" for c in df_processed.columns if c not in keys}
    df_processed = df_processed.rename(columns=new_cols)
    
    return df_processed.sort_values('ts_vol')

def _enrich_with_auctions(df_hourly: pd.DataFrame, auc_path: Path, tenor_map_rev: Dict[str, float]) -> pd.DataFrame:
    """
    Calculates 'hours_to_auction' using merge_asof forward.
    """
    if not auc_path.exists() or df_hourly.empty:
        df_hourly['hours_to_auction'] = 999.0
        return df_hourly
        
    try:
        df_auc = pd.read_csv(auc_path)
    except:
        df_hourly['hours_to_auction'] = 999.0
        return df_hourly

    # Ensure correct column names
    ts_col = 'Date Time' if 'Date Time' in df_auc.columns else 'timestamp'
    type_col = 'event_type'
    
    if ts_col not in df_auc.columns: 
        return df_hourly

    df_auc[ts_col] = pd.to_datetime(df_auc[ts_col], utc=True)
    
    # Map auction tickers to floats
    df_auc['tenor_yrs'] = df_auc[type_col].map(tenor_map_rev)
    df_auc = df_auc.dropna(subset=['tenor_yrs']).sort_values(ts_col)
    
    df_hourly = df_hourly.sort_values('ts')
    
    # Use merge_asof with 'by' (Exact match on tenor, Nearest Forward match on Time)
    merged = pd.merge_asof(
        df_hourly,
        df_auc[[ts_col, 'tenor_yrs']], 
        left_on='ts',
        right_on=ts_col,
        by='tenor_yrs',
        direction='forward',
        tolerance=pd.Timedelta(days=30),
        suffixes=('', '_auc')
    )
    
    # Calculate hours
    diff = (merged[ts_col] - merged['ts']).dt.total_seconds() / 3600.0
    df_hourly['hours_to_auction'] = diff.fillna(999.0).values
    
    return df_hourly

# ==============================================================================
# 4. MATH LOGIC: PHYSICS (Core), PCA, SPLINE, HURST
# ==============================================================================

def _calc_physics_features(snap_df: pd.DataFrame) -> pd.DataFrame:
    """
    PHYSICS ENGINE: Uses math_core.SplineCurve to calculate Drift, Carry, Roll, DV01.
    """
    if snap_df.empty: return pd.DataFrame()

    valid = snap_df[snap_df["tenor_yrs"] >= 0.25].dropna()
    if valid.shape[0] < 3: return pd.DataFrame()
    
    tenors = valid["tenor_yrs"].values
    rates = valid["rate"].values
    
    try:
        curve = mc.SplineCurve(tenors, rates)
    except:
        return pd.DataFrame()
        
    results = []
    dt = 1.0 / 360.0
    funding = curve.get_funding_rate()
    
    for t in tenors:
        r_t = curve.get_rate(t)
        r_rolled = curve.get_rate(t - dt)
        
        # Bps per Day
        carry = (r_t - funding) * 100.0 * dt 
        roll = (r_t - r_rolled) * 100.0      
        dv01 = curve.get_dv01(t) 
        
        results.append({
            "tenor_yrs": t,
            "carry_bps_day": carry,
            "roll_bps_day": roll,
            "total_drift_day": carry + roll,
            "dv01": dv01
        })
        
    return pd.DataFrame(results)

def _spline_fit_safe(snap_long: pd.DataFrame) -> Tuple[pd.Series, float]:
    """
    SIGNAL ENGINE: Fits a SMOOTHING spline (UnivariateSpline).
    Returns Residuals (Z-Scores) and Scale.
    """
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

def _calc_hurst_rs(series: np.ndarray, min_chunk: int = 8) -> float:
    """
    Hurst Exponent (R/S Analysis).
    H < 0.5: Mean Reverting. H > 0.5: Trending.
    """
    series = np.array(series)
    N = len(series)
    if N < 100: return 0.5 
    
    max_chunk = N // 2
    if max_chunk < min_chunk: return 0.5
    
    chunks = np.unique(np.linspace(min_chunk, max_chunk, num=10).astype(int))
    rs_values = []
    
    for n in chunks:
        num_splits = N // n
        tmp = series[:num_splits * n].reshape(num_splits, n)
        
        means = np.mean(tmp, axis=1, keepdims=True)
        y = tmp - means
        z = np.cumsum(y, axis=1)
        r = np.max(z, axis=1) - np.min(z, axis=1)
        s = np.std(tmp, axis=1, ddof=1)
        s[s == 0] = 1e-9 
        
        rs = np.mean(r / s)
        rs_values.append(rs)
        
    try:
        y_reg = np.log(rs_values)
        x_reg = np.log(chunks)
        H, _ = np.polyfit(x_reg, y_reg, 1)
        return float(H)
    except:
        return 0.5

def _calc_ou_halflife(series: np.ndarray) -> Tuple[float, float]:
    """
    Ornstein-Uhlenbeck Half-Life.
    Returns (HalfLife_Days, R_Squared).
    """
    if len(series) < 10: return (np.nan, 0.0)
    
    y = series[1:]
    x = series[:-1]
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    
    if denominator == 0: return (np.nan, 0.0)
    
    beta = numerator / denominator
    alpha = y_mean - beta * x_mean
    
    y_pred = alpha + beta * x
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y_mean)**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    if beta >= 0.999: return (999.0, r2)
    if beta <= 0.0: return (0.1, r2)
        
    hl = -np.log(2) / np.log(beta)
    return (float(hl), float(r2))

def _pca_fit_panel_robust(panel_long: pd.DataFrame, cols_ordered: List[float], n_comps: int) -> Optional[Dict[str, Any]]:
    """
    Fits PCA on DIFFERENCED data (Yield Changes).
    Returns model + Historical Residuals for Hurst.
    """
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
    
    # Sign Flip Fix
    for i in range(n_comps):
        if i == 0: 
            if np.sum(comps[i]) < 0: comps[i] = -comps[i]
        elif i == 1: 
            if comps[i][-1] < comps[i][0]: comps[i] = -comps[i]
        elif i == 2: 
            mid_idx = len(comps[i]) // 2
            if comps[i][mid_idx] < 0: comps[i] = -comps[i]

    evr = (S**2) / (S**2).sum()
    
    factors = comps @ Z.T 
    recon_z = (comps.T @ factors).T 
    resid_z = Z - recon_z 
    
    return {
        "cols": list(W.columns), 
        "mean_diff": mu,
        "sigma_diff": sigma,
        "components": comps, 
        "evr": evr[:n_comps],
        "last_level": X_levels[-1, :], 
        "hist_resid_z": resid_z 
    }

def _pca_apply_hybrid(df_hourly: pd.DataFrame, pca_model: dict) -> Tuple[pd.Series, float]:
    """
    Applies PCA using LIVE SHOCK (Current - Prev Close).
    """
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
    recon_z_move = pca_model["components"].T @ factors
    resid_z = live_z_input - recon_z_move
    
    med = np.median(resid_z)
    mad = np.median(np.abs(resid_z - med))
    scale = (1.4826 * mad) if mad > 0 else np.std(resid_z, ddof=1)
    if scale < 1e-4: scale = 0.01
    
    final_z = (resid_z - resid_z.mean()) / scale
    
    return df_hourly["tenor_yrs"].map(dict(zip(cols, final_z))), scale

# ==============================================================================
# 5. ORCHESTRATORS (The Build Process)
# ==============================================================================

def _process_instantaneous_bucket(dts, df_bucket, df_history_daily, pca_config):
    """
    PHASE 1: Compute Instantaneous features (Snapshot only).
    Spline, PCA, Physics.
    """
    if "ts" not in df_history_daily.columns and df_history_daily.index.name == "ts":
        df_history_daily = df_history_daily.reset_index()

    out = df_bucket.copy()
    
    # 1. Get History (Strictly < Bucket Date)
    t_date = pd.to_datetime(dts).normalize()
    hist_window = df_history_daily[df_history_daily["ts"] < t_date]
    
    # 2. PCA & Hurst
    pca_z = np.nan
    pca_scale = np.nan
    hurst_map = {}
    halflife_map = {}
    
    out["z_pca"] = pca_z
    if pca_config['enable']:
        cols = sorted(out["tenor_yrs"].unique().tolist())
        model = _pca_fit_panel_robust(hist_window, cols, pca_config['n_comps'])
        
        if model:
            # A. Live PCA Signal
            pca_z, pca_scale = _pca_apply_hybrid(out, model)
            out["z_pca"] = pca_z
            
            # B. Hurst/OU on Historical Residuals
            resid_hist = model["hist_resid_z"]
            for i, tenor in enumerate(model["cols"]):
                if i < resid_hist.shape[1]:
                    h_val = _calc_hurst_rs(resid_hist[:, i])
                    hurst_map[tenor] = h_val
                    hl, _ = _calc_ou_halflife(resid_hist[:, i])
                    halflife_map[tenor] = hl

    # 3. Spline (Intraday)
    z_spline, spline_scale = _spline_fit_safe(out)
    out["z_spline"] = z_spline
    
    # 4. Physics (Drift)
    df_phys = _calc_physics_features(out)
    if not df_phys.empty:
        out = out.merge(df_phys, on='tenor_yrs', how='left')
    else:
        out['total_drift_day'] = 0.0
        out['carry_bps_day'] = 0.0
        out['roll_bps_day'] = 0.0
        out['dv01'] = 0.0

    # 5. Map & Combine
    if hurst_map: out["hurst"] = out["tenor_yrs"].map(hurst_map)
    else: out["hurst"] = 0.5

    if halflife_map: out["halflife"] = out["tenor_yrs"].map(halflife_map)
    else: out["halflife"] = 999.0 
    
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

    # --- 1. CONFIG & EXOG SETUP ---
    tenor_dict = getattr(cr, "TENOR_YEARS", {})
    # Map for Vol/Auctions (String -> Float)
    tenor_map_rev = {k: float(v) for k, v in tenor_dict.items()}
    # Fallback map for common tickers if config differs
    for t_str in ["2Y", "5Y", "10Y", "30Y"]:
        if t_str not in tenor_map_rev:
             # simple parser logic if needed, or rely on config
             pass

    # --- 2. PREP EXOG DATA (Context) ---
    # We must process Volatility FIRST to have it ready for the merge.
    print(f"[{_now()}] [PREP] Processing Volatility & Auctions...")
    # Assumes files exist in local directory or path defined in config
    path_vol = Path("VolSOFR_1M_concat.csv") 
    path_auc = Path("auc_data.csv")
    
    df_vol_daily = _process_daily_vol_stats(path_vol, tenor_map_rev)

    # --- 3. LOAD RATES ---
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

    df_long = _melt_long(df_wide, tenor_dict)
    
    # --- 4. BUCKET (Hourly & Daily) ---
    print(f"[{_now()}] [PREP] Bucketing Rates...")
    df_hourly = _make_decision_buckets(df_long, 'H', mode='head')
    df_daily = _make_decision_buckets(df_long, 'D', mode='tail')

    # --- 5. MERGE VOLATILITY (Look-Back Safety) ---
    if not df_vol_daily.empty:
        print(f"[{_now()}] [MERGE] Integrating Volatility (Backward Asof)...")
        # Ensure Types match
        df_hourly['tenor_yrs'] = df_hourly['tenor_yrs'].astype(float)
        df_vol_daily['tenor_yrs'] = df_vol_daily['tenor_yrs'].astype(float)
        
        # Backward merge: Finds the last available Close <= Hourly Time
        df_hourly = pd.merge_asof(
            df_hourly.sort_values('ts'),
            df_vol_daily.sort_values('ts_vol'),
            left_on='ts',
            right_on='ts_vol',
            by='tenor_yrs',
            direction='backward',
            tolerance=pd.Timedelta(days=7) 
        ).drop(columns=['ts_vol'])

    # --- 6. HISTORY CONTEXT (For Phase 1 PCA/Hurst) ---
    # Load previous month daily summary for continuity
    df_past_history = pd.DataFrame()
    # Logic to find prev month files (simplified):
    try:
        prev_dt = datetime.datetime.strptime(yymm, "%y%m") - relativedelta(months=1)
        prev_yymm = prev_dt.strftime("%y%m")
        prev_path = path_enh / f"{prev_yymm}_summary_D.parquet"
        if prev_path.exists():
            df_past_history = pd.read_parquet(prev_path)
    except: pass
    
    df_full_context = pd.concat([df_past_history, df_daily], ignore_index=True)
    df_full_context = df_full_context.drop_duplicates(subset=['ts', 'tenor_yrs']).sort_values('ts')

    # --- 7. PHASE 1: INSTANTANEOUS SIGNAL (Parallel) ---
    # Spline, PCA, Physics (Drift)
    buckets = np.sort(df_hourly["ts"].unique())
    pca_cfg = {
        'enable': bool(getattr(cr, "PCA_ENABLE", True)),
        'n_comps': int(getattr(cr, "PCA_COMPONENTS", 3))
    }
    
    N_JOBS = int(getattr(cr, "N_JOBS", 1))
    jobs = max(1, min((os.cpu_count() // 2), 8)) if N_JOBS == 0 else N_JOBS

    print(f"[{_now()}] [PHASE 1] Calculating Physics/Signal on {len(buckets)} buckets...")

    def _one(dts):
        snap = df_hourly[df_hourly["ts"] == dts]
        return _process_instantaneous_bucket(dts, snap, df_full_context, pca_cfg)

    parts = Parallel(n_jobs=jobs, backend="loky")(delayed(_one)(d) for d in buckets)
    
    if not parts:
        print(f"[WARN] {yymm} produced EMPTY Phase 1 output.")
        return

    df_instant = pd.concat(parts, ignore_index=True).sort_values(['ts','tenor_yrs'])

    # --- 8. PHASE 2: HOURLY REGIME & DRIFT ACCRUAL (Vectorized) ---
    print(f"[{_now()}] [PHASE 2] Calculating Hourly Rolling Stats & Accruals...")
    
    # Needs history from previous hourlies for correct rolling windows
    # (Assuming prev month _enh file exists, load tail. Simplified here.)
    df_full_hourly = df_instant.copy() 
    
    # 2. Vectorized Rolling (Kitchen Sink)
    # Applied to Rates, Signals, and Physics
    target_cols = ['rate', 'z_comb', 'z_pca', 'z_spline', 'total_drift_day', 'dv01']
    hourly_windows = [5, 10, 50] # Buckets (approx 0.5d, 1d, 1wk)
    
    features = [df_full_hourly]
    
    for col in target_cols:
        if col in df_full_hourly.columns:
            stats = _calc_kitchen_sink_stats(df_full_hourly, col, hourly_windows)
            features.append(stats)
            
    df_final = pd.concat(features, axis=1)
    # Cleanup duplicates
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    
    # 3. Time-Aware Drift Accrual (The Weekend Problem)
    # Calc Time Elapsed between buckets per tenor
    df_final['hours_elapsed'] = df_final.groupby('tenor_yrs')['ts'].diff().dt.total_seconds() / 3600.0
    # First item in group is NaN, assume 1 hour or 0? 1 hr is safer for trading logic.
    df_final['hours_elapsed'] = df_final['hours_elapsed'].fillna(1.0)
    
    # Accrue & CumSum
    for d_col in ['carry_bps_day', 'roll_bps_day', 'total_drift_day']:
        if d_col in df_final.columns:
            # Accrued = (Daily_Rate / 24) * Hours_Elapsed
            accrued_col = d_col.replace('_day', '_accrued')
            df_final[accrued_col] = (df_final[d_col] / 24.0) * df_final['hours_elapsed']
            
            # CumSum = Running Total (To scan via subtraction later)
            cumsum_col = d_col.replace('_day', '_cumsum')
            df_final[cumsum_col] = df_final.groupby('tenor_yrs')[accrued_col].cumsum().fillna(0.0)

    # --- 9. MERGE AUCTIONS (Final Step) ---
    print(f"[{_now()}] [MERGE] Integrating Auction Countdowns...")
    df_final = _enrich_with_auctions(df_final, path_auc, tenor_map_rev)

    # --- 10. SAVE ---
    out_name = f"{yymm}_enh{getattr(cr, 'ENH_SUFFIX', '')}.parquet"
    out_path = path_enh / out_name
    
    if not df_final.empty and 'z_comb' in df_final.columns:
        df_final.to_parquet(out_path, index=False)
        
        # Save Daily Summary for next month's history
        df_daily.to_parquet(path_enh / f"{yymm}_summary_D.parquet", index=False)
        
        zr = pd.to_numeric(df_final['z_comb'], errors='coerce')
        valid_pct = float(np.isfinite(zr).mean() * 100)
        print(f"[DONE] {yymm} -> {out_path} (Rows: {len(df_final)}, Valid Z: {valid_pct:.1f}%)")
    else:
        print(f"[WARN] {yymm} produced EMPTY output.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_creation.py 2304")
        sys.exit(1)
    for m in sys.argv[1:]:
        build_month(m)
