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
import volscript as vs
import ecoscript as ec

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

def _get_ql_calendar():
    if not getattr(cr, "USE_QL_CALENDAR", False): 
        return None
    try:
        import QuantLib as ql
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
    Applies rolling stats. 
    Robust implementation: Decouples math from index to prevent alignment errors.
    """
    # 1. Setup Output with Original Index
    res = pd.DataFrame(index=df.index)
    
    # 2. Internal Working Frame (RangeIndex 0..N)
    # We use this for all groupby/shift operations to guarantee alignment
    work_df = df[[col, group_col]].copy().reset_index(drop=True)
    grp = work_df.groupby(group_col)[col]
    
    # Pre-extract numpy array for raw values to avoid Series overhead/alignment issues
    raw_values = work_df[col].values
    
    for w in windows:
        suffix = f"{w}d" if "exog" in col else f"{w}b" 
        
        # --- A. KINETICS ---
        # Note: grp.shift(w) returns a Series aligned to work_df.index (RangeIndex)
        shift_w = grp.shift(w)
        
        # Slope: (Current - Old) / Window
        # We use .values on shift_w to ensure numpy subtraction (safe)
        slope_arr = (raw_values - shift_w.values) / w
        
        # Accel: Slope - Slope_Old
        # We need to wrap slope in a Series (with work_df's grouping) to shift it correctly
        slope_series = pd.Series(slope_arr, index=work_df.index)
        shift_w_half = slope_series.groupby(work_df[group_col]).shift(w // 2)
        accel_arr = slope_arr - shift_w_half.values

        # --- B. DISTRIBUTION ---
        roll = grp.rolling(w)
        
        mean_val = roll.mean().values
        std_val  = roll.std().values
        max_val  = roll.max().values
        min_val  = roll.min().values
        
        # --- C. DERIVED (Numpy Math) ---
        max_abs = np.maximum(np.abs(max_val), np.abs(min_val))
        z_local = (raw_values - mean_val) / (std_val + 1e-8)
        rng_pos = (raw_values - min_val) / ((max_val - min_val) + 1e-8)
        
        # --- D. QUANTILES ---
        q25 = roll.quantile(0.25).values
        q75 = roll.quantile(0.75).values

        # --- ASSIGNMENT ---
        # Direct numpy assignment to the result DataFrame
        # Since res has same length as work_df, this aligns by position
        # and ignores any index mismatches.
        col_slope = f"{col}_slope_{suffix}"
        res[col_slope] = slope_arr
        res[f"{col}_accel_{suffix}"] = accel_arr
        
        res[f"{col}_mean_{suffix}"] = mean_val
        res[f"{col}_std_{suffix}"]  = std_val
        res[f"{col}_max_{suffix}"]  = max_val
        res[f"{col}_min_{suffix}"]  = min_val
        
        res[f"{col}_max_abs_{suffix}"] = max_abs
        res[f"{col}_zlocal_{suffix}"] = z_local
        res[f"{col}_rng_pos_{suffix}"] = rng_pos
        
        res[f"{col}_q25_{suffix}"] = q25
        res[f"{col}_q75_{suffix}"] = q75
        
    return res

# ==============================================================================
# 3. EXOGENOUS PROCESSORS (Vol & Auctions)
# ==============================================================================

def _process_daily_vol_df(df_vol: pd.DataFrame, tenor_map_rev: Dict[str, float]) -> pd.DataFrame:
    """
    Takes the dataframe from volscript.run_all(), maps tickers, calcs stats.
    """
    if df_vol.empty: return pd.DataFrame()
    
    # 1. Parse Timestamp (file_modified is the reference time)
    # Ensure UTC
    if 'file_modified' in df_vol.columns:
        df_vol['ts_vol'] = pd.to_datetime(df_vol['file_modified'], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        return pd.DataFrame() # Can't process without time
    
    # 2. Map Tickers
    df_vol['tenor_yrs'] = df_vol['ticker'].map(tenor_map_rev)
    df = df_vol.dropna(subset=['tenor_yrs']).sort_values(['tenor_yrs', 'ts_vol'])
    
    # 3. Rename
    if 'Implied Vol' in df.columns: 
        df = df.rename(columns={'Implied Vol': 'vol_implied'})
        df['vol_implied'] = pd.to_numeric(df['vol_implied'], errors='coerce')
    if 'Skew' in df.columns: 
        df = df.rename(columns={'Skew': 'vol_skew'})
        df['vol_skew'] = pd.to_numeric(df['vol_skew'], errors='coerce')
    
    # 4. Kitchen Sink (Daily Windows)
    vol_windows = [5, 20, 60] 
    features = []
    base_cols = ['ts_vol', 'tenor_yrs', 'vol_implied', 'vol_skew']
    features.append(df[[c for c in base_cols if c in df.columns]])
    
    for col in ['vol_implied', 'vol_skew']:
        if col in df.columns:
            stats = _calc_kitchen_sink_stats(df, col, vol_windows, group_col='tenor_yrs')
            features.append(stats)
        
    df_processed = pd.concat(features, axis=1)
    
    # 5. Prefix
    keys = ['ts_vol', 'tenor_yrs']
    new_cols = {c: f"exog_{c}" for c in df_processed.columns if c not in keys}
    df_processed = df_processed.rename(columns=new_cols)
    
    return df_processed.sort_values('ts_vol')

def _enrich_with_auctions(df_hourly: pd.DataFrame, df_auc: pd.DataFrame, tenor_map_rev: Dict[str, float]) -> pd.DataFrame:
    """
    Calculates 'hours_to_auction' (Tenor Specific).
    """
    if df_auc.empty or df_hourly.empty:
        df_hourly['hours_to_auction'] = 999.0
        return df_hourly

    # Prep Auction Data
    ts_col = 'Date Time'
    type_col = 'event_type' # Contains ticker
    
    if ts_col not in df_auc.columns: return df_hourly

    # Convert to matching timezone-naive UTC
    auc_ts = pd.to_datetime(df_auc[ts_col], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df_auc['ts_auc'] = auc_ts
    
    # Map Tickers
    df_auc['tenor_yrs'] = df_auc[type_col].map(tenor_map_rev)
    df_auc = df_auc.dropna(subset=['tenor_yrs']).sort_values('ts_auc')
    
    df_hourly = df_hourly.sort_values('ts')
    
    # Merge Asof (Forward) matching on Tenor
    merged = pd.merge_asof(
        df_hourly,
        df_auc[['ts_auc', 'tenor_yrs']], 
        left_on='ts',
        right_on='ts_auc',
        by='tenor_yrs',
        direction='forward',
        tolerance=pd.Timedelta(days=30),
        suffixes=('', '_auc')
    )
    
    diff = (merged['ts_auc'] - merged['ts']).dt.total_seconds() / 3600.0
    df_hourly['hours_to_auction'] = diff.fillna(999.0).values
    return df_hourly

def _enrich_with_econ(df_hourly: pd.DataFrame, df_eco: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 'hours_to_econ' (Market Wide / Systemic).
    Does NOT match on tenor (Econ affects everyone).
    """
    if df_eco.empty or df_hourly.empty:
        df_hourly['hours_to_econ'] = 999.0
        return df_hourly
        
    ts_col = 'Date Time'
    if ts_col not in df_eco.columns: return df_hourly
    
    # Prep Eco Data
    eco_ts = pd.to_datetime(df_eco[ts_col], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    # We only need the timestamps, sorted
    eco_events = pd.DataFrame({'ts_eco': eco_ts}).sort_values('ts_eco').dropna()
    
    df_hourly = df_hourly.sort_values('ts')
    
    # Merge Asof (Forward) - No 'by' clause because it's global
    merged = pd.merge_asof(
        df_hourly,
        eco_events,
        left_on='ts',
        right_on='ts_eco',
        direction='forward',
        tolerance=pd.Timedelta(days=14), # 2 week lookahead max
        suffixes=('', '_eco')
    )
    
    diff = (merged['ts_eco'] - merged['ts']).dt.total_seconds() / 3600.0
    df_hourly['hours_to_econ'] = diff.fillna(999.0).values
    return df_hourly

# ==============================================================================
# 4. MATH LOGIC: PHYSICS (Core), PCA, SPLINE, HURST
# ==============================================================================

def _calc_physics_features(snap_df: pd.DataFrame) -> pd.DataFrame:
    """
    PHYSICS ENGINE: Uses math_core.SplineCurve to calculate Drift, Carry, Roll, DV01.
    """
    if snap_df.empty: return pd.DataFrame()

    valid = snap_df[snap_df["tenor_yrs"] >= 0.0].dropna()
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
    
    s_fit = snap_long[snap_long["tenor_yrs"] >= 0.0].dropna().sort_values("tenor_yrs")
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
    Hurst Exponent (Rescaled Range Analysis).
    0.0 < H < 0.5: Mean Reverting.
    0.5 < H < 1.0: Trending.
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
    Ornstein-Uhlenbeck Half-Life via Regression.
    Returns: (HalfLife_Days, R_Squared)
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
    
    # Calculate R2
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

def _pca_apply_hybrid(df_hourly: pd.DataFrame, pca_model: dict) -> Tuple[pd.Series, float, Dict[str, Any]]:
    """
    Applies PCA and extracts Regime (Vol/Drift) and State (Factors/Error) features.
    """
    out = pd.Series(index=df_hourly.index, dtype=float)
    scale = np.nan
    extras = {} # New container for meta features
    
    if not pca_model or df_hourly.empty: return out, scale, extras

    cols = pca_model["cols"]
    # Reindex ensures we align with the model's structure
    snap = df_hourly.set_index("tenor_yrs")["rate"].reindex(cols)
    if snap.isnull().any(): return out, scale, extras 

    current_level = snap.values.astype(float)
    
    # 1. Calculate Core Signal
    live_diff = current_level - pca_model["last_level"]
    live_z_input = (live_diff - pca_model["mean_diff"]) / pca_model["sigma_diff"]
    
    factors = pca_model["components"] @ live_z_input
    recon_z_move = pca_model["components"].T @ factors
    resid_z = live_z_input - recon_z_move
    
    # 2. Robust Scaling (Existing Logic)
    med = np.median(resid_z)
    mad = np.median(np.abs(resid_z - med))
    scale = (1.4826 * mad) if mad > 0 else np.std(resid_z, ddof=1)
    if scale < 1e-4: scale = 0.01
    
    final_z = (resid_z - resid_z.mean()) / scale
    out_series = df_hourly["tenor_yrs"].map(dict(zip(cols, final_z)))

    # --- NEW: Extract Meta Features ---
    
    # A. Regime Features (Vector Mapped to Tenor)
    # These tell the model "What is normal volatility/drift for THIS tenor?"
    # We map the model arrays back to the dataframe index
    extras["pca_vol_regime"] = df_hourly["tenor_yrs"].map(dict(zip(cols, pca_model["sigma_diff"])))
    extras["pca_drift_regime"] = df_hourly["tenor_yrs"].map(dict(zip(cols, pca_model["mean_diff"])))

    # B. State Features (Scalars applied to entire timestamp)
    # Factor Scores (Level, Slope, Curve)
    for i, score in enumerate(factors):
        extras[f"pca_factor_{i}"] = float(score)
        
    # Reconstruction Error (Q-Statistic / "Weirdness")
    extras["pca_error_norm"] = float(np.linalg.norm(resid_z))
    
    # Confidence (Explained Variance)
    extras["pca_evr_sum"] = float(np.sum(pca_model["evr"]))

    return out_series, scale, extras

# ==============================================================================
# 5. ORCHESTRATORS (The Build Process)
# ==============================================================================

def _process_instantaneous_bucket(dts, df_bucket, df_history_daily, pca_config):
    """PHASE 1: Compute Instantaneous features (Snapshot only)."""
    # Ensure History has 'ts' column
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
    
    # Only run PCA if we have history
    if pca_config['enable'] and not hist_window.empty:
        cols = sorted(out["tenor_yrs"].unique().tolist())
        model = _pca_fit_panel_robust(hist_window, cols, pca_config['n_comps'])
        
        if model:
            # A. Live PCA Signal
            pca_z, pca_scale, pca_extras = _pca_apply_hybrid(out, model)
            out["z_pca"] = pca_z
            
            # Loop through extras and assign to DataFrame
            # Scalars (Factors) will broadcast automatically; Series (Regime) will align
            for k, v in pca_extras.items():
                out[k] = v
            
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
    
    # Robust Scale Logic
    raw_scale = 0.01
    if np.isfinite(pca_scale) and pca_scale > 1e-6:
        raw_scale = pca_scale
    elif np.isfinite(spline_scale):
        raw_scale = spline_scale
        
    out["scale"] = raw_scale
    out["z_comb"] = out[["z_pca", "z_spline"]].mean(axis=1)
    
    return out

def build_month(yymm: str) -> None:
    path_data = Path(getattr(cr, "PATH_DATA", "."))
    path_enh  = Path(getattr(cr, "PATH_ENH", "."))
    path_enh.mkdir(parents=True, exist_ok=True)

    # --- 1. CONFIG & EXOG SETUP ---
    tenor_dict = getattr(cr, "TENOR_YEARS", {})
    tenor_map_rev = {k: float(v) for k, v in tenor_dict.items()}

    # --- 2. GENERATE EXTERNAL DATA (Live) ---
    print(f"[{_now()}] [PREP] Running Volatility & Eco Generators...")
    
    df_vol_daily = pd.DataFrame()
    df_eco_raw = pd.DataFrame()
    df_auc_raw = pd.DataFrame()

    try:
        # Run Vol Script
        df_vol_raw = vs.run_all() 
        df_vol_daily = _process_daily_vol_df(df_vol_raw, tenor_map_rev)
    except Exception as e:
        print(f"[WARN] Volatility Generation Failed: {e}")

    try:
        # Run Eco Script (Returns Tuple: Eco, Auc)
        df_eco_raw, df_auc_raw = ec.run_all()
    except Exception as e:
        print(f"[WARN] Eco/Auction Generation Failed: {e}")

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
        df_hourly['tenor_yrs'] = df_hourly['tenor_yrs'].astype(float)
        df_vol_daily['tenor_yrs'] = df_vol_daily['tenor_yrs'].astype(float)
        
        # Merge Asof: Matches hourly TS to the most recent previous Vol Close
        df_hourly = pd.merge_asof(
            df_hourly.sort_values('ts'),
            df_vol_daily.sort_values('ts_vol'),
            left_on='ts',
            right_on='ts_vol',
            by='tenor_yrs',
            direction='backward',
            tolerance=pd.Timedelta(days=7) 
        ).drop(columns=['ts_vol'])

    # --- 6. HISTORY CONTEXT (For Phase 1 PCA) ---
    df_past_history = pd.DataFrame()
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

    # --- 8. PHASE 2: ROLLING STATS (With Warm Start) ---
    print(f"[{_now()}] [PHASE 2] Calculating Rolling Stats & Accruals...")
    
    # A. WARM START BUFFER
    MAX_LOOKBACK_HRS = 120 # 5 Days
    df_prev_tail = pd.DataFrame()
    drift_offsets = {}
    
    try:
        curr_dt = datetime.datetime.strptime(yymm, "%y%m")
        prev_dt = curr_dt - relativedelta(months=1)
        prev_yymm = prev_dt.strftime("%y%m")
        prev_enh_path = path_enh / f"{prev_yymm}_enh{getattr(cr, 'ENH_SUFFIX', '')}.parquet"

        if prev_enh_path.exists():
            df_prev = pd.read_parquet(prev_enh_path)
            
            # 1. Capture Drift Offsets
            for d_col in ['total_drift_day', 'carry_bps_day', 'roll_bps_day']:
                cs_col = d_col.replace('_day', '_cumsum')
                if cs_col in df_prev.columns:
                    drift_offsets[cs_col] = df_prev.groupby('tenor_yrs')[cs_col].last().to_dict()

            # 2. Slice Tail
            last_ts = df_prev['ts'].max()
            start_buffer = last_ts - pd.Timedelta(hours=MAX_LOOKBACK_HRS)
            common_cols = [c for c in df_instant.columns if c in df_prev.columns]
            df_prev_tail = df_prev[df_prev['ts'] > start_buffer][common_cols].copy()
    except: pass

    # B. CONCAT
    if not df_prev_tail.empty:
        df_full_hourly = pd.concat([df_prev_tail, df_instant], ignore_index=True)
    else:
        df_full_hourly = df_instant.copy()
    df_full_hourly = df_full_hourly.sort_values(['tenor_yrs', 'ts'])

    # C. KITCHEN SINK (Including Rate!)
    # We include 'rate' so XGBoost can see Rate Velocity (Slope) and Acceleration.
    target_cols = ['rate', 'z_comb', 'z_pca', 'z_spline', 'total_drift_day', 'dv01']
    
    # Also include Vol/Skew if they exist
    if 'exog_vol_implied' in df_full_hourly.columns: target_cols.append('exog_vol_implied')
    if 'exog_vol_skew' in df_full_hourly.columns: target_cols.append('exog_vol_skew')
    
    features = [df_full_hourly]
    hourly_windows = [5, 10, 50]
    
    for col in target_cols:
        if col in df_full_hourly.columns:
            stats = _calc_kitchen_sink_stats(df_full_hourly, col, hourly_windows)
            features.append(stats)
            
    df_final_buffered = pd.concat(features, axis=1)
    # Remove duplicate columns
    df_final_buffered = df_final_buffered.loc[:, ~df_final_buffered.columns.duplicated()]

    # D. DRIFT ACCRUAL & OFFSET
    df_final_buffered['hours_elapsed'] = df_final_buffered.groupby('tenor_yrs')['ts'].diff().dt.total_seconds() / 3600.0
    df_final_buffered['hours_elapsed'] = df_final_buffered['hours_elapsed'].fillna(1.0)
    
    for d_col in ['carry_bps_day', 'roll_bps_day', 'total_drift_day']:
        if d_col in df_final_buffered.columns:
            accrued_col = d_col.replace('_day', '_accrued')
            cumsum_col = d_col.replace('_day', '_cumsum')
            
            df_final_buffered[accrued_col] = (df_final_buffered[d_col] / 24.0) * df_final_buffered['hours_elapsed']
            df_final_buffered[cumsum_col] = df_final_buffered.groupby('tenor_yrs')[accrued_col].cumsum().fillna(0.0)
            
            # Apply Offset from Prev Month
            if cumsum_col in drift_offsets:
                offset_series = df_final_buffered['tenor_yrs'].map(drift_offsets[cumsum_col]).fillna(0.0)
                df_final_buffered[cumsum_col] += offset_series

    # E. SLICE TO CURRENT MONTH
    min_ts_current = df_instant['ts'].min()
    df_final = df_final_buffered[df_final_buffered['ts'] >= min_ts_current].copy()

    # --- 9. MERGE EVENTS (Events & Econ) ---
    print(f"[{_now()}] [MERGE] Integrating Events (Auctions & Econ)...")
    
    # A. Auctions (Tenor Specific)
    df_final = _enrich_with_auctions(df_final, df_auc_raw, tenor_map_rev)
    
    # B. Econ (Market Wide)
    # Note: df_eco_raw comes from ecoscript.run_all()
    df_final = _enrich_with_econ(df_final, df_eco_raw)

    # --- 10. SAVE ---
    out_name = f"{yymm}_enh{getattr(cr, 'ENH_SUFFIX', '')}.parquet"
    out_path = path_enh / out_name
    
    if not df_final.empty and 'z_comb' in df_final.columns:
        df_final.to_parquet(out_path, index=False)
        df_daily.to_parquet(path_enh / f"{yymm}_summary_D.parquet", index=False)
        
        zr = pd.to_numeric(df_final['z_comb'], errors='coerce')
        valid_pct = float(np.isfinite(zr).mean() * 100)
        print(f"[DONE] {yymm} -> {out_path} (Rows: {len(df_final)}, Valid Z: {valid_pct:.1f}%)")
    else:
        print(f"[WARN] {yymm} produced EMPTY output.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_creation.py 2304")
    for m in sys.argv[1:]:
        build_month(m)
