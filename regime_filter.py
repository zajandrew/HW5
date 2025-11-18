# regime_filter.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

import cr_config as cr


# ---------------------------------------------------------------------------
# Helpers to read enhanced files month-by-month
# ---------------------------------------------------------------------------

def _enhanced_in_path(yymm: str) -> Path:
    """
    Same convention as portfolio_test_new/overlay_diag:
    enh_2304_d.parquet, enh_2304_h.parquet, etc.
    """
    return Path(cr.PATH_ENH) / f"enh_{yymm}{cr.ENH_SUFFIX}.parquet"


def _iter_month_snapshots(
    yymms: Iterable[str],
    decision_freq: Optional[str] = None,
):
    """
    Generator yielding (bucket, snap_last_df) where:
      bucket: decision timestamp (daily or hourly)
      snap_last_df: last snapshot per tenor_yrs within that bucket.
    This is memory-safe: it streams month by month.
    """
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()

    for yymm in yymms:
        path = _enhanced_in_path(yymm)
        if not path.exists():
            print(f"[regime_filter] Missing enhanced file for {yymm}: {path}")
            continue

        df = pd.read_parquet(path)
        if df.empty:
            print(f"[regime_filter] Empty enhanced file for {yymm}: {path}")
            continue

        df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
        if decision_freq == "D":
            df["bucket"] = df["ts"].dt.floor("D")
        elif decision_freq == "H":
            df["bucket"] = df["ts"].dt.floor("H")
        else:
            raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

        # Group by decision bucket within the month.
        for bucket, snap in df.groupby("bucket", sort=True):
            if snap.empty:
                continue

            snap_last = (
                snap.sort_values("ts")
                    .groupby("tenor_yrs", as_index=False)
                    .tail(1)
                    .reset_index(drop=True)
            )
            if snap_last.empty:
                continue

            yield bucket, snap_last


# ---------------------------------------------------------------------------
# Config knobs
# ---------------------------------------------------------------------------

@dataclass
class RegimeFilterConfig:
    """
    All knobs for signal construction and regime gating.

    Defaults are a reasonable starting point based on your plots:
      - windows ~10 days,
      - health in [0,1] (1 = good),
      - bad regime: health < 0.4, trendiness > -0.2, residual > 1.8
      - recovery: health > 0.6, trendiness < -0.4, residual < 1.2
      - shocks: large day-over-day moves in these variables,
        scaled by rolling volatility.
    """

    # ----- signal windows -----
    window_health: int = 10         # smoothing window for health
    window_trend: int = 10          # window for rolling AC1 of Δmean_z
    window_resid: int = 10          # smoothing window for residuals

    # window for estimating volatility of day-over-day changes (for shocks)
    shock_vol_window: int = 20

    # ----- which signals to use -----
    use_health: bool = True
    use_trend: bool = True
    use_resid: bool = True

    # ----- shock thresholds (ON -> OFF) -----
    # Absolute day-over-day move thresholds:
    shock_abs_dhealth: Optional[float] = 0.25   # drop in health
    shock_abs_dtrend: Optional[float] = 0.25    # jump in trendiness
    shock_abs_dresid: Optional[float] = 0.50    # jump in residuals

    # Z-score thresholds vs rolling std of ΔS (None = don't use z-score filter)
    shock_zscore_dhealth: Optional[float] = 1.5
    shock_zscore_dtrend: Optional[float] = 1.5
    shock_zscore_dresid: Optional[float] = 1.5

    # ----- bad-regime level thresholds (ON -> OFF) -----
    # If any active signal is inside its "bad" region we can flip OFF.
    level_bad_health_min: Optional[float] = 0.40     # health < this is bad
    level_bad_trend_max: Optional[float] = -0.20     # trend > this is bad
    level_bad_resid_max: Optional[float] = 1.80      # resid > this is bad

    level_window: int = 10   # smoothing for level checks

    # ----- recovery thresholds (OFF -> ON) -----
    # Need all active signals in their "good" regions for K consecutive days.
    recovery_good_health_min: Optional[float] = 0.60   # health > this is good
    recovery_good_trend_max: Optional[float] = -0.40   # trend < this is good
    recovery_good_resid_max: Optional[float] = 1.20    # resid < this is good

    recovery_window: int = 10   # smoothing for recovery checks

    # base number of consecutive good days required
    K_base: int = 10            # ~2 trading weeks

    # scaling of K with shock severity (>=0)
    # K = K_base * (1 + alpha * (severity - 1))
    K_alpha: float = 0.5


DEFAULT_CONFIG = RegimeFilterConfig()


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def _rolling_ac1(x: pd.Series, window: int) -> pd.Series:
    """
    Rolling lag-1 autocorrelation of x over the given window.
    """
    if window <= 1:
        return pd.Series(index=x.index, dtype=float)

    def ac1(arr: np.ndarray) -> float:
        arr = arr.astype(float)
        if np.all(np.isnan(arr)):
            return np.nan
        # drop nans
        arr = arr[~np.isnan(arr)]
        if len(arr) < 3:
            return np.nan
        x1 = arr[:-1]
        x2 = arr[1:]
        if x1.std(ddof=0) == 0 or x2.std(ddof=0) == 0:
            return 0.0
        return float(np.corrcoef(x1, x2)[0, 1])

    return x.rolling(window=window, min_periods=window).apply(ac1, raw=True)


def build_base_series(
    yymms: Iterable[str],
    decision_freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    Construct the *base* time series used for regime signals.

    Returns a DataFrame indexed by decision bucket (D/H) with columns:
      - mean_z_comb       : cross-sectional mean z_comb
      - mean_abs_z_comb   : mean |z_comb|
      - median_abs_resid  : median |z_spline - z_pca|
    """
    records = []

    for bucket, snap_last in _iter_month_snapshots(yymms, decision_freq):
        zc = snap_last["z_comb"].astype(float)
        zs = snap_last["z_spline"].astype(float)
        zp = snap_last["z_pca"].astype(float)

        resid = (zs - zp).abs()

        records.append(
            {
                "bucket": bucket,
                "mean_z_comb": float(zc.mean()),
                "mean_abs_z_comb": float(zc.abs().mean()),
                "median_abs_resid": float(resid.median()),
            }
        )

    if not records:
        raise ValueError("No enhanced snapshots found for the given yymms.")

    base = (
        pd.DataFrame(records)
        .dropna()
        .sort_values("bucket")
        .set_index("bucket")
    )

    return base


def add_derived_signals(
    base: pd.DataFrame,
    cfg: RegimeFilterConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Take the base series and add:
      - d_mean_z_comb          : day-over-day change in mean_z_comb
      - trendiness             : rolling AC1 of Δmean_z (higher = more trending)
      - resid_smoothed         : smoothed median_abs_resid
      - health_core            : health in [0,1] based on residuals only
      - health_combined        : health including trend (also [0,1])
    """
    df = base.copy()
    df = df.sort_index()

    # Δ mean z
    df["d_mean_z_comb"] = df["mean_z_comb"].diff()

    # Trendiness: AC1 of Δmean_z over rolling window
    df["trendiness"] = _rolling_ac1(
        df["d_mean_z_comb"], window=cfg.window_trend
    )

    # Smoothed residuals
    df["resid_smoothed"] = (
        df["median_abs_resid"]
        .rolling(window=cfg.window_resid, min_periods=1)
        .median()
    )

    # Normalize residuals to [0,1] via in-sample quantiles
    q_lo_resid = df["resid_smoothed"].quantile(0.1)
    q_hi_resid = df["resid_smoothed"].quantile(0.9)
    denom_resid = max(1e-6, q_hi_resid - q_lo_resid)

    resid_norm = (df["resid_smoothed"] - q_lo_resid) / denom_resid
    resid_norm = resid_norm.clip(0.0, 1.0)

    # Normalize trendiness: more positive = more trending = worse
    q_lo_tr = df["trendiness"].quantile(0.1)
    q_hi_tr = df["trendiness"].quantile(0.9)
    denom_tr = max(1e-6, q_hi_tr - q_lo_tr)

    trend_norm = (df["trendiness"] - q_lo_tr) / denom_tr
    trend_norm = trend_norm.clip(0.0, 1.0)

    # Health: 1 = good, 0 = bad.
    df["health_core"] = 1.0 - resid_norm
    df["health_combined"] = 1.0 - 0.5 * (resid_norm + trend_norm)

    return df


# ---------------------------------------------------------------------------
# Regime state machine (no lookahead)
# ---------------------------------------------------------------------------

def _shock_flags(
    df: pd.DataFrame,
    cfg: RegimeFilterConfig,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute per-day shock flags and a severity score (>=0).
    Uses *previous day's* metrics to avoid lookahead when we later
    align the state with trading decisions.
    """
    # work on copies to avoid SettingWithCopy noise
    d_health = df["health_core"].diff()
    d_trend = df["trendiness"].diff()
    d_resid = df["resid_smoothed"].diff()

    # rolling std of day-over-day changes
    vol_d_health = d_health.rolling(cfg.shock_vol_window, min_periods=5).std()
    vol_d_trend = d_trend.rolling(cfg.shock_vol_window, min_periods=5).std()
    vol_d_resid = d_resid.rolling(cfg.shock_vol_window, min_periods=5).std()

    shock_flag = pd.Series(False, index=df.index, dtype=bool)
    severity = pd.Series(0.0, index=df.index, dtype=float)

    # Helper to register shock for one signal
    def apply_signal(
        use: bool,
        dS: pd.Series,
        vol_dS: pd.Series,
        abs_thresh: Optional[float],
        z_thresh: Optional[float],
    ):
        nonlocal shock_flag, severity

        if (not use) or (abs_thresh is None):
            return

        move = dS.copy()

        # We care about magnitude of the move; direction handled by sign of dS
        big_abs = move.abs() >= abs_thresh

        if z_thresh is not None:
            z = move / (vol_dS.replace(0.0, np.nan))
            big_z = z.abs() >= z_thresh
            hit = big_abs & big_z
        else:
            hit = big_abs

        # Severity = how many "abs_thresh" units the move represents.
        sev = (move.abs() / abs_thresh).fillna(0.0)

        # Update overall flags and max severity
        shock_flag |= hit
        severity = np.where(hit, np.maximum(severity, sev), severity)

    apply_signal(
        cfg.use_health,
        d_health,
        vol_d_health,
        cfg.shock_abs_dhealth,
        cfg.shock_zscore_dhealth,
    )
    apply_signal(
        cfg.use_trend,
        d_trend,
        vol_d_trend,
        cfg.shock_abs_dtrend,
        cfg.shock_zscore_dtrend,
    )
    apply_signal(
        cfg.use_resid,
        d_resid,
        vol_d_resid,
        cfg.shock_abs_dresid,
        cfg.shock_zscore_dresid,
    )

    return shock_flag, severity


def _level_flags(df: pd.DataFrame, cfg: RegimeFilterConfig) -> pd.Series:
    """
    Bad-regime level flags (True when we consider the environment "bad").
    Uses smoothed series and will later be shifted by 1 for no lookahead.
    """
    # Smooth series
    H = df["health_core"].rolling(cfg.level_window, min_periods=1).mean()
    T = df["trendiness"].rolling(cfg.level_window, min_periods=1).mean()
    R = df["resid_smoothed"].rolling(cfg.level_window, min_periods=1).mean()

    bad = pd.Series(False, index=df.index, dtype=bool)

    if cfg.use_health and cfg.level_bad_health_min is not None:
        bad |= H < cfg.level_bad_health_min

    if cfg.use_trend and cfg.level_bad_trend_max is not None:
        bad |= T > cfg.level_bad_trend_max

    if cfg.use_resid and cfg.level_bad_resid_max is not None:
        bad |= R > cfg.level_bad_resid_max

    return bad


def _recovery_good_flags(df: pd.DataFrame, cfg: RegimeFilterConfig) -> pd.Series:
    """
    Flags indicating days where *all active signals* are in their
    "good" regions (used during OFF regime for counting toward recovery).
    """
    H = df["health_core"].rolling(cfg.recovery_window, min_periods=1).mean()
    T = df["trendiness"].rolling(cfg.recovery_window, min_periods=1).mean()
    R = df["resid_smoothed"].rolling(cfg.recovery_window, min_periods=1).mean()

    good = pd.Series(True, index=df.index, dtype=bool)

    if cfg.use_health and cfg.recovery_good_health_min is not None:
        good &= H >= cfg.recovery_good_health_min

    if cfg.use_trend and cfg.recovery_good_trend_max is not None:
        good &= T <= cfg.recovery_good_trend_max

    if cfg.use_resid and cfg.recovery_good_resid_max is not None:
        good &= R <= cfg.recovery_good_resid_max

    return good


def build_regime_state(
    signals: pd.DataFrame,
    cfg: RegimeFilterConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Given a signal DataFrame from add_derived_signals, build a regime
    ON/OFF state with *no lookahead bias*.

    The returned DataFrame has columns:
      - state         : 1 = ON (strategy allowed), 0 = OFF
      - shock_flag    : True if a shock occurred (based on yesterday's data)
      - level_flag    : True if environment is bad (based on yesterday's data)
      - off_counter   : consecutive "good" days seen in current OFF episode
      - K_current     : recovery horizon (days) for current OFF episode
      - shock_severity: severity for the last shock that flipped us OFF
    """
    df = signals.copy().sort_index()

    # Shock & level flags in "raw" time (based on same-day metrics)
    shock_raw, severity_raw = _shock_flags(df, cfg)
    level_raw = _level_flags(df, cfg)
    recovery_good_raw = _recovery_good_flags(df, cfg)

    # To avoid lookahead:
    #   state used for decisions on day t will be based on
    #   shock/level/recovery signals up to day t-1.
    shock = shock_raw.shift(1).fillna(False)
    severity = severity_raw.shift(1).fillna(0.0)
    level_bad = level_raw.shift(1).fillna(False)
    recovery_good = recovery_good_raw.shift(1).fillna(False)

    dates = df.index

    state = []
    off_counter = []
    K_current_series = []
    last_severity_series = []

    # State machine
    on = True
    K_curr = float(cfg.K_base)
    off_run = 0
    last_sev = 0.0

    for t in dates:
        sh = bool(shock.loc[t])
        lv = bool(level_bad.loc[t])
        rec_good = bool(recovery_good.loc[t])
        sev = float(severity.loc[t])

        if on:
            # Check if we should flip OFF due to shock or bad level
            if sh or lv:
                on = False
                last_sev = max(1.0, sev) if sev > 0 else 1.0
                # dynamic K: larger after bigger shocks
                K_curr = max(
                    cfg.K_base,
                    cfg.K_base * (1.0 + cfg.K_alpha * (last_sev - 1.0)),
                )
                off_run = 0
        else:
            # We are OFF; count consecutive "good" days
            if rec_good:
                off_run += 1
            else:
                off_run = 0

            if off_run >= K_curr:
                # Flip back ON and reset
                on = True
                off_run = 0
                K_curr = float(cfg.K_base)
                last_sev = 0.0

        state.append(1 if on else 0)
        off_counter.append(off_run)
        K_current_series.append(K_curr)
        last_severity_series.append(last_sev)

    out = pd.DataFrame(
        {
            "state": state,
            "shock_flag": shock,
            "level_flag": level_bad,
            "off_counter": off_counter,
            "K_current": K_current_series,
            "shock_severity": last_severity_series,
        },
        index=dates,
    )

    return out


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_regime_filter(
    yymms: Iterable[str],
    decision_freq: Optional[str] = None,
    cfg: RegimeFilterConfig = DEFAULT_CONFIG,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline:
      1) build_base_series(yymms)
      2) add_derived_signals(...)
      3) build_regime_state(...)

    Returns (signals_df, regime_df).
    """
    base = build_base_series(yymms, decision_freq)
    signals = add_derived_signals(base, cfg)
    regime = build_regime_state(signals, cfg)
    return signals, regime
