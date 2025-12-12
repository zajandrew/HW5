"""
hybrid_filter.py

Regime filter logic for the overlay / strategy.

Responsibilities
----------------
1) Build and cache a time series of "hybrid_signals" based purely on the
   enhanced curve panel (Curve Environment).

2) Regime filter:
   - From hybrid_signals, build composite "signal health", "trendiness", etc.
   - Produce a boolean series indicating when we consider the environment tradable.
   - Can pull defaults directly from cr_config.

All functions are written to avoid lookahead by design: any signal used to
decide about bucket t is lagged to t-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import cr_config as cr


# ======================================================================
# Paths / basic config
# ======================================================================

HYBRID_SIGNALS_NAME = f"hybrid_signals{cr.OUT_SUFFIX}.parquet"
HYBRID_SIGNALS_PATH = Path(cr.PATH_OUT) / HYBRID_SIGNALS_NAME


# ======================================================================
# 1) BUILDING HYBRID SIGNALS FROM ENHANCED PANEL (REGIME LAYER)
# ======================================================================

@dataclass
class RegimeConfig:
    # rolling window length in buckets (days if DECISION_FREQ='D')
    base_window: int = 5

    # which rolling features to build (all per-bucket)
    use_mean: bool = True
    use_std: bool = True
    use_slope: bool = True
    use_max_abs: bool = True

    # composite “health” weights (sign convention: higher = better)
    w_health_mean: float = 0.5
    w_health_std: float = -0.5
    w_health_slope: float = -0.5

    # allow override of bucket definitions if desired
    buckets: Optional[Dict[str, Tuple[float, float]]] = None


def _iter_enhanced_paths() -> List[Path]:
    """Return list of enhanced parquet paths respecting ENH_SUFFIX."""
    root = Path(cr.PATH_ENH)
    suffix = getattr(cr, "ENH_SUFFIX", "")
    return sorted(root.glob(f"*{suffix}.parquet"))


def _compute_z_slope(tenor: np.ndarray, z: np.ndarray) -> float:
    """OLS slope of z vs tenor, or NaN if not enough points."""
    mask = np.isfinite(tenor) & np.isfinite(z)
    tenor = tenor[mask]
    z = z[mask]
    if tenor.size < 3:
        return np.nan
    # center tenor to reduce numerical issues
    x = tenor - tenor.mean()
    try:
        beta, _ = np.polyfit(x, z, 1)
        return float(beta)
    except Exception:
        return np.nan


def _compute_cross_sectional_row(
    dts: pd.Timestamp,
    snap: pd.DataFrame,
    buckets: Dict[str, Tuple[float, float]],
) -> Dict:
    """
    Build one row of cross-sectional stats for a single decision_ts snapshot.
    snap must already be reduced to last tick per tenor_yrs.
    """
    out = {"decision_ts": dts}

    z = pd.to_numeric(snap["z_comb"], errors="coerce").to_numpy()
    ten = pd.to_numeric(snap["tenor_yrs"], errors="coerce").to_numpy()
    
    # --- Capture Scale (New) ---
    # Since scale is uniform for the bucket (mostly), mean is fine.
    if "scale" in snap.columns:
        out["scale_mean"] = float(snap["scale"].mean())

    # Core cross-sectional stats
    out["z_xs_mean"] = float(np.nanmean(z)) if np.isfinite(z).any() else np.nan
    out["z_xs_std"] = float(np.nanstd(z)) if np.isfinite(z).any() else np.nan
    out["z_xs_max_abs"] = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else np.nan
    out["z_xs_slope"] = _compute_z_slope(ten, z)

    # Bucket-level means
    for name, (lo, hi) in buckets.items():
        mask = (ten >= lo) & (ten <= hi)
        if not mask.any():
            out[f"z_{name}_mean"] = np.nan
        else:
            out[f"z_{name}_mean"] = float(np.nanmean(z[mask]))

    return out


def build_hybrid_signals(
    *,
    regime_cfg: Optional[RegimeConfig] = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Build (or load) hybrid_signals from enhanced parquets.
    Lag all regime metrics by 1 bucket to avoid lookahead.
    """
    regime_cfg = regime_cfg or RegimeConfig()
    buckets = regime_cfg.buckets or cr.BUCKETS

    if HYBRID_SIGNALS_PATH.exists() and not force_rebuild:
        return pd.read_parquet(HYBRID_SIGNALS_PATH)

    rows: List[Dict] = []
    paths = _iter_enhanced_paths()
    if not paths:
        raise FileNotFoundError(f"No enhanced files found under {cr.PATH_ENH}")

    for pth in paths:
        df = pd.read_parquet(pth)
        if df.empty:
            continue

        df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
        if "decision_ts" not in df.columns:
            if cr.DECISION_FREQ.upper() == "D":
                df["decision_ts"] = df["ts"].dt.floor("D")
            elif cr.DECISION_FREQ.upper() == "H":
                df["decision_ts"] = df["ts"].dt.floor("H")
            else:
                raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

        # Last tick per tenor_yrs per decision_ts
        df = (
            df.sort_values("ts")
              .groupby(["decision_ts", "tenor_yrs"], as_index=False)
              .tail(1)
        )

        for dts, snap in df.groupby("decision_ts", sort=True):
            if snap.empty:
                continue
            rows.append(_compute_cross_sectional_row(dts, snap, buckets))

    if not rows:
        raise RuntimeError("No cross-sectional rows built; check enhanced files and columns.")

    sig = pd.DataFrame(rows).sort_values("decision_ts").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Rolling regime metrics
    # ------------------------------------------------------------------
    win = int(regime_cfg.base_window)
    if win <= 1:
        raise ValueError("RegimeConfig.base_window must be > 1")

    sig = sig.set_index("decision_ts").sort_index()

    # Rolling Stats
    if regime_cfg.use_mean and "z_xs_mean" in sig.columns:
        sig["z_xs_mean_roll"] = sig["z_xs_mean"].rolling(win, min_periods=2).mean()
        # "Is the curve richness trending?"
        sig["z_xs_mean_roll_z"] = (
            (sig["z_xs_mean"] - sig["z_xs_mean_roll"])
            / sig["z_xs_mean"].rolling(win, min_periods=2).std(ddof=0)
        )

    if regime_cfg.use_std and "z_xs_std" in sig.columns:
        sig["z_xs_std_roll"] = sig["z_xs_std"].rolling(win, min_periods=2).mean()
        # "Is dispersion expanding?"
        sig["z_xs_std_roll_z"] = (
            (sig["z_xs_std"] - sig["z_xs_std_roll"])
            / sig["z_xs_std"].rolling(win, min_periods=2).std(ddof=0)
        )

    if regime_cfg.use_slope and "z_xs_slope" in sig.columns:
        sig["z_xs_slope_roll"] = sig["z_xs_slope"].rolling(win, min_periods=2).mean()
        # "Is the cheapness slope flipping?"
        sig["z_xs_slope_roll_z"] = (
            (sig["z_xs_slope"] - sig["z_xs_slope_roll"])
            / sig["z_xs_slope"].rolling(win, min_periods=2).std(ddof=0)
        )
        
    # NEW: Rolling Scale (Vol Regime)
    if "scale_mean" in sig.columns:
        sig["scale_roll"] = sig["scale_mean"].rolling(win, min_periods=2).mean()
        # "Is Vol spiking?"
        sig["scale_roll_z"] = (
            (sig["scale_mean"] - sig["scale_roll"]) 
            / sig["scale_mean"].rolling(win, min_periods=2).std(ddof=0)
        )

    # ------------------------------------------------------------------
    # Composite "signal_health" and "trendiness"
    # ------------------------------------------------------------------
    # Definition of Health: High Mean Reversion (Low Autocorrelation)
    # If Mean/Std/Slope are moving violently (High Z), health is LOW.
    health_terms = []
    if "z_xs_mean_roll_z" in sig.columns:
        health_terms.append(regime_cfg.w_health_mean * sig["z_xs_mean_roll_z"].pow(2) * -1.0)
    if "z_xs_std_roll_z" in sig.columns:
        health_terms.append(regime_cfg.w_health_std * sig["z_xs_std_roll_z"].pow(2) * -1.0)
    if "z_xs_slope_roll_z" in sig.columns:
        health_terms.append(regime_cfg.w_health_slope * sig["z_xs_slope_roll_z"].pow(2) * -1.0)

    if health_terms:
        sig["signal_health_raw"] = sum(health_terms)
        # normalize to Z-score
        mu = sig["signal_health_raw"].rolling(win, min_periods=2).mean()
        sd = sig["signal_health_raw"].rolling(win, min_periods=2).std(ddof=0)
        sig["signal_health_z"] = (sig["signal_health_raw"] - mu) / sd

    # Trendiness = How directional is the Z-score structure?
    if "z_xs_slope_roll_z" in sig.columns:
        sig["trendiness_abs"] = sig["z_xs_slope_roll_z"].abs()

    # ------------------------------------------------------------------
    # Avoid lookahead: shift all regime-derived signals by 1 bucket
    # ------------------------------------------------------------------
    # Everything calculated above uses data up to T. 
    # To trade at T, we must only use data from T-1 (Yesterday's close).
    regime_cols = [c for c in sig.columns if c not in ["z_xs_mean", "z_xs_std", "z_xs_slope", "z_xs_max_abs", "scale_mean"]]
    sig[regime_cols] = sig[regime_cols].shift(1)

    sig = sig.reset_index()

    HYBRID_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sig.to_parquet(HYBRID_SIGNALS_PATH, index=False)

    return sig


# ======================================================================
# 2) REGIME FILTER: TURN SIGNALS INTO MASKS OR MULTIPLIERS
# ======================================================================

@dataclass
class RegimeThresholds:
    # if not None, enforce these; if None, column ignored
    min_signal_health_z: Optional[float] = -0.5  # require health >= this
    max_trendiness_abs: Optional[float] = 2.0    # require |trendiness| <= this
    max_z_xs_mean_abs_z: Optional[float] = 2.0   # keep cross-sectional mean in bounds


def regime_mask_from_signals(
    signals: pd.DataFrame,
    *,
    thresholds: Optional[RegimeThresholds] = None,
) -> pd.Series:
    """
    Construct a boolean mask 'ok_to_trade' indexed by decision_ts from signals.
    """
    if thresholds is None:
        thresholds = RegimeThresholds(
            min_signal_health_z=float(getattr(cr, "MIN_SIGNAL_HEALTH_Z", -0.5)),
            max_trendiness_abs=float(getattr(cr, "MAX_TRENDINESS_ABS", 2.0)),
            max_z_xs_mean_abs_z=float(getattr(cr, "MAX_Z_XS_MEAN_ABS_Z", 2.0)),
        )

    sig = signals.copy()
    if "decision_ts" not in sig.columns:
        raise ValueError("signals must have a 'decision_ts' column.")

    sig = sig.set_index("decision_ts").sort_index()

    ok = pd.Series(True, index=sig.index, dtype=bool)

    if thresholds.min_signal_health_z is not None and "signal_health_z" in sig.columns:
        ok &= sig["signal_health_z"] >= thresholds.min_signal_health_z

    if thresholds.max_trendiness_abs is not None and "trendiness_abs" in sig.columns:
        ok &= sig["trendiness_abs"] <= thresholds.max_trendiness_abs

    if thresholds.max_z_xs_mean_abs_z is not None and "z_xs_mean_roll_z" in sig.columns:
        ok &= sig["z_xs_mean_roll_z"].abs() <= thresholds.max_z_xs_mean_abs_z

    ok.name = "ok_regime"
    return ok


# ======================================================================
# 3) CONVENIENCE WRAPPERS
# ======================================================================

def get_or_build_hybrid_signals(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Convenience wrapper: always use this to obtain hybrid_signals.
    """
    return build_hybrid_signals(force_rebuild=force_rebuild)
