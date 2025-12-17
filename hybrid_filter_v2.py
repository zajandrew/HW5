"""
hybrid_filter.py

Regime filter logic for the Fly Overlay strategy.

Responsibilities
----------------
1) Consumes the HOURLY enhanced Z-scores (_enh.parquet).
2) Calculates Cross-Sectional Stats (Mean, Std, Slope) per hour.
3) Determines "Regime Health":
   - Is Volatility (Scale) spiking?
   - Are Z-scores trending (Momentum)?
   - Are Z-scores unstable (Flipping signs)?
4) Produces a boolean 'ok_regime' mask to enable/disable trading.

NOTE: All signals are LAGGED by 1 bucket (Shift=1) to prevent lookahead.
      We trade at 11:00 based on the Regime stats of 10:00.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import cr_config_new as cr

# ======================================================================
# Paths / Config
# ======================================================================

HYBRID_SIGNALS_NAME = f"hybrid_signals{getattr(cr, 'OUT_SUFFIX', '')}.parquet"
HYBRID_SIGNALS_PATH = Path(getattr(cr, "PATH_OUT", ".")) / HYBRID_SIGNALS_NAME

@dataclass
class RegimeConfig:
    # Rolling window in BUCKETS (Hours). 
    # 10-12 is good (covers ~1 trading day of context).
    base_window: int = 12

    # Which features to monitor
    use_mean: bool = True       # Is the curve generally Rich/Cheap?
    use_std: bool = True        # Is dispersion (opportunity) expanding?
    use_slope: bool = True      # Is the Z-structure trending (e.g. 2s vs 30s)?
    use_scale: bool = True      # Is raw volatility rising?

    # Health Weights (Negative = Stability is Good, Volatility is Bad)
    w_health_mean: float = 0.5   # Mean reversion is good
    w_health_std: float = -0.5   # Explosive dispersion is suspicious
    w_health_slope: float = -0.5 # Trending Z-scores = Macro Move (Bad for Fly)
    w_health_scale: float = -1.0 # Rising Scale = Rising Risk

    # Bucket definitions for granular stats (optional)
    buckets: Optional[Dict[str, Tuple[float, float]]] = None


def _iter_enhanced_paths() -> List[Path]:
    """Return list of enhanced parquet paths respecting ENH_SUFFIX."""
    root = Path(getattr(cr, "PATH_ENH", "."))
    suffix = getattr(cr, "ENH_SUFFIX", "")
    # Matches {yymm}_enh{suffix}.parquet
    return sorted(root.glob(f"*{suffix}.parquet"))


def _compute_z_slope(tenor: np.ndarray, z: np.ndarray) -> float:
    """OLS slope of Z-score vs Tenor. High slope = Directional Curve Move."""
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


def _compute_hourly_stats(
    dts: pd.Timestamp,
    snap: pd.DataFrame,
    bucket_map: Dict[str, Tuple[float, float]],
) -> Dict:
    """
    Calculates summary stats for ONE hourly bucket.
    """
    out = {"decision_ts": dts}

    # Extract Arrays
    z = pd.to_numeric(snap["z_comb"], errors="coerce").to_numpy()
    ten = pd.to_numeric(snap["tenor_yrs"], errors="coerce").to_numpy()
    
    # 1. Scale (Volatility) Check
    # In V2, 'scale' is constant for the bucket (PCA Scale), or per-tenor (Spline).
    # Taking the mean gives the average "Cost of Business" for this hour.
    if "scale" in snap.columns:
        out["scale_mean"] = float(snap["scale"].mean())
    else:
        out["scale_mean"] = np.nan

    # 2. Cross-Sectional Stats (The "Shape" of the Z-scores)
    # If Mean is High (+2.0): The whole curve is Cheap (Level dislocation).
    # If Slope is High: The curve is Steepening vs Model (Slope dislocation).
    # If Std is High: The curve is jagged (Fly opportunity).
    
    valid = np.isfinite(z)
    if valid.any():
        out["z_xs_mean"] = float(np.mean(z[valid]))
        out["z_xs_std"] = float(np.std(z[valid]))
        out["z_xs_max_abs"] = float(np.max(np.abs(z[valid])))
        out["z_xs_slope"] = _compute_z_slope(ten, z)
    else:
        out["z_xs_mean"] = np.nan
        out["z_xs_std"] = np.nan
        out["z_xs_max_abs"] = np.nan
        out["z_xs_slope"] = np.nan

    # 3. Sub-Bucket Stats (e.g. "Belly" specific heat)
    if bucket_map:
        for name, (lo, hi) in bucket_map.items():
            mask = (ten >= lo) & (ten <= hi)
            if mask.any():
                out[f"z_{name}_mean"] = float(np.nanmean(z[mask]))
            else:
                out[f"z_{name}_mean"] = np.nan

    return out


def build_hybrid_signals(
    *,
    regime_cfg: Optional[RegimeConfig] = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Consumes hourly _enh files and produces a regime time-series.
    """
    regime_cfg = regime_cfg or RegimeConfig()
    buckets = regime_cfg.buckets or getattr(cr, "BUCKETS", None)

    if HYBRID_SIGNALS_PATH.exists() and not force_rebuild:
        print(f"[REGIME] Loading cached signals: {HYBRID_SIGNALS_PATH}")
        return pd.read_parquet(HYBRID_SIGNALS_PATH)

    rows: List[Dict] = []
    paths = _iter_enhanced_paths()
    
    if not paths:
        print("[WARN] No enhanced files found. Regime filter cannot be built.")
        return pd.DataFrame()

    print(f"[REGIME] Building signals from {len(paths)} months of hourly data...")

    for pth in paths:
        try:
            df = pd.read_parquet(pth)
            if df.empty: continue
            
            # V2 Clean-up: The 'ts' column in _enh IS the bucket time.
            # We don't need complex grouping. Just ensure datetime type.
            df["ts"] = pd.to_datetime(df["ts"])
            
            # Group by Hour
            for dts, snap in df.groupby("ts"):
                rows.append(_compute_hourly_stats(dts, snap, buckets))
                
        except Exception as e:
            print(f"[WARN] Failed reading {pth}: {e}")

    if not rows:
        return pd.DataFrame()

    sig = pd.DataFrame(rows).sort_values("decision_ts").reset_index(drop=True)
    sig = sig.set_index("decision_ts").sort_index()

    # ------------------------------------------------------------------
    # Rolling Logic (The "Memory" of the Regime)
    # ------------------------------------------------------------------
    win = int(regime_cfg.base_window)
    
    # Helper for Z-score of a rolling window
    def roll_z(series):
        roll_mean = series.rolling(win, min_periods=2).mean()
        roll_std = series.rolling(win, min_periods=2).std(ddof=0)
        # Avoid division by zero
        return (series - roll_mean) / roll_std.replace(0, np.nan)

    # 1. Trendiness (Slope Stability)
    if regime_cfg.use_slope and "z_xs_slope" in sig.columns:
        sig["z_slope_roll_z"] = roll_z(sig["z_xs_slope"])
        # If Slope Z is high, the curve is strictly trending (Steepening/Flattening).
        # We generally want to pause Fly trading during massive macro trends.
        sig["trendiness_abs"] = sig["z_slope_roll_z"].abs()

    # 2. Volatility Expansion (Scale Stability)
    if regime_cfg.use_scale and "scale_mean" in sig.columns:
        sig["scale_roll_z"] = roll_z(sig["scale_mean"])
        
    # 3. Level Stability (Mean Stability)
    if regime_cfg.use_mean and "z_xs_mean" in sig.columns:
        sig["z_mean_roll_z"] = roll_z(sig["z_xs_mean"])

    # 4. Dispersion Stability (Std Stability)
    if regime_cfg.use_std and "z_xs_std" in sig.columns:
        sig["z_std_roll_z"] = roll_z(sig["z_xs_std"])

    # ------------------------------------------------------------------
    # Composite "Signal Health"
    # ------------------------------------------------------------------
    # We sum the negative squares of the instabilities.
    # Logic: If any metric is moving 3 sigmas (9 squared), it penalizes health heavily.
    health_terms = []
    
    if "z_mean_roll_z" in sig.columns:
        health_terms.append(regime_cfg.w_health_mean * sig["z_mean_roll_z"].pow(2))
        
    if "z_std_roll_z" in sig.columns:
        health_terms.append(regime_cfg.w_health_std * sig["z_std_roll_z"].pow(2))
        
    if "z_slope_roll_z" in sig.columns:
        health_terms.append(regime_cfg.w_health_slope * sig["z_slope_roll_z"].pow(2))
        
    if "scale_roll_z" in sig.columns:
        health_terms.append(regime_cfg.w_health_scale * sig["scale_roll_z"].pow(2))

    if health_terms:
        # Sum of penalties (e.g. -0.5 * 3^2 = -4.5)
        sig["signal_health_raw"] = sum(health_terms)
        
        # Normalize Health to a Z-score for easier thresholding
        # (e.g. "Health is -2.0 sigmas below normal")
        sig["signal_health_z"] = roll_z(sig["signal_health_raw"])

    # ------------------------------------------------------------------
    # CRITICAL: Lag by 1 Bucket (No Lookahead)
    # ------------------------------------------------------------------
    # To decide on 11:00 AM trade, we use stats derived from 10:00 AM data.
    # shift(1) moves the 10:00 row to the 11:00 index slot.
    cols_to_lag = [c for c in sig.columns if c != "decision_ts"]
    sig[cols_to_lag] = sig[cols_to_lag].shift(1)

    sig = sig.dropna(how='all').reset_index()

    # Save
    HYBRID_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sig.to_parquet(HYBRID_SIGNALS_PATH, index=False)
    print(f"[REGIME] Saved hybrid signals: {HYBRID_SIGNALS_PATH}")

    return sig


# ======================================================================
# REGIME MASK (The "Go/No-Go" Switch)
# ======================================================================

@dataclass
class RegimeThresholds:
    min_health_z: float = -1.5    # Kill switch if composite health is terrible
    max_trendiness: float = 2.5   # Kill switch if macro trend is massive
    max_vol_z: float = 3.0        # Kill switch if Vol (Scale) is exploding


def regime_mask_from_signals(
    signals: pd.DataFrame,
    thresholds: Optional[RegimeThresholds] = None,
) -> pd.Series:
    """
    Returns a Boolean Series (True = Trade, False = Block).
    Indexed by decision_ts.
    """
    if thresholds is None:
        thresholds = RegimeThresholds()

    sig = signals.set_index("decision_ts").sort_index()
    ok = pd.Series(True, index=sig.index, dtype=bool)

    # 1. Health Check
    if "signal_health_z" in sig.columns:
        ok &= (sig["signal_health_z"] >= thresholds.min_health_z)

    # 2. Trend Check (Slope)
    if "trendiness_abs" in sig.columns:
        ok &= (sig["trendiness_abs"] <= thresholds.max_trendiness)

    # 3. Volatility Check (Scale)
    # If scale is rising 3 sigmas faster than normal, market is panicking.
    if "scale_roll_z" in sig.columns:
        ok &= (sig["scale_roll_z"] <= thresholds.max_vol_z)

    ok.name = "ok_regime"
    return ok

def get_or_build_hybrid_signals(force_rebuild: bool = False) -> pd.DataFrame:
    return build_hybrid_signals(force_rebuild=force_rebuild)

if __name__ == "__main__":
    # Test run
    df = build_hybrid_signals(force_rebuild=True)
    print(df.tail())
