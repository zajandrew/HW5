"""
hybrid_filter.py

Combined regime + shock filter logic for the overlay / strategy.

Responsibilities
----------------
1) Build and cache a time series of "hybrid_signals" based purely on the
   enhanced curve panel (no PnL here; curve environment only).

2) Regime filter:
   - From hybrid_signals, build composite "signal health", "trendiness", etc.
   - Produce a boolean series indicating when we consider the environment tradable.
   - Can pull defaults directly from cr_config.

3) Shock-blocker:
   - From closed-positions PnL and hybrid_signals, detect local PnL shocks 
     and block new risk for K buckets after each shock.
   - Can pull defaults directly from cr_config.
   - Used primarily for OFF-LINE analysis. (On-line logic is in portfolio_test).

All functions are written to avoid lookahead by design: any signal used to
decide about bucket t is lagged to t-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    base_window: int = 10

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

    # Mean / std rolling
    if regime_cfg.use_mean and "z_xs_mean" in sig.columns:
        sig["z_xs_mean_roll"] = sig["z_xs_mean"].rolling(win, min_periods=2).mean()
        sig["z_xs_mean_roll_z"] = (
            (sig["z_xs_mean"] - sig["z_xs_mean_roll"])
            / sig["z_xs_mean"].rolling(win, min_periods=2).std(ddof=0)
        )

    if regime_cfg.use_std and "z_xs_std" in sig.columns:
        sig["z_xs_std_roll"] = sig["z_xs_std"].rolling(win, min_periods=2).mean()
        sig["z_xs_std_roll_z"] = (
            (sig["z_xs_std"] - sig["z_xs_std_roll"])
            / sig["z_xs_std"].rolling(win, min_periods=2).std(ddof=0)
        )

    if regime_cfg.use_slope and "z_xs_slope" in sig.columns:
        sig["z_xs_slope_roll"] = sig["z_xs_slope"].rolling(win, min_periods=2).mean()
        sig["z_xs_slope_roll_z"] = (
            (sig["z_xs_slope"] - sig["z_xs_slope_roll"])
            / sig["z_xs_slope"].rolling(win, min_periods=2).std(ddof=0)
        )

    if regime_cfg.use_max_abs and "z_xs_max_abs" in sig.columns:
        sig["z_xs_max_abs_roll"] = sig["z_xs_max_abs"].rolling(win, min_periods=2).mean()
        sig["z_xs_max_abs_roll_z"] = (
            (sig["z_xs_max_abs"] - sig["z_xs_max_abs_roll"])
            / sig["z_xs_max_abs"].rolling(win, min_periods=2).std(ddof=0)
        )

    # ------------------------------------------------------------------
    # Composite "signal_health" and "trendiness"
    # ------------------------------------------------------------------
    health_terms = []
    if "z_xs_mean_roll_z" in sig.columns:
        health_terms.append(regime_cfg.w_health_mean * sig["z_xs_mean_roll_z"].pow(2) * -1.0)
    if "z_xs_std_roll_z" in sig.columns:
        health_terms.append(regime_cfg.w_health_std * sig["z_xs_std_roll_z"].pow(2) * -1.0)
    if "z_xs_slope_roll_z" in sig.columns:
        health_terms.append(regime_cfg.w_health_slope * sig["z_xs_slope_roll_z"].pow(2) * -1.0)

    if health_terms:
        sig["signal_health_raw"] = sum(health_terms)
        # normalize
        mu = sig["signal_health_raw"].rolling(win, min_periods=2).mean()
        sd = sig["signal_health_raw"].rolling(win, min_periods=2).std(ddof=0)
        sig["signal_health_z"] = (sig["signal_health_raw"] - mu) / sd

    if "z_xs_slope_roll_z" in sig.columns:
        sig["trendiness_abs"] = sig["z_xs_slope_roll_z"].abs()

    # ------------------------------------------------------------------
    # Avoid lookahead: shift all regime-derived signals by 1 bucket
    # ------------------------------------------------------------------
    regime_cols = [c for c in sig.columns if c not in ["z_xs_mean", "z_xs_std", "z_xs_slope", "z_xs_max_abs"]]
    sig[regime_cols] = sig[regime_cols].shift(1)

    sig = sig.reset_index()

    HYBRID_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sig.to_parquet(HYBRID_SIGNALS_PATH, index=False)

    return sig


# ======================================================================
# 2) REGIME FILTER: TURN SIGNALS INTO A SIMPLE "GOOD/BAD" MASK
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
    AUTO-LOADS defaults from cr_config if thresholds=None.
    """
    # --- NEW: Load defaults from config if missing ---
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
# 3) SHOCK BLOCKER: PNL-DRIVEN INTERMITTENT BLOCKS (NO LOOKAHEAD)
# ======================================================================

@dataclass
class ShockConfig:
    # horizon over which to compute rolling PnL stats (in buckets)
    pnl_window: int = 10

    # shock detection: we use both raw PnL and residual vs linear model
    use_raw_pnl: bool = True
    use_residuals: bool = True

    # thresholds: negative, in units of rolling std
    raw_pnl_z_thresh: float = -1.5
    resid_z_thresh: float = -1.5

    # which signal columns to regress on; if empty/None => raw PnL only
    regression_cols: Optional[Sequence[str]] = None

    # block length (K) after each detected shock, in buckets
    block_length: int = 10
    
    # --- NEW FIELDS ---
    metric_type: str = "BPS"      # "BPS" or "CASH"
    shock_mode: str = "ROLL_OFF"  # "ROLL_OFF" or "EXIT_ALL"


def _make_pnl_ts_from_closed(
    pos_overlay: pd.DataFrame,
    *,
    decision_freq: Optional[str] = None,
    pnl_col: str = "pnl_net_bp",
) -> pd.DataFrame:
    """
    Convert closed-positions df into a time series of PnL by decision bucket.
    """
    if pnl_col not in pos_overlay.columns:
        raise ValueError(f"pos_overlay missing column {pnl_col!r}")

    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()

    df = pos_overlay.copy()
    df["close_ts"] = pd.to_datetime(df["close_ts"], utc=False, errors="coerce")

    if decision_freq == "D":
        df["decision_ts"] = df["close_ts"].dt.floor("D")
    elif decision_freq == "H":
        df["decision_ts"] = df["close_ts"].dt.floor("H")
    else:
        raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

    pnl_ts = (
        df.groupby("decision_ts", as_index=False)[pnl_col]
          .sum()
          .rename(columns={pnl_col: "pnl"})
    )

    return pnl_ts.sort_values("decision_ts").reset_index(drop=True)


def run_shock_blocker(
    pos_overlay: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    shock_cfg: Optional[ShockConfig] = None,
    decision_freq: Optional[str] = None,
) -> Dict[str, object]:
    """
    Core shock-blocker logic (OFF-LINE ANALYSIS).
    AUTO-LOADS defaults from cr_config if shock_cfg=None.
    """
    # --- NEW: Load defaults from config if missing ---
    if shock_cfg is None:
        shock_cfg = ShockConfig(
            pnl_window=int(getattr(cr, "SHOCK_PNL_WINDOW", 10)),
            use_raw_pnl=bool(getattr(cr, "SHOCK_USE_RAW_PNL", True)),
            use_residuals=bool(getattr(cr, "SHOCK_USE_RESIDUALS", True)),
            raw_pnl_z_thresh=float(getattr(cr, "SHOCK_RAW_PNL_Z_THRESH", -1.5)),
            resid_z_thresh=float(getattr(cr, "SHOCK_RESID_Z_THRESH", -1.5)),
            regression_cols=list(getattr(cr, "SHOCK_REGRESSION_COLS", [])),
            block_length=int(getattr(cr, "SHOCK_BLOCK_LENGTH", 10)),
            metric_type=str(getattr(cr, "SHOCK_METRIC_TYPE", "BPS")),
            shock_mode=str(getattr(cr, "SHOCK_MODE", "ROLL_OFF"))
        )

    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()

    # 1) Build pnl_ts per decision bucket (BPS or CASH based on config)
    target_col = "pnl_net_bp" if shock_cfg.metric_type == "BPS" else "pnl_net_cash"
    pnl_ts = _make_pnl_ts_from_closed(
        pos_overlay,
        decision_freq=decision_freq,
        pnl_col=target_col,
    )

    # 2) Align with signals on union of decision_ts
    sig = signals.copy()
    if "decision_ts" not in sig.columns:
        raise ValueError("signals must have a 'decision_ts' column.")

    sig = sig.set_index("decision_ts").sort_index()
    pnl_ts = pnl_ts.set_index("decision_ts").sort_index()

    idx = sig.index.union(pnl_ts.index).sort_values()
    sig = sig.reindex(idx)
    pnl = pnl_ts.reindex(idx)["pnl"].fillna(0.0)

    # 3) Prepare design matrix X_t for regression (t-1, to avoid lookahead)
    if shock_cfg.regression_cols:
        cols = [c for c in shock_cfg.regression_cols if c in sig.columns]
    else:
        cols = []

    if cols:
        X = sig[cols].copy()
        X = X.astype(float)
        # lag X by 1 bucket: decision for t uses signals up to t-1 only
        X = X.shift(1)
    else:
        X = None

    # 4) Compute rolling PnL stats & (optionally) linear residuals
    win = int(shock_cfg.pnl_window)
    if win <= 1:
        raise ValueError("ShockConfig.pnl_window must be > 1")

    pnl_mu = pnl.rolling(win, min_periods=2).mean()
    pnl_sd = pnl.rolling(win, min_periods=2).std(ddof=0)
    pnl_z = (pnl - pnl_mu) / pnl_sd

    if X is not None:
        # rolling window regression
        resid = pd.Series(np.nan, index=idx, dtype=float)
        for i in range(len(idx)):
            end = i
            if end <= 0:
                continue
            start = max(0, end - win)
            y_win = pnl.iloc[start:end]
            X_win = X.iloc[start:end]
            mask = np.isfinite(y_win) & np.isfinite(X_win).all(axis=1)
            if mask.sum() < len(cols) + 1:
                continue
            y_sub = y_win[mask]
            X_sub = X_win[mask]
            # add intercept
            A = np.column_stack([np.ones(len(X_sub)), X_sub.values])
            try:
                beta, _, _, _ = np.linalg.lstsq(A, y_sub.values, rcond=None)
                y_hat = A @ beta
                resid.iloc[start:end] = y_sub.values - y_hat
            except Exception:
                continue

        resid_mu = resid.rolling(win, min_periods=2).mean()
        resid_sd = resid.rolling(win, min_periods=2).std(ddof=0)
        resid_z = (resid - resid_mu) / resid_sd
    else:
        resid = None
        resid_z = None

    # 5) Detect shocks
    shock = pd.Series(False, index=idx, dtype=bool)

    if shock_cfg.use_raw_pnl:
        shock_raw = pnl_z <= shock_cfg.raw_pnl_z_thresh
        shock |= shock_raw.fillna(False)

    if shock_cfg.use_residuals and resid_z is not None:
        shock_resid = resid_z <= shock_cfg.resid_z_thresh
        shock |= shock_resid.fillna(False)

    # 6) Build block mask
    block = pd.Series(False, index=idx, dtype=bool)
    K = int(shock_cfg.block_length)

    shock_indices = np.where(shock.values)[0]
    for s_idx in shock_indices:
        start = s_idx + 1      # block AFTER the shock bucket itself
        end = min(len(idx), s_idx + 1 + K)
        if start < end:
            block.iloc[start:end] = True

    block.name = "block"

    # 7) Apply block to PnL (for evaluation only)
    pnl_blocked = pnl.copy()
    pnl_blocked[block] = 0.0

    result = {
        "ts": idx,
        "pnl_raw": pnl,
        "pnl_blocked": pnl_blocked,
        "shock_mask": shock,
        "block_mask": block,
        "regression_cols": cols,
    }
    return result


# ======================================================================
# 4) SMALL CONVENIENCE WRAPPERS
# ======================================================================

def get_or_build_hybrid_signals(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Convenience wrapper: always use this to obtain hybrid_signals.
    """
    return build_hybrid_signals(force_rebuild=force_rebuild)


def attach_regime_and_shock_masks(
    pos_overlay: pd.DataFrame,
    *,
    regime_thresholds: Optional[RegimeThresholds] = None,
    shock_cfg: Optional[ShockConfig] = None,
    force_rebuild_signals: bool = False,
) -> Dict[str, object]:
    """
    High-level helper (OFFLINE).
    """
    signals = get_or_build_hybrid_signals(force_rebuild=force_rebuild_signals)

    # Regime mask
    reg_mask = regime_mask_from_signals(signals, thresholds=regime_thresholds)

    # Shock blocker
    shock_res = run_shock_blocker(pos_overlay, signals, shock_cfg=shock_cfg)

    out = {
        "signals": signals,
        "regime_mask": reg_mask,
        "shock_results": shock_res,
    }
    return out
