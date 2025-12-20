"""
regime_engine.py

A generalized framework for converting raw market features into 
strategy-specific Regime Multipliers.

Supports both Mean Reversion (Fade) and Momentum (Trend) logic
by decoupling 'Signal Generation' from 'Strategy Scoring'.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
import numpy as np
import pandas as pd
import cr_config_new as cr

# ======================================================================
# 1. GENERIC SIGNAL EXTRACTORS
# ======================================================================

def _acceleration(x: np.ndarray) -> float:
    """
    Calculates the change in slope (2nd Derivative).
    Returns positive if accelerating up, negative if decelerating/curving down.
    """
    valid = x[np.isfinite(x)]
    if len(valid) < 5: return np.nan
    
    # Split the window in half to compare recent slope vs old slope
    mid = len(valid) // 2
    first_half = valid[:mid]
    second_half = valid[mid:]
    
    slope_1 = _slope(first_half)
    slope_2 = _slope(second_half)
    
    if np.isnan(slope_1) or np.isnan(slope_2):
        return np.nan
        
    return slope_2 - slope_1

def _slope(x: np.ndarray) -> float:
    """Calculates linear slope of an array."""
    valid = x[np.isfinite(x)]
    if len(valid) < 3: return np.nan
    # Center x to avoid numerical issues
    t = np.arange(len(valid))
    t = t - t.mean()
    try:
        slope, _ = np.polyfit(t, valid, 1)
        return float(slope)
    except: return np.nan

def _max_abs(x: np.ndarray) -> float:
    valid = x[np.isfinite(x)]
    if not len(valid): return np.nan
    return float(np.max(np.abs(valid)))

def _clean_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x))

# Map string names to functions for easy config
AGG_FUNCS = {
    "mean": _clean_mean,
    "std": np.nanstd,
    "max": np.nanmax,
    "min": np.nanmin,
    "max_abs": _max_abs,
    "slope": _slope,
    "accel": _accel
}

# ======================================================================
# 2. CONFIGURATION OBJECTS
# ======================================================================

@dataclass
class SignalDef:
    """Defines a raw signal to extract from the hourly snapshot."""
    input_col: str          # e.g., 'hurst', 'z_comb', 'scale'
    agg_func: str           # e.g., 'mean', 'max', 'slope'
    output_name: str        # e.g., 'hurst_max', 'z_slope'
    rolling_window: int = 12 # Smoothing window (0 = no smoothing)

@dataclass
class RegimeRule:
    """Defines how a signal impacts a specific strategy."""
    signal_name: str        # Must match an output_name from SignalDef
    condition: str          # 'min', 'max', 'target_low', 'target_high'
    threshold: float
    impact: str = "multiplier" # 'multiplier' or 'additive'
    weight: float = 1.0     # The value to apply

    # Logic:
    # if condition='max' and val > threshold: apply weight (e.g. 0.0 to kill)
    # if condition='min' and val < threshold: apply weight

@dataclass
class StrategyProfile:
    """A collection of rules for ONE strategy type."""
    name: str # e.g., "MeanReversion_Fly", "Momentum_Curve"
    rules: List[RegimeRule]
    base_score: float = 1.0 # Default state

# ======================================================================
# 3. THE ENGINE
# ======================================================================

class RegimeEngine:
    def __init__(self, signals: List[SignalDef], strategies: List[StrategyProfile]):
        self.signal_defs = signals
        self.strategies = strategies

    def _compute_snapshot_stats(self, dts: pd.Timestamp, snap: pd.DataFrame) -> Dict:
        """Turns one hourly dataframe (Tenor x Cols) into a flat dict of stats."""
        row = {"decision_ts": dts}
        
        for sig in self.signal_defs:
            if sig.input_col not in snap.columns:
                row[sig.output_name] = np.nan
                continue
            
            arr = pd.to_numeric(snap[sig.input_col], errors='coerce').values
            func = AGG_FUNCS.get(sig.agg_func, _clean_mean)
            row[sig.output_name] = func(arr)
            
        return row

    def build_signals(self, enh_paths: List[Path]) -> pd.DataFrame:
        """Iterates over files and builds the raw signal history."""
        rows = []
        print(f"[REGIME] Extracting features from {len(enh_paths)} files...")
        
        for pth in enh_paths:
            try:
                df = pd.read_parquet(pth)
                if df.empty: continue
                df["ts"] = pd.to_datetime(df["ts"])
                
                # Group by Timestamp (Hourly Snapshot)
                for dts, snap in df.groupby("ts"):
                    rows.append(self._compute_snapshot_stats(dts, snap))
            except Exception as e:
                print(f"[WARN] Failed {pth}: {e}")
                
        if not rows: return pd.DataFrame()
        
        df_sig = pd.DataFrame(rows).sort_values("decision_ts").set_index("decision_ts")
        
        # Apply Rolling Smoothing
        for sig in self.signal_defs:
            if sig.rolling_window > 1:
                col = sig.output_name
                # Z-Score normalization optional? Let's stick to raw levels for explicit thresholds
                # or just simple moving average for noise reduction
                df_sig[f"{col}_smooth"] = df_sig[col].rolling(sig.rolling_window, min_periods=1).mean()
        
        return df_sig

    def score_strategies(self, df_signals: pd.DataFrame) -> pd.DataFrame:
        """Applies strategy rules to generate multipliers."""
        scores = pd.DataFrame(index=df_signals.index)
        
        # Lag Signals (CRITICAL: No Lookahead)
        # We trade at 11:00 using data from 10:00
        df_lagged = df_signals.shift(1) 

        for strat in self.strategies:
            # Start with base score
            s_series = pd.Series(strat.base_score, index=df_lagged.index)
            
            for rule in strat.rules:
                # Handle smoothed vs raw names
                col = rule.signal_name
                if f"{col}_smooth" in df_lagged.columns:
                    col = f"{col}_smooth"
                
                if col not in df_lagged.columns: continue
                
                val = df_lagged[col]
                
                # Apply Logic
                mask = pd.Series(False, index=df_lagged.index)
                
                if rule.condition == "max":     # Bad if Value > Threshold
                    mask = val > rule.threshold
                elif rule.condition == "min":   # Bad if Value < Threshold
                    mask = val < rule.threshold
                
                # Apply Impact
                if rule.impact == "multiplier":
                    # e.g. weight 0.0 (Kill) or 1.5 (Boost)
                    s_series.loc[mask] *= rule.weight
                elif rule.impact == "additive":
                    s_series.loc[mask] += rule.weight
            
            scores[f"mult_{strat.name}"] = s_series
            
        return pd.concat([df_signals, scores], axis=1)

# ======================================================================
# 4. FACTORY: Define Your Logic Here
# ======================================================================

def get_default_engine() -> RegimeEngine:
    """
    Constructs the specific logic for your RV Strategies.
    """
    
    # --- A. Define Features ---
    signals = [
        # Hurst: The Regime Detector
        SignalDef("hurst", "max", "hurst_max", rolling_window=3), 
        SignalDef("hurst", "mean", "hurst_mean", rolling_window=3),
        
        # Scale: The Risk Detector
        SignalDef("scale", "mean", "scale_mean", rolling_window=12),
        
        # Z-Scores: The Shape Detector
        SignalDef("z_comb", "slope", "z_slope", rolling_window=6), # Trendiness
        SignalDef("z_comb", "std", "z_dispersion", rolling_window=6) # Opportunity Size
    ]
    
    # --- B. Define Strategies ---
    
    # Strategy 1: PCA-Neutral Butterfly (Mean Reversion)
    # Logic: Hates trends (High Hurst), Hates High Vol (Scale)
    strat_rv = StrategyProfile(
        name="MeanReversion",
        base_score=1.0,
        rules=[
            # If Max Hurst > 0.55 (Trending), Kill the trade (Mult=0)
            RegimeRule("hurst_max", "max", 0.55, "multiplier", 0.0),
            
            # If Vol is crazy (> 2.0 scale), Cut size in half
            RegimeRule("scale_mean", "max", 2.0, "multiplier", 0.5),
            
            # If Slope is extreme (Curve is shifting hard), Pause
            RegimeRule("z_slope", "max", 0.5, "multiplier", 0.0), # Example threshold
             RegimeRule("z_slope", "min", -0.5, "multiplier", 0.0),
        ]
    )
    
    # Strategy 2: Directional Curve (Momentum)
    # Logic: Loves trends (High Hurst), Tolerates Vol
    strat_mom = StrategyProfile(
        name="Momentum",
        base_score=0.0, # Default to OFF, only turn on when trend detected
        rules=[
            # If Hurst > 0.60, We have a trend. Enable (Base=0 + 1 = 1)
            RegimeRule("hurst_max", "min", 0.60, "additive", 1.0),
            
            # If Vol is high, that's actually okay for momentum, maybe boost?
            RegimeRule("scale_mean", "min", 1.5, "multiplier", 1.2),
        ]
    )
    
    return RegimeEngine(signals, [strat_rv, strat_mom])

# ======================================================================
# 5. RUNNER
# ======================================================================

def run_regime_pipeline():
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    suffix = getattr(cr, "ENH_SUFFIX", "")
    paths = sorted(path_enh.glob(f"*{suffix}.parquet"))
    
    engine = get_default_engine()
    
    # 1. Build Signals
    df_sig = engine.build_signals(paths)
    
    # 2. Score Strategies
    df_scored = engine.score_strategies(df_sig)
    
    # 3. Save
    out_path = Path(getattr(cr, "PATH_OUT", ".")) / f"regime_multipliers{suffix}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_parquet(out_path)
    
    print(f"[DONE] Saved Regime Multipliers to {out_path}")
    print(df_scored[['mult_MeanReversion', 'mult_Momentum']].tail(10))

if __name__ == "__main__":
    run_regime_pipeline()
