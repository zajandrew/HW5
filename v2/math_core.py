"""
math_core.py

The mathematical engine for the All-Weather RV Strategy.
Responsibilities:
1. Yield Curve Construction (Cubic Splines).
2. Canonical PnL Calculations (Price, Carry, Rolldown).
3. Hedge Tape Parsing & Validation.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from typing import Tuple, Optional, Dict, Union

# ==============================================================================
# 1. YIELD CURVE ENGINE
# ==============================================================================

class SplineCurve:
    """
    A high-performance wrapper around Scipy's CubicSpline.
    Handles sorting, funding rate extraction, and DV01 calculation.
    """
    def __init__(self, tenors: np.ndarray, rates: np.ndarray):
        """
        Args:
            tenors: Array of years (floats).
            rates: Array of rates in PERCENT (e.g., 4.25 for 4.25%).
        """
        # 1. Sort Data (Crucial for Spline)
        if len(tenors) != len(rates):
            raise ValueError("Tenor and Rate arrays must be same length")
            
        sort_idx = np.argsort(tenors)
        self._t = tenors[sort_idx]
        self._r = rates[sort_idx]
        
        # 2. Fit Spline (Natural boundary conditions often best for rates)
        # We use rates directly (4.25). 
        self.spline = CubicSpline(self._t, self._r, bc_type='natural')
        
        # 3. Cache Funding Rate (Shortest available tenor)
        self.funding_rate = self._r[0] if len(self._r) > 0 else 0.0

    def get_rate(self, tenor: float) -> float:
        """Returns Interpolated Rate in Percent."""
        return float(self.spline(tenor))

    def get_rates(self, tenors: np.ndarray) -> np.ndarray:
        """Vectorized version of get_rate."""
        return self.spline(tenors)

    def get_dv01(self, tenor: float) -> float:
        """
        Calculates exact DV01 for a generic par swap at this tenor/rate.
        Formula Approx: PV01 = (1 - (1+r)^-n) / r
        But for relative value, a simpler Risk proxy is often: Tenor / (1 + Rate)
        
        Returns:
            Float representing 'Risk Factor' (approx Duration).
        """
        r_dec = self.get_rate(tenor) / 100.0
        # Protect against negative/zero rates for division
        denom = max(1e-4, 1.0 + r_dec)
        return tenor / denom

    def get_funding_rate(self) -> float:
        return self.funding_rate

# ==============================================================================
# 2. CANONICAL PNL CALCULATIONS
# ==============================================================================

def calc_price_pnl(
    entry_rate: float,
    curr_rate: float,
    curr_dv01: float,
    direction: float
) -> float:
    """
    Calculates Capital Gains due to rate movement (Delta).
    Args:
        entry_rate: % (e.g. 4.00)
        curr_rate: % (e.g. 4.10)
        curr_dv01: Dollar Value of 1bp move
        direction: +1.0 (Receiver/Long), -1.0 (Payer/Short)
    Returns:
        Cash PnL
    """
    # If Receiver (+1): Rate falls -> Profit
    # If Payer (-1): Rate rises -> Profit
    # Formula: (Entry - Current) * 100 * DV01 * Dir
    # Note: If Dir is already +1/-1 attached to DV01, logic simplifies.
    # Assuming 'direction' is separate here.
    return (entry_rate - curr_rate) * 100.0 * curr_dv01 * direction


def calc_carry_pnl(
    entry_rate: float,
    funding_rate: float,
    initial_notional: float,
    days_held: float,
    direction: float
) -> float:
    """
    Calculates Net Interest Income (Carry).
    Args:
        entry_rate: Fixed Coupon % (e.g. 4.25)
        funding_rate: Floating Rate % (e.g. 4.00)
        initial_notional: Trade size / Tenor (approx)
        days_held: Number of days for this specific accrual period
        direction: +1.0 (Rec Fixed), -1.0 (Pay Fixed)
    """
    # Receiver (+1): Earns Fixed (Entry), Pays Float (Funding)
    # Payer (-1): Pays Fixed (Entry), Earns Float (Funding)
    
    # Rate Diff * Notional * Time
    # (Rate - Funding) is the "Spread" earned by a Receiver
    spread = (entry_rate - funding_rate) / 100.0 # Decimals
    time_frac = days_held / 360.0
    
    return spread * initial_notional * time_frac * direction


def calc_rolldown_pnl(
    curr_rate: float,
    rolled_rate: float,
    curr_dv01: float,
    direction: float
) -> float:
    """
    Calculates PnL due to the passage of time on the curve shape (Theta).
    Args:
        curr_rate: Rate at tenor T (e.g. 7.00y)
        rolled_rate: Rate at tenor T-dt (e.g. 6.99y)
        direction: +1.0 (Rec), -1.0 (Pay)
    """
    # If curve is steep (upward sloping), rolled rate is lower.
    # Receiver (+1): "Sold" high yield, "Buys back" lower yield -> Profit.
    # (Curr - Rolled) is Positive.
    return (curr_rate - rolled_rate) * 100.0 * curr_dv01 * direction


def calc_signal_drift(
    tenor: float,
    direction: float,
    curve: SplineCurve
) -> float:
    """
    Calculates the expected 1-day PnL (Carry + Roll) in Basis Points.
    Used for ENTRY SCORING (Forward looking).
    """
    dt = 1.0 / 360.0
    rate = curve.get_rate(tenor)
    funding = curve.get_funding_rate()
    
    # 1. Carry Bps (Rate - Fund)
    carry_bps = (rate - funding) * 100.0 * dt
    
    # 2. Roll Bps (Rate - RolledRate) * 100
    rolled_rate = curve.get_rate(tenor - dt)
    roll_bps = (rate - rolled_rate) * 100.0
    
    # Total Drift
    return (carry_bps + roll_bps) * direction

# ==============================================================================
# 3. TAPE INGESTION
# ==============================================================================

def clean_hedge_tape(
    df: pd.DataFrame, 
    decision_freq: str = "H",
    ticker_map: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Standardizes the hedge tape columns and types.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    out = df.copy()
    
    # 1. Time Normalization
    # Support 'tradetimeUTC' (common) or 'ts'
    if "tradetimeUTC" in out.columns:
        out["trade_ts"] = pd.to_datetime(out["tradetimeUTC"], utc=True).dt.tz_localize(None)
    elif "ts" in out.columns:
        out["trade_ts"] = pd.to_datetime(out["ts"], utc=True).dt.tz_localize(None)
    else:
        # Fallback if no time provided (rare)
        out["trade_ts"] = pd.Timestamp.utcnow().tz_localize(None)
        
    # 2. Decision Bucket
    freq = decision_freq.upper()
    if freq == "H":
        out["decision_ts"] = out["trade_ts"].dt.floor("h")
    else:
        out["decision_ts"] = out["trade_ts"].dt.floor("d")
        
    # 3. Instrument / Tenor Mapping
    if "instrument" in out.columns and ticker_map:
        # Helper to map "USOSFR10" -> 10.0
        def _map(x):
            s = str(x).strip()
            return float(ticker_map.get(s, ticker_map.get(s.split()[0], np.nan)))
        
        out["tenor_yrs"] = out["instrument"].apply(_map)
    elif "tenor_yrs" not in out.columns:
        # If no mapping provided, assume tenor_yrs must exist or fail
        return pd.DataFrame() # Fail safe
        
    out = out.dropna(subset=["tenor_yrs"])
    
    # 4. Side Normalization
    # Expect: "CPAY" (Payer), "CRCV" (Receiver), or "Pay"/"Rec"
    if "side" in out.columns:
        out["side"] = out["side"].astype(str).str.upper()
        # Filter for valid swaps only
        valid_sides = ["CPAY", "CRCV", "PAY", "REC", "PAYER", "RECEIVER"]
        out = out[out["side"].isin(valid_sides)]
    
    # 5. DV01 / Risk
    # Map 'EqVolDelta' or 'dv01' to 'dv01'
    if "EqVolDelta" in out.columns:
        out["dv01"] = pd.to_numeric(out["EqVolDelta"], errors="coerce").abs()
    elif "dv01" in out.columns:
        out["dv01"] = pd.to_numeric(out["dv01"], errors="coerce").abs()
    
    out = out.dropna(subset=["dv01"])
    out = out[out["dv01"] > 1.0] # Filter noise
    
    # 6. ID
    if "trade_id" not in out.columns:
        out = out.reset_index(drop=True)
        out["trade_id"] = out.index.astype(int)

    # 7. Return Clean Subset
    cols = ["decision_ts", "trade_ts", "trade_id", "tenor_yrs", "side", "dv01"]
    # Preserve other metadata if exists
    extra = [c for c in out.columns if c not in cols]
    
    return out[cols + extra]
