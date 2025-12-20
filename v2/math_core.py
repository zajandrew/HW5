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
import config as cr

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
        duration_proxy = tenor / denom
        return duration_proxy * 0.0001

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
def _map_instrument_to_tenor(instr: str) -> Optional[float]:
    if instr is None or not isinstance(instr, str): return None
    instr = instr.strip()
    mapped = cr.BBG_DICT.get(instr, instr)
    tenor = cr.TENOR_YEARS.get(mapped)
    return float(tenor) if tenor is not None else None

def clean_hedge_tape(raw_df: pd.DataFrame, decision_freq: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty: return pd.DataFrame()
    df = raw_df.copy()
    # Ensure UTC conversion is robust
    df["trade_ts"] = pd.to_datetime(df["tradetimeUTC"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    
    decision_freq = str(decision_freq).upper()
    if decision_freq == "D": df["decision_ts"] = df["trade_ts"].dt.floor("d")
    elif decision_freq == "H": df["decision_ts"] = df["trade_ts"].dt.floor("h")
    else: raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

    df["side"] = df["side"].astype(str).str.upper()
    df = df[df["side"].isin(["CPAY", "CRCV"])]
    df["tenor_yrs"] = df["instrument"].map(_map_instrument_to_tenor)
    df = df[np.isfinite(df["tenor_yrs"])]
    df["dv01"] = pd.to_numeric(df["EqVolDelta"], errors="coerce")
    df = df[np.isfinite(df["dv01"]) & (df["dv01"] > 0)]
    df = df.dropna(subset=["trade_ts", "decision_ts", "tenor_yrs", "dv01"])
    
    if "trade_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["trade_id"] = df.index.astype(int)
        
    cols = ["trade_id", "trade_ts", "decision_ts", "tenor_yrs", "side", "dv01"]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra]

# ==============================================================================
# 4. LIVE CURVE PARSER
# ==============================================================================
# Add this to math_core.py

def build_live_curve(
    row: pd.Series
) -> Optional[SplineCurve]:
    """
    Scrapes wide-format columns from your hedge tape row to build a SplineCurve.
    Uses your existing BBG_DICT and TENOR_YEARS to find columns.
    """
    tenors = []
    rates = []
    
    # Reverse map: We need to know that "10Y" corresponds to column "USOSFR10"
    # or just iterate the columns present in the row.
    
    # Robust Strategy: Iterate your known BBG keys (USOSFR1, etc.)
    for ticker_col, standard_key in cr.TENOR_YEARS.items():
        val = None
        
        # Check if the exact ticker is a column in the row
        if ticker_col in row.index:
            val = row[ticker_col]
        # Check for common suffixes if exact match fails
        elif f"{ticker_col} Curncy" in row.index:
            val = row[f"{ticker_col} Curncy"]
        elif f"{ticker_col}_mid" in row.index:
            val = row[f"{ticker_col}_mid"]
            
        if val is not None:
            try:
                f_val = float(val)
                # Sanity Check: Rates between -5% and 20%
                if np.isfinite(f_val) and -5.0 < f_val < 20.0:
                    t_yr = standard_key
                    if t_yr:
                        tenors.append(t_yr)
                        rates.append(f_val)
            except:
                continue
    
    # Need at least 3 points for a Cubic Spline
    if len(tenors) < 3: 
        return None
        
    try:
        return SplineCurve(np.array(tenors), np.array(rates))
    except:
        return None
