"""
instruments.py

Defines the stateful trading objects (Legs, Pairs, Flys).
Inherits calculation logic from math_core.py.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# Import Canon Math
import math_core as mc
# Import Config for constraints
import config as cr 

# ==============================================================================
# 1. ATOMIC UNIT: SWAP LEG
# ==============================================================================

@dataclass
class SwapLeg:
    """
    Represents a single position on the curve (e.g., Pay 10Y).
    Manages its own state, rate updates, and PnL accumulation.
    """
    tenor: float
    direction: float        # +1.0 (Receiver), -1.0 (Payer)
    notional: float         # Risk / DV01_Unit
    entry_rate: float       # In Percent (4.25)
    open_ts: pd.Timestamp
    
    # State Attributes
    curr_rate: float = field(init=False)
    curr_dv01: float = field(init=False)
    
    # Accumulated PnL
    pnl_price: float = 0.0
    pnl_carry: float = 0.0
    pnl_roll: float  = 0.0
    
    # Helpers
    tenor_orig: float = field(init=False)

    def __post_init__(self):
        self.curr_rate = self.entry_rate
        self.tenor_orig = self.tenor
        self.curr_dv01 = 0.0 # Will be set on first mark

    def mark(self, curve: mc.SplineCurve, dt_days: float):
        """
        Updates rates and accumulates PnL.
        """
        # 1. Update Market Data
        # We assume linear decay of tenor for the rate lookup (10y becomes 9.99y)
        # But for DV01 calc, we stick to the risk profile of the original tenor 
        # (Overlay style) or decaying tenor (Strategy style). 
        # Let's decay the tenor for lookup to capture rolldown physics.
        
        decay_years = (dt_days / 360.0) if dt_days > 0 else 0.0
        
        # Update the Remaining Tenor
        self.tenor = max(0.01, self.tenor - decay_years)
        t_lookup = self.tenor
        
        # 2. Update Rate & Risk
        self.curr_rate = curve.get_rate(t_lookup)
        
        # Risk Factor (DV01 per unit of notional)
        # We re-calculate this to capture the changing duration of the swap
        unit_dv01 = curve.get_dv01(t_lookup)
        
        # Total Risk ($ per bp)
        self.curr_dv01 = unit_dv01 * self.notional
        
        # 2. Price PnL (Snapshot vs Entry)
        self.pnl_price = mc.calc_price_pnl(
            self.entry_rate, 
            self.curr_rate, 
            self.curr_dv01, 
            self.direction
        )
        
        # 3. Incremental Carry & Roll (Path Dependent)
        if dt_days > 0:
            funding = curve.get_funding_rate()
            
            # Carry
            inc_carry = mc.calc_carry_pnl(
                self.entry_rate, 
                funding, 
                self.notional, 
                dt_days, 
                self.direction
            )
            self.pnl_carry += inc_carry
            
            # Rolldown (Theta)
            # What would the rate be if we were dt_days further down the curve?
            rolled_rate = curve.get_rate(max(0.01, t_lookup - (1/360.0)))
            
            inc_roll = mc.calc_rolldown_pnl(
                self.curr_rate, 
                rolled_rate, 
                self.curr_dv01, 
                self.direction
            )
            # We scale roll by days passed (Theta * dt)
            # Note: calc_rolldown returns the instantaneous 1-day roll.
            # We multiply by dt_days to approx the period.
            self.pnl_roll += (inc_roll * dt_days)

# ==============================================================================
# 2. CURVE TRADE: PAIR POSITION
# ==============================================================================

class PairPos:
    def __init__(
        self, 
        ts: pd.Timestamp, 
        leg_i: Dict, # {tenor, rate}
        leg_j: Dict, 
        curve: mc.SplineCurve,
        target_dv01: float,
        regime_meta: Dict = None
    ):
        self.open_ts = ts
        self.meta = regime_meta or {}
        self.scale_dv01 = target_dv01
        self.closed = False
        self.exit_reason = None
        self.last_mark_ts = ts
        self.txn_cost_bps = 0.1
        self.txn_cost_cash = self.txn_cost_bps * self.scale_dv01
        
        # 1. Determine Direction (Weights)
        # Standard: DV01 Neutral
        # w_i * dv01_i + w_j * dv01_j = 0
        
        t_i, r_i = leg_i['tenor'], leg_i['rate']
        t_j, r_j = leg_j['tenor'], leg_j['rate']
        
        unit_dv01_i = curve.get_dv01(t_i)
        unit_dv01_j = curve.get_dv01(t_j)
        
        # Default: Buy i / Sell j (Slope view? or Z-Score view?)
        # We assume leg_i is the "Primary" leg (e.g. the Cheap one we are buying)
        # So we want +DV01 on i
        
        # Notional = TargetRisk / UnitRisk
        not_i = target_dv01 / unit_dv01_i
        
        # Hedge Ratio
        hedge_ratio = unit_dv01_i / unit_dv01_j
        not_j = not_i * hedge_ratio
        
        # Directions (passed from strategy logic via leg dicts usually, or inferred)
        # If leg_i is "Long", dir_i = 1.
        dir_i = leg_i.get('direction', 1.0)
        dir_j = leg_j.get('direction', -1.0) # Opposite
        
        self.legs = [
            SwapLeg(t_i, dir_i, not_i, r_i, ts),
            SwapLeg(t_j, dir_j, not_j, r_j, ts)
        ]
        
        # Initial Stats
        self.entry_spread = (r_i * dir_i) + (r_j * dir_j) 
        # Note: Spread definition varies. Simplest is Rate_i - Rate_j.
        # But for PnL tracking, we sum the legs.
        
    def mark(self, curve: mc.SplineCurve, ts: pd.Timestamp):
        if self.closed: return
        
        delta = ts - self.last_mark_ts
        dt_days = delta.total_seconds() / 86400.0 
        
        for leg in self.legs: 
            leg.mark(curve, dt_days) # Pass fractional incremental time
            
        self.last_mark_ts = ts
        
    @property
    def pnl_total(self):
        return sum(l.pnl_price + l.pnl_carry for l in self.legs) - self.txn_cost_cash
        
    @property
    def pnl_bps(self):
        return self.pnl_total / self.scale_dv01

# ==============================================================================
# 3. BUTTERFLY TRADE: FLY POSITION
# ==============================================================================

class FlyPos:
    def __init__(
        self, 
        ts: pd.Timestamp, 
        belly: Dict, # {tenor, rate}
        left: Dict, 
        right: Dict, 
        curve: mc.SplineCurve,
        target_dv01: float, # Risk on the Belly
        weight_method: str = "convexity",
        regime_meta: Dict = None
    ):
        self.open_ts = ts
        self.meta = regime_meta or {}
        self.scale_dv01 = target_dv01
        self.closed = False
        self.exit_reason = None
        self.last_mark_ts = ts
        self.txn_cost_bps = 0.1
        self.txn_cost_cash = self.txn_cost_bps * self.scale_dv01
        
        # 1. Belly Setup
        t_b, r_b = belly['tenor'], belly['rate']
        unit_dv01_b = curve.get_dv01(t_b)
        not_b = target_dv01 / unit_dv01_b
        dir_b = belly.get('direction', 1.0) # e.g. Rec Belly (Long Fly)
        
        self.leg_belly = SwapLeg(t_b, dir_b, not_b, r_b, ts)
        
        # 2. Wing Weighting
        t_l, r_l = left['tenor'], left['rate']
        t_r, r_r = right['tenor'], right['rate']
        
        unit_dv01_l = curve.get_dv01(t_l)
        unit_dv01_r = curve.get_dv01(t_r)
        
        # Total Risk to Hedge = TargetDV01
        risk_to_hedge = target_dv01
        
        # Calculate Split Ratios (Distance weighted to kill Slope)
        # Ratio_L = (Tr - Tb) / (Tr - Tl)
        dist_total = t_r - t_l
        w_l_ratio = (t_r - t_b) / dist_total
        w_r_ratio = (t_b - t_l) / dist_total
        
        # Allocate Risk
        risk_l = risk_to_hedge * w_l_ratio
        risk_r = risk_to_hedge * w_r_ratio
        
        # Calculate Notionals
        not_l = risk_l / unit_dv01_l
        not_r = risk_r / unit_dv01_r
        
        # Wing Direction is opposite of Belly
        dir_wings = -1.0 * dir_b
        
        self.leg_left = SwapLeg(t_l, dir_wings, not_l, r_l, ts)
        self.leg_right = SwapLeg(t_r, dir_wings, not_r, r_r, ts)
        
        self.legs = [self.leg_belly, self.leg_left, self.leg_right]
        
    def mark(self, curve: mc.SplineCurve, ts: pd.Timestamp):
        if self.closed: return
        
        delta = ts - self.last_mark_ts
        dt_days = delta.total_seconds() / 86400.0 
        
        for leg in self.legs: 
            leg.mark(curve, dt_days) # Pass fractional incremental time
            
        self.last_mark_ts = ts

    @property
    def pnl_total(self):
        return sum(l.pnl_price + l.pnl_carry for l in self.legs) - self.txn_cost_cash
        
    @property
    def pnl_bps(self):
        return self.pnl_total / self.scale_dv01
