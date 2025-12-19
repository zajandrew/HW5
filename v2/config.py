"""
config.py

Central configuration for the All-Weather RV Strategy.
Holds paths, physics constants, feature parameters, and strategy logic.

SECTIONS:
1. I/O & PATHS
2. GLOBAL PHYSICS & MAPPINGS
3. FEATURE CREATION SETTINGS
4. REGIME DEFINITIONS
5. STRATEGY PARAMETERS (PAIRS VS FLYS)
"""

from pathlib import Path

# ==============================================================================
# 1. FILE PATHS & I/O
# ==============================================================================
BASE_DIR = Path(".")
PATH_DATA = BASE_DIR / "data" / "features"   # Raw Feature Inputs
PATH_ENH  = BASE_DIR / "data" / "enhanced"   # Hourly Enhanced Data (Anchor)
PATH_OUT  = BASE_DIR / "data" / "output"     # Backtest Results

# File Naming Conventions
ENH_SUFFIX = ""                # Suffix for intermediate files (e.g., _v1)
OUT_SUFFIX = "_v1_allweather"  # Suffix for final output ledgers
TRADE_TYPES = "trades"         # Name of hedge tape pickle file (w/o extension)

# ==============================================================================
# 2. GLOBAL PHYSICS & MAPPINGS
# ==============================================================================
DECISION_FREQ = "H"   # 'D' for Daily, 'H' for Hourly
CALC_METHOD   = "cubic" # Spline interpolation method

# Standard Tenor Map (Internal Keys -> Years)
TENOR_YEARS = {
    "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, 
    "7Y": 7.0, "10Y": 10.0, "12Y": 12.0, "15Y": 15.0,
    "20Y": 20.0, "25Y": 25.0, "30Y": 30.0
}

# Bloomberg Ticker Map (Tape Columns -> Internal Keys)
# Used by clean_hedge_tape to parse raw input
BBG_DICT = {
    "USOSFR1": "1Y",  "USOSFR1 Curncy": "1Y",
    "USOSFR2": "2Y",  "USOSFR2 Curncy": "2Y",
    "USOSFR3": "3Y",  "USOSFR3 Curncy": "3Y",
    "USOSFR5": "5Y",  "USOSFR5 Curncy": "5Y",
    "USOSFR7": "7Y",  "USOSFR7 Curncy": "7Y",
    "USOSFR10": "10Y", "USOSFR10 Curncy": "10Y",
    "USOSFR12": "12Y", "USOSFR12 Curncy": "12Y",
    "USOSFR15": "15Y", "USOSFR15 Curncy": "15Y",
    "USOSFR20": "20Y", "USOSFR20 Curncy": "20Y",
    "USOSFR25": "25Y", "USOSFR25 Curncy": "25Y",
    "USOSFR30": "30Y", "USOSFR30 Curncy": "30Y",
}

# ==============================================================================
# 3. FEATURE CREATION SETTINGS
# ==============================================================================
# PCA: 6 Months (~126 trading days)
PCA_LOOKBACK_DAYS = 126   

# Hurst: 6 Months (~126 trading days)
# We want the regime signal to be aligned with our PCA model
HURST_WINDOW      = 126    ## Not implemented, assumed to be the same as PCA window

# Volatility: 1 Month (~21 days)
VOL_LOOKBACK      = 21

# ==============================================================================
# 4. REGIME DEFINITIONS (The "Brain")
# ==============================================================================
STRATEGY_MODE = "both" # Options: 'curve', 'fly', 'both'

REGIME_CONFIG = {
    # MOMENTUM REGIME (High Trend) -> Favors Curve Trades
    "curve": {
        "threshold": {
            "hurst_max": (0.45, "greater"), # Hurst > 0.45 enables Curve
        },
        "multiplier": {
            # In high vol, we widen the entry gate (require higher signal)
            "scale_mean": (1.5, "multiply", 1.2), 
        }
    },
    # REVERSION REGIME (Low Trend) -> Favors Fly Trades
    "fly": {
        "threshold": {
            "hurst_max": (0.55, "less"),    # Hurst < 0.55 enables Flys
            "scale_mean": (2.5, "less")     # Block Flys in extreme crisis Vol
        },
        "multiplier": {
            "scale_mean": (0.5, "multiply", 0.8) # Tighten gates in calm markets
        }
    },
    # ARBITRATION LOGIC
    "both": {
        "priority": {
            "signal": "hurst_max",
            "split_level": 0.55,      # Above 0.55 = Trend (Curve), Below = Reversion (Fly)
            "above_split": "curve",
            "below_split": "fly"
        }
    }
}

# ==============================================================================
# 5. STRATEGY PARAMETERS
# ==============================================================================
# Global Limits
MIN_TENOR = 0.5
MAX_CONCURRENT = 50

# --- STRATEGY A: CURVE (PAIRS) - MOMENTUM LOGIC ---
PARAMS_PAIR = {
    "Z_ENTRY": 0.50,          # Lower entry bar allowed because we require Trend Alignment
    "Z_EXIT_MOMENTUM": -0.1,  # Exit when Trend Flips (Momentum < -0.1)
    "Z_STOP": 3.0,            # Hard Stop (Approx 30bps)
    
    "DRIFT_GATE_BPS": 0.0,    # Momentum trades just need positive carry (not massive)
    "DRIFT_WEIGHT": 0.50,     # We weight Carry heavily to ensure we are paid to wait
    
    "MOMENTUM_WINDOW": 10,    # Lookback for Trend Calculation (Days)
    "PIVOT_POINT": 5.0        # Segmentation: Trade <5Y vs <5Y OR >5Y vs >5Y. No crossing.
}

# --- STRATEGY B: FLY (BUTTERFLIES) - MEAN REVERSION LOGIC ---
PARAMS_FLY = {
    "Z_ENTRY": 1.25,          # High entry bar (Require extreme dislocation)
    "Z_EXIT_REVERSION": 0.25, # Exit when Fair Value is restored (Z -> 0)
    "Z_STOP": 3.0,            # Hard Stop
    
    "DRIFT_GATE_BPS": -2.0,   # Can tolerate slight negative drift if Z is huge
    "DRIFT_WEIGHT": 0.20,     # Weight Z-score (Reversion) higher than Carry
    
    "CONVEXITY_PREMIUM_BPS": 2.0, # Extra drift required to sell Gamma
    "MAX_HALFLIFE_DAYS": 20.0,    # "Zombie Filter": Kill trade if reversion takes >20 days
    
    "FLY_WING_WIDTH": (1.5, 7.0), # Min/Max wing width in years
    "FLY_WEIGHT_METHOD": "convexity" # Neutralize slope/curvature
}

# --- LEGACY / FALLBACK (If needed for generic calls) ---
PARAMS = {
    "MOMENTUM_WINDOW": 10,
    "Z_STOP": 3.0,
    "MIN_TENOR": 0.5,
    "MAX_CONCURRENT": 50,
    "MAX_HALFLIFE_DAYS": 20.0
}
