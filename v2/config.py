"""
config.py

Central configuration for the All-Weather RV Strategy.
Holds paths, physics constants, and the Regime Logic dictionaries.
"""

from pathlib import Path

# ==============================================================================
# 1. FILE PATHS & I/O
# ==============================================================================
# Paths should be relative or absolute based on your environment
BASE_DIR = Path(".")
PATH_DATA = BASE_DIR / "data" / "features"
PATH_ENH  = BASE_DIR / "data" / "enhanced"
PATH_OUT  = BASE_DIR / "data" / "output"

# File Suffixes (for versioning)
ENH_SUFFIX = ""
OUT_SUFFIX = "_v1_allweather"

# ==============================================================================
# 2. EXECUTION PHYSICS
# ==============================================================================
DECISION_FREQ = "H"  # 'D' for Daily, 'H' for Hourly
CALC_METHOD   = "cubic" # 'linear' or 'cubic' (Spline type)

# Tenor Definitions (Tickers to Years)
TENOR_YEARS = {
    "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, 
    "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0
}

# Trade sizing
DEFAULT_SCALE_DV01 = 10_000.0

# ==============================================================================
# 3. REGIME LOGIC (The "Brain")
# ==============================================================================

# Strategy Modes: 'curve', 'fly', 'both'
STRATEGY_MODE = "both" 

# The Master Regime Dictionary
# Structure:
#   Type -> Mode (Threshold/Multiplier) -> Signal -> Value
REGIME_CONFIG = {
    # --------------------------------------------------------------------------
    # CURVE TRADES (Momentum/Directional)
    # --------------------------------------------------------------------------
    "curve": {
        "threshold": {
            # Gating Logic: Only trade Curve if...
            # Example: Block if Hurst is too low (Range bound market)
            "hurst_max": (0.45, "greater"), 
        },
        "multiplier": {
            # Scoring Logic: Adjust Z-Entry based on signal
            # Example: If Vol (scale) is high, lower the entry bar (chase the trend)
            "scale_mean": (1.5, "multiply", 0.8), # If scale > 1.5, multiply entry Z by 0.8
            # Example: If Trend (slope) agrees, lower bar
            "z_slope_abs": (0.5, "subtract", 0.5) # If slope > 0.5, subtract 0.5 from entry Z
        }
    },

    # --------------------------------------------------------------------------
    # FLY TRADES (Mean Reversion)
    # --------------------------------------------------------------------------
    "fly": {
        "threshold": {
            # Gating Logic: Only trade Fly if...
            # Example: Block if Hurst is too high (Trending market)
            "hurst_max": (0.55, "less"),
            # Example: Block if Vol is exploding
            "scale_mean": (2.0, "less")
        },
        "multiplier": {
            # Scoring Logic:
            # Example: If Vol is super low, demand higher premium (raise bar)
            "scale_mean": (0.5, "multiply", 1.2)
        }
    },

    # --------------------------------------------------------------------------
    # ARBITRATION (When Mode == 'Both')
    # --------------------------------------------------------------------------
    "both": {
        "priority": {
            # If signals conflict, which wins?
            # Logic: If Hurst > 0.6, force Curve mode. Else Fly.
            "signal": "hurst_max",
            "split_level": 0.55,
            "above_split": "curve",
            "below_split": "fly"
        }
    }
}

# ==============================================================================
# 4. ENTRY / EXIT PARAMETERS
# ==============================================================================
PARAMS = {
    # Base Z-Scores
    "Z_ENTRY": 0.75,
    "Z_EXIT": 0.25,
    "Z_STOP": 3.00,
    
    # Drift Logic
    "DRIFT_GATE_BPS": -5.0,  # Minimum daily carry/roll allowed (don't bleed)
    "DRIFT_WEIGHT": 0.20,    # How much drift boosts the Z-score (score = z + drift*w)
    
    # Execution
    "MIN_TENOR": 0.5,
    "MAX_CONCURRENT": 50,
    
    # Fly Specifics
    "FLY_WEIGHT_METHOD": "convexity", # 'convexity' or 'duration'
    "FLY_WING_WIDTH": (1.5, 7.0),     # Min/Max distance for wings
    
    # Momentum (Falling Knife)
    "MOMENTUM_WINDOW": 5,    # Days to look back for fast trend
    "MOMENTUM_GATE": 0.25    # Z-score velocity threshold
}
