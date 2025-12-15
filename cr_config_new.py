#cr_config.py

from zoneinfo import ZoneInfo
from pathlib import Path

# ========= Paths =========
PATH_DATA   = "x"
PATH_ENH    = "x"
PATH_MODELS = "x"
PATH_OUT    = "x"

Path(PATH_ENH).mkdir(parents=True, exist_ok=True)
Path(PATH_MODELS).mkdir(parents=True, exist_ok=True)
Path(PATH_OUT).mkdir(parents=True, exist_ok=True)

# ========= Instruments (explicit) =========
TENOR_YEARS = {
    "USOSFRA BGN Curncy": 1/12,   "USOSFRB BGN Curncy": 2/12,   "USOSFRC BGN Curncy": 3/12,
    "USOSFRD BGN Curncy": 4/12,   "USOSFRE BGN Curncy": 5/12,   "USOSFRF BGN Curncy": 6/12,
    "USOSFRG BGN Curncy": 7/12,   "USOSFRH BGN Curncy": 8/12,   "USOSFRI BGN Curncy": 9/12,
    "USOSFRJ BGN Curncy": 10/12,  "USOSFRK BGN Curncy": 11/12,  "USOSFR1 BGN Curncy": 1,
    "USOSFR1F BGN Curncy": 18/12, "USOSFR2 BGN Curncy": 2,      "USOSFR3 BGN Curncy": 3,
    "USOSFR4 BGN Curncy": 4,      "USOSFR5 BGN Curncy": 5,      "USOSFR6 BGN Curncy": 6,
    "USOSFR7 BGN Curncy": 7,      "USOSFR8 BGN Curncy": 8,      "USOSFR9 BGN Curncy": 9,
    "USOSFR10 BGN Curncy": 10,    "USOSFR12 BGN Curncy": 12,    "USOSFR15 BGN Curncy": 15,
    "USOSFR20 BGN Curncy": 20,    "USOSFR25 BGN Curncy": 25,    "USOSFR30 BGN Curncy": 30,
    "USOSFR40 BGN Curncy": 40,
}

# ========= Trade tape instrument mapping (overlay mode) =========
# Map your hedge tape "instrument" to the curve instrument used in TENOR_YEARS.
# Example:
# BBG_DICT = {
#     "USSWAP5 Curncy": "USOSFR5 BGN Curncy",
#     "USSWAP10 Curncy": "USOSFR10 BGN Curncy",
# }
#Below is placeholder, local version has the BBG_DICT filled out.
BBG_DICT = {}

TRADE_TYPES = "synth_trades"

# ========= Calendar filtering =========
USE_QL_CALENDAR = True
QL_US_MARKET    = "FederalReserve"
CAL_TZ          = "America/New_York"
TRADING_HOURS = ("08:00", "17:00")

# ========= Feature (builder) settings =========
# PCA trained on a rolling panel built at the chosen decision frequency (D/H).
PCA_COMPONENTS     = 3         # number of components to keep
PCA_LOOKBACK_DAYS  = 126       # original setting (≈ 6 months of trading days)

# Decision frequency for BOTH feature buckets and the backtest layer
DECISION_FREQ      = 'D'       # 'D' (daily) or 'H' (hourly)
RUN_TAG   = DECISION_FREQ                       # choose 'D' or 'H'
ENH_SUFFIX = f"_{RUN_TAG.lower()}"    # -> "_h"
OUT_SUFFIX = f"_{RUN_TAG.lower()}"    # -> "_h"

# Enable/disable PCA; when disabled we still compute spline residuals and z_comb will fallback.
PCA_ENABLE         = True

# Parallelism for feature creation (0 = auto ~ half the cores up to a small cap)
N_JOBS             = 0

# ---- Derived: convert day lookback to "bucket" lookback used by the builder
# If daily, 126 days means 126 buckets.
# If hourly, we expand to hours but cap to avoid huge fits (2 weeks cap by default).
PCA_LOOKBACK_CAP_HOURS = 24 * 14
if DECISION_FREQ == 'D':
    PCA_LOOKBACK = int(PCA_LOOKBACK_DAYS)                    # 126 --> 126 daily buckets
else:  # 'H'
    PCA_LOOKBACK = max(1, min(PCA_LOOKBACK_DAYS * 24, PCA_LOOKBACK_CAP_HOURS))

# ========= Backtest decision layer =========
Z_ENTRY       = 1.85     # enter when cheap-rich z-spread >= Z_ENTRY
Z_EXIT        = 0.65     # take profit when |z-spread| <= Z_EXIT
Z_STOP        = 2.45     # stop if divergence since entry >= Z_STOP
MAX_HOLD_DAYS = 60       # max holding period for a pair (days when DECISION_FREQ='D')
DRIFT_GATE_BPS = 0.0
DRIFT_WEIGHT = 0.0

# ========= Dynamic Thresholds (The "Transmission") =========
# Enables regime-based scaling of Z_ENTRY, Z_EXIT, Z_STOP
DYN_THRESH_ENABLE = True

# Sensitivity Weights:
# Multiplier = 1.0 + (Trend_Z * SENS_TRENDINESS) + (Health_Z * SENS_HEALTH)
# 
# SENS_TRENDINESS > 0: High Trend (Bad) -> Multiplier > 1 -> Harder Entry (Safety)
# SENS_HEALTH < 0: High Health (Good) -> Multiplier < 1 -> Easier Entry (Aggression)

SENS_TRENDINESS   = 0.0   # Start at 0.0 for Phase 1 (Baseline) optimization
SENS_HEALTH       = 0.0   # Start at 0.0 for Phase 1 (Baseline) optimization

# ========= Safety: The "20% Rule" =========
# We enforce that Tenor >= 5x Holding Period to ensure linear decay assumptions hold.
# 73.0 comes from 365 / 5.
MIN_TENOR_SAFETY_FACTOR = 73.0 

# Apply this to the execution thresholds
EXEC_LEG_TENOR_YEARS = 0.085
ALT_LEG_TENOR_YEARS  = 0.083

# ========= Risk & selection =========
BUCKETS = {
    "short": (0.0, 0.99),   # ~6M–<2Y
    "front": (1.0, 3.0),
    "belly": (3.1, 9.0),
    "long" : (10.0, 40.0),
}
MIN_SEP_YEARS        = 0.05

# Extra entry threshold if any leg is in short bucket (kept from earlier design)
SHORT_END_EXTRA_Z     = 0.30

# ========= Overlay mode settings =========
OVERLAY_SWITCH_COST_BP = 0.10   # 0.10 bp × DV01 per round-trip switch

# ========= Stalemate dead money =========
STALE_ENABLE = True
STALE_START_DAYS = 5.0
STALE_MIN_VELOCITY_Z = 0.015

# ========= Naive Risk (Curve Balance) =========
RISK_NAIVE_ENABLE = True
RISK_NAIVE_PIVOT  = 5.0      # Tenors <= 5.0 are "Front", > 5.0 are "Back"
RISK_NAIVE_LIMIT  = 80_000   # Max Absolute Net DV01 allowed on either side

# ========= Butterfly Router (Curvature Bonus) =========
FLY_ENABLE     = True
FLY_WEIGHT     = 0.15   # Bonus weight (e.g., 0.15 sigmas per unit of Fly Z)
FLY_MIN_DIST   = 1.5    # Wings must be at least 1.5y away (avoids noisy 9s/10s/11s)

# ==========================================
# BUTTERFLY / FLY OVERLAY CONFIG
# ==========================================

# 1. RUN MODE
# Options: "fly" (New Engine), "overlay" (Legacy Pair Engine)
RUN_MODE = "fly" 

# 2. FLY CONSTRUCTION RULES
# "STRICT": Belly must match Hedge Tenor exactly (Zero Basis Risk).
# "FLEXIBLE": Belly can be +/- 2.5y from Hedge Tenor (Basis Risk for Alpha).
# Recommendation: Start with STRICT.
FLY_ANCHOR_MODE = "STRICT" 

# Wings must be at least 2y away and at most 7y away from belly.
# Example: For 10y Belly -> Scans 8y/12y (min) up to 3y/17y (max).
FLY_WING_WIDTH_RANGE = (2.0, 7.0) 

# 3. TREND LOGIC
# Days to calculate the moving average Z-score.
# 20 Days = approx 1 trading month. Standard for "Short Term Trend".
Z_TREND_WINDOW = 20 

# 4. DRIFT WEIGHT (Updated for Flies)
# Flies rely heavily on roll-down. Increase this from 0.2 to 0.75 or 1.0.
DRIFT_WEIGHT = 0.75 
