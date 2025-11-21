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
BBG_DICT = {}

RUN_MODE = "overlay"
TRADE_TYPES = "synth_trades"

# ========= Calendar filtering =========
USE_QL_CALENDAR = True
QL_US_MARKET    = "FederalReserve"
CAL_TZ          = "America/New_York"
TRADING_HOURS = ("07:00", "17:30")

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
Z_ENTRY       = 2.00     # enter when cheap-rich z-spread >= Z_ENTRY
Z_EXIT        = 0.75     # take profit when |z-spread| <= Z_EXIT
Z_STOP        = 1.00     # stop if divergence since entry >= Z_STOP
BPS_PNL_STOP = 0.00
MAX_HOLD_DAYS = 10       # max holding period for a pair (days when DECISION_FREQ='D')
ALT_LEG_TENOR_YEARS = 0.0
EXEC_LEG_TENOR_YEARS = 0.084

# ========= Regime Filter Settings (Curve Environment) =========
# These control the "Regime Mask" (exogenous filter)
MIN_SIGNAL_HEALTH_Z     = -0.5   # Require composite health >= this (higher is better)
MAX_TRENDINESS_ABS      = 2.0    # Require absolute trendiness <= this
MAX_Z_XS_MEAN_ABS_Z     = 2.0    # Require cross-sectional mean Z-score <= this

# ========= Shock Filter Settings (PnL Feedback) =========
# These control the "Shock Blocker" (endogenous filter)
SHOCK_PNL_WINDOW        = 10     # Lookback window (in decision buckets) for PnL stats
SHOCK_USE_RAW_PNL       = True   # Enable simple Z-score check on PnL?
SHOCK_USE_RESIDUALS     = True   # Enable regression residual check?
SHOCK_RAW_PNL_Z_THRESH  = -1.5   # Z-score below this triggers a block
SHOCK_RESID_Z_THRESH    = -1.5   # Residual Z-score below this triggers a block
SHOCK_BLOCK_LENGTH      = 10     # Number of buckets to block AFTER a shock
SHOCK_REGRESSION_COLS   = [      # Columns from hybrid_signals to regress against
    "signal_health_z", 
    "trendiness_abs", 
    "z_xs_mean_roll_z"
]
SHOCK_METRIC_TYPE       = "BPS"  # "BPS" or "CASH"
SHOCK_MODE              = "ROLL_OFF" #"ROLL_OFF" or "EXIT_ALL"

# ========= Risk & selection =========
BUCKETS = {
    "short": (0.0, 0.99),   # ~6M–<2Y
    "front": (1.0, 3.0),
    "belly": (3.1, 9.0),
    "long" : (10.0, 40.0),
}
MIN_SEP_YEARS        = 0.05
MAX_SPAN_YEARS = 10
MAX_CONCURRENT_PAIRS = 100_000
PER_BUCKET_DV01_CAP  = 1e10
TOTAL_DV01_CAP       = 1e10
FRONT_END_DV01_CAP   = 1e10

# ========= Fly-alignment gate (curvature sanity) =========
FLY_ENABLE            = True          # master on/off
FLY_MODE              = "strict"    # "off" | "loose" | "tolerant" | "strict"

# Which flies to evaluate (triplets in years, sorted a<b<c)
FLY_DEFS              = [(1.0, 3.0, 7.0), (2.0, 5.0, 10.0), (3.0, 7.0, 15.0), (4.0, 8.0, 20.0)]

# Strength and locality
FLY_Z_MIN             = 1.0           # require |fly_z| >= this to consider it “strong”
FLY_NEIGHBOR_ONLY     = True          # only evaluate flies near the leg tenor
FLY_WINDOW_YEARS      = 2.0           # neighborhood half-width in years (± around leg tenor)

# Robust blocking logic
FLY_REQUIRE_COUNT     = 1             # need ≥K contradictory strong flies to block (tolerant)
FLY_SKIP_SHORT_UNDER  = 1.0           # skip gate entirely for legs < this tenor (yrs)
FLY_ALLOW_BIG_ZDISP   = True          # never block if pair dispersion already big
FLY_BIG_ZDISP_MARGIN  = 0.10          # allow when zdisp ≥ Z_ENTRY + margin
FLY_TENOR_TOL_YEARS   = 0.02

# Extra entry threshold if any leg is in short bucket (kept from earlier design)
SHORT_END_EXTRA_Z     = 0.30

# ========= Overlay mode settings =========
OVERLAY_SWITCH_COST_BP = 0.10   # 0.10 bp × DV01 per round-trip switch

# 1) Hard per-trade DV01 cap (absolute DV01 used in overlay pair)
OVERLAY_DV01_CAP_PER_TRADE = 1e10   # good first pass

# 2) Per-bucket per-trade DV01 caps (overrides global cap when present)
OVERLAY_DV01_CAP_PER_TRADE_BUCKET = {
    "short": 1e10,
    "front": 1e10,
    "belly": 1e10,
    "long": 1e10,
    "other": 1e10,
}

# 3) Per decision timestamp DV01 gate (skip overlay on extreme flow days)
# This is sum(abs(dv01)) across hedge tape at that decision timestamp.
OVERLAY_DV01_TS_CAP = 1e10         # try 300k; later test 200k vs 500k

# ======== Overlay Z-entry scaling with DV01 =======

# Base reference DV01 for which Z_ENTRY applies "as-is"
OVERLAY_Z_ENTRY_DV01_REF = 1e10      # roughly median hedge size

# Slope for DV01 scaling:
# Z_ENTRY_eff = Z_ENTRY + k * log(dv01 / OVERLAY_Z_ENTRY_DV01_REF)
OVERLAY_Z_ENTRY_DV01_K = 0.0           # try 0.45 first; 0.35–0.55 reasonable

# ======== Overlay max-hold scaling with DV01 ======

# For small trades: use global MAX_HOLD_DAYS (your existing config)
# For medium/large trades: tighten max-hold.

OVERLAY_MAX_HOLD_DV01_MED  = 1e10   # dv01 >= this → use MED max-hold
OVERLAY_MAX_HOLD_DAYS_MED  = 10          # instead of 10

OVERLAY_MAX_HOLD_DV01_HI   = 1e10   # dv01 >= this → use HI max-hold
OVERLAY_MAX_HOLD_DAYS_HI   = 10          # very tight leash on big trades
