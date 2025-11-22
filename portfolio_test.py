#portfolio_test.py

import os, sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd

# All config access via module namespace
import cr_config as cr
from hybrid_filter import ShockConfig

# ------------------------
# Utilities / conventions
# ------------------------
Path(getattr(cr, "PATH_OUT", ".")).mkdir(parents=True, exist_ok=True)


def _to_float(x, default=np.nan):
    """Safe scalar float extraction (works for scalar, 1-elem Series, or np types)."""
    try:
        if isinstance(x, (pd.Series, pd.Index)):
            if len(x) == 0:
                return default
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return default


def pv01_proxy(tenor_yrs, rate_pct):
    """Simple PV01 proxy so pair is roughly DV01-neutral."""
    return tenor_yrs / max(1e-6, 1.0 + 0.01 * rate_pct)


def assign_bucket(tenor):
    buckets = getattr(cr, "BUCKETS", {})
    for name, (lo, hi) in buckets.items():
        if (tenor >= lo) and (tenor <= hi):
            return name
    return "other"


# ------------------------
# Hedge tape helpers (overlay mode)
# ------------------------
def _map_instrument_to_tenor(instr: str) -> Optional[float]:
    """
    Map a hedge-tape 'instrument' to tenor_yrs using BBG_DICT and TENOR_YEARS.
    - First apply BBG_DICT (tape ticker -> curve ticker),
    - Then look up in TENOR_YEARS.
    """
    if instr is None or not isinstance(instr, str):
        return None
    instr = instr.strip()
    # Apply mapping if present
    mapped = cr.BBG_DICT.get(instr, instr)
    tenor = cr.TENOR_YEARS.get(mapped)
    return float(tenor) if tenor is not None else None


def prepare_hedge_tape(raw_df: pd.DataFrame, decision_freq: str) -> pd.DataFrame:
    """
    Clean hedge tape for overlay mode.
    Expects columns: side ('CPAY'/'CRCV'), instrument, EqVolDelta, tradetimeUTC.
    Returns dataframe with: trade_id, trade_ts, decision_ts, tenor_yrs, side, dv01.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    # Time: parse as UTC, then drop tz info (consistent with enhanced files)
    df["trade_ts"] = (
        pd.to_datetime(df["tradetimeUTC"], utc=True, errors="coerce")
          .dt.tz_convert("UTC")
          .dt.tz_localize(None)
    )

    decision_freq = str(decision_freq).upper()
    if decision_freq == "D":
        df["decision_ts"] = df["trade_ts"].dt.floor("d")
    elif decision_freq == "H":
        df["decision_ts"] = df["trade_ts"].dt.floor("h")
    else:
        raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

    # Side
    df["side"] = df["side"].astype(str).str.upper()
    df = df[df["side"].isin(["CPAY", "CRCV"])]

    # Tenor mapping
    df["tenor_yrs"] = df["instrument"].map(_map_instrument_to_tenor)
    df = df[np.isfinite(df["tenor_yrs"])]

    # DV01
    df["dv01"] = pd.to_numeric(df["EqVolDelta"], errors="coerce")
    df = df[np.isfinite(df["dv01"]) & (df["dv01"] > 0)]

    # Drop obvious nulls
    df = df.dropna(subset=["trade_ts", "decision_ts", "tenor_yrs", "dv01"])

    # Stable trade_id
    if "trade_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["trade_id"] = df.index.astype(int)

    cols = ["trade_id", "trade_ts", "decision_ts", "tenor_yrs", "side", "dv01"]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra]


# ------------------------
# Fly alignment (optional)
# ------------------------
def _row_for_tenor(snap_last: pd.DataFrame, tenor: float) -> pd.Series | None:
    r = snap_last.loc[snap_last["tenor_yrs"] == tenor]
    if r.empty:
        return None
    return r.iloc[0]


def _get_z_at_tenor(snap_last: pd.DataFrame, tenor: float, tol: float | None = None) -> float | None:
    if tol is None:
        tol = float(getattr(cr, "FLY_TENOR_TOL_YEARS", 0.02))
    t = float(tenor)
    s = snap_last[["tenor_yrs", "z_comb"]].dropna()
    if s.empty:
        return None
    # choose nearest tenor within tolerance
    s = s.assign(_dist=(s["tenor_yrs"] - t).abs())
    row = s.loc[s["_dist"].idxmin()]
    if row["_dist"] <= tol:
        return float(row["z_comb"])
    return None


def compute_fly_z(snap_last: pd.DataFrame, a: float, b: float, c: float) -> float | None:
    try:
        z_a = _get_z_at_tenor(snap_last, float(a))
        z_b = _get_z_at_tenor(snap_last, float(b))
        z_c = _get_z_at_tenor(snap_last, float(c))
        if any(v is None for v in (z_a, z_b, z_c)):
            return None
        xs = snap_last["z_comb"].astype(float).to_numpy()
        sd = np.nanstd(xs, ddof=1) if xs.size > 1 else 1.0
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        fly_raw = 0.5*(z_a + z_c) - z_b
        return fly_raw / sd
    except Exception:
        return None


def fly_alignment_ok(
    leg_tenor: float,
    leg_sign_z: float,                 # +1 for cheap (expect z↓), -1 for rich (expect z↑)
    snap_last: pd.DataFrame,
    *,
    zdisp_for_pair: float | None = None
) -> bool:
    FLY_ENABLE          = bool(getattr(cr, "FLY_ENABLE", True))
    FLY_MODE            = str(getattr(cr, "FLY_MODE", "tolerant")).lower()
    FLY_DEFS            = list(getattr(cr, "FLY_DEFS", []))
    FLY_Z_MIN           = float(getattr(cr, "FLY_Z_MIN", 0.8))
    FLY_REQUIRE_COUNT   = int(getattr(cr, "FLY_REQUIRE_COUNT", 2))
    FLY_NEIGHBOR_ONLY   = bool(getattr(cr, "FLY_NEIGHBOR_ONLY", True))
    FLY_WINDOW_YEARS    = float(getattr(cr, "FLY_WINDOW_YEARS", 3.0))
    FLY_SKIP_SHORT_UNDER= getattr(cr, "FLY_SKIP_SHORT_UNDER", None)
    FLY_ALLOW_BIG_ZDISP = bool(getattr(cr, "FLY_ALLOW_BIG_ZDISP", True))
    FLY_BIG_ZDISP_MARGIN= float(getattr(cr, "FLY_BIG_ZDISP_MARGIN", 0.20))
    Z_ENTRY             = float(getattr(cr, "Z_ENTRY", 0.75))

    if not FLY_ENABLE or FLY_MODE == "off":
        return True

    # Big dispersion waiver
    if (FLY_ALLOW_BIG_ZDISP and (zdisp_for_pair is not None) and
        (float(zdisp_for_pair) >= float(Z_ENTRY) + float(FLY_BIG_ZDISP_MARGIN))):
        return True

    # Skip short end
    if (FLY_SKIP_SHORT_UNDER is not None) and (leg_tenor < float(FLY_SKIP_SHORT_UNDER)):
        return True

    triplets = FLY_DEFS
    if FLY_NEIGHBOR_ONLY:
        W = float(FLY_WINDOW_YEARS)
        triplets = [(a,b,c) for (a,b,c) in FLY_DEFS if abs(float(b) - float(leg_tenor)) <= W]
        if not triplets:
            return True

    contradictions = 0
    for (a, b, c) in triplets:
        fz = compute_fly_z(snap_last, a, b, c)
        if fz is None or not np.isfinite(fz) or abs(fz) < float(FLY_Z_MIN):
            continue
        # CONTRADICTION when sign(fly)*sign(leg) < 0
        if np.sign(fz) * np.sign(leg_sign_z) < 0:
            contradictions += 1

    if FLY_MODE == "strict":
        return contradictions == 0
    if FLY_MODE == "loose":
        return contradictions <= 1
    if FLY_MODE == "tolerant":
        return contradictions <= int(FLY_REQUIRE_COUNT)
    return True

def tenor_to_ticker(tenor_yrs: float, tol: float = 1e-6) -> Optional[str]:
    """
    Reverse lookup: given tenor_yrs, find the curve ticker in TENOR_YEARS
    whose tenor is nearest, within tolerance tol.

    Returns the ticker string or None if nothing within tol.
    """
    if tenor_yrs is None or not np.isfinite(tenor_yrs):
        return None

    t = float(tenor_yrs)
    best_key = None
    best_diff = float("inf")

    for k, v in cr.TENOR_YEARS.items():
        try:
            vv = float(v)
        except Exception:
            continue
        diff = abs(vv - t)
        if diff < best_diff:
            best_diff = diff
            best_key = k

    if best_key is not None and best_diff <= tol:
        return best_key
    return None

# ------------------------
# Pair object
# ------------------------
class PairPos:
    def __init__(
        self,
        open_ts,
        cheap_row,
        rich_row,
        w_i,
        w_j,
        decisions_per_day: int,
        *,
        scale_dv01: float = 1.0,
        mode: str = "strategy",
        meta: Optional[Dict] = None,
        dir_sign: Optional[float] = None,
        entry_rate_i: Optional[float] = None,
        entry_rate_j: Optional[float] = None,
    ):
        """
        cheap_row / rich_row naming is historical; in overlay mode, these can be
        interpreted as (alt_leg, exec_leg). Orientation is fully captured by w_i/w_j
        and side sign when we construct the pair.

        dir_sign:
          - If provided, this is the "direction" in z-space we care about for exits.
          - If None, we default to sign(entry_zspread) (cheap minus rich).

        entry_rate_i / entry_rate_j:
          - If provided, these are the TRUE trade-time entry rates per leg.
          - If None, we fall back to cheap_row['rate'] / rich_row['rate'].

        DV01 semantics (overlay mode only):
          - scale_dv01 = hedge-tape DV01 (per 1bp) for the pair.
          - Per-leg entry DV01:
                dv01_i_entry = w_i * scale_dv01
                dv01_j_entry = w_j * scale_dv01
          - At mark t, we compute remaining tenor in years via actual/360:
                years_passed = days(open_ts -> decision_ts) / 360
                rem_i = max(tenor_i_orig - years_passed, 0)
                rem_j = max(tenor_j_orig - years_passed, 0)
                frac_i = rem_i / tenor_i_orig
                frac_j = rem_j / tenor_j_orig
                dv01_i_curr = dv01_i_entry * frac_i
                dv01_j_curr = dv01_j_entry * frac_j
          - PnL cash:
                (entry_rate_i - rate_i_t)*100*dv01_i_curr
              + (entry_rate_j - rate_j_t)*100*dv01_j_curr
          - pnl_bp = pnl_cash / scale_dv01  (matches old behavior at t=0).
        """
        self.open_ts = open_ts

        # Tenors
        self.tenor_i = _to_float(cheap_row["tenor_yrs"])
        self.tenor_j = _to_float(rich_row["tenor_yrs"])
        self.tenor_i_orig = self.tenor_i
        self.tenor_j_orig = self.tenor_j

        # Default entry rates from snapshot rows
        default_rate_i = _to_float(cheap_row["rate"])
        default_rate_j = _to_float(rich_row["rate"])

        # Trade-time entry rates (can be overridden in overlay mode)
        self.entry_rate_i = (
            _to_float(entry_rate_i, default=default_rate_i)
            if entry_rate_i is not None else default_rate_i
        )
        self.entry_rate_j = (
            _to_float(entry_rate_j, default=default_rate_j)
            if entry_rate_j is not None else default_rate_j
        )

        # For PnL we treat these as the fixed entry anchors
        self.rate_i = self.entry_rate_i
        self.rate_j = self.entry_rate_j

        # Current / last marked rates (updated in mark)
        self.last_rate_i = self.entry_rate_i
        self.last_rate_j = self.entry_rate_j

        self.w_i = float(w_i)
        self.w_j = float(w_j)

        zi = _to_float(cheap_row["z_comb"])
        zj = _to_float(rich_row["z_comb"])
        self.entry_zspread = zi - zj

        # Direction in z-space (cheap-rich) that we care about for exits.
        if dir_sign is None or not np.isfinite(dir_sign) or dir_sign == 0.0:
            if np.isfinite(self.entry_zspread) and self.entry_zspread != 0.0:
                self.dir_sign = float(np.sign(self.entry_zspread))
            else:
                # Fallback: treat positive direction as default
                self.dir_sign = 1.0
        else:
            self.dir_sign = float(dir_sign)

        # Directional z at entry (used for reversion/stop logic)
        self.entry_z_dir = self.dir_sign * self.entry_zspread

        self.closed = False
        self.close_ts = None
        self.exit_reason = None

        # Mode / DV01 setup
        self.scale_dv01: float = float(scale_dv01)
        self.mode: str = str(mode)
        self.meta: Dict = meta or {}

        # Pair-level "initial dv01" (signed, per pair)
        # For overlay, this is the hedge-tape DV01; for strategy, typically 1.0.
        self.initial_dv01: float = self.scale_dv01

        # Per-leg DV01 at entry
        # Overlay: dv01_i_entry = w_i * dv01_pair, etc.
        # Strategy: keep same definition for logging, even if not used for PnL.
        self.dv01_i_entry: float = self.scale_dv01 * self.w_i
        self.dv01_j_entry: float = self.scale_dv01 * self.w_j

        # Current per-leg DV01 (updated at each mark)
        self.dv01_i_curr: float = self.dv01_i_entry
        self.dv01_j_curr: float = self.dv01_j_entry

        # Remaining tenor (years) used for DV01 decay (overlay)
        self.rem_tenor_i: float = self.tenor_i_orig
        self.rem_tenor_j: float = self.tenor_j_orig

        # PnL bookkeeping
        self.pnl_bp: float = 0.0      # bp-equivalent
        self.pnl_cash: float = 0.0    # cash units
        self.pnl: float = 0.0         # alias

        # Transaction cost placeholders (overlay mode)
        self.tcost_bp: float = 0.0
        self.tcost_cash: float = 0.0

        # bookkeeping / aging
        self.decisions_per_day = decisions_per_day
        self.age_decisions = 0  # increments by 1 each mark

        self.bucket_i = assign_bucket(self.tenor_i)
        self.bucket_j = assign_bucket(self.tenor_j)

        # attribution proxy
        self.last_zspread = self.entry_zspread
        self.last_z_dir = self.entry_z_dir
        self.conv_pnl_proxy = 0.0

    def _update_overlay_dv01(self, decision_ts: pd.Timestamp):
        """
        Overlay-only: update per-leg DV01 via linear decay in remaining tenor.
        Uses actual/360 day count between open_ts and current decision_ts.
        """
        if not isinstance(decision_ts, pd.Timestamp):
            return

        # Calendar days difference (actual/360)
        days = (decision_ts.normalize() - self.open_ts.normalize()).days
        if days < 0:
            days = 0
        years_passed = days / 360.0

        # Remaining tenor in years (floor at 0)
        rem_i = max(self.tenor_i_orig - years_passed, 0.0)
        rem_j = max(self.tenor_j_orig - years_passed, 0.0)
        self.rem_tenor_i = rem_i
        self.rem_tenor_j = rem_j

        # Fractions of original tenor (0..1)
        Ti0 = max(self.tenor_i_orig, 1e-6)
        Tj0 = max(self.tenor_j_orig, 1e-6)
        frac_i = rem_i / Ti0
        frac_j = rem_j / Tj0

        # Current per-leg DV01 in cash units (signed)
        self.dv01_i_curr = self.dv01_i_entry * frac_i
        self.dv01_j_curr = self.dv01_j_entry * frac_j

    def mark(self, snap_last: pd.DataFrame, decision_ts: Optional[pd.Timestamp] = None):
        """
        Mark-to-market at decision time using last rate per tenor; update convergence proxy.

        Strategy mode:
          - PnL is unchanged from your existing implementation
            (unit world: bp × scale_dv01).

        Overlay mode:
          - Per-leg DV01 decays linearly with remaining tenor (actual/360),
            and PnL is computed using those time-varying per-leg DV01s
            against trade-time entry rates.
        """
        # Current curve rates at this decision bucket
        ri = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "rate"])
        rj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "rate"])

        # Store current marks for diagnostics
        self.last_rate_i = ri
        self.last_rate_j = rj

        if self.mode == "overlay":
            # --- Overlay: DV01-decaying PnL ---
            if decision_ts is not None:
                self._update_overlay_dv01(decision_ts)

            # PnL cash using per-leg DV01 and entry→current rate change
            d_i_cash = (self.entry_rate_i - ri) * 100.0 * self.dv01_i_curr
            d_j_cash = (self.entry_rate_j - rj) * 100.0 * self.dv01_j_curr
            pnl_cash = d_i_cash + d_j_cash

            self.pnl_cash = pnl_cash
            self.pnl = pnl_cash

            if self.scale_dv01 != 0.0 and np.isfinite(self.scale_dv01):
                self.pnl_bp = pnl_cash / self.scale_dv01
            else:
                self.pnl_bp = 0.0
        else:
            # --- Strategy mode: keep original bp-based logic ---
            d_i = (self.entry_rate_i - ri) * self.w_i * 100.0
            d_j = (self.entry_rate_j - rj) * self.w_j * 100.0
            pnl_bp = d_i + d_j
            self.pnl_bp = pnl_bp

            pnl_cash = pnl_bp * self.scale_dv01
            self.pnl_cash = pnl_cash
            self.pnl = pnl_cash

            # For logging, keep per-leg DV01 consistent (no decay in strategy)
            self.dv01_i_curr = self.dv01_i_entry
            self.dv01_j_curr = self.dv01_j_entry
            self.rem_tenor_i = self.tenor_i_orig
            self.rem_tenor_j = self.tenor_j_orig

        # z-spread and convergence proxy (unchanged)
        zi = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "z_comb"])
        zj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "z_comb"])
        zsp = zi - zj

        # Update convergence proxy in raw z-spread space
        if np.isfinite(zsp) and np.isfinite(self.last_zspread):
            self.conv_pnl_proxy += (self.last_zspread - zsp) * 10.0
        self.last_zspread = zsp

        # Also keep a directional z for exit logic
        if np.isfinite(zsp) and np.isfinite(self.dir_sign):
            self.last_z_dir = self.dir_sign * zsp
        else:
            self.last_z_dir = np.nan

        # age by one decision step
        self.age_decisions += 1
        return zsp

# ------------------------
# Greedy selector with caps (Mode A: strategy)
# ------------------------
def choose_pairs_under_caps(
    snap_last: pd.DataFrame,
    max_pairs: int,
    per_bucket_cap: float,
    total_cap: float,
    front_end_cap: float,
    extra_z_entry: float
):
    """
    Returns list of (cheap_row, rich_row, w_i, w_j).
    Greedy: rank by dispersion, enforce tenor uniqueness & DV01 caps by bucket and total.
    Uses tolerant fly gate; big zdisp can waive fly blocks.
    """
    # minimal columns required
    cols_need = {"tenor_yrs", "rate", "z_comb"}
    if not cols_need.issubset(snap_last.columns):
        return []

    sig = snap_last[list(cols_need)].dropna().copy()
    if sig.empty:
        return []

    # Sort cross-section by z to find extremes
    sig = sig.sort_values("z_comb", kind="mergesort")  # stable sort

    candidates = []
    used_tenors = set()

    # Build pool from a few lowest and highest z points
    low_take  = min(5, len(sig))
    high_take = min(8, len(sig))

    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    MAX_SPAN_YEARS = float(getattr(cr, "MAX_SPAN_YEARS", 10))
    Z_ENTRY       = float(getattr(cr, "Z_ENTRY", 0.75))
    SHORT_END_EXTRA_Z = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))

    for k_low in range(low_take):
        rich = sig.iloc[k_low]
        for k_hi in range(1, high_take + 1):
            cheap = sig.iloc[-k_hi]

            Ti, Tj = float(cheap["tenor_yrs"]), float(rich["tenor_yrs"])
            if Ti in used_tenors or Tj in used_tenors:
                continue

            MIN_LEG_TENOR = float(getattr(cr, "MIN_LEG_TENOR_YEARS", 0.0))
            if (Ti < MIN_LEG_TENOR) or (Tj < MIN_LEG_TENOR):
                continue

            # minimum tenor separation
            if abs(Ti - Tj) < MIN_SEP_YEARS:
                continue
            # maximum tenor separation
            if abs(Ti - Tj) > MAX_SPAN_YEARS:
                continue

            zdisp = float(cheap["z_comb"] - rich["z_comb"])
            if zdisp < (Z_ENTRY + float(extra_z_entry)):
                continue

            # tolerant fly gate (can waive if signal already large per config)
            if not fly_alignment_ok(Ti, +1.0, snap_last, zdisp_for_pair=zdisp):
                continue
            if not fly_alignment_ok(Tj, -1.0, snap_last, zdisp_for_pair=zdisp):
                continue

            candidates.append((zdisp, cheap, rich))

    # Greedy pack by decreasing dispersion under DV01 caps
    BUCKETS              = getattr(cr, "BUCKETS", {})
    PER_BUCKET_DV01_CAP  = float(getattr(cr, "PER_BUCKET_DV01_CAP", 1.0))
    TOTAL_DV01_CAP       = float(getattr(cr, "TOTAL_DV01_CAP", 3.0))
    FRONT_END_DV01_CAP   = float(getattr(cr, "FRONT_END_DV01_CAP", 1.0))

    bucket_dv01 = {b: 0.0 for b in BUCKETS.keys()}
    total_dv01 = 0.0
    selected = []

    for zdisp, cheap, rich in sorted(candidates, key=lambda x: x[0], reverse=True):
        if len(selected) >= max_pairs:
            break

        Ti, Tj = float(cheap["tenor_yrs"]), float(rich["tenor_yrs"])
        if Ti in used_tenors or Tj in used_tenors:
            continue

        # PV01-neutral weights within the pair
        pv_i = pv01_proxy(Ti, float(cheap["rate"]))
        pv_j = pv01_proxy(Tj, float(rich["rate"]))
        w_i =  1.0
        w_j = - w_i * pv_i / (pv_j if pv_j != 0 else 1e-12)

        # Bucket caps
        b_i = assign_bucket(Ti)
        b_j = assign_bucket(Tj)
        dv_i = abs(w_i) * pv_i
        dv_j = abs(w_j) * pv_j

        # per-bucket caps
        if b_i in bucket_dv01 and (bucket_dv01[b_i] + dv_i) > PER_BUCKET_DV01_CAP:
            continue
        if b_j in bucket_dv01 and (bucket_dv01[b_j] + dv_j) > PER_BUCKET_DV01_CAP:
            continue

        # front-end aggregate cap
        short_add = (dv_i if b_i == "short" else 0.0) + (dv_j if b_j == "short" else 0.0)
        short_tot = bucket_dv01.get("short", 0.0)
        if (short_tot + short_add) > FRONT_END_DV01_CAP:
            continue

        # total cap
        if (total_dv01 + dv_i + dv_j) > TOTAL_DV01_CAP:
            continue

        # extra threshold if any leg sits in the short bucket
        if (b_i == "short" or b_j == "short") and (zdisp < (Z_ENTRY + SHORT_END_EXTRA_Z)):
            continue

        # accept
        used_tenors.add(Ti); used_tenors.add(Tj)
        bucket_dv01[b_i] = bucket_dv01.get(b_i, 0.0) + dv_i
        bucket_dv01[b_j] = bucket_dv01.get(b_j, 0.0) + dv_j
        total_dv01 += (dv_i + dv_j)

        selected.append((cheap, rich, w_i, w_j))

    return selected


# ------------------------
# Filename helpers (suffixed IO)
# ------------------------
def _enhanced_in_path(yymm: str) -> Path:
    """Use cr.enh_fname if present; else {yymm}_enh{ENH_SUFFIX}.parquet; fallback {yymm}_enh.parquet."""
    if hasattr(cr, "enh_fname") and callable(cr.enh_fname):
        name = cr.enh_fname(yymm)
    else:
        suffix = getattr(cr, "ENH_SUFFIX", "")
        name = f"{yymm}_enh{suffix}.parquet" if suffix else f"{yymm}_enh.parquet"
    return Path(getattr(cr, "PATH_ENH", ".")) / name


def _positions_out_path() -> Path:
    if hasattr(cr, "positions_fname") and callable(cr.positions_fname):
        name = cr.positions_fname()
    else:
        suffix = getattr(cr, "OUT_SUFFIX", "")
        name = f"positions_ledger{suffix}.parquet" if suffix else "positions_ledger.parquet"
    return Path(getattr(cr, "PATH_OUT", ".")) / name


def _marks_out_path() -> Path:
    if hasattr(cr, "marks_fname") and callable(cr.marks_fname):
        name = cr.marks_fname()
    else:
        suffix = getattr(cr, "OUT_SUFFIX", "")
        name = f"marks_ledger{suffix}.parquet" if suffix else "marks_ledger.parquet"
    return Path(getattr(cr, "PATH_OUT", ".")) / name


def _pnl_out_path() -> Path:
    if hasattr(cr, "pnl_fname") and callable(cr.pnl_fname):
        name = cr.pnl_fname()
    else:
        suffix = getattr(cr, "OUT_SUFFIX", "")
        name = f"pnl_by_bucket{suffix}.parquet" if suffix else "pnl_by_bucket.parquet"
    return Path(getattr(cr, "PATH_OUT", ".")) / name


def _pnl_curve_png(yymm: str) -> Path:
    if hasattr(cr, "pnl_curve_png") and callable(cr.pnl_curve_png):
        name = cr.pnl_curve_png(yymm)
    else:
        suffix = getattr(cr, "OUT_SUFFIX", "")
        name = f"pnl_curve_{yymm}{suffix}.png" if suffix else f"pnl_curve_{yymm}.png"
    return Path(getattr(cr, "PATH_OUT", ".")) / name

def run_month(
    yymm: str,
    *,
    decision_freq: str | None = None,
    open_positions: Optional[List[PairPos]] | None = None,
    carry_in: bool = True,
    mode: str = "strategy",
    hedges: Optional[pd.DataFrame] = None,
    overlay_use_caps: Optional[bool] = None,
    
    # --- FILTER ARGUMENTS ---
    regime_mask: Optional[pd.Series] = None,       
    hybrid_signals: Optional[pd.DataFrame] = None, 
    shock_cfg: Optional[ShockConfig] = None,       
):
    """
    Run a single month.
    NO LOOKAHEAD VERSION: 
    - Entries are gated by the shock state as of Yesterday (Start of Bucket).
    - Panic Exits are triggered by the shock state as of Today (End of Bucket).
    """
    import math
    try:
        np
    except NameError:
        import numpy as np

    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    mode = mode.lower()

    enh_path = _enhanced_in_path(yymm)
    if not enh_path.exists():
        raise FileNotFoundError(f"Missing enhanced file {enh_path}. Run feature_creation.py first.")

    df = pd.read_parquet(enh_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), (open_positions or [])

    need = {"ts", "tenor_yrs", "rate", "z_spline", "z_pca", "z_comb"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{enh_path} missing columns: {missing}")

    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    if decision_freq == "D":
        df["decision_ts"] = df["ts"].dt.floor("d")
        decisions_per_day = 1
    elif decision_freq == "H":
        df["decision_ts"] = df["ts"].dt.floor("h")
        per_day_counts = df.groupby(df["decision_ts"].dt.floor("d"))["decision_ts"].nunique()
        decisions_per_day = int(per_day_counts.mean()) if len(per_day_counts) else 24
    else:
        raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

    base_max_hold_decisions = cr.MAX_HOLD_DAYS * decisions_per_day

    OVERLAY_MAX_HOLD_DV01_MED = float(getattr(cr, "OVERLAY_MAX_HOLD_DV01_MED", 20_000.0))
    OVERLAY_MAX_HOLD_DV01_HI = float(getattr(cr, "OVERLAY_MAX_HOLD_DV01_HI", 50_000.0))
    OVERLAY_MAX_HOLD_DAYS_MED = float(getattr(cr, "OVERLAY_MAX_HOLD_DAYS_MED", 5.0))
    OVERLAY_MAX_HOLD_DAYS_HI = float(getattr(cr, "OVERLAY_MAX_HOLD_DAYS_HI", 2.0))

    def _overlay_max_hold_decisions(dv01_cash: float) -> int:
        dv = abs(float(dv01_cash))
        if dv >= OVERLAY_MAX_HOLD_DV01_HI:
            return int(round(OVERLAY_MAX_HOLD_DAYS_HI * decisions_per_day))
        elif dv >= OVERLAY_MAX_HOLD_DV01_MED:
            return int(round(OVERLAY_MAX_HOLD_DAYS_MED * decisions_per_day))
        return int(round(float(cr.MAX_HOLD_DAYS) * decisions_per_day))

    BASE_Z_ENTRY = float(getattr(cr, "Z_ENTRY", 0.75))
    Z_REF = float(getattr(cr, "OVERLAY_Z_ENTRY_DV01_REF", 5_000.0))
    Z_K = float(getattr(cr, "OVERLAY_Z_ENTRY_DV01_K", 0.0))

    def _overlay_effective_z_entry(dv01_cash: float) -> float:
        dv = abs(float(dv01_cash))
        if dv <= 0 or Z_REF <= 0 or Z_K == 0.0:
            return BASE_Z_ENTRY
        return BASE_Z_ENTRY + Z_K * math.log(dv / Z_REF)

    OVERLAY_DV01_CAP_PER_TRADE = float(getattr(cr, "OVERLAY_DV01_CAP_PER_TRADE", float("inf")))
    OVERLAY_DV01_CAP_PER_TRADE_BUCKET = dict(getattr(cr, "OVERLAY_DV01_CAP_PER_TRADE_BUCKET", {}))
    OVERLAY_DV01_TS_CAP = float(getattr(cr, "OVERLAY_DV01_TS_CAP", float("inf")))

    def _per_trade_dv01_cap_for_bucket(bucket: str) -> float:
        return float(OVERLAY_DV01_CAP_PER_TRADE_BUCKET.get(bucket, OVERLAY_DV01_CAP_PER_TRADE_BUCKET.get("other", OVERLAY_DV01_CAP_PER_TRADE)))

    _bps_stop_val = getattr(cr, "BPS_PNL_STOP", None)
    BPS_PNL_STOP = float(_bps_stop_val) if _bps_stop_val is not None else 0.0

    daily_pnl_history: list[float] = [] 
    daily_pnl_dates: list[pd.Timestamp] = []
    shock_block_remaining = 0
    
    valid_reg_cols = []
    sig_lookup = pd.DataFrame()
    if shock_cfg is not None:
        if hybrid_signals is not None:
            if "decision_ts" in hybrid_signals.columns:
                sig_lookup = hybrid_signals.drop_duplicates("decision_ts").set_index("decision_ts").sort_index()
            else:
                sig_lookup = hybrid_signals
            
            if shock_cfg.regression_cols:
                valid_reg_cols = [c for c in shock_cfg.regression_cols if c in sig_lookup.columns]

    SHOCK_MODE = str(getattr(cr, "SHOCK_MODE", "ROLL_OFF")).upper()
    open_positions = (open_positions or []) if carry_in else []

    if mode == "overlay" and hedges is not None and not hedges.empty:
        valid_decisions = df["decision_ts"].dropna().unique()
        hedges = hedges[hedges["decision_ts"].isin(valid_decisions)].copy()
    else:
        hedges = None

    ledger_rows: list[dict] = []
    closed_rows: list[dict] = []

    PER_BUCKET_DV01_CAP = float(getattr(cr, "PER_BUCKET_DV01_CAP", 1.0))
    TOTAL_DV01_CAP = float(getattr(cr, "TOTAL_DV01_CAP", 3.0))
    FRONT_END_DV01_CAP = float(getattr(cr, "FRONT_END_DV01_CAP", 1.0))
    Z_ENTRY = float(getattr(cr, "Z_ENTRY", 0.75))
    SHORT_END_EXTRA_Z = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))
    Z_EXIT = float(getattr(cr, "Z_EXIT", 0.40))
    Z_STOP = float(getattr(cr, "Z_STOP", 3.00))
    
    EXEC_LEG_TENOR_THRESHOLD = float(getattr(cr, "EXEC_LEG_TENOR_YEARS", 0.084))
    ALT_LEG_TENOR_THRESHOLD  = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0))
    
    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    MAX_SPAN_YEARS = float(getattr(cr, "MAX_SPAN_YEARS", 10.0))

    for dts, snap in df.groupby("decision_ts", sort=True):
        snap_last = (
            snap.sort_values("ts")
                .groupby("tenor_yrs", as_index=False)
                .tail(1)
                .reset_index(drop=True)
        )
        if snap_last.empty:
            continue

        # ============================================================
        # A) CAPTURE STATE AT OPEN (FOR ENTRIES)
        # ============================================================
        # CRITICAL: We must decide if we are blocked BEFORE we see today's PnL.
        # If shock_block_remaining > 0 from YESTERDAY, we block entries TODAY.
        was_shock_active_at_open = (shock_block_remaining > 0)
        
        # Decrement the counter for the *next* day logic, but 'was_shock_active_at_open'
        # remains the truth source for gating entries in this bucket.
        if shock_block_remaining > 0:
            shock_block_remaining -= 1

        # ============================================================
        # 1) MARK POSITIONS & NATURAL EXITS
        # ============================================================
        period_pnl_cash = 0.0       
        period_pnl_bps_mtm = 0.0    
        period_pnl_bps_realized = 0.0 
        
        still_open: list[PairPos] = []
        
        for pos in open_positions:
            prev_pnl_cash = pos.pnl_cash
            prev_pnl_bp = pos.pnl_bp
            
            zsp = pos.mark(snap_last, decision_ts=dts)
            
            period_pnl_cash += (pos.pnl_cash - prev_pnl_cash)
            period_pnl_bps_mtm += (pos.pnl_bp - prev_pnl_bp)

            entry_z = pos.entry_zspread
            exit_flag = None

            if np.isfinite(zsp) and np.isfinite(entry_z):
                entry_dir = getattr(pos, "entry_z_dir", pos.dir_sign * entry_z)
                curr_dir = getattr(pos, "last_z_dir", pos.dir_sign * zsp)

                if np.isfinite(entry_dir) and np.isfinite(curr_dir):
                    sign_entry = np.sign(entry_dir)
                    sign_curr = np.sign(curr_dir)
                    same_side = (sign_entry != 0) and (sign_entry == sign_curr)
                    moved_towards_zero = abs(curr_dir) <= abs(entry_dir)
                    within_exit_band = abs(curr_dir) <= Z_EXIT
                    dz_dir = curr_dir - entry_dir
                    moved_away = same_side and (abs(curr_dir) >= abs(entry_dir)) and (abs(dz_dir) >= Z_STOP)

                    if same_side and moved_towards_zero and within_exit_band:
                        exit_flag = "reversion"
                    elif moved_away:
                        exit_flag = "stop"

            if exit_flag is None and BPS_PNL_STOP > 0.0:
                if np.isfinite(pos.pnl_bp) and pos.pnl_bp <= -BPS_PNL_STOP:
                    exit_flag = "pnl_stop"

            if exit_flag is None:
                limit = _overlay_max_hold_decisions(pos.scale_dv01) if pos.mode == "overlay" else base_max_hold_decisions
                if pos.age_decisions >= limit:
                    exit_flag = "max_hold"

            if exit_flag is not None:
                pos.closed = True
                pos.close_ts = dts
                pos.exit_reason = exit_flag

            ledger_rows.append({
                "decision_ts": dts, "event": "mark",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "pnl_bp": pos.pnl_bp, "pnl_cash": pos.pnl_cash,
                "z_spread": zsp, "closed": pos.closed, "mode": pos.mode,
                "rate_i": getattr(pos, "last_rate_i", np.nan),
                "rate_j": getattr(pos, "last_rate_j", np.nan),
                "w_i": pos.w_i, "w_j": pos.w_j,
            })

            if pos.closed:
                tcost_bp = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10)) if pos.mode == "overlay" else 0.0
                tcost_cash = tcost_bp * pos.scale_dv01
                
                pos.tcost_bp = tcost_bp
                pos.tcost_cash = tcost_cash
                
                period_pnl_cash -= tcost_cash
                period_pnl_bps_mtm -= tcost_bp
                period_pnl_bps_realized += (pos.pnl_bp - tcost_bp)

                closed_rows.append({
                    "open_ts": pos.open_ts, 
                    "close_ts": pos.close_ts, 
                    "exit_reason": pos.exit_reason,
                    "tenor_i": pos.tenor_i, 
                    "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i,
                    "w_j": pos.w_j,
                    "leg_dir_i": float(np.sign(pos.w_i)),
                    "leg_dir_j": float(np.sign(pos.w_j)),
                    "entry_rate_i": pos.entry_rate_i,
                    "entry_rate_j": pos.entry_rate_j,
                    "close_rate_i": getattr(pos, "last_rate_i", np.nan),
                    "close_rate_j": getattr(pos, "last_rate_j", np.nan),
                    "dv01_i_entry": pos.dv01_i_entry,
                    "dv01_j_entry": pos.dv01_j_entry,
                    "dv01_i_close": pos.dv01_i_curr,
                    "dv01_j_close": pos.dv01_j_curr,
                    "initial_dv01": pos.initial_dv01,
                    "scale_dv01": pos.scale_dv01,
                    "entry_zspread": pos.entry_zspread,
                    "conv_proxy": pos.conv_pnl_proxy,
                    "pnl_gross_bp": pos.pnl_bp, 
                    "pnl_gross_cash": pos.pnl_cash,
                    "tcost_bp": tcost_bp, 
                    "tcost_cash": tcost_cash,
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, 
                    "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "days_held_equiv": pos.age_decisions / max(1, decisions_per_day),
                    "mode": pos.mode,
                    "trade_id": pos.meta.get("trade_id"),
                    "side": pos.meta.get("side"),
                })
            else:
                still_open.append(pos)

        open_positions = still_open

        # ============================================================
        # 2) UPDATE HISTORY & DETECT NEW SHOCK (AT CLOSE)
        # ============================================================
        metric_type = "MTM_BPS"
        if shock_cfg is not None:
            metric_type = getattr(shock_cfg, "metric_type", "MTM_BPS")
        
        if metric_type == "REALIZED_BPS":
            metric_val = period_pnl_bps_realized
        elif metric_type == "MTM_BPS" or metric_type == "BPS":
            metric_val = period_pnl_bps_mtm
        else:
            metric_val = period_pnl_cash

        daily_pnl_history.append(metric_val)
        daily_pnl_dates.append(dts)

        # We now check if TODAY'S results triggered a NEW shock.
        # This new state will affect:
        #   a) Panic Exits (Today) -> Because we can observe the close and exit MOC.
        #   b) Entry Gating (Tomorrow) -> Because we can't travel back in time.
        
        is_new_shock_triggered = False
        
        if shock_cfg is not None:
            win = int(shock_cfg.pnl_window)
            if len(daily_pnl_history) >= win + 2:
                pnl_slice_raw = np.array(daily_pnl_history[-win:])
                
                if shock_cfg.use_raw_pnl:
                    mask_raw = np.isfinite(pnl_slice_raw)
                    if mask_raw.sum() >= 2:
                        pnl_clean = pnl_slice_raw[mask_raw]
                        mu = np.mean(pnl_clean)
                        sd = np.std(pnl_clean, ddof=1)
                        if sd > 1e-9:
                            if np.isfinite(pnl_slice_raw[-1]):
                                last_z = (pnl_slice_raw[-1] - mu) / sd
                                if last_z <= shock_cfg.raw_pnl_z_thresh:
                                    is_new_shock_triggered = True

                if shock_cfg.use_residuals and not is_new_shock_triggered and valid_reg_cols:
                    rel_dates = daily_pnl_dates[-win:]
                    try:
                        sig_slice_raw = sig_lookup.reindex(rel_dates)[valid_reg_cols].values
                        if len(sig_slice_raw) == len(pnl_slice_raw):
                            valid_mask = np.isfinite(pnl_slice_raw) & np.isfinite(sig_slice_raw).all(axis=1)
                            n_obs = valid_mask.sum()
                            n_cols = sig_slice_raw.shape[1] + 1
                            if n_obs >= n_cols + 1:
                                Y_clean = pnl_slice_raw[valid_mask]
                                X_clean = sig_slice_raw[valid_mask]
                                X_final = np.column_stack([np.ones(len(X_clean)), X_clean])
                                try:
                                    beta, _, _, _ = np.linalg.lstsq(X_final, Y_clean, rcond=None)
                                    y_hat = X_final @ beta
                                    resid = Y_clean - y_hat
                                    r_mu = np.mean(resid)
                                    r_sd = np.std(resid, ddof=1)
                                    if r_sd > 1e-9:
                                        last_r_z = (resid[-1] - r_mu) / r_sd
                                        if last_r_z <= shock_cfg.resid_z_thresh:
                                            is_new_shock_triggered = True
                                except np.linalg.LinAlgError:
                                    pass
                    except Exception:
                         pass

        # If new shock, set block for FUTURE
        if is_new_shock_triggered:
            shock_block_remaining = int(shock_cfg.block_length)

        # ============================================================
        # 3) CHECK REGIME & EXECUTE PANIC (IF SHOCK TRIGGERED TODAY)
        # ============================================================
        # Panic logic uses TODAY'S shock status (is_new_shock_triggered or existing block)
        # If we are in a shock (old or new), and mode is EXIT_ALL, get out.
        
        is_in_shock_state = (shock_block_remaining > 0) or is_new_shock_triggered
        
        if is_in_shock_state and SHOCK_MODE == "EXIT_ALL" and len(open_positions) > 0:
            panic_pnl_realized_net = 0.0 
            panic_tcost_bps = 0.0        
            panic_tcost_cash = 0.0
            
            for pos in open_positions:
                pos.closed = True
                pos.close_ts = dts
                pos.exit_reason = "shock_exit"
                
                tcost_bp = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10)) if pos.mode == "overlay" else 0.0
                tcost_cash = tcost_bp * pos.scale_dv01
                
                pos.tcost_bp = tcost_bp
                pos.tcost_cash = tcost_cash
                
                panic_tcost_bps += tcost_bp
                panic_tcost_cash += tcost_cash
                panic_pnl_realized_net += (pos.pnl_bp - tcost_bp)

                closed_rows.append({
                    "open_ts": pos.open_ts, 
                    "close_ts": pos.close_ts, 
                    "exit_reason": pos.exit_reason,
                    "tenor_i": pos.tenor_i, 
                    "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i,
                    "w_j": pos.w_j,
                    "leg_dir_i": float(np.sign(pos.w_i)),
                    "leg_dir_j": float(np.sign(pos.w_j)),
                    "entry_rate_i": pos.entry_rate_i,
                    "entry_rate_j": pos.entry_rate_j,
                    "close_rate_i": getattr(pos, "last_rate_i", np.nan),
                    "close_rate_j": getattr(pos, "last_rate_j", np.nan),
                    "dv01_i_entry": pos.dv01_i_entry,
                    "dv01_j_entry": pos.dv01_j_entry,
                    "dv01_i_close": pos.dv01_i_curr,
                    "dv01_j_close": pos.dv01_j_curr,
                    "initial_dv01": pos.initial_dv01,
                    "scale_dv01": pos.scale_dv01,
                    "entry_zspread": pos.entry_zspread,
                    "conv_proxy": pos.conv_pnl_proxy,
                    "pnl_gross_bp": pos.pnl_bp, 
                    "pnl_gross_cash": pos.pnl_cash,
                    "tcost_bp": tcost_bp, 
                    "tcost_cash": tcost_cash,
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, 
                    "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "days_held_equiv": pos.age_decisions / max(1, decisions_per_day),
                    "mode": pos.mode,
                    "trade_id": pos.meta.get("trade_id"),
                    "side": pos.meta.get("side"),
                })
            
            open_positions = [] 
            
            # RETROACTIVE HISTORY UPDATE
            if metric_type == "REALIZED_BPS":
                daily_pnl_history[-1] += panic_pnl_realized_net
            elif metric_type == "MTM_BPS" or metric_type == "BPS":
                daily_pnl_history[-1] -= panic_tcost_bps
            else: 
                daily_pnl_history[-1] -= panic_tcost_cash

        # ============================================================
        # 4) CHECK REGIME & GATE ENTRIES
        # ============================================================
        # CRITICAL: We use 'was_shock_active_at_open' (Yesterday's state) to gate entries.
        # This prevents looking ahead at today's PnL to avoid today's trades.
        
        regime_ok = True
        if regime_mask is not None:
            if dts in regime_mask.index:
                regime_ok = bool(regime_mask.at[dts])
            else:
                regime_ok = False 

        if (not regime_ok) or was_shock_active_at_open:
            continue

        # ============================================================
        # 5) NEW ENTRIES
        # ============================================================
        remaining_slots = max(0, cr.MAX_CONCURRENT_PAIRS - len(open_positions))
        if remaining_slots <= 0:
            continue

        if mode == "strategy":
            selected = choose_pairs_under_caps(
                snap_last, remaining_slots, PER_BUCKET_DV01_CAP, TOTAL_DV01_CAP, FRONT_END_DV01_CAP, 0.0
            )
            for (cheap, rich, w_i, w_j) in selected:
                t_i, t_j = _to_float(cheap["tenor_yrs"]), _to_float(rich["tenor_yrs"])
                if (assign_bucket(t_i) == "short" or assign_bucket(t_j) == "short"):
                    zd = _to_float(cheap["z_comb"]) - _to_float(rich["z_comb"])
                    if zd < (Z_ENTRY + SHORT_END_EXTRA_Z): continue
                
                pos = PairPos(dts, cheap, rich, w_i, w_j, decisions_per_day, scale_dv01=1.0, mode="strategy")
                open_positions.append(pos)
                ledger_rows.append({"decision_ts": dts, "event": "open", "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j, "mode": "strategy"})

        elif mode == "overlay":
            if hedges is None or hedges.empty: continue
            hedges_here = hedges[hedges["decision_ts"] == dts]
            if hedges_here.empty: continue
            
            if float(hedges_here["dv01"].abs().sum()) > OVERLAY_DV01_TS_CAP: continue

            snap_srt = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
            
            for _, h in hedges_here.iterrows():
                if len(open_positions) >= cr.MAX_CONCURRENT_PAIRS: break
                
                trade_tenor = float(h["tenor_yrs"])
                if trade_tenor < EXEC_LEG_TENOR_THRESHOLD: continue
                
                dv01_cash = float(h["dv01"])
                if abs(dv01_cash) > _per_trade_dv01_cap_for_bucket(assign_bucket(trade_tenor)): continue

                side_sign = 1.0 if str(h["side"]).upper() == "CRCV" else -1.0
                z_entry_eff = _overlay_effective_z_entry(dv01_cash)

                exec_z = _get_z_at_tenor(snap_srt, trade_tenor)
                if exec_z is None: continue
                exec_row = snap_srt.iloc[(snap_srt["tenor_yrs"] - trade_tenor).abs().idxmin()]
                exec_tenor = float(exec_row["tenor_yrs"])

                best_cand = None
                best_zd = 0.0
                
                for _, alt_row in snap_srt.iterrows():
                    alt_tenor = float(alt_row["tenor_yrs"])
                    if alt_tenor < ALT_LEG_TENOR_THRESHOLD: continue
                    if alt_tenor == exec_tenor: continue
                    if not (MIN_SEP_YEARS <= abs(alt_tenor - exec_tenor) <= MAX_SPAN_YEARS): continue
                    
                    z_alt = _to_float(alt_row["z_comb"])
                    zdisp = (z_alt - exec_z) if side_sign > 0 else (exec_z - z_alt)
                    
                    if zdisp < z_entry_eff: continue

                    if (assign_bucket(alt_tenor)=="short" or assign_bucket(exec_tenor)=="short") and (zdisp < z_entry_eff + SHORT_END_EXTRA_Z):
                        continue
                    
                    c_t, r_t = (alt_tenor, exec_tenor) if z_alt > exec_z else (exec_tenor, alt_tenor)
                    if not (fly_alignment_ok(c_t, 1, snap_srt, zdisp_for_pair=zdisp) and fly_alignment_ok(r_t, -1, snap_srt, zdisp_for_pair=zdisp)):
                        continue

                    if zdisp > best_zd:
                        best_zd = zdisp
                        best_cand = (alt_row, exec_row)

                if best_cand:
                    alt_row, exec_row = best_cand
                    
                    rate_i, rate_j = None, None
                    ti, tj = tenor_to_ticker(float(alt_row["tenor_yrs"])), tenor_to_ticker(float(exec_row["tenor_yrs"]))
                    if ti and f"{ti}_mid" in h: rate_i = _to_float(h[f"{ti}_mid"])
                    if tj and f"{tj}_mid" in h: rate_j = _to_float(h[f"{tj}_mid"])
                    
                    if rate_i is None: rate_i = _to_float(alt_row["rate"])
                    if rate_j is None: rate_j = _to_float(exec_row["rate"])

                    pos = PairPos(dts, alt_row, exec_row, side_sign*1.0, side_sign*-1.0, decisions_per_day, scale_dv01=dv01_cash, mode="overlay",
                                  meta={"trade_id": h.get("trade_id"), "side": h.get("side")}, entry_rate_i=rate_i, entry_rate_j=rate_j)
                    open_positions.append(pos)
                    ledger_rows.append({"decision_ts": dts, "event": "open", "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j, "mode": "overlay"})

    pos_df = pd.DataFrame(closed_rows)
    ledger = pd.DataFrame(ledger_rows)
    if not ledger.empty:
        idx = ledger[ledger["event"]=="mark"]["decision_ts"].dt.floor("d" if decision_freq=="D" else "h")
        pnl_by = ledger[ledger["event"]=="mark"].groupby(idx)["pnl_cash"].sum().reset_index(name="pnl_cash").rename(columns={"decision_ts": "bucket"})
    else:
        pnl_by = pd.DataFrame(columns=["bucket", "pnl_cash"])

    return pos_df, ledger, pnl_by, open_positions

# ------------------------
# Multi-month runner
# ------------------------
def run_all(
    yymms: List[str],
    *,
    decision_freq: str | None = None,
    carry: bool = True,
    force_close_end: bool = False,
    mode: str = "strategy",
    hedge_df: Optional[pd.DataFrame] = None,
    overlay_use_caps: Optional[bool] = None,
    
    # --- NEW ARGUMENTS ---
    regime_mask: Optional[pd.Series] = None,
    hybrid_signals: Optional[pd.DataFrame] = None,
    shock_cfg: Optional[ShockConfig] = None,
):
    """
    Run multiple months with support for Regime and Shock filters.
    """
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    mode = mode.lower()

    clean_hedges = None
    if mode == "overlay":
        if hedge_df is None or hedge_df.empty:
            raise ValueError("Overlay mode requires a non-empty hedge_df.")
        clean_hedges = prepare_hedge_tape(hedge_df, decision_freq)

    all_pos, all_ledger, all_by = [], [], []
    open_positions: List[PairPos] = []

    filter_status = "ON" if (regime_mask is not None or shock_cfg is not None) else "OFF"
    print(f"[INFO] months: {len(yymms)} | mode={mode} | filter={filter_status}")
    
    for yymm in yymms:
        hedges_month = None
        if mode == "overlay" and clean_hedges is not None:
            year, month = 2000 + int(yymm[:2]), int(yymm[2:])
            start = pd.Timestamp(year, month, 1)
            end = (start + MonthEnd(1)) + pd.Timedelta(days=1)
            hedges_month = clean_hedges[(clean_hedges["decision_ts"] >= start) & (clean_hedges["decision_ts"] < end)].copy()

        print(f"[RUN] month {yymm}")
        p, l, b, open_positions = run_month(
            yymm,
            decision_freq=decision_freq,
            open_positions=open_positions,
            carry_in=carry,
            mode=mode,
            hedges=hedges_month,
            overlay_use_caps=overlay_use_caps,
            
            # Pass Filter Args
            regime_mask=regime_mask,
            hybrid_signals=hybrid_signals,
            shock_cfg=shock_cfg
        )
        
        if not p.empty: all_pos.append(p.assign(yymm=yymm))
        if not l.empty: all_ledger.append(l.assign(yymm=yymm))
        if not b.empty: all_by.append(b.assign(yymm=yymm))
        print(f"[DONE] {yymm} | closed={len(p)} | open={len(open_positions)}")

    if force_close_end and open_positions:
        final_ts = pd.Timestamp.now()
        if all_ledger: final_ts = max(x["decision_ts"].max() for x in all_ledger if not x.empty)
        
        closed_rows = []
        for pos in open_positions:
            pos.closed, pos.close_ts, pos.exit_reason = True, final_ts, "eoc"
            tcost_bp = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10)) if pos.mode == "overlay" else 0.0
            tcost_cash = tcost_bp * pos.scale_dv01
            closed_rows.append({
                "open_ts": pos.open_ts, "close_ts": pos.close_ts, "exit_reason": "eoc",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "pnl_net_bp": pos.pnl_bp - tcost_bp, "pnl_net_cash": pos.pnl_cash - tcost_cash,
                "mode": pos.mode, "scale_dv01": pos.scale_dv01
            })
        if closed_rows:
            all_pos.append(pd.DataFrame(closed_rows).assign(yymm=yymms[-1]))

    return (
        pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame(),
        pd.concat(all_ledger, ignore_index=True) if all_ledger else pd.DataFrame(),
        pd.concat(all_by, ignore_index=True) if all_by else pd.DataFrame()
    )

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    import hybrid_filter as hf  # Need access to signal builders
    
    if len(sys.argv) < 2:
        print("Usage: python portfolio_test.py 2304 [2305 2306 ...]")
        sys.exit(1)
        
    months = sys.argv[1:]
    
    # 1. Load Trade Tape
    trades_path = Path(f"{cr.TRADE_TYPES}.pkl")
    if not trades_path.exists():
        print(f"[WARN] Trade tape {trades_path} not found. Running in Strategy mode (no overlay).")
        trades = None
        mode_ run = "strategy"
    else:
        trades = pd.read_pickle(trades_path)
        mode_run = cr.RUN_MODE

    # 2. Prepare Filters (Curve + Shock)
    print(f"[INIT] Preparing Hybrid Filters using config settings...")
    
    # A) Get Signals (Feature Engineering)
    signals = hf.get_or_build_hybrid_signals()
    
    # B) Build Regime Mask (Exogenous Curve Filter)
    #    Uses thresholds from cr_config automatically (via hf defaults) or explicit pass:
    regime_mask = hf.regime_mask_from_signals(
        signals, 
        thresholds=hf.RegimeThresholds(
            min_signal_health_z=getattr(cr, "MIN_SIGNAL_HEALTH_Z", -0.5),
            max_trendiness_abs=getattr(cr, "MAX_TRENDINESS_ABS", 2.0),
            max_z_xs_mean_abs_z=getattr(cr, "MAX_Z_XS_MEAN_ABS_Z", 2.0),
        )
    )

    # C) Configure Shock Filter (Endogenous PnL Filter)
    #     Explicitly pull from cr_config so the config file is the boss.
    shock_config = ShockConfig(
        pnl_window=int(getattr(cr, "SHOCK_PNL_WINDOW", 10)),
        use_raw_pnl=bool(getattr(cr, "SHOCK_USE_RAW_PNL", True)),
        use_residuals=bool(getattr(cr, "SHOCK_USE_RESIDUALS", True)),
        raw_pnl_z_thresh=float(getattr(cr, "SHOCK_RAW_PNL_Z_THRESH", -1.5)),
        resid_z_thresh=float(getattr(cr, "SHOCK_RESID_Z_THRESH", -1.5)),
        regression_cols=list(getattr(cr, "SHOCK_REGRESSION_COLS", [])),
        block_length=int(getattr(cr, "SHOCK_BLOCK_LENGTH", 10)),
    )

    # 3. Run Backtest
    print(f"[EXEC] Running {mode_run} on {len(months)} months with filters ACTIVE.")
    pos, led, by = run_all(
        months, 
        carry=True, 
        force_close_end=False, 
        mode=mode_run, 
        hedge_df=trades,
        
        # Inject the filters
        regime_mask=regime_mask,
        hybrid_signals=signals,
        shock_cfg=shock_config
    )

    # 4. Save Outputs
    out_dir = Path(cr.PATH_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = getattr(cr, "OUT_SUFFIX", "")
    
    if not pos.empty: 
        pos.to_parquet(out_dir / f"positions_ledger{suffix}.parquet")
    if not led.empty: 
        led.to_parquet(out_dir / f"marks_ledger{suffix}.parquet")
    if not by.empty:  
        by.to_parquet(out_dir / f"pnl_by_bucket{suffix}.parquet")
        
    print(f"[DONE] Results saved to {out_dir}")
