import os, sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd

# All config access via module namespace
import cr_config as cr


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
        df["decision_ts"] = df["trade_ts"].dt.floor("D")
    elif decision_freq == "H":
        df["decision_ts"] = df["trade_ts"].dt.floor("H")
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


# ------------------------
# Pair object
# ------------------------
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
    ):
        """
        cheap_row / rich_row naming is historical; in overlay mode, these can be
        interpreted as (alt_leg, exec_leg). Orientation is fully captured by w_i/w_j
        and side sign when we construct the pair.

        dir_sign:
          - If provided, this is the "direction" in z-space we care about for exits.
          - If None, we default to sign(entry_zspread) (cheap minus rich).
        """
        self.open_ts = open_ts

        self.tenor_i = _to_float(cheap_row["tenor_yrs"])
        self.rate_i  = _to_float(cheap_row["rate"])
        self.tenor_j = _to_float(rich_row["tenor_yrs"])
        self.rate_j  = _to_float(rich_row["rate"])

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

        # PnL bookkeeping
        self.pnl_bp: float = 0.0      # per-unit in bps (bp × unit-DV01)
        self.pnl_cash: float = 0.0    # scaled by scale_dv01
        self.pnl: float = 0.0         # alias, for backward compatibility (cash units)
        self.scale_dv01: float = float(scale_dv01)
        self.mode: str = str(mode)
        self.meta: Dict = meta or {}

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

    def mark(self, snap_last: pd.DataFrame):
        """Mark-to-market at decision time using last rate per tenor; update convergence proxy."""
        ri = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "rate"])
        rj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "rate"])

        # Pair PnL per unit: (old - new) * weight * 100  (rates in % → *100 to bps)
        d_i = (self.rate_i - ri) * self.w_i * 100.0
        d_j = (self.rate_j - rj) * self.w_j * 100.0
        pnl_bp = d_i + d_j
        self.pnl_bp = pnl_bp

        pnl_cash = pnl_bp * self.scale_dv01
        self.pnl_cash = pnl_cash
        self.pnl = pnl_cash  # alias

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


# ------------------------
# Month runner
# ------------------------
def run_month(
    yymm: str,
    *,
    decision_freq: str | None = None,
    open_positions: Optional[List[PairPos]] | None = None,   # carry-in
    carry_in: bool = True,
    mode: str = "strategy",                                  # "strategy" or "overlay"
    hedges: Optional[pd.DataFrame] = None,
    overlay_use_caps: Optional[bool] = None
):
    """
    Run a single month.

    mode="strategy": use cheap–rich selector (Mode A).
    mode="overlay" : open pairs driven by hedge tape (hedges df) as an overlay.

    Returns: (closed_positions_df, ledger_df, pnl_by_df, open_positions_out)
    where open_positions_out are the still-open positions to carry into next month.
    """
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    mode = mode.lower()
    overlay_use_caps = cr.OVERLAY_USE_CAPS if overlay_use_caps is None else bool(overlay_use_caps)

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

    # Decision buckets & decisions-per-day
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    if decision_freq == "D":
        df["decision_ts"] = df["ts"].dt.floor("D")
        decisions_per_day = 1
    elif decision_freq == "H":
        df["decision_ts"] = df["ts"].dt.floor("H")
        # compute actual # decisions per trading day in this month
        per_day_counts = df.groupby(df["decision_ts"].dt.floor("D"))["decision_ts"].nunique()
        decisions_per_day = int(per_day_counts.mean()) if len(per_day_counts) else 24
    else:
        raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

    max_hold_decisions = cr.MAX_HOLD_DAYS * decisions_per_day

    # Carry-in
    open_positions = (open_positions or []) if carry_in else []

    # Filter hedges for the decision_ts that actually exist in this month
    if mode == "overlay" and hedges is not None and not hedges.empty:
        valid_decisions = df["decision_ts"].dropna().unique()
        hedges = hedges[hedges["decision_ts"].isin(valid_decisions)].copy()
    else:
        hedges = None

    ledger_rows: list[dict] = []
    closed_rows: list[dict] = []

    # Overlay caps (per decision snapshot)
    OVERLAY_CAPS_ON = overlay_use_caps
    PER_BUCKET_DV01_CAP  = float(getattr(cr, "PER_BUCKET_DV01_CAP", 1.0))
    TOTAL_DV01_CAP       = float(getattr(cr, "TOTAL_DV01_CAP", 3.0))
    FRONT_END_DV01_CAP   = float(getattr(cr, "FRONT_END_DV01_CAP", 1.0))
    Z_ENTRY              = float(getattr(cr, "Z_ENTRY", 0.75))
    SHORT_END_EXTRA_Z    = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))
    Z_EXIT               = float(getattr(cr, "Z_EXIT", 0.40))
    Z_STOP               = float(getattr(cr, "Z_STOP", 3.00))

    for dts, snap in df.groupby("decision_ts", sort=True):
        snap_last = (
            snap.sort_values("ts")
                .groupby("tenor_yrs", as_index=False)
                .tail(1)
                .reset_index(drop=True)
        )
        if snap_last.empty:
            continue

        # 1) Mark & evaluate exits on any open (carried + this month’s)
        still_open: list[PairPos] = []
        for pos in open_positions:
            zsp = pos.mark(snap_last)
            entry_z = pos.entry_zspread

            Z_EXIT = float(getattr(cr, "Z_EXIT", 0.40))
            Z_STOP = float(getattr(cr, "Z_STOP", 3.00))

            exit_flag = None

            if np.isfinite(zsp) and np.isfinite(entry_z):
                # Directional z at entry / current (cheap–rich space)
                entry_dir = getattr(pos, "entry_z_dir", pos.dir_sign * entry_z)
                curr_dir  = getattr(pos, "last_z_dir", pos.dir_sign * zsp)

                if np.isfinite(entry_dir) and np.isfinite(curr_dir):
                    sign_entry = np.sign(entry_dir)
                    sign_curr  = np.sign(curr_dir)

                    same_side = (sign_entry != 0) and (sign_entry == sign_curr)

                    # Has z moved toward zero relative to entry?
                    moved_towards_zero = abs(curr_dir) <= abs(entry_dir)

                    # Is current z inside the exit band?
                    within_exit_band = abs(curr_dir) <= Z_EXIT

                    # Directional change from entry
                    dz_dir = curr_dir - entry_dir
                    moved_away = (
                        same_side
                        and (abs(curr_dir) >= abs(entry_dir))
                        and (abs(dz_dir) >= Z_STOP)
                    )

                    # 1) Reversion: same side of zero, closer to zero, inside exit band
                    if same_side and moved_towards_zero and within_exit_band:
                        exit_flag = "reversion"
                    # 2) Stop: same side, but materially further from zero than entry
                    elif moved_away:
                        exit_flag = "stop"

            # 3) Max-hold if nothing else triggered
            if exit_flag is None and pos.age_decisions >= max_hold_decisions:
                exit_flag = "max_hold"

            if exit_flag is not None:
                pos.closed = True
                pos.close_ts = dts
                pos.exit_reason = exit_flag

            # Ledger mark (pnl in bps and cash)
            ledger_rows.append({
                "decision_ts": dts,
                "event": "mark",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "w_i": pos.w_i, "w_j": pos.w_j,
                "pnl_bp": pos.pnl_bp,
                "pnl_cash": pos.pnl_cash,
                "z_spread": zsp,
                "conv_proxy": pos.conv_pnl_proxy,
                "open_ts": pos.open_ts,
                "closed": pos.closed,
                "exit_reason": pos.exit_reason,
                "mode": pos.mode,
                "scale_dv01": pos.scale_dv01
            })

            if pos.closed:
                # Transaction cost only for overlay mode
                if pos.mode == "overlay":
                    tcost_bp = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10))
                    tcost_cash = tcost_bp * pos.scale_dv01
                else:
                    tcost_bp = 0.0
                    tcost_cash = 0.0
                pos.tcost_bp = tcost_bp
                pos.tcost_cash = tcost_cash

                closed_rows.append({
                    "open_ts": pos.open_ts,
                    "close_ts": pos.close_ts,
                    "exit_reason": pos.exit_reason,
                    "tenor_i": pos.tenor_i,
                    "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i,
                    "w_j": pos.w_j,
                    "entry_zspread": pos.entry_zspread,
                    "pnl_gross_bp": pos.pnl_bp,
                    "pnl_gross_cash": pos.pnl_cash,
                    "tcost_bp": tcost_bp,
                    "tcost_cash": tcost_cash,
                    "pnl_net_bp": pos.pnl_bp - tcost_bp,
                    "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "days_held_equiv": pos.age_decisions / decisions_per_day,
                    "conv_proxy": pos.conv_pnl_proxy,
                    "mode": pos.mode,
                    "scale_dv01": pos.scale_dv01
                })
            else:
                still_open.append(pos)

        open_positions = still_open

        # 2) New entries under caps / hedge overlay
        remaining_slots = max(0, cr.MAX_CONCURRENT_PAIRS - len(open_positions))
        if remaining_slots <= 0:
            continue

        if mode == "strategy":
            # Mode A: cheap–rich selector (unchanged logic)
            selected = choose_pairs_under_caps(
                snap_last=snap_last,
                max_pairs=remaining_slots,
                per_bucket_cap=cr.PER_BUCKET_DV01_CAP,
                total_cap=cr.TOTAL_DV01_CAP,
                front_end_cap=cr.FRONT_END_DV01_CAP,
                extra_z_entry=0.0  # short-end extra handled inside chooser using SHORT_END_EXTRA_Z
            )

            for (cheap, rich, w_i, w_j) in selected:
                t_i = _to_float(cheap["tenor_yrs"])
                t_j = _to_float(rich["tenor_yrs"])
                if (assign_bucket(t_i) == "short") or (assign_bucket(t_j) == "short"):
                    zdisp = _to_float(cheap["z_comb"]) - _to_float(rich["z_comb"])
                    if not np.isfinite(zdisp) or (zdisp < (Z_ENTRY + SHORT_END_EXTRA_Z)):
                        continue

                pos = PairPos(
                    open_ts=dts,
                    cheap_row=cheap,
                    rich_row=rich,
                    w_i=w_i,
                    w_j=w_j,
                    decisions_per_day=decisions_per_day,
                    scale_dv01=1.0,
                    mode="strategy",
                    meta={}
                )
                open_positions.append(pos)
                ledger_rows.append({
                    "decision_ts": dts,
                    "event": "open",
                    "tenor_i": pos.tenor_i,
                    "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i,
                    "w_j": pos.w_j,
                    "entry_zspread": pos.entry_zspread,
                    "mode": pos.mode,
                    "scale_dv01": pos.scale_dv01
                })

        elif mode == "overlay":
            # Mode B: hedge overlay – open synthetic pairs driven by hedge tape
            if hedges is None or hedges.empty:
                continue

            hedges_here = hedges[hedges["decision_ts"] == dts]
            if hedges_here.empty:
                continue

            # Overlay caps per decision snapshot, in DV01 space
            bucket_dv01 = {b: 0.0 for b in getattr(cr, "BUCKETS", {}).keys()}
            total_dv01 = 0.0

            # Pre-index snap_last by tenor_yrs for quick lookups
            snap_last_sorted = snap_last.sort_values("tenor_yrs").reset_index(drop=True)

            for _, h in hedges_here.iterrows():
                if len(open_positions) >= cr.MAX_CONCURRENT_PAIRS:
                    break

                trade_tenor = float(h["tenor_yrs"])

                MIN_LEG_TENOR = float(getattr(cr, "MIN_LEG_TENOR_YEARS", 0.0))
                # If the executed hedge itself is too short, skip overlay entirely
                if trade_tenor < MIN_LEG_TENOR:
                    continue

                side = str(h["side"]).upper()
                dv01_cash = float(h["dv01"])
                trade_id = h.get("trade_id", None)

                # Side sign is only used for better-tenor logic; pair is DV01-native unit.
                if side == "CRCV":
                    side_sign = +1.0
                elif side == "CPAY":
                    side_sign = -1.0
                else:
                    continue

                # Find executed tenor row (nearest within tolerance)
                exec_z = _get_z_at_tenor(snap_last_sorted, trade_tenor)
                if exec_z is None or not np.isfinite(exec_z):
                    continue
                exec_row = snap_last_sorted.iloc[
                    (snap_last_sorted["tenor_yrs"] - trade_tenor).abs().idxmin()
                ]
                exec_tenor = float(exec_row["tenor_yrs"])

                # Scan for alternative tenors
                best_candidate = None
                best_zdisp = 0.0
                MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
                MAX_SPAN_YEARS = float(getattr(cr, "MAX_SPAN_YEARS", 10.0))

                for _, alt_row in snap_last_sorted.iterrows():
                    alt_tenor = float(alt_row["tenor_yrs"])
                    if alt_tenor == exec_tenor:
                        continue
                
                    # Enforce min tenor on both legs of the potential pair
                    if (alt_tenor < MIN_LEG_TENOR) or (exec_tenor < MIN_LEG_TENOR):
                        continue

                    diff = abs(alt_tenor - exec_tenor)
                    if diff < MIN_SEP_YEARS or diff > MAX_SPAN_YEARS:
                        continue

                    z_alt = _to_float(alt_row["z_comb"])
                    if not np.isfinite(z_alt):
                        continue

                    # Direction-dependent "better tenor" logic:
                    #   CRCV: prefer higher z than executed
                    #   CPAY: prefer lower z than executed
                    if side == "CRCV":
                        if z_alt <= exec_z:
                            continue
                        zdisp = z_alt - exec_z
                    else:  # CPAY
                        if z_alt >= exec_z:
                            continue
                        zdisp = exec_z - z_alt

                    if zdisp < Z_ENTRY:
                        continue

                    # Fly gate: classify cheap/rich purely by z
                    if z_alt > exec_z:
                        cheap_tenor, rich_tenor = alt_tenor, exec_tenor
                    else:
                        cheap_tenor, rich_tenor = exec_tenor, alt_tenor

                    ok_i = fly_alignment_ok(cheap_tenor, +1.0, snap_last_sorted, zdisp_for_pair=zdisp)
                    ok_j = fly_alignment_ok(rich_tenor, -1.0, snap_last_sorted, zdisp_for_pair=zdisp)
                    if not (ok_i and ok_j):
                        continue

                    # Keep best (largest zdisp) candidate
                    if zdisp > best_zdisp:
                        best_zdisp = zdisp
                        best_candidate = (alt_row, exec_row, z_alt, exec_z)

                if best_candidate is None:
                    continue

                alt_row, exec_row, z_alt, z_exec = best_candidate
                alt_tenor = float(alt_row["tenor_yrs"])
                exec_tenor = float(exec_row["tenor_yrs"])

                # DV01-native unit pair:
                #   - no PV01 scaling
                #   - |w_i| = |w_j| = 1
                #   - orientation via side_sign
                w_alt = side_sign * 1.0
                w_exec = side_sign * -1.0

                # Optional overlay caps – apply directly to dv01_trade by bucket & total
                if OVERLAY_CAPS_ON:
                    b_alt = assign_bucket(alt_tenor)
                    b_exec = assign_bucket(exec_tenor)
                    dv = abs(dv01_cash)

                    # Per-bucket caps (each leg consumes dv01_trade in its own bucket)
                    if b_alt in bucket_dv01 and (bucket_dv01[b_alt] + dv) > PER_BUCKET_DV01_CAP:
                        continue
                    if b_exec in bucket_dv01 and (bucket_dv01[b_exec] + dv) > PER_BUCKET_DV01_CAP:
                        continue

                    # Front-end aggregate cap
                    short_add = (dv if b_alt == "short" else 0.0) + (dv if b_exec == "short" else 0.0)
                    short_tot = bucket_dv01.get("short", 0.0)
                    if (short_tot + short_add) > FRONT_END_DV01_CAP:
                        continue

                    # Total cap (sum of abs DV01 across both legs)
                    if (total_dv01 + 2.0 * dv) > TOTAL_DV01_CAP:
                        continue

                    # Short-end extra threshold (overlay) if either leg is short bucket
                    if (b_alt == "short" or b_exec == "short") and (best_zdisp < (Z_ENTRY + SHORT_END_EXTRA_Z)):
                        continue

                    bucket_dv01[b_alt] = bucket_dv01.get(b_alt, 0.0) + dv
                    bucket_dv01[b_exec] = bucket_dv01.get(b_exec, 0.0) + dv
                    total_dv01 += 2.0 * dv

                # Build PairPos with scale_dv01 = hedge DV01 (cash)
                # Treat alt_row as "cheap_row" and exec_row as "rich_row" for zspread bookkeeping.
                pos = PairPos(
                    open_ts=dts,
                    cheap_row=alt_row,
                    rich_row=exec_row,
                    w_i=w_alt,
                    w_j=w_exec,
                    decisions_per_day=decisions_per_day,
                    scale_dv01=dv01_cash,
                    mode="overlay",
                    meta={"trade_id": trade_id, "side": side}
                )
                open_positions.append(pos)

                ledger_rows.append({
                    "decision_ts": dts,
                    "event": "open",
                    "tenor_i": pos.tenor_i,
                    "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i,
                    "w_j": pos.w_j,
                    "entry_zspread": pos.entry_zspread,
                    "mode": pos.mode,
                    "scale_dv01": pos.scale_dv01,
                    "trade_id": trade_id,
                    "side": side
                })

        # else: unknown mode – nothing opened

    # Outputs for this month (closed only)
    pos_df = pd.DataFrame(closed_rows)
    ledger = pd.DataFrame(ledger_rows)

    # PnL by bucket from marks (we aggregate in cash terms; bp is still in ledger)
    if not ledger.empty:
        marks = ledger[ledger["event"] == "mark"].copy()
        idx = marks["decision_ts"].dt.floor("D" if decision_freq == "D" else "H")
        pnl_by = marks.groupby(idx)["pnl_cash"].sum().rename("pnl_cash").to_frame().reset_index()
        pnl_by = pnl_by.rename(columns={"decision_ts": "bucket"})
    else:
        pnl_by = pd.DataFrame(columns=["bucket", "pnl_cash"])

    return pos_df, ledger, pnl_by, open_positions  # carry-out

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
    overlay_use_caps: Optional[bool] = None
):
    """
    Run multiple months, carrying open positions across months when carry=True.

    mode="strategy": existing cheap–rich strategy (no hedge tape).
    mode="overlay" : hedge-overlay mode driven by hedge_df (trade tape).
                     hedge_df must contain: side, instrument, EqVolDelta, tradetimeUTC.

    If force_close_end=True, anything still open after the last month is
    closed at the final bucket time (exit_reason='eoc' end-of-cycle).

    Returns concatenated (positions_closed, ledger, pnl_by).
    """
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    mode = mode.lower()

    # Prepare hedge tape up front for overlay mode
    clean_hedges = None
    if mode == "overlay":
        if hedge_df is None or hedge_df.empty:
            raise ValueError("Overlay mode requires a non-empty hedge_df (trade tape).")
        clean_hedges = prepare_hedge_tape(hedge_df, decision_freq)

    all_pos, all_ledger, all_by = [], [], []
    open_positions: List[PairPos] = []

    print(f"[INFO] months: {len(yymms)} -> {yymms} | mode={mode}")
    for yymm in yymms:
        # Slice hedges for this month (overlay mode only)
        hedges_month = None
        if mode == "overlay" and clean_hedges is not None and not clean_hedges.empty:
            year = 2000 + int(yymm[:2])
            month = int(yymm[2:])
            start = pd.Timestamp(year=year, month=month, day=1)
            end = (start + MonthEnd(1)) + pd.Timedelta(days=1)
            hedges_month = clean_hedges[
                (clean_hedges["decision_ts"] >= start) &
                (clean_hedges["decision_ts"] < end)
            ].copy()

        print(f"[RUN] month {yymm}")
        p, l, b, open_positions = run_month(
            yymm,
            decision_freq=decision_freq,
            open_positions=open_positions,
            carry_in=carry,
            mode=mode,
            hedges=hedges_month,
            overlay_use_caps=overlay_use_caps
        )
        if not p.empty:
            all_pos.append(p.assign(yymm=yymm))
        if not l.empty:
            all_ledger.append(l.assign(yymm=yymm))
        if not b.empty:
            all_by.append(b.assign(yymm=yymm))
        print(f"[DONE] {yymm} | closed={len(p)} | open_carry={len(open_positions)}")

    # Optionally force-close what's left after the last month for tidy reporting
    if force_close_end and open_positions:
        # final timestamp = last bucket of last processed month (from ledger/by if present)
        if all_ledger:
            final_ts = max(x["decision_ts"].max() for x in all_ledger if not x.empty)
        else:
            # Fallback: synthesize month end
            final_ts = pd.Timestamp.now()
        closed_rows = []
        for pos in open_positions:
            pos.closed, pos.close_ts, pos.exit_reason = True, final_ts, "eoc"
            # Transaction cost on forced close only for overlay
            if pos.mode == "overlay":
                tcost_bp = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10))
                tcost_cash = tcost_bp * pos.scale_dv01
            else:
                tcost_bp = 0.0
                tcost_cash = 0.0
            closed_rows.append({
                "open_ts": pos.open_ts,
                "close_ts": pos.close_ts,
                "exit_reason": pos.exit_reason,
                "tenor_i": pos.tenor_i,
                "tenor_j": pos.tenor_j,
                "w_i": pos.w_i,
                "w_j": pos.w_j,
                "entry_zspread": pos.entry_zspread,
                "pnl_gross_bp": pos.pnl_bp,
                "pnl_gross_cash": pos.pnl_cash,
                "tcost_bp": tcost_bp,
                "tcost_cash": tcost_cash,
                "pnl_net_bp": pos.pnl_bp - tcost_bp,
                "pnl_net_cash": pos.pnl_cash - tcost_cash,
                "days_held_equiv": pos.age_decisions / max(1, pos.decisions_per_day),
                "conv_proxy": pos.conv_pnl_proxy,
                "mode": pos.mode,
                "scale_dv01": pos.scale_dv01
            })
        if closed_rows:
            all_pos.append(pd.DataFrame(closed_rows).assign(yymm=yymms[-1]))

    pos = pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame()
    led = pd.concat(all_ledger, ignore_index=True) if all_ledger else pd.DataFrame()
    by  = pd.concat(all_by,  ignore_index=True) if all_by  else pd.DataFrame()
    return pos, led, by


# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python portfolio_test.py 2304 [2305 2306 ...]")
        sys.exit(1)
    months = sys.argv[1:]
    # CLI uses strategy mode by default (no hedge tape)
    pos, led, by = run_all(months, carry=True, force_close_end=False, mode="strategy")
    out_dir = Path(cr.PATH_OUT); out_dir.mkdir(parents=True, exist_ok=True)
    suffix = getattr(cr, "OUT_SUFFIX", "")
    if not pos.empty: pos.to_parquet(out_dir / f"positions_ledger{suffix}.parquet")
    if not led.empty: led.to_parquet(out_dir / f"marks_ledger{suffix}.parquet")
    if not by.empty:  by.to_parquet(out_dir / f"pnl_by_bucket{suffix}.parquet")
    print("[DONE]")
