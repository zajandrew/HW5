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
    """Safe scalar float extraction."""
    try:
        if isinstance(x, (pd.Series, pd.Index)):
            if len(x) == 0: return default
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
# Hedge tape helpers
# ------------------------
def _map_instrument_to_tenor(instr: str) -> Optional[float]:
    if instr is None or not isinstance(instr, str): return None
    instr = instr.strip()
    mapped = cr.BBG_DICT.get(instr, instr)
    tenor = cr.TENOR_YEARS.get(mapped)
    return float(tenor) if tenor is not None else None

def prepare_hedge_tape(raw_df: pd.DataFrame, decision_freq: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty: return pd.DataFrame()
    df = raw_df.copy()
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

# ------------------------
# Fly & Curve Logic
# ------------------------
def _get_z_at_tenor(snap_last: pd.DataFrame, tenor: float, tol: float | None = None) -> float | None:
    if tol is None: tol = float(getattr(cr, "FLY_TENOR_TOL_YEARS", 0.02))
    t = float(tenor)
    s = snap_last[["tenor_yrs", "z_comb"]].dropna()
    if s.empty: return None
    s = s.assign(_dist=(s["tenor_yrs"] - t).abs())
    row = s.loc[s["_dist"].idxmin()]
    if row["_dist"] <= tol: return float(row["z_comb"])
    return None

def compute_fly_z(snap_last: pd.DataFrame, a: float, b: float, c: float) -> float | None:
    try:
        z_a, z_b, z_c = _get_z_at_tenor(snap_last, a), _get_z_at_tenor(snap_last, b), _get_z_at_tenor(snap_last, c)
        if any(v is None for v in (z_a, z_b, z_c)): return None
        xs = snap_last["z_comb"].astype(float).to_numpy()
        sd = np.nanstd(xs, ddof=1) if xs.size > 1 else 1.0
        return (0.5*(z_a + z_c) - z_b) / (sd if sd > 0 else 1.0)
    except Exception: return None

def fly_alignment_ok(leg_tenor, leg_sign_z, snap_last, *, zdisp_for_pair=None):
    if not getattr(cr, "FLY_ENABLE", True) or getattr(cr, "FLY_MODE", "tolerant") == "off": return True
    if getattr(cr, "FLY_ALLOW_BIG_ZDISP", True) and zdisp_for_pair and (zdisp_for_pair >= getattr(cr, "Z_ENTRY", 0.75) + getattr(cr, "FLY_BIG_ZDISP_MARGIN", 0.20)): return True
    
    skip = getattr(cr, "FLY_SKIP_SHORT_UNDER", None)
    if skip and leg_tenor < float(skip): return True

    triplets = getattr(cr, "FLY_DEFS", [])
    if getattr(cr, "FLY_NEIGHBOR_ONLY", True):
        W = float(getattr(cr, "FLY_WINDOW_YEARS", 3.0))
        triplets = [t for t in triplets if abs(t[1] - leg_tenor) <= W]
        if not triplets: return True

    contras = 0
    min_z = float(getattr(cr, "FLY_Z_MIN", 0.8))
    for (a,b,c) in triplets:
        fz = compute_fly_z(snap_last, a, b, c)
        if fz and abs(fz) >= min_z and np.sign(fz) * np.sign(leg_sign_z) < 0:
            contras += 1

    mode = getattr(cr, "FLY_MODE", "tolerant").lower()
    req = int(getattr(cr, "FLY_REQUIRE_COUNT", 2))
    if mode == "strict": return contras == 0
    if mode == "loose": return contras <= 1
    return contras <= req

def tenor_to_ticker(tenor_yrs, tol=1e-6):
    if tenor_yrs is None: return None
    t = float(tenor_yrs)
    best_k, best_d = None, float("inf")
    for k, v in cr.TENOR_YEARS.items():
        try: 
            d = abs(float(v) - t)
            if d < best_d: best_d, best_k = d, k
        except: continue
    return best_k if best_d <= tol else None

def _get_funding_rate(snap_last: pd.DataFrame) -> float:
    """Finds proxy for funding rate (shortest available tenor)."""
    try:
        if snap_last.empty: return 0.0
        sorted_snap = snap_last.sort_values("tenor_yrs")
        return float(sorted_snap.iloc[0]["rate"])
    except:
        return 0.0

# ------------------------
# Pair object (With Carry/Roll Logic)
# ------------------------
class PairPos:
    def __init__(self, open_ts, cheap_row, rich_row, w_i, w_j, decisions_per_day, *, 
                 scale_dv01=1.0, mode="strategy", meta=None, dir_sign=None, 
                 entry_rate_i=None, entry_rate_j=None):
        self.open_ts = open_ts
        self.tenor_i = _to_float(cheap_row["tenor_yrs"])
        self.tenor_j = _to_float(rich_row["tenor_yrs"])
        self.tenor_i_orig, self.tenor_j_orig = self.tenor_i, self.tenor_j
        
        def_rate_i = _to_float(cheap_row["rate"])
        def_rate_j = _to_float(rich_row["rate"])
        self.entry_rate_i = _to_float(entry_rate_i, default=def_rate_i) if entry_rate_i is not None else def_rate_i
        self.entry_rate_j = _to_float(entry_rate_j, default=def_rate_j) if entry_rate_j is not None else def_rate_j
        
        self.last_rate_i, self.last_rate_j = self.entry_rate_i, self.entry_rate_j
        self.w_i, self.w_j = float(w_i), float(w_j)
        
        zi, zj = _to_float(cheap_row["z_comb"]), _to_float(rich_row["z_comb"])
        self.entry_zspread = zi - zj
        
        if dir_sign is None or dir_sign == 0:
            self.dir_sign = float(np.sign(self.entry_zspread)) if self.entry_zspread != 0 else 1.0
        else:
            self.dir_sign = float(dir_sign)
        self.entry_z_dir = self.dir_sign * self.entry_zspread

        self.closed, self.close_ts, self.exit_reason = False, None, None
        self.scale_dv01, self.mode = float(scale_dv01), str(mode)
        self.meta = meta or {}
        self.initial_dv01 = self.scale_dv01
        
        self.dv01_i_entry = self.scale_dv01 * self.w_i
        self.dv01_j_entry = self.scale_dv01 * self.w_j
        self.dv01_i_curr, self.dv01_j_curr = self.dv01_i_entry, self.dv01_j_entry
        self.rem_tenor_i, self.rem_tenor_j = self.tenor_i_orig, self.tenor_j_orig
        
        # State for Incremental Carry
        self.last_mark_ts = open_ts 
        
        # PnL Components (Cash)
        self.pnl_price_cash = 0.0
        self.pnl_carry_cash = 0.0
        self.pnl_roll_cash = 0.0
        self.pnl_cash = 0.0
        
        # PnL Components (Bps)
        self.pnl_price_bp = 0.0
        self.pnl_carry_bp = 0.0
        self.pnl_roll_bp = 0.0
        self.pnl_bp = 0.0
        
        self.tcost_bp, self.tcost_cash = 0.0, 0.0
        self.decisions_per_day = decisions_per_day
        self.age_decisions = 0
        self.bucket_i, self.bucket_j = assign_bucket(self.tenor_i), assign_bucket(self.tenor_j)
        self.last_zspread, self.last_z_dir, self.conv_pnl_proxy = self.entry_zspread, self.entry_z_dir, 0.0

    def _update_overlay_dv01(self, decision_ts):
        if not isinstance(decision_ts, pd.Timestamp): return
        days = max(0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        yr_pass = days / 360.0
        self.rem_tenor_i = max(self.tenor_i_orig - yr_pass, 0.0)
        self.rem_tenor_j = max(self.tenor_j_orig - yr_pass, 0.0)
        
        fi = self.rem_tenor_i / max(self.tenor_i_orig, 1e-6)
        fj = self.rem_tenor_j / max(self.tenor_j_orig, 1e-6)
        self.dv01_i_curr = self.dv01_i_entry * fi
        self.dv01_j_curr = self.dv01_j_entry * fj

    def mark(self, snap_last: pd.DataFrame, decision_ts: Optional[pd.Timestamp] = None):
        # 0. Update Decay
        if self.mode == "overlay" and decision_ts:
            self._update_overlay_dv01(decision_ts)

        # 1. Get Market Data
        ri = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "rate"])
        rj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "rate"])
        r_float = _get_funding_rate(snap_last)
        self.last_rate_i, self.last_rate_j = ri, rj

        # 2. Price PnL (Delta)
        pnl_price_i = (self.entry_rate_i - ri) * 100.0 * self.dv01_i_curr
        pnl_price_j = (self.entry_rate_j - rj) * 100.0 * self.dv01_j_curr
        self.pnl_price_cash = pnl_price_i + pnl_price_j

        # 3. Carry PnL (Coupon) - Incremental Accumulation
        if decision_ts and self.last_mark_ts:
            dt_days = max(0.0, (decision_ts.normalize() - self.last_mark_ts.normalize()).days)
            if dt_days > 0:
                inc_carry_i = (self.entry_rate_i - r_float) * 100.0 * self.dv01_i_curr * (dt_days / 360.0)
                inc_carry_j = (self.entry_rate_j - r_float) * 100.0 * self.dv01_j_curr * (dt_days / 360.0)
                self.pnl_carry_cash += (inc_carry_i + inc_carry_j)
            self.last_mark_ts = decision_ts

        # 4. Roll-Down PnL
        days_total = 0.0
        if decision_ts:
            days_total = max(0.0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        
        xp = snap_last["tenor_yrs"].values
        fp = snap_last["rate"].values
        
        t_roll_i = max(0.0, self.tenor_i_orig - (days_total/360.0))
        y_roll_i = np.interp(t_roll_i, xp, fp)
        roll_gain_i = (ri - y_roll_i) * 100.0 * self.dv01_i_curr
        
        t_roll_j = max(0.0, self.tenor_j_orig - (days_total/360.0))
        y_roll_j = np.interp(t_roll_j, xp, fp)
        roll_gain_j = (rj - y_roll_j) * 100.0 * self.dv01_j_curr
        
        self.pnl_roll_cash = roll_gain_i + roll_gain_j

        # 5. Total Cash
        self.pnl_cash = self.pnl_price_cash + self.pnl_carry_cash + self.pnl_roll_cash
        
        # 6. Total Bps (and components)
        if self.scale_dv01 != 0.0 and np.isfinite(self.scale_dv01):
            self.pnl_bp = self.pnl_cash / self.scale_dv01
            self.pnl_price_bp = self.pnl_price_cash / self.scale_dv01
            self.pnl_carry_bp = self.pnl_carry_cash / self.scale_dv01
            self.pnl_roll_bp = self.pnl_roll_cash / self.scale_dv01
        else:
            self.pnl_bp = 0.0
            self.pnl_price_bp = 0.0
            self.pnl_carry_bp = 0.0
            self.pnl_roll_bp = 0.0

        # Z-Score Logic
        zi = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "z_comb"])
        zj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "z_comb"])
        zsp = zi - zj

        if np.isfinite(zsp) and np.isfinite(self.last_zspread):
            self.conv_pnl_proxy += (self.last_zspread - zsp) * 10.0
        self.last_zspread = zsp

        if np.isfinite(zsp) and np.isfinite(self.dir_sign):
            self.last_z_dir = self.dir_sign * zsp
        else:
            self.last_z_dir = np.nan

        self.age_decisions += 1
        return zsp

# ------------------------
# Selection Logic
# ------------------------
def choose_pairs_under_caps(snap_last, max_pairs, per_bucket_cap, total_cap, front_end_cap, extra_z_entry):
    cols_need = {"tenor_yrs", "rate", "z_comb"}
    if not cols_need.issubset(snap_last.columns): return []
    sig = snap_last[list(cols_need)].dropna().copy().sort_values("z_comb", kind="mergesort")
    if sig.empty: return []

    candidates, used = [], set()
    low_take, high_take = min(5, len(sig)), min(8, len(sig))
    
    MIN_SEP = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    MAX_SPAN = float(getattr(cr, "MAX_SPAN_YEARS", 10))
    Z_ENT = float(getattr(cr, "Z_ENTRY", 0.75))
    SHORT_EXTRA = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))
    MIN_TEN = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.5))

    for k_low in range(low_take):
        rich = sig.iloc[k_low]
        for k_hi in range(1, high_take + 1):
            cheap = sig.iloc[-k_hi]
            Ti, Tj = float(cheap["tenor_yrs"]), float(rich["tenor_yrs"])
            if Ti in used or Tj in used: continue
            if (Ti < MIN_TEN) or (Tj < MIN_TEN): continue
            if abs(Ti - Tj) < MIN_SEP or abs(Ti - Tj) > MAX_SPAN: continue
            
            zdisp = float(cheap["z_comb"] - rich["z_comb"])
            if zdisp < (Z_ENT + extra_z_entry): continue
            if not fly_alignment_ok(Ti, +1.0, snap_last, zdisp_for_pair=zdisp): continue
            if not fly_alignment_ok(Tj, -1.0, snap_last, zdisp_for_pair=zdisp): continue
            candidates.append((zdisp, cheap, rich))

    BUCKETS = getattr(cr, "BUCKETS", {})
    bucket_dv01 = {b: 0.0 for b in BUCKETS.keys()}
    total_dv01, selected = 0.0, []

    for zdisp, cheap, rich in sorted(candidates, key=lambda x: x[0], reverse=True):
        if len(selected) >= max_pairs: break
        Ti, Tj = float(cheap["tenor_yrs"]), float(rich["tenor_yrs"])
        if Ti in used or Tj in used: continue
        
        pv_i, pv_j = pv01_proxy(Ti, float(cheap["rate"])), pv01_proxy(Tj, float(rich["rate"]))
        w_i, w_j = 1.0, -1.0 * pv_i / (pv_j if pv_j != 0 else 1e-12)
        b_i, b_j = assign_bucket(Ti), assign_bucket(Tj)
        dv_i, dv_j = abs(w_i)*pv_i, abs(w_j)*pv_j
        
        if b_i in bucket_dv01 and (bucket_dv01[b_i] + dv_i) > per_bucket_cap: continue
        if b_j in bucket_dv01 and (bucket_dv01[b_j] + dv_j) > per_bucket_cap: continue
        
        short_add = (dv_i if b_i=="short" else 0) + (dv_j if b_j=="short" else 0)
        if (bucket_dv01.get("short",0) + short_add) > front_end_cap: continue
        if (total_dv01 + dv_i + dv_j) > total_cap: continue
        if (b_i=="short" or b_j=="short") and (zdisp < (Z_ENT + SHORT_EXTRA)): continue

        used.add(Ti); used.add(Tj)
        bucket_dv01[b_i] += dv_i; bucket_dv01[b_j] += dv_j; total_dv01 += (dv_i + dv_j)
        selected.append((cheap, rich, w_i, w_j))

    return selected

# ------------------------
# Filenames & I/O
# ------------------------
def _enhanced_in_path(yymm: str) -> Path:
    suffix = getattr(cr, "ENH_SUFFIX", "")
    name = cr.enh_fname(yymm) if hasattr(cr, "enh_fname") else f"{yymm}_enh{suffix}.parquet" if suffix else f"{yymm}_enh.parquet"
    return Path(getattr(cr, "PATH_ENH", ".")) / name

def _positions_out_path() -> Path:
    if hasattr(cr, "positions_fname") and callable(cr.positions_fname): name = cr.positions_fname()
    else: suffix = getattr(cr, "OUT_SUFFIX", ""); name = f"positions_ledger{suffix}.parquet" if suffix else "positions_ledger.parquet"
    return Path(getattr(cr, "PATH_OUT", ".")) / name

def _marks_out_path() -> Path:
    if hasattr(cr, "marks_fname") and callable(cr.marks_fname): name = cr.marks_fname()
    else: suffix = getattr(cr, "OUT_SUFFIX", ""); name = f"marks_ledger{suffix}.parquet" if suffix else "marks_ledger.parquet"
    return Path(getattr(cr, "PATH_OUT", ".")) / name

def _pnl_out_path() -> Path:
    if hasattr(cr, "pnl_fname") and callable(cr.pnl_fname): name = cr.pnl_fname()
    else: suffix = getattr(cr, "OUT_SUFFIX", ""); name = f"pnl_by_bucket{suffix}.parquet" if suffix else "pnl_by_bucket.parquet"
    return Path(getattr(cr, "PATH_OUT", ".")) / name

def _pnl_curve_png(yymm: str) -> Path:
    if hasattr(cr, "pnl_curve_png") and callable(cr.pnl_curve_png): name = cr.pnl_curve_png(yymm)
    else: suffix = getattr(cr, "OUT_SUFFIX", ""); name = f"pnl_curve_{yymm}{suffix}.png" if suffix else f"pnl_curve_{yymm}.png"
    return Path(getattr(cr, "PATH_OUT", ".")) / name

# ------------------------
# RUN MONTH (Corrected Logic)
# ------------------------
def run_month(yymm, *, decision_freq=None, open_positions=None, carry_in=True, mode="strategy", hedges=None, overlay_use_caps=None, regime_mask=None, hybrid_signals=None, shock_cfg=None, shock_state=None):
    try: np
    except NameError: import numpy as np

    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    mode = mode.lower()
    enh_path = _enhanced_in_path(yymm)
    if not enh_path.exists(): raise FileNotFoundError(f"Missing {enh_path}")
    
    df = pd.read_parquet(enh_path)
    if df.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), (open_positions or [])
    
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df["decision_ts"] = df["ts"].dt.floor("d" if decision_freq=="D" else "h")
    decisions_per_day = 1 if decision_freq=="D" else 24
    
    BASE_Z_ENTRY = float(getattr(cr, "Z_ENTRY", 0.75))
    Z_EXIT = float(getattr(cr, "Z_EXIT", 0.40))
    Z_STOP = float(getattr(cr, "Z_STOP", 3.00))
    BPS_PNL_STOP = float(getattr(cr, "BPS_PNL_STOP", 0.0))
    SHOCK_MODE = str(getattr(cr, "SHOCK_MODE", "ROLL_OFF")).upper()
    
    if shock_state is None: shock_state = {"history": [], "dates": [], "remaining": 0}
    open_positions = (open_positions or []) if carry_in else []

    if mode == "overlay" and hedges is not None:
        valid_decisions = df["decision_ts"].dropna().unique()
        hedges = hedges[hedges["decision_ts"].isin(valid_decisions)].copy()
    else: hedges = None

    ledger_rows, closed_rows = [], []
    
    EXEC_LEG_THRESHOLD = float(getattr(cr, "EXEC_LEG_TENOR_YEARS", 0.084))
    ALT_LEG_THRESHOLD = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0))
    MIN_SEP = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    MAX_SPAN = float(getattr(cr, "MAX_SPAN_YEARS", 10.0))
    SHORT_EXTRA = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))

    # Prepare Shock Filter Signals (Safe Lookup)
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

    for dts, snap in df.groupby("decision_ts", sort=True):
        snap_last = snap.sort_values("ts").groupby("tenor_yrs", as_index=False).tail(1).reset_index(drop=True)
        if snap_last.empty: continue

        # A) START-OF-DAY GATING
        was_shock_active = (shock_state["remaining"] > 0)
        if shock_state["remaining"] > 0: shock_state["remaining"] -= 1
        
        # B) MARK POSITIONS & NATURAL EXITS
        per_pnl_cash, per_pnl_bps_mtm = 0.0, 0.0
        per_pnl_bps_real, per_pnl_cash_real = 0.0, 0.0
        
        still_open = []
        for pos in open_positions:
            prev_cash, prev_bp = pos.pnl_cash, pos.pnl_bp
            zsp = pos.mark(snap_last, decision_ts=dts)
            
            # MTM drift
            per_pnl_cash += (pos.pnl_cash - prev_cash)
            per_pnl_bps_mtm += (pos.pnl_bp - prev_bp)
            
            exit_flag = None
            if np.isfinite(zsp) and np.isfinite(pos.entry_zspread):
                entry_dir = pos.dir_sign * pos.entry_zspread
                curr_dir = pos.dir_sign * zsp
                if abs(curr_dir) <= Z_EXIT and abs(curr_dir) <= abs(entry_dir): exit_flag = "reversion"
                elif abs(curr_dir - entry_dir) >= Z_STOP and abs(curr_dir) >= abs(entry_dir): exit_flag = "stop"
            
            if exit_flag is None and BPS_PNL_STOP > 0 and pos.pnl_bp <= -BPS_PNL_STOP: exit_flag = "pnl_stop"
            
            limit = _overlay_max_hold_decisions(pos.scale_dv01) if pos.mode == "overlay" else base_max_hold_decisions
            if exit_flag is None and pos.age_decisions >= limit: exit_flag = "max_hold"
            
            if exit_flag: 
                pos.closed, pos.close_ts, pos.exit_reason = True, dts, exit_flag

            # LOGGING (With all components)
            ledger_rows.append({
                "decision_ts": dts, "event": "mark",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "pnl_bp": pos.pnl_bp, "pnl_cash": pos.pnl_cash,
                # COMPONENTS
                "pnl_price_cash": pos.pnl_price_cash, "pnl_price_bp": pos.pnl_price_bp,
                "pnl_carry_cash": pos.pnl_carry_cash, "pnl_carry_bp": pos.pnl_carry_bp,
                "pnl_roll_cash": pos.pnl_roll_cash, "pnl_roll_bp": pos.pnl_roll_bp,
                "z_spread": zsp, "closed": pos.closed, "mode": pos.mode,
                "rate_i": getattr(pos, "last_rate_i", np.nan), "rate_j": getattr(pos, "last_rate_j", np.nan),
                "w_i": pos.w_i, "w_j": pos.w_j,
            })

            if pos.closed:
                t_bp = 0.10 if pos.mode=="overlay" else 0.0 
                t_cash = t_bp * pos.scale_dv01
                pos.tcost_bp, pos.tcost_cash = t_bp, t_cash
                
                per_pnl_cash -= t_cash
                per_pnl_bps_mtm -= t_bp
                
                per_pnl_bps_real += (pos.pnl_bp - t_bp)
                per_pnl_cash_real += (pos.pnl_cash - t_cash)
                
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": dts, "exit_reason": pos.exit_reason,
                    "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                    "pnl_gross_bp": pos.pnl_bp, "pnl_gross_cash": pos.pnl_cash,
                    # COMPONENTS
                    "pnl_carry_cash": pos.pnl_carry_cash, "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_cash": pos.pnl_roll_cash, "pnl_roll_bp": pos.pnl_roll_bp,  
                    "pnl_price_cash": pos.pnl_price_cash, "pnl_price_bp": pos.pnl_price_bp,
                    "tcost_bp": t_bp, "tcost_cash": t_cash,
                    "pnl_net_bp": pos.pnl_bp - t_bp, "pnl_net_cash": pos.pnl_cash - t_cash,
                    "days_held_equiv": pos.age_decisions / max(1, decisions_per_day),
                    "mode": pos.mode,
                    "trade_id": pos.meta.get("trade_id"), "side": pos.meta.get("side"),
                })
            else:
                still_open.append(pos)
        
        open_positions = still_open
        
        # C) UPDATE SHOCK STATE
        metric_type = getattr(shock_cfg, "metric_type", "MTM_BPS") if shock_cfg else "MTM_BPS"
        if metric_type == "REALIZED_CASH": metric_val = per_pnl_cash_real
        elif metric_type == "REALIZED_BPS": metric_val = per_pnl_bps_real
        elif metric_type == "MTM_BPS" or metric_type == "BPS": metric_val = per_pnl_bps_mtm
        else: metric_val = per_pnl_cash

        shock_state["history"].append(metric_val)
        shock_state["dates"].append(dts)
        
        is_new_shock = False
        if shock_cfg:
            win = int(shock_cfg.pnl_window)
            if len(shock_state["history"]) >= win + 2:
                hist = np.array(shock_state["history"][-win:])
                if shock_cfg.use_raw_pnl:
                    mask_raw = np.isfinite(hist)
                    if mask_raw.sum() >= 2:
                        clean = hist[mask_raw]
                        mu, sd = np.mean(clean), np.std(clean, ddof=1)
                        if sd > 1e-9 and np.isfinite(hist[-1]):
                            if (hist[-1] - mu)/sd <= shock_cfg.raw_pnl_z_thresh: is_new_shock = True
                
                if shock_cfg.use_residuals and not is_new_shock and valid_reg_cols:
                    try:
                         rel_dates = shock_state["dates"][-win:]
                         sig_slice_raw = sig_lookup.reindex(rel_dates)[valid_reg_cols].values
                         if len(sig_slice_raw) == len(hist):
                             valid_mask = np.isfinite(hist) & np.isfinite(sig_slice_raw).all(axis=1)
                             if valid_mask.sum() >= sig_slice_raw.shape[1] + 1:
                                 Y_cl, X_cl = hist[valid_mask], sig_slice_raw[valid_mask]
                                 X_f = np.column_stack([np.ones(len(X_cl)), X_cl])
                                 try:
                                     beta, _, _, _ = np.linalg.lstsq(X_f, Y_cl, rcond=None)
                                     y_hat = X_f @ beta
                                     resid = Y_cl - y_hat
                                     r_mu, r_sd = np.mean(resid), np.std(resid, ddof=1)
                                     if r_sd > 1e-9:
                                         last_r_z = (resid[-1] - r_mu) / r_sd
                                         if last_r_z <= shock_cfg.resid_z_thresh: is_new_shock = True
                                 except np.linalg.LinAlgError: pass
                    except Exception: pass
        
        if is_new_shock: shock_state["remaining"] = int(shock_cfg.block_length)

        # D) PANIC EXIT
        is_shock_active = (shock_state["remaining"] > 0) or is_new_shock
        if is_shock_active and SHOCK_MODE == "EXIT_ALL" and len(open_positions) > 0:
            panic_bp_real, panic_cash_real = 0.0, 0.0
            panic_t_bp, panic_t_cash = 0.0, 0.0
            
            for pos in open_positions:
                pos.closed, pos.close_ts, pos.exit_reason = True, dts, "shock_exit"
                t_bp = 0.10 if pos.mode=="overlay" else 0.0
                t_cash = t_bp * pos.scale_dv01
                pos.tcost_bp, pos.tcost_cash = t_bp, t_cash
                
                panic_t_bp += t_bp; panic_t_cash += t_cash
                panic_bp_real += (pos.pnl_bp - t_bp)
                panic_cash_real += (pos.pnl_cash - t_cash)
                
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": dts, "exit_reason": "shock_exit",
                    "pnl_net_bp": pos.pnl_bp - t_bp, "pnl_net_cash": pos.pnl_cash - t_cash,
                    "pnl_carry_cash": pos.pnl_carry_cash, "pnl_roll_cash": pos.pnl_roll_cash,
                    "pnl_carry_bp": pos.pnl_carry_bp, "pnl_roll_bp": pos.pnl_roll_bp,
                    "pnl_price_cash": pos.pnl_price_cash, "pnl_price_bp": pos.pnl_price_bp,
                    "mode": pos.mode, "trade_id": pos.meta.get("trade_id"), "side": pos.meta.get("side")
                })
            open_positions = []
            
            if metric_type == "REALIZED_CASH": shock_state["history"][-1] += panic_cash_real
            elif metric_type == "REALIZED_BPS": shock_state["history"][-1] += panic_bp_real
            elif metric_type == "MTM_BPS" or metric_type == "BPS": shock_state["history"][-1] -= panic_t_bp
            else: shock_state["history"][-1] -= panic_t_cash

        # E) GATE & ENTRIES
        regime_ok = True
        if regime_mask is not None:
            regime_ok = bool(regime_mask.at[dts]) if dts in regime_mask.index else False
            
        gate = (not regime_ok) or was_shock_active
        if SHOCK_MODE == "EXIT_ALL" and is_shock_active: gate = True
        if gate: continue
        
        rem_slots = max(0, cr.MAX_CONCURRENT_PAIRS - len(open_positions))
        if rem_slots <= 0: continue

        if mode == "strategy":
            selected = choose_pairs_under_caps(snap_last, rem_slots, PER_BUCKET_DV01_CAP, TOTAL_DV01_CAP, FRONT_END_DV01_CAP, 0.0)
            for (cheap, rich, w_i, w_j) in selected:
                pos = PairPos(dts, cheap, rich, w_i, w_j, decisions_per_day, mode="strategy")
                open_positions.append(pos)
                ledger_rows.append({"decision_ts": dts, "event": "open", "mode": "strategy"})
                
        elif mode == "overlay":
            if hedges is None or hedges.empty: continue
            h_here = hedges[hedges["decision_ts"] == dts]
            if h_here.empty: continue
            if float(h_here["dv01"].abs().sum()) > OVERLAY_DV01_TS_CAP: continue
            
            snap_srt = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
            
            for _, h in h_here.iterrows():
                if len(open_positions) >= cr.MAX_CONCURRENT_PAIRS: break
                t_trade = float(h["tenor_yrs"])
                if t_trade < EXEC_LEG_THRESHOLD: continue
                if abs(float(h["dv01"])) > _per_trade_dv01_cap_for_bucket(assign_bucket(t_trade)): continue

                side_s = 1.0 if str(h["side"]).upper() == "CRCV" else -1.0
                z_ent_eff = _overlay_effective_z_entry(float(h["dv01"]))
                exec_z = _get_z_at_tenor(snap_srt, t_trade)
                if exec_z is None: continue
                exec_row = snap_srt.iloc[(snap_srt["tenor_yrs"] - t_trade).abs().idxmin()]
                
                best_c, best_z = None, 0.0
                for _, alt in snap_srt.iterrows():
                    t_alt = float(alt["tenor_yrs"])
                    if t_alt < ALT_LEG_THRESHOLD or t_alt == float(exec_row["tenor_yrs"]): continue
                    if not (MIN_SEP <= abs(t_alt - float(exec_row["tenor_yrs"])) <= MAX_SPAN): continue
                    
                    z_alt = _to_float(alt["z_comb"])
                    disp = (z_alt - exec_z) if side_s > 0 else (exec_z - z_alt)
                    if disp < z_ent_eff: continue
                    
                    if (assign_bucket(alt_tenor)=="short" or assign_bucket(exec_tenor)=="short") and (disp < z_ent_eff + SHORT_EXTRA):
                        continue
                    
                    c_t, r_t = (t_alt, float(exec_row["tenor_yrs"])) if z_alt > exec_z else (float(exec_row["tenor_yrs"]), t_alt)
                    if not (fly_alignment_ok(c_t, 1, snap_srt, zdisp_for_pair=disp) and fly_alignment_ok(r_t, -1, snap_srt, zdisp_for_pair=disp)): continue
                    
                    if disp > best_z: best_z, best_c = disp, alt
                
                if best_c is not None:
                    rate_i, rate_j = None, None
                    ti, tj = tenor_to_ticker(float(best_c["tenor_yrs"])), tenor_to_ticker(float(exec_row["tenor_yrs"]))
                    if ti and f"{ti}_mid" in h: rate_i = _to_float(h[f"{ti}_mid"])
                    if tj and f"{tj}_mid" in h: rate_j = _to_float(h[f"{tj}_mid"])
                    
                    if rate_i is None: rate_i = _to_float(best_c["rate"])
                    if rate_j is None: rate_j = _to_float(exec_row["rate"])

                    pos = PairPos(dts, best_c, exec_row, side_s*1.0, side_s*-1.0, decisions_per_day, 
                                  scale_dv01=float(h["dv01"]), mode="overlay", 
                                  meta={"trade_id": h.get("trade_id"), "side": h.get("side")},
                                  entry_rate_i=rate_i, entry_rate_j=rate_j)
                    open_positions.append(pos)
                    ledger_rows.append({"decision_ts": dts, "event": "open", "mode": "overlay"})

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions

def run_all(yymms, *, decision_freq=None, carry=True, force_close_end=False, mode="strategy", hedge_df=None, overlay_use_caps=None, regime_mask=None, hybrid_signals=None, shock_cfg=None):
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    mode = mode.lower()
    
    clean_hedges = None
    if mode == "overlay":
        if hedge_df is None: raise ValueError("Overlay requires hedge_df")
        clean_hedges = prepare_hedge_tape(hedge_df, decision_freq)
        
    all_pos, all_led, all_by = [], [], []
    open_pos = []
    shock_state = {"history": [], "dates": [], "remaining": 0}
    
    for yymm in yymms:
        h_mon = None
        if mode == "overlay" and clean_hedges is not None:
            y, m = 2000+int(yymm[:2]), int(yymm[2:])
            start, end = pd.Timestamp(y, m, 1), (pd.Timestamp(y, m, 1) + MonthEnd(1) + pd.Timedelta(days=1))
            h_mon = clean_hedges[(clean_hedges["decision_ts"] >= start) & (clean_hedges["decision_ts"] < end)].copy()
            
        p, l, b, open_pos = run_month(yymm, decision_freq=decision_freq, open_positions=open_pos, carry_in=carry, mode=mode, hedges=h_mon, regime_mask=regime_mask, hybrid_signals=hybrid_signals, shock_cfg=shock_cfg, shock_state=shock_state)
        if not p.empty: all_pos.append(p.assign(yymm=yymm))
        if not l.empty: all_led.append(l.assign(yymm=yymm))
        if not b.empty: all_by.append(b.assign(yymm=yymm))
        
    if force_close_end and open_pos:
        # EOC Close
        pass

    return (pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame(),
            pd.concat(all_led, ignore_index=True) if all_led else pd.DataFrame(),
            pd.concat(all_by, ignore_index=True) if all_by else pd.DataFrame())

if __name__ == "__main__":
    # ... CLI ...
    pass
