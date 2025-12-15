import os, sys
import math
from pathlib import Path
from typing import Optional, List, Dict
from collections import deque

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from scipy.interpolate import CubicSpline

# All config access via module namespace
import cr_config as cr

# ------------------------
# Utilities / conventions
# ------------------------
Path(getattr(cr, "PATH_OUT", ".")).mkdir(parents=True, exist_ok=True)

def _to_float(x, default=np.nan):
    try:
        if isinstance(x, (pd.Series, pd.Index)):
            if len(x) == 0: return default
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return default

def assign_bucket(tenor):
    buckets = getattr(cr, "BUCKETS", {})
    for name, (lo, hi) in buckets.items():
        if (tenor >= lo) and (tenor <= hi):
            return name
    return "other"

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

def _get_z_at_tenor(snap_last: pd.DataFrame, tenor: float) -> float | None:
    t = float(tenor)
    s = snap_last[["tenor_yrs", "z_comb"]].dropna()
    if s.empty: return None
    s = s.assign(_dist=(s["tenor_yrs"] - t).abs())
    row = s.loc[s["_dist"].idxmin()]
    if row["_dist"] > 0.05: return None
    return float(row["z_comb"])

def _map_instrument_to_tenor(instr: str) -> Optional[float]:
    if instr is None or not isinstance(instr, str): return None
    instr = instr.strip()
    mapped = cr.BBG_DICT.get(instr, instr)
    tenor = cr.TENOR_YEARS.get(mapped)
    return float(tenor) if tenor is not None else None

def prepare_hedge_tape(raw_df: pd.DataFrame, decision_freq: str) -> pd.DataFrame:
    """Robust Hedge Tape Loader with Debugging."""
    if raw_df is None or raw_df.empty: 
        print("[WARN] Hedge tape is empty or None.")
        return pd.DataFrame()
    
    df = raw_df.copy()
    print(f"[DEBUG] Loading Hedge Tape. Raw Cols: {list(df.columns)}")
    
    # 1. Standardize Time
    if "tradetimeUTC" in df.columns:
        df["trade_ts"] = pd.to_datetime(df["tradetimeUTC"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    elif "ts" in df.columns:
        df["trade_ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    else:
        # Fallback
        print("[WARN] No timestamp column found in hedge tape. Using current time.")
        df["trade_ts"] = pd.Timestamp.now()

    decision_freq = str(decision_freq).upper()
    if decision_freq == "D": df["decision_ts"] = df["trade_ts"].dt.floor("d")
    elif decision_freq == "H": df["decision_ts"] = df["trade_ts"].dt.floor("h")
    else: raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

    # 2. Standardize Side
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.upper()
        # Ensure we only keep valid sides
        df = df[df["side"].isin(["CPAY", "CRCV"])]
    else:
        print("[ERROR] 'side' column missing in hedge tape.")
        return pd.DataFrame()

    # 3. Map Tenor
    if "instrument" in df.columns:
        df["tenor_yrs"] = df["instrument"].map(_map_instrument_to_tenor)
    else:
        print("[ERROR] 'instrument' column missing.")
        return pd.DataFrame()
    
    df = df[np.isfinite(df["tenor_yrs"])]
    
    # 4. Extract DV01 (The Critical Fix)
    if "EqVolDelta" in df.columns:
        df["dv01"] = pd.to_numeric(df["EqVolDelta"], errors="coerce").abs() # Ensure positive risk size
    elif "dv01" in df.columns:
        df["dv01"] = pd.to_numeric(df["dv01"], errors="coerce").abs()
    else:
        print("[ERROR] No 'EqVolDelta' or 'dv01' column found.")
        return pd.DataFrame()
        
    df = df[np.isfinite(df["dv01"]) & (df["dv01"] > 1.0)] # Filter tiny/zero risks
    df = df.dropna(subset=["trade_ts", "decision_ts", "tenor_yrs", "dv01"])
    
    if "trade_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["trade_id"] = df.index.astype(int)
        
    cols = ["trade_id", "trade_ts", "decision_ts", "tenor_yrs", "side", "dv01"]
    print(f"[DEBUG] Hedge Tape Loaded. {len(df)} valid rows.")
    if not df.empty:
        print(f"[DEBUG] Sample Row:\n{df[cols].iloc[0]}")
    
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra]

def calc_trade_drift(tenor, direction, rate_fixed, rate_float, xp_sorted, fp_sorted):
    """Fast linear estimation for legacy pairs."""
    day_fraction = 1.0 / 360.0
    t_roll = max(0.0, tenor - day_fraction)
    rate_rolled = np.interp(t_roll, xp_sorted, fp_sorted)
    raw_carry_bps = (rate_fixed - rate_float) * 100.0 * day_fraction
    raw_roll_bps = (rate_fixed - rate_rolled) * 100.0
    return (raw_carry_bps + raw_roll_bps) * direction

# ------------------------
# POSITION OBJECTS
# ------------------------

class PairPos:
    """Legacy Pair Object (Slope/Overlay)"""
    def __init__(self, open_ts, cheap_row, rich_row, w_i, w_j, decisions_per_day, *, 
                 scale_dv01=1.0, meta=None, dir_sign=None, 
                 entry_rate_i=None, entry_rate_j=None,
                 fly_bonus=0.0, regime_mult=1.0, 
                 z_entry_base=0.0, z_entry_final=0.0):
        
        self.open_ts = open_ts
        self.tenor_i = _to_float(cheap_row["tenor_yrs"])
        self.tenor_j = _to_float(rich_row["tenor_yrs"])
        self.mode = "overlay" 
        
        # Safety Cap
        safety_factor = float(getattr(cr, "MIN_TENOR_SAFETY_FACTOR", 73.0))
        self.max_days_i = self.tenor_i * safety_factor
        self.max_days_j = self.tenor_j * safety_factor
        self.max_days_pair = min(self.max_days_i, self.max_days_j)
        
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
        self.scale_dv01 = float(scale_dv01)
        self.meta = meta or {}
        
        self.fly_bonus = fly_bonus
        self.regime_mult = regime_mult
        self.z_entry_base = z_entry_base
        self.z_entry_final = z_entry_final
        
        self.initial_dv01 = self.scale_dv01
        self.dv01_i_entry = self.scale_dv01 * self.w_i
        self.dv01_j_entry = self.scale_dv01 * self.w_j
        self.dv01_i_curr, self.dv01_j_curr = self.dv01_i_entry, self.dv01_j_entry
        self.rem_tenor_i, self.rem_tenor_j = self.tenor_i_orig, self.tenor_j_orig
        
        self.not_i_entry = self.dv01_i_entry / max(1e-6, self.tenor_i)
        self.not_j_entry = self.dv01_j_entry / max(1e-6, self.tenor_j)
        
        self.last_mark_ts = open_ts 
        self.pnl_price_cash = 0.0
        self.pnl_carry_cash = 0.0
        self.pnl_roll_cash = 0.0
        self.pnl_cash = 0.0
        self.pnl_bp = 0.0
        self.pnl_price_bp = 0.0
        self.pnl_carry_bp = 0.0
        self.pnl_roll_bp = 0.0

        self.tcost_bp, self.tcost_cash = 0.0, 0.0
        self.decisions_per_day = decisions_per_day
        self.age_decisions = 0
        self.bucket_i, self.bucket_j = assign_bucket(self.tenor_i), assign_bucket(self.tenor_j)
        self.last_zspread, self.last_z_dir = self.entry_zspread, self.entry_z_dir

    def _update_risk_decay(self, decision_ts):
        if not isinstance(decision_ts, pd.Timestamp): return
        days = max(0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        yr_pass = days / 360.0
        self.rem_tenor_i = max(self.tenor_i_orig - yr_pass, 0.0)
        self.rem_tenor_j = max(self.tenor_j_orig - yr_pass, 0.0)
        
        fi = self.rem_tenor_i / max(self.tenor_i_orig, 1e-6)
        fj = self.rem_tenor_j / max(self.tenor_j_orig, 1e-6)
        self.dv01_i_curr = self.dv01_i_entry * fi
        self.dv01_j_curr = self.dv01_j_entry * fj

    def mark(self, snap_last: pd.DataFrame, xp_sorted: np.ndarray, fp_sorted: np.ndarray, r_float: float, decision_ts: Optional[pd.Timestamp] = None):
        if decision_ts: self._update_risk_decay(decision_ts)
        ri = np.interp(self.tenor_i, xp_sorted, fp_sorted)
        rj = np.interp(self.tenor_j, xp_sorted, fp_sorted)
        self.last_rate_i, self.last_rate_j = ri, rj

        pnl_price_i = (self.entry_rate_i - ri) * 100.0 * self.dv01_i_curr
        pnl_price_j = (self.entry_rate_j - rj) * 100.0 * self.dv01_j_curr
        self.pnl_price_cash = pnl_price_i + pnl_price_j

        if decision_ts and self.last_mark_ts:
            dt_days = max(0.0, (decision_ts.normalize() - self.last_mark_ts.normalize()).days)
            if dt_days > 0:
                inc_carry_i = (self.entry_rate_i - r_float) * 100.0 * self.not_i_entry * (dt_days / 360.0)
                inc_carry_j = (self.entry_rate_j - r_float) * 100.0 * self.not_j_entry * (dt_days / 360.0)
                self.pnl_carry_cash += (inc_carry_i + inc_carry_j)
            self.last_mark_ts = decision_ts

        days_total = 0.0
        if decision_ts:
            days_total = max(0.0, (decision_ts.normalize() - self.open_ts.normalize()).days)

        if len(xp_sorted) > 1:
            t_roll_i = max(0.0, self.tenor_i_orig - (days_total/360.0))
            y_roll_i = np.interp(t_roll_i, xp_sorted, fp_sorted)
            gain_i = (ri - y_roll_i) * 100.0 * self.dv01_i_curr
            t_roll_j = max(0.0, self.tenor_j_orig - (days_total/360.0))
            y_roll_j = np.interp(t_roll_j, xp_sorted, fp_sorted)
            gain_j = (rj - y_roll_j) * 100.0 * self.dv01_j_curr
            self.pnl_roll_cash = gain_i + gain_j
        else:
            self.pnl_roll_cash = 0.0

        self.pnl_cash = self.pnl_price_cash + self.pnl_carry_cash + self.pnl_roll_cash
        if self.scale_dv01 != 0.0:
            self.pnl_bp = self.pnl_cash / self.scale_dv01
            self.pnl_price_bp = self.pnl_price_cash / self.scale_dv01
            self.pnl_carry_bp = self.pnl_carry_cash / self.scale_dv01
            self.pnl_roll_bp = self.pnl_roll_cash / self.scale_dv01

        zi = _get_z_at_tenor(snap_last, self.tenor_i)
        zj = _get_z_at_tenor(snap_last, self.tenor_j)
        if zi is not None and zj is not None:
            zsp = zi - zj
            if np.isfinite(self.last_zspread): self.conv_pnl_proxy += (self.last_zspread - zsp) * 10.0
            self.last_zspread = zsp
            if np.isfinite(self.dir_sign): self.last_z_dir = self.dir_sign * zsp
            else: self.last_z_dir = np.nan
        else:
            zsp = self.last_zspread
        self.age_decisions += 1
        return zsp

class FlyPos:
    """Butterfly Object (Curvature) with Cubic Spline Drift & Decay"""
    def __init__(self, open_ts, belly_row, left_row, right_row, decisions_per_day, *, 
                 scale_dv01=10_000.0, meta=None, 
                 z_score_current=0.0, z_score_trend=0.0,
                 z_entry_base=0.0, z_entry_final=0.0, regime_mult=1.0,
                 direction_multiplier=1.0): # <--- NEW: Explicit Flip Control
        
        self.open_ts = open_ts
        self.meta = meta or {}
        self.mode = "fly"
        self.scale_dv01 = float(scale_dv01)
        
        # --- Tenors ---
        self.t_belly = float(belly_row["tenor_yrs"])
        self.t_left  = float(left_row["tenor_yrs"])
        self.t_right = float(right_row["tenor_yrs"])
        self.t_belly_orig, self.t_left_orig, self.t_right_orig = self.t_belly, self.t_left, self.t_right
        
        # --- Safety Cap: Shortest Tenor ---
        safety_factor = float(getattr(cr, "MIN_TENOR_SAFETY_FACTOR", 73.0))
        self.max_days_fly = min(self.t_belly, self.t_left, self.t_right) * safety_factor
        
        # --- Weights (Logic: Belly is Anchor, Wings are Legs) ---
        # Default (Neutral): Belly=1.0 (Rec), Wings=-0.5 (Pay)
        # Multiplier: -1.0 means Pay Belly, Rec Wings
        dm = float(direction_multiplier)
        
        self.w_belly = 1.0 * dm
        self.w_left  = -0.5 * dm
        self.w_right = -0.5 * dm
        
        # --- Risk Init (DV01) ---
        # We multiply by scale_dv01 (which is ABSOLUTE risk size)
        self.dv01_belly_entry = self.scale_dv01 * self.w_belly
        self.dv01_left_entry  = self.scale_dv01 * self.w_left
        self.dv01_right_entry = self.scale_dv01 * self.w_right
        
        self.dv01_belly_curr = self.dv01_belly_entry
        self.dv01_left_curr  = self.dv01_left_entry
        self.dv01_right_curr = self.dv01_right_entry

        # --- Entry Rates ---
        self.r_entry_belly = float(belly_row["rate"])
        self.r_entry_left  = float(left_row["rate"])
        self.r_entry_right = float(right_row["rate"])
        
        # --- Stats ---
        self.entry_z = z_score_current
        self.trend_z = z_score_trend
        
        # Direction for Exit Logic:
        # If we are Short Belly (dm = -1), we want Rates to Rise? 
        # No, Fly PnL is (Price Diff).
        # We assume Z tracks "Richness" (High Z = Cheap).
        # If we are Short Belly, we want Z to go Up? 
        # Let's keep it simple: We track PnL.
        self.dir_sign = -1.0 * np.sign(self.entry_z) 
        self.side_desc = meta.get("side", "Unknown")

        self.regime_mult = regime_mult
        self.z_entry_final = z_entry_final
        self.z_entry_base = z_entry_base
        
        self.pnl_cash = 0.0
        self.pnl_bp = 0.0
        self.closed = False
        self.exit_reason = None
        self.age_decisions = 0
        self.decisions_per_day = decisions_per_day
        self.last_mark_ts = open_ts
        
        self.last_z_val = self.entry_z
        self.last_z_dir = self.entry_z * self.dir_sign

        self.pnl_price_cash = 0.0
        self.pnl_carry_cash = 0.0
        self.pnl_roll_cash = 0.0
        self.pnl_price_bp = 0.0
        self.pnl_carry_bp = 0.0
        self.pnl_roll_bp = 0.0

    def _update_risk_decay(self, decision_ts):
        if not isinstance(decision_ts, pd.Timestamp): return
        days = max(0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        yr_pass = days / 360.0
        
        self.t_belly = max(self.t_belly_orig - yr_pass, 0.0)
        self.t_left  = max(self.t_left_orig  - yr_pass, 0.0)
        self.t_right = max(self.t_right_orig - yr_pass, 0.0)
        
        fb = self.t_belly / max(self.t_belly_orig, 1e-6)
        fl = self.t_left  / max(self.t_left_orig, 1e-6)
        fr = self.t_right / max(self.t_right_orig, 1e-6)
        
        self.dv01_belly_curr = self.dv01_belly_entry * fb
        self.dv01_left_curr  = self.dv01_left_entry  * fl
        self.dv01_right_curr = self.dv01_right_entry * fr

    def mark(self, snap_last, xp, fp, r_float, decision_ts=None):
        if decision_ts: self._update_risk_decay(decision_ts)
        
        cs = CubicSpline(xp, fp)
        
        curr_belly = float(cs(self.t_belly))
        curr_left  = float(cs(self.t_left))
        curr_right = float(cs(self.t_right))
        
        # PnL = (Entry - Current) * DV01
        # DV01 already contains the correct Sign/Direction from __init__
        pnl_belly = (self.r_entry_belly - curr_belly) * 100.0 * self.dv01_belly_curr
        pnl_left  = (self.r_entry_left - curr_left)   * 100.0 * self.dv01_left_curr
        pnl_right = (self.r_entry_right - curr_right) * 100.0 * self.dv01_right_curr
        
        self.pnl_price_cash = pnl_belly + pnl_left + pnl_right
        
        if decision_ts and self.last_mark_ts:
            dt_days = (decision_ts - self.last_mark_ts).days
            if dt_days > 0:
                yld_b = self.r_entry_belly * self.dv01_belly_curr
                yld_l = self.r_entry_left  * self.dv01_left_curr
                yld_r = self.r_entry_right * self.dv01_right_curr
                self.pnl_carry_cash += (yld_b + yld_l + yld_r) * 100.0 * (dt_days/360.0)
                
                t_roll = 1.0/360.0
                roll_belly = float(cs(self.t_belly - t_roll))
                roll_left  = float(cs(self.t_left  - t_roll))
                roll_right = float(cs(self.t_right - t_roll))
                
                gain_belly = (curr_belly - roll_belly) * 100.0 * self.dv01_belly_curr
                gain_left  = (curr_left - roll_left)   * 100.0 * self.dv01_left_curr
                gain_right = (curr_right - roll_right) * 100.0 * self.dv01_right_curr
                
                self.pnl_roll_cash += (gain_belly + gain_left + gain_right) * dt_days 
                
            self.last_mark_ts = decision_ts

        self.pnl_cash = self.pnl_price_cash + self.pnl_carry_cash + self.pnl_roll_cash
        self.pnl_bp = self.pnl_cash / self.scale_dv01 if self.scale_dv01 else 0.0
        self.age_decisions += 1
        
        if self.scale_dv01:
            self.pnl_price_bp = self.pnl_price_cash / self.scale_dv01
            self.pnl_carry_bp = self.pnl_carry_cash / self.scale_dv01
            self.pnl_roll_bp = self.pnl_roll_cash / self.scale_dv01

        curr_z = _get_z_at_tenor(snap_last, self.t_belly)
        if curr_z is not None:
            self.last_z_val = curr_z
            self.last_z_dir = curr_z * self.dir_sign
            return curr_z
        return self.last_z_val

# ------------------------
# RUN MONTH
# ------------------------
def run_month(
    yymm: str,
    *,
    decision_freq: str | None = None,
    open_positions: Optional[List] | None = None,
    carry_in: bool = True,
    mode: str = "fly", 
    hedges: Optional[pd.DataFrame] = None,
    regime_signals: Optional[pd.DataFrame] = None,
    z_history: Optional[Dict] = None 
):
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()

    enh_path = _enhanced_in_path(yymm)
    if not enh_path.exists():
        raise FileNotFoundError(f"Missing enhanced file {enh_path}")

    df = pd.read_parquet(enh_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), (open_positions or []), z_history

    need = {"ts", "tenor_yrs", "rate", "z_comb"}
    if not need.issubset(df.columns):
        raise ValueError(f"{enh_path} missing columns: {need - set(df.columns)}")

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

    # --- PARAMS ---
    Z_ENTRY_BASE = float(getattr(cr, "Z_ENTRY", 0.75))
    Z_EXIT_BASE  = float(getattr(cr, "Z_EXIT", 0.40))
    Z_STOP_BASE  = float(getattr(cr, "Z_STOP", 3.00))
    SWITCH_COST_BP = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10))
    
    FLY_ANCHOR_MODE = str(getattr(cr, "FLY_ANCHOR_MODE", "STRICT")).upper() 
    FLY_WING_MIN, FLY_WING_MAX = getattr(cr, "FLY_WING_WIDTH_RANGE", (2.0, 7.0))
    TREND_WINDOW = int(getattr(cr, "Z_TREND_WINDOW", 20))
    
    # --- LIQUIDITY GRID (Fix for Phantom Tenors) ---
    VALID_TENORS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0]
    def is_tradeable(t_check):
        for v in VALID_TENORS:
            if abs(t_check - v) < 0.1: return True
        return False

    sig_lookup = None
    if regime_signals is not None and not regime_signals.empty:
        sig_lookup = regime_signals.drop_duplicates("decision_ts").set_index("decision_ts").sort_index()
    
    DYN_THRESH_ENABLE = getattr(cr, "DYN_THRESH_ENABLE", False)

    open_positions = (open_positions or []) if carry_in else []

    # FILTER HEDGES (Crucial for Overlay)
    if hedges is not None and not hedges.empty:
        valid_decisions = df["decision_ts"].dropna().unique()
        hedges = hedges[hedges["decision_ts"].isin(valid_decisions)].copy()
    else:
        hedges = pd.DataFrame()

    ledger_rows: list[dict] = []
    closed_rows: list[dict] = []

    if z_history is None: z_history = {} 

    for dts, snap in df.groupby("decision_ts", sort=True):
        snap_last = (
            snap.sort_values("ts")
                .groupby("tenor_yrs", as_index=False)
                .tail(1)
                .reset_index(drop=True)
        )
        if snap_last.empty: continue

        valid_curve = snap_last[["tenor_yrs", "rate"]].dropna()
        xp_sorted = valid_curve["tenor_yrs"].values.astype(float)
        fp_sorted = valid_curve["rate"].values.astype(float)
        sort_idx = np.argsort(xp_sorted)
        xp_sorted = xp_sorted[sort_idx]
        fp_sorted = fp_sorted[sort_idx]
        r_float = fp_sorted[0] if len(fp_sorted) > 0 else 0.0

        current_z_map = dict(zip(snap_last["tenor_yrs"], snap_last["z_comb"]))
        for t, z_val in current_z_map.items():
            if t not in z_history: z_history[t] = deque(maxlen=TREND_WINDOW)
            z_history[t].append(z_val)

        z_entry_curr = Z_ENTRY_BASE
        z_exit_curr  = Z_EXIT_BASE
        z_stop_curr  = Z_STOP_BASE
        regime_mult  = 1.0
        
        if DYN_THRESH_ENABLE and sig_lookup is not None:
            if dts in sig_lookup.index:
                row = sig_lookup.loc[dts]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                
                raw_vol = float(row["scale_mean"]) if "scale_mean" in row else 0.01
                trend_z = raw_vol / 0.01
                health_z = float(row["z_xs_std"]) if "z_xs_std" in row else 1.0
                
                sens_trend = float(getattr(cr, "SENS_TRENDINESS", 0.0))
                sens_health = float(getattr(cr, "SENS_HEALTH", 0.0))
                
                raw_mult = 1.0 + (trend_z * sens_trend) + (health_z * sens_health)
                regime_mult = max(0.5, min(raw_mult, 3.0)) 
                
                z_entry_curr *= regime_mult
                z_stop_curr  *= regime_mult
                z_exit_curr  *= regime_mult

        still_open: list = []
        for pos in open_positions:
            zsp = pos.mark(snap_last, xp_sorted, fp_sorted, r_float, decision_ts=dts)
            exit_flag = None
            
            if hasattr(pos, 'last_z_dir') and np.isfinite(pos.last_z_dir):
                if abs(pos.last_z_val) <= z_exit_curr: exit_flag = "reversion"
                if pos.mode == "fly":
                    raw_move = pos.last_z_val - pos.entry_z
                    bad_move = raw_move * -pos.dir_sign 
                    if bad_move >= z_stop_curr: exit_flag = "stop"
                else:
                    entry_dir = pos.entry_z_dir
                    curr_dir_val = pos.last_z_dir
                    dz_dir = curr_dir_val - entry_dir
                    if (np.sign(entry_dir) == np.sign(curr_dir_val)) and (abs(curr_dir_val) >= abs(entry_dir)) and (abs(dz_dir) >= z_stop_curr):
                        exit_flag = "stop"

            STALE_ENABLE = getattr(cr, "STALE_ENABLE", False)
            if STALE_ENABLE and exit_flag is None:
                STALE_START = float(getattr(cr, "STALE_START_DAYS", 3.0))
                days_held = pos.age_decisions / max(1, decisions_per_day)
                if days_held >= STALE_START:
                    if pos.pnl_bp < 0 and days_held > 10: exit_flag = "stalemate_loss"
                    elif pos.pnl_bp > 0 and (pos.pnl_bp / days_held) < 0.1: exit_flag = "stalemate_slow"

            if exit_flag is None:
                limit = getattr(pos, 'max_days_fly', getattr(pos, 'max_days_pair', 60))
                if pos.age_decisions >= (limit * decisions_per_day): exit_flag = "safety_cap"

            if exit_flag is not None:
                pos.closed = True
                pos.close_ts = dts
                pos.exit_reason = exit_flag

            row = {
                "decision_ts": dts, "event": "mark",
                "tenor_i": getattr(pos, "t_belly", getattr(pos, "tenor_i", np.nan)),
                "pnl_bp": pos.pnl_bp, "pnl_cash": pos.pnl_cash,
                "pnl_price_bp": pos.pnl_price_bp, "pnl_carry_bp": pos.pnl_carry_bp,
                "pnl_roll_bp": pos.pnl_roll_bp, "mode": pos.mode, "closed": pos.closed,
                "scale_dv01": pos.scale_dv01
            }
            if pos.mode == "fly":
                row.update({
                    "tenor_left": pos.t_left, "tenor_right": pos.t_right,
                    "dv01_belly": pos.dv01_belly_curr, "dv01_left": pos.dv01_left_curr, "dv01_right": pos.dv01_right_curr,
                    "side": pos.side_desc
                })
            ledger_rows.append(row)

            if pos.closed:
                tcost_bp = SWITCH_COST_BP
                tcost_cash = tcost_bp * pos.scale_dv01
                cl_row = {
                    "open_ts": pos.open_ts, "close_ts": pos.close_ts, 
                    "exit_reason": pos.exit_reason, "mode": pos.mode,
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "pnl_price_bp": pos.pnl_price_bp, "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp": pos.pnl_roll_bp, "tcost_bp": tcost_bp,
                    "days_held": pos.age_decisions / decisions_per_day,
                    "regime_mult": pos.regime_mult,
                    "trade_id": pos.meta.get("trade_id", -1),
                    "scale_dv01": pos.scale_dv01
                }
                if pos.mode == "fly":
                    cl_row.update({
                        "tenor_i": pos.t_belly, "tenor_left": pos.t_left, "tenor_right": pos.t_right,
                        "dv01_belly_entry": pos.dv01_belly_entry, "dv01_left_entry": pos.dv01_left_entry, "dv01_right_entry": pos.dv01_right_entry,
                        "side": pos.side_desc
                    })
                else:
                    cl_row.update({"tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j})
                closed_rows.append(cl_row)
            else:
                still_open.append(pos)
        open_positions = still_open
        
        # ============================================================
        # 5) NEW ENTRIES (SMART FLY OVERLAY)
        # ============================================================
        if mode == "fly":
            if hedges.empty: continue
            h_here = hedges[hedges["decision_ts"] == dts]
            if h_here.empty: continue

            curve = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
            
            for _, h in h_here.iterrows():
                if len(open_positions) >= 50: break

                # A. Read Hedge Details & VALIDATE DV01
                t_hedge = float(h["tenor_yrs"])
                hedge_side = str(h["side"]).upper()
                hedge_dv01 = float(h["dv01"])
                
                # SANITY CHECK: Print debug if skipping
                if hedge_dv01 < 100: 
                    # print(f"[SKIP] Small DV01: {hedge_dv01}")
                    continue 

                # B. DIRECTION LOGIC: Flip Anchor, Keep Legs
                # CRCV (Long) -> Anchor Pay (-1), Legs Rec (+1)
                # CPAY (Short) -> Anchor Rec (+1), Legs Pay (-1)
                
                # Multiplier Logic:
                # If CRCV: dm = -1.0
                #   -> Belly W = 1.0 * -1.0 = -1.0 (Pay)
                #   -> Wing W  = -0.5 * -1.0 = +0.5 (Rec)
                # Matches requirements perfectly.
                
                needed_dm = -1.0 if hedge_side == "CRCV" else 1.0
                scan_range = 0.5 if FLY_ANCHOR_MODE == "STRICT" else 2.5
                
                best_fly = None
                best_score = -999.0
                
                for i in range(1, len(curve)-1):
                    belly = curve.iloc[i]
                    t_belly = float(belly["tenor_yrs"])
                    
                    if abs(t_belly - t_hedge) > scan_range: continue
                    if not is_tradeable(t_belly): continue 
                    
                    z_curr = float(belly["z_comb"])
                    
                    # ALPHA SCORE: Does direction help?
                    # needed_dm tells us if we are Shorting Belly (-1) or Buying (+1)
                    # If Shorting (-1), we want High Z (Cheap). Wait, Shorting Cheap = Profit? No.
                    # Shorting Cheap = Selling Top. YES.
                    # So if dm=-1, we want Z > 0.
                    # Score = -1 * 2.0 = -2.0? No.
                    # We want alignment.
                    # Alignment = -1 (Short) * 2.0 (Cheap/High) = -2.0. This implies bad.
                    # Let's check: Selling High is Good.
                    # So if dm=-1 and Z>0, Score should be positive.
                    # Score = -1 * Z * -1? 
                    # Simpler: Score = (needed_dm * -1) * Z ?
                    # Let's trust Z-Trend logic or just filter extremes.
                    
                    # For Overlay, we just do it unless it's terrible.
                    # Terrible = Buying Top (Long + Cheap) or Selling Bottom (Short + Rich).
                    # Buying Top: dm=1, Z>2. Selling Bottom: dm=-1, Z<-2.
                    if needed_dm == 1.0 and z_curr > 2.0: continue
                    if needed_dm == -1.0 and z_curr < -2.0: continue
                    
                    valid_wings = False
                    cand_left, cand_right, cand_comp_z = None, None, 0.0
                    
                    for j in range(i-1, -1, -1):
                        left = curve.iloc[j]
                        t_left = float(left["tenor_yrs"])
                        width_l = t_belly - t_left
                        if width_l > FLY_WING_MAX: break
                        if width_l < FLY_WING_MIN: continue
                        if not is_tradeable(t_left): continue 
                        
                        for k in range(i+1, len(curve)):
                            right = curve.iloc[k]
                            t_right = float(right["tenor_yrs"])
                            width_r = t_right - t_belly
                            if width_r > FLY_WING_MAX: break
                            if width_r < FLY_WING_MIN: continue
                            if not is_tradeable(t_right): continue 
                            
                            # Maximize Spread?
                            # Just pick widest valid? Or random?
                            # Let's pick widest valid for stability.
                            if not valid_wings or (width_l + width_r) > cand_comp_z:
                                cand_left, cand_right = left, right
                                cand_comp_z = width_l + width_r # Using width as score for now
                                valid_wings = True
                    
                    if not valid_wings: continue
                    
                    # Selection
                    if best_fly is None:
                        best_fly = (belly, cand_left, cand_right, z_curr)

                if best_fly:
                    f_belly, f_left, f_right, f_z = best_fly
                    
                    # EXECUTE using HEDGE_DV01 and DIRECTION_MULTIPLIER
                    pos = FlyPos(dts, f_belly, f_left, f_right, decisions_per_day,
                                 scale_dv01=hedge_dv01, # <--- EXPLICIT
                                 z_score_current=f_z, z_score_trend=0.0,
                                 z_entry_final=z_entry_curr, regime_mult=regime_mult,
                                 direction_multiplier=needed_dm, # <--- EXPLICIT FLIP
                                 meta={"trade_id": h.get("trade_id"), "side": hedge_side})
                    
                    open_positions.append(pos)
                    
                    ledger_rows.append({
                        "decision_ts": dts, "event": "open", "mode": "fly", 
                        "tenor_i": float(f_belly["tenor_yrs"]), 
                        "tenor_left": float(f_left["tenor_yrs"]), "tenor_right": float(f_right["tenor_yrs"]),
                        "scale_dv01": hedge_dv01, "side": pos.side_desc
                    })

        elif mode == "overlay":
            # Legacy Pair Mode
            if hedges.empty: continue
            h_here = hedges[hedges["decision_ts"] == dts]
            if h_here.empty: continue
            
            snap_srt = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
            RISK_PIVOT = float(getattr(cr, "RISK_NAIVE_PIVOT", 5.0))
            
            for _, h in h_here.iterrows():
                t_trade = float(h["tenor_yrs"])
                if t_trade < EXEC_LEG_THRESHOLD: continue
                side_s = 1.0 if str(h["side"]).upper() == "CRCV" else -1.0
                exec_z = _get_z_at_tenor(snap_srt, t_trade)
                if exec_z is None: continue
                exec_row = snap_srt.iloc[(snap_srt["tenor_yrs"] - t_trade).abs().idxmin()]
                exec_tenor = float(exec_row["tenor_yrs"])
                drift_exec = calc_trade_drift(exec_tenor, side_s, float(exec_row["rate"]), r_float, xp_sorted, fp_sorted)
                
                best_c_row, best_score = None, -999.0
                for _, alt in snap_srt.iterrows():
                    alt_tenor = float(alt["tenor_yrs"])
                    if abs(alt_tenor - exec_tenor) < MIN_SEP_YEARS: continue
                    drift_alt = calc_trade_drift(alt_tenor, side_s, float(alt["rate"]), r_float, xp_sorted, fp_sorted)
                    net_drift = drift_alt - drift_exec
                    if net_drift < DRIFT_GATE: continue
                    z_alt = _to_float(alt["z_comb"])
                    disp = (z_alt - exec_z) if side_s > 0 else (exec_z - z_alt)
                    score = disp + (net_drift / max(0.1, abs(alt_tenor - exec_tenor)) * DRIFT_W)
                    if score > z_entry_curr and score > best_score:
                        best_score = score
                        best_c_row = alt
                
                if best_c_row is not None:
                    pos = PairPos(dts, best_c_row, exec_row, side_s*1.0, side_s*-1.0, decisions_per_day, 
                                  scale_dv01=float(h["dv01"]), meta={"trade_id": h.get("trade_id"), "side": h.get("side")},
                                  regime_mult=regime_mult, z_entry_base=Z_ENTRY_BASE, z_entry_final=z_entry_curr)
                    open_positions.append(pos)
                    ledger_rows.append({"decision_ts": dts, "event": "open", "mode": "overlay", "tenor_i": float(best_c_row["tenor_yrs"]), "tenor_j": exec_tenor})

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions, z_history

def run_all(yymms, *, decision_freq=None, carry=True, force_close_end=False, hedge_df=None, regime_signals=None, mode="fly"):
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    clean_hedges = None
    
    # UNIFIED LOADING
    if mode in ["overlay", "fly"]:
        if hedge_df is None: raise ValueError("Mode requires hedge_df")
        clean_hedges = prepare_hedge_tape(hedge_df, decision_freq)
        
    all_pos, all_led, all_by = [], [], []
    open_pos = []
    z_history_state = {} 
    
    for yymm in yymms:
        h_mon = None
        if clean_hedges is not None:
            y, m = 2000+int(yymm[:2]), int(yymm[2:])
            start, end = pd.Timestamp(y, m, 1), (pd.Timestamp(y, m, 1) + MonthEnd(1) + pd.Timedelta(days=1))
            h_mon = clean_hedges[(clean_hedges["decision_ts"] >= start) & (clean_hedges["decision_ts"] < end)].copy()
            
        p, l, b, open_pos, z_history_state = run_month(
            yymm, decision_freq=decision_freq, open_positions=open_pos, carry_in=carry, 
            hedges=h_mon, regime_signals=regime_signals, mode=mode, z_history=z_history_state
        )
        if not p.empty: all_pos.append(p.assign(yymm=yymm))
        if not l.empty: all_led.append(l.assign(yymm=yymm))
        if not b.empty: all_by.append(b.assign(yymm=yymm))
        
    if force_close_end and open_pos:
        final_ts = pd.Timestamp.now()
        if all_led: final_ts = max(x["decision_ts"].max() for x in all_led if not x.empty)
        closed_rows = []
        for pos in open_pos:
            pos.closed, pos.close_ts, pos.exit_reason = True, final_ts, "eoc"
            tcost_bp = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10))
            tcost_cash = tcost_bp * pos.scale_dv01
            row = {
                "open_ts": pos.open_ts, "close_ts": pos.close_ts, "exit_reason": "eoc",
                "mode": pos.mode, "pnl_net_bp": pos.pnl_bp - tcost_bp, 
                "pnl_net_cash": pos.pnl_cash - tcost_cash, "tcost_bp": tcost_bp,
                "scale_dv01": pos.scale_dv01
            }
            if pos.mode == "fly":
                row.update({"tenor_i": pos.t_belly, "tenor_left": pos.t_left, "tenor_right": pos.t_right})
            else:
                row.update({"tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j})
            closed_rows.append(row)
        if closed_rows:
            all_pos.append(pd.DataFrame(closed_rows).assign(yymm=yymms[-1]))

    return (pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame(),
            pd.concat(all_led, ignore_index=True) if all_led else pd.DataFrame(),
            pd.concat(all_by, ignore_index=True) if all_by else pd.DataFrame())

if __name__ == "__main__":
    import hybrid_filter as hf
    if len(sys.argv) < 2:
        print("Usage: python portfolio_test.py 2304 [2305 ...]")
        sys.exit(1)
    months = sys.argv[1:]
    trades_path = Path(f"{getattr(cr, 'TRADE_TYPES', 'trades')}.pkl")
    trades = pd.read_pickle(trades_path) if trades_path.exists() else None
    print(f"[INIT] Hybrid Filters...")
    signals = hf.get_or_build_hybrid_signals()
    run_mode = getattr(cr, "RUN_MODE", "fly") 
    print(f"[EXEC] Running {run_mode.upper()}...")
    pos, led, by = run_all(months, carry=True, force_close_end=True, hedge_df=trades, regime_signals=signals, mode=run_mode)
    out_dir = Path(cr.PATH_OUT)
    if not pos.empty: pos.to_parquet(out_dir / f"positions_ledger.parquet")
    if not led.empty: led.to_parquet(out_dir / f"marks_ledger.parquet")
    print(f"[DONE] Results saved to {out_dir}")
