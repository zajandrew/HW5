import os, sys
import math
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

# All config access via module namespace
import cr_config as cr

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
    """Simple helper to get Z-score at a specific tenor."""
    t = float(tenor)
    s = snap_last[["tenor_yrs", "z_comb"]].dropna()
    if s.empty: return None
    
    s = s.assign(_dist=(s["tenor_yrs"] - t).abs())
    row = s.loc[s["_dist"].idxmin()]
    
    if row["_dist"] > 0.05: return None
    return float(row["z_comb"])

def _calc_local_fly_z(snap_last, center_tenor, min_dist=1.5):
    """
    Calculates the Z-score of the 'Fly' (Curvature).
    Positive Fly Z = Center is Cheap vs Wings (Humped).
    """
    valid = snap_last[["tenor_yrs", "rate", "scale"]].dropna().sort_values("tenor_yrs")
    if valid.empty: return 0.0
    
    # 1. Find Center
    center_row = valid[np.isclose(valid["tenor_yrs"], center_tenor, atol=0.01)]
    if center_row.empty: return 0.0
    c_rate = float(center_row["rate"].iloc[0])
    
    # 2. Find Left Wing
    left_candidates = valid[valid["tenor_yrs"] <= (center_tenor - min_dist)]
    if left_candidates.empty: return 0.0
    l_row = left_candidates.iloc[-1]
    l_rate = float(l_row["rate"])
    
    # 3. Find Right Wing
    right_candidates = valid[valid["tenor_yrs"] >= (center_tenor + min_dist)]
    if right_candidates.empty: return 0.0
    r_row = right_candidates.iloc[0]
    r_rate = float(r_row["rate"])
    
    # 4. Calculate Fly Rate
    fly_rate = (2 * c_rate) - (l_rate + r_rate)
    
    # 5. Normalize
    scale = float(center_row["scale"].iloc[0]) if "scale" in center_row.columns else 0.01
    z_fly = fly_rate / max(1e-6, scale)
    return z_fly

def calc_trade_drift(tenor, direction, rate_fixed, rate_float, xp_sorted, fp_sorted):
    """
    Fast estimation of Daily Drift using pre-sorted arrays (Optimization).
    Args:
        rate_fixed: The specific rate of the tenor being traded.
        rate_float: The funding rate (pre-calculated).
        xp_sorted: Pre-sorted tenor arrays.
        fp_sorted: Pre-sorted rate arrays.
    """
    # 1. Rolldown Setup
    day_fraction = 1.0 / 360.0
    t_roll = max(0.0, tenor - day_fraction)
    
    # Fast interpolation using pre-sorted arrays
    rate_rolled = np.interp(t_roll, xp_sorted, fp_sorted)
    
    # 2. Calculate Components 
    # Carry PnL = (Fixed - Float) * 100 * (1/360)
    raw_carry_bps = (rate_fixed - rate_float) * 100.0 * day_fraction
    
    # Roll PnL = (Current - Rolled) * 100
    raw_roll_bps = (rate_fixed - rate_rolled) * 100.0
    
    # 3. Apply Direction
    total_drift_bps = (raw_carry_bps + raw_roll_bps) * direction
    
    return total_drift_bps

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
# Pair object
# ------------------------
class PairPos:
    def __init__(self, open_ts, cheap_row, rich_row, w_i, w_j, decisions_per_day, *, 
                 scale_dv01=1.0, meta=None, dir_sign=None, 
                 entry_rate_i=None, entry_rate_j=None,
                 # --- METADATA ---
                 fly_bonus=0.0, regime_mult=1.0, 
                 z_entry_base=0.0, z_entry_final=0.0):
        
        self.open_ts = open_ts
        self.tenor_i = _to_float(cheap_row["tenor_yrs"])
        self.tenor_j = _to_float(rich_row["tenor_yrs"])
        
        # Safety Cap (20% Rule)
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
        
        # Store metadata
        self.fly_bonus = fly_bonus
        self.regime_mult = regime_mult
        self.z_entry_base = z_entry_base
        self.z_entry_final = z_entry_final
        
        self.initial_dv01 = self.scale_dv01
        self.dv01_i_entry = self.scale_dv01 * self.w_i
        self.dv01_j_entry = self.scale_dv01 * self.w_j
        self.dv01_i_curr, self.dv01_j_curr = self.dv01_i_entry, self.dv01_j_entry
        self.rem_tenor_i, self.rem_tenor_j = self.tenor_i_orig, self.tenor_j_orig
        
        # Notional (Unit check: DV01 is risk, Tenor is time. Notional = Risk/Time)
        # Used for Carry calc
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
        self.last_zspread, self.last_z_dir, self.conv_pnl_proxy = self.entry_zspread, self.entry_z_dir, 0.0

    def _update_risk_decay(self, decision_ts):
        """Linearly decay DV01 based on Act/360 time passed."""
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
        """
        Optimized Mark: Uses pre-calculated curve vectors (xp, fp) and funding rate (r_float).
        """
        if decision_ts: self._update_risk_decay(decision_ts)

        # 1. Get Rates (Optimized: Linear Interp from pre-sorted curve)
        # Replaces slow DataFrame lookups with fast numpy interpolation
        ri = np.interp(self.tenor_i, xp_sorted, fp_sorted)
        rj = np.interp(self.tenor_j, xp_sorted, fp_sorted)
        
        self.last_rate_i, self.last_rate_j = ri, rj

        # 2. Price PnL
        pnl_price_i = (self.entry_rate_i - ri) * 100.0 * self.dv01_i_curr
        pnl_price_j = (self.entry_rate_j - rj) * 100.0 * self.dv01_j_curr
        self.pnl_price_cash = pnl_price_i + pnl_price_j

        # 3. Carry PnL (Incremental)
        if decision_ts and self.last_mark_ts:
            dt_days = max(0.0, (decision_ts.normalize() - self.last_mark_ts.normalize()).days)
            if dt_days > 0:
                inc_carry_i = (self.entry_rate_i - r_float) * 100.0 * self.not_i_entry * (dt_days / 360.0)
                inc_carry_j = (self.entry_rate_j - r_float) * 100.0 * self.not_j_entry * (dt_days / 360.0)
                self.pnl_carry_cash += (inc_carry_i + inc_carry_j)
            self.last_mark_ts = decision_ts

        # 4. Roll-Down PnL (Fast Interpolation)
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

        # 5. Totals
        self.pnl_cash = self.pnl_price_cash + self.pnl_carry_cash + self.pnl_roll_cash
        if self.scale_dv01 != 0.0 and np.isfinite(self.scale_dv01):
            self.pnl_bp = self.pnl_cash / self.scale_dv01
            self.pnl_price_bp = self.pnl_price_cash / self.scale_dv01
            self.pnl_carry_bp = self.pnl_carry_cash / self.scale_dv01
            self.pnl_roll_bp = self.pnl_roll_cash / self.scale_dv01
        else:
            self.pnl_bp = 0.0; self.pnl_price_bp = 0.0; self.pnl_carry_bp = 0.0; self.pnl_roll_bp = 0.0

        # 6. Z-Score (Kept robust as Z-scores are non-linear, lookup is safer than interp for Z)
        zi = _get_z_at_tenor(snap_last, self.tenor_i)
        zj = _get_z_at_tenor(snap_last, self.tenor_j)
        
        if zi is not None and zj is not None:
            zsp = zi - zj
            if np.isfinite(self.last_zspread):
                self.conv_pnl_proxy += (self.last_zspread - zsp) * 10.0
            self.last_zspread = zsp
            if np.isfinite(self.dir_sign):
                self.last_z_dir = self.dir_sign * zsp
            else:
                self.last_z_dir = np.nan
        else:
            zsp = self.last_zspread

        self.age_decisions += 1
        return zsp

# ------------------------
# RUN MONTH
# ------------------------
def run_month(
    yymm: str,
    *,
    decision_freq: str | None = None,
    open_positions: Optional[List[PairPos]] | None = None,
    carry_in: bool = True,
    hedges: Optional[pd.DataFrame] = None,
    regime_signals: Optional[pd.DataFrame] = None
):
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()

    enh_path = _enhanced_in_path(yymm)
    if not enh_path.exists():
        raise FileNotFoundError(f"Missing enhanced file {enh_path}. Run feature_creation.py first.")

    df = pd.read_parquet(enh_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), (open_positions or [])

    # Minimal validation
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

    # --- BASE PARAMS (Fair Weather) ---
    Z_ENTRY_BASE = float(getattr(cr, "Z_ENTRY", 0.75))
    Z_EXIT_BASE  = float(getattr(cr, "Z_EXIT", 0.40))
    Z_STOP_BASE  = float(getattr(cr, "Z_STOP", 3.00))
    
    EXEC_LEG_THRESHOLD = float(getattr(cr, "EXEC_LEG_TENOR_YEARS", 0.084))
    ALT_LEG_THRESHOLD  = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0))
    MIN_SEP_YEARS      = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    SHORT_END_EXTRA_Z  = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))
    SWITCH_COST_BP     = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10))
    
    DRIFT_GATE = float(getattr(cr, "DRIFT_GATE_BPS", -100.0)) 
    DRIFT_W    = float(getattr(cr, "DRIFT_WEIGHT", 0.0)) 

    # --- REGIME PREP ---
    sig_lookup = None
    if regime_signals is not None and not regime_signals.empty:
        # Index for fast daily lookup
        sig_lookup = regime_signals.drop_duplicates("decision_ts").set_index("decision_ts").sort_index()
    
    DYN_THRESH_ENABLE = getattr(cr, "DYN_THRESH_ENABLE", False)

    open_positions = (open_positions or []) if carry_in else []

    if hedges is not None and not hedges.empty:
        valid_decisions = df["decision_ts"].dropna().unique()
        hedges = hedges[hedges["decision_ts"].isin(valid_decisions)].copy()
    else:
        hedges = pd.DataFrame()

    ledger_rows: list[dict] = []
    closed_rows: list[dict] = []

    for dts, snap in df.groupby("decision_ts", sort=True):
        snap_last = (
            snap.sort_values("ts")
                .groupby("tenor_yrs", as_index=False)
                .tail(1)
                .reset_index(drop=True)
        )
        if snap_last.empty: continue

        # ============================================================
        # OPTIMIZATION: PRE-CALC CURVE VECTORS (ONCE PER DAY)
        # ============================================================
        valid_curve = snap_last[["tenor_yrs", "rate"]].dropna()
        xp_sorted = valid_curve["tenor_yrs"].values.astype(float)
        fp_sorted = valid_curve["rate"].values.astype(float)
        
        # Sort is required for np.interp
        sort_idx = np.argsort(xp_sorted)
        xp_sorted = xp_sorted[sort_idx]
        fp_sorted = fp_sorted[sort_idx]
        
        # Funding rate (Shortest tenor)
        r_float = fp_sorted[0] if len(fp_sorted) > 0 else 0.0

        # ============================================================
        # 0) CALCULATE DYNAMIC THRESHOLDS (The Transmission)
        # ============================================================
        z_entry_curr = Z_ENTRY_BASE
        z_exit_curr  = Z_EXIT_BASE
        z_stop_curr  = Z_STOP_BASE
        regime_mult  = 1.0
        
        if DYN_THRESH_ENABLE and sig_lookup is not None:
            if dts in sig_lookup.index:
                row = sig_lookup.loc[dts]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                
                # --- NEW MAPPING: Volatility (Defense) and Dispersion (Offense) ---
                # 1. Defense: scale_roll_z (Vol Spike) or normalized scale_mean
                trend_z = float(row["scale_roll_z"]) if "scale_roll_z" in row else 0.0
                
                # 2. Offense: z_xs_std (Dispersion)
                health_z = float(row["z_xs_std"]) if "z_xs_std" in row else 1.0
                
                sens_trend = float(getattr(cr, "SENS_TRENDINESS", 0.0))
                sens_health = float(getattr(cr, "SENS_HEALTH", 0.0))
                
                # Formula: 1 + (Vol * DefensiveWeight) + (Dispersion * OffensiveWeight)
                raw_mult = 1.0 + (trend_z * sens_trend) + (health_z * sens_health)
                regime_mult = max(0.5, min(raw_mult, 3.0)) 
                
                z_entry_curr *= regime_mult
                z_stop_curr  *= regime_mult
                z_exit_curr  *= regime_mult

        # ============================================================
        # 1) MARK POSITIONS & EXITS
        # ============================================================
        still_open: list[PairPos] = []
        
        for pos in open_positions:
            # OPTIMIZATION: PASS PRE-CALCULATED VECTORS
            zsp = pos.mark(snap_last, xp_sorted, fp_sorted, r_float, decision_ts=dts)
            
            entry_z = pos.entry_zspread
            exit_flag = None

            if np.isfinite(zsp) and np.isfinite(entry_z):
                entry_dir = getattr(pos, "entry_z_dir", pos.dir_sign * entry_z)
                curr_dir = getattr(pos, "last_z_dir", pos.dir_sign * zsp)

                if np.isfinite(entry_dir) and np.isfinite(curr_dir):
                    sign_entry = np.sign(entry_dir)
                    sign_curr = np.sign(curr_dir)
                    same_side = (sign_entry != 0) and (sign_entry == sign_curr)
                    
                    # Profit Take: Use DYNAMIC z_exit_curr
                    if same_side and (abs(curr_dir) <= abs(entry_dir)) and (abs(curr_dir) <= z_exit_curr):
                        exit_flag = "reversion"
                    
                    # Stop Loss: Use DYNAMIC z_stop_curr
                    dz_dir = curr_dir - entry_dir
                    if same_side and (abs(curr_dir) >= abs(entry_dir)) and (abs(dz_dir) >= z_stop_curr):
                        exit_flag = "stop"

            # --- DYNAMIC STALEMATE (Velocity) ---
            STALE_ENABLE = getattr(cr, "STALE_ENABLE", False)
            if STALE_ENABLE and exit_flag is None:
                STALE_START = float(getattr(cr, "STALE_START_DAYS", 3.0))
                MIN_VELOCITY = float(getattr(cr, "STALE_MIN_VELOCITY_Z", 0.015))
                
                days_held = pos.age_decisions / max(1, decisions_per_day)
                if days_held >= STALE_START:
                    # Price Velocity
                    z_improvement = (pos.entry_zspread - zsp) * pos.dir_sign
                    vel_price = z_improvement / days_held
                    
                    # Carry Velocity (Scaled to Z)
                    # Approx Scale (Fall back to 0.5bp if missing)
                    pair_scale = 0.0050
                    if "scale" in snap_last.columns:
                        pair_scale = float(snap_last["scale"].median())
                    
                    # Realized Daily Drift (Rate %)
                    # (Carry_BP + Roll_BP) is in Bps. Divide by 100 to get Rate Units (%).
                    total_drift_bps = (pos.pnl_carry_bp + pos.pnl_roll_bp)
                    daily_drift_rate = (total_drift_bps / days_held) / 100.0
                    
                    vel_carry = daily_drift_rate / max(1e-6, pair_scale)
                    
                    if (vel_price + vel_carry) < MIN_VELOCITY:
                        exit_flag = "stalemate"

            # --- SAFETY CAP (20% Rule) ---
            if exit_flag is None:
                if pos.age_decisions >= (pos.max_days_pair * decisions_per_day):
                    exit_flag = "safety_cap"

            if exit_flag is not None:
                pos.closed = True
                pos.close_ts = dts
                pos.exit_reason = exit_flag

            ledger_rows.append({
                "decision_ts": dts, "event": "mark",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "pnl_bp": pos.pnl_bp, "pnl_cash": pos.pnl_cash,
                "pnl_price_bp": pos.pnl_price_bp, "pnl_carry_bp": pos.pnl_carry_bp,
                "pnl_roll_bp": pos.pnl_roll_bp, "pnl_price_cash": pos.pnl_price_cash,
                "pnl_carry_cash": pos.pnl_carry_cash, "pnl_roll_cash": pos.pnl_roll_cash,
                "z_spread": zsp, "closed": pos.closed, 
                "rate_i": getattr(pos, "last_rate_i", np.nan),
                "rate_j": getattr(pos, "last_rate_j", np.nan),
            })

            if pos.closed:
                tcost_bp = SWITCH_COST_BP
                tcost_cash = tcost_bp * pos.scale_dv01
                pos.tcost_bp = tcost_bp
                pos.tcost_cash = tcost_cash
                
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": pos.close_ts, 
                    "exit_reason": pos.exit_reason,
                    "mode": "overlay", 
                    "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i, "w_j": pos.w_j,
                    "entry_rate_i": pos.entry_rate_i, "entry_rate_j": pos.entry_rate_j,
                    "close_rate_i": getattr(pos, "last_rate_i", np.nan),
                    "close_rate_j": getattr(pos, "last_rate_j", np.nan),
                    "scale_dv01": pos.scale_dv01,
                    "entry_zspread": pos.entry_zspread,
                    
                    # PnL
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, 
                    "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "tcost_bp": tcost_bp, "tcost_cash": tcost_cash,
                    
                    # Attribution
                    "pnl_price_bp": pos.pnl_price_bp, "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp": pos.pnl_roll_bp, "pnl_price_cash": pos.pnl_price_cash,
                    "pnl_carry_cash": pos.pnl_carry_cash, "pnl_roll_cash": pos.pnl_roll_cash,
                    
                    # Analytics
                    "days_held_equiv": pos.age_decisions / max(1, decisions_per_day),
                    "trade_id": pos.meta.get("trade_id"),
                    "side": pos.meta.get("side"),
                    
                    # METADATA
                    "z_entry_base": pos.z_entry_base,
                    "z_entry_final": pos.z_entry_final,
                    "regime_mult": pos.regime_mult,
                    "fly_bonus": pos.fly_bonus,
                })
            else:
                still_open.append(pos)
        
        open_positions = still_open
        
        # ============================================================
        # 3) SCAN HEDGES (OVERLAY)
        # ============================================================
        if hedges.empty: continue
        h_here = hedges[hedges["decision_ts"] == dts]
        if h_here.empty: continue
        
        snap_srt = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
        
        # --- NAIVE RISK PRE-CALC ---
        RISK_NAIVE_ENABLE = getattr(cr, "RISK_NAIVE_ENABLE", False)
        curr_left, curr_right = 0.0, 0.0
        RISK_PIVOT = float(getattr(cr, "RISK_NAIVE_PIVOT", 5.0))
        RISK_LIMIT = float(getattr(cr, "RISK_NAIVE_LIMIT", 80_000))
        
        if RISK_NAIVE_ENABLE:
            for p in open_positions:
                if p.tenor_i <= RISK_PIVOT: curr_left += p.dv01_i_curr * np.sign(p.w_i)
                else: curr_right += p.dv01_i_curr * np.sign(p.w_i)
                if p.tenor_j <= RISK_PIVOT: curr_left += p.dv01_j_curr * np.sign(p.w_j)
                else: curr_right += p.dv01_j_curr * np.sign(p.w_j)

        for _, h in h_here.iterrows():
            t_trade = float(h["tenor_yrs"])
            if t_trade < EXEC_LEG_THRESHOLD: continue

            side_s = 1.0 if str(h["side"]).upper() == "CRCV" else -1.0
            trade_dv01 = float(h["dv01"])
            
            exec_z = _get_z_at_tenor(snap_srt, t_trade)
            if exec_z is None: continue
            
            exec_row = snap_srt.iloc[(snap_srt["tenor_yrs"] - t_trade).abs().idxmin()]
            exec_tenor = float(exec_row["tenor_yrs"])
            exec_bucket = assign_bucket(exec_tenor)

            # Pre-calc Execution Leg Risk Impact
            exec_is_left = (exec_tenor <= RISK_PIVOT)
            delta_j = trade_dv01 * -side_s

            best_c_row, best_score = None, -999.0

            # --- BUTTERFLY ROUTER PRE-CALC ---
            fly_bonus_base = 0.0
            FLY_ENABLE = getattr(cr, "FLY_ENABLE", False)
            if FLY_ENABLE:
                fly_z_exec = _calc_local_fly_z(snap_srt, exec_tenor, 
                                               min_dist=float(getattr(cr, "FLY_MIN_DIST", 1.5)))
                
                # Bonus if hedge is bad (Fly says Switch)
                hedge_dir = -side_s
                fly_alignment = hedge_dir * fly_z_exec
                fly_bonus_base = -1.0 * fly_alignment * float(getattr(cr, "FLY_WEIGHT", 0.15))

            # Drift of Execution Leg
            drift_exec = calc_trade_drift(exec_tenor, side_s, float(exec_row["rate"]), r_float, xp_sorted, fp_sorted)

            # Search Alternates
            for _, alt in snap_srt.iterrows():
                alt_tenor = float(alt["tenor_yrs"])
                
                # 1. Tenor constraints
                if alt_tenor < ALT_LEG_THRESHOLD: continue
                if alt_tenor == exec_tenor: continue
                if abs(alt_tenor - exec_tenor) < MIN_SEP_YEARS: continue
                
                # 2. Bucket constraints
                alt_bucket = assign_bucket(alt_tenor)
                if alt_bucket == "short" and exec_bucket == "long": continue
                if exec_bucket == "short" and alt_bucket == "long": continue
                
                # 3. Naive Risk Check
                if RISK_NAIVE_ENABLE:
                    alt_is_left = (alt_tenor <= RISK_PIVOT)
                    delta_i = trade_dv01 * side_s
                    
                    proj_left = curr_left
                    proj_right = curr_right
                    
                    if exec_is_left: proj_left += delta_j
                    else: proj_right += delta_j
                        
                    if alt_is_left: proj_left += delta_i
                    else: proj_right += delta_i
                    
                    if abs(proj_left) > RISK_LIMIT or abs(proj_right) > RISK_LIMIT:
                        continue 

                # 4. Z-Score Opportunity
                z_alt = _to_float(alt["z_comb"])
                disp = (z_alt - exec_z) if side_s > 0 else (exec_z - z_alt)
                
                # 5. Thresholds (Dynamic Entry!)
                thresh = z_entry_curr
                if alt_bucket == "short" or exec_bucket == "short":
                    thresh += SHORT_END_EXTRA_Z
                
                # 6. Drift Logic
                drift_alt = calc_trade_drift(alt_tenor, side_s, float(alt["rate"]), r_float, xp_sorted, fp_sorted)
                net_drift_bps = drift_alt - drift_exec 
                
                if net_drift_bps < DRIFT_GATE: continue 
                
                dist_years = abs(alt_tenor - exec_tenor)
                norm_drift_bps = net_drift_bps / max(0.1, dist_years)
                
                # Score
                score = disp + (norm_drift_bps * DRIFT_W) + fly_bonus_base
                
                if score < thresh: continue
                
                if score > best_score:
                    best_score = score
                    best_c_row = alt

            # Execute Best
            if best_c_row is not None:
                rate_i, rate_j = None, None
                ti, tj = tenor_to_ticker(float(best_c_row["tenor_yrs"])), tenor_to_ticker(t_trade)
                if ti and f"{ti}_mid" in h: rate_i = _to_float(h[f"{ti}_mid"])
                if tj and f"{tj}_mid" in h: rate_j = _to_float(h[f"{tj}_mid"])
                
                if rate_i is None: rate_i = _to_float(best_c_row["rate"])
                if rate_j is None: rate_j = _to_float(exec_row["rate"])

                pos = PairPos(dts, best_c_row, exec_row, side_s*1.0, side_s*-1.0, decisions_per_day, 
                              scale_dv01=float(h["dv01"]), 
                              meta={"trade_id": h.get("trade_id"), "side": h.get("side")},
                              entry_rate_i=rate_i, entry_rate_j=rate_j,
                              # --- PASS METADATA ---
                              fly_bonus=fly_bonus_base,
                              regime_mult=regime_mult,
                              z_entry_base=Z_ENTRY_BASE,
                              z_entry_final=z_entry_curr) # The threshold used to pass
                open_positions.append(pos)
                ledger_rows.append({"decision_ts": dts, "event": "open"})

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions

def run_all(yymms, *, decision_freq=None, carry=True, force_close_end=False, hedge_df=None, regime_signals=None):
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    
    clean_hedges = None
    if hedge_df is None: 
        raise ValueError("Overlay mode requires hedge_df")
    clean_hedges = prepare_hedge_tape(hedge_df, decision_freq)
        
    all_pos, all_led, all_by = [], [], []
    open_pos = []
    
    for yymm in yymms:
        h_mon = None
        if clean_hedges is not None:
            y, m = 2000+int(yymm[:2]), int(yymm[2:])
            start, end = pd.Timestamp(y, m, 1), (pd.Timestamp(y, m, 1) + MonthEnd(1) + pd.Timedelta(days=1))
            h_mon = clean_hedges[(clean_hedges["decision_ts"] >= start) & (clean_hedges["decision_ts"] < end)].copy()
            
        p, l, b, open_pos = run_month(yymm, decision_freq=decision_freq, open_positions=open_pos, carry_in=carry, hedges=h_mon, regime_signals=regime_signals)
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
            closed_rows.append({
                "open_ts": pos.open_ts, "close_ts": pos.close_ts, "exit_reason": "eoc",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "pnl_net_bp": pos.pnl_bp - tcost_bp, "pnl_net_cash": pos.pnl_cash - tcost_cash,
                "tcost_bp": tcost_bp, "tcost_cash": tcost_cash,
                "scale_dv01": pos.scale_dv01
            })
        if closed_rows:
            all_pos.append(pd.DataFrame(closed_rows).assign(yymm=yymms[-1]))

    return (pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame(),
            pd.concat(all_led, ignore_index=True) if all_led else pd.DataFrame(),
            pd.concat(all_by, ignore_index=True) if all_by else pd.DataFrame())

if __name__ == "__main__":
    import hybrid_filter as hf
    
    if len(sys.argv) < 2:
        print("Usage: python portfolio_test.py 2304 [2305 2306 ...]")
        sys.exit(1)
        
    months = sys.argv[1:]
    
    trades_path = Path(f"{getattr(cr, 'TRADE_TYPES', 'trades')}.pkl")
    trades = pd.read_pickle(trades_path) if trades_path.exists() else None
    
    if trades is None:
        print("[ERROR] No trades.pkl found. Strategy mode is deprecated. Overlay requires trades.")
        sys.exit(1)

    print(f"[INIT] Hybrid Filters (Regime Only)...")
    signals = hf.get_or_build_hybrid_signals()
    # Note: We skip the regime_mask creation, we pass signals for dynamic logic

    print(f"[EXEC] Running Overlay on {len(months)} months.")
    pos, led, by = run_all(months, carry=True, force_close_end=True, hedge_df=trades, regime_signals=signals)

    out_dir = Path(cr.PATH_OUT)
    suffix = getattr(cr, "OUT_SUFFIX", "")
    if not pos.empty: pos.to_parquet(out_dir / f"positions_ledger{suffix}.parquet")
    if not led.empty: led.to_parquet(out_dir / f"marks_ledger{suffix}.parquet")
    if not by.empty: by.to_parquet(out_dir / f"pnl_by_bucket{suffix}.parquet")
    print(f"[DONE] Results saved to {out_dir}")
