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
    if raw_df is None or raw_df.empty: 
        print("[WARN] Hedge tape is empty.")
        return pd.DataFrame()
    
    df = raw_df.copy()
    if "tradetimeUTC" in df.columns:
        df["trade_ts"] = pd.to_datetime(df["tradetimeUTC"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    elif "ts" in df.columns:
        df["trade_ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    else:
        df["trade_ts"] = pd.Timestamp.now()

    decision_freq = str(decision_freq).upper()
    if decision_freq == "D": df["decision_ts"] = df["trade_ts"].dt.floor("d")
    elif decision_freq == "H": df["decision_ts"] = df["trade_ts"].dt.floor("h")
    else: raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.upper()
        df = df[df["side"].isin(["CPAY", "CRCV"])]
    else:
        print("[ERROR] Hedge tape missing 'side'.")
        return pd.DataFrame()

    if "instrument" in df.columns:
        df["tenor_yrs"] = df["instrument"].map(_map_instrument_to_tenor)
    else:
        print("[ERROR] Hedge tape missing 'instrument'.")
        return pd.DataFrame()
    
    df = df[np.isfinite(df["tenor_yrs"])]
    
    if "EqVolDelta" in df.columns:
        df["dv01"] = pd.to_numeric(df["EqVolDelta"], errors="coerce").abs()
    elif "dv01" in df.columns:
        df["dv01"] = pd.to_numeric(df["dv01"], errors="coerce").abs()
    else:
        print("[ERROR] Hedge tape missing 'EqVolDelta' or 'dv01'.")
        return pd.DataFrame()

    df = df[np.isfinite(df["dv01"]) & (df["dv01"] > 1.0)]
    df = df.dropna(subset=["trade_ts", "decision_ts", "tenor_yrs", "dv01"])
    
    if "trade_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["trade_id"] = df.index.astype(int)
    
    cols = ["trade_id", "trade_ts", "decision_ts", "tenor_yrs", "side", "dv01"]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra]

def calc_leg_drift(tenor, rate, r_float, cs_func):
    dt = 1.0/360.0
    carry_bps = (rate - r_float) * 100.0 * dt
    rolled_rate = float(cs_func(tenor - dt))
    roll_bps = (rate - rolled_rate) * 100.0
    return carry_bps + roll_bps

# ------------------------
# POSITION OBJECT (FLY ONLY)
# ------------------------

class FlyPos:
    def __init__(self, open_ts, belly_row, left_row, right_row, decisions_per_day, *, 
                 scale_dv01=10_000.0, meta=None, 
                 z_score_current=0.0, z_score_trend=0.0,
                 z_entry_base=0.0, z_entry_final=0.0, regime_mult=1.0,
                 direction_multiplier=1.0): 
        
        self.open_ts = open_ts
        self.meta = meta or {}
        self.mode = "overlay"  
        self.scale_dv01 = float(scale_dv01)
        
        # --- Tenors ---
        self.t_belly = float(belly_row["tenor_yrs"])
        self.t_left  = float(left_row["tenor_yrs"])
        self.t_right = float(right_row["tenor_yrs"])
        self.t_belly_orig, self.t_left_orig, self.t_right_orig = self.t_belly, self.t_left, self.t_right
        
        # --- Safety Cap ---
        # "Minimum Tenor Safety Factor": e.g. 73.0 (from 365/5). 
        # Max Days = Tenor * Factor.
        safety_factor = float(getattr(cr, "MIN_TENOR_SAFETY_FACTOR", 73.0))
        self.max_days_fly = min(self.t_belly, self.t_left, self.t_right) * safety_factor

        # --- Weights & Direction ---
        self.dm = direction_multiplier
        self.w_belly = 1.0 * self.dm
        self.w_left  = -0.5 * self.dm
        self.w_right = -0.5 * self.dm
        
        self.dir_belly = 1 if self.w_belly > 0 else -1
        self.dir_left  = 1 if self.w_left > 0 else -1
        self.dir_right = 1 if self.w_right > 0 else -1

        # --- Risk Init (DV01) ---
        self.dv01_belly_entry = self.scale_dv01 * abs(self.w_belly)
        self.dv01_left_entry  = self.scale_dv01 * abs(self.w_left)
        self.dv01_right_entry = self.scale_dv01 * abs(self.w_right)
        
        self.dv01_belly_curr = self.dv01_belly_entry
        self.dv01_left_curr  = self.dv01_left_entry
        self.dv01_right_curr = self.dv01_right_entry

        # --- Notional Init ---
        self.not_belly = self.dv01_belly_entry / max(1e-6, self.t_belly)
        self.not_left  = self.dv01_left_entry  / max(1e-6, self.t_left)
        self.not_right = self.dv01_right_entry / max(1e-6, self.t_right)

        # --- Entry Rates ---
        self.r_entry_belly = float(belly_row["rate"])
        self.r_entry_left  = float(left_row["rate"])
        self.r_entry_right = float(right_row["rate"])
        
        self.r_curr_belly = self.r_entry_belly
        self.r_curr_left  = self.r_entry_left
        self.r_curr_right = self.r_entry_right
        
        # --- Stats ---
        self.entry_z = z_score_current
        self.trend_z = z_score_trend
        
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
        
        self.r_curr_belly = float(cs(self.t_belly))
        self.r_curr_left  = float(cs(self.t_left))
        self.r_curr_right = float(cs(self.t_right))
        
        pnl_p_b = (self.r_entry_belly - self.r_curr_belly) * 100.0 * self.dv01_belly_curr * self.dir_belly
        pnl_p_l = (self.r_entry_left - self.r_curr_left)   * 100.0 * self.dv01_left_curr  * self.dir_left
        pnl_p_r = (self.r_entry_right - self.r_curr_right) * 100.0 * self.dv01_right_curr * self.dir_right
        self.pnl_price_cash = pnl_p_b + pnl_p_l + pnl_p_r
        
        if decision_ts and self.last_mark_ts:
            dt_days = (decision_ts - self.last_mark_ts).days
            if dt_days > 0:
                t_frac = dt_days / 360.0
                carry_b = (self.r_entry_belly - r_float) * 100.0 * self.not_belly * t_frac * self.dir_belly
                carry_l = (self.r_entry_left  - r_float) * 100.0 * self.not_left  * t_frac * self.dir_left
                carry_r = (self.r_entry_right - r_float) * 100.0 * self.not_right * t_frac * self.dir_right
                self.pnl_carry_cash += (carry_b + carry_l + carry_r)
                
                step = 1.0/360.0
                r_roll_b = float(cs(self.t_belly - step))
                r_roll_l = float(cs(self.t_left  - step))
                r_roll_r = float(cs(self.t_right - step))
                
                roll_b = (self.r_curr_belly - r_roll_b) * 100.0 * self.dv01_belly_curr * self.dir_belly * dt_days
                roll_l = (self.r_curr_left  - r_roll_l) * 100.0 * self.dv01_left_curr  * self.dir_left  * dt_days
                roll_r = (self.r_curr_right - r_roll_r) * 100.0 * self.dv01_right_curr * self.dir_right * dt_days
                self.pnl_roll_cash += (roll_b + roll_l + roll_r)
                
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
    hedges: Optional[pd.DataFrame] = None,
    regime_signals: Optional[pd.DataFrame] = None,
    z_history: Optional[Dict] = None 
):
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    enh_path = _enhanced_in_path(yymm)
    if not enh_path.exists(): raise FileNotFoundError(f"Missing {enh_path}")
    df = pd.read_parquet(enh_path)
    if df.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), (open_positions or []), z_history

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
    FLY_WING_MIN, FLY_WING_MAX = getattr(cr, "FLY_WING_WIDTH_RANGE", (2.0, 7.0))
    TREND_WINDOW = int(getattr(cr, "Z_TREND_WINDOW", 20))
    MIN_TENOR_FLOOR = 0.5 
    DRIFT_GATE = float(getattr(cr, "DRIFT_GATE_BPS", -100.0)) 
    DRIFT_W = float(getattr(cr, "DRIFT_WEIGHT", 0.0))

    # --- REGIME ---
    sig_lookup = None
    if regime_signals is not None and not regime_signals.empty:
        sig_lookup = regime_signals.drop_duplicates("decision_ts").set_index("decision_ts").sort_index()
    DYN_THRESH_ENABLE = getattr(cr, "DYN_THRESH_ENABLE", False)

    open_positions = (open_positions or []) if carry_in else []
    if hedges is not None and not hedges.empty:
        valid_decisions = df["decision_ts"].dropna().unique()
        hedges = hedges[hedges["decision_ts"].isin(valid_decisions)].copy()
    else: hedges = pd.DataFrame()

    ledger_rows: list[dict] = []
    closed_rows: list[dict] = []
    if z_history is None: z_history = {} 

    for dts, snap in df.groupby("decision_ts", sort=True):
        snap_last = (
            snap.sort_values("ts").groupby("tenor_yrs", as_index=False).tail(1).reset_index(drop=True)
        )
        if snap_last.empty: continue

        valid_curve = snap_last[["tenor_yrs", "rate"]].dropna()
        xp_sorted = valid_curve["tenor_yrs"].values.astype(float)
        fp_sorted = valid_curve["rate"].values.astype(float)
        sort_idx = np.argsort(xp_sorted)
        xp_sorted = xp_sorted[sort_idx]
        fp_sorted = fp_sorted[sort_idx]
        r_float = fp_sorted[0] if len(fp_sorted) > 0 else 0.0

        daily_cs = CubicSpline(xp_sorted, fp_sorted)

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
                
                scale_z = float(row["scale_roll_z"]) / 0.01 if "scale_roll_z" in row else 0.0
                disp_z  = float(row["z_xs_std"]) if "z_xs_std" in row else 1.0
                health_z = float(row["signal_health_z"]) if "signal_health_z" in row else 1.0

                sens_scale = float(getattr(cr, "SENS_SCALE", 0.0))
                sens_disp  = float(getattr(cr, "SENS_DISP", 0.0))
                sens_health = float(getattr(cr, "SENS_HEALTH", 0.0))

                raw_mult = 1.0 + (scale_z * sens_scale) + (disp_z * sens_disp) + (health_z * sens_health)
                regime_mult = max(0.5, min(raw_mult, 3.0))
                
                z_entry_curr *= regime_mult
                z_stop_curr  *= regime_mult
                z_exit_curr  *= regime_mult

        # ----------------------------------------
        # 1. MARK POSITIONS
        # ----------------------------------------
        still_open: list = []
        for pos in open_positions:
            zsp = pos.mark(snap_last, xp_sorted, fp_sorted, r_float, decision_ts=dts)
            exit_flag = None
            
            if hasattr(pos, 'last_z_val'):
                curr_z = pos.last_z_val
                
                # A. Reversion (Profit Take)
                if abs(curr_z) <= z_exit_curr: 
                    exit_flag = "reversion"
                
                # B. Stop Loss (Based on Z-Score)
                # If we entered Short (-1) at +2.0, we stop if it goes to +5.0 (Rising is bad).
                # If we entered Long (+1) at -2.0, we stop if it goes to -5.0 (Falling is bad).
                entry_z = pos.entry_z
                bad_move = 0.0
                if entry_z > 0: bad_move = curr_z - entry_z
                else: bad_move = entry_z - curr_z
                
                if bad_move >= z_stop_curr: 
                    exit_flag = "stop"

            # C. Dynamic Stalemate (Velocity)
            STALE_ENABLE = getattr(cr, "STALE_ENABLE", False)
            if STALE_ENABLE and exit_flag is None:
                STALE_START = float(getattr(cr, "STALE_START_DAYS", 3.0))
                MIN_VELOCITY = float(getattr(cr, "STALE_MIN_VELOCITY_Z", 0.015))
                
                days_held = pos.age_decisions / max(1, decisions_per_day)
                if days_held >= STALE_START:
                    # 1. Price Velocity (Z-Score improvement per day)
                    # Note: We need 'dir_sign' logic. If we want Z to drop, Z_imp = (Entry - Curr).
                    # If we want Z to rise, Z_imp = (Curr - Entry).
                    # Simplified: We use 'needed_dir' logic. 
                    # If needed_dir=-1 (Pay), we want Z down. Imp = Entry - Curr.
                    # If needed_dir=+1 (Rec), we want Z up. Imp = Curr - Entry.
                    needed_dir = pos.dm # Stored as direction_multiplier
                    z_improvement = (pos.entry_z - curr_z) if needed_dir == -1.0 else (curr_z - pos.entry_z)
                    vel_price = z_improvement / days_held

                    # 2. Carry Velocity (Scaled)
                    pair_scale = 0.0050
                    if "scale" in snap_last.columns:
                        pair_scale = float(snap_last["scale"].median())
                    
                    # Realized Daily Drift (%)
                    total_drift_bps = (pos.pnl_carry_bp + pos.pnl_roll_bp)
                    daily_drift_rate = (total_drift_bps / days_held) / 100.0
                    vel_carry = daily_drift_rate / max(1e-6, pair_scale)
                    
                    if (vel_price + vel_carry) < MIN_VELOCITY:
                        exit_flag = "stalemate"

            # D. Safety Cap (Min Tenor Rule)
            if exit_flag is None:
                if pos.age_decisions >= (pos.max_days_fly * decisions_per_day): 
                    exit_flag = "safety_cap"

            if exit_flag is not None:
                pos.closed = True
                pos.close_ts = dts
                pos.exit_reason = exit_flag

            row = {
                "decision_ts": dts, "event": "mark", "mode": pos.mode,
                "tenor_i": pos.t_belly, 
                "pnl_bp": pos.pnl_bp, "pnl_cash": pos.pnl_cash,
                "pnl_price_bp": pos.pnl_price_bp, "pnl_carry_bp": pos.pnl_carry_bp,
                "pnl_roll_bp": pos.pnl_roll_bp,
                "pnl_price_cash": pos.pnl_price_cash,
                "pnl_carry_cash": pos.pnl_carry_cash,
                "pnl_roll_cash": pos.pnl_roll_cash,
                "rate_belly": pos.r_curr_belly, "closed": pos.closed,
                "entry_z": pos.entry_z, "current_z": curr_z
            }
            ledger_rows.append(row)

            if pos.closed:
                tcost_bp = SWITCH_COST_BP
                tcost_cash = tcost_bp * pos.scale_dv01
                
                # --- EXPLICIT TIE-OUT SUMS ---
                pnl_net_cash_final = pos.pnl_price_cash + pos.pnl_carry_cash + pos.pnl_roll_cash - tcost_cash
                pnl_net_bp_final = pos.pnl_price_bp + pos.pnl_carry_bp + pos.pnl_roll_bp - tcost_bp

                cl_row = {
                    "open_ts": pos.open_ts, "close_ts": pos.close_ts, 
                    "exit_reason": pos.exit_reason, "mode": pos.mode,
                    "pnl_net_bp": pnl_net_bp_final, 
                    "pnl_net_cash": pnl_net_cash_final,
                    "tcost_bp": tcost_bp, "tcost_cash": tcost_cash,
                    "days_held": pos.age_decisions / decisions_per_day,
                    "trade_id": pos.meta.get("trade_id", -1),
                    "scale_dv01": pos.scale_dv01,
                    "tenor_i": pos.t_belly, "tenor_left": pos.t_left, "tenor_right": pos.t_right,
                    "side": pos.side_desc,
                    "pnl_price_cash": pos.pnl_price_cash,
                    "pnl_carry_cash": pos.pnl_carry_cash,
                    "pnl_roll_cash":  pos.pnl_roll_cash,
                    "pnl_price_bp": pos.pnl_price_bp,
                    "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp":  pos.pnl_roll_bp,
                    "dir_belly": pos.dir_belly, "dir_left": pos.dir_left, "dir_right": pos.dir_right,
                    "rate_belly_entry": pos.r_entry_belly, "rate_belly_exit": pos.r_curr_belly,
                    "rate_left_entry":  pos.r_entry_left,  "rate_left_exit":  pos.r_curr_left,
                    "rate_right_entry": pos.r_entry_right, "rate_right_exit": pos.r_curr_right,
                    "entry_z": pos.entry_z, "exit_z": curr_z
                }
                closed_rows.append(cl_row)
            else:
                still_open.append(pos)
        open_positions = still_open
        
        # ----------------------------------------
        # 2. SCAN FOR NEW TRADES
        # ----------------------------------------
        if hedges.empty: continue
        h_here = hedges[hedges["decision_ts"] == dts]
        if h_here.empty: continue

        curve = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
        
        for _, h in h_here.iterrows():
            if len(open_positions) >= 50: break

            t_hedge = float(h["tenor_yrs"])
            hedge_side = str(h["side"]).upper()
            hedge_dv01 = float(h["dv01"])
            
            if hedge_dv01 < 100: continue
            if t_hedge < MIN_TENOR_FLOOR: continue

            needed_dir = 1.0 if hedge_side == "CPAY" else -1.0
            
            belly_rows = curve.iloc[(curve["tenor_yrs"] - t_hedge).abs().argsort()[:1]]
            if belly_rows.empty: continue
            belly = belly_rows.iloc[0]
            t_belly = float(belly["tenor_yrs"])
            if abs(t_belly - t_hedge) > 0.05: continue 
            
            z_curr = float(belly["z_comb"])
            if needed_dir == -1.0 and z_curr > 2.0: continue
            if needed_dir == 1.0 and z_curr < -2.0: continue
            
            drift_b = calc_leg_drift(t_belly, float(belly["rate"]), r_float, daily_cs)

            valid_wings = False
            cand_left, cand_right, cand_score = None, None, -999.0
            
            for j in range(0, len(curve)):
                left = curve.iloc[j]
                t_left = float(left["tenor_yrs"])
                if t_left < MIN_TENOR_FLOOR: continue
                if t_left >= t_belly: break 
                width_l = t_belly - t_left
                if width_l > FLY_WING_MAX: continue
                if width_l < FLY_WING_MIN: continue
                
                drift_l = calc_leg_drift(t_left, float(left["rate"]), r_float, daily_cs)
                
                for k in range(j+1, len(curve)):
                    right = curve.iloc[k]
                    t_right = float(right["tenor_yrs"])
                    if t_right < MIN_TENOR_FLOOR: continue
                    if t_right <= t_belly: continue
                    width_r = t_right - t_belly
                    if width_r > FLY_WING_MAX: break
                    if width_r < FLY_WING_MIN: continue
                    
                    z_l = float(left["z_comb"])
                    z_r = float(right["z_comb"])
                    z_fly = z_curr - 0.5*(z_l + z_r)
                    
                    drift_r = calc_leg_drift(t_right, float(right["rate"]), r_float, daily_cs)
                    
                    net_drift_bps = needed_dir * (drift_b - 0.5*(drift_l + drift_r))
                    if net_drift_bps < DRIFT_GATE: continue

                    score_z = needed_dir * z_fly
                    score_drift = net_drift_bps * DRIFT_W
                    total_score = score_z + score_drift
                    
                    if not valid_wings or total_score > cand_score:
                        cand_left, cand_right = left, right
                        cand_score = total_score
                        valid_wings = True
            
            if not valid_wings: continue
            
            z_hist = z_history.get(t_belly, [])
            z_slow = np.mean(z_hist) if len(z_hist) > 5 else z_curr
            
            pos = FlyPos(dts, belly, cand_left, cand_right, decisions_per_day,
                         scale_dv01=hedge_dv01,
                         z_score_current=z_curr, z_score_trend=z_slow,
                         z_entry_final=z_entry_curr, regime_mult=regime_mult,
                         direction_multiplier=needed_dir, 
                         meta={"trade_id": h.get("trade_id"), "side": hedge_side})
            
            open_positions.append(pos)
            ledger_rows.append({
                "decision_ts": dts, "event": "open", "mode": pos.mode,
                "tenor_i": t_belly, 
                "scale_dv01": hedge_dv01, 
                "side": pos.side_desc
            })

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions, z_history

def _enhanced_in_path(yymm):
    base = Path(getattr(cr, "PATH_ENH", "."))
    freq = getattr(cr, "RUN_TAG", "D").lower()
    return base / f"{yymm}_enh_{freq}.parquet"

def run_all(yymms, *, decision_freq=None, carry=True, force_close_end=False, hedge_df=None, regime_signals=None):
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    
    clean_hedges = None
    if hedge_df is None: raise ValueError("Strategy requires hedge_df input.")
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
            hedges=h_mon, regime_signals=regime_signals, z_history=z_history_state
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
            
            pnl_net_cash_final = pos.pnl_price_cash + pos.pnl_carry_cash + pos.pnl_roll_cash - tcost_cash
            pnl_net_bp_final = pos.pnl_price_bp + pos.pnl_carry_bp + pos.pnl_roll_bp - tcost_bp
            
            row = {
                "open_ts": pos.open_ts, "close_ts": pos.close_ts, "exit_reason": "eoc", "mode": pos.mode,
                "pnl_net_bp": pnl_net_bp_final, "pnl_net_cash": pnl_net_cash_final, "tcost_bp": tcost_bp, "tcost_cash": tcost_cash,
                "scale_dv01": pos.scale_dv01,
                "tenor_i": pos.t_belly, "tenor_left": pos.t_left, "tenor_right": pos.t_right,
                # Granular PnL
                "pnl_price_cash": pos.pnl_price_cash,
                "pnl_carry_cash": pos.pnl_carry_cash,
                "pnl_roll_cash":  pos.pnl_roll_cash,
                "pnl_price_bp": pos.pnl_price_bp,
                "pnl_carry_bp": pos.pnl_carry_bp,
                "pnl_roll_bp":  pos.pnl_roll_bp,
                # Tie-Out Data
                "dir_belly": pos.dir_belly, "dir_left": pos.dir_left, "dir_right": pos.dir_right,
                "rate_belly_entry": pos.r_entry_belly, "rate_belly_exit": pos.r_curr_belly,
                "rate_left_entry":  pos.r_entry_left,  "rate_left_exit":  pos.r_curr_left,
                "rate_right_entry": pos.r_entry_right, "rate_right_exit": pos.r_curr_right,
            }
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
    print(f"[EXEC] Running FLY ENGINE on {len(months)} months...")
    pos, led, by = run_all(months, carry=True, force_close_end=True, hedge_df=trades, regime_signals=signals)
    out_dir = Path(cr.PATH_OUT)
    if not pos.empty: pos.to_parquet(out_dir / f"positions_ledger.parquet")
    if not led.empty: led.to_parquet(out_dir / f"marks_ledger.parquet")
    print(f"[DONE] Results saved to {out_dir}")
