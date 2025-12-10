import os, sys
import math
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

# All config access via module namespace
import cr_config as cr
from hybrid_filter import RegimeThresholds

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

def _get_funding_rate(snap_last: pd.DataFrame) -> float:
    """Finds proxy for funding rate (shortest available tenor)."""
    try:
        if snap_last.empty: return 0.0
        # Sort by tenor to find the shortest (e.g., 1M or 0.083y)
        sorted_snap = snap_last.sort_values("tenor_yrs")
        return float(sorted_snap.iloc[0]["rate"])
    except:
        return 0.0

def _get_z_at_tenor(snap_last: pd.DataFrame, tenor: float) -> float | None:
    """Simple helper to get Z-score at a specific tenor."""
    t = float(tenor)
    s = snap_last[["tenor_yrs", "z_comb"]].dropna()
    if s.empty: return None
    
    # Find nearest tenor in snapshot
    s = s.assign(_dist=(s["tenor_yrs"] - t).abs())
    row = s.loc[s["_dist"].idxmin()]
    
    # If nearest is too far (e.g. > 0.05y), return None
    if row["_dist"] > 0.05: return None
    return float(row["z_comb"])

def calc_trade_drift(tenor, direction, snap_last):
    """
    Estimates Daily Drift (Carry + Rolldown) in Basis Points of Price (BP) per unit of DV01.
    """
    # 1. Get Data
    row = snap_last.loc[snap_last["tenor_yrs"] == tenor]
    if row.empty: return -999.0
    
    # Use .iloc[0] to avoid FutureWarning
    rate_fixed = float(row["rate"].iloc[0])
    rate_float = _get_funding_rate(snap_last) 
    
    # 2. Rolldown Setup (Linear Interpolation for speed)
    # We roll down 1 day (1/360)
    day_fraction = 1.0 / 360.0
    t_roll = max(0.0, tenor - day_fraction)
    
    xp = snap_last["tenor_yrs"].values
    fp = snap_last["rate"].values
    
    # Ensure sorted for np.interp
    sort_idx = np.argsort(xp)
    xp_sorted = xp[sort_idx]
    fp_sorted = fp[sort_idx]
    
    rate_rolled = np.interp(t_roll, xp_sorted, fp_sorted)
    
    # 3. Calculate Components 
    raw_carry_bps = (rate_fixed - rate_float) * 100.0 * day_fraction
    raw_roll_bps = (rate_fixed - rate_rolled) * 100.0
    
    # 4. Apply Direction (+1 Receiver, -1 Payer)
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
    # Ensure UTC conversion is robust
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
        self.scale_dv01 = float(scale_dv01)
        self.meta = meta or {}
        self.initial_dv01 = self.scale_dv01
        
        self.dv01_i_entry = self.scale_dv01 * self.w_i
        self.dv01_j_entry = self.scale_dv01 * self.w_j
        self.dv01_i_curr, self.dv01_j_curr = self.dv01_i_entry, self.dv01_j_entry
        self.rem_tenor_i, self.rem_tenor_j = self.tenor_i_orig, self.tenor_j_orig
        
        # --- Notional Calculation for Carry ---
        # Raw division (no safety) as requested
        self.not_i_entry = self.dv01_i_entry / self.tenor_i
        self.not_j_entry = self.dv01_j_entry / self.tenor_j
        
        # State for Incremental Carry
        self.last_mark_ts = open_ts 
        
        # PnL Components (Cumulative)
        self.pnl_price_cash = 0.0
        self.pnl_carry_cash = 0.0
        self.pnl_roll_cash = 0.0
        self.pnl_cash = 0.0
        self.pnl_bp = 0.0
        
        # Breakdown Bps
        self.pnl_price_bp = 0.0
        self.pnl_carry_bp = 0.0
        self.pnl_roll_bp = 0.0

        self.tcost_bp, self.tcost_cash = 0.0, 0.0
        self.decisions_per_day = decisions_per_day
        self.age_decisions = 0
        self.bucket_i, self.bucket_j = assign_bucket(self.tenor_i), assign_bucket(self.tenor_j)
        self.last_zspread, self.last_z_dir, self.conv_pnl_proxy = self.entry_zspread, self.entry_z_dir, 0.0

    def _update_risk_decay(self, decision_ts):
        """Linearly decay DV01 based on Act/360 time passed (Price Risk only)."""
        if not isinstance(decision_ts, pd.Timestamp): return
        days = max(0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        yr_pass = days / 360.0
        self.rem_tenor_i = max(self.tenor_i_orig - yr_pass, 0.0)
        self.rem_tenor_j = max(self.tenor_j_orig - yr_pass, 0.0)
        
        # Avoid div by zero
        fi = self.rem_tenor_i / max(self.tenor_i_orig, 1e-6)
        fj = self.rem_tenor_j / max(self.tenor_j_orig, 1e-6)
        self.dv01_i_curr = self.dv01_i_entry * fi
        self.dv01_j_curr = self.dv01_j_entry * fj

    def mark(self, snap_last: pd.DataFrame, decision_ts: Optional[pd.Timestamp] = None):
        # 0. Update Decay
        if decision_ts:
            self._update_risk_decay(decision_ts)

        # 1. Get Market Data (Rates & Funding)
        ri = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "rate"])
        rj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "rate"])
        r_float = _get_funding_rate(snap_last)
        self.last_rate_i, self.last_rate_j = ri, rj

        # 2. Price PnL (Capital Gains due to Rate Change)
        # Formula: (Entry - Current) * 100 * DV01_Decayed
        pnl_price_i = (self.entry_rate_i - ri) * 100.0 * self.dv01_i_curr
        pnl_price_j = (self.entry_rate_j - rj) * 100.0 * self.dv01_j_curr
        self.pnl_price_cash = pnl_price_i + pnl_price_j

        # 3. Carry PnL (Net Interest Income) - Incremental Accumulation
        # Formula: (FixedEntry - FundingFloat) * 100 * Notional * Time
        if decision_ts and self.last_mark_ts:
            dt_days = max(0.0, (decision_ts.normalize() - self.last_mark_ts.normalize()).days)
            if dt_days > 0:
                inc_carry_i = (self.entry_rate_i - r_float) * 100.0 * self.not_i_entry * (dt_days / 360.0)
                inc_carry_j = (self.entry_rate_j - r_float) * 100.0 * self.not_j_entry * (dt_days / 360.0)
                self.pnl_carry_cash += (inc_carry_i + inc_carry_j)
            self.last_mark_ts = decision_ts

        # 4. Roll-Down PnL (Price gain due to sliding down yield curve)
        days_total = 0.0
        if decision_ts:
            days_total = max(0.0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        
        xp = snap_last["tenor_yrs"].values
        fp = snap_last["rate"].values
        sort_idx = np.argsort(xp)
        xp_sorted = xp[sort_idx]
        fp_sorted = fp[sort_idx]
        
        t_roll_i = max(0.0, self.tenor_i_orig - (days_total/360.0))
        y_roll_i = np.interp(t_roll_i, xp_sorted, fp_sorted)
        roll_gain_i = (ri - y_roll_i) * 100.0 * self.dv01_i_curr
        
        t_roll_j = max(0.0, self.tenor_j_orig - (days_total/360.0))
        y_roll_j = np.interp(t_roll_j, xp_sorted, fp_sorted)
        roll_gain_j = (rj - y_roll_j) * 100.0 * self.dv01_j_curr
        
        self.pnl_roll_cash = roll_gain_i + roll_gain_j

        # 5. Total Cash
        self.pnl_cash = self.pnl_price_cash + self.pnl_carry_cash + self.pnl_roll_cash
        
        # 6. Total Bps and Breakdown
        if self.scale_dv01 != 0.0 and np.isfinite(self.scale_dv01):
            self.pnl_bp = self.pnl_cash / self.scale_dv01
            self.pnl_price_bp = self.pnl_price_cash / self.scale_dv01
            self.pnl_carry_bp = self.pnl_carry_cash / self.scale_dv01
            self.pnl_roll_bp = self.pnl_roll_cash / self.scale_dv01
        else:
            self.pnl_bp = 0.0; self.pnl_price_bp = 0.0; self.pnl_carry_bp = 0.0; self.pnl_roll_bp = 0.0

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
# Filenames & I/O
# ------------------------
def _enhanced_in_path(yymm: str) -> Path:
    suffix = getattr(cr, "ENH_SUFFIX", "")
    name = cr.enh_fname(yymm) if hasattr(cr, "enh_fname") else f"{yymm}_enh{suffix}.parquet" if suffix else f"{yymm}_enh.parquet"
    return Path(getattr(cr, "PATH_ENH", ".")) / name

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
    regime_mask: Optional[pd.Series] = None
):
    import math
    try: np
    except NameError: import numpy as np

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

    # --- Configs ---
    MAX_HOLD_DECISIONS = int(round(float(getattr(cr, "MAX_HOLD_DAYS", 10)) * decisions_per_day))
    Z_ENTRY = float(getattr(cr, "Z_ENTRY", 0.75))
    Z_EXIT = float(getattr(cr, "Z_EXIT", 0.40))
    Z_STOP = float(getattr(cr, "Z_STOP", 3.00))
    
    EXEC_LEG_THRESHOLD = float(getattr(cr, "EXEC_LEG_TENOR_YEARS", 0.084))
    ALT_LEG_THRESHOLD  = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0))
    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    SHORT_END_EXTRA_Z = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))
    
    SWITCH_COST_BP = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10))
    
    # --- Re-enabled Drift Logic ---
    DRIFT_GATE = float(getattr(cr, "DRIFT_GATE_BPS", -100.0)) 
    DRIFT_W = float(getattr(cr, "DRIFT_WEIGHT", 0.0)) 

    open_positions = (open_positions or []) if carry_in else []

    # Filter hedges to month
    if hedges is not None and not hedges.empty:
        valid_decisions = df["decision_ts"].dropna().unique()
        hedges = hedges[hedges["decision_ts"].isin(valid_decisions)].copy()
    else:
        # If no hedges, we can't run overlay mode
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
        # 1) MARK POSITIONS & NATURAL EXITS
        # ============================================================
        still_open: list[PairPos] = []
        
        for pos in open_positions:
            # Mark
            zsp = pos.mark(snap_last, decision_ts=dts)
            
            # Logic: Reversion or Stop
            entry_z = pos.entry_zspread
            exit_flag = None

            if np.isfinite(zsp) and np.isfinite(entry_z):
                entry_dir = getattr(pos, "entry_z_dir", pos.dir_sign * entry_z)
                curr_dir = getattr(pos, "last_z_dir", pos.dir_sign * zsp)

                if np.isfinite(entry_dir) and np.isfinite(curr_dir):
                    sign_entry = np.sign(entry_dir)
                    sign_curr = np.sign(curr_dir)
                    same_side = (sign_entry != 0) and (sign_entry == sign_curr)
                    
                    # Profit Take (Reversion)
                    if same_side and (abs(curr_dir) <= abs(entry_dir)) and (abs(curr_dir) <= Z_EXIT):
                        exit_flag = "reversion"
                    
                    # Stop Loss (Divergence)
                    # Requires: Same side (still distorted), Moved away from zero, Disp change > Z_STOP
                    dz_dir = curr_dir - entry_dir
                    if same_side and (abs(curr_dir) >= abs(entry_dir)) and (abs(dz_dir) >= Z_STOP):
                        exit_flag = "stop"

            if exit_flag is None:
                if pos.age_decisions >= MAX_HOLD_DECISIONS:
                    exit_flag = "max_hold"

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
                # Trade Cost fixed at X bp of DV01
                tcost_bp = SWITCH_COST_BP
                tcost_cash = tcost_bp * pos.scale_dv01
                pos.tcost_bp = tcost_bp
                pos.tcost_cash = tcost_cash
                
                # --- UPDATED OUTPUT DICTIONARY FOR ANALYTICS ---
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": pos.close_ts, 
                    "exit_reason": pos.exit_reason,
                    "mode": "overlay", # Analytics script uses this filter
                    "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i, "w_j": pos.w_j,
                    "entry_rate_i": pos.entry_rate_i, "entry_rate_j": pos.entry_rate_j,
                    "close_rate_i": getattr(pos, "last_rate_i", np.nan),
                    "close_rate_j": getattr(pos, "last_rate_j", np.nan),
                    "scale_dv01": pos.scale_dv01,
                    "entry_zspread": pos.entry_zspread,
                    
                    # Net PnL
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, 
                    "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    
                    # T-Costs
                    "tcost_bp": tcost_bp,
                    "tcost_cash": tcost_cash,

                    # Attribution Breakdown (Required for Analytics Charts)
                    "pnl_price_bp": pos.pnl_price_bp,
                    "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp": pos.pnl_roll_bp,
                    "pnl_price_cash": pos.pnl_price_cash,
                    "pnl_carry_cash": pos.pnl_carry_cash,
                    "pnl_roll_cash": pos.pnl_roll_cash,

                    # Stats
                    "days_held_equiv": pos.age_decisions / max(1, decisions_per_day),
                    "trade_id": pos.meta.get("trade_id"),
                    "side": pos.meta.get("side"),
                })
            else:
                still_open.append(pos)
        
        open_positions = still_open
        
        # ============================================================
        # 2) GATE ENTRIES
        # ============================================================
        # Regime Gate
        regime_ok = True
        if regime_mask is not None and dts in regime_mask.index:
            regime_ok = bool(regime_mask.at[dts])
        
        if not regime_ok: continue
        
        # ============================================================
        # 3) SCAN HEDGES (OVERLAY LOGIC)
        # ============================================================
        if hedges.empty: continue
        h_here = hedges[hedges["decision_ts"] == dts]
        if h_here.empty: continue
        
        snap_srt = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
        
        for _, h in h_here.iterrows():
            t_trade = float(h["tenor_yrs"])
            if t_trade < EXEC_LEG_THRESHOLD: continue

            # Determine direction: 
            # If Client Pays -> We Rec -> Side +1. We replace Rec T with Rec Alt.
            # Spread = (Alt - Exec) if Rec, (Exec - Alt) if Pay.
            side_s = 1.0 if str(h["side"]).upper() == "CRCV" else -1.0
            
            exec_z = _get_z_at_tenor(snap_srt, t_trade)
            if exec_z is None: continue
            
            # Find exact row for execution leg
            exec_row = snap_srt.iloc[(snap_srt["tenor_yrs"] - t_trade).abs().idxmin()]
            exec_tenor = float(exec_row["tenor_yrs"])
            exec_bucket = assign_bucket(exec_tenor)

            best_c_row, best_score = None, -999.0

            # --- Drift Baseline ---
            drift_exec = calc_trade_drift(exec_tenor, side_s, snap_srt)

            # Scan Curve for Alternate
            for _, alt in snap_srt.iterrows():
                alt_tenor = float(alt["tenor_yrs"])
                
                # 1. Tenor constraints
                if alt_tenor < ALT_LEG_THRESHOLD: continue
                if alt_tenor == exec_tenor: continue
                if abs(alt_tenor - exec_tenor) < MIN_SEP_YEARS: continue
                
                # 2. Bucket constraints (No Short vs Long)
                alt_bucket = assign_bucket(alt_tenor)
                if alt_bucket == "short" and exec_bucket == "long": continue
                if exec_bucket == "short" and alt_bucket == "long": continue
                
                # 3. Z-Score Opportunity
                z_alt = _to_float(alt["z_comb"])
                # "Dispersion" = Gain from switch
                disp = (z_alt - exec_z) if side_s > 0 else (exec_z - z_alt)
                
                # 4. Entry Thresholds
                thresh = Z_ENTRY
                if alt_bucket == "short" or exec_bucket == "short":
                    thresh += SHORT_END_EXTRA_Z
                
                if disp < thresh: continue
                
                # 5. Drift Logic (Restored)
                drift_alt = calc_trade_drift(alt_tenor, side_s, snap_srt)
                net_drift_bps = drift_alt - drift_exec 
                
                # Gate on raw Net Drift
                if net_drift_bps < DRIFT_GATE: continue 
                
                # Normalize drift by distance (cost of extension)
                dist_years = abs(alt_tenor - exec_tenor)
                # No safety here, MIN_SEP_YEARS guarantees dist_years > 0
                norm_drift_bps = net_drift_bps / dist_years 
                
                # Score = Z-Spread + Weighted Normalized Drift
                score = disp + (norm_drift_bps * DRIFT_W)
                
                if score > best_score:
                    best_score = score
                    best_c_row = alt

            # Execute Best Candidate
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
                              entry_rate_i=rate_i, entry_rate_j=rate_j)
                open_positions.append(pos)
                ledger_rows.append({"decision_ts": dts, "event": "open"})

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions

def run_all(yymms, *, decision_freq=None, carry=True, force_close_end=False, hedge_df=None, regime_mask=None, hybrid_signals=None):
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
            
        p, l, b, open_pos = run_month(yymm, decision_freq=decision_freq, open_positions=open_pos, carry_in=carry, hedges=h_mon, regime_mask=regime_mask)
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
                "mode": "overlay",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "pnl_net_bp": pos.pnl_bp - tcost_bp, "pnl_net_cash": pos.pnl_cash - tcost_cash,
                "tcost_bp": tcost_bp, "tcost_cash": tcost_cash,
                "pnl_price_bp": pos.pnl_price_bp, "pnl_carry_bp": pos.pnl_carry_bp, "pnl_roll_bp": pos.pnl_roll_bp,
                "pnl_price_cash": pos.pnl_price_cash, "pnl_carry_cash": pos.pnl_carry_cash, "pnl_roll_cash": pos.pnl_roll_cash,
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
    regime_mask = hf.regime_mask_from_signals(signals)

    print(f"[EXEC] Running Overlay on {len(months)} months.")
    pos, led, by = run_all(months, carry=True, force_close_end=True, hedge_df=trades, regime_mask=regime_mask)

    out_dir = Path(cr.PATH_OUT)
    suffix = getattr(cr, "OUT_SUFFIX", "")
    if not pos.empty: pos.to_parquet(out_dir / f"positions_ledger{suffix}.parquet")
    if not led.empty: led.to_parquet(out_dir / f"marks_ledger{suffix}.parquet")
    if not by.empty: by.to_parquet(out_dir / f"pnl_by_bucket{suffix}.parquet")
    print(f"[DONE] Results saved to {out_dir}")





import os, sys
import math
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

# All config access via module namespace
import cr_config as cr
from hybrid_filter import RegimeThresholds

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

def _get_funding_rate(snap_last: pd.DataFrame) -> float:
    """Finds proxy for funding rate (shortest available tenor)."""
    try:
        if snap_last.empty: return 0.0
        # Sort by tenor to find the shortest (e.g., 1M or 0.083y)
        sorted_snap = snap_last.sort_values("tenor_yrs")
        return float(sorted_snap.iloc[0]["rate"])
    except:
        return 0.0

def _get_z_at_tenor(snap_last: pd.DataFrame, tenor: float) -> float | None:
    """Simple helper to get Z-score at a specific tenor."""
    t = float(tenor)
    s = snap_last[["tenor_yrs", "z_comb"]].dropna()
    if s.empty: return None
    
    # Find nearest tenor in snapshot
    s = s.assign(_dist=(s["tenor_yrs"] - t).abs())
    row = s.loc[s["_dist"].idxmin()]
    
    # If nearest is too far (e.g. > 0.05y), return None
    if row["_dist"] > 0.05: return None
    return float(row["z_comb"])

def calc_trade_drift(tenor, direction, snap_last):
    """
    Estimates Daily Drift (Carry + Rolldown) in Basis Points of Price (BP) per unit of DV01.
    """
    # 1. Get Data
    row = snap_last.loc[snap_last["tenor_yrs"] == tenor]
    if row.empty: return -999.0
    
    rate_fixed = float(row["rate"])
    rate_float = _get_funding_rate(snap_last) 
    
    # 2. Rolldown Setup (Linear Interpolation for speed)
    # We roll down 1 day (1/360)
    day_fraction = 1.0 / 360.0
    t_roll = max(0.0, tenor - day_fraction)
    
    xp = snap_last["tenor_yrs"].values
    fp = snap_last["rate"].values
    
    rate_rolled = np.interp(t_roll, xp, fp)
    
    # 3. Calculate Components 
    raw_carry_bps = (rate_fixed - rate_float) * 100.0 * day_fraction
    raw_roll_bps = (rate_fixed - rate_rolled) * 100.0
    
    # 4. Apply Direction (+1 Receiver, -1 Payer)
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
    # Ensure UTC conversion is robust
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
        self.scale_dv01 = float(scale_dv01)
        self.meta = meta or {}
        self.initial_dv01 = self.scale_dv01
        
        self.dv01_i_entry = self.scale_dv01 * self.w_i
        self.dv01_j_entry = self.scale_dv01 * self.w_j
        self.dv01_i_curr, self.dv01_j_curr = self.dv01_i_entry, self.dv01_j_entry
        self.rem_tenor_i, self.rem_tenor_j = self.tenor_i_orig, self.tenor_j_orig
        
        # --- Notional Calculation for Carry ---
        # NO SAFETY as requested: assumed inputs are valid
        self.not_i_entry = self.dv01_i_entry / self.tenor_i
        self.not_j_entry = self.dv01_j_entry / self.tenor_j
        
        # State for Incremental Carry
        self.last_mark_ts = open_ts 
        
        # PnL Components (Cumulative)
        self.pnl_price_cash = 0.0
        self.pnl_carry_cash = 0.0
        self.pnl_roll_cash = 0.0
        self.pnl_cash = 0.0
        self.pnl_bp = 0.0
        
        # Breakdown Bps
        self.pnl_price_bp = 0.0
        self.pnl_carry_bp = 0.0
        self.pnl_roll_bp = 0.0

        self.tcost_bp, self.tcost_cash = 0.0, 0.0
        self.decisions_per_day = decisions_per_day
        self.age_decisions = 0
        self.bucket_i, self.bucket_j = assign_bucket(self.tenor_i), assign_bucket(self.tenor_j)
        self.last_zspread, self.last_z_dir, self.conv_pnl_proxy = self.entry_zspread, self.entry_z_dir, 0.0

    def _update_risk_decay(self, decision_ts):
        """Linearly decay DV01 based on Act/360 time passed (Price Risk only)."""
        if not isinstance(decision_ts, pd.Timestamp): return
        days = max(0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        yr_pass = days / 360.0
        self.rem_tenor_i = max(self.tenor_i_orig - yr_pass, 0.0)
        self.rem_tenor_j = max(self.tenor_j_orig - yr_pass, 0.0)
        
        # Avoid div by zero
        fi = self.rem_tenor_i / max(self.tenor_i_orig, 1e-6)
        fj = self.rem_tenor_j / max(self.tenor_j_orig, 1e-6)
        self.dv01_i_curr = self.dv01_i_entry * fi
        self.dv01_j_curr = self.dv01_j_entry * fj

    def mark(self, snap_last: pd.DataFrame, decision_ts: Optional[pd.Timestamp] = None):
        # 0. Update Decay
        if decision_ts:
            self._update_risk_decay(decision_ts)

        # 1. Get Market Data (Rates & Funding)
        ri = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "rate"])
        rj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "rate"])
        r_float = _get_funding_rate(snap_last)
        self.last_rate_i, self.last_rate_j = ri, rj

        # 2. Price PnL (Capital Gains due to Rate Change)
        # Formula: (Entry - Current) * 100 * DV01_Decayed
        pnl_price_i = (self.entry_rate_i - ri) * 100.0 * self.dv01_i_curr
        pnl_price_j = (self.entry_rate_j - rj) * 100.0 * self.dv01_j_curr
        self.pnl_price_cash = pnl_price_i + pnl_price_j

        # 3. Carry PnL (Net Interest Income) - Incremental Accumulation
        # Formula: (FixedEntry - FundingFloat) * 100 * Notional * Time
        if decision_ts and self.last_mark_ts:
            dt_days = max(0.0, (decision_ts.normalize() - self.last_mark_ts.normalize()).days)
            if dt_days > 0:
                inc_carry_i = (self.entry_rate_i - r_float) * 100.0 * self.not_i_entry * (dt_days / 360.0)
                inc_carry_j = (self.entry_rate_j - r_float) * 100.0 * self.not_j_entry * (dt_days / 360.0)
                self.pnl_carry_cash += (inc_carry_i + inc_carry_j)
            self.last_mark_ts = decision_ts

        # 4. Roll-Down PnL (Price gain due to sliding down yield curve)
        days_total = 0.0
        if decision_ts:
            days_total = max(0.0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        
        xp = snap_last["tenor_yrs"].values
        fp = snap_last["rate"].values
        sort_idx = np.argsort(xp)
        xp_sorted = xp[sort_idx]
        fp_sorted = fp[sort_idx]
        
        t_roll_i = max(0.0, self.tenor_i_orig - (days_total/360.0))
        y_roll_i = np.interp(t_roll_i, xp_sorted, fp_sorted)
        roll_gain_i = (ri - y_roll_i) * 100.0 * self.dv01_i_curr
        
        t_roll_j = max(0.0, self.tenor_j_orig - (days_total/360.0))
        y_roll_j = np.interp(t_roll_j, xp_sorted, fp_sorted)
        roll_gain_j = (rj - y_roll_j) * 100.0 * self.dv01_j_curr
        
        self.pnl_roll_cash = roll_gain_i + roll_gain_j

        # 5. Total Cash
        self.pnl_cash = self.pnl_price_cash + self.pnl_carry_cash + self.pnl_roll_cash
        
        # 6. Total Bps and Breakdown
        if self.scale_dv01 != 0.0 and np.isfinite(self.scale_dv01):
            self.pnl_bp = self.pnl_cash / self.scale_dv01
            self.pnl_price_bp = self.pnl_price_cash / self.scale_dv01
            self.pnl_carry_bp = self.pnl_carry_cash / self.scale_dv01
            self.pnl_roll_bp = self.pnl_roll_cash / self.scale_dv01
        else:
            self.pnl_bp = 0.0; self.pnl_price_bp = 0.0; self.pnl_carry_bp = 0.0; self.pnl_roll_bp = 0.0

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
# Filenames & I/O
# ------------------------
def _enhanced_in_path(yymm: str) -> Path:
    suffix = getattr(cr, "ENH_SUFFIX", "")
    name = cr.enh_fname(yymm) if hasattr(cr, "enh_fname") else f"{yymm}_enh{suffix}.parquet" if suffix else f"{yymm}_enh.parquet"
    return Path(getattr(cr, "PATH_ENH", ".")) / name

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
    regime_mask: Optional[pd.Series] = None
):
    import math
    try: np
    except NameError: import numpy as np

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

    # --- Configs ---
    MAX_HOLD_DECISIONS = int(round(float(getattr(cr, "MAX_HOLD_DAYS", 10)) * decisions_per_day))
    Z_ENTRY = float(getattr(cr, "Z_ENTRY", 0.75))
    Z_EXIT = float(getattr(cr, "Z_EXIT", 0.40))
    Z_STOP = float(getattr(cr, "Z_STOP", 3.00))
    
    EXEC_LEG_THRESHOLD = float(getattr(cr, "EXEC_LEG_TENOR_YEARS", 0.084))
    ALT_LEG_THRESHOLD  = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0))
    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    SHORT_END_EXTRA_Z = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))
    
    SWITCH_COST_BP = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10))
    
    # --- Re-enabled Drift Logic ---
    DRIFT_GATE = float(getattr(cr, "DRIFT_GATE_BPS", -100.0)) 
    DRIFT_W = float(getattr(cr, "DRIFT_WEIGHT", 0.0)) 

    open_positions = (open_positions or []) if carry_in else []

    # Filter hedges to month
    if hedges is not None and not hedges.empty:
        valid_decisions = df["decision_ts"].dropna().unique()
        hedges = hedges[hedges["decision_ts"].isin(valid_decisions)].copy()
    else:
        # If no hedges, we can't run overlay mode
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
        # 1) MARK POSITIONS & NATURAL EXITS
        # ============================================================
        still_open: list[PairPos] = []
        
        for pos in open_positions:
            # Mark
            zsp = pos.mark(snap_last, decision_ts=dts)
            
            # Logic: Reversion or Stop
            entry_z = pos.entry_zspread
            exit_flag = None

            if np.isfinite(zsp) and np.isfinite(entry_z):
                entry_dir = getattr(pos, "entry_z_dir", pos.dir_sign * entry_z)
                curr_dir = getattr(pos, "last_z_dir", pos.dir_sign * zsp)

                if np.isfinite(entry_dir) and np.isfinite(curr_dir):
                    sign_entry = np.sign(entry_dir)
                    sign_curr = np.sign(curr_dir)
                    same_side = (sign_entry != 0) and (sign_entry == sign_curr)
                    
                    # Profit Take (Reversion)
                    if same_side and (abs(curr_dir) <= abs(entry_dir)) and (abs(curr_dir) <= Z_EXIT):
                        exit_flag = "reversion"
                    
                    # Stop Loss (Divergence)
                    # Requires: Same side (still distorted), Moved away from zero, Disp change > Z_STOP
                    dz_dir = curr_dir - entry_dir
                    if same_side and (abs(curr_dir) >= abs(entry_dir)) and (abs(dz_dir) >= Z_STOP):
                        exit_flag = "stop"

            if exit_flag is None:
                if pos.age_decisions >= MAX_HOLD_DECISIONS:
                    exit_flag = "max_hold"

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
                # Trade Cost fixed at X bp of DV01
                tcost_bp = SWITCH_COST_BP
                tcost_cash = tcost_bp * pos.scale_dv01
                pos.tcost_bp = tcost_bp
                pos.tcost_cash = tcost_cash
                
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": pos.close_ts, 
                    "exit_reason": pos.exit_reason,
                    "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i, "w_j": pos.w_j,
                    "entry_rate_i": pos.entry_rate_i, "entry_rate_j": pos.entry_rate_j,
                    "close_rate_i": getattr(pos, "last_rate_i", np.nan),
                    "close_rate_j": getattr(pos, "last_rate_j", np.nan),
                    "scale_dv01": pos.scale_dv01,
                    "entry_zspread": pos.entry_zspread,
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, 
                    "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "tcost_bp": tcost_bp,
                    "tcost_cash": tcost_cash,
                    "days_held_equiv": pos.age_decisions / max(1, decisions_per_day),
                    "trade_id": pos.meta.get("trade_id"),
                    "side": pos.meta.get("side"),
                })
            else:
                still_open.append(pos)
        
        open_positions = still_open
        
        # ============================================================
        # 2) GATE ENTRIES
        # ============================================================
        # Regime Gate
        regime_ok = True
        if regime_mask is not None and dts in regime_mask.index:
            regime_ok = bool(regime_mask.at[dts])
        
        if not regime_ok: continue
        
        # ============================================================
        # 3) SCAN HEDGES (OVERLAY LOGIC)
        # ============================================================
        if hedges.empty: continue
        h_here = hedges[hedges["decision_ts"] == dts]
        if h_here.empty: continue
        
        snap_srt = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
        
        for _, h in h_here.iterrows():
            t_trade = float(h["tenor_yrs"])
            if t_trade < EXEC_LEG_THRESHOLD: continue

            # Determine direction: 
            # If Client Pays -> We Rec -> Side +1. We replace Rec T with Rec Alt.
            # Spread = (Alt - Exec) if Rec, (Exec - Alt) if Pay.
            side_s = 1.0 if str(h["side"]).upper() == "CRCV" else -1.0
            
            exec_z = _get_z_at_tenor(snap_srt, t_trade)
            if exec_z is None: continue
            
            # Find exact row for execution leg
            exec_row = snap_srt.iloc[(snap_srt["tenor_yrs"] - t_trade).abs().idxmin()]
            exec_tenor = float(exec_row["tenor_yrs"])
            exec_bucket = assign_bucket(exec_tenor)

            best_c_row, best_score = None, -999.0

            # --- Drift Baseline ---
            drift_exec = calc_trade_drift(exec_tenor, side_s, snap_srt)

            # Scan Curve for Alternate
            for _, alt in snap_srt.iterrows():
                alt_tenor = float(alt["tenor_yrs"])
                
                # 1. Tenor constraints
                if alt_tenor < ALT_LEG_THRESHOLD: continue
                if alt_tenor == exec_tenor: continue
                if abs(alt_tenor - exec_tenor) < MIN_SEP_YEARS: continue
                
                # 2. Bucket constraints (No Short vs Long)
                alt_bucket = assign_bucket(alt_tenor)
                if alt_bucket == "short" and exec_bucket == "long": continue
                if exec_bucket == "short" and alt_bucket == "long": continue
                
                # 3. Z-Score Opportunity
                z_alt = _to_float(alt["z_comb"])
                # "Dispersion" = Gain from switch
                disp = (z_alt - exec_z) if side_s > 0 else (exec_z - z_alt)
                
                # 4. Entry Thresholds
                thresh = Z_ENTRY
                if alt_bucket == "short" or exec_bucket == "short":
                    thresh += SHORT_END_EXTRA_Z
                
                if disp < thresh: continue
                
                # 5. Drift Logic (Restored)
                drift_alt = calc_trade_drift(alt_tenor, side_s, snap_srt)
                net_drift_bps = drift_alt - drift_exec 
                
                # Gate on raw Net Drift
                if net_drift_bps < DRIFT_GATE: continue 
                
                # Normalize drift by distance (cost of extension)
                dist_years = abs(alt_tenor - exec_tenor)
                # No safety here, MIN_SEP_YEARS guarantees dist_years > 0
                norm_drift_bps = net_drift_bps / dist_years 
                
                # Score = Z-Spread + Weighted Normalized Drift
                score = disp + (norm_drift_bps * DRIFT_W)
                
                if score > best_score:
                    best_score = score
                    best_c_row = alt

            # Execute Best Candidate
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
                              entry_rate_i=rate_i, entry_rate_j=rate_j)
                open_positions.append(pos)
                ledger_rows.append({"decision_ts": dts, "event": "open"})

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions

def run_all(yymms, *, decision_freq=None, carry=True, force_close_end=False, hedge_df=None, regime_mask=None, hybrid_signals=None):
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
            
        p, l, b, open_pos = run_month(yymm, decision_freq=decision_freq, open_positions=open_pos, carry_in=carry, hedges=h_mon, regime_mask=regime_mask)
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
    regime_mask = hf.regime_mask_from_signals(signals)

    print(f"[EXEC] Running Overlay on {len(months)} months.")
    pos, led, by = run_all(months, carry=True, force_close_end=True, hedge_df=trades, regime_mask=regime_mask)

    out_dir = Path(cr.PATH_OUT)
    suffix = getattr(cr, "OUT_SUFFIX", "")
    if not pos.empty: pos.to_parquet(out_dir / f"positions_ledger{suffix}.parquet")
    if not led.empty: led.to_parquet(out_dir / f"marks_ledger{suffix}.parquet")
    if not by.empty: by.to_parquet(out_dir / f"pnl_by_bucket{suffix}.parquet")
    print(f"[DONE] Results saved to {out_dir}")




import os, sys
import math
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

# All config access via module namespace
import cr_config as cr
# Only importing RegimeThresholds as ShockConfig is removed
from hybrid_filter import RegimeThresholds

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

def _get_funding_rate(snap_last: pd.DataFrame) -> float:
    """Finds proxy for funding rate (shortest available tenor)."""
    try:
        if snap_last.empty: return 0.0
        # Sort by tenor to find the shortest (e.g., 1M or 0.083y)
        sorted_snap = snap_last.sort_values("tenor_yrs")
        return float(sorted_snap.iloc[0]["rate"])
    except:
        return 0.0

def _get_z_at_tenor(snap_last: pd.DataFrame, tenor: float) -> float | None:
    """Simple helper to get Z-score at a specific tenor."""
    t = float(tenor)
    # Filter for reasonable proximity
    s = snap_last[["tenor_yrs", "z_comb"]].dropna()
    if s.empty: return None
    
    # Find nearest tenor in snapshot
    s = s.assign(_dist=(s["tenor_yrs"] - t).abs())
    row = s.loc[s["_dist"].idxmin()]
    
    # If nearest is too far (e.g. > 0.05y), return None
    if row["_dist"] > 0.05: return None
    return float(row["z_comb"])

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
    # Ensure UTC conversion is robust
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
        self.scale_dv01 = float(scale_dv01)
        self.meta = meta or {}
        self.initial_dv01 = self.scale_dv01
        
        self.dv01_i_entry = self.scale_dv01 * self.w_i
        self.dv01_j_entry = self.scale_dv01 * self.w_j
        self.dv01_i_curr, self.dv01_j_curr = self.dv01_i_entry, self.dv01_j_entry
        self.rem_tenor_i, self.rem_tenor_j = self.tenor_i_orig, self.tenor_j_orig
        
        # --- Notional Calculation for Carry ---
        # Avoiding division by zero for very short tenors
        self.not_i_entry = self.dv01_i_entry / max(0.1, self.tenor_i)
        self.not_j_entry = self.dv01_j_entry / max(0.1, self.tenor_j)
        
        # State for Incremental Carry
        self.last_mark_ts = open_ts 
        
        # PnL Components (Cumulative)
        self.pnl_price_cash = 0.0
        self.pnl_carry_cash = 0.0
        self.pnl_roll_cash = 0.0
        self.pnl_cash = 0.0
        self.pnl_bp = 0.0
        
        # Breakdown Bps
        self.pnl_price_bp = 0.0
        self.pnl_carry_bp = 0.0
        self.pnl_roll_bp = 0.0

        self.tcost_bp, self.tcost_cash = 0.0, 0.0
        self.decisions_per_day = decisions_per_day
        self.age_decisions = 0
        self.bucket_i, self.bucket_j = assign_bucket(self.tenor_i), assign_bucket(self.tenor_j)
        self.last_zspread, self.last_z_dir, self.conv_pnl_proxy = self.entry_zspread, self.entry_z_dir, 0.0

    def _update_risk_decay(self, decision_ts):
        """Linearly decay DV01 based on Act/360 time passed (Price Risk only)."""
        if not isinstance(decision_ts, pd.Timestamp): return
        days = max(0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        yr_pass = days / 360.0
        self.rem_tenor_i = max(self.tenor_i_orig - yr_pass, 0.0)
        self.rem_tenor_j = max(self.tenor_j_orig - yr_pass, 0.0)
        
        # Avoid div by zero
        fi = self.rem_tenor_i / max(self.tenor_i_orig, 1e-6)
        fj = self.rem_tenor_j / max(self.tenor_j_orig, 1e-6)
        self.dv01_i_curr = self.dv01_i_entry * fi
        self.dv01_j_curr = self.dv01_j_entry * fj

    def mark(self, snap_last: pd.DataFrame, decision_ts: Optional[pd.Timestamp] = None):
        # 0. Update Decay
        if decision_ts:
            self._update_risk_decay(decision_ts)

        # 1. Get Market Data (Rates & Funding)
        ri = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "rate"])
        rj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "rate"])
        r_float = _get_funding_rate(snap_last)
        self.last_rate_i, self.last_rate_j = ri, rj

        # 2. Price PnL (Capital Gains due to Rate Change)
        # Formula: (Entry - Current) * 100 * DV01_Decayed
        pnl_price_i = (self.entry_rate_i - ri) * 100.0 * self.dv01_i_curr
        pnl_price_j = (self.entry_rate_j - rj) * 100.0 * self.dv01_j_curr
        self.pnl_price_cash = pnl_price_i + pnl_price_j

        # 3. Carry PnL (Net Interest Income) - Incremental Accumulation
        # Formula: (FixedEntry - FundingFloat) * 100 * Notional * Time
        if decision_ts and self.last_mark_ts:
            dt_days = max(0.0, (decision_ts.normalize() - self.last_mark_ts.normalize()).days)
            if dt_days > 0:
                inc_carry_i = (self.entry_rate_i - r_float) * 100.0 * self.not_i_entry * (dt_days / 360.0)
                inc_carry_j = (self.entry_rate_j - r_float) * 100.0 * self.not_j_entry * (dt_days / 360.0)
                self.pnl_carry_cash += (inc_carry_i + inc_carry_j)
            self.last_mark_ts = decision_ts

        # 4. Roll-Down PnL (Price gain due to sliding down yield curve)
        days_total = 0.0
        if decision_ts:
            days_total = max(0.0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        
        xp = snap_last["tenor_yrs"].values
        fp = snap_last["rate"].values
        sort_idx = np.argsort(xp)
        xp_sorted = xp[sort_idx]
        fp_sorted = fp[sort_idx]
        
        t_roll_i = max(0.0, self.tenor_i_orig - (days_total/360.0))
        y_roll_i = np.interp(t_roll_i, xp_sorted, fp_sorted)
        roll_gain_i = (ri - y_roll_i) * 100.0 * self.dv01_i_curr
        
        t_roll_j = max(0.0, self.tenor_j_orig - (days_total/360.0))
        y_roll_j = np.interp(t_roll_j, xp_sorted, fp_sorted)
        roll_gain_j = (rj - y_roll_j) * 100.0 * self.dv01_j_curr
        
        self.pnl_roll_cash = roll_gain_i + roll_gain_j

        # 5. Total Cash
        self.pnl_cash = self.pnl_price_cash + self.pnl_carry_cash + self.pnl_roll_cash
        
        # 6. Total Bps and Breakdown
        if self.scale_dv01 != 0.0 and np.isfinite(self.scale_dv01):
            self.pnl_bp = self.pnl_cash / self.scale_dv01
            self.pnl_price_bp = self.pnl_price_cash / self.scale_dv01
            self.pnl_carry_bp = self.pnl_carry_cash / self.scale_dv01
            self.pnl_roll_bp = self.pnl_roll_cash / self.scale_dv01
        else:
            self.pnl_bp = 0.0; self.pnl_price_bp = 0.0; self.pnl_carry_bp = 0.0; self.pnl_roll_bp = 0.0

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
# Filenames & I/O
# ------------------------
def _enhanced_in_path(yymm: str) -> Path:
    suffix = getattr(cr, "ENH_SUFFIX", "")
    name = cr.enh_fname(yymm) if hasattr(cr, "enh_fname") else f"{yymm}_enh{suffix}.parquet" if suffix else f"{yymm}_enh.parquet"
    return Path(getattr(cr, "PATH_ENH", ".")) / name

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
    regime_mask: Optional[pd.Series] = None
):
    import math
    try: np
    except NameError: import numpy as np

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

    # --- Configs ---
    MAX_HOLD_DECISIONS = int(round(float(getattr(cr, "MAX_HOLD_DAYS", 10)) * decisions_per_day))
    Z_ENTRY = float(getattr(cr, "Z_ENTRY", 0.75))
    Z_EXIT = float(getattr(cr, "Z_EXIT", 0.40))
    Z_STOP = float(getattr(cr, "Z_STOP", 3.00))
    BPS_PNL_STOP = float(getattr(cr, "BPS_PNL_STOP", 0.0) or 0.0)
    
    EXEC_LEG_THRESHOLD = float(getattr(cr, "EXEC_LEG_TENOR_YEARS", 0.084))
    ALT_LEG_THRESHOLD  = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0))
    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    SHORT_END_EXTRA_Z = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))
    
    SWITCH_COST_BP = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10))

    open_positions = (open_positions or []) if carry_in else []

    # Filter hedges to month
    if hedges is not None and not hedges.empty:
        valid_decisions = df["decision_ts"].dropna().unique()
        hedges = hedges[hedges["decision_ts"].isin(valid_decisions)].copy()
    else:
        # If no hedges, we can't run overlay mode
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
        # 1) MARK POSITIONS & NATURAL EXITS
        # ============================================================
        still_open: list[PairPos] = []
        
        for pos in open_positions:
            # Mark
            zsp = pos.mark(snap_last, decision_ts=dts)
            
            # Logic: Reversion or Stop
            entry_z = pos.entry_zspread
            exit_flag = None

            if np.isfinite(zsp) and np.isfinite(entry_z):
                entry_dir = getattr(pos, "entry_z_dir", pos.dir_sign * entry_z)
                curr_dir = getattr(pos, "last_z_dir", pos.dir_sign * zsp)

                if np.isfinite(entry_dir) and np.isfinite(curr_dir):
                    sign_entry = np.sign(entry_dir)
                    sign_curr = np.sign(curr_dir)
                    same_side = (sign_entry != 0) and (sign_entry == sign_curr)
                    
                    # Profit Take (Reversion)
                    if same_side and (abs(curr_dir) <= abs(entry_dir)) and (abs(curr_dir) <= Z_EXIT):
                        exit_flag = "reversion"
                    
                    # Stop Loss (Divergence)
                    # Requires: Same side (still distorted), Moved away from zero, Disp change > Z_STOP
                    dz_dir = curr_dir - entry_dir
                    if same_side and (abs(curr_dir) >= abs(entry_dir)) and (abs(dz_dir) >= Z_STOP):
                        exit_flag = "stop"

            if exit_flag is None and BPS_PNL_STOP > 0.0:
                if np.isfinite(pos.pnl_bp) and pos.pnl_bp <= -BPS_PNL_STOP:
                    exit_flag = "pnl_stop"

            if exit_flag is None:
                if pos.age_decisions >= MAX_HOLD_DECISIONS:
                    exit_flag = "max_hold"

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
                # Trade Cost fixed at X bp of DV01
                tcost_bp = SWITCH_COST_BP
                tcost_cash = tcost_bp * pos.scale_dv01
                pos.tcost_bp = tcost_bp
                pos.tcost_cash = tcost_cash
                
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": pos.close_ts, 
                    "exit_reason": pos.exit_reason,
                    "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i, "w_j": pos.w_j,
                    "entry_rate_i": pos.entry_rate_i, "entry_rate_j": pos.entry_rate_j,
                    "close_rate_i": getattr(pos, "last_rate_i", np.nan),
                    "close_rate_j": getattr(pos, "last_rate_j", np.nan),
                    "scale_dv01": pos.scale_dv01,
                    "entry_zspread": pos.entry_zspread,
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, 
                    "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "days_held_equiv": pos.age_decisions / max(1, decisions_per_day),
                    "trade_id": pos.meta.get("trade_id"),
                    "side": pos.meta.get("side"),
                })
            else:
                still_open.append(pos)
        
        open_positions = still_open
        
        # ============================================================
        # 2) GATE ENTRIES
        # ============================================================
        # Regime Gate
        regime_ok = True
        if regime_mask is not None and dts in regime_mask.index:
            regime_ok = bool(regime_mask.at[dts])
        
        if not regime_ok: continue
        
        # ============================================================
        # 3) SCAN HEDGES (OVERLAY LOGIC)
        # ============================================================
        if hedges.empty: continue
        h_here = hedges[hedges["decision_ts"] == dts]
        if h_here.empty: continue
        
        snap_srt = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
        
        for _, h in h_here.iterrows():
            t_trade = float(h["tenor_yrs"])
            if t_trade < EXEC_LEG_THRESHOLD: continue

            # Determine direction: 
            # If Client Pays -> We Rec -> Side +1. We replace Rec T with Rec Alt.
            # Spread = (Alt - Exec) if Rec, (Exec - Alt) if Pay.
            side_s = 1.0 if str(h["side"]).upper() == "CRCV" else -1.0
            
            exec_z = _get_z_at_tenor(snap_srt, t_trade)
            if exec_z is None: continue
            
            # Find exact row for execution leg
            exec_row = snap_srt.iloc[(snap_srt["tenor_yrs"] - t_trade).abs().idxmin()]
            exec_tenor = float(exec_row["tenor_yrs"])
            exec_bucket = assign_bucket(exec_tenor)

            best_c_row, best_disp = None, -999.0

            # Scan Curve for Alternate
            for _, alt in snap_srt.iterrows():
                alt_tenor = float(alt["tenor_yrs"])
                
                # 1. Tenor constraints
                if alt_tenor < ALT_LEG_THRESHOLD: continue
                if alt_tenor == exec_tenor: continue
                if abs(alt_tenor - exec_tenor) < MIN_SEP_YEARS: continue
                
                # 2. Bucket constraints (No Short vs Long)
                alt_bucket = assign_bucket(alt_tenor)
                if alt_bucket == "short" and exec_bucket == "long": continue
                if exec_bucket == "short" and alt_bucket == "long": continue
                
                # 3. Z-Score Opportunity
                z_alt = _to_float(alt["z_comb"])
                # "Dispersion" = Gain from switch
                disp = (z_alt - exec_z) if side_s > 0 else (exec_z - z_alt)
                
                # 4. Entry Thresholds
                thresh = Z_ENTRY
                if alt_bucket == "short" or exec_bucket == "short":
                    thresh += SHORT_END_EXTRA_Z
                
                if disp < thresh: continue
                
                # Optimization: Pick max dispersion (Mean Reversion only)
                if disp > best_disp:
                    best_disp = disp
                    best_c_row = alt

            # Execute Best Candidate
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
                              entry_rate_i=rate_i, entry_rate_j=rate_j)
                open_positions.append(pos)
                ledger_rows.append({"decision_ts": dts, "event": "open"})

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions

def run_all(yymms, *, decision_freq=None, carry=True, force_close_end=False, hedge_df=None, regime_mask=None, hybrid_signals=None):
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
            
        p, l, b, open_pos = run_month(yymm, decision_freq=decision_freq, open_positions=open_pos, carry_in=carry, hedges=h_mon, regime_mask=regime_mask)
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
    regime_mask = hf.regime_mask_from_signals(signals)

    print(f"[EXEC] Running Overlay on {len(months)} months.")
    pos, led, by = run_all(months, carry=True, force_close_end=True, hedge_df=trades, regime_mask=regime_mask)

    out_dir = Path(cr.PATH_OUT)
    suffix = getattr(cr, "OUT_SUFFIX", "")
    if not pos.empty: pos.to_parquet(out_dir / f"positions_ledger{suffix}.parquet")
    if not led.empty: led.to_parquet(out_dir / f"marks_ledger{suffix}.parquet")
    if not by.empty: by.to_parquet(out_dir / f"pnl_by_bucket{suffix}.parquet")
    print(f"[DONE] Results saved to {out_dir}")



def run_month(
    yymm: str,
    *,
    decision_freq: str | None = None,
    open_positions: Optional[List[PairPos]] | None = None,
    carry_in: bool = True,
    mode: str = "strategy",
    hedges: Optional[pd.DataFrame] = None,
    overlay_use_caps: Optional[bool] = None,
    regime_mask: Optional[pd.Series] = None,        
    hybrid_signals: Optional[pd.DataFrame] = None, 
    shock_cfg: Optional[ShockConfig] = None,
    shock_state: Optional[Dict] = None 
):
    import math
    try: np
    except NameError: import numpy as np

    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    mode = mode.lower()

    enh_path = _enhanced_in_path(yymm)
    if not enh_path.exists():
        raise FileNotFoundError(f"Missing enhanced file {enh_path}. Run feature_creation.py first.")

    df = pd.read_parquet(enh_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), (open_positions or [])

    # Ensure required columns
    need = {"ts", "tenor_yrs", "rate", "z_spline", "z_pca", "z_comb"}
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

    base_max_hold_decisions = cr.MAX_HOLD_DAYS * decisions_per_day
    
    # Overlay configs
    OVERLAY_MAX_HOLD_DV01_MED = float(getattr(cr, "OVERLAY_MAX_HOLD_DV01_MED", 20_000.0))
    OVERLAY_MAX_HOLD_DV01_HI = float(getattr(cr, "OVERLAY_MAX_HOLD_DV01_HI", 50_000.0))
    OVERLAY_MAX_HOLD_DAYS_MED = float(getattr(cr, "OVERLAY_MAX_HOLD_DAYS_MED", 5.0))
    OVERLAY_MAX_HOLD_DAYS_HI = float(getattr(cr, "OVERLAY_MAX_HOLD_DAYS_HI", 2.0))
    
    _check_days = getattr(cr, "OVERLAY_MIN_CHECK_DAYS", None)
    min_check_decisions = 0
    if _check_days is not None and int(_check_days) > 0:
        min_check_decisions = int(_check_days) * decisions_per_day

    def _overlay_max_hold_decisions(dv01_cash: float) -> int:
        dv = abs(float(dv01_cash))
        if dv >= OVERLAY_MAX_HOLD_DV01_HI:
            return int(round(OVERLAY_MAX_HOLD_DAYS_HI * decisions_per_day))
        elif dv >= OVERLAY_MAX_HOLD_DV01_MED:
            return int(round(OVERLAY_MAX_HOLD_DAYS_MED * decisions_per_day))
        return int(round(float(cr.MAX_HOLD_DAYS) * decisions_per_day))

    def _overlay_effective_z_entry(dv01_cash: float) -> float:
        Z_REF = float(getattr(cr, "OVERLAY_Z_ENTRY_DV01_REF", 5_000.0))
        Z_K = float(getattr(cr, "OVERLAY_Z_ENTRY_DV01_K", 0.0))
        base = float(getattr(cr, "Z_ENTRY", 0.75))
        dv = abs(float(dv01_cash))
        if dv <= 0 or Z_REF <= 0 or Z_K == 0.0: return base
        return base + Z_K * math.log(dv / Z_REF)

    OVERLAY_DV01_CAP_PER_TRADE = float(getattr(cr, "OVERLAY_DV01_CAP_PER_TRADE", float("inf")))
    OVERLAY_DV01_CAP_PER_TRADE_BUCKET = dict(getattr(cr, "OVERLAY_DV01_CAP_PER_TRADE_BUCKET", {}))
    OVERLAY_DV01_TS_CAP = float(getattr(cr, "OVERLAY_DV01_TS_CAP", float("inf")))

    def _per_trade_dv01_cap_for_bucket(bucket: str) -> float:
        return float(OVERLAY_DV01_CAP_PER_TRADE_BUCKET.get(bucket, OVERLAY_DV01_CAP_PER_TRADE_BUCKET.get("other", OVERLAY_DV01_CAP_PER_TRADE)))

    _bps_stop_val = getattr(cr, "BPS_PNL_STOP", None)
    BPS_PNL_STOP = float(_bps_stop_val) if _bps_stop_val is not None else 0.0

    # --- Shock Filter State ---
    if shock_state is None:
        shock_state = {"history": [], "dates": [], "remaining": 0}
    
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
    
    # --- REVERSION SETTINGS ---
    Z_ENTRY_REV = float(getattr(cr, "Z_ENTRY", 0.75))
    Z_EXIT_REV  = float(getattr(cr, "Z_EXIT", 0.40))
    Z_STOP_REV  = float(getattr(cr, "Z_STOP", 3.00))
    
    # --- MOMENTUM SETTINGS ---
    Z_ENTRY_MOM = float(getattr(cr, "Z_ENTRY_MOM", 2.00))
    Z_EXIT_MOM  = float(getattr(cr, "Z_EXIT_MOM", 1.00))
    Z_STOP_MOM  = float(getattr(cr, "Z_STOP_MOM", 1.00))
    
    SHORT_END_EXTRA_Z = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))
    EXEC_LEG_THRESHOLD = float(getattr(cr, "EXEC_LEG_TENOR_YEARS", 0.084))
    ALT_LEG_THRESHOLD  = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0))
    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    SHORT_EXTRA = SHORT_END_EXTRA_Z 

    for dts, snap in df.groupby("decision_ts", sort=True):
        snap_last = (
            snap.sort_values("ts")
                .groupby("tenor_yrs", as_index=False)
                .tail(1)
                .reset_index(drop=True)
        )
        if snap_last.empty: continue

        # ============================================================
        # A) START-OF-DAY GATING
        # ============================================================
        was_shock_active_at_open = (shock_state["remaining"] > 0)
        if shock_state["remaining"] > 0: shock_state["remaining"] -= 1

        regime_ok = True
        if regime_mask is not None:
            regime_ok = bool(regime_mask.at[dts]) if dts in regime_mask.index else False
        
        current_strat_mode = "reversion" if regime_ok else "momentum"
        gate = was_shock_active_at_open
        if SHOCK_MODE == "EXIT_ALL" and (shock_state["remaining"] > 0): gate = True
        
        # ============================================================
        # 1) MARK POSITIONS & EXITS
        # ============================================================
        period_pnl_cash = 0.0        
        period_pnl_bps_mtm = 0.0     
        period_pnl_bps_realized = 0.0 
        period_pnl_cash_realized = 0.0
        
        still_open: list[PairPos] = []
        
        # --- CRITICAL FIX 1: Track Active IDs ---
        # We rebuild this set every day to ensure we know exactly what is currently held.
        # This prevents the loop below from re-entering a trade that is already alive.
        active_ids = set()

        for pos in open_positions:
            prev_cash, prev_bp = pos.pnl_cash, pos.pnl_bp
            zsp = pos.mark(snap_last, decision_ts=dts)
            
            period_pnl_cash += (pos.pnl_cash - prev_cash)
            period_pnl_bps_mtm += (pos.pnl_bp - prev_bp)

            entry_z = pos.entry_zspread
            exit_flag = None

            if np.isfinite(zsp) and np.isfinite(entry_z):
                entry_dir = getattr(pos, "entry_z_dir", pos.dir_sign * entry_z)
                curr_dir = getattr(pos, "last_z_dir", pos.dir_sign * zsp)

                if np.isfinite(entry_dir) and np.isfinite(curr_dir):
                    # --- REVERSION EXITS ---
                    if pos.strat_type == "reversion":
                        sign_entry = np.sign(entry_dir)
                        sign_curr = np.sign(curr_dir)
                        same_side = (sign_entry != 0) and (sign_entry == sign_curr)
                        moved_towards_zero = abs(curr_dir) <= abs(entry_dir)
                        within_exit_band = abs(curr_dir) <= Z_EXIT_REV
                        dz_dir = curr_dir - entry_dir
                        moved_away = same_side and (abs(curr_dir) >= abs(entry_dir)) and (abs(dz_dir) >= Z_STOP_REV)

                        if same_side and moved_towards_zero and within_exit_band:
                            exit_flag = "reversion"
                        elif moved_away:
                            exit_flag = "stop"
                            
                    # --- MOMENTUM EXITS ---
                    elif pos.strat_type == "momentum":
                        if abs(curr_dir) <= (abs(entry_dir) - Z_STOP_MOM):
                             exit_flag = "stop_converge"
                        elif abs(curr_dir) >= (abs(entry_dir) + Z_EXIT_MOM):
                             exit_flag = "profit_diverge"

            # Stagnation Check
            if exit_flag is None and pos.mode == "overlay" and min_check_decisions > 0:
                if pos.age_decisions > 0 and (pos.age_decisions % min_check_decisions == 0):
                    if pos.strat_type == "reversion":
                        if curr_dir >= pos.z_at_last_check:
                            exit_flag = "stagnation"
                        else:
                            pos.z_at_last_check = curr_dir
                    elif pos.strat_type == "momentum":
                        if abs(curr_dir) <= abs(pos.z_at_last_check):
                            exit_flag = "stagnation"
                        else:
                            pos.z_at_last_check = curr_dir

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
                "pnl_price_bp": pos.pnl_price_bp,
                "pnl_carry_bp": pos.pnl_carry_bp,
                "pnl_roll_bp": pos.pnl_roll_bp,
                "pnl_price_cash": pos.pnl_price_cash,
                "pnl_carry_cash": pos.pnl_carry_cash,
                "pnl_roll_cash": pos.pnl_roll_cash,
                "z_spread": zsp, "closed": pos.closed, "mode": pos.mode,
                "strat_type": getattr(pos, "strat_type", "reversion"),
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
                period_pnl_cash_realized += (pos.pnl_cash - tcost_cash)

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
                    "pnl_price_bp": pos.pnl_price_bp,
                    "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp": pos.pnl_roll_bp,
                    "pnl_price_cash": pos.pnl_price_cash,
                    "pnl_carry_cash": pos.pnl_carry_cash, 
                    "pnl_roll_cash": pos.pnl_roll_cash,   
                    "tcost_bp": tcost_bp, 
                    "tcost_cash": tcost_cash,
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, 
                    "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "days_held_equiv": pos.age_decisions / max(1, decisions_per_day),
                    "mode": pos.mode,
                    "strat_type": getattr(pos, "strat_type", "reversion"),
                    "trade_id": pos.meta.get("trade_id"),
                    "side": pos.meta.get("side"),
                })
            else:
                still_open.append(pos)
                # --- CRITICAL FIX 2: Register Active ID ---
                # If the position remains open, we register its ID.
                # This ensures the Entry loop (below) knows it exists.
                tid = pos.meta.get("trade_id")
                if tid is not None:
                    active_ids.add(tid)
        
        open_positions = still_open
        
        # ... [Shock History Update - Same as before] ...
        metric_type = getattr(shock_cfg, "metric_type", "MTM_BPS") if shock_cfg else "MTM_BPS"
        if metric_type == "REALIZED_CASH": metric_val = period_pnl_cash_realized
        elif metric_type == "REALIZED_BPS": metric_val = period_pnl_bps_realized
        elif metric_type == "MTM_BPS" or metric_type == "BPS": metric_val = period_pnl_bps_mtm
        else: metric_val = period_pnl_cash

        shock_state["history"].append(metric_val)
        shock_state["dates"].append(dts)
        
        is_new_shock = False
        if shock_cfg is not None:
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
                                           if last_r_z <= shock_cfg.resid_z_thresh:
                                               is_new_shock = True
                                  except np.linalg.LinAlgError:
                                      pass
                    except Exception:
                         pass
        if is_new_shock: shock_state["remaining"] = int(shock_cfg.block_length)

        # ... [Panic Exit - Same as before] ...
        is_in_shock_state = (shock_state["remaining"] > 0) or is_new_shock
        if is_in_shock_state and SHOCK_MODE == "EXIT_ALL" and len(open_positions) > 0:
            panic_bp_real, panic_cash_real = 0.0, 0.0
            panic_t_bp, panic_t_cash = 0.0, 0.0
            for pos in open_positions:
                pos.closed = True
                pos.close_ts = dts
                pos.exit_reason = "shock_exit"
                tcost_bp = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10)) if pos.mode == "overlay" else 0.0
                tcost_cash = tcost_bp * pos.scale_dv01
                pos.tcost_bp, pos.tcost_cash = tcost_bp, tcost_cash
                
                panic_t_bp += tcost_bp
                panic_t_cash += tcost_cash
                panic_bp_real += (pos.pnl_bp - tcost_bp)
                panic_cash_real += (pos.pnl_cash - tcost_cash)
                
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": dts, "exit_reason": "shock_exit",
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "pnl_price_bp": pos.pnl_price_bp,
                    "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp": pos.pnl_roll_bp,
                    "pnl_carry_cash": pos.pnl_carry_cash, "pnl_roll_cash": pos.pnl_roll_cash,
                    "mode": pos.mode, "trade_id": pos.meta.get("trade_id"), "side": pos.meta.get("side"),
                    "strat_type": getattr(pos, "strat_type", "reversion")
                })
            open_positions = []
            if metric_type == "REALIZED_CASH": shock_state["history"][-1] += panic_cash_real
            elif metric_type == "REALIZED_BPS": shock_state["history"][-1] += panic_bp_real
            elif metric_type == "MTM_BPS" or metric_type == "BPS": shock_state["history"][-1] -= panic_t_bp
            else: shock_state["history"][-1] -= panic_t_cash

        # ============================================================
        # 4) NEW ENTRIES (With Re-Entry Guard)
        # ============================================================
        if gate: continue
        
        rem_slots = max(0, cr.MAX_CONCURRENT_PAIRS - len(open_positions))
        if rem_slots <= 0: continue

        if mode == "strategy":
            selected = choose_pairs_under_caps(snap_last, rem_slots, PER_BUCKET_DV01_CAP, TOTAL_DV01_CAP, FRONT_END_DV01_CAP, 0.0)
            for (cheap, rich, w_i, w_j) in selected:
                pos = PairPos(dts, cheap, rich, w_i, w_j, decisions_per_day, mode="strategy", strat_type="reversion")
                open_positions.append(pos)
                ledger_rows.append({"decision_ts": dts, "event": "open", "mode": "strategy", "strat_type": "reversion"})
                
        elif mode == "overlay":
            if hedges is None or hedges.empty: continue
            h_here = hedges[hedges["decision_ts"] == dts]
            if h_here.empty: continue
            if float(h_here["dv01"].abs().sum()) > OVERLAY_DV01_TS_CAP: continue
            
            snap_srt = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
            
            DRIFT_GATE = float(getattr(cr, "DRIFT_GATE_BPS", -100.0)) 
            DRIFT_W = float(getattr(cr, "DRIFT_WEIGHT", 0.0)) 

            for _, h in h_here.iterrows():
                if len(open_positions) >= cr.MAX_CONCURRENT_PAIRS: break
                
                # --- CRITICAL FIX 3: Re-Entry Guard ---
                # Check if this trade_id is ALREADY in open_positions (from the set active_ids)
                tid = h.get("trade_id")
                if tid is not None and tid in active_ids: 
                    # If we already have it, we skip re-evaluation.
                    # This protects against duplicate entries in the hedge tape OR logic errors.
                    continue
                # --------------------------------------
                
                t_trade = float(h["tenor_yrs"])
                
                if t_trade < EXEC_LEG_THRESHOLD: continue
                if abs(float(h["dv01"])) > _per_trade_dv01_cap_for_bucket(assign_bucket(t_trade)): continue

                side_s = 1.0 if str(h["side"]).upper() == "CRCV" else -1.0
                
                z_ent_eff = _overlay_effective_z_entry(float(h["dv01"]))
                if current_strat_mode == "momentum":
                    z_ent_eff = Z_ENTRY_MOM 
                
                exec_z = _get_z_at_tenor(snap_srt, t_trade)
                if exec_z is None: continue
                exec_row = snap_srt.iloc[(snap_srt["tenor_yrs"] - t_trade).abs().idxmin()]
                exec_tenor = float(exec_row["tenor_yrs"])

                best_c_row, best_score = None, -999.0

                for _, alt in snap_srt.iterrows():
                    alt_tenor = float(alt["tenor_yrs"])
                    
                    if alt_tenor < ALT_LEG_THRESHOLD: continue
                    if alt_tenor == exec_tenor: continue 
                    if assign_bucket(alt_tenor) == "short" and assign_bucket(exec_tenor) == "long": continue
                    if assign_bucket(exec_tenor) == "short" and assign_bucket(alt_tenor) == "long": continue
                                        
                    z_alt = _to_float(alt["z_comb"])
                    disp = (z_alt - exec_z) if side_s > 0 else (exec_z - z_alt)
                    
                    if current_strat_mode == "reversion":
                        if disp < z_ent_eff: continue
                        if (assign_bucket(alt_tenor)=="short" or assign_bucket(exec_tenor)=="short") and (disp < z_ent_eff + SHORT_EXTRA):
                            continue
                            
                    elif current_strat_mode == "momentum":
                        if disp > -z_ent_eff: continue

                    c_t, r_t = (alt_tenor, exec_tenor) if z_alt > exec_z else (exec_tenor, alt_tenor)
                    if not (fly_alignment_ok(c_t, 1, snap_srt, zdisp_for_pair=disp) and fly_alignment_ok(r_t, -1, snap_srt, zdisp_for_pair=disp)): continue
                    
                    drift_exec = calc_trade_drift(exec_tenor, side_s, snap_srt)
                    drift_alt = calc_trade_drift(alt_tenor, side_s, snap_srt)
                    
                    if drift_exec == -999.0 or drift_alt == -999.0: continue

                    net_drift_bps = drift_alt - drift_exec 
                    dist_years = abs(alt_tenor - exec_tenor)
                    scaling_factor = dist_years
                    norm_drift_bps = net_drift_bps / scaling_factor
                    
                    if net_drift_bps < DRIFT_GATE: continue 
                    
                    score = disp + (norm_drift_bps * DRIFT_WEIGHT)
                    
                    if current_strat_mode == "momentum":
                        score = -score
                    
                    if score > best_score: 
                        best_score, best_c_row = score, alt
                
                if best_c_row is not None:
                    rate_i, rate_j = None, None
                    ti, tj = tenor_to_ticker(float(best_c_row["tenor_yrs"])), tenor_to_ticker(t_trade)
                    if ti and f"{ti}_mid" in h: rate_i = _to_float(h[f"{ti}_mid"])
                    if tj and f"{tj}_mid" in h: rate_j = _to_float(h[f"{tj}_mid"])
                    
                    if rate_i is None: rate_i = _to_float(best_c_row["rate"])
                    if rate_j is None: rate_j = _to_float(exec_row["rate"])

                    pos = PairPos(dts, best_c_row, exec_row, side_s*1.0, side_s*-1.0, decisions_per_day, 
                                  scale_dv01=float(h["dv01"]), mode="overlay", 
                                  meta={"trade_id": h.get("trade_id"), "side": h.get("side")},
                                  entry_rate_i=rate_i, entry_rate_j=rate_j,
                                  strat_type=current_strat_mode) 
                    
                    open_positions.append(pos)
                    # --- Update Active Set immediately ---
                    if tid is not None: active_ids.add(tid)
                    
                    ledger_rows.append({"decision_ts": dts, "event": "open", "mode": "overlay", "strat_type": current_strat_mode})

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions





def run_month(
    yymm: str,
    *,
    decision_freq: str | None = None,
    open_positions: Optional[List[PairPos]] | None = None,
    carry_in: bool = True,
    mode: str = "strategy",
    hedges: Optional[pd.DataFrame] = None,
    overlay_use_caps: Optional[bool] = None,
    regime_mask: Optional[pd.Series] = None,        
    hybrid_signals: Optional[pd.DataFrame] = None, 
    shock_cfg: Optional[ShockConfig] = None,
    shock_state: Optional[Dict] = None 
):
    import math
    try: np
    except NameError: import numpy as np

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
    
    # Overlay configs
    OVERLAY_MAX_HOLD_DV01_MED = float(getattr(cr, "OVERLAY_MAX_HOLD_DV01_MED", 20_000.0))
    OVERLAY_MAX_HOLD_DV01_HI = float(getattr(cr, "OVERLAY_MAX_HOLD_DV01_HI", 50_000.0))
    OVERLAY_MAX_HOLD_DAYS_MED = float(getattr(cr, "OVERLAY_MAX_HOLD_DAYS_MED", 5.0))
    OVERLAY_MAX_HOLD_DAYS_HI = float(getattr(cr, "OVERLAY_MAX_HOLD_DAYS_HI", 2.0))
    
    _check_days = getattr(cr, "OVERLAY_MIN_CHECK_DAYS", None)
    min_check_decisions = 0
    if _check_days is not None and int(_check_days) > 0:
        min_check_decisions = int(_check_days) * decisions_per_day

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

    # --- Shock Filter State ---
    if shock_state is None:
        shock_state = {"history": [], "dates": [], "remaining": 0}
    
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
    
    # --- REVERSION SETTINGS ---
    Z_ENTRY_REV = float(getattr(cr, "Z_ENTRY", 0.75))
    Z_EXIT_REV  = float(getattr(cr, "Z_EXIT", 0.40))
    Z_STOP_REV  = float(getattr(cr, "Z_STOP", 3.00))
    
    # --- MOMENTUM SETTINGS (NEW) ---
    Z_ENTRY_MOM = float(getattr(cr, "Z_ENTRY_MOM", 2.00))
    Z_EXIT_MOM  = float(getattr(cr, "Z_EXIT_MOM", 1.00)) # Profit Dist
    Z_STOP_MOM  = float(getattr(cr, "Z_STOP_MOM", 1.00)) # Stop Dist
    
    SHORT_END_EXTRA_Z = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))
    
    EXEC_LEG_THRESHOLD = float(getattr(cr, "EXEC_LEG_TENOR_YEARS", 0.084))
    ALT_LEG_THRESHOLD  = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0))
    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    
    SHORT_EXTRA = SHORT_END_EXTRA_Z 

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
        # A) START-OF-DAY GATING
        # ============================================================
        was_shock_active_at_open = (shock_state["remaining"] > 0)
        
        if shock_state["remaining"] > 0:
            shock_state["remaining"] -= 1

        regime_ok = True
        if regime_mask is not None:
            regime_ok = bool(regime_mask.at[dts]) if dts in regime_mask.index else False
        
        # Determine Mode for NEW trades
        current_strat_mode = "reversion" if regime_ok else "momentum"
        
        gate = was_shock_active_at_open
        if SHOCK_MODE == "EXIT_ALL" and (shock_state["remaining"] > 0): gate = True
        
        # ============================================================
        # 1) MARK POSITIONS & EXITS
        # ============================================================
        period_pnl_cash = 0.0        
        period_pnl_bps_mtm = 0.0     
        period_pnl_bps_realized = 0.0 
        period_pnl_cash_realized = 0.0
        
        still_open: list[PairPos] = []
        
        for pos in open_positions:
            prev_cash, prev_bp = pos.pnl_cash, pos.pnl_bp
            zsp = pos.mark(snap_last, decision_ts=dts)
            
            period_pnl_cash += (pos.pnl_cash - prev_cash)
            period_pnl_bps_mtm += (pos.pnl_bp - prev_bp)

            entry_z = pos.entry_zspread
            exit_flag = None

            if np.isfinite(zsp) and np.isfinite(entry_z):
                entry_dir = getattr(pos, "entry_z_dir", pos.dir_sign * entry_z)
                curr_dir = getattr(pos, "last_z_dir", pos.dir_sign * zsp)

                if np.isfinite(entry_dir) and np.isfinite(curr_dir):
                    
                    # --- REVERSION EXIT LOGIC ---
                    if pos.strat_type == "reversion":
                        sign_entry = np.sign(entry_dir)
                        sign_curr = np.sign(curr_dir)
                        same_side = (sign_entry != 0) and (sign_entry == sign_curr)
                        moved_towards_zero = abs(curr_dir) <= abs(entry_dir)
                        within_exit_band = abs(curr_dir) <= Z_EXIT_REV
                        dz_dir = curr_dir - entry_dir
                        moved_away = same_side and (abs(curr_dir) >= abs(entry_dir)) and (abs(dz_dir) >= Z_STOP_REV)

                        if same_side and moved_towards_zero and within_exit_band:
                            exit_flag = "reversion"
                        elif moved_away:
                            exit_flag = "stop"
                            
                    # --- MOMENTUM EXIT LOGIC ---
                    elif pos.strat_type == "momentum":
                        # Convergence = Stop (Losing the trend)
                        # "Distance Reverted" > STOP_MOM
                        # i.e. Current Abs < (Entry Abs - STOP)
                        if abs(curr_dir) <= (abs(entry_dir) - Z_STOP_MOM):
                             exit_flag = "stop_converge"
                        
                        # Divergence = Profit (Trend Extension)
                        # "Distance Gained" > EXIT_MOM
                        # i.e. Current Abs >= (Entry Abs + EXIT)
                        elif abs(curr_dir) >= (abs(entry_dir) + Z_EXIT_MOM):
                             exit_flag = "profit_diverge"

            # Stagnation Check
            if exit_flag is None and pos.mode == "overlay" and min_check_decisions > 0:
                if pos.age_decisions > 0 and (pos.age_decisions % min_check_decisions == 0):
                    if pos.strat_type == "reversion":
                        if curr_dir >= pos.z_at_last_check:
                            exit_flag = "stagnation"
                        else:
                            pos.z_at_last_check = curr_dir
                            
                    elif pos.strat_type == "momentum":
                        # Momentum wants growth (higher Abs). Stagnation = No growth.
                        if abs(curr_dir) <= abs(pos.z_at_last_check):
                            exit_flag = "stagnation"
                        else:
                            pos.z_at_last_check = curr_dir

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
                "pnl_price_bp": pos.pnl_price_bp,
                "pnl_carry_bp": pos.pnl_carry_bp,
                "pnl_roll_bp": pos.pnl_roll_bp,
                "pnl_price_cash": pos.pnl_price_cash,
                "pnl_carry_cash": pos.pnl_carry_cash,
                "pnl_roll_cash": pos.pnl_roll_cash,
                "z_spread": zsp, "closed": pos.closed, "mode": pos.mode,
                "strat_type": getattr(pos, "strat_type", "reversion"),
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
                period_pnl_cash_realized += (pos.pnl_cash - tcost_cash)

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
                    "pnl_price_bp": pos.pnl_price_bp,
                    "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp": pos.pnl_roll_bp,
                    "pnl_price_cash": pos.pnl_price_cash,
                    "pnl_carry_cash": pos.pnl_carry_cash, 
                    "pnl_roll_cash": pos.pnl_roll_cash,   
                    "tcost_bp": tcost_bp, 
                    "tcost_cash": tcost_cash,
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, 
                    "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "days_held_equiv": pos.age_decisions / max(1, decisions_per_day),
                    "mode": pos.mode,
                    "strat_type": getattr(pos, "strat_type", "reversion"),
                    "trade_id": pos.meta.get("trade_id"),
                    "side": pos.meta.get("side"),
                })
            else:
                still_open.append(pos)
        
        # ... [Shock History Update (Standard)] ...
        metric_type = getattr(shock_cfg, "metric_type", "MTM_BPS") if shock_cfg else "MTM_BPS"
        if metric_type == "REALIZED_CASH": metric_val = period_pnl_cash_realized
        elif metric_type == "REALIZED_BPS": metric_val = period_pnl_bps_realized
        elif metric_type == "MTM_BPS" or metric_type == "BPS": metric_val = period_pnl_bps_mtm
        else: metric_val = period_pnl_cash

        shock_state["history"].append(metric_val)
        shock_state["dates"].append(dts)
        
        is_new_shock = False
        if shock_cfg is not None:
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
                                           if last_r_z <= shock_cfg.resid_z_thresh:
                                               is_new_shock = True
                                  except np.linalg.LinAlgError:
                                      pass
                    except Exception:
                         pass
        
        if is_new_shock: shock_state["remaining"] = int(shock_cfg.block_length)

        # ... [Panic Exit Logic (Standard)] ...
        is_in_shock_state = (shock_state["remaining"] > 0) or is_new_shock
        if is_in_shock_state and SHOCK_MODE == "EXIT_ALL" and len(open_positions) > 0:
            panic_bp_real, panic_cash_real = 0.0, 0.0
            panic_t_bp, panic_t_cash = 0.0, 0.0
            
            for pos in open_positions:
                pos.closed = True
                pos.close_ts = dts
                pos.exit_reason = "shock_exit"
                tcost_bp = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10)) if pos.mode == "overlay" else 0.0
                tcost_cash = tcost_bp * pos.scale_dv01
                pos.tcost_bp, pos.tcost_cash = tcost_bp, tcost_cash
                
                panic_t_bp += tcost_bp
                panic_t_cash += tcost_cash
                panic_bp_real += (pos.pnl_bp - tcost_bp)
                panic_cash_real += (pos.pnl_cash - tcost_cash)
                
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": dts, "exit_reason": "shock_exit",
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "pnl_price_bp": pos.pnl_price_bp,
                    "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp": pos.pnl_roll_bp,
                    "pnl_carry_cash": pos.pnl_carry_cash, "pnl_roll_cash": pos.pnl_roll_cash,
                    "mode": pos.mode, "trade_id": pos.meta.get("trade_id"), "side": pos.meta.get("side"),
                    "strat_type": getattr(pos, "strat_type", "reversion")
                })
            open_positions = []
            
            if metric_type == "REALIZED_CASH": shock_state["history"][-1] += panic_cash_real
            elif metric_type == "REALIZED_BPS": shock_state["history"][-1] += panic_bp_real
            elif metric_type == "MTM_BPS" or metric_type == "BPS": shock_state["history"][-1] -= panic_t_bp
            else: shock_state["history"][-1] -= panic_t_cash

        # ============================================================
        # 4) NEW ENTRIES
        # ============================================================
        if gate: continue
        
        rem_slots = max(0, cr.MAX_CONCURRENT_PAIRS - len(open_positions))
        if rem_slots <= 0: continue

        if mode == "strategy":
            selected = choose_pairs_under_caps(snap_last, rem_slots, PER_BUCKET_DV01_CAP, TOTAL_DV01_CAP, FRONT_END_DV01_CAP, 0.0)
            for (cheap, rich, w_i, w_j) in selected:
                pos = PairPos(dts, cheap, rich, w_i, w_j, decisions_per_day, mode="strategy", strat_type="reversion")
                open_positions.append(pos)
                ledger_rows.append({"decision_ts": dts, "event": "open", "mode": "strategy", "strat_type": "reversion"})
                
        elif mode == "overlay":
            if hedges is None or hedges.empty: continue
            h_here = hedges[hedges["decision_ts"] == dts]
            if h_here.empty: continue
            if float(h_here["dv01"].abs().sum()) > OVERLAY_DV01_TS_CAP: continue
            
            snap_srt = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
            
            DRIFT_GATE = float(getattr(cr, "DRIFT_GATE_BPS", -100.0)) 
            DRIFT_W = float(getattr(cr, "DRIFT_WEIGHT", 0.0)) 

            for _, h in h_here.iterrows():
                if len(open_positions) >= cr.MAX_CONCURRENT_PAIRS: break
                t_trade = float(h["tenor_yrs"])
                
                if t_trade < EXEC_LEG_THRESHOLD: continue
                if abs(float(h["dv01"])) > _per_trade_dv01_cap_for_bucket(assign_bucket(t_trade)): continue

                side_s = 1.0 if str(h["side"]).upper() == "CRCV" else -1.0
                
                # Z_ENTRY varies by Strategy Type
                z_ent_eff = _overlay_effective_z_entry(float(h["dv01"]))
                if current_strat_mode == "momentum":
                    # For momentum, use the MOM entry threshold (plus scaling if desired, here just raw)
                    z_ent_eff = Z_ENTRY_MOM 
                
                exec_z = _get_z_at_tenor(snap_srt, t_trade)
                if exec_z is None: continue
                exec_row = snap_srt.iloc[(snap_srt["tenor_yrs"] - t_trade).abs().idxmin()]
                exec_tenor = float(exec_row["tenor_yrs"])

                best_c_row, best_score = None, -999.0

                for _, alt in snap_srt.iterrows():
                    alt_tenor = float(alt["tenor_yrs"])
                    
                    if alt_tenor < ALT_LEG_THRESHOLD: continue
                    if alt_tenor == exec_tenor: continue 
                    if assign_bucket(alt_tenor) == "short" and assign_bucket(exec_tenor) == "long": continue
                    if assign_bucket(exec_tenor) == "short" and assign_bucket(alt_tenor) == "long": continue
                                        
                    z_alt = _to_float(alt["z_comb"])
                    disp = (z_alt - exec_z) if side_s > 0 else (exec_z - z_alt)
                    
                    # --- REVERSION ENTRY ---
                    if current_strat_mode == "reversion":
                        if disp < z_ent_eff: continue
                        if (assign_bucket(alt_tenor)=="short" or assign_bucket(exec_tenor)=="short") and (disp < z_ent_eff + SHORT_EXTRA):
                            continue
                            
                    # --- MOMENTUM ENTRY ---
                    elif current_strat_mode == "momentum":
                        # We want negative dispersion (Betting against model)
                        # disp must be "worse" than -Z_ENTRY_MOM
                        if disp > -z_ent_eff: continue

                    c_t, r_t = (alt_tenor, exec_tenor) if z_alt > exec_z else (exec_tenor, alt_tenor)
                    if not (fly_alignment_ok(c_t, 1, snap_srt, zdisp_for_pair=disp) and fly_alignment_ok(r_t, -1, snap_srt, zdisp_for_pair=disp)): continue
                    
                    drift_exec = calc_trade_drift(exec_tenor, side_s, snap_srt)
                    drift_alt = calc_trade_drift(alt_tenor, side_s, snap_srt)
                    
                    if drift_exec == -999.0 or drift_alt == -999.0: continue

                    net_drift_bps = drift_alt - drift_exec 
                    dist_years = abs(alt_tenor - exec_tenor)
                    scaling_factor = dist_years
                    norm_drift_bps = net_drift_bps / scaling_factor
                    
                    if net_drift_bps < DRIFT_GATE: continue 
                    
                    score = disp + (norm_drift_bps * DRIFT_WEIGHT)
                    
                    # Flip score for Momentum (Want most negative)
                    if current_strat_mode == "momentum":
                        score = -score
                    
                    if score > best_score: 
                        best_score, best_c_row = score, alt
                
                if best_c_row is not None:
                    rate_i, rate_j = None, None
                    ti, tj = tenor_to_ticker(float(best_c_row["tenor_yrs"])), tenor_to_ticker(t_trade)
                    if ti and f"{ti}_mid" in h: rate_i = _to_float(h[f"{ti}_mid"])
                    if tj and f"{tj}_mid" in h: rate_j = _to_float(h[f"{tj}_mid"])
                    
                    if rate_i is None: rate_i = _to_float(best_c_row["rate"])
                    if rate_j is None: rate_j = _to_float(exec_row["rate"])

                    pos = PairPos(dts, best_c_row, exec_row, side_s*1.0, side_s*-1.0, decisions_per_day, 
                                  scale_dv01=float(h["dv01"]), mode="overlay", 
                                  meta={"trade_id": h.get("trade_id"), "side": h.get("side")},
                                  entry_rate_i=rate_i, entry_rate_j=rate_j,
                                  strat_type=current_strat_mode) # Pass Strategy
                    
                    open_positions.append(pos)
                    ledger_rows.append({"decision_ts": dts, "event": "open", "mode": "overlay", "strat_type": current_strat_mode})

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions





class PairPos:
    def __init__(self, open_ts, cheap_row, rich_row, w_i, w_j, decisions_per_day, *, 
                 scale_dv01=1.0, mode="strategy", meta=None, dir_sign=None, 
                 entry_rate_i=None, entry_rate_j=None,
                 # NEW: Strategy Type ('reversion' or 'momentum')
                 strat_type="reversion"):
        
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

        # NEW: Store Strategy Type
        self.strat_type = strat_type 

        self.closed, self.close_ts, self.exit_reason = False, None, None
        self.scale_dv01, self.mode = float(scale_dv01), str(mode)
        self.meta = meta or {}
        self.initial_dv01 = self.scale_dv01
        
        self.dv01_i_entry = self.scale_dv01 * self.w_i
        self.dv01_j_entry = self.scale_dv01 * self.w_j
        self.dv01_i_curr, self.dv01_j_curr = self.dv01_i_entry, self.dv01_j_entry
        self.rem_tenor_i, self.rem_tenor_j = self.tenor_i_orig, self.tenor_j_orig
        
        self.last_mark_ts = open_ts 
        
        # PnL Components (Cumulative)
        self.pnl_price_cash = 0.0
        self.pnl_carry_cash = 0.0
        self.pnl_roll_cash = 0.0
        self.pnl_cash = 0.0
        self.pnl_bp = 0.0
        
        # Breakdown Bps
        self.pnl_price_bp = 0.0
        self.pnl_carry_bp = 0.0
        self.pnl_roll_bp = 0.0

        self.tcost_bp, self.tcost_cash = 0.0, 0.0
        self.decisions_per_day = decisions_per_day
        self.age_decisions = 0
        self.bucket_i, self.bucket_j = assign_bucket(self.tenor_i), assign_bucket(self.tenor_j)
        self.last_zspread, self.last_z_dir, self.conv_pnl_proxy = self.entry_zspread, self.entry_z_dir, 0.0

        # NEW: Stagnation Check State (Initialized to entry level)
        self.z_at_last_check = self.entry_z_dir

    def _update_overlay_dv01(self, decision_ts):
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

    def mark(self, snap_last: pd.DataFrame, decision_ts: Optional[pd.Timestamp] = None):
        if self.mode == "overlay" and decision_ts:
            self._update_overlay_dv01(decision_ts)

        ri = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "rate"])
        rj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "rate"])
        r_float = _get_funding_rate(snap_last)
        self.last_rate_i, self.last_rate_j = ri, rj

        # Price PnL
        pnl_price_i = (self.entry_rate_i - ri) * 100.0 * self.dv01_i_curr
        pnl_price_j = (self.entry_rate_j - rj) * 100.0 * self.dv01_j_curr
        self.pnl_price_cash = pnl_price_i + pnl_price_j

        # Carry PnL
        if decision_ts and self.last_mark_ts:
            dt_days = max(0.0, (decision_ts.normalize() - self.last_mark_ts.normalize()).days)
            if dt_days > 0:
                inc_carry_i = (self.entry_rate_i - r_float) * 100.0 * self.dv01_i_curr * (dt_days / 360.0)
                inc_carry_j = (self.entry_rate_j - r_float) * 100.0 * self.dv01_j_curr * (dt_days / 360.0)
                self.pnl_carry_cash += (inc_carry_i + inc_carry_j)
            self.last_mark_ts = decision_ts

        # Roll-Down PnL
        days_total = 0.0
        if decision_ts:
            days_total = max(0.0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        
        xp = snap_last["tenor_yrs"].values
        fp = snap_last["rate"].values
        sort_idx = np.argsort(xp)
        xp_sorted = xp[sort_idx]
        fp_sorted = fp[sort_idx]
        
        t_roll_i = max(0.0, self.tenor_i_orig - (days_total/360.0))
        y_roll_i = np.interp(t_roll_i, xp_sorted, fp_sorted)
        roll_gain_i = (ri - y_roll_i) * 100.0 * self.dv01_i_curr
        
        t_roll_j = max(0.0, self.tenor_j_orig - (days_total/360.0))
        y_roll_j = np.interp(t_roll_j, xp_sorted, fp_sorted)
        roll_gain_j = (rj - y_roll_j) * 100.0 * self.dv01_j_curr
        
        self.pnl_roll_cash = roll_gain_i + roll_gain_j

        # Totals
        self.pnl_cash = self.pnl_price_cash + self.pnl_carry_cash + self.pnl_roll_cash
        
        if self.scale_dv01 != 0.0 and np.isfinite(self.scale_dv01):
            self.pnl_bp = self.pnl_cash / self.scale_dv01
            self.pnl_price_bp = self.pnl_price_cash / self.scale_dv01
            self.pnl_carry_bp = self.pnl_carry_cash / self.scale_dv01
            self.pnl_roll_bp = self.pnl_roll_cash / self.scale_dv01
        else:
            self.pnl_bp = 0.0; self.pnl_price_bp = 0.0; self.pnl_carry_bp = 0.0; self.pnl_roll_bp = 0.0

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


def run_month(
    yymm: str,
    *,
    decision_freq: str | None = None,
    open_positions: Optional[List[PairPos]] | None = None,
    carry_in: bool = True,
    mode: str = "strategy",
    hedges: Optional[pd.DataFrame] = None,
    overlay_use_caps: Optional[bool] = None,
    regime_mask: Optional[pd.Series] = None,        
    hybrid_signals: Optional[pd.DataFrame] = None, 
    shock_cfg: Optional[ShockConfig] = None,
    shock_state: Optional[Dict] = None 
):
    import math
    try: np
    except NameError: import numpy as np

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
    
    # Overlay configs
    OVERLAY_MAX_HOLD_DV01_MED = float(getattr(cr, "OVERLAY_MAX_HOLD_DV01_MED", 20_000.0))
    OVERLAY_MAX_HOLD_DV01_HI = float(getattr(cr, "OVERLAY_MAX_HOLD_DV01_HI", 50_000.0))
    OVERLAY_MAX_HOLD_DAYS_MED = float(getattr(cr, "OVERLAY_MAX_HOLD_DAYS_MED", 5.0))
    OVERLAY_MAX_HOLD_DAYS_HI = float(getattr(cr, "OVERLAY_MAX_HOLD_DAYS_HI", 2.0))
    
    # Stagnation Check Config
    _check_days = getattr(cr, "OVERLAY_MIN_CHECK_DAYS", None)
    min_check_decisions = 0
    if _check_days is not None and int(_check_days) > 0:
        min_check_decisions = int(_check_days) * decisions_per_day

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

    # --- Shock Filter State ---
    if shock_state is None:
        shock_state = {"history": [], "dates": [], "remaining": 0}
    
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
    
    EXEC_LEG_THRESHOLD = float(getattr(cr, "EXEC_LEG_TENOR_YEARS", 0.084))
    ALT_LEG_THRESHOLD  = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0))
    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    
    SHORT_EXTRA = SHORT_END_EXTRA_Z 

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
        # A) START-OF-DAY GATING & REGIME
        # ============================================================
        was_shock_active_at_open = (shock_state["remaining"] > 0)
        
        if shock_state["remaining"] > 0:
            shock_state["remaining"] -= 1

        regime_ok = True
        if regime_mask is not None:
            regime_ok = bool(regime_mask.at[dts]) if dts in regime_mask.index else False
        
        # --- MODE SWITCHING ---
        current_strat_mode = "reversion" if regime_ok else "momentum"
        
        gate = was_shock_active_at_open
        if SHOCK_MODE == "EXIT_ALL" and (shock_state["remaining"] > 0): gate = True
        
        # ============================================================
        # 1) MARK POSITIONS & NATURAL EXITS
        # ============================================================
        period_pnl_cash = 0.0        
        period_pnl_bps_mtm = 0.0     
        period_pnl_bps_realized = 0.0 
        period_pnl_cash_realized = 0.0
        
        still_open: list[PairPos] = []
        
        for pos in open_positions:
            prev_cash, prev_bp = pos.pnl_cash, pos.pnl_bp
            
            # Mark
            zsp = pos.mark(snap_last, decision_ts=dts)
            
            # Incremental MTM
            period_pnl_cash += (pos.pnl_cash - prev_cash)
            period_pnl_bps_mtm += (pos.pnl_bp - prev_bp)

            entry_z = pos.entry_zspread
            exit_flag = None

            if np.isfinite(zsp) and np.isfinite(entry_z):
                entry_dir = getattr(pos, "entry_z_dir", pos.dir_sign * entry_z)
                curr_dir = getattr(pos, "last_z_dir", pos.dir_sign * zsp)

                if np.isfinite(entry_dir) and np.isfinite(curr_dir):
                    
                    # --- REVERSION EXIT LOGIC ---
                    if pos.strat_type == "reversion":
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
                            
                    # --- MOMENTUM EXIT LOGIC ---
                    elif pos.strat_type == "momentum":
                        # Convergence = Stop (Losing the trend)
                        if abs(curr_dir) < abs(entry_dir):
                             exit_flag = "stop_converge"
                        # Divergence = Profit
                        elif abs(curr_dir) > (abs(entry_dir) + Z_STOP):
                             exit_flag = "profit_diverge"

            # Check Stagnation (Only applies to Reversion?)
            # Usually momentum moves fast, so we keep stagnation check for both.
            if exit_flag is None and pos.mode == "overlay" and min_check_decisions > 0:
                if pos.age_decisions > 0 and (pos.age_decisions % min_check_decisions == 0):
                    # For Reversion: Improvement = Lower Z.
                    # For Momentum: Improvement = Higher Z (More Divergence).
                    
                    if pos.strat_type == "reversion":
                        if curr_dir >= pos.z_at_last_check:
                            exit_flag = "stagnation"
                        else:
                            pos.z_at_last_check = curr_dir
                            
                    elif pos.strat_type == "momentum":
                        # We want it to be HIGHER (more divergent) than last check
                        if abs(curr_dir) <= abs(pos.z_at_last_check):
                            exit_flag = "stagnation"
                        else:
                            pos.z_at_last_check = curr_dir

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

            # 
            # Ensure STRAT_TYPE is added but no existing keys removed
            ledger_rows.append({
                "decision_ts": dts, "event": "mark",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "pnl_bp": pos.pnl_bp, "pnl_cash": pos.pnl_cash,
                "pnl_price_bp": pos.pnl_price_bp,
                "pnl_carry_bp": pos.pnl_carry_bp,
                "pnl_roll_bp": pos.pnl_roll_bp,
                "pnl_price_cash": pos.pnl_price_cash,
                "pnl_carry_cash": pos.pnl_carry_cash,
                "pnl_roll_cash": pos.pnl_roll_cash,
                "z_spread": zsp, "closed": pos.closed, "mode": pos.mode,
                "strat_type": getattr(pos, "strat_type", "reversion"), # <--- NEW
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
                period_pnl_cash_realized += (pos.pnl_cash - tcost_cash)

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
                    "pnl_price_bp": pos.pnl_price_bp,
                    "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp": pos.pnl_roll_bp,
                    "pnl_price_cash": pos.pnl_price_cash,
                    "pnl_carry_cash": pos.pnl_carry_cash, 
                    "pnl_roll_cash": pos.pnl_roll_cash,   
                    "tcost_bp": tcost_bp, 
                    "tcost_cash": tcost_cash,
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, 
                    "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "days_held_equiv": pos.age_decisions / max(1, decisions_per_day),
                    "mode": pos.mode,
                    "strat_type": getattr(pos, "strat_type", "reversion"), # <--- NEW
                    "trade_id": pos.meta.get("trade_id"),
                    "side": pos.meta.get("side"),
                })
            else:
                still_open.append(pos)
        
        open_positions = still_open
        
        # ============================================================
        # 2) UPDATE HISTORY (SHOCK)
        # ============================================================
        metric_type = getattr(shock_cfg, "metric_type", "MTM_BPS") if shock_cfg else "MTM_BPS"
        if metric_type == "REALIZED_CASH": metric_val = period_pnl_cash_realized
        elif metric_type == "REALIZED_BPS": metric_val = period_pnl_bps_realized
        elif metric_type == "MTM_BPS" or metric_type == "BPS": metric_val = period_pnl_bps_mtm
        else: metric_val = period_pnl_cash

        shock_state["history"].append(metric_val)
        shock_state["dates"].append(dts)
        
        is_new_shock = False
        if shock_cfg is not None:
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
                                           if last_r_z <= shock_cfg.resid_z_thresh:
                                               is_new_shock = True
                                  except np.linalg.LinAlgError:
                                      pass
                    except Exception:
                         pass
        
        if is_new_shock: shock_state["remaining"] = int(shock_cfg.block_length)

        # ============================================================
        # 3) EXECUTE PANIC EXIT
        # ============================================================
        is_in_shock_state = (shock_state["remaining"] > 0) or is_new_shock
        if is_in_shock_state and SHOCK_MODE == "EXIT_ALL" and len(open_positions) > 0:
            panic_bp_real, panic_cash_real = 0.0, 0.0
            panic_t_bp, panic_t_cash = 0.0, 0.0
            
            for pos in open_positions:
                pos.closed = True
                pos.close_ts = dts
                pos.exit_reason = "shock_exit"
                tcost_bp = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10)) if pos.mode == "overlay" else 0.0
                tcost_cash = tcost_bp * pos.scale_dv01
                pos.tcost_bp, pos.tcost_cash = tcost_bp, tcost_cash
                
                panic_t_bp += tcost_bp
                panic_t_cash += tcost_cash
                panic_bp_real += (pos.pnl_bp - tcost_bp)
                panic_cash_real += (pos.pnl_cash - tcost_cash)
                
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": dts, "exit_reason": "shock_exit",
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "pnl_price_bp": pos.pnl_price_bp,
                    "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp": pos.pnl_roll_bp,
                    "pnl_carry_cash": pos.pnl_carry_cash, "pnl_roll_cash": pos.pnl_roll_cash,
                    "mode": pos.mode, "trade_id": pos.meta.get("trade_id"), "side": pos.meta.get("side"),
                    "strat_type": getattr(pos, "strat_type", "reversion") # NEW
                })
            open_positions = []
            
            if metric_type == "REALIZED_CASH": shock_state["history"][-1] += panic_cash_real
            elif metric_type == "REALIZED_BPS": shock_state["history"][-1] += panic_bp_real
            elif metric_type == "MTM_BPS" or metric_type == "BPS": shock_state["history"][-1] -= panic_t_bp
            else: shock_state["history"][-1] -= panic_t_cash

        # ============================================================
        # 4) NEW ENTRIES
        # ============================================================
        if gate: continue
        
        rem_slots = max(0, cr.MAX_CONCURRENT_PAIRS - len(open_positions))
        if rem_slots <= 0: continue

        if mode == "strategy":
            selected = choose_pairs_under_caps(snap_last, rem_slots, PER_BUCKET_DV01_CAP, TOTAL_DV01_CAP, FRONT_END_DV01_CAP, 0.0)
            for (cheap, rich, w_i, w_j) in selected:
                pos = PairPos(dts, cheap, rich, w_i, w_j, decisions_per_day, mode="strategy", strat_type="reversion")
                open_positions.append(pos)
                ledger_rows.append({"decision_ts": dts, "event": "open", "mode": "strategy", "strat_type": "reversion"})
                
        elif mode == "overlay":
            if hedges is None or hedges.empty: continue
            h_here = hedges[hedges["decision_ts"] == dts]
            if h_here.empty: continue
            if float(h_here["dv01"].abs().sum()) > OVERLAY_DV01_TS_CAP: continue
            
            snap_srt = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
            
            DRIFT_GATE = float(getattr(cr, "DRIFT_GATE_BPS", -100.0)) 
            DRIFT_W = float(getattr(cr, "DRIFT_WEIGHT", 0.0)) 

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
                exec_tenor = float(exec_row["tenor_yrs"])

                best_c_row, best_score = None, -999.0

                for _, alt in snap_srt.iterrows():
                    alt_tenor = float(alt["tenor_yrs"])
                    
                    if alt_tenor < ALT_LEG_THRESHOLD: continue
                    if alt_tenor == exec_tenor: continue 
                    if assign_bucket(alt_tenor) == "short" and assign_bucket(exec_tenor) == "long": continue
                    if assign_bucket(exec_tenor) == "short" and assign_bucket(alt_tenor) == "long": continue
                                        
                    z_alt = _to_float(alt["z_comb"])
                    disp = (z_alt - exec_z) if side_s > 0 else (exec_z - z_alt)
                    
                    # --- REVERSION LOGIC ---
                    if current_strat_mode == "reversion":
                        if disp < z_ent_eff: continue
                        if (assign_bucket(alt_tenor)=="short" or assign_bucket(exec_tenor)=="short") and (disp < z_ent_eff + SHORT_EXTRA):
                            continue
                            
                    # --- MOMENTUM LOGIC ---
                    elif current_strat_mode == "momentum":
                        # Look for negative spread (Divergence)
                        if disp > -z_ent_eff: continue

                    c_t, r_t = (alt_tenor, exec_tenor) if z_alt > exec_z else (exec_tenor, alt_tenor)
                    
                    # Fly Gate Logic (Kept Strict to avoid structural breakage)
                    if not (fly_alignment_ok(c_t, 1, snap_srt, zdisp_for_pair=disp) and fly_alignment_ok(r_t, -1, snap_srt, zdisp_for_pair=disp)): continue
                    
                    drift_exec = calc_trade_drift(exec_tenor, side_s, snap_srt)
                    drift_alt = calc_trade_drift(alt_tenor, side_s, snap_srt)
                    
                    if drift_exec == -999.0 or drift_alt == -999.0: continue

                    net_drift_bps = drift_alt - drift_exec 
                    dist_years = abs(alt_tenor - exec_tenor)
                    scaling_factor = dist_years
                    norm_drift_bps = net_drift_bps / scaling_factor
                    
                    if net_drift_bps < DRIFT_GATE: continue 
                    
                    score = disp + (norm_drift_bps * DRIFT_WEIGHT)
                    
                    # In Momentum, we want 'Most Negative' spread.
                    # Standard logic seeks Max Score.
                    # So we flip score sign for Momentum.
                    if current_strat_mode == "momentum":
                        score = -score
                    
                    if score > best_score: 
                        best_score, best_c_row = score, alt
                
                if best_c_row is not None:
                    rate_i, rate_j = None, None
                    ti, tj = tenor_to_ticker(float(best_c_row["tenor_yrs"])), tenor_to_ticker(t_trade)
                    if ti and f"{ti}_mid" in h: rate_i = _to_float(h[f"{ti}_mid"])
                    if tj and f"{tj}_mid" in h: rate_j = _to_float(h[f"{tj}_mid"])
                    
                    if rate_i is None: rate_i = _to_float(best_c_row["rate"])
                    if rate_j is None: rate_j = _to_float(exec_row["rate"])

                    pos = PairPos(dts, best_c_row, exec_row, side_s*1.0, side_s*-1.0, decisions_per_day, 
                                  scale_dv01=float(h["dv01"]), mode="overlay", 
                                  meta={"trade_id": h.get("trade_id"), "side": h.get("side")},
                                  entry_rate_i=rate_i, entry_rate_j=rate_j,
                                  strat_type=current_strat_mode) # Pass Type
                    
                    open_positions.append(pos)
                    ledger_rows.append({"decision_ts": dts, "event": "open", "mode": "overlay", "strat_type": current_strat_mode})

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions






# ------------------------
# Pair object (Modified for Regime Switching)
# ------------------------
class PairPos:
    def __init__(self, open_ts, cheap_row, rich_row, w_i, w_j, decisions_per_day, *, 
                 scale_dv01=1.0, mode="strategy", meta=None, dir_sign=None, 
                 entry_rate_i=None, entry_rate_j=None, 
                 # NEW: Strategy Type ('reversion' or 'momentum')
                 strat_type="reversion"):
        
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

        # NEW: Store Strategy Type
        self.strat_type = strat_type 

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
        
        # PnL Components (Cumulative)
        self.pnl_price_cash = 0.0
        self.pnl_carry_cash = 0.0
        self.pnl_roll_cash = 0.0
        self.pnl_cash = 0.0
        self.pnl_bp = 0.0
        
        # Breakdown Bps
        self.pnl_price_bp = 0.0
        self.pnl_carry_bp = 0.0
        self.pnl_roll_bp = 0.0

        self.tcost_bp, self.tcost_cash = 0.0, 0.0
        self.decisions_per_day = decisions_per_day
        self.age_decisions = 0
        self.bucket_i, self.bucket_j = assign_bucket(self.tenor_i), assign_bucket(self.tenor_j)
        self.last_zspread, self.last_z_dir, self.conv_pnl_proxy = self.entry_zspread, self.entry_z_dir, 0.0

    def _update_overlay_dv01(self, decision_ts):
        """Linearly decay DV01 based on Act/360 time passed."""
        if not isinstance(decision_ts, pd.Timestamp): return
        days = max(0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        yr_pass = days / 360.0
        self.rem_tenor_i = max(self.tenor_i_orig - yr_pass, 0.0)
        self.rem_tenor_j = max(self.tenor_j_orig - yr_pass, 0.0)
        
        # Avoid div by zero
        fi = self.rem_tenor_i / max(self.tenor_i_orig, 1e-6)
        fj = self.rem_tenor_j / max(self.tenor_j_orig, 1e-6)
        self.dv01_i_curr = self.dv01_i_entry * fi
        self.dv01_j_curr = self.dv01_j_entry * fj

    def mark(self, snap_last: pd.DataFrame, decision_ts: Optional[pd.Timestamp] = None):
        # 0. Update Decay (Overlay Only)
        if self.mode == "overlay" and decision_ts:
            self._update_overlay_dv01(decision_ts)

        # 1. Get Market Data (Rates & Funding)
        ri = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "rate"])
        rj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "rate"])
        r_float = _get_funding_rate(snap_last)
        self.last_rate_i, self.last_rate_j = ri, rj

        # 2. Price PnL
        pnl_price_i = (self.entry_rate_i - ri) * 100.0 * self.dv01_i_curr
        pnl_price_j = (self.entry_rate_j - rj) * 100.0 * self.dv01_j_curr
        self.pnl_price_cash = pnl_price_i + pnl_price_j

        # 3. Carry PnL
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
        sort_idx = np.argsort(xp)
        xp_sorted = xp[sort_idx]
        fp_sorted = fp[sort_idx]
        
        t_roll_i = max(0.0, self.tenor_i_orig - (days_total/360.0))
        y_roll_i = np.interp(t_roll_i, xp_sorted, fp_sorted)
        roll_gain_i = (ri - y_roll_i) * 100.0 * self.dv01_i_curr
        
        t_roll_j = max(0.0, self.tenor_j_orig - (days_total/360.0))
        y_roll_j = np.interp(t_roll_j, xp_sorted, fp_sorted)
        roll_gain_j = (rj - y_roll_j) * 100.0 * self.dv01_j_curr
        
        self.pnl_roll_cash = roll_gain_i + roll_gain_j

        # 5. Total Cash
        self.pnl_cash = self.pnl_price_cash + self.pnl_carry_cash + self.pnl_roll_cash
        
        # 6. Total Bps
        if self.scale_dv01 != 0.0 and np.isfinite(self.scale_dv01):
            self.pnl_bp = self.pnl_cash / self.scale_dv01
            self.pnl_price_bp = self.pnl_price_cash / self.scale_dv01
            self.pnl_carry_bp = self.pnl_carry_cash / self.scale_dv01
            self.pnl_roll_bp = self.pnl_roll_cash / self.scale_dv01
        else:
            self.pnl_bp = 0.0; self.pnl_price_bp = 0.0; self.pnl_carry_bp = 0.0; self.pnl_roll_bp = 0.0

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
# RUN MONTH (Modified for Reversion/Momentum Switch)
# ------------------------
def run_month(
    yymm: str,
    *,
    decision_freq: str | None = None,
    open_positions: Optional[List[PairPos]] | None = None,
    carry_in: bool = True,
    mode: str = "strategy",
    hedges: Optional[pd.DataFrame] = None,
    overlay_use_caps: Optional[bool] = None,
    regime_mask: Optional[pd.Series] = None,        
    hybrid_signals: Optional[pd.DataFrame] = None, 
    shock_cfg: Optional[ShockConfig] = None,
    shock_state: Optional[Dict] = None 
):
    import math
    try: np
    except NameError: import numpy as np

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
    
    # Overlay specific configs
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

    # --- Shock Filter State ---
    if shock_state is None:
        shock_state = {"history": [], "dates": [], "remaining": 0}
    
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
    
    EXEC_LEG_THRESHOLD = float(getattr(cr, "EXEC_LEG_TENOR_YEARS", 0.084))
    ALT_LEG_THRESHOLD  = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0))
    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    
    SHORT_EXTRA = SHORT_END_EXTRA_Z 

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
        # A) START-OF-DAY GATING & REGIME DETECTION
        # ============================================================
        was_shock_active_at_open = (shock_state["remaining"] > 0)
        
        if shock_state["remaining"] > 0:
            shock_state["remaining"] -= 1

        # Check Regime Mask
        regime_ok = True
        if regime_mask is not None:
            regime_ok = bool(regime_mask.at[dts]) if dts in regime_mask.index else False
        
        # --- NEW: MODE SWITCHING ---
        # If Regime is OK -> Mean Reversion (Standard)
        # If Regime is BAD -> Momentum (Flipped)
        # Note: Shock active typically kills trading, but here we focus on Regime.
        current_strat_mode = "reversion" if regime_ok else "momentum"
        
        # If Shock is active, we might still want to gate completely (or maybe momentum overrides shock?)
        # For now, let's say Shock still blocks new entries completely, but Regime flips logic.
        gate = was_shock_active_at_open
        if SHOCK_MODE == "EXIT_ALL" and (shock_state["remaining"] > 0): gate = True
        
        # ============================================================
        # 1) MARK POSITIONS & EXITS
        # ============================================================
        period_pnl_cash = 0.0        
        period_pnl_bps_mtm = 0.0     
        period_pnl_bps_realized = 0.0 
        period_pnl_cash_realized = 0.0
        
        still_open: list[PairPos] = []
        
        for pos in open_positions:
            prev_cash, prev_bp = pos.pnl_cash, pos.pnl_bp
            
            # Mark
            zsp = pos.mark(snap_last, decision_ts=dts)
            
            # Incremental MTM
            period_pnl_cash += (pos.pnl_cash - prev_cash)
            period_pnl_bps_mtm += (pos.pnl_bp - prev_bp)

            entry_z = pos.entry_zspread
            exit_flag = None

            if np.isfinite(zsp) and np.isfinite(entry_z):
                # entry_dir is usually Positive (Abs distance from mean)
                entry_dir = getattr(pos, "entry_z_dir", pos.dir_sign * entry_z)
                curr_dir = getattr(pos, "last_z_dir", pos.dir_sign * zsp)

                if np.isfinite(entry_dir) and np.isfinite(curr_dir):
                    
                    # --- REVERSION LOGIC (Standard) ---
                    if pos.strat_type == "reversion":
                        sign_entry = np.sign(entry_dir)
                        sign_curr = np.sign(curr_dir)
                        same_side = (sign_entry != 0) and (sign_entry == sign_curr)
                        moved_towards_zero = abs(curr_dir) <= abs(entry_dir)
                        within_exit_band = abs(curr_dir) <= Z_EXIT
                        dz_dir = curr_dir - entry_dir
                        moved_away = same_side and (abs(curr_dir) >= abs(entry_dir)) and (abs(dz_dir) >= Z_STOP)

                        if same_side and moved_towards_zero and within_exit_band:
                            exit_flag = "reversion"  # Profit
                        elif moved_away:
                            exit_flag = "stop"       # Loss

                    # --- MOMENTUM LOGIC (Flipped) ---
                    elif pos.strat_type == "momentum":
                        # In Momentum, we WANT divergence.
                        # Stop Loss = Reverting back to mean (Convergence)
                        # Take Profit = Exploding further out (Divergence)
                        
                        # Check Convergence (Stop Loss for Momentum)
                        # If we move closer to 0 than entry, we are losing the trend.
                        if abs(curr_dir) < abs(entry_dir):
                            # Use Z_EXIT as a tolerance? 
                            # Strict stop: if abs(curr) < abs(entry) - buffer
                            exit_flag = "stop_reversion" 
                        
                        # Check Divergence (Profit for Momentum)
                        # If we move further out by Z_STOP amount
                        elif abs(curr_dir) > (abs(entry_dir) + Z_STOP):
                             exit_flag = "profit_momentum"

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
                "strat_type": getattr(pos, "strat_type", "reversion") # Log strategy type
            })

            if pos.closed:
                tcost_bp = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10)) if pos.mode == "overlay" else 0.0
                tcost_cash = tcost_bp * pos.scale_dv01
                
                pos.tcost_bp = tcost_bp
                pos.tcost_cash = tcost_cash
                
                period_pnl_cash -= tcost_cash
                period_pnl_bps_mtm -= tcost_bp
                
                period_pnl_bps_realized += (pos.pnl_bp - tcost_bp)
                period_pnl_cash_realized += (pos.pnl_cash - tcost_cash)

                closed_rows.append({
                    "open_ts": pos.open_ts, 
                    "close_ts": pos.close_ts, 
                    "exit_reason": pos.exit_reason,
                    "tenor_i": pos.tenor_i, 
                    "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i, 
                    "w_j": pos.w_j,
                    "entry_zspread": pos.entry_zspread,
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, 
                    "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "mode": pos.mode,
                    "strat_type": pos.strat_type, # Save strat type
                    "trade_id": pos.meta.get("trade_id"),
                    "side": pos.meta.get("side"),
                })
            else:
                still_open.append(pos)
        
        open_positions = still_open
        
        # ============================================================
        # 2) UPDATE HISTORY & DETECT NEW SHOCK
        # ============================================================
        # (Shock logic remains identical - Endogenous PnL check)
        metric_type = getattr(shock_cfg, "metric_type", "MTM_BPS") if shock_cfg else "MTM_BPS"
        if metric_type == "REALIZED_CASH": metric_val = period_pnl_cash_realized
        elif metric_type == "REALIZED_BPS": metric_val = period_pnl_bps_realized
        elif metric_type == "MTM_BPS" or metric_type == "BPS": metric_val = period_pnl_bps_mtm
        else: metric_val = period_pnl_cash

        shock_state["history"].append(metric_val)
        shock_state["dates"].append(dts)
        
        is_new_shock = False
        if shock_cfg is not None:
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
                
                # (Residual logic skipped for brevity, assumed same as original)
        
        if is_new_shock: shock_state["remaining"] = int(shock_cfg.block_length)

        # ============================================================
        # 3) EXECUTE PANIC EXIT
        # ============================================================
        # (Panic logic remains identical)
        is_in_shock_state = (shock_state["remaining"] > 0) or is_new_shock
        if is_in_shock_state and SHOCK_MODE == "EXIT_ALL" and len(open_positions) > 0:
             # ... Panic exit code ...
             pass # (Implied same as original)

        # ============================================================
        # 4) NEW ENTRIES
        # ============================================================
        if gate: continue
        
        rem_slots = max(0, cr.MAX_CONCURRENT_PAIRS - len(open_positions))
        if rem_slots <= 0: continue

        if mode == "strategy":
            # Strategy mode logic (omitted or kept vanilla as requested)
            pass
                
        elif mode == "overlay":
            if hedges is None or hedges.empty: continue
            h_here = hedges[hedges["decision_ts"] == dts]
            if h_here.empty: continue
            if float(h_here["dv01"].abs().sum()) > OVERLAY_DV01_TS_CAP: continue
            
            snap_srt = snap_last.sort_values("tenor_yrs").reset_index(drop=True)
            
            DRIFT_GATE = float(getattr(cr, "DRIFT_GATE_BPS", -100.0)) 
            DRIFT_W = float(getattr(cr, "DRIFT_WEIGHT", 0.0)) 

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
                exec_tenor = float(exec_row["tenor_yrs"])

                best_c_row, best_score = None, -999.0

                for _, alt in snap_srt.iterrows():
                    alt_tenor = float(alt["tenor_yrs"])
                    
                    if alt_tenor < ALT_LEG_THRESHOLD: continue
                    if alt_tenor == exec_tenor: continue 
                    if assign_bucket(alt_tenor) == "short" and assign_bucket(exec_tenor) == "long": continue
                    if assign_bucket(exec_tenor) == "short" and assign_bucket(alt_tenor) == "long": continue
                                        
                    z_alt = _to_float(alt["z_comb"])
                    disp = (z_alt - exec_z) if side_s > 0 else (exec_z - z_alt)
                    
                    # --- REVERSION LOGIC (Standard) ---
                    if current_strat_mode == "reversion":
                        if disp < z_ent_eff: continue
                        if (assign_bucket(alt_tenor)=="short" or assign_bucket(exec_tenor)=="short") and (disp < z_ent_eff + SHORT_EXTRA):
                            continue
                    
                    # --- MOMENTUM LOGIC (Flipped) ---
                    # We look for DISLOCATION in the opposite direction
                    # disp defined as: Cheap - Rich.
                    # Reversion wants Positive (Buy Cheap).
                    # Momentum wants Negative (Buy Rich, betting it gets Richer / Sell Cheap betting it gets Cheaper).
                    elif current_strat_mode == "momentum":
                        if disp > -z_ent_eff: continue # Must be VERY negative
                        # (Skip short bucket extra logic for momentum for now, or apply symmetrically)

                    c_t, r_t = (alt_tenor, exec_tenor) if z_alt > exec_z else (exec_tenor, alt_tenor)
                    
                    # Fly Gates (Check if flies allow this direction)
                    # Note: For Momentum, we are entering the 'bad' side of the trade, so we check if that is allowed.
                    # This implies checking the Fly logic is still valid for the flow we are putting on.
                    if not (fly_alignment_ok(c_t, 1, snap_srt, zdisp_for_pair=disp) and fly_alignment_ok(r_t, -1, snap_srt, zdisp_for_pair=disp)): continue
                    
                    drift_exec = calc_trade_drift(exec_tenor, side_s, snap_srt)
                    drift_alt = calc_trade_drift(alt_tenor, side_s, snap_srt)
                    
                    if drift_exec == -999.0 or drift_alt == -999.0: continue

                    net_drift_bps = drift_alt - drift_exec 
                    dist_years = abs(alt_tenor - exec_tenor)
                    scaling_factor = dist_years
                    norm_drift_bps = net_drift_bps / scaling_factor
                    
                    # Drift Gate (Flip for Momentum?)
                    # Usually if betting on momentum (divergence), we might ignore drift or want it to align.
                    # For simplicity: Keep drift check standard (we don't want to bleed carry even in momentum).
                    if net_drift_bps < DRIFT_GATE: continue 
                    
                    score = disp + (norm_drift_bps * DRIFT_WEIGHT)
                    
                    # In Momentum, 'disp' is negative. Score will be negative.
                    # We want the MOST negative score (Biggest divergence).
                    # But 'best_score' logic searches for Max.
                    # So for Momentum, we might want to invert score or logic.
                    if current_strat_mode == "momentum":
                        # Invert score so 'most negative' becomes 'highest positive'
                        score = -score
                    
                    if score > best_score: 
                        best_score, best_c_row = score, alt
                
                if best_c_row is not None:
                    rate_i, rate_j = None, None
                    ti, tj = tenor_to_ticker(float(best_c_row["tenor_yrs"])), tenor_to_ticker(t_trade)
                    if ti and f"{ti}_mid" in h: rate_i = _to_float(h[f"{ti}_mid"])
                    if tj and f"{tj}_mid" in h: rate_j = _to_float(h[f"{tj}_mid"])
                    
                    if rate_i is None: rate_i = _to_float(best_c_row["rate"])
                    if rate_j is None: rate_j = _to_float(exec_row["rate"])

                    pos = PairPos(dts, best_c_row, exec_row, side_s*1.0, side_s*-1.0, decisions_per_day, 
                                  scale_dv01=float(h["dv01"]), mode="overlay", 
                                  meta={"trade_id": h.get("trade_id"), "side": h.get("side")},
                                  entry_rate_i=rate_i, entry_rate_j=rate_j,
                                  # PASS STRATEGY MODE
                                  strat_type=current_strat_mode)
                    
                    open_positions.append(pos)
                    ledger_rows.append({"decision_ts": dts, "event": "open", "mode": "overlay", "strat": current_strat_mode})

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from math import sqrt
import sys

# Import config
import cr_config as cr

# ==============================================================================
# 0. MODE SELECTION & CONFIGURATION
# ==============================================================================
# Detect mode from command line argument
REPORT_MODE = "BPS"  # Default
if len(sys.argv) > 1 and sys.argv[1].lower() == "cash":
    REPORT_MODE = "CASH"

print(f"[INIT] Running Report in {REPORT_MODE} Mode...")

# Configuration Mapping
if REPORT_MODE == "CASH":
    cfg = {
        'col_net': 'pnl_net_cash',
        'col_price': 'pnl_price_cash',
        'col_carry': 'pnl_carry_cash',
        'col_roll': 'pnl_roll_cash',
        'col_tcost': 'tcost_cash',
        'unit': '$',
        'label': 'USD',
        'fmt': '${:,.0f}'  # e.g., $1,500
    }
else:
    cfg = {
        'col_net': 'pnl_net_bp',
        'col_price': 'pnl_price_bp',
        'col_carry': 'pnl_carry_bp',
        'col_roll': 'pnl_roll_bp',
        'col_tcost': 'tcost_bp',
        'unit': 'bps',
        'label': 'Basis Points',
        'fmt': '{:,.1f}'   # e.g., 12.5
    }

# ==============================================================================
# STYLE & HELPERS
# ==============================================================================
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
colors = {'pnl': '#2ecc71', 'dd': '#e74c3c', 'price': '#3498db', 'carry': '#f1c40f', 'roll': '#9b59b6'}

def safe_divide(n, d, default=0.0):
    return n / d if d != 0 else default

def fmt_val(val):
    return cfg['fmt'].format(val)

def human_format(x, pos):
    """Formats 1500 as 1.5k, 1000000 as 1M"""
    if x == 0: return '0'
    abs_x = abs(x)
    if abs_x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif abs_x >= 1e3:
        return f'{x*1e-3:.0f}k'
    else:
        return f'{x:.0f}'

def assign_bucket_simple(tenor):
    """Re-derives buckets for analysis if not present"""
    if pd.isna(tenor): return "Unknown"
    if tenor < 2.0: return "Short (<2Y)"
    if tenor < 5.0: return "Front (2-5Y)"
    if tenor < 10.0: return "Belly (5-10Y)"
    return "Long (10Y+)"

# ==============================================================================
# 1. DATA LOADING & PREP
# ==============================================================================
out_dir = Path(cr.PATH_OUT)
suffix = getattr(cr, "OUT_SUFFIX", "")
pos_path = out_dir / f"positions_ledger{suffix}.parquet"

if not pos_path.exists():
    raise FileNotFoundError(f"[ERROR] {pos_path} not found.")

print(f"[LOAD] Reading {pos_path}...")
df = pd.read_parquet(pos_path)

# Filter Overlay only
if "mode" in df.columns:
    df = df[df["mode"] == "overlay"].copy()

# Sort by Close Time (Critical for Equity Curve)
df["close_ts"] = pd.to_datetime(df["close_ts"])
df = df.sort_values("close_ts").reset_index(drop=True)

# Derive Bucket if missing (Assuming 'tenor_i' exists)
if "bucket" not in df.columns and "tenor_i" in df.columns:
    df["bucket"] = df["tenor_i"].apply(assign_bucket_simple)

# --- Create Equity Curves using the Selected Column ---
df["equity_curve"] = df[cfg['col_net']].cumsum()

# ==============================================================================
# 2. METRICS ENGINE
# ==============================================================================

# --- A. Trade Statistics ---
total_trades = len(df)
win_trades = df[df[cfg['col_net']] > 0]
loss_trades = df[df[cfg['col_net']] <= 0]

win_rate = safe_divide(len(win_trades), total_trades)
gross_win = win_trades[cfg['col_net']].sum()
gross_loss = abs(loss_trades[cfg['col_net']].sum())
profit_factor = safe_divide(gross_win, gross_loss)

avg_trade = df[cfg['col_net']].mean()
avg_win = win_trades[cfg['col_net']].mean()
avg_loss = loss_trades[cfg['col_net']].mean()

# Hold Times
if "days_held_equiv" in df.columns:
    avg_hold = df["days_held_equiv"].mean()
else:
    avg_hold = (df["close_ts"] - pd.to_datetime(df["open_ts"])).dt.days.mean()

# --- B. Time-Series Statistics (Sharpe/Sortino) ---
# Resample to Daily for correct Sharpe math
daily_idx = pd.date_range(start=df["close_ts"].min(), end=df["close_ts"].max(), freq='D')
daily_pnl = df.set_index("close_ts")[cfg['col_net']].resample('D').sum().reindex(daily_idx, fill_value=0.0)

ANN_FACTOR = 252
mean_daily = daily_pnl.mean()
std_daily = daily_pnl.std()
downside_daily = daily_pnl[daily_pnl < 0].std()

sharpe = safe_divide(mean_daily * ANN_FACTOR, std_daily * sqrt(ANN_FACTOR))
sortino = safe_divide(mean_daily * ANN_FACTOR, downside_daily * sqrt(ANN_FACTOR))

cum_equity_daily = daily_pnl.cumsum()
running_max = cum_equity_daily.cummax()
dd_series = cum_equity_daily - running_max
max_dd = dd_series.min()
calmar = safe_divide(cum_equity_daily.iloc[-1], abs(max_dd))

# --- C. Attribution ---
total_net = df[cfg['col_net']].sum()
total_price = df[cfg['col_price']].sum()
total_carry = df[cfg['col_carry']].sum()
total_roll = df[cfg['col_roll']].sum()

# ==============================================================================
# 3. PRINT PROFESSIONAL TABLE
# ==============================================================================
print("\n" + "="*60)
print(f"{f'SYSTEMATIC OVERLAY REPORT ({REPORT_MODE})':^60}")
print("="*60)

stats = [
    (f"Total Net PnL ({cfg['unit']})", fmt_val(total_net)),
    ("Total Trades", f"{total_trades}"),
    ("Win Rate", f"{win_rate:.1%}"),
    ("-" * 20, "-" * 20),
    ("Profit Factor", f"{profit_factor:.2f}"),
    (f"Avg Trade ({cfg['unit']})", fmt_val(avg_trade)),
    ("Avg Win / Avg Loss", f"{abs(avg_win/avg_loss):.2f}"),
    ("Avg Hold (Days)", f"{avg_hold:.1f}"),
    ("-" * 20, "-" * 20),
    ("Sharpe Ratio (Ann.)", f"{sharpe:.2f}"),
    ("Sortino Ratio (Ann.)", f"{sortino:.2f}"),
    (f"Max Drawdown ({cfg['unit']})", fmt_val(max_dd)),
    ("Return / DD (Calmar)", f"{abs(calmar):.2f}"),
]

col_width = 35
for label, val in stats:
    if "---" in label:
        print(f"{label}   {val}")
    else:
        print(f"{label:<{col_width}} {val:>15}")
print("="*60)
print(f"ATTRIBUTION:\n Price: {fmt_val(total_price)} | Carry: {fmt_val(total_carry)} | Roll: {fmt_val(total_roll)}")
print("="*60 + "\n")

# ==============================================================================
# 4. PLOTTING SUITE
# ==============================================================================

# --- FIGURE 1: EXECUTIVE DASHBOARD ---
fig = plt.figure(figsize=(16, 12), constrained_layout=True)
gs = fig.add_gridspec(3, 2)

# 1. Equity Curve
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum_equity_daily.index, cum_equity_daily.values, color=colors['pnl'], lw=2)
ax1.fill_between(cum_equity_daily.index, cum_equity_daily.values, 0, color=colors['pnl'], alpha=0.1)
ax1.set_title(f"Realized Equity Curve (Net {cfg['label']})", fontweight='bold')
ax1.set_ylabel(f"Cumulative {cfg['label']}")
ax1.margins(x=0)
if REPORT_MODE == "CASH":
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(human_format))

# 2. Drawdown
ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
ax2.fill_between(dd_series.index, dd_series.values, 0, color=colors['dd'], alpha=0.3)
ax2.plot(dd_series.index, dd_series.values, color=colors['dd'], lw=1)
ax2.set_title("Drawdown Profile", fontsize=11)
ax2.set_ylabel(f"Drawdown ({cfg['unit']})")
if REPORT_MODE == "CASH":
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(human_format))

# 3. Monthly Returns (Bar)
ax3 = fig.add_subplot(gs[2, 0])
monthly = df.set_index("close_ts")[cfg['col_net']].resample('M').sum()
clrs = ['#e74c3c' if x < 0 else '#2ecc71' for x in monthly.values]
# Only show last 12 months if too many
plot_monthly = monthly.iloc[-24:] if len(monthly) > 24 else monthly
plot_monthly.index = plot_monthly.index.strftime('%Y-%m')
plot_monthly.plot(kind='bar', ax=ax3, color=clrs[-len(plot_monthly):], width=0.8)
ax3.set_title(f"Monthly Net PnL ({cfg['unit']})")
ax3.set_xlabel("")
ax3.tick_params(axis='x', rotation=45, labelsize=9)

# 4. PnL Distribution (Histogram) - FIXED LABELS
ax4 = fig.add_subplot(gs[2, 1])
sns.histplot(df[cfg['col_net']], kde=True, ax=ax4, color='#34495e', bins=30)
ax4.axvline(0, color='black', linestyle='--')
ax4.axvline(avg_trade, color=colors['pnl'], linestyle='-', label=f'Mean: {human_format(avg_trade,0)}')
ax4.set_title(f"Trade Distribution ({cfg['unit']})")
ax4.set_xlabel(f"PnL Net {cfg['label']}") # Explicit Label
# Use k formatter
ax4.xaxis.set_major_formatter(mtick.FuncFormatter(human_format))
ax4.legend()

plt.show()

# --- FIGURE 2: ADVANCED ANALYTICS (New Requests) ---
fig2 = plt.figure(figsize=(16, 10), constrained_layout=True)
gs2 = fig2.add_gridspec(2, 2)

# 1. PnL Attribution (Drift vs Price)
ax_attr = fig2.add_subplot(gs2[0, 0])
cum_price = df[cfg['col_price']].cumsum()
cum_drift = (df[cfg['col_carry']] + df[cfg['col_roll']]).cumsum()
ax_attr.plot(df["close_ts"], cum_price, label="Price (Timing)", color=colors['price'], alpha=0.8)
ax_attr.plot(df["close_ts"], cum_drift, label="Drift (Carry+Roll)", color=colors['carry'], lw=2)
ax_attr.plot(df["close_ts"], df["equity_curve"], label="Total Net", color='black', linestyle='--')
ax_attr.set_title("PnL Source Attribution", fontweight='bold')
ax_attr.legend()
if REPORT_MODE == "CASH":
    ax_attr.yaxis.set_major_formatter(mtick.FuncFormatter(human_format))

# 2. PnL by Curve Sector (Tenor Bucket)
ax_buck = fig2.add_subplot(gs2[0, 1])
if "bucket" in df.columns:
    bucket_perf = df.groupby("bucket")[cfg['col_net']].sum().sort_values()
    bucket_perf.plot(kind='barh', ax=ax_buck, color='#2980b9')
    ax_buck.set_title(f"Total PnL by Curve Sector ({cfg['unit']})")
    ax_buck.set_xlabel(cfg['unit'])
    if REPORT_MODE == "CASH":
        ax_buck.xaxis.set_major_formatter(mtick.FuncFormatter(human_format))

# 3. Monthly Heatmap (The 'Calendar' View)
ax_heat = fig2.add_subplot(gs2[1, 0])
heat_data = daily_pnl.resample('M').sum()
heat_df = pd.DataFrame({'year': heat_data.index.year, 'month': heat_data.index.month, 'pnl': heat_data.values})
heat_piv = heat_df.pivot(index='year', columns='month', values='pnl')

# --- FIX START: Custom k-formatting for Cash ---
if REPORT_MODE == "CASH":
    # Create a DataFrame of strings like "15k", "-2k"
    annot_labels = heat_piv.applymap(lambda x: f'{x/1000:.0f}k' if pd.notnull(x) else '')
    fmt_param = ""  # Formatting is already done in the strings
else:
    # Standard behavior for Bps
    annot_labels = True 
    fmt_param = ".0f"
# -----------------------------------------------

sns.heatmap(heat_piv, ax=ax_heat, cmap="RdYlGn", center=0, annot=annot_labels, fmt=fmt_param, cbar=False)
ax_heat.set_title(f"Monthly Returns Heatmap ({cfg['unit']})")


# 4. Rolling Sharpe Ratio (6-Month Lookback)
ax_roll = fig2.add_subplot(gs2[1, 1])
roll_window = 126 # ~6 months
rolling_sharpe = daily_pnl.rolling(roll_window).mean() / daily_pnl.rolling(roll_window).std() * sqrt(252)
ax_roll.plot(rolling_sharpe.index, rolling_sharpe, color='#8e44ad')
ax_roll.axhline(0, color='black', lw=0.5)
ax_roll.axhline(1, color='gray', linestyle='--', alpha=0.5, label="Target > 1.0")
ax_roll.set_title(f"Rolling {roll_window}D Sharpe Ratio")
ax_roll.legend()

plt.show()







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
            
        # This is the directional Z (Positive = Distance from Mean). 
        # Mean reversion implies this value should decrease (go to 0 or negative).
        self.entry_z_dir = self.dir_sign * self.entry_zspread

        # --- NEW: Stagnation Check State ---
        # We initialize the "last check" as the entry level.
        self.z_at_last_check = self.entry_z_dir

        self.closed, self.close_ts, self.exit_reason = False, None, None
        self.scale_dv01, self.mode = float(scale_dv01), str(mode)
        self.meta = meta or {}
        self.initial_dv01 = self.scale_dv01
        
        self.dv01_i_entry = self.scale_dv01 * self.w_i
        self.dv01_j_entry = self.scale_dv01 * self.w_j
        self.dv01_i_curr, self.dv01_j_curr = self.dv01_i_entry, self.dv01_j_entry
        self.rem_tenor_i, self.rem_tenor_j = self.tenor_i_orig, self.tenor_j_orig
        
        self.last_mark_ts = open_ts 
        
        # PnL Components (Cumulative)
        self.pnl_price_cash = 0.0
        self.pnl_carry_cash = 0.0
        self.pnl_roll_cash = 0.0
        self.pnl_cash = 0.0
        self.pnl_bp = 0.0
        
        # Breakdown Bps
        self.pnl_price_bp = 0.0
        self.pnl_carry_bp = 0.0
        self.pnl_roll_bp = 0.0

        self.tcost_bp, self.tcost_cash = 0.0, 0.0
        self.decisions_per_day = decisions_per_day
        self.age_decisions = 0
        self.bucket_i, self.bucket_j = assign_bucket(self.tenor_i), assign_bucket(self.tenor_j)
        self.last_zspread, self.last_z_dir, self.conv_pnl_proxy = self.entry_zspread, self.entry_z_dir, 0.0

    def _update_overlay_dv01(self, decision_ts):
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

    def mark(self, snap_last: pd.DataFrame, decision_ts: Optional[pd.Timestamp] = None):
        if self.mode == "overlay" and decision_ts:
            self._update_overlay_dv01(decision_ts)

        ri = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "rate"])
        rj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "rate"])
        r_float = _get_funding_rate(snap_last)
        self.last_rate_i, self.last_rate_j = ri, rj

        # Price PnL
        pnl_price_i = (self.entry_rate_i - ri) * 100.0 * self.dv01_i_curr
        pnl_price_j = (self.entry_rate_j - rj) * 100.0 * self.dv01_j_curr
        self.pnl_price_cash = pnl_price_i + pnl_price_j

        # Carry PnL (Incremental)
        if decision_ts and self.last_mark_ts:
            dt_days = max(0.0, (decision_ts.normalize() - self.last_mark_ts.normalize()).days)
            if dt_days > 0:
                inc_carry_i = (self.entry_rate_i - r_float) * 100.0 * self.dv01_i_curr * (dt_days / 360.0)
                inc_carry_j = (self.entry_rate_j - r_float) * 100.0 * self.dv01_j_curr * (dt_days / 360.0)
                self.pnl_carry_cash += (inc_carry_i + inc_carry_j)
            self.last_mark_ts = decision_ts

        # Roll-Down PnL
        days_total = 0.0
        if decision_ts:
            days_total = max(0.0, (decision_ts.normalize() - self.open_ts.normalize()).days)
        
        xp = snap_last["tenor_yrs"].values
        fp = snap_last["rate"].values
        sort_idx = np.argsort(xp)
        xp_sorted = xp[sort_idx]
        fp_sorted = fp[sort_idx]
        
        t_roll_i = max(0.0, self.tenor_i_orig - (days_total/360.0))
        y_roll_i = np.interp(t_roll_i, xp_sorted, fp_sorted)
        roll_gain_i = (ri - y_roll_i) * 100.0 * self.dv01_i_curr
        
        t_roll_j = max(0.0, self.tenor_j_orig - (days_total/360.0))
        y_roll_j = np.interp(t_roll_j, xp_sorted, fp_sorted)
        roll_gain_j = (rj - y_roll_j) * 100.0 * self.dv01_j_curr
        
        self.pnl_roll_cash = roll_gain_i + roll_gain_j

        # Totals
        self.pnl_cash = self.pnl_price_cash + self.pnl_carry_cash + self.pnl_roll_cash
        
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

        # Update Z stats
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


def run_month(
    yymm: str,
    *,
    decision_freq: str | None = None,
    open_positions: Optional[List[PairPos]] | None = None,
    carry_in: bool = True,
    mode: str = "strategy",
    hedges: Optional[pd.DataFrame] = None,
    overlay_use_caps: Optional[bool] = None,
    regime_mask: Optional[pd.Series] = None,        
    hybrid_signals: Optional[pd.DataFrame] = None, 
    shock_cfg: Optional[ShockConfig] = None,
    shock_state: Optional[Dict] = None 
):
    import math
    try: np
    except NameError: import numpy as np

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
    
    # --- Overlay Configs ---
    OVERLAY_MAX_HOLD_DV01_MED = float(getattr(cr, "OVERLAY_MAX_HOLD_DV01_MED", 20_000.0))
    OVERLAY_MAX_HOLD_DV01_HI = float(getattr(cr, "OVERLAY_MAX_HOLD_DV01_HI", 50_000.0))
    OVERLAY_MAX_HOLD_DAYS_MED = float(getattr(cr, "OVERLAY_MAX_HOLD_DAYS_MED", 5.0))
    OVERLAY_MAX_HOLD_DAYS_HI = float(getattr(cr, "OVERLAY_MAX_HOLD_DAYS_HI", 2.0))
    
    # NEW: Stagnation Check Config
    _check_days = getattr(cr, "OVERLAY_MIN_CHECK_DAYS", None)
    min_check_decisions = 0
    if _check_days is not None and int(_check_days) > 0:
        min_check_decisions = int(_check_days) * decisions_per_day

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

    # --- Shock Filter State ---
    if shock_state is None:
        shock_state = {"history": [], "dates": [], "remaining": 0}
    
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
    
    EXEC_LEG_THRESHOLD = float(getattr(cr, "EXEC_LEG_TENOR_YEARS", 0.084))
    ALT_LEG_THRESHOLD  = float(getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0))
    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    
    SHORT_EXTRA = SHORT_END_EXTRA_Z 

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
        # A) START-OF-DAY GATING (NO LOOKAHEAD)
        # ============================================================
        was_shock_active_at_open = (shock_state["remaining"] > 0)
        
        if shock_state["remaining"] > 0:
            shock_state["remaining"] -= 1

        # ============================================================
        # 1) MARK POSITIONS & NATURAL EXITS
        # ============================================================
        period_pnl_cash = 0.0        
        period_pnl_bps_mtm = 0.0     
        period_pnl_bps_realized = 0.0 
        period_pnl_cash_realized = 0.0
        
        still_open: list[PairPos] = []
        
        for pos in open_positions:
            prev_cash, prev_bp = pos.pnl_cash, pos.pnl_bp
            
            # Mark
            zsp = pos.mark(snap_last, decision_ts=dts)
            
            # Incremental MTM
            period_pnl_cash += (pos.pnl_cash - prev_cash)
            period_pnl_bps_mtm += (pos.pnl_bp - prev_bp)

            entry_z = pos.entry_zspread
            exit_flag = None

            if np.isfinite(zsp) and np.isfinite(entry_z):
                entry_dir = getattr(pos, "entry_z_dir", pos.dir_sign * entry_z)
                curr_dir = getattr(pos, "last_z_dir", pos.dir_sign * zsp)

                if np.isfinite(entry_dir) and np.isfinite(curr_dir):
                    
                    # --- NEW: Stagnation Check (Overlay Only) ---
                    # Logic: If it's a check day, and Z has not improved (is >=) compared 
                    # to the last checkpoint, exit.
                    if exit_flag is None and pos.mode == "overlay" and min_check_decisions > 0:
                        if pos.age_decisions > 0 and (pos.age_decisions % min_check_decisions == 0):
                            # Note: curr_dir is Signed Positive (Distance from Mean).
                            # Improvement = Lower Value.
                            if curr_dir >= pos.z_at_last_check:
                                exit_flag = "stagnation"
                            else:
                                # Update trailing reference (it improved, so new hurdle is lower)
                                pos.z_at_last_check = curr_dir
                    
                    # --- Standard Strategy Exits ---
                    if exit_flag is None:
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
                "pnl_price_bp": pos.pnl_price_bp,
                "pnl_carry_bp": pos.pnl_carry_bp,
                "pnl_roll_bp": pos.pnl_roll_bp,
                "pnl_price_cash": pos.pnl_price_cash,
                "pnl_carry_cash": pos.pnl_carry_cash,
                "pnl_roll_cash": pos.pnl_roll_cash,
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
                period_pnl_cash_realized += (pos.pnl_cash - tcost_cash)

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
                    "pnl_price_bp": pos.pnl_price_bp,
                    "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp": pos.pnl_roll_bp,
                    "pnl_price_cash": pos.pnl_price_cash,
                    "pnl_carry_cash": pos.pnl_carry_cash, 
                    "pnl_roll_cash": pos.pnl_roll_cash,   
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
        # 2) UPDATE HISTORY & DETECT NEW SHOCK
        # ============================================================
        metric_type = getattr(shock_cfg, "metric_type", "MTM_BPS") if shock_cfg else "MTM_BPS"
        if metric_type == "REALIZED_CASH": metric_val = period_pnl_cash_realized
        elif metric_type == "REALIZED_BPS": metric_val = period_pnl_bps_realized
        elif metric_type == "MTM_BPS" or metric_type == "BPS": metric_val = period_pnl_bps_mtm
        else: metric_val = period_pnl_cash

        shock_state["history"].append(metric_val)
        shock_state["dates"].append(dts)
        
        is_new_shock = False
        if shock_cfg is not None:
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
                                           if last_r_z <= shock_cfg.resid_z_thresh:
                                               is_new_shock = True
                                  except np.linalg.LinAlgError:
                                      pass
                    except Exception:
                         pass
        
        if is_new_shock: shock_state["remaining"] = int(shock_cfg.block_length)

        # ============================================================
        # 3) EXECUTE PANIC EXIT
        # ============================================================
        is_in_shock_state = (shock_state["remaining"] > 0) or is_new_shock
        if is_in_shock_state and SHOCK_MODE == "EXIT_ALL" and len(open_positions) > 0:
            panic_bp_real, panic_cash_real = 0.0, 0.0
            panic_t_bp, panic_t_cash = 0.0, 0.0
            
            for pos in open_positions:
                pos.closed = True
                pos.close_ts = dts
                pos.exit_reason = "shock_exit"
                tcost_bp = float(getattr(cr, "OVERLAY_SWITCH_COST_BP", 0.10)) if pos.mode == "overlay" else 0.0
                tcost_cash = tcost_bp * pos.scale_dv01
                pos.tcost_bp, pos.tcost_cash = tcost_bp, tcost_cash
                
                panic_t_bp += tcost_bp
                panic_t_cash += tcost_cash
                panic_bp_real += (pos.pnl_bp - tcost_bp)
                panic_cash_real += (pos.pnl_cash - tcost_cash)
                
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": dts, "exit_reason": "shock_exit",
                    "pnl_net_bp": pos.pnl_bp - tcost_bp, "pnl_net_cash": pos.pnl_cash - tcost_cash,
                    "pnl_price_bp": pos.pnl_price_bp,
                    "pnl_carry_bp": pos.pnl_carry_bp,
                    "pnl_roll_bp": pos.pnl_roll_bp,
                    "pnl_carry_cash": pos.pnl_carry_cash, "pnl_roll_cash": pos.pnl_roll_cash,
                    "mode": pos.mode, "trade_id": pos.meta.get("trade_id"), "side": pos.meta.get("side")
                })
            open_positions = []
            
            # Retroactive History Update
            if metric_type == "REALIZED_CASH": shock_state["history"][-1] += panic_cash_real
            elif metric_type == "REALIZED_BPS": shock_state["history"][-1] += panic_bp_real
            elif metric_type == "MTM_BPS" or metric_type == "BPS": shock_state["history"][-1] -= panic_t_bp
            else: shock_state["history"][-1] -= panic_t_cash

        # ============================================================
        # 4) GATE ENTRIES
        # ============================================================
        regime_ok = True
        if regime_mask is not None:
            regime_ok = bool(regime_mask.at[dts]) if dts in regime_mask.index else False
        
        gate = (not regime_ok) or was_shock_active_at_open
        if SHOCK_MODE == "EXIT_ALL" and is_in_shock_state: gate = True
        if gate: continue
        
        # ============================================================
        # 5) NEW ENTRIES
        # ============================================================
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
            
            DRIFT_GATE = float(getattr(cr, "DRIFT_GATE_BPS", -100.0)) 
            DRIFT_W = float(getattr(cr, "DRIFT_WEIGHT", 0.0)) 

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
                exec_tenor = float(exec_row["tenor_yrs"])

                best_c_row, best_score = None, -999.0

                for _, alt in snap_srt.iterrows():
                    alt_tenor = float(alt["tenor_yrs"])
                    
                    if alt_tenor < ALT_LEG_THRESHOLD: continue
                    if alt_tenor == exec_tenor: continue 
                    if assign_bucket(alt_tenor) == "short" and assign_bucket(exec_tenor) == "long": continue
                    if assign_bucket(exec_tenor) == "short" and assign_bucket(alt_tenor) == "long": continue
                                        
                    z_alt = _to_float(alt["z_comb"])
                    disp = (z_alt - exec_z) if side_s > 0 else (exec_z - z_alt)
                    if disp < z_ent_eff: continue
                    
                    if (assign_bucket(alt_tenor)=="short" or assign_bucket(exec_tenor)=="short") and (disp < z_ent_eff + SHORT_EXTRA):
                        continue
                    
                    c_t, r_t = (alt_tenor, exec_tenor) if z_alt > exec_z else (exec_tenor, alt_tenor)
                    if not (fly_alignment_ok(c_t, 1, snap_srt, zdisp_for_pair=disp) and fly_alignment_ok(r_t, -1, snap_srt, zdisp_for_pair=disp)): continue
                    
                    drift_exec = calc_trade_drift(exec_tenor, side_s, snap_srt)
                    drift_alt = calc_trade_drift(alt_tenor, side_s, snap_srt)
                    
                    if drift_exec == -999.0 or drift_alt == -999.0: continue

                    net_drift_bps = drift_alt - drift_exec 
                    dist_years = abs(alt_tenor - exec_tenor)
                    scaling_factor = dist_years
                    norm_drift_bps = net_drift_bps / scaling_factor
                    
                    if net_drift_bps < DRIFT_GATE: continue 
                    
                    score = disp + (norm_drift_bps * DRIFT_WEIGHT)
                    if score > best_score: 
                        best_score, best_c_row = score, alt
                
                if best_c_row is not None:
                    rate_i, rate_j = None, None
                    ti, tj = tenor_to_ticker(float(best_c_row["tenor_yrs"])), tenor_to_ticker(t_trade)
                    if ti and f"{ti}_mid" in h: rate_i = _to_float(h[f"{ti}_mid"])
                    if tj and f"{tj}_mid" in h: rate_j = _to_float(h[f"{tj}_mid"])
                    
                    if rate_i is None: rate_i = _to_float(best_c_row["rate"])
                    if rate_j is None: rate_j = _to_float(exec_row["rate"])

                    pos = PairPos(dts, best_c_row, exec_row, side_s*1.0, side_s*-1.0, decisions_per_day, 
                                  scale_dv01=float(h["dv01"]), mode="overlay", 
                                  meta={"trade_id": h.get("trade_id"), "side": h.get("side")},
                                  entry_rate_i=rate_i, entry_rate_j=rate_j)
                    open_positions.append(pos)
                    ledger_rows.append({"decision_ts": dts, "event": "open", "mode": "overlay"})

    return pd.DataFrame(closed_rows), pd.DataFrame(ledger_rows), pd.DataFrame(), open_positions











import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from math import sqrt
import sys

# Import config
import cr_config as cr

# ==============================================================================
# 0. MODE SELECTION & CONFIGURATION
# ==============================================================================
# Detect mode from command line argument
REPORT_MODE = "BPS"  # Default
if len(sys.argv) > 1 and sys.argv[1].lower() == "cash":
    REPORT_MODE = "CASH"

print(f"[INIT] Running Report in {REPORT_MODE} Mode...")

# Configuration Mapping
if REPORT_MODE == "CASH":
    cfg = {
        'col_net': 'pnl_net_cash',
        'col_price': 'pnl_price_cash',
        'col_carry': 'pnl_carry_cash',
        'col_roll': 'pnl_roll_cash',
        'col_tcost': 'tcost_cash',
        'unit': '$',
        'label': 'USD',
        'fmt': '${:,.0f}'  # e.g., $1,500
    }
else:
    cfg = {
        'col_net': 'pnl_net_bp',
        'col_price': 'pnl_price_bp',
        'col_carry': 'pnl_carry_bp',
        'col_roll': 'pnl_roll_bp',
        'col_tcost': 'tcost_bp',
        'unit': 'bps',
        'label': 'Basis Points',
        'fmt': '{:,.1f}'   # e.g., 12.5
    }

# ==============================================================================
# STYLE SETUP
# ==============================================================================
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 300
colors = {'pnl': '#2ecc71', 'dd': '#e74c3c', 'price': '#3498db', 'carry': '#f1c40f', 'roll': '#9b59b6'}

def safe_divide(n, d, default=0.0):
    return n / d if d != 0 else default

def fmt_val(val):
    return cfg['fmt'].format(val)

# ==============================================================================
# 1. DATA LOADING & PREP
# ==============================================================================
out_dir = Path(cr.PATH_OUT)
suffix = getattr(cr, "OUT_SUFFIX", "")
pos_path = out_dir / f"positions_ledger{suffix}.parquet"

if not pos_path.exists():
    raise FileNotFoundError(f"[ERROR] {pos_path} not found.")

print(f"[LOAD] Reading {pos_path}...")
df = pd.read_parquet(pos_path)

# Filter Overlay only
if "mode" in df.columns:
    df = df[df["mode"] == "overlay"].copy()

# Sort by Close Time (Critical for Equity Curve)
df["close_ts"] = pd.to_datetime(df["close_ts"])
df = df.sort_values("close_ts").reset_index(drop=True)

# --- Create Equity Curves using the Selected Column ---
# We use cfg['col_net'] to toggle between 'pnl_net_bp' and 'pnl_net_cash'
df["equity_curve"] = df[cfg['col_net']].cumsum()

# ==============================================================================
# 2. METRICS ENGINE
# ==============================================================================

# --- A. Trade Statistics ---
total_trades = len(df)
win_trades = df[df[cfg['col_net']] > 0]
loss_trades = df[df[cfg['col_net']] <= 0]

win_rate = safe_divide(len(win_trades), total_trades)
gross_win = win_trades[cfg['col_net']].sum()
gross_loss = abs(loss_trades[cfg['col_net']].sum())
profit_factor = safe_divide(gross_win, gross_loss)

avg_trade = df[cfg['col_net']].mean()
avg_win = win_trades[cfg['col_net']].mean()
avg_loss = loss_trades[cfg['col_net']].mean()

# Hold Times
if "days_held_equiv" in df.columns:
    avg_hold = df["days_held_equiv"].mean()
else:
    avg_hold = (df["close_ts"] - pd.to_datetime(df["open_ts"])).dt.days.mean()

# --- B. Time-Series Statistics (Sharpe/Sortino) ---
# Resample to Daily for correct Sharpe math (works for both Cash and Bps)
daily_idx = pd.date_range(start=df["close_ts"].min(), end=df["close_ts"].max(), freq='D')
daily_pnl = df.set_index("close_ts")[cfg['col_net']].resample('D').sum().reindex(daily_idx, fill_value=0.0)

# Annualization Factor (Assuming 252 trading days)
ANN_FACTOR = 252

mean_daily = daily_pnl.mean()
std_daily = daily_pnl.std()
downside_daily = daily_pnl[daily_pnl < 0].std()

# Metrics
sharpe = safe_divide(mean_daily * ANN_FACTOR, std_daily * sqrt(ANN_FACTOR))
sortino = safe_divide(mean_daily * ANN_FACTOR, downside_daily * sqrt(ANN_FACTOR))

# Drawdown Calculation (Time Series)
cum_equity_daily = daily_pnl.cumsum()
running_max = cum_equity_daily.cummax()
dd_series = cum_equity_daily - running_max
max_dd = dd_series.min()
calmar = safe_divide(cum_equity_daily.iloc[-1], abs(max_dd))

# --- C. Attribution (The Drift Check) ---
total_net = df[cfg['col_net']].sum()
total_price = df[cfg['col_price']].sum()
total_carry = df[cfg['col_carry']].sum()
total_roll = df[cfg['col_roll']].sum()

# ==============================================================================
# 3. PRINT PROFESSIONAL TABLE
# ==============================================================================
print("\n" + "="*60)
print(f"{f'SYSTEMATIC OVERLAY REPORT ({REPORT_MODE})':^60}")
print("="*60)

stats = [
    (f"Total Net PnL ({cfg['unit']})", fmt_val(total_net)),
    ("Total Trades", f"{total_trades}"),
    ("Win Rate", f"{win_rate:.1%}"),
    ("-" * 20, "-" * 20),
    ("Profit Factor", f"{profit_factor:.2f}"),
    (f"Avg Trade ({cfg['unit']})", fmt_val(avg_trade)),
    ("Avg Win / Avg Loss", f"{abs(avg_win/avg_loss):.2f}"),
    ("Avg Hold (Days)", f"{avg_hold:.1f}"),
    ("-" * 20, "-" * 20),
    ("Sharpe Ratio (Ann.)", f"{sharpe:.2f}"),
    ("Sortino Ratio (Ann.)", f"{sortino:.2f}"),
    (f"Max Drawdown ({cfg['unit']})", fmt_val(max_dd)),
    ("Return / DD (Calmar)", f"{abs(calmar):.2f}"),
]

col_width = 35
for label, val in stats:
    if "---" in label:
        print(f"{label}   {val}")
    else:
        print(f"{label:<{col_width}} {val:>15}")
print("="*60)
print(f"ATTRIBUTION:\n Price: {fmt_val(total_price)} | Carry: {fmt_val(total_carry)} | Roll: {fmt_val(total_roll)}")
print("="*60 + "\n")

# ==============================================================================
# 4. PLOTTING SUITE
# ==============================================================================

# --- FIGURE 1: EXECUTIVE DASHBOARD ---
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2)

# 1. Equity Curve (Main - Daily Resampled)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum_equity_daily.index, cum_equity_daily.values, color=colors['pnl'], lw=2, label=f'Net Equity ({cfg["unit"]})')
ax1.fill_between(cum_equity_daily.index, cum_equity_daily.values, 0, color=colors['pnl'], alpha=0.1)
ax1.set_title(f"Realized Equity Curve (Net {cfg['label']})", fontweight='bold')
ax1.set_ylabel(f"Cumulative {cfg['label']}")
ax1.legend(loc="upper left")
ax1.margins(x=0)

if REPORT_MODE == "CASH":
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

# 2. Drawdown (Underwater)
ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
ax2.fill_between(dd_series.index, dd_series.values, 0, color=colors['dd'], alpha=0.3)
ax2.plot(dd_series.index, dd_series.values, color=colors['dd'], lw=1)
ax2.set_title("Drawdown Profile", fontsize=11)
ax2.set_ylabel(f"Drawdown ({cfg['unit']})")
ax2.grid(True, linestyle='--', alpha=0.5)

if REPORT_MODE == "CASH":
    ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

# 3. Monthly Returns (Bar)
ax3 = fig.add_subplot(gs[2, 0])
monthly = df.set_index("close_ts")[cfg['col_net']].resample('M').sum()
clrs = ['#e74c3c' if x < 0 else '#2ecc71' for x in monthly.values]
monthly.index = monthly.index.strftime('%Y-%m')
monthly.plot(kind='bar', ax=ax3, color=clrs, width=0.8)
ax3.set_title(f"Monthly Net PnL ({cfg['unit']})")
ax3.set_xlabel("")
ax3.tick_params(axis='x', rotation=45, labelsize=9)
ax3.grid(axis='y', linestyle='--', alpha=0.5)

# 4. PnL Distribution (Histogram)
ax4 = fig.add_subplot(gs[2, 1])
sns.histplot(df[cfg['col_net']], kde=True, ax=ax4, color='#34495e', bins=30)
ax4.axvline(0, color='black', linestyle='--')
ax4.axvline(avg_trade, color=colors['pnl'], linestyle='-', label=f'Mean: {fmt_val(avg_trade)}')
ax4.set_title(f"Trade Distribution ({cfg['unit']})")
ax4.legend()

if REPORT_MODE == "CASH":
    ax4.xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

plt.tight_layout()
plt.show()

# --- FIGURE 2: THE "DRIFT" THESIS (Attribution) ---
cum_price = df[cfg['col_price']].cumsum()
cum_carry = df[cfg['col_carry']].cumsum()
cum_roll = df[cfg['col_roll']].cumsum()
cum_drift = cum_carry + cum_roll

fig2, ax = plt.subplots(figsize=(14, 6))

ax.plot(df["close_ts"], cum_price, label="Price PnL (Luck)", color=colors['price'], lw=1.5, alpha=0.8)
ax.plot(df["close_ts"], cum_drift, label="Drift PnL (Carry + Roll)", color=colors['carry'], lw=2.5)
ax.plot(df["close_ts"], df["equity_curve"], label="Total Net PnL", color='black', lw=2, linestyle='--')

ax.set_title(f"PnL Attribution: Drift vs Price ({REPORT_MODE})", fontsize=14, fontweight='bold')
ax.set_ylabel(f"Cumulative {cfg['label']}")
ax.legend(fontsize=11)
ax.grid(True, which='major', linestyle='-', alpha=0.6)
ax.minorticks_on()

if REPORT_MODE == "CASH":
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

plt.tight_layout()
plt.show()











import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from math import sqrt

# Import config
import cr_config as cr

# ==============================================================================
# CONFIG & STYLE
# ==============================================================================
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 300
colors = {'pnl': '#2ecc71', 'dd': '#e74c3c', 'price': '#3498db', 'carry': '#f1c40f', 'roll': '#9b59b6'}

def safe_divide(n, d, default=0.0):
    return n / d if d != 0 else default

# ==============================================================================
# 1. DATA LOADING & PREP
# ==============================================================================
out_dir = Path(cr.PATH_OUT)
suffix = getattr(cr, "OUT_SUFFIX", "")
pos_path = out_dir / f"positions_ledger{suffix}.parquet"

if not pos_path.exists():
    raise FileNotFoundError(f"[ERROR] {pos_path} not found.")

print(f"[LOAD] Reading {pos_path}...")
df = pd.read_parquet(pos_path)

# Filter Overlay only
if "mode" in df.columns:
    df = df[df["mode"] == "overlay"].copy()

# Sort by Close Time (Critical for Equity Curve)
df["close_ts"] = pd.to_datetime(df["close_ts"])
df = df.sort_values("close_ts").reset_index(drop=True)

# --- FIX: Create the Trade-Level Equity Curve Columns ---
df["equity_bp"] = df["pnl_net_bp"].cumsum()  # <--- THIS WAS MISSING
df["equity_cash"] = df["pnl_net_cash"].cumsum()

# ==============================================================================
# 2. METRICS ENGINE
# ==============================================================================

# --- A. Trade Statistics ---
total_trades = len(df)
win_trades = df[df["pnl_net_bp"] > 0]
loss_trades = df[df["pnl_net_bp"] <= 0]

win_rate = safe_divide(len(win_trades), total_trades)
gross_win = win_trades["pnl_net_bp"].sum()
gross_loss = abs(loss_trades["pnl_net_bp"].sum())
profit_factor = safe_divide(gross_win, gross_loss)

avg_trade_bp = df["pnl_net_bp"].mean()
avg_win_bp = win_trades["pnl_net_bp"].mean()
avg_loss_bp = loss_trades["pnl_net_bp"].mean()

# Hold Times
if "days_held_equiv" in df.columns:
    avg_hold = df["days_held_equiv"].mean()
else:
    avg_hold = (df["close_ts"] - pd.to_datetime(df["open_ts"])).dt.days.mean()

# --- B. Time-Series Statistics (Sharpe/Sortino) ---
# We convert the discrete trade ledger into a Daily Equity Curve for proper risk stats
daily_idx = pd.date_range(start=df["close_ts"].min(), end=df["close_ts"].max(), freq='D')
daily_pnl = df.set_index("close_ts")["pnl_net_bp"].resample('D').sum().reindex(daily_idx, fill_value=0.0)

# Annualization Factor (Assuming 252 trading days)
ANN_FACTOR = 252

mean_daily = daily_pnl.mean()
std_daily = daily_pnl.std()
downside_daily = daily_pnl[daily_pnl < 0].std()

# Metrics
sharpe = safe_divide(mean_daily * ANN_FACTOR, std_daily * sqrt(ANN_FACTOR))
sortino = safe_divide(mean_daily * ANN_FACTOR, downside_daily * sqrt(ANN_FACTOR))

# Drawdown Calculation (Time Series)
cum_equity_daily = daily_pnl.cumsum()
running_max = cum_equity_daily.cummax()
dd_series = cum_equity_daily - running_max
max_dd_bp = dd_series.min()
calmar = safe_divide(cum_equity_daily.iloc[-1], abs(max_dd_bp))

# --- C. Attribution (The Drift Check) ---
total_price = df["pnl_price_bp"].sum()
total_carry = df["pnl_carry_bp"].sum()
total_roll = df["pnl_roll_bp"].sum()

# ==============================================================================
# 3. PRINT PROFESSIONAL TABLE
# ==============================================================================
print("\n" + "="*60)
print(f"{'SYSTEMATIC OVERLAY PERFORMANCE REPORT':^60}")
print("="*60)

stats = [
    ("Total Net PnL (bps)", f"{df['pnl_net_bp'].sum():,.1f}"),
    ("Total Net Cash ($)", f"${df['pnl_net_cash'].sum():,.0f}"),
    ("Total Trades", f"{total_trades}"),
    ("Win Rate", f"{win_rate:.1%}"),
    ("-" * 20, "-" * 20),
    ("Profit Factor", f"{profit_factor:.2f}"),
    ("Avg Trade (bps)", f"{avg_trade_bp:.2f}"),
    ("Avg Win / Avg Loss", f"{abs(avg_win_bp/avg_loss_bp):.2f}"),
    ("Avg Hold (Days)", f"{avg_hold:.1f}"),
    ("-" * 20, "-" * 20),
    ("Sharpe Ratio (Ann.)", f"{sharpe:.2f}"),
    ("Sortino Ratio (Ann.)", f"{sortino:.2f}"),
    ("Max Drawdown (bps)", f"{max_dd_bp:,.1f}"),
    ("Return / DD (Calmar)", f"{abs(calmar):.2f}"),
]

col_width = 35
for label, val in stats:
    if "---" in label:
        print(f"{label}   {val}")
    else:
        print(f"{label:<{col_width}} {val:>15}")
print("="*60)
print(f"ATTRIBUTION:\n Price: {total_price:,.0f} bps | Carry: {total_carry:,.0f} bps | Roll: {total_roll:,.0f} bps")
print("="*60 + "\n")

# ==============================================================================
# 4. PLOTTING SUITE
# ==============================================================================

# --- FIGURE 1: EXECUTIVE DASHBOARD ---
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2)

# 1. Equity Curve (Main - Daily Resampled for Smoothness)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum_equity_daily.index, cum_equity_daily.values, color=colors['pnl'], lw=2, label='Net Equity (bps)')
ax1.fill_between(cum_equity_daily.index, cum_equity_daily.values, 0, color=colors['pnl'], alpha=0.1)
ax1.set_title("Realized Equity Curve (Net Bps)", fontweight='bold')
ax1.set_ylabel("Cumulative Bps")
ax1.legend(loc="upper left")
ax1.margins(x=0)

# 2. Drawdown (Underwater)
ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
ax2.fill_between(dd_series.index, dd_series.values, 0, color=colors['dd'], alpha=0.3)
ax2.plot(dd_series.index, dd_series.values, color=colors['dd'], lw=1)
ax2.set_title("Drawdown Profile", fontsize=11)
ax2.set_ylabel("Drawdown (bps)")
ax2.grid(True, linestyle='--', alpha=0.5)

# 3. Monthly Returns (Bar)
ax3 = fig.add_subplot(gs[2, 0])
monthly = df.set_index("close_ts")["pnl_net_bp"].resample('M').sum()
norm = plt.Normalize(monthly.min(), monthly.max())
clrs = ['#e74c3c' if x < 0 else '#2ecc71' for x in monthly.values]
monthly.index = monthly.index.strftime('%Y-%m')
monthly.plot(kind='bar', ax=ax3, color=clrs, width=0.8)
ax3.set_title("Monthly Net PnL (bps)")
ax3.set_xlabel("")
ax3.tick_params(axis='x', rotation=45, labelsize=9)
ax3.grid(axis='y', linestyle='--', alpha=0.5)

# 4. PnL Distribution (Histogram)
ax4 = fig.add_subplot(gs[2, 1])
sns.histplot(df['pnl_net_bp'], kde=True, ax=ax4, color='#34495e', bins=30)
ax4.axvline(0, color='black', linestyle='--')
ax4.axvline(avg_trade_bp, color=colors['pnl'], linestyle='-', label=f'Mean: {avg_trade_bp:.1f}')
ax4.set_title("Trade Distribution (bps)")
ax4.legend()

plt.tight_layout()
plt.show()

# --- FIGURE 2: THE "DRIFT" THESIS (Attribution) ---
# Create cumulative series for components based on Trade Ledger
# (df is already sorted by close_ts)
cum_price = df["pnl_price_bp"].cumsum()
cum_carry = df["pnl_carry_bp"].cumsum()
cum_roll = df["pnl_roll_bp"].cumsum()
cum_drift = cum_carry + cum_roll

fig2, ax = plt.subplots(figsize=(14, 6))

ax.plot(df["close_ts"], cum_price, label="Price PnL (Luck/Mean Rev)", color=colors['price'], lw=1.5, alpha=0.8)
ax.plot(df["close_ts"], cum_drift, label="Drift PnL (Carry + Roll)", color=colors['carry'], lw=2.5)
# FIX: Now "equity_bp" definitely exists
ax.plot(df["close_ts"], df["equity_bp"], label="Total Net PnL", color='black', lw=2, linestyle='--')

ax.set_title("PnL Attribution: Drift (Predictable) vs Price (Volatile)", fontsize=14, fontweight='bold')
ax.set_ylabel("Cumulative Bps")
ax.legend(fontsize=11)
ax.grid(True, which='major', linestyle='-', alpha=0.6)
ax.grid(True, which='minor', linestyle=':', alpha=0.3)
ax.minorticks_on()

plt.tight_layout()
plt.show()





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from math import sqrt

# Import config
import cr_config as cr

# ==============================================================================
# CONFIG & STYLE
# ==============================================================================
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 300
colors = {'pnl': '#2ecc71', 'dd': '#e74c3c', 'price': '#3498db', 'carry': '#f1c40f', 'roll': '#9b59b6'}

def safe_divide(n, d, default=0.0):
    return n / d if d != 0 else default

# ==============================================================================
# 1. DATA LOADING & PREP
# ==============================================================================
out_dir = Path(cr.PATH_OUT)
suffix = getattr(cr, "OUT_SUFFIX", "")
pos_path = out_dir / f"positions_ledger{suffix}.parquet"

if not pos_path.exists():
    raise FileNotFoundError(f"[ERROR] {pos_path} not found.")

print(f"[LOAD] Reading {pos_path}...")
df = pd.read_parquet(pos_path)

# Filter Overlay only
if "mode" in df.columns:
    df = df[df["mode"] == "overlay"].copy()

# Sort
df["close_ts"] = pd.to_datetime(df["close_ts"])
df = df.sort_values("close_ts").reset_index(drop=True)

# ==============================================================================
# 2. METRICS ENGINE
# ==============================================================================

# --- A. Trade Statistics ---
total_trades = len(df)
win_trades = df[df["pnl_net_bp"] > 0]
loss_trades = df[df["pnl_net_bp"] <= 0]

win_rate = safe_divide(len(win_trades), total_trades)
gross_win = win_trades["pnl_net_bp"].sum()
gross_loss = abs(loss_trades["pnl_net_bp"].sum())
profit_factor = safe_divide(gross_win, gross_loss)

avg_trade_bp = df["pnl_net_bp"].mean()
avg_win_bp = win_trades["pnl_net_bp"].mean()
avg_loss_bp = loss_trades["pnl_net_bp"].mean()

# Hold Times
if "days_held_equiv" in df.columns:
    avg_hold = df["days_held_equiv"].mean()
else:
    # Fallback
    avg_hold = (df["close_ts"] - pd.to_datetime(df["open_ts"])).dt.days.mean()

# --- B. Time-Series Statistics (Sharpe/Sortino) ---
# We convert the discrete trade ledger into a Daily Equity Curve for proper risk stats
daily_idx = pd.date_range(start=df["close_ts"].min(), end=df["close_ts"].max(), freq='D')
daily_pnl = df.set_index("close_ts")["pnl_net_bp"].resample('D').sum().reindex(daily_idx, fill_value=0.0)

# Annualization Factor (Assuming 252 trading days)
ANN_FACTOR = 252

mean_daily = daily_pnl.mean()
std_daily = daily_pnl.std()
downside_daily = daily_pnl[daily_pnl < 0].std()

# Metrics
sharpe = safe_divide(mean_daily * ANN_FACTOR, std_daily * sqrt(ANN_FACTOR))
sortino = safe_divide(mean_daily * ANN_FACTOR, downside_daily * sqrt(ANN_FACTOR))

# Drawdown Calculation (Time Series)
cum_equity = daily_pnl.cumsum()
running_max = cum_equity.cummax()
dd_series = cum_equity - running_max
max_dd_bp = dd_series.min()
calmar = safe_divide(cum_equity.iloc[-1], abs(max_dd_bp)) # Simple realized calmar

# --- C. Attribution (The Drift Check) ---
total_price = df["pnl_price_bp"].sum()
total_carry = df["pnl_carry_bp"].sum()
total_roll = df["pnl_roll_bp"].sum()
total_costs = df["tcost_bp"].sum()

# ==============================================================================
# 3. PRINT PROFESSIONAL TABLE
# ==============================================================================
print("\n" + "="*60)
print(f"{'SYSTEMATIC OVERLAY PERFORMANCE REPORT':^60}")
print("="*60)

stats = [
    ("Total Net PnL (bps)", f"{df['pnl_net_bp'].sum():,.1f}"),
    ("Total Net Cash ($)", f"${df['pnl_net_cash'].sum():,.0f}"),
    ("Total Trades", f"{total_trades}"),
    ("Win Rate", f"{win_rate:.1%}"),
    ("-" * 20, "-" * 20),
    ("Profit Factor", f"{profit_factor:.2f}"),
    ("Avg Trade (bps)", f"{avg_trade_bp:.2f}"),
    ("Avg Win / Avg Loss", f"{abs(avg_win_bp/avg_loss_bp):.2f}"),
    ("Avg Hold (Days)", f"{avg_hold:.1f}"),
    ("-" * 20, "-" * 20),
    ("Sharpe Ratio (Ann.)", f"{sharpe:.2f}"),
    ("Sortino Ratio (Ann.)", f"{sortino:.2f}"),
    ("Max Drawdown (bps)", f"{max_dd_bp:,.1f}"),
    ("Return / DD (Calmar)", f"{abs(calmar):.2f}"),
]

col_width = 35
for label, val in stats:
    if "---" in label:
        print(f"{label}   {val}")
    else:
        print(f"{label:<{col_width}} {val:>15}")
print("="*60)
print(f"ATTRIBUTION:\n Price: {total_price:,.0f} bps | Carry: {total_carry:,.0f} bps | Roll: {total_roll:,.0f} bps")
print("="*60 + "\n")

# ==============================================================================
# 4. PLOTTING SUITE
# ==============================================================================

# --- FIGURE 1: EXECUTIVE DASHBOARD ---
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2)

# 1. Equity Curve (Main)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum_equity.index, cum_equity.values, color=colors['pnl'], lw=2, label='Net Equity (bps)')
ax1.fill_between(cum_equity.index, cum_equity.values, 0, color=colors['pnl'], alpha=0.1)
ax1.set_title("Realized Equity Curve (Net Bps)", fontweight='bold')
ax1.set_ylabel("Cumulative Bps")
ax1.legend(loc="upper left")
ax1.margins(x=0)

# 2. Drawdown (Underwater)
ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
ax2.fill_between(dd_series.index, dd_series.values, 0, color=colors['dd'], alpha=0.3)
ax2.plot(dd_series.index, dd_series.values, color=colors['dd'], lw=1)
ax2.set_title("Drawdown Profile", fontsize=11)
ax2.set_ylabel("Drawdown (bps)")
ax2.grid(True, linestyle='--', alpha=0.5)

# 3. Monthly Returns (Bar)
ax3 = fig.add_subplot(gs[2, 0])
monthly = df.set_index("close_ts")["pnl_net_bp"].resample('M').sum()
norm = plt.Normalize(monthly.min(), monthly.max())
clrs = ['#e74c3c' if x < 0 else '#2ecc71' for x in monthly.values]
monthly.index = monthly.index.strftime('%Y-%m')
monthly.plot(kind='bar', ax=ax3, color=clrs, width=0.8)
ax3.set_title("Monthly Net PnL (bps)")
ax3.set_xlabel("")
ax3.tick_params(axis='x', rotation=45, labelsize=9)
ax3.grid(axis='y', linestyle='--', alpha=0.5)

# 4. PnL Distribution (Histogram)
ax4 = fig.add_subplot(gs[2, 1])
sns.histplot(df['pnl_net_bp'], kde=True, ax=ax4, color='#34495e', bins=30)
ax4.axvline(0, color='black', linestyle='--')
ax4.axvline(avg_trade_bp, color=colors['pnl'], linestyle='-', label=f'Mean: {avg_trade_bp:.1f}')
ax4.set_title("Trade Distribution (bps)")
ax4.legend()

plt.tight_layout()
plt.show()

# --- FIGURE 2: THE "DRIFT" THESIS (Attribution) ---
# Create cumulative series for components
# We must re-sort original DF to ensure cumsum is correct timeline
df_sorted = df.sort_values("close_ts")
cum_price = df_sorted["pnl_price_bp"].cumsum()
cum_carry = df_sorted["pnl_carry_bp"].cumsum()
cum_roll = df_sorted["pnl_roll_bp"].cumsum()
cum_drift = cum_carry + cum_roll

fig2, ax = plt.subplots(figsize=(14, 6))

ax.plot(df_sorted["close_ts"], cum_price, label="Price PnL (Luck/Mean Rev)", color=colors['price'], lw=1.5, alpha=0.8)
ax.plot(df_sorted["close_ts"], cum_drift, label="Drift PnL (Carry + Roll)", color=colors['carry'], lw=2.5)
ax.plot(df_sorted["close_ts"], df_sorted["equity_bp"], label="Total Net PnL", color='black', lw=2, linestyle='--')

ax.set_title("PnL Attribution: Drift (Predictable) vs Price (Volatile)", fontsize=14, fontweight='bold')
ax.set_ylabel("Cumulative Bps")
ax.legend(fontsize=11)
ax.grid(True, which='major', linestyle='-', alpha=0.6)
ax.grid(True, which='minor', linestyle=':', alpha=0.3)
ax.minorticks_on()

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import numpy as np

# Import your config to get paths automatically
import cr_config as cr

# ==============================================================================
# 1. LOAD CLOSED POSITIONS
# ==============================================================================
out_dir = Path(cr.PATH_OUT)
suffix = getattr(cr, "OUT_SUFFIX", "")
pos_path = out_dir / f"positions_ledger{suffix}.parquet"

if not pos_path.exists():
    print(f"[ERROR] File not found: {pos_path}")
else:
    print(f"[LOAD] Reading {pos_path}...")
    df = pd.read_parquet(pos_path)

    # 1. Filter for Overlay Mode (if mixed)
    if "mode" in df.columns:
        df = df[df["mode"] == "overlay"].copy()

    # 2. Sort by CLOSE TIME (Critical for realized equity curve)
    df["close_ts"] = pd.to_datetime(df["close_ts"])
    df = df.sort_values("close_ts").reset_index(drop=True)

    # ==============================================================================
    # 2. CALCULATE REALIZED EQUITY CURVE
    # ==============================================================================
    # Cumulative Sum of Net PnL (Bps and Cash)
    df["equity_bp"] = df["pnl_net_bp"].cumsum()
    df["equity_cash"] = df["pnl_net_cash"].cumsum()

    # Calculate Drawdown (Distance from All-Time High Realized Equity)
    running_max_bp = df["equity_bp"].cummax()
    df["drawdown_bp"] = df["equity_bp"] - running_max_bp

    # Stats
    total_bp = df["pnl_net_bp"].sum()
    total_cash = df["pnl_net_cash"].sum()
    max_dd = df["drawdown_bp"].min()
    win_rate = (df["pnl_net_bp"] > 0).mean()
    avg_trade = df["pnl_net_bp"].mean()

    print("-" * 40)
    print(f"Total Realized PnL:   {total_bp:,.1f} bps")
    print(f"Total Realized Cash:  ${total_cash:,.0f}")
    print(f"Max Realized DD:      {max_dd:,.1f} bps")
    print(f"Win Rate:             {win_rate:.1%}")
    print(f"Avg Trade:            {avg_trade:.2f} bps")
    print("-" * 40)

    # ==============================================================================
    # 3. PLOT
    # ==============================================================================
    # We use a STEP plot because realized PnL is discrete.
    # It implies: "My banked money stayed X until trade T closed."
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})

    # --- Top: Equity Curve ---
    ax1.step(df["close_ts"], df["equity_bp"], where='post', color='#1f77b4', lw=2, label="Realized Net PnL")
    
    # Optional: Color background for "Winning Streaks" vs "Losing Streaks"?
    # Keeping it simple for now.
    
    ax1.set_title("Closed Position Equity Curve (Net Bps)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cumulative Bps")
    ax1.grid(True, alpha=0.4)
    ax1.legend(loc="upper left")

    # --- Bottom: Drawdown ---
    ax2.fill_between(df["close_ts"], df["drawdown_bp"], 0, step='post', color='#d62728', alpha=0.3)
    ax2.step(df["close_ts"], df["drawdown_bp"], where='post', color='#d62728', lw=1)
    
    ax2.set_title("Drawdown Profile (High Water Mark)", fontsize=12)
    ax2.set_ylabel("Drawdown (Bps)")
    ax2.set_xlabel("Trade Close Date")
    ax2.grid(True, alpha=0.4)

    # Format X-Axis Dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.show()






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import cr_config as cr
import hybrid_filter as hf
from hybrid_filter import RegimeThresholds, ShockConfig

# =========================================================
# 1) Load closed overlay positions (pnl_net_bp, close_ts)
# =========================================================
out_dir = Path(cr.PATH_OUT)

# ⚠️ Adjust this filename if yours is different
# This should be the *closed positions* output, not the mark-to-market ledger.
pos_path = out_dir / f"pos_overlay{cr.OUT_SUFFIX}.parquet"
pos_overlay = pd.read_parquet(pos_path)

# If the file also contains strategy-mode trades, keep only overlay:
if "mode" in pos_overlay.columns:
    pos_overlay = pos_overlay[pos_overlay["mode"] == "overlay"].copy()

pos_overlay["close_ts"] = pd.to_datetime(pos_overlay["close_ts"], utc=False, errors="coerce")
pos_overlay = pos_overlay.sort_values("close_ts")
print("Closed overlay positions:", len(pos_overlay))

# =========================================================
# 2) Build/load hybrid signals + regime & shock masks
# =========================================================

# Example regime thresholds (tweak later):
reg_thresholds = RegimeThresholds(
    min_signal_health_z=-0.5,   # require OK-ish health
    max_trendiness_abs=2.0,     # avoid highly trending / one-way regimes
    max_z_xs_mean_abs_z=2.0,    # avoid very extreme cross-sectional mean
)

# Example shock config (tweak later):
shock_cfg = ShockConfig(
    pnl_window=10,                       # 10-bucket window (days if DECISION_FREQ='D')
    use_raw_pnl=True,
    use_residuals=True,
    raw_pnl_z_thresh=-1.5,              # raw PnL shock threshold
    resid_z_thresh=-1.5,                # residual-based shock threshold
    regression_cols=[
        "signal_health_z",
        "trendiness_abs",
        "z_xs_mean_roll_z",
    ],                                  # will auto-drop any missing cols
    block_length=10,                    # block 10 buckets after each shock
)

hyb = hf.attach_regime_and_shock_masks(
    pos_overlay=pos_overlay,
    regime_thresholds=reg_thresholds,
    shock_cfg=shock_cfg,
    force_rebuild_signals=False,        # set True if you change RegimeConfig/base_window
)

signals      = hyb["signals"]
regime_mask  = hyb["regime_mask"]       # pd.Series indexed by decision_ts (ok_regime)
shock_res    = hyb["shock_results"]     # dict from run_shock_blocker

print("Signals shape:", signals.shape)
print("Regime mask length:", len(regime_mask))

# =========================================================
# 3) Build a diagnostic DataFrame on the common index
# =========================================================
idx          = shock_res["ts"]                      # DatetimeIndex
pnl_raw      = shock_res["pnl_raw"]                 # per-bucket PnL (bp) before blocking
pnl_blocked  = shock_res["pnl_blocked"]             # after shock-blocker (zeroed on blocks)
shock_mask   = shock_res["shock_mask"]              # True where bucket is shock
block_mask   = shock_res["block_mask"]              # True where we would BLOCK *after* shock

# Align regime mask to this index
reg = regime_mask.reindex(idx).fillna(False)

diag = pd.DataFrame({
    "pnl_raw":     pnl_raw,
    "pnl_blocked": pnl_blocked,
    "shock":       shock_mask,
    "block_shock": block_mask,
    "ok_regime":   reg,
}).sort_index()

# Hybrid "block" flag: either bad regime OR in a shock-blocked zone
diag["hybrid_block"] = (~diag["ok_regime"]) | diag["block_shock"]

print("\nDiag head:")
print(diag.head())

# =========================================================
# 4) Plot cumulative PnL: original vs shock-blocked vs hybrid-blocked
# =========================================================
pnl_raw_cum        = diag["pnl_raw"].cumsum()
pnl_shock_cum      = diag["pnl_blocked"].cumsum()
pnl_hybrid_cum     = diag["pnl_raw"].where(~diag["hybrid_block"], 0.0).cumsum()

plt.figure()
pnl_raw_cum.plot(label="Original (no filter)", linewidth=1.2)
pnl_shock_cum.plot(label="Shock-blocker only", linewidth=1.2)
pnl_hybrid_cum.plot(label="Hybrid (regime + shock)", linewidth=1.2)
plt.title("Overlay cumulative PnL (bp) – filters comparison")
plt.xlabel("Decision bucket")
plt.ylabel("Cumulative PnL (bp)")
plt.legend()
plt.grid(True)
plt.show()

# =========================================================
# 5) Quick stats: on vs off blocks (using ORIGINAL pnl_raw)
# =========================================================
def pnl_stats(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "label": label,
            "n_days": 0,
            "mean_pnl": np.nan,
            "median_pnl": np.nan,
            "p05": np.nan,
            "p95": np.nan,
        }
    return {
        "label": label,
        "n_days": len(df),
        "mean_pnl": df["pnl_raw"].mean(),
        "median_pnl": df["pnl_raw"].median(),
        "p05": df["pnl_raw"].quantile(0.05),
        "p95": df["pnl_raw"].quantile(0.95),
    }

on_hybrid  = diag[diag["hybrid_block"]]
off_hybrid = diag[~diag["hybrid_block"]]

stats_on  = pnl_stats(on_hybrid,  "on_hybrid_block")
stats_off = pnl_stats(off_hybrid, "off_hybrid_block")

print("\n=== PnL per bucket: on vs off hybrid blocks (using ORIGINAL pnl_raw) ===")
print(pd.DataFrame([stats_on, stats_off]).set_index("label"))

# =========================================================
# 6) Optional: visualize hybrid-block days as vertical stripes
# =========================================================
plt.figure()
ax = pnl_raw_cum.plot(label="Original", linewidth=1.0)
pnl_hybrid_cum.plot(ax=ax, label="Hybrid-filtered", linewidth=1.0)

for dt, row in diag.iterrows():
    if row["hybrid_block"]:
        ax.axvspan(dt, dt, alpha=0.12)  # thin vertical highlight

ax.set_title("Cumulative PnL with hybrid-blocked periods shaded")
ax.set_xlabel("Decision bucket")
ax.set_ylabel("Cumulative PnL (bp)")
ax.legend()
ax.grid(True)
plt.show()


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import cr_config as cr
import hybrid_filter as hf  # combined regime + shock filter

# -------------------------
# 1) Discover months and build/load signals
# -------------------------
enh_dir = Path(cr.PATH_ENH)

# Assumes filenames like "202404_enh_d.parquet"
yymms = sorted({
    p.stem.split("_")[0]
    for p in enh_dir.glob(f"*{cr.ENH_SUFFIX}.parquet")
})
print("YYMMs detected:", yymms)

# This matches the expected signature in hybrid_filter.py:
# def build_hybrid_signals(yymms, decision_freq, force_rebuild=False, cache_path=None)
signals = hf.build_hybrid_signals(
    yymms=yymms,
    decision_freq=cr.DECISION_FREQ,
    force_rebuild=False,   # set True if you want to rebuild from scratch
)
print("Signals shape:", signals.shape)
print("Signal columns:", signals.columns.tolist())

# -------------------------
# 2) Load overlay PnL from your backtest
# -------------------------
out_dir = Path(cr.PATH_OUT)

# Adjust filename if your ledger name differs
ledger_path = out_dir / f"ledger_overlay{cr.OUT_SUFFIX}.parquet"
ledger = pd.read_parquet(ledger_path)

# We only care about overlay trades
pos_overlay = ledger[ledger["mode"] == "overlay"].copy()
pos_overlay["decision_ts"] = pd.to_datetime(pos_overlay["decision_ts"])
pos_overlay = pos_overlay.sort_values("decision_ts")

print("Overlay marks:", len(pos_overlay))

# -------------------------
# 3) Run the hybrid filter (regime + shock)
# -------------------------
# This matches the expected signature:
# def run_hybrid_filter(signals, pos_overlay, shock_method="B", shock_block_days=10,
#                       use_regime_filter=True, use_shock_blocker=True, **kwargs)
results = hf.run_hybrid_filter(
    signals=signals,
    pos_overlay=pos_overlay,
    # shock-blocker knobs
    shock_method="B",          # "A" = all strong, "B" = top-K, "C" = single best
    shock_block_days=10,       # how long to block after a shock
    # regime filter knobs
    use_regime_filter=True,
    use_shock_blocker=True,
)

pnl_original = results["pnl_original"]   # pd.Series indexed by date
pnl_filtered = results["pnl_filtered"]   # same index
flags       = results["flags"]           # DataFrame with boolean columns
summary     = results["summary"]         # DataFrame of stats

print("\n=== Hybrid summary ===")
print(summary)

print("\nFlag columns:", flags.columns.tolist())
print(flags.head())

# -------------------------
# 4) Plot original vs hybrid-filtered cumulative PnL
# -------------------------
plt.figure()
pnl_original.cumsum().plot(label="Original", linewidth=1.2)
pnl_filtered.cumsum().plot(label="Hybrid-filtered", linewidth=1.2)
plt.title("Overlay PnL: original vs hybrid-filtered")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL (bp or cash)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# 5) Quick look at where the filter blocks
# -------------------------
diag = pd.concat(
    [
        pnl_original.rename("pnl_orig"),
        pnl_filtered.rename("pnl_filt"),
        flags,
    ],
    axis=1,
).dropna(subset=["pnl_orig"])

blocked_days = diag[diag["hybrid_block"]].copy() if "hybrid_block" in diag.columns else pd.DataFrame()
print("\nFirst few blocked days:")
print(blocked_days.head(10))

# -------------------------
# 6) Compare average daily PnL on vs off hybrid blocks (using ORIGINAL PnL)
# -------------------------
if "hybrid_block" in diag.columns:
    on_block  = diag[diag["hybrid_block"]]
    off_block = diag[~diag["hybrid_block"]]
else:
    # Fallback: treat no blocks if column missing
    on_block  = diag.iloc[0:0]
    off_block = diag

def pnl_stats(x, label):
    if len(x) == 0:
        return {"label": label, "n_days": 0, "mean_pnl": np.nan,
                "median_pnl": np.nan, "p05": np.nan, "p95": np.nan}
    return {
        "label": label,
        "n_days": len(x),
        "mean_pnl": x["pnl_orig"].mean(),
        "median_pnl": x["pnl_orig"].median(),
        "p05": x["pnl_orig"].quantile(0.05),
        "p95": x["pnl_orig"].quantile(0.95),
    }

stats_on  = pnl_stats(on_block,  "on_block (hybrid_block=True)")
stats_off = pnl_stats(off_block, "off_block (hybrid_block=False)")

print("\n=== PnL per day: on vs off hybrid blocks (using ORIGINAL PnL) ===")
print(pd.DataFrame([stats_on, stats_off]).set_index("label"))

# -------------------------
# 7) Optional: visualize blocks as vertical stripes
# -------------------------
plt.figure()
cum_orig = pnl_original.cumsum()
cum_filt = pnl_filtered.cumsum()

ax = cum_orig.plot(label="Original", linewidth=1.0)
cum_filt.plot(ax=ax, label="Hybrid-filtered", linewidth=1.0)

if "hybrid_block" in flags.columns:
    for dt, row in flags.iterrows():
        if row.get("hybrid_block", False):
            ax.axvspan(dt, dt, alpha=0.15)  # thin vertical stripe

ax.set_title("Cumulative PnL with hybrid block days shaded")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative PnL")
ax.legend()
ax.grid(True)
plt.show()

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import cr_config as cr
import hybrid_filter as hf  # <- your new hybrid filter file

# -------------------------
# 1) Discover months and build/load signals
# -------------------------
enh_dir = Path(cr.PATH_ENH)

# Assumes filenames like "202404_enh_d.parquet" or similar with yymm first
yymms = sorted({
    p.stem.split("_")[0]
    for p in enh_dir.glob(f"*{cr.ENH_SUFFIX}.parquet")
})
print("YYMMs detected:", yymms)

signals = hf.build_or_load_hybrid_signals(
    yymms=yymms,
    decision_freq=cr.DECISION_FREQ,
    force_rebuild=False,   # flip to True if you want to regenerate from scratch
)
print("Signals shape:", signals.shape)
print("Signal columns:", signals.columns.tolist())

# -------------------------
# 2) Load overlay PnL from your backtest
# -------------------------
out_dir = Path(cr.PATH_OUT)

# Adjust filename if yours differs
ledger_path = out_dir / f"ledger_overlay{cr.OUT_SUFFIX}.parquet"
ledger = pd.read_parquet(ledger_path)

# We only care about overlay trades here
pos_overlay = ledger[ledger["mode"] == "overlay"].copy()
pos_overlay["decision_ts"] = pd.to_datetime(pos_overlay["decision_ts"])
pos_overlay = pos_overlay.sort_values("decision_ts")

print("Overlay marks:", len(pos_overlay))

# -------------------------
# 3) Run the hybrid filter
# -------------------------
results = hf.run_hybrid_filter(
    signals=signals,
    pos_overlay=pos_overlay,
    # shock-blocker knobs
    shock_method="B",          # "A" = all strong, "B" = top-K, "C" = single best
    shock_block_days=10,       # how long to block after a shock
    # regime filter knobs (whatever you wired in – adjust if needed)
    use_regime_filter=True,
    use_shock_blocker=True,
)

# Expected structure (adjust if your implementation uses slightly different keys):
pnl_original = results["pnl_original"]      # pd.Series indexed by date, in bp or cash
pnl_filtered = results["pnl_filtered"]      # same index, after hybrid blocks
flags       = results["flags"]              # DataFrame with e.g. ["shock_block", "regime_block", "hybrid_block"]
summary     = results["summary"]            # high-level stats table

print("\n=== Hybrid summary ===")
print(summary)

print("\nFlag columns:", flags.columns.tolist())
print(flags.head())

# -------------------------
# 4) Plot original vs hybrid-filtered cumulative PnL
# -------------------------
plt.figure()
pnl_original.cumsum().plot(label="Original", linewidth=1.2)
pnl_filtered.cumsum().plot(label="Hybrid-filtered", linewidth=1.2)
plt.title("Overlay PnL: original vs hybrid-filtered")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# 5) Quick look at where the filter blocks
# -------------------------
# Combine PnL + flags into one frame
diag = pd.concat(
    [
        pnl_original.rename("pnl_orig"),
        pnl_filtered.rename("pnl_filt"),
        flags,
    ],
    axis=1,
).dropna(subset=["pnl_orig"])

# Example: days with hybrid_block = True
blocked_days = diag[diag["hybrid_block"]].copy()
print("\nFirst few blocked days:")
print(blocked_days.head(10))

# -------------------------
# 6) Compare average daily PnL in/on vs off blocks
# -------------------------
on_block  = diag[diag["hybrid_block"]]
off_block = diag[~diag["hybrid_block"]]

def pnl_stats(x, label):
    return {
        "label": label,
        "n_days": len(x),
        "mean_pnl": x["pnl_orig"].mean(),
        "median_pnl": x["pnl_orig"].median(),
        "p05": x["pnl_orig"].quantile(0.05),
        "p95": x["pnl_orig"].quantile(0.95),
    }

stats_on  = pnl_stats(on_block, "on_block (hybrid_block=True)")
stats_off = pnl_stats(off_block, "off_block (hybrid_block=False)")

print("\n=== PnL per day: on vs off hybrid blocks (using ORIGINAL PnL) ===")
print(pd.DataFrame([stats_on, stats_off]).set_index("label"))

# -------------------------
# 7) Optional: visualize blocks as vertical stripes
# -------------------------
plt.figure()
cum_orig = pnl_original.cumsum()
cum_filt = pnl_filtered.cumsum()

ax = cum_orig.plot(label="Original", linewidth=1.0)
cum_filt.plot(ax=ax, label="Hybrid-filtered", linewidth=1.0)

for dt, row in flags.iterrows():
    if row.get("hybrid_block", False):
        ax.axvspan(dt, dt, alpha=0.15)  # thin vertical stripe

ax.set_title("Cumulative PnL with hybrid block days shaded")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative PnL")
ax.legend()
ax.grid(True)
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Extract outputs
daily_pnl = results["daily_pnl"]
daily_pnl_f = results["daily_pnl_filtered"]
block = results["block_series"]

# Cumulative PnL
cum_pnl = daily_pnl.cumsum()
cum_pnl_f = daily_pnl_f.cumsum()

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 10), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

# --- Top panel: cumulative PnL ---
ax1.plot(cum_pnl.index, cum_pnl.values, label="Original", linewidth=2)
ax1.plot(cum_pnl_f.index, cum_pnl_f.values, label="With Shock Blocker", linewidth=2)

# Shade blocked days
for d in block[block].index:
    ax1.axvspan(d, d + pd.Timedelta(days=1), color="red", alpha=0.08)

ax1.set_title("Cumulative PnL (bp/day) Before vs After Shock Blocker")
ax1.set_ylabel("Cumulative PnL (bp)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Bottom panel: block indicator ---
ax2.plot(block.index, block.astype(int), drawstyle="steps-post", color="red")
ax2.set_ylim(-0.1, 1.1)
ax2.set_ylabel("Blocked")
ax2.set_title("Shock Block Regime")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd

def analyze_shock_blocker(
    signals: pd.DataFrame,
    pos_overlay: pd.DataFrame,
    *,
    pnl_col: str = "pnl_net_bp",
    horizon_days: int = 7,             # Q1: short-term horizon (we chose 7)
    cool_off_days: int = 10,           # Q4: default cooling period
    trigger_mode: str = "A",           # Q2: "A" (any), "B" (confirm), "C" (weighted)
    min_signals_confirm: int = 2,      # for mode "B"
    score_threshold: float = 0.7,      # for mode "C"
    quantile_bad: float = 0.8,         # which tail to use for "danger" (80% tail)
    signal_cols: list[str] | None = None,
    verbose: bool = True,
):
    """
    Analyze short-term 'shock' regimes and build an intermittent blocking rule
    based on your signal dataframe and overlay PnL.

    Parameters
    ----------
    signals : DataFrame
        Index: daily bucket (DatetimeIndex, e.g. 'bucket' from regime_filter).
        Columns: signal series like:
            ['mean_z_comb', 'mean_abs_z_comb', 'median_abs_resid',
             'd_mean_z_comb', 'trendiness', 'resid_smoothed',
             'health_core', 'health_combined', ...]
    pos_overlay : DataFrame
        Closed overlay positions from portfolio_test_new.
        Must contain:
            - 'open_ts'  (datetime)
            - 'close_ts' (datetime)
            - pnl_col    (default 'pnl_net_bp')
    trigger_mode : {"A","B","C"}
        A: block if ANY signal is in its danger zone on a given day.
        B: block if >= min_signals_confirm signals are in danger.
        C: block if weighted danger-score >= score_threshold,
           with weights ~ |corr(signal, future_pnl)|.
    Lookahead handling
    ------------------
    - We build a 7d *future* PnL target for analysis ONLY.
    - The actual block flag for a given date t uses ONLY signals up to t-1:
          raw_trigger_t-1  -> block starting at t for cool_off_days.
    Returns
    -------
    results : dict with keys:
        'signal_stats' : per-signal regression / threshold info
        'block_series' : Boolean Series (index = dates, True = blocked)
        'daily_pnl'    : original daily pnl series (bp)
        'daily_pnl_filtered' : daily pnl with blocks applied
        'pos_filtered' : pos_overlay with blocked trades removed
        'summary'      : small DataFrame comparing before/after stats
    """
    # ----- 0) Basic setup -----
    if signal_cols is None:
        signal_cols = list(signals.columns)

    # Ensure datetime index
    sig = signals.copy()
    if not isinstance(sig.index, pd.DatetimeIndex):
        sig = sig.copy()
        sig.index = pd.to_datetime(sig.index)

    # ----- 1) Build daily PnL series from pos_overlay -----
    po = pos_overlay.copy()
    po["close_date"] = pd.to_datetime(po["close_ts"]).dt.floor("D")
    daily_pnl = (
        po.groupby("close_date")[pnl_col]
          .sum()
          .sort_index()
    )

    # Align daily pnl to signal index (fill missing days with 0 pnl)
    daily_pnl = daily_pnl.reindex(sig.index).fillna(0.0)

    # ----- 2) Build 7-day forward PnL target for analysis -----
    # future_pnl_t = sum_{d=1..horizon} pnl[t+d]
    future_pnl = (
        daily_pnl.shift(-1)
                 .rolling(horizon_days, min_periods=1)
                 .sum()
    )

    # Align with signals (drop last horizon_days where future_pnl is incomplete)
    valid_idx = future_pnl.index[:-horizon_days] if horizon_days > 0 else future_pnl.index
    future_pnl = future_pnl.loc[valid_idx]
    sig_aligned = sig.loc[valid_idx]

    # ----- 3) Per-signal analysis: sign of correlation & danger threshold -----
    stats_rows = []
    for col in signal_cols:
        x = sig_aligned[col].astype(float)
        y = future_pnl.astype(float)

        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]
        if len(x) < 50:  # not enough points
            continue

        x_mean = x.mean()
        y_mean = y.mean()
        cov_xy = ((x - x_mean) * (y - y_mean)).mean()
        var_x = ((x - x_mean) ** 2).mean()
        var_y = ((y - y_mean) ** 2).mean()

        if var_x <= 0 or var_y <= 0:
            continue

        beta = cov_xy / var_x
        corr = cov_xy / np.sqrt(var_x * var_y)

        # direction: which tail is "bad" (more negative future pnl)
        # If corr < 0: higher x -> lower y (worse PnL) -> high tail is bad
        if corr < 0:
            bad_side = "high"
            q = quantile_bad
        else:
            bad_side = "low"
            q = 1.0 - quantile_bad

        thr = x.quantile(q)

        if bad_side == "high":
            bad_mask = x >= thr
        else:
            bad_mask = x <= thr

        mean_pnl_bad = y[bad_mask].mean()
        mean_pnl_good = y[~bad_mask].mean()

        stats_rows.append({
            "signal": col,
            "beta": beta,
            "corr": corr,
            "bad_side": bad_side,
            "quantile": q,
            "threshold": thr,
            "mean_future_pnl_bad": mean_pnl_bad,
            "mean_future_pnl_good": mean_pnl_good,
            "n_bad": int(bad_mask.sum()),
            "n_good": int((~bad_mask).sum())
        })

    if not stats_rows:
        raise ValueError("No usable signals for shock analysis (too few points or zero variance).")

    signal_stats = pd.DataFrame(stats_rows).sort_values("mean_future_pnl_bad")

    if verbose:
        print("=== Shock-blocker signal stats (sorted by worst future pnl in danger zone) ===")
        display(signal_stats)

    # ----- 4) Build per-day 'danger' indicators using these thresholds -----
    danger_df = pd.DataFrame(index=sig.index)
    for _, row in signal_stats.iterrows():
        col = row["signal"]
        bad_side = row["bad_side"]
        thr = row["threshold"]
        x_full = sig[col].astype(float)

        if bad_side == "high":
            danger_df[col] = x_full >= thr
        else:
            danger_df[col] = x_full <= thr

    # ----- 5) Combine into a raw trigger series (today, pre-lookahead adjustment) -----
    trigger_mode = trigger_mode.upper()
    if trigger_mode == "A":
        # Block if ANY signal in danger
        raw_trigger = danger_df.any(axis=1)

    elif trigger_mode == "B":
        # Block if >= min_signals_confirm in danger
        raw_trigger = (danger_df.sum(axis=1) >= int(min_signals_confirm))

    elif trigger_mode == "C":
        # Weighted score: weights ~ |corr|
        w = signal_stats.set_index("signal")["corr"].abs()
        w = w / w.sum()
        # align to columns
        w_vec = w.reindex(danger_df.columns).fillna(0.0)
        risk_score = (danger_df.astype(float) * w_vec).sum(axis=1)
        raw_trigger = risk_score >= float(score_threshold)
    else:
        raise ValueError("trigger_mode must be 'A', 'B', or 'C'.")

    # ----- 6) MAKE IT LOOKAHEAD-SAFE + apply cool-off -----
    # Important:
    #  - raw_trigger_t = function of signals_t (end of day t info)
    #  - block should start at t+1 (decisions from next day onward)
    # So we shift by +1: block_base_{t+1} = raw_trigger_t
    block_base = raw_trigger.shift(1).fillna(False)

    # Now expand each trigger forward by cool_off_days
    block = block_base.copy().astype(bool)
    for i in range(1, cool_off_days):
        block |= block_base.shift(i).fillna(False)

    block = block.reindex(sig.index).fillna(False)

    if verbose:
        print("\n=== Shock-blocker summary ===")
        print(f"Total days: {len(block)}")
        print(f"Blocked days: {int(block.sum())} ({block.mean()*100:.1f}%)")

    # ----- 7) Apply block to trades by OPEN DATE (not close) -----
    po = pos_overlay.copy()
    po["open_date"] = pd.to_datetime(po["open_ts"]).dt.floor("D")
    blocked_dates = set(block[block].index)
    po["blocked"] = po["open_date"].isin(blocked_dates)

    pos_filtered = po[~po["blocked"]].copy()

    # Rebuild daily pnl from filtered trades
    daily_pnl_filtered = (
        pos_filtered.groupby("close_date")[pnl_col]
                    .sum()
                    .reindex(sig.index)
                    .fillna(0.0)
    )

    # ----- 8) Summary stats before / after -----
    def _stats(series):
        series = series.astype(float)
        mean = series.mean()
        std = series.std(ddof=1)
        sharpe_252 = (mean / std) * np.sqrt(252) if std > 0 else np.nan
        return mean, std, sharpe_252

    mean0, std0, sh0 = _stats(daily_pnl)
    mean1, std1, sh1 = _stats(daily_pnl_filtered)

    summary = pd.DataFrame({
        "mean_pnl_bp": [mean0, mean1],
        "std_pnl_bp": [std0, std1],
        "sharpe_252": [sh0, sh1],
        "n_trades": [len(pos_overlay), len(pos_filtered)],
        "n_days_blocked": [0, int(block.sum())],
        "pct_days_blocked": [0.0, block.mean()*100],
    }, index=["original", "with_shock_blocker"])

    if verbose:
        print("\n=== PnL summary (bp per day) ===")
        display(summary)

    return {
        "signal_stats": signal_stats,
        "block_series": block,
        "daily_pnl": daily_pnl,
        "daily_pnl_filtered": daily_pnl_filtered,
        "pos_filtered": pos_filtered,
        "summary": summary,
    }
    
results = analyze_shock_blocker(
    signals=signals,
    pos_overlay=pos_overlay,
    trigger_mode="A",          # or "B" / "C"
    cool_off_days=10,
    horizon_days=7,
    quantile_bad=0.8,
    verbose=True,
)

block = results["block_series"]
summary = results["summary"]
signal_stats = results["signal_stats"]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_regime_filter(signals, pos_overlay,
                           thresholds=None,
                           verbose=True):
    """
    Apply regime filter to overlay positions and compute:
    - cumulative pnl ON vs OFF
    - sharpe, hit rate
    - distribution plots

    signals: df indexed by 'bucket' (date)
    pos_overlay: df with open_ts, close_ts, pnl_net_bp
    """

    # -------------------------------------------------
    # 1. Default thresholds if not provided
    # -------------------------------------------------
    if thresholds is None:
        thresholds = {
            'z_change_high': signals['d_mean_z_comb'].quantile(0.70),
            'z_change_low':  signals['d_mean_z_comb'].quantile(0.30),

            'trend_low':    signals['trendiness'].quantile(0.30),
            'trend_high':   signals['trendiness'].quantile(0.70),

            'resid_high':   signals['median_abs_resid'].quantile(0.70),
            'absz_max':     signals['mean_abs_z_comb'].quantile(0.80),

            'health_bad':   signals['health_core'].quantile(0.70),
            'health_good':  signals['health_core'].quantile(0.30),
        }

    # -------------------------------------------------
    # 2. Merge signals onto positions at entry
    # -------------------------------------------------
    pos = pos_overlay.copy()
    pos['open_date'] = pos['open_ts'].dt.floor('D')

    merged = pos.merge(
        signals,
        left_on='open_date',
        right_index=True,
        how='left'
    )

    # -------------------------------------------------
    # 3. Apply regime ON/OFF rules
    # -------------------------------------------------
    m = merged

    regime_on = (
        (m['d_mean_z_comb'] > thresholds['z_change_high']) &
        (m['trendiness'] < thresholds['trend_low']) &
        (m['median_abs_resid'] > thresholds['resid_high']) &
        (m['mean_abs_z_comb'] < thresholds['absz_max'])
    )

    m['regime'] = np.where(regime_on, 'ON', 'OFF')

    # -------------------------------------------------
    # 4. Compute stats
    # -------------------------------------------------
    stats = m.groupby('regime')['pnl_net_bp'].agg(['count','mean','std'])
    stats['sharpe_252'] = (stats['mean'] / stats['std']) * np.sqrt(252)

    if verbose:
        print("\n=== Regime Summary ===")
        print(stats)

    # -------------------------------------------------
    # 5. Plots
    # -------------------------------------------------
    m['cum_pnl'] = m.groupby('regime')['pnl_net_bp'].cumsum()

    plt.figure(figsize=(12,6))
    for r in ['ON','OFF']:
        tmp = m[m['regime']==r]
        plt.plot(tmp['close_ts'], tmp['cum_pnl'], label=r)
    plt.legend(); plt.title("Cumulative PnL by Regime")
    plt.grid(True)
    plt.show()

    return m, stats, thresholds


import numpy as np
import pandas as pd

def analyze_regime_and_overlay(
    signals: pd.DataFrame,
    pos_overlay: pd.DataFrame,
    *,
    pnl_col: str = "pnl_net_bp",
    regime_quantiles: tuple[float, float] = (0.25, 0.75),
    fwd_windows: tuple[int, ...] = (1, 3, 5, 10),
):
    """
    Analyze how overlay PnL depends on regime signals.

    Parameters
    ----------
    signals : DataFrame
        From regime_filter.py. Index must be dates (bucket), columns are signal series
        like mean_z_comb, trendiness, health_core, health_combined, etc.
    pos_overlay : DataFrame
        Closed overlay positions from portfolio_test_new (pos_overlay).
        Must contain 'close_ts' and the PnL column (default 'pnl_net_bp').
    pnl_col : str
        Column in pos_overlay to use as PnL (bp). We'll sum by close date.
    regime_quantiles : (low_q, high_q)
        Quantile cutoffs for low/mid/high regimes per signal.
    fwd_windows : tuple of ints
        Horizons (in days) for forward cumulative PnL from each date t
        (t+1 .. t+k), used to assess predictability without lookahead.

    Returns
    -------
    results : dict
        {
          "daily_pnl": daily_pnl_df,
          "merged": merged_df,
          "regime_stats": regime_stats_df,
          "forward_stats": forward_stats_df,
        }
    """

    # --- 1) Build daily PnL series from pos_overlay ---
    po = pos_overlay.copy()
    po["close_ts"] = pd.to_datetime(po["close_ts"])
    po["bucket"] = po["close_ts"].dt.floor("D")

    daily_pnl = (
        po.groupby("bucket")[pnl_col]
        .sum()
        .rename("pnl_daily")
        .to_frame()
        .sort_index()
    )

    # --- 2) Align with signals on date index ---
    sig = signals.copy()
    # your signals index is already "bucket", but ensure datetime:
    sig.index = pd.to_datetime(sig.index)

    merged = sig.join(daily_pnl, how="left")
    # days with no overlay trades → 0 pnl
    merged["pnl_daily"] = merged["pnl_daily"].fillna(0.0)

    # Automatically pick numeric signal columns
    signal_cols = [
        c for c in merged.columns
        if c != "pnl_daily" and np.issubdtype(merged[c].dtype, np.number)
    ]

    low_q, high_q = regime_quantiles

    # --- 3) Build regimes for each signal ---
    for col in signal_cols:
        series = merged[col].dropna()
        if series.empty:
            continue
        lo = series.quantile(low_q)
        hi = series.quantile(high_q)

        reg_col = f"{col}_regime"
        merged[reg_col] = np.where(
            merged[col] <= lo,
            "low",
            np.where(merged[col] >= hi, "high", "mid"),
        )

    # --- 4) Per-regime same-day PnL stats ---
    regime_rows = []
    for col in signal_cols:
        reg_col = f"{col}_regime"
        if reg_col not in merged.columns:
            continue

        tmp = merged[[col, reg_col, "pnl_daily"]].dropna(subset=[col])
        for regime in ["low", "mid", "high"]:
            sub = tmp[tmp[reg_col] == regime]
            if sub.empty:
                continue

            pnl = sub["pnl_daily"]
            mean_pnl = pnl.mean()
            std_pnl = pnl.std(ddof=0)
            sharpe = (
                mean_pnl / std_pnl * np.sqrt(252)
                if std_pnl > 0
                else np.nan
            )
            hit_rate = (pnl > 0).mean()
            avg_signal = sub[col].mean()

            regime_rows.append(
                {
                    "signal": col,
                    "regime": regime,
                    "n_days": len(sub),
                    "mean_pnl_bp": mean_pnl,
                    "std_pnl_bp": std_pnl,
                    "sharpe_252": sharpe,
                    "hit_rate": hit_rate,
                    "avg_signal": avg_signal,
                }
            )

    regime_stats = pd.DataFrame(regime_rows).set_index(["signal", "regime"]).sort_index()

    # --- 5) Forward-window PnL stats (no lookahead) ---
    merged = merged.sort_index()
    for k in fwd_windows:
        # forward PnL from t+1 .. t+k
        merged[f"pnl_fwd_{k}d"] = (
            merged["pnl_daily"]
            .shift(-1)
            .rolling(window=k, min_periods=k)
            .sum()
        )

    fwd_rows = []
    for col in signal_cols:
        reg_col = f"{col}_regime"
        if reg_col not in merged.columns:
            continue

        base = merged[[col, reg_col, "pnl_daily"] +
                      [f"pnl_fwd_{k}d" for k in fwd_windows]].dropna(subset=[col])

        for regime in ["low", "mid", "high"]:
            sub = base[base[reg_col] == regime]
            if sub.empty:
                continue

            for k in fwd_windows:
                fwd_col = f"pnl_fwd_{k}d"
                pnl = sub[fwd_col].dropna()
                if pnl.empty:
                    continue

                mean_pnl = pnl.mean()
                std_pnl = pnl.std(ddof=0)
                sharpe = (
                    mean_pnl / std_pnl * np.sqrt(252 / k)
                    if std_pnl > 0
                    else np.nan
                )
                hit_rate = (pnl > 0).mean()

                fwd_rows.append(
                    {
                        "signal": col,
                        "regime": regime,
                        "horizon_days": k,
                        "n_obs": len(pnl),
                        "mean_fwd_pnl_bp": mean_pnl,
                        "std_fwd_pnl_bp": std_pnl,
                        "sharpe_252_scaled": sharpe,
                        "hit_rate": hit_rate,
                    }
                )

    forward_stats = (
        pd.DataFrame(fwd_rows)
        .set_index(["signal", "regime", "horizon_days"])
        .sort_index()
    )

    # --- 6) Print a quick summary for inspection ---
    print("\n=== Same-day PnL by regime (bp) ===")
    print(regime_stats[["n_days", "mean_pnl_bp", "sharpe_252", "hit_rate"]])

    print("\n=== Forward PnL by regime & horizon (bp) ===")
    print(
        forward_stats[["mean_fwd_pnl_bp", "sharpe_252_scaled", "hit_rate"]]
        .groupby(level=[0, 1, 2])
        .first()
    )

    return {
        "daily_pnl": daily_pnl,
        "merged": merged,
        "regime_stats": regime_stats,
        "forward_stats": forward_stats,
    }
    
results = analyze_regime_and_overlay(signals, pos_overlay)

import numpy as np
import pandas as pd

def analyze_regime_and_overlay(
    signals: pd.DataFrame,
    pos_overlay: pd.DataFrame,
    *,
    signal_col: str = "signal_health",
    pnl_col: str = "pnl_cash",
    horizons=(1, 5, 10),
    thresholds: list[float] | None = None,
    K_grid: list[int] | None = None,
):
    """
    Drop-in analysis:
      - Builds daily overlay PnL series from pos_overlay ledger.
      - Aligns with signals by timestamp index.
      - 1) Threshold vs future PnL stats on `signal_col`.
      - 2) Regime blocking sims: if signal <= threshold, block for K days.

    Returns dict:
      {
        "baseline": {...},
        "threshold_results": DataFrame,
        "blocking_results": DataFrame,
        "aligned_signals": DataFrame,
        "aligned_pnl": Series
      }
    """

    # --------- Helpers inside so it's fully self-contained ---------
    def _max_drawdown(series: pd.Series) -> float:
        cum = series.cumsum()
        running_max = cum.cummax()
        dd = cum - running_max
        return dd.min()  # negative

    def _simple_sharpe(pnl: pd.Series) -> float:
        if len(pnl) < 2:
            return np.nan
        mu = pnl.mean()
        sig = pnl.std(ddof=1)
        return np.nan if sig == 0 else mu / sig

    def _make_future_pnl(pnl: pd.Series, horizons_):
        fut = {}
        for H in horizons_:
            fut[f"future_{H}d_pnl"] = (
                pnl.rolling(H, min_periods=H).sum().shift(-H)
            )
        return pd.DataFrame(fut)

    # --------- 1) Build pnl_series from pos_overlay ---------
    df_ledger = pos_overlay.copy()

    # If ledger has both strategy + overlay, keep only overlay marks
    if "mode" in df_ledger.columns:
        df_ledger = df_ledger[df_ledger["mode"] == "overlay"]
    if "event" in df_ledger.columns:
        df_ledger = df_ledger[df_ledger["event"] == "mark"]

    # Ensure we have decision_ts + pnl_col
    if "decision_ts" not in df_ledger.columns:
        raise ValueError("pos_overlay must have a 'decision_ts' column.")
    if pnl_col not in df_ledger.columns:
        raise ValueError(f"pos_overlay must have '{pnl_col}' column.")

    df_ledger = df_ledger.copy()
    df_ledger["decision_ts"] = pd.to_datetime(df_ledger["decision_ts"])

    # Aggregate to one PnL number per decision bucket (usually daily)
    pnl_series = (
        df_ledger
        .groupby("decision_ts")[pnl_col]
        .sum()
        .sort_index()
    )

    # --------- 2) Align with signals ---------
    sig = signals.copy()

    # If signal_ts is a column, move it to index
    if signal_col not in sig.columns:
        raise ValueError(f"{signal_col} not found in signals columns.")

    if not isinstance(sig.index, pd.DatetimeIndex):
        # Try to infer a time column if index is not datetime
        if "decision_ts" in sig.columns:
            sig_index = pd.to_datetime(sig["decision_ts"])
            sig = sig.set_index(sig_index)
        else:
            sig.index = pd.to_datetime(sig.index)

    if not isinstance(pnl_series.index, pd.DatetimeIndex):
        pnl_series.index = pd.to_datetime(pnl_series.index)

    common_index = sig.index.intersection(pnl_series.index)
    sig = sig.loc[common_index].sort_index()
    pnl_series = pnl_series.loc[common_index].sort_index()

    if sig.empty or pnl_series.empty:
        raise ValueError("No overlapping dates between signals and pos_overlay PnL.")

    # --------- 3) Baseline stats (no regime filter) ---------
    baseline = {
        "total_pnl": float(pnl_series.sum()),
        "pnl_per_day": float(pnl_series.mean()),
        "max_drawdown": float(_max_drawdown(pnl_series)),
        "sharpe_like": float(_simple_sharpe(pnl_series)),
        "n_days": int(len(pnl_series)),
    }

    # --------- 4) Threshold vs future PnL analysis ---------
    df = pd.concat(
        [
            sig[[signal_col]].rename(columns={signal_col: "signal"}),
            pnl_series.rename("pnl"),
        ],
        axis=1,
    ).dropna()

    fut = _make_future_pnl(df["pnl"], horizons)
    df = df.join(fut).dropna()

    if thresholds is None:
        # Use lower quantiles as candidate grid
        qs = np.linspace(0.05, 0.95, 19)
        thresholds = list(np.quantile(df["signal"], qs))

    rows_thr = []
    for thr in thresholds:
        bad_mask = df["signal"] <= thr
        frac_bad = bad_mask.mean()

        stats = {"threshold": thr, "frac_bad": frac_bad}

        for H in horizons:
            col = f"future_{H}d_pnl"
            x_bad = df.loc[bad_mask, col]
            x_good = df.loc[~bad_mask, col]

            if len(x_bad) < 20 or len(x_good) < 20:
                mu_bad = mu_good = tstat = np.nan
            else:
                mu_bad = x_bad.mean()
                mu_good = x_good.mean()
                s_bad = x_bad.var(ddof=1)
                s_good = x_good.var(ddof=1)
                denom = np.sqrt(s_bad / len(x_bad) + s_good / len(x_good))
                tstat = np.nan if denom == 0 else (mu_bad - mu_good) / denom

            stats[f"mu_bad_{H}d"] = mu_bad
            stats[f"mu_good_{H}d"] = mu_good
            stats[f"tstat_bad_vs_good_{H}d"] = tstat

        rows_thr.append(stats)

    threshold_results = (
        pd.DataFrame(rows_thr)
        .sort_values("threshold")
        .reset_index(drop=True)
    )

    # --------- 5) Regime blocking simulation (threshold + K) ---------
    if K_grid is None:
        K_grid = [2, 5, 10, 15, 20]

    idx = df.index
    n = len(df)
    blocking_rows = []

    triggers_all = {thr: (df["signal"] <= thr) for thr in thresholds}

    for thr, triggers in triggers_all.items():
        for K in K_grid:
            blocked = np.zeros(n, dtype=bool)
            block_until = -1

            # If signal is bad at t, block PnL starting at t+1 for K days
            for i, (_, is_bad) in enumerate(triggers.items()):
                if i <= block_until:
                    blocked[i] = True
                    continue

                if is_bad:
                    start_idx = i + 1
                    end_idx = min(n - 1, start_idx + K - 1)
                    if start_idx < n:
                        blocked[start_idx : end_idx + 1] = True
                        block_until = end_idx

            df_blocked = df.copy()
            df_blocked["blocked"] = blocked
            df_blocked["pnl_effective"] = np.where(
                df_blocked["blocked"], 0.0, df_blocked["pnl"]
            )

            pnl_eff = df_blocked["pnl_effective"]
            pnl_total = pnl_eff.sum()
            pnl_per_day = pnl_total / len(pnl_eff)
            dd = _max_drawdown(pnl_eff)
            sharpe = _simple_sharpe(pnl_eff)
            frac_blocked = df_blocked["blocked"].mean()
            frac_trading = 1.0 - frac_blocked

            blocking_rows.append(
                {
                    "threshold": thr,
                    "K_days": K,
                    "total_pnl": pnl_total,
                    "pnl_per_day": pnl_per_day,
                    "max_drawdown": dd,
                    "sharpe_like": sharpe,
                    "frac_blocked": frac_blocked,
                    "frac_trading": frac_trading,
                }
            )

    blocking_results = (
        pd.DataFrame(blocking_rows)
        .sort_values(by=["sharpe_like", "total_pnl"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return {
        "baseline": baseline,
        "threshold_results": threshold_results,
        "blocking_results": blocking_results,
        "aligned_signals": sig,
        "aligned_pnl": pnl_series,
    }
    
res = analyze_regime_and_overlay(signals_df, pos_overlay)

res["baseline"]           # raw overlay PnL stats
res["threshold_results"]  # where signal actually predicts bad forward PnL
res["blocking_results"]   # candidate (threshold, K) combos to try


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Assumes you already have these from regime_filter.build_regime_state(...) ---
# signals_df: DataFrame indexed by decision_ts (or date), with signal columns
# regime_df : DataFrame indexed similarly, with at least a 'regime' column (0/1)

# 1) Basic safety copies
signals_df = signals_df.copy()
regime_df = regime_df.copy()

# 2) Ensure DateTimeIndex if possible
if not isinstance(signals_df.index, pd.DatetimeIndex):
    signals_df.index = pd.to_datetime(signals_df.index)

if not isinstance(regime_df.index, pd.DatetimeIndex):
    regime_df.index = pd.to_datetime(regime_df.index)

# 3) Align on common index to avoid mismatches
common_index = signals_df.index.intersection(regime_df.index)
signals_df = signals_df.loc[common_index].sort_index()
regime_df  = regime_df.loc[common_index].sort_index()

# 4) Pull the regime series (required)
if "regime" not in regime_df.columns:
    raise ValueError("regime_df must contain a 'regime' column (0/1).")

reg = regime_df["regime"].astype(float).fillna(0.0)

# 5) Determine which signal columns to plot (all numeric, excluding 'regime' if present)
signal_cols = []
for col in signals_df.columns:
    if col == "regime":
        continue
    if pd.api.types.is_numeric_dtype(signals_df[col]):
        signal_cols.append(col)

if not signal_cols:
    raise ValueError("No numeric signal columns found in signals_df to plot.")

n_plots = len(signal_cols) + 1  # signals + regime

# 6) Find contiguous blocked intervals (regime == 1) to shade
starts = (reg == 1) & (reg.shift(1, fill_value=0) == 0)
ends   = (reg == 0) & (reg.shift(1, fill_value=0) == 1)

start_times = reg.index[starts]
end_times   = reg.index[ends]

# If regime ends in blocked state, close last interval at final timestamp
if len(end_times) < len(start_times):
    end_times = end_times.append(pd.Index([reg.index[-1]]))

blocked_intervals = list(zip(start_times, end_times))

def shade_blocked(ax):
    """Shade regime==1 intervals on the given axis."""
    for s, e in blocked_intervals:
        ax.axvspan(s, e, alpha=0.15)  # let Matplotlib choose color; just a light band

# 7) Build figure with all signal panels + regime panel
fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3 * n_plots), sharex=True)
if n_plots == 1:
    axes = [axes]  # make iterable if only one

# 7a) Plot each signal column with blocked shading
for i, col in enumerate(signal_cols):
    ax = axes[i]
    ax.plot(signals_df.index, signals_df[col], linewidth=1.6, label=col)
    shade_blocked(ax)
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

# 7b) Final panel: regime itself (0/1) as a step function
ax_reg = axes[-1]
ax_reg.step(reg.index, reg.values, where="post", linewidth=1.6, label="regime (0=OK, 1=blocked)")
shade_blocked(ax_reg)
ax_reg.set_ylabel("regime")
ax_reg.set_ylim(-0.1, 1.1)
ax_reg.grid(True, alpha=0.3)
ax_reg.legend(loc="upper left")

plt.suptitle("Signals and Regime (shaded = blocked regime)", y=0.99)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Assuming these exist:
# signals_df, regime_df

fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# --- 1) Signal Health ---
ax[0].plot(signals_df.index, signals_df["health"], label="Signal Health", linewidth=1.8)
ax[0].axhline(0, color='black', linewidth=0.8)
ax[0].set_title("Signal Health Over Time")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

# --- 2) Raw shocks (day-over-day) ---
ax[1].plot(signals_df.index, signals_df["shock_raw"].astype(int), 
           drawstyle="steps-post", label="Raw Shock Flags")
ax[1].set_title("Shock Flags (Before Moving Window)")
ax[1].grid(True, alpha=0.3)
ax[1].legend()

# --- 3) Final regime ---
ax[2].plot(regime_df.index, regime_df["regime"].astype(int),
           drawstyle="steps-post", label="Final Regime")
ax[2].set_title("Final Regime State (0 = OK, 1 = Blocked)")
ax[2].grid(True, alpha=0.3)
ax[2].legend()

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(14, 5))

# Signal health line
ax.plot(signals_df.index, signals_df["health"], label="Signal Health", linewidth=1.8)

# Add shaded regions where regime == 1 (blocked)
for ts, val in regime_df["regime"].iteritems():
    if val == 1:
        ax.axvspan(ts, ts, color="red", alpha=0.15)

ax.set_title("Signal Health with Regime Blocking")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# If you have z_comb series for comparison:
z_comb = signals_df["z_comb"] if "z_comb" in signals_df else None

if z_comb is not None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(z_comb.index, z_comb, label="z_comb", linewidth=1.5)

    # Shade blocks
    for ts, val in regime_df["regime"].items():
        if val == 1:
            ax.axvspan(ts, ts, color="red", alpha=0.15)

    ax.set_title("z_comb with Regime Filter")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()
    
fig, ax = plt.subplots(figsize=(15, 5))

# main signal — pick whichever you rely on most.
ax.plot(signals_df.index, signals_df["health"], label="Health", linewidth=1.8)

# Regime shading
reg = regime_df["regime"]
starts = reg[(reg == 1) & (reg.shift(1, fill_value=0) == 0)].index
ends   = reg[(reg == 0) & (reg.shift(1, fill_value=0) == 1)].index

# ensure ends align
if len(ends) < len(starts):
    ends = ends.append(pd.Index([reg.index[-1]]))

for s, e in zip(starts, ends):
    ax.axvspan(s, e, color='red', alpha=0.15)

ax.set_title("Regime Filter Applied to Signal Health")
ax.legend()
ax.grid(True)
plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import cr_config as cr  # assumes cr_config.py is importable from this notebook

plt.rcParams["figure.figsize"] = (12, 4)

# ---------- FILE ITERATION ----------

def iter_enhanced_files():
    """
    Yield enhanced parquet files one by one from PATH_ENH matching ENH_SUFFIX,
    e.g. *_d.parquet or *_h.parquet.
    """
    enh_path = Path(cr.PATH_ENH)
    pattern = f"*{cr.ENH_SUFFIX}.parquet"
    files = sorted(enh_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No enhanced parquet files matching {pattern} in {enh_path}")
    for f in files:
        yield f


# ---------- PER-FILE STATS (SMALL) ----------

def compute_file_stats(parquet_path: Path) -> pd.DataFrame:
    """
    Load ONE enhanced parquet file and compute per-bucket summary stats.
    Returns a small DataFrame (one row per decision bucket in that file).
    """
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return pd.DataFrame()

    if "ts" not in df.columns or "tenor_yrs" not in df.columns:
        raise ValueError(f"{parquet_path} missing 'ts' or 'tenor_yrs'.")

    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "tenor_yrs"])

    freq = str(cr.DECISION_FREQ).upper()
    if freq == "D":
        df["bucket"] = df["ts"].dt.floor("D")
    elif freq == "H":
        df["bucket"] = df["ts"].dt.floor("H")
    else:
        raise ValueError("cr.DECISION_FREQ must be 'D' or 'H'.")

    if "z_comb" not in df.columns:
        raise ValueError(f"{parquet_path} has no 'z_comb' column.")

    have_z_pca    = ("z_pca" in df.columns) and df["z_pca"].notna().any()
    have_z_spline = ("z_spline" in df.columns) and df["z_spline"].notna().any()

    if have_z_pca and have_z_spline:
        df["abs_resid"] = (df["z_spline"] - df["z_pca"]).abs()
    else:
        df["abs_resid"] = np.nan

    g = df.groupby("bucket", sort=True)

    stats = g.apply(
        lambda g_: pd.Series({
            "n_tenors": g_["tenor_yrs"].nunique(),
            "z_std": g_["z_comb"].std(skipna=True),
            "z_iqr": g_["z_comb"].quantile(0.75) - g_["z_comb"].quantile(0.25),
            "z_max_abs": g_["z_comb"].abs().max(),
            "median_abs_resid": g_["abs_resid"].median(skipna=True),
            "p95_abs_resid": (
                g_["abs_resid"].quantile(0.95)
                if g_["abs_resid"].notna().any()
                else np.nan
            ),
            "mean_z": g_["z_comb"].mean(skipna=True),
        })
    ).reset_index()

    return stats


# ---------- STREAM ALL FILES & BUILD GLOBAL STATS ----------

def build_global_stats_streaming() -> pd.DataFrame:
    """
    Iterate over all enhanced files, compute small per-bucket stats per file,
    and concatenate them. This keeps memory usage low.
    """
    all_stats = []
    for f in iter_enhanced_files():
        print(f"[INFO] Processing {f.name} ...")
        s = compute_file_stats(f)
        if not s.empty:
            all_stats.append(s)

    if not all_stats:
        raise ValueError("No non-empty per-bucket stats produced from enhanced files.")

    stats = pd.concat(all_stats, ignore_index=True)

    # Ensure sorted by time
    stats = stats.sort_values("bucket").reset_index(drop=True)
    return stats


# ---------- STRESS / HEALTH METRICS (NO RAW PANEL NEEDED) ----------

def add_signal_health(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Given per-bucket stats with columns:
      bucket, z_std, median_abs_resid, mean_z
    compute:
      - z_stress
      - resid_stress
      - trend_stress (from Δ mean_z autocorr)
      - signal_health (PCA+z)
      - signal_health_combined (incl. trend)
    """
    out = stats.copy()

    # Robust scale for z_std
    z_std_med = out["z_std"].median(skipna=True)
    z_std_hi  = out["z_std"].quantile(0.9) if out["z_std"].notna().any() else np.nan

    resid_med = out["median_abs_resid"].median(skipna=True)
    resid_hi  = out["median_abs_resid"].quantile(0.9) if out["median_abs_resid"].notna().any() else np.nan

    def _normalize(x, mid, hi):
        if not np.isfinite(mid) or not np.isfinite(hi) or hi <= mid:
            return np.zeros_like(x)
        return np.clip((x - mid) / (hi - mid), 0.0, 1.0)

    out["z_stress"] = _normalize(out["z_std"].values, z_std_med, z_std_hi)
    out["resid_stress"] = _normalize(out["median_abs_resid"].values, resid_med, resid_hi)

    out["stress_score"] = 0.5 * out["z_stress"] + 0.5 * out["resid_stress"]
    out["signal_health"] = 1.0 - out["stress_score"]

    # --- trend / persistence based only on mean_z series ---
    out = out.sort_values("bucket").reset_index(drop=True)
    out["d_mean_z"] = out["mean_z"].diff()

    window = 20  # you can tune this
    ac_vals = []
    d = out["d_mean_z"]
    for i in range(len(d)):
        if i < window:
            ac_vals.append(np.nan)
            continue
        x = d.iloc[i-window+1:i+1]
        if x.isna().any():
            ac_vals.append(np.nan)
            continue
        ac_vals.append(x.autocorr(lag=1))

    out["d_mean_z_ac1"] = ac_vals

    ac_series = out["d_mean_z_ac1"]
    ac_med = ac_series.median(skipna=True)
    ac_hi  = ac_series.quantile(0.9) if ac_series.notna().any() else np.nan

    if np.isfinite(ac_med) and np.isfinite(ac_hi) and ac_hi > ac_med:
        trend_stress = np.clip((ac_series - ac_med) / (ac_hi - ac_med), 0.0, 1.0)
    else:
        trend_stress = pd.Series(np.zeros(len(ac_series)), index=ac_series.index)

    out["trend_stress"] = trend_stress
    out["trend_health"] = 1.0 - out["trend_stress"]

    # Combined health
    out["stress_score_combined"] = (
        0.4 * out["z_stress"] +
        0.4 * out["resid_stress"] +
        0.2 * out["trend_stress"].fillna(0.0)
    )
    out["signal_health_combined"] = 1.0 - out["stress_score_combined"]

    return out


# ---------- RUN & PLOT ----------

stats = build_global_stats_streaming()
stats = add_signal_health(stats)

display(stats.head())

# Signal health
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["signal_health"], label="PCA / z signal_health")
ax.plot(stats["bucket"], stats["signal_health_combined"], label="Combined (incl. trend)", alpha=0.7)
ax.set_title("Signal Health Over Time")
ax.set_ylabel("Health (1 = good, 0 = bad)")
ax.legend()
plt.tight_layout()
plt.show()

# PCA residuals
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["median_abs_resid"], label="median |z_spline - z_pca|")
ax.set_title("PCA Residuals Over Time")
ax.legend()
plt.tight_layout()
plt.show()

# Trend metric
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["d_mean_z_ac1"], label="AC1 of Δ mean_z")
ax.set_title("Trendiness of z_comb (higher = more trending / less mean-reverting)")
ax.legend()
plt.tight_layout()
plt.show()

# --- imports & config ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import cr_config as cr  # assumes cr_config.py is importable from this notebook

plt.rcParams["figure.figsize"] = (12, 4)

# --- helper: load all enhanced parquet files into one panel ---

def load_enhanced_panel():
    """
    Load all enhanced files from PATH_ENH that match ENH_SUFFIX.
    Returns a single DataFrame with ts, tenor_yrs, z_spline, z_pca, z_comb, etc.
    """
    enh_path = Path(cr.PATH_ENH)
    pattern = f"*{cr.ENH_SUFFIX}.parquet"  # e.g. *_d.parquet or *_h.parquet
    files = sorted(enh_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No enhanced parquet files matching {pattern} in {enh_path}")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        df["src_file"] = f.name
        dfs.append(df)

    if not dfs:
        raise ValueError("All enhanced parquet files are empty.")

    panel = pd.concat(dfs, ignore_index=True)
    if "ts" not in panel.columns or "tenor_yrs" not in panel.columns:
        raise ValueError("Enhanced files must contain at least 'ts' and 'tenor_yrs' columns.")

    panel["ts"] = pd.to_datetime(panel["ts"], utc=False, errors="coerce")
    panel = panel.dropna(subset=["ts", "tenor_yrs"])

    # decision bucket consistent with backtest
    freq = str(cr.DECISION_FREQ).upper()
    if freq == "D":
        panel["bucket"] = panel["ts"].dt.floor("D")
    elif freq == "H":
        panel["bucket"] = panel["ts"].dt.floor("H")
    else:
        raise ValueError("cr.DECISION_FREQ must be 'D' or 'H'.")

    return panel


# --- helper: compute diagnostics per decision bucket ---

def compute_signal_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each decision bucket, compute:
      - cross-sectional dispersion of z_comb
      - PCA residual stats: |z_spline - z_pca|
      - basic counts
      And build a simple signal_health index in [0, 1].
    """
    have_z_pca = ("z_pca" in panel.columns) and panel["z_pca"].notna().any()
    have_z_spline = ("z_spline" in panel.columns) and panel["z_spline"].notna().any()

    if "z_comb" not in panel.columns:
        raise ValueError("Expected 'z_comb' in enhanced panel.")

    if not have_z_pca or not have_z_spline:
        print("[WARN] z_pca and/or z_spline not present/usable. PCA residual diagnostics will be skipped.")

    df = panel.copy()

    if have_z_pca and have_z_spline:
        df["resid"] = df["z_spline"] - df["z_pca"]
        df["abs_resid"] = df["resid"].abs()
    else:
        df["abs_resid"] = np.nan

    # group by decision bucket
    grouped = df.groupby("bucket", sort=True)

    stats = grouped.apply(
        lambda g: pd.Series({
            "n_tenors": g["tenor_yrs"].nunique(),
            "z_std": g["z_comb"].std(skipna=True),
            "z_iqr": (g["z_comb"].quantile(0.75) - g["z_comb"].quantile(0.25)),
            "z_max_abs": g["z_comb"].abs().max(),
            "median_abs_resid": g["abs_resid"].median(skipna=True),
            "p95_abs_resid": g["abs_resid"].quantile(0.95) if g["abs_resid"].notna().any() else np.nan,
        })
    ).reset_index()

    # --- build a very simple "signal_health" index ---
    # Higher dispersion and larger residuals => worse health.

    # robust scales (avoid division by zero)
    z_std_med = stats["z_std"].median(skipna=True)
    z_std_hi = stats["z_std"].quantile(0.9) if stats["z_std"].notna().any() else np.nan

    resid_med = stats["median_abs_resid"].median(skipna=True)
    resid_hi = stats["median_abs_resid"].quantile(0.9) if stats["median_abs_resid"].notna().any() else np.nan

    def _normalize(x, mid, hi):
        if not np.isfinite(mid) or not np.isfinite(hi) or hi <= mid:
            return np.zeros_like(x)
        # 0 near mid, 1 near hi, clipped
        return np.clip((x - mid) / (hi - mid), 0.0, 1.0)

    stats["z_stress"] = _normalize(stats["z_std"].values, z_std_med, z_std_hi)  # shape change / stress
    stats["resid_stress"] = _normalize(stats["median_abs_resid"].values, resid_med, resid_hi)

    # composite stress and health
    stats["stress_score"] = 0.5 * stats["z_stress"] + 0.5 * stats["resid_stress"]
    stats["signal_health"] = 1.0 - stats["stress_score"]

    return stats.sort_values("bucket").reset_index(drop=True)


# --- helper: rough "trending vs reversion" metric using z_comb changes ---

def add_trend_metric(stats: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a simple trending metric based on cross-sectional mean z_comb changes.
    This is very coarse but gives a sense of whether z moves are persistent.
    """
    df = panel.copy()
    # cross-sectional mean z per bucket
    mean_z = df.groupby("bucket")["z_comb"].mean().rename("mean_z").to_frame().reset_index()
    mean_z["d_mean_z"] = mean_z["mean_z"].diff()

    # rolling window for autocorr / trendiness
    window = 20  # ~1 month if daily, or adjust
    d = mean_z["d_mean_z"]

    ac_vals = []
    for i in range(len(d)):
        if i < window:
            ac_vals.append(np.nan)
            continue
        x = d.iloc[i-window+1:i+1]
        if x.isna().any():
            ac_vals.append(np.nan)
            continue
        # lag-1 autocorr approx
        ac = x.autocorr(lag=1)
        ac_vals.append(ac)

    mean_z["d_mean_z_ac1"] = ac_vals

    # normalize trendiness into [0,1] "trend_stress"
    ac_series = mean_z["d_mean_z_ac1"]
    ac_med = ac_series.median(skipna=True)
    ac_hi = ac_series.quantile(0.9) if ac_series.notna().any() else np.nan

    if np.isfinite(ac_med) and np.isfinite(ac_hi) and ac_hi > ac_med:
        trend_stress = np.clip((ac_series - ac_med) / (ac_hi - ac_med), 0.0, 1.0)
    else:
        trend_stress = pd.Series(np.zeros(len(ac_series)), index=ac_series.index)

    mean_z["trend_stress"] = trend_stress
    mean_z["trend_health"] = 1.0 - mean_z["trend_stress"]

    # merge into stats
    out = stats.merge(mean_z[["bucket", "mean_z", "d_mean_z", "d_mean_z_ac1", "trend_stress", "trend_health"]],
                      on="bucket", how="left")
    # update overall signal_health to include trend
    out["stress_score_combined"] = (
        0.4 * out["z_stress"] +
        0.4 * out["resid_stress"] +
        0.2 * out["trend_stress"].fillna(0.0)
    )
    out["signal_health_combined"] = 1.0 - out["stress_score_combined"]

    return out


# --- run everything and plot ---

panel = load_enhanced_panel()
stats = compute_signal_diagnostics(panel)
stats = add_trend_metric(stats, panel)

display(stats.head())

# Plot basic signal health over time
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["signal_health"], label="PCA/residual signal_health")
ax.plot(stats["bucket"], stats["signal_health_combined"], label="Combined (incl. trend)", alpha=0.7)
ax.set_title("Signal Health Over Time")
ax.set_ylabel("Health (1 = good, 0 = bad)")
ax.legend()
plt.tight_layout()
plt.show()

# Optionally look at residual stress directly
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["median_abs_resid"], label="median |z_spline - z_pca|")
ax.set_title("PCA Residuals Over Time")
ax.legend()
plt.tight_layout()
plt.show()

# Optionally inspect trending metric
fig, ax = plt.subplots()
ax.plot(stats["bucket"], stats["d_mean_z_ac1"], label="AC1 of Δ mean z_comb")
ax.set_title("Trendiness of z_comb (higher = more trending / less mean-reverting)")
ax.legend()
plt.tight_layout()
plt.show()





import cr_analytics as cra

cra.overlay_full_report(pos_overlay, diag_overlay, label="overlay_D")


from pathlib import Path

PATH_OUT = Path(cr.PATH_OUT)

# Example: use a different OUT_SUFFIX for overlay vs pure strategy if you like
suffix_overlay = cr.OUT_SUFFIX  # or "_overlay_d" if you set it that way

pos_overlay = pd.read_parquet(PATH_OUT / f"positions_ledger{suffix_overlay}.parquet")
diag_overlay = pd.read_parquet(PATH_OUT / f"overlay_diag{suffix_overlay}.parquet")

summarize_overlay_backtest(pos_overlay, diag_overlay, label="overlay D")



import pandas as pd
import overlay_diag as od

hedge_df = pd.read_parquet("my_trade_tape.parquet")
months = ["2304", "2305"]

diag = od.analyze_overlay(months, hedge_df)

# See why things are failing:
diag["reason"].value_counts()

# Look at failed ones by category:
diag[diag["reason"] == "no_exec_tenor"].head()
diag[diag["reason"] == "no_zdisp_ge_entry"].head()
diag[diag["reason"] == "fly_block"].head()
diag[diag["reason"] == "caps_block"].head()

# Check that any would have opened:
diag[diag["reason"] == "opened"].head()






ledger["dv01_leg_i"] = np.abs(ledger["w_i"]) * ledger["tenor_i"] / (1 + 0.01*ledger["rate_i"])
ledger["dv01_leg_j"] = np.abs(ledger["w_j"]) * ledger["tenor_j"] / (1 + 0.01*ledger["rate_j"])
ledger["pair_dv01"]  = ledger["dv01_leg_i"] + ledger["dv01_leg_j"]

# average and peak concurrent exposure
avg_dv01 = ledger.groupby("decision_ts")["pair_dv01"].sum().mean()
max_dv01 = ledger.groupby("decision_ts")["pair_dv01"].sum().max()
print(f"Average concurrent DV01 units: {avg_dv01:.2f}, peak: {max_dv01:.2f}")

# === RV analytics (drop-in) ==============================================
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import cr_config as cr

PATH_OUT = Path(cr.PATH_OUT)
SFX = (getattr(cr, "ENH_SUFFIX", "") or "").lower()

files = {
    "positions": PATH_OUT / f"positions_ledger{SFX}.parquet",
    "marks":     PATH_OUT / f"marks_ledger{SFX}.parquet",
    "pnl":       PATH_OUT / f"pnl_by_bucket{SFX}.parquet",
}

dfs = {}
for k, p in files.items():
    if p.exists():
        dfs[k] = pd.read_parquet(p)
        print(f"[OK] {k}: {len(dfs[k]):,} rows ({p.name})")
    else:
        print(f"[MISS] {k}: {p.name}")

def _assign_bucket(tenor):
    for name, (lo, hi) in cr.BUCKETS.items():
        if (tenor >= lo) and (tenor <= hi): return name
    return "other"

def _safe_div(a, b):
    try:
        return float(a) / float(b) if (b not in (0, None) and pd.notna(b) and float(b)!=0) else np.nan
    except Exception:
        return np.nan

# -------------------- Position-level stats --------------------
pos = dfs.get("positions", pd.DataFrame()).copy()
marks = dfs.get("marks", pd.DataFrame()).copy()
by   = dfs.get("pnl", pd.DataFrame()).copy()

if not pos.empty:
    # Buckets for each leg
    pos["bucket_i"] = pos["tenor_i"].astype(float).map(_assign_bucket)
    pos["bucket_j"] = pos["tenor_j"].astype(float).map(_assign_bucket)
    pos["win"] = pos["pnl"] > 0
    pos["days_held_equiv"] = pd.to_numeric(pos["days_held_equiv"], errors="coerce")

    # Exit reason distribution
    exit_ct = pos["exit_reason"].value_counts().rename_axis("exit_reason").to_frame("count")

    # Core stats
    n_trades = len(pos)
    win_rate = _safe_div(pos["win"].sum(), n_trades)
    avg_pnl  = pos["pnl"].mean()
    med_pnl  = pos["pnl"].median()
    p90_pnl  = pos["pnl"].quantile(0.90)
    hold_mean = pos["days_held_equiv"].mean()
    hold_p90  = pos["days_held_equiv"].quantile(0.90)

    # By bucket contribution (using leg i as proxy; you can do both legs if preferred)
    by_bucket = pos.groupby("bucket_i")["pnl"].sum().rename("cum_pnl").sort_values(ascending=False).to_frame()

# -------------------- Time-series PnL & risk --------------------
daily = pd.DataFrame()
if not by.empty:
    # Normalize to calendar day for a clean equity curve (works for D and H)
    daily = by.copy()
    daily["date"] = pd.to_datetime(daily["bucket"]).dt.floor("D")
    daily = (daily.groupby("date")["pnl"].sum()
                   .rename("daily_pnl").to_frame().reset_index())
    daily = daily.sort_values("date")
    daily["cum_pnl"] = daily["daily_pnl"].cumsum()

    # Max drawdown on the daily curve
    eq = daily["cum_pnl"].astype(float)
    roll_max = eq.cummax()
    dd = roll_max - eq
    max_dd = float(dd.max())
    dd_end_idx = int(dd.idxmax()) if len(dd) else 0
    dd_end_date = daily.iloc[dd_end_idx]["date"] if len(daily) else None

    # Simple Sharpe-like on (bucket) PnL
    pnl_mean = by["pnl"].astype(float).mean()
    pnl_std  = by["pnl"].astype(float).std(ddof=1)
    sharpe_like = _safe_div(pnl_mean, pnl_std)

# -------------------- Open vs Closed sanity (position-level) --------------------
open_now_ct = np.nan
if not marks.empty:
    key = ["open_ts", "tenor_i", "tenor_j"]
    last_mark = (marks.loc[marks["event"]=="mark"]
                      .sort_values(key + ["decision_ts"])
                      .groupby(key, as_index=False)
                      .tail(1))
    open_now_ct = int((last_mark["closed"]==False).sum())

# -------------------- Display --------------------
display(HTML("<h3>Summary</h3>"))
summary_rows = []

if not pos.empty:
    summary_rows += [
        ("Trades (closed)", n_trades),
        ("Win rate", f"{100.0*win_rate:.1f}%"),
        ("Avg PnL / trade", f"{avg_pnl:,.2f}"),
        ("Median PnL / trade", f"{med_pnl:,.2f}"),
        ("90th pct PnL / trade", f"{p90_pnl:,.2f}"),
        ("Mean hold (days eq.)", f"{hold_mean:.2f}"),
        ("P90 hold (days eq.)", f"{hold_p90:.2f}"),
    ]
if not by.empty:
    summary_rows += [
        ("Cum PnL", f"{float(daily['cum_pnl'].iloc[-1]):,.2f}" if len(daily) else "—"),
        ("Max drawdown", f"{max_dd:,.2f}"),
        ("Sharpe-like (mean/std, bucket PnL)", f"{sharpe_like:.3f}"),
    ]
if not marks.empty:
    summary_rows += [("Open positions at end", open_now_ct)]

display(pd.DataFrame(summary_rows, columns=["metric", "value"]))

if not pos.empty:
    display(HTML("<h4>Exit reason counts</h4>"))
    display(exit_ct)
    display(HTML("<h4>PnL by bucket (leg i proxy)</h4>"))
    display(by_bucket)

# -------------------- Plots --------------------
if not daily.empty:
    plt.figure()
    plt.plot(daily["date"], daily["cum_pnl"])
    plt.title("Cumulative PnL")
    plt.xlabel("Date"); plt.ylabel("Cum PnL")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure()
    plt.plot(daily["date"], daily["daily_pnl"])
    plt.title("Daily PnL")
    plt.xlabel("Date"); plt.ylabel("PnL")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Drawdown curve (optional)
    plt.figure()
    plt.plot(daily["date"], (daily["cum_pnl"].cummax() - daily["cum_pnl"]))
    plt.title("Drawdown")
    plt.xlabel("Date"); plt.ylabel("DD")
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("No time-series PnL loaded; skipping plots.")
# =======================================================================

# Jupyter one-off: add "_D" suffix to enhanced files and sweeper outputs

from pathlib import Path
import re

import cr_config as cr  # uses your configured PATH_ENH / PATH_OUT

# -------- Settings --------
SUFFIX = "_D"
DRY_RUN = True   # set to False to actually rename

# Sweeper artifacts to rename (prefixes without suffix)
# Add/remove prefixes if you use different names
SWEEPER_PREFIXES = ["sweep_results"]   # will catch parquet/csv + *_best.json

# --------------------------

enh_dir = Path(getattr(cr, "PATH_ENH", "."))
out_dir = Path(getattr(cr, "PATH_OUT", "."))

def add_suffix_before_ext(p: Path, suffix: str) -> Path:
    """Return a new path with suffix inserted before the final extension."""
    return p.with_name(p.stem + suffix + p.suffix)

def has_any_suffix(name: str, suffixes=("_D", "_H")) -> bool:
    """Detect if a filename stem already ends with a known suffix."""
    return any(name.endswith(s) for s in suffixes)

def resolve_collision(target: Path) -> Path:
    """If target exists, append a numeric counter _D2, _D3, ..."""
    if not target.exists():
        return target
    base, ext = target.stem, target.suffix
    m = re.match(r"^(.*?)(_D|_H)(\d+)?$", base)
    if m:
        root, sfx, num = m.groups()
        n = int(num) + 1 if num else 2
        return target.with_name(f"{root}{sfx}{n}{ext}")
    # generic fallback
    n = 2
    while True:
        cand = target.with_name(f"{base}{n}{ext}")
        if not cand.exists():
            return cand
        n += 1

plan = []

# 1) Enhanced parquet files: {yymm}_enh.parquet  ->  {yymm}_enh_D.parquet
# Only pick files that look like *_enh.parquet AND do not already carry _D/_H
for p in sorted(enh_dir.glob("*_enh.parquet")):
    stem = p.stem  # e.g., "2304_enh"
    if has_any_suffix(stem):  # already suffixed like *_enh_D or *_enh_H
        continue
    target = add_suffix_before_ext(p, SUFFIX)
    target = resolve_collision(target)
    plan.append((p, target))

# 2) Sweeper outputs:
#    sweep_results.parquet / sweep_results.csv / sweep_results_best.json  ->  *_D.*
for prefix in SWEEPER_PREFIXES:
    # core tables
    for ext in (".parquet", ".csv"):
        p = out_dir / f"{prefix}{ext}"
        if p.exists():
            stem = p.stem
            if not has_any_suffix(stem):
                target = add_suffix_before_ext(p, SUFFIX)
                target = resolve_collision(target)
                plan.append((p, target))
    # best-variant JSON
    pbest = out_dir / f"{prefix}_best.json"
    if pbest.exists():
        stem = pbest.stem  # e.g., "sweep_results_best"
        if not has_any_suffix(stem):
            # insert suffix before extension (after entire stem)
            target = add_suffix_before_ext(pbest, SUFFIX)
            target = resolve_collision(target)
            plan.append((pbest, target))

# ---- Show the plan ----
if not plan:
    print("[INFO] Nothing to rename (files already suffixed or not found).")
else:
    print(f"[INFO] Planned renames ({'DRY RUN' if DRY_RUN else 'COMMIT'}):")
    for src, dst in plan:
        print(f"  {src}  ->  {dst}")

    if not DRY_RUN:
        moved = 0
        for src, dst in plan:
            try:
                src.rename(dst)
                moved += 1
            except Exception as e:
                print(f"[ERROR] Failed to rename {src} -> {dst}: {e}")
        print(f"[DONE] Renamed {moved}/{len(plan)} files.")
    else:
        print("Set DRY_RUN=False and re-run to perform the renames.")


