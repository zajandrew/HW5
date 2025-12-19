"""
strategy_main.py

The Execution Engine for the All-Weather RV Strategy.
Orchestrates Data -> Regime -> Alpha Search -> Position Management.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from collections import deque

# --- MODULE IMPORTS ---
import config as cr
import math_core as mc
import instruments as inst
from regime_manager import RegimeManager

# ==============================================================================
# 1. HELPER: Z-SCORE PROJECTION (DELTA METHOD)
# ==============================================================================

def get_live_z(
    tenor: float, 
    anchor_curve: mc.SplineCurve, 
    live_curve: mc.SplineCurve,
    anchor_scale: float
) -> float:
    """
    Projects the Live Z-Score using the Anchor Model + Live Delta.
    Z_live = Z_anchor + (Rate_live - Rate_anchor) / Scale
    """
    # 1. Get Base Z (The Structural View)
    z_anchor = anchor_curve.z_scores.get(tenor, 0.0)
    
    # 2. Get Rates
    r_anchor = anchor_curve.get_rate(tenor)
    r_live = live_curve.get_rate(tenor)
    
    # 3. Adjust
    # Scale is in rate units (e.g., 5bps = 0.05). Avoid div/0.
    valid_scale = max(1e-4, anchor_scale)
    delta_z = (r_live - r_anchor) / valid_scale
    
    return z_anchor + delta_z

# ==============================================================================
# 2. HELPER: ALPHA SCANNER
# ==============================================================================

def scan_for_alpha(
    dts: pd.Timestamp,
    hedge_row: pd.Series, 
    anchor_curve: mc.SplineCurve, 
    live_curve: mc.SplineCurve,
    anchor_scale: float,
    regime_mgr: RegimeManager,
    regime_state: Dict,
    z_history: Dict[float, deque]
) -> Optional[Union[inst.PairPos, inst.FlyPos]]:
    """
    Scans the curve for the best relative value trade to replace the hedge.
    Returns a Position object (PairPos or FlyPos) or None.
    """
    # 1. Parse Hedge Details
    h_tenor = float(hedge_row["tenor_yrs"])
    h_dv01 = float(hedge_row["dv01"])
    h_side = str(hedge_row["side"]).upper() 
    h_id = hedge_row.get("trade_id", -1)
    
    # 2. Determine Alpha Direction (The "Pivot")
    # If Desk is RECEIVING (Long Duration), we normally PAY (Short Duration).
    # So our Alpha Trade starts with a PAY leg at h_tenor.
    pivot_dir = -1.0 if h_side in ["CRCV", "REC", "RECEIVER"] else 1.0
    
    # 3. Get Rules of Engagement
    allow_curve = regime_mgr.check_trade_allowed("curve", regime_state)
    allow_fly = regime_mgr.check_trade_allowed("fly", regime_state)
    
    # Live Context for Pivot
    pivot_z = get_live_z(h_tenor, anchor_curve, live_curve, anchor_scale)
    pivot_drift = mc.calc_signal_drift(h_tenor, pivot_dir, live_curve)
    
    best_pair, best_pair_score = None, -999.0
    best_fly, best_fly_score = None, -999.0
    
    # ==========================================================================
    # STRATEGY A: CURVE (PAIRS)
    # ==========================================================================
    if allow_curve:
        z_entry_thresh = regime_mgr.get_z_adjustment("curve", regime_state, cr.PARAMS["Z_ENTRY"])
        
        # Scan known universe (Anchor tenors) using Live Rates
        for t_cand in anchor_curve.z_scores.keys():
            if abs(t_cand - h_tenor) < cr.PARAMS["MIN_TENOR"]: continue
            
            # Candidate is OPPOSITE to Pivot
            cand_dir = -1.0 * pivot_dir
            cand_z = get_live_z(t_cand, anchor_curve, live_curve, anchor_scale)
            
            # Z-Score Spread (Long - Short)
            # If cand_dir=1 (Rec), we want High Z. Spread = Z_cand - Z_pivot
            z_spread = (cand_z - pivot_z) if cand_dir > 0 else (pivot_z - cand_z)
            
            if z_spread < z_entry_thresh: continue
            
            # MOMENTUM CHECK (Falling Knife)
            # If Spread is CHEAP but falling fast, wait.
            # (Implementation omitted for brevity, relies on z_history logic)
            
            cand_drift = mc.calc_signal_drift(t_cand, cand_dir, live_curve)
            net_drift = cand_drift + pivot_drift
            
            if net_drift < cr.PARAMS["DRIFT_GATE_BPS"]: continue
            
            score = z_spread + (net_drift * cr.PARAMS["DRIFT_WEIGHT"])
            
            if score > best_pair_score:
                best_pair_score = score
                best_pair = {
                    "type": "pair", "cand_tenor": t_cand, 
                    "cand_rate": live_curve.get_rate(t_cand), 
                    "cand_dir": cand_dir, "score": score, 
                    "z_entry_thresh": z_entry_thresh, "net_drift": net_drift
                }

    # ==========================================================================
    # STRATEGY B: FLY (BUTTERFLY)
    # ==========================================================================
    if allow_fly:
        z_entry_thresh = regime_mgr.get_z_adjustment("fly", regime_state, cr.PARAMS["Z_ENTRY"])
        
        # 1. GAMMA CHARGE (Convexity Premium)
        # We are Selling Gamma (Long Fly), so demand extra drift
        gamma_charge = cr.PARAMS.get("CONVEXITY_PREMIUM_BPS", 0.0)
        drift_gate_fly = cr.PARAMS["DRIFT_GATE_BPS"] + gamma_charge
        
        min_w, max_w = cr.PARAMS["FLY_WING_WIDTH"]
        
        for w in np.arange(min_w, max_w + 0.5, 0.5):
            t_left, t_right = h_tenor - w, h_tenor + w
            if t_left < cr.PARAMS["MIN_TENOR"] or t_right > 30.0: continue
            if t_left not in anchor_curve.z_scores or t_right not in anchor_curve.z_scores: continue
            
            # 2. ZOMBIE FILTER (Half-Life)
            # Pivot (Belly) Half-Life check
            hl_pivot = anchor_curve.halflives.get(h_tenor, 999.0)
            if hl_pivot > cr.PARAMS.get("MAX_HALFLIFE_DAYS", 20.0): continue
            
            # Live Zs
            z_left = get_live_z(t_left, anchor_curve, live_curve, anchor_scale)
            z_right = get_live_z(t_right, anchor_curve, live_curve, anchor_scale)
            
            # Fly Z: Pivot - Wings
            # If Pivot=-1 (Pay Belly), we benefit if Belly is RICH (High Z)
            fly_z = pivot_z - 0.5 * (z_left + z_right)
            trade_z = fly_z * (-1.0 * pivot_dir) # Sign flip to get "Long the Alpha"
            
            if trade_z < z_entry_thresh: continue
            
            drift_l = mc.calc_signal_drift(t_left, -pivot_dir, live_curve)
            drift_r = mc.calc_signal_drift(t_right, -pivot_dir, live_curve)
            net_drift = pivot_drift + 0.5*(drift_l + drift_r)
            
            if net_drift < drift_gate_fly: continue
            
            score = trade_z + (net_drift * cr.PARAMS["DRIFT_WEIGHT"])
            
            if score > best_fly_score:
                best_fly_score = score
                best_fly = {
                    "type": "fly", "left_tenor": t_left, "right_tenor": t_right,
                    "left_rate": live_curve.get_rate(t_left), 
                    "right_rate": live_curve.get_rate(t_right),
                    "score": score, "z_entry_thresh": z_entry_thresh, 
                    "net_drift": net_drift, "trade_z": trade_z
                }

    # ==========================================================================
    # SELECTION & CONSTRUCTION
    # ==========================================================================
    chosen = None
    if best_pair and best_fly:
        chosen = best_pair if best_pair_score > best_fly_score else best_fly
    elif best_pair: chosen = best_pair
    elif best_fly: chosen = best_fly
        
    if not chosen: return None

    # Metadata tracking
    meta = {
        "trade_id": h_id, "hedge_side": h_side,
        "drift_score": chosen["net_drift"], "total_score": chosen["score"],
        "z_entry_threshold": chosen["z_entry_thresh"], 
        "regime_state": str(regime_state)
    }

    # Initialize Instruments with LIVE CURVE Rates
    if chosen["type"] == "pair":
        meta["z_entry_val"] = best_pair_score - (chosen["net_drift"] * cr.PARAMS["DRIFT_WEIGHT"])
        return inst.PairPos(
            ts=dts,
            leg_i={"tenor": chosen["cand_tenor"], "rate": chosen["cand_rate"], "direction": chosen["cand_dir"]},
            leg_j={"tenor": h_tenor, "rate": live_curve.get_rate(h_tenor), "direction": pivot_dir},
            curve=live_curve,
            target_dv01=h_dv01, regime_meta=meta
        )
    else:
        meta["z_entry_val"] = chosen["trade_z"]
        return inst.FlyPos(
            ts=dts,
            belly={"tenor": h_tenor, "rate": live_curve.get_rate(h_tenor), "direction": pivot_dir},
            left={"tenor": chosen["left_tenor"], "rate": chosen["left_rate"]},
            right={"tenor": chosen["right_tenor"], "rate": chosen["right_rate"]},
            curve=live_curve,
            target_dv01=h_dv01, weight_method=cr.PARAMS.get("FLY_WEIGHT_METHOD", "convexity"), 
            regime_meta=meta
        )

# ==============================================================================
# 3. OUTPUT SERIALIZATION
# ==============================================================================

def flatten_position(pos: Union[inst.PairPos, inst.FlyPos], state: str) -> Dict:
    """Flattens a position object for Parquet."""
    base = {
        "trade_id": pos.meta.get("trade_id"),
        "open_ts": pos.open_ts,
        "state": state,
        "type": type(pos).__name__,
        "scale_dv01": pos.scale_dv01,
        "pnl_total_bps": pos.pnl_bps,
        "pnl_total_cash": pos.pnl_total,
        "drift_score": pos.meta.get("drift_score"),
        "z_entry_val": pos.meta.get("z_entry_val"),
        "z_threshold": pos.meta.get("z_entry_threshold"),
    }
    
    if state == "closed":
        base["close_ts"] = pos.last_mark_ts
        base["exit_reason"] = pos.exit_reason
    else:
        base["last_mark_ts"] = pos.last_mark_ts
    
    # Iterate Legs
    for i, leg in enumerate(pos.legs):
        pfx = f"leg{i}"
        base[f"{pfx}_tenor"] = leg.tenor
        base[f"{pfx}_dir"]   = leg.direction
        base[f"{pfx}_entry_rate"] = leg.entry_rate
        base[f"{pfx}_curr_rate"]  = leg.curr_rate
        base[f"{pfx}_orig_dv01"] = leg.notional * leg.curr_dv01 
        base[f"{pfx}_curr_dv01"] = leg.curr_dv01
        
    return base

def flatten_mark(pos: Union[inst.PairPos, inst.FlyPos], dts: pd.Timestamp, curve: mc.SplineCurve) -> Dict:
    """Creates a mark row."""
    row = flatten_position(pos, "open")
    row["mark_ts"] = dts
    row["regime_state"] = pos.meta.get("regime_state")
    return row

# ==============================================================================
# 4. MAIN EXECUTION LOOP
# ==============================================================================

def run_strategy(yymms: List[str]):
    
    print(f"[INIT] Loading Regime Manager from {cr.PATH_OUT}")
    regime_path = cr.PATH_OUT / f"regime_multipliers{cr.ENH_SUFFIX}.parquet"
    regime_mgr = RegimeManager(regime_path)
    
    open_positions = []
    closed_pos_data, mark_ledger_data = [], []
    z_history = {} 

    for yymm in yymms:
        print(f"[EXEC] Processing {yymm}...")
        enh_path = cr.PATH_ENH / f"{yymm}_enh{cr.ENH_SUFFIX}.parquet"
        if not enh_path.exists(): continue
        df_enh = pd.read_parquet(enh_path)
        
        trade_path = cr.BASE_DIR / "trades.pkl" # Load Hedge Tape
        df_trades = pd.read_pickle(trade_path) if trade_path.exists() else pd.DataFrame()
        
        # We clean the tape but preserve ALL columns for live curve building
        clean_hedges = mc.clean_hedge_tape(df_trades, decision_freq=cr.DECISION_FREQ)
        
        # HOURLY LOOP
        for dts, snap in df_enh.groupby("decision_ts"):
            
            # --- A. ANCHOR CURVE (Hourly) ---
            valid = snap.dropna(subset=["rate", "tenor_yrs"])
            if valid.empty: continue
            
            anchor_curve = mc.SplineCurve(valid["tenor_yrs"].values, valid["rate"].values)
            anchor_curve.z_scores = dict(zip(valid["tenor_yrs"], valid["z_comb"]))
            
            # Zombie Filter: Load Half-Life
            if "halflife" in valid.columns:
                anchor_curve.halflives = dict(zip(valid["tenor_yrs"], valid["halflife"]))
            else:
                anchor_curve.halflives = {}
                
            # Anchor Scale (Mean of bucket)
            anchor_scale = float(snap["scale"].mean()) if "scale" in snap.columns else 0.05
            
            # Update History
            for t, z in anchor_curve.z_scores.items():
                if t not in z_history: z_history[t] = deque(maxlen=cr.PARAMS["MOMENTUM_WINDOW"])
                z_history[t].append(z)
                
            regime_state = regime_mgr.get_state(dts)
            
            # --- B. MARK POSITIONS (Using Anchor) ---
            still_open = []
            for pos in open_positions:
                pos.mark(anchor_curve, dts)
                
                exit_reason = None
                
                # 1. Stop Loss (Total PnL Bps)
                # Note: pos.pnl_bps calc logic resides in Instrument class
                if pos.pnl_bps <= -cr.PARAMS["Z_STOP"] * 10.0: 
                     exit_reason = "StopLoss_PnL"
                
                # 2. Take Profit (Z-Reversion)
                # Need to check Z of position. 
                # Simplification: If PnL > Target or Z reverts to 0. 
                # (Omitted for brevity, standard RV logic applies)
                
                if exit_reason:
                    pos.closed = True
                    pos.exit_reason = exit_reason
                    closed_pos_data.append(flatten_position(pos, "closed"))
                else:
                    still_open.append(pos)
                
                # Record Mark
                mark_ledger_data.append(flatten_mark(pos, dts, anchor_curve))
                    
            open_positions = still_open
            
            # --- C. SCAN HEDGES (Dual Curve) ---
            if clean_hedges.empty: continue
            current_hedges = clean_hedges[clean_hedges["decision_ts"] == dts]
            
            for _, hedge in current_hedges.iterrows():
                if len(open_positions) >= cr.PARAMS["MAX_CONCURRENT"]: break
                
                # 1. Build Live Curve
                live_curve = mc.build_live_curve(hedge, cr.TENOR_YEARS)
                if live_curve is None: live_curve = anchor_curve # Fallback
                
                # 2. Scan
                new_pos = scan_for_alpha(
                    dts, hedge, 
                    anchor_curve, live_curve, anchor_scale, 
                    regime_mgr, regime_state, z_history
                )
                
                if new_pos:
                    open_positions.append(new_pos)
                    mark_ledger_data.append(flatten_mark(new_pos, dts, anchor_curve))

    # 4. Save
    print(f"[DONE] Saving outputs to {cr.PATH_OUT}")
    if closed_pos_data:
        pd.DataFrame(closed_pos_data).to_parquet(cr.PATH_OUT / f"positions_ledger{cr.OUT_SUFFIX}.parquet")
    if mark_ledger_data:
        pd.DataFrame(mark_ledger_data).to_parquet(cr.PATH_OUT / f"marks_ledger{cr.OUT_SUFFIX}.parquet")

if __name__ == "__main__":
    run_strategy(sys.argv[1:])
