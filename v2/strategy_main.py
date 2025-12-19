"""
strategy_main.py

The Execution Engine for the All-Weather RV Strategy.
Orchestrates Data -> Regime -> Alpha Search -> Position Management.
Outputs detailed Position and Mark ledgers.
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
# 1. HELPER: ALPHA SCANNER
# ==============================================================================

def scan_for_alpha(
    dts: pd.Timestamp,
    hedge_row: pd.Series, 
    curve: mc.SplineCurve, 
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
    
    pivot_z = curve.z_scores.get(h_tenor, 0.0)
    pivot_drift = mc.calc_signal_drift(h_tenor, pivot_dir, curve)
    
    # ==========================================================================
    # STRATEGY A: CURVE (PAIRS)
    # ==========================================================================
    best_pair = None
    best_pair_score = -999.0
    
    if allow_curve:
        z_entry_thresh = regime_mgr.get_z_adjustment("curve", regime_state, cr.PARAMS["Z_ENTRY"])
        
        for t_cand, z_cand in curve.z_scores.items():
            if abs(t_cand - h_tenor) < cr.PARAMS["MIN_TENOR"]: continue
            
            cand_dir = -1.0 * pivot_dir
            z_spread = (z_cand - pivot_z) if cand_dir > 0 else (pivot_z - z_cand)
            
            if z_spread < z_entry_thresh: continue
            
            cand_drift = mc.calc_signal_drift(t_cand, cand_dir, curve)
            net_drift = cand_drift + pivot_drift
            
            if net_drift < cr.PARAMS["DRIFT_GATE_BPS"]: continue
            
            score = z_spread + (net_drift * cr.PARAMS["DRIFT_WEIGHT"])
            
            if score > best_pair_score:
                best_pair_score = score
                best_pair = {
                    "type": "pair", "cand_tenor": t_cand, "cand_rate": curve.get_rate(t_cand),
                    "cand_dir": cand_dir, "score": score, "z_entry_thresh": z_entry_thresh,
                    "net_drift": net_drift
                }

    # ==========================================================================
    # STRATEGY B: FLY (BUTTERFLY)
    # ==========================================================================
    best_fly = None
    best_fly_score = -999.0
    
    if allow_fly:
        z_entry_thresh = regime_mgr.get_z_adjustment("fly", regime_state, cr.PARAMS["Z_ENTRY"])
        min_w, max_w = cr.PARAMS["FLY_WING_WIDTH"]
        
        for w in np.arange(min_w, max_w + 0.5, 0.5):
            t_left, t_right = h_tenor - w, h_tenor + w
            if t_left < cr.PARAMS["MIN_TENOR"] or t_right > 30.0: continue
            if t_left not in curve.z_scores or t_right not in curve.z_scores: continue
            
            z_left, z_right = curve.z_scores[t_left], curve.z_scores[t_right]
            
            # Pivot is the Belly.
            # If Pivot=-1 (Pay Belly), we want (Belly - Wings) to be POSITIVE (High Belly).
            fly_z = pivot_z - 0.5 * (z_left + z_right)
            trade_z = fly_z * (-1.0 * pivot_dir) 
            
            if trade_z < z_entry_thresh: continue
            
            drift_l = mc.calc_signal_drift(t_left, -pivot_dir, curve)
            drift_r = mc.calc_signal_drift(t_right, -pivot_dir, curve)
            net_drift = pivot_drift + 0.5*(drift_l + drift_r)
            
            if net_drift < cr.PARAMS["DRIFT_GATE_BPS"]: continue
            
            score = trade_z + (net_drift * cr.PARAMS["DRIFT_WEIGHT"])
            
            if score > best_fly_score:
                best_fly_score = score
                best_fly = {
                    "type": "fly", "left_tenor": t_left, "right_tenor": t_right,
                    "left_rate": curve.get_rate(t_left), "right_rate": curve.get_rate(t_right),
                    "score": score, "z_entry_thresh": z_entry_thresh, "net_drift": net_drift,
                    "trade_z": trade_z
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

    # Common Metadata to track "Why we entered"
    meta = {
        "trade_id": h_id,
        "hedge_side": h_side,
        "drift_score": chosen["net_drift"],
        "total_score": chosen["score"],
        "z_entry_threshold": chosen["z_entry_thresh"],
        "regime_state": str(regime_state) # Snapshot of regime for debugging
    }

    if chosen["type"] == "pair":
        meta["z_entry_val"] = best_pair_score - (chosen["net_drift"] * cr.PARAMS["DRIFT_WEIGHT"]) # Back out raw Z
        return inst.PairPos(
            ts=dts,
            leg_i={"tenor": chosen["cand_tenor"], "rate": chosen["cand_rate"], "direction": chosen["cand_dir"]},
            leg_j={"tenor": h_tenor, "rate": curve.get_rate(h_tenor), "direction": pivot_dir},
            curve=curve,
            target_dv01=h_dv01,
            regime_meta=meta
        )
    else:
        meta["z_entry_val"] = chosen["trade_z"]
        return inst.FlyPos(
            ts=dts,
            belly={"tenor": h_tenor, "rate": curve.get_rate(h_tenor), "direction": pivot_dir},
            left={"tenor": chosen["left_tenor"], "rate": chosen["left_rate"]},
            right={"tenor": chosen["right_tenor"], "rate": chosen["right_rate"]},
            curve=curve,
            target_dv01=h_dv01,
            weight_method=cr.PARAMS.get("FLY_WEIGHT_METHOD", "convexity"),
            regime_meta=meta
        )

# ==============================================================================
# 2. OUTPUT SERIALIZATION (FLATTENING)
# ==============================================================================

def flatten_position(pos: Union[inst.PairPos, inst.FlyPos], state: str) -> Dict:
    """
    Flattens a position object into a dictionary for Parquet saving.
    state: "open" or "closed"
    """
    # 1. Base PnL & Metadata
    base = {
        "trade_id": pos.meta.get("trade_id"),
        "open_ts": pos.open_ts,
        "state": state,
        "type": type(pos).__name__,
        "scale_dv01": pos.scale_dv01,
        
        # PnL (Bps & Cash)
        "pnl_total_bps": pos.pnl_bps,
        "pnl_total_cash": pos.pnl_total,
        
        # Breakdown
        "pnl_price_cash": sum(l.pnl_price for l in pos.legs),
        "pnl_carry_cash": sum(l.pnl_carry for l in pos.legs),
        "pnl_roll_cash":  sum(l.pnl_roll for l in pos.legs),
        
        # Entry Reason
        "drift_score": pos.meta.get("drift_score"),
        "z_entry_val": pos.meta.get("z_entry_val"),
        "z_threshold": pos.meta.get("z_entry_threshold"),
    }
    
    if state == "closed":
        base["close_ts"] = pos.last_mark_ts
        base["exit_reason"] = pos.exit_reason
    else:
        base["last_mark_ts"] = pos.last_mark_ts
    
    # 2. Leg Specifics (Dynamic Columns)
    # We iterate legs and suffix them _0, _1, _2
    for i, leg in enumerate(pos.legs):
        # Identify leg role if possible (Belly/Wing vs Leg1/2)
        # For now, index is safest.
        pfx = f"leg{i}"
        
        base[f"{pfx}_tenor"] = leg.tenor
        base[f"{pfx}_dir"]   = leg.direction
        base[f"{pfx}_entry_rate"] = leg.entry_rate
        base[f"{pfx}_curr_rate"]  = leg.curr_rate
        
        # Risk State
        base[f"{pfx}_orig_dv01"] = leg.notional * leg.curr_dv01 # Approximation of original
        base[f"{pfx}_curr_dv01"] = leg.curr_dv01
        
    return base

def flatten_mark(pos: Union[inst.PairPos, inst.FlyPos], dts: pd.Timestamp, curve: mc.SplineCurve) -> Dict:
    """
    Creates a detailed mark row for the ledger.
    """
    row = flatten_position(pos, "open")
    row["mark_ts"] = dts
    # Add context
    row["regime_state"] = pos.meta.get("regime_state")
    return row

# ==============================================================================
# 3. MAIN EXECUTION LOOP
# ==============================================================================

def run_strategy(yymms: List[str]):
    
    print(f"[INIT] Loading Regime Manager from {cr.PATH_OUT}")
    regime_path = cr.PATH_OUT / f"regime_multipliers{cr.ENH_SUFFIX}.parquet"
    regime_mgr = RegimeManager(regime_path)
    
    open_positions = []
    
    # OUTPUT LISTS
    closed_pos_data = []
    mark_ledger_data = []
    
    z_history = {} 

    for yymm in yymms:
        print(f"[EXEC] Processing {yymm}...")
        enh_path = cr.PATH_ENH / f"{yymm}_enh{cr.ENH_SUFFIX}.parquet"
        if not enh_path.exists(): continue
        df_enh = pd.read_parquet(enh_path)
        
        # Load Trades
        trade_path = cr.BASE_DIR / "trades.pkl" # Adjust path as needed
        if not trade_path.exists():
             df_trades = pd.DataFrame()
        else:
             df_trades = pd.read_pickle(trade_path)
        
        clean_hedges = mc.clean_hedge_tape(df_trades, decision_freq=cr.DECISION_FREQ)
        
        # ----------------------------------------------------------------------
        # HOURLY LOOP
        # ----------------------------------------------------------------------
        for dts, snap in df_enh.groupby("decision_ts"):
            
            # A. Curve Construction
            valid = snap.dropna(subset=["rate", "tenor_yrs"])
            if valid.empty: continue
            
            curve = mc.SplineCurve(valid["tenor_yrs"].values, valid["rate"].values)
            curve.z_scores = dict(zip(valid["tenor_yrs"], valid["z_comb"]))
            
            # B. Momentum History Update
            for t, z in curve.z_scores.items():
                if t not in z_history: z_history[t] = deque(maxlen=cr.PARAMS["MOMENTUM_WINDOW"])
                z_history[t].append(z)
                
            # C. Regime
            regime_state = regime_mgr.get_state(dts)
            
            # D. Mark Open Positions
            still_open = []
            for pos in open_positions:
                pos.mark(curve, dts)
                
                # Exit Logic ---------------------------------------------------
                exit_reason = None
                
                # 1. Stop Loss (Total PnL Bps)
                if pos.pnl_bps <= -cr.PARAMS["Z_STOP"] * 10.0: # Approx 1Z = 10bps roughly? Config dependent
                     exit_reason = "StopLoss_PnL"
                
                # 2. Take Profit (Z-Reversion)
                # Note: We need to re-calc the Z-score of the position structure
                # This requires logic in the Instrument class to get_current_z(curve)
                # For now, we assume simple PnL target or Time stop
                
                # 3. Max Hold
                # if (dts - pos.open_ts).days > 5: exit_reason = "TimeStop"
                
                # --------------------------------------------------------------

                # Record Mark
                mark_ledger_data.append(flatten_mark(pos, dts, curve))
                
                if exit_reason:
                    pos.closed = True
                    pos.exit_reason = exit_reason
                    closed_pos_data.append(flatten_position(pos, "closed"))
                else:
                    still_open.append(pos)
                    
            open_positions = still_open
            
            # E. Scan Hedges
            if clean_hedges.empty: continue
            current_hedges = clean_hedges[clean_hedges["decision_ts"] == dts]
            
            for _, hedge in current_hedges.iterrows():
                if len(open_positions) >= cr.PARAMS["MAX_CONCURRENT"]: break
                
                new_pos = scan_for_alpha(dts, hedge, curve, regime_mgr, regime_state, z_history)
                if new_pos:
                    open_positions.append(new_pos)
                    # Initial Mark (Open State)
                    mark_ledger_data.append(flatten_mark(new_pos, dts, curve))

    # 4. Save Final Outputs
    print(f"[DONE] Saving outputs to {cr.PATH_OUT}")
    
    if closed_pos_data:
        df_closed = pd.DataFrame(closed_pos_data)
        df_closed.to_parquet(cr.PATH_OUT / f"positions_ledger{cr.OUT_SUFFIX}.parquet")
        print(f" -> Saved {len(df_closed)} closed positions.")
        
    if mark_ledger_data:
        df_marks = pd.DataFrame(mark_ledger_data)
        df_marks.to_parquet(cr.PATH_OUT / f"marks_ledger{cr.OUT_SUFFIX}.parquet")
        print(f" -> Saved {len(df_marks)} marks.")

if __name__ == "__main__":
    run_strategy(sys.argv[1:])
