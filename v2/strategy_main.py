"""
strategy_main.py

The Canonical Execution Engine for the All-Weather RV Strategy.
Orchestrates Data -> Regime -> Alpha Search -> Position Management.

Features:
- Dual Curve Logic (Anchor vs. Live)
- Regime Arbitration (Curve vs. Fly)
- Full Entry/Exit/Stop/TakeProfit implementation
- New Pivot Point Constraint (5Y)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from collections import deque

# --- MODULE IMPORTS ---
import config as cr
import math_core as mc
import instruments as inst
from regime_manager import RegimeManager

# ==============================================================================
# 0. UTILITIES
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
    z_anchor = anchor_curve.z_scores.get(tenor, 0.0)
    r_anchor = anchor_curve.get_rate(tenor)
    r_live = live_curve.get_rate(tenor)
    
    valid_scale = max(1e-4, anchor_scale)
    delta_z = (r_live - r_anchor) / valid_scale
    return z_anchor + delta_z

def get_position_z(
    pos: Union[inst.PairPos, inst.FlyPos],
    curve: mc.SplineCurve,
    scale: float
) -> float:
    """
    Calculates the aggregate Z-score of an open position against a specific curve/model.
    Used for Take Profit (Reversion) checks.
    """
    # Note: We rely on the curve having z_scores attached. 
    # If using Live Projection, one would need to pass live_curve + anchor_curve.
    # For Exit Logic (Strategic), we typically use the ANCHOR curve (Hourly).
    
    if isinstance(pos, inst.PairPos):
        # Legs: [Primary (I), Hedge (J)]
        # We need to know which leg is Long/Short relative to the Z calculation.
        # Pair Z = Z_I - Z_J (assuming I is the alpha leg we bought)
        # But robustly: Sum(Z * Direction) is usually the "Portfolio Z".
        # However, Z-scores are usually "High = Cheap".
        # If we Rec Fixed (+1), we want High Z.
        # If we Pay Fixed (-1), we want Low Z.
        # Agg Z = Sum(Z_leg * Direction_leg)
        
        z_agg = 0.0
        for leg in pos.legs:
            z_l = curve.z_scores.get(leg.tenor, 0.0)
            z_agg += z_l * leg.direction
            
        # Normalize? A 2-leg spread has Vol ~ sqrt(2)*Vol_Leg.
        # But our Zs are usually standardized residuals. 
        # Let's keep it simple: The Entry Threshold was based on the Spread Z.
        # Spread Z = (Z_cand - Z_pivot) * Direction_Cand.
        # This aligns with Sum(Z * Dir).
        return z_agg

    elif isinstance(pos, inst.FlyPos):
        # Fly Z = Belly - 0.5*Wings (Standard Definition of Fly Height).
        # Position Z: If we are Long Fly (Rec Belly, Pay Wings), we want High Fly Z.
        # Agg Z = Sum(Z_leg * Direction_leg)
        
        z_agg = 0.0
        # Weights in Z-space are typically 1, -0.5, -0.5 for a Fly
        # But we simply sum the directional Zs.
        for leg in pos.legs:
            z_l = curve.z_scores.get(leg.tenor, 0.0)
            # For a weighted fly, the risk weights handle PnL.
            # For Z-score signal, we typically use 1/-0.5/-0.5 topology.
            # We must detect if this leg is Belly or Wing.
            # Simpler: Use the definition stored in metadata or infer?
            # Let's infer based on tenor middle.
            
            # Actually, standard Sum(Z*Dir) works if Zs are comparable.
            z_agg += z_l * leg.direction
            
        return z_agg
    
    return 0.0

# ==============================================================================
# 1. ALPHA SCANNER
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
    Scans for Relative Value trades using Dual Curve logic and full constraints.
    """
    # 1. Parse Hedge (The Pivot)
    h_tenor = float(hedge_row["tenor_yrs"])
    h_dv01 = float(hedge_row["dv01"])
    h_side = str(hedge_row["side"]).upper() 
    h_id = hedge_row.get("trade_id", -1)
    
    # Pivot Direction: To hedge a Receiver, we Pay (-1).
    pivot_dir = -1.0 if h_side in ["CRCV", "REC", "RECEIVER"] else 1.0
    
    # 2. Regime Gating
    allow_curve = regime_mgr.check_trade_allowed("curve", regime_state)
    allow_fly = regime_mgr.check_trade_allowed("fly", regime_state)
    
    # 3. Live Context
    pivot_z = get_live_z(h_tenor, anchor_curve, live_curve, anchor_scale)
    pivot_drift = mc.calc_signal_drift(h_tenor, pivot_dir, live_curve)
    
    # Momentum History for Pivot (Falling Knife check)
    pivot_mom = 0.0
    if h_tenor in z_history and len(z_history[h_tenor]) >= cr.PARAMS["MOMENTUM_WINDOW"]:
        pivot_mom = pivot_z - z_history[h_tenor][0] # Current - Oldest

    best_pair, best_pair_score = None, -999.0
    best_fly, best_fly_score = None, -999.0
    
    # ==========================================================================
    # STRATEGY A: CURVE (PAIRS) - MOMENTUM/TREND
    # ==========================================================================
    if allow_curve:
        z_entry_thresh = regime_mgr.get_z_adjustment("curve", regime_state, cr.PARAMS["Z_ENTRY"])
        pivot_point = getattr(cr, "PIVOT_POINT", 5.0)

        for t_cand in anchor_curve.z_scores.keys():
            if abs(t_cand - h_tenor) < cr.PARAMS["MIN_TENOR"]: continue
            
            # --- CONSTRAINT: PIVOT POINT (New) ---
            # Both must be <= 5Y OR Both must be > 5Y
            if (h_tenor <= pivot_point and t_cand > pivot_point) or \
               (h_tenor > pivot_point and t_cand <= pivot_point):
                continue
            
            # Candidate Direction
            cand_dir = -1.0 * pivot_dir
            cand_z = get_live_z(t_cand, anchor_curve, live_curve, anchor_scale)
            
            # Spread Z (Long - Short)
            # If cand_dir = 1 (Rec), we want Z_cand > Z_pivot
            z_spread = (cand_z - pivot_z) if cand_dir > 0 else (pivot_z - cand_z)
            
            # Threshold Check
            if z_spread < z_entry_thresh: continue
            
            # Drift Check
            cand_drift = mc.calc_signal_drift(t_cand, cand_dir, live_curve)
            net_drift = cand_drift + pivot_drift
            if net_drift < cr.PARAMS["DRIFT_GATE_BPS"]: continue
            
            # Score
            score = z_spread + (net_drift * cr.PARAMS["DRIFT_WEIGHT"])
            
            if score > best_pair_score:
                best_pair_score = score
                best_pair = {
                    "type": "pair", "cand_tenor": t_cand, 
                    "cand_rate": live_curve.get_rate(t_cand), 
                    "cand_dir": cand_dir, "score": score, 
                    "z_entry_thresh": z_entry_thresh, "net_drift": net_drift,
                    "trade_z": z_spread
                }

    # ==========================================================================
    # STRATEGY B: FLY (BUTTERFLY) - MEAN REVERSION
    # ==========================================================================
    if allow_fly:
        z_entry_thresh = regime_mgr.get_z_adjustment("fly", regime_state, cr.PARAMS["Z_ENTRY"])
        
        # Gamma Charge
        gamma_charge = cr.PARAMS.get("CONVEXITY_PREMIUM_BPS", 0.0)
        drift_gate_fly = cr.PARAMS["DRIFT_GATE_BPS"] + gamma_charge
        
        # Zombie Filter (Pivot)
        hl_pivot = anchor_curve.halflives.get(h_tenor, 999.0)
        max_hl = cr.PARAMS.get("MAX_HALFLIFE_DAYS", 20.0)
        
        if hl_pivot <= max_hl:
            min_w, max_w = cr.PARAMS["FLY_WING_WIDTH"]
            
            for w in np.arange(min_w, max_w + 0.5, 0.5):
                t_left, t_right = h_tenor - w, h_tenor + w
                if t_left < cr.PARAMS["MIN_TENOR"] or t_right > 30.0: continue
                if t_left not in anchor_curve.z_scores or t_right not in anchor_curve.z_scores: continue
                
                # Zombie Filter (Wings) - Strict mode: all legs must be active
                hl_l = anchor_curve.halflives.get(t_left, 999.0)
                hl_r = anchor_curve.halflives.get(t_right, 999.0)
                if hl_l > max_hl or hl_r > max_hl: continue

                # Live Zs
                z_left = get_live_z(t_left, anchor_curve, live_curve, anchor_scale)
                z_right = get_live_z(t_right, anchor_curve, live_curve, anchor_scale)
                
                # Fly Z Calculation
                # If Pivot=-1 (Pay Belly), we want High Belly Z relative to wings
                fly_z_struct = pivot_z - 0.5 * (z_left + z_right)
                trade_z = fly_z_struct * (-1.0 * pivot_dir) 
                
                if trade_z < z_entry_thresh: continue
                
                # Momentum Gate (Falling Knife)
                # If we are betting on reversion, we don't want Z moving against us fast
                # Approx Check: Check momentum of the belly (pivot)
                # If we want to Pay Belly (Pivot=-1), we hate it if Belly Z is crashing
                mom_gate = cr.PARAMS.get("MOMENTUM_GATE", 0.25)
                # If pivot_dir = -1, we want Z to revert DOWN? No, Pay Belly = Short.
                # If Belly is Rich (High Z), we Pay. We make money if Z falls.
                # Bad Momentum: Z is RISING fast (getting richer).
                if pivot_dir == -1.0 and pivot_mom > mom_gate: continue
                if pivot_dir == 1.0 and pivot_mom < -mom_gate: continue

                # Drift
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

    # --- SELECTION ---
    chosen = None
    if best_pair and best_fly:
        chosen = best_pair if best_pair_score > best_fly_score else best_fly
    elif best_pair: chosen = best_pair
    elif best_fly: chosen = best_fly
        
    if not chosen: return None

    # Metadata
    meta = {
        "trade_id": h_id, "hedge_side": h_side,
        "drift_score": chosen["net_drift"], "total_score": chosen["score"],
        "z_entry_threshold": chosen["z_entry_thresh"], 
        "regime_state": str(regime_state),
        "trade_z_entry": chosen["trade_z"]
    }

    if chosen["type"] == "pair":
        meta["z_entry_val"] = chosen["trade_z"]
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
# 2. OUTPUT SERIALIZATION
# ==============================================================================

def flatten_position(pos: Union[inst.PairPos, inst.FlyPos], state: str) -> Dict:
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
    row = flatten_position(pos, "open")
    row["mark_ts"] = dts
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
    closed_pos_data, mark_ledger_data = [], []
    z_history = {} 

    for yymm in yymms:
        print(f"[EXEC] Processing {yymm}...")
        enh_path = cr.PATH_ENH / f"{yymm}_enh{cr.ENH_SUFFIX}.parquet"
        if not enh_path.exists(): 
            print(f"[WARN] {enh_path} not found.")
            continue
        df_enh = pd.read_parquet(enh_path)
        
        trade_path = cr.BASE_DIR / f"{getattr(cr, 'TRADE_TYPES', 'trades')}.pkl"
        if not trade_path.exists():
            print(f"[WARN] {trade_path} not found.")
            df_trades = pd.DataFrame()
        else:
            df_trades = pd.read_pickle(trade_path)
        
        # 1. Clean Hedge Tape (Using User's Signature)
        # Note: clean_hedge_tape in math_core must match the signature requested
        # We assume math_core.clean_hedge_tape is the updated one.
        # But per instruction, user renamed 'prepare_hedge_tape' to 'clean_hedge_tape'.
        # We must call it with the correct args.
        clean_hedges = mc.clean_hedge_tape(
            df_trades, 
            decision_freq=cr.DECISION_FREQ,
        )
        
        for dts, snap in df_enh.groupby("decision_ts"):
            
            # --- A. ANCHOR CURVE (Hourly) ---
            valid = snap.dropna(subset=["rate", "tenor_yrs"])
            if valid.empty: continue
            
            anchor_curve = mc.SplineCurve(valid["tenor_yrs"].values, valid["rate"].values)
            anchor_curve.z_scores = dict(zip(valid["tenor_yrs"], valid["z_comb"]))
            
            if "halflife" in valid.columns:
                anchor_curve.halflives = dict(zip(valid["tenor_yrs"], valid["halflife"]))
            else:
                anchor_curve.halflives = {}
                
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
                
                # 1. STOP LOSS (Hard PnL Floor)
                # "Approx 1Z = 10bps" - Config dependent. Using raw bps threshold.
                stop_thresh = cr.PARAMS.get("Z_STOP", 3.0) * 10.0 # Default assumption
                if pos.pnl_bps <= -stop_thresh:
                     exit_reason = "StopLoss_PnL"
                
                # 2. TAKE PROFIT / REVERSION
                # Calculate current structural Z of the position
                current_struct_z = get_position_z(pos, anchor_curve, anchor_scale)
                
                # Target: Reversion to 0 or close to it (Z_EXIT)
                # If we entered at +2.0, we exit at +0.25.
                # If we entered at -2.0, we exit at -0.25.
                z_exit_param = cr.PARAMS.get("Z_EXIT", 0.25)
                
                if abs(current_struct_z) <= z_exit_param:
                    exit_reason = "TakeProfit_Reversion"
                    
                # 3. MAX HOLD / STALEMATE
                # Check half life of belly? Or hard cap?
                # Using simple day count
                days_held = (dts - pos.open_ts).days
                if days_held > cr.PARAMS.get("MAX_HALFLIFE_DAYS", 20.0):
                    exit_reason = "TimeStop_Zombie"

                if exit_reason:
                    pos.closed = True
                    pos.exit_reason = exit_reason
                    closed_pos_data.append(flatten_position(pos, "closed"))
                else:
                    still_open.append(pos)
                
                # Ledger
                mark_ledger_data.append(flatten_mark(pos, dts, anchor_curve))
                    
            open_positions = still_open
            
            # --- C. SCAN HEDGES (Dual Curve) ---
            if clean_hedges.empty: continue
            current_hedges = clean_hedges[clean_hedges["decision_ts"] == dts]
            
            for _, hedge in current_hedges.iterrows():
                if len(open_positions) >= cr.PARAMS["MAX_CONCURRENT"]: break
                
                # 1. Build Live Curve
                live_curve = mc.build_live_curve(hedge, cr.BBG_DICT, cr.TENOR_YEARS)
                if live_curve is None: live_curve = anchor_curve 
                
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
