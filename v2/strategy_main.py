"""
strategy_main.py

The Canonical Execution Engine for the All-Weather RV Strategy.
Orchestrates Data -> Regime -> Alpha Search -> Position Management.

Research Alignment:
- PAIRS (Slope/PC2): Momentum-based Entry/Exit. Constrained by 5Y Pivot.
- FLYS (Curvature/PC3): Mean-Reversion Entry/Exit. Constrained by Half-Life & Gamma.
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
# 0. CONFIGURATION PROXIES
# ==============================================================================
# We define default split parameters here to align with the research.
# Ideally, these should be moved to config.py as 'PARAMS_PAIR' and 'PARAMS_FLY'.

CFG_PAIR = getattr(cr, "PARAMS_PAIR", {
    "Z_ENTRY": 0.50,         # Lower bar for entry if trend is strong
    "Z_EXIT_MOMENTUM": -0.1, # Exit if trend reverses (Momentum < -0.1)
    "Z_STOP": 3.0,           # Hard Stop
    "DRIFT_GATE_BPS": 0.0,   # Trend trades just need positive carry
    "DRIFT_WEIGHT": 0.50,    # High weight on Carry for trend trades
    "MOMENTUM_WINDOW": 10,   # Longer window for Slope trends
    "PIVOT_POINT": 5.0       # 5Y Segmentation Constraint
})

CFG_FLY = getattr(cr, "PARAMS_FLY", {
    "Z_ENTRY": 1.25,          # High bar for entry (Mean Reversion)
    "Z_EXIT_REVERSION": 0.25, # Exit when Z reverts to near zero
    "Z_STOP": 3.0,            # Hard Stop
    "DRIFT_GATE_BPS": -2.0,   # Can tolerate slight negative drift if Z is huge
    "DRIFT_WEIGHT": 0.20,     # Lower weight on carry, high on Z
    "CONVEXITY_PREMIUM_BPS": 2.0, # Gamma Charge
    "MAX_HALFLIFE_DAYS": 20.0,    # Zombie Filter
    "FLY_WING_WIDTH": (1.5, 7.0)
})

# ==============================================================================
# 1. UTILITIES & CALCULATIONS
# ==============================================================================

def get_live_z(
    tenor: float, 
    anchor_curve: mc.SplineCurve, 
    live_curve: mc.SplineCurve,
    anchor_scale: float
) -> float:
    """Projects Live Z-Score using Anchor Model + Live Delta."""
    z_anchor = anchor_curve.z_scores.get(tenor, 0.0)
    r_anchor = anchor_curve.get_rate(tenor)
    r_live = live_curve.get_rate(tenor)
    valid_scale = max(1e-4, anchor_scale)
    return z_anchor + ((r_live - r_anchor) / valid_scale)

def calculate_spread_momentum(
    leg_tenors: List[float],
    leg_dirs: List[float],
    z_history: Dict[float, deque],
    window: int
) -> float:
    """
    Calculates the momentum of the spread package.
    Mom = Sum( (CurrentZ - OldZ) * Direction )
    Positive Momentum means the trade is moving IN THE MONEY.
    """
    mom_agg = 0.0
    valid_legs = 0
    
    for t, d in zip(leg_tenors, leg_dirs):
        if t in z_history and len(z_history[t]) >= window:
            # Momentum = Current Z - Z N-days ago
            z_curr = z_history[t][-1]
            z_old = z_history[t][0] # Oldest in deque
            
            # If we are Long (+1), we want Z to rise (+).
            # If we are Short (-1), we want Z to fall (-).
            # Spread Mom = Delta_Z * Direction
            mom_agg += (z_curr - z_old) * d
            valid_legs += 1
            
    if valid_legs == 0: return 0.0
    return mom_agg

def get_position_z(pos: Union[inst.PairPos, inst.FlyPos], curve: mc.SplineCurve) -> float:
    """Calculates current structural Z-score of an open position."""
    z_agg = 0.0
    for leg in pos.legs:
        z_l = curve.z_scores.get(leg.tenor, 0.0)
        z_agg += z_l * leg.direction
    return z_agg

# ==============================================================================
# 2. ALPHA SCANNER
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
    Scans for trades respecting Research Logic:
    - Pairs: Momentum Entry (Trend), 5Y Pivot.
    - Flys: Mean Reversion Entry (Value), Gamma Charge, Zombie Filter.
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
    
    best_pair, best_pair_score = None, -999.0
    best_fly, best_fly_score = None, -999.0
    
    # ==========================================================================
    # STRATEGY A: CURVE PAIRS (MOMENTUM)
    # Research: "Freight Train". Enter when trend aligns.
    # ==========================================================================
    if allow_curve:
        # Dynamic Entry Threshold
        z_entry_thresh = regime_mgr.get_z_adjustment("curve", regime_state, CFG_PAIR["Z_ENTRY"])
        pivot_point = CFG_PAIR["PIVOT_POINT"]
        mom_window = int(CFG_PAIR["MOMENTUM_WINDOW"])

        for t_cand in anchor_curve.z_scores.keys():
            if abs(t_cand - h_tenor) < getattr(cr, "MIN_TENOR", 0.5): continue
            
            # 1. PIVOT POINT CONSTRAINT (5Y)
            # Cannot cross the 5Y line.
            if (h_tenor <= pivot_point and t_cand > pivot_point) or \
               (h_tenor > pivot_point and t_cand <= pivot_point):
                continue
            
            cand_dir = -1.0 * pivot_dir
            cand_z = get_live_z(t_cand, anchor_curve, live_curve, anchor_scale)
            
            # 2. MOMENTUM CHECK (The "Freight Train")
            # We calculate the momentum of the PROPOSED Pair.
            # Long Cand / Long Pivot (Directions already set)
            pair_mom = calculate_spread_momentum(
                [t_cand, h_tenor], 
                [cand_dir, pivot_dir], 
                z_history, mom_window
            )
            
            # Rule: Momentum must be Positive (Trend Alignment)
            # We allow slight noise (-0.05) but generally we want to ride the wave.
            if pair_mom < -0.05: continue 
            
            # 3. Z-Spread & Drift
            z_spread = (cand_z * cand_dir) + (pivot_z * pivot_dir)
            # Note: For Pairs, since we are trend following, we might buy "Expensive" 
            # if momentum is strong. But we still set a floor Z.
            if z_spread < z_entry_thresh: continue

            cand_drift = mc.calc_signal_drift(t_cand, cand_dir, live_curve)
            net_drift = cand_drift + pivot_drift
            if net_drift < CFG_PAIR["DRIFT_GATE_BPS"]: continue
            
            # Score: Weighted heavily towards Drift & Momentum for Curve trades
            score = z_spread + (net_drift * CFG_PAIR["DRIFT_WEIGHT"]) + (pair_mom * 1.0)
            
            if score > best_pair_score:
                best_pair_score = score
                best_pair = {
                    "type": "pair", "cand_tenor": t_cand, 
                    "cand_rate": live_curve.get_rate(t_cand), 
                    "cand_dir": cand_dir, "score": score, 
                    "z_entry_thresh": z_entry_thresh, "net_drift": net_drift,
                    "trade_z": z_spread, "trade_mom": pair_mom
                }

    # ==========================================================================
    # STRATEGY B: FLY (BUTTERFLY) - MEAN REVERSION
    # Research: "Rubber Band". Enter on dislocation, exit on reversion.
    # ==========================================================================
    if allow_fly:
        z_entry_thresh = regime_mgr.get_z_adjustment("fly", regime_state, CFG_FLY["Z_ENTRY"])
        
        # 1. GAMMA CHARGE & DRIFT
        gamma_charge = CFG_FLY["CONVEXITY_PREMIUM_BPS"]
        drift_gate_fly = CFG_FLY["DRIFT_GATE_BPS"] + gamma_charge
        
        # 2. ZOMBIE FILTER (Pivot)
        hl_pivot = anchor_curve.halflives.get(h_tenor, 999.0)
        max_hl = CFG_FLY["MAX_HALFLIFE_DAYS"]
        
        if hl_pivot <= max_hl:
            min_w, max_w = CFG_FLY["FLY_WING_WIDTH"]
            
            for w in np.arange(min_w, max_w + 0.5, 0.5):
                t_left, t_right = h_tenor - w, h_tenor + w
                if t_left < getattr(cr, "MIN_TENOR", 0.5) or t_right > 30.0: continue
                if t_left not in anchor_curve.z_scores or t_right not in anchor_curve.z_scores: continue
                
                # Zombie Filter (Wings)
                hl_l = anchor_curve.halflives.get(t_left, 999.0)
                hl_r = anchor_curve.halflives.get(t_right, 999.0)
                if hl_l > max_hl or hl_r > max_hl: continue

                # Live Zs
                z_left = get_live_z(t_left, anchor_curve, live_curve, anchor_scale)
                z_right = get_live_z(t_right, anchor_curve, live_curve, anchor_scale)
                
                # 3. MEAN REVERSION SIGNAL
                # Fly Z = Belly - Wings.
                # If Pivot=-1 (Pay Belly), we want High Belly Z.
                # Trade Z = (Z_Pivot * Dir) + (Z_Left * -Dir) + (Z_Right * -Dir)
                # Note: Fly structure usually equal weight Z.
                fly_z_struct = pivot_z - 0.5 * (z_left + z_right)
                trade_z = fly_z_struct * (-1.0 * pivot_dir) 
                
                if trade_z < z_entry_thresh: continue
                
                # Drift
                drift_l = mc.calc_signal_drift(t_left, -pivot_dir, live_curve)
                drift_r = mc.calc_signal_drift(t_right, -pivot_dir, live_curve)
                net_drift = pivot_drift + 0.5*(drift_l + drift_r)
                
                if net_drift < drift_gate_fly: continue
                
                score = trade_z + (net_drift * CFG_FLY["DRIFT_WEIGHT"])
                
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
        # Normalize scores (Drift weighting helps comparison)
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
        "trade_z_entry": chosen["trade_z"],
        "trade_mom_entry": chosen.get("trade_mom", 0.0) # Only for pairs
    }

    if chosen["type"] == "pair":
        return inst.PairPos(
            ts=dts,
            leg_i={"tenor": chosen["cand_tenor"], "rate": chosen["cand_rate"], "direction": chosen["cand_dir"]},
            leg_j={"tenor": h_tenor, "rate": live_curve.get_rate(h_tenor), "direction": pivot_dir},
            curve=live_curve,
            target_dv01=h_dv01, regime_meta=meta
        )
    else:
        return inst.FlyPos(
            ts=dts,
            belly={"tenor": h_tenor, "rate": live_curve.get_rate(h_tenor), "direction": pivot_dir},
            left={"tenor": chosen["left_tenor"], "rate": chosen["left_rate"]},
            right={"tenor": chosen["right_tenor"], "rate": chosen["right_rate"]},
            curve=live_curve,
            target_dv01=h_dv01, weight_method=getattr(cr, "FLY_WEIGHT_METHOD", "convexity"), 
            regime_meta=meta
        )

# ==============================================================================
# 3. OUTPUT SERIALIZATION
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
        "txn_cost": getattr(pos, "txn_cost_cash", 0.0),
        "drift_score": pos.meta.get("drift_score"),
        "z_entry_val": pos.meta.get("trade_z_entry"),
        "mom_entry_val": pos.meta.get("trade_mom_entry"),
        "z_threshold": pos.meta.get("z_entry_threshold"),
    }
    
    if state == "closed":
        base["close_ts"] = pos.last_mark_ts
        base["exit_reason"] = pos.exit_reason
    else:
        base["last_mark_ts"] = pos.last_mark_ts
    
    for i, leg in enumerate(pos.legs):
        pfx = f"leg{i}"
        base[f"{pfx}_tenor"] = leg.tenor
        base[f"{pfx}_dir"]   = leg.direction
        base[f"{pfx}_entry_rate"] = leg.entry_rate
        base[f"{pfx}_curr_rate"]  = leg.curr_rate
        base[f"{pfx}_curr_dv01"] = leg.curr_dv01
        
    return base

def flatten_mark(pos: Union[inst.PairPos, inst.FlyPos], dts: pd.Timestamp) -> Dict:
    row = flatten_position(pos, "open")
    row["mark_ts"] = dts
    row["regime_state"] = pos.meta.get("regime_state")
    return row

# ==============================================================================
# 4. MAIN EXECUTION LOOP
# ==============================================================================

def run_strategy(yymms: List[str]):
    print(f"[INIT] Loading Regime Manager from {cr.PATH_OUT}")
    regime_mgr = RegimeManager(cr.PATH_OUT / f"regime_multipliers{cr.ENH_SUFFIX}.parquet")
    
    open_positions = []
    closed_pos_data, mark_ledger_data = [], []
    z_history = {} 

    for yymm in yymms:
        print(f"[EXEC] Processing {yymm}...")
        enh_path = cr.PATH_ENH / f"{yymm}_enh{cr.ENH_SUFFIX}.parquet"
        if not enh_path.exists(): continue
        df_enh = pd.read_parquet(enh_path)
        
        trade_path = cr.BASE_DIR / f"{getattr(cr, 'TRADE_TYPES', 'trades')}.pkl"
        clean_hedges = mc.clean_hedge_tape(
            pd.read_pickle(trade_path) if trade_path.exists() else pd.DataFrame(),
            decision_freq=cr.DECISION_FREQ,
            bbg_map=cr.BBG_DICT, tenor_map=cr.TENOR_YEARS
        )
        
        for dts, snap in df_enh.groupby("decision_ts"):
            
            # --- A. ANCHOR CURVE ---
            valid = snap.dropna(subset=["rate", "tenor_yrs"])
            if valid.empty: continue
            
            anchor_curve = mc.SplineCurve(valid["tenor_yrs"].values, valid["rate"].values)
            anchor_curve.z_scores = dict(zip(valid["tenor_yrs"], valid["z_comb"]))
            anchor_curve.halflives = dict(zip(valid["tenor_yrs"], valid["halflife"])) if "halflife" in valid.columns else {}
            anchor_scale = float(snap["scale"].mean()) if "scale" in snap.columns else 0.05
            
            # Update Momentum History
            for t, z in anchor_curve.z_scores.items():
                if t not in z_history: z_history[t] = deque(maxlen=CFG_PAIR["MOMENTUM_WINDOW"])
                z_history[t].append(z)
                
            regime_state = regime_mgr.get_state(dts)
            
            # --- B. MARK & EXIT ---
            still_open = []
            for pos in open_positions:
                pos.mark(anchor_curve, dts)
                exit_reason = None
                
                # 1. HARD STOP (All Types)
                if pos.pnl_bps <= -CFG_PAIR["Z_STOP"] * 10.0:
                     exit_reason = "StopLoss_PnL"

                # 2. EXIT LOGIC (Type Specific)
                current_z = get_position_z(pos, anchor_curve)
                
                if isinstance(pos, inst.PairPos):
                    # CURVE: Momentum Exit
                    # Calculate current momentum of the position
                    leg_tenors = [l.tenor for l in pos.legs]
                    leg_dirs = [l.direction for l in pos.legs]
                    pos_mom = calculate_spread_momentum(leg_tenors, leg_dirs, z_history, int(CFG_PAIR["MOMENTUM_WINDOW"]))
                    
                    # Exit if Momentum flips against us (Trend Exhaustion)
                    if pos_mom < CFG_PAIR["Z_EXIT_MOMENTUM"]:
                        exit_reason = "TakeProfit_TrendExhaustion"
                        
                elif isinstance(pos, inst.FlyPos):
                    # FLY: Mean Reversion Exit
                    # Exit if Z reverts to near zero (Fair Value)
                    if abs(current_z) <= CFG_FLY["Z_EXIT_REVERSION"]:
                        exit_reason = "TakeProfit_Reversion"
                        
                    # Fly Timeout (Zombie)
                    if (dts - pos.open_ts).days > CFG_FLY["MAX_HALFLIFE_DAYS"]:
                        exit_reason = "TimeStop_Zombie"

                # Ledger
                mark_ledger_data.append(flatten_mark(pos, dts))
                
                if exit_reason:
                    pos.closed = True
                    pos.exit_reason = exit_reason
                    closed_pos_data.append(flatten_position(pos, "closed"))
                else:
                    still_open.append(pos)
                    
            open_positions = still_open
            
            # --- C. ENTRY SCAN ---
            if clean_hedges.empty: continue
            current_hedges = clean_hedges[clean_hedges["decision_ts"] == dts]
            
            for _, hedge in current_hedges.iterrows():
                if len(open_positions) >= getattr(cr, "MAX_CONCURRENT", 50): break
                
                live_curve = mc.build_live_curve(hedge, cr.BBG_DICT, cr.TENOR_YEARS)
                if live_curve is None: live_curve = anchor_curve 
                
                new_pos = scan_for_alpha(
                    dts, hedge, 
                    anchor_curve, live_curve, anchor_scale, 
                    regime_mgr, regime_state, z_history
                )
                
                if new_pos:
                    open_positions.append(new_pos)
                    mark_ledger_data.append(flatten_mark(new_pos, dts))

    # 4. Save
    print(f"[DONE] Saving outputs to {cr.PATH_OUT}")
    if closed_pos_data:
        pd.DataFrame(closed_pos_data).to_parquet(cr.PATH_OUT / f"positions_ledger{cr.OUT_SUFFIX}.parquet")
    if mark_ledger_data:
        pd.DataFrame(mark_ledger_data).to_parquet(cr.PATH_OUT / f"marks_ledger{cr.OUT_SUFFIX}.parquet")

if __name__ == "__main__":
    run_strategy(sys.argv[1:])
