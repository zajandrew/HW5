"""
strategy_generator.py (v7.0 - Triple Barrier & Full Feature Projection)

1. Dynamic Feature Projection: Automatically captures ALL windowed stats (Slope, Accel, Z).
2. Strict Triple Barrier:
   - Upper Barrier (Win): Total PnL >= 2bp AND Price PnL >= 1bp.
   - Lower Barrier (Stop): Total PnL <= -1.5bp.
   - Vertical Barrier (Time): 120 Hours (1 Week) cap for Momentum.
3. Feature Groups:
   - NET: Directional (Spread Slopes, Net Z).
   - REGIME: Weighted Avgs (Vol, Skew, Halflife).
   - LEGS: Raw Sniper stats for specific leg stress.
"""

import pandas as pd
import numpy as np
import numba
from itertools import combinations
from pathlib import Path
import sys
import gc
import config as cr

# ==============================================================================
# 1. NUMBA SCANNER (Triple Barrier Audit)
# ==============================================================================
@numba.njit(parallel=True)
def scan_pnl_audit(
    entry_rates, 
    drift_cumsums, 
    halflife_idxs, 
    price_hurdle, 
    total_hurdle, 
    stop_loss, 
    max_hold_steps=120
):
    """
    Implements Triple Barrier Method.
    Barriers:
    1. Profit (Win): Total > 2bps AND Price > 1bps (Quality Control).
    2. Stop Loss (Loss): Total < -1.5bps.
    3. Time (Vertical): Max Hold (120h) or 2*HalfLife.
    """
    n_time, n_trades = entry_rates.shape
    
    # Audit Containers
    out_label = np.zeros((n_time, n_trades), dtype=np.int8)
    out_price = np.zeros((n_time, n_trades), dtype=np.float32)
    out_carry = np.zeros((n_time, n_trades), dtype=np.float32)
    out_total = np.zeros((n_time, n_trades), dtype=np.float32)
    out_exit_idx  = np.zeros((n_time, n_trades), dtype=np.int32)
    out_exit_rate = np.zeros((n_time, n_trades), dtype=np.float32)
    
    for j in numba.prange(n_trades):
        for i in range(n_time):
            # --- Vertical Barrier (Time) ---
            # If HL is 999 (Momentum), we use max_hold_steps (120h).
            # If HL is 10 (Reversion), we use 20h.
            hl_step = int(halflife_idxs[i, j] * 2.0)
            steps = min(hl_step, max_hold_steps)
            if steps < 5: steps = 5
            
            end_idx = min(i + steps, n_time)
            if end_idx <= i + 1: continue

            # Entry State
            rate_in = entry_rates[i, j]
            drift_in = drift_cumsums[i, j]
            
            best_price, best_carry, best_total = 0.0, 0.0, -999.0
            final_idx = end_idx - 1
            won, stopped = False, False
            
            # --- Path Traversal ---
            for k in range(i + 1, end_idx):
                # 1. Calc PnL
                curr_price = (rate_in - entry_rates[k, j]) * 100.0 # Price PnL
                curr_carry = drift_cumsums[k, j] - drift_in        # Carry PnL
                curr_total = curr_price + curr_carry               # Total PnL
                
                # 2. Lower Barrier (Stop Loss)
                if curr_total <= stop_loss:
                    out_label[i, j] = 0
                    out_price[i, j] = curr_price
                    out_carry[i, j] = curr_carry
                    out_total[i, j] = curr_total
                    final_idx = k
                    stopped = True
                    break
                
                # 3. Upper Barrier (Take Profit - Triple Barrier Logic)
                # MUST make money on Price (Momentum/Reversion) AND Total.
                # Prevents "Carry Traps" where Price is losing but Carry is huge.
                if (curr_price >= price_hurdle) and (curr_total >= total_hurdle):
                    out_label[i, j] = 1
                    out_price[i, j] = curr_price
                    out_carry[i, j] = curr_carry
                    out_total[i, j] = curr_total
                    final_idx = k
                    won = True
                    break
                
                # Track 'Exit at Vertical Barrier' state
                best_price, best_carry, best_total = curr_price, curr_carry, curr_total
            
            # 4. Vertical Barrier Exit (Time Expired)
            if not won and not stopped:
                out_price[i, j] = best_price
                out_carry[i, j] = best_carry
                out_total[i, j] = best_total
                
            out_exit_idx[i, j] = final_idx
            out_exit_rate[i, j] = entry_rates[final_idx, j]

    return out_label, out_price, out_carry, out_total, out_exit_idx, out_exit_rate

# ==============================================================================
# 2. DYNAMIC FEATURE PROJECTION (The Kitchen Sink)
# ==============================================================================
def get_pivots(df):
    """
    Pivots EVERY column found in the input DataFrame.
    This ensures slopes, accels, and z-locals are all captured automatically.
    """
    pivots = {}
    cols = [c for c in df.columns if c not in ['ts', 'tenor_yrs']]
    for c in cols:
        pivots[c] = df.pivot(index='ts', columns='tenor_yrs', values=c).ffill().astype(np.float32)
    return pivots

def calc_modified_carry(z_arr, drift_arr):
    # Sigmoid function to weight Z-score by Drift direction
    safe_drift = np.clip(drift_arr * 2.0, -20, 20)
    sigmoid = 2.0 / (1.0 + np.exp(-safe_drift))
    return z_arr * sigmoid

def project_standard_features(pivots, W, W_abs, trade_names):
    """
    Dynamically projects ALL input features into NET or REGIME stats.
    """
    data = {}
    
    # 1. Identify Feature Types
    REGIME_KEYS = ['halflife', 'scale', 'dv01', 'exog_', 'pca_error_norm']
    SKIP_KEYS   = ['cumsum', 'hours_to_'] # Handled manually or skipped

    for name, mat in pivots.items():
        if any(x in name for x in SKIP_KEYS): continue
        
        # Decide: Is this a Regime Feature (Avg) or Directional (Net)?
        is_regime = any(k in name for k in REGIME_KEYS)
        
        vals = mat.values
        # Clean naming (e.g., z_comb_slope_50b -> NET_z_slope_50b)
        clean = name.replace("z_comb", "z").replace("total_drift_day", "drift").replace("rate", "rate")
        clean = clean.replace("exog_", "").replace("signal_sharpe", "signal_sharpe")

        if is_regime:
            # Weighted Average (Environment)
            norm = np.sum(W_abs, axis=0)
            norm[norm == 0] = 1.0
            data[f"NET_{clean}"] = (vals @ W_abs) / norm
        else:
            # Net Difference (Directional Signal)
            # This captures: Slope, Accel, Z-Score, etc.
            data[f"NET_{clean}"] = vals @ W
            
    for k in data: data[k] = data[k].ravel()
    return data

def make_inverse(df):
    """Creates Short positions for every Long position."""
    df_inv = df.copy()
    df_inv['trade_id'] += "_INV"
    cols_to_flip = ['aux_price_pnl', 'aux_carry_pnl', 'aux_total_pnl']
    
    for c in df_inv.columns:
        if c.startswith('NET_'):
            # Flip anything Directional. 
            # Note: Slope/Accel IS directional, so it gets flipped.
            if any(x in c for x in ['z', 'drift', 'rate', 'slope', 'accel', 'divergence', 'ratio', 'modified', 'leakage', 'stress', 'sharpe']):
                cols_to_flip.append(c)
                
    for c in cols_to_flip:
        if c in df_inv.columns: df_inv[c] *= -1
    
    # Recalculate Labels for the Inverse Trades
    df_inv['target_label'] = (
        (df_inv['aux_price_pnl'] >= 1.0) & 
        (df_inv['aux_total_pnl'] >= 2.0)
    ).astype(np.int8)
    
    return df_inv

# ==============================================================================
# 3. BUILDER: CURVES
# ==============================================================================
def build_curves(pivots, tenors, month_str):
    print(f"[{month_str}] Building CURVES...")
    combos = list(combinations(tenors, 2))
    n_trades = len(combos)
    n_time = len(pivots['rate'].index)
    idx_time = np.repeat(pivots['rate'].index, n_trades)
    
    ids, dist_arr = [], []
    for t1, t2 in combos:
        ids.append(f"C_{t1:g}_{t2:g}")
        dist_arr.append(abs(t2 - t1))
        
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2) in enumerate(combos):
        i1, i2 = tenors.index(t1), tenors.index(t2)
        W[i1, j] = 1.0; W[i2, j] = -1.0
    W_abs = np.abs(W)
    
    # 1. Dynamic Projection (Includes Slopes, Accels, etc.)
    data = project_standard_features(pivots, W, W_abs, ids)
    
    # 2. Leg Features (Sniper Selection)
    idx_map_1 = [tenors.index(c[0]) for c in combos]
    idx_map_2 = [tenors.index(c[1]) for c in combos]
    
    target_legs = ['rate', 'total_drift_day', 'z_comb_slope_5b', 'z_comb', 'z_pca', 'z_spline', 'signal_sharpe', 'z_pca_vol_adj']
    L1_drift, L2_drift = None, None
    
    for feat in pivots.keys():
        is_target = any(t in feat for t in target_legs)
        is_event = 'hours_to_' in feat
        if not (is_target or is_event): continue
        
        vals = pivots[feat].values
        l1 = vals[:, idx_map_1].ravel()
        l2 = vals[:, idx_map_2].ravel()
        
        clean = feat.replace("z_comb", "z").replace("total_drift_day", "drift").replace("exog_", "").replace("hours_to_", "h_")
        
        if is_event:
            data[f"NET_{clean}_min"] = np.minimum(l1, l2)
        else:
            data[f"L1_{clean}"] = l1; data[f"L2_{clean}"] = l2
            if feat == 'total_drift_day': L1_drift, L2_drift = l1, l2
            if feat == 'z_comb': data['NET_leg_stress'] = np.maximum(np.abs(l1), np.abs(l2))

    # 3. Physics & Derived
    if 'NET_drift' in data and 'NET_vol_implied' in data:
        safe_vol = np.maximum(data['NET_vol_implied'], 0.1)
        data['NET_drift_vol_ratio'] = data['NET_drift'] / safe_vol
        
    if 'NET_drift' in data:
        for z_name in ['NET_z', 'NET_z_pca', 'NET_z_spline']:
            if z_name in data:
                data[f"{z_name}_modified"] = calc_modified_carry(data[z_name], data['NET_drift'])
                
    if L1_drift is not None:
         if 'L1_z' in data: data['L1_z_modified'] = calc_modified_carry(data['L1_z'], L1_drift)
         if 'L2_z' in data: data['L2_z_modified'] = calc_modified_carry(data['L2_z'], L2_drift)

    # 4. Audit & Labels
    data['ts'] = idx_time; data['trade_id'] = np.tile(ids, n_time); data['meta_dist'] = np.tile(dist_arr, n_time)
    
    net_rates = pivots['rate'].values @ W
    net_drift = pivots['total_drift_day_cumsum'].values @ W
    hl_buckets = data.get('NET_halflife', np.full(n_time*n_trades, 100.0)) * 24.0
    
    # TRIPLE BARRIER PARAMS: Price >= 1.0, Total >= 2.0
    labels, prices, carries, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, hl_buckets.reshape(n_time, n_trades), 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5, max_hold_steps=120)
    
    data['target_label'] = labels.ravel()
    data['aux_price_pnl'] = prices.ravel(); data['aux_carry_pnl'] = carries.ravel()
    data['aux_total_pnl'] = totals.ravel()
    data['aux_entry_rate'] = net_rates.ravel(); data['aux_exit_rate'] = ex_rate.ravel()
    
    df = pd.DataFrame(data)
    df_inv = make_inverse(df)
    final = pd.concat([df, df_inv], ignore_index=True).dropna()
    
    out_p = Path(getattr(cr, "PATH_ENH", ".")) / f"training_curves_{month_str}.parquet"
    final.to_parquet(out_p, index=False)
    print(f"   Saved {len(final)} CURVES -> {out_p}")
    del df, df_inv, final, data; gc.collect()

# ==============================================================================
# 4. BUILDER: FLYS
# ==============================================================================
def build_flys(pivots, tenors, month_str):
    print(f"[{month_str}] Building FLYS...")
    combos = list(combinations(tenors, 3))
    n_trades = len(combos)
    n_time = len(pivots['rate'].index)
    idx_time = np.repeat(pivots['rate'].index, n_trades)
    
    ids, t1_arr, t2_arr, t3_arr = [], [], [], []
    for t1, t2, t3 in combos:
        ids.append(f"F_{t1:g}_{t2:g}_{t3:g}")
        t1_arr.append(t1); t2_arr.append(t2); t3_arr.append(t3)
        
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2, t3) in enumerate(combos):
        i1, i2, i3 = tenors.index(t1), tenors.index(t2), tenors.index(t3)
        W[i1, j] = 0.5; W[i2, j] = -1.0; W[i3, j] = 0.5
    W_abs = np.abs(W)
    
    data = project_standard_features(pivots, W, W_abs, ids)
    
    idx_map_1 = [tenors.index(c[0]) for c in combos]
    idx_map_2 = [tenors.index(c[1]) for c in combos]
    idx_map_3 = [tenors.index(c[2]) for c in combos]
    
    target_legs = ['rate', 'total_drift_day', 'z_comb_slope_5b', 'z_comb', 'z_pca', 'z_spline', 'signal_sharpe', 'z_pca_vol_adj']
    L1_drift, L2_drift, L3_drift = None, None, None
    
    for feat in pivots.keys():
        is_target = any(t in feat for t in target_legs)
        is_event = 'hours_to_' in feat
        if not (is_target or is_event): continue
        
        vals = pivots[feat].values
        l1 = vals[:, idx_map_1].ravel(); l2 = vals[:, idx_map_2].ravel(); l3 = vals[:, idx_map_3].ravel()
        clean = feat.replace("z_comb", "z").replace("total_drift_day", "drift").replace("exog_", "").replace("hours_to_", "h_")
        
        if is_event: data[f"NET_{clean}_min"] = np.minimum(np.minimum(l1, l2), l3)
        else:
            data[f"L1_{clean}"] = l1; data[f"L2_{clean}"] = l2; data[f"L3_{clean}"] = l3
            if feat == 'total_drift_day': L1_drift, L2_drift, L3_drift = l1, l2, l3

    t1 = np.tile(t1_arr, n_time); t2 = np.tile(t2_arr, n_time); t3 = np.tile(t3_arr, n_time)
    dist = t3 - t1; dist[dist==0]=1.0
    data['NET_slope_leakage'] = 0.5 - ((t3 - t2) / dist)
    
    if 'NET_drift' in data and 'NET_vol_implied' in data:
        safe_vol = np.maximum(data['NET_vol_implied'], 0.1)
        data['NET_drift_vol_ratio'] = data['NET_drift'] / safe_vol

    if 'NET_drift' in data:
        for z_name in ['NET_z', 'NET_z_pca', 'NET_z_spline']:
            if z_name in data: data[f"{z_name}_modified"] = calc_modified_carry(data[z_name], data['NET_drift'])
            
    if L1_drift is not None:
         if 'L1_z' in data: data['L1_z_modified'] = calc_modified_carry(data['L1_z'], L1_drift)
         if 'L2_z' in data: data['L2_z_modified'] = calc_modified_carry(data['L2_z'], L2_drift)
         if 'L3_z' in data: data['L3_z_modified'] = calc_modified_carry(data['L3_z'], L3_drift)

    data['ts'] = idx_time; data['trade_id'] = np.tile(ids, n_time); data['meta_t2'] = t2
    net_rates = pivots['rate'].values @ W
    net_drift = pivots['total_drift_day_cumsum'].values @ W
    hl_buckets = data.get('NET_halflife', np.full(n_time*n_trades, 100.0)) * 24.0
    
    labels, prices, carries, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, hl_buckets.reshape(n_time, n_trades), 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5, max_hold_steps=120)
    
    data['target_label'] = labels.ravel()
    data['aux_price_pnl'] = prices.ravel(); data['aux_carry_pnl'] = carries.ravel()
    data['aux_total_pnl'] = totals.ravel()
    data['aux_entry_rate'] = net_rates.ravel(); data['aux_exit_rate'] = ex_rate.ravel()
    
    df = pd.DataFrame(data)
    df_inv = make_inverse(df)
    final = pd.concat([df, df_inv], ignore_index=True).dropna()
    out_p = Path(getattr(cr, "PATH_ENH", ".")) / f"training_flys_{month_str}.parquet"
    final.to_parquet(out_p, index=False)
    print(f"   Saved {len(final)} FLYS -> {out_p}")
    del df, df_inv, final, data; gc.collect()

def process_month(month_str):
    p = Path(getattr(cr, "PATH_ENH", ".")) / f"{month_str}_enh{getattr(cr, 'ENH_SUFFIX', '')}.parquet"
    if not p.exists(): print(f"[{month_str}] Not found: {p}"); return
    print(f"[{month_str}] Loading Features..."); df = pd.read_parquet(p)
    pivots = get_pivots(df); tenors = sorted(pivots['rate'].columns)
    build_curves(pivots, tenors, month_str); build_flys(pivots, tenors, month_str)
    print(f"[{month_str}] Done.")

if __name__ == "__main__":
    for m in sys.argv[1:]: process_month(m)
