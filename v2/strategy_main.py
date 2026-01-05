"""
strategy_generator.py (v13.0 - Audit Precision & PnL Breakdown)

Role:
1. Ingests full history.
2. Constructs synthetic instruments.
3. Implements strict naming convention (Features vs. _drop).
4. Simulates execution via Triple Barrier with explicit PnL breakdown.
   - Separate columns for Price, Roll, and Pure Carry PnL.
5. Slices output to monthly files.
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
# 1. NUMBA SCANNER
# ==============================================================================
@numba.njit(parallel=True)
def scan_pnl_audit(
    entry_rates, 
    drift_cumsums,    # Total Drift (Carry + Roll)
    roll_cumsums,     # Rolldown Component
    halflife_idxs, 
    price_hurdle, 
    total_hurdle, 
    stop_loss, 
    use_roll_in_quality, # <--- NEW: Logic Flag (True for Flys, False for Curves)
    min_hold_steps=120
):
    """
    Triple Barrier Method with Granular Audit.
    
    Logic:
    1. Total PnL   = Price + Total Drift
    2. Roll PnL    = Roll_Exit - Roll_Entry
    3. Carry PnL   = Total_Drift_PnL - Roll_PnL (Derived Pure Carry)
    4. Quality PnL = Price + (Roll if use_roll_in_quality else 0)
    
    Win Condition: Total >= 2bps AND Quality >= 1bp
    """
    n_time, n_trades = entry_rates.shape
    
    # Audit Containers
    out_label = np.zeros((n_time, n_trades), dtype=np.int8)
    out_price = np.zeros((n_time, n_trades), dtype=np.float32)
    out_roll  = np.zeros((n_time, n_trades), dtype=np.float32) # Explicit Roll
    out_carry = np.zeros((n_time, n_trades), dtype=np.float32) # Explicit Pure Carry
    out_total = np.zeros((n_time, n_trades), dtype=np.float32)
    out_exit_idx  = np.zeros((n_time, n_trades), dtype=np.int32)
    out_exit_rate = np.zeros((n_time, n_trades), dtype=np.float32)
    
    for j in numba.prange(n_trades):
        for i in range(n_time):
            # Dynamic Vertical Barrier
            hl_val = halflife_idxs[i, j]
            if hl_val > 500: steps = 120 
            else:
                steps = int(hl_val * 2.0)
                if steps < 5: steps = 5
            
            end_idx = min(i + steps, n_time)
            if end_idx <= i + 1: continue

            # Entry State
            rate_in  = entry_rates[i, j]
            drift_in = drift_cumsums[i, j]
            roll_in  = roll_cumsums[i, j]
            
            best_price, best_roll, best_carry, best_total = 0.0, 0.0, 0.0, -999.0
            final_idx = end_idx - 1
            won, stopped = False, False
            
            # Path Traversal
            for k in range(i + 1, end_idx):
                # 1. Component Calculation
                curr_price = (rate_in - entry_rates[k, j]) * 100.0
                
                # Accruals
                total_drift_pnl = drift_cumsums[k, j] - drift_in
                curr_roll       = roll_cumsums[k, j] - roll_in
                curr_carry      = total_drift_pnl - curr_roll # Pure Carry
                
                curr_total = curr_price + total_drift_pnl
                
                # 2. Quality Logic (Strategy Specific)
                if use_roll_in_quality:
                    curr_quality = curr_price + curr_roll # Flys
                else:
                    curr_quality = curr_price # Curves
                
                # 3. Stop Loss
                if curr_total <= stop_loss:
                    out_label[i, j] = 0
                    out_price[i, j] = curr_price; out_roll[i, j] = curr_roll
                    out_carry[i, j] = curr_carry; out_total[i, j] = curr_total
                    final_idx = k; stopped = True
                    break
                
                # 4. Take Profit (Win)
                if (curr_quality >= price_hurdle) and (curr_total >= total_hurdle):
                    out_label[i, j] = 1
                    out_price[i, j] = curr_price; out_roll[i, j] = curr_roll
                    out_carry[i, j] = curr_carry; out_total[i, j] = curr_total
                    final_idx = k; won = True
                    break
                
                best_price, best_roll, best_carry, best_total = curr_price, curr_roll, curr_carry, curr_total
            
            if not won and not stopped:
                out_price[i, j] = best_price; out_roll[i, j] = best_roll
                out_carry[i, j] = best_carry; out_total[i, j] = best_total
                
            out_exit_idx[i, j] = final_idx
            out_exit_rate[i, j] = entry_rates[final_idx, j]

    return out_label, out_price, out_roll, out_carry, out_total, out_exit_idx, out_exit_rate

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def get_pivots(df):
    pivots = {}
    potential_cols = [c for c in df.columns if c not in ['ts', 'tenor_yrs']]
    for c in potential_cols:
        if not pd.api.types.is_numeric_dtype(df[c]): continue
        try:
            pivots[c] = df.pivot(index='ts', columns='tenor_yrs', values=c).ffill().astype(np.float32)
        except ValueError: continue
    return pivots

def calc_modified_carry(z_arr, drift_arr):
    safe_drift = np.clip(drift_arr * 2.0, -20, 20)
    sigmoid = 2.0 / (1.0 + np.exp(-safe_drift))
    return z_arr * sigmoid

def project_standard_features(pivots, W, W_abs, trade_names):
    data = {}
    REGIME_KEYS = ['halflife', 'scale', 'dv01', 'exog_', 'pca_error_norm']
    # Explicitly drop 'hours_elapsed', 'accrued', 'cumsum' from FEATURES
    # We will grab cumsum later specifically for PnL, but they shouldn't be features.
    DROP_KEYWORDS = ['cumsum', 'hours_to_', 'hours_elapsed', 'accrued'] 

    for name, mat in pivots.items():
        if any(x in name for x in DROP_KEYWORDS): continue
        is_regime = any(k in name for k in REGIME_KEYS)
        vals = mat.values
        
        clean = name.replace("z_comb", "z").replace("total_drift_day", "drift").replace("rate", "rate")
        clean = clean.replace("exog_", "").replace("signal_sharpe", "signal_sharpe")

        if is_regime:
            norm = np.sum(W_abs, axis=0); norm[norm == 0] = 1.0
            data[f"NET_{clean}"] = (vals @ W_abs) / norm
        else:
            data[f"NET_{clean}"] = vals @ W
    for k in data: data[k] = data[k].ravel()
    return data

def make_inverse(df):
    """
    Creates Inverse trades.
    Strictly flips Features AND Audit PnL columns.
    """
    df_inv = df.copy()
    df_inv['trade_id'] += "_INV"
    df_inv['meta_direction'] = -1.0
    
    cols_to_flip = []
    
    # 1. Flip Directional Features
    directional_keywords = [
        'z', 'drift', 'rate', 'slope', 'accel', 'divergence', 'ratio', 
        'modified', 'leakage', 'stress', 'sharpe', 'roll', 'carry'
    ]
    for c in df_inv.columns:
        if c.startswith('NET_') and any(x in c for x in directional_keywords):
            cols_to_flip.append(c)
            
    # 2. Flip Audit PnL (Future Data)
    # We look for '_drop' columns that imply PnL
    for c in df_inv.columns:
        if c.endswith('_drop') and any(x in c for x in ['pnl', 'drift', 'roll', 'carry']):
            cols_to_flip.append(c)

    for c in cols_to_flip:
        if c in df_inv.columns: df_inv[c] *= -1
    
    # Re-calc Label using FLIPPED Audit Cols
    df_inv['target_label'] = ((df_inv['audit_price_pnl_drop'] >= 1.0) & (df_inv['audit_total_pnl_drop'] >= 2.0)).astype(np.int8)
    
    return df_inv

def save_by_month_incremental(data, idx_time, prefix):
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    ts_series = pd.to_datetime(idx_time)
    months = ts_series.strftime("%y%m")
    unique_months = np.unique(months)
    
    print(f"   Streaming {len(unique_months)} months to disk...")
    for m in unique_months:
        mask = (months == m)
        if not np.any(mask): continue
        batch_data = {k: v[mask] for k, v in data.items()}
        try:
            df_chunk = pd.DataFrame(batch_data)
            df_inv = make_inverse(df_chunk)
            final = pd.concat([df_chunk, df_inv], ignore_index=True)
            out_p = path_enh / f"training_{prefix}_{m}.parquet"
            final.to_parquet(out_p, index=False)
            print(f"      -> {out_p.name} ({len(final)} rows)")
        except Exception as e:
            print(f"      [ERR] Failed to save {m}: {e}")
        del df_chunk, df_inv, final, batch_data
        gc.collect()

# ==============================================================================
# 3. BUILDER: CURVES
# ==============================================================================
def build_curves(pivots, tenors):
    print(f"   Building CURVES on full history ({len(pivots['rate'])} rows)...")
    combos = list(combinations(tenors, 2))
    n_trades = len(combos)
    n_time = len(pivots['rate'].index)
    idx_time = np.repeat(pivots['rate'].index, n_trades)
    
    ids, dist_arr = [], []
    for t1, t2 in combos: ids.append(f"C_{t1:g}_{t2:g}"); dist_arr.append(abs(t2 - t1))
        
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2) in enumerate(combos):
        i1, i2 = tenors.index(t1), tenors.index(t2)
        W[i1, j] = 1.0; W[i2, j] = -1.0
    W_abs = np.abs(W)
    
    data = project_standard_features(pivots, W, W_abs, ids)
    data['meta_direction'] = np.ones(n_time * n_trades, dtype=np.float32)
    
    idx_map_1 = [tenors.index(c[0]) for c in combos]
    idx_map_2 = [tenors.index(c[1]) for c in combos]
    rate_vals = pivots['rate'].values
    L1_rate_mat = rate_vals[:, idx_map_1]; L2_rate_mat = rate_vals[:, idx_map_2]
    
    target_legs = ['rate', 'total_drift_day', 'z_comb_slope_5b', 'z_comb', 'z_pca', 'z_spline', 'signal_sharpe', 'z_pca_vol_adj']
    L1_drift, L2_drift = None, None
    for feat in pivots.keys():
        is_target = any(t in feat for t in target_legs)
        is_event = 'hours_to_' in feat
        if not (is_target or is_event): continue
        vals = pivots[feat].values
        l1 = vals[:, idx_map_1].ravel(); l2 = vals[:, idx_map_2].ravel()
        clean = feat.replace("z_comb", "z").replace("total_drift_day", "drift").replace("exog_", "").replace("hours_to_", "h_")
        
        if is_event: 
            data[f"NET_{clean}_min"] = np.minimum(l1, l2)
        else:
            if feat == 'rate':
                data[f"L1_{clean}_raw_drop"] = l1; data[f"L2_{clean}_raw_drop"] = l2
            else:
                data[f"L1_{clean}"] = l1; data[f"L2_{clean}"] = l2
            if feat == 'total_drift_day': L1_drift, L2_drift = l1, l2
            if feat == 'z_comb': data['NET_leg_stress'] = np.maximum(np.abs(l1), np.abs(l2))

    if 'NET_drift' in data and 'NET_vol_implied' in data:
        safe_vol = np.maximum(data['NET_vol_implied'], 0.1)
        data['NET_drift_vol_ratio'] = data['NET_drift'] / safe_vol
    if 'NET_drift' in data:
        for z_name in ['NET_z', 'NET_z_pca', 'NET_z_spline']:
            if z_name in data: data[f"{z_name}_modified"] = calc_modified_carry(data[z_name], data['NET_drift'])
    if L1_drift is not None:
         if 'L1_z' in data: data['L1_z_modified'] = calc_modified_carry(data['L1_z'], L1_drift)
         if 'L2_z' in data: data['L2_z_modified'] = calc_modified_carry(data['L2_z'], L2_drift)

    data['ts'] = idx_time; data['trade_id'] = np.tile(ids, n_time); data['meta_dist'] = np.tile(dist_arr, n_time)
    
    # Calculate Net PnL Vectors
    drift_key = 'total_drift_day_cumsum' if 'total_drift_day_cumsum' in pivots else 'total_drift_cumsum'
    if drift_key not in pivots: raise KeyError("Missing Drift Column")
    
    roll_key = 'roll_bps_day_cumsum' if 'roll_bps_day_cumsum' in pivots else 'roll_bps_cumsum'
    # Fallback to zero if roll not found, but it should be there for correct audit
    if roll_key in pivots:
        net_roll = pivots[roll_key].values @ W
    else:
        net_roll = np.zeros((n_time, n_trades), dtype=np.float32)

    net_rates = rate_vals @ W
    net_drift = pivots[drift_key].values @ W
    hl_buckets = data.get('NET_halflife', np.full(n_time*n_trades, 100.0)) * 24.0
    
    # --- CURVES: Pass True Roll, but use_roll_in_quality=False ---
    labels, prices, rolls, carries, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, net_roll,
        hl_buckets.reshape(n_time, n_trades), 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5,
        use_roll_in_quality=False # Strict Price Check
    )
    
    # Audit Columns (Strict _drop)
    data['audit_L1_entry_rate_drop'] = L1_rate_mat.ravel()
    data['audit_L2_entry_rate_drop'] = L2_rate_mat.ravel()
    data['audit_L1_exit_rate_drop'] = np.take_along_axis(L1_rate_mat, ex_idx, axis=0).ravel()
    data['audit_L2_exit_rate_drop'] = np.take_along_axis(L2_rate_mat, ex_idx, axis=0).ravel()
    
    all_ts = pivots['rate'].index.values
    data['audit_exit_ts_drop'] = all_ts[ex_idx.ravel()]
    
    data['target_label'] = labels.ravel()
    data['audit_price_pnl_drop'] = prices.ravel()
    data['audit_roll_pnl_drop']  = rolls.ravel()  # NEW
    data['audit_carry_pnl_drop'] = carries.ravel()
    data['audit_total_pnl_drop'] = totals.ravel()
    data['audit_entry_rate_drop'] = net_rates.ravel()
    data['audit_exit_rate_drop'] = ex_rate.ravel()
    data['audit_exit_idx_drop'] = ex_idx.ravel()
    
    save_by_month_incremental(data, idx_time, "curves")
    del data, net_rates, net_drift, net_roll, labels, prices, rolls, carries, totals
    gc.collect()

# ==============================================================================
# 4. BUILDER: FLYS
# ==============================================================================
def build_flys(pivots, tenors):
    print(f"   Building FLYS on full history ({len(pivots['rate'])} rows)...")
    combos = list(combinations(tenors, 3))
    n_trades = len(combos)
    n_time = len(pivots['rate'].index)
    idx_time = np.repeat(pivots['rate'].index, n_trades)
    
    ids, t1_arr, t2_arr, t3_arr = [], [], [], []
    fly_width_arr, belly_ratio_arr = [], []

    for t1, t2, t3 in combos:
        ids.append(f"F_{t1:g}_{t2:g}_{t3:g}")
        t1_arr.append(t1); t2_arr.append(t2); t3_arr.append(t3)
        width = t3 - t1
        ratio = (t2 - t1) / width if width != 0 else 0.5
        fly_width_arr.append(width); belly_ratio_arr.append(ratio)
        
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2, t3) in enumerate(combos):
        i1, i2, i3 = tenors.index(t1), tenors.index(t2), tenors.index(t3)
        W[i1, j] = 0.5; W[i2, j] = -1.0; W[i3, j] = 0.5
    W_abs = np.abs(W)
    
    data = project_standard_features(pivots, W, W_abs, ids)
    data['meta_direction'] = np.ones(n_time * n_trades, dtype=np.float32)

    idx_map_1 = [tenors.index(c[0]) for c in combos]
    idx_map_2 = [tenors.index(c[1]) for c in combos]
    idx_map_3 = [tenors.index(c[2]) for c in combos]
    rate_vals = pivots['rate'].values
    L1_rate_mat = rate_vals[:, idx_map_1]; L2_rate_mat = rate_vals[:, idx_map_2]; L3_rate_mat = rate_vals[:, idx_map_3]
    
    target_legs = ['rate', 'total_drift_day', 'z_comb_slope_5b', 'z_comb', 'z_pca', 'z_spline', 'signal_sharpe', 'z_pca_vol_adj']
    L1_drift, L2_drift, L3_drift = None, None, None
    for feat in pivots.keys():
        is_target = any(t in feat for t in target_legs)
        is_event = 'hours_to_' in feat
        if not (is_target or is_event): continue
        vals = pivots[feat].values
        l1 = vals[:, idx_map_1].ravel(); l2 = vals[:, idx_map_2].ravel(); l3 = vals[:, idx_map_3].ravel()
        clean = feat.replace("z_comb", "z").replace("total_drift_day", "drift").replace("exog_", "").replace("hours_to_", "h_")
        
        if is_event: 
            data[f"NET_{clean}_min"] = np.minimum(np.minimum(l1, l2), l3)
        else:
            if feat == 'rate':
                data[f"L1_{clean}_raw_drop"] = l1; data[f"L2_{clean}_raw_drop"] = l2; data[f"L3_{clean}_raw_drop"] = l3
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
    data['meta_fly_width']  = np.tile(fly_width_arr, n_time)
    data['meta_belly_ratio'] = np.tile(belly_ratio_arr, n_time)
    
    drift_key = 'total_drift_day_cumsum' if 'total_drift_day_cumsum' in pivots else 'total_drift_cumsum'
    if drift_key not in pivots: raise KeyError("Missing Drift Column")
    
    roll_key = 'roll_bps_day_cumsum' if 'roll_bps_day_cumsum' in pivots else 'roll_bps_cumsum'
    if roll_key in pivots:
        net_roll = pivots[roll_key].values @ W
    else:
        net_roll = np.zeros((n_time, n_trades), dtype=np.float32)

    net_rates = rate_vals @ W
    net_drift = pivots[drift_key].values @ W
    hl_buckets = data.get('NET_halflife', np.full(n_time*n_trades, 100.0)) * 24.0 
    
    # --- FLYS: Pass True Roll, use_roll_in_quality=True ---
    labels, prices, rolls, carries, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, net_roll,
        hl_buckets.reshape(n_time, n_trades), 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5,
        use_roll_in_quality=True # Price + Roll Check
    )
    
    data['audit_L1_entry_rate_drop'] = L1_rate_mat.ravel()
    data['audit_L2_entry_rate_drop'] = L2_rate_mat.ravel()
    data['audit_L3_entry_rate_drop'] = L3_rate_mat.ravel()
    data['audit_L1_exit_rate_drop'] = np.take_along_axis(L1_rate_mat, ex_idx, axis=0).ravel()
    data['audit_L2_exit_rate_drop'] = np.take_along_axis(L2_rate_mat, ex_idx, axis=0).ravel()
    data['audit_L3_exit_rate_drop'] = np.take_along_axis(L3_rate_mat, ex_idx, axis=0).ravel()
    all_ts = pivots['rate'].index.values; data['audit_exit_ts_drop'] = all_ts[ex_idx.ravel()]
    
    data['target_label'] = labels.ravel()
    data['audit_price_pnl_drop'] = prices.ravel()
    data['audit_roll_pnl_drop']  = rolls.ravel()
    data['audit_carry_pnl_drop'] = carries.ravel()
    data['audit_total_pnl_drop'] = totals.ravel()
    data['audit_entry_rate_drop'] = net_rates.ravel()
    data['audit_exit_rate_drop'] = ex_rate.ravel()
    data['audit_exit_idx_drop'] = ex_idx.ravel()
    
    save_by_month_incremental(data, idx_time, "flys")
    del data, net_rates, net_drift, net_roll, labels, prices, rolls, carries, totals
    gc.collect()

# ==============================================================================
# 5. ORCHESTRATOR
# ==============================================================================
def process_full_history():
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    files = sorted(list(path_enh.glob(f"*_enh{getattr(cr, 'ENH_SUFFIX', '')}.parquet")))
    
    if not files:
        print("No files found."); return

    print(f"Loading {len(files)} files for Full History Stitch...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except: pass
    
    if not dfs: return
    df_full = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['ts', 'tenor_yrs']).sort_values(['ts', 'tenor_yrs'])
    print(f"Full History Loaded: {len(df_full)} rows. Pivoting...")
    
    pivots = get_pivots(df_full)
    tenors = sorted(pivots['rate'].columns)
    
    build_curves(pivots, tenors)
    build_flys(pivots, tenors)
    
    print("Done.")

if __name__ == "__main__":
    process_full_history()
