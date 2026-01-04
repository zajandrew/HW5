"""
strategy_generator.py (v9.0 - Full History Stitch)

Role:
1. Loads ENTIRE history into memory (Stitching).
2. Constructs synthetic instruments (Curves, Flys).
3. Scans PnL on the continuous timeline (No cutoffs).
4. Slices output back to monthly files for storage.

Key Fixes:
- Full History Context: Solves the "Long Half-Life" cutoff issue.
- Dynamic Time Barrier: Respects HL=300+ signals.
- Bucket Consistency: HL in Trading Hours, Drift in Wall Clock (Verified).
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
    min_hold_steps=120
):
    """
    Implements Triple Barrier Method.
    Time Barrier is Dynamic: Max(2 * HalfLife, min_hold_steps).
    This allows long-term mean reversion (HL=300) to play out,
    while giving momentum (HL=999) at least 'min_hold_steps' (1 week).
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
            # --- Dynamic Vertical Barrier ---
            # If HL=300 (30 days), we give it 600 hours.
            # If HL=10 (1 day), we give it 20 hours (but floored at min_hold if needed?)
            # Actually, standard logic is just 2 * HL. 
            # But for Momentum (999), we cap/floor it.
            
            hl_val = halflife_idxs[i, j]
            
            # Logic: If HL is "Infinite" (Momentum > 500), cap at 120 (1 week).
            # If HL is Real (Reversion < 500), give it 2 * HL.
            if hl_val > 500:
                steps = 120 # 1 Week for Momentum
            else:
                steps = int(hl_val * 2.0)
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
                curr_price = (rate_in - entry_rates[k, j]) * 100.0 
                curr_carry = drift_cumsums[k, j] - drift_in        
                curr_total = curr_price + curr_carry               
                
                # 2. Lower Barrier (Stop Loss)
                if curr_total <= stop_loss:
                    out_label[i, j] = 0
                    out_price[i, j] = curr_price
                    out_carry[i, j] = curr_carry
                    out_total[i, j] = curr_total
                    final_idx = k
                    stopped = True
                    break
                
                # 3. Upper Barrier (Take Profit)
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
            
            # 4. Vertical Barrier Exit
            if not won and not stopped:
                out_price[i, j] = best_price
                out_carry[i, j] = best_carry
                out_total[i, j] = best_total
                
            out_exit_idx[i, j] = final_idx
            out_exit_rate[i, j] = entry_rates[final_idx, j]

    return out_label, out_price, out_carry, out_total, out_exit_idx, out_exit_rate

# ==============================================================================
# 2. FEATURE PROJECTION
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
    SKIP_KEYS   = ['cumsum', 'hours_to_'] 

    for name, mat in pivots.items():
        if any(x in name for x in SKIP_KEYS): continue
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
    df_inv = df.copy()
    df_inv['trade_id'] += "_INV"
    cols_to_flip = ['aux_price_pnl', 'aux_carry_pnl', 'aux_total_pnl']
    for c in df_inv.columns:
        if c.startswith('NET_'):
            if any(x in c for x in ['z', 'drift', 'rate', 'slope', 'accel', 'divergence', 'ratio', 'modified', 'leakage', 'stress', 'sharpe']):
                cols_to_flip.append(c)
    for c in cols_to_flip:
        if c in df_inv.columns: df_inv[c] *= -1
    df_inv['target_label'] = ((df_inv['aux_price_pnl'] >= 1.0) & (df_inv['aux_total_pnl'] >= 2.0)).astype(np.int8)
    return df_inv
    
def save_by_month_incremental(data, idx_time, prefix):
    """
    Slices the massive numpy arrays by month and saves incrementally.
    Prevents holding a 20M+ row DataFrame in RAM.
    """
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    
    # 1. Identify Months from the Timestamp Array
    # idx_time corresponds 1:1 with the rows in 'data'
    ts_series = pd.to_datetime(idx_time)
    months = ts_series.strftime("%y%m")
    unique_months = np.unique(months)
    
    print(f"   Streaming {len(unique_months)} months to disk...")
    
    for m in unique_months:
        # 2. Boolean Mask (Fast)
        mask = (months == m)
        if not np.any(mask): continue
        
        # 3. Slice Dictionary (Cheap)
        # We only materialize the DataFrame for this specific month
        batch_data = {k: v[mask] for k, v in data.items()}
        
        # 4. Create & Save Small DataFrame
        try:
            df_chunk = pd.DataFrame(batch_data)
            df_inv = make_inverse(df_chunk)
            final = pd.concat([df_chunk, df_inv], ignore_index=True)
            
            out_p = path_enh / f"training_{prefix}_{m}.parquet"
            final.to_parquet(out_p, index=False)
            print(f"      -> {out_p.name} ({len(final)} rows)")
        except Exception as e:
            print(f"      [ERR] Failed to save {m}: {e}")
        
        # 5. Immediate Cleanup
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
    
    # 1. Full History Calculations (Fast in Numpy)
    data = project_standard_features(pivots, W, W_abs, ids)
    
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
        if is_event: data[f"NET_{clean}_min"] = np.minimum(l1, l2)
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
    
    drift_key = 'total_drift_day_cumsum' if 'total_drift_day_cumsum' in pivots else 'total_drift_cumsum'
    if drift_key not in pivots: raise KeyError("Missing Drift Column")
    
    net_rates = rate_vals @ W; net_drift = pivots[drift_key].values @ W
    hl_buckets = data.get('NET_halflife', np.full(n_time*n_trades, 100.0)) * 24.0
    
    labels, prices, carries, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, hl_buckets.reshape(n_time, n_trades), 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5)
    
    data['L1_entry_rate'] = L1_rate_mat.ravel(); data['L2_entry_rate'] = L2_rate_mat.ravel()
    data['L1_exit_rate'] = np.take_along_axis(L1_rate_mat, ex_idx, axis=0).ravel()
    data['L2_exit_rate'] = np.take_along_axis(L2_rate_mat, ex_idx, axis=0).ravel()
    all_ts = pivots['rate'].index.values; data['aux_exit_ts'] = all_ts[ex_idx.ravel()]
    
    data['target_label'] = labels.ravel(); data['aux_price_pnl'] = prices.ravel()
    data['aux_carry_pnl'] = carries.ravel(); data['aux_total_pnl'] = totals.ravel()
    data['aux_entry_rate'] = net_rates.ravel(); data['aux_exit_rate'] = ex_rate.ravel()
    data['aux_exit_idx'] = ex_idx.ravel()
    
    # 2. STREAMING SAVE (Memory Fix)
    save_by_month_incremental(data, idx_time, "curves")
    
    # Clear Memory
    del data, net_rates, net_drift, labels, prices, carries, totals
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
    for t1, t2, t3 in combos:
        ids.append(f"F_{t1:g}_{t2:g}_{t3:g}"); t1_arr.append(t1); t2_arr.append(t2); t3_arr.append(t3)
        
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2, t3) in enumerate(combos):
        i1, i2, i3 = tenors.index(t1), tenors.index(t2), tenors.index(t3)
        W[i1, j] = 0.5; W[i2, j] = -1.0; W[i3, j] = 0.5
    W_abs = np.abs(W)
    
    data = project_standard_features(pivots, W, W_abs, ids)
    idx_map_1 = [tenors.index(c[0]) for c in combos]; idx_map_2 = [tenors.index(c[1]) for c in combos]; idx_map_3 = [tenors.index(c[2]) for c in combos]
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
    drift_key = 'total_drift_day_cumsum' if 'total_drift_day_cumsum' in pivots else 'total_drift_cumsum'
    if drift_key not in pivots: raise KeyError("Missing Drift Column")
    net_rates = rate_vals @ W; net_drift = pivots[drift_key].values @ W
    hl_buckets = data.get('NET_halflife', np.full(n_time*n_trades, 100.0)) * 24.0 
    
    labels, prices, carries, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, hl_buckets.reshape(n_time, n_trades), 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5)
    
    data['L1_entry_rate'] = L1_rate_mat.ravel(); data['L2_entry_rate'] = L2_rate_mat.ravel(); data['L3_entry_rate'] = L3_rate_mat.ravel()
    data['L1_exit_rate'] = np.take_along_axis(L1_rate_mat, ex_idx, axis=0).ravel()
    data['L2_exit_rate'] = np.take_along_axis(L2_rate_mat, ex_idx, axis=0).ravel()
    data['L3_exit_rate'] = np.take_along_axis(L3_rate_mat, ex_idx, axis=0).ravel()
    all_ts = pivots['rate'].index.values; data['aux_exit_ts'] = all_ts[ex_idx.ravel()]
    
    data['target_label'] = labels.ravel(); data['aux_price_pnl'] = prices.ravel()
    data['aux_carry_pnl'] = carries.ravel(); data['aux_total_pnl'] = totals.ravel()
    data['aux_entry_rate'] = net_rates.ravel(); data['aux_exit_rate'] = ex_rate.ravel()
    data['aux_exit_idx'] = ex_idx.ravel()
    
    # 2. STREAMING SAVE
    save_by_month_incremental(data, idx_time, "flys")
    
    del data, net_rates, net_drift, labels, prices, carries, totals
    gc.collect()

# ==============================================================================
# 5. ORCHESTRATOR (Full History Stitch)
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
    
    # The builders now handle saving internally to manage memory
    build_curves(pivots, tenors)
    build_flys(pivots, tenors)
    
    print("Done.")

if __name__ == "__main__":
    process_full_history()