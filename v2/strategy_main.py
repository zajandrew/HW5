"""
strategy_generator.py (v15.0 - Multiclass & Binary Targets)

Role:
1. Ingests full history.
2. Constructs synthetic instruments.
3. Implements strict naming convention (Features vs. _drop).
4. Simulates execution via Triple Barrier with Dual Targets:
   - target_binary: 1 (Alpha) vs 0 (Else)
   - target_multiclass: 2 (Alpha), 1 (Weak/Carry), 0 (Loss)
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
# 1. NUMBA SCANNER (Multiclass Logic)
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
    use_roll_in_quality, 
    min_hold_steps=120
):
    """
    Triple Barrier Method with 50% Quality Ratio Logic.
    
    Classes:
    2 = Alpha Win (Total >= Hurdle AND Quality Ratio >= 50%)
    1 = Weak Win  (Total >= Hurdle but Quality Ratio < 50% OR Drifted Positive)
    0 = Loss      (Stop Loss or Time Exit <= 0)
    """
    n_time, n_trades = entry_rates.shape
    
    # Outputs
    out_multi = np.zeros((n_time, n_trades), dtype=np.int8) # 0, 1, 2
    out_price = np.zeros((n_time, n_trades), dtype=np.float32)
    out_roll  = np.zeros((n_time, n_trades), dtype=np.float32) 
    out_carry = np.zeros((n_time, n_trades), dtype=np.float32) 
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

            # Entry
            rate_in  = entry_rates[i, j]
            drift_in = drift_cumsums[i, j]
            roll_in  = roll_cumsums[i, j]
            
            best_price, best_roll, best_carry, best_total = 0.0, 0.0, 0.0, -999.0
            final_idx = end_idx - 1
            won, stopped = False, False
            
            # Forward Scan
            for k in range(i + 1, end_idx):
                curr_price = (rate_in - entry_rates[k, j]) * 100.0
                total_drift_pnl = drift_cumsums[k, j] - drift_in
                curr_roll       = roll_cumsums[k, j] - roll_in
                curr_carry      = total_drift_pnl - curr_roll
                curr_total      = curr_price + total_drift_pnl
                
                # Quality Check
                if use_roll_in_quality:
                    curr_quality = curr_price + curr_roll 
                else:
                    curr_quality = curr_price 
                
                # 1. Stop Loss (Class 0)
                if curr_total <= stop_loss:
                    out_multi[i, j] = 0
                    out_price[i, j] = curr_price; out_roll[i, j] = curr_roll
                    out_carry[i, j] = curr_carry; out_total[i, j] = curr_total
                    final_idx = k; stopped = True
                    break
                
                # 2. Profit Target Hit (Viability Floor)
                if curr_total >= total_hurdle:
                    # Calculate Ratio (Avoid Div/0)
                    ratio = curr_quality / curr_total if curr_total > 0 else 0.0
                    
                    if ratio >= 0.50:
                        # Alpha Win (Class 2)
                        out_multi[i, j] = 2
                    else:
                        # Weak/Carry Win (Class 1)
                        out_multi[i, j] = 1
                        
                    out_price[i, j] = curr_price; out_roll[i, j] = curr_roll
                    out_carry[i, j] = curr_carry; out_total[i, j] = curr_total
                    final_idx = k; won = True
                    break
                
                best_price, best_roll, best_carry, best_total = curr_price, curr_roll, curr_carry, curr_total
            
            # 3. Time Exit
            if not won and not stopped:
                out_price[i, j] = best_price; out_roll[i, j] = best_roll
                out_carry[i, j] = best_carry; out_total[i, j] = best_total
                
                # If drifted positive at exit, mark as Weak Win (Class 1)
                # This ensures we don't accidentally train on these as 'Alpha'
                if best_total > 0.0:
                    out_multi[i, j] = 1 
                else:
                    out_multi[i, j] = 0 # Loss
                
            out_exit_idx[i, j] = final_idx
            out_exit_rate[i, j] = entry_rates[final_idx, j]

    return out_multi, out_price, out_roll, out_carry, out_total, out_exit_idx, out_exit_rate
   
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
    Creates Inverse trades (Shorts).
    
    Logic Update:
    1. Flips signs of directional features and Audit PnL columns.
    2. Re-calculates 'target_multiclass' and 'target_binary' using the 
       new 50% Quality Ratio logic on the FLIPPED PnL.
       
    Quality Definition:
    - Flys: Quality = Price PnL + Roll PnL
    - Curves: Quality = Price PnL only
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
    for c in df_inv.columns:
        if c.endswith('_drop') and any(x in c for x in ['pnl', 'drift', 'roll', 'carry']):
            cols_to_flip.append(c)

    for c in cols_to_flip:
        if c in df_inv.columns: df_inv[c] *= -1
    
    # --- RE-CALCULATE LABELS (Ratio Logic) ---
    
    # A. Define Components
    pnl   = df_inv['audit_total_pnl_drop']
    price = df_inv['audit_price_pnl_drop']
    roll  = df_inv['audit_roll_pnl_drop']
    
    # B. Define Quality based on Instrument Type
    # We check for 'meta_fly_width' which is only present in Flys.
    if 'meta_fly_width' in df_inv.columns:
        # Fly Logic: Quality includes Roll
        quality = price + roll
    else:
        # Curve Logic: Quality is Price only
        quality = price
    
    # C. Vectorized Classification
    new_labels = np.zeros(len(df_inv), dtype=np.int8)
    
    # Viability Floor (Must clear 2.0 bps)
    is_viable = (pnl >= 2.0)
    
    # Ratio Check (Quality must be >= 50% of Total)
    # Handle Division by Zero safely (if pnl is 0, ratio is 0)
    ratio = np.zeros_like(pnl)
    nonzero = (pnl != 0)
    ratio[nonzero] = quality[nonzero] / pnl[nonzero]
    
    is_quality = (ratio >= 0.50)
    
    # Class 2 (Alpha): Viable AND Quality
    mask_2 = is_viable & is_quality
    new_labels[mask_2] = 2
    
    # Class 1 (Weak): Positive PnL but failed Class 2 requirements
    # (Either < 2.0bps profit OR < 50% Quality)
    mask_1 = (pnl > 0) & (~mask_2)
    new_labels[mask_1] = 1
    
    # Class 0 is default (0 or negative PnL)
    
    df_inv['target_multiclass'] = new_labels
    df_inv['target_binary'] = (new_labels == 2).astype(np.int8)
    
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
            final.rename(columns={'ts':'ts_drop', 'trade_id':'trade_id_drop'}, inplace=True)
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
        
        if is_event: data[f"NET_{clean}_min"] = np.minimum(l1, l2)
        else:
            if feat == 'rate':
                data[f"L1_{clean}_raw_drop"] = l1; data[f"L2_{clean}_raw_drop"] = l2
            else:
                data[f"L1_{clean}"] = l1; data[f"L2_{clean}"] = l2
            if feat == 'total_drift_day': L1_drift, L2_drift = l1, l2
            if feat == 'z_comb': data['NET_leg_stress'] = np.maximum(np.abs(l1), np.abs(l2))

    if 'NET_drift' in data:
        for z_name in ['NET_z', 'NET_z_pca', 'NET_z_spline']:
            if z_name in data: data[f"{z_name}_modified"] = calc_modified_carry(data[z_name], data['NET_drift'])
    if L1_drift is not None:
         if 'L1_z' in data: data['L1_z_modified'] = calc_modified_carry(data['L1_z'], L1_drift)
         if 'L2_z' in data: data['L2_z_modified'] = calc_modified_carry(data['L2_z'], L2_drift)

    data['ts'] = idx_time; data['trade_id'] = np.tile(ids, n_time); data['meta_dist'] = np.tile(dist_arr, n_time)
    
    drift_key = 'total_drift_day_cumsum' if 'total_drift_day_cumsum' in pivots else 'total_drift_cumsum'
    roll_key = 'roll_bps_day_cumsum' if 'roll_bps_day_cumsum' in pivots else 'roll_bps_cumsum'
    
    if roll_key in pivots: net_roll = pivots[roll_key].values @ W
    else: net_roll = np.zeros((n_time, n_trades), dtype=np.float32)

    net_rates = rate_vals @ W
    net_drift = pivots[drift_key].values @ W
    hl_buckets = data.get('NET_halflife', np.full(n_time*n_trades, 100.0)) * 24.0
    
    # SCANNER (Curve Logic: use_roll_in_quality=False)
    multiclass, prices, rolls, carries, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, net_roll,
        hl_buckets.reshape(n_time, n_trades), 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5,
        use_roll_in_quality=False
    )
    
    # Audit Columns
    data['audit_L1_entry_rate_drop'] = L1_rate_mat.ravel()
    data['audit_L2_entry_rate_drop'] = L2_rate_mat.ravel()
    data['audit_L1_exit_rate_drop'] = np.take_along_axis(L1_rate_mat, ex_idx, axis=0).ravel()
    data['audit_L2_exit_rate_drop'] = np.take_along_axis(L2_rate_mat, ex_idx, axis=0).ravel()
    data['audit_exit_ts_drop'] = pivots['rate'].index.values[ex_idx.ravel()]
    
    data['audit_price_pnl_drop'] = prices.ravel()
    data['audit_roll_pnl_drop']  = rolls.ravel()
    data['audit_carry_pnl_drop'] = carries.ravel()
    data['audit_total_pnl_drop'] = totals.ravel()
    data['audit_entry_rate_drop'] = net_rates.ravel()
    data['audit_exit_rate_drop'] = ex_rate.ravel()
    data['audit_exit_idx_drop'] = ex_idx.ravel()
    
    # TARGETS
    data['target_multiclass'] = multiclass.ravel()
    data['target_binary'] = (multiclass.ravel() == 2).astype(np.int8)
    
    save_by_month_incremental(data, idx_time, "curves")
    del data, net_rates, net_drift, net_roll, multiclass, prices, rolls, carries, totals
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
    data['meta_fly_width']  = np.tile(fly_width_arr, n_time)
    data['meta_belly_ratio'] = np.tile(belly_ratio_arr, n_time)

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
        
        if is_event: data[f"NET_{clean}_min"] = np.minimum(np.minimum(l1, l2), l3)
        else:
            if feat == 'rate':
                data[f"L1_{clean}_raw_drop"] = l1; data[f"L2_{clean}_raw_drop"] = l2; data[f"L3_{clean}_raw_drop"] = l3
            else:
                data[f"L1_{clean}"] = l1; data[f"L2_{clean}"] = l2; data[f"L3_{clean}"] = l3
            if feat == 'total_drift_day': L1_drift, L2_drift, L3_drift = l1, l2, l3

    t1 = np.tile(t1_arr, n_time); t2 = np.tile(t2_arr, n_time); t3 = np.tile(t3_arr, n_time)
    dist = t3 - t1; dist[dist==0]=1.0
    data['NET_slope_leakage'] = 0.5 - ((t3 - t2) / dist)
    
    if 'NET_drift' in data:
        for z_name in ['NET_z', 'NET_z_pca', 'NET_z_spline']:
            if z_name in data: data[f"{z_name}_modified"] = calc_modified_carry(data[z_name], data['NET_drift'])
    if L1_drift is not None:
         if 'L1_z' in data: data['L1_z_modified'] = calc_modified_carry(data['L1_z'], L1_drift)
         if 'L2_z' in data: data['L2_z_modified'] = calc_modified_carry(data['L2_z'], L2_drift)
         if 'L3_z' in data: data['L3_z_modified'] = calc_modified_carry(data['L3_z'], L3_drift)

    data['ts'] = idx_time; data['trade_id'] = np.tile(ids, n_time); data['meta_t2'] = t2
    
    drift_key = 'total_drift_day_cumsum' if 'total_drift_day_cumsum' in pivots else 'total_drift_cumsum'
    roll_key = 'roll_bps_day_cumsum' if 'roll_bps_day_cumsum' in pivots else 'roll_bps_cumsum'
    
    if roll_key in pivots: net_roll = pivots[roll_key].values @ W
    else: net_roll = np.zeros((n_time, n_trades), dtype=np.float32)

    net_rates = rate_vals @ W
    net_drift = pivots[drift_key].values @ W
    hl_buckets = data.get('NET_halflife', np.full(n_time*n_trades, 100.0)) * 24.0 
    
    # SCANNER (Fly Logic: use_roll_in_quality=True)
    multiclass, prices, rolls, carries, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, net_roll,
        hl_buckets.reshape(n_time, n_trades), 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5,
        use_roll_in_quality=True
    )
    
    data['audit_L1_entry_rate_drop'] = L1_rate_mat.ravel()
    data['audit_L2_entry_rate_drop'] = L2_rate_mat.ravel()
    data['audit_L3_entry_rate_drop'] = L3_rate_mat.ravel()
    data['audit_L1_exit_rate_drop'] = np.take_along_axis(L1_rate_mat, ex_idx, axis=0).ravel()
    data['audit_L2_exit_rate_drop'] = np.take_along_axis(L2_rate_mat, ex_idx, axis=0).ravel()
    data['audit_L3_exit_rate_drop'] = np.take_along_axis(L3_rate_mat, ex_idx, axis=0).ravel()
    data['audit_exit_ts_drop'] = pivots['rate'].index.values[ex_idx.ravel()]
    
    data['audit_price_pnl_drop'] = prices.ravel()
    data['audit_roll_pnl_drop']  = rolls.ravel()
    data['audit_carry_pnl_drop'] = carries.ravel()
    data['audit_total_pnl_drop'] = totals.ravel()
    data['audit_entry_rate_drop'] = net_rates.ravel()
    data['audit_exit_rate_drop'] = ex_rate.ravel()
    data['audit_exit_idx_drop'] = ex_idx.ravel()
    
    # TARGETS
    data['target_multiclass'] = multiclass.ravel()
    data['target_binary'] = (multiclass.ravel() == 2).astype(np.int8)
    
    save_by_month_incremental(data, idx_time, "flys")
    del data, net_rates, net_drift, net_roll, multiclass, prices, rolls, carries, totals
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
