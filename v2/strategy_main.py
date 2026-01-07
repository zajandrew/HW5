import pandas as pd
import numpy as np
import numba
from itertools import combinations
from pathlib import Path
import sys
import gc
import config as cr

# ==============================================================================
# 1. NUMBA SCANNER (Unchanged)
# ==============================================================================
@numba.njit(parallel=True)
def scan_pnl_audit(
    entry_rates, 
    drift_cumsums,    
    roll_cumsums,     
    halflife_idxs, 
    price_hurdle, 
    total_hurdle, 
    stop_loss, 
    use_roll_in_quality, 
    min_hold_steps=120
):
    """
    Triple Barrier Method with 50% Quality Ratio Logic.
    """
    n_time, n_trades = entry_rates.shape
    
    # Outputs
    out_multi = np.zeros((n_time, n_trades), dtype=np.int8) 
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
                
                # 1. Stop Loss
                if curr_total <= stop_loss:
                    out_multi[i, j] = 0
                    out_price[i, j] = curr_price; out_roll[i, j] = curr_roll
                    out_carry[i, j] = curr_carry; out_total[i, j] = curr_total
                    final_idx = k; stopped = True
                    break
                
                # 2. Profit Target Hit
                if curr_total >= total_hurdle:
                    ratio = curr_quality / curr_total if curr_total > 0 else 0.0
                    if ratio >= 0.50: out_multi[i, j] = 2 # Alpha
                    else:             out_multi[i, j] = 1 # Weak
                        
                    out_price[i, j] = curr_price; out_roll[i, j] = curr_roll
                    out_carry[i, j] = curr_carry; out_total[i, j] = curr_total
                    final_idx = k; won = True
                    break
                
                best_price, best_roll, best_carry, best_total = curr_price, curr_roll, curr_carry, curr_total
            
            # 3. Time Exit
            if not won and not stopped:
                out_price[i, j] = best_price; out_roll[i, j] = best_roll
                out_carry[i, j] = best_carry; out_total[i, j] = best_total
                if best_total > 0.0: out_multi[i, j] = 1 
                else:                out_multi[i, j] = 0
                
            out_exit_idx[i, j] = final_idx
            out_exit_rate[i, j] = entry_rates[final_idx, j]

    return out_multi, out_price, out_roll, out_carry, out_total, out_exit_idx, out_exit_rate

# ==============================================================================
# 2. FEATURE INTELLIGENCE
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

def classify_feature(col_name):
    """
    Smart classification of features to determine netting strategy.
    """
    # 1. GLOBAL / MACRO (Single instance, no netting)
    # These are identical across tenors.
    if any(k in col_name for k in ['pca_factor', 'hours_to_econ', 'pca_evr', 'pca_error']):
        return 'GLOBAL'

    # 2. EVENT (Min aggregation)
    if 'hours_to_' in col_name:
        return 'EVENT'

    # 3. RAW LEVELS (Non-stationary)
    # We want the Spread (NET), but L1/L2 must be dropped/audit only.
    if col_name == 'rate':
        return 'RAW'

    # 4. REGIME (Weighted Average)
    # Properties like Vol, Curvature, Halflife. 
    # Netting (L1-L2) is less useful than "System Vol" (Avg).
    if any(k in col_name for k in ['vol_', 'halflife', 'scale', 'curvature', 'dv01']):
        return 'REGIME'

    # 5. SIGNAL / PHYSICS (Standard Netting)
    # Drift, Z-Scores, Carry, Slope.
    # We want NET (Spread) + L1 + L2 (Components).
    return 'SIGNAL'

def project_smart_features(pivots, W, W_abs, trade_names, leg_indices):
    """
    Intelligent projection based on feature type.
    """
    data = {}
    
    for col_name, mat in pivots.items():
        ftype = classify_feature(col_name)
        vals = mat.values
        
        # --- A. GLOBAL (Macro) ---
        if ftype == 'GLOBAL':
            # Take the first column (tenor) since all are identical
            # Naming: MACRO_pca_factor_0
            data[f"MACRO_{col_name}"] = np.repeat(vals[:, 0], len(trade_names))
            
        # --- B. EVENT (Min) ---
        elif ftype == 'EVENT':
            # Find min across legs for each trade
            # leg_indices is a list of lists: [[i1, i2], [i1, i2]...]
            # We construct a matrix of shape (Time, Trades)
            n_time = vals.shape[0]
            n_trades = len(trade_names)
            res = np.zeros((n_time, n_trades), dtype=np.float32)
            
            for t_idx in range(n_trades):
                # Get columns for this trade
                cols = vals[:, leg_indices[t_idx]] 
                # Row-wise min
                res[:, t_idx] = np.min(cols, axis=1)
            
            data[f"EVENT_{col_name}_min"] = res.ravel()
            
        # --- C. RAW LEVELS (Rate) ---
        elif ftype == 'RAW':
            # NET: The Spread (Valid Price)
            data[f"NET_{col_name}"] = (vals @ W).ravel()
            # L1/L2: Audit/Drop (Pollutes Model)
            for i, leg_name in enumerate([f"L{k+1}" for k in range(len(leg_indices[0]))]):
                # Create a specialized gather for each leg
                # This is a bit slow in python loop, but safe
                leg_vals = np.zeros((vals.shape[0], len(trade_names)), dtype=np.float32)
                for t_idx in range(len(trade_names)):
                    leg_vals[:, t_idx] = vals[:, leg_indices[t_idx][i]]
                data[f"audit_{leg_name}_{col_name}_drop"] = leg_vals.ravel()

        # --- D. REGIME (Basket Average) ---
        elif ftype == 'REGIME':
            # BASKET: Weighted Average (Using W_abs)
            norm = np.sum(W_abs, axis=0); norm[norm == 0] = 1.0
            data[f"BASKET_{col_name}"] = ((vals @ W_abs) / norm).ravel()
            
            # Keep L1/L2 for context (e.g. skew difference)
            for i, leg_name in enumerate([f"L{k+1}" for k in range(len(leg_indices[0]))]):
                leg_vals = np.zeros((vals.shape[0], len(trade_names)), dtype=np.float32)
                for t_idx in range(len(trade_names)):
                    leg_vals[:, t_idx] = vals[:, leg_indices[t_idx][i]]
                data[f"{leg_name}_{col_name}"] = leg_vals.ravel()

        # --- E. SIGNAL (Standard Net + Legs) ---
        else:
            # NET: Spread (L1 - L2)
            data[f"NET_{col_name}"] = (vals @ W).ravel()
            
            # L1/L2: Drivers
            for i, leg_name in enumerate([f"L{k+1}" for k in range(len(leg_indices[0]))]):
                leg_vals = np.zeros((vals.shape[0], len(trade_names)), dtype=np.float32)
                for t_idx in range(len(trade_names)):
                    leg_vals[:, t_idx] = vals[:, leg_indices[t_idx][i]]
                data[f"{leg_name}_{col_name}"] = leg_vals.ravel()

    return data

def calc_modified_carry(z_arr, drift_arr):
    safe_drift = np.clip(drift_arr * 2.0, -20, 20)
    sigmoid = 2.0 / (1.0 + np.exp(-safe_drift))
    return z_arr * sigmoid

def make_inverse(df):
    """
    Creates Inverse trades (Shorts).
    
    CORRECTED LOGIC:
    1. FLIP: NET_ features (Trade properties: Drift, Spread Z) and Audit PnL.
       Why? "High Drift" is good. For Short, positive drift becomes negative carry.
       We want the model to learn 'High Feature = Win' universally.
       
    2. KEEP: L1_, L2_, MACRO_, EVENT_ (Asset/Market properties).
       Why? The 5Y rate is 4.0% regardless of whether we buy or sell it.
       Flipping this creates fake data (e.g., -4.0% rates).
       We rely on 'meta_direction' to tell the model the orientation.
    """
    df_inv = df.copy()
    df_inv['trade_id'] += "_INV"
    df_inv['meta_direction'] = -1.0
    
    cols_to_flip = []
    
    # 1. Flip Trade-Level Directional Features (NET_)
    # We only flip NET columns that are strictly directional (Drift, Z, etc.)
    # We PROTECT Regime columns (Vol, Scale) even if they are NET
    
    FLIP_PREFIX = 'NET_'
    PROTECTED_SUBSTRINGS = ['halflife', 'scale', 'vol_', 'curvature', 'dv01', 'stress', 'dist']
    
    for c in df_inv.columns:
        if not pd.api.types.is_numeric_dtype(df_inv[c]): continue
        
        # Logic: Must be NET_ AND not a Regime feature
        if c.startswith(FLIP_PREFIX):
            if not any(p in c for p in PROTECTED_SUBSTRINGS):
                cols_to_flip.append(c)

    # 2. Flip PnL Targets (Audit columns)
    for c in df_inv.columns:
        if 'pnl_drop' in c: 
            cols_to_flip.append(c)
            
    # Apply the Flip
    for c in cols_to_flip:
        df_inv[c] *= -1
    
    # --- RE-CALCULATE LABELS ---
    # (Same logic as before, just on the new flipped PnL)
    pnl   = df_inv['audit_total_pnl_drop']
    price = df_inv['audit_price_pnl_drop']
    roll  = df_inv['audit_roll_pnl_drop']
    
    if 'meta_fly_width' in df_inv.columns: quality = price + roll
    else: quality = price
    
    new_labels = np.zeros(len(df_inv), dtype=np.int8)
    is_viable = (pnl >= 2.0)
    
    ratio = np.zeros_like(pnl)
    nonzero = (pnl != 0)
    ratio[nonzero] = quality[nonzero] / pnl[nonzero]
    
    mask_2 = is_viable & (ratio >= 0.50)
    new_labels[mask_2] = 2
    mask_1 = (pnl > 0) & (~mask_2)
    new_labels[mask_1] = 1
    
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
    leg_indices = [] # Stores [i1, i2] for every trade
    
    for t1, t2 in combos: 
        ids.append(f"C_{t1:g}_{t2:g}")
        dist_arr.append(abs(t2 - t1))
        leg_indices.append([tenors.index(t1), tenors.index(t2)])
        
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2) in enumerate(combos):
        i1, i2 = tenors.index(t1), tenors.index(t2)
        W[i1, j] = 1.0; W[i2, j] = -1.0
    W_abs = np.abs(W)
    
    # 1. SMART PROJECTION
    data = project_smart_features(pivots, W, W_abs, ids, leg_indices)
    
    data['meta_direction'] = np.ones(n_time * n_trades, dtype=np.float32)
    data['ts'] = idx_time; data['trade_id'] = np.tile(ids, n_time); data['meta_dist'] = np.tile(dist_arr, n_time)

    # 2. Modified Drift Checks
    if 'NET_total_drift_day' in data:
        net_drift = data['NET_total_drift_day']
        for k in list(data.keys()):
            if k.startswith('NET_z') and 'modified' not in k:
                data[f"{k}_modified"] = calc_modified_carry(data[k], net_drift)
                
    # 3. Stress Meta
    if 'L1_z_comb' in data and 'L2_z_comb' in data:
        data['NET_leg_stress'] = np.maximum(np.abs(data['L1_z_comb']), np.abs(data['L2_z_comb']))

    # 4. Scanner
    drift_key = 'total_drift_day_cumsum' 
    roll_key  = 'roll_bps_day_cumsum' 
    if drift_key not in pivots: drift_key = 'total_drift_cumsum'
    if roll_key not in pivots: roll_key = 'roll_bps_cumsum'
    
    net_rates = pivots['rate'].values @ W
    net_drift = pivots[drift_key].values @ W
    if roll_key in pivots: net_roll = pivots[roll_key].values @ W
    else: net_roll = np.zeros((n_time, n_trades), dtype=np.float32)

    hl_buckets = data.get('BASKET_halflife', np.full(n_time*n_trades, 100.0)) * 24.0
    
    multiclass, prices, rolls, carries, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, net_roll,
        hl_buckets.reshape(n_time, n_trades), 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5,
        use_roll_in_quality=False
    )
    
    # 5. Audit
    rate_vals = pivots['rate'].values
    # Get exit rates efficiently
    # We already have audit_L1_entry_rate_drop from project_smart
    # We need exit rates.
    
    # Map back indices
    idx_map_1 = [x[0] for x in leg_indices]
    idx_map_2 = [x[1] for x in leg_indices]
    
    L1_rate_mat = rate_vals[:, idx_map_1]
    L2_rate_mat = rate_vals[:, idx_map_2]
    
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
    
    data['target_multiclass'] = multiclass.ravel()
    data['target_binary'] = (multiclass.ravel() == 2).astype(np.int8)
    
    save_by_month_incremental(data, idx_time, "curves")
    del data, net_rates, net_drift, net_roll
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
    leg_indices = []

    for t1, t2, t3 in combos:
        ids.append(f"F_{t1:g}_{t2:g}_{t3:g}")
        t1_arr.append(t1); t2_arr.append(t2); t3_arr.append(t3)
        width = t3 - t1
        ratio = (t2 - t1) / width if width != 0 else 0.5
        fly_width_arr.append(width); belly_ratio_arr.append(ratio)
        leg_indices.append([tenors.index(t1), tenors.index(t2), tenors.index(t3)])
        
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2, t3) in enumerate(combos):
        i1, i2, i3 = tenors.index(t1), tenors.index(t2), tenors.index(t3)
        W[i1, j] = 0.5; W[i2, j] = -1.0; W[i3, j] = 0.5
    W_abs = np.abs(W)
    
    # 1. Smart Projection
    data = project_smart_features(pivots, W, W_abs, ids, leg_indices)
    
    data['meta_direction'] = np.ones(n_time * n_trades, dtype=np.float32)
    data['meta_fly_width']  = np.tile(fly_width_arr, n_time)
    data['meta_belly_ratio'] = np.tile(belly_ratio_arr, n_time)
    data['meta_t2'] = np.tile(t2_arr, n_time)
    
    t1 = np.tile(t1_arr, n_time); t3 = np.tile(t3_arr, n_time); t2 = np.tile(t2_arr, n_time)
    dist = t3 - t1; dist[dist==0]=1.0
    data['NET_slope_leakage'] = 0.5 - ((t3 - t2) / dist)

    # 2. Modified Drift
    if 'NET_total_drift_day' in data:
        net_drift = data['NET_total_drift_day']
        for k in list(data.keys()):
            if k.startswith('NET_z') and 'modified' not in k:
                data[f"{k}_modified"] = calc_modified_carry(data[k], net_drift)

    data['ts'] = idx_time; data['trade_id'] = np.tile(ids, n_time)
    
    # 3. Scanner
    drift_key = 'total_drift_day_cumsum' 
    roll_key  = 'roll_bps_day_cumsum' 
    if drift_key not in pivots: drift_key = 'total_drift_cumsum'
    if roll_key not in pivots: roll_key = 'roll_bps_cumsum'

    net_rates = pivots['rate'].values @ W
    net_drift = pivots[drift_key].values @ W
    if roll_key in pivots: net_roll = pivots[roll_key].values @ W
    else: net_roll = np.zeros((n_time, n_trades), dtype=np.float32)

    hl_buckets = data.get('BASKET_halflife', np.full(n_time*n_trades, 100.0)) * 24.0 
    
    multiclass, prices, rolls, carries, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, net_roll,
        hl_buckets.reshape(n_time, n_trades), 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5,
        use_roll_in_quality=True
    )
    
    # 4. Audit
    rate_vals = pivots['rate'].values
    idx_map_1 = [x[0] for x in leg_indices]
    idx_map_2 = [x[1] for x in leg_indices]
    idx_map_3 = [x[2] for x in leg_indices]
    
    L1_rate_mat = rate_vals[:, idx_map_1]
    L2_rate_mat = rate_vals[:, idx_map_2]
    L3_rate_mat = rate_vals[:, idx_map_3]
    
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
    
    data['target_multiclass'] = multiclass.ravel()
    data['target_binary'] = (multiclass.ravel() == 2).astype(np.int8)
    
    save_by_month_incremental(data, idx_time, "flys")
    del data, net_rates, net_drift, net_roll
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
    
    dfs = None
    gc.collect()
    
    pivots = get_pivots(df_full)
    tenors = sorted(pivots['rate'].columns)
    
    del df_full
    gc.collect()
    
    build_curves(pivots, tenors)
    build_flys(pivots, tenors)
    
    print("Done.")

if __name__ == "__main__":
    process_full_history()
