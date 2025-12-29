"""
strategy_generator.py (v5.0 - Research Aligned & Fully Fleshed)

1. Splits processing into 'Curves' and 'Flys'.
2. Projects Features:
   - Directional (Z, Drift): Net Difference (Signed Weights).
   - Regime (Vol, Scale): Weighted Average (Abs Weights).
   - Events (Auctions): MINIMUM (Proximity).
3. Derived Physics (Research Aligned):
   - Modified Carry (Sigmoid): Z * (2 / (1 + e^-Drift)). Calculated for NET and ALL LEGS.
   - Slope Leakage (Convexity): Deviation of 50/50 weights from Slope Neutrality.
   - Signal/Noise Ratios: Drift/Vol, Z/Vol.
   - Model Divergence: PCA vs Spline Z-scores.
4. Output: Fully enriched dataset with Audit Trails.
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
# 1. NUMBA SCANNER (Universal)
# ==============================================================================
@numba.njit(parallel=True)
def scan_pnl_audit(
    entry_rates, drift_cumsums, halflife_idxs, 
    price_hurdle, total_hurdle, stop_loss
):
    n_time, n_trades = entry_rates.shape
    out_label = np.zeros((n_time, n_trades), dtype=np.int8)
    out_price = np.zeros((n_time, n_trades), dtype=np.float32)
    out_total = np.zeros((n_time, n_trades), dtype=np.float32)
    out_exit_idx  = np.zeros((n_time, n_trades), dtype=np.int32)
    out_exit_rate = np.zeros((n_time, n_trades), dtype=np.float32)
    
    for j in numba.prange(n_trades):
        for i in range(n_time):
            steps = int(halflife_idxs[i, j] * 2)
            if steps < 5: steps = 5        
            if steps > 24*90: steps = 24*90 
            
            end_idx = min(i + steps, n_time)
            rate_in = entry_rates[i, j]
            drift_in = drift_cumsums[i, j]
            
            best_price = 0.0
            best_total = 0.0
            final_idx = end_idx - 1
            won = False
            stopped = False
            
            for k in range(i + 1, end_idx):
                curr_price = (rate_in - entry_rates[k, j]) * 100.0
                curr_income = drift_cumsums[k, j] - drift_in
                curr_total = curr_price + curr_income
                
                if curr_total <= stop_loss:
                    out_label[i, j] = 0 
                    out_price[i, j] = curr_price
                    out_total[i, j] = curr_total
                    final_idx = k
                    stopped = True
                    break
                
                if (curr_price >= price_hurdle) and (curr_total >= total_hurdle):
                    out_label[i, j] = 1 
                    out_price[i, j] = curr_price
                    out_total[i, j] = curr_total
                    final_idx = k
                    won = True
                    break
                
                best_price = curr_price
                best_total = curr_total
            
            if not won and not stopped:
                out_price[i, j] = best_price
                out_total[i, j] = best_total
                
            out_exit_idx[i, j] = final_idx
            out_exit_rate[i, j] = entry_rates[final_idx, j]

    return out_label, out_price, out_total, out_exit_idx, out_exit_rate

# ==============================================================================
# 2. HELPER: MATH & PROJECTION
# ==============================================================================
def get_pivots(df):
    """Pivots ALL columns into Time x Tenor matrices."""
    pivots = {}
    cols = [c for c in df.columns if c not in ['ts', 'tenor_yrs']]
    print(f"   Pivoting {len(cols)} features...")
    for c in cols:
        pivots[c] = df.pivot(index='ts', columns='tenor_yrs', values=c).fillna(method='ffill').astype(np.float32)
    return pivots

def project_features(pivots, W, W_abs, trade_names):
    """
    Projects Features based on Type:
    1. Directional (Z, Drift) -> Signed Net.
    2. Regime (Vol, Scale) -> Weighted Avg.
    """
    data = {}
    
    REGIME_KEYS = ['halflife', 'scale', 'dv01', 'exog_']
    EVENT_KEYS  = ['hours_to_'] # Handle manually
    SKIP_KEYS   = ['cumsum']

    for name, mat in pivots.items():
        if any(x in name for x in SKIP_KEYS + EVENT_KEYS): continue
        
        clean = name.replace("z_comb", "z").replace("total_drift_day", "drift").replace("rate", "rate")
        clean = clean.replace("exog_", "")
        
        vals = mat.values
        
        if any(x in name for x in REGIME_KEYS):
            # REGIME (Weighted Average)
            norm_factor = np.sum(W_abs, axis=0)
            norm_factor[norm_factor == 0] = 1.0 
            data[f"NET_{clean}"] = (vals @ W_abs) / norm_factor
        else:
            # DIRECTIONAL (Signed Net)
            data[f"NET_{clean}"] = vals @ W
            
    for k in data:
        data[k] = data[k].ravel()
    return data

def calc_modified_carry(z_arr, drift_arr):
    """
    Research Paper Logic: Z * (2 / (1 + exp(-Drift))).
    Amplifies Z if Drift supports it, dampens if opposed.
    Assumes Drift is in bps/day. Scaling drift by 2.0 makes the sigmoid steeper.
    """
    sigmoid = 2.0 / (1.0 + np.exp(-drift_arr * 2.0))
    return z_arr * sigmoid

def make_inverse(df):
    """Generates Inverse trades (Flipping Directional features)."""
    df_inv = df.copy()
    df_inv['trade_id'] += "_INV"
    
    cols_to_flip = ['aux_price_pnl', 'aux_total_pnl', 'aux_entry_rate', 'aux_exit_rate']
    
    for c in df_inv.columns:
        if c.startswith('NET_'):
            # Flip Directional: Z, Drift, Rate, Slope, Accel, Divergence, Ratio, ModZ, Leakage
            if any(x in c for x in ['z', 'drift', 'rate', 'slope', 'accel', 'divergence', 'ratio', 'modified', 'leakage']):
                cols_to_flip.append(c)
                
    for c in cols_to_flip:
        df_inv[c] *= -1
        
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
    
    ids, t1_arr, t2_arr, dist_arr = [], [], [], []
    for t1, t2 in combos:
        ids.append(f"C_{t1:g}_{t2:g}")
        t1_arr.append(t1)
        t2_arr.append(t2)
        dist_arr.append(t2 - t1)
        
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2) in enumerate(combos):
        i1, i2 = tenors.index(t1), tenors.index(t2)
        W[i1, j] = 1.0
        W[i2, j] = -1.0
    W_abs = np.abs(W)
    
    # --- 1. Project Standard Features ---
    data = project_features(pivots, W, W_abs, ids)
    
    # --- 2. Leg Features & Events ---
    print(f"   Gathering Leg Features & Events...")
    
    # Grab all Z types to do Modified Carry on all of them
    z_types = ['z_comb', 'z_pca', 'z_spline']
    target_leg_feats = ['rate', 'total_drift_day', 'z_comb_slope_5b'] + z_types
    
    pivot_cols = list(pivots.keys())
    cols_to_grab = []
    
    for base in target_leg_feats:
        if base in pivot_cols: cols_to_grab.append(base)
    for c in pivot_cols:
        if 'hours_to_' in c or 'exog_' in c: cols_to_grab.append(c)
    cols_to_grab = list(set(cols_to_grab))
    
    idx_map_1 = [tenors.index(c[0]) for c in combos]
    idx_map_2 = [tenors.index(c[1]) for c in combos]
    
    # Holders for calculation
    L1_drift, L2_drift = None, None
    
    # First Pass: Get Raw Values
    for feat in cols_to_grab:
        vals = pivots[feat].values
        l1 = vals[:, idx_map_1].ravel()
        l2 = vals[:, idx_map_2].ravel()
        
        clean = feat.replace("z_comb", "z").replace("total_drift_day", "drift").replace("exog_", "").replace("hours_to_", "h_")
        data[f"L1_{clean}"] = l1
        data[f"L2_{clean}"] = l2
        
        if feat == 'total_drift_day':
            L1_drift, L2_drift = l1, l2
        
        if 'hours_to_' in feat:
            data[f"NET_{clean}_min"] = np.minimum(l1, l2)
            
        if feat == 'z_comb':
            data['NET_leg_stress'] = np.maximum(np.abs(l1), np.abs(l2))

    # --- 3. Derived Physics (Research Aligned) ---
    print(f"   Calculating Physics (Modified Carry, Divergence)...")
    
    # A. Modified Carry (Drift Adjusted Z) - For NET and LEGS
    # NET Level
    if 'NET_drift' in data:
        for z_name in ['NET_z', 'NET_z_pca', 'NET_z_spline']:
            if z_name in data:
                data[f"{z_name}_modified"] = calc_modified_carry(data[z_name], data['NET_drift'])
    
    # LEG Level (Requires L1_drift/L2_drift to be found above)
    if L1_drift is not None:
        for z_key in z_types:
            clean_z = z_key.replace("z_comb", "z")
            if f"L1_{clean_z}" in data:
                data[f"L1_{clean_z}_modified"] = calc_modified_carry(data[f"L1_{clean_z}"], L1_drift)
            if f"L2_{clean_z}" in data:
                data[f"L2_{clean_z}_modified"] = calc_modified_carry(data[f"L2_{clean_z}"], L2_drift)

    # B. Divergence
    if 'NET_z_pca' in data and 'NET_z_spline' in data:
        data['NET_z_divergence'] = data['NET_z_pca'] - data['NET_z_spline']
        
    # C. Signal-to-Noise
    vol_col = next((k for k in data if 'vol_implied' in k), None)
    if vol_col:
        safe_vol = np.maximum(data[vol_col], 0.1)
        data['NET_drift_vol_ratio'] = data['NET_drift'] / safe_vol
        data['NET_z_vol_ratio'] = data['NET_z'] / safe_vol

    # --- 4. Metadata & PnL ---
    data['ts'] = idx_time
    data['trade_id'] = np.tile(ids, n_time)
    data['meta_t1'] = np.tile(t1_arr, n_time)
    data['meta_t2'] = np.tile(t2_arr, n_time)
    data['meta_dist'] = np.tile(dist_arr, n_time)

    net_rates = pivots['rate'].values @ W
    net_drift = pivots['total_drift_day_cumsum'].values @ W
    hl_buckets = data['NET_halflife'] * 24.0
    hl_reshaped = hl_buckets.reshape(n_time, n_trades)
    
    labels, prices, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, hl_reshaped, 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5
    )
    
    data['target_label'] = labels.ravel()
    data['aux_price_pnl'] = prices.ravel()
    data['aux_total_pnl'] = totals.ravel()
    data['aux_entry_rate'] = net_rates.ravel()
    data['aux_exit_rate'] = ex_rate.ravel()
    data['aux_exit_idx'] = ex_idx.ravel()
    
    df = pd.DataFrame(data)
    df_inv = make_inverse(df)
    final = pd.concat([df, df_inv], ignore_index=True).dropna()
    
    out_p = Path(getattr(cr, "PATH_ENH", ".")) / f"training_curves_{month_str}.parquet"
    final.to_parquet(out_p, index=False)
    print(f"   Saved {len(final)} CURVES -> {out_p}")
    del df, df_inv, final
    gc.collect()

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
        t1_arr.append(t1)
        t2_arr.append(t2)
        t3_arr.append(t3)
        
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2, t3) in enumerate(combos):
        i1, i2, i3 = tenors.index(t1), tenors.index(t2), tenors.index(t3)
        W[i1, j] = 0.5
        W[i2, j] = -1.0
        W[i3, j] = 0.5
    W_abs = np.abs(W)
    
    # --- 1. Project Standard Features ---
    data = project_features(pivots, W, W_abs, ids)
    
    # --- 2. Leg Features & Events ---
    print(f"   Gathering Leg Features & Events...")
    
    z_types = ['z_comb', 'z_pca', 'z_spline']
    target_leg_feats = ['rate', 'total_drift_day', 'z_comb_slope_5b'] + z_types
    
    pivot_cols = list(pivots.keys())
    cols_to_grab = []
    for base in target_leg_feats:
        if base in pivot_cols: cols_to_grab.append(base)
    for c in pivot_cols:
        if 'hours_to_' in c or 'exog_' in c: cols_to_grab.append(c)
    cols_to_grab = list(set(cols_to_grab))
    
    idx_map_1 = [tenors.index(c[0]) for c in combos]
    idx_map_2 = [tenors.index(c[1]) for c in combos]
    idx_map_3 = [tenors.index(c[2]) for c in combos]
    
    L1_drift, L2_drift, L3_drift = None, None, None
    
    for feat in cols_to_grab:
        vals = pivots[feat].values
        l1 = vals[:, idx_map_1].ravel()
        l2 = vals[:, idx_map_2].ravel()
        l3 = vals[:, idx_map_3].ravel()
        
        clean = feat.replace("z_comb", "z").replace("total_drift_day", "drift").replace("exog_", "").replace("hours_to_", "h_")
        data[f"L1_{clean}"] = l1
        data[f"L2_{clean}"] = l2
        data[f"L3_{clean}"] = l3
        
        if feat == 'total_drift_day':
            L1_drift, L2_drift, L3_drift = l1, l2, l3
        
        if 'hours_to_' in feat:
            data[f"NET_{clean}_min"] = np.minimum(np.minimum(l1, l2), l3)

    # --- 3. Derived Physics (Research Aligned) ---
    print(f"   Calculating Physics (Slope Leakage, Modified Carry)...")
    
    # A. Slope Leakage (Convexity Bias)
    # Calculate Ideal Slope Neutral Weight for Left Wing: (Tr - Tb) / (Tr - Tl)
    t1 = np.tile(t1_arr, n_time)
    t2 = np.tile(t2_arr, n_time)
    t3 = np.tile(t3_arr, n_time)
    
    dist_total = t3 - t1
    dist_total[dist_total == 0] = 1.0 # Safety
    w_left_ideal = (t3 - t2) / dist_total
    
    # Feature: Difference between our 0.5 weight and the ideal slope neutral weight
    data['NET_slope_leakage'] = 0.5 - w_left_ideal
    
    # B. Modified Carry (Drift Adjusted Z)
    if 'NET_drift' in data:
        for z_name in ['NET_z', 'NET_z_pca', 'NET_z_spline']:
            if z_name in data:
                data[f"{z_name}_modified"] = calc_modified_carry(data[z_name], data['NET_drift'])
                
    if L1_drift is not None:
        for z_key in z_types:
            clean_z = z_key.replace("z_comb", "z")
            if f"L1_{clean_z}" in data:
                data[f"L1_{clean_z}_modified"] = calc_modified_carry(data[f"L1_{clean_z}"], L1_drift)
            if f"L2_{clean_z}" in data:
                data[f"L2_{clean_z}_modified"] = calc_modified_carry(data[f"L2_{clean_z}"], L2_drift)
            if f"L3_{clean_z}" in data:
                data[f"L3_{clean_z}_modified"] = calc_modified_carry(data[f"L3_{clean_z}"], L3_drift)

    # C. Divergence & Ratios
    if 'NET_z_pca' in data and 'NET_z_spline' in data:
        data['NET_z_divergence'] = data['NET_z_pca'] - data['NET_z_spline']
        
    vol_col = next((k for k in data if 'vol_implied' in k), None)
    if vol_col:
        safe_vol = np.maximum(data[vol_col], 0.1)
        data['NET_drift_vol_ratio'] = data['NET_drift'] / safe_vol
        data['NET_z_vol_ratio'] = data['NET_z'] / safe_vol
        
    raw_vol_name = next(c for c in pivots if 'vol_implied' in c and 'exog' in c)
    data['NET_vol_smile'] = (pivots[raw_vol_name].values @ W).ravel()
    
    # --- 4. Metadata & PnL ---
    data['ts'] = idx_time
    data['trade_id'] = np.tile(ids, n_time)
    data['meta_t1'] = t1
    data['meta_t2'] = t2
    data['meta_t3'] = t3
    
    net_rates = pivots['rate'].values @ W
    net_drift = pivots['total_drift_day_cumsum'].values @ W
    hl_buckets = data['NET_halflife'] * 24.0
    hl_reshaped = hl_buckets.reshape(n_time, n_trades)
    
    labels, prices, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, hl_reshaped, 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5
    )
    
    data['target_label'] = labels.ravel()
    data['aux_price_pnl'] = prices.ravel()
    data['aux_total_pnl'] = totals.ravel()
    data['aux_entry_rate'] = net_rates.ravel()
    data['aux_exit_rate'] = ex_rate.ravel()
    
    df = pd.DataFrame(data)
    df_inv = make_inverse(df)
    final = pd.concat([df, df_inv], ignore_index=True).dropna()
    
    out_p = Path(getattr(cr, "PATH_ENH", ".")) / f"training_flys_{month_str}.parquet"
    final.to_parquet(out_p, index=False)
    print(f"   Saved {len(final)} FLYS -> {out_p}")
    del df, df_inv, final
    gc.collect()

# ==============================================================================
# 5. ORCHESTRATOR
# ==============================================================================
def process_month(month_str):
    p = Path(getattr(cr, "PATH_ENH", ".")) / f"{month_str}_enh{getattr(cr, 'ENH_SUFFIX', '')}.parquet"
    if not p.exists(): 
        print(f"[{month_str}] Not found: {p}")
        return

    print(f"[{month_str}] Loading Features...")
    df = pd.read_parquet(p)
    pivots = get_pivots(df)
    tenors = sorted(pivots['rate'].columns)
    
    build_curves(pivots, tenors, month_str)
    build_flys(pivots, tenors, month_str)
    print(f"[{month_str}] Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python strategy_generator.py 2304")
    for m in sys.argv[1:]:
        process_month(m)