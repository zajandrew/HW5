"""
strategy_generator.py (v3.0 - Split Models & Full Audit)

1. Splits processing into 'Curves' and 'Flys'.
2. Generates 'NET' features (Matrix Algebra) AND 'LEG' features (Direct Lookup).
3. Saves fully enriched datasets ready for XGBoost X/y construction.
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
    entry_rates,       # (Time, Trades) - Net Weighted Entry Rate
    drift_cumsums,     # (Time, Trades) - Net Weighted Drift CumSum
    halflife_idxs,     # (Time, Trades) - Holding Period in Buckets
    price_hurdle,      
    total_hurdle,      
    stop_loss          
):
    n_time, n_trades = entry_rates.shape
    
    # Results
    out_label = np.zeros((n_time, n_trades), dtype=np.int8)
    out_price = np.zeros((n_time, n_trades), dtype=np.float32)
    out_total = np.zeros((n_time, n_trades), dtype=np.float32)
    
    # Audit
    out_exit_idx  = np.zeros((n_time, n_trades), dtype=np.int32)
    out_exit_rate = np.zeros((n_time, n_trades), dtype=np.float32)
    
    for j in numba.prange(n_trades):
        for i in range(n_time):
            
            # Dynamic Horizon
            steps = int(halflife_idxs[i, j] * 2)
            if steps < 5: steps = 5        
            if steps > 24*90: steps = 24*90 
            
            end_idx = min(i + steps, n_time)
            
            # Entry State
            rate_in = entry_rates[i, j]
            drift_in = drift_cumsums[i, j]
            
            # Default: Time Limit
            best_price = 0.0
            best_total = 0.0
            final_idx = end_idx - 1
            
            won = False
            stopped = False
            
            for k in range(i + 1, end_idx):
                # 1. Net Price PnL (Includes Direction via weights)
                curr_price = (rate_in - entry_rates[k, j]) * 100.0
                
                # 2. Net Income PnL (Includes Direction via weights)
                curr_income = drift_cumsums[k, j] - drift_in
                
                curr_total = curr_price + curr_income
                
                # Check Stop
                if curr_total <= stop_loss:
                    out_label[i, j] = 0 
                    out_price[i, j] = curr_price
                    out_total[i, j] = curr_total
                    final_idx = k
                    stopped = True
                    break
                
                # Check Win
                if (curr_price >= price_hurdle) and (curr_total >= total_hurdle):
                    out_label[i, j] = 1 
                    out_price[i, j] = curr_price
                    out_total[i, j] = curr_total
                    final_idx = k
                    won = True
                    break
                
                # Track latest
                best_price = curr_price
                best_total = curr_total
            
            if not won and not stopped:
                out_price[i, j] = best_price
                out_total[i, j] = best_total
                
            out_exit_idx[i, j] = final_idx
            out_exit_rate[i, j] = entry_rates[final_idx, j]

    return out_label, out_price, out_total, out_exit_idx, out_exit_rate

# ==============================================================================
# 2. HELPER: MATRIX GENERATION
# ==============================================================================
def get_pivots(df):
    """Pivots ALL columns into Time x Tenor matrices."""
    pivots = {}
    
    # 1. Identify Columns
    # We want everything in 'df' that is feature-like
    cols = [c for c in df.columns if c not in ['ts', 'tenor_yrs']]
    
    print(f"   Pivoting {len(cols)} features...")
    for c in cols:
        # FFill is critical for data integrity
        pivots[c] = df.pivot(index='ts', columns='tenor_yrs', values=c).fillna(method='ffill').astype(np.float32)
        
    return pivots

def make_inverse(df, type_tag):
    """Generates the Inverse (Payer/Short) trades."""
    df_inv = df.copy()
    df_inv['trade_id'] += "_INV"
    
    # Columns to flip: Targets + Net Directional Features
    # We do NOT flip Leg Features (L1_z) because the Payer version still sees the same L1 z-score, 
    # it just interprets it differently via the Net feature.
    # Actually: If we want the Model to learn "High Z = Buy", then for the Inverse trade, 
    # we should flip the Net features.
    
    cols_to_flip = ['aux_price_pnl', 'aux_total_pnl', 'aux_entry_rate', 'aux_exit_rate']
    
    for c in df_inv.columns:
        if c.startswith('NET_'):
            # Flip Net Z, Net Drift, etc.
            if any(x in c for x in ['z', 'drift', 'rate', 'slope', 'accel']):
                cols_to_flip.append(c)
                
    for c in cols_to_flip:
        df_inv[c] *= -1
        
    # Recalculate Label
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
    
    # A. Define Universe (T1, T2)
    # T1 = Front (Buy/Rec), T2 = Back (Sell/Pay) -> Flattener
    # Weights: [+1, -1]
    combos = list(combinations(tenors, 2))
    n_trades = len(combos)
    n_time = len(pivots['rate'].index)
    
    # B. Metadata Arrays
    idx_time = np.repeat(pivots['rate'].index, n_trades)
    
    # Trade IDs and Attributes
    ids = []
    t1_arr = []
    t2_arr = []
    dist_arr = []
    
    for t1, t2 in combos:
        ids.append(f"C_{t1:g}_{t2:g}")
        t1_arr.append(t1)
        t2_arr.append(t2)
        dist_arr.append(t2 - t1)
        
    # Tile attributes for the DataFrame
    ids_tiled = np.tile(ids, n_time)
    t1_tiled = np.tile(t1_arr, n_time)
    t2_tiled = np.tile(t2_arr, n_time)
    dist_tiled = np.tile(dist_arr, n_time)
    
    # C. Build Weight Matrix W (n_tenors, n_trades)
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2) in enumerate(combos):
        i1 = tenors.index(t1)
        i2 = tenors.index(t2)
        W[i1, j] = 1.0
        W[i2, j] = -1.0
        
    W_abs = np.abs(W) # For Vol/HalfLife
    
    # D. Feature Dictionary
    data = {
        'ts': idx_time,
        'trade_id': ids_tiled,
        'meta_t1': t1_tiled,
        'meta_t2': t2_tiled,
        'meta_dist': dist_tiled
    }
    
    def flat(x): return x.ravel()
    
    # --- NET FEATURES (Matrix Mult) ---
    # "What is the Z-score of the spread?"
    for name, mat in pivots.items():
        if any(x in name for x in ['halflife', 'exog_', 'cumsum']): continue
        # Clean name
        clean = name.replace("z_comb", "z").replace("total_drift_day", "drift")
        data[f"NET_{clean}"] = flat(mat.values @ W)
        
    # Regime Features (Weighted Avg)
    if 'halflife' in pivots:
        data['NET_halflife'] = flat((pivots['halflife'].values @ W_abs) / 2.0)
    for k in pivots.keys():
        if 'exog_vol' in k:
            clean = k.replace("exog_", "")
            data[f"NET_{clean}"] = flat((pivots[k].values @ W_abs) / 2.0)
            
    # --- LEG FEATURES (Direct Indexing) ---
    # "What is the Z-score of the 2Y leg specifically?"
    # Since we can't efficiently store 300 leg cols for every trade via matrix,
    # we construct them by repeating the source tenor columns.
    # Optimization: Use 'take' or indexing logic.
    
    # Extract indices for T1 and T2
    # This is slightly slow but robust
    print(f"   Gathering Leg Features...")
    
    # Key features to keep per leg (Don't keep everything or RAM explodes)
    keep_leg_feats = ['rate', 'z_comb', 'total_drift_day', 'z_comb_slope_5b', 'rate_slope_50b']
    
    for feat in keep_leg_feats:
        if feat not in pivots: continue
        # Matrix: (Time, Tenors)
        vals = pivots[feat].values 
        
        # We need to construct a (Time*Trades) array where each row corresponds to t1 or t2
        # Vectorized approach: 
        # 1. Create index map: trade_j -> tenor_idx_1, tenor_idx_2
        idx_map_1 = [tenors.index(c[0]) for c in combos]
        idx_map_2 = [tenors.index(c[1]) for c in combos]
        
        # 2. Select columns from vals using these indices
        # Result: (Time, Trades)
        L1_vals = vals[:, idx_map_1]
        L2_vals = vals[:, idx_map_2]
        
        data[f"L1_{feat}"] = flat(L1_vals)
        data[f"L2_{feat}"] = flat(L2_vals)

    # E. PnL Scan
    net_rates = pivots['rate'].values @ W
    net_drift = pivots['total_drift_day_cumsum'].values @ W
    hl_buckets = data['NET_halflife'] * 24.0
    hl_reshaped = hl_buckets.reshape(n_time, n_trades)
    
    labels, prices, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, hl_reshaped, 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5
    )
    
    data['target_label'] = flat(labels)
    data['aux_price_pnl'] = flat(prices)
    data['aux_total_pnl'] = flat(totals)
    data['aux_entry_rate'] = flat(net_rates)
    data['aux_exit_rate'] = flat(ex_rate)
    data['aux_exit_idx'] = flat(ex_idx)
    
    # F. Save
    df = pd.DataFrame(data)
    df_inv = make_inverse(df, "Curve")
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
    
    # A. Define Universe (T1, T2, T3)
    # Weights: [+0.5, -1.0, +0.5]
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
        
    ids_tiled = np.tile(ids, n_time)
    
    # C. Weight Matrix
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2, t3) in enumerate(combos):
        i1, i2, i3 = tenors.index(t1), tenors.index(t2), tenors.index(t3)
        W[i1, j] = 0.5
        W[i2, j] = -1.0
        W[i3, j] = 0.5
        
    W_abs = np.abs(W)
    
    data = {
        'ts': idx_time,
        'trade_id': ids_tiled,
        'meta_t1': np.tile(t1_arr, n_time),
        'meta_t2': np.tile(t2_arr, n_time), # Belly
        'meta_t3': np.tile(t3_arr, n_time)
    }
    
    def flat(x): return x.ravel()
    
    # --- NET FEATURES ---
    for name, mat in pivots.items():
        if any(x in name for x in ['halflife', 'exog_', 'cumsum']): continue
        clean = name.replace("z_comb", "z").replace("total_drift_day", "drift")
        data[f"NET_{clean}"] = flat(mat.values @ W)

    if 'halflife' in pivots:
        data['NET_halflife'] = flat((pivots['halflife'].values @ W_abs) / 2.0)
    for k in pivots.keys():
        if 'exog_vol' in k:
            clean = k.replace("exog_", "")
            data[f"NET_{clean}"] = flat((pivots[k].values @ W_abs) / 2.0)

    # --- LEG FEATURES ---
    print(f"   Gathering Leg Features...")
    keep_leg_feats = ['rate', 'z_comb', 'total_drift_day'] # Keep leaner for Flys (more trades)
    
    idx_map_1 = [tenors.index(c[0]) for c in combos]
    idx_map_2 = [tenors.index(c[1]) for c in combos]
    idx_map_3 = [tenors.index(c[2]) for c in combos]
    
    for feat in keep_leg_feats:
        if feat not in pivots: continue
        vals = pivots[feat].values
        
        data[f"L1_{feat}"] = flat(vals[:, idx_map_1])
        data[f"L2_{feat}"] = flat(vals[:, idx_map_2]) # Belly
        data[f"L3_{feat}"] = flat(vals[:, idx_map_3])
        
    # E. PnL Scan
    net_rates = pivots['rate'].values @ W
    net_drift = pivots['total_drift_day_cumsum'].values @ W
    hl_buckets = data['NET_halflife'] * 24.0
    hl_reshaped = hl_buckets.reshape(n_time, n_trades)
    
    labels, prices, totals, ex_idx, ex_rate = scan_pnl_audit(
        net_rates, net_drift, hl_reshaped, 
        price_hurdle=1.0, total_hurdle=2.0, stop_loss=-1.5
    )
    
    data['target_label'] = flat(labels)
    data['aux_price_pnl'] = flat(prices)
    data['aux_total_pnl'] = flat(totals)
    data['aux_entry_rate'] = flat(net_rates)
    data['aux_exit_rate'] = flat(ex_rate)
    
    df = pd.DataFrame(data)
    df_inv = make_inverse(df, "Fly")
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
    
    # 1. Prepare Data
    pivots = get_pivots(df)
    tenors = sorted(pivots['rate'].columns)
    
    # 2. Build Models
    build_curves(pivots, tenors, month_str)
    build_flys(pivots, tenors, month_str)
    
    print(f"[{month_str}] Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python strategy_generator.py 2304")
    for m in sys.argv[1:]:
        process_month(m)
