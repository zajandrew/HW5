"""
strategy_generator.py (v3.1 - The XGBoost Matrix Generator)

1. Splits processing into 'Curves' and 'Flys'.
2. Projects ALL features (Slope, Accel, Z) via Matrix Algebra.
3. Calculates Derived Physics (Sharpe Ratios, Dislocation, Fly Smile).
4. Scans PnL using Total Drift (Carry+Roll) + Constant Maturity Price.
5. Outputs 'XGBoost Ready' rows with Targets, Features, and Audit trails.
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
# 1. NUMBA SCANNER (Universal PnL Engine)
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
            
            # Dynamic Horizon (2x HalfLife, Min 5h, Max 3 Months)
            steps = int(halflife_idxs[i, j] * 2)
            if steps < 5: steps = 5        
            if steps > 24*90: steps = 24*90 
            
            end_idx = min(i + steps, n_time)
            
            # Entry State
            rate_in = entry_rates[i, j]
            drift_in = drift_cumsums[i, j]
            
            # Default: Time Limit Exit
            best_price = 0.0
            best_total = 0.0
            final_idx = end_idx - 1
            
            won = False
            stopped = False
            
            for k in range(i + 1, end_idx):
                # 1. Net Price PnL (Direction handled by Weights in Entry/Exit)
                # Curve Shift = (Entry - Curr) * 100
                curr_price = (rate_in - entry_rates[k, j]) * 100.0
                
                # 2. Net Income PnL (Direction handled by Weights)
                # Integrated Drift = CurrCumSum - EntryCumSum
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
                
                # Track latest for Time Limit
                best_price = curr_price
                best_total = curr_total
            
            if not won and not stopped:
                out_price[i, j] = best_price
                out_total[i, j] = best_total
                
            out_exit_idx[i, j] = final_idx
            out_exit_rate[i, j] = entry_rates[final_idx, j]

    return out_label, out_price, out_total, out_exit_idx, out_exit_rate

# ==============================================================================
# 2. HELPER: FEATURE PROJECTION
# ==============================================================================
def get_pivots(df):
    """Pivots ALL columns into Time x Tenor matrices."""
    pivots = {}
    
    # 1. Identify Columns
    cols = [c for c in df.columns if c not in ['ts', 'tenor_yrs']]
    
    print(f"   Pivoting {len(cols)} features...")
    for c in cols:
        # FFill is critical for data integrity
        pivots[c] = df.pivot(index='ts', columns='tenor_yrs', values=c).fillna(method='ffill').astype(np.float32)
        
    return pivots

def project_features(pivots, W, W_abs, trade_names):
    """
    Core Logic: Projects Leg Features onto Trades.
    Handles Directional (Net) vs Regime (Avg) logic automatically.
    """
    data = {}
    
    # A. Define Categories
    # REGIME: Weighted Average (Use W_abs)
    # DIRECTIONAL: Weighted Net (Use W)
    
    # Context columns that are Regime based
    REGIME_KEYWORDS = ['halflife', 'scale', 'dv01', 'exog_', 'hours_to_auction']
    
    # PnL Columns to skip in Feature Set
    SKIP_KEYWORDS = ['cumsum']

    for name, mat in pivots.items():
        if any(x in name for x in SKIP_KEYWORDS): continue
        
        # XGBoost Friendly Name
        # e.g. z_comb_slope_5b -> z_slope_5b
        clean = name.replace("z_comb", "z").replace("total_drift_day", "drift").replace("rate", "rate")
        clean = clean.replace("exog_", "")
        
        vals = mat.values
        
        if any(x in name for x in REGIME_KEYWORDS):
            # REGIME: Weighted Average
            # Logic: If 2Y is Volatile and 10Y is Volatile, Trade is Volatile.
            # Normalization: Sum(W_abs) per trade handles 2-leg vs 3-leg normalization.
            norm_factor = np.sum(W_abs, axis=0)
            norm_factor[norm_factor == 0] = 1.0 # Safety
            data[f"NET_{clean}"] = (vals @ W_abs) / norm_factor
        else:
            # DIRECTIONAL: Net Difference
            # Logic: Buy 2Y (Slope +1), Sell 10Y (Slope +1) -> Net Slope 0.
            data[f"NET_{clean}"] = vals @ W
            
    # Flatten all arrays
    for k in data:
        data[k] = data[k].ravel()
        
    return data

def make_inverse(df):
    """Generates the Inverse (Payer/Short) trades for Data Augmentation."""
    df_inv = df.copy()
    df_inv['trade_id'] += "_INV"
    
    # Columns to flip: 
    # 1. Targets (PnL)
    # 2. Audit (Entry/Exit Rates)
    # 3. Directional Features (Net Z, Net Drift, Net Slope)
    
    cols_to_flip = ['aux_price_pnl', 'aux_total_pnl', 'aux_entry_rate', 'aux_exit_rate']
    
    for c in df_inv.columns:
        if c.startswith('NET_'):
            # Flip Net Z, Drift, Rate, Divergence, Ratios
            # Do NOT flip Halflife, Vol, Scale, Smile (Convexity is unsigned cost)
            if any(x in c for x in ['z', 'drift', 'rate', 'slope', 'accel', 'divergence', 'ratio']):
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
    
    # A. Define Universe
    combos = list(combinations(tenors, 2))
    n_trades = len(combos)
    n_time = len(pivots['rate'].index)
    
    idx_time = np.repeat(pivots['rate'].index, n_trades)
    
    # Metadata
    ids, t1_arr, t2_arr, dist_arr = [], [], [], []
    for t1, t2 in combos:
        ids.append(f"C_{t1:g}_{t2:g}")
        t1_arr.append(t1)
        t2_arr.append(t2)
        dist_arr.append(t2 - t1)
        
    # B. Weight Matrix
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2) in enumerate(combos):
        i1, i2 = tenors.index(t1), tenors.index(t2)
        W[i1, j] = 1.0
        W[i2, j] = -1.0
        
    W_abs = np.abs(W)
    
    # C. Project Features
    data = project_features(pivots, W, W_abs, ids)
    
    # D. Derived Physics (The "Step 4.5" Logic)
    print(f"   Calculating Derived Physics (Signal/Noise, Divergence)...")
    
    # 1. Divergence (PCA vs Spline)
    if 'NET_z_pca' in data and 'NET_z_spline' in data:
        data['NET_z_divergence'] = data['NET_z_pca'] - data['NET_z_spline']
        
    # 2. Signal-to-Noise Ratios
    # Use implied vol as the denominator (Regime). Clip to avoid div/0.
    vol_col = next((k for k in data if 'vol_implied' in k), None)
    if vol_col:
        safe_vol = np.maximum(data[vol_col], 0.1)
        data['NET_drift_vol_ratio'] = data['NET_drift'] / safe_vol
        data['NET_z_vol_ratio'] = data['NET_z'] / safe_vol
        
    # E. Leg Features (Stress Detection)
    print(f"   Gathering Leg Features...")
    keep_leg_feats = ['rate', 'z_comb', 'total_drift_day', 'z_comb_slope_5b']
    
    # Map indices
    idx_map_1 = [tenors.index(c[0]) for c in combos]
    idx_map_2 = [tenors.index(c[1]) for c in combos]
    
    leg_stress = [] # Holder for MaxAbs(Z)
    
    for feat in keep_leg_feats:
        if feat not in pivots: continue
        vals = pivots[feat].values
        
        l1 = vals[:, idx_map_1].ravel()
        l2 = vals[:, idx_map_2].ravel()
        
        data[f"L1_{feat}"] = l1
        data[f"L2_{feat}"] = l2
        
        if feat == 'z_comb':
            # Calculate Leg Stress: Max(Abs(L1), Abs(L2))
            # If correlation breaks, both legs might be huge.
            stress = np.maximum(np.abs(l1), np.abs(l2))
            data['NET_leg_stress'] = stress

    # F. Base Metadata
    data['ts'] = idx_time
    data['trade_id'] = np.tile(ids, n_time)
    data['meta_t1'] = np.tile(t1_arr, n_time)
    data['meta_t2'] = np.tile(t2_arr, n_time)
    data['meta_dist'] = np.tile(dist_arr, n_time)

    # G. PnL Scan
    net_rates = pivots['rate'].values @ W
    net_drift = pivots['total_drift_day_cumsum'].values @ W
    # Convert Halflife (Days) to Buckets (Hours)
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
    
    # H. Save
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
    
    # A. Define Universe
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
        
    # B. Weight Matrix [0.5, -1.0, 0.5]
    W = np.zeros((len(tenors), n_trades), dtype=np.float32)
    for j, (t1, t2, t3) in enumerate(combos):
        i1, i2, i3 = tenors.index(t1), tenors.index(t2), tenors.index(t3)
        W[i1, j] = 0.5
        W[i2, j] = -1.0
        W[i3, j] = 0.5
        
    W_abs = np.abs(W)
    
    # C. Project Features
    data = project_features(pivots, W, W_abs, ids)
    
    # D. Derived Physics
    print(f"   Calculating Derived Physics (Smile, Ratios)...")
    
    if 'NET_z_pca' in data and 'NET_z_spline' in data:
        data['NET_z_divergence'] = data['NET_z_pca'] - data['NET_z_spline']
        
    vol_col = next((k for k in data if 'vol_implied' in k), None)
    if vol_col:
        safe_vol = np.maximum(data[vol_col], 0.1)
        data['NET_drift_vol_ratio'] = data['NET_drift'] / safe_vol
        data['NET_z_vol_ratio'] = data['NET_z'] / safe_vol
        
    # *** Fly Specific: Vol Smile (Convexity Cost) ***
    # W is [0.5, -1.0, 0.5].
    # NET_vol_implied (Weighted Avg) was calculated using W_abs [0.5, 1.0, 0.5].
    # We want "Smile" = 0.5*Vol_L + 0.5*Vol_R - 1.0*Vol_B.
    # This matches exactly applying W (signed) to the Vol Surface.
    # We need to manually calculate this because project_features used W_abs for all REGIME_KEYWORDS.
    raw_vol_name = next(c for c in pivots if 'vol_implied' in c and 'exog' in c)
    data['NET_vol_smile'] = (pivots[raw_vol_name].values @ W).ravel()
    
    # E. Leg Features
    print(f"   Gathering Leg Features...")
    keep_leg_feats = ['rate', 'z_comb', 'total_drift_day'] 
    
    idx_map_1 = [tenors.index(c[0]) for c in combos]
    idx_map_2 = [tenors.index(c[1]) for c in combos]
    idx_map_3 = [tenors.index(c[2]) for c in combos]
    
    for feat in keep_leg_feats:
        if feat not in pivots: continue
        vals = pivots[feat].values
        data[f"L1_{feat}"] = vals[:, idx_map_1].ravel()
        data[f"L2_{feat}"] = vals[:, idx_map_2].ravel()
        data[f"L3_{feat}"] = vals[:, idx_map_3].ravel()
    
    # F. Metadata
    data['ts'] = idx_time
    data['trade_id'] = np.tile(ids, n_time)
    data['meta_t1'] = np.tile(t1_arr, n_time)
    data['meta_t2'] = np.tile(t2_arr, n_time)
    data['meta_t3'] = np.tile(t3_arr, n_time)
    
    # G. PnL Scan
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
