"""
training_pipeline_dask.py (v17.1 - Bulletproof Alignment)

Role:
1. Solves 'AssertionError' by keeping X and y in the same Dask DataFrame.
2. Solves 'CancelledError' by lowering worker count to prevent OOM.
3. Uses Native DaskDMatrix column specification.

Usage:
    python training_pipeline_dask.py
"""

import gc
import numpy as np
import pandas as pd
import xgboost as xgb
import xgboost.dask # CRITICAL IMPORT
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import config as cr

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# v16.2 Params
XGB_PARAMS = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'device': 'cuda',
    'max_bin': 256,
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 0.0,
    'subsample': 0.70,
    'colsample_bytree': 0.50,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1.0
}

TRAIN_WINDOW_MONTHS = 18
NUM_ROUNDS = 5000
EARLY_STOPPING = 200

# ==============================================================================
# 1. DASK DATA LOADER (Unified)
# ==============================================================================
def get_file_list(strategy_type):
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    pattern = f"training_{strategy_type}_*.parquet"
    files = sorted(list(path_enh.glob(pattern)))
    return [str(f) for f in files]

def load_dask_data(files, purge_mode):
    """
    Loads data but KEEPS X and y together to prevent partition misalignment.
    """
    # 1. Read Parquet
    ddf = dd.read_parquet(files)
    
    # 2. Filter Rows (Purge Logic)
    if purge_mode:
        ddf = ddf[ddf['target_multiclass'] != 1]
        
    # 3. Create Binary Target Column IN PLACE
    # We add this column to the dataframe so it stays aligned
    ddf['target_binary'] = (ddf['target_multiclass'] == 2).astype('float32')
    
    # 4. Drop Unused Columns
    # We KEEP 'target_binary' for the label, and DROP everything else we don't need
    all_cols = ddf.columns
    cols_to_drop = [
        c for c in all_cols 
        if c.endswith('_drop') or c.startswith('target_multiclass') or c in ['ts', 'trade_id']
    ]
    
    # Drop columns but keep the unified dataframe
    ddf_clean = ddf.drop(columns=cols_to_drop)
    
    # Cast features to float32 (Label is already float32)
    ddf_clean = ddf_clean.astype('float32')
    
    return ddf_clean

# ==============================================================================
# 2. WALK-FORWARD ENGINE
# ==============================================================================
def run_walk_forward(client, strategy_type, purge_mode):
    files = get_file_list(strategy_type)
    if len(files) < (TRAIN_WINDOW_MONTHS + 1):
        print(f"[{strategy_type}] Not enough data.")
        return

    mode_name = "PURGED" if purge_mode else "STANDARD"
    mode_suffix = "purged" if purge_mode else "standard"
    print(f"\n=== Starting Walk-Forward: {strategy_type.upper()} [{mode_name}] ===")
    
    results = []
    
    for i in range(TRAIN_WINDOW_MONTHS, len(files)):
        test_file = files[i]
        train_files = files[i - TRAIN_WINDOW_MONTHS : i]
        
        test_month = Path(test_file).stem.split('_')[-1]
        train_start = Path(train_files[0]).stem.split('_')[-1]
        train_end = Path(train_files[-1]).stem.split('_')[-1]
        
        print(f"[{mode_name}] Round {i}: Train {train_start}-{train_end} -> Test {test_month}")
        
        # --- UNIFIED LOADING ---
        # We get a single dataframe containing features AND 'target_binary'
        ddf_train = load_dask_data(train_files, purge_mode)
        ddf_test = load_dask_data([test_file], purge_mode)
        
        # --- ROBUST DMatrix CREATION ---
        # We pass the WHOLE dataframe and tell Dask which column is the label.
        # This prevents the 'partitions inconsistent' error.
        dtrain = xgb.dask.DaskDMatrix(client, data=ddf_train, label='target_binary')
        dtest = xgb.dask.DaskDMatrix(client, data=ddf_test, label='target_binary')
        
        # --- TRAINING ---
        output = xgb.dask.train(
            client,
            XGB_PARAMS,
            dtrain,
            num_boost_round=NUM_ROUNDS,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=False
        )
        
        model = output['booster']
        
        # --- SCORING ---
        y_prob_dask = xgb.dask.predict(client, model, dtest)
        y_prob = y_prob_dask.compute()
        
        # Get Truth (Slice column locally)
        y_true = ddf_test['target_binary'].compute()
        
        y_pred = (y_prob > 0.52).astype(int)
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
            
        print(f"   -> AUC: {auc:.3f} | Prec: {prec:.2%} | Rec: {rec:.2%} | Trees: {model.best_iteration}")
        
        # Save Results
        model_sub = f"{strategy_type}_{mode_suffix}"
        model_dir = Path("models") / model_sub
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(model_dir / f"model_{test_month}.json")
        
        results.append({
            'test_month': test_month,
            'train_start': train_start,
            'train_end': train_end,
            'auc': auc,
            'precision': prec,
            'recall': rec,
            'best_iter': model.best_iteration
        })
        
        # Explicit cleanup
        del dtrain, dtest, output, model, ddf_train, ddf_test
        gc.collect()

    out_csv = f"audit_{strategy_type}_{mode_suffix}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)

# ==============================================================================
# 3. ORCHESTRATOR
# ==============================================================================
def run_both_modes():
    print("Initializing Dask Cluster...")
    
    # SAFE CONFIG FOR 64GB RAM:
    # Reduced to 3 Workers (leaves ~20GB headroom for OS/Client)
    # This prevents the 'CancelledError' OOM crash.
    cluster = LocalCluster(
        n_workers=3,          
        threads_per_worker=1, 
        memory_limit='14GB'   
    )
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")
    
    try:
        run_walk_forward(client, "curves", purge_mode=True)
        run_walk_forward(client, "curves", purge_mode=False)
        run_walk_forward(client, "flys", purge_mode=True)
        run_walk_forward(client, "flys", purge_mode=False)
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    run_both_modes()
