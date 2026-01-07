"""
training_pipeline_dask.py (v17.0 - Dask Industrial Grade)

Role:
1. Replaces manual iterators with Dask Distributed (Parallel Loading).
2. Manages Memory automatically (Spills to disk if RAM fills up).
3. Feeds XGBoost efficiently for Multi-GPU or Single-GPU training.

Usage:
    python training_pipeline_dask.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import config as cr

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# v16.2 Params (Aggressive / Alpha Hunting)
XGB_PARAMS = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',       # Dask works best with 'hist'
    'device': 'cuda',            # GPU Acceleration
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
# 1. DASK DATA LOADER
# ==============================================================================
def get_file_list(strategy_type):
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    pattern = f"training_{strategy_type}_*.parquet"
    files = sorted(list(path_enh.glob(pattern)))
    return [str(f) for f in files] # Dask prefers string paths

def load_dask_data(files, purge_mode):
    """
    Lazy loads data using Dask. No memory is consumed until compute() or train().
    """
    # 1. Read Parquet (Lazy)
    # Dask reads the metadata first, so this is instant.
    ddf = dd.read_parquet(files)
    
    # 2. Filter (Lazy)
    if purge_mode:
        ddf = ddf[ddf['target_multiclass'] != 1]
        
    # 3. Define Target (Lazy)
    # Convert 2 -> 1.0, else 0.0
    y = (ddf['target_multiclass'] == 2).astype('float32')
    
    # 4. Drop Columns (Lazy)
    # Identify columns to drop based on naming convention
    # Note: Dask needs to know columns upfront, so we check the meta
    all_cols = ddf.columns
    cols_to_drop = [
        c for c in all_cols 
        if c.endswith('_drop') or c.startswith('target_') or c in ['ts', 'trade_id']
    ]
    
    X = ddf.drop(columns=cols_to_drop)
    
    # Cast to float32 for GPU efficiency
    X = X.astype('float32')
    
    return X, y

# ==============================================================================
# 2. WALK-FORWARD ENGINE (DASK)
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
        
        # Extract Month string for logging
        test_month = Path(test_file).stem.split('_')[-1]
        train_start = Path(train_files[0]).stem.split('_')[-1]
        train_end = Path(train_files[-1]).stem.split('_')[-1]
        
        print(f"[{mode_name}] Round {i}: Train {train_start}-{train_end} -> Test {test_month}")
        
        # --- DASK LOADING ---
        # This creates the Computation Graph, it doesn't load RAM yet.
        X_train, y_train = load_dask_data(train_files, purge_mode)
        X_test, y_test = load_dask_data([test_file], purge_mode)
        
        # --- DASK TRAINING ---
        # dask_xgboost handles the transfer to GPU automatically
        # It will chunk the data on CPU and feed it to the GPU Device
        dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
        dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)
        
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
        
        # --- PREDICTION ---
        # We use the model to predict on the dtest matrix
        # Dask returns a Dask Series/Array, we compute() to bring it to local RAM for scoring
        y_prob_dask = xgb.dask.predict(client, model, dtest)
        y_prob = y_prob_dask.compute() # Bring to CPU RAM
        
        # Get Truth (Compute validation set labels to CPU RAM)
        y_true = y_test.compute()
        
        # --- SCORING ---
        y_pred = (y_prob > 0.52).astype(int)
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.5
            
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
        
        # Explicit cleanup to help Dask scheduler
        del dtrain, dtest, output, model, X_train, y_train
        gc.collect()

    # Save CSV
    out_csv = f"audit_{strategy_type}_{mode_suffix}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)

# ==============================================================================
# 3. ORCHESTRATOR
# ==============================================================================
def run_both_modes():
    print("Initializing Dask Local Cluster...")
    # This sets up the 'Virtual Cluster' on your machine.
    # n_workers: How many parallel processes (Recommended: # Physical Cores - 2)
    # threads_per_worker: Keep low for XGBoost interactions
    # memory_limit: Limits per worker to prevent OOM
    cluster = LocalCluster(
        n_workers=4,          # Adjust based on your CPU cores (e.g., 4 or 6)
        threads_per_worker=1, 
        memory_limit='12GB'   # 12GB * 4 = 48GB Total (Leaves room for OS)
    )
    client = Client(cluster)
    print(f"Dask Dashboard available at: {client.dashboard_link}")
    
    try:
        # Run Curves
        run_walk_forward(client, "curves", purge_mode=True)
        run_walk_forward(client, "curves", purge_mode=False)
        
        # Run Flys
        run_walk_forward(client, "flys", purge_mode=True)
        run_walk_forward(client, "flys", purge_mode=False)
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    # Dask MUST be run inside main block on Windows
    run_both_modes()
