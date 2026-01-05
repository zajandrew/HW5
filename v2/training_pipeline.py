"""
training_pipeline.py

Role:
1. Implements Out-of-Core Training using ParquetIterator (Low RAM).
2. Performs Rolling Walk-Forward Validation (e.g., Train 6 months -> Test 1 month).
3. Optimized "Anti-Overfitting" Hyperparameters.
4. Outputs monthly performance audits.

Usage:
    python training_pipeline.py
"""

import os
import gc
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import config as cr

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# LEAKAGE PROTECTION: Columns to strictly ignore
DROP_COLS = [
    'ts', 'trade_id', 'target_label', 
    'aux_price_pnl', 'aux_carry_pnl', 'aux_total_pnl', 
    'aux_entry_rate', 'aux_exit_rate', 'aux_exit_idx', 'aux_exit_ts',
    'L1_entry_rate', 'L2_entry_rate', 'L3_entry_rate',
    'L1_exit_rate', 'L2_exit_rate', 'L3_exit_rate',
    'meta_t1', 'meta_t2', 'meta_t3', 'meta_dist'
]

# TRAINING HYPERPARAMETERS (Conservative / Anti-Overfit)
XGB_PARAMS = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',       # CRITICAL for speed/memory
    'device': 'cuda',            # Change to 'cpu' if no GPU
    'max_bin': 256,
    
    # --- Anti-Overfitting Config ---
    'learning_rate': 0.05,       # Slow learning
    'max_depth': 4,              # Shallow trees (prevents memorization)
    'min_child_weight': 100,     # High threshold (needs many trades to split)
    'gamma': 0.2,                # Minimum loss reduction to split
    'subsample': 0.7,            # Row sampling
    'colsample_bytree': 0.7,     # Feature sampling
    'reg_lambda': 1.0,           # L2 Regularization
    'reg_alpha': 0.1,            # L1 Regularization
    'scale_pos_weight': 1.0      # Adjust if classes are super imbalanced
}

TRAIN_WINDOW_MONTHS = 6   # Lookback period
NUM_ROUNDS = 500          # Max trees
EARLY_STOPPING = 50       # Stop if no improvement

# ==============================================================================
# 1. DATA ITERATOR (Streaming from Disk)
# ==============================================================================
class ParquetIterator(xgb.DataIter):
    """
    Streams X and y from a list of Parquet files without loading all to RAM.
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.it = 0
        super().__init__()

    def next(self, input_data):
        if self.it == len(self.file_paths):
            return 0 # Stop
        
        file = self.file_paths[self.it]
        try:
            df = pd.read_parquet(file)
            
            # Extract Label and Features
            y = df['target_label'].values.astype(np.float32)
            
            # Drop Audit Cols
            cols_to_drop = [c for c in DROP_COLS if c in df.columns]
            X = df.drop(columns=cols_to_drop)
            
            # XGBoost needs strictly float/int (no objects)
            # Strategy Generator already ensures float32, but good to check
            X_vals = X.values.astype(np.float32)
            
            input_data(data=X_vals, label=y, 
                       feature_names=X.columns.tolist(),
                       feature_types=['float'] * len(X.columns))
            
            self.it += 1
            return 1
        except Exception as e:
            print(f"Error reading {file}: {e}")
            self.it += 1
            return 1

    def reset(self):
        self.it = 0

def get_file_list(strategy_type):
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    pattern = f"training_{strategy_type}_*.parquet"
    files = sorted(list(path_enh.glob(pattern)))
    return files

# ==============================================================================
# 2. WALK-FORWARD ENGINE
# ==============================================================================
def run_walk_forward(strategy_type):
    files = get_file_list(strategy_type)
    if len(files) < (TRAIN_WINDOW_MONTHS + 1):
        print(f"[{strategy_type}] Not enough data for {TRAIN_WINDOW_MONTHS}m window.")
        return

    print(f"\n=== Starting Walk-Forward: {strategy_type} ({len(files)} months) ===")
    
    results = []
    
    # Loop: Start at [Window] and move forward
    # i is the index of the TEST month
    for i in range(TRAIN_WINDOW_MONTHS, len(files)):
        
        # 1. Define Slices
        test_file = files[i]
        train_files = files[i - TRAIN_WINDOW_MONTHS : i]
        
        test_month = test_file.stem.split('_')[-1] # Extract YYMM
        train_months = [f.stem.split('_')[-1] for f in train_files]
        
        print(f"\nRound {i}: Train {train_months[0]}-{train_months[-1]} -> Test {test_month}")
        
        # 2. Prepare Iterators
        # Note: We use QuantileDMatrix with iterator for low memory usage
        # 'nthread' controls parallel loading
        dtrain = xgb.QuantileDMatrix(ParquetIterator(train_files))
        dtest  = xgb.QuantileDMatrix(ParquetIterator([test_file]))
        
        # 3. Train
        evals = [(dtrain, 'train'), (dtest, 'eval')]
        
        model = xgb.train(
            params=XGB_PARAMS,
            dtrain=dtrain,
            num_boost_round=NUM_ROUNDS,
            evals=evals,
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=False
        )
        
        # 4. Predict & Score (Out of Sample)
        # Note: We need to reload y_test manually to score, as DMatrix hides it
        df_test_audit = pd.read_parquet(test_file, columns=['target_label'])
        y_true = df_test_audit['target_label'].values
        y_prob = model.predict(dtest)
        y_pred = (y_prob > 0.52).astype(int) # Slight hurdle above 0.5
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        
        print(f"   -> AUC: {auc:.3f} | Prec: {prec:.2%} | Rec: {rec:.2%} | Trees: {model.best_iteration}")
        
        # 5. Save Monthly Model (Optional - Good for Analysis)
        model_dir = Path("models") / strategy_type
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(model_dir / f"model_{test_month}.json")
        
        # 6. Log Result
        results.append({
            'test_month': test_month,
            'train_start': train_months[0],
            'train_end': train_months[-1],
            'auc': auc,
            'precision': prec,
            'recall': recall,
            'best_iter': model.best_iteration
        })
        
        # Cleanup RAM
        del dtrain, dtest, model, df_test_audit, y_true, y_prob
        gc.collect()

    # Save Audit CSV
    pd.DataFrame(results).to_csv(f"audit_{strategy_type}_walkforward.csv", index=False)
    print(f"Finished {strategy_type}. Results saved.")

if __name__ == "__main__":
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU detected. Switching XGBoost to CPU mode.")
            XGB_PARAMS['device'] = 'cpu'
    except:
        XGB_PARAMS['device'] = 'cpu'

    # Run
    run_walk_forward("curves")
    run_walk_forward("flys")
