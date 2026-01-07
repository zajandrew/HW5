"""
training_pipeline.py (v16.2 - Batch Streaming & A/B Test)

Role:
1. Implements Out-of-Core Training using PyArrow Batch Streaming (Low RAM).
2. Performs Rolling Walk-Forward Validation (18-Month Window).
3. Supports "Purge Mode" (Drop Class 1) vs "Standard Mode".
4. Outputs distinct CSV audits and Model files.

Usage:
    python training_pipeline.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import gc
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
import pyarrow.parquet as pq  # <--- NEW: For batch streaming
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import config as cr
import os
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# v16.1 HYPERPARAMETERS (Aggressive / Alpha Hunting)
# Optimized for the Purged method and sparse Fly data
XGB_PARAMS = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',       # CRITICAL for speed/memory/QuantileDMatrix
    'device': 'cuda',            # GPU Acceleration
    'max_bin': 256,
    
    # --- Alpha Hunting Config ---
    'learning_rate': 0.01,       # Fast enough to capture signal
    'max_depth': 6,              # Sweet spot for tabular generalization
    'min_child_weight': 1,       # Low threshold allows splits on sparse data
    'gamma': 0.0,                # No minimum loss reduction (let trees grow)
    
    # --- Robustness ---
    'subsample': 0.70,           # Row sampling
    'colsample_bytree': 0.50,    # Feature sampling (Force Plan B features)
    'reg_lambda': 1.0,           # Lower L2 tax (Encourage Recall)
    'scale_pos_weight': 1.0      
}

# 18-Month Window to prevent "0 Tree" Starvation
TRAIN_WINDOW_MONTHS = 18  
NUM_ROUNDS = 5000         
EARLY_STOPPING = 200      

# ==============================================================================
# 1. DATA ITERATOR (Low-Memory Batch Streaming)
# ==============================================================================
class PyArrowIterator(xgb.DataIter):
    """
    Streams small batches (chunks) from Parquet files to prevent RAM spikes.
    Uses PyArrow to read row-groups instead of loading full Pandas DataFrames.
    """
    def __init__(self, file_paths, purge_mode, batch_size=50000):
        self.file_paths = file_paths
        self.purge_mode = purge_mode # True = Drop Class 1, False = Keep Class 1 as 0
        self.batch_size = batch_size
        self._file_idx = 0
        self._batch_gen = None
        super().__init__()

    def _get_next_batch_generator(self):
        """Opens the next file and yields small filtered batches."""
        while self._file_idx < len(self.file_paths):
            file_path = self.file_paths[self._file_idx]
            self._file_idx += 1
            
            try:
                # Open Parquet file stream
                parquet_file = pq.ParquetFile(file_path)
                
                # Iterate over batches to keep RAM usage low (<2GB)
                for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                    df_chunk = batch.to_pandas()
                    
                    # --- SURGICAL LOGIC: Purge vs. Keep ---
                    if self.purge_mode:
                        # OPTION A: The Purge (Remove Class 1)
                        mask = df_chunk['target_multiclass'] != 1
                        df_chunk = df_chunk.loc[mask].copy()
                        if df_chunk.empty: continue
                    else:
                        # OPTION B: Standard (Keep Class 1, treat as 0)
                        pass

                    # Create Binary Target (Alpha vs Rest)
                    # Class 2 is the ONLY target (1.0), everything else is 0.0
                    y = (df_chunk['target_multiclass'] == 2).values.astype(np.float32)
                    
                    # Dynamic Drops
                    cols_to_drop = [
                        c for c in df_chunk.columns 
                        if c.endswith('_drop') or c.startswith('target_') or c in ['ts', 'trade_id']
                    ]
                    X = df_chunk.drop(columns=cols_to_drop)
                    
                    yield X, y
                    
            except Exception as e:
                print(f"[Warn] Error streaming {file_path}: {e}")
                continue
                
        yield None, None # Signal End of Data

    def next(self, input_data):
        # Initialize generator if first run
        if self._batch_gen is None:
            self._batch_gen = self._get_next_batch_generator()
            
        try:
            X, y = next(self._batch_gen)
            
            # End of all files
            if X is None:
                return 0 
            
            # Pass data to XGBoost
            # Note: We cast to float32 to ensure compatibility with GPU Hist
            input_data(data=X.values.astype(np.float32), 
                       label=y,
                       feature_names=X.columns.tolist(),
                       feature_types=['float'] * len(X.columns))
            return 1
            
        except StopIteration:
            return 0

    def reset(self):
        self._file_idx = 0
        self._batch_gen = None

def get_file_list(strategy_type):
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    pattern = f"training_{strategy_type}_*.parquet"
    files = sorted(list(path_enh.glob(pattern)))
    return files

# ==============================================================================
# 2. WALK-FORWARD ENGINE
# ==============================================================================
def run_walk_forward(strategy_type, purge_mode):
    """
    Runs the full history simulation for a specific strategy and mode.
    """
    files = get_file_list(strategy_type)
    if len(files) < (TRAIN_WINDOW_MONTHS + 1):
        print(f"[{strategy_type}] Not enough data for {TRAIN_WINDOW_MONTHS}m window.")
        return

    # Define Labels
    mode_name = "PURGED" if purge_mode else "STANDARD"
    mode_suffix = "purged" if purge_mode else "standard"
    
    print(f"\n=== Starting Walk-Forward: {strategy_type.upper()} [{mode_name}] ===")
    
    results = []
    
    # Loop: Start at [Window] and move forward
    for i in range(TRAIN_WINDOW_MONTHS, len(files)):
        
        # 1. Define Slices
        test_file = files[i]
        train_files = files[i - TRAIN_WINDOW_MONTHS : i]
        
        test_month = test_file.stem.split('_')[-1] # Extract YYMM
        train_months = [f.stem.split('_')[-1] for f in train_files]
        
        print(f"[{mode_name}] Round {i}: Train {train_months[0]}-{train_months[-1]} -> Test {test_month}")
        
        # 2. Prepare Iterators (Batched for Low RAM)
        # 50k batch size is safe for 64GB RAM even with 3-leg Flys
        dtrain = xgb.QuantileDMatrix(PyArrowIterator(train_files, purge_mode=purge_mode, batch_size=50000))
        dtest  = xgb.QuantileDMatrix(PyArrowIterator([test_file], purge_mode=purge_mode, batch_size=50000))
        
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
        
        # 4. Predict & Score
        # Manually load y_true for scoring (XGBoost DMatrix hides labels)
        df_test = pd.read_parquet(test_file, columns=['target_multiclass'])
        
        if purge_mode:
            # If testing in Purge Mode, we must also drop Class 1 from validation 
            # to match the dtest/dtrain distribution for AUC calculation.
            df_test = df_test[df_test['target_multiclass'] != 1]
            
        y_true = (df_test['target_multiclass'] == 2).values.astype(int)
        y_prob = model.predict(dtest)
        
        # Fixed Threshold for now
        y_pred = (y_prob > 0.52).astype(int)
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.5
        
        print(f"   -> AUC: {auc:.3f} | Prec: {prec:.2%} | Rec: {rec:.2%} | Trees: {model.best_iteration}")
        
        # 5. Save Model
        model_sub = f"{strategy_type}_{mode_suffix}"
        model_dir = Path("models") / model_sub
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(model_dir / f"model_{test_month}.json")
        
        # 6. Log Result
        results.append({
            'test_month': test_month,
            'train_start': train_months[0],
            'train_end': train_months[-1],
            'auc': auc,
            'precision': prec,
            'recall': rec,
            'best_iter': model.best_iteration
        })
        
        # Cleanup
        del dtrain, dtest, model, df_test, y_true, y_prob
        gc.collect()

    # Save Audit CSV
    out_csv = f"audit_{strategy_type}_{mode_suffix}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Finished {strategy_type.upper()} [{mode_name}]. Saved to {out_csv}")

# ==============================================================================
# 3. ORCHESTRATOR
# ==============================================================================
def run_both_modes():
    print("Starting Global A/B Test Run (18-Month Window)...")
    
    # 1. Run Curves (Purged vs Standard)
    run_walk_forward("curves", purge_mode=True)
    run_walk_forward("curves", purge_mode=False)
    
    # 2. Run Flys (Purged vs Standard)
    # The new Batch Streamer should prevent RAM crashes here
    run_walk_forward("flys", purge_mode=True)
    run_walk_forward("flys", purge_mode=False)
    
    print("\nAll experiments completed.")

if __name__ == "__main__":
    # Fallback check
    try:
        import torch
        if not torch.cuda.is_available():
            print("No GPU detected by Torch. Switching XGBoost to CPU.")
            XGB_PARAMS['device'] = 'cpu'
    except:
        pass

    run_both_modes()
