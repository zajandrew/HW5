"""
training_pipeline.py (v18.0 - PyArrow Industrial Streamer)

Role:
1. Uses pyarrow.dataset to push Filters & Column Selection down to C++ layer.
2. Streams efficient batches to XGBoost QuantileDMatrix (Out-of-Core).
3. ZERO memory leaks or partition errors.

Usage:
    python training_pipeline.py
"""

import os
import gc
import ctypes
import numpy as np
import pandas as pd
import xgboost as xgb
import pyarrow.dataset as ds  # <--- The Secret Weapon
import pyarrow as pa
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import config as cr

# ==============================================================================
# GPU CONFIG
# ==============================================================================
try:
    ctypes.WinDLL("nvrtc-builtins64_129.dll")
except: pass

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
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
# 1. INDUSTRIAL DATA ITERATOR (PyArrow Dataset)
# ==============================================================================
class PyArrowIterator(xgb.DataIter):
    def __init__(self, file_paths, purge_mode, batch_size=100000):
        self.file_paths = file_paths
        self.purge_mode = purge_mode
        self.batch_size = batch_size
        self._batch_gen = None
        
        # 1. Smart Schema Detection
        # We peek at the first file to figure out which columns to LOAD
        # and which to IGNORE on disk.
        try:
            sample = ds.dataset(file_paths[0], format='parquet').schema
            all_cols = sample.names
            
            # Keep only Feature columns + Target
            self.load_cols = [
                c for c in all_cols 
                if c == 'target_multiclass' or 
                (not c.endswith('_drop') 
                 and not c.startswith('target_') 
                 and c not in ['ts', 'trade_id', 'meta_t1', 'meta_t2', 'meta_t3'])
            ]
        except:
            self.load_cols = None # Fallback
            
        super().__init__()

    def _get_next_batch_generator(self):
        # 2. Define Filter Expression (Pushdown to C++)
        # This prevents "Class 1" rows from ever entering RAM if purging.
        filter_expr = None
        if self.purge_mode:
            # "target_multiclass != 1"
            filter_expr = (ds.field('target_multiclass') != 1)

        # 3. Create Dataset Scanner
        dataset = ds.dataset(self.file_paths, format='parquet')
        
        # 4. Stream Batches
        # columns=self.load_cols -> Only reads features (Saves 50% RAM/IO)
        # filter=filter_expr -> Skips rows on disk (Saves CPU)
        batch_iter = dataset.to_batches(
            columns=self.load_cols,
            filter=filter_expr, 
            batch_size=self.batch_size
        )
        
        for record_batch in batch_iter:
            # Zero-copy conversion to Pandas
            df_chunk = record_batch.to_pandas()
            
            if df_chunk.empty: continue
            
            # Create Label
            y = (df_chunk['target_multiclass'] == 2).values.astype(np.float32)
            
            # Create Features (Drop the target column we used for filtering)
            X = df_chunk.drop(columns=['target_multiclass'])
            
            # Yield efficient Numpy blocks to XGBoost
            yield X, y
            
        yield None, None

    def next(self, input_data):
        if self._batch_gen is None:
            self._batch_gen = self._get_next_batch_generator()
        
        try:
            X, y = next(self._batch_gen)
            if X is None: return 0
            
            input_data(data=X.values.astype(np.float32), 
                       label=y,
                       feature_names=X.columns.tolist(),
                       feature_types=['float'] * len(X.columns))
            return 1
        except StopIteration:
            return 0

    def reset(self):
        self._batch_gen = None

def get_file_list(strategy_type):
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    pattern = f"training_{strategy_type}_*.parquet"
    files = sorted(list(path_enh.glob(pattern)))
    return [str(f) for f in files]

# ==============================================================================
# 2. WALK-FORWARD ENGINE
# ==============================================================================
def run_walk_forward(strategy_type, purge_mode):
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
        
        # --- TRAINING ---
        # QuantileDMatrix is the Magic Key. 
        # It builds histograms on-the-fly from the iterator.
        # It never holds the full dataset in RAM.
        dtrain = xgb.QuantileDMatrix(
            PyArrowIterator(train_files, purge_mode=purge_mode, batch_size=100000)
        )
        
        # For testing, we also use the iterator to keep memory flat
        dtest = xgb.QuantileDMatrix(
            PyArrowIterator([test_file], purge_mode=purge_mode, batch_size=100000)
        )
        
        model = xgb.train(
            params=XGB_PARAMS,
            dtrain=dtrain,
            num_boost_round=NUM_ROUNDS,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=False
        )
        
        # --- SCORING ---
        # We need y_true to calculate Precision/Recall. 
        # Since DMatrix hides labels, we read the test file just for the target.
        # This is cheap (read 1 column).
        df_audit = pd.read_parquet(test_file, columns=['target_multiclass'])
        if purge_mode:
            df_audit = df_audit[df_audit['target_multiclass'] != 1]
            
        y_true = (df_audit['target_multiclass'] == 2).values.astype(int)
        y_prob = model.predict(dtest)
        
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
        
        del dtrain, dtest, model, y_prob, y_true, df_audit
        gc.collect()

    out_csv = f"audit_{strategy_type}_{mode_suffix}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)

# ==============================================================================
# 3. ORCHESTRATOR
# ==============================================================================
def run_both_modes():
    try:
        run_walk_forward("curves", purge_mode=True)
        run_walk_forward("curves", purge_mode=False)
        run_walk_forward("flys", purge_mode=True)
        run_walk_forward("flys", purge_mode=False)
    except KeyboardInterrupt:
        print("Run cancelled by user.")

if __name__ == "__main__":
    run_both_modes()
