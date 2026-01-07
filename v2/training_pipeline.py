"""
training_pipeline.py (v19.0 - Simple Iterator with Feature Diet)

Role:
1. Reverts to simple ParquetIterator (No PyArrow Streaming/Dask).
2. Implements "Feature Diet": Drops L1/L2/L3 raw columns at load time.
3. Supports 18-Month Walk-Forward & Purge/Standard Modes.

Usage:
    python training_pipeline.py
"""

import os
import gc
import ctypes
import numpy as np
import pandas as pd
import xgboost as xgb
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import config as cr

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Windows GPU DLL Fix (Kept silent to prevent import crashes)
try:
    ctypes.WinDLL("nvrtc-builtins64_129.dll")
except: pass

XGB_PARAMS = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'device': 'cuda',  # XGBoost will find your A4000 automatically
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
# 1. SIMPLE DATA ITERATOR (With Column Pruning)
# ==============================================================================
class ParquetIterator(xgb.DataIter):
    """
    Standard Iterator (Loads 1 file at a time) but with aggressive column pruning.
    """
    def __init__(self, file_paths, purge_mode):
        self.file_paths = file_paths
        self.purge_mode = purge_mode
        self.it = 0
        
        # --- FEATURE DIET ---
        # We determine which columns to load ONCE.
        # This prevents loading 'L1_', 'audit_', etc. into RAM.
        try:
            # Read schema only (instant)
            schema = pq.read_schema(file_paths[0])
            all_cols = schema.names
            
            self.load_cols = []
            for c in all_cols:
                # 1. Always keep Label
                if c == 'target_multiclass':
                    self.load_cols.append(c)
                    continue
                
                # 2. Drop Administrative
                if c.endswith('_drop') or c.startswith('target_') or c in ['ts', 'trade_id']:
                    continue
                
                # 3. Drop Raw Legs (L1, L2, L3) -> Keep only NET_ features
                if c.startswith(('L1_', 'L2_', 'L3_')):
                    continue
                    
                self.load_cols.append(c)
                
            # print(f"   [Diet] Loading {len(self.load_cols)} columns (dropped {len(all_cols) - len(self.load_cols)})")
            
        except Exception as e:
            print(f"   [Warn] Schema read failed, loading all: {e}")
            self.load_cols = None

        super().__init__()

    def next(self, input_data):
        if self.it == len(self.file_paths):
            return 0 # Stop
        
        file = self.file_paths[self.it]
        self.it += 1
        
        try:
            # Load specific columns only (The optimization)
            df = pd.read_parquet(file, columns=self.load_cols)
            
            # --- PURGE LOGIC ---
            if self.purge_mode:
                # Remove Class 1 (Weak Wins)
                df = df[df['target_multiclass'] != 1]
            
            # Create Binary Target (2 = Win, Else = Loss)
            y = (df['target_multiclass'] == 2).values.astype(np.float32)
            
            # Drop Target from Features
            X = df.drop(columns=['target_multiclass'])
            
            # Pass to XGBoost
            input_data(data=X.values.astype(np.float32), 
                       label=y,
                       feature_names=X.columns.tolist(),
                       feature_types=['float'] * len(X.columns))
            
            return 1
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
            return 1 # Skip file but keep going

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
        
        test_month = test_file.stem.split('_')[-1]
        train_months = [f.stem.split('_')[-1] for f in train_files]
        
        print(f"[{mode_name}] Round {i}: Train {train_months[0]}-{train_months[-1]} -> Test {test_month}")
        
        # 1. Train Iterator (18 months)
        dtrain = xgb.QuantileDMatrix(ParquetIterator(train_files, purge_mode))
        
        # 2. Test Iterator (1 month)
        dtest  = xgb.QuantileDMatrix(ParquetIterator([test_file], purge_mode))
        
        # 3. Train
        model = xgb.train(
            params=XGB_PARAMS,
            dtrain=dtrain,
            num_boost_round=NUM_ROUNDS,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=False
        )
        
        # 4. Score
        # Minimal load for audit
        df_audit = pd.read_parquet(test_file, columns=['target_multiclass'])
        
        if purge_mode:
             df_audit = df_audit[df_audit['target_multiclass'] != 1]

        y_true = (df_audit['target_multiclass'] == 2).values.astype(int)
        y_prob = model.predict(dtest)
        
        y_pred = (y_prob > 0.52).astype(int)
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        
        print(f"   -> AUC: {auc:.3f} | Prec: {prec:.2%} | Rec: {rec:.2%} | Trees: {model.best_iteration}")
        
        # 5. Save
        model_sub = f"{strategy_type}_{mode_suffix}"
        model_dir = Path("models") / model_sub
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(model_dir / f"model_{test_month}.json")
        
        results.append({
            'test_month': test_month,
            'train_start': train_months[0],
            'train_end': train_months[-1],
            'auc': auc,
            'precision': prec,
            'recall': rec,
            'best_iter': model.best_iteration
        })
        
        del dtrain, dtest, model, df_audit, y_true, y_prob
        gc.collect()

    out_csv = f"audit_{strategy_type}_{mode_suffix}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)

# ==============================================================================
# 3. ORCHESTRATOR
# ==============================================================================
def run_both_modes():
    try:
        # Curves
        run_walk_forward("curves", purge_mode=True)
        run_walk_forward("curves", purge_mode=False)
        
        # Flys (With column pruning, should handle 18m memory load)
        run_walk_forward("flys", purge_mode=True)
        run_walk_forward("flys", purge_mode=False)
        
    except KeyboardInterrupt:
        print("\nRun cancelled by user.")

if __name__ == "__main__":
    run_both_modes()
