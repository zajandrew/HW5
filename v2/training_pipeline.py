"""
training_pipeline.py (v19.0 - Robust Walk-Forward with Feature Diet)

Role:
1. Implements "Rich Net / Lean Legs" loading logic (Filters columns on read).
2. Uses Robust "Financial" XGBoost settings (Depth 4, Min Child 250).
3. Dynamically calculates Class Imbalance (scale_pos_weight) for each window.
4. Supports Purge Mode (removing weak wins from training).

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

# Windows GPU DLL Fix
try:
    ctypes.WinDLL("nvrtc-builtins64_129.dll")
except: pass

# ROBUST FINANCIAL SETTINGS
# High min_child_weight and low max_depth prevent overfitting specific noise events.
BASE_XGB_PARAMS = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',          # AUC is better for ranking quality than logloss
    'tree_method': 'hist',
    'device': 'cuda',
    'max_bin': 256,
    
    # --- ROBUSTNESS ---
    'learning_rate': 0.05,         # Higher than 0.01 for faster research iteration
    'max_depth': 4,                # Shallow trees force broad theme learning
    'min_child_weight': 250,       # CRITICAL: Requires ~250 samples to make a decision rule
    
    # --- RANDOMNESS ---
    'gamma': 0.1,
    'subsample': 0.70,
    'colsample_bytree': 0.40,      # Force model to look at diverse features
    
    # --- REGULARIZATION ---
    'reg_lambda': 1.0,             # L2
    'reg_alpha': 0.5,              # L1 (Lasso) helps zero out bad features
    
    # 'scale_pos_weight': SET DYNAMICALLY IN LOOP
}

TRAIN_WINDOW_MONTHS = 18
NUM_ROUNDS = 1000                  # Lowered to match LR 0.05
EARLY_STOPPING = 50

# ==============================================================================
# 1. SMART ITERATOR (With Feature Diet)
# ==============================================================================
class ParquetIterator(xgb.DataIter):
    """
    Loads Parquet files one-by-one with "Rich Net / Lean Legs" filtering.
    """
    def __init__(self, file_paths, purge_mode):
        self.file_paths = file_paths
        self.purge_mode = purge_mode
        self.it = 0
        
        # --- FEATURE DIET LOGIC ---
        # We determine the "Safe" columns once.
        # Rule: Keep all NET/MACRO. Drop noisy variations of L1/L2/L3.
        
        # Blocklist: Leg-level features we don't need (Redundant/Noisy)
        # We keep 'z_comb' and 'total_drift' as the "Anchors" for the legs.
        LEG_BLOCKLIST = ['slope', 'accel', 'z_pca', 'z_spline', 'carry', 'roll', 
                         'z_local', 'rng_pos', 'audit']

        try:
            # Read schema from first file
            schema = pq.read_schema(file_paths[0])
            all_cols = schema.names
            
            self.load_cols = []
            for c in all_cols:
                # 1. Always keep Target
                if c == 'target_multiclass':
                    self.load_cols.append(c)
                    continue
                
                # 2. Drop Administrative
                if c.endswith('_drop') or c.startswith('target_') or c in ['ts', 'trade_id']:
                    continue
                
                # 3. LEAN LEGS FILTER
                # If column belongs to a Leg (L1/L2/L3)
                if c.startswith(('L1_', 'L2_', 'L3_')):
                    # If it contains a blocked keyword, skip it
                    if any(b in c for b in LEG_BLOCKLIST):
                        continue
                    
                self.load_cols.append(c)
                
            # print(f"   [Diet] Selected {len(self.load_cols)} columns (dropped {len(all_cols) - len(self.load_cols)})")
            
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
            df = pd.read_parquet(file, columns=self.load_cols)
            
            # --- PURGE LOGIC ---
            if self.purge_mode:
                # Remove Class 1 (Weak/Drift Wins)
                df = df[df['target_multiclass'] != 1]
            
            # Binary Target (2 = Alpha Win, Else = 0)
            y = (df['target_multiclass'] == 2).values.astype(np.float32)
            
            X = df.drop(columns=['target_multiclass'])
            
            input_data(data=X.values.astype(np.float32), 
                       label=y,
                       feature_names=X.columns.tolist(),
                       feature_types=['float'] * len(X.columns))
            
            return 1
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
            return 1 

    def reset(self):
        self.it = 0

def get_file_list(strategy_type):
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    pattern = f"training_{strategy_type}_*.parquet"
    files = sorted(list(path_enh.glob(pattern)))
    return files

# ==============================================================================
# 2. HELPER: DYNAMIC BALANCE CALC
# ==============================================================================
def get_window_balance(files, purge_mode):
    """
    Quickly scans the targets of the training window to calculate scale_pos_weight.
    Avoids lookahead bias by only scanning the current 'train_files'.
    """
    total_pos = 0
    total_neg = 0
    
    # We only need the target column, very fast load
    for f in files:
        try:
            df = pd.read_parquet(f, columns=['target_multiclass'])
            if purge_mode:
                df = df[df['target_multiclass'] != 1]
            
            n_pos = (df['target_multiclass'] == 2).sum()
            n_total = len(df)
            n_neg = n_total - n_pos
            
            total_pos += n_pos
            total_neg += n_neg
        except: pass
        
    if total_pos == 0: return 1.0
    return total_neg / total_pos

# ==============================================================================
# 3. WALK-FORWARD ENGINE
# ==============================================================================
def run_walk_forward(strategy_type, purge_mode):
    files = get_file_list(strategy_type)
    if len(files) < (TRAIN_WINDOW_MONTHS + 1):
        print(f"[{strategy_type}] Not enough data (Need {TRAIN_WINDOW_MONTHS}+ months).")
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
        
        # 1. Calculate Dynamic Balance for THIS window
        # This ensures we adapt if the market gets harder/easier over time
        balance_ratio = get_window_balance(train_files, purge_mode)
        
        current_params = BASE_XGB_PARAMS.copy()
        current_params['scale_pos_weight'] = balance_ratio
        
        print(f"[{mode_name}] Round {i}: Train {train_months[0]}-{train_months[-1]} -> Test {test_month} (Ratio: {balance_ratio:.1f})")
        
        # 2. Iterators
        dtrain = xgb.QuantileDMatrix(ParquetIterator(train_files, purge_mode))
        dtest  = xgb.QuantileDMatrix(ParquetIterator([test_file], purge_mode))
        
        # 3. Train
        model = xgb.train(
            params=current_params,
            dtrain=dtrain,
            num_boost_round=NUM_ROUNDS,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=False
        )
        
        # 4. Score
        df_audit = pd.read_parquet(test_file, columns=['target_multiclass'])
        if purge_mode:
             df_audit = df_audit[df_audit['target_multiclass'] != 1]

        y_true = (df_audit['target_multiclass'] == 2).values.astype(int)
        y_prob = model.predict(dtest)
        
        # Simple threshold check (Can optimize this later)
        y_pred = (y_prob > 0.50).astype(int)
        
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
            'auc': auc,
            'precision': prec,
            'recall': rec,
            'best_iter': model.best_iteration,
            'pos_ratio': balance_ratio
        })
        
        del dtrain, dtest, model, df_audit, y_true, y_prob
        gc.collect()

    out_csv = f"audit_{strategy_type}_{mode_suffix}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)

# ==============================================================================
# 4. ORCHESTRATOR
# ==============================================================================
def run_both_modes():
    try:
        # Run Curves
        run_walk_forward("curves", purge_mode=True)
        # run_walk_forward("curves", purge_mode=False) # Optional: Run standard if needed
        
        # Run Flys
        run_walk_forward("flys", purge_mode=True)
        
    except KeyboardInterrupt:
        print("\nRun cancelled by user.")

if __name__ == "__main__":
    run_both_modes()