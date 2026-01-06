"""
training_pipeline.py (v16.1 - Purge vs Standard A/B Test)

Role:
1. Implements Out-of-Core Training using ParquetIterator (Low RAM).
2. Performs Rolling Walk-Forward Validation.
3. Supports "Purge Mode" (Drop Class 1) vs "Standard Mode" (Class 1 = 0).
4. Outputs distinct CSV audits and Model files for comparison.

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

# v16.1 HYPERPARAMETERS (Optimized for Alpha Hunting & Purged Data)
# These are looser than v15 to prevent the model from starving on smaller (purged) datasets.
XGB_PARAMS = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',       # CRITICAL for speed/memory
    'device': 'cuda',            # GPU Acceleration
    'max_bin': 256,
    
    # --- Alpha Hunting Config ---
    'learning_rate': 0.01,       # Fast enough to capture signal, slow enough to generalize
    'max_depth': 6,              # Depth 6 is the "sweet spot" for tabular finance
    'min_child_weight': 1,       # Low threshold allows splits on sparse "Fly" data
    'gamma': 0.0,                # No minimum loss reduction (let trees grow)
    
    # --- Robustness ---
    'subsample': 0.70,           # Row sampling to prevent memorization
    'colsample_bytree': 0.50,    # Feature sampling to force "Plan B" logic
    'reg_lambda': 1.0,           # Lower L2 tax to encourage higher Recall
    'scale_pos_weight': 1.0      
}

TRAIN_WINDOW_MONTHS = 6   # Lookback period
NUM_ROUNDS = 5000         # High ceiling (let Early Stopping decide)
EARLY_STOPPING = 200      # Patience

# ==============================================================================
# 1. DATA ITERATOR (Smart Purging)
# ==============================================================================
class ParquetIterator(xgb.DataIter):
    """
    Streams X and y from Parquet files.
    Dynamically applies "The Purge" logic based on initialization flags.
    """
    def __init__(self, file_paths, purge_mode):
        self.file_paths = file_paths
        self.purge_mode = purge_mode  # True = Drop Class 1, False = Keep Class 1 as 0
        self.it = 0
        super().__init__()

    def next(self, input_data):
        if self.it == len(self.file_paths):
            return 0 # Stop
        
        file = self.file_paths[self.it]
        try:
            df = pd.read_parquet(file)
            
            # --- SURGICAL LOGIC: Purge vs. Keep ---
            if self.purge_mode:
                # OPTION A: The Purge
                # Remove "Weak Wins" (Class 1) entirely.
                # Train only on Alpha (2) vs Loss (0).
                mask = df['target_multiclass'] != 1
                df = df.loc[mask].copy()
            else:
                # OPTION B: Standard
                # Keep Class 1, but it will naturally become 0 below
                # because we only map Class 2 to 1.
                pass 

            # Create Binary Target (Alpha vs Rest)
            y = (df['target_multiclass'] == 2).values.astype(np.float32)
            
            # Dynamic Drops: Remove anything future-looking or target-related
            # We assume features do NOT end with '_drop' and are NOT 'target_*'
            cols_to_drop = [
                c for c in df.columns 
                if c.endswith('_drop') or c.startswith('target_') or c in ['ts', 'trade_id']
            ]
            X = df.drop(columns=cols_to_drop)
            
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
    # Looks for 'training_curves_2301.parquet' etc.
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
        
        # 2. Prepare Iterators (Injecting the purge_mode flag)
        dtrain = xgb.QuantileDMatrix(ParquetIterator(train_files, purge_mode=purge_mode))
        # Note: We usually test on FULL data (purge_mode=False) to see real world performance,
        # OR we test on Purged data to see model stability. 
        # Standard practice: Train on Purged, Test on Reality (Standard).
        # HOWEVER, for apples-to-apples validation metrics, we align test processing with train for now.
        # If you want to test on "Reality", set purge_mode=False for dtest always.
        # Let's stick to the mode for consistency of metrics first.
        dtest  = xgb.QuantileDMatrix(ParquetIterator([test_file], purge_mode=purge_mode))
        
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
        # We need to manually load y_true using the same logic as the iterator
        df_test = pd.read_parquet(test_file, columns=['target_multiclass'])
        if purge_mode:
            df_test = df_test[df_test['target_multiclass'] != 1]
            
        y_true = (df_test['target_multiclass'] == 2).values.astype(int)
        y_prob = model.predict(dtest)
        
        # Dynamic Threshold or Fixed? Fixed 0.52 for now
        y_pred = (y_prob > 0.52).astype(int)
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        # Handle single-class edge case in AUC
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.5
        
        print(f"   -> AUC: {auc:.3f} | Prec: {prec:.2%} | Rec: {rec:.2%} | Trees: {model.best_iteration}")
        
        # 5. Save Model (With distinct subfolder)
        # Folder: models/curves_purged/model_2301.json
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
    print("Starting Global A/B Test Run...")
    
    # 1. Run Curves (Purged vs Standard)
    run_walk_forward("curves", purge_mode=True)
    run_walk_forward("curves", purge_mode=False)
    
    # 2. Run Flys (Purged vs Standard)
    run_walk_forward("flys", purge_mode=True)
    run_walk_forward("flys", purge_mode=False)
    
    print("\nAll experiments completed.")

if __name__ == "__main__":
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU detected. Switching to CPU.")
            XGB_PARAMS['device'] = 'cpu'
    except:
        XGB_PARAMS['device'] = 'cpu'

    run_both_modes()