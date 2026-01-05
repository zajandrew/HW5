"""
data_converter.py

Role:
1. Ingests monthly training_*.parquet files.
2. Separates Features (X) from Labels (y) and Metadata.
3. Converts data into XGBoost Binary Buffers (.buffer).
4. enables Out-of-Core training for datasets larger than RAM.

Output:
- curves_train.buffer
- flys_train.buffer
"""

import os
import gc
import sys
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import config as cr

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Define columns to EXCLUDE from features (Leakage Prevention)
# We keep 'target_label' as y, and everything else is X.
# We explicitly drop audit columns that contain the answer or future data.
DROP_COLS = [
    'ts', 'trade_id', 'target_label', 
    'aux_price_pnl', 'aux_carry_pnl', 'aux_total_pnl', 
    'aux_entry_rate', 'aux_exit_rate', 'aux_exit_idx', 'aux_exit_ts',
    'L1_entry_rate', 'L2_entry_rate', 'L3_entry_rate',
    'L1_exit_rate', 'L2_exit_rate', 'L3_exit_rate',
    'meta_t1', 'meta_t2', 'meta_t3', 'meta_dist'
]

def convert_to_binary(strategy_type):
    """
    Reads all parquet files for a strategy type (curves/flys),
    creates a DMatrix, and saves it as a binary buffer.
    """
    path_enh = Path(getattr(cr, "PATH_ENH", "."))
    pattern = f"training_{strategy_type}_*.parquet"
    files = sorted(list(path_enh.glob(pattern)))
    
    if not files:
        print(f"[{strategy_type}] No files found matching {pattern}")
        return

    output_buffer = path_enh / f"{strategy_type}_full.buffer"
    print(f"[{strategy_type}] Found {len(files)} files. Converting to {output_buffer.name}...")

    # We cannot load all DFs at once. We must iterate.
    # XGBoost does not natively support "appending" to a buffer file easily from Python API 
    # without using external memory text formats (LibSVM). 
    # However, constructing a single DMatrix from a concatenation of files 
    # is often efficient enough if we do it in chunks or use QuantileDMatrix.
    
    # STRATEGY: 
    # For robust out-of-core, we ideally use the text format or an iterator.
    # But simpler for "Large but not Petabyte" data: 
    # Load all X/y into Numpy arrays (if 64GB RAM allows) or concatenate via DMatrix.
    # If strictly RAM constrained, we use the iterator method.
    
    # Below is the Iterator Method (True Low-Memory Approach).
    
    class ParquetIterator(xgb.DataIter):
        def __init__(self, file_list):
            self.file_list = file_list
            self.it = 0
            super().__init__()

        def next(self, input_data):
            if self.it == len(self.file_list):
                return 0 # End of iteration
            
            # Load One Month
            file_path = self.file_list[self.it]
            try:
                df = pd.read_parquet(file_path)
                
                # Split X and y
                # 1. Label
                y_batch = df['target_label'].values.astype(np.float32)
                
                # 2. Features (Drop audit/meta columns)
                # Ensure we only drop columns that actually exist
                existing_drop = [c for c in DROP_COLS if c in df.columns]
                X_batch = df.drop(columns=existing_drop)
                
                # Force float32 for efficiency
                feature_names = X_batch.columns.tolist()
                feature_types = ['float'] * len(feature_names)
                X_data = X_batch.values.astype(np.float32)

                input_data(data=X_data, label=y_batch,
                           feature_names=feature_names, feature_types=feature_types)
                
                self.it += 1
                return 1 # Continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                self.it += 1
                return 1 # Skip but continue?

        def reset(self):
            self.it = 0

    # Initialize Iterator
    it = ParquetIterator(files)
    
    # Create DMatrix from Iterator
    # QuantileDMatrix is faster and memory efficient for large datasets
    print(f"   -> Building QuantileDMatrix (This may take time)...")
    dtrain = xgb.QuantileDMatrix(it)
    
    # Save to Binary
    print(f"   -> Saving binary buffer...")
    dtrain.save_binary(output_buffer)
    print(f"[{strategy_type}] Done. Binary size: {output_buffer.stat().st_size / 1e6:.2f} MB")
    
    # Cleanup
    del dtrain, it
    gc.collect()

if __name__ == "__main__":
    # Convert Curves
    convert_to_binary("curves")
    
    # Convert Flys
    convert_to_binary("flys")
