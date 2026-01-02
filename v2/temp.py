import pandas as pd
import numpy as np
from pathlib import Path
import sys

def audit_file(file_path):
    print(f"--- AUDITING: {Path(file_path).name} ---")
    df = pd.read_parquet(file_path)
    
    # Filter for numeric columns only
    df_num = df.select_dtypes(include=[np.number])
    if 'ts' in df_num.columns: df_num = df_num.drop(columns=['ts'])
    
    # OUTPUT CONTAINER
    drop_candidates = []

    # 1. CHECK FOR "STEP FUNCTIONS" (Daily Data treated as Hourly)
    print("...Scanning for Step Functions (Daily data disguised as Hourly)...")
    for col in df_num.columns:
        if 'slope' in col or 'accel' in col:
            # Check if value changes rarely (daily update on hourly file)
            change_density = (df_num[col].diff() != 0).mean()
            if 0.001 < change_density < 0.08: 
                print(f"  [STEP DETECTED] {col} (Updates ~{change_density*100:.1f}% of time)")
                drop_candidates.append(col)

    # 2. CHECK FOR REDUNDANCY (Window vs Window)
    print("...Scanning for Window Redundancy (Corr > 0.99)...")
    
    # Only check columns that look like window stats
    # We explicitly exclude base physics like 'dv01', 'rate', 'scale', 'vol'
    window_keywords = ['_slope_', '_accel_', '_zlocal_', '_mean_', '_std_', '_max_', '_min_']
    window_cols = [c for c in df_num.columns if any(k in c for k in window_keywords)]
    
    if len(window_cols) > 1:
        df_sample = df_num[window_cols].sample(n=min(10000, len(df_num)))
        corr_matrix = df_sample.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        redundant = [column for column in upper.columns if any(upper[column] > 0.99)]
        
        for col in redundant:
            print(f"  [REDUNDANT] {col}")
            drop_candidates.append(col)

    # --- FINAL OUTPUT ---
    print("\n" + "="*60)
    print("COPY THIS LIST TO PRUNE FEATURES:")
    print("="*60)
    
    # De-duplicate and format as Python list
    final_list = sorted(list(set(drop_candidates)))
    print(f"cols_to_drop = {final_list}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audit_file(sys.argv[1])
    else:
        print("Usage: python audit_features.py path/to/file.parquet")
