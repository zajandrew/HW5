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
    
    results = {
        'suspicious_step': [],
        'suspicious_spike': [],
        'redundant': []
    }

    # 1. CHECK FOR "STEP FUNCTIONS" (Daily Data treated as Hourly)
    # Logic: If a column changes value in < 5% of rows, it's likely Daily (1/24 = ~4.1%)
    print("...Scanning for Step Functions (Daily data disguised as Hourly)...")
    for col in df_num.columns:
        # distinct_changes / total_rows
        # We use .diff() != 0 to find changes
        change_density = (df_num[col].diff() != 0).mean()
        
        # If it changes rarely (but is not just a static 0), flag it
        if 0.001 < change_density < 0.08: 
            # Check if it looks like a slope calculation (e.g., 'slope' in name)
            # A daily slope on hourly data looks like: 0, 0, 0, 0, HUGE_JUMP, 0, 0...
            if 'slope' in col or 'accel' in col:
                results['suspicious_step'].append((col, f"{change_density:.1%} change rate"))

    # 2. CHECK FOR SPIKES / BAD SCALING
    # Logic: Kurtosis > 50 implies massive outliers relative to distribution
    print("...Scanning for Spikes (Kurtosis & Outliers)...")
    kurt = df_num.kurtosis()
    spiky_cols = kurt[kurt > 50].sort_values(ascending=False).head(10)
    
    for col, k_val in spiky_cols.items():
        # Double check limits
        c_min, c_max = df_num[col].min(), df_num[col].max()
        c_std = df_num[col].std()
        
        # If the range is massive compared to std dev, it's a spike
        if c_std > 0 and (c_max - c_min) / c_std > 20:
            results['suspicious_spike'].append((col, f"Kurtosis: {k_val:.1f}"))

    # 3. CHECK FOR REDUNDANCY (Collinearity)
    # Logic: Correlation > 0.99
    print("...Scanning for Redundancy (Corr > 0.99)...")
    # We sample if the file is huge to speed up correlation
    df_sample = df_num.sample(n=min(10000, len(df_num))) if len(df_num) > 10000 else df_num
    corr_matrix = df_sample.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find index of feature columns with correlation greater than 0.99
    to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
    
    for col in to_drop:
        # Find who it is correlated with
        partners = upper.index[upper[col] > 0.99].tolist()
        results['redundant'].append((col, f"Correlated with: {partners}"))

    # --- REPORT ---
    print("\n" + "="*60)
    print("AUDIT REPORT")
    print("="*60)
    
    if results['suspicious_step']:
        print("\n[CRITICAL] POTENTIAL BAD WINDOWS (Daily vars with Hourly Slopes?):")
        for c, reason in results['suspicious_step']:
            print(f"  X {c} ({reason})")
            print(f"    -> RECOMMENDATION: Remove this column or use Daily Window.")
            
    if results['suspicious_spike']:
        print("\n[WARN] EXTREME SPIKES (Data Errors or Structural Breaks?):")
        for c, reason in results['suspicious_spike']:
            print(f"  ! {c} ({reason})")
            
    if results['redundant']:
        print("\n[INFO] HIGHLY REDUNDANT (Safe to drop for XGBoost speed):")
        for c, reason in results['redundant']:
            print(f"  - {c}")
            
    print("\n" + "="*60)

if __name__ == "__main__":
    # Usage: python audit_features.py path/to/2304_enh.parquet
    if len(sys.argv) > 1:
        audit_file(sys.argv[1])
    else:
        print("Please provide a parquet file path.")
