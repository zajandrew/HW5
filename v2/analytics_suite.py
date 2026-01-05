"""
analytics_suite.py

Role:
1. Loads the 'audit_*.csv' performance logs.
2. Loads every individual XGBoost model from 'models/*/*.json'.
3. Aggregates Feature Importance (Gain) over time to find stable alpha.
4. Generates professional plots for Model Audits.

Usage:
    python analytics_suite.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set Plot Style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_models_and_features(strategy_type):
    """
    Iterates through saved JSON models and extracts feature importance.
    Returns a DataFrame where Index=Feature, Cols=Months.
    """
    model_dir = Path("models") / strategy_type
    model_files = sorted(list(model_dir.glob("model_*.json")))
    
    if not model_files:
        print(f"[{strategy_type}] No models found in {model_dir}")
        return None

    print(f"[{strategy_type}] Analyzing {len(model_files)} models for feature stability...")
    
    importance_history = {}
    
    for f in model_files:
        month = f.stem.split('_')[-1] # model_2304 -> 2304
        
        # Load Model
        model = xgb.Booster()
        model.load_model(f)
        
        # Extract Importance (Gain = Predictive Power, Weight = Frequency)
        # We use 'gain' because we want to know what DRIVES the decision.
        scores = model.get_score(importance_type='gain')
        
        # Normalize to % for fair comparison across months
        total_gain = sum(scores.values())
        if total_gain > 0:
            scores = {k: v / total_gain for k, v in scores.items()}
            
        importance_history[month] = scores

    # Convert to DataFrame (Features x Months)
    df_imp = pd.DataFrame(importance_history).fillna(0.0)
    return df_imp

def plot_performance_timeline(strategy_type):
    """
    Plots AUC, Precision, and Recall from the audit CSV.
    """
    csv_path = f"audit_{strategy_type}_walkforward.csv"
    if not os.path.exists(csv_path):
        print(f"[{strategy_type}] Audit CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df['test_month'] = df['test_month'].astype(str)
    
    # 1. Create Figure
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot AUC (Left Axis)
    sns.lineplot(data=df, x='test_month', y='auc', marker='o', label='AUC (Robustness)', ax=ax1, color='navy', linewidth=2)
    ax1.set_ylabel('AUC Score', color='navy', fontweight='bold')
    ax1.set_ylim(0.5, 0.85)
    ax1.tick_params(axis='y', labelcolor='navy')
    
    # Plot Precision/Recall (Right Axis)
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='test_month', y='precision', marker='s', label='Precision (Hit Rate)', ax=ax2, color='forestgreen', linestyle='--')
    sns.lineplot(data=df, x='test_month', y='recall', marker='^', label='Recall (Opportunity)', ax=ax2, color='darkorange', linestyle=':')
    
    ax2.set_ylabel('Precision / Recall', color='black', fontweight='bold')
    ax2.set_ylim(0, 1.0)
    
    # Formatting
    plt.title(f"{strategy_type.upper()} Strategy: Walk-Forward Performance", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Test Month", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Combine Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(f"analytics_{strategy_type}_performance.png")
    print(f"   Saved plot: analytics_{strategy_type}_performance.png")
    plt.close()

def plot_global_feature_importance(df_imp, strategy_type):
    """
    Bar chart of the top 20 features averaged over all time.
    """
    # Calculate Mean and Std
    df_stats = pd.DataFrame({
        'mean': df_imp.mean(axis=1),
        'std': df_imp.std(axis=1)
    }).sort_values('mean', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=df_stats['mean'], y=df_stats.index, palette="viridis")
    plt.errorbar(x=df_stats['mean'], y=np.arange(len(df_stats)), xerr=df_stats['std'], fmt='none', c='black', capsize=3)
    
    plt.title(f"{strategy_type.upper()}: Top 20 Alpha Drivers (Avg Gain)", fontsize=16)
    plt.xlabel("Relative Importance (Normalized Gain)", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"analytics_{strategy_type}_importance_global.png")
    print(f"   Saved plot: analytics_{strategy_type}_importance_global.png")
    plt.close()

def plot_feature_evolution(df_imp, strategy_type):
    """
    Heatmap showing how the Top 10 features change rank over time.
    """
    # 1. Identify Top 15 Global Features to track
    top_features = df_imp.mean(axis=1).nlargest(15).index
    df_subset = df_imp.loc[top_features]
    
    # 2. Plot Heatmap
    plt.figure(figsize=(16, 9))
    sns.heatmap(df_subset, cmap="magma", linewidths=.5, annot=False, cbar_kws={'label': 'Feature Importance'})
    
    plt.title(f"{strategy_type.upper()}: Alpha Evolution (Feature Stability)", fontsize=16)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"analytics_{strategy_type}_importance_heatmap.png")
    print(f"   Saved plot: analytics_{strategy_type}_importance_heatmap.png")
    plt.close()

def run_analytics(strategy_type):
    print(f"\n--- Generating Analytics for {strategy_type.upper()} ---")
    
    # 1. Performance Plots
    plot_performance_timeline(strategy_type)
    
    # 2. Feature Analysis
    df_imp = load_models_and_features(strategy_type)
    if df_imp is not None:
        plot_global_feature_importance(df_imp, strategy_type)
        plot_feature_evolution(df_imp, strategy_type)

if __name__ == "__main__":
    run_analytics("curves")
    run_analytics("flys")
