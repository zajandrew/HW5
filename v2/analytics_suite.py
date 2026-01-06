"""
analytics_suite.py (v2.0 - Analytics & CSV Export)

Role:
1. Loads 'audit_*.csv' logs and 'models/*/*.json' models.
2. Visualizes performance stability (AUC/Precision over time).
3. Aggregates Feature Importance into a "Master Alpha CSV".
4. Plots feature evolution heatmaps (regime detection).

Outputs:
- analytics_{type}_performance.png
- analytics_{type}_importance_heatmap.png
- analytics_{type}_feature_stats.csv  <-- NEW: The Excel Analysis File
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
        try:
            model.load_model(f)
        except Exception as e:
            print(f"   [Warn] Could not load {f.name}: {e}")
            continue
        
        # Extract Importance (Gain = Predictive Power)
        scores = model.get_score(importance_type='gain')
        
        # Normalize to % for fair comparison across months
        total_gain = sum(scores.values())
        if total_gain > 0:
            scores = {k: v / total_gain for k, v in scores.items()}
            
        importance_history[month] = scores

    # Convert to DataFrame (Features x Months) and fill missing with 0
    df_imp = pd.DataFrame(importance_history).fillna(0.0)
    return df_imp

def generate_feature_csv(df_imp, strategy_type):
    """
    Generates a high-value CSV analyzing which features are 'Real Alpha'
    vs 'Noise'.
    """
    if df_imp is None or df_imp.empty: return

    # 1. Calculate Core Metrics
    stats = pd.DataFrame()
    stats['avg_gain'] = df_imp.mean(axis=1)           # How strong is it usually?
    stats['std_gain'] = df_imp.std(axis=1)            # How volatile is it?
    stats['max_gain'] = df_imp.max(axis=1)            # Peak influence
    
    # 2. Calculate "Presence" (How often is it in the top 20?)
    # A feature that appears 100% of the time is "Structural Alpha"
    # A feature that appears 10% of the time is "Regime Alpha" or Noise
    
    # Get ranks for every month
    ranks = df_imp.rank(ascending=False, method='min', axis=0)
    
    # Count how many months it was in Top 10 or Top 50
    stats['months_in_top10'] = (ranks <= 10).sum(axis=1)
    stats['months_in_top50'] = (ranks <= 50).sum(axis=1)
    stats['total_months'] = df_imp.shape[1]
    
    stats['stability_score'] = stats['months_in_top10'] / stats['total_months']
    
    # 3. Sort by Average Power
    stats = stats.sort_values('avg_gain', ascending=False)
    
    # 4. Save
    out_path = f"analytics_{strategy_type}_feature_stats.csv"
    stats.to_csv(out_path)
    print(f"   Saved Analysis CSV: {out_path}")
    print(f"      -> Top Feature: {stats.index[0]} (Stability: {stats['stability_score'].iloc[0]:.0%})")

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
    
    # Auto-scale Y axis but keep reasonable bounds
    auc_min = max(0.4, df['auc'].min() - 0.05)
    auc_max = min(1.0, df['auc'].max() + 0.05)
    ax1.set_ylim(auc_min, auc_max)
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

def plot_feature_evolution(df_imp, strategy_type):
    """
    Heatmap showing how the Top 15 features change rank over time.
    """
    if df_imp is None or df_imp.empty: return

    # 1. Identify Top 15 Global Features
    top_features = df_imp.mean(axis=1).nlargest(15).index
    df_subset = df_imp.loc[top_features]
    
    # 2. Plot Heatmap
    plt.figure(figsize=(16, 9))
    sns.heatmap(df_subset, cmap="magma", linewidths=.5, annot=False, cbar_kws={'label': 'Feature Importance (Gain)'})
    
    plt.title(f"{strategy_type.upper()}: Alpha Evolution (Feature Stability)", fontsize=16)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"analytics_{strategy_type}_importance_heatmap.png")
    print(f"   Saved plot: analytics_{strategy_type}_importance_heatmap.png")
    plt.close()

def run_analytics(strategy_type):
    print(f"\n--- Generating Analytics for {strategy_type.upper()} ---")
    
    # 1. Performance Plots (from CSV logs)
    plot_performance_timeline(strategy_type)
    
    # 2. Feature Analysis (from Model files)
    df_imp = load_models_and_features(strategy_type)
    
    if df_imp is not None:
        # Generate the Excel-ready CSV
        generate_feature_csv(df_imp, strategy_type)
        
        # Generate the Heatmap
        plot_feature_evolution(df_imp, strategy_type)

if __name__ == "__main__":
    # Ensure models exist
    if not Path("models").exists():
        print("Error: 'models/' directory not found. Run training_pipeline.py first.")
    else:
        run_analytics("curves")
        run_analytics("flys")
