"""
evaluate.py
-----------
Model evaluation: MAE, RMSE, MAPE metrics and comparison visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def compute_metrics(actual, predicted, model_name='Model'):
    """Compute MAE, RMSE, and MAPE for a forecast."""
    actual    = np.array(actual)
    predicted = np.array(predicted)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}


def build_comparison_table(results):
    """
    Build a formatted comparison DataFrame from a list of metric dicts.
    Each dict should have keys: Model, MAE, RMSE, MAPE (%).
    """
    df = pd.DataFrame(results).set_index('Model')
    return df


def plot_all_forecasts(test_index, actual, forecasts, save_name='model_comparison.png'):
    """
    Overlay all model forecasts against actual prices.
    forecasts: dict of {model_name: array_of_predictions}
    """
    styles = {
        'ARIMA':  ('green',     '--',  1.5),
        'SARIMA': ('royalblue', '-.',  1.5),
        'LSTM':   ('purple',    ':',   2.0),
    }
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_index, actual, label='Actual', color='black', linewidth=2)
    for name, preds in forecasts.items():
        color, ls, lw = styles.get(name, ('gray', '-', 1.5))
        ax.plot(test_index, preds, label=name, color=color, linestyle=ls, linewidth=lw)
    ax.set_title('TSLA — All Model Forecasts vs Actual (Test Period)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (USD)')
    ax.set_xlabel('Date')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


def plot_metrics_bar(results_df, save_name='metrics_comparison.png'):
    """Bar chart comparing MAE, RMSE, MAPE across models."""
    metrics  = ['MAE', 'RMSE', 'MAPE (%)']
    colors   = ['#2CA02C', '#1F77B4', '#9467BD']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(results_df.index, results_df[metric],
                      color=colors[:len(results_df)], edgecolor='white', width=0.5)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=15)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


def print_summary(results_df):
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON TABLE")
    print("=" * 55)
    print(results_df.to_string())
    print("=" * 55)
    best = results_df['RMSE'].idxmin()
    print(f"\n  Best model by RMSE: {best}")
