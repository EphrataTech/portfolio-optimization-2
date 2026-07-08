"""
eda.py
------
Exploratory Data Analysis: returns, rolling stats, outliers,
correlation, seasonal decomposition, and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os

COLORS = {'TSLA': '#E31937', 'BND': '#1F77B4', 'SPY': '#2CA02C'}
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def compute_daily_returns(dfs):
    """Add Daily_Return (%) column to each DataFrame."""
    for ticker, df in dfs.items():
        df['Daily_Return'] = df['Adj Close'].pct_change() * 100
    return dfs


def compute_rolling_stats(series, window=30):
    """Return rolling mean and rolling std for a price series."""
    return series.rolling(window).mean(), series.rolling(window).std()


def detect_outliers(returns, threshold=3):
    """Return boolean mask of outliers beyond threshold standard deviations."""
    mean, std = returns.mean(), returns.std()
    return (returns > mean + threshold * std) | (returns < mean - threshold * std)


def compute_correlation(dfs):
    """Return correlation matrix of daily returns across all assets."""
    combined = pd.DataFrame({
        ticker: dfs[ticker]['Daily_Return'] for ticker in dfs
    }).dropna()
    return combined.corr()


def plot_closing_prices(dfs, save_dir=PROCESSED_DIR):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for ax, ticker in zip(axes, dfs):
        ax.plot(dfs[ticker].index, dfs[ticker]['Adj Close'],
                color=COLORS[ticker], linewidth=1.2)
        ax.set_title(f'{ticker} - Adjusted Closing Price', fontsize=13, fontweight='bold')
        ax.set_ylabel('Price (USD)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel('Date')
    plt.suptitle('Adjusted Closing Prices (2015–2026)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'closing_prices.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_daily_returns(dfs, save_dir=PROCESSED_DIR):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, ticker in zip(axes, dfs):
        returns = dfs[ticker]['Daily_Return'].dropna()
        ax.plot(returns.index, returns, color=COLORS[ticker], linewidth=0.7, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f'{ticker} - Daily % Return  |  Std: {returns.std():.2f}%', fontsize=12)
        ax.set_ylabel('Return (%)')
    axes[-1].set_xlabel('Date')
    plt.suptitle('Daily Percentage Returns (2015–2026)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'daily_returns.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_rolling_stats(dfs, window=30, save_dir=PROCESSED_DIR):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for ax, ticker in zip(axes, dfs):
        price = dfs[ticker]['Adj Close']
        roll_mean, roll_std = compute_rolling_stats(price, window)
        ax.plot(price.index, price, color=COLORS[ticker], linewidth=0.8, alpha=0.5, label='Adj Close')
        ax.plot(roll_mean.index, roll_mean, color='black', linewidth=1.5, label=f'{window}-day Mean')
        ax.fill_between(price.index, roll_mean - roll_std, roll_mean + roll_std,
                        alpha=0.2, color=COLORS[ticker], label='±1 Std Dev')
        ax.set_title(f'{ticker} - Rolling Mean & Volatility Band', fontsize=12)
        ax.set_ylabel('Price (USD)')
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('Date')
    plt.suptitle(f'{window}-Day Rolling Statistics', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rolling_stats.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_return_distributions(dfs, save_dir=PROCESSED_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ticker in zip(axes, dfs):
        returns = dfs[ticker]['Daily_Return'].dropna()
        ax.hist(returns, bins=80, color=COLORS[ticker], edgecolor='white', alpha=0.85)
        ax.axvline(returns.mean(), color='black', linestyle='--', linewidth=1.5,
                   label=f'Mean: {returns.mean():.2f}%')
        ax.set_title(f'{ticker} Return Distribution', fontsize=12)
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)
    plt.suptitle('Daily Return Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'return_distributions.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(dfs, save_dir=PROCESSED_DIR):
    corr = compute_correlation(dfs)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Daily Returns Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_seasonal_decomposition(df, ticker='TSLA', save_dir=PROCESSED_DIR):
    monthly = df['Adj Close'].resample('ME').last()
    decomp  = seasonal_decompose(monthly, model='multiplicative', period=12)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    for ax, (label, data) in zip(axes, [
        ('Observed', decomp.observed), ('Trend', decomp.trend),
        ('Seasonal', decomp.seasonal), ('Residual', decomp.resid)
    ]):
        ax.plot(data.index, data, color=COLORS.get(ticker, '#333'), linewidth=1.2)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'{ticker} - Seasonal Decomposition (Monthly)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'seasonal_decomposition.png'), dpi=150, bbox_inches='tight')
    plt.show()
