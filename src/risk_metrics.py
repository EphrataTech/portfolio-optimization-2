"""
risk_metrics.py
---------------
Computes foundational portfolio risk metrics:
Value at Risk (VaR), Sharpe Ratio, annualized return and volatility.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

TRADING_DAYS     = 252
RISK_FREE_ANNUAL = 0.05
RISK_FREE_DAILY  = RISK_FREE_ANNUAL / TRADING_DAYS
PROCESSED_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
COLORS           = {'TSLA': '#E31937', 'BND': '#1F77B4', 'SPY': '#2CA02C'}


def compute_var(returns, confidence=0.95):
    """
    Historical Value at Risk at given confidence level.
    Returns the loss threshold (negative number).
    """
    return np.percentile(returns, (1 - confidence) * 100)


def compute_sharpe(returns, risk_free_daily=RISK_FREE_DAILY):
    """Annualized Sharpe Ratio using daily returns."""
    excess = returns - risk_free_daily
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def compute_annualized_volatility(returns):
    """Annualized volatility from daily returns."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def compute_annualized_return(returns):
    """Annualized return from daily returns."""
    return returns.mean() * TRADING_DAYS


def compute_all_metrics(dfs):
    """
    Compute VaR, Sharpe Ratio, annualized return and volatility
    for each asset. Returns a summary DataFrame.
    """
    records = {}
    for ticker, df in dfs.items():
        returns = df['Daily_Return'].dropna() / 100
        records[ticker] = {
            'Ann. Return':     compute_annualized_return(returns),
            'Ann. Volatility': compute_annualized_volatility(returns),
            'Sharpe Ratio':    compute_sharpe(returns),
            'VaR (95%)':       compute_var(returns, confidence=0.95),
        }
    return pd.DataFrame(records)


def plot_var(dfs, save_dir=PROCESSED_DIR):
    """Plot return distributions with VaR threshold lines."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ticker in zip(axes, dfs):
        returns = dfs[ticker]['Daily_Return'].dropna()
        var_95  = compute_var(returns / 100, confidence=0.95) * 100
        ax.hist(returns, bins=80, color=COLORS[ticker], edgecolor='white', alpha=0.75)
        ax.axvline(var_95, color='red', linestyle='--', linewidth=2,
                   label=f'VaR 95%: {var_95:.2f}%')
        ax.set_title(f'{ticker} - Value at Risk', fontsize=12)
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)
    plt.suptitle('Historical Value at Risk (95% Confidence)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'var_visualization.png'), dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    from data_loader import fetch_data, clean_data
    from eda import compute_daily_returns

    dfs = clean_data(fetch_data())
    dfs = compute_daily_returns(dfs)
    metrics = compute_all_metrics(dfs)
    print(metrics.to_string())
    plot_var(dfs)
