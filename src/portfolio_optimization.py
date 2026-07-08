"""
portfolio_optimization.py
--------------------------
Modern Portfolio Theory (MPT) implementation:
- Expected returns vector (forecast + historical)
- Covariance matrix from historical returns
- Efficient Frontier via Monte Carlo simulation
- Maximum Sharpe Ratio and Minimum Volatility portfolios
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.optimize import minimize
import os

PROCESSED_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRADING_DAYS   = 252
RISK_FREE_RATE = 0.05          # annual
N_PORTFOLIOS   = 10_000
TICKERS        = ['TSLA', 'BND', 'SPY']


# ── Expected Returns ──────────────────────────────────────────────────────────

def get_expected_returns(dfs, tsla_forecast_annual_return):
    """
    Build expected returns vector:
    - TSLA: from model forecast (annualized)
    - BND, SPY: historical mean daily return * 252
    """
    expected = {}
    for ticker in TICKERS:
        if ticker == 'TSLA':
            expected[ticker] = tsla_forecast_annual_return
        else:
            daily_ret = dfs[ticker]['Adj Close'].pct_change().dropna()
            expected[ticker] = daily_ret.mean() * TRADING_DAYS
    return pd.Series(expected)


# ── Covariance Matrix ─────────────────────────────────────────────────────────

def compute_cov_matrix(dfs):
    """
    Compute annualized covariance matrix from historical daily returns
    for all three assets.
    """
    returns = pd.DataFrame({
        ticker: dfs[ticker]['Adj Close'].pct_change().dropna()
        for ticker in TICKERS
    }).dropna()
    return returns.cov() * TRADING_DAYS


def plot_cov_heatmap(cov_matrix, save_name='covariance_heatmap.png'):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cov_matrix, annot=True, fmt='.6f', cmap='YlOrRd',
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Annualized Covariance Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


# ── Portfolio Metrics ─────────────────────────────────────────────────────────

def portfolio_performance(weights, expected_returns, cov_matrix):
    """Return (annual_return, annual_volatility, sharpe_ratio) for a weight vector."""
    weights = np.array(weights)
    ret  = np.dot(weights, expected_returns)
    vol  = np.sqrt(weights @ cov_matrix.values @ weights)
    shrp = (ret - RISK_FREE_RATE) / vol
    return ret, vol, shrp


# ── Efficient Frontier (Monte Carlo) ─────────────────────────────────────────

def simulate_efficient_frontier(expected_returns, cov_matrix, n=N_PORTFOLIOS, seed=42):
    """
    Randomly sample N portfolios and record return, volatility, Sharpe.
    Returns a DataFrame with columns: Return, Volatility, Sharpe, TSLA, BND, SPY.
    """
    np.random.seed(seed)
    n_assets = len(TICKERS)
    records  = []

    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        ret, vol, shrp = portfolio_performance(w, expected_returns, cov_matrix)
        records.append([ret, vol, shrp] + list(w))

    cols = ['Return', 'Volatility', 'Sharpe'] + TICKERS
    return pd.DataFrame(records, columns=cols)


# ── Optimal Portfolios (scipy) ────────────────────────────────────────────────

def max_sharpe_portfolio(expected_returns, cov_matrix):
    """Find weights that maximise the Sharpe Ratio."""
    n = len(TICKERS)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds      = tuple((0, 1) for _ in range(n))
    init        = np.ones(n) / n

    def neg_sharpe(w):
        r, v, _ = portfolio_performance(w, expected_returns, cov_matrix)
        return -(r - RISK_FREE_RATE) / v

    result = minimize(neg_sharpe, init, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    w = result.x
    ret, vol, shrp = portfolio_performance(w, expected_returns, cov_matrix)
    return dict(zip(TICKERS, w)), ret, vol, shrp


def min_volatility_portfolio(expected_returns, cov_matrix):
    """Find weights that minimise portfolio volatility."""
    n = len(TICKERS)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds      = tuple((0, 1) for _ in range(n))
    init        = np.ones(n) / n

    def port_vol(w):
        return np.sqrt(w @ cov_matrix.values @ w)

    result = minimize(port_vol, init, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    w = result.x
    ret, vol, shrp = portfolio_performance(w, expected_returns, cov_matrix)
    return dict(zip(TICKERS, w)), ret, vol, shrp


# ── Efficient Frontier Plot ───────────────────────────────────────────────────

def plot_efficient_frontier(frontier_df, max_sharpe, min_vol,
                             save_name='efficient_frontier.png'):
    """
    Plot the Efficient Frontier scatter coloured by Sharpe Ratio,
    with Max Sharpe and Min Volatility portfolios marked.
    """
    ms_weights, ms_ret, ms_vol, ms_shrp = max_sharpe
    mv_weights, mv_ret, mv_vol, mv_shrp = min_vol

    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(frontier_df['Volatility'], frontier_df['Return'],
                    c=frontier_df['Sharpe'], cmap='viridis', alpha=0.5, s=8)
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')

    # Max Sharpe
    ax.scatter(ms_vol, ms_ret, marker='*', color='red', s=300, zorder=5,
               label=f'Max Sharpe  (SR={ms_shrp:.2f})')
    ax.annotate(f'Max Sharpe\n{_fmt_weights(ms_weights)}',
                xy=(ms_vol, ms_ret), xytext=(ms_vol + 0.02, ms_ret + 0.02),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1))

    # Min Volatility
    ax.scatter(mv_vol, mv_ret, marker='D', color='blue', s=150, zorder=5,
               label=f'Min Volatility  (SR={mv_shrp:.2f})')
    ax.annotate(f'Min Vol\n{_fmt_weights(mv_weights)}',
                xy=(mv_vol, mv_ret), xytext=(mv_vol + 0.02, mv_ret - 0.04),
                fontsize=8, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1))

    ax.set_xlabel('Annual Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Annual Expected Return', fontsize=12)
    ax.set_title('Efficient Frontier — TSLA / BND / SPY', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


def _fmt_weights(w):
    return '\n'.join([f'{t}: {v*100:.1f}%' for t, v in w.items()])


# ── Summary Printer ───────────────────────────────────────────────────────────

def print_portfolio_summary(label, weights, ret, vol, shrp):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    for ticker, w in weights.items():
        print(f"  {ticker:<6}: {w*100:.2f}%")
    print(f"  {'─'*40}")
    print(f"  Expected Annual Return   : {ret*100:.2f}%")
    print(f"  Expected Annual Volatility: {vol*100:.2f}%")
    print(f"  Sharpe Ratio             : {shrp:.4f}")
    print(f"{'='*55}")
