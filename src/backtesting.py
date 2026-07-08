"""
backtesting.py
--------------
Backtesting engine:
- Simulates optimal portfolio strategy vs 60/40 benchmark
- Supports buy-and-hold and monthly rebalancing
- Computes total return, annualized return, Sharpe, max drawdown
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

PROCESSED_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRADING_DAYS   = 252
RISK_FREE_RATE = 0.05
TICKERS        = ['TSLA', 'BND', 'SPY']

# 60/40 benchmark: 60% SPY, 40% BND
BENCHMARK_WEIGHTS = {'TSLA': 0.0, 'BND': 0.4, 'SPY': 0.6}


# ── Data Preparation ──────────────────────────────────────────────────────────

def get_backtest_returns(dfs, start='2025-01-01', end=None):
    """
    Extract daily returns for all assets over the backtest window.
    Returns a DataFrame of daily returns.
    """
    prices = pd.DataFrame({
        ticker: dfs[ticker]['Adj Close'] for ticker in TICKERS
    })
    if end:
        prices = prices.loc[start:end]
    else:
        prices = prices.loc[start:]
    return prices.pct_change().dropna()


# ── Portfolio Simulation ──────────────────────────────────────────────────────

def simulate_portfolio(daily_returns, weights, rebalance_monthly=False):
    """
    Simulate portfolio cumulative returns.
    - weights: dict {ticker: weight}
    - rebalance_monthly: if True, rebalance back to target weights each month
    Returns a Series of cumulative portfolio values (starting at 1.0).
    """
    w = np.array([weights[t] for t in TICKERS])
    port_returns = daily_returns[TICKERS].values @ w

    if rebalance_monthly:
        port_returns = _monthly_rebalance(daily_returns, weights)

    cum_returns = pd.Series(
        (1 + port_returns).cumprod(),
        index=daily_returns.index,
        name='Portfolio'
    )
    return cum_returns


def _monthly_rebalance(daily_returns, weights):
    """
    Simulate monthly rebalancing: at the start of each month,
    reset weights to target allocation.
    Returns array of daily portfolio returns.
    """
    w_target = np.array([weights[t] for t in TICKERS])
    port_rets = []
    current_w = w_target.copy()

    for i, (date, row) in enumerate(daily_returns[TICKERS].iterrows()):
        # Rebalance at start of each month
        if i > 0 and date.month != daily_returns.index[i - 1].month:
            current_w = w_target.copy()
        daily_ret = np.dot(current_w, row.values)
        port_rets.append(daily_ret)
        # Drift weights
        asset_growth = (1 + row.values)
        current_w    = current_w * asset_growth
        current_w   /= current_w.sum()

    return np.array(port_rets)


# ── Performance Metrics ───────────────────────────────────────────────────────

def compute_performance_metrics(cum_returns, label='Portfolio'):
    """
    Compute total return, annualized return, Sharpe Ratio, and max drawdown.
    """
    daily_rets = cum_returns.pct_change().dropna()
    total_ret  = cum_returns.iloc[-1] - 1
    n_days     = len(daily_rets)
    ann_ret    = (1 + total_ret) ** (TRADING_DAYS / n_days) - 1
    ann_vol    = daily_rets.std() * np.sqrt(TRADING_DAYS)
    sharpe     = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan
    max_dd     = _max_drawdown(cum_returns)

    return {
        'Portfolio':          label,
        'Total Return (%)':   round(total_ret * 100, 2),
        'Ann. Return (%)':    round(ann_ret  * 100, 2),
        'Ann. Volatility (%)':round(ann_vol  * 100, 2),
        'Sharpe Ratio':       round(sharpe,          4),
        'Max Drawdown (%)':   round(max_dd   * 100, 2),
    }


def _max_drawdown(cum_returns):
    """Compute maximum drawdown from a cumulative returns series."""
    rolling_max = cum_returns.cummax()
    drawdown    = (cum_returns - rolling_max) / rolling_max
    return drawdown.min()


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_cumulative_returns(strategy_cum, benchmark_cum,
                             save_name='backtest_cumulative_returns.png'):
    """Plot strategy vs benchmark cumulative returns over the backtest period."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(strategy_cum.index,  strategy_cum.values,
            color='green',     linewidth=2.0, label='Optimal Strategy')
    ax.plot(benchmark_cum.index, benchmark_cum.values,
            color='steelblue', linewidth=2.0, linestyle='--', label='Benchmark (60% SPY / 40% BND)')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_title('Backtest: Strategy vs Benchmark — Cumulative Returns',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Value (Starting = 1.0)')
    ax.set_xlabel('Date')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


def plot_drawdown(strategy_cum, benchmark_cum,
                  save_name='backtest_drawdown.png'):
    """Plot drawdown curves for both portfolios."""
    def drawdown_series(cum):
        rolling_max = cum.cummax()
        return (cum - rolling_max) / rolling_max * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(strategy_cum.index,  drawdown_series(strategy_cum),
                    color='green',     alpha=0.4, label='Strategy Drawdown')
    ax.fill_between(benchmark_cum.index, drawdown_series(benchmark_cum),
                    color='steelblue', alpha=0.4, label='Benchmark Drawdown')
    ax.set_title('Drawdown Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Date')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


def print_metrics_table(strategy_metrics, benchmark_metrics):
    df = pd.DataFrame([strategy_metrics, benchmark_metrics]).set_index('Portfolio')
    print("\n" + "=" * 65)
    print("  BACKTEST PERFORMANCE METRICS")
    print("=" * 65)
    print(df.to_string())
    print("=" * 65)
    winner = ('Strategy' if strategy_metrics['Sharpe Ratio'] > benchmark_metrics['Sharpe Ratio']
              else 'Benchmark')
    print(f"\n  Higher Sharpe Ratio: {winner}")
    return df
