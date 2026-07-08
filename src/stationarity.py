"""
stationarity.py
---------------
Augmented Dickey-Fuller stationarity tests on price and return series.
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series, name=''):
    """
    Run ADF test on a series.
    Returns a dict with statistic, p-value, critical values, and result.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    stationary = result[1] < 0.05
    return {
        'name':        name,
        'adf_stat':    result[0],
        'p_value':     result[1],
        'critical_5%': result[4]['5%'],
        'stationary':  stationary,
        'result':      'STATIONARY' if stationary else 'NON-STATIONARY',
    }


def run_stationarity_tests(dfs):
    """
    Run ADF tests on closing prices and daily returns for all assets.
    Prints and returns a summary DataFrame.
    """
    rows = []
    for ticker, df in dfs.items():
        rows.append(adf_test(df['Adj Close'],    name=f'{ticker} - Closing Price'))
        rows.append(adf_test(df['Daily_Return'], name=f'{ticker} - Daily Return'))

    summary = pd.DataFrame(rows).set_index('name')

    print("\nAugmented Dickey-Fuller Stationarity Test Results")
    print("=" * 65)
    for idx, row in summary.iterrows():
        print(f"\n  {idx}")
        print(f"    ADF Statistic : {row['adf_stat']:.4f}")
        print(f"    p-value       : {row['p_value']:.4f}")
        print(f"    Critical (5%) : {row['critical_5%']:.4f}")
        print(f"    Result        : {row['result']}")

    return summary


if __name__ == '__main__':
    from data_loader import fetch_data, clean_data
    from eda import compute_daily_returns

    dfs = clean_data(fetch_data())
    dfs = compute_daily_returns(dfs)
    run_stationarity_tests(dfs)
