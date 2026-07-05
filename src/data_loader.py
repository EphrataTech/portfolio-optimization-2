"""
data_loader.py
--------------
Handles data extraction from YFinance and all cleaning steps.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os

TICKERS     = ['TSLA', 'BND', 'SPY']
START_DATE  = '2015-01-01'
END_DATE    = '2026-06-30'
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def fetch_data(tickers=TICKERS, start=START_DATE, end=END_DATE):
    """Download historical OHLCV data from YFinance for given tickers."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False)
    dfs = {}
    for ticker in tickers:
        df = raw.xs(ticker, axis=1, level=1).copy()
        df.index = pd.to_datetime(df.index)
        dfs[ticker] = df
    return dfs


def clean_data(dfs):
    """
    Clean each asset DataFrame:
    - Forward fill then backward fill missing values
    - Enforce correct dtypes (float64 for prices, int64 for volume)
    """
    cleaned = {}
    for ticker, df in dfs.items():
        df = df.ffill().bfill()
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            df[col] = df[col].astype(np.float64)
        df['Volume'] = df['Volume'].astype(np.int64)
        cleaned[ticker] = df
    return cleaned


def save_data(dfs, out_dir=PROCESSED_DIR):
    """Save cleaned DataFrames to CSV in the processed data directory."""
    os.makedirs(out_dir, exist_ok=True)
    for ticker, df in dfs.items():
        path = os.path.join(out_dir, f'{ticker}_cleaned.csv')
        df.to_csv(path)
        print(f"Saved: {path}")


def load_cleaned(ticker, data_dir=PROCESSED_DIR):
    """Load a previously saved cleaned CSV for a given ticker."""
    path = os.path.join(data_dir, f'{ticker}_cleaned.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    return df


if __name__ == '__main__':
    print("Fetching data...")
    dfs = fetch_data()
    print("Cleaning data...")
    dfs = clean_data(dfs)
    save_data(dfs)
    print("Done.")
