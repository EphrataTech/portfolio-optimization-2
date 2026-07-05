"""
arima_model.py
--------------
ARIMA/SARIMA model: parameter search via auto_arima,
model fitting with SARIMAX, and forecast generation.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def plot_acf_pacf(series, lags=40):
    """Plot ACF and PACF of a differenced series to guide parameter selection."""
    diff = series.diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(diff,  lags=lags, ax=axes[0], title='ACF  (1st Difference)')
    plot_pacf(diff, lags=lags, ax=axes[1], title='PACF (1st Difference)', method='ywm')
    plt.tight_layout()
    plt.show()


def find_best_order(train_series, seasonal=False, m=12):
    """
    Use auto_arima to find optimal (p,d,q) and seasonal (P,D,Q,m) orders.
    Returns the fitted auto_arima model.
    """
    print(f"Running auto_arima (seasonal={seasonal}, m={m})...")
    model = auto_arima(
        train_series,
        start_p=0, start_q=0,
        max_p=3,   max_q=3,
        d=1,
        seasonal=seasonal, m=m,
        start_P=0, start_Q=0,
        max_P=1,   max_Q=1,
        D=1 if seasonal else None,
        information_criterion='aic',
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )
    print(f"Best order: {model.order}  Seasonal: {model.seasonal_order}")
    return model


def fit_sarimax(train_series, order, seasonal_order=(0, 0, 0, 0)):
    """Fit a SARIMAX model on the training series."""
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    return model


def forecast(fitted_model, steps, index=None):
    """
    Generate point forecast and 95% confidence intervals.
    Returns a DataFrame with columns: forecast, lower_ci, upper_ci.
    """
    pred = fitted_model.get_forecast(steps=steps)
    mean = pred.predicted_mean
    ci   = pred.conf_int(alpha=0.05)

    result = pd.DataFrame({
        'forecast':  mean.values,
        'lower_ci':  ci.iloc[:, 0].values,
        'upper_ci':  ci.iloc[:, 1].values,
    }, index=index if index is not None else mean.index)
    return result


def plot_forecast(train, test, forecast_df, title='ARIMA Forecast', save_name='arima_forecast.png'):
    """Plot train, actual test, and forecast with confidence interval."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index[-120:], train.iloc[-120:], label='Train (last 120 days)', color='steelblue')
    ax.plot(test.index,  test.values,                label='Actual',   color='darkorange')
    ax.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='green', linestyle='--')
    ax.fill_between(forecast_df.index,
                    forecast_df['lower_ci'],
                    forecast_df['upper_ci'],
                    alpha=0.2, color='green', label='95% CI')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    from data_loader import load_cleaned

    tsla = load_cleaned('TSLA')[['Adj Close']].rename(columns={'Adj Close': 'Close'})
    train = tsla[tsla.index < '2025-01-01']['Close']
    test  = tsla[tsla.index >= '2025-01-01']['Close']

    plot_acf_pacf(train)
    auto   = find_best_order(train, seasonal=False)
    fitted = fit_sarimax(train, order=auto.order)
    fc     = forecast(fitted, steps=len(test), index=test.index)
    plot_forecast(train, test, fc, title=f'ARIMA{auto.order} Forecast vs Actual')
    print(fc.head())
