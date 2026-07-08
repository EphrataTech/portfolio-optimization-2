"""
forecast.py
-----------
Generates 6-12 month future forecasts using ARIMA and LSTM models.
Includes confidence interval computation and all visualizations.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
FORECAST_MONTHS = 12   # 6–12 month horizon
TRADING_DAYS_PER_MONTH = 21


def generate_future_dates(last_date, n_days):
    """Generate n_days of future business dates starting after last_date."""
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)


# ── ARIMA Future Forecast ─────────────────────────────────────────────────────

def arima_future_forecast(fitted_model, last_date, n_months=FORECAST_MONTHS):
    """
    Generate future point forecast + 95% CI from a fitted SARIMAX model.
    Returns a DataFrame with columns: forecast, lower_ci, upper_ci.
    """
    n_days   = n_months * TRADING_DAYS_PER_MONTH
    fut_idx  = generate_future_dates(last_date, n_days)
    pred     = fitted_model.get_forecast(steps=n_days)
    mean     = pred.predicted_mean.values
    ci       = pred.conf_int(alpha=0.05)

    return pd.DataFrame({
        'forecast':  mean,
        'lower_ci':  ci.iloc[:, 0].values,
        'upper_ci':  ci.iloc[:, 1].values,
    }, index=fut_idx)


# ── LSTM Future Forecast ──────────────────────────────────────────────────────

def lstm_future_forecast(model, last_window_scaled, scaler, n_months=FORECAST_MONTHS):
    """
    Iterative multi-step LSTM forecast.
    Feeds each prediction back as input for the next step.
    Returns a DataFrame with columns: forecast, lower_ci, upper_ci.
    Confidence intervals are estimated via Monte Carlo Dropout (20 passes).
    """
    n_days  = n_months * TRADING_DAYS_PER_MONTH
    window  = last_window_scaled.shape[0]

    # --- Point forecast (deterministic) ---
    current = last_window_scaled.copy().tolist()
    point_preds = []
    for _ in range(n_days):
        x    = np.array(current[-window:]).reshape(1, window, 1)
        pred = model.predict(x, verbose=0)[0, 0]
        point_preds.append(pred)
        current.append([pred])

    # --- Uncertainty via Monte Carlo Dropout (20 stochastic passes) ---
    mc_preds = []
    for _ in range(20):
        current_mc = last_window_scaled.copy().tolist()
        run = []
        for _ in range(n_days):
            x    = np.array(current_mc[-window:]).reshape(1, window, 1)
            pred = model(x, training=True).numpy()[0, 0]   # dropout active
            run.append(pred)
            current_mc.append([pred])
        mc_preds.append(run)

    mc_array = np.array(mc_preds)                          # (20, n_days)
    lower_scaled = np.percentile(mc_array, 2.5,  axis=0)
    upper_scaled = np.percentile(mc_array, 97.5, axis=0)

    # Inverse transform
    forecast_prices = scaler.inverse_transform(
        np.array(point_preds).reshape(-1, 1)).flatten()
    lower_prices = scaler.inverse_transform(
        lower_scaled.reshape(-1, 1)).flatten()
    upper_prices = scaler.inverse_transform(
        upper_scaled.reshape(-1, 1)).flatten()

    return forecast_prices, lower_prices, upper_prices


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_arima_future(history, test_actual, test_forecast_df,
                      future_df, title='ARIMA — Future Forecast',
                      save_name='arima_future_forecast.png'):
    """
    Plot: historical prices | test forecast | future forecast with CI.
    Clearly distinguishes all three segments.
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # Historical (last 2 years for clarity)
    hist_tail = history.iloc[-504:]
    ax.plot(hist_tail.index, hist_tail.values,
            color='steelblue', linewidth=1.2, label='Historical Price')

    # Test period forecast
    ax.plot(test_forecast_df.index, test_forecast_df['forecast'],
            color='darkorange', linewidth=1.5, linestyle='--', label='Test Forecast')
    ax.fill_between(test_forecast_df.index,
                    test_forecast_df['lower_ci'], test_forecast_df['upper_ci'],
                    alpha=0.15, color='darkorange')

    # Actual test prices
    ax.plot(test_actual.index, test_actual.values,
            color='black', linewidth=1.0, linestyle=':', label='Actual (Test)')

    # Future forecast
    ax.plot(future_df.index, future_df['forecast'],
            color='green', linewidth=2.0, label='Future Forecast (12 months)')
    ax.fill_between(future_df.index,
                    future_df['lower_ci'], future_df['upper_ci'],
                    alpha=0.2, color='green', label='95% Confidence Interval')

    # Dividers
    ax.axvline(test_actual.index[0],   color='gray',  linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(future_df.index[0],     color='green', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (USD)')
    ax.set_xlabel('Date')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


def plot_lstm_future(history, test_actual, test_forecast,
                     future_dates, future_forecast, lower_ci, upper_ci,
                     save_name='lstm_future_forecast.png'):
    """
    Plot: historical prices | test forecast | LSTM future forecast with MC CI.
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    hist_tail = history.iloc[-504:]
    ax.plot(hist_tail.index, hist_tail.values,
            color='steelblue', linewidth=1.2, label='Historical Price')

    ax.plot(test_actual.index, test_forecast,
            color='darkorange', linewidth=1.5, linestyle='--', label='Test Forecast (LSTM)')
    ax.plot(test_actual.index, test_actual.values,
            color='black', linewidth=1.0, linestyle=':', label='Actual (Test)')

    ax.plot(future_dates, future_forecast,
            color='purple', linewidth=2.0, label='Future Forecast (12 months)')
    ax.fill_between(future_dates, lower_ci, upper_ci,
                    alpha=0.2, color='purple', label='95% CI (MC Dropout)')

    ax.axvline(test_actual.index[0],  color='gray',   linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(future_dates[0],       color='purple',  linestyle='--', linewidth=1, alpha=0.7)

    ax.set_title('LSTM — Future Forecast with Monte Carlo Confidence Interval',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (USD)')
    ax.set_xlabel('Date')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


def plot_ci_width_over_time(future_df=None,
                             future_dates=None, lower_ci=None, upper_ci=None,
                             save_name='ci_width_over_time.png'):
    """
    Plot how confidence interval width grows over the forecast horizon
    for both ARIMA and LSTM forecasts.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    if future_df is not None:
        arima_width = future_df['upper_ci'] - future_df['lower_ci']
        ax.plot(future_df.index, arima_width,
                color='green', linewidth=1.5, label='ARIMA 95% CI Width')

    if future_dates is not None and lower_ci is not None:
        lstm_width = upper_ci - lower_ci
        ax.plot(future_dates, lstm_width,
                color='purple', linewidth=1.5, linestyle='--', label='LSTM 95% CI Width (MC)')

    ax.set_title('Confidence Interval Width Over Forecast Horizon', fontsize=13, fontweight='bold')
    ax.set_ylabel('CI Width (USD)')
    ax.set_xlabel('Date')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


def trend_analysis(future_df_or_array, label='Forecast'):
    """
    Print a simple trend analysis: direction, % change, and CI width growth.
    """
    if isinstance(future_df_or_array, pd.DataFrame):
        values = future_df_or_array['forecast'].values
        ci_start = (future_df_or_array['upper_ci'].iloc[0]  - future_df_or_array['lower_ci'].iloc[0])
        ci_end   = (future_df_or_array['upper_ci'].iloc[-1] - future_df_or_array['lower_ci'].iloc[-1])
    else:
        values   = np.array(future_df_or_array)
        ci_start = ci_end = None

    pct_change = (values[-1] - values[0]) / values[0] * 100
    direction  = 'UPWARD' if pct_change > 1 else ('DOWNWARD' if pct_change < -1 else 'STABLE')

    print(f"\n{'='*55}")
    print(f"  {label} — Trend Analysis")
    print(f"{'='*55}")
    print(f"  Start Price  : ${values[0]:.2f}")
    print(f"  End Price    : ${values[-1]:.2f}")
    print(f"  Change       : {pct_change:+.2f}%")
    print(f"  Direction    : {direction}")
    if ci_start is not None:
        print(f"  CI Width (start) : ${ci_start:.2f}")
        print(f"  CI Width (end)   : ${ci_end:.2f}")
        print(f"  CI Expansion     : {(ci_end/ci_start - 1)*100:+.1f}%")
    print(f"{'='*55}")
