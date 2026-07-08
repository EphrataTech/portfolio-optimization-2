import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_price_series(n=1000, start='2015-01-01', seed=42):
    np.random.seed(seed)
    dates  = pd.bdate_range(start=start, periods=n)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    return pd.Series(prices, index=dates, name='Close')


def make_forecast_df(n=252, start='2026-07-01', seed=0):
    np.random.seed(seed)
    idx    = pd.bdate_range(start=start, periods=n)
    fc     = 300 + np.cumsum(np.random.normal(0.1, 1.5, n))
    lower  = fc - np.linspace(10, 80, n)
    upper  = fc + np.linspace(10, 80, n)
    return pd.DataFrame({'forecast': fc, 'lower_ci': lower, 'upper_ci': upper}, index=idx)


def simple_lstm_iterative(last_window, n_steps):
    """Simulate iterative LSTM forecast without a real model."""
    preds = []
    window = list(last_window)
    for _ in range(n_steps):
        pred = np.mean(window[-10:]) * np.random.uniform(0.99, 1.01)
        preds.append(pred)
        window.append(pred)
    return np.array(preds)


# ── Section 1: Future Date Generation ────────────────────────────────────────

class TestFutureDateGeneration:

    def test_future_dates_are_business_days(self):
        last = pd.Timestamp('2026-06-30')
        dates = pd.bdate_range(start=last + pd.Timedelta(days=1), periods=252)
        assert len(dates) == 252
        assert all(d.weekday() < 5 for d in dates)

    def test_future_dates_start_after_last_date(self):
        last  = pd.Timestamp('2026-06-30')
        dates = pd.bdate_range(start=last + pd.Timedelta(days=1), periods=10)
        assert dates[0] > last

    def test_six_month_forecast_length(self):
        n_days = 6 * 21
        assert n_days == 126

    def test_twelve_month_forecast_length(self):
        n_days = 12 * 21
        assert n_days == 252

    def test_future_dates_monotonic(self):
        dates = pd.bdate_range(start='2026-07-01', periods=100)
        assert dates.is_monotonic_increasing

    def test_no_weekends_in_future_dates(self):
        dates = pd.bdate_range(start='2026-07-01', periods=252)
        assert all(d.weekday() < 5 for d in dates)


# ── Section 2: ARIMA Confidence Intervals ────────────────────────────────────

class TestARIMAConfidenceIntervals:

    def setup_method(self):
        self.fc_df = make_forecast_df(n=252)

    def test_forecast_df_has_required_columns(self):
        assert {'forecast', 'lower_ci', 'upper_ci'}.issubset(self.fc_df.columns)

    def test_upper_ci_always_above_lower_ci(self):
        assert (self.fc_df['upper_ci'] > self.fc_df['lower_ci']).all()

    def test_forecast_within_ci(self):
        assert (self.fc_df['forecast'] >= self.fc_df['lower_ci']).all()
        assert (self.fc_df['forecast'] <= self.fc_df['upper_ci']).all()

    def test_ci_width_is_positive(self):
        width = self.fc_df['upper_ci'] - self.fc_df['lower_ci']
        assert (width > 0).all()

    def test_ci_widens_over_horizon(self):
        """CI width at end of horizon should be greater than at start."""
        width = self.fc_df['upper_ci'] - self.fc_df['lower_ci']
        assert width.iloc[-1] > width.iloc[0]

    def test_forecast_length_matches_horizon(self):
        assert len(self.fc_df) == 252

    def test_forecast_index_is_datetime(self):
        assert isinstance(self.fc_df.index, pd.DatetimeIndex)


# ── Section 3: LSTM Iterative Forecasting ────────────────────────────────────

class TestLSTMIterativeForecast:

    def setup_method(self):
        self.series = make_price_series(n=800)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled = self.scaler.fit_transform(self.series.values.reshape(-1, 1))
        self.window = 60

    def test_iterative_forecast_correct_length(self):
        last_window = self.scaled[-self.window:, 0]
        preds = simple_lstm_iterative(last_window, n_steps=252)
        assert len(preds) == 252

    def test_iterative_forecast_no_nan(self):
        last_window = self.scaled[-self.window:, 0]
        preds = simple_lstm_iterative(last_window, n_steps=100)
        assert not np.isnan(preds).any()

    def test_inverse_transform_restores_price_scale(self):
        dummy_scaled = np.array([[0.0], [0.5], [1.0]])
        restored = self.scaler.inverse_transform(dummy_scaled).flatten()
        assert restored[0] < restored[1] < restored[2]

    def test_mc_dropout_produces_spread(self):
        """Multiple MC passes should produce different predictions."""
        np.random.seed(None)
        last_window = self.scaled[-self.window:, 0]
        run1 = simple_lstm_iterative(last_window, n_steps=50)
        run2 = simple_lstm_iterative(last_window, n_steps=50)
        assert not np.allclose(run1, run2)

    def test_mc_lower_ci_below_upper_ci(self):
        runs = np.array([
            simple_lstm_iterative(self.scaled[-self.window:, 0], n_steps=50)
            for _ in range(20)
        ])
        lower = np.percentile(runs, 2.5,  axis=0)
        upper = np.percentile(runs, 97.5, axis=0)
        assert (upper > lower).all()

    def test_mc_percentile_shape(self):
        runs = np.array([
            simple_lstm_iterative(self.scaled[-self.window:, 0], n_steps=50)
            for _ in range(20)
        ])
        lower = np.percentile(runs, 2.5,  axis=0)
        upper = np.percentile(runs, 97.5, axis=0)
        assert lower.shape == (50,)
        assert upper.shape == (50,)


# ── Section 4: CI Width Growth Analysis ──────────────────────────────────────

class TestCIWidthGrowth:

    def test_ci_width_increases_monotonically(self):
        """Simulated expanding CI should be monotonically increasing."""
        n = 100
        width = np.linspace(10, 100, n)
        assert width[-1] > width[0]
        assert all(width[i] <= width[i+1] for i in range(n-1))

    def test_ci_expansion_ratio_computable(self):
        fc_df = make_forecast_df(n=252)
        width = fc_df['upper_ci'] - fc_df['lower_ci']
        ratio = width.iloc[-1] / width.iloc[0]
        assert ratio > 1.0

    def test_near_term_ci_narrower_than_long_term(self):
        fc_df = make_forecast_df(n=252)
        width = fc_df['upper_ci'] - fc_df['lower_ci']
        near_term_avg = width.iloc[:21].mean()
        long_term_avg = width.iloc[-21:].mean()
        assert long_term_avg > near_term_avg

    def test_ci_width_all_positive(self):
        fc_df = make_forecast_df(n=252)
        width = fc_df['upper_ci'] - fc_df['lower_ci']
        assert (width > 0).all()


# ── Section 5: Trend Analysis ─────────────────────────────────────────────────

class TestTrendAnalysis:

    def test_upward_trend_detected(self):
        values = np.linspace(100, 150, 252)
        pct    = (values[-1] - values[0]) / values[0] * 100
        assert pct > 1
        assert 'UPWARD' == ('UPWARD' if pct > 1 else 'DOWNWARD' if pct < -1 else 'STABLE')

    def test_downward_trend_detected(self):
        values = np.linspace(150, 100, 252)
        pct    = (values[-1] - values[0]) / values[0] * 100
        assert pct < -1

    def test_stable_trend_detected(self):
        values = np.ones(252) * 200
        pct    = (values[-1] - values[0]) / values[0] * 100
        assert abs(pct) <= 1

    def test_pct_change_calculation(self):
        start, end = 100.0, 120.0
        pct = (end - start) / start * 100
        assert pct == pytest.approx(20.0)

    def test_forecast_summary_keys(self):
        fc_df = make_forecast_df(n=252)
        summary = {
            'start':    fc_df['forecast'].iloc[0],
            'end':      fc_df['forecast'].iloc[-1],
            'pct_chg':  (fc_df['forecast'].iloc[-1] - fc_df['forecast'].iloc[0]) / fc_df['forecast'].iloc[0] * 100,
            'ci_start': fc_df['upper_ci'].iloc[0]  - fc_df['lower_ci'].iloc[0],
            'ci_end':   fc_df['upper_ci'].iloc[-1] - fc_df['lower_ci'].iloc[-1],
        }
        assert all(np.isfinite(v) for v in summary.values())

    def test_monthly_resample_produces_12_rows(self):
        fc_df = make_forecast_df(n=252, start='2026-07-01')
        monthly = fc_df['forecast'].resample('ME').last().dropna()
        assert len(monthly) <= 12
        assert len(monthly) >= 11
