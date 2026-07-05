import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_price_series(n=1000, start="2015-01-01", seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range(start=start, periods=n)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    return pd.Series(prices, index=dates, name="Close")


def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def compute_metrics(actual, predicted):
    actual    = np.array(actual)
    predicted = np.array(predicted)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ── Section 1: Train / Test Split ─────────────────────────────────────────────

class TestTrainTestSplit:

    def setup_method(self):
        self.series = make_price_series(n=1000)
        self.split  = "2019-01-01"
        self.train  = self.series[self.series.index < self.split]
        self.test   = self.series[self.series.index >= self.split]

    def test_no_data_leakage(self):
        assert self.train.index.max() < self.test.index.min()

    def test_split_is_chronological(self):
        assert self.train.index.is_monotonic_increasing
        assert self.test.index.is_monotonic_increasing

    def test_train_and_test_cover_full_series(self):
        assert len(self.train) + len(self.test) == len(self.series)

    def test_train_larger_than_test(self):
        assert len(self.train) > len(self.test)

    def test_no_overlap_between_splits(self):
        overlap = self.train.index.intersection(self.test.index)
        assert len(overlap) == 0

    def test_train_not_empty(self):
        assert len(self.train) > 0

    def test_test_not_empty(self):
        assert len(self.test) > 0


# ── Section 2: ARIMA Preparation ─────────────────────────────────────────────

class TestARIMAPreparation:

    def setup_method(self):
        self.series = make_price_series(n=500)

    def test_first_difference_reduces_trend(self):
        diff = self.series.diff().dropna()
        assert abs(diff.mean()) < abs(self.series.mean())

    def test_first_difference_length(self):
        diff = self.series.diff().dropna()
        assert len(diff) == len(self.series) - 1

    def test_differenced_series_has_no_nan(self):
        diff = self.series.diff().dropna()
        assert not diff.isnull().any()

    def test_arima_order_tuple_valid(self):
        order = (1, 1, 1)
        assert len(order) == 3
        assert all(isinstance(v, int) and v >= 0 for v in order)

    def test_sarima_seasonal_order_valid(self):
        seasonal_order = (1, 1, 1, 12)
        assert len(seasonal_order) == 4
        assert seasonal_order[3] > 1  # seasonal period must be > 1

    def test_forecast_length_matches_test(self):
        """Simulated forecast array should match test set length."""
        test = self.series.iloc[-50:]
        fake_forecast = np.random.normal(self.series.mean(), self.series.std(), len(test))
        assert len(fake_forecast) == len(test)


# ── Section 3: LSTM Data Preparation ─────────────────────────────────────────

class TestLSTMDataPreparation:

    def setup_method(self):
        self.series      = make_price_series(n=800)
        self.window_size = 60
        self.scaler      = MinMaxScaler(feature_range=(0, 1))

    def test_scaler_output_range(self):
        scaled = self.scaler.fit_transform(self.series.values.reshape(-1, 1))
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0

    def test_inverse_transform_recovers_original(self):
        values = self.series.values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)
        recovered = self.scaler.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(values, recovered, decimal=5)

    def test_sequence_X_shape(self):
        scaled = self.scaler.fit_transform(self.series.values.reshape(-1, 1))
        X, y = create_sequences(scaled, self.window_size)
        assert X.shape[1] == self.window_size

    def test_sequence_y_length(self):
        scaled = self.scaler.fit_transform(self.series.values.reshape(-1, 1))
        X, y = create_sequences(scaled, self.window_size)
        assert len(y) == len(self.series) - self.window_size

    def test_sequence_X_y_aligned(self):
        scaled = self.scaler.fit_transform(self.series.values.reshape(-1, 1))
        X, y = create_sequences(scaled, self.window_size)
        assert len(X) == len(y)

    def test_lstm_input_reshape(self):
        scaled = self.scaler.fit_transform(self.series.values.reshape(-1, 1))
        X, y = create_sequences(scaled, self.window_size)
        X_3d = X.reshape(X.shape[0], X.shape[1], 1)
        assert X_3d.ndim == 3
        assert X_3d.shape[2] == 1

    def test_no_nan_in_sequences(self):
        scaled = self.scaler.fit_transform(self.series.values.reshape(-1, 1))
        X, y = create_sequences(scaled, self.window_size)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_window_size_effect_on_samples(self):
        scaled = self.scaler.fit_transform(self.series.values.reshape(-1, 1))
        X30, _ = create_sequences(scaled, 30)
        X60, _ = create_sequences(scaled, 60)
        assert len(X30) > len(X60)


# ── Section 4: Model Evaluation Metrics ──────────────────────────────────────

class TestModelEvaluation:

    def setup_method(self):
        np.random.seed(42)
        self.actual    = make_price_series(n=100).values
        self.perfect   = self.actual.copy()
        self.noisy     = self.actual + np.random.normal(0, 5, len(self.actual))
        self.bad       = self.actual + np.random.normal(0, 50, len(self.actual))

    def test_mae_perfect_forecast_is_zero(self):
        m = compute_metrics(self.actual, self.perfect)
        assert m["MAE"] == pytest.approx(0.0)

    def test_rmse_perfect_forecast_is_zero(self):
        m = compute_metrics(self.actual, self.perfect)
        assert m["RMSE"] == pytest.approx(0.0)

    def test_mape_perfect_forecast_is_zero(self):
        m = compute_metrics(self.actual, self.perfect)
        assert m["MAPE"] == pytest.approx(0.0)

    def test_mae_non_negative(self):
        m = compute_metrics(self.actual, self.noisy)
        assert m["MAE"] >= 0

    def test_rmse_non_negative(self):
        m = compute_metrics(self.actual, self.noisy)
        assert m["RMSE"] >= 0

    def test_mape_non_negative(self):
        m = compute_metrics(self.actual, self.noisy)
        assert m["MAPE"] >= 0

    def test_rmse_gte_mae(self):
        """RMSE is always >= MAE."""
        m = compute_metrics(self.actual, self.noisy)
        assert m["RMSE"] >= m["MAE"]

    def test_better_forecast_lower_rmse(self):
        m_noisy = compute_metrics(self.actual, self.noisy)
        m_bad   = compute_metrics(self.actual, self.bad)
        assert m_noisy["RMSE"] < m_bad["RMSE"]

    def test_better_forecast_lower_mae(self):
        m_noisy = compute_metrics(self.actual, self.noisy)
        m_bad   = compute_metrics(self.actual, self.bad)
        assert m_noisy["MAE"] < m_bad["MAE"]

    def test_metrics_dict_has_required_keys(self):
        m = compute_metrics(self.actual, self.noisy)
        assert {"MAE", "RMSE", "MAPE"}.issubset(m.keys())

    def test_all_metrics_finite(self):
        m = compute_metrics(self.actual, self.noisy)
        for key, val in m.items():
            assert np.isfinite(val), f"{key} is not finite"
