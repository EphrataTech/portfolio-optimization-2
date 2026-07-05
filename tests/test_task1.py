import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from statsmodels.tsa.stattools import adfuller


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_price_series(n=500, start="2015-01-01", seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range(start=start, periods=n)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    return pd.Series(prices, index=dates, name="Adj Close")


def make_asset_df(n=500, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range(start="2015-01-01", periods=n)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    return pd.DataFrame({
        "Open":      prices * np.random.uniform(0.99, 1.00, n),
        "High":      prices * np.random.uniform(1.00, 1.02, n),
        "Low":       prices * np.random.uniform(0.98, 1.00, n),
        "Close":     prices,
        "Adj Close": prices,
        "Volume":    np.random.randint(1_000_000, 50_000_000, n).astype(np.int64),
    }, index=dates)


# ── Section 1: Data Extraction ────────────────────────────────────────────────

class TestDataExtraction:

    def test_tickers_defined(self):
        tickers = ["TSLA", "BND", "SPY"]
        assert len(tickers) == 3
        assert "TSLA" in tickers
        assert "BND" in tickers
        assert "SPY" in tickers

    def test_date_range_valid(self):
        start = pd.Timestamp("2015-01-01")
        end   = pd.Timestamp("2026-06-30")
        assert start < end
        assert (end - start).days > 365 * 5

    def test_dataframe_columns(self):
        df = make_asset_df()
        expected = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        assert expected.issubset(set(df.columns))

    def test_dataframe_index_is_datetime(self):
        df = make_asset_df()
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_dataframe_not_empty(self):
        df = make_asset_df()
        assert len(df) > 0


# ── Section 2: Data Cleaning ──────────────────────────────────────────────────

class TestDataCleaning:

    def test_no_missing_after_ffill_bfill(self):
        df = make_asset_df()
        df.iloc[5, 0] = np.nan
        df.iloc[10, 1] = np.nan
        df = df.ffill().bfill()
        assert df.isnull().sum().sum() == 0

    def test_price_columns_float64(self):
        df = make_asset_df()
        for col in ["Open", "High", "Low", "Close", "Adj Close"]:
            assert df[col].dtype == np.float64, f"{col} should be float64"

    def test_volume_column_int64(self):
        df = make_asset_df()
        assert df["Volume"].dtype == np.int64

    def test_no_negative_prices(self):
        df = make_asset_df()
        for col in ["Open", "High", "Low", "Close", "Adj Close"]:
            assert (df[col] > 0).all(), f"{col} contains non-positive values"

    def test_high_gte_low(self):
        df = make_asset_df()
        assert (df["High"] >= df["Low"]).all()

    def test_ffill_preserves_length(self):
        df = make_asset_df(n=100)
        df.iloc[3, 0] = np.nan
        filled = df.ffill().bfill()
        assert len(filled) == len(df)


# ── Section 3: EDA ────────────────────────────────────────────────────────────

class TestEDA:

    def test_daily_return_calculation(self):
        s = make_price_series(n=100)
        returns = s.pct_change() * 100
        assert returns.iloc[0] != returns.iloc[0]   # first value is NaN
        assert not returns.iloc[1:].isnull().any()

    def test_daily_return_length(self):
        s = make_price_series(n=200)
        returns = s.pct_change().dropna()
        assert len(returns) == 199

    def test_rolling_mean_length(self):
        s = make_price_series(n=300)
        rolling = s.rolling(30).mean()
        assert len(rolling) == len(s)
        assert rolling.iloc[:29].isnull().all()
        assert not rolling.iloc[29:].isnull().any()

    def test_rolling_std_non_negative(self):
        s = make_price_series(n=300)
        rolling_std = s.rolling(30).std().dropna()
        assert (rolling_std >= 0).all()

    def test_outlier_detection(self):
        np.random.seed(0)
        returns = pd.Series(np.random.normal(0, 1, 1000))
        mean, std = returns.mean(), returns.std()
        outliers = returns[(returns > mean + 3*std) | (returns < mean - 3*std)]
        assert len(outliers) < len(returns) * 0.05

    def test_correlation_matrix_shape(self):
        dfs = {t: make_asset_df(seed=i) for i, t in enumerate(["TSLA", "BND", "SPY"])}
        combined = pd.DataFrame({
            t: dfs[t]["Adj Close"].pct_change() for t in dfs
        }).dropna()
        corr = combined.corr()
        assert corr.shape == (3, 3)

    def test_correlation_diagonal_is_one(self):
        dfs = {t: make_asset_df(seed=i) for i, t in enumerate(["TSLA", "BND", "SPY"])}
        combined = pd.DataFrame({
            t: dfs[t]["Adj Close"].pct_change() for t in dfs
        }).dropna()
        corr = combined.corr()
        np.testing.assert_array_almost_equal(np.diag(corr.values), np.ones(3))


# ── Section 4: Stationarity ───────────────────────────────────────────────────

class TestStationarity:

    def test_price_series_non_stationary(self):
        """Random walk price series should be non-stationary (p > 0.05)."""
        s = make_price_series(n=500)
        result = adfuller(s, autolag="AIC")
        p_value = result[1]
        assert p_value > 0.05, f"Expected non-stationary price series, got p={p_value:.4f}"

    def test_returns_series_stationary(self):
        """Daily returns should be stationary (p < 0.05)."""
        s = make_price_series(n=500)
        returns = s.pct_change().dropna()
        result = adfuller(returns, autolag="AIC")
        p_value = result[1]
        assert p_value < 0.05, f"Expected stationary returns, got p={p_value:.4f}"

    def test_adf_returns_four_elements(self):
        s = make_price_series(n=300)
        result = adfuller(s.pct_change().dropna(), autolag="AIC")
        assert len(result) >= 4

    def test_adf_critical_values_present(self):
        s = make_price_series(n=300)
        result = adfuller(s.pct_change().dropna(), autolag="AIC")
        assert "5%" in result[4]

    def test_differenced_series_stationary(self):
        """First difference of price series should be stationary."""
        s = make_price_series(n=500)
        diff = s.diff().dropna()
        result = adfuller(diff, autolag="AIC")
        assert result[1] < 0.05


# ── Section 5: Risk Metrics ───────────────────────────────────────────────────

class TestRiskMetrics:

    def setup_method(self):
        s = make_price_series(n=500)
        self.returns = s.pct_change().dropna()

    def test_var_95_is_negative(self):
        var_95 = np.percentile(self.returns, 5)
        assert var_95 < 0, "VaR at 95% confidence should be a negative number"

    def test_var_99_more_extreme_than_var_95(self):
        var_95 = np.percentile(self.returns, 5)
        var_99 = np.percentile(self.returns, 1)
        assert var_99 < var_95

    def test_sharpe_ratio_is_finite(self):
        risk_free = 0.05 / 252
        excess = self.returns - risk_free
        sharpe = (excess.mean() / excess.std()) * np.sqrt(252)
        assert np.isfinite(sharpe)

    def test_annualized_volatility_positive(self):
        ann_vol = self.returns.std() * np.sqrt(252)
        assert ann_vol > 0

    def test_annualized_return_finite(self):
        ann_return = self.returns.mean() * 252
        assert np.isfinite(ann_return)

    def test_var_within_return_range(self):
        var_95 = np.percentile(self.returns, 5)
        assert self.returns.min() <= var_95 <= self.returns.max()

    def test_sharpe_higher_for_better_asset(self):
        """Asset with higher mean and same std should have higher Sharpe."""
        risk_free = 0.05 / 252
        np.random.seed(1)
        good = pd.Series(np.random.normal(0.001, 0.01, 500))
        bad  = pd.Series(np.random.normal(0.0001, 0.01, 500))
        sharpe_good = ((good - risk_free).mean() / (good - risk_free).std()) * np.sqrt(252)
        sharpe_bad  = ((bad  - risk_free).mean() / (bad  - risk_free).std()) * np.sqrt(252)
        assert sharpe_good > sharpe_bad
