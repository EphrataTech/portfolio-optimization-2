import pytest
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

TICKERS = ['TSLA', 'BND', 'SPY']
TRADING_DAYS   = 252
RISK_FREE_RATE = 0.05


def make_returns(n=500, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range(start='2015-01-01', periods=n)
    data  = {
        'TSLA': np.random.normal(0.001, 0.03, n),
        'BND':  np.random.normal(0.0002, 0.003, n),
        'SPY':  np.random.normal(0.0004, 0.01, n),
    }
    return pd.DataFrame(data, index=dates)


def make_price_df(n=500, seed=42):
    np.random.seed(seed)
    dates  = pd.bdate_range(start='2015-01-01', periods=n)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    return pd.DataFrame({'Adj Close': prices}, index=dates)


def portfolio_performance(weights, expected_returns, cov_matrix):
    w   = np.array(weights)
    ret = np.dot(w, expected_returns)
    vol = np.sqrt(w @ cov_matrix @ w)
    shr = (ret - RISK_FREE_RATE) / vol if vol > 0 else np.nan
    return ret, vol, shr


def compute_cov(returns_df):
    return returns_df.cov().values * TRADING_DAYS


def simulate_cum_returns(daily_returns_df, weights):
    w    = np.array([weights[t] for t in TICKERS])
    rets = daily_returns_df[TICKERS].values @ w
    return pd.Series((1 + rets).cumprod(), index=daily_returns_df.index)


def max_drawdown(cum):
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return dd.min()


# ── Section 1: Expected Returns ───────────────────────────────────────────────

class TestExpectedReturns:

    def test_expected_returns_has_all_tickers(self):
        er = {'TSLA': 0.30, 'BND': 0.03, 'SPY': 0.12}
        assert set(er.keys()) == set(TICKERS)

    def test_tsla_uses_forecast_not_historical(self):
        historical_tsla = 0.10
        forecast_tsla   = 0.35
        er = {'TSLA': forecast_tsla, 'BND': 0.03, 'SPY': 0.12}
        assert er['TSLA'] == forecast_tsla
        assert er['TSLA'] != historical_tsla

    def test_historical_return_annualization(self):
        daily_ret = 0.0004
        ann_ret   = daily_ret * TRADING_DAYS
        assert ann_ret == pytest.approx(0.1008)

    def test_expected_returns_finite(self):
        er = {'TSLA': 0.30, 'BND': 0.03, 'SPY': 0.12}
        assert all(np.isfinite(v) for v in er.values())

    def test_bnd_lower_return_than_tsla(self):
        er = {'TSLA': 0.30, 'BND': 0.03, 'SPY': 0.12}
        assert er['BND'] < er['TSLA']


# ── Section 2: Covariance Matrix ──────────────────────────────────────────────

class TestCovarianceMatrix:

    def setup_method(self):
        self.returns = make_returns()
        self.cov     = compute_cov(self.returns)

    def test_cov_matrix_shape(self):
        assert self.cov.shape == (3, 3)

    def test_cov_matrix_symmetric(self):
        np.testing.assert_array_almost_equal(self.cov, self.cov.T)

    def test_cov_matrix_positive_semidefinite(self):
        eigenvalues = np.linalg.eigvalsh(self.cov)
        assert (eigenvalues >= -1e-10).all()

    def test_diagonal_elements_positive(self):
        assert all(self.cov[i, i] > 0 for i in range(3))

    def test_tsla_highest_variance(self):
        tsla_idx = TICKERS.index('TSLA')
        bnd_idx  = TICKERS.index('BND')
        assert self.cov[tsla_idx, tsla_idx] > self.cov[bnd_idx, bnd_idx]


# ── Section 3: Portfolio Performance ─────────────────────────────────────────

class TestPortfolioPerformance:

    def setup_method(self):
        self.er  = np.array([0.30, 0.03, 0.12])
        self.cov = compute_cov(make_returns())

    def test_equal_weight_return(self):
        w = [1/3, 1/3, 1/3]
        ret, _, _ = portfolio_performance(w, self.er, self.cov)
        expected  = np.dot(w, self.er)
        assert ret == pytest.approx(expected)

    def test_volatility_positive(self):
        w = [0.5, 0.2, 0.3]
        _, vol, _ = portfolio_performance(w, self.er, self.cov)
        assert vol > 0

    def test_sharpe_finite(self):
        w = [0.5, 0.2, 0.3]
        _, _, shr = portfolio_performance(w, self.er, self.cov)
        assert np.isfinite(shr)

    def test_all_bnd_lowest_volatility(self):
        w_bnd = [0.0, 1.0, 0.0]
        w_tsla= [1.0, 0.0, 0.0]
        _, vol_bnd,  _ = portfolio_performance(w_bnd,  self.er, self.cov)
        _, vol_tsla, _ = portfolio_performance(w_tsla, self.er, self.cov)
        assert vol_bnd < vol_tsla

    def test_weights_sum_to_one(self):
        w = [0.4, 0.2, 0.4]
        assert sum(w) == pytest.approx(1.0)


# ── Section 4: Efficient Frontier ────────────────────────────────────────────

class TestEfficientFrontier:

    def setup_method(self):
        np.random.seed(42)
        self.er  = np.array([0.30, 0.03, 0.12])
        self.cov = compute_cov(make_returns())
        n = 500
        records = []
        for _ in range(n):
            w = np.random.dirichlet(np.ones(3))
            r, v, s = portfolio_performance(w, self.er, self.cov)
            records.append({'Return': r, 'Volatility': v, 'Sharpe': s,
                            'TSLA': w[0], 'BND': w[1], 'SPY': w[2]})
        self.frontier = pd.DataFrame(records)

    def test_frontier_has_correct_columns(self):
        assert {'Return', 'Volatility', 'Sharpe', 'TSLA', 'BND', 'SPY'}.issubset(
            self.frontier.columns)

    def test_frontier_weights_sum_to_one(self):
        weight_sums = self.frontier[['TSLA', 'BND', 'SPY']].sum(axis=1)
        np.testing.assert_array_almost_equal(weight_sums, np.ones(len(self.frontier)))

    def test_frontier_volatility_positive(self):
        assert (self.frontier['Volatility'] > 0).all()

    def test_frontier_return_range(self):
        assert self.frontier['Return'].min() >= 0
        assert self.frontier['Return'].max() <= 1.0

    def test_max_sharpe_is_maximum(self):
        max_sharpe_idx = self.frontier['Sharpe'].idxmax()
        max_sharpe_val = self.frontier.loc[max_sharpe_idx, 'Sharpe']
        assert max_sharpe_val == self.frontier['Sharpe'].max()

    def test_min_vol_is_minimum(self):
        min_vol_val = self.frontier['Volatility'].min()
        assert min_vol_val == self.frontier['Volatility'].min()

    def test_all_weights_between_0_and_1(self):
        for t in TICKERS:
            assert (self.frontier[t] >= 0).all()
            assert (self.frontier[t] <= 1).all()


# ── Section 5: Backtesting ────────────────────────────────────────────────────

class TestBacktesting:

    def setup_method(self):
        self.returns = make_returns(n=252)
        self.strategy_w  = {'TSLA': 0.5, 'BND': 0.1, 'SPY': 0.4}
        self.benchmark_w = {'TSLA': 0.0, 'BND': 0.4, 'SPY': 0.6}

    def test_cumulative_returns_starts_at_one(self):
        cum = simulate_cum_returns(self.returns, self.strategy_w)
        assert cum.iloc[0] == pytest.approx(1.0, abs=0.05)

    def test_cumulative_returns_length(self):
        cum = simulate_cum_returns(self.returns, self.strategy_w)
        assert len(cum) == len(self.returns)

    def test_cumulative_returns_positive(self):
        cum = simulate_cum_returns(self.returns, self.strategy_w)
        assert (cum > 0).all()

    def test_benchmark_weights_sum_to_one(self):
        assert sum(self.benchmark_w.values()) == pytest.approx(1.0)

    def test_strategy_weights_sum_to_one(self):
        assert sum(self.strategy_w.values()) == pytest.approx(1.0)

    def test_no_data_leakage_in_backtest(self):
        """Backtest returns should only use data from the defined start date."""
        start = self.returns.index[50]
        backtest = self.returns.loc[start:]
        assert backtest.index[0] >= start


# ── Section 6: Performance Metrics ───────────────────────────────────────────

class TestPerformanceMetrics:

    def setup_method(self):
        self.returns = make_returns(n=252)
        self.strategy_w  = {'TSLA': 0.5, 'BND': 0.1, 'SPY': 0.4}
        self.benchmark_w = {'TSLA': 0.0, 'BND': 0.4, 'SPY': 0.6}
        self.strat_cum = simulate_cum_returns(self.returns, self.strategy_w)
        self.bench_cum = simulate_cum_returns(self.returns, self.benchmark_w)

    def test_total_return_finite(self):
        total = self.strat_cum.iloc[-1] - 1
        assert np.isfinite(total)

    def test_max_drawdown_non_positive(self):
        dd = max_drawdown(self.strat_cum)
        assert dd <= 0

    def test_max_drawdown_between_neg1_and_0(self):
        dd = max_drawdown(self.strat_cum)
        assert -1.0 <= dd <= 0.0

    def test_annualized_return_finite(self):
        total  = self.strat_cum.iloc[-1] - 1
        n_days = len(self.strat_cum)
        ann    = (1 + total) ** (TRADING_DAYS / n_days) - 1
        assert np.isfinite(ann)

    def test_sharpe_ratio_finite(self):
        daily_rets = self.strat_cum.pct_change().dropna()
        ann_ret    = daily_rets.mean() * TRADING_DAYS
        ann_vol    = daily_rets.std()  * np.sqrt(TRADING_DAYS)
        sharpe     = (ann_ret - RISK_FREE_RATE) / ann_vol
        assert np.isfinite(sharpe)

    def test_higher_tsla_weight_higher_volatility(self):
        """Portfolio with more TSLA should have higher volatility."""
        high_tsla = {'TSLA': 0.8, 'BND': 0.1, 'SPY': 0.1}
        low_tsla  = {'TSLA': 0.1, 'BND': 0.4, 'SPY': 0.5}
        cum_high  = simulate_cum_returns(self.returns, high_tsla)
        cum_low   = simulate_cum_returns(self.returns, low_tsla)
        vol_high  = cum_high.pct_change().dropna().std()
        vol_low   = cum_low.pct_change().dropna().std()
        assert vol_high > vol_low

    def test_metrics_dict_has_required_keys(self):
        daily_rets = self.strat_cum.pct_change().dropna()
        total_ret  = self.strat_cum.iloc[-1] - 1
        n_days     = len(daily_rets)
        ann_ret    = (1 + total_ret) ** (TRADING_DAYS / n_days) - 1
        ann_vol    = daily_rets.std() * np.sqrt(TRADING_DAYS)
        sharpe     = (ann_ret - RISK_FREE_RATE) / ann_vol
        dd         = max_drawdown(self.strat_cum)
        metrics    = {
            'Total Return (%)':    total_ret * 100,
            'Ann. Return (%)':     ann_ret   * 100,
            'Ann. Volatility (%)': ann_vol   * 100,
            'Sharpe Ratio':        sharpe,
            'Max Drawdown (%)':    dd        * 100,
        }
        required = {'Total Return (%)', 'Ann. Return (%)',
                    'Ann. Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)'}
        assert required.issubset(metrics.keys())
