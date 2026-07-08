# Portfolio Optimization

![Unit Tests](https://github.com/EphrataTech/portfolio-optimization-2/actions/workflows/unittests.yml/badge.svg)

A data-driven portfolio management project built for **GMF Investments**, applying time series forecasting and Modern Portfolio Theory to historical financial data to enhance investment strategies. The project covers end-to-end analysis — from data extraction and EDA through forecasting, portfolio optimization, and strategy backtesting.

---

## Business Context

GMF Investments is a financial advisory firm that leverages cutting-edge technology to provide clients with tailored investment strategies. This project integrates advanced time series forecasting to:

- Predict market trends and volatility
- Optimize asset allocation across risk profiles
- Enhance portfolio performance while managing downside risk

> Per the Efficient Market Hypothesis, these models are used to forecast volatility and identify momentum factors rather than as standalone price prediction tools.

---

## Assets Analyzed

| Ticker | Asset | Risk Profile |
|--------|-------|-------------|
| TSLA | Tesla Inc. | High risk, high potential return |
| BND | Vanguard Total Bond Market ETF | Low risk, stability and income |
| SPY | S&P 500 ETF | Moderate risk, broad market exposure |

**Data period:** January 1, 2015 – June 30, 2026 (sourced via [YFinance](https://pypi.org/project/yfinance/))

---

## Project Structure

```
portfolio-optimization/
├── .github/
│   └── workflows/
│       └── unittests.yml           # CI pipeline (GitHub Actions)
├── .vscode/
│   └── settings.json
├── data/
│   └── processed/                  # Cleaned CSVs and saved visualizations
├── notebooks/
│   ├── task-1.ipynb                # EDA, cleaning, stationarity, risk metrics
│   ├── task-2.ipynb                # ARIMA/SARIMA and LSTM forecasting models
│   ├── task-3.ipynb                # Future forecasting with confidence intervals
│   ├── task-4-5.ipynb              # Portfolio optimization and backtesting
│   └── interim_report.md           # Written interim report for Task 1
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data extraction and cleaning
│   ├── eda.py                      # EDA computations and visualizations
│   ├── stationarity.py             # ADF stationarity tests
│   ├── risk_metrics.py             # VaR, Sharpe Ratio, annualized metrics
│   ├── arima_model.py              # ARIMA/SARIMA fitting and forecasting
│   ├── lstm_model.py               # LSTM architecture, training, forecasting
│   ├── evaluate.py                 # MAE, RMSE, MAPE model evaluation
│   ├── forecast.py                 # 6-12 month future forecasting + CI
│   ├── portfolio_optimization.py   # MPT, efficient frontier, optimal portfolios
│   └── backtesting.py              # Strategy simulation and performance metrics
├── tests/
│   ├── __init__.py
│   ├── test_task1.py               # Unit tests for Task 1
│   ├── test_task2.py               # Unit tests for Task 2
│   ├── test_task3.py               # Unit tests for Task 3
│   └── test_task4_5.py             # Unit tests for Tasks 4 & 5
├── scripts/
│   └── __init__.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tasks

### Task 1 — Preprocess and Explore the Data (`task-1` branch)

**Notebook:** `notebooks/task-1.ipynb` | **Scripts:** `src/data_loader.py`, `src/eda.py`, `src/stationarity.py`, `src/risk_metrics.py`

- Extract historical data for TSLA, BND, SPY using YFinance
- Clean data: handle missing values (ffill/bfill), enforce correct dtypes
- Exploratory Data Analysis: closing prices, daily returns, rolling stats, distributions, correlation heatmap, seasonal decomposition
- Stationarity testing using the Augmented Dickey-Fuller (ADF) test
- Risk metrics: Value at Risk (VaR 95%), Sharpe Ratio, annualized volatility

**Key findings:**
- TSLA annualized volatility ~50% vs SPY ~17% vs BND ~5%
- All closing price series are non-stationary; all return series are stationary
- BND provides genuine diversification (near-zero correlation with equities)

**Interim Report:** `notebooks/interim_report.md`

---

### Task 2 — Build Time Series Forecasting Models (`task-2` branch)

**Notebook:** `notebooks/task-2.ipynb` | **Scripts:** `src/arima_model.py`, `src/lstm_model.py`, `src/evaluate.py`

- Chronological train/test split: train on 2015–2024, test on 2025–2026
- ARIMA/SARIMA: ACF/PACF analysis, `auto_arima` parameter search, SARIMAX fitting
- LSTM: 60-day sliding window, 2-layer architecture (64→32 units), Dropout, EarlyStopping
- Model evaluation: MAE, RMSE, MAPE comparison table

---

### Task 3 — Forecast Future Market Trends (`task-3` branch)

**Notebook:** `notebooks/task-3.ipynb` | **Script:** `src/forecast.py`

- 12-month future forecasts using best-performing model from Task 2
- ARIMA: `get_forecast()` with 95% confidence intervals
- LSTM: iterative multi-step forecasting with Monte Carlo Dropout uncertainty bounds
- CI width analysis over the forecast horizon
- Trend analysis, market opportunities and risks assessment
- Forecast reliability assessment by time horizon

---

### Task 4 — Optimize Portfolio Based on Forecast (`task-4` branch)

**Notebook:** `notebooks/task-4-5.ipynb` | **Script:** `src/portfolio_optimization.py`

- Expected returns: TSLA from forecast, BND/SPY from historical averages
- Annualized covariance matrix from historical daily returns
- Efficient Frontier via 10,000 Monte Carlo portfolio simulations
- Maximum Sharpe Ratio Portfolio and Minimum Volatility Portfolio (scipy optimization)
- Recommended portfolio with weights, expected return, volatility, and Sharpe Ratio

---

### Task 5 — Strategy Backtesting (`task-4` branch)

**Notebook:** `notebooks/task-4-5.ipynb` | **Script:** `src/backtesting.py`

- Backtest window: January 2025 – June 2026 (out-of-sample)
- Benchmark: static 60% SPY / 40% BND portfolio
- Simulations: buy-and-hold and monthly rebalancing
- Performance metrics: total return, annualized return, Sharpe Ratio, max drawdown
- Cumulative returns and drawdown comparison plots
- Written conclusion on strategy viability and backtest limitations

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/EphrataTech/portfolio-optimization-2.git
cd portfolio-optimization
pip install -r requirements.txt
```

### Run Notebooks

```bash
jupyter notebook notebooks/task-1.ipynb
jupyter notebook notebooks/task-2.ipynb
jupyter notebook notebooks/task-3.ipynb
jupyter notebook notebooks/task-4-5.ipynb
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Requirements

```
pytest
yfinance
pandas
numpy
matplotlib
seaborn
statsmodels
pmdarima
scikit-learn
tensorflow
keras
scipy
```

---

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Base project structure and README |
| `task-1` | Data preprocessing, EDA, stationarity, risk metrics |
| `task-2` | ARIMA/SARIMA and LSTM forecasting models |
| `task-3` | Future forecasting with confidence intervals |
| `task-4` | Portfolio optimization (MPT) and strategy backtesting |

---

## CI/CD

GitHub Actions runs unit tests automatically on every push and pull request via `.github/workflows/unittests.yml`. Tests cover:

- Data extraction and cleaning logic
- EDA computations (rolling stats, returns, correlations)
- Stationarity test behavior (ADF)
- Risk metric calculations (VaR, Sharpe Ratio)
- Train/test split integrity (no data leakage)
- LSTM sequence preparation and scaler correctness
- Model evaluation metric properties (MAE, RMSE, MAPE)
- Future date generation and CI width growth
- LSTM iterative forecasting and Monte Carlo Dropout
- MPT covariance matrix properties and portfolio performance
- Efficient Frontier weight constraints
- Backtesting cumulative returns and performance metrics

---

## License

This project is for educational and analytical purposes at GMF Investments.
