# Portfolio Optimization

![Unit Tests](https://github.com/EphrataTech/portfolio-optimization/actions/workflows/unittests.yml/badge.svg)

A data-driven portfolio management project built for **GMF Investments**, applying time series forecasting to historical financial data to enhance investment strategies. The project covers end-to-end analysis — from data extraction and EDA through to predictive modeling using ARIMA/SARIMA and LSTM.

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
│       └── unittests.yml       # CI pipeline (GitHub Actions)
├── .vscode/
│   └── settings.json
├── data/
│   └── processed/              # Cleaned CSVs and saved visualizations
├── notebooks/
│   ├── task-1.ipynb            # EDA, cleaning, stationarity, risk metrics
│   ├── task-2.ipynb            # ARIMA/SARIMA and LSTM forecasting models
│   └── interim_report.md       # Written interim report for Task 1
├── src/
│   └── __init__.py
├── tests/
│   ├── test_task1.py           # Unit tests for Task 1
│   └── test_task2.py           # Unit tests for Task 2
├── scripts/
│   └── __init__.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tasks

### Task 1 — Preprocess and Explore the Data (`task-1` branch)

**Notebook:** `notebooks/task-1.ipynb`

- Extract historical data for TSLA, BND, SPY using YFinance
- Clean data: handle missing values (ffill/bfill), enforce correct dtypes
- Exploratory Data Analysis:
  - Adjusted closing prices over time
  - Daily percentage returns and volatility
  - 30-day rolling mean and standard deviation bands
  - Return distributions and correlation heatmap
  - Seasonal decomposition (monthly, multiplicative)
- Stationarity testing using the Augmented Dickey-Fuller (ADF) test
- Risk metrics: Value at Risk (VaR 95%), Sharpe Ratio, annualized volatility

**Key findings:**
- TSLA annualized volatility ~50% vs SPY ~17% vs BND ~5%
- All closing price series are non-stationary; all return series are stationary
- BND provides genuine diversification (near-zero correlation with equities)

**Interim Report:** `notebooks/interim_report.md`

---

### Task 2 — Build Time Series Forecasting Models (`task-2` branch)

**Notebook:** `notebooks/task-2.ipynb`

- Chronological train/test split: train on 2015–2024, test on 2025–2026
- **ARIMA/SARIMA:**
  - ACF/PACF analysis on differenced series
  - Parameter search via `auto_arima` (pmdarima)
  - Fit and forecast using `SARIMAX`
- **LSTM:**
  - 60-day sliding window sequences
  - 2-layer LSTM (64 → 32 units) with Dropout and EarlyStopping
  - MinMaxScaler normalization with inverse transform for evaluation
- Model evaluation: MAE, RMSE, MAPE
- Comparative analysis of all models

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/EphrataTech/portfolio-optimization.git
cd portfolio-optimization
pip install -r requirements.txt
```

### Run Notebooks

```bash
jupyter notebook notebooks/task-1.ipynb
jupyter notebook notebooks/task-2.ipynb
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
| `main` | Base project structure |
| `task-1` | Data preprocessing, EDA, stationarity, risk metrics |
| `task-2` | ARIMA/SARIMA and LSTM forecasting models |

---

## CI/CD

GitHub Actions runs unit tests automatically on every push and pull request via `.github/workflows/unittests.yml`. Tests cover:

- Data extraction and cleaning logic
- EDA computations (rolling stats, returns, correlations)
- Stationarity test behavior
- Risk metric calculations (VaR, Sharpe Ratio)
- Train/test split integrity (no data leakage)
- LSTM sequence preparation and scaler correctness
- Model evaluation metric properties (MAE, RMSE, MAPE)

---

## License

This project is for educational and analytical purposes at GMF Investments.
