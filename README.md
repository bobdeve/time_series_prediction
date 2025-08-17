Task 1: Data Loading, Preprocessing, and Exploratory Data Analysis
Data Loading and Preprocessing
Historical daily price and volume data for TSLA, BND, and SPY were fetched using the yfinance Python library, spanning from July 1, 2015, to July 31, 2025.

Data was cleaned to ensure correct data types, with missing values carefully checked and found to be negligible or non-existent.

Price data was normalized by calculating daily returns (percentage change) to standardize across assets for further analysis.

Exploratory Data Analysis and Financial Metrics
Closing prices and daily returns were visualized over time, with rolling means and rolling standard deviations computed to reveal trends and volatility clusters.

The Augmented Dickey-Fuller (ADF) test was performed on both closing prices and returns to assess stationarity, which is critical for subsequent time series modeling.

Key financial risk metrics were calculated:

Value at Risk (VaR) at 95% confidence level to quantify potential losses.

Sharpe Ratio to evaluate risk-adjusted returns.

Code Organization and Readability
The code is structured into clearly labeled sections for loading, cleaning, analysis, and visualization.

Descriptive variable names and comments ensure the workflow is easy to follow and reproducible.

Completion Status
All requirements outlined in Task 1 have been implemented.

This repository reflects a robust initial analysis that forms the foundation for the forecasting and portfolio optimization tasks to follow.


# Momentum Trading Strategy Analysis

## Overview
This project implements and evaluates a momentum-based trading strategy compared to a benchmark index. The analysis covers data preparation, strategy development, backtesting, and performance evaluation.

---

## Task 4: Data Preparation & Signal Generation
1. **Data Cleaning**  
   - Historical daily price data was used for both the strategy asset and the benchmark.
   - Missing values were handled to ensure continuous time series.
   
2. **Momentum Signal Calculation**  
   - Momentum was calculated using rolling returns over a chosen lookback period.
   - A binary trading signal was generated:
     - **1** → Buy (if momentum is positive)
     - **0** → Stay out of the market (if momentum is negative)

---

## Task 5: Strategy Backtesting
1. **Position Implementation**  
   - Strategy returns were calculated by applying the trading signal to the daily returns.
   - Benchmark returns were computed using a simple buy-and-hold approach.

2. **Performance Metrics**  
   - **Total Return**:
     - Strategy: `-83.39%`
     - Benchmark: `-87.53%`
   - **Sharpe Ratio**:
     - Strategy: `0.953`
     - Benchmark: `1.021`

3. **Cumulative Return Plot**  
   - A cumulative performance plot was generated to visually compare the strategy vs. the benchmark.  
   - ![Cumulative Performance Plot](cmulativeplot.png)

---

## Task 6: Risk-Adjusted Performance Evaluation
1. **Sharpe Ratio Analysis**  
   - The Sharpe ratio measures risk-adjusted returns.
   - Both the strategy and benchmark had positive Sharpe ratios, indicating positive returns relative to volatility, but overall returns were negative.

2. **Observations**  
   - While the strategy slightly reduced total loss compared to the benchmark, it did not produce positive returns.
   - The benchmark Sharpe ratio was marginally higher, suggesting better consistency despite larger drawdowns.

---

## Conclusion
- **Key Finding:** The momentum strategy underperformed in terms of generating profits, ending with an overall negative return, though it slightly outperformed the benchmark in loss reduction.
- **Sharpe Ratio Insight:** The benchmark had a slightly higher Sharpe ratio, implying more consistent performance despite heavier losses.
- **Recommendation:**  
  - Test different momentum lookback periods.
  - Combine momentum with other indicators (e.g., moving averages, volatility filters).
  - Explore risk management techniques like stop-loss orders to reduce drawdowns.

---

## How to Run
1. Clone the repository.
2. Install required Python libraries:
   ```bash
   pip install pandas numpy matplotlib


# Time Series Forecasting & Portfolio Optimization (Tasks 1–5)

## Overview

This repository implements a full workflow for time series forecasting and portfolio optimization for three assets — **TSLA**, **BND**, and **SPY** — and validates the results with backtesting.

The project covers:
- Task 1 — Data loading, cleaning, and exploratory analysis (EDA).
- Task 2 — Time series models: ARIMA (statsmodels / pmdarima) and LSTM (Keras/TensorFlow).
- Task 3 — Forecast generation and analysis (6–12 month horizon).
- Task 4 — Portfolio optimization (Efficient Frontier, max-Sharpe and min-volatility portfolios).
- Task 5 — Strategy backtesting vs benchmark (60% SPY / 40% BND).

> **Recommended Python version:** 3.10.x (3.13 caused binary incompatibility for some packages in this project)

---

## Repo structure (recommended)

time_series_prediction/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│ ├─ README.md # explain what should be here (no CSVs committed)
│ ├─ combined_data.csv # (NOT committed if large — keep out via .gitignore)
│ └─ outputs/ # forecasts, models, backtest csvs (not committed)
├─ notebooks/
│ ├─ 01_data_exploration.ipynb
│ ├─ 02_forecasting_models.ipynb
│ ├─ 03_forecast_analysis.ipynb
│ ├─ 04_portfolio_optimization.ipynb
│ └─ 05_backtesting.ipynb
├─ src/
│ ├─ data_utils.py # load/save, preprocess functions
│ ├─ eda_utils.py # plots, rolling stats, adf test helper
│ ├─ models.py # ARIMA/LSTM wrappers, training & predict
│ ├─ optimize.py # efficient frontier & optimization helpers
│ └─ backtest.py # backtest simulation functions
├─ scripts/
│ ├─ run_all.sh # optional script to run pipeline
│ └─ export_results.py # command-line exporting of outputs
└─ reports/
├─ interim_report.md
└─ figures/
└─ cumulative_plot.png


Download data (if not committed):

You can use the notebooks/01_data_exploration.ipynb to fetch data via yfinance and save to data/combined_data.csv. The notebook contains the exact snippet:

import yfinance as yf
tickers = ['TSLA','BND','SPY']
df = yf.download(tickers, start='2015-07-01', end='2025-07-31')['Close']
df.columns = ['TSLA','BND','SPY']
df.to_csv("data/combined_data.csv")
Open notebooks in order in Jupyter and run:


jupyter lab    # or jupyter notebook
# then open notebooks/01_data_exploration.ipynb and run cells in order

Files of interest (short descriptions)
notebooks/01_data_exploration.ipynb — Task 1 work: data download, cleaning, EDA, ADF tests, VaR, Sharpe.

notebooks/02_forecasting_models.ipynb — Task 2: ARIMA and LSTM model implementations, training, evaluation.

notebooks/03_forecast_analysis.ipynb — Task 3: Forecast generation, confidence intervals, plots.

notebooks/04_portfolio_optimization.ipynb — Task 4: expected returns vector, covariance matrix, efficient frontier, identify max-Sharpe & min-vol.

notebooks/05_backtesting.ipynb — Task 5: backtesting strategy vs benchmark, cumulative returns, sharpe, drawdown.

src/ — modular functions used in notebooks so work is reproducible.

Key results (from my runs)
ARIMA best model: ARIMA(0,1,0) (AIC reported from stepwise search)

Model comparison (test set):

ARIMA → MAE: 64.79, RMSE: 81.17, MAPE: 23.32%

LSTM → MAE: 75.60, RMSE: 94.63, MAPE: 26.43%

Optimization (Task 4):

Max Sharpe portfolio weights ≈ [TSLA 0.00, BND 0.12, SPY 0.87]

Min volatility weights ≈ [TSLA 0.00, BND 0.95, SPY 0.05]

Backtest (Task 5):

Backtest period: 2024-08-01 → 2025-07-30 (249 trading days)

Strategy total return: -83.39%; Sharpe: 0.953

Benchmark total return: -87.53%; Sharpe: 1.021

Notes & troubleshooting
Use Python 3.10 for best compatibility. Some packages (pmdarima, old tensorflow wheels) may fail on newer Python versions.

If pmdarima fails to install:

Ensure numpy is installed first (pip install numpy),

Or remove pmdarima from requirements.txt and rely on statsmodels ARIMA.

LSTM requires TensorFlow. If you only need ARIMA, you can skip TF/Keras.

Checklist before submission
 data/ contains your saved combined_data.csv (if your repo policy allows; otherwise keep in local and add instructions).

 notebooks/ run start-to-finish (restart kernel and run all) without errors.

 requirements.txt present and installs on a clean venv (Python 3.10).

 README.md updated with final results and instructions.

 .gitignore prevents large data files committed.


 4) Minimal src module outlines (create these files)
src/data_utils.py


import pandas as pd

def load_combined(path="data/combined_data.csv"):
    return pd.read_csv(path, index_col=0, parse_dates=True)
src/eda_utils.py


import matplotlib.pyplot as plt
import pandas as pd

def plot_close(df, ticker):
    df[ticker].plot(title=f"{ticker} Close")
src/optimize.py


import numpy as np

def portfolio_return(weights, returns):
    return np.dot(weights, returns)

def portfolio_vol(weights, cov):
    return np.sqrt(weights.T @ cov @ weights)
src/backtest.py


import pandas as pd
def run_holdbacktest(prices, weights, initial_capital=100000):
    returns = prices.pct_change().dropna()
    port_ret = returns.dot(weights)
    cum = (1+port_ret).cumprod()*initial_capital
    return port_ret, cum
These helpers keep notebooks clean and readable.

5) Quick Git commands (create branch & push)
From repo root:

# create branch
git checkout -b task-4

# add all changes
git add .

# commit
git commit -m "Task 4: portfolio optimization results and notebooks"

# push branch
git push origin task-4
For Task 5:


git checkout -b task-5
git add notebooks/05_backtesting.ipynb src/backtest.py data/backtest_summary.csv
git commit -m "Task 5: backtesting - results and notebooks"
git push origin task-5
6) Final notes before submission
Confirm you do not commit large raw data files if not allowed. Instead include data/README.md describing how to re-download via the notebook.

Double-check notebooks: Kernel → Restart & Run All (in Jupyter) to ensure reproducibility from a clean state.

Include a short reports/interim_report.md that summarizes Tasks 1–5 (you already have content we prepared — paste it there).


# 📈 Capstone Project: Portfolio Risk & Performance Analysis

## 🔍 Overview
This project applies **quantitative finance methods** to evaluate portfolio performance and risk.  
It demonstrates reliability and risk reduction through **robust engineering and reproducible analysis**.

The primary goals were:
- Assess **portfolio returns and volatility** over time.  
- Apply **statistical tests** (ADF) to evaluate stationarity.  
- Calculate **Value-at-Risk (VaR)** and **Sharpe Ratios** as core risk metrics.  
- Build a structured, maintainable, and finance-focused pipeline.  

---

## 🏦 Business Problem
In modern financial markets, investors face **uncertainty and hidden risk exposure**.  
Our business objective:  
- Quantify **risk-adjusted returns**.  
- Provide **transparent insights** into volatility, downside risk, and potential business impact.  

**Impact**: This analysis framework supports **better portfolio allocation decisions** and reduces unexpected downside risks.

---

## ⚙️ Project Workflow

### 1. Data Loading & Preprocessing
- Historical daily price data (2015–2025) from `yfinance`.  
- Cleaning: checked for missing values, aligned timeframes, ensured consistent datatypes.  

### 2. Exploratory Data Analysis
- Calculated **daily returns** and visualized time series.  
- Performed **rolling mean & rolling volatility** analysis.  

### 3. Statistical Testing
- **ADF Test** to check for stationarity.  
- Results used to determine model validity for time series forecasting.  

### 4. Risk & Performance Metrics
- **Cumulative Return**: 514.98%  
- **Average Daily Return**: 0.0008  
- **Volatility (σ)**: 0.0148  
- **Sharpe Ratio**: 0.06  
- **Value-at-Risk (VaR)** calculated at 95% and 99% confidence levels.  

---

## 📊 Results & Insights
- Portfolio achieved **significant cumulative growth (514%)**, but **low Sharpe Ratio (0.06)** shows limited risk-adjusted efficiency.  
- **Volatility (1.48%) daily** indicates moderate risk exposure.  
- **VaR analysis** quantified potential losses at different confidence intervals, supporting better downside protection.  

---

## 🚀 How to Reproduce

### Installation
```bash
git clone https://github.com/bobdeve/time_series_prediction.git
cd time_series_prediction
pip install -r requirements.txt
