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
