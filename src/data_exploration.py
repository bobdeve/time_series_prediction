import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

plt.style.use('seaborn-v0_8-darkgrid')  # fallback if seaborn-darkgrid fails

# -------------------- Config --------------------
tickers = ['TSLA', 'BND', 'SPY']
start = '2015-07-01'
end = '2025-07-31'

# -------------------- Load and Clean Data --------------------
all_data = {}

for ticker in tickers:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    df.ffill(inplace=True)  # Fill missing data
    df['Ticker'] = ticker
    all_data[ticker] = df

# -------------------- Combine --------------------
combined_df = pd.concat(all_data.values(), axis=0)
combined_df.index.name = 'Date'

# -------------------- Explore Each Ticker --------------------
for ticker in tickers:
    print(f"\n--- {ticker} ---")
    df = all_data[ticker].copy()

    print(df.info())
    print(df.describe())
    print("Missing values:\n", df.isnull().sum())

    # Daily Return
    df['Daily Return'] = df['Adj Close'].pct_change()

    # Rolling volatility
    df['Rolling Mean'] = df['Adj Close'].rolling(window=21).mean()
    df['Rolling Std'] = df['Adj Close'].rolling(window=21).std()

    # ADF Test
    print(f"\nADF Test for {ticker} Adj Close:")
    result = adfuller(df['Adj Close'].dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Stationary" if result[1] < 0.05 else "Non-stationary")

    # -------------------- Plots --------------------
    plt.figure(figsize=(10, 4))
    df['Adj Close'].plot(title=f"{ticker} - Adjusted Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    df['Daily Return'].plot(title=f"{ticker} - Daily Return")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    df['Rolling Std'].plot(title=f"{ticker} - Rolling Volatility (21 days)")
    plt.xlabel("Date")
    plt.ylabel("Std Dev")
    plt.tight_layout()
    plt.show()

    # -------------------- Risk Metrics --------------------
    print(f"\nRisk Metrics for {ticker}:")
    var_95 = np.percentile(df['Daily Return'].dropna(), 5)
    sharpe = (df['Daily Return'].mean() / df['Daily Return'].std()) * np.sqrt(252)
    print(f"95% Value at Risk (VaR): {var_95:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
