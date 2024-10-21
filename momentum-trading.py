import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download historical data from Yahoo Finance
ticker = 'SPY'
data = yf.download(ticker, start='2015-01-01', end='2023-04-07')

# Calculate the 10-day and 50-day rate of change (ROC)
data['10d_roc'] = data['Close'].pct_change(periods=10)
data['50d_roc'] = data['Close'].pct_change(periods=50)

# Create a new column called "position" that signals buy/sell orders based on the momentum
data['position'] = 0
data.loc[(data['10d_roc'] > 0) & (data['50d_roc'] > 0), 'position'] = 1
data.loc[(data['10d_roc'] < 0) & (data['50d_roc'] < 0), 'position'] = -1

# Set up initial capital and position size
capital = 10000
position_size = capital / len(data)

# Calculate daily returns
data['daily_returns'] = data['Close'].pct_change()

# Calculate strategy returns
data['strategy_returns'] = data['position'].shift(1) * data['daily_returns'] * position_size

# Calculate cumulative returns
data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod() * capital

# Plot cumulative returns
plt.figure(figsize=(15,8))
plt.plot(data['cumulative_returns'], 'b')
plt.title('Momentum Trading Strategy on S&P 500')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns $')
plt.show()