import yfinance as yf

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.dates import date2num, DateFormatter, WeekdayLocator, DayLocator, MONDAY
import seaborn as sns

import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc

from scipy import stats
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

import datetime
from datetime import date, timedelta

import warnings
warnings.filterwarnings('ignore')

# Function to calculate the M2 Money Flow Divergence
def money_flow_divergence(stock_data, m2_data, period='1M'):
    # Ensure that both stock_data and m2_data have a DatetimeIndex
    stock_data.index = pd.to_datetime(stock_data.index)
    m2_data.index = pd.to_datetime(m2_data.index)
    
    # Resample stock and M2 data to the specified period (e.g., monthly)
    stock_returns = stock_data['Adj Close'].resample(period).ffill().pct_change().dropna()
    m2_growth = m2_data['M2SL'].resample(period).ffill().pct_change().dropna()

    # Compute divergence: M2 Growth - Stock Growth
    divergence = m2_growth - stock_returns
    return divergence

# Function to simulate buy/sell signals based on the divergence
def backtest_strategy(divergence, stock_data, threshold=0.05):
    signals = []
    for date, diff in divergence.items():
        if diff > threshold:
            signals.append((date, 'Buy'))
        else:
            signals.append((date, 'Hold'))
    return signals

# Function to calculate total earnings between two dates
def calculate_total_earnings(stock_data, signals, start_date, end_date):
    total_return = 1  # Start with $1 for simplicity, we will multiply returns
    holding = False
    buy_price = None

    # Ensure we have data for every date by forward filling
    stock_data = stock_data.ffill()

    for date, signal in signals:
        # Only consider dates within the specified range
        if date < pd.to_datetime(start_date) or date > pd.to_datetime(end_date):
            continue

        if signal == 'Buy' and not holding:
            # Check if the date exists in stock_data
            if date in stock_data.index:
                buy_price = stock_data.loc[date]['Adj Close']
                holding = True
        elif holding:
            # Check if the date exists in stock_data
            if date in stock_data.index:
                sell_price = stock_data.loc[date]['Adj Close']
                total_return *= (sell_price / buy_price)
                holding = False  # Reset holding status

    # If still holding at the end date, sell at the last available price
    if holding:
        # Check if end_date exists in stock_data
        if pd.to_datetime(end_date) in stock_data.index:
            final_price = stock_data.loc[pd.to_datetime(end_date)]['Adj Close']
        else:
            final_price = stock_data.iloc[-1]['Adj Close']  # Use the last available price if end_date isn't found
        total_return *= (final_price / buy_price)

    return total_return - 1  # Return the profit/loss percentage

# Test on 3 selected stocks 
stocks = ['AAPL', 'MSFT', 'TSLA']
start_date = '2010-01-01'
end_date = '2023-01-01'
calc_start_date = '2015-01-01'  # Start of the calculation period
calc_end_date = '2020-01-01'    # End of the calculation period

# Download stock price data
stock_data = {stock: yf.download(stock, start=start_date, end=end_date) for stock in stocks}

# Load M2 money supply data from a CSV
m2_data = pd.read_csv('data/M2SL.csv')  # Ensure the M2 data is loaded with a 'Date' and 'M2SL' column
m2_data['DATE'] = pd.to_datetime(m2_data['DATE'])
m2_data.set_index('DATE', inplace=True)

# Iterate over each stock and backtest the strategy
for stock in stocks:
    print(f"Backtesting on {stock}")
    
    # Get the stock data and M2 divergence
    divergence = money_flow_divergence(stock_data[stock], m2_data)

    # Simulate strategy based on the calculated divergence
    signals = backtest_strategy(divergence, stock_data[stock])
    
    # Calculate total earnings between the calculation period
    total_earnings = calculate_total_earnings(stock_data[stock], signals, calc_start_date, calc_end_date)
    
    print(f"Total earnings for {stock} from {calc_start_date} to {calc_end_date}: {total_earnings * 100:.2f}%")
    
    # Optional: Plot the divergence and stock price
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data[stock]['Adj Close'], label=f'{stock} Price')
    plt.title(f'{stock} Price vs Money Flow Divergence')
    plt.legend()
    plt.show()
