# Support and Reistance Levels 8

import pandas as pd
import yfinance as yf
import numpy as np
import math
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt


def get_stock_data(stock_symbol):
  data = yf.download(stock_symbol, start='2023-01-01', threads= False)
  data['Date'] = pd.to_datetime(data.index)
  data['Date'] = data['Date'].apply(mpl_dates.date2num)
  data = data.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
  return data


def is_far_from_level(value, levels, data):    
  ave =  np.mean(data['High'] - data['Low'])    
  return np.sum([abs(value-level)<ave for _,level in levels])==0

def plot_all(levels, data):    
  fig, ax = plt.subplots(figsize=(15, 10))   
  candlestick_ohlc(ax,data.values,width=0.6, colorup='green', colordown='red', alpha=0.8)    
  date_format = mpl_dates.DateFormatter('%d %b %Y')
  
  ax.xaxis.set_major_formatter(date_format)    
  for level in levels:        
    plt.hlines(level[1], xmin = data['Date'][level[0]], xmax = max(data['Date']), colors='m', linestyle='--')
  plt.plot(data['Close'], 'k')
  plt.ylabel('Price $') 
  plt.xlabel('Time') 
  plt.title("S&P 500 Support and Resistance Levels")    
  fig.show()

stock_symbol = 'SPY'
data = get_stock_data(stock_symbol)

pivot_levels = []
maxi_list = []
mini_list = []
for i in range(5, len(data)-5):
  # taking a window of 9 candles
  high_range = data['High'][i-5:i+4]
  current_maxi = high_range.max()
  # if we find a new maximum value, empty the maxi_list 
  if current_maxi not in maxi_list:
    maxi_list = []
  maxi_list.append(current_maxi)
  # if the maximum value remains the same after shifting 5 times
  if len(maxi_list)==5 and is_far_from_level(current_maxi,pivot_levels,data):
      pivot_levels.append((high_range.idxmax(), current_maxi))
    
  low_range = data['Low'][i-5:i+5]
  current_mini = low_range.min()
  if current_mini not in mini_list:
    mini_list = []
  mini_list.append(current_mini)
  if len(mini_list)==5 and is_far_from_level(current_mini,pivot_levels,data):
    pivot_levels.append((low_range.idxmin(), current_mini))
print(pivot_levels)
plot_all(pivot_levels, data)