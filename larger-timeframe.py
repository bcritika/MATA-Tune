import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Download stock data
ticker = 'TSLA'
data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
data.reset_index(inplace=True)
data['Close'] = data['Close'].astype(float)

# Normalize by dividing Close by a 200-day rolling average
data['Normalized_Close'] = data['Close'] / data['Close'].rolling(window=200).mean()

# Parameters for fitting
window_size = 50  # Larger window size for broader patterns
half_window = window_size // 2

# Initialize columns for quadratic fit results
data['quad_extreme'] = np.nan
data['extreme_type'] = np.nan

# Loop over data with sliding window for quadratic fit
for i in range(half_window, len(data) - half_window):
    # Extract windowed data on normalized close price
    window_data = data['Normalized_Close'][i - half_window:i + half_window + 1]
    x_vals = np.arange(len(window_data))
    
    # Fit a quadratic (degree 2 polynomial) to the windowed data
    coeffs = np.polyfit(x_vals, window_data, 2)
    a, b, _ = coeffs
    vertex_x = -b / (2 * a)  # Vertex of the parabola
    
    # Check if the vertex is within the window range
    if 0 <= vertex_x < len(window_data):
        vertex_y = np.polyval(coeffs, vertex_x)
        data.at[i, 'quad_extreme'] = vertex_y
        data.at[i, 'extreme_type'] = 'max' if a < 0 else 'min'

# Initialize signal column
data['signal'] = -1  # Default signal value

# Detect Cup and Handle patterns and set signals
cup_handle_patterns = []
for i in range(len(data) - window_size):
    if data['extreme_type'].iloc[i] == 'max' and data['extreme_type'].iloc[i + half_window] == 'min':
        left_max = data['quad_extreme'].iloc[i]
        cup_min = data['quad_extreme'].iloc[i + half_window]
        right_max = data['quad_extreme'].iloc[i + window_size - 1]
        
        if right_max >= 0.95 * left_max:  # Check cup shape
            depth = left_max - cup_min
            handle_range = data['Normalized_Close'][i + half_window + 1:i + window_size - 1]
            
            if handle_range.max() < left_max and handle_range.min() > cup_min:
                sell_price = left_max + depth
                sell_date = data['Date'].iloc[i + window_size - 1]
                
                # Mark buy and sell points
                data.at[i + half_window, 'signal'] = 0  # Buy signal
                data.at[i + window_size - 1, 'signal'] = 1  # Sell signal
                
                buy_price = left_max  # Resistance level
                buy_date = data['Date'].iloc[i + half_window]
                cup_handle_patterns.append({
                    'start_date': data['Date'].iloc[i],
                    'end_date': data['Date'].iloc[i + window_size - 1],
                    'left_max': left_max,
                    'cup_min': cup_min,
                    'sell_price': sell_price,
                    'sell_date': sell_date,
                    'buy_price': buy_price,
                    'buy_date': buy_date,
                })

# Simulating trading strategy
initial_balance = 100000
balance = initial_balance
stock_holding = 0
trading_log = []

for idx, row in data.iterrows():
    if row['signal'] == 0:  # Buy signal
        if balance > 0:  # Ensure we have cash to buy
            stock_holding = balance // row['Close']
            balance -= stock_holding * row['Close']
            trading_log.append({
                'date': row['Date'],
                'action': 'BUY',
                'price': row['Close'],
                'shares': stock_holding,
                'balance': balance
            })
    elif row['signal'] == 1:  # Sell signal
        if stock_holding > 0:  # Ensure we have stocks to sell
            balance += stock_holding * row['Close']
            trading_log.append({
                'date': row['Date'],
                'action': 'SELL',
                'price': row['Close'],
                'shares': stock_holding,
                'balance': balance
            })
            stock_holding = 0

# Final forced sell if stocks remain
if stock_holding > 0:
    balance += stock_holding * data['Close'].iloc[-1]
    trading_log.append({
        'date': data['Date'].iloc[-1],
        'action': 'FORCED SELL',
        'price': data['Close'].iloc[-1],
        'shares': stock_holding,
        'balance': balance
    })
    stock_holding = 0

# Final portfolio value
final_value = balance
print(f"Initial Balance: ${initial_balance:.2f}")
print(f"Final Portfolio Value: ${final_value:.2f}")
print(f"Profit: ${final_value - initial_balance:.2f}")

# Display trading log
trading_log_df = pd.DataFrame(trading_log)
print(trading_log_df)

# Plotting
plt.figure(figsize=(14, 7))

# Plot normalized close price
plt.plot(data['Date'], data['Normalized_Close'], label='Normalized Close Price', color='blue', alpha=0.6)

# Plot buy and sell signals
buy_signals = data[data['signal'] == 0]
sell_signals = data[data['signal'] == 1]
plt.scatter(buy_signals['Date'], buy_signals['Normalized_Close'], color='green', label='Buy Signal', marker='^', alpha=1)
plt.scatter(sell_signals['Date'], sell_signals['Normalized_Close'], color='red', label='Sell Signal', marker='v', alpha=1)

# Annotate detected Cup and Handle patterns
for pattern in cup_handle_patterns:
    plt.axvspan(pattern['start_date'], pattern['end_date'], color='yellow', alpha=0.3, label='Cup and Handle' if 'Cup and Handle' not in plt.gca().get_legend_handles_labels()[1] else "")

# Labels and legend
plt.title(f"{ticker} Stock Price with Cup and Handle Patterns")
plt.xlabel("Date")
plt.ylabel("Normalized Close Price")
plt.legend()
plt.show()
