import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Download stock data
ticker = 'TRAK'
data = yf.download(ticker, start='2023-10-01', end='2024-10-01')
data.reset_index(inplace=True)
data['Close'] = data['Close'].astype(float)

# Normalize by dividing Close by a 50-day rolling average
data['Normalized_Close'] = data['Close'] / data['Close'].rolling(window=50).mean()

# Parameters for fitting
window_size = 11
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
                buy_price = cup_min  # Buy at the minimum of the cup
                buy_date = data['Date'].iloc[i + half_window]
                
                # Mark buy and sell points
                data.at[i + half_window, 'signal'] = 0  # Buy signal
                data.at[i + window_size - 1, 'signal'] = 1  # Sell signal
                
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

# Example function to handle signals
def handle_signal(price,date, signal_value):
    """
    Placeholder for signal handling logic. This could involve calling an API.
    """
    if signal_value == 1:
        print(f"[SELL] Action triggered on {date} for {price}")
    elif signal_value == 0:
        print(f"[BUY] Action triggered on {date} for {price}")
    else:
        print(f"[HOLD] No action for {date} for {price}")

# Apply signal detection and trigger API calls
for idx, row in data.iterrows():
    if row['signal'] in [0, 1]:  # Only take action on buy or sell signals
        handle_signal(row['Close'],row['Date'], row['signal'])

# Plotting
plt.figure(figsize=(14, 7))

# Plot normalized close price
plt.plot(data['Date'], data['Normalized_Close'], label='Normalized Close Price', color='blue', alpha=0.6)

# Plot quadratic extremes
max_points = data[data['extreme_type'] == 'max']
min_points = data[data['extreme_type'] == 'min']
plt.scatter(max_points['Date'], max_points['quad_extreme'], color='red', marker='^', label='Local Max (Quadratic Fit)', alpha=0.8)
plt.scatter(min_points['Date'], min_points['quad_extreme'], color='green', marker='v', label='Local Min (Quadratic Fit)', alpha=0.8)

# Highlight Cup and Handle patterns
for pattern in cup_handle_patterns:
    plt.axvspan(pattern['start_date'], pattern['end_date'], color='orange', alpha=0.2)
    plt.axhline(y=pattern['sell_price'], color='red', linestyle='--', label=f'Sell Target: {pattern["sell_price"]:.2f}')
    plt.scatter(pattern['sell_date'], pattern['sell_price'], color='black', marker='o', s=100, label='Sell Point')
    plt.scatter(pattern['buy_date'], pattern['buy_price'], color='green', marker='o', s=100, label='Buy Point')
    plt.text(pattern['buy_date'], pattern['buy_price'], f'Buy: {pattern["buy_price"]:.2f}', 
             verticalalignment='top', horizontalalignment='left', color='darkgreen', fontsize=8)

# Labels and legend
plt.title(f"{ticker} Stock Price with Cup and Handle Pattern Detection and Signals")
plt.xlabel("Date")
plt.ylabel("Normalized Close Price")
plt.legend()
plt.show()