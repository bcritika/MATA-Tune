import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
# Download stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2015-01-01', end='2020-01-01')
data.reset_index(inplace=True)
data['Close'] = data['Close'].astype(float)

# Parameters for fitting
window_size = 11  # Use an odd number to center the window around each point
half_window = window_size // 2

# Initialize columns for quadratic fit results
data['quad_extreme'] = np.nan
data['extreme_type'] = np.nan  # To store if it's a max or min

# Loop over data with sliding window
for i in range(half_window, len(data) - half_window):
    # Extract the windowed data
    window_data = data['Close'][i - half_window:i + half_window + 1]
    x_vals = np.arange(len(window_data))
    
    # Fit a quadratic (degree 2 polynomial) to the windowed data
    coeffs = np.polyfit(x_vals, window_data, 2)  # [a, b, c]
    
    # Vertex (extreme point) x = -b / (2a)
    a, b, _ = coeffs
    vertex_x = -b / (2 * a)
    
    # Check if vertex is within the window
    if 0 <= vertex_x < len(window_data):
        # Calculate the extreme value at the vertex
        vertex_y = np.polyval(coeffs, vertex_x)
        data.at[i, 'quad_extreme'] = vertex_y
        
        # Determine if it's a max or min based on 'a' coefficient
        data.at[i, 'extreme_type'] = 'max' if a < 0 else 'min'

# Display data with quadratic extremes
print(data[['Date', 'Close', 'quad_extreme', 'extreme_type']].dropna(subset=['quad_extreme']))

plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Close Price', color='blue', alpha=0.6)

# Plot detected extremes
max_points = data[data['extreme_type'] == 'max']
min_points = data[data['extreme_type'] == 'min']
plt.scatter(max_points['Date'], max_points['quad_extreme'], color='red', marker='^', label='Local Max (Quadratic Fit)', alpha=0.8)

plt.scatter(min_points['Date'], min_points['quad_extreme'], color='green', marker='v', label='Local Min (Quadratic Fit)', alpha=0.8)


plt.title(f"{ticker} Stock Price with Quadratic Fit Extremes")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

cup_handle_patterns = []
for i in range(len(data) - window_size):
    # Check for a "cup" pattern
    if data['extreme_type'].iloc[i] == 'max' and data['extreme_type'].iloc[i + half_window] == 'min':
        left_max = data['quad_extreme'].iloc[i]
        cup_min = data['quad_extreme'].iloc[i + half_window]
        right_max = data['quad_extreme'].iloc[i + window_size - 1]
        
        # Check if the right side returns to a similar max level, forming a cup
        if right_max >= 0.95 * left_max:  # Condition to confirm cup shape
            depth = left_max - cup_min  # Depth of the cup
            handle_range = data['Close'][i + half_window + 1:i + window_size - 1]
            
            # Check for handle consolidation
            if handle_range.max() < left_max and handle_range.min() > cup_min:
                # Calculate the target sell price
                sell_price = left_max + depth
                
                # Store the pattern information
                cup_handle_patterns.append({
                    'start_date': data['Date'].iloc[i],
                    'end_date': data['Date'].iloc[i + window_size - 1],
                    'left_max': left_max,
                    'cup_min': cup_min,
                    'sell_price': sell_price
                })

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Close Price', color='blue', alpha=0.6)

# Plot identified "Cup and Handle" patterns
for pattern in cup_handle_patterns:
    plt.axvspan(pattern['start_date'], pattern['end_date'], color='orange', alpha=0.2)
    plt.axhline(y=pattern['sell_price'], color='red', linestyle='--', label=f'Sell Target: {pattern["sell_price"]:.2f}')
    plt.scatter(data['Date'], data['quad_extreme'], color='purple', marker='o', label='Quadratic Extreme')

# Labels and legend
plt.title(f"{ticker} Stock Price with Cup and Handle Pattern Detection")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()