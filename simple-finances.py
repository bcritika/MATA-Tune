import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Fetch historical data for Apple Inc.
ticker = '^GSPC'
data = yf.download(ticker)

# Display the first few rows of the dataset
print(data.tail())

# Calculate moving averages
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()

# Drop NaN values
data = data.dropna()

# Define features and target
X = data[['Close', 'MA_10', 'MA_50']]
y = data['Close'].shift(-1).dropna()

# Align X and y (to handle the shift properly)
X = X[:-1]

# Split data into training and testing sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("X Train:",X_train.head(10))
print("Y Train:", y_train.head(10))

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Plot Actual vs. Predicted Stock Prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Actual Price')
plt.plot(y_test.index, predictions, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()

# Simulate a trading strategy
initial_balance = 1000  # Starting balance in USD
balance = initial_balance
position = 0  # Number of shares

print("Predictions:", predictions[0:5])
print("X_test:", X_test[0:5])

for i in range(len(X_test)):
    current_price = X_test.iloc[i]['Close']
    predicted_price = predictions[i]
    date = X_test.index[i].strftime('%Y-%m-%d')  # Extract date from index

    if predicted_price > current_price and balance >= current_price:
        # Buy stock
        shares_to_buy = int(balance // current_price)  # Buy whole shares only
        if shares_to_buy > 0:  # Ensure we are buying at least one share
            position += shares_to_buy
            balance -= shares_to_buy * current_price
            print(f"Buying {shares_to_buy} shares at {current_price:.2f} on {date}")

    elif predicted_price < current_price and position > 0:
        # Sell stock
        balance += position * current_price
        print(f"Selling {position} shares at {current_price:.2f} on {date}")
        position = 0

# Calculate final balance including the value of the remaining shares
final_balance = balance + (position * X_test.iloc[-1]['Close'])
profit = final_balance - initial_balance
print(f"Final balance: ${final_balance:.2f}")
print(f"Profit: ${profit:.2f}")
