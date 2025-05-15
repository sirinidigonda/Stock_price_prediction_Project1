
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# Step 1: Download stock data
ticker = "AAPL"  # Change to any other stock if needed
start_date = "2015-01-01"
end_date = "2024-12-31"
data = yf.download(ticker, start=start_date, end=end_date)

# Step 2: Preprocess and normalize data
close_data = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_data)

# Step 3: Create sequences
def create_sequences(data, sequence_length):
    x = []
    y = []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

sequence_length = 60
x, y = create_sequences(scaled_data, sequence_length)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Step 4: Split data into training and testing sets
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 5: Hyperparameter Tuning
def build_model(units=50, sequence_length=60, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Experiment with different batch sizes and units
batch_size = 32
epochs = 15  # Increased epochs for better training results

# Build and train the model
model = build_model(units=60, sequence_length=x.shape[1], optimizer='adam')
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# Step 6: Model Validation
# Make predictions on the test set
predicted_prices_test = model.predict(x_test)
predicted_prices_test = scaler.inverse_transform(predicted_prices_test)
real_prices_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot test predictions vs real prices
plt.figure(figsize=(14, 5))
plt.plot(real_prices_test, label="Real Price (Test)", color='blue')
plt.plot(predicted_prices_test, label="Predicted Price (Test)", color='red')
plt.title("Stock Price Prediction (Test Data)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig('stock_price_prediction_test.png')
plt.show()

# Step 7: Model Evaluation - Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(real_prices_test, predicted_prices_test)
rmse = np.sqrt(mean_squared_error(real_prices_test, predicted_prices_test))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 8: Improving Graph Visualization
# Enhanced plot with grid and labels
plt.figure(figsize=(14, 5))
plt.plot(real_prices_test, label="Real Price (Test)", color='blue')
plt.plot(predicted_prices_test, label="Predicted Price (Test)", color='red')
plt.title("Stock Price Prediction (Test Data)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig('stock_price_prediction_test_enhanced.png')  # Save the plot as an image
plt.show()

# Step 9: Add Indicators (optional)
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

print(data.tail())  # Shows the last few rows of the DataFrame

# Save the trained model weights
model.save('stock_price_model.h5')  # Save the model weights

print("Script executed successfully!")
print("Reached end of script.")
model.save("stock_price_model.h5")
=======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# Step 1: Download stock data
ticker = "AAPL"  # Change to any other stock if needed
start_date = "2015-01-01"
end_date = "2024-12-31"
data = yf.download(ticker, start=start_date, end=end_date)

# Step 2: Preprocess and normalize data
close_data = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_data)

# Step 3: Create sequences
def create_sequences(data, sequence_length):
    x = []
    y = []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

sequence_length = 60
x, y = create_sequences(scaled_data, sequence_length)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Step 4: Split data into training and testing sets
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 5: Hyperparameter Tuning
def build_model(units=50, sequence_length=60, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Experiment with different batch sizes and units
batch_size = 32
epochs = 15  # Increased epochs for better training results

# Build and train the model
model = build_model(units=60, sequence_length=x.shape[1], optimizer='adam')
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# Step 6: Model Validation
# Make predictions on the test set
predicted_prices_test = model.predict(x_test)
predicted_prices_test = scaler.inverse_transform(predicted_prices_test)
real_prices_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot test predictions vs real prices
plt.figure(figsize=(14, 5))
plt.plot(real_prices_test, label="Real Price (Test)", color='blue')
plt.plot(predicted_prices_test, label="Predicted Price (Test)", color='red')
plt.title("Stock Price Prediction (Test Data)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig('stock_price_prediction_test.png')
plt.show()

# Step 7: Model Evaluation - Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(real_prices_test, predicted_prices_test)
rmse = np.sqrt(mean_squared_error(real_prices_test, predicted_prices_test))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 8: Improving Graph Visualization
# Enhanced plot with grid and labels
plt.figure(figsize=(14, 5))
plt.plot(real_prices_test, label="Real Price (Test)", color='blue')
plt.plot(predicted_prices_test, label="Predicted Price (Test)", color='red')
plt.title("Stock Price Prediction (Test Data)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig('stock_price_prediction_test_enhanced.png')  # Save the plot as an image
plt.show()

# Step 9: Add Indicators (optional)
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

print(data.tail())  # Shows the last few rows of the DataFrame

# Save the trained model weights
model.save('stock_price_model.h5')  # Save the model weights

print("Script executed successfully!")
print("Reached end of script.")
model.save("stock_price_model.h5")

