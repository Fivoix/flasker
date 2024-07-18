import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Fetch current hour average price
def get_current_price():
    response = requests.get('https://hourlypricing.comed.com/api?type=currenthouraverage&format=json')
    current_price_data = response.json()
    return float(current_price_data[0]['price'])

# Fetch historical 5-minute prices for the last 24 hours
def get_historical_prices():
    response = requests.get('https://hourlypricing.comed.com/api?type=5minutefeed&format=json')
    historical_data = response.json()
    prices = [float(entry['price']) for entry in historical_data]
    timestamps = [entry['millisUTC'] for entry in historical_data]
    return pd.DataFrame({'timestamp': pd.to_datetime(timestamps, unit='ms'), 'price': prices})

# Prepare data for LSTM
def prepare_data(prices, look_back=12):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))

    X, Y = [], []
    for i in range(len(scaled_prices) - look_back):
        X.append(scaled_prices[i:i + look_back, 0])
        Y.append(scaled_prices[i + look_back, 0])
    return np.array(X), np.array(Y), scaler

# Build and train the LSTM model
def train_lstm(X, Y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, epochs=20, batch_size=32, verbose=1)
    return model

# Predict future prices
def predict_future_prices(model, scaler, historical_prices, n_future=24):
    future_predictions = []
    current_input = historical_prices.reshape((1, historical_prices.shape[0], 1))

    for _ in range(n_future):
        prediction = model.predict(current_input, verbose=0)
        future_predictions.append(prediction[0, 0])
        prediction = prediction.reshape(1, 1, 1)
        current_input = np.append(current_input[:, 1:, :], prediction, axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten()

# Main script
if __name__ == "__main__":
    # Get current and historical prices
    current_price = get_current_price()
    historical_prices_df = get_historical_prices()
    
    # Prepare data
    historical_prices = historical_prices_df['price'].values
    X, Y, scaler = prepare_data(historical_prices)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Train LSTM model
    model = train_lstm(X, Y)
    
    # Predict future prices
    future_predictions = predict_future_prices(model, scaler, historical_prices[-12:])
    
    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(historical_prices_df['timestamp'], historical_prices, label='Historical Prices')
    
    # Generate future timestamps for prediction plot
    future_timestamps = [historical_prices_df['timestamp'].iloc[-1] + timedelta(minutes=5 * (i+1)) for i in range(len(future_predictions))]
    
    plt.plot(future_timestamps, future_predictions, label='Predicted Prices', color='red')
    
    # Plot vertical line for current time
    plt.axvline(x=historical_prices_df['timestamp'].iloc[-1], color='gray', linestyle='--', label='Current Time')
    
    plt.xlabel('Time (hh:mm AM/PM)')
    plt.xticks(rotation=45)
    plt.ylabel('Price (cents per kWh)')
    plt.title('Electricity Price Prediction')
    plt.legend()
    
    # Set major formatter for x-axis to show time in 12-hour format
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))

    plt.tight_layout()
    plt.show()

    # Decision making based on price
    if current_price < np.mean(future_predictions):
        print("Current price is low. You may use high-energy appliances now.")
    else:
        print("Current price is high. Consider reducing energy usage or shifting it to later.")

    # Print current price and predictions
    print(f"Current Hour Average Price: {current_price} cents per kWh")
    print(f"Predicted Prices: {future_predictions.flatten()} cents per kWh")