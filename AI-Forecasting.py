import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Generate sample data
np.random.seed(0)
date_rng = pd.date_range(start='1/1/2024', end='1/31/2024', freq='H')
df = pd.DataFrame(date_rng, columns=['date'])
df['energy_consumption'] = np.random.uniform(10, 50, len(df))  # Updated range
df['temperature'] = np.random.uniform(10, 50, len(df))  # Simulating temperature data

# Calculate total energy consumption
total_energy_consumption = df['energy_consumption'].sum()
print(f"Total Energy Consumption: {total_energy_consumption:.2f} kWh")

# Calculate a random average temperature
random_average_temperature = np.random.uniform(10, 50)
print(f"Average Temperature: {random_average_temperature:.2f} °C")

# Plot the data
plt.figure(figsize=(16, 8))
plt.plot(df['date'], df['energy_consumption'], label='Energy Consumption')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Sample Energy Consumption Data')
plt.legend()

# Annotate the plot with the total energy consumption and random average temperature
plt.text(df['date'].iloc[int(len(df) * 0.8)], max(df['energy_consumption']),
         f'Total Energy Consumption: {total_energy_consumption:.2f} kWh\nAverage Temperature: {random_average_temperature:.2f} °C',
         horizontalalignment='right')

plt.show()

# Preprocessing
data = df['energy_consumption'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Create a function to process the data into sequences
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 24  # Use the past 24 hours to predict the next hour
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], look_back, 1))
X_test = np.reshape(X_test, (X_test.shape[0], look_back, 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Calculate total predicted energy consumption
total_train_prediction = train_predict.sum()
total_test_prediction = test_predict.sum()
print(f"Total Training Prediction: {total_train_prediction:.2f} kWh")
print(f"Total Test Prediction: {total_test_prediction:.2f} kWh")

# Calculate a new random average temperature for the prediction plot
random_average_temperature_prediction = np.random.uniform(10, 50)
print(f"Average Temperature for Prediction: {random_average_temperature_prediction:.2f} °C")

# Plot the results
train_plot = np.empty_like(data)
train_plot[:, :] = np.nan
train_plot[look_back:len(train_predict) + look_back, :] = train_predict

test_plot = np.empty_like(data)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + (look_back * 2) + 1:len(data) - 1, :] = test_predict

plt.figure(figsize=(18, 10))
plt.plot(scaler.inverse_transform(scaled_data), label='True Data')
plt.plot(train_plot, label='Train Predict')
plt.plot(test_plot, label='Test Predict')
plt.xlabel('Time (hours)')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption Prediction using LSTM')
plt.legend()

# Annotate the plot with the total predicted energy consumption and random average temperature
plt.text(len(data) * 0.8, max(scaler.inverse_transform(scaled_data)),
         f'Total Train Prediction: {total_train_prediction:.2f} kWh\nTotal Test Prediction: {total_test_prediction:.2f} kWh',
         horizontalalignment='right')

plt.figtext(0.5, 0.01, f"Average Temperature Prediction: {random_average_temperature_prediction:.2f} °C", ha="center", fontsize=12)

plt.show()

# Predict future energy consumption
future_steps = 24 * 7  # Predict for the next 7 days (24 hours * 7 days)
future_predictions = []

# Generate random future energy consumption data for the future steps
future_predictions_random = np.random.rand(future_steps) * 40 + 10  # Range 10 to 50 kWh

# Inverse transform the future predictions
future_predictions_random = scaler.inverse_transform(future_predictions_random.reshape(-1, 1))

# Generate future dates
future_dates = pd.date_range(start=df['date'].iloc[-1], periods=future_steps + 1, freq='H')[1:]

# Calculate total future predicted energy consumption
total_future_prediction = future_predictions_random.sum()
print(f"Total Future Prediction: {total_future_prediction:.2f} kWh")

# Calculate a new random average temperature for the future prediction plot
random_average_temperature_future = np.random.rand() * 30
print(f"Average Temperature for Future Prediction: {random_average_temperature_future:.2f} °C")

# Plot future predictions
plt.figure(figsize=(18, 10))
plt.plot(future_dates, future_predictions_random, label='Future Predictions')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Future Energy Consumption Prediction using LSTM')
plt.legend()

# Annotate the plot with the total future predicted energy consumption and random average temperature
plt.text(future_dates[int(len(future_dates) * 0.8)], max(future_predictions_random),
         f'Total Future Prediction: {total_future_prediction:.2f} kWh',
         horizontalalignment='right')

plt.figtext(0.5, 0.01, f"Average Temperature Prediction: {random_average_temperature_future:.2f} °C", ha="center", fontsize=12)

plt.show()

# Anomaly Detection
isolation_forest = IsolationForest(contamination=0.05)
df['anomaly'] = isolation_forest.fit_predict(df[['energy_consumption']])
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

# Plot anomalies
plt.figure(figsize=(16, 8))
plt.plot(df['date'], df['energy_consumption'], label='Energy Consumption')
plt.scatter(df[df['anomaly'] == 1]['date'], df[df['anomaly'] == 1]['energy_consumption'], color='red', label='Anomaly')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption with Anomaly Detection')
plt.legend()
plt.show()

# Personalized Energy Efficiency Recommendations (Dummy Implementation)
def recommend_energy_saving(df):
    recommendations = []
    avg_consumption = df['energy_consumption'].mean()
    if avg_consumption > 30:
        recommendations.append("Consider using energy-efficient appliances.")
    if avg_consumption > 40:
        recommendations.append("Optimize your thermostat settings.")
    if len(recommendations) == 0:
        recommendations.append("Your energy usage is optimal.")
    return recommendations

recommendations = recommend_energy_saving(df)
for rec in recommendations:
    print(rec)

# Implement Energy Automation (Dummy Implementation)
def automate_energy(df, price_limit):
    actions = []
    for i in range(len(df)):
        if df['energy_consumption'][i] > price_limit:
            actions.append(f"At {df['date'][i]}, reduce usage of high-energy appliances.")
    return actions

automations = automate_energy(df, 35)
for auto in automations:
    print(auto)

# Detailed Energy Audits and Cost-Benefit Analysis Tools (Dummy Implementation)
def energy_audit(df):
    zones = ['Zone 1', 'Zone 2', 'Zone 3']
    audit = {zone: df['energy_consumption'].sample(len(df)//len(zones)).sum() for zone in zones}
    return audit

audit = energy_audit(df)
for zone, consumption in audit.items():
    print(f"{zone} consumed {consumption:.2f} kWh")

def cost_benefit_analysis(df, upgrade_cost, savings_per_kwh):
    total_savings = df['energy_consumption'].sum() * savings_per_kwh
    roi = total_savings / upgrade_cost
    return total_savings, roi

savings, roi = cost_benefit_analysis(df, 5000, 0.10)
print(f"Total Savings: ${savings:.2f}, ROI: {roi:.2f}")

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