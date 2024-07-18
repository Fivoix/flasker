import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
from bs4 import BeautifulSoup

# Fetch real-time price data
def fetch_real_time_price():
    url = "https://hourlypricing.comed.com/live-prices/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    price = float(soup.find('div', {'class': 'live-price'}).text.strip().split(' ')[0])
    return price

# Generate sample data
np.random.seed(0)
date_rng = pd.date_range(start='1/1/2024', end='1/31/2024', freq='H')
df = pd.DataFrame(date_rng, columns=['date'])
df['energy_consumption'] = np.random.uniform(10, 50, len(df))  # Simulated energy consumption data
df['temperature'] = np.random.uniform(0, 30, len(df))  # Simulated temperature data
df['price'] = fetch_real_time_price()  # Fetch the current real-time price

# Calculate total energy consumption
total_energy_consumption = df['energy_consumption'].sum()
print(f"Total Energy Consumption: {total_energy_consumption:.2f} kWh")

# Calculate a random average temperature
random_average_temperature = np.random.uniform(0, 30)
print(f"Average Temperature: {random_average_temperature:.2f} °C")

# Preprocessing
data = df[['energy_consumption', 'price']].values  # Including price as a feature
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Create a function to process the data into sequences
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 24  # Use the past 24 hours to predict the next hour
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 2)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(np.concatenate([train_predict, np.zeros_like(train_predict)], axis=1))[:,0]
test_predict = scaler.inverse_transform(np.concatenate([test_predict, np.zeros_like(test_predict)], axis=1))[:,0]
y_train = scaler.inverse_transform(np.concatenate([y_train.reshape(-1, 1), np.zeros_like(y_train.reshape(-1, 1))], axis=1))[:,0]
y_test = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1))], axis=1))[:,0]

# Calculate total predicted energy consumption
total_train_prediction = train_predict.sum()
total_test_prediction = test_predict.sum()
print(f"Total Training Prediction: {total_train_prediction:.2f} kWh")
print(f"Total Test Prediction: {total_test_prediction:.2f} kWh")

# Plot the results
plt.figure(figsize=(18, 10))
plt.plot(df['date'], df['energy_consumption'], label='True Data')
plt.plot(df['date'][look_back:len(train_predict) + look_back], train_predict, label='Train Predict')
plt.plot(df['date'][len(train_predict) + (look_back * 2) + 1:len(data) - 1], test_predict, label='Test Predict')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption Prediction using LSTM with Real-Time Pricing')
plt.legend()
plt.show()

# Anomaly Detection
def detect_anomalies(consumption, threshold=1.5):
    mean = np.mean(consumption)
    std_dev = np.std(consumption)
    anomalies = np.where(np.abs(consumption - mean) > threshold * std_dev)
    return anomalies

anomalies = detect_anomalies(df['energy_consumption'])
print("Anomalies detected at indices:", anomalies)

# Personalized Recommendations
def provide_recommendations(price):
    recommendations = []
    if price > 3.0:  # Example threshold in cents per kWh
        recommendations.append("Reduce usage of high-energy appliances.")
        recommendations.append("Increase thermostat by 4 degrees.")
    else:
        recommendations.append("Normal energy usage is fine.")
    return recommendations

recommendations = provide_recommendations(df['price'][0])
print("Recommendations based on current price:", recommendations)

# Simulate Automated Adjustments
def simulate_automations(price, consumption):
    if price > 3.0:  # Example threshold in cents per kWh
        adjusted_consumption = consumption * 0.8  # Simulating 20% reduction
        print("Adjusted consumption to reduce cost:", adjusted_consumption)
    else:
        print("No adjustments needed.")

simulate_automations(df['price'][0], df['energy_consumption'][0])
