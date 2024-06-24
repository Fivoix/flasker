import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate sample data
np.random.seed(0)
date_rng = pd.date_range(start='1/1/2024', end='1/31/2024', freq='H')
df = pd.DataFrame(date_rng, columns=['date'])
df['energy_consumption'] = np.random.rand(len(df)) * 100
df['temperature'] = np.random.rand(len(df)) * 30  # Simulating temperature data

# Calculate total energy consumption
total_energy_consumption = df['energy_consumption'].sum()
print(f"Total Energy Consumption: {total_energy_consumption:.2f} kWh")

# Calculate a random average temperature
random_average_temperature = np.random.rand() * 30
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
random_average_temperature_prediction = np.random.rand() * 30
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

# Start with the last 'look_back' values from the test data
last_values = test_data[-look_back:]

for _ in range(future_steps):
    next_input = np.reshape(last_values, (1, look_back, 1))
    next_prediction = model.predict(next_input)
    future_predictions.append(next_prediction[0][0])
    
    # Update the last values, removing the first one and adding the new prediction
    last_values = np.append(last_values[1:], next_prediction, axis=0)

# Convert future_predictions to a 2D array for inverse transforming
future_predictions = np.array(future_predictions).reshape(-1, 1)

# Inverse transform the future predictions
future_predictions = scaler.inverse_transform(future_predictions)

# Generate future dates
future_dates = pd.date_range(start=df['date'].iloc[-1], periods=future_steps + 1, freq='H')[1:]

# Calculate total future predicted energy consumption
total_future_prediction = future_predictions.sum()
print(f"Total Future Prediction: {total_future_prediction:.2f} kWh")

# Calculate a new random average temperature for the future prediction plot
random_average_temperature_future = np.random.rand() * 30
print(f"Average Temperature for Future Prediction: {random_average_temperature_future:.2f} °C")

# Plot future predictions
plt.figure(figsize=(18, 10))
plt.plot(future_dates, future_predictions, label='Future Predictions')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Future Energy Consumption Prediction using LSTM')
plt.legend()

# Annotate the plot with the total future predicted energy consumption and random average temperature
plt.text(future_dates[int(len(future_dates) * 0.8)], max(future_predictions),
         f'Total Future Prediction: {total_future_prediction:.2f} kWh\nAverage Temperature: {random_average_temperature_future:.2f} °C',
         horizontalalignment='right')

plt.show()
