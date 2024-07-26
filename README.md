# Stock-Price-Prediction-Using-LSTM-Neural-Network

# Stock Price Prediction Using LSTM Neural Network

This project aims to predict stock prices using a Long Short-Term Memory (LSTM) neural network. The dataset consists of synthetic stock data for the year 2024.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
The goal of this project is to leverage the power of LSTM neural networks to predict future stock prices based on historical data. LSTMs are particularly well-suited for time series prediction due to their ability to learn long-term dependencies.

## Dataset
The dataset used in this project is synthetic stock data for the year 2024. The data includes the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

### Sample Data
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100,105,95,102,1000
2024-01-02,102,107,97,104,1100
...


Installation
To run this project, you will need to have Python and the following libraries installed:

numpy
pandas
matplotlib
scikit-learn
tensorflow
You can install these libraries using pip:   pip install numpy pandas matplotlib scikit-learn tensorflow


Usage
1. Generate the Dataset:
Run the following script to generate the synthetic stock data for 2024:
import csv
from datetime import datetime, timedelta

# Generate a sample CSV file with stock data for 2024
start_date = datetime.strptime('2024-01-01', '%Y-%m-%d')
num_days = 365  # Number of days for one year

with open('stock_data_2024.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    for i in range(num_days):
        date = start_date + timedelta(days=i)
        open_price = 100 + (i * 0.1)  # Example: gradually increasing open price
        high_price = open_price + 5
        low_price = open_price - 5
        close_price = open_price + 1  # Example: slightly higher close price
        volume = 1000 + (i * 10)
        writer.writerow([date.strftime('%Y-%m-%d'), open_price, high_price, low_price, close_price, volume])

2. Run the LSTM Model:
Use the following script to train the LSTM model and make predictions:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Load the dataset
df = pd.read_csv('stock_data_2024.csv')  # Replace with your dataset path

# Check the dataframe structure
print(df.head())

# Preprocessing
# Select the 'Close' price column and convert to numpy array
data = df['Close'].values.reshape(-1, 1)

# Scale the data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predict the stock prices
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual stock prices
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform([y_train])
y_test_actual = scaler.inverse_transform([y_test])

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(df['Close'], label='Actual Stock Price')
train_plot = np.empty_like(data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict) + time_step, :] = train_predict
plt.plot(train_plot, label='Training Prediction')

test_plot = np.empty_like(data)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + (time_step * 2) + 1:len(data) - 1, :] = test_predict
plt.plot(test_plot, label='Testing Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()



Results
The results of the LSTM model are visualized by plotting the actual stock prices and the model's predictions for both the training and testing datasets.

Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.


Contact
Email: msharifk2304@gmail.com
LinkedIn: Mudhathir Sharif Khaja
Thank you for checking out my project!



