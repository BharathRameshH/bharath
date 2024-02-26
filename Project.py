import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Define a Streamlit app
st.title('Stock Price Prediction and Visualization')

# Sidebar for user input
st.sidebar.header('User Input')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2019-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2023-06-30'))
st.sidebar.write('You selected:', start_date, 'to', end_date)

# Download and preprocess the data
data = yf.download('IBM', start=start_date, end=end_date)
data['Open-Close'] = (data.Open - data.Close) / data.Open
data['High-Low'] = (data.High - data.Low) / data.Low
data['percent_change'] = data['Adj Close'].pct_change()
data['std_5'] = data['percent_change'].rolling(5).std()
data['ret_5'] = data['percent_change'].rolling(5).mean()
data.dropna(inplace=True)

# Split the data
X = data[['Open-Close', 'High-Low', 'std_5', 'ret_5']]
y = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, -1)
dataset_length = data.shape[0]
split = int(dataset_length * 0.75)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train the model
clf = RandomForestClassifier()
model = clf.fit(X_train, y_train)

# Display model accuracy
st.write('Model Accuracy:')
accuracy = accuracy_score(y_test, model.predict(X_test), normalize=True) * 100.0
st.write(f'Correct Prediction (%): {accuracy:.2f}%')

# Display classification report
st.write('Classification Report:')
report = classification_report(y_test, model.predict(X_test))
st.write(report)

# Plot strategy returns
data['strategy_returns'] = data.percent_change.shift(-1) * model.predict(X)
st.write('Strategy Returns Histogram:')
fig, ax = plt.subplots()
ax.hist(data.strategy_returns[split:], bins=20)
st.pyplot(fig)

st.write('Strategy Returns Cumulative Returns:')
fig, ax = plt.subplots()
ax.plot((data.strategy_returns[split:] + 1).cumprod())
st.pyplot(fig)