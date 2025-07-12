import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime

# Load raw data
df = pd.read_csv("data/raw_data.csv", skiprows=3, names=["Datetime", "Adj Close", "Close", "High", "Low", "Open", "Volume"])
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Feature engineering (must match training pipeline)
df['hour'] = df.index.hour
df['volatility'] = df['High'] - df['Low']
df['price_change'] = df['Close'] - df['Open']
df['percent_change'] = (df['Close'] - df['Open']) / df['Open']
df['rolling_mean_3'] = df['Close'].rolling(window=3).mean()
df['rolling_std_3'] = df['Close'].rolling(window=3).std()
df['prev_close'] = df['Close'].shift(1)
df['volume_change'] = df['Volume'] - df['Volume'].shift(1)

df.dropna(inplace=True)

# Select latest feature row
X_latest = df[['Open', 'Close', 'Volume', 'hour', 'volatility', 'price_change',
               'percent_change', 'rolling_mean_3', 'rolling_std_3', 'prev_close', 'volume_change']].iloc[-1:]

# Load model
model = joblib.load("models/linear_regression.pkl")  # Change if needed

# Predict
prediction = model.predict(X_latest)[0]
print(f"📈 Predicted next hour's closing price: ${prediction:.2f}")

# Save to CSV
os.makedirs("data", exist_ok=True)
log_path = "data/predicted_prices.csv"
log_entry = pd.DataFrame([{
    "Prediction_Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "Predicted_Close": prediction
}])

# Append or create
if os.path.exists(log_path):
    log_entry.to_csv(log_path, mode='a', header=False, index=False)
else:
    log_entry.to_csv(log_path, mode='w', header=True, index=False)
