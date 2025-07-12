import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv("data/raw_data.csv", skiprows=3, names=["Datetime", "Adj Close", "Close", "High", "Low", "Open", "Volume"])
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

df['hour'] = df.index.hour
df['volatility'] = df['High'] - df['Low']
df['price_change'] = df['Close']- df['Open']
df['percent_change'] = (df['Close'] - df['Open']) / df['Open']
df['rolling_mean_3'] = df['Close'].rolling(window=3).mean()
df['rolling_std_3'] = df['Close'].rolling(window=3).std()
df['prev_close'] = df['Close'].shift(1)
df['volume_change'] = df['Volume'] - df['Volume'].shift(1)

df.dropna(inplace=True)

X = df[['Open', 'Close', 'Volume', 'hour', 'volatility', 'price_change', 'percent_change', 'rolling_mean_3', 'rolling_std_3', 'prev_close', 'volume_change']]

X = X.iloc[:-1]
df = df.iloc[:-1]

# Define models to load
model_names = ["random_forest", "xgboost", "linear_regression"]

plt.figure(figsize=(14, 6))

# Plot actual stock price
plt.plot(df.index, df['Close'], label='Actual Close', color='black', linewidth=2, linestyle='--')

# Plot predictions for each model
for name in model_names:
    model = joblib.load(f"models/{name}.pkl")
    preds = model.predict(X)
    df[name + "_pred"] = preds
    plt.plot(df.index, preds, marker='o', linestyle='-', label=name.replace("_", " ").title())

# Finalize plot
plt.title("Model Predictions vs Actual AAPL Stock Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

