import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv("data/raw_data.csv", skiprows=3, names=["Datetime", "Adj Close", "Close", "High", "Low", "Open", "Volume"])
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

df['hour'] = df.index.hour
df['volatility'] = df['High'] - df['Low']

X = df[['Open', 'Close', 'Volume', 'hour', 'volatility']]

# Define models to load
model_names = ["random_forest", "xgboost", "logistic_regression"]

plt.figure(figsize=(14, 6))

for name in model_names:
    model = joblib.load(f"models/{name}.pkl")
    preds = model.predict(X)
    df[name + "_pred"] = preds
    plt.plot(df.index, preds, marker='o', linestyle='-', label=name.replace("_", " ").title())

# Finalize plot
plt.title("Model Predictions Over Time")
plt.xlabel("Time")
plt.ylabel("Prediction (1 = Up, 0 = Down)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
