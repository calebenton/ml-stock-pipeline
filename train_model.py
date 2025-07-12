import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load data
df = pd.read_csv("data/raw_data.csv", skiprows=3, names=["Datetime", "Adj Close", "Close", "High", "Low", "Open", "Volume"])
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Feature engineering
df['hour'] = df.index.hour
df['volatility'] = df['High'] - df['Low']
df['price_change'] = df['Close']- df['Open']
df['percent_change'] = (df['Close'] - df['Open']) / df['Open']
df['rolling_mean_3'] = df['Close'].rolling(window=3).mean()
df['rolling_std_3'] = df['Close'].rolling(window=3).std()
df['prev_close'] = df['Close'].shift(1)
df['volume_change'] = df['Volume'] - df['Volume'].shift(1)

df.dropna(inplace=True)

# Define features and target
X = df[['Open', 'Close', 'Volume', 'hour', 'volatility', 'price_change', 'percent_change', 'rolling_mean_3', 'rolling_std_3', 'prev_close', 'volume_change']]
y = df['Close'].shift(-1)

# Drop last row (NaN target)
X = X.iloc[:-1]
y = y.iloc[:-1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=False)

# Define models
models = {
    "random_forest": RandomForestRegressor(
        n_estimators=100, max_depth=None, random_state=42
        # you can tune these params similarly
    ),
    "xgboost": XGBRegressor(
        subsample=0.7,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric='rmse'
    ),
    "linear_regression": LinearRegression()
}


# Store results for CSV
results = []

# Train, evaluate, and save each model
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Print to terminal
    print(f"\n{name.upper()} Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    # Save model
    joblib.dump(model, f"models/{name}.pkl")
    print(f"  Saved model to models/{name}.pkl")

    # Save predictions with actual prices
    df_out = pd.DataFrame({
        "Prediction_Time": X_test.index,
        "Actual_Close": y_test,
        "Predicted_Close": preds
    })
    os.makedirs("predictions", exist_ok=True)
    df_out.to_csv(f"predictions/{name}_predicted_prices.csv", index=False)
    print(f"  Saved predictions to predictions/{name}_predicted_prices.csv")

    # Append to results
    results.append({
        "Model": name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4)
    })

# Save all metrics to CSV
os.makedirs("metrics", exist_ok=True)
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("metrics/model_scores.csv", index=False)
print("\nModel scores saved to metrics/model_scores.csv")
