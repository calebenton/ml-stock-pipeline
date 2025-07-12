import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

files = glob.glob("predictions/*_predicted_prices.csv")

for file_path in files:
    df = pd.read_csv(file_path, parse_dates=["Prediction_Time"])
    df = df.dropna(subset=['Actual_Close'])
    
    rmse = np.sqrt(mean_squared_error(df['Actual_Close'], df['Predicted_Close']))
    mae = mean_absolute_error(df['Actual_Close'], df['Predicted_Close'])
    r2 = r2_score(df['Actual_Close'], df['Predicted_Close'])
    
    model_name = os.path.basename(file_path).replace("_predicted_prices.csv", "")
    print(f"\n{model_name.upper()} Evaluation Metrics over {len(df)} predictions:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    plt.figure(figsize=(12,6))
    plt.plot(df['Prediction_Time'], df['Actual_Close'], label='Actual Close', marker='o')
    plt.plot(df['Prediction_Time'], df['Predicted_Close'], label='Predicted Close', marker='x')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"{model_name.title()} Predicted vs Actual Closing Prices Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
