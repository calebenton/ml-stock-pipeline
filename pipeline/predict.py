import joblib
import pandas as pd
import os
from datetime import datetime

def run_prediction(X):
    model_names = ["random_forest", "xgboost", "linear_regression"]

    os.makedirs("predictions", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for name in model_names:
        model_path = f"models/{name}.pkl"
        model = joblib.load(model_path)
        preds = model.predict(X)

        print(f"\n{name.upper()} predictions:\n", preds)

        df_preds = pd.DataFrame({
            "Prediction_Time": pd.to_datetime(X.index),
            "Predicted_Close": preds
        })

        pred_filename = f"predictions/{name}_predicted_prices_{timestamp}.csv"
        df_preds.to_csv(pred_filename, index=False)
        print(f"Saved {name} predictions to {pred_filename}")
