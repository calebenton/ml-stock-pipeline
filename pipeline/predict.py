from sqlalchemy import create_engine, text
import joblib
import pandas as pd
import os
from datetime import datetime

def create_prediction_table_if_not_exists(engine, table_name):
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        prediction_time TIMESTAMP PRIMARY KEY,
        predicted_close FLOAT
    );
    """
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()
    print(f"Ensured table '{table_name}' exists.")

def run_prediction(X):
    model_names = ["random_forest", "xgboost", "linear_regression"]
    engine = create_engine("postgresql+psycopg2://calebbenton@localhost:5432/ml_pipeline")

    os.makedirs("predictions", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for name in model_names:
        table_name = f"{name}_predicted_prices"

        # Create table if it doesn't exist
        create_prediction_table_if_not_exists(engine, table_name)

        model_path = f"models/{name}.pkl"
        model = joblib.load(model_path)
        preds = model.predict(X)

        print(f"\n{name.upper()} predictions:\n", preds)

        df_preds = pd.DataFrame({
            "prediction_time": pd.to_datetime(X.index),
            "predicted_close": preds
        })

        # Save CSV file as before
        pred_filename = f"predictions/{name}_predicted_prices_{timestamp}.csv"
        df_preds.to_csv(pred_filename, index=False)
        print(f"Saved {name} predictions to {pred_filename}")

        # Safely filter out existing timestamps to avoid duplicates
        existing_times = pd.read_sql(f"SELECT prediction_time FROM {table_name}", engine)

        if not existing_times.empty and 'prediction_time' in existing_times.columns:
            df_preds = df_preds[~df_preds["prediction_time"].isin(existing_times["prediction_time"])]

        if not df_preds.empty:
            df_preds.to_sql(
                table_name,
                engine,
                if_exists="append",
                index=False,
                method='multi'
            )
            print(f"Appended new predictions to Postgres table '{table_name}'")
        else:
            print(f"No new predictions to append for {name}")
