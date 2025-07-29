from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
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
        conn.execute(text(create_table_sql))  # ✅ Removed conn.commit()
    print(f"Ensured table '{table_name}' exists.")

def upsert_df(df, table_name, engine):
    from sqlalchemy import Table, MetaData

    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    with engine.connect() as conn:
        for _, row in df.iterrows():
            stmt = insert(table).values(**row.to_dict())
            stmt = stmt.on_conflict_do_nothing(index_elements=['prediction_time'])
            conn.execute(stmt)
        # ✅ Removed conn.commit()


def run_prediction(X):
    from sqlalchemy.exc import ProgrammingError

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
        print(f"{name} prediction time range:", X.index.min(), "to", X.index.max())

        df_preds = pd.DataFrame({
            "prediction_time": pd.to_datetime(X.index),
            "predicted_close": preds
        })

        # Save CSV
        pred_filename = f"predictions/{name}_predicted_prices_{timestamp}.csv"
        df_preds.to_csv(pred_filename, index=False)
        print(f"Saved {name} predictions to {pred_filename}")

        # Insert/update predictions
        upsert_df(df_preds, table_name, engine)
        print(f"Upserted predictions to Postgres table '{table_name}'")

        # Step 1: Add 'actual_close' column if missing
        with engine.connect() as conn:
            try:
                conn.execute(text(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN IF NOT EXISTS actual_close FLOAT;
                """))
                print(f"Ensured 'actual_close' column exists in {table_name}")
            except ProgrammingError as e:
                print(f"Error adding 'actual_close' column to {table_name}: {e}")

        # Step 2: Update actual_close values by joining with apple_stock_data
        with engine.connect() as conn:
            update_sql = f"""
            UPDATE {table_name} p
            SET actual_close = a."Close"
            FROM apple_stock_data a
            WHERE p.prediction_time::timestamp(0) = a."Datetime"::timestamp(0);
            """
            conn.execute(text(update_sql))
            print(f"Updated 'actual_close' values in {table_name}")
