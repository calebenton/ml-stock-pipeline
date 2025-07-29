from sqlalchemy import create_engine
import pandas as pd

# Load or define your DataFrame
df_processed = pd.read_csv("data/processed_features.csv")  # example

# Create SQLAlchemy engine (do NOT open a separate connection)
engine = create_engine("postgresql+psycopg2://calebbenton@localhost:5432/ml_pipeline")

# Pass the engine itself, NOT a connection object, to to_sql
df_processed.to_sql(
    "apple_stock_data",
    con=engine,      # pass engine here, not engine.connect()
    if_exists="append",
    index=False
)

print("Data uploaded successfully.")
