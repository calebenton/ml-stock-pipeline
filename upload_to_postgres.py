import pandas as pd
from sqlalchemy import create_engine
from pipeline.preprocess import preprocess  # ✅ import your shared function

# Load the raw CSV
df_raw = pd.read_csv("data/raw_data.csv", skiprows=3,
                     names=["Datetime", "Adj Close", "Close", "High", "Low", "Open", "Volume"])
df_raw['Datetime'] = pd.to_datetime(df_raw['Datetime'])
df_raw.set_index('Datetime', inplace=True)

# Use the same preprocessing logic
df_processed = preprocess(df_raw)

# Optionally reset index if needed
df_processed.reset_index(inplace=True)

# Upload to PostgreSQL
engine = create_engine("postgresql://calebbenton@localhost:5432/ml_pipeline")
df_processed.to_sql("apple_stock_data", engine, if_exists="append", index=False)

print("Uploaded feature-engineered data to PostgreSQL.")