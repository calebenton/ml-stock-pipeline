from pipeline.fetch_data import fetch_data
from pipeline.preprocess import preprocess
from pipeline.predict import run_prediction

def main():
    print("Starting ML pipeline...")
    df = fetch_data()
    X = preprocess(df)

    print("Columns before cleaning:", X.columns)

    # Flatten MultiIndex columns by joining the tuple parts with a space,
    # then strip and remove "AAPL"
    X.columns = [
        ' '.join(filter(None, col)).strip().replace("AAPL", "").strip()
        for col in X.columns
    ]

    print("Columns after cleaning:", X.columns)

    train_cols = ['Open', 'Close', 'Volume', 'hour', 'volatility', 'price_change',
                  'percent_change', 'rolling_mean_3', 'rolling_std_3', 'prev_close', 'volume_change']

    X = X[train_cols]

    # ✅ Save to CSV so it can be used elsewhere (e.g., PostgreSQL)
    X.to_csv("data/processed_features.csv", index=False)
    print("✅ Saved preprocessed features to data/processed_features.csv")

    run_prediction(X)

if __name__ == '__main__':
    main()