from pipeline.fetch_data import fetch_data
from pipeline.preprocess import preprocess
from pipeline.predict import run_prediction

def main():
    print("Starting ML pipeline...")
    df = fetch_data()
    X = preprocess(df)
    run_prediction(X)

if __name__ == '__main__':
    main()
