import joblib
import pandas as pd

def run_prediction(X):
    model = joblib.load('models/model.pkl')
    preds = model.predict(X)
    print(preds)
