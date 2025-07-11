import pandas as pd

def preprocess(df):
    df = df.dropna()
    df['hour'] = df.index.hour
    df['volatility'] = df['High'] - df['Low']
    X = df[['Open', 'Close', 'Volume', 'hour', 'volatility']]
    return X
