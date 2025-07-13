import pandas as pd

def preprocess(df):
    df = df.dropna()

    df['hour'] = df.index.hour
    df['volatility'] = df['High'] - df['Low']
    df['price_change'] = df['Close'] - df['Open']
    df['percent_change'] = (df['Close'] - df['Open']) / df['Open']
    df['rolling_mean_3'] = df['Close'].rolling(window=3).mean()
    df['rolling_std_3'] = df['Close'].rolling(window=3).std()
    df['prev_close'] = df['Close'].shift(1)
    df['volume_change'] = df['Volume'] - df['Volume'].shift(1)

    df = df.dropna()

    # Select features in exact same order as training
    X = df[['Open', 'Close', 'Volume', 'hour', 'volatility', 'price_change',
            'percent_change', 'rolling_mean_3', 'rolling_std_3', 'prev_close', 'volume_change']]

    return X
