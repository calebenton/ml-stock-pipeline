import yfinance as yf
import pandas as pd


def fetch_data(ticker="AAPL"):
    df = yf.download(ticker, period="60d", interval="1h", auto_adjust=False)
    df.to_csv("data/raw_data.csv")
    return df
