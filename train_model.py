import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/raw_data.csv", skiprows=3, names=["Datetime", "Adj Close", "Close", "High", "Low", "Open", "Volume"])
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Feature engineering
df['hour'] = df.index.hour
df['volatility'] = df['High'] - df['Low']

X = df[['Open', 'Close', 'Volume', 'hour', 'volatility']]
y = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop the last row (it has NaN label)
X = X.iloc[:-1]
y = y.iloc[:-1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define models
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "logistic_regression": LogisticRegression(max_iter=1000)
}

# Train and save each model
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}.pkl")
    print(f"Saved {name}.pkl")
