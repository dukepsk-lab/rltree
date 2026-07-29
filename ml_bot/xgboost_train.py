import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from rl_train import fetch_data, add_features
from features import FEATURES
from sklearn.metrics import accuracy_score
import MetaTrader5 as mt5

SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
DATA_LIMIT = 5000
WINDOW_SIZE = 20

def main():
    if not mt5.initialize():
        print("initialize() failed")
        return
        
    df = fetch_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    mt5.shutdown()
    
    if df is None:
        return
        
    df = add_features(df).dropna()
    features = FEATURES

    # Previously a bare `except:` here refitted the scaler and overwrote
    # rl_scaler.save whenever transform() raised — silently breaking the PPO
    # model that had been trained against the old one (ML_ANALYSIS.md #1).
    # Fail loudly instead; rl_train.py is the only writer of this file.
    scaler = joblib.load('ml_bot/rl_scaler.save')
    if scaler.n_features_in_ != len(features):
        raise SystemExit(
            f"rl_scaler.save was fitted on {scaler.n_features_in_} features, this script "
            f"needs {len(features)}. Re-run rl_train.py first so the scaler matches."
        )
    scaled_data = scaler.transform(df[features])

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_data) - 1):
        # Flatten the window for XGBoost (WINDOW_SIZE * NUM_FEATURES)
        flattened_window = scaled_data[i - WINDOW_SIZE:i].flatten()
        X.append(flattened_window)
        
        # Target: 0 = Buy (Close > Open), 1 = Sell (Close <= Open)
        next_close = df['close'].iloc[i+1]
        next_open = df['open'].iloc[i+1]
        target = 0 if next_close > next_open else 1
        y.append(target)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("Training XGBoost Model on GPU...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        device='cuda'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=20) # early_stopping_rounds removed in newer xgboost, using callbacks if needed, but it's fast anyway
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    model.save_model('ml_bot/xgboost_model.json')
    print("Saved XGBoost model to ml_bot/xgboost_model.json")

if __name__ == "__main__":
    main()
