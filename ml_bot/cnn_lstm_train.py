import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
import joblib
from rl_train import fetch_data, add_features
from sklearn.preprocessing import MinMaxScaler
import MetaTrader5 as mt5

SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
DATA_LIMIT = 5000
WINDOW_SIZE = 20

def create_cnn_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid') # 1 = Sell, 0 = Buy
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if not mt5.initialize():
        print("initialize() failed")
        return
        
    df = fetch_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    mt5.shutdown()
    
    if df is None:
        return
        
    df = add_features(df).dropna()
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y', 'atr_14', 'day_of_week']
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    joblib.dump(scaler, 'ml_bot/rl_scaler.save') # Overwrite global scaler with 14 features
    
    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_data) - 1): # -1 because we predict the NEXT day
        X.append(scaled_data[i - WINDOW_SIZE:i])
        
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
    
    model = create_cnn_lstm_model((WINDOW_SIZE, len(features)))
    print("Training CNN+LSTM Model...")
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])
    
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    model.save('ml_bot/cnn_lstm_model.keras')
    print("Saved CNN+LSTM model to ml_bot/cnn_lstm_model.keras")

if __name__ == "__main__":
    main()
