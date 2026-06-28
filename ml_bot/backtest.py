import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# --- Configuration ---
SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
DATA_LIMIT = 5000     # Historical bars for backtest
SEQUENCE_LENGTH = 20
TP_PERCENT = 0.01      # 1% Take Profit
INITIAL_CAPITAL = 10000.0

# --- 1. Initialize MT5 ---
def init_mt5():
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code: {mt5.last_error()}")
        quit()
    print("MT5 Initialized Successfully")

# --- 2. Fetch Data ---
def fetch_data(symbol, timeframe, n_bars):
    # Ensure symbol is visible in Market Watch
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol '{symbol}' not found in MT5. Please check the SYMBOL name (e.g. 'XAUUSD.m', 'GOLD').")
        return None
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select symbol '{symbol}' in Market Watch.")
            return None
            
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        print(f"Failed to get rates for '{symbol}'. MT5 Error Code: {mt5.last_error()}")
        print("Tip: If error is (1, 'Success'), the broker might need time to download history. Try running again or open the chart manually.")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume']]

# --- 3. Preprocess Data ---
def add_features(df):
    df = df.copy()
    # SMA
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # RSI 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # ADX 14
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    up = high - high.shift()
    down = low.shift() - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_14 = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).sum() / (tr_14 + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).sum() / (tr_14 + 1e-9))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['adx_14'] = dx.rolling(14).mean()
    
    # Linear Regression 20
    window = 20
    x = np.arange(window)
    sum_x = x.sum()
    sum_x2 = (x**2).sum()
    denom = window * sum_x2 - sum_x**2
    def slope_func(y):
        return (window * (x * y).sum() - sum_x * y.sum()) / denom
    df['linreg_20'] = df['close'].rolling(window=window).apply(slope_func, raw=True)
    
    return df

def prepare_data(df, seq_length):
    df = add_features(df)
    
    # Target: 1 if next close > current close, else 0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # We drop the last row (no target) and rows with NaN from rolling
    df_valid = df.dropna()
    
    scaler = MinMaxScaler()
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20']
    scaled_data = scaler.fit_transform(df_valid[features])
    targets = df_valid['target'].values

    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i : i + seq_length])
        y.append(targets[i + seq_length])
        
    return np.array(X), np.array(y), scaler, df_valid.iloc[seq_length:]

from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input

# --- 4. Build Model ---
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 5. Main Backtest Logic ---
def main():
    init_mt5()
    
    print(f"Fetching {DATA_LIMIT} historical bars...")
    df = fetch_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    mt5.shutdown() # We don't need MT5 connection anymore after fetching data
    
    if df is None or len(df) < SEQUENCE_LENGTH * 2:
        print("Not enough data.")
        return
        
    print("Preparing data...")
    X, y, scaler, df_prices = prepare_data(df, SEQUENCE_LENGTH)
    
    # Split Data (80% Train, 20% Test) chronologically
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    df_test = df_prices.iloc[split_index:].copy()
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    print("Building and training model...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    print("Running Backtest on Test Data...")
    predictions = model.predict(X_test, verbose=0)
    
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    
    trades = 0
    wins = 0
    losses = 0
    
    # Loop over Test Set
    for i in range(len(df_test) - 1):
        pred = predictions[i][0]
        
        # We predict at index i (representing the close of the current bar).
        # We trade on index i+1 (the next bar).
        next_bar = df_test.iloc[i + 1]
        
        entry_price = next_bar['open']
        high_price = next_bar['high']
        low_price = next_bar['low']
        close_price = next_bar['close']
        
        trade_profit = 0.0
        traded = False
        
        if pred > 0.505: # BUY SIGNAL
            traded = True
            tp_price = entry_price * (1 + TP_PERCENT)
            
            # Check if high hits TP
            if high_price >= tp_price:
                # Won: Hit TP
                trade_profit = (tp_price - entry_price) / entry_price * equity
                wins += 1
            else:
                # Close at the end of the bar
                trade_profit = (close_price - entry_price) / entry_price * equity
                if trade_profit > 0:
                    wins += 1
                else:
                    losses += 1
                    
        if traded:
            trades += 1
            equity += trade_profit
            
        equity_curve.append(equity)
        
    # --- Statistics ---
    net_profit = equity - INITIAL_CAPITAL
    win_rate = (wins / trades * 100) if trades > 0 else 0
    
    # Drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    print("\n=== BACKTEST RESULTS ===")
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Final Capital:   ${equity:.2f}")
    print(f"Net Profit:      ${net_profit:.2f} ({(net_profit/INITIAL_CAPITAL*100):.2f}%)")
    print(f"Total Trades:    {trades}")
    print(f"Win Rate:        {win_rate:.2f}% ({wins} W / {losses} L)")
    print(f"Max Drawdown:    {max_dd:.2f}%")
    print("========================\n")
    
    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label='Equity Curve', color='blue')
    plt.title(f"CNN+LSTM Backtest on {SYMBOL} (H1) - TP 1%")
    plt.xlabel('Bars / Time (Test Set)')
    plt.ylabel('Account Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
