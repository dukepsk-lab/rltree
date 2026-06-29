import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# --- Configuration ---
SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
MAGIC_NUMBER = 999999
LOT_SIZE = 0.01
SEQUENCE_LENGTH = 20  # Number of past bars to use for prediction
DATA_LIMIT = 5000     # Historical bars to train on
MAX_SPREAD_POINTS = 50 # Max spread allowed (e.g. 50 points = 5 pips)
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

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
    
    # Calculate target: 1 if next close > current close, else 0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    scaler = MinMaxScaler()
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20']
    scaled_data = scaler.fit_transform(df[features])
    targets = df['target'].values

    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i : i + seq_length])
        y.append(targets[i + seq_length])
        
    return np.array(X), np.array(y), scaler

from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input

# --- 4. Build CNN+LSTM Model ---
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

# --- 5. Trading Filters ---
def is_high_impact_news_approaching():
    try:
        # Fetch ForexFactory JSON for current week
        response = requests.get(NEWS_URL, timeout=10)
        data = response.json()
        now = datetime.utcnow()
        
        for event in data:
            if event['country'] == 'USD' and event['impact'] == 'High':
                # Parse timezone-aware string like "2023-10-12T08:30:00-04:00"
                event_time_str = event['date']
                # python < 3.11 doesn't parse Z/offsets easily in strptime, so we can use pd.to_datetime
                event_time = pd.to_datetime(event_time_str).tz_convert('UTC').tz_localize(None)
                
                time_diff = (event_time - now).total_seconds()
                
                # If news is within the next 2 hours
                if 0 < time_diff <= 2 * 3600:
                    print(f"High impact news approaching: '{event['title']}' in {time_diff/3600:.1f} hours.")
                    return True
    except Exception as e:
        print(f"Failed to fetch or parse news: {e}")
    return False

def is_spread_ok(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return False
    spread = symbol_info.spread
    print(f"Current spread: {spread} points")
    return spread <= MAX_SPREAD_POINTS

# --- 6. Trading Execution ---
def close_all_positions(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return
        
    for p in positions:
        if p.magic == magic:
            tick = mt5.symbol_info_tick(symbol)
            type_dict = {mt5.POSITION_TYPE_BUY: mt5.ORDER_TYPE_SELL, mt5.POSITION_TYPE_SELL: mt5.ORDER_TYPE_BUY}
            price_dict = {mt5.POSITION_TYPE_BUY: tick.bid, mt5.POSITION_TYPE_SELL: tick.ask}
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": p.volume,
                "type": type_dict[p.type],
                "position": p.ticket,
                "price": price_dict[p.type],
                "deviation": 20,
                "magic": magic,
                "comment": "Close at bar end",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Failed to close position {p.ticket}, retcode={result.retcode}")
            else:
                print(f"Position {p.ticket} closed successfully at bar end.")

def open_trade(symbol, action, tp_price_diff=0.01):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return

    ask = mt5.symbol_info_tick(symbol).ask
    bid = mt5.symbol_info_tick(symbol).bid
    
    if action == "BUY":
        price = ask
        tp = price * (1 + tp_percent)
        type_ = mt5.ORDER_TYPE_BUY
    elif action == "SELL":
        price = bid
        tp = price * (1 - tp_percent)
        type_ = mt5.ORDER_TYPE_SELL
    else:
        return

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": LOT_SIZE,
        "type": type_,
        "price": price,
        "tp": tp,
        # SL is omitted completely as requested
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "ML CNN-LSTM",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed, retcode={result.retcode}")
    else:
        print(f"Order placed successfully: {action} at {price}, TP: {tp}")

def wait_for_new_bar(current_bar_time):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for the next bar to open...")
    while True:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
        if rates is not None and len(rates) > 0:
            latest_time = rates[0]['time']
            if latest_time != current_bar_time:
                return latest_time
        time.sleep(60) # Check every 60 seconds

# --- Main Logic ---
def main():
    init_mt5()
    
    print("Fetching historical data...")
    df = fetch_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    if df is None or len(df) < SEQUENCE_LENGTH * 2:
        print("Not enough data to train model.")
        return
        
    print("Preparing data...")
    X, y, scaler = prepare_data(df, SEQUENCE_LENGTH)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("Building and training CNN+LSTM model (this may take a moment)...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    print("--- Starting Live Trading Loop ---")
    
    # Get current bar time to know when a new one starts
    initial_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
    current_bar_time = initial_rates[0]['time'] if initial_rates is not None else 0
    
    # Optional: if you start the script mid-bar, wait until the next bar to start properly
    current_bar_time = wait_for_new_bar(current_bar_time)
    
    while True:
        print(f"--- New Bar Started: {datetime.now().strftime('%H:%M:%S')} ---")
        
        # 1. Close open positions from previous bar
        close_all_positions(SYMBOL, MAGIC_NUMBER)
        
        # 2. Check News Filter
        if is_high_impact_news_approaching():
            print("News filter active. Skipping trade this bar.")
            current_bar_time = wait_for_new_bar(current_bar_time)
            continue
            
        # 3. Predict direction
        latest_df = fetch_data(SYMBOL, TIMEFRAME, SEQUENCE_LENGTH + 40) # Fetch more for ADX/LinReg windows
        
        # Calculate all indicators for the live data
        latest_df = add_features(latest_df)
        latest_df = latest_df.dropna()
        
        latest_features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20']
        scaled_latest = scaler.transform(latest_df[latest_features])
        X_live = np.array([scaled_latest[-SEQUENCE_LENGTH:]])
        prediction = model.predict(X_live, verbose=0)[0][0]
        
        print(f"ML Prediction (Up probability): {prediction:.4f}")
        
        # 4. Evaluate Prediction and Spread
        if prediction > 0.505:
            print("Strong Up trend detected -> BUY")
            if is_spread_ok(SYMBOL):
                open_trade(SYMBOL, "BUY", tp_price_diff=0.01)
            else:
                print("Spread too high, skipped.")
                
        else:
            print("No clear trend or Down trend. Skipping trade (Long Only Mode).")
            
        # 5. Wait until next bar starts
        current_bar_time = wait_for_new_bar(current_bar_time)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Bot stopped by user.")
    finally:
        mt5.shutdown()
