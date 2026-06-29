import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import joblib
from stable_baselines3 import PPO

# --- Configuration ---
SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
TP_PERCENT = 0.03
WINDOW_SIZE = 20

def fetch_data(symbol, timeframe, n_bars):
    if not mt5.initialize(): return None
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume']]

def add_features(df):
    df = df.copy()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    up, down = high - high.shift(), low.shift() - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_14 = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).sum() / (tr_14 + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).sum() / (tr_14 + 1e-9))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['adx_14'] = dx.rolling(14).mean()
    
    window = 20
    x = np.arange(window)
    sum_x, sum_x2 = x.sum(), (x**2).sum()
    denom = window * sum_x2 - sum_x**2
    def slope_func(y):
        return (window * (x * y).sum() - sum_x * y.sum()) / denom
    df['linreg_20'] = df['close'].rolling(window=window).apply(slope_func, raw=True)
    return df

def open_trade(symbol, action_type, tp_percent):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found.")
        return
        
    price = mt5.symbol_info_tick(symbol).ask if action_type == "BUY" else mt5.symbol_info_tick(symbol).bid
    tp = price + (price * tp_percent) if action_type == "BUY" else price - (price * tp_percent)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY if action_type == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": 0.0,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": f"RL_Agent_{action_type}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order send failed, retcode={result.retcode}")
    else:
        print(f"Order sent successfully! Ticket: {result.order}")

def main():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return
        
    print(f"Starting Legacy RL Live Trading Agent for {SYMBOL}...")
    try:
        model = PPO.load("ml_bot/rl_model")
        scaler = joblib.load("ml_bot/rl_scaler_legacy.save")
        print("Legacy Model and Scaler loaded successfully.")
    except Exception as e:
        print(f"Failed to load legacy model/scaler: {e}")
        return
        
    while True:
        try:
            df = fetch_data(SYMBOL, TIMEFRAME, WINDOW_SIZE + 40)
            df = add_features(df).dropna()
            
            features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20']
            scaled_data = scaler.transform(df[features])
            
            obs = np.array([scaled_data[-WINDOW_SIZE:]], dtype=np.float32)
            
            action, _states = model.predict(obs, deterministic=True)
            action_idx = action[0]
            
            print(f"[{datetime.now()}] Current state analyzed. ", end="")
            
            if action_idx == 1:
                print("RL Agent decided to: BUY")
                open_trade(SYMBOL, "BUY", tp_percent=TP_PERCENT)
            else:
                print("RL Agent decided to: HOLD/FLAT")
                
            time.sleep(3600)
            
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
