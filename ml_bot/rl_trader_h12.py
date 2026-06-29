import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import joblib
from stable_baselines3 import PPO
from rl_train_h12 import fetch_data, add_features

# --- Configuration ---
SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_H12
MAGIC_NUMBER = 999999
LOT_SIZE = 0.01
WINDOW_SIZE = 20
TP_PRICE_DIFF = 0.01

def init_mt5():
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code: {mt5.last_error()}")
        quit()
    print("MT5 Initialized Successfully")

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
                "comment": "RL Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Failed to close position {p.ticket}, retcode={result.retcode}")
            else:
                print(f"Position {p.ticket} closed successfully.")

def open_trade(symbol, action_type, tp_price_diff=0.01):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return

    ask = mt5.symbol_info_tick(symbol).ask
    bid = mt5.symbol_info_tick(symbol).bid
    
    if action_type == "BUY":
        price = ask
        tp = price * (1 + tp_percent)
        type_ = mt5.ORDER_TYPE_BUY
    elif action_type == "SELL":
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
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "RL PPO Agent",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed, retcode={result.retcode}")
    else:
        print(f"Order placed successfully: {action_type} at {price}, TP: {tp}")

def wait_for_new_bar(current_bar_time):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for the next H12 bar to open...")
    while True:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
        if rates is not None and len(rates) > 0:
            latest_time = rates[0]['time']
            if latest_time != current_bar_time:
                return latest_time
        time.sleep(60)

def main():
    init_mt5()
    
    print("Loading RL Model and Scaler...")
    try:
        model = PPO.load("ml_bot/rl_model_h12")
        scaler = joblib.load('ml_bot/rl_scaler_h12.save')
    except Exception as e:
        print(f"Failed to load model or scaler: {e}")
        return
        
    print("--- RL Live Trading Loop Started ---")
    
    initial_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
    current_bar_time = initial_rates[0]['time'] if initial_rates is not None else 0
    
    current_bar_time = wait_for_new_bar(current_bar_time)
    
    while True:
        print(f"--- New Bar Started: {datetime.now().strftime('%H:%M:%S')} ---")
        
        close_all_positions(SYMBOL, MAGIC_NUMBER)
        
        df = fetch_data(SYMBOL, TIMEFRAME, WINDOW_SIZE + 40)
        df = add_features(df).dropna()
        
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y']
        scaled_data = scaler.transform(df[features])
        
        obs = np.array([scaled_data[-WINDOW_SIZE:]], dtype=np.float32)
        
        action, _states = model.predict(obs, deterministic=True)
        action_idx = action[0]
        
        if action_idx == 1:
            print("RL Agent decided to: BUY")
            open_trade(SYMBOL, "BUY", tp_price_diff=TP_PRICE_DIFF)
        else:
            print("RL Agent decided to: HOLD/FLAT")
            
        current_bar_time = wait_for_new_bar(current_bar_time)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Bot stopped by user.")
    finally:
        mt5.shutdown()
