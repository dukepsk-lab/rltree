import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import joblib
from stable_baselines3 import PPO
from rl_train import fetch_data, add_features

# --- Configuration ---
SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
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

def open_trade(symbol, action_type, tp_multiplier, sl_multiplier):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found.")
        return
        
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get account info")
        return
        
    # Get recent ATR
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 15)
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(rates)
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    if pd.isna(atr): atr = 1.0
    
    tp_dist = min(atr * tp_multiplier, 3.00)
    sl_dist = atr * sl_multiplier
        
    # Dynamic lot size: 0.01 lot per $100 of equity
    equity = account_info.equity
    lot_size = (equity / 100.0) * 0.01
    lot_size = min(lot_size, 10.0) # Cap max lot size at 10.0
    lot_size = round(lot_size, 2)
    
    if lot_size < symbol_info.volume_min:
        lot_size = symbol_info.volume_min
    elif lot_size > symbol_info.volume_max:
        lot_size = symbol_info.volume_max
        
    price = mt5.symbol_info_tick(symbol).ask if action_type == "BUY" else mt5.symbol_info_tick(symbol).bid
    
    if action_type == "BUY":
        tp = price + tp_dist
        sl = price - sl_dist
        order_type = mt5.ORDER_TYPE_BUY
    else:
        tp = price - tp_dist
        sl = price + sl_dist
        order_type = mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
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
        print(f"Order sent successfully! Ticket: {result.order}, Volume: {lot_size}, TP_dist: {tp_dist:.2f}, SL_dist: {sl_dist:.2f}")

def main():
    init_mt5()
    
    print("Loading RL Model and Scaler...")
    try:
        model = PPO.load("ml_bot/rl_model")
        scaler = joblib.load('ml_bot/rl_scaler.save')
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
        
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y', 'atr_14']
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
