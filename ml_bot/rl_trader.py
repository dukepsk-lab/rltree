import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import joblib
import json
import os
import tensorflow as tf
import xgboost as xgb
from stable_baselines3 import PPO

# Load AI Levels
LEVELS_FILE = 'ml_bot/ai_levels.json'
if not os.path.exists(LEVELS_FILE):
    default_levels = {
        "RL": {"exp": 0, "level": 1, "last_pred": None, "last_date": None},
        "CNN": {"exp": 0, "level": 1, "last_pred": None, "last_date": None},
        "XGB": {"exp": 0, "level": 1, "last_pred": None, "last_date": None}
    }
    with open(LEVELS_FILE, 'w') as f:
        json.dump(default_levels, f)

with open(LEVELS_FILE, 'r') as f:
    ai_levels = json.load(f)

def evaluate_yesterday(df, ai_levels):
    # Check if we have a prediction to evaluate
    if ai_levels["RL"]["last_pred"] is None:
        return ai_levels
        
    last_date = ai_levels["RL"]["last_date"]
    
    # Simple check: Did yesterday close higher than it opened?
    if len(df) >= 2:
        yesterday_bar = df.iloc[-2] # Current is -1 (today's open), yesterday is -2
        actual_dir = 0 if yesterday_bar['close'] > yesterday_bar['open'] else 1
        
        for m in ai_levels:
            pred = ai_levels[m]["last_pred"]
            if pred == actual_dir:
                ai_levels[m]["exp"] += 10
            else:
                ai_levels[m]["exp"] = max(0, ai_levels[m]["exp"] - 5)
            
            ai_levels[m]["level"] = (ai_levels[m]["exp"] // 100) + 1
            
    return ai_levels

def get_ensemble_action(obs, rl_model, cnn_model, xgb_model, ai_levels, today_date):
    # 1. Get Predictions
    rl_pred = int(rl_model.predict(obs, deterministic=True)[0])
    
    obs_cnn = np.expand_dims(obs, axis=0)
    cnn_prob = cnn_model.predict(obs_cnn, verbose=0)[0][0]
    cnn_pred = 0 if cnn_prob <= 0.5 else 1
    
    obs_xgb = obs.flatten().reshape(1, -1)
    xgb_pred = int(xgb_model.predict(obs_xgb)[0])
    
    # 2. Update levels state for tomorrow
    ai_levels["RL"]["last_pred"] = rl_pred
    ai_levels["RL"]["last_date"] = str(today_date)
    ai_levels["CNN"]["last_pred"] = cnn_pred
    ai_levels["CNN"]["last_date"] = str(today_date)
    ai_levels["XGB"]["last_pred"] = xgb_pred
    ai_levels["XGB"]["last_date"] = str(today_date)
    
    with open(LEVELS_FILE, 'w') as f:
        json.dump(ai_levels, f, indent=4)
        
    # 3. Weighted Voting
    buy_score = 0
    sell_score = 0
    
    if rl_pred == 0: buy_score += ai_levels["RL"]["level"]
    else: sell_score += ai_levels["RL"]["level"]
    
    if cnn_pred == 0: buy_score += ai_levels["CNN"]["level"]
    else: sell_score += ai_levels["CNN"]["level"]
    
    if xgb_pred == 0: buy_score += ai_levels["XGB"]["level"]
    else: sell_score += ai_levels["XGB"]["level"]
    
    print(f"--- VOTING RESULTS ---")
    print(f"RL  (Lv.{ai_levels['RL']['level']}): {'BUY' if rl_pred==0 else 'SELL'}")
    print(f"CNN (Lv.{ai_levels['CNN']['level']}): {'BUY' if cnn_pred==0 else 'SELL'}")
    print(f"XGB (Lv.{ai_levels['XGB']['level']}): {'BUY' if xgb_pred==0 else 'SELL'}")
    print(f"Total Score -> BUY: {buy_score}, SELL: {sell_score}")
    
    return 0 if buy_score > sell_score else 1


from rl_train import fetch_data, add_features
from features import build_observation

# For a single forecast (with a compatibility gate, a per-trade risk cap and a
# dry-run default) prefer `python ml_bot/d1_forecast.py`. This module is the
# always-on loop version and has no risk cap beyond the ATR stop.

# --- Configuration ---
SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
MAGIC_NUMBER = 999999
LOT_SIZE = 0.01
WINDOW_SIZE = 20
TP_MULTIPLIER = 1.0
SL_MULTIPLIER = 2.0

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
        # Must match MAGIC_NUMBER, otherwise close_all_positions() never finds
        # the positions this function opens and they are held indefinitely.
        "magic": MAGIC_NUMBER,
        "comment": f"RL_Agent_{action_type}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order send failed, retcode={result.retcode}")
    else:
        print(f"Order sent successfully! Ticket: {result.order}, Volume: {lot_size}, TP_dist: {tp_dist:.2f}, SL_dist: {sl_dist:.2f}")

def wait_for_new_bar(current_bar_time):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for the next D1 bar to open...")
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

        # +1 bar because the last row returned by fetch_data is the bar currently
        # forming; the policy was only ever trained on completed bars.
        df = fetch_data(SYMBOL, TIMEFRAME, WINDOW_SIZE + 41)
        df = add_features(df).dropna().iloc[:-1]

        # Observation layout must match TradingEnv: scaled features + the raw
        # spread_cost / atr_14 columns, in the env's own column order.
        obs, _cols = build_observation(df, scaler, WINDOW_SIZE)

        expected = tuple(model.observation_space.shape)
        if tuple(obs.shape) != expected:
            print(f"Model/scaler mismatch: policy expects {expected}, pipeline built {obs.shape}. "
                  f"Retrain with rl_train.py. Not trading.")
            return

        action, _states = model.predict(obs, deterministic=True)
        action_idx = int(action)

        # TradingEnv convention: 0 = BUY, 1 = SELL. There is no flat action.
        if action_idx == 0:
            print("RL Agent decided to: BUY")
            open_trade(SYMBOL, "BUY", TP_MULTIPLIER, SL_MULTIPLIER)
        else:
            print("RL Agent decided to: SELL")
            open_trade(SYMBOL, "SELL", TP_MULTIPLIER, SL_MULTIPLIER)

        current_bar_time = wait_for_new_bar(current_bar_time)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Bot stopped by user.")
    finally:
        mt5.shutdown()
