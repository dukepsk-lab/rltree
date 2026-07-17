import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from stable_baselines3 import PPO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from rl_train_h8 import fetch_data, add_features

# NOTE: rl_model_h8.zip was trained on the legacy TradingEnv (long-only,
# action 1 = BUY at open, flat TP = entry + $3.00, EOD close, no SL,
# fixed 0.01 lot). This trader mirrors that behavior and adds a
# broker-side disaster SL that the training env did not have.

# --- Configuration ---
SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_H8
MAGIC_NUMBER = 999999
LOT_SIZE = 0.01
WINDOW_SIZE = 20
TP_PRICE_DIFF = 3.00   # must match TP_PRICE_DIFF in rl_train_h8.py
SL_ATR_MULTIPLIER = 2.0  # disaster stop, not part of the trained policy
RETRY_SECONDS = 300

SCALED_FEATURES = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10',
                   'sma_20', 'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y']
RAW_FEATURES = ['spread_cost']


def init_mt5():
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code: {mt5.last_error()}")
        sys.exit(1)
    print("MT5 Initialized Successfully")


def build_obs(df, scaler):
    scaled = scaler.transform(df[SCALED_FEATURES])
    raw = df[RAW_FEATURES].values
    stacked = np.hstack([scaled, raw])
    return np.array([stacked[-WINDOW_SIZE:]], dtype=np.float32)


def close_all_positions(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return

    for p in positions:
        if p.magic != magic:
            continue
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print("No tick data; cannot close position now.")
            return
        type_dict = {mt5.POSITION_TYPE_BUY: mt5.ORDER_TYPE_SELL,
                     mt5.POSITION_TYPE_SELL: mt5.ORDER_TYPE_BUY}
        price_dict = {mt5.POSITION_TYPE_BUY: tick.bid,
                      mt5.POSITION_TYPE_SELL: tick.ask}

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
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else mt5.last_error()
            print(f"Failed to close position {p.ticket}, retcode={code}")
        else:
            print(f"Position {p.ticket} closed successfully.")


def open_trade(symbol, atr):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found.")
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("No tick data; skipping entry.")
        return

    price = tick.ask
    tp_dist = TP_PRICE_DIFF
    sl_dist = atr * SL_ATR_MULTIPLIER

    min_stop = symbol_info.trade_stops_level * symbol_info.point
    tp_dist = max(tp_dist, min_stop)
    sl_dist = max(sl_dist, min_stop)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": price - sl_dist,
        "tp": price + tp_dist,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "RL PPO Agent H8",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else mt5.last_error()
        print(f"Order failed, retcode={code}")
    else:
        print(f"BUY placed at {price}, TP: {price + tp_dist:.2f}, SL: {price - sl_dist:.2f}")


def compute_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return 1.0 if pd.isna(atr) else float(atr)


def wait_for_new_bar(current_bar_time):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for the next H8 bar to open...")
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
        model = PPO.load(os.path.join(BASE_DIR, "rl_model_h8"))
        scaler = joblib.load(os.path.join(BASE_DIR, "rl_scaler_h8.save"))
    except Exception as e:
        print(f"Failed to load model or scaler: {e}")
        return

    expected_shape = (WINDOW_SIZE, len(SCALED_FEATURES) + len(RAW_FEATURES))
    if tuple(model.observation_space.shape) != expected_shape:
        print(f"Model observation space {model.observation_space.shape} does not "
              f"match trader features {expected_shape}. Retrain or fix features.")
        return

    print("--- RL Live Trading Loop Started ---")

    initial_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
    current_bar_time = initial_rates[0]['time'] if initial_rates is not None else 0

    current_bar_time = wait_for_new_bar(current_bar_time)

    while True:
        print(f"--- New Bar Started: {datetime.now().strftime('%H:%M:%S')} ---")

        try:
            close_all_positions(SYMBOL, MAGIC_NUMBER)

            df = fetch_data(SYMBOL, TIMEFRAME, WINDOW_SIZE + 60)
            if df is None or len(df) == 0:
                print(f"No rates returned; retrying in {RETRY_SECONDS}s.")
                time.sleep(RETRY_SECONDS)
                continue

            df = add_features(df).dropna()
            if len(df) < WINDOW_SIZE:
                print(f"Not enough bars after features ({len(df)}); retrying in {RETRY_SECONDS}s.")
                time.sleep(RETRY_SECONDS)
                continue

            obs = build_obs(df, scaler)
            action, _states = model.predict(obs, deterministic=True)
            action_idx = int(action[0])

            # Legacy env: action 1 = BUY, action 0 = Flat/Skip
            if action_idx == 1:
                print("RL Agent decided to: BUY")
                open_trade(SYMBOL, compute_atr(df))
            else:
                print("RL Agent decided to: HOLD/FLAT")
        except Exception as e:
            print(f"Error in trading loop: {e}; retrying in {RETRY_SECONDS}s.")
            time.sleep(RETRY_SECONDS)
            continue

        current_bar_time = wait_for_new_bar(current_bar_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Bot stopped by user.")
    finally:
        mt5.shutdown()
