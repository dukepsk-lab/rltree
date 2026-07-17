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

from rl_train import fetch_data, add_features

# NOTE: the committed rl_model.zip (1M-step PPO) was trained on the legacy
# TradingEnv (git 6150c7d): long-only, action 1 = BUY at the daily open,
# flat TP = entry + $3.00, forced close at end of day, no SL, dynamic lot
# 0.01 per $100 equity capped at 10. Its observation is 12 MinMax-scaled
# features + raw spread_cost => shape (20, 13). Its matching scaler is
# rl_scaler_d1_legacy.save (the first 12 columns of rl_scaler.save, which
# was fitted the same day on the same broker data; the committed
# rl_scaler_legacy.save is a 10-feature scaler from an older model and
# does NOT match). This trader reproduces exactly that behavior and
# adds a broker-side disaster SL the training env did not have.
# If you retrain with the current rl_train.py/rl_env.py (Always-In, 16
# features), update SCALED_FEATURES/RAW_FEATURES, SCALER_FILE and the
# action mapping below.

# --- Configuration ---
SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
MAGIC_NUMBER = 999999
WINDOW_SIZE = 20
TP_PRICE_DIFF = 3.00      # flat $ TP distance, must match legacy training
# Disaster stop as % of equity (not part of the trained policy). At the
# mandated sizing (0.01 lot per $100) a $1/oz move is ~1% of equity, so an
# ATR-based stop (~2xATR = $130) would exceed 100% of equity in one bar.
SL_EQUITY_PCT = 5.0
MAX_LOT = 10.0
RETRY_SECONDS = 300
MODEL_FILE = "rl_model"
SCALER_FILE = "rl_scaler_d1_legacy.save"

SCALED_FEATURES = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10',
                   'sma_20', 'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y']
RAW_FEATURES = ['spread_cost']


def init_mt5():
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code: {mt5.last_error()}")
        sys.exit(1)
    print("MT5 Initialized Successfully")


def build_obs(df, scaler):
    """(1, WINDOW_SIZE, 13) observation exactly as in legacy training:
    12 MinMax-scaled features followed by raw spread_cost."""
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


def compute_lot(symbol_info, equity):
    # Legacy sizing: 0.01 lot per $100 of equity, capped
    lot = (equity / 100.0) * 0.01
    lot = min(lot, MAX_LOT)
    step = symbol_info.volume_step or 0.01
    lot = round(lot / step) * step
    lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
    return round(lot, 2)


def open_trade(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found.")
        return

    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get account info")
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("No tick data; skipping entry.")
        return

    price = tick.ask
    lot_size = compute_lot(symbol_info, account_info.equity)

    tp_dist = TP_PRICE_DIFF
    # price distance that loses SL_EQUITY_PCT% of equity at this lot size
    contract = symbol_info.trade_contract_size or 100.0
    sl_dist = (SL_EQUITY_PCT / 100.0) * account_info.equity / (lot_size * contract)

    min_stop = symbol_info.trade_stops_level * symbol_info.point
    tp_dist = max(tp_dist, min_stop)
    sl_dist = max(sl_dist, min_stop)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": price - sl_dist,
        "tp": price + tp_dist,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "RL_Agent_BUY",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else mt5.last_error()
        print(f"Order send failed, retcode={code}")
    else:
        print(f"BUY order sent! Ticket: {result.order}, Lot: {lot_size}, "
              f"TP: {price + tp_dist:.2f}, SL: {price - sl_dist:.2f}")


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
        model = PPO.load(os.path.join(BASE_DIR, MODEL_FILE))
        scaler = joblib.load(os.path.join(BASE_DIR, SCALER_FILE))
    except Exception as e:
        print(f"Failed to load model or scaler: {e}")
        return

    expected_shape = (WINDOW_SIZE, len(SCALED_FEATURES) + len(RAW_FEATURES))
    if tuple(model.observation_space.shape) != expected_shape:
        print(f"Model observation space {model.observation_space.shape} does not "
              f"match trader features {expected_shape}. Retrain or fix features.")
        return
    if getattr(scaler, 'n_features_in_', None) != len(SCALED_FEATURES):
        print(f"Scaler expects {getattr(scaler, 'n_features_in_', '?')} features, "
              f"trader provides {len(SCALED_FEATURES)}. Wrong scaler file.")
        return

    print("--- RL Live Trading Loop Started ---")

    initial_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
    current_bar_time = initial_rates[0]['time'] if initial_rates is not None else 0

    current_bar_time = wait_for_new_bar(current_bar_time)

    while True:
        print(f"--- New Bar Started: {datetime.now().strftime('%H:%M:%S')} ---")

        try:
            # Legacy behavior: position never held across the daily bar
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

            # Legacy env: action 1 = BUY at open, action 0 = Flat/Skip the day
            if action_idx == 1:
                print("RL Agent decided to: BUY")
                open_trade(SYMBOL)
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
