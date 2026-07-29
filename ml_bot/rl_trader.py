import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import MetaTrader5 as mt5
from stable_baselines3 import PPO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from rl_train import fetch_data, add_features

# Matches the CURRENT rl_env.py / rl_train.py (Always-In strategy):
#   - observation: 20 x (14 MinMax-scaled features + raw spread_cost + raw
#     atr_14) = shape (20, 16)
#   - action 0 = BUY, action 1 = SELL -- a position is opened EVERY bar,
#     there is no flat/skip state
#   - TP = min(atr * TP_MULTIPLIER, $3.00), SL (in training) = atr *
#     SL_MULTIPLIER with NO cap
#   - dynamic lot 0.01 per $100 equity, capped at 10 lots, $100/oz/lot
#     (i.e. $1 price move = $100 * lot_size)
#
# IMPORTANT: the trained SL (atr * 2.0, uncapped) is not safe to send to a
# broker at this position sizing. At equity $10,000 (lot 1.0), a $100 ATR
# gives an SL distance of $200 -> a $20,000 loss, 200% of equity, in one
# bar. This trader instead attaches a broker-side SL sized as a % of
# equity (SL_EQUITY_PCT) -- the deployed risk is therefore NOT identical
# to what the agent was trained against; see backtest_offline.py for a
# side-by-side simulation of both.

# --- Configuration ---
SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
MAGIC_NUMBER = 999999
WINDOW_SIZE = 20
TP_MULTIPLIER = 1.0    # must match TP_MULTIPLIER in rl_train.py
TP_CAP = 3.00          # env caps TP distance at $3.00 (rl_env.py)
SL_EQUITY_PCT = 5.0    # broker-side disaster stop, see note above
MAX_LOT = 10.0
RETRY_SECONDS = 300
MODEL_FILE = "rl_model"
SCALER_FILE = "rl_scaler.save"

# 14 scaled inputs; the model additionally sees raw spread_cost and atr_14
# (see prepare_rl_data in rl_train.py / feature_cols in rl_env.py)
SCALED_FEATURES = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10',
                   'sma_20', 'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y',
                   'atr_14', 'day_of_week']
RAW_FEATURES = ['spread_cost', 'atr_14']


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


def compute_lot(symbol_info, equity):
    # Mandated sizing: 0.01 lot per $100 of equity
    lot = (equity / 100.0) * 0.01
    lot = min(lot, MAX_LOT)
    step = symbol_info.volume_step or 0.01
    lot = round(lot / step) * step
    lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
    return round(lot, 2)


def open_trade(symbol, action_type, atr):
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

    lot_size = compute_lot(symbol_info, account_info.equity)

    tp_dist = min(atr * TP_MULTIPLIER, TP_CAP)
    # price distance that loses SL_EQUITY_PCT% of equity at this lot size
    contract = symbol_info.trade_contract_size or 100.0
    sl_dist = (SL_EQUITY_PCT / 100.0) * account_info.equity / (lot_size * contract)

    min_stop = symbol_info.trade_stops_level * symbol_info.point
    tp_dist = max(tp_dist, min_stop)
    sl_dist = max(sl_dist, min_stop)

    if action_type == "BUY":
        price = tick.ask
        tp = price + tp_dist
        sl = price - sl_dist
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
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
        "magic": MAGIC_NUMBER,
        "comment": f"RL_Agent_{action_type}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else mt5.last_error()
        print(f"Order send failed, retcode={code}")
    else:
        print(f"{action_type} order sent! Ticket: {result.order}, Lot: {lot_size}, "
              f"TP_dist: {tp_dist:.2f}, SL_dist: {sl_dist:.2f}")


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
            # Env is Always-In (a position is opened every bar and always
            # closed by EOD/TP/SL), so any leftover position at this point
            # is a stale one from a previous crash/restart -- clear it.
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
            atr = float(df['atr_14'].iloc[-1])

            if action_idx == 0:
                print("RL Agent decided to: BUY")
                open_trade(SYMBOL, "BUY", atr)
            else:
                print("RL Agent decided to: SELL")
                open_trade(SYMBOL, "SELL", atr)
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
