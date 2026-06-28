import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import TradingEnv

def fetch_data(symbol, timeframe, n_bars):
    if not mt5.initialize(): return None
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    mt5.shutdown()
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

def run_backtest(symbol, timeframe, data_limit, window_size, tp_percent, model_path, scaler_path):
    df = fetch_data(symbol, timeframe, data_limit)
    df = add_features(df).dropna()
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20']
    
    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=[f"scaled_{f}" for f in features], index=df.index)
    final_df = pd.concat([scaled_df, df[['open', 'high', 'low', 'close']]], axis=1)
    
    split = int(len(final_df) * 0.8)
    test_df = final_df.iloc[split:].copy()
    
    model = PPO.load(model_path)
    env = DummyVecEnv([lambda: TradingEnv(test_df, window_size, tp_percent)])
    obs = env.reset()
    
    dones = [False]
    equity_curve = [10000.0]
    trades = 0
    
    while not dones[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        equity_curve.append(infos[0]['balance'])
        if action[0] != 0:
            trades += 1
            
    final_balance = equity_curve[-1]
    net_profit = final_balance - 10000.0
    return net_profit, trades

def optimize():
    tps = [0.01, 0.015, 0.02, 0.025, 0.03]
    print("--- Optimizing D1 (Day) ---")
    for tp in tps:
        profit, trades = run_backtest("XAUUSD.", mt5.TIMEFRAME_D1, 5000, 20, tp, "ml_bot/rl_model", "ml_bot/rl_scaler.save")
        print(f"TP {int(tp*100)}% -> Net Profit: ${profit:.2f} ({(profit/10000*100):.2f}%) | Trades: {trades}")

    print("\n--- Optimizing H12 ---")
    for tp in tps:
        profit, trades = run_backtest("XAUUSD.", mt5.TIMEFRAME_H12, 5000, 20, tp, "ml_bot/rl_model_h12", "ml_bot/rl_scaler_h12.save")
        print(f"TP {int(tp*100)}% -> Net Profit: ${profit:.2f} ({(profit/10000*100):.2f}%) | Trades: {trades}")

if __name__ == '__main__':
    optimize()
