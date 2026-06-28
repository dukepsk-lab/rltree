import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import TradingEnv
import os
import joblib

# --- Configuration ---
SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_H8
DATA_LIMIT = 5000
WINDOW_SIZE = 20
TP_PERCENT = 0.03 # Default to 3% as analyzed
TIMESTEPS = 200000

def init_mt5():
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code: {mt5.last_error()}")
        quit()
    print("MT5 Initialized Successfully")

def fetch_data(symbol, timeframe, n_bars):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None or not symbol_info.visible:
        mt5.symbol_select(symbol, True)
            
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Calculate Spread Cost
    point = symbol_info.point
    df['spread_cost'] = df['spread'] * point
    
    return df[['open', 'high', 'low', 'close', 'tick_volume', 'spread_cost']]

def add_macro_data(df):
    start_date = df.index.min()
    end_date = df.index.max() + pd.Timedelta(days=1)
    
    # Fetch DXY and US10Y silently
    dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date, progress=False)['Close']
    us10y = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close']
    
    # Ensure they are Series
    if isinstance(dxy, pd.DataFrame): dxy = dxy.iloc[:, 0]
    if isinstance(us10y, pd.DataFrame): us10y = us10y.iloc[:, 0]
    
    macro_df = pd.DataFrame({'dxy': dxy, 'us10y': us10y})
    
    # tz-localize macro data to None since MT5 is naive
    if macro_df.index.tz is not None:
        macro_df.index = macro_df.index.tz_localize(None)
    
    # Merge and forward fill for holidays/weekends
    df = df.join(macro_df, how='left')
    df['dxy'] = df['dxy'].ffill().bfill()
    df['us10y'] = df['us10y'].ffill().bfill()
    
    return df

def add_features(df):
    df = df.copy()
    df = add_macro_data(df)
    
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

def prepare_rl_data(df):
    df = add_features(df).dropna()
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y']
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    scaled_df = pd.DataFrame(scaled_data, columns=[f"scaled_{f}" for f in features], index=df.index)
    final_df = pd.concat([scaled_df, df[['open', 'high', 'low', 'close', 'spread_cost']]], axis=1)
    return final_df, scaler

def main():
    init_mt5()
    print("Fetching data for training...")
    df = fetch_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    mt5.shutdown()
    
    if df is None:
        print("Failed to fetch data.")
        return
        
    print("Preparing data and fetching Macro variables...")
    rl_df, scaler = prepare_rl_data(df)
    joblib.dump(scaler, 'ml_bot/rl_scaler_h8.save')
    
    env = DummyVecEnv([lambda: TradingEnv(rl_df, WINDOW_SIZE, TP_PERCENT)])
    
    print("Training PPO Agent on GPU with Deep Architecture...")
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, 
                device='cuda', policy_kwargs=policy_kwargs)
    
    model.learn(total_timesteps=TIMESTEPS)
    
    model.save("ml_bot/rl_model_h8")
    print("Training complete! Model saved as rl_model_h8.zip")

if __name__ == "__main__":
    main()
