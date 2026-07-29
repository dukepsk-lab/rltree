import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import TradingEnv
# Feature engineering lives in features.py so training and live serving cannot
# drift apart. Re-exported here because the other scripts import it from rl_train.
from features import FEATURES, add_macro_data, add_features
import os
import json
import joblib

# --- Configuration ---
SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
DATA_LIMIT = 5000
WINDOW_SIZE = 20
TP_MULTIPLIER = 1.0 # Will be updated by Optuna
SL_MULTIPLIER = 2.0 # Will be updated by Optuna
TIMESTEPS = 1000000

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

def prepare_rl_data(df):
    df = add_features(df).dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])

    scaled_df = pd.DataFrame(scaled_data, columns=[f"scaled_{f}" for f in FEATURES], index=df.index)
    final_df = pd.concat([scaled_df, df[['open', 'high', 'low', 'close', 'spread_cost', 'atr_14']]], axis=1)
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
    joblib.dump(scaler, 'ml_bot/rl_scaler.save')
    
    env = DummyVecEnv([lambda: TradingEnv(rl_df, WINDOW_SIZE, TP_MULTIPLIER, SL_MULTIPLIER)])
    
    print("Training PPO Agent on GPU with Deep Architecture...")
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=0.0033597434479013337, 
                gamma=0.9517781177590708,
                n_steps=4096, 
                ent_coef=0.002157175081922516,
                device='auto', policy_kwargs=policy_kwargs)
    
    model.learn(total_timesteps=TIMESTEPS)

    model.save("ml_bot/rl_model")

    # Record what this model was actually trained on. d1_forecast.py checks this
    # so a model/scaler pair from different feature sets is rejected loudly
    # instead of producing a shape error (or worse, a silently wrong forecast).
    meta = {
        "features": FEATURES,
        "window_size": WINDOW_SIZE,
        "observation_columns": [c for c in env.get_attr("feature_cols")[0]],
        "observation_shape": list(env.observation_space.shape),
        "tp_multiplier": TP_MULTIPLIER,
        "sl_multiplier": SL_MULTIPLIER,
        "timesteps": TIMESTEPS,
        "bars": len(rl_df),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    with open("ml_bot/rl_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Training complete! Model saved as rl_model.zip (+ rl_model_meta.json)")

if __name__ == "__main__":
    main()
