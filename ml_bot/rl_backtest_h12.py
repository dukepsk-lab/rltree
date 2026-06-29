import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import joblib
from rl_env import TradingEnv
from rl_train_h12 import fetch_data, add_features

SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_H12
DATA_LIMIT = 5000
WINDOW_SIZE = 20
TP_PRICE_DIFF = 0.01

def main():
    if not mt5.initialize():
        print("MT5 init failed")
        return
        
    print("Fetching data for testing...")
    df = fetch_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    mt5.shutdown()
    
    if df is None:
        return
        
    df = add_features(df).dropna()
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y']
    
    # Load Scaler
    try:
        scaler = joblib.load('ml_bot/rl_scaler_h12.save')
    except Exception as e:
        print(f"Scaler not found or error loading: {e}. Train the model first using rl_train_h12.py.")
        return
        
    scaled_data = scaler.transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=[f"scaled_{f}" for f in features], index=df.index)
    final_df = pd.concat([scaled_df, df[['open', 'high', 'low', 'close', 'spread_cost']]], axis=1)
    
    # We will test on the last 20% of data
    split = int(len(final_df) * 0.8)
    test_df = final_df.iloc[split:].copy()
    
    print(f"Testing on {len(test_df)} samples...")
    
    # Load Model
    try:
        model = PPO.load("ml_bot/rl_model_h12")
    except Exception as e:
        print(f"Model not found or error loading: {e}. Train the model first using rl_train_h12.py.")
        return
        
    # Create test env
    env = DummyVecEnv([lambda: TradingEnv(test_df, WINDOW_SIZE, tp_price_diff=TP_PRICE_DIFF)])
    obs = env.reset()
    
    # DummyVecEnv wraps returns in arrays
    dones = [False]
    equity_curve = [10000.0]
    trades = 0
    
    print("Running Backtest...")
    while not dones[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        equity_curve.append(infos[0]['balance'])
        
        # Simple trade counter: action 1 or 2 is a new decision, 0 is hold/flat. 
        # In reality, this requires tracking position state from env, but we'll keep it simple here.
        if action[0] != 0:
            trades += 1
            
    final_balance = equity_curve[-1]
    net_profit = final_balance - 10000.0
    
    print("\n=== RL BACKTEST RESULTS ===")
    print(f"Initial Capital: $10000.00")
    print(f"Final Capital:   ${final_balance:.2f}")
    print(f"Net Profit:      ${net_profit:.2f} ({(net_profit/10000.0*100):.2f}%)")
    print(f"Decision Changes (Actions taken): {trades}")
    print("===========================\n")
        
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label='RL Equity Curve', color='green')
    plt.title(f"RL Agent (PPO) Backtest on {SYMBOL} (H12)")
    plt.xlabel("Time Step (Test Set)")
    plt.ylabel("Account Balance ($)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
