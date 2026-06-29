import optuna
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import joblib
from rl_env import TradingEnv
from rl_train import fetch_data, add_features
import MetaTrader5 as mt5

SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
DATA_LIMIT = 5000
WINDOW_SIZE = 20

def optimize_agent(trial):
    if not mt5.initialize():
        return 0.0
        
    df = fetch_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    mt5.shutdown()
    
    if df is None:
        return 0.0
        
    df = add_features(df).dropna()
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y', 'atr_14', 'day_of_week']
    
    try:
        scaler = joblib.load('ml_bot/rl_scaler.save')
        scaled_data = scaler.transform(df[features])
    except Exception as e:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(df[features])
        scaled_data = scaler.transform(df[features])
        
    scaled_df = pd.DataFrame(scaled_data, columns=[f"scaled_{f}" for f in features], index=df.index)
    final_df = pd.concat([scaled_df, df[['open', 'high', 'low', 'close', 'atr_14']]], axis=1)
    
    split = int(len(final_df) * 0.8)
    train_df = final_df.iloc[:split].copy()
    test_df = final_df.iloc[split:].copy()
    
    # We lock PPO parameters to the best found values
    learning_rate = 0.00336
    gamma = 0.9517
    n_steps = 4096
    ent_coef = 0.00215
    policy_kwargs = dict(net_arch=[256, 256])
    
    # Let Optuna optimize environment rules
    tp_multiplier = trial.suggest_float('tp_multiplier', 0.1, 3.0)
    sl_multiplier = trial.suggest_float('sl_multiplier', 0.5, 5.0)
    
    train_env = DummyVecEnv([lambda: TradingEnv(train_df, WINDOW_SIZE, tp_multiplier, sl_multiplier)])
    test_env = DummyVecEnv([lambda: TradingEnv(test_df, WINDOW_SIZE, tp_multiplier, sl_multiplier)])
    
    model = PPO("MlpPolicy", train_env, verbose=0, learning_rate=learning_rate, gamma=gamma, 
                n_steps=n_steps, ent_coef=ent_coef, device='auto', policy_kwargs=policy_kwargs)
    
    # Quick train for optimization
    model.learn(total_timesteps=15000)
    
    # Evaluate
    obs = test_env.reset()
    dones = [False]
    equity_curve = [10000.0]
    
    while not dones[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = test_env.step(action)
        equity_curve.append(infos[0]['balance'])
        
    net_profit = equity_curve[-1] - 10000.0
    return net_profit

if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Optimization (D1)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=30)
    
    print("--- OPTIMIZATION COMPLETE ---")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Net Profit): {trial.value}")
    print("  Optimal Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    print("\nNext Step: Insert these Multipliers into ml_bot/rl_train.py and rl_trader.py!")
