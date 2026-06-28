import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import optuna
import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from rl_env import TradingEnv
from rl_train import fetch_data, add_features

SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
DATA_LIMIT = 5000
WINDOW_SIZE = 20
TP_PERCENT = 0.03

def optimize_agent(trial):
    # Hyperparameters to search
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 1e-2)
    
    # Fetch Data
    mt5.initialize()
    df = fetch_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    mt5.shutdown()
    
    df = add_features(df).dropna()
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y']
    
    try:
        scaler = joblib.load('ml_bot/rl_scaler.save')
    except:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(df[features])
        
    scaled_data = scaler.transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=[f"scaled_{f}" for f in features], index=df.index)
    final_df = pd.concat([scaled_df, df[['open', 'high', 'low', 'close', 'spread_cost']]], axis=1)
    
    # Train / Test split
    split = int(len(final_df) * 0.8)
    train_df = final_df.iloc[:split]
    test_df = final_df.iloc[split:]
    
    train_env = DummyVecEnv([lambda: TradingEnv(train_df, WINDOW_SIZE, TP_PERCENT)])
    test_env = DummyVecEnv([lambda: TradingEnv(test_df, WINDOW_SIZE, TP_PERCENT)])
    
    model = PPO("MlpPolicy", train_env, verbose=0, learning_rate=learning_rate, gamma=gamma, 
                n_steps=n_steps, ent_coef=ent_coef, device='cuda')
    
    # Quick train for optimization
    model.learn(total_timesteps=15000)
    
    # Evaluate
    mean_reward, _ = evaluate_policy(model, test_env, n_eval_episodes=1)
    return mean_reward

if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Optimization (D1)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_agent, n_trials=30)
    
    print("--- OPTIMIZATION COMPLETE ---")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Mean Reward): {trial.value}")
    print("  Optimal Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("\nNext Step: Manually insert these Optimal Parameters into ml_bot/rl_train.py!")
