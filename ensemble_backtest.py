import os
import json
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import tensorflow as tf
import xgboost as xgb
import joblib

from rl_train import fetch_data, add_features
from rl_env import TradingEnv

SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
DATA_LIMIT = 5000
WINDOW_SIZE = 20

# Optimized parameters
TP_MULTIPLIER = 1.0
SL_MULTIPLIER = 2.0

class LevelSystem:
    def __init__(self):
        self.state = {
            "RL": {"exp": 0, "level": 1, "last_pred": None},
            "CNN": {"exp": 0, "level": 1, "last_pred": None},
            "XGB": {"exp": 0, "level": 1, "last_pred": None}
        }
    
    def get_weight(self, model_name):
        return float(self.state[model_name]["level"])
        
    def evaluate(self, actual_direction):
        for m in self.state:
            pred = self.state[m]["last_pred"]
            if pred is not None:
                if pred == actual_direction:
                    self.state[m]["exp"] += 10
                else:
                    self.state[m]["exp"] = max(0, self.state[m]["exp"] - 5)
                self.state[m]["level"] = (self.state[m]["exp"] // 100) + 1

def main():
    if not mt5.initialize():
        print("initialize() failed")
        return
        
    df = fetch_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    mt5.shutdown()
    
    if df is None: return
        
    df = add_features(df).dropna()
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y', 'atr_14', 'day_of_week']
    
    scaler = joblib.load('ml_bot/rl_scaler.save')
    scaled_data = scaler.transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=[f"scaled_{f}" for f in features], index=df.index)
    final_df = pd.concat([scaled_df, df[['open', 'high', 'low', 'close', 'atr_14']]], axis=1)
    
    split = int(len(final_df) * 0.8)
    test_df = final_df.iloc[split:].copy()
    
    print("Loading 3 Models...")
    try:
        rl_model = PPO.load("ml_bot/rl_model")
        cnn_model = tf.keras.models.load_model('ml_bot/cnn_lstm_model.keras')
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('ml_bot/xgboost_model.json')
    except Exception as e:
        print("Models not found. Train them first!")
        print(e)
        return
        
    print("Starting Ensemble Backtest with EXP Leveling...")
    exp_sys = LevelSystem()
    
    initial_balance = 10000.0
    balance = initial_balance
    
    wins = 0
    losses = 0
    
    for i in range(WINDOW_SIZE, len(test_df) - 1):
        current_bar = test_df.iloc[i]
        
        # 1. Check yesterday's prediction against today's opening
        # Wait, the backtest moves bar by bar. Let's look at what the models predicted for bar `i-1`
        # and how bar `i-1` closed.
        if i > WINDOW_SIZE:
            prev_bar = test_df.iloc[i-1]
            actual_dir = 0 if prev_bar['close'] > prev_bar['open'] else 1 # 0=Buy, 1=Sell
            exp_sys.evaluate(actual_dir)
            
        # 2. Get predictions for today
        # RL requires observation wrapper
        # We simulate the dummy env observation
        obs_rl = test_df[[f"scaled_{f}" for f in features]].iloc[i - WINDOW_SIZE : i].values.astype(np.float32)
        rl_pred, _ = rl_model.predict(obs_rl, deterministic=True)
        rl_pred = int(rl_pred)
        
        # CNN prediction
        obs_cnn = np.expand_dims(obs_rl, axis=0)
        cnn_prob = cnn_model.predict(obs_cnn, verbose=0)[0][0]
        cnn_pred = 0 if cnn_prob <= 0.5 else 1 # Wait, our target was 0=Buy, 1=Sell. If prob > 0.5 -> 1 (Sell). If prob <= 0.5 -> 0 (Buy).
        
        # XGB prediction
        obs_xgb = obs_rl.flatten().reshape(1, -1)
        xgb_pred = int(xgb_model.predict(obs_xgb)[0])
        
        # Save predictions
        exp_sys.state["RL"]["last_pred"] = rl_pred
        exp_sys.state["CNN"]["last_pred"] = cnn_pred
        exp_sys.state["XGB"]["last_pred"] = xgb_pred
        
        # 3. Weighted Voting
        buy_score = 0
        sell_score = 0
        
        if rl_pred == 0: buy_score += exp_sys.get_weight("RL")
        else: sell_score += exp_sys.get_weight("RL")
        
        if cnn_pred == 0: buy_score += exp_sys.get_weight("CNN")
        else: sell_score += exp_sys.get_weight("CNN")
        
        if xgb_pred == 0: buy_score += exp_sys.get_weight("XGB")
        else: sell_score += exp_sys.get_weight("XGB")
        
        final_action = 0 if buy_score > sell_score else 1
        
        # 4. Simulate Trade
        open_price = current_bar['open']
        high_price = current_bar['high']
        low_price = current_bar['low']
        close_price = current_bar['close']
        atr = current_bar['atr_14']
        
        spread_cost = 0.15
        lot_size = min((balance / 100.0) * 0.01, 10.0)
        
        tp_dist = min(atr * TP_MULTIPLIER, 3.00)
        sl_dist = atr * SL_MULTIPLIER
        
        if final_action == 0: # BUY
            entry_price = open_price + spread_cost
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
            
            if low_price <= sl_price: price_diff = -sl_dist
            elif high_price >= tp_price: price_diff = tp_dist
            else: price_diff = (close_price - entry_price)
            
        else: # SELL
            entry_price = open_price - spread_cost
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist
            
            if high_price >= sl_price: price_diff = -sl_dist
            elif low_price <= tp_price: price_diff = tp_dist
            else: price_diff = (entry_price - close_price)
            
        profit_usd = price_diff * 100.0 * lot_size
        balance += profit_usd
        
        if profit_usd > 0: wins += 1
        else: losses += 1
        
    print("\n=== ENSEMBLE BACKTEST RESULTS ===")
    print(f"Total Trades: {wins + losses}")
    print(f"Win Rate: {wins / (wins + losses) * 100:.2f}%")
    print(f"Final Balance: ${balance:.2f} (Net: ${balance - initial_balance:.2f})")
    print("\n--- FINAL AI LEVELS ---")
    for m in exp_sys.state:
        print(f"{m} -> Lv.{exp_sys.state[m]['level']} (EXP: {exp_sys.state[m]['exp']})")
        
if __name__ == "__main__":
    main()
