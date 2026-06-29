import re

def update_live_trader():
    filepath = 'ml_bot/rl_trader.py'
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    # The new imports and logic
    new_logic = """import json
import os
import tensorflow as tf
import xgboost as xgb
from stable_baselines3 import PPO

# Load AI Levels
LEVELS_FILE = 'ml_bot/ai_levels.json'
if not os.path.exists(LEVELS_FILE):
    default_levels = {
        "RL": {"exp": 0, "level": 1, "last_pred": None, "last_date": None},
        "CNN": {"exp": 0, "level": 1, "last_pred": None, "last_date": None},
        "XGB": {"exp": 0, "level": 1, "last_pred": None, "last_date": None}
    }
    with open(LEVELS_FILE, 'w') as f:
        json.dump(default_levels, f)

with open(LEVELS_FILE, 'r') as f:
    ai_levels = json.load(f)

def evaluate_yesterday(df, ai_levels):
    # Check if we have a prediction to evaluate
    if ai_levels["RL"]["last_pred"] is None:
        return ai_levels
        
    last_date = ai_levels["RL"]["last_date"]
    
    # Simple check: Did yesterday close higher than it opened?
    if len(df) >= 2:
        yesterday_bar = df.iloc[-2] # Current is -1 (today's open), yesterday is -2
        actual_dir = 0 if yesterday_bar['close'] > yesterday_bar['open'] else 1
        
        for m in ai_levels:
            pred = ai_levels[m]["last_pred"]
            if pred == actual_dir:
                ai_levels[m]["exp"] += 10
            else:
                ai_levels[m]["exp"] = max(0, ai_levels[m]["exp"] - 5)
            
            ai_levels[m]["level"] = (ai_levels[m]["exp"] // 100) + 1
            
    return ai_levels

def get_ensemble_action(obs, rl_model, cnn_model, xgb_model, ai_levels, today_date):
    # 1. Get Predictions
    rl_pred = int(rl_model.predict(obs, deterministic=True)[0])
    
    obs_cnn = np.expand_dims(obs, axis=0)
    cnn_prob = cnn_model.predict(obs_cnn, verbose=0)[0][0]
    cnn_pred = 0 if cnn_prob <= 0.5 else 1
    
    obs_xgb = obs.flatten().reshape(1, -1)
    xgb_pred = int(xgb_model.predict(obs_xgb)[0])
    
    # 2. Update levels state for tomorrow
    ai_levels["RL"]["last_pred"] = rl_pred
    ai_levels["RL"]["last_date"] = str(today_date)
    ai_levels["CNN"]["last_pred"] = cnn_pred
    ai_levels["CNN"]["last_date"] = str(today_date)
    ai_levels["XGB"]["last_pred"] = xgb_pred
    ai_levels["XGB"]["last_date"] = str(today_date)
    
    with open(LEVELS_FILE, 'w') as f:
        json.dump(ai_levels, f, indent=4)
        
    # 3. Weighted Voting
    buy_score = 0
    sell_score = 0
    
    if rl_pred == 0: buy_score += ai_levels["RL"]["level"]
    else: sell_score += ai_levels["RL"]["level"]
    
    if cnn_pred == 0: buy_score += ai_levels["CNN"]["level"]
    else: sell_score += ai_levels["CNN"]["level"]
    
    if xgb_pred == 0: buy_score += ai_levels["XGB"]["level"]
    else: sell_score += ai_levels["XGB"]["level"]
    
    print(f"--- VOTING RESULTS ---")
    print(f"RL  (Lv.{ai_levels['RL']['level']}): {'BUY' if rl_pred==0 else 'SELL'}")
    print(f"CNN (Lv.{ai_levels['CNN']['level']}): {'BUY' if cnn_pred==0 else 'SELL'}")
    print(f"XGB (Lv.{ai_levels['XGB']['level']}): {'BUY' if xgb_pred==0 else 'SELL'}")
    print(f"Total Score -> BUY: {buy_score}, SELL: {sell_score}")
    
    return 0 if buy_score > sell_score else 1

"""

    # We need to inject this into rl_trader.py
    # Remove the old action prediction logic and load all 3 models
    
    code = code.replace("from stable_baselines3 import PPO", new_logic)
    
    # Replace the model loading
    old_load = """        print("Loading trained model...")
        model = PPO.load("ml_bot/rl_model")"""
        
    new_load = """        print("Loading 3 Models for Ensemble...")
        rl_model = PPO.load("ml_bot/rl_model")
        cnn_model = tf.keras.models.load_model("ml_bot/cnn_lstm_model.keras")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("ml_bot/xgboost_model.json")
        
        global ai_levels
        ai_levels = evaluate_yesterday(df, ai_levels)"""
        
    code = code.replace(old_load, new_load)
    
    # Replace action prediction
    old_pred = """        print("Predicting action...")
        action, _ = model.predict(obs, deterministic=True)"""
        
    new_pred = """        print("Voting...")
        today_date = df.index[-1].date()
        action = get_ensemble_action(obs, rl_model, cnn_model, xgb_model, ai_levels, today_date)"""
        
    code = code.replace(old_pred, new_pred)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)

update_live_trader()
