import os
import glob

# 1. Fix rl_env.py
with open('ml_bot/rl_env.py', 'r', encoding='utf-8') as f:
    env_code = f.read()

env_code = env_code.replace('tp_percent=0.01', 'tp_price_diff=3.00')
env_code = env_code.replace('self.tp_percent = tp_percent', 'self.tp_price_diff = tp_price_diff')
env_code = env_code.replace('tp_price = entry_price * (1 + self.tp_percent)', 'tp_price = entry_price + self.tp_price_diff')
# Update reward calculation. We want reward based on USD profit.
env_code = env_code.replace('profit = (tp_price - entry_price) / entry_price', 'profit = (tp_price - entry_price)')
env_code = env_code.replace('profit = (close_price - entry_price) / entry_price', 'profit = (close_price - entry_price)')
# reward += profit * 100 -> we can leave it as reward += profit, since $3 profit gives +3 reward, which is perfect.
env_code = env_code.replace('reward += profit * 100', 'reward += profit')
env_code = env_code.replace('self.balance *= (1 + profit)', 'self.balance += (profit * 1.00) # $1 per $1 movement for 0.01 lot')

with open('ml_bot/rl_env.py', 'w', encoding='utf-8') as f:
    f.write(env_code)

# 2. Fix all other python scripts to pass TP_PRICE_DIFF instead of TP_PERCENT
scripts = glob.glob('ml_bot/*.py')
for script in scripts:
    with open(script, 'r', encoding='utf-8') as f:
        code = f.read()
    
    if 'TP_PERCENT = 0.03' in code:
        code = code.replace('TP_PERCENT = 0.03 # Default to 3% as analyzed', 'TP_PRICE_DIFF = 3.00 # $3.00 price movement for XAUUSD')
        code = code.replace('TP_PERCENT = 0.03', 'TP_PRICE_DIFF = 3.00')
    
    code = code.replace('TP_PERCENT', 'TP_PRICE_DIFF')
    code = code.replace('tp_percent=TP_PERCENT', 'tp_price_diff=TP_PRICE_DIFF')
    code = code.replace('tp_percent=TP_PRICE_DIFF', 'tp_price_diff=TP_PRICE_DIFF')
    code = code.replace(', tp_percent', ', tp_price_diff')
    
    # In rl_trader_legacy.py and rl_trader.py
    code = code.replace('tp = price + (price * tp_percent)', 'tp = price + tp_price_diff')
    code = code.replace('tp = price + (price * tp_price_diff)', 'tp = price + tp_price_diff')
    code = code.replace('price - (price * tp_percent)', 'price - tp_price_diff')
    code = code.replace('price - (price * tp_price_diff)', 'price - tp_price_diff')
    
    # In optimize_rl.py
    code = code.replace('TradingEnv(train_df, WINDOW_SIZE, TP_PRICE_DIFF)', 'TradingEnv(train_df, WINDOW_SIZE, tp_price_diff=TP_PRICE_DIFF)')
    code = code.replace('TradingEnv(test_df, WINDOW_SIZE, TP_PRICE_DIFF)', 'TradingEnv(test_df, WINDOW_SIZE, tp_price_diff=TP_PRICE_DIFF)')
    
    # rl_train.py
    code = code.replace('TradingEnv(rl_df, WINDOW_SIZE, TP_PRICE_DIFF)', 'TradingEnv(rl_df, WINDOW_SIZE, tp_price_diff=TP_PRICE_DIFF)')
    
    with open(script, 'w', encoding='utf-8') as f:
        f.write(code)

print("Fixed all scripts!")
