import os

tfs = ['H4', 'H8', 'H12']
for tf in tfs:
    tf_lower = tf.lower()
    
    with open('ml_bot/rl_train.py', 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('TIMEFRAME_D1', f'TIMEFRAME_{tf}')
    content = content.replace('rl_model', f'rl_model_{tf_lower}')
    content = content.replace('rl_scaler', f'rl_scaler_{tf_lower}')
    content = content.replace('D1', tf)
    with open(f'ml_bot/rl_train_{tf_lower}.py', 'w', encoding='utf-8') as f:
        f.write(content)
        
    with open('ml_bot/rl_backtest.py', 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('TIMEFRAME_D1', f'TIMEFRAME_{tf}')
    content = content.replace('rl_model', f'rl_model_{tf_lower}')
    content = content.replace('rl_scaler', f'rl_scaler_{tf_lower}')
    content = content.replace('D1', tf)
    content = content.replace('rl_train', f'rl_train_{tf_lower}')
    with open(f'ml_bot/rl_backtest_{tf_lower}.py', 'w', encoding='utf-8') as f:
        f.write(content)
        
    with open('ml_bot/rl_trader.py', 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('TIMEFRAME_D1', f'TIMEFRAME_{tf}')
    content = content.replace('rl_model', f'rl_model_{tf_lower}')
    content = content.replace('rl_scaler', f'rl_scaler_{tf_lower}')
    content = content.replace('D1', tf)
    content = content.replace('rl_train', f'rl_train_{tf_lower}')
    with open(f'ml_bot/rl_trader_{tf_lower}.py', 'w', encoding='utf-8') as f:
        f.write(content)

print("Propagated successfully!")
