import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO

import sys
sys.path.append("ml_bot")
from rl_trader_legacy import fetch_data, add_features

SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
WINDOW_SIZE = 20
TP_PERCENT = 0.03

def main():
    if not mt5.initialize():
        print("MT5 init failed")
        return

    model = PPO.load("ml_bot/rl_model")
    expected_shape = model.policy.observation_space.shape[1]
    
    df_hist = fetch_data(SYMBOL, TIMEFRAME, 5000)
    df_hist = add_features(df_hist).dropna()
    
    if expected_shape == 10:
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20']
    elif expected_shape == 13:
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14']
    elif expected_shape == 14:
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20']
    else:
        print("Unknown model shape")
        return
        
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(df_hist[features])
    
    # We only want to backtest the last 7 days.
    test_bars = WINDOW_SIZE + 7
    df_test = df_hist.iloc[-test_bars:].copy()
    
    scaled_data = scaler.transform(df_test[features])
    scaled_df = pd.DataFrame(scaled_data, columns=[f"scaled_{f}" for f in features], index=df_test.index)
    if expected_shape == 10:
        final_df = scaled_df
    else:
        final_df = pd.concat([scaled_df, df_test[['open', 'high', 'low', 'close']]], axis=1)
    
    print("=== Backtest Results (Last 7 Days) ===")
    capital = 10000.0
    
    for i in range(WINDOW_SIZE, len(final_df)):
        obs_window = final_df.values[i-WINDOW_SIZE:i]
        obs = np.array([obs_window], dtype=np.float32)
        
        action, _ = model.predict(obs, deterministic=True)
        action_idx = action[0]
        
        current_date = df_test.index[i].strftime("%Y-%m-%d")
        open_price = df_test.iloc[i]['open']
        high_price = df_test.iloc[i]['high']
        close_price = df_test.iloc[i]['close']
        
        if action_idx == 1:
            tp_price = open_price + 3.00 # $3 price movement for XAUUSD
            if high_price >= tp_price:
                profit_usd = 3.00 # $3 profit per 0.01 lot
                result = "HIT TP (+$3.00)"
            else:
                profit_usd = (close_price - open_price) * 1.00 # 1 oz for 0.01 lot
                sign = "+" if profit_usd > 0 else ""
                result = f"Closed End of Day ({sign}${profit_usd:.2f})"
            
            capital += profit_usd
            print(f"[{current_date}] ACTION: BUY  | Open: {open_price:.2f} | High: {high_price:.2f} | Result: {result:25} | Profit: ${profit_usd:7.2f} | Balance: ${capital:.2f}")
        else:
            print(f"[{current_date}] ACTION: HOLD | Balance: ${capital:.2f}")
            
    print(f"======================================")
    print(f"Final Balance: ${capital:.2f}")
    mt5.shutdown()

if __name__ == '__main__':
    main()
