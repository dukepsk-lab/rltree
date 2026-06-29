import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom Trading Environment for RL agent using Gymnasium.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=20, tp_price_diff=3.00):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.tp_price_diff = tp_price_diff
        
        # Assume df contains 'open', 'high', 'low', 'close' for execution
        # and feature columns prefixed with 'feat_' or just use all columns except OHLCV as features
        self.feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'tick_volume', 'time', 'target']]
        
        # Define action space: 0 = Flat/Close, 1 = Buy (Long)
        self.action_space = spaces.Discrete(2)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, len(self.feature_cols)), dtype=np.float32
        )
        
        self.current_step = self.window_size
        self.end_step = len(self.df) - 1
        
        self.position = 0 # 0 = flat, 1 = long
        self.entry_price = 0.0
        
        self.balance = 10000.0
        self.initial_balance = 10000.0
        self.equity_curve = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.balance = 10000.0
        self.equity_curve = [self.balance]
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.df[self.feature_cols].iloc[self.current_step - self.window_size : self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        current_bar = self.df.iloc[self.current_step]
        open_price = current_bar['open']
        high_price = current_bar['high']
        close_price = current_bar['close']
        
        reward = 0
        done = False
        
        # 0 = Flat/Skip, 1 = Buy at Open
        if action == 1:
            spread_cost = current_bar['spread_cost'] if 'spread_cost' in current_bar else 0
            entry_price = open_price + spread_cost
            tp_price = entry_price + self.tp_price_diff
            
            # Simulate inside the bar
            if high_price >= tp_price:
                # Hit Take Profit!
                profit = (tp_price - entry_price)
                reward += profit
                self.balance += (profit * 1.00) # $1 per $1 movement for 0.01 lot
            else:
                # Didn't hit TP, forcefully close at the end of the bar at Bid price (close_price)
                profit = (close_price - entry_price)
                reward += profit
                self.balance += (profit * 1.00) # $1 per $1 movement for 0.01 lot
        
        # Position is always flat at the end of the step
        self.position = 0
        self.entry_price = 0

        self.current_step += 1
        
        if self.current_step >= self.end_step:
            done = True
        
        if self.balance < self.initial_balance * 0.5:
            done = True
            reward -= 100

        self.equity_curve.append(self.balance)
        return self._get_obs(), reward, done, False, {"balance": self.balance}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}")
