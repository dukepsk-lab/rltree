import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom Trading Environment for RL agent using Gymnasium.
    Action 0: BUY
    Action 1: SELL
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=20, tp_multiplier=1.0, sl_multiplier=1.0):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        
        # We need atr_14 in the features, but also in the raw data
        # feature_cols are all columns except non-features
        self.feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'tick_volume', 'time', 'target']]
        
        # Define action space: 0 = Buy, 1 = Sell
        self.action_space = spaces.Discrete(2)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, len(self.feature_cols)), dtype=np.float32
        )
        
        self.current_step = self.window_size
        self.end_step = len(self.df) - 1
        
        self.position = 0
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
        low_price = current_bar['low']
        close_price = current_bar['close']
        atr = current_bar.get('atr_14', 1.0)
        
        reward = 0
        done = False
        
        # Fixed simulated spread representing 01:00 execution (15 points = $0.15 for Gold)
        spread_cost = 0.15 
        
        lot_size = (self.balance / 100.0) * 0.01
        lot_size = min(lot_size, 10.0)
        
        # Capped dynamic TP and SL
        tp_dist = min(atr * self.tp_multiplier, 3.00)
        sl_dist = atr * self.sl_multiplier
        
        if action == 0: # BUY
            entry_price = open_price + spread_cost
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
            
            if low_price <= sl_price:
                price_diff = -sl_dist
            elif high_price >= tp_price:
                price_diff = tp_dist
            else:
                price_diff = (close_price - entry_price)
                
            profit_usd = price_diff * 100.0 * lot_size
            reward += profit_usd
            self.balance += profit_usd
            
        elif action == 1: # SELL
            entry_price = open_price - spread_cost
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist
            
            if high_price >= sl_price:
                price_diff = -sl_dist
            elif low_price <= tp_price:
                price_diff = tp_dist
            else:
                price_diff = (entry_price - close_price)
                
            profit_usd = price_diff * 100.0 * lot_size
            reward += profit_usd
            self.balance += profit_usd
            
        self.current_step += 1
        
        if self.current_step >= self.end_step:
            done = True
        
        if self.balance < self.initial_balance * 0.5:
            done = True
            reward -= 100

        self.equity_curve.append(self.balance)
        return self._get_obs(), reward, done, False, {"balance": self.balance}

    def render(self):
        pass
