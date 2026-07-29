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

    def __init__(self, df, window_size=20, tp_multiplier=1.0, sl_multiplier=1.0,
                 allow_flat=False, reward_mode='usd', use_bar_spread=False,
                 dd_penalty=0.0, ruin_penalty=100.0, random_start=False):
        """
        The keyword arguments after `sl_multiplier` all default to the original
        behaviour, so existing scripts are unaffected. `train_d1.py` turns them on:

        allow_flat      adds action 2 = stay out. Without it the agent must hold a
                        position on every bar and pays the spread every day.
        reward_mode     'usd' rewards raw dollars, which grow with the compounding
                        balance; 'pct' rewards percent of balance, which is
                        stationary and is what PPO's value function needs.
        use_bar_spread  use each bar's own spread_cost instead of a flat $0.15.
        dd_penalty      subtract dd_penalty x (drawdown % from the equity high) from
                        every reward, so drawdown has a price.
        random_start    start each episode at a random bar instead of always the
                        first one, so PPO sees more than a single trajectory.
        """
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier

        self.allow_flat = allow_flat
        self.reward_mode = reward_mode
        self.use_bar_spread = use_bar_spread
        self.dd_penalty = dd_penalty
        self.ruin_penalty = ruin_penalty
        self.random_start = random_start

        # We need atr_14 in the features, but also in the raw data
        # feature_cols are all columns except non-features
        self.feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'tick_volume', 'time', 'target']]

        # Define action space: 0 = Buy, 1 = Sell (, 2 = Flat when allow_flat)
        self.action_space = spaces.Discrete(3 if allow_flat else 2)

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
        self.high_water_mark = 10000.0
        self.equity_curve = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start and self.end_step > self.window_size + 1:
            self.current_step = int(self.np_random.integers(self.window_size, self.end_step))
        else:
            self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.balance = 10000.0
        self.high_water_mark = self.balance
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
        balance_before = self.balance

        # Fixed simulated spread representing 01:00 execution (15 points = $0.15 for Gold)
        spread_cost = 0.15
        if self.use_bar_spread:
            spread_cost = float(current_bar.get('spread_cost', spread_cost))

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

        # action == 2 (only reachable when allow_flat): stay out, no P&L, no spread.

        if self.reward_mode == 'pct':
            # Percent of the balance at the start of the bar: stationary as the
            # account compounds, unlike raw dollars.
            reward = reward / balance_before * 100.0 if balance_before else 0.0

        self.high_water_mark = max(self.high_water_mark, self.balance)
        if self.dd_penalty:
            drawdown_pct = (self.high_water_mark - self.balance) / self.high_water_mark * 100.0
            reward -= self.dd_penalty * drawdown_pct

        self.current_step += 1

        if self.current_step >= self.end_step:
            done = True

        if self.balance < self.initial_balance * 0.5:
            done = True
            reward -= self.ruin_penalty

        self.equity_curve.append(self.balance)
        return self._get_obs(), reward, done, False, {"balance": self.balance}

    def render(self):
        pass
