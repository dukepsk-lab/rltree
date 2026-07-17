"""Offline backtest of the D1 PPO model (rl_model.zip) without MT5.

Matches the CURRENT rl_env.py / rl_train.py (Always-In strategy):
  - observation: 20 x (14 MinMax-scaled features + raw spread_cost + raw
    atr_14) = shape (20, 16), scaler: rl_scaler.save (14 features)
  - action 0 = BUY, action 1 = SELL -- a position is opened EVERY bar (no
    flat/skip state)
  - TP = min(atr * TP_MULTIPLIER, $3.00) intrabar, otherwise forced close
    at the bar's close
  - trained SL = atr * SL_MULTIPLIER (uncapped) -- see the SL_EQUITY_PCT
    note below, this is not safe to deploy as-is
  - dynamic lot 0.01 per $100 equity, capped at 10 lots, $100/oz/lot

This script replays those rules on yfinance gold futures (GC=F), or on a
local CSV exported by export_data_mt5.py, as a proxy for broker XAUUSD.
It reports results BOTH with the trained (uncapped ATR) SL and with the
broker-side 5%-of-equity disaster SL that rl_trader.py actually attaches
-- the two are not the same risk profile, see the note above.

Caveats:
- GC=F prices/volumes differ slightly from broker XAUUSD spot feeds.
- spread_cost is a constant assumption (default $0.15/oz).
- This model is whatever is on disk at ml_bot/rl_model.zip when you run
  this script -- if you retrained locally, that's an IN-SAMPLE model over
  however much of the data you trained on. Only bars strictly after your
  own training run are true out-of-sample.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WINDOW_SIZE = 20
TP_MULTIPLIER = 1.0
TP_CAP = 3.00
SL_MULTIPLIER = 2.0    # trained SL multiplier (uncapped -- dangerous, see module docstring)
SL_EQUITY_PCT = 5.0    # broker-side disaster SL actually deployed by rl_trader.py
SPREAD_COST = 0.15
START_BALANCE = 10000.0
MAX_LOT = 10.0

FEATURES = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20',
            'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y', 'atr_14',
            'day_of_week']


def fetch_yf(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_data(start, end):
    gold = fetch_yf("GC=F", start, end)
    df = pd.DataFrame({
        'open': gold['Open'],
        'high': gold['High'],
        'low': gold['Low'],
        'close': gold['Close'],
        'tick_volume': gold['Volume'],
    })
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df['spread_cost'] = SPREAD_COST
    return df.dropna()


def load_csv(path):
    """Load OHLC data exported by export_data_mt5.py (or any CSV with
    time,open,high,low,close,tick_volume[,spread_cost] columns)."""
    df = pd.read_csv(path, parse_dates=['time'], index_col='time')
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if 'spread_cost' not in df.columns:
        df['spread_cost'] = SPREAD_COST
    cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread_cost']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing columns: {missing}")
    return df[cols].dropna()


def add_features(df):
    # Mirrors rl_train.add_features (rl_train imports MetaTrader5 at module
    # level, so it cannot be imported on non-Windows machines)
    df = df.copy()

    start_date = df.index.min()
    end_date = df.index.max() + pd.Timedelta(days=1)
    dxy = fetch_yf('DX-Y.NYB', start_date, end_date)['Close']
    us10y = fetch_yf('^TNX', start_date, end_date)['Close']
    if isinstance(dxy, pd.DataFrame):
        dxy = dxy.iloc[:, 0]
    if isinstance(us10y, pd.DataFrame):
        us10y = us10y.iloc[:, 0]
    macro_df = pd.DataFrame({'dxy': dxy, 'us10y': us10y})
    if macro_df.index.tz is not None:
        macro_df.index = macro_df.index.tz_localize(None)
    df = df.join(macro_df, how='left')
    df['dxy'] = df['dxy'].ffill().bfill()
    df['us10y'] = df['us10y'].ffill().bfill()

    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['day_of_week'] = df.index.dayofweek
    up, down = high - high.shift(), low.shift() - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_14 = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).sum() / (tr_14 + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).sum() / (tr_14 + 1e-9))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['adx_14'] = dx.rolling(14).mean()

    window = 20
    x = np.arange(window)
    sum_x, sum_x2 = x.sum(), (x ** 2).sum()
    denom = window * sum_x2 - sum_x ** 2

    def slope_func(y):
        return (window * (x * y).sum() - sum_x * y.sum()) / denom

    df['linreg_20'] = df['close'].rolling(window=window).apply(slope_func, raw=True)
    return df


def precompute_actions(model, df, scaler):
    """Predict the action for every bar (obs = the 20 bars BEFORE the bar)."""
    scaled = scaler.transform(df[FEATURES])
    stacked = np.hstack([scaled, df[['spread_cost', 'atr_14']].values]).astype(np.float32)
    actions = np.zeros(len(df), dtype=int)
    for i in range(WINDOW_SIZE, len(df)):
        obs = stacked[i - WINDOW_SIZE:i][None, ...]
        action, _ = model.predict(obs, deterministic=True)
        actions[i] = int(np.asarray(action).reshape(-1)[0])
    return actions


def simulate(df, actions, sl_mode, seed=0):
    """Replay the Always-In trade rules bar by bar (a trade is taken every
    bar, direction from `actions`: 0=BUY, 1=SELL).

    sl_mode: 'none' (no SL, as literally trained), 'atr' (trained SL = atr *
    SL_MULTIPLIER, uncapped), or 'equity' (broker-side 5%-of-equity SL that
    rl_trader.py actually deploys).

    A D1 bar cannot tell whether TP or SL was touched first when the bar's
    range covers both. For those ambiguous bars the outcome is drawn with the
    continuous-barrier win probability SL/(TP+SL) (same math as
    sim_robustness.py), with a fixed seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    balance = START_BALANCE
    curve = [balance]
    trade_pnls = []
    ambiguous = 0
    for i in range(WINDOW_SIZE, len(df)):
        bar = df.iloc[i]
        atr = bar['atr_14'] if not pd.isna(bar['atr_14']) else 1.0
        lot = min((balance / 100.0) * 0.01, MAX_LOT)
        tp_dist = min(atr * TP_MULTIPLIER, TP_CAP)

        if sl_mode == 'none':
            sl_dist = np.inf
        elif sl_mode == 'atr':
            sl_dist = atr * SL_MULTIPLIER
        else:  # 'equity'
            sl_dist = (SL_EQUITY_PCT / 100.0) * balance / (lot * 100.0)

        is_buy = actions[i] == 0
        if is_buy:
            entry = bar['open'] + SPREAD_COST
            tp_price, sl_price = entry + tp_dist, entry - sl_dist
            hit_tp, hit_sl = bar['high'] >= tp_price, bar['low'] <= sl_price
        else:
            entry = bar['open'] - SPREAD_COST
            tp_price, sl_price = entry - tp_dist, entry + sl_dist
            hit_tp, hit_sl = bar['low'] <= tp_price, bar['high'] >= sl_price

        if hit_tp and hit_sl:
            ambiguous += 1
            p_win = sl_dist / (tp_dist + sl_dist) if np.isfinite(sl_dist) else 1.0
            diff = tp_dist if rng.random() < p_win else -sl_dist
        elif hit_sl:
            diff = -sl_dist
        elif hit_tp:
            diff = tp_dist
        else:
            diff = (bar['close'] - entry) if is_buy else (entry - bar['close'])

        pnl = diff * 100.0 * lot
        balance += pnl
        trade_pnls.append(pnl)
        curve.append(balance)
        if balance <= 0:
            break
    if ambiguous:
        print(f"[note] {ambiguous} bars touched both TP and SL; outcomes drawn "
              f"with p(win)=SL/(TP+SL)")
    return np.array(curve), np.array(trade_pnls)


def report(label, curve, pnls):
    print(f"\n=== {label} ===")
    print(f"Trades:          {len(pnls)}")
    if len(pnls) == 0:
        print("No trades taken.")
        return
    wins = (pnls > 0).sum()
    losses = (pnls < 0).sum()
    gross_win = pnls[pnls > 0].sum()
    gross_loss = -pnls[pnls < 0].sum()
    peak = np.maximum.accumulate(curve)
    max_dd = ((peak - curve) / peak).max() * 100
    rets = np.diff(curve) / curve[:-1]
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    print(f"Final balance:   ${curve[-1]:,.2f}  (start ${START_BALANCE:,.0f})")
    print(f"Net profit:      {curve[-1] - START_BALANCE:+,.2f} "
          f"({(curve[-1] / START_BALANCE - 1) * 100:+.2f}%)")
    print(f"Win rate:        {wins / max(wins + losses, 1) * 100:.1f}% ({wins}W/{losses}L)")
    print(f"Profit factor:   {gross_win / gross_loss if gross_loss > 0 else float('inf'):.2f}")
    print(f"Avg win/loss:    +{pnls[pnls > 0].mean() if wins else 0:.2f} / "
          f"{pnls[pnls < 0].mean() if losses else 0:.2f}")
    print(f"Worst trade:     {pnls.min():+,.2f}")
    print(f"Max drawdown:    {max_dd:.1f}%")
    print(f"Sharpe (ann.):   {sharpe:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--csv', default=None,
                        help='Backtest from a local CSV (see export_data_mt5.py) '
                             'instead of downloading GC=F from yfinance')
    parser.add_argument('--plot', default='backtest_offline.png')
    args = parser.parse_args()

    print("Loading model and scaler...")
    model = PPO.load(os.path.join(BASE_DIR, "rl_model"))
    scaler = joblib.load(os.path.join(BASE_DIR, "rl_scaler.save"))
    assert tuple(model.observation_space.shape) == (WINDOW_SIZE, 16), \
        (f"unexpected model obs {model.observation_space.shape}, expected "
         f"(20, 16) for the current Always-In env -- is rl_model.zip stale?")
    assert scaler.n_features_in_ == len(FEATURES), \
        f"scaler expects {scaler.n_features_in_} features, need {len(FEATURES)}"

    if args.csv:
        print(f"Loading gold data from {args.csv} (macro from yfinance)...")
        df = load_csv(args.csv)
        if args.start:
            df = df[df.index >= args.start]
        if args.end:
            df = df[df.index <= args.end]
    else:
        print("Fetching gold + macro data from yfinance...")
        df = fetch_data(args.start, args.end)
    df = add_features(df).dropna()
    print(f"{len(df)} bars ready ({df.index.min().date()} -> {df.index.max().date()})")

    print("Predicting actions for every bar...")
    actions = precompute_actions(model, df, scaler)
    n_bars = len(df) - WINDOW_SIZE
    n_buy = (actions[WINDOW_SIZE:] == 0).sum()
    print(f"BUY days: {n_buy}/{n_bars} ({n_buy / n_bars * 100:.0f}%), "
          f"SELL days: {n_bars - n_buy}/{n_bars}")

    sub = df
    curves = {}
    for sl_mode, label in [('none', 'no SL (as literally trained)'),
                           ('atr', f'trained SL ({SL_MULTIPLIER}x ATR, uncapped)'),
                           ('equity', f'broker-side {SL_EQUITY_PCT:.0f}%-equity SL (as deployed)')]:
        curve, pnls = simulate(sub, actions, sl_mode)
        report(label, curve, pnls)
        curves[label] = (sub.index[WINDOW_SIZE:WINDOW_SIZE + len(curve) - 1], curve)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(curves), 1, figsize=(11, 3.5 * len(curves)))
        for ax, (label, (dates, curve)) in zip(axes, curves.items()):
            ax.plot(dates, curve[1:len(dates) + 1], color='green')
            ax.set_title(label)
            ax.set_ylabel('Balance ($)')
            ax.grid(True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\nPlot saved to {args.plot}")
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()
