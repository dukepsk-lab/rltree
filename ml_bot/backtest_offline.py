"""Offline backtest of the committed D1 PPO model (rl_model.zip) without MT5.

The committed model is the legacy long-only agent (git 6150c7d):
  - observation: 20 x (12 MinMax-scaled features + raw spread_cost)
  - scaler: rl_scaler_d1_legacy.save (first 12 columns of rl_scaler.save)
  - action 1 = BUY at the daily open, action 0 = skip the day
  - TP = entry + $3.00 (intrabar), otherwise forced close at the daily close
  - dynamic lot 0.01 per $100 equity, capped at 10 lots, $1/oz/0.01-lot
This script replays exactly those rules on yfinance gold futures (GC=F) as a
proxy for broker XAUUSD, and additionally simulates the same policy WITH the
disaster stop-loss (5% of equity) that the fixed rl_trader.py now attaches.

Caveats:
- GC=F prices/volumes differ slightly from broker XAUUSD spot feeds.
- spread_cost is a constant assumption (default $0.15/oz).
- The model was trained 2026-06-29 on ~5000 D1 broker bars, so everything
  before that date is IN-SAMPLE; only later bars are true out-of-sample.
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
TP_PRICE_DIFF = 3.00
SL_EQUITY_PCT = 5.0
SPREAD_COST = 0.15
START_BALANCE = 10000.0
MAX_LOT = 10.0
TRAIN_CUTOFF = '2026-06-29'

FEATURES = ['open', 'high', 'low', 'close', 'tick_volume', 'sma_10', 'sma_20',
            'rsi_14', 'adx_14', 'linreg_20', 'dxy', 'us10y']


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
    stacked = np.hstack([scaled, df[['spread_cost']].values]).astype(np.float32)
    actions = np.zeros(len(df), dtype=int)
    for i in range(WINDOW_SIZE, len(df)):
        obs = stacked[i - WINDOW_SIZE:i][None, ...]
        action, _ = model.predict(obs, deterministic=True)
        actions[i] = int(np.asarray(action).reshape(-1)[0])
    return actions


def simulate(df, actions, use_sl, seed=0):
    """Replay the legacy trade rules bar by bar.

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
        if actions[i] == 1:
            entry = bar['open'] + SPREAD_COST
            lot = min((balance / 100.0) * 0.01, MAX_LOT)
            sl_dist = (SL_EQUITY_PCT / 100.0) * balance / (lot * 100.0)
            tp_price = entry + TP_PRICE_DIFF
            sl_price = entry - sl_dist
            hit_tp = bar['high'] >= tp_price
            hit_sl = use_sl and bar['low'] <= sl_price
            if hit_tp and hit_sl:
                ambiguous += 1
                p_win = sl_dist / (TP_PRICE_DIFF + sl_dist)
                diff = TP_PRICE_DIFF if rng.random() < p_win else -sl_dist
            elif hit_sl:
                diff = -sl_dist
            elif hit_tp:
                diff = TP_PRICE_DIFF
            else:
                diff = bar['close'] - entry
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


def report(label, df, curve, pnls):
    dates = df.index[WINDOW_SIZE:WINDOW_SIZE + len(curve) - 1]
    print(f"\n=== {label} ===")
    if len(dates):
        print(f"Period:          {dates[0].date()} -> {dates[-1].date()} "
              f"({len(dates)} bars, {len(pnls)} trades)")
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
    scaler = joblib.load(os.path.join(BASE_DIR, "rl_scaler_d1_legacy.save"))
    assert tuple(model.observation_space.shape) == (WINDOW_SIZE, 13), \
        f"unexpected model obs {model.observation_space.shape}"
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
    print(f"BUY days: {actions.sum()} / {n_bars} ({actions.sum() / n_bars * 100:.0f}%)")

    windows = [
        ('FULL WINDOW (in-sample)', df.index.min(), None),
        ('2026 YTD (in-sample)', pd.Timestamp('2026-01-01'), None),
        (f'OUT-OF-SAMPLE (after {TRAIN_CUTOFF} training date)',
         pd.Timestamp(TRAIN_CUTOFF), None),
    ]

    results = {}
    for label, start, end in windows:
        # keep WINDOW_SIZE bars of history before the window for the obs
        idx = df.index.searchsorted(start)
        idx = max(idx - WINDOW_SIZE, 0)
        sub = df.iloc[idx:]
        sub_actions = actions[idx:]
        if len(sub) <= WINDOW_SIZE:
            print(f"\n=== {label} === skipped (not enough bars)")
            continue
        for sl_label, use_sl in [('no SL (as trained)', False),
                                 ('with 5%-equity disaster SL (as deployed)', True)]:
            curve, pnls = simulate(sub, sub_actions, use_sl)
            report(f"{label} | {sl_label}", sub, curve, pnls)
            results[(label, sl_label)] = (sub.index[WINDOW_SIZE:], curve)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = [k for k in results if 'as deployed' in k[1]]
        fig, axes = plt.subplots(len(labels), 1, figsize=(11, 3.5 * len(labels)))
        if len(labels) == 1:
            axes = [axes]
        for ax, key in zip(axes, labels):
            dates, curve = results[key]
            ax.plot(dates, curve[1:len(dates) + 1], color='green')
            ax.set_title(f"{key[0]} — {key[1]}")
            ax.set_ylabel('Balance ($)')
            ax.grid(True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\nPlot saved to {args.plot}")
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()
