#!/usr/bin/env python3
"""Walk-forward training and honest out-of-sample evaluation for the D1 PPO agent.

    python ml_bot/train_d1.py --timesteps 300000            # walk-forward + final model
    python ml_bot/train_d1.py --folds 5 --no-final          # validation only
    python ml_bot/train_d1.py --synthetic --timesteps 3000  # offline smoke test, no MT5

What this fixes relative to `rl_train.py` + `rl_backtest.py` (see ML_ANALYSIS.md §3):

* The scaler is fitted on the training slice of each fold only, so test-period
  minima and maxima never leak into training.
* Each fold trains on data strictly before its test window and is scored on data
  the agent has never seen.
* Every fold is reported against always-BUY, always-SELL and (when --allow-flat)
  always-FLAT run through the same environment, plus the majority-class direction
  rate. A number with no baseline next to it says nothing.
* The final model, its scaler and a metadata file are written from one run, so
  they cannot drift apart the way the committed artifacts did.

The final artifacts go to `--out-prefix` (default `ml_bot/rl_model`), i.e.
rl_model.zip / rl_scaler.save / rl_model_meta.json. Pass --no-final to leave the
existing artifacts untouched.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import FEATURES, add_features, add_indicators, build_env_frame  # noqa: E402
from rl_env import TradingEnv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------- data

def load_mt5_bars(symbol, n_bars):
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    try:
        info = mt5.symbol_info(symbol)
        if info is None or not info.visible:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol {symbol!r} not found")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 1, n_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no rates for {symbol}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df['spread_cost'] = df['spread'] * info.point
        return df[['open', 'high', 'low', 'close', 'tick_volume', 'spread_cost']]
    finally:
        mt5.shutdown()


def synthetic_bars(n, seed=0):
    """Random-walk bars with macro columns — for exercising the code, not for research."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2010-01-01', periods=n)
    close = 1200 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, n)))
    spread = np.abs(rng.normal(0, 0.008, n)) * close
    return pd.DataFrame({
        'open': close * (1 + rng.normal(0, 0.004, n)),
        'high': close + spread,
        'low': close - spread,
        'close': close,
        'tick_volume': rng.integers(1000, 200000, n).astype(float),
        'spread_cost': np.full(n, 0.15),
        'dxy': 95 + np.cumsum(rng.normal(0, 0.12, n)),
        'us10y': 2.5 + np.cumsum(rng.normal(0, 0.02, n)),
    }, index=idx)


# ------------------------------------------------------------------- evaluation

def make_env(frame, args, *, random_start=False):
    return TradingEnv(
        frame, args.window, args.tp_mult, args.sl_mult,
        allow_flat=args.allow_flat,
        reward_mode=args.reward_mode,
        use_bar_spread=args.use_bar_spread,
        dd_penalty=args.dd_penalty,
        random_start=random_start,
    )


def roll_out(env, act):
    """Run one deterministic pass. `act(obs) -> int`. Returns balances and actions."""
    obs, _ = env.reset()
    balances, actions = [env.balance], []
    done = False
    while not done:
        action = act(obs)
        actions.append(action)
        obs, _reward, done, _trunc, info = env.step(action)
        balances.append(info['balance'])
    return np.asarray(balances), np.asarray(actions)


def max_drawdown_pct(balances):
    peak = np.maximum.accumulate(balances)
    return float((1 - balances / peak).max() * 100)


def score(balances, actions, frame, window):
    """Return %, max drawdown %, action mix, and directional accuracy."""
    traded = frame.iloc[window:window + len(actions)]
    up = (traded['close'] > traded['open']).values
    mask = actions < 2
    correct = ((actions[mask] == 0) == up[mask]).mean() * 100 if mask.any() else float('nan')
    return {
        'final_balance': float(balances[-1]),
        'return_pct': float((balances[-1] / balances[0] - 1) * 100),
        'max_dd_pct': max_drawdown_pct(balances),
        'buy_pct': float((actions == 0).mean() * 100),
        'sell_pct': float((actions == 1).mean() * 100),
        'flat_pct': float((actions == 2).mean() * 100),
        'direction_accuracy_pct': float(correct),
        'majority_class_pct': float(max(up.mean(), 1 - up.mean()) * 100),
    }


def evaluate(model, frame, args):
    env = make_env(frame, args)
    balances, actions = roll_out(env, lambda o: int(model.predict(o, deterministic=True)[0]))
    result = score(balances, actions, frame, args.window)

    for name, fixed in [('always_buy', 0), ('always_sell', 1)] + ([('always_flat', 2)] if args.allow_flat else []):
        b, a = roll_out(make_env(frame, args), lambda _o, f=fixed: f)
        result[name + '_return_pct'] = float((b[-1] / b[0] - 1) * 100)
    return result


# --------------------------------------------------------------------- training

def train_ppo(frame, args, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: make_env(frame, args, random_start=args.random_start)])
    model = PPO(
        "MlpPolicy", env, verbose=0, seed=seed,
        learning_rate=args.learning_rate, gamma=args.gamma,
        n_steps=args.n_steps, ent_coef=args.ent_coef,
        device=args.device, policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=args.timesteps)
    return model, env


def fold_frames(df, train_end, test_end, args):
    """Fit the scaler on the training slice only, then build both env frames."""
    train_raw, test_raw = df.iloc[:train_end], df.iloc[train_end:test_end]
    scaler = MinMaxScaler().fit(train_raw[FEATURES])
    return build_env_frame(train_raw, scaler), build_env_frame(test_raw, scaler), scaler


# ------------------------------------------------------------------------- main

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--symbol', default='XAUUSD.')
    p.add_argument('--bars', type=int, default=5000)
    p.add_argument('--window', type=int, default=20)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--timesteps', type=int, default=300_000)
    p.add_argument('--min-train-frac', type=float, default=0.4,
                   help='fraction of the series reserved for the first fold\'s training slice')

    p.add_argument('--tp-mult', type=float, default=1.0)
    p.add_argument('--sl-mult', type=float, default=2.0)
    p.add_argument('--allow-flat', action='store_true', default=True,
                   help='give the agent a stay-out action (default on)')
    p.add_argument('--no-allow-flat', dest='allow_flat', action='store_false')
    p.add_argument('--reward-mode', choices=['usd', 'pct'], default='pct')
    p.add_argument('--use-bar-spread', action='store_true', default=True)
    p.add_argument('--no-use-bar-spread', dest='use_bar_spread', action='store_false')
    p.add_argument('--dd-penalty', type=float, default=0.05)
    p.add_argument('--random-start', action='store_true', default=True)
    p.add_argument('--no-random-start', dest='random_start', action='store_false')

    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--n-steps', type=int, default=2048)
    p.add_argument('--ent-coef', type=float, default=0.005)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--macro-lag-days', type=int, default=1)
    p.add_argument('--synthetic', action='store_true', help='random-walk data, no MT5 or network')
    p.add_argument('--out-prefix', default=os.path.join(HERE, 'rl_model'))
    p.add_argument('--scaler-path', default=os.path.join(HERE, 'rl_scaler.save'))
    p.add_argument('--report', default=None, help='write the fold report as JSON')
    p.add_argument('--no-final', dest='save_final', action='store_false',
                   help='skip the final full-data model, leave existing artifacts alone')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.synthetic:
        print(f"loading synthetic bars ({args.bars})")
        df = add_indicators(synthetic_bars(args.bars, args.seed)).dropna()
    else:
        print(f"loading {args.bars} closed D1 bars for {args.symbol} from MT5")
        df = add_features(load_mt5_bars(args.symbol, args.bars),
                          macro_lag_days=args.macro_lag_days).dropna()

    print(f"{len(df)} usable bars: {df.index[0].date()} .. {df.index[-1].date()}")

    n = len(df)
    first_train = int(n * args.min_train_frac)
    step = (n - first_train) // args.folds
    if step <= args.window * 2:
        raise SystemExit(f"folds too small ({step} bars each); reduce --folds or raise --bars")

    results = []
    for k in range(args.folds):
        train_end = first_train + k * step
        test_end = first_train + (k + 1) * step if k < args.folds - 1 else n
        train_frame, test_frame, _ = fold_frames(df, train_end, test_end, args)

        print(f"\n--- fold {k + 1}/{args.folds}: "
              f"train {df.index[0].date()}..{df.index[train_end - 1].date()} ({train_end} bars) | "
              f"test {df.index[train_end].date()}..{df.index[test_end - 1].date()} ({test_end - train_end} bars)")

        model, _env = train_ppo(train_frame, args, seed=args.seed + k)
        res = evaluate(model, test_frame, args)
        res['fold'] = k + 1
        res['train_bars'] = train_end
        res['test_bars'] = test_end - train_end
        res['test_start'] = str(df.index[train_end].date())
        res['test_end'] = str(df.index[test_end - 1].date())
        results.append(res)

        print(f"    agent   {res['return_pct']:+8.2f}%   maxDD {res['max_dd_pct']:5.1f}%   "
              f"dir.acc {res['direction_accuracy_pct']:5.1f}% (majority {res['majority_class_pct']:.1f}%)")
        print(f"    always-BUY {res['always_buy_return_pct']:+8.2f}%   "
              f"always-SELL {res['always_sell_return_pct']:+8.2f}%"
              + (f"   always-FLAT {res['always_flat_return_pct']:+.2f}%" if args.allow_flat else ""))
        print(f"    action mix: BUY {res['buy_pct']:.0f}% / SELL {res['sell_pct']:.0f}% / FLAT {res['flat_pct']:.0f}%")

    agent = np.array([r['return_pct'] for r in results])
    buy = np.array([r['always_buy_return_pct'] for r in results])
    acc = np.array([r['direction_accuracy_pct'] for r in results])
    maj = np.array([r['majority_class_pct'] for r in results])

    edge = float(np.nanmean(acc) - maj.mean())
    beats_buy = int((agent > buy).sum())

    print("\n=== walk-forward summary ({} folds, out-of-sample) ===".format(args.folds))
    print(f"agent return      : median {np.median(agent):+.2f}%   worst {agent.min():+.2f}%   "
          f"positive folds {int((agent > 0).sum())}/{len(agent)}")
    print(f"always-BUY return : median {np.median(buy):+.2f}%   "
          f"agent beats it in {beats_buy}/{len(agent)} folds")
    print(f"direction accuracy: {np.nanmean(acc):.2f}%   majority class {maj.mean():.2f}%   "
          f"edge {edge:+.2f} points")
    print(f"max drawdown      : median {np.median([r['max_dd_pct'] for r in results]):.1f}%   "
          f"worst {max(r['max_dd_pct'] for r in results):.1f}%")

    # Returns here compound at ~1% of balance per bar, so their distribution is
    # lognormal: the mean is dominated by a handful of lucky paths and says
    # almost nothing. Judge on directional edge and on how many folds beat
    # always-BUY, not on the headline percentage.
    if edge <= 0:
        print("verdict           : NO directional edge — accuracy is at or below the "
              "majority class. Any positive return here is the compounding tail, not skill.")
    elif beats_buy <= len(agent) // 2:
        print(f"verdict           : weak — {edge:+.2f} points of accuracy, but it beats "
              f"always-BUY in only {beats_buy}/{len(agent)} folds.")
    else:
        print(f"verdict           : {edge:+.2f} points of directional edge and beats "
              f"always-BUY in {beats_buy}/{len(agent)} folds. Re-run with another --seed "
              f"to check the result is not seed luck.")

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'data': 'synthetic' if args.synthetic else args.symbol,
        'bars': int(n), 'folds': results, 'config': vars(args),
    }
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"report written to {args.report}")

    if not args.save_final:
        print("\n--no-final: existing artifacts left untouched")
        return 0

    print(f"\ntraining the final model on all {n} bars...")
    import joblib

    scaler = MinMaxScaler().fit(df[FEATURES])
    frame = build_env_frame(df, scaler)
    model, env = train_ppo(frame, args, seed=args.seed)

    model.save(args.out_prefix)
    joblib.dump(scaler, args.scaler_path)
    meta = {
        'features': FEATURES,
        'window_size': args.window,
        'observation_columns': list(env.get_attr('feature_cols')[0]),
        'observation_shape': list(env.observation_space.shape),
        'action_space_n': int(env.action_space.n),
        'allow_flat': args.allow_flat,
        'reward_mode': args.reward_mode,
        'tp_multiplier': args.tp_mult,
        'sl_multiplier': args.sl_mult,
        'timesteps': args.timesteps,
        'bars': int(n),
        'data': 'synthetic' if args.synthetic else args.symbol,
        'macro_lag_days': args.macro_lag_days,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'walk_forward': {
            'folds': args.folds,
            'mean_return_pct': float(agent.mean()),
            'mean_always_buy_return_pct': float(buy.mean()),
            'mean_direction_accuracy_pct': float(np.nanmean(acc)),
            'mean_majority_class_pct': float(maj.mean()),
        },
    }
    with open(args.out_prefix + '_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"saved {args.out_prefix}.zip, {args.scaler_path}, {args.out_prefix}_meta.json")
    print("run `python ml_bot/d1_forecast.py` to get today's forecast from it")
    return 0


if __name__ == '__main__':
    sys.exit(main())
