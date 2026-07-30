#!/usr/bin/env python3
"""Is there any predictive information in these features at all?

    python ml_bot/signal_test.py                     # walk-forward test on MT5 D1 data
    python ml_bot/signal_test.py --synthetic         # offline check, no MT5 needed
    python ml_bot/signal_test.py --horizon 5         # 5-bar direction instead of 1

Before spending hours of PPO training, answer the cheap question first: can *any*
straightforward classifier beat the base rate on next-bar direction using this
feature set? PPO's returns are noisy and its policy conflates "which direction"
with "how much to risk"; a plain classifier isolates the directional signal and
runs in about a minute.

The test is deliberately generous to the hypothesis — logistic regression and
gradient boosting, walk-forward, scaler fitted per fold — so a negative result
is informative: if these can't find signal, the RL agent is not going to either.

A shuffled-label control is run alongside. Any classifier scores a bit above 50%
on shuffled labels through sheer fitting noise; that number is the floor a real
result has to clear, and printing it stops a 51% accuracy from being mistaken for
an edge.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import FEATURES, add_features, add_indicators  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def build_matrix(df, window, horizon):
    """Flattened `window` bars of features -> direction of the bar `horizon` ahead.

    Row i uses bars [i-window, i) and is labelled by bar i+horizon-1, so nothing
    from the labelled bar or any bar after it enters the features.
    """
    values = df[FEATURES].values
    X, y, idx = [], [], []
    for i in range(window, len(df) - horizon + 1):
        target_bar = df.iloc[i + horizon - 1]
        X.append(values[i - window:i].ravel())
        y.append(int(target_bar['close'] > target_bar['open']))
        idx.append(df.index[i + horizon - 1])
    return np.asarray(X, dtype=np.float64), np.asarray(y), pd.DatetimeIndex(idx)


def evaluate_fold(X_tr, y_tr, X_te, y_te, seed):
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    out = {}
    majority = DummyClassifier(strategy='most_frequent').fit(X_tr_s, y_tr)
    out['majority'] = {'accuracy': accuracy_score(y_te, majority.predict(X_te_s)) * 100, 'auc': 50.0}

    models = {
        'logistic': LogisticRegression(max_iter=2000, C=0.1),
        'gbm': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              max_depth=4, random_state=seed),
    }
    for name, model in models.items():
        fit_X = X_tr_s if name == 'logistic' else X_tr
        pred_X = X_te_s if name == 'logistic' else X_te
        model.fit(fit_X, y_tr)
        proba = model.predict_proba(pred_X)[:, 1]
        out[name] = {
            'accuracy': accuracy_score(y_te, model.predict(pred_X)) * 100,
            'auc': roc_auc_score(y_te, proba) * 100 if len(set(y_te)) > 1 else float('nan'),
        }

    # Control: same model, labels shuffled. Whatever it scores is the noise floor.
    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(y_tr)
    ctrl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          max_depth=4, random_state=seed).fit(X_tr, y_shuf)
    out['gbm_shuffled'] = {
        'accuracy': accuracy_score(y_te, ctrl.predict(X_te)) * 100,
        'auc': roc_auc_score(y_te, ctrl.predict_proba(X_te)[:, 1]) * 100
        if len(set(y_te)) > 1 else float('nan'),
    }
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--symbol', default='XAUUSD.')
    p.add_argument('--bars', type=int, default=5000)
    p.add_argument('--window', type=int, default=20)
    p.add_argument('--horizon', type=int, default=1, help='predict the direction N bars ahead')
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--min-train-frac', type=float, default=0.4)
    p.add_argument('--macro-lag-days', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--report', default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.synthetic:
        from train_d1 import synthetic_bars
        df = add_indicators(synthetic_bars(args.bars, args.seed)).dropna()
    else:
        from train_d1 import load_mt5_bars
        df = add_features(load_mt5_bars(args.symbol, args.bars),
                          macro_lag_days=args.macro_lag_days).dropna()

    X, y, idx = build_matrix(df, args.window, args.horizon)
    print(f"{len(X)} samples, {X.shape[1]} inputs ({args.window} bars x {len(FEATURES)} features), "
          f"horizon {args.horizon}")
    print(f"base rate: {y.mean() * 100:.2f}% up bars\n")

    n = len(X)
    first_train = int(n * args.min_train_frac)
    step = (n - first_train) // args.folds

    names = ['majority', 'logistic', 'gbm', 'gbm_shuffled']
    rows = []
    print(f"{'fold':>4} {'test window':<24} " + "".join(f"{m:>22}" for m in names))
    for k in range(args.folds):
        train_end = first_train + k * step
        test_end = first_train + (k + 1) * step if k < args.folds - 1 else n
        res = evaluate_fold(X[:train_end], y[:train_end], X[train_end:test_end],
                            y[train_end:test_end], args.seed + k)
        res['fold'] = k + 1
        res['test_start'], res['test_end'] = str(idx[train_end].date()), str(idx[test_end - 1].date())
        res['test_samples'] = test_end - train_end
        rows.append(res)
        cells = "".join(f"{res[m]['accuracy']:>11.2f}%{res[m]['auc']:>10.2f}" for m in names)
        print(f"{k + 1:>4} {res['test_start'] + '..' + res['test_end']:<24}" + cells)

    print(" " * 29 + "".join(f"{'acc':>12}{'auc':>10}" for _ in names))

    print("\n=== summary (mean across folds) ===")
    summary = {}
    for m in names:
        acc = float(np.mean([r[m]['accuracy'] for r in rows]))
        auc = float(np.nanmean([r[m]['auc'] for r in rows]))
        summary[m] = {'accuracy': acc, 'auc': auc}
        print(f"{m:>14}: accuracy {acc:6.2f}%   AUC {auc:6.2f}")

    best = max(('logistic', 'gbm'), key=lambda m: summary[m]['auc'])
    floor = max(summary['majority']['accuracy'], summary['gbm_shuffled']['accuracy'])
    auc_over_chance = summary[best]['auc'] - 50.0

    print(f"\nbest model        : {best}")
    print(f"accuracy vs floor : {summary[best]['accuracy']:.2f}% vs {floor:.2f}% "
          f"(majority / shuffled-label control)")
    print(f"AUC over chance   : {auc_over_chance:+.2f} points")

    if auc_over_chance < 1.0 and summary[best]['accuracy'] <= floor:
        print("verdict           : NO usable signal in this feature set. A classifier given "
              "every advantage cannot beat the base rate, so more RL timesteps, a bigger "
              "network or another ensemble member will not help. Change the inputs.")
    elif auc_over_chance < 1.0:
        print("verdict           : marginal — accuracy edges the floor but AUC is at chance. "
              "Most likely base-rate fitting, not signal. Re-run with another --seed.")
    else:
        print(f"verdict           : {auc_over_chance:+.2f} AUC points above chance. Worth "
              f"pursuing — confirm across seeds and horizons before trusting it.")

    if args.report:
        with open(args.report, 'w') as f:
            json.dump({'generated_at': datetime.now(timezone.utc).isoformat(),
                       'data': 'synthetic' if args.synthetic else args.symbol,
                       'samples': int(n), 'base_rate_pct': float(y.mean() * 100),
                       'folds': rows, 'summary': summary, 'config': vars(args)},
                      f, indent=2, default=str)
        print(f"report written to {args.report}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
