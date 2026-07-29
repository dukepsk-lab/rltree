#!/usr/bin/env python3
"""D1 forecast from the PPO agent, with optional order placement.

    python ml_bot/d1_forecast.py                    # forecast only (default, no orders)
    python ml_bot/d1_forecast.py --json signal.json # forecast + write signal file
    python ml_bot/d1_forecast.py --live --yes       # forecast AND send the order

Requires a running MetaTrader 5 terminal on the same machine (the `MetaTrader5`
package is Windows-only), so it cannot run on a Linux CI box or in a container.

Differences from `rl_trader.py`, all deliberate — see ML_ANALYSIS.md:

* Refuses to trade instead of guessing when the model and the scaler disagree
  about the number of inputs.
* Uses only *closed* D1 bars; `copy_rates_from_pos(..., 0, n)` returns the
  still-forming bar as the last row, which training never saw.
* Uses the env's action convention (0 = BUY, 1 = SELL) rather than treating
  action 1 as BUY and action 0 as flat.
* Attaches broker-side SL and TP, caps risk per trade, checks spread and stops
  level, detects the broker's filling mode, and never opens a second position.
* Never sends anything without both --live and --yes.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import (  # noqa: E402
    FEATURES,
    add_features,
    build_observation,
    compute_order_plan,
    out_of_range_features,
)

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SYMBOL = "XAUUSD."
WINDOW_SIZE = 20
ACTION_NAMES = {0: "BUY", 1: "SELL"}


# --------------------------------------------------------------------------- data

def fetch_closed_bars(mt5, symbol, timeframe, n_bars):
    """Closed D1 bars only, plus the broker's spread in price units."""
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol {symbol!r} not found in Market Watch")
    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)

    # start_pos=1 skips the bar currently forming.
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"no rates returned for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df['spread_cost'] = df['spread'] * info.point
    return df[['open', 'high', 'low', 'close', 'tick_volume', 'spread_cost']], info


# ------------------------------------------------------------------- compatibility

def check_compatibility(model, scaler, obs, cols):
    """Hard gate: the policy's input shape must match what we just built."""
    expected = tuple(model.observation_space.shape)
    actual = tuple(obs.shape)
    problems = []

    if scaler.n_features_in_ != len(FEATURES):
        problems.append(
            f"scaler was fitted on {scaler.n_features_in_} features "
            f"{list(scaler.feature_names_in_)}, but this code builds {len(FEATURES)}: {FEATURES}"
        )
    if expected != actual:
        problems.append(
            f"policy expects observations of shape {expected}, "
            f"this pipeline produces {actual} (columns: {cols})"
        )
    return problems


def policy_probabilities(model, obs):
    """Action probabilities from the PPO policy, or None if unavailable."""
    try:
        obs_tensor, _ = model.policy.obs_to_tensor(obs[None, ...])
        dist = model.policy.get_distribution(obs_tensor)
        return dist.distribution.probs.detach().cpu().numpy()[0].tolist()
    except Exception:
        return None


# ---------------------------------------------------------------------- execution

def resolve_filling_mode(mt5, info):
    """Pick a filling mode the broker actually supports instead of assuming IOC."""
    mode = getattr(info, 'filling_mode', 0)
    if mode & 1:  # SYMBOL_FILLING_FOK
        return mt5.ORDER_FILLING_FOK
    if mode & 2:  # SYMBOL_FILLING_IOC
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def send_order(mt5, symbol, info, side, lot, tp_dist, sl_dist, magic, deviation, comment):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"no tick for {symbol}: {mt5.last_error()}")

    digits = info.digits
    stops_level = getattr(info, 'trade_stops_level', 0) * info.point

    if side == "BUY":
        price = tick.ask
        tp, sl = price + tp_dist, price - sl_dist
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        tp, sl = price - tp_dist, price + sl_dist
        order_type = mt5.ORDER_TYPE_SELL

    if stops_level and (abs(price - tp) < stops_level or abs(price - sl) < stops_level):
        raise RuntimeError(
            f"TP/SL closer than the broker's stops level ({stops_level:.{digits}f}); "
            f"widen --tp-mult / --sl-mult"
        )

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": order_type,
        "price": round(price, digits),
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": resolve_filling_mode(mt5, info),
    }

    check = mt5.order_check(request)
    if check is not None and check.retcode != 0:
        raise RuntimeError(f"order_check rejected the request: retcode={check.retcode} {check.comment}")

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send returned None: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"order_send failed: retcode={result.retcode} {result.comment}")
    return result, request


# --------------------------------------------------------------------------- main

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--symbol', default=DEFAULT_SYMBOL)
    p.add_argument('--model', default=os.path.join(HERE, 'rl_model'))
    p.add_argument('--scaler', default=os.path.join(HERE, 'rl_scaler.save'))
    p.add_argument('--bars', type=int, default=WINDOW_SIZE + 60,
                   help='closed bars to pull (needs >= window + 40 for indicator warm-up)')
    p.add_argument('--window', type=int, default=WINDOW_SIZE)
    p.add_argument('--macro-lag-days', type=int, default=0,
                   help='shift DXY/US10Y by N sessions; 0 reproduces the shipped artifacts')
    p.add_argument('--tp-mult', type=float, default=1.0)
    p.add_argument('--sl-mult', type=float, default=2.0)
    p.add_argument('--tp-cap', type=float, default=3.00, help='hard cap on TP distance in price units')
    p.add_argument('--risk-pct', type=float, default=5.0, help='max loss per trade as %% of equity')
    p.add_argument('--max-spread-points', type=int, default=80)
    p.add_argument('--magic', type=int, default=20260729)
    p.add_argument('--deviation', type=int, default=20)
    p.add_argument('--json', dest='json_path', default=None, help='write the signal to this file')
    p.add_argument('--live', action='store_true', help='place the order (requires --yes)')
    p.add_argument('--yes', action='store_true', help='confirm live order placement')
    p.add_argument('--allow-incompatible', action='store_true',
                   help=argparse.SUPPRESS)  # forecast-only escape hatch for debugging
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.live and not args.yes:
        print("--live requires --yes. Nothing was sent.", file=sys.stderr)
        return 2

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 is not installed (it is Windows-only). "
              "Run this on the machine hosting the MT5 terminal.", file=sys.stderr)
        return 3

    import joblib
    from stable_baselines3 import PPO

    if not mt5.initialize():
        print(f"MT5 initialize() failed: {mt5.last_error()}", file=sys.stderr)
        return 3

    signal = {'generated_at': datetime.now(timezone.utc).isoformat(), 'symbol': args.symbol,
              'timeframe': 'D1', 'executed': False}
    exit_code = 0

    try:
        raw, info = fetch_closed_bars(mt5, args.symbol, mt5.TIMEFRAME_D1, args.bars)
        df = add_features(raw, macro_lag_days=args.macro_lag_days).dropna()
        if len(df) < args.window:
            raise RuntimeError(f"only {len(df)} usable bars after warm-up; raise --bars")

        scaler = joblib.load(args.scaler)
        model = PPO.load(args.model)

        obs, cols = build_observation(df, scaler, args.window)
        problems = check_compatibility(model, scaler, obs, cols)

        last_bar = df.index[-1]
        atr = float(df['atr_14'].iloc[-1])
        signal.update({
            'last_closed_bar': str(last_bar),
            'observation_window': [str(df.index[-args.window]), str(last_bar)],
            'observation_shape': list(obs.shape),
            'atr_14': atr,
            'model_expects': list(model.observation_space.shape),
            'compatible': not problems,
        })

        print(f"=== D1 forecast · {args.symbol} ===")
        print(f"observation : {df.index[-args.window].date()} .. {last_bar.date()}  "
              f"({args.window} closed bars, shape {obs.shape})")
        print(f"ATR(14)     : {atr:.2f}")

        if problems:
            print("\nARTIFACT MISMATCH — no forecast, no order:")
            for p in problems:
                print(f"  * {p}")
            print("\nThe committed model and scaler come from different feature sets. "
                  "Retrain with `python ml_bot/rl_train.py` so rl_model.zip and rl_scaler.save "
                  "are written from one run, then re-run this script. See ML_ANALYSIS.md.")
            signal['error'] = problems
            return 4 if not args.allow_incompatible else 0

        stale = out_of_range_features(df, scaler)
        if stale:
            signal['out_of_range_features'] = stale
            print("\nWARNING — features outside the scaler's fitted range "
                  "(the policy is extrapolating):")
            for name, rng in stale.items():
                print(f"  * {name}={rng['value']:.4f} outside "
                      f"[{rng['fitted_min']:.4f}, {rng['fitted_max']:.4f}]")

        action = int(model.predict(obs, deterministic=True)[0])
        side = ACTION_NAMES[action]
        probs = policy_probabilities(model, obs)

        signal.update({'action': action, 'side': side, 'action_probabilities': probs})
        print(f"\nforecast    : {side} (action={action})")
        if probs:
            print(f"policy probs: BUY {probs[0]:.3f} / SELL {probs[1]:.3f}")

        account = mt5.account_info()
        if account is None:
            raise RuntimeError("account_info() returned None")
        equity = float(account.equity)

        plan = compute_order_plan(
            equity, atr,
            tp_multiplier=args.tp_mult, sl_multiplier=args.sl_mult, tp_cap=args.tp_cap,
            risk_pct=args.risk_pct, contract_size=float(info.trade_contract_size),
            volume_min=float(info.volume_min), volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
        )
        signal['plan'] = plan
        signal['equity'] = equity

        print(f"\nequity      : ${equity:,.2f}")
        print(f"TP / SL dist: {plan['tp_dist']:.2f} / {plan['sl_dist']:.2f} price units")
        print(f"lot (mandate 0.01/$100): {plan['mandate_lot']:.4f}   "
              f"lot (risk cap {args.risk_pct}%): {plan['risk_capped_lot']:.4f}")
        if plan['lot'] is None:
            print(f"lot         : none — {plan['reason']}")
        else:
            print(f"lot         : {plan['lot']:.2f}  "
                  f"(risk ${plan['risk_money']:,.2f} = {plan['risk_pct_of_equity']:.2f}% of equity)")

        if not args.live:
            print("\ndry run — no order sent. Add --live --yes to place it.")
            return 0

        # ---- live path -------------------------------------------------------
        blockers = []
        if plan['lot'] is None:
            blockers.append(plan['reason'])

        spread_points = mt5.symbol_info(args.symbol).spread
        if spread_points > args.max_spread_points:
            blockers.append(f"spread {spread_points} points > --max-spread-points {args.max_spread_points}")

        existing = mt5.positions_get(symbol=args.symbol) or []
        if any(p.magic == args.magic for p in existing):
            blockers.append(f"a position with magic {args.magic} is already open")

        if blockers:
            print("\nORDER SKIPPED:")
            for b in blockers:
                print(f"  * {b}")
            signal['skipped'] = blockers
            return 5

        result, request = send_order(
            mt5, args.symbol, info, side, plan['lot'], plan['tp_dist'], plan['sl_dist'],
            args.magic, args.deviation, f"d1_forecast_{side}",
        )
        signal.update({
            'executed': True,
            'ticket': int(result.order),
            'fill_price': float(result.price),
            'volume': float(result.volume),
            'sl': request['sl'],
            'tp': request['tp'],
        })
        print(f"\nORDER SENT  : ticket {result.order}  {side} {result.volume} @ {result.price} "
              f"SL {request['sl']} TP {request['tp']}")

    except Exception as exc:
        signal['error'] = str(exc)
        print(f"\nERROR: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if args.json_path:
            with open(args.json_path, 'w') as fh:
                json.dump(signal, fh, indent=2, default=str)
            print(f"signal written to {args.json_path}")
        mt5.shutdown()

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
