"""Offline checks for the serving-side feature/observation/sizing code.

Runs without MetaTrader5, torch or stable_baselines3:

    python ml_bot/test_features.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F


def synthetic_bars(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2025-01-01', periods=n)
    close = 4000 + np.cumsum(rng.normal(0, 30, n))
    high = close + rng.uniform(5, 60, n)
    low = close - rng.uniform(5, 60, n)
    open_ = close - rng.normal(0, 20, n)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
        'tick_volume': rng.integers(1000, 90000, n).astype(float),
        'spread_cost': np.full(n, 0.15),
        'dxy': 100 + rng.normal(0, 1, n),
        'us10y': 4 + rng.normal(0, 0.1, n),
    }, index=idx)


def test_observation_matches_env_layout():
    """The observation must reproduce TradingEnv's column selection exactly."""
    import joblib
    df = F.add_indicators(synthetic_bars()).dropna()
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'rl_scaler.save'))

    obs, cols = F.build_observation(df, scaler, window_size=20)

    assert cols == [f"scaled_{f}" for f in F.FEATURES] + ['spread_cost', 'atr_14'], cols
    assert obs.shape == (20, 16), obs.shape
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    # Same rule TradingEnv applies, run against the same frame.
    env_frame = F.build_env_frame(df, scaler)
    env_cols = [c for c in env_frame.columns
                if c not in ['open', 'high', 'low', 'close', 'tick_volume', 'time', 'target']]
    assert env_cols == cols
    print(f"ok  observation layout: {obs.shape}, {len(cols)} columns")


def test_shipped_artifacts_are_incompatible():
    """Documents the current repo state: the D1 model predates the D1 scaler."""
    import json
    import zipfile
    import base64
    import joblib

    here = os.path.dirname(os.path.abspath(__file__))
    data = json.loads(zipfile.ZipFile(os.path.join(here, 'rl_model.zip')).read('data').decode())
    blob = base64.b64decode(data['observation_space'][':serialized:'])
    # ...'_shape' MEMOIZE BININT1 <window> BININT1 <n_cols> TUPLE2...
    marker = blob.find(b'_shape')
    assert blob[marker + 7] == blob[marker + 9] == ord('K'), 'unexpected pickle layout'
    window, n_cols = blob[marker + 8], blob[marker + 10]

    scaler = joblib.load(os.path.join(here, 'rl_scaler.save'))
    pipeline_cols = len(F.FEATURES) + 2  # + spread_cost + atr_14

    print(f"ok  rl_model.zip expects (window={window}, features={n_cols}); "
          f"rl_scaler.save has {scaler.n_features_in_} features -> "
          f"pipeline builds {pipeline_cols}")
    assert n_cols != pipeline_cols, (
        "artifacts now agree — update ML_ANALYSIS.md #1, the mismatch is fixed"
    )


def test_risk_cap_shrinks_the_mandated_lot():
    """0.01 lot per $100 with a 2xATR stop risks far more than the account."""
    plan = F.compute_order_plan(10_000.0, 65.0, sl_multiplier=2.0, risk_pct=5.0)
    assert plan['mandate_lot'] == 1.0
    assert plan['lot'] < plan['mandate_lot']
    assert plan['risk_pct_of_equity'] <= 5.0 + 1e-9
    unclamped_risk = plan['mandate_lot'] * plan['sl_dist'] * 100.0
    print(f"ok  risk cap: mandate lot 1.00 would risk ${unclamped_risk:,.0f} "
          f"({unclamped_risk / 100:.0f}% of equity) -> capped to {plan['lot']:.2f} lot "
          f"(${plan['risk_money']:,.0f}, {plan['risk_pct_of_equity']:.1f}%)")


def test_no_tradeable_size_is_reported_not_guessed():
    plan = F.compute_order_plan(200.0, 65.0, sl_multiplier=2.0, risk_pct=5.0)
    assert plan['lot'] is None and plan['reason']
    print(f"ok  small account: {plan['reason']}")


def test_out_of_range_detection():
    import joblib
    df = F.add_indicators(synthetic_bars()).dropna()
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'rl_scaler.save'))
    df.loc[df.index[-1], 'close'] = 99_999.0
    flagged = F.out_of_range_features(df, scaler)
    assert 'close' in flagged, flagged
    print(f"ok  out-of-range detection flagged: {sorted(flagged)}")


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
