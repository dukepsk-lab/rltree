"""Feature engineering shared by training, backtesting and live serving.

This module is deliberately free of `MetaTrader5`, `torch` and `stable_baselines3`
imports so it can be imported (and unit-tested) anywhere. `rl_train.py` re-exports
`add_features` / `add_macro_data` from here, so training and serving can never drift
apart in feature order or definition.

Action convention (matches `rl_env.TradingEnv`): 0 = BUY, 1 = SELL.
"""

import numpy as np
import pandas as pd

# The D1 feature set, in the exact order the scaler was fitted on.
FEATURES = [
    'open', 'high', 'low', 'close', 'tick_volume',
    'sma_10', 'sma_20', 'rsi_14', 'adx_14', 'linreg_20',
    'dxy', 'us10y', 'atr_14', 'day_of_week',
]

# Columns TradingEnv drops when it builds `feature_cols`. Kept here verbatim so
# the observation layout used at serving time is derived by the same rule as in
# training instead of being hand-maintained.
ENV_EXCLUDED_COLS = ['open', 'high', 'low', 'close', 'tick_volume', 'time', 'target']


def add_macro_data(df, macro_lag_days=0):
    """Join DXY and US10Y daily closes onto a bar frame.

    `macro_lag_days=1` shifts the macro series one row forward so a bar can only
    ever see the *previous* session's macro close. The default of 0 reproduces the
    behaviour the shipped scaler/model were built with; see ML_ANALYSIS.md #6.
    """
    import yfinance as yf

    start_date = df.index.min()
    end_date = df.index.max() + pd.Timedelta(days=1)

    dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date, progress=False)['Close']
    us10y = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close']

    if isinstance(dxy, pd.DataFrame):
        dxy = dxy.iloc[:, 0]
    if isinstance(us10y, pd.DataFrame):
        us10y = us10y.iloc[:, 0]

    macro_df = pd.DataFrame({'dxy': dxy, 'us10y': us10y})

    if macro_df.index.tz is not None:
        macro_df.index = macro_df.index.tz_localize(None)

    if macro_lag_days:
        macro_df = macro_df.shift(macro_lag_days)

    df = df.join(macro_df, how='left')
    df['dxy'] = df['dxy'].ffill().bfill()
    df['us10y'] = df['us10y'].ffill().bfill()

    return df


def add_indicators(df):
    """Price-derived indicators only — no network access, no macro join."""
    df = df.copy()

    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
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


def add_features(df, macro_lag_days=0):
    """Macro join + indicators, in the order the shipped artifacts expect."""
    df = df.copy()
    df = add_macro_data(df, macro_lag_days=macro_lag_days)
    return add_indicators(df)


def build_env_frame(df, scaler):
    """Scale `FEATURES` and append the raw columns TradingEnv keeps.

    Returns the same frame layout `rl_train.prepare_rl_data` produces, so the
    observation columns can be selected with the env's own exclusion rule.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    scaled = scaler.transform(df[FEATURES])
    scaled_df = pd.DataFrame(scaled, columns=[f"scaled_{f}" for f in FEATURES], index=df.index)
    raw_cols = [c for c in ['open', 'high', 'low', 'close', 'spread_cost', 'atr_14'] if c in df.columns]
    return pd.concat([scaled_df, df[raw_cols]], axis=1)


def observation_columns(env_frame):
    """Column order TradingEnv feeds the policy — derived, not hardcoded."""
    return [c for c in env_frame.columns if c not in ENV_EXCLUDED_COLS]


def build_observation(df, scaler, window_size=20):
    """Latest `window_size` rows as a float32 observation of shape (window, n_cols)."""
    env_frame = build_env_frame(df, scaler)
    cols = observation_columns(env_frame)
    if len(env_frame) < window_size:
        raise ValueError(f"need {window_size} rows, have {len(env_frame)}")
    obs = env_frame[cols].iloc[-window_size:].values.astype(np.float32)
    if not np.isfinite(obs).all():
        raise ValueError("observation contains NaN/inf — check the warm-up window and macro join")
    return obs, cols


def out_of_range_features(df, scaler, tolerance=0.0):
    """Features whose latest value sits outside the scaler's fitted min/max.

    MinMaxScaler does not clip, so out-of-range inputs are silently extrapolated
    into observation values outside [0, 1] the policy never saw in training.
    """
    last = df[FEATURES].iloc[-1]
    flagged = {}
    for name, value, lo, hi in zip(scaler.feature_names_in_, last.values, scaler.data_min_, scaler.data_max_):
        span = (hi - lo) or 1.0
        if value < lo - tolerance * span or value > hi + tolerance * span:
            flagged[name] = {'value': float(value), 'fitted_min': float(lo), 'fitted_max': float(hi)}
    return flagged


def compute_order_plan(equity, atr, *, tp_multiplier=1.0, sl_multiplier=2.0, tp_cap=3.00,
                       risk_pct=5.0, contract_size=100.0, volume_min=0.01,
                       volume_max=100.0, volume_step=0.01, max_lot=10.0):
    """Size a trade under both the repo's sizing rule and a hard per-trade risk cap.

    The repo's mandate is 0.01 lot per $100 of equity. With an ATR-based stop that
    rule alone risks far more than the account per trade (ANALYSIS_REPORT.md §3),
    so the lot is additionally shrunk until `sl_dist` costs at most `risk_pct` of
    equity. Returns a dict; `lot` is None when no size satisfies both rules.
    """
    tp_dist = min(atr * tp_multiplier, tp_cap)
    sl_dist = atr * sl_multiplier

    mandate_lot = min((equity / 100.0) * 0.01, max_lot)
    risk_budget = equity * (risk_pct / 100.0)
    per_lot_risk = sl_dist * contract_size

    if per_lot_risk <= 0:
        return {'lot': None, 'reason': 'non-positive stop distance (ATR unavailable)',
                'tp_dist': tp_dist, 'sl_dist': sl_dist, 'mandate_lot': mandate_lot}

    risk_capped_lot = risk_budget / per_lot_risk
    lot = min(mandate_lot, risk_capped_lot, volume_max)
    lot = np.floor(lot / volume_step) * volume_step
    lot = float(round(lot, 8))

    plan = {
        'tp_dist': tp_dist,
        'sl_dist': sl_dist,
        'mandate_lot': mandate_lot,
        'risk_capped_lot': risk_capped_lot,
        'risk_budget': risk_budget,
        'lot': lot,
        'risk_money': lot * per_lot_risk,
        'risk_pct_of_equity': (lot * per_lot_risk / equity * 100.0) if equity else float('inf'),
        'reason': None,
    }

    if lot < volume_min:
        plan['lot'] = None
        plan['reason'] = (f"minimum lot {volume_min} would risk "
                          f"${volume_min * per_lot_risk:,.2f} = "
                          f"{volume_min * per_lot_risk / equity * 100:.1f}% of equity "
                          f"(cap {risk_pct}%) — no tradeable size")
    return plan
