# ML Pipeline Analysis — D1 forecast & order placement

**Scope.** The machine-learning side of the repo (`ml_bot/`): the PPO agent (`rl_train.py`,
`rl_env.py`), the CNN+LSTM and XGBoost models, the 3-model ensemble, and the live traders.
`ANALYSIS_REPORT.md` already covers the MQL5 expert advisor and the risk math; this document
does not repeat it.

**Headline.** The committed D1 artifacts cannot produce a forecast. `rl_model.zip` expects an
observation of shape `(20, 13)`; `rl_scaler.save` was fitted on 14 features; the code path in
`rl_trader.py` builds 14 columns and the training environment builds 16. Those are four
different numbers for the same tensor. `model.predict()` raises before any forecast exists, and
the live D1 loop crashes earlier still on an undefined function. Every backtest number in the
repo is in-sample.

---

## 1. The model, the scaler and the code disagree about the input

Decoded straight from the artifacts (`python ml_bot/test_features.py` re-checks this):

| Artifact | Inputs | Evidence |
|---|---|---|
| `rl_model.zip` | `Box(20, 13)` | `observation_space` in the archive's `data` member |
| `rl_scaler.save` | 14 features | `n_features_in_ = 14`, `feature_names_in_` = the D1 list |
| `rl_trader.py` (before this change) | 14 columns | scaled features only, `rl_trader.py:238` |
| `rl_env.TradingEnv` | 16 columns | 14 scaled + raw `spread_cost` + raw `atr_14` |

13 is what you get from the *older* 12-feature list (no `atr_14`, no `day_of_week`) plus
`spread_cost`. So `rl_model.zip` was trained before `atr_14`/`day_of_week` were added, and the
scaler next to it has since been replaced.

Loading the model and feeding it the pipeline's observation reproduces it exactly:

```
>>> PPO.load('ml_bot/rl_model').observation_space
Box(-inf, inf, (20, 13), float32)
>>> model.predict(obs)          # obs.shape == (20, 16)
ValueError: Error: Unexpected observation shape (20, 16) for Box environment,
please use (20, 13) or (n_env, 20, 13) for the observation shape.
```

**Root cause — three trainers wrote the same file.**

* `cnn_lstm_train.py:47` refitted a `MinMaxScaler` and wrote it to `ml_bot/rl_scaler.save`, with
  the comment *"Overwrite global scaler with 14 features"*. That silently invalidates whatever
  PPO model was trained against the previous scaler.
* `xgboost_train.py:31-37` wrapped `scaler.transform()` in a bare `except:` that, on any error —
  including exactly the feature-count error this situation produces — refitted a new scaler and
  overwrote the same file. A failure that should have stopped the run instead destroyed the
  artifact that caused it.

Both are fixed here: `rl_train.py` is the only writer of `rl_scaler.save`, and the other two
load it and exit with a message if the feature count doesn't match.

## 2. The live D1 trader could not have run

`rl_trader.py` — the script that is supposed to do "forecast D1, then open the order":

| Line (before) | Problem |
|---|---|
| `225`, `249` | `wait_for_new_bar(...)` is called but never defined in this module → `NameError` on the first call, before the loop starts |
| `245` | `open_trade(SYMBOL, "BUY", tp_price_diff=TP_PRICE_DIFF)` against `def open_trade(symbol, action_type, tp_multiplier, sl_multiplier)` → `TypeError` |
| `243-247` | Treats action `1` as BUY and action `0` as HOLD. `rl_env.TradingEnv` defines **0 = BUY, 1 = SELL**, and has no flat action at all. Had the call signature been right, the bot would have bought precisely on the bars the agent wanted to sell, and stood aside on the rest |
| `197` vs `99` | Orders are sent with magic `234000` while `close_all_positions()` looks for magic `999999`, so nothing the bot opens is ever closed by it |
| `232`, `238` | Observation built from the scaled features only (14 columns, missing `spread_cost`/`atr_14`), and from `copy_rates_from_pos(..., 0, n)` whose last row is the **bar currently forming** — a partial close/high/low the model never saw in training |

All five are fixed in `rl_trader.py`, but prefer `d1_forecast.py` (§6) for a single forecast.

## 3. Every reported performance number is in-sample

* `rl_train.py` trains on all `DATA_LIMIT = 5000` bars. `rl_backtest.py:45` then evaluates on the
  last 20% **of the same 5000 bars**. The equity curve in `backtest_result.png` is a plot of the
  agent replaying data it was trained on.
* `cnn_lstm_train.py:46` and `xgboost_train.py` fit the `MinMaxScaler` on the whole series before
  the 80/20 split, so the training rows carry min/max information from the test period.
* No walk-forward, no purge/embargo around the split, and no baseline: neither script prints the
  majority-class rate, so "Test Accuracy: 54%" cannot be read as better than always predicting up.
* `optimize_rl.py`'s hyperparameters (`learning_rate=3.36e-3`, `gamma=0.952`) were selected on
  the same in-sample window, so they are fitted to it as well. `3.36e-3` is roughly 7× the usual
  PPO setting for a 256×256 MLP.
* `rl_env.reset()` always restarts at `current_step = window_size` — one fixed trajectory, no
  random starts. With `TIMESTEPS = 1_000_000` over ~4,900 usable bars the agent makes ~200
  identical passes over the same path. That is memorization, and it is what an in-sample equity
  curve rewards.

## 4. The inputs make the D1 problem harder than it needs to be

* **Raw price levels through a `MinMaxScaler`.** The scaler's `close` range is
  \$1,052.87 → \$5,512.91. A \$65 daily ATR move is **1.5% of the scaled range**, while `rsi_14`
  uses the full 0–1 span. The network's price inputs are effectively a slow-moving regime
  indicator, not a description of the day. Gold's level is also non-stationary — new highs map
  outside `[0, 1]` (`MinMaxScaler` does not clip), and the policy extrapolates on inputs it never
  saw. `d1_forecast.py` now warns when any feature leaves its fitted range.
  Returns, `close/sma - 1`, and ATR-normalized distances would fix this.
* **Mixed magnitudes in one vector.** 14 columns in `[0, 1]` sit next to raw `spread_cost` (~0.15)
  and raw `atr_14` (7 → 295). `atr_14` appears twice, once scaled and once raw.
* **Weekend bars.** `day_of_week` reaches 6, so Saturday/Sunday broker bars are in the training
  set. Those micro-range bars distort the ATR/RSI warm-up and the label distribution.
* **Macro data has no lag.** `add_macro_data` joins the DXY and US10Y close of calendar day *T*
  onto bar *T*. For a UTC+2/+3 broker, bar *T* opens around 21:00–22:00 UTC on day *T−1*, so the
  day-*T* macro close is not knowable at entry and the day-*T−1* close lands on the boundary.
  `features.add_macro_data(..., macro_lag_days=1)` now shifts it; the default is `0` so the
  shipped artifacts keep their original semantics.

## 5. Reward, environment and ensemble

**`rl_env.TradingEnv`**

* `spaces.Discrete(2)` — BUY or SELL, **no flat**. The agent is forced into the market on every
  single bar and pays the spread every day. That is the same structural drag `ANALYSIS_REPORT.md`
  §3 measured at ~0.2–0.3%/day; no policy can learn its way out of a cost it must pay every step.
* Reward is raw USD P&L, and `lot_size` scales with `self.balance`, so the reward magnitude grows
  with the equity curve. Nothing normalizes it (`VecNormalize` is not used), which makes the
  value function's target non-stationary.
* The ruin penalty is `reward -= 100` (`rl_env.py:119`) while ordinary daily rewards run into the
  hundreds of dollars at a \$10,000 balance. Blowing up the account costs the agent less than one
  average day. If drawdown is supposed to matter, it has to be priced into the reward.
* No commission or swap; the spread is hardcoded to `0.15` (`rl_env.py:71`) even though the real
  per-bar spread is available (and is fed in as a feature).
* `lot_size` is never rounded to the broker's `volume_step`/`volume_min`, so simulated fills are
  sizes the broker would reject.
* One thing that *is* right: when a bar's range covers both TP and SL, the SL is taken first —
  the pessimistic assumption.

**Ensemble (`ml_trader.py`, `ensemble_backtest.py`)**

* The three models don't answer the same question. CNN and XGB are trained on
  "next bar closes above its open" (`cnn_lstm_train.py:56`); the RL agent is trained on the P&L of
  an ATR-based TP/SL barrier race. A vote between a direction forecast and a barrier outcome is
  not an ensemble of the same estimator.
* CNN/XGB pair the window `[i-20, i)` with the target of bar `i+1` — **bar `i` is skipped**. They
  are trained to predict two bars ahead and served as a one-bar-ahead signal.
* **The EXP level system cannot discriminate.** `+10` for a correct call, `-5` for a wrong one, so
  EXP grows whenever accuracy exceeds **1/3**. Every model above 33.3% levels up forever and
  without bound. A 55%-accurate and a 50%-accurate model converge to a fixed ~1.3 weight ratio —
  never enough for one model to outvote the other two — so weighted voting degenerates to a plain
  majority vote plus an ever-growing state file.
* `evaluate_yesterday()` (`ml_trader.py:27`) returns early unless *RL* has a stored prediction,
  then grades all three models against `df.iloc[-2]`'s close-vs-open regardless of which bar each
  prediction was made for.
* Ties go to SELL: `return 0 if buy_score > sell_score else 1`.
* `ensemble_backtest.py:100` feeds the 14-column observation to the RL model, so it hits the same
  shape error as §1.

## 6. What is in this change

| File | What it does |
|---|---|
| `ml_bot/features.py` | Single source of truth for feature engineering, observation assembly, out-of-range detection and position sizing. No `MetaTrader5`/`torch` imports, so it is testable anywhere. `rl_train.py` re-exports from it |
| `ml_bot/d1_forecast.py` | D1 forecast, and optionally the order. Dry-run by default |
| `ml_bot/train_d1.py` | Walk-forward training and out-of-sample evaluation; writes model, scaler and metadata from one run |
| `ml_bot/rl_env.py` | Optional flat action, percent-of-balance reward, per-bar spread, drawdown penalty, random episode starts — all off by default |
| `ml_bot/test_features.py` | Offline checks: observation layout matches the env's own column rule, the artifact mismatch is real, the risk cap behaves, the env's defaults are unchanged |
| `ml_bot/rl_trader.py` | The five bugs in §2 fixed |
| `ml_bot/xgboost_train.py`, `ml_bot/cnn_lstm_train.py` | Stop overwriting `rl_scaler.save`; fail loudly on a feature-count mismatch |
| `ml_bot/rl_train.py` | Uses the shared feature module; writes `rl_model_meta.json` alongside the model |

### `d1_forecast.py`

```bash
python ml_bot/d1_forecast.py                        # forecast only — sends nothing
python ml_bot/d1_forecast.py --json signal.json     # also write the signal file
python ml_bot/d1_forecast.py --live --yes           # forecast AND place the order
```

It requires a running MetaTrader 5 terminal on the same machine (the `MetaTrader5` package is
Windows-only). Compared with `rl_trader.py` it:

* **refuses to trade on an artifact mismatch** instead of guessing a shape — with the artifacts as
  committed it exits with code 4 and tells you to retrain;
* uses **closed bars only** (`copy_rates_from_pos(..., 1, n)`);
* uses the env's action convention (0 = BUY, 1 = SELL) and prints the policy's action
  probabilities, so the forecast comes with a confidence number rather than a bare label;
* caps risk per trade. The repo's mandate (0.01 lot per \$100) with a 2×ATR stop risks **130% of
  equity** on a \$10,000 account at ATR \$65 — the sizing rule and the stop were never reconciled.
  `--risk-pct` (default 5%) shrinks the lot until the stop is affordable, and reports when no
  size works at all;
* attaches broker-side SL/TP, respects `trade_stops_level`, gates on spread, detects the broker's
  filling mode instead of assuming IOC, and refuses to open a second position on the same magic;
* sends nothing without both `--live` and `--yes`.

### `train_d1.py` — retraining

```bash
python ml_bot/train_d1.py --timesteps 300000        # walk-forward, then the final model
python ml_bot/train_d1.py --folds 5 --no-final      # validation only, artifacts untouched
python ml_bot/train_d1.py --synthetic --timesteps 3000   # offline smoke test, no MT5 needed
```

It fixes the three things that made the old numbers meaningless:

* the scaler is fitted on **each fold's training slice only**, so test-period minima and maxima
  never reach the training rows;
* each fold trains strictly before its test window and is scored on unseen bars;
* every fold is printed next to always-BUY, always-SELL, always-FLAT and the majority-class
  direction rate.

Defaults turn on the §5 fixes: a flat action, percent-of-balance reward, per-bar spread, a
drawdown penalty and random episode starts. `--no-allow-flat --reward-mode usd --no-random-start`
reproduces the old environment. `learning_rate` defaults to `3e-4` rather than the in-sample
Optuna value of `3.36e-3`.

**Read the direction-accuracy line, not the return line.** With 0.01 lot per \$100 the balance
compounds ~1% per bar, so returns are lognormal — a handful of lucky paths dominate the mean and
a run can print +800% with accuracy *below* the majority class. The smoke run on random-walk data
does exactly that, which is the point: the script calls it correctly.

```
=== walk-forward summary (3 folds, out-of-sample) ===
agent return      : median +813.53%   worst +216.35%   positive folds 3/3
direction accuracy: 47.82%   majority class 52.34%   edge -4.52 points
verdict           : NO directional edge — accuracy is at or below the majority class.
                    Any positive return here is the compounding tail, not skill.
```

## 7. To get a forecast worth acting on

1. **Retrain the pair on real broker data.** `python ml_bot/train_d1.py` writes `rl_model.zip`,
   `rl_scaler.save` and `rl_model_meta.json` from one run. Until then §1 stands and
   `d1_forecast.py` will not trade.
2. **Judge it on the walk-forward edge.** For daily gold direction, anything under roughly
   +2 points over the majority class, out-of-sample and stable across seeds, is noise.
3. **Fix the representation before adding models.** Returns and ATR-normalized distances instead
   of raw price levels; drop weekend bars; keep `--macro-lag-days 1`.
4. Then re-read `ANALYSIS_REPORT.md` §7: even a genuinely predictive model still has to clear the
   ~0.2–0.3%/day cost of trading every session at this size.
