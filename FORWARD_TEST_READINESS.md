# Forward-Test Readiness — Fix Log & How to Backtest

**Scope.** Follow-up to `ANALYSIS_REPORT.md`: fixes to the Python RL bot so it can be
forward-tested on a demo account, plus an offline backtester. Branch:
`claude/forward-test-readiness-kgtcoo`.

---

## 1. What the committed model artifacts actually are

Inspecting the binaries (not the code) shows which pipeline each artifact belongs to:

| Artifact | Observation | Matches |
|---|---|---|
| `rl_model.zip` (D1, 1M steps) | (20, **13**) = 12 scaled features + raw `spread_cost` | legacy env at git `6150c7d`: **long-only**, action 1 = BUY at daily open, flat TP = entry + $3.00, EOD close, no SL, lot 0.01/$100 capped 10 |
| `rl_scaler.save` | 14 features | current Always-In pipeline — **not** the D1 model's scaler |
| `rl_scaler_legacy.save` | 10 features | an older 10-feature model that is no longer in the repo |
| `rl_model_h4/h8/h12.zip` | (20, **10**) | a pre-macro feature set whose training code was never committed |

Consequences:

- **No committed scaler matched the D1 model.** Fixed by slicing the first 12 columns
  of `rl_scaler.save` (fitted the same day, on the same broker data, and MinMaxScaler
  is per-column) → committed as **`rl_scaler_d1_legacy.save`**.
- **The current `rl_env.py`/`rl_train.py` (Always-In, 16-feature obs) has no trained
  model.** If you retrain with them, update `rl_trader.py`'s feature lists and action
  mapping (comments in the file show where).
- **h4/h8/h12 models are stale** (obs 10 ≠ anything the current code produces). Their
  traders now verify the observation shape at startup and will refuse to run until the
  models are retrained with `rl_train_h4/h8/h12.py`.

## 2. Bugs fixed in `ml_bot/`

`rl_trader.py` (rewritten):

1. `wait_for_new_bar()` was called but never defined → crashed at startup.
2. `open_trade()` called with a keyword that didn't exist in its signature → crash on
   first BUY signal.
3. Opened positions with magic `234000` but closed only magic `999999` → the bot could
   never close its own positions. Now one `MAGIC_NUMBER` everywhere.
4. Action mapping now matches the model's training env (legacy: 1 = BUY, 0 = skip).
   The old code was written for a different env's mapping.
5. Observation now built exactly as in training (12 scaled + raw `spread_cost`),
   with **startup guards**: model obs shape and scaler feature count are checked, and
   the bot refuses to trade on any mismatch.
6. Broker-side TP (+$3.00 as trained) **and a disaster SL at 5% of equity** attached
   to every order. At the mandated sizing $1/oz ≈ 1% of equity, so an ATR-based stop
   (2×ATR ≈ $130) would be >100% of equity — the stop must be equity-based. 5% matches
   `InpMaxDailyLossPct` in the robust EA.
7. Half-applied ensemble patch removed (its `cnn_lstm_model.keras` /
   `xgboost_model.json` were never committed; `update_trader_ensemble.py`'s string
   replacements had silently failed).
8. Paths are now relative to the script location, error handling + retry around the
   whole loop, dead-tick guards on all order sends.

`rl_trader_h4/h8/h12.py` (rewritten): undefined-variable crash in `open_trade`
(`tp_percent`), no-SL entries, and 12-vs-13 feature mismatch fixed the same way; plus
the startup shape guard that currently (correctly) blocks the stale models.

`requirements.txt`: added `xgboost`, `joblib`.

## 3. Offline backtest

`ml_bot/backtest_offline.py` replays the exact legacy trade rules with the real
model + scaler, no MT5 needed:

```
# on any machine with internet:
python ml_bot/backtest_offline.py                      # uses yfinance GC=F
# with broker data (preferred – run export on the MT5 machine first):
python ml_bot/export_data_mt5.py
python ml_bot/backtest_offline.py --csv ml_bot/xauusd_d1.csv
```

It reports full-window, 2026-YTD and post-2026-06-29 (true out-of-sample) stats, each
with and without the 5% disaster SL, and saves an equity-curve plot. When a D1 bar
touches both TP and SL the outcome is drawn with the continuous-barrier probability
`SL/(TP+SL)` (same math as `sim_robustness.py`) — OHLC alone cannot order intrabar
touches.

**Status of this run:** this sandbox's network policy blocks all market-data hosts
(Yahoo, Stooq, FRED all 403), so real-data numbers could not be produced here. The
pipeline was verified end-to-end with clearly-labeled synthetic data: the model loads,
predicts a non-degenerate mix (~51% BUY days on the synthetic path), trades resolve,
stats/plots generate, and the 5% SL caps every loss near 5% as designed. Run one of
the commands above on your machine to get the real numbers before going to demo.

## 4. Demo forward-test checklist

- [ ] `python ml_bot/backtest_offline.py --csv ...` on broker data — confirm the
      out-of-sample window isn't already falling apart.
- [ ] EA side: compile `XAUUSD_DayOpen_Long_Robust.mq5` (the committed `.ex5` is
      still v1.20) and A/B it per `ANALYSIS_REPORT.md` §5.
- [ ] Start `rl_trader.py` on the **demo** account; the startup guards should print
      the model/scaler check and then wait for the next D1 bar.
- [ ] Expectations: the model's profits in backtests came mostly from gold's uptrend;
      the ~$0.30/day spread toll and the $3-TP barrier race mean flat/down regimes
      will bleed. The 5% SL bounds the damage; it does not create edge.
- [ ] Do **not** run the h4/h8/h12 traders until retrained (they will refuse anyway).
