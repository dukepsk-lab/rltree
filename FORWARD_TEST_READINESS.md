# Forward-Test Readiness — Fix Log & How to Backtest

**Scope.** Follow-up to `ANALYSIS_REPORT.md`: fixes to the Python RL bot so it can be
forward-tested on a demo account, plus an offline backtester. Branch:
`claude/forward-test-readiness-kgtcoo`.

---

## 1. Which model this targets — IMPORTANT, read before running anything

The repo's `ml_bot/rl_env.py` / `rl_train.py` implement an **Always-In D1 strategy**:
action 0 = BUY, action 1 = SELL, a position is opened on *every* bar (no flat/skip
state), TP = `min(atr * TP_MULTIPLIER, $3.00)`, SL (as trained) = `atr *
SL_MULTIPLIER` with **no cap**, lot = 0.01 per $100 equity capped at 10, $100/oz/lot.
Observation shape: `(20, 16)` — 14 MinMax-scaled features (`rl_scaler.save`) + raw
`spread_cost` + raw `atr_14`.

The `rl_model.zip` that was committed to git before this branch was actually trained
against an **older, different env** (git `6150c7d`: long-only, action 1 = BUY,
flat $3 TP, obs shape `(20, 13)`) — that mismatch is what the earlier version of this
fix worked around with a sliced 12-feature scaler.

**As of this branch, `rl_trader.py` and `backtest_offline.py` target the CURRENT
Always-In env above**, because that's what you're training locally. **You need to
commit and push your locally retrained `ml_bot/rl_model.zip` and
`ml_bot/rl_scaler.save`** — right now those two files only exist on your machine
(uncommitted), so anyone else pulling this branch still gets the old
git-committed model, which will fail the shape check in both scripts
(`unexpected model obs (20, 13)`, not `(20, 16)`).

```
git add ml_bot/rl_model.zip ml_bot/rl_scaler.save
git commit -m "Commit retrained Always-In D1 model"
git push
```

If you intended to keep testing the *old* git-committed legacy model instead, say so —
that needs a different trader config (BUY-only, flat $3 TP, 13-dim obs); ask and it'll
be restored.

## 2. Why the trained SL can't be sent to the broker as-is

`SL_MULTIPLIER = 2.0`, uncapped. At the mandated sizing (lot = equity/100 * 0.01), a
$1 gold move ≈ 1% of equity. Gold's ATR-14 is routinely $30-100+, so `atr * 2` can be
$60-200+ — **that SL distance alone can exceed 100% of equity in a single bar**. TP is
capped at $3 but SL is not: the training env's reward function never needed the SL to
be survivable because `done = balance < 0.5 * initial_balance` just ends the episode,
it doesn't prevent the loss.

`rl_trader.py` instead attaches a **broker-side SL sized as 5% of equity**
(`SL_EQUITY_PCT`, converted to a price distance via lot size and contract size) —
matching `InpMaxDailyLossPct` in the robust EA. This means **the deployed risk is not
identical to what the agent was trained against**; `backtest_offline.py` reports all
three variants (no SL / trained uncapped ATR SL / deployed 5%-equity SL) side by side
so you can see the gap before going live.

## 3. Bugs fixed in `ml_bot/`

`rl_trader.py` (rewritten):

1. `wait_for_new_bar()` was called but never defined → crashed at startup.
2. `open_trade()` called with a keyword that didn't exist in its signature → crash on
   first BUY signal.
3. Opened positions with magic `234000` but closed only magic `999999` → the bot could
   never close its own positions. Now one `MAGIC_NUMBER` everywhere.
4. Action mapping matches the Always-In env (0 = BUY, 1 = SELL, always trades).
5. Observation built exactly as in training (14 scaled + raw `spread_cost` + raw
   `atr_14`), with **startup guards**: model obs shape and scaler feature count are
   checked, and the bot refuses to trade on any mismatch.
6. Broker-side TP (as trained, capped at $3) **and a disaster SL at 5% of equity**
   attached to every order — see §2.
7. Half-applied ensemble patch removed (its `cnn_lstm_model.keras` /
   `xgboost_model.json` were never committed; `update_trader_ensemble.py`'s string
   replacements had silently failed).
8. Paths are now relative to the script location, error handling + retry around the
   whole loop, dead-tick guards on all order sends.

`rl_trader_h4/h8/h12.py` (rewritten, **unaffected by §1** — separate model family):
undefined-variable crash in `open_trade` (`tp_percent`), no-SL entries, and a 10-vs-9
feature mismatch fixed the same way; startup shape guard currently (correctly) blocks
these stale 10-feature models until retrained with `rl_train_h4/h8/h12.py`.

`requirements.txt`: added `xgboost`, `joblib`.

## 4. Offline backtest

`ml_bot/backtest_offline.py` replays the exact Always-In trade rules with the real
model + scaler, no MT5 needed:

```
# on any machine with internet:
python ml_bot/backtest_offline.py                      # uses yfinance GC=F
# with broker data (preferred – run export on the MT5 machine first):
python ml_bot/export_data_mt5.py
python ml_bot/backtest_offline.py --csv ml_bot/xauusd_d1.csv
```

It reports three risk variants (no SL / trained uncapped ATR SL / deployed 5%-equity
SL) and saves an equity-curve plot for each. When a D1 bar touches both TP and SL the
outcome is drawn with the continuous-barrier probability `SL/(TP+SL)` (same math as
`sim_robustness.py`) — OHLC alone cannot order intrabar touches.

**Status of this run:** this sandbox's network policy blocks all market-data hosts
(Yahoo, Stooq, FRED all 403), and the sandbox does not have your locally-retrained
`rl_model.zip`/`rl_scaler.save` (see §1) — so no real numbers could be produced here.
The full pipeline (model load, shape asserts, per-bar prediction, all three SL modes,
stats, plotting) was verified end-to-end with a throwaway model trained on
clearly-labeled synthetic data, confirming the code runs correctly; it says nothing
about the actual strategy's performance. Run one of the commands above on your machine
— with your real model and (ideally) `--csv` broker data — for numbers that matter.

## 5. Demo forward-test checklist

- [ ] Commit + push your retrained `rl_model.zip` / `rl_scaler.save` (§1).
- [ ] `python ml_bot/backtest_offline.py --csv ...` on broker data — look at all three
      SL variants, not just "no SL" (which is closest to what training optimized for
      but is not what you can safely deploy).
- [ ] EA side: compile `XAUUSD_DayOpen_Long_Robust.mq5` (the committed `.ex5` is
      still v1.20) and A/B it per `ANALYSIS_REPORT.md` §5.
- [ ] Start `rl_trader.py` on the **demo** account; the startup guards should print
      the model/scaler check and then wait for the next D1 bar.
- [ ] Expectations: the model's profits in backtests came mostly from gold's uptrend;
      the ~$0.30/day spread toll and the $3-TP barrier race mean flat/down regimes
      will bleed. The SL bounds the damage; it does not create edge — and note it
      changes the model's risk profile from what it was trained against (§2).
- [ ] Do **not** run the h4/h8/h12 traders until retrained (they will refuse anyway).
