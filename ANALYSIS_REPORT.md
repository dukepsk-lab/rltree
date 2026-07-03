# XAUUSD Day-Open Long — Backtest Analysis & Robustness Fixes

**Scope.** Analysis of the MT5 strategy-tester report in `log.txt` (EA `XAUUSD_DayOpen_Long.mq5`
v1.20, XAUUSD D1, 2026.01.01–2026.06.27 window per `test.ini`), plus the RL-agent equity curve in
`backtest_result.png`. The mandate is kept unchanged: **one trade per daily bar, profit target of
1% of equity per day, sizing of 0.01 lot per $100 of equity**. Deliverables:

| File | What it is |
|---|---|
| `XAUUSD_DayOpen_Long_Robust.mq5` | v2.00 of the EA with the robustness fixes (needs compiling in MetaEditor) |
| `sim_robustness.py` | Monte Carlo experiment quantifying each risk-control choice |
| `ANALYSIS_REPORT.md` | this report |

---

## 1. What the report actually shows

Two tester runs are in `log.txt`, and both die the same way:

| Run | Deposit | Outcome |
|---|---|---|
| 1 | $10,000 (lots 1.04+) | **Final balance −$6,708.55** (negative!) after margin stop-out |
| 2 | $200 (lots 0.02–0.03) | **Final balance $1.05** (−99.5%) after margin stop-out |

The pattern in run 2: **78 consecutive wins** at the fixed $2 profit target (equity $200 → $392.88,
~+96% in six weeks), then on 2026.02.17 gold fell from $4,989.93 to $4,859.32 — about −$130 —
while the EA held a 0.03-lot long **with no stop loss**. The tester reports
`stop out occurred on 27% of testing interval` and the account was effectively destroyed in one day.
The RL agent curve in `backtest_result.png` shows the same signature: gains, a flat dead zone, then
a terminal collapse below zero.

This is the classic negative-skew profile: a high win rate built from many small capped wins and an
uncapped left tail. The 98% win rate was not edge — it was unpriced tail risk plus a rising gold
market.

Secondary finding: **6,104 log lines of failed `market closed` buy orders** — the EA retried a
market order every cooldown tick through the Sunday-night close (e.g. 2026.01.04 23:03 onward).
Harmless in the tester, but on a live account this is a retry storm that can get an account
rate-limited or flagged.

## 2. Root causes, ranked

1. **No stop loss at ~50:1 effective exposure.** At the mandated sizing (0.01 lot = 1 oz per $100),
   a $1 gold move = 1% of equity. Gold's daily range in 2026 is routinely $50–150, i.e. a normal bad
   day is −50% to −150% of equity if held to the close. Ruin was not a risk; it was a schedule.
2. **Exits existed only in `OnTick`.** No broker-side TP/SL, so a disconnect/VPS crash would leave a
   naked leveraged position.
3. **Fixed $2 price target ≠ the mandate.** $2 is 2% of a $100-per-0.01-lot account regardless of
   equity only by accident of the sizing formula; after lot rounding (e.g. equity $250 → lot 0.02)
   the realized percentage drifts. The mandate ("1% profit per day") is a **money** target.
4. **Entry at the midnight rollover.** The daily open is the worst spread of the session. With a ~$1
   target, a $0.40–1.00 rollover spread is 40–100% of the target.
5. **No circuit breaker / attempt caps** — the retry storm, and nothing to stop the EA re-leveraging
   a shrinking account day after day.

## 3. The math of the mandate (what cannot be fixed by code)

With 0.01 lot per $100 and a 1%-of-equity target, the TP sits ~$1 above entry and any daily SL of
X% sits ~$X below. Gold moves ~$65/day (1.3% vol at $5,000), so the TP/SL race resolves **within
minutes** of entry. Two consequences:

- The trade captures almost none of gold's trend/drift (it's in the market for minutes), but pays
  the **full spread every day**: ~$0.30/oz round trip ≈ **−0.3% of equity per day** against a +1%
  target. That is a structural cost drag no filter can remove.
- Win probability of the barrier race is ≈ `SL/(TP+SL)` — e.g. a 5% SL wins ~83% of days. The high
  win rate is mechanical, not predictive.

**Monte Carlo results** (`sim_robustness.py`: $200 start, 125 trading days, 5,000 paths, fat-tailed
1.3% daily vol, $0.30 spread; exact continuous-time barrier math, so no step-size artifacts):

| SL (% equity/day) | Ruin % (zero drift) | Median final $ | Ruin % (bull +0.1%/day) | Median final $ |
|---|---|---|---|---|
| none (v1.20)      | **52.2** | 46  | **46.2** | 99  |
| 30%               | 24.8     | 111 | 18.7     | 160 |
| 10%               | 2.5      | 155 | 2.1      | 155 |
| **5% (default)**  | **0.0**  | 150 | **0.0**  | 160 |
| 2%                | 0.0      | 158 | 0.0      | 158 |

Interpretation:

- **The no-SL configuration ruins roughly half of all accounts within six months even in a bull
  market** — fully consistent with what `log.txt` shows happening once in reality.
- **A daily disaster stop of ~5% of equity eliminates ruin** and cuts median max drawdown from ~80%
  to ~30%.
- No SL setting makes the median outcome profitable: the ~0.2–0.3%/day spread drag dominates. The
  strategy as mandated is a *risk-shaping* system, not an *edge* system; its historical profits came
  from gold rising, and robustness work can only make it survive the days gold doesn't.

## 4. What v2.00 (`XAUUSD_DayOpen_Long_Robust.mq5`) changes

Mandate-preserving: still one BUY per D1 bar, still 1%/day equity target, still 0.01 lot/$100,
still EOD + Friday close, same trend filters, same `xau_filter.cfg` batch-test mechanism.

| # | Failure observed | Fix in v2.00 | Input (default) |
|---|---|---|---|
| 1 | −$130 day wiped the account | Broker-side **disaster SL** at entry, sized as % of entry equity; money-based emergency close in `OnTick` backs it up against gaps | `InpMaxDailyLossPct` (5.0) |
| 2 | Exits lived only in `OnTick` | Broker-side **TP and SL attached to the order** (survive disconnects), respecting `SYMBOL_TRADE_STOPS_LEVEL` | — |
| 3 | Fixed $2 target | **1% of entry equity in money**, net of swap+commission, converted to price via tick value (correct under lot rounding) | `InpDailyProfitPct` (1.0) |
| 4 | Entry at rollover spread | **Entry delay** after the daily open + **max-spread gate** | `InpEntryDelayMinutes` (5), `InpMaxSpreadPoints` (80) |
| 5 | 6,104 failed-order retries | **Attempt cap per day** + **entry window** (no late entries with no time left to reach the target) | `InpMaxEntryAttempts` (20), `InpEntryWindowHours` (12) |
| 6 | Account re-leveraged into the ground | **Equity circuit breaker**: entries halt when equity is X% below its high-water mark (peak persisted in a terminal global variable across restarts) | `InpMaxTotalDDPct` (30) |
| 7 | Margin-reject skipped whole days | Lot **shrinks to fit free margin** (with buffer) instead of rejecting; skips the day only if the minimum lot doesn't fit | `InpMarginUseFraction` (0.8) |
| 8 | Sunday micro-bar trades | Optional **Sunday-bar skip** | `InpSkipSundayBar` (true) |
| 9 | Trend filter defaulted to "UP" on the attach day | Filter evaluated on the **first tick of every day**, including attach day | — |

New default magic number (`20260703`) so v1 and v2 can run side by side in comparison backtests
without touching each other's positions.

## 5. How to validate

1. Compile `XAUUSD_DayOpen_Long_Robust.mq5` in MetaEditor (the committed `.ex5` in this repo is the
   old v1.20 binary — it must be rebuilt).
2. Re-run the same `test.ini` window with `Expert=XAUUSD_DayOpen_Long_Robust.ex5`. Compare
   `results_<tag>.csv` (now includes a `sharpe` column) against the v1.20 runs. The A/B expectation
   from the Monte Carlo: v2 shows a *lower* peak return over Jan–Feb but does **not** hand the
   account back on 2026.02.17 — the worst day is capped near −5%.
3. `python3 sim_robustness.py` (needs `numpy`) reproduces the risk table above; edit the constants
   at the top to test other spreads/vol regimes/SL levels.

## 6. Honest limitations & recommendations

- **Robust ≠ profitable.** v2.00 converts "certain eventual ruin" into "bounded drawdowns with a
  ~0.2–0.3%/day cost headwind". Expected 1%/day compounding is not achievable with these
  parameters — that would be ~1,100%/year with no drawdown, which no spread-paying barrier race
  delivers.
- The levers that actually improve expectancy, if the mandate is ever relaxed: trade **fewer** days
  (a strict trend filter cuts the daily spread toll — keep `TREND_ADX` or `DONCHIAN` on), use a
  **wider target** relative to spread, or reduce the sizing so daily vol is survivable and the
  position can hold long enough to capture trend.
- Keep `InpMaxDailyLossPct` in the 3–7% range; `0` (off) reproduces v1.20 behavior and its ~50%
  six-month ruin probability.
- Test on a demo account first; the risk warning from v1.20 still applies to leveraged gold trading
  in general.
