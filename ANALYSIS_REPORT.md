# XAUUSD Day-Open Long — Backtest Analysis & Robustness Fixes

**Scope.** Analysis of the MT5 strategy-tester report in `log.txt` (EA `XAUUSD_DayOpen_Long.mq5`
v1.20, XAUUSD D1, 2026.01.01–2026.06.27 window per `test.ini`), plus the RL-agent equity curve in
`backtest_result.png`. The mandate is kept unchanged: **one trade per daily bar, profit target of
1% of equity per day, sizing of 0.01 lot per $100 of equity**. Deliverables:

| File | What it is |
|---|---|
| `XAUUSD_DayOpen_Long_Robust.mq5` | v2.10 of the EA with the robustness fixes; TP/SL optimizable (needs compiling in MetaEditor) |
| `sim_robustness.py` | Monte Carlo experiment quantifying each risk-control choice |
| `optimize_tp_sweep.py` | Monte Carlo grid sweep of the daily TP% x SL% (TP not fixed at 1%) |
| `optimize_tp.ini` | MT5 strategy-tester preset that optimizes `InpDailyProfitPct` x `InpMaxDailyLossPct` on real data |
| `reports/ReportOptimizer2101170120.xml` | actual MT5 optimizer output (180 passes, 2026 H1 tick data, IUXMarkets demo) |
| `analyze_optimizer_report.py` | parses an optimizer XML, prints the TP×SL grid, and ranks cells by worst-neighbor profit |
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

## 4. What v2.10 (`XAUUSD_DayOpen_Long_Robust.mq5`) changes

Mandate-preserving: still one BUY per D1 bar, still 1%/day equity target, still 0.01 lot/$100,
still EOD + Friday close, same trend filters, same `xau_filter.cfg` batch-test mechanism.

| # | Failure observed | Fix in v2.10 | Input (default) |
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

## 6. TP optimization — the target does not have to be 1%

The daily target is now a first-class optimizable parameter:

- **In the simulator:** `python3 optimize_tp_sweep.py` sweeps TP ∈ {0.25…12}% × SL ∈ {2…10}%
  under three gold-drift regimes (zero / +0.1%/day / +0.3%/day).
- **In MT5 on real data:** run the tester with `optimize_tp.ini` — it optimizes
  `InpDailyProfitPct` (0.25→5.0 step 0.25) × `InpMaxDailyLossPct` (2→10 step 1) using the
  **Custom max** criterion. `OnTester` returns `netProfit × (100 − maxDD%) / 100`, so a pass that
  "won" through near-ruin drawdowns cannot top the ranking. Each pass writes its own
  `results_<tag>_tp<TP>_sl<SL>.csv` (files no longer overwrite each other), and
  `tppct=` / `slpct=` keys in `xau_filter.cfg` allow scripted batch runs.

What the sweep shows (median final $ from $200 over 125 days, ruin % in parentheses):

| TP% \ SL% | 2% | 5% | 10% |
|---|---|---|---|
| 0.25 | 158–162 (0.0) | 160 (0.0) | 159 (0.1–0.2) |
| 1.00 | 158–163 (0.0) | 150–160 (0.0) | 155–174 (1–2.5) |
| 3.00 | 150–158 (0.0) | 141–166 (1.5–3.4) | 123–184 (7–17) |
| 8.00 | 136–165 (0.7–1.5) | 120–200 (8–19) | 77–230 (19–42) |

(ranges span the zero-drift → strong-bull scenarios)

Three conclusions:

1. **The TP knob shapes risk, not expectancy.** With 0.01 lot/$100 sizing the TP/SL race ends
   within minutes, so drift capture (≈ drift × TP×SL / vol²) never covers the ~0.3%/day spread.
   No cell reaches a median above the $200 start under zero or normal bull drift.
2. **The sane region is TP 0.25–3% with SL 2–5%**: ruin ≈ 0%, best medians, lowest variance.
   The mandated 1%/5% sits comfortably inside it; there is no strong reason to move within the
   region unless real-data optimization says otherwise.
3. **Big targets are a trap.** TP 8–12% with SL 10% shows the best medians *only* in a relentless
   uptrend — and with 19–42% ruin odds. That corner is statistically the same bet that destroyed
   v1.20. If the MT5 optimizer crowns a corner cell on one historical window, check its drawdown
   column before believing it.

### 6.1 Real-data optimization results (2026 H1, 180 passes)

The `optimize_tp.ini` sweep was run on real tick data (XAUUSD Daily, 2026.01.01–2026.06.27,
$200 deposit, 1:100, IUXMarkets demo). Raw output is committed as
`reports/ReportOptimizer2101170120.xml`; reproduce the analysis with
`python3 analyze_optimizer_report.py reports/ReportOptimizer2101170120.xml`. Headlines:

- **MT5's raw winner is an overfit spike.** TP 2.25 / SL 10 tops the ranking at **+$114.61**, but
  **every one of its grid neighbors loses ~$50–60** (all halted by the circuit breaker). One
  quarter-point of TP separates +57% from −30% — that is luck, not signal. Do not deploy it.
- **The genuine plateau is TP 0.75–1.5% × SL 7–10%**: a contiguous block where every cell is
  profitable (+$23…+$83). Ranked by worst-neighbor profit, the deployable picks are
  **TP 1.0 / SL 7** (+$78.6 ≈ +39%, DD 10.9%, PF 2.39, Sharpe 4.79, 62 trades) and
  **TP 1.25 / SL 10** (worst neighbor still +$44). The mandated 1%/day target sits in the middle
  of the plateau — the optimization *confirms* it rather than replacing it.
- **Tight stops lose on real data.** The SL 2–3 columns are almost uniformly negative: gold dips
  $2–5 intraday before resuming, so a 2–3%-of-equity stop converts normal noise into realized
  losses. This is where the synthetic sim (which favored tight stops for survival) and real
  trending data disagree → **EA default `InpMaxDailyLossPct` raised from 5.0 to 7.0** (v2.11).
  Note SL 7–10% of equity per day is still a meaningful risk appetite; the circuit breaker is the
  backstop behind it.
- **The circuit breaker did its job.** 52 of 180 configurations were halted after 8–10 trades at
  roughly −30%; the worst cell in the whole grid lost $59.98. The same configurations under
  v1.20 rules would have ridden to near −100%.
- Caveat: this is **one six-month window** containing both the February crash and a strong
  uptrend. The plateau pick should be re-validated on a second window (e.g. walk-forward on
  2026 H2 as data accrues) before any live use.

### 6.2 Cross-check at $1,000 deposit + anatomy of the top configuration

A second full sweep at a $1,000 deposit (`reports/ReportOptimizer2101171557_Deposit1000.xml`)
independently reproduces the plateau, one notch tighter: **TP 1.0–1.5% × SL 8–10 is solidly
positive in both runs**, and this time MT5's raw winner — **TP 1.25 / SL 10, +$637.36 (+64%)** —
sits *on* the plateau (worst grid neighbor still +$255), so it is legitimate rather than a spike.
The larger deposit also scales better because lot rounding is finer (0.10 lots vs 0.02–0.03 at
$200). Since the two runs agree that SL 7 is the plateau's weak edge and SL 8–10 its core, the
EA default `InpMaxDailyLossPct` moves 7.0 → **9.0** (v2.12); TP default stays at the mandated 1.0
with TP 1.25 as the optimizer's slightly stronger alternative.

The single-run report of the top cell (`reports/ReportTester2101171557_Top_Deposit1000.xlsx`,
99% history quality, 685k ticks) shows exactly the anatomy predicted in section 3:

| Metric | Value | Reading |
|---|---|---|
| Net profit | **+$637.36 (+64%)** in 6 months | plateau cell, both deposits agree |
| Trades | 62 long, **96.8% win rate** | ADX filter traded ~half of all days |
| Avg holding time | **4 min 18 s** (min 20 s) | confirms the "TP/SL race resolves in minutes" math |
| Avg win / avg loss | +$15.38 / **−$117.21** | negative skew is still the engine — now capped |
| Max equity DD | 13.66% | vs 100% wipe-out of v1.20 |
| Profit factor / Sharpe | 3.23 / 6.06 | LR correlation 0.91 — smooth equity line |

The two losing trades (−$119, −$115) are the February-crash days that destroyed v1.20; here the
10%-of-equity SL converted "account gone" into "one bad day". The structural caveats stand: the
win rate is mechanical (SL 10 vs TP 1.25 wins ~89% of races by geometry alone), profitability
still requires gold's uptrend, and a Z-score of 1.47 on 62 trades is thin evidence — walk-forward
on H2 2026 before risking real money.

## 7. Honest limitations & recommendations

- **Robust ≠ profitable.** v2.10 converts "certain eventual ruin" into "bounded drawdowns with a
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
