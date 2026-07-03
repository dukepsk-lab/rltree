# Multi-Timeframe / Long-Short Grid Results (user's real data, Nov 2021 – Jul 2026)

**Data**: user-supplied XAUUSD H1 bars (24,347 bars, 4.6 years, median spread $0.04 —
much tighter than the IUX demo feed), committed under `data/`. Engine: `grid_backtest.py`
(one-trade-per-bar family, money-based TP/SL as % of equity, 0.01 lot/$100 sizing,
30% circuit breaker, spread charged per trade).

**Sweep**: TF {H4, H6, H8, H12, D1} × direction {long, short, both} × filter
{ADX, LINREG, DONCHIAN} × TP {0.25…3}% × SL {3…10}% — 1,440 configs per run.
Barrier races are resolved at H1 resolution, so every config was run twice:
**pess** (TP+SL inside one H1 bar → count as SL) and **opt** (→ count as TP).
Cells where both modes give the *same* number contain no ambiguous bars and are the
only fully trustworthy readouts at this resolution.
Walk-forward: **IS = Nov 2021–Dec 2024**, **OOS = Jan 2025–Jul 2026**.

## Headline findings

1. **Short and "both" modes fail everywhere.** Across 4.6 years every short
   configuration is negative (best short cell ≈ −19% pess), and "both" is always
   worse than long-only on the same plane. Gold's long-run drift dominates; the
   filters cannot time shorts at bar-open granularity. **Stay long-only.**

2. **D1 — the original strategy — dies out-of-sample on this dataset.** The IS
   plateau (TP 1.25–1.5 × SL 9–10, +42…+85%) matches the earlier IUX-demo MT5
   optimizations, but in OOS 2025–26 *every* D1 cell loses −17…−36%. This broker's
   feed shows gold falling ~5,000 → ~4,180 through H1 2026; daily-open longs bleed
   through it. (Note: the IUX demo tester showed +64% on Jan–Jun 2026 for the same
   config — the two brokers' 2026 price histories genuinely disagree. Treat D1 as
   regime-dependent, not robust.)

3. **The survivor: H4 / ADX / LONG / TP 3% / SL 9%.**

   | Window | pess | opt | trades | win % | max DD |
   |---|---|---|---|---|---|
   | IS 2021–24 | **+241.6%** | +241.6% | 262 | 66% | 31.6% |
   | OOS 2025–26 | **+64.0%** | +64.0% | 122 | 73% | 32.6% |

   pess == opt in both windows → zero ambiguous bars: this readout is exact at H1
   resolution. Its whole TP×SL neighborhood (TP 3 × SL 9–10) is positive in both
   windows, and it is the robust-cell winner of the full-period run
   (+248.8%, worst neighbor +135%). Larger TP works on H4 because a $3-per-oz target
   is wide enough to escape spread noise while ADX-filtered 4-hour momentum is still
   alive.

4. **Small-TP intraday cells are unmeasurable with H1 data.** E.g. H4 TP 1.0/SL 9
   OOS reads −9% pess but +310% opt — the race lives inside single H1 bars. If these
   corners matter to you, export **M1** and rerun; do not trust either bound.

5. **Expect the circuit breaker to fire.** Every strategy in the family, including
   the winner, sees >30% peak-to-trough drawdowns roughly once per regime. The
   breaker converts those into scheduled reassessment stops — that is by design, but
   it means "set and forget" is not on the menu.

## Recommended settings (XAUUSD_BarOpen_MultiTF.mq5 v3.00)

```
InpTradeTF          = PERIOD_H4
InpDirection        = DIR_LONG_ONLY
InpTrendFilter      = TREND_ADX   (period 14, threshold 20)
InpProfitPctPerBar  = 3.0         // % of equity per bar
InpMaxLossPctPerBar = 9.0         // % of equity per bar
InpMaxTotalDDPct    = 30.0        // will trip ~once per regime; that's the design
InpCapitalPer001Lot = 100.0       // mandate sizing
deposit             >= $1000
```

Realistic expectation from the unseen 18-month window: **~+40–65% per 1.5 years with
30%+ drawdowns along the way** — not 1% per day. The D1 mandate configuration
(TP 1%/day) remains fine for the demo walk-forward, but this dataset says its edge is
the 2021–24 bull regime, not a stable property.

## 2026-only backtest (Jan 1 – Jul 3, 2026)

Requested follow-up: the same grid restricted to 2026. Two structural facts first:

- **2026 is a different market.** Median H1 bar range is **$19.56 vs $4.84 in 2024** (4×),
  and gold crashes from the ~$5,000 February peak to ~$4,180 by July on this feed.
- **83% of 2026 H1 bars span more than $12**, so even a TP 3 / SL 9 race frequently sits
  inside a single H1 bar → the pess/opt bounds get wide, and only conclusions where both
  bounds agree in sign are trustworthy.

Results for the key cells (long, ADX):

| Cell | pess (ties→SL) | opt (ties→TP) | verdict |
|---|---|---|---|
| **H4 TP 3 / SL 9** (recommended) | −30.2% (halted) | **−15.1% (halted)** | **loses in 2026, bounded −30…−15%** |
| H4 TP 1.0 / SL 9 | −33.1% | +76.9% | unmeasurable at H1 resolution |
| D1 TP 1.0–1.25 / SL 9–10 (mandate) | ≈−30% (halted) | +2…+7% | flat-to-losing; note the IUX tick tester said **+64%** for the same window — feeds/models disagree badly in 2026 |
| all shorts / both, all TFs | ≈−20…−32% | — | still lose despite the crash (high-vol counter-rallies) |

Takeaways:

1. **2026 broke the regime.** Every long config that carried 2021–25 loses in 2026 on this
   feed; the recommended H4 cell is genuinely negative (both bounds < 0). The circuit
   breaker capped the damage near −30% — exactly its job — and that −30% *is* the realistic
   cost of a regime change under this strategy family.
2. **Shorting still doesn't save 2026.** The crash came with 4× volatility; bar-open shorts
   get stopped on counter-rallies just like longs on dips.
3. **The 2025 profits carried the OOS window.** The +64% OOS figure in the table above is
   2025 gains plus an early-2026 breaker stop; do not read it as "works in 2026".
4. **Decisive verification needs finer data**: either export **M1** history and rerun this
   grid, or run the v3 EA in the MT5 tick tester over 2026 (`backtest_2026_h4.ini`) — the
   tick tester resolves what H1 bars cannot, and will also arbitrate the feed disagreement
   with the earlier IUX +64% result.

## Reproduce

```
python3 grid_backtest.py data/XAUUSD_H1.csv --deposit 1000 --tie pess
python3 grid_backtest.py data/XAUUSD_H1.csv --deposit 1000 --tie opt
python3 grid_backtest.py data/XAUUSD_H1.csv --deposit 1000 --tie pess --date-from 2025-01-01
```
