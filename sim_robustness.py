"""
Monte Carlo robustness check for the XAUUSD Day-Open Long strategy.

Strategy under test (fixed mandate):
  - ONE market BUY per D1 bar, at the daily open.
  - Profit target = 1% of account equity per day.
  - Position sizing = 0.01 lot per $100 of equity  (=> a $1.00 gold move ~= 1% of equity).
  - Position always closed by end of day (no cross-day hold).

What we vary: the disaster stop-loss, expressed as % of equity risked per day
(None = the original no-SL configuration that blew up in the MT5 backtest).

Market model (synthetic, clearly labelled as such):
  - Gold ~ $5000, intraday arithmetic Brownian motion (exact continuous-time
    treatment: analytic first-passage probabilities for TP-vs-SL races and
    Brownian-bridge max/min sampling for the no-SL case, so results do not
    depend on a simulation step size).
  - Base daily volatility 1.3%, with lognormal day-to-day vol mixing to create
    fat tails / shock days (calibrated loosely to 2025-26 gold, incl. days
    like 2026-02-17's -$130 move seen in the real backtest log).
  - Spread cost $0.30/oz round trip, paid on every trade.

This is NOT a price forecast. It only answers: given realistic volatility,
how does each risk-control setting change the distribution of outcomes?
"""

import numpy as np

S0 = 5000.0            # gold price level
BASE_DAILY_VOL = 0.013 # 1.3% daily vol ~= $65/day at $5000
VOL_MIX_SIGMA = 0.45   # lognormal vol-of-vol -> fat-tailed daily moves
SPREAD = 0.30          # $/oz round-trip transaction cost
START_EQUITY = 200.0   # matches the MT5 backtest deposit
N_DAYS = 125           # ~6 months of trading days (test.ini window)
N_PATHS = 5000
CONTRACT = 100.0       # oz per 1.0 lot
LOT_STEP = 0.01
CAPITAL_PER_001 = 100.0  # $ equity per 0.01 lot (mandate)
TARGET_PCT = 1.0         # daily profit target, % of equity (mandate)
MIN_MARGIN = 50.0        # margin for 0.01 lot @ 1:100 at $5000 -> ruin floor


def p_hit_up_before_down(a, b, mu, sig):
    """P(BM with drift mu, vol sig hits +a before -b), continuous time."""
    theta = 2.0 * mu / sig**2
    p0 = b / (a + b)                                   # driftless limit
    tb = np.clip(theta * b, -500.0, 500.0)
    ta = np.clip(-theta * a, -500.0, 500.0)
    with np.errstate(over="ignore", invalid="ignore"):
        num = np.expm1(tb)
        den = np.exp(tb) - np.exp(ta)
        p = np.where(np.abs(theta) * (a + b) < 1e-8, p0, num / den)
    p = np.where(np.isfinite(p), p, p0)
    return np.clip(p, 0.0, 1.0)


def run(sl_pct, drift=0.0, tp_pct=TARGET_PCT):
    """sl_pct: stop-loss as % of equity per day, or None for no stop.
    drift: expected daily log-return of gold (0 = no view; the 2025-26 gold
    bull run was roughly +0.10%/day).
    tp_pct: daily profit target as % of equity (mandate default 1.0)."""
    rng = np.random.default_rng(42)   # common random numbers across scenarios
    equity = np.full(N_PATHS, START_EQUITY)
    peak = equity.copy()
    ruined = np.zeros(N_PATHS, dtype=bool)
    for _ in range(N_DAYS):
        alive = ~ruined & (equity >= MIN_MARGIN)
        n = int(alive.sum())
        if n == 0:
            break
        eq = equity[alive]
        # fat-tailed daily vol, in $ terms
        vol = BASE_DAILY_VOL * np.exp(VOL_MIX_SIGMA * rng.standard_normal(n)
                                      - 0.5 * VOL_MIX_SIGMA**2)
        sig = S0 * vol                    # $ std-dev of the day's move
        mu = S0 * drift                   # $ drift of the day's move

        lot = np.maximum(np.floor(eq * 0.01 / CAPITAL_PER_001 / LOT_STEP)
                         * LOT_STEP, LOT_STEP)
        upd = lot * CONTRACT              # P/L in $ per $1 gold move
        a = (eq * tp_pct / 100.0) / upd + SPREAD          # TP distance ($)
        pnl = np.empty(n)

        if sl_pct is not None:
            # TP-vs-SL race decided analytically (both barriers are tiny vs
            # daily vol, so the race resolves well before end of day)
            b = np.maximum((eq * sl_pct / 100.0) / upd - SPREAD, 0.01)
            p = p_hit_up_before_down(a, b, mu, sig)
            win = rng.random(n) < p
            pnl[win] = eq[win] * tp_pct / 100.0
            pnl[~win] = -eq[~win] * sl_pct / 100.0
        else:
            # no SL: sample the day's close, then the path max/min via the
            # exact Brownian-bridge extreme distributions
            c = mu + sig * rng.standard_normal(n)
            u = rng.random(n)
            mmax = 0.5 * (c + np.sqrt(c**2 - 2.0 * sig**2 * np.log(u)))
            v = rng.random(n)
            mmin = 0.5 * (c - np.sqrt(c**2 - 2.0 * sig**2 * np.log(v)))
            win = mmax >= a
            pnl[win] = eq[win] * tp_pct / 100.0
            pnl[~win] = (c[~win] - SPREAD) * upd[~win]
            # margin stop-out: unrealized loss reached ~100% of equity
            blown = (~win) & (eq + (mmin - SPREAD) * upd <= 0.0)
            pnl[blown] = -eq[blown]

        eq_new = np.maximum(eq + pnl, 0.0)
        equity[alive] = eq_new
        ruined[alive] |= eq_new < MIN_MARGIN
        peak[alive] = np.maximum(peak[alive], eq_new)

    dd = 1.0 - equity / peak
    return {
        "sl": "none" if sl_pct is None else f"{sl_pct:.0f}%",
        "ruin%": 100 * np.mean(equity < MIN_MARGIN),
        "median_final": float(np.median(equity)),
        "p5_final": float(np.percentile(equity, 5)),
        "p95_final": float(np.percentile(equity, 95)),
        "median_maxDD%": 100 * float(np.median(dd)),
    }


if __name__ == "__main__":
    print(f"start=${START_EQUITY:.0f}  days={N_DAYS}  paths={N_PATHS}  "
          f"target={TARGET_PCT}%/day  sizing=0.01lot/${CAPITAL_PER_001:.0f}  "
          f"spread=${SPREAD:.2f}")
    for drift, label in ((0.0, "ZERO DRIFT (no view on gold)"),
                         (0.001, "BULL DRIFT +0.10%/day (2025-26-like gold)")):
        print(f"\n== {label} ==")
        hdr = (f"{'SL/day':>8} {'ruin %':>8} {'median $':>10} {'p5 $':>8} "
               f"{'p95 $':>10} {'med DD %':>9}")
        print(hdr)
        print("-" * len(hdr))
        for sl in (None, 30.0, 20.0, 10.0, 5.0, 3.0, 2.0):
            r = run(sl, drift)
            print(f"{r['sl']:>8} {r['ruin%']:>8.1f} {r['median_final']:>10.2f} "
                  f"{r['p5_final']:>8.2f} {r['p95_final']:>10.2f} "
                  f"{r['median_maxDD%']:>9.1f}")
    print("\nNote: with a ~$1 TP on an asset moving ~$65/day, the TP/SL race "
          "resolves in minutes;\nthe trade captures almost no drift but pays "
          "the full spread daily (~-0.3%/day drag).")
