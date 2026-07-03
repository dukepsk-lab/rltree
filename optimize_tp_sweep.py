"""
TP x SL grid sweep for the XAUUSD Day-Open Long strategy.

Uses the same continuous-time Monte Carlo engine as sim_robustness.py, but
sweeps the daily profit target (tp_pct, % of equity) instead of pinning it at
the 1% mandate. Everything else stays on the mandate: one trade per D1 bar,
0.01 lot per $100 equity, EOD close.

Why the TP size matters at all (there is no free parameter here):
  - With barriers tiny vs gold's ~$65/day range, a day resolves to either
    +TP% or -SL% with P(win) ~= SL/(TP+SL). Pre-cost expectancy is ~zero
    (martingale); the spread (~0.3% of equity/day) is the constant drag.
  - Drift capture ~= drift * expected-hold-time ~= mu * TP*SL / sigma^2:
    LARGER barriers hold the trade longer and capture more of a gold
    uptrend, but larger SL also means deeper drawdowns. The sweep shows
    where that trade-off lands.

Ranking metric: median final equity (5,000 paths, 125 days, $200 start),
with ruin probability shown alongside - a high median with meaningful ruin
risk is not an acceptable optimum.
"""

import numpy as np
import sim_robustness as sim

TP_GRID = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]   # % of equity per day
SL_GRID = [2.0, 3.0, 5.0, 10.0]                        # % of equity per day
DRIFTS = ((0.0, "ZERO DRIFT (no view on gold)"),
          (0.001, "BULL DRIFT +0.10%/day (2025-26-like gold)"),
          (0.003, "STRONG BULL +0.30%/day (relentless uptrend)"))

if __name__ == "__main__":
    print(f"start=${sim.START_EQUITY:.0f}  days={sim.N_DAYS}  "
          f"paths={sim.N_PATHS}  spread=${sim.SPREAD:.2f}  "
          f"sizing=0.01lot/${sim.CAPITAL_PER_001:.0f}")
    print("cell = median final $ (ruin %) ; start capital $200\n")

    for drift, label in DRIFTS:
        print(f"== {label} ==")
        corner = "TP% / SL%"
        header = f"{corner:>10} |" + "".join(f"{sl:>16.0f}" for sl in SL_GRID)
        print(header)
        print("-" * len(header))
        best = (None, -1.0)
        for tp in TP_GRID:
            cells = []
            for sl in SL_GRID:
                r = sim.run(sl, drift, tp_pct=tp)
                cells.append(f"{r['median_final']:>8.0f} ({r['ruin%']:>4.1f})")
                # optimum must be ruin-free to count
                if r["ruin%"] < 0.5 and r["median_final"] > best[1]:
                    best = ((tp, sl), r["median_final"])
            print(f"{tp:>10.2f} |" + "".join(f"{c:>16}" for c in cells))
        if best[0] is not None:
            tp, sl = best[0]
            print(f"best ruin-free cell: TP={tp}% SL={sl}% "
                  f"-> median ${best[1]:.0f}\n")
        else:
            print("no ruin-free cell in grid\n")

    print("Reading guide:")
    print(" - No cell reaches median > $200 under zero or normal bull drift:")
    print("   at 0.01 lot/$100 the TP/SL race is over in minutes, so drift")
    print("   capture (~drift * TP*SL / vol^2) never covers the ~0.3%/day")
    print("   spread. The TP knob shapes RISK, not expectancy.")
    print(" - Small targets (TP 0.25-1%, SL 2-5%) are the survival-optimal")
    print("   region: ruin ~0%, best medians, lowest variance.")
    print(" - Big targets only turn positive in a relentless uptrend, and")
    print("   with double-digit ruin odds - exactly the trap the original")
    print("   no-SL EA fell into. Do not optimize into that corner.")
    print(" - Cross-check any chosen cell on real tick data with the MT5")
    print("   optimizer preset in optimize_tp.ini (criterion = DD-penalized")
    print("   profit written by OnTester).")
