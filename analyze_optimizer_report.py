"""
Analyze an MT5 strategy-tester OPTIMIZER report (XML / Excel-XML export) for
the XAUUSD_DayOpen_Long_Robust TP x SL sweep.

Usage:  python3 analyze_optimizer_report.py <ReportOptimizer*.xml>

What it does beyond MT5's own ranking:
  - prints the full TP x SL profit grid, marking cells where the equity
    circuit breaker halted trading early (trade count collapsed);
  - ranks cells by WORST-NEIGHBOR profit instead of own profit. A parameter
    set whose 8 grid neighbors also make money sits on a plateau and has a
    chance of generalizing; a winner surrounded by losers is an overfit
    spike and must not be deployed.
"""

import re
import sys
import xml.etree.ElementTree as ET

TP_COL = "InpDailyProfitPct"
SL_COL = "InpMaxDailyLossPct"
BREAKER_TRADES = 20   # below this, the run was halted by the circuit breaker


def load(path):
    raw = open(path, encoding="utf-8").read()
    raw = re.sub(r'xmlns(:\w+)?="[^"]*"', "", raw)
    raw = re.sub(r"<(/?)(\w+):", r"<\1", raw)
    raw = raw.replace("ss:", "")
    root = ET.fromstring(raw)
    rows = []
    for row in root.iter("Row"):
        cells = []
        for c in row.findall("Cell"):
            d = c.find("Data")
            cells.append(d.text if d is not None else "")
        rows.append(cells)
    header, out = rows[0], []
    for r in rows[1:]:
        if len(r) != len(header):
            continue
        try:
            out.append({k: float(v) for k, v in zip(header, r)})
        except (TypeError, ValueError):
            pass
    return out


def main(path):
    data = load(path)
    grid = {(d[TP_COL], d[SL_COL]): d for d in data}
    tps = sorted({k[0] for k in grid})
    sls = sorted({k[1] for k in grid})
    print(f"{len(data)} passes, TP {tps[0]}..{tps[-1]} x SL {sls[0]}..{sls[-1]}")

    print("\nPROFIT grid ($). rows=TP%, cols=SL%. * = circuit breaker tripped")
    print(f"{'TP/SL':>6}|" + "".join(f"{sl:>8.0f}" for sl in sls))
    for tp in tps:
        row = ""
        for sl in sls:
            d = grid.get((tp, sl))
            if d is None:
                row += f"{'-':>8}"
                continue
            mark = "*" if d["Trades"] < BREAKER_TRADES else " "
            row += f"{d['Profit']:>7.0f}{mark}"
        print(f"{tp:>6.2f}|{row}")

    def neighbors(tp, sl):
        ti, si = tps.index(tp), sls.index(sl)
        out = []
        for dt in (-1, 0, 1):
            for ds in (-1, 0, 1):
                t2, s2 = ti + dt, si + ds
                if 0 <= t2 < len(tps) and 0 <= s2 < len(sls):
                    d = grid.get((tps[t2], sls[s2]))
                    if d is not None:
                        out.append(d)
        return out

    scored = []
    for tp in tps:
        for sl in sls:
            if (tp, sl) not in grid:
                continue
            ns = [n["Profit"] for n in neighbors(tp, sl)]
            scored.append((min(ns), sum(ns) / len(ns), tp, sl, grid[(tp, sl)]))
    scored.sort(key=lambda x: (-x[0], -x[1]))

    print("\nTop 10 by WORST-NEIGHBOR profit (deploy from this list, not MT5's):")
    print(f"{'TP%':>5} {'SL%':>4} {'ownP$':>7} {'minNbr':>7} {'avgNbr':>7} "
          f"{'DD%':>6} {'PF':>5} {'Sharpe':>6} {'Trades':>6}")
    for mn, avg, tp, sl, d in scored[:10]:
        print(f"{tp:>5.2f} {sl:>4.0f} {d['Profit']:>7.1f} {mn:>7.1f} {avg:>7.1f} "
              f"{d['Equity DD %']:>6.1f} {d['Profit Factor']:>5.2f} "
              f"{d['Sharpe Ratio']:>6.2f} {d['Trades']:>6.0f}")

    raw_best = max(data, key=lambda d: d["Custom"])
    ns = neighbors(raw_best[TP_COL], raw_best[SL_COL])
    others = [n["Profit"] for n in ns if n is not raw_best]
    print(f"\nMT5's raw winner: TP {raw_best[TP_COL]:.2f} SL {raw_best[SL_COL]:.0f} "
          f"profit ${raw_best['Profit']:.2f}")
    if others and max(others) < 0:
        print("  -> ALL its neighbors lose money: overfit spike, do NOT deploy.")
    else:
        print(f"  -> neighbor profits: min {min(others):.1f} / max {max(others):.1f}")

    pos = sum(1 for d in data if d["Profit"] > 0)
    trip = sum(1 for d in data if d["Trades"] < BREAKER_TRADES)
    worst = min(data, key=lambda d: d["Profit"])
    print(f"\npositive cells {pos}/{len(data)} | breaker-halted {trip}/{len(data)} "
          f"| worst cell ${worst['Profit']:.2f} "
          f"(v1.20 equivalent would have been near -100%)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/ReportOptimizer2101170120.xml")
