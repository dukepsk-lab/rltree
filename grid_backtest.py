"""
Full grid backtester for the "one trade per bar" XAUUSD strategy family
(XAUUSD_BarOpen_MultiTF.mq5) on user-supplied fine-grained OHLC data.

Sweeps: timeframe (H4/H8/H12/D1) x direction (long/short/both) x trend filter
(ADX/LINREG/DONCHIAN/NONE) x TP% x SL%, ranks TP x SL cells by worst-neighbor
profit (anti-overfit), and prints the best settings.

DATA FORMAT (M1 or M5 bars; finer = more accurate barrier resolution):
  MT5 export (Symbols -> Bars -> Export, or Charts -> Save As) is accepted
  directly: tab- or comma-separated with headers like
    <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
  Minimum columns: date+time (or datetime), open, high, low, close.
  SPREAD column (points) is used if present; otherwise pass --spread-points.

USAGE:
  python3 grid_backtest.py XAUUSD_M1.csv --deposit 1000 --spread-points 30

MODEL NOTES (kept consistent with the MQL5 EA):
  - entry at first fine bar's open of each TF bar, ask = open + spread/2;
  - money-based TP/SL as % of entry equity; with the mandate sizing
    (0.01 lot / $100) a TP of X% == a $X price move, so barrier distances
    are equity-independent (lot rounding ignored - validated within a few
    $ against the MT5 tester reports in reports/);
  - barrier race resolved on fine bars; if TP and SL are both inside the
    same fine bar the PESSIMISTIC assumption (SL first) is used;
  - end-of-bar close at the TF bar's last close, minus half-spread;
  - full spread charged per round trip; 30% equity circuit breaker;
    ruin floor at $50 (min-lot margin).
"""

import argparse
import csv
import sys
from datetime import datetime

import numpy as np

TF_HOURS = {"H4": 4, "H6": 6, "H8": 8, "H12": 12, "D1": 24}
TP_GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
SL_GRID = [3.0, 5.0, 7.0, 9.0, 10.0]
BREAKER_DD = 30.0
RUIN_FLOOR = 50.0
POINT = 0.01                       # XAUUSD point size


# ---------------------------------------------------------------- data
def load_fine_bars(path, spread_points):
    """Return dict of numpy arrays: ts (epoch s), o, h, l, c, spread ($)."""
    with open(path, encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = "\t" if "\t" in sample.splitlines()[0] else ","
        rdr = csv.reader(f, delimiter=delim)
        head = [x.strip().lstrip("<").rstrip(">").upper() for x in next(rdr)]

        def col(*names):
            for n in names:
                if n in head:
                    return head.index(n)
            return None

        i_date, i_time = col("DATE"), col("TIME")
        i_dt = col("DATETIME", "TIMESTAMP")
        i_o, i_h = col("OPEN"), col("HIGH")
        i_l, i_c = col("LOW"), col("CLOSE")
        i_sp = col("SPREAD")
        if i_o is None or i_c is None or (i_date is None and i_dt is None):
            sys.exit(f"unrecognized columns: {head}")

        ts, o, h, l, c, sp = [], [], [], [], [], []
        for row in rdr:
            if len(row) < 4:
                continue
            try:
                if i_dt is not None:
                    t = datetime.fromisoformat(row[i_dt].strip())
                else:
                    s = row[i_date].strip().replace(".", "-")
                    t = datetime.fromisoformat(
                        s + " " + (row[i_time].strip() if i_time is not None else "00:00"))
                ts.append(t.timestamp())
                o.append(float(row[i_o])); h.append(float(row[i_h]))
                l.append(float(row[i_l])); c.append(float(row[i_c]))
                sp.append(float(row[i_sp]) * POINT if i_sp is not None
                          else spread_points * POINT)
            except (ValueError, IndexError):
                continue
    order = np.argsort(np.array(ts))
    return {k: np.array(v)[order] for k, v in
            dict(ts=ts, o=o, h=h, l=l, c=c, sp=sp).items()}


def aggregate(fine, hours):
    """Group fine bars into TF bars aligned to midnight. Returns list of
    (o, h, l, c, entry_spread, slice) per TF bar, plus per-bar day-of-week."""
    secs = hours * 3600
    key = (fine["ts"] // secs).astype(np.int64)
    bars, idx = [], np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    idx = np.r_[idx, len(key)]
    for a, b in zip(idx[:-1], idx[1:]):
        sl_ = slice(a, b)
        dow = datetime.utcfromtimestamp(fine["ts"][a]).weekday()  # Mon=0..Sun=6
        bars.append(dict(o=fine["o"][a], h=fine["h"][sl_].max(),
                         l=fine["l"][sl_].min(), c=fine["c"][b - 1],
                         sp=fine["sp"][a], sl=sl_, dow=dow))
    return bars


# ---------------------------------------------------------------- filters
def wilder_smooth(x, n):
    out = np.empty_like(x)
    out[:n] = np.nan
    if len(x) <= n:
        return out
    out[n - 1] = np.nansum(x[:n])
    for i in range(n, len(x)):
        out[i] = out[i - 1] - out[i - 1] / n + x[i]
    return out


def adx_verdict(o, h, l, c, period=14, thr=20.0):
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])
    atr = wilder_smooth(tr, period)
    pdi = 100 * wilder_smooth(plus_dm, period) / atr
    mdi = 100 * wilder_smooth(minus_dm, period) / atr
    dx = 100 * np.abs(pdi - mdi) / np.where(pdi + mdi == 0, np.nan, pdi + mdi)
    adx = np.full_like(dx, np.nan)
    start = 2 * period - 1
    if len(dx) > start:
        adx[start] = np.nanmean(dx[period - 1:start + 1])
        for i in range(start + 1, len(dx)):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    v = np.zeros(len(c), dtype=int)
    ok = ~np.isnan(adx) & (adx >= thr)
    v[1:][ok] = np.where(pdi[ok] > mdi[ok], 1, -1)
    return v


def linreg_verdict(c, n=20):
    v = np.zeros(len(c), dtype=int)
    x = np.arange(n) - (n - 1) / 2
    denom = (x * x).sum()
    for i in range(n, len(c)):
        slope = (x * c[i - n:i]).sum() / denom
        v[i] = 1 if slope > 0 else (-1 if slope < 0 else 0)
    return v


def donchian_verdict(h, l, c, n=20):
    v = np.zeros(len(c), dtype=int)
    for i in range(n + 1, len(c)):
        hh, ll = h[i - n - 1:i - 1].max(), l[i - n - 1:i - 1].min()
        if c[i - 1] > hh:
            v[i] = 1
        elif c[i - 1] < ll:
            v[i] = -1
    return v


def verdicts(bars, name):
    o = np.array([b["o"] for b in bars]); h = np.array([b["h"] for b in bars])
    l = np.array([b["l"] for b in bars]); c = np.array([b["c"] for b in bars])
    if name == "ADX":
        return adx_verdict(o, h, l, c)
    if name == "LINREG":
        return linreg_verdict(c)
    if name == "DONCHIAN":
        return donchian_verdict(h, l, c)
    return np.ones(len(bars), dtype=int)          # NONE -> long side


# ---------------------------------------------------------------- engine
def precompute_moves(fine, bars):
    """Per TF bar: cumulative max up-move and max down-move series (vs the
    bar's entry price) as monotone arrays for O(log n) barrier search."""
    pre = []
    for b in bars:
        entry = fine["o"][b["sl"]][0]
        upmove = np.maximum.accumulate(fine["h"][b["sl"]] - entry)
        dnmove = np.maximum.accumulate(entry - fine["l"][b["sl"]])
        pre.append((entry, upmove, dnmove, fine["c"][b["sl"]][-1]))
    return pre


def simulate(bars, pre, verd, direction, tp_pct, sl_pct, deposit, tie="pess"):
    """direction: 'long', 'short', 'both'. tie: what wins when TP and SL sit
    inside the same fine bar - 'pess' (SL) or 'opt' (TP). With H1 fine bars
    the truth lies between the two runs; trust cells where both agree."""
    eq, peak, halted = deposit, deposit, False
    trades = wins = 0
    worst_dd = 0.0
    for b, (entry, upmove, dnmove, close), v in zip(bars, pre, verd):
        if eq < RUIN_FLOOR or halted:
            break
        if b["dow"] == 6:               # Sunday bar
            continue
        d = v if direction == "both" else (1 if direction == "long" else -1)
        if direction == "long" and v <= 0:
            continue
        if direction == "short" and v >= 0:
            continue
        if direction == "both" and v == 0:
            continue
        spread = b["sp"]
        tp_move = tp_pct + spread       # $ distances (equity-independent)
        sl_move = sl_pct - spread if sl_pct - spread > 0.05 else 0.05
        fav, adv = (upmove, dnmove) if d > 0 else (dnmove, upmove)
        i_tp = int(np.searchsorted(fav, tp_move))
        i_sl = int(np.searchsorted(adv, sl_move))
        n = len(fav)
        tp_first = (i_tp < n) and (i_tp <= i_sl if tie == "opt" else i_tp < i_sl)
        if tp_first:
            pnl = eq * tp_pct / 100.0
        elif i_sl < n:                  # SL first (ties per tie-mode)
            pnl = -eq * sl_pct / 100.0
        else:                           # end-of-bar close
            move = (close - entry) * d - spread
            pnl = eq * move / 100.0     # $1 move == 1% of equity (mandate)
        eq = max(eq + pnl, 0.0)
        trades += 1
        wins += pnl > 0
        peak = max(peak, eq)
        dd = 100 * (1 - eq / peak) if peak > 0 else 100.0
        worst_dd = max(worst_dd, dd)
        if dd >= BREAKER_DD:
            halted = True
    ret = 100 * (eq / deposit - 1)
    return dict(final=eq, ret=ret, dd=worst_dd, trades=trades,
                winrate=100 * wins / trades if trades else 0.0, halted=halted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--deposit", type=float, default=1000.0)
    ap.add_argument("--spread-points", type=float, default=30.0,
                    help="fallback spread in points if no SPREAD column")
    ap.add_argument("--tfs", nargs="+", default=["H4", "H8", "H12", "D1"])
    ap.add_argument("--dirs", nargs="+", default=["long", "short", "both"])
    ap.add_argument("--filters", nargs="+", default=["ADX", "LINREG", "DONCHIAN"])
    ap.add_argument("--tie", choices=["pess", "opt"], default="pess",
                    help="TP/SL both inside one fine bar: pess=SL wins, opt=TP wins")
    ap.add_argument("--date-from", default=None, help="e.g. 2025-01-01")
    ap.add_argument("--date-to", default=None)
    ap.add_argument("--out", default="grid_results.csv")
    args = ap.parse_args()

    fine = load_fine_bars(args.csv_path, args.spread_points)
    if args.date_from or args.date_to:
        lo = datetime.fromisoformat(args.date_from).timestamp() if args.date_from else -1e18
        hi = datetime.fromisoformat(args.date_to).timestamp() if args.date_to else 1e18
        keep = (fine["ts"] >= lo) & (fine["ts"] < hi)
        fine = {k: v[keep] for k, v in fine.items()}
    span_days = (fine["ts"][-1] - fine["ts"][0]) / 86400
    print(f"{len(fine['ts'])} fine bars, {span_days:.0f} days, "
          f"median spread ${np.median(fine['sp']):.2f}")

    rows = []
    for tf in args.tfs:
        bars = aggregate(fine, TF_HOURS[tf])
        pre = precompute_moves(fine, bars)
        for filt in args.filters:
            verd = verdicts(bars, filt)
            for direction in args.dirs:
                grid = {}
                for tp in TP_GRID:
                    for sl in SL_GRID:
                        m = simulate(bars, pre, verd, direction, tp, sl,
                                     args.deposit, args.tie)
                        grid[(tp, sl)] = m
                        rows.append(dict(tf=tf, filter=filt, dir=direction,
                                         tp=tp, sl=sl, **m))
                # robust cell = best worst-neighbor return
                def worst_nbr(tp, sl):
                    ti, si = TP_GRID.index(tp), SL_GRID.index(sl)
                    vals = [grid[(TP_GRID[t2], SL_GRID[s2])]["ret"]
                            for t2 in range(max(0, ti - 1), min(len(TP_GRID), ti + 2))
                            for s2 in range(max(0, si - 1), min(len(SL_GRID), si + 2))]
                    return min(vals)
                best = max(grid, key=lambda k: (worst_nbr(*k), grid[k]["ret"]))
                m = grid[best]
                print(f"{tf:>4} {filt:>8} {direction:>5} | robust cell "
                      f"TP {best[0]:.2f} SL {best[1]:.0f} -> {m['ret']:+7.1f}% "
                      f"(minNbr {worst_nbr(*best):+.1f}%, DD {m['dd']:.1f}%, "
                      f"{m['trades']} trades, win {m['winrate']:.0f}%"
                      f"{', HALTED' if m['halted'] else ''})")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nfull grid written to {args.out} "
          f"({len(rows)} configs). Rank there by ret with dd/halted in view.")


if __name__ == "__main__":
    main()
