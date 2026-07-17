"""Export broker XAUUSD D1 bars to CSV for backtest_offline.py.

Run on the Windows machine with the MT5 terminal installed:
    python ml_bot/export_data_mt5.py
Then:
    python ml_bot/backtest_offline.py --csv ml_bot/xauusd_d1.csv
"""

import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "XAUUSD."
TIMEFRAME = mt5.TIMEFRAME_D1
N_BARS = 5000
OUT_FILE = "ml_bot/xauusd_d1.csv"


def main():
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code: {mt5.last_error()}")
        return

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None or not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)
        symbol_info = mt5.symbol_info(SYMBOL)

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, N_BARS)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("No rates returned.")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['spread_cost'] = df['spread'] * symbol_info.point
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread_cost']]
    df.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(df)} bars to {OUT_FILE} "
          f"({df['time'].min()} -> {df['time'].max()})")


if __name__ == "__main__":
    main()
