import re

# Update rl_trader.py and rl_trader_legacy.py to use ATR
def update_trader(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    # Find the def open_trade signature
    new_open_trade = """def open_trade(symbol, action_type, tp_multiplier, sl_multiplier):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found.")
        return
        
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get account info")
        return
        
    # Get recent ATR
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 15)
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(rates)
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    if pd.isna(atr): atr = 1.0
    
    tp_dist = min(atr * tp_multiplier, 3.00)
    sl_dist = atr * sl_multiplier
        
    # Dynamic lot size: 0.01 lot per $100 of equity
    equity = account_info.equity
    lot_size = (equity / 100.0) * 0.01
    lot_size = min(lot_size, 10.0) # Cap max lot size at 10.0
    lot_size = round(lot_size, 2)
    
    if lot_size < symbol_info.volume_min:
        lot_size = symbol_info.volume_min
    elif lot_size > symbol_info.volume_max:
        lot_size = symbol_info.volume_max
        
    price = mt5.symbol_info_tick(symbol).ask if action_type == "BUY" else mt5.symbol_info_tick(symbol).bid
    
    if action_type == "BUY":
        tp = price + tp_dist
        sl = price - sl_dist
        order_type = mt5.ORDER_TYPE_BUY
    else:
        tp = price - tp_dist
        sl = price + sl_dist
        order_type = mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": f"RL_Agent_{action_type}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order send failed, retcode={result.retcode}")
    else:
        print(f"Order sent successfully! Ticket: {result.order}, Volume: {lot_size}, TP_dist: {tp_dist:.2f}, SL_dist: {sl_dist:.2f}")

def main("""
    
    # Replace the old open_trade
    code = re.sub(r'def open_trade\((.*?)\):(.*?)def main\(', new_open_trade, code, flags=re.DOTALL)
    
    # In main(), change TP_PRICE_DIFF to TP_MULTIPLIER and SL_MULTIPLIER
    code = code.replace('TP_PRICE_DIFF = 3.00', 'TP_MULTIPLIER = 1.0\nSL_MULTIPLIER = 2.0')
    
    # In action logic
    code = code.replace('open_trade(SYMBOL, "BUY", TP_PRICE_DIFF)', 'open_trade(SYMBOL, "BUY", TP_MULTIPLIER, SL_MULTIPLIER)')
    code = code.replace('open_trade(SYMBOL, "BUY", tp_price_diff)', 'open_trade(SYMBOL, "BUY", tp_multiplier, sl_multiplier)')
    
    # Replace action == 1 logic to handle SELL
    action_logic_old = """    if action == 1:
        print("Model predicts BUY signal.")
        # open_trade(SYMBOL, "BUY", TP_PRICE_DIFF)
        print("LIVE TRADING IS COMMENTED OUT. UNCOMMENT TO EXECUTE.")
    else:
        print("Model predicts HOLD/FLAT signal. No trade executed.")"""
        
    action_logic_new = """    if action == 0:
        print("Model predicts BUY signal.")
        # open_trade(SYMBOL, "BUY", TP_MULTIPLIER, SL_MULTIPLIER)
        print("LIVE TRADING IS COMMENTED OUT. UNCOMMENT TO EXECUTE.")
    elif action == 1:
        print("Model predicts SELL signal.")
        # open_trade(SYMBOL, "SELL", TP_MULTIPLIER, SL_MULTIPLIER)
        print("LIVE TRADING IS COMMENTED OUT. UNCOMMENT TO EXECUTE.")"""
        
    code = code.replace(action_logic_old, action_logic_new)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)

update_trader('ml_bot/rl_trader.py')
update_trader('ml_bot/rl_trader_legacy.py')
