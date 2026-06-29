import os
import re

# 1. Update rl_env.py
with open('ml_bot/rl_env.py', 'r', encoding='utf-8') as f:
    env_code = f.read()

# Replace the block inside `if action == 1:`
old_block = """            # Simulate inside the bar
            if high_price >= tp_price:
                # Hit Take Profit!
                profit = (tp_price - entry_price)
                reward += profit
                self.balance += (profit * 1.00) # $1 per $1 movement for 0.01 lot
            else:
                # Didn't hit TP, forcefully close at the end of the bar at Bid price (close_price)
                profit = (close_price - entry_price)
                reward += profit
                self.balance += (profit * 1.00) # $1 per $1 movement for 0.01 lot"""

new_block = """            # Calculate dynamic lot size: 0.01 lot per $100 of equity
            lot_size = (self.balance / 100.0) * 0.01
            
            # Simulate inside the bar
            if high_price >= tp_price:
                # Hit Take Profit!
                price_diff = (tp_price - entry_price)
            else:
                # Didn't hit TP, forcefully close at the end of the bar
                price_diff = (close_price - entry_price)
                
            # Profit USD = Price Difference * 100 (Contract Size) * Lot Size
            profit_usd = price_diff * 100.0 * lot_size
            
            # For RL agent, we can use the USD profit as reward (scales with account size)
            # Or use normalized points. Let's use profit_usd.
            reward += profit_usd
            self.balance += profit_usd"""

env_code = env_code.replace(old_block, new_block)
with open('ml_bot/rl_env.py', 'w', encoding='utf-8') as f:
    f.write(env_code)

# 2. Update rl_trader.py
with open('ml_bot/rl_trader.py', 'r', encoding='utf-8') as f:
    trader_code = f.read()

# Replace open_trade function
import re

def replace_open_trade(code):
    match = re.search(r'def open_trade\((.*?)\):(.*?)def main\(', code, re.DOTALL)
    if not match:
        return code
    
    old_open_trade = match.group(0)
    
    new_open_trade = """def open_trade(symbol, action_type, tp_price_diff):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found.")
        return
        
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get account info")
        return
        
    # Dynamic lot size: 0.01 lot per $100 of equity
    equity = account_info.equity
    lot_size = (equity / 100.0) * 0.01
    lot_size = round(lot_size, 2)
    
    if lot_size < symbol_info.volume_min:
        lot_size = symbol_info.volume_min
    elif lot_size > symbol_info.volume_max:
        lot_size = symbol_info.volume_max
        
    price = mt5.symbol_info_tick(symbol).ask if action_type == "BUY" else mt5.symbol_info_tick(symbol).bid
    tp = price + tp_price_diff if action_type == "BUY" else price - tp_price_diff
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if action_type == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": 0.0,
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
        print(f"Order sent successfully! Ticket: {result.order}, Volume: {lot_size}")

def main("""
    return code.replace(old_open_trade, new_open_trade)

trader_code = replace_open_trade(trader_code)
with open('ml_bot/rl_trader.py', 'w', encoding='utf-8') as f:
    f.write(trader_code)

# 3. Update rl_trader_legacy.py
with open('ml_bot/rl_trader_legacy.py', 'r', encoding='utf-8') as f:
    legacy_code = f.read()

legacy_code = replace_open_trade(legacy_code)
with open('ml_bot/rl_trader_legacy.py', 'w', encoding='utf-8') as f:
    f.write(legacy_code)

print("Updated lot size logic in all files.")
