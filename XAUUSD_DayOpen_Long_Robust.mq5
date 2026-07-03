//+------------------------------------------------------------------+
//|                                   XAUUSD_DayOpen_Long_Robust.mq5 |
//|                                                                  |
//|=== STRATEGY (unchanged mandate) =================================|
//|  Long-only XAUUSD Expert Advisor.                                |
//|  - Exactly ONE market BUY per trading day (one D1 bar = one      |
//|    trade), opened near the open of each new daily bar.           |
//|  - Profit target = 1% of account equity per day (money-based,    |
//|    net of swap+commission), NOT a fixed $ price move.            |
//|  - Position sizing: 0.01 lot per $100 of equity                  |
//|    (lot = equity * 0.01 / InpCapitalPer001Lot).                  |
//|  - Position never held across the daily bar (EOD close) and is   |
//|    force-closed before the weekend.                              |
//|  - Optional trend filter (ADX / Donchian / LinReg / SAR).        |
//|                                                                  |
//|=== ROBUSTNESS CHANGES vs XAUUSD_DayOpen_Long v1.20 ==============|
//|  The v1.20 backtest (log.txt) ended in a margin stop-out:        |
//|  78 straight $2-target wins, then one -$130 day (2026.02.17)     |
//|  wiped the account (200 -> 1.05 USD; 10k run -> -6708 USD).      |
//|  Fixes, in order of importance:                                  |
//|   1. DISASTER STOP-LOSS: broker-side SL placed at entry, sized   |
//|      as InpMaxDailyLossPct of entry equity (default 5%). A       |
//|      money-based emergency close in OnTick backs it up against   |
//|      gaps. This is the change that prevents the observed ruin.   |
//|   2. Broker-side TP also placed at entry, so target/stop survive |
//|      terminal/VPS disconnects (v1.20 exits lived only in OnTick).|
//|   3. Profit target is 1% of entry equity in MONEY (mandate),     |
//|      converted to a price distance via tick value - correct even |
//|      when lot rounding makes $1-move != 1% of equity.            |
//|   4. Spread filter + entry delay: v1.20 bought at the midnight   |
//|      rollover, the worst spread of the day. With a ~$1 target a  |
//|      $0.50 spread is 50% of the target. Entry waits              |
//|      InpEntryDelayMinutes and requires spread <= InpMaxSpread.   |
//|   5. Entry attempt cap + entry window: v1.20 logged 6104 failed  |
//|      "market closed" orders retrying all night. Attempts are     |
//|      capped per day and entries stop after InpEntryWindowHours   |
//|      (late entries have little time left to reach the target).   |
//|   6. Equity circuit breaker: trading halts if equity falls       |
//|      InpMaxTotalDDPct below its high-water mark (persisted via a |
//|      terminal global variable across restarts).                  |
//|   7. Lot is reduced to fit free margin (with a safety buffer)    |
//|      instead of being rejected outright; skips the day if even   |
//|      the minimum lot does not fit.                               |
//|   8. Optional Sunday-bar skip (short bar, bad spreads).          |
//|   9. Trend filter is evaluated on the attach day too (v1.20      |
//|      defaulted to "UP" until the next daily bar).                |
//|                                                                  |
//|=== HONEST LIMITATION ============================================|
//|  These changes cap catastrophic risk; they cannot create edge.   |
//|  At 0.01 lot / $100 equity a $1 gold move = 1% of equity, i.e.   |
//|  ~50:1 effective exposure at $5000 gold. The 1%-target/SL race   |
//|  resolves within minutes of the open, so the trade pays the      |
//|  full spread daily while capturing almost no trend. Expect the   |
//|  spread (~0.2-0.4% of equity/day) as a structural cost. See      |
//|  ANALYSIS_REPORT.md and sim_robustness.py in this repo.          |
//+------------------------------------------------------------------+
#property copyright "2026"
#property version   "2.00"
#property description "Robust long-only XAUUSD daily-open EA: 1%/day money target, disaster SL, spread filter, circuit breaker."

#include <Trade\Trade.mqh>

//--- trend filter selection ----------------------------------------
enum ENUM_TREND_FILTER
{
   TREND_NONE     = 0, // None (buy every day)
   TREND_ADX      = 1, // ADX: ADX>=thr AND +DI>-DI
   TREND_DONCHIAN = 2, // Donchian: close[1] > highest high of prior N days
   TREND_LINREG   = 3, // Linear-regression slope>0 over N daily closes
   TREND_SAR      = 4  // Parabolic SAR below close[1]
};

//--- inputs ---------------------------------------------------------
input group "=== Strategy (mandate) ==="
input double  InpDailyProfitPct       = 1.0;      // Daily profit target (% of entry equity)
input double  InpCapitalPer001Lot     = 100.0;    // $ equity per 0.01 lot (position sizing)
input bool    InpCloseBeforeWeekend   = true;     // Force-close before weekend
input int     InpFridayCloseHour      = 22;       // Friday server hour to force-close (0-23)
input bool    InpEnterOnAttachSameDay = false;    // Enter immediately on attach (mid-day)

input group "=== Risk control ==="
input double  InpMaxDailyLossPct      = 5.0;      // Disaster SL (% of entry equity, 0=off - NOT recommended)
input double  InpMaxTotalDDPct        = 30.0;     // Circuit breaker: halt below this % drawdown from peak (0=off)
input int     InpMaxSpreadPoints      = 80;       // Max spread (points) allowed at entry (0=off)
input int     InpEntryDelayMinutes    = 5;        // Wait after daily open before entering (rollover spread)
input int     InpEntryWindowHours     = 12;       // Stop trying to enter this many hours into the day
input int     InpMaxEntryAttempts     = 20;       // Max order attempts per day (stops retry storms)
input bool    InpSkipSundayBar        = true;     // Skip the short Sunday bar (if broker has one)
input double  InpMarginUseFraction    = 0.8;      // Max fraction of free margin the position may use

input group "=== Trend Filter (only BUY when uptrend) ==="
input ENUM_TREND_FILTER InpTrendFilter    = TREND_ADX; // Trend filter
input int     InpAdxPeriod                = 14;     // ADX period
input double  InpAdxThreshold             = 20.0;   // ADX min strength to count as trending
input int     InpDonchianPeriod           = 20;     // Donchian lookback (days)
input int     InpLinRegPeriod             = 20;     // Linear-regression lookback (days)
input double  InpSarStep                  = 0.02;   // Parabolic SAR step
input double  InpSarMax                   = 0.2;    // Parabolic SAR max
input int     InpEntryCooldownSec         = 30;     // Cooldown (sec) between entry retries
input string  InpRunTag                   = "run";  // Results-file tag (for batch backtests)

input group "=== Execution ==="
input long    InpMagic                = 20260703; // Magic number (EA id)
input ulong   InpMaxSlippagePoints    = 50;       // Max slippage (points)
input string  InpTradeComment         = "XAU_DayOpen_Robust"; // Trade comment

//--- globals --------------------------------------------------------
CTrade        trade;
datetime      g_lastD1BarTime    = 0;     // open-time of the last processed D1 bar
bool          g_tradedToday      = false; // already entered today (re-entry guard)
double        g_entryEquity      = 0.0;   // equity captured at entry decision
double        g_targetMoney      = 0.0;   // $ profit that equals InpDailyProfitPct of entry equity
double        g_maxLossMoney     = 0.0;   // $ loss cap for the day
int           g_adxHandle        = INVALID_HANDLE;
int           g_sarHandle        = INVALID_HANDLE;
bool          g_todayUpTrend     = true;
bool          g_trendEvaluated   = false; // trend verdict computed for current day?
datetime      g_lastEntryAttempt = 0;
int           g_entryAttempts    = 0;     // order attempts today
bool          g_haltedByDD       = false; // circuit breaker latched
string        g_peakGvName       = "";    // terminal global-variable name for peak equity

// runtime filter params (defaults from inputs; overridden by config file)
ENUM_TREND_FILTER g_trendFilter = TREND_NONE;
int     g_adxPeriod      = 14;
double  g_adxThreshold   = 20.0;
int     g_donchianPeriod = 20;
int     g_linRegPeriod   = 20;
double  g_sarStep        = 0.02;
double  g_sarMax         = 0.2;
string  g_runTag         = "run";

//+------------------------------------------------------------------+
//| Trading permission check                                          |
//+------------------------------------------------------------------+
bool IsTradingAllowed(void)
{
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))   return(false);
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))             return(false);
   if((ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE)
        == SYMBOL_TRADE_MODE_DISABLED)                return(false);
   return(true);
}

//+------------------------------------------------------------------+
//| Auto-detect best filling mode supported by the symbol             |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING DetectFilling(void)
{
   int filling = (int)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_IOC) != 0) return(ORDER_FILLING_IOC);
   if((filling & SYMBOL_FILLING_FOK) != 0) return(ORDER_FILLING_FOK);
   return(ORDER_FILLING_RETURN);
}

//+------------------------------------------------------------------+
//| Decimal count implied by a volume step                            |
//+------------------------------------------------------------------+
int VolumeStepDigits(double step)
{
   if(step >= 1.0)   return(0);
   if(step >= 0.1)   return(1);
   if(step >= 0.01)  return(2);
   return(3);
}

//+------------------------------------------------------------------+
//| $ of P/L produced by a 1.0 price-unit move of 1.0 lot             |
//+------------------------------------------------------------------+
double MoneyPerUnitPerLot(void)
{
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickValue <= 0.0 || tickSize <= 0.0)
      return(SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE)); // fallback
   return(tickValue / tickSize);
}

//+------------------------------------------------------------------+
//| Position size: AccountEquity * 0.01 / InpCapitalPer001Lot         |
//| (mandate), then reduced to fit free margin with a buffer.         |
//+------------------------------------------------------------------+
double CalcLot(double equity, double price)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0.0) step = 0.01;
   if(vmin  < 0.0) vmin = 0.01;
   if(vmax <= 0.0) vmax = vmin;

   double lot = equity * 0.01 / InpCapitalPer001Lot;
   lot = MathFloor(lot / step) * step;
   if(lot < vmin) lot = vmin;
   if(lot > vmax) lot = vmax;
   lot = NormalizeDouble(lot, VolumeStepDigits(step));

   // shrink to fit free margin (v1.20 rejected the whole day instead)
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE) * InpMarginUseFraction;
   for(int i = 0; i < 200 && lot >= vmin; i++)
   {
      double need = 0.0;
      if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lot, price, need))
         return(0.0);
      if(need <= freeMargin)
         return(lot);
      lot = NormalizeDouble(lot - step, VolumeStepDigits(step));
   }
   return(0.0);   // not affordable even at minimum volume
}

//+------------------------------------------------------------------+
//| Does the EA hold an open position on this symbol/magic?           |
//+------------------------------------------------------------------+
bool HasOpenPosition(ulong &ticketOut)
{
   ticketOut = 0;
   int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionSelectByTicket(t) &&
         PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == InpMagic)
      {
         ticketOut = t;
         return(true);
      }
   }
   return(false);
}

//+------------------------------------------------------------------+
//| Sum commission over the deals belonging to a position             |
//+------------------------------------------------------------------+
double GetPositionCommission(ulong positionTicket)
{
   double comm = 0.0;
   if(HistorySelectByPosition(positionTicket))
   {
      int deals = HistoryDealsTotal();
      for(int i = 0; i < deals; i++)
      {
         ulong dt = HistoryDealGetTicket(i);
         if(dt != 0)
            comm += HistoryDealGetDouble(dt, DEAL_COMMISSION);
      }
   }
   return(comm);
}

//+------------------------------------------------------------------+
//| Net P/L (incl. swap + commission) of an open position             |
//+------------------------------------------------------------------+
double PositionNetProfit(ulong ticket)
{
   if(!PositionSelectByTicket(ticket)) return(0.0);
   return(PositionGetDouble(POSITION_PROFIT)
        + PositionGetDouble(POSITION_SWAP)
        + GetPositionCommission(ticket));
}

//+------------------------------------------------------------------+
//| Close a position (single retry, clear log)                        |
//+------------------------------------------------------------------+
bool ClosePosition(ulong ticket, const string reason)
{
   if(!PositionSelectByTicket(ticket)) return(false);

   double vol = PositionGetDouble(POSITION_VOLUME);
   double pl  = PositionNetProfit(ticket);

   for(int attempt = 0; attempt < 2; attempt++)
   {
      if(trade.PositionClose(ticket, InpMaxSlippagePoints))
      {
         PrintFormat("CLOSED ticket=%I64u vol=%.2f netP/L=%.2f | reason=%s",
                     ticket, vol, pl, reason);
         return(true);
      }
      PrintFormat("Close failed (attempt %d): retcode=%u err=%d (%s)",
                  attempt + 1, trade.ResultRetcode(), GetLastError(),
                  trade.ResultRetcodeDescription());
      if(attempt == 0) Sleep(500);
   }
   return(false);
}

//+------------------------------------------------------------------+
//| Peak-equity high-water mark (persisted across restarts)           |
//+------------------------------------------------------------------+
double LoadPeakEquity(double fallback)
{
   if(GlobalVariableCheck(g_peakGvName))
      return(GlobalVariableGet(g_peakGvName));
   GlobalVariableSet(g_peakGvName, fallback);
   return(fallback);
}

void UpdatePeakEquity(double equity)
{
   double peak = LoadPeakEquity(equity);
   if(equity > peak)
      GlobalVariableSet(g_peakGvName, equity);
}

//+------------------------------------------------------------------+
//| Circuit breaker: equity too far below its high-water mark?        |
//+------------------------------------------------------------------+
bool CircuitBreakerTripped(void)
{
   if(InpMaxTotalDDPct <= 0.0) return(false);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   UpdatePeakEquity(equity);
   double peak = LoadPeakEquity(equity);
   if(peak <= 0.0) return(false);
   double ddPct = 100.0 * (1.0 - equity / peak);
   if(ddPct >= InpMaxTotalDDPct)
   {
      if(!g_haltedByDD)
         PrintFormat("CIRCUIT BREAKER: equity %.2f is %.1f%% below peak %.2f "
                     "(limit %.1f%%). New entries HALTED.",
                     equity, ddPct, peak, InpMaxTotalDDPct);
      g_haltedByDD = true;
      return(true);
   }
   return(false);
}

//+------------------------------------------------------------------+
//| Spread in points, and entry-time spread check                     |
//+------------------------------------------------------------------+
bool SpreadOk(void)
{
   if(InpMaxSpreadPoints <= 0) return(true);
   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   return(spread > 0 && spread <= InpMaxSpreadPoints);
}

//+------------------------------------------------------------------+
//| Try to open the daily BUY with broker-side SL/TP attached.        |
//+------------------------------------------------------------------+
bool TryOpenLong(void)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double ask    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
   {
      Print("TryOpenLong: invalid quotes, skipping.");
      return(false);
   }

   double lot = CalcLot(equity, ask);
   if(lot <= 0.0)
   {
      PrintFormat("WARN: minimum lot does not fit free margin (equity=%.2f) - skipping today.", equity);
      g_tradedToday = true;    // give up for today, not just this tick
      return(false);
   }

   double mpu = MoneyPerUnitPerLot();            // $ per 1.0 move per lot
   if(mpu <= 0.0) return(false);

   // money targets from the MANDATE: 1% of equity up, InpMaxDailyLossPct down
   double targetMoney  = equity * InpDailyProfitPct   / 100.0;
   double maxLossMoney = equity * InpMaxDailyLossPct  / 100.0;

   double tpMove = targetMoney / (lot * mpu);    // price distance for +1% equity
   double slMove = (InpMaxDailyLossPct > 0.0 ? maxLossMoney / (lot * mpu) : 0.0);

   // respect broker minimum stop distance
   double point      = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double minStop    = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   double tpPrice    = NormalizeDouble(ask + MathMax(tpMove, minStop), _Digits);
   double slPrice    = 0.0;
   if(slMove > 0.0)
   {
      slMove  = MathMax(slMove, minStop + (ask - bid));
      slPrice = NormalizeDouble(ask - slMove, _Digits);
      if(slPrice <= 0.0) slPrice = 0.0;
   }

   for(int attempt = 0; attempt < 2; attempt++)
   {
      if(trade.Buy(lot, _Symbol, ask, slPrice, tpPrice, InpTradeComment))
      {
         g_entryEquity = equity;
         g_targetMoney  = targetMoney;
         g_maxLossMoney = maxLossMoney;
         PrintFormat("BUY opened lot=%.2f @ %.*f | TP=%.*f (+$%.2f = %.2f%%) | SL=%.*f (-$%.2f = %.2f%%) | ticket=%I64u",
                     lot, _Digits, ask,
                     _Digits, tpPrice, targetMoney, InpDailyProfitPct,
                     _Digits, slPrice, maxLossMoney, InpMaxDailyLossPct,
                     trade.ResultOrder());
         return(true);
      }
      PrintFormat("Buy failed (attempt %d): retcode=%u err=%d (%s)",
                  attempt + 1, trade.ResultRetcode(), GetLastError(),
                  trade.ResultRetcodeDescription());
      if(attempt == 0) Sleep(500);
   }
   return(false);
}

//+------------------------------------------------------------------+
//| Map a filter name/number string to the enum                       |
//+------------------------------------------------------------------+
ENUM_TREND_FILTER FilterFromString(string s)
{
   StringTrimLeft(s); StringTrimRight(s);
   if(s == "NONE" || s == "none" || s == "0") return(TREND_NONE);
   if(s == "ADX"  || s == "adx"  || s == "1") return(TREND_ADX);
   if(s == "DONCHIAN" || s == "donchian" || s == "2") return(TREND_DONCHIAN);
   if(s == "LINREG" || s == "linreg" || s == "3") return(TREND_LINREG);
   if(s == "SAR" || s == "sar" || s == "4") return(TREND_SAR);
   return(TREND_NONE);
}

//+------------------------------------------------------------------+
//| Read optional filter config from <Common>\Files\xau_filter.cfg    |
//+------------------------------------------------------------------+
void ReadFilterConfig(void)
{
   int h = FileOpen("xau_filter.cfg", FILE_READ | FILE_COMMON | FILE_ANSI | FILE_TXT);
   if(h == INVALID_HANDLE)
      return;

   int loaded = 0;
   while(!FileIsEnding(h))
   {
      string line = FileReadString(h);
      StringTrimLeft(line); StringTrimRight(line);
      if(StringLen(line) == 0) continue;
      if(StringGetCharacter(line, 0) == '#') continue;
      int sep = StringFind(line, "=");
      if(sep <= 0) continue;
      string key = StringSubstr(line, 0, sep);
      string val = StringSubstr(line, sep + 1);
      StringTrimLeft(key); StringTrimRight(key);
      StringTrimLeft(val); StringTrimRight(val);
      if(key == "filter")          { g_trendFilter   = FilterFromString(val); loaded++; }
      else if(key == "runtag")     { g_runTag        = val; loaded++; }
      else if(key == "adxperiod")  { g_adxPeriod     = (int)StringToInteger(val); loaded++; }
      else if(key == "adxthreshold"){ g_adxThreshold = StringToDouble(val); loaded++; }
      else if(key == "donchian")   { g_donchianPeriod= (int)StringToInteger(val); loaded++; }
      else if(key == "linreg")     { g_linRegPeriod  = (int)StringToInteger(val); loaded++; }
      else if(key == "sarstep")    { g_sarStep       = StringToDouble(val); loaded++; }
      else if(key == "sarmax")     { g_sarMax        = StringToDouble(val); loaded++; }
   }
   FileClose(h);
   if(loaded > 0)
      PrintFormat("Filter config loaded from xau_filter.cfg (%d fields): filter=%s runtag=%s",
                  loaded, EnumToString(g_trendFilter), g_runTag);
}

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   if(!IsTradingAllowed())
   {
      Print("OnInit: trading is NOT allowed (terminal/auto-trading/symbol).");
      return(INIT_FAILED);
   }

   trade.SetExpertMagicNumber((ulong)InpMagic);
   trade.SetDeviationInPoints(InpMaxSlippagePoints);
   trade.SetTypeFilling(DetectFilling());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_ALL);

   g_trendFilter    = InpTrendFilter;
   g_adxPeriod      = InpAdxPeriod;
   g_adxThreshold   = InpAdxThreshold;
   g_donchianPeriod = InpDonchianPeriod;
   g_linRegPeriod   = InpLinRegPeriod;
   g_sarStep        = InpSarStep;
   g_sarMax         = InpSarMax;
   g_runTag         = InpRunTag;
   ReadFilterConfig();

   if(g_trendFilter == TREND_ADX)
   {
      g_adxHandle = iADX(_Symbol, PERIOD_D1, g_adxPeriod);
      if(g_adxHandle == INVALID_HANDLE)
         PrintFormat("WARN: iADX handle creation failed err=%d", GetLastError());
   }
   if(g_trendFilter == TREND_SAR)
   {
      g_sarHandle = iSAR(_Symbol, PERIOD_D1, g_sarStep, g_sarMax);
      if(g_sarHandle == INVALID_HANDLE)
         PrintFormat("WARN: iSAR handle creation failed err=%d", GetLastError());
   }

   g_peakGvName = StringFormat("XAU_DOLR_peak_%I64d", InpMagic);

   datetime curD1 = iTime(_Symbol, PERIOD_D1, 0);
   if(InpEnterOnAttachSameDay)
   {
      g_lastD1BarTime = curD1;   // treat attach day as the current day...
      g_tradedToday   = false;   // ...but allow one entry
   }
   else
   {
      g_lastD1BarTime = curD1;
      g_tradedToday   = true;    // wait for the next daily bar
   }
   g_trendEvaluated = false;     // evaluate filter on first tick either way
   g_entryEquity    = AccountInfoDouble(ACCOUNT_EQUITY);
   g_entryAttempts  = 0;
   g_haltedByDD     = false;
   UpdatePeakEquity(g_entryEquity);

   PrintFormat("OnInit OK v2.00: magic=%I64d equity=%.2f filter=%s runtag=%s "
               "target=%.2f%%/day SL=%.2f%%/day maxDD=%.1f%%",
               InpMagic, g_entryEquity, EnumToString(g_trendFilter), g_runTag,
               InpDailyProfitPct, InpMaxDailyLossPct, InpMaxTotalDDPct);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_adxHandle != INVALID_HANDLE) IndicatorRelease(g_adxHandle);
   if(g_sarHandle != INVALID_HANDLE) IndicatorRelease(g_sarHandle);
}

//+------------------------------------------------------------------+
//| Trend filters (unchanged from v1.20)                              |
//+------------------------------------------------------------------+
bool UpByAdx(void)
{
   if(g_adxHandle == INVALID_HANDLE) return(true);
   double adx[], plus[], minus[];
   if(CopyBuffer(g_adxHandle, 0, 1, 1, adx)   < 1) return(false);
   if(CopyBuffer(g_adxHandle, 1, 1, 1, plus)  < 1) return(false);
   if(CopyBuffer(g_adxHandle, 2, 1, 1, minus) < 1) return(false);
   return(adx[0] >= g_adxThreshold && plus[0] > minus[0]);
}

bool UpByDonchian(void)
{
   int n = g_donchianPeriod;
   if(n < 2) n = 2;
   double maxHigh = 0.0;
   for(int s = 2; s <= n + 1; s++)
   {
      double h = iHigh(_Symbol, PERIOD_D1, s);
      if(h > maxHigh) maxHigh = h;
   }
   double close1 = iClose(_Symbol, PERIOD_D1, 1);
   return(close1 > maxHigh);
}

bool UpByLinReg(void)
{
   int n = g_linRegPeriod;
   if(n < 3) n = 3;
   double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0;
   for(int i = 0; i < n; i++)
   {
      int shift = n - i;
      double y = iClose(_Symbol, PERIOD_D1, shift);
      if(y <= 0.0) return(false);
      double x = (double)i;
      sumX += x; sumY += y; sumXY += x * y; sumX2 += x * x;
   }
   double denom = (double)n * sumX2 - sumX * sumX;
   if(denom == 0.0) return(false);
   double slope = ((double)n * sumXY - sumX * sumY) / denom;
   return(slope > 0.0);
}

bool UpBySar(void)
{
   if(g_sarHandle == INVALID_HANDLE) return(true);
   double sar[];
   if(CopyBuffer(g_sarHandle, 0, 1, 1, sar) < 1) return(false);
   double close1 = iClose(_Symbol, PERIOD_D1, 1);
   return(sar[0] < close1);
}

bool IsUpTrend(void)
{
   switch(g_trendFilter)
   {
      case TREND_ADX:      return(UpByAdx());
      case TREND_DONCHIAN: return(UpByDonchian());
      case TREND_LINREG:   return(UpByLinReg());
      case TREND_SAR:      return(UpBySar());
      default:             return(true);
   }
}

//+------------------------------------------------------------------+
//| Main tick handler                                                 |
//+------------------------------------------------------------------+
void OnTick(void)
{
   if(!IsTradingAllowed()) return;

   datetime curD1 = iTime(_Symbol, PERIOD_D1, 0);
   if(curD1 == 0) return;

   //-----------------------------------------------------------------
   // 1) NEW D1 BAR: close leftover position, reset the day
   //-----------------------------------------------------------------
   if(curD1 != g_lastD1BarTime)
   {
      ulong tk = 0;
      if(HasOpenPosition(tk))
         ClosePosition(tk, "End-of-day (new D1 bar)");

      g_tradedToday      = false;
      g_entryEquity      = AccountInfoDouble(ACCOUNT_EQUITY);
      g_lastD1BarTime    = curD1;
      g_lastEntryAttempt = 0;
      g_entryAttempts    = 0;
      g_trendEvaluated   = false;
   }

   if(!g_trendEvaluated)
   {
      g_todayUpTrend   = IsUpTrend();
      g_trendEvaluated = true;
      PrintFormat("D1 bar %s | equity=%.2f | filter=%s trend=%s",
                  TimeToString(curD1, TIME_DATE),
                  AccountInfoDouble(ACCOUNT_EQUITY),
                  EnumToString(g_trendFilter),
                  (g_todayUpTrend ? "UP" : "down(NO-TRADE)"));
   }

   //-----------------------------------------------------------------
   // 2) WEEKEND GUARD
   //-----------------------------------------------------------------
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(InpCloseBeforeWeekend && dt.day_of_week == 5 && dt.hour >= InpFridayCloseHour)
   {
      ulong tk = 0;
      if(HasOpenPosition(tk))
         ClosePosition(tk, StringFormat("Weekend guard (Friday %02d:00)", dt.hour));
      g_tradedToday = true;
   }

   //-----------------------------------------------------------------
   // 3) MONEY-BASED TARGET / EMERGENCY LOSS CAP
   //    (broker-side TP/SL are the primary exits; this nets in
   //     swap+commission and covers gaps past the broker SL)
   //-----------------------------------------------------------------
   ulong tk = 0;
   if(HasOpenPosition(tk))
   {
      double net = PositionNetProfit(tk);
      if(g_targetMoney > 0.0 && net >= g_targetMoney)
      {
         ClosePosition(tk, StringFormat("Profit target hit (net $%.2f >= $%.2f = %.2f%% of equity)",
                                        net, g_targetMoney, InpDailyProfitPct));
         g_tradedToday = true;
      }
      else if(g_maxLossMoney > 0.0 && net <= -g_maxLossMoney)
      {
         ClosePosition(tk, StringFormat("EMERGENCY loss cap (net $%.2f <= -$%.2f = %.2f%% of equity)",
                                        net, g_maxLossMoney, InpMaxDailyLossPct));
         g_tradedToday = true;
      }
   }

   //-----------------------------------------------------------------
   // 4) ENTRY (one per day, gated by every robustness check)
   //-----------------------------------------------------------------
   tk = 0;
   if(g_tradedToday || HasOpenPosition(tk) || !g_todayUpTrend) return;
   if(CircuitBreakerTripped())                                 return;
   if(InpSkipSundayBar && dt.day_of_week == 0)                 return;

   datetime now = TimeCurrent();
   long secondsIntoDay = (long)(now - curD1);
   if(secondsIntoDay < (long)InpEntryDelayMinutes * 60)        return; // rollover spread
   if(InpEntryWindowHours > 0 &&
      secondsIntoDay > (long)InpEntryWindowHours * 3600)
   {
      g_tradedToday = true;   // window closed - no late entries today
      return;
   }
   if(g_entryAttempts >= InpMaxEntryAttempts)
   {
      g_tradedToday = true;   // stop retry storms (6104 failures in v1.20 log)
      PrintFormat("Entry attempts exhausted (%d) - giving up for today.", g_entryAttempts);
      return;
   }
   if(!SpreadOk())                                             return;
   if(g_lastEntryAttempt != 0 && now - g_lastEntryAttempt < InpEntryCooldownSec)
      return;

   g_entryAttempts++;
   g_lastEntryAttempt = now;
   if(TryOpenLong())
      g_tradedToday = true;
}

//+------------------------------------------------------------------+
//| Tester: write results to <Common>\Files\results_<runtag>.csv      |
//+------------------------------------------------------------------+
double OnTester(void)
{
   double netProfit = TesterStatistics(STAT_PROFIT);
   double pf        = TesterStatistics(STAT_PROFIT_FACTOR);
   double ddPct     = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   double sharpe    = TesterStatistics(STAT_SHARPE_RATIO);
   long   trades    = (long)TesterStatistics(STAT_TRADES);
   long   winT      = (long)TesterStatistics(STAT_PROFIT_TRADES);
   long   lossT     = (long)TesterStatistics(STAT_LOSS_TRADES);
   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);

   string fn = StringFormat("results_%s.csv", g_runTag);
   int h = FileOpen(fn, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(h != INVALID_HANDLE)
   {
      FileWrite(h, "tag","filter","netProfit","profitFactor","maxDDpct","sharpe",
                   "trades","wins","losses","finalBalance");
      FileWrite(h, g_runTag, EnumToString(g_trendFilter),
                DoubleToString(netProfit,2), DoubleToString(pf,3),
                DoubleToString(ddPct,2), DoubleToString(sharpe,3),
                trades, winT, lossT, DoubleToString(balance,2));
      FileClose(h);
      PrintFormat("OnTester CSV written: %s | net=%.2f PF=%.3f DD=%.2f%% trades=%I64d",
                  fn, netProfit, pf, ddPct, trades);
   }
   return(netProfit);
}
//+------------------------------------------------------------------+
