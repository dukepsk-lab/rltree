//+------------------------------------------------------------------+
//|                                          XAUUSD_DayOpen_Long.mq5 |
//|                                                                  |
//|=== STRATEGY SUMMARY =============================================|
//|  Long-only XAUUSD Expert Advisor.                                |
//|  - Exactly ONE market BUY per trading day, opened at the open    |
//|    of each new daily (D1) bar (at the current Ask).              |
//|  - NO STOP LOSS (SL = 0).                                        |
//|  - Exit when EITHER:                                             |
//|      (a) gold price risen >= InpProfitTargetPrice $ from entry   |
//|           (e.g. 2.0 = $2 up) -> close early, no re-entry;        |
//|      (b) the daily bar ends (a new D1 bar forms) -> close at     |
//|          market (hold only WITHIN the bar, no cross-day hold);   |
//|      (c) Friday guard closes the position before the weekend.    |
//|  - TREND FILTER: only buys when the selected filter says UP.     |
//|                                                                  |
//|  For batch backtests the filter can be driven by a config file:  |
//|     <Common>\Files\xau_filter.cfg   (overrides the input defaults)|
//|                                                                  |
//|=== POSITION SIZING =============================================|
//|  Lot = AccountEquity * 0.01 / InpCapitalPer001Lot                |
//|      (default InpCapitalPer001Lot = 100  =>  0.01 lot per $100)  |
//|                                                                  |
//|=== RISK WARNING ================================================|
//|  ****************************************************************|
//|   THIS EA TRADES WITHOUT ANY STOP LOSS.                          |
//|   Use ONLY with capital you can afford to lose; test on DEMO.    |
//|  ****************************************************************|
//+------------------------------------------------------------------+
#property copyright "2026"
#property version   "1.20"
#property description "Long-only XAUUSD daily-open EA with switchable trend filter. NO STOP LOSS."

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
input group "=== Strategy ==="
input double  InpProfitTargetPrice    = 2.0;      // Profit target (gold price $ move, e.g. 2.0 = $2 up)
input double  InpCapitalPer001Lot     = 100.0;    // $ equity per 0.01 lot (position sizing)
input bool    InpCloseBeforeWeekend   = true;     // Force-close before weekend
input int     InpFridayCloseHour      = 22;       // Friday server hour to force-close (0-23)
input bool    InpEnterOnAttachSameDay = false;    // Enter immediately on attach (mid-day)

input group "=== Trend Filter (only BUY when uptrend) ==="
input ENUM_TREND_FILTER InpTrendFilter     = TREND_ADX; // Trend filter (ADX = buy only when ADX>=thr AND +DI>-DI)
input int     InpAdxPeriod                = 14;     // ADX period
input double  InpAdxThreshold             = 20.0;   // ADX min strength to count as trending
input int     InpDonchianPeriod           = 20;     // Donchian lookback (days)
input int     InpLinRegPeriod             = 20;     // Linear-regression lookback (days)
input double  InpSarStep                  = 0.02;   // Parabolic SAR step
input double  InpSarMax                   = 0.2;    // Parabolic SAR max
input int     InpEntryCooldownSec         = 30;     // Cooldown (sec) between entry retries (market-closed)
input string  InpRunTag                   = "run";  // Results-file tag (for batch backtests)

input group "=== Execution ==="
input long    InpMagic                = 20260625; // Magic number (EA id)
input ulong   InpMaxSlippagePoints    = 50;       // Max slippage (points)
input string  InpTradeComment         = "XAU_DayOpen_Long"; // Trade comment

//--- globals --------------------------------------------------------
CTrade        trade;                 // trade helper
datetime      g_lastD1BarTime   = 0; // open-time of the last processed D1 bar
bool          g_tradedToday     = false; // already entered today (re-entry guard)
double        g_entryEquity     = 0.0;   // account equity captured at entry
bool          g_firstTickDone   = false; // first-tick initialisation flag
int           g_adxHandle       = INVALID_HANDLE;
int           g_sarHandle       = INVALID_HANDLE;
bool          g_todayUpTrend    = true;  // trend verdict for the current day
datetime      g_lastEntryAttempt= 0;     // last entry attempt time (cooldown)

// runtime filter params (defaults from inputs; overridden by config file)
ENUM_TREND_FILTER g_trendFilter   = TREND_NONE;
int     g_adxPeriod      = 14;
double  g_adxThreshold   = 20.0;
int     g_donchianPeriod = 20;
int     g_linRegPeriod   = 20;
double  g_sarStep        = 0.02;
double  g_sarMax         = 0.2;
string  g_runTag         = "run";

//+------------------------------------------------------------------+
//| Trading permission check                                         |
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
//| Auto-detect best filling mode supported by the symbol            |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING DetectFilling(void)
{
   int filling = (int)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_IOC) != 0) return(ORDER_FILLING_IOC);
   if((filling & SYMBOL_FILLING_FOK) != 0) return(ORDER_FILLING_FOK);
   return(ORDER_FILLING_RETURN);
}

//+------------------------------------------------------------------+
//| Decimal count implied by a volume step                           |
//+------------------------------------------------------------------+
int VolumeStepDigits(double step)
{
   if(step >= 1.0)   return(0);
   if(step >= 0.1)   return(1);
   if(step >= 0.01)  return(2);
   return(3);
}

//+------------------------------------------------------------------+
//| Position size: AccountEquity * 0.01 / InpCapitalPer001Lot        |
//+------------------------------------------------------------------+
double CalcLot(double equity)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0.0) step = 0.01;
   if(vmin  < 0.0) vmin = 0.01;
   if(vmax <= 0.0) vmax = vmin;

   double lot = equity * 0.01 / InpCapitalPer001Lot;
   if(step > 0.0)
      lot = MathFloor(lot / step) * step;
   if(lot < vmin) lot = vmin;
   if(lot > vmax) lot = vmax;

   return(NormalizeDouble(lot, VolumeStepDigits(step)));
}

//+------------------------------------------------------------------+
//| Does the EA hold an open position on this symbol/magic?          |
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
//| Sum commission over the deals belonging to a position            |
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
//| Close a position (single retry, clear log)                       |
//+------------------------------------------------------------------+
bool ClosePosition(ulong ticket, const string reason)
{
   if(!PositionSelectByTicket(ticket)) return(false);

   double vol  = PositionGetDouble(POSITION_VOLUME);
   double pl   = PositionGetDouble(POSITION_PROFIT)
               + PositionGetDouble(POSITION_SWAP)
               + GetPositionCommission(ticket);

   for(int attempt = 0; attempt < 2; attempt++)
   {
      if(trade.PositionClose(ticket, InpMaxSlippagePoints))
      {
         PrintFormat("CLOSED ticket=%I64u vol=%.2f netP/L=%.2f | reason=%s",
                     ticket, vol, pl, reason);
         return(true);
      }
      else
      {
         PrintFormat("Close failed (attempt %d): retcode=%u err=%d (%s)",
                     attempt + 1, trade.ResultRetcode(), GetLastError(),
                     trade.ResultRetcodeDescription());
         if(attempt == 0) Sleep(500);
      }
   }
   return(false);
}

//+------------------------------------------------------------------+
//| Try to open a market BUY at Ask (SL=0, TP=0).                    |
//+------------------------------------------------------------------+
bool TryOpenLong(void)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double lot    = CalcLot(equity);
   double ask    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(ask <= 0.0)
   {
      Print("TryOpenLong: invalid Ask price, skipping.");
      return(false);
   }

   double marginNeeded = 0.0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lot, ask, marginNeeded))
   {
      PrintFormat("OrderCalcMargin failed err=%d - trade rejected.", GetLastError());
      return(false);
   }
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   if(marginNeeded > freeMargin)
   {
      PrintFormat("WARN insufficient margin: needed=%.2f free=%.2f lot=%.2f - trade rejected, giving up for today.",
                  marginNeeded, freeMargin, lot);
      g_tradedToday = true;
      return(false);
   }

   for(int attempt = 0; attempt < 2; attempt++)
   {
      if(trade.Buy(lot, _Symbol, ask, 0.0, 0.0, InpTradeComment))
      {
         g_entryEquity = equity;
         PrintFormat("BUY opened lot=%.2f @ %.*f | targetPrice=%.*f (+%.2f) | ticket=%I64u",
                     lot, _Digits, ask,
                     _Digits, ask + InpProfitTargetPrice, InpProfitTargetPrice,
                     trade.ResultOrder());
         return(true);
      }
      else
      {
         PrintFormat("Buy failed (attempt %d): retcode=%u err=%d (%s)",
                     attempt + 1, trade.ResultRetcode(), GetLastError(),
                     trade.ResultRetcodeDescription());
         if(attempt == 0) Sleep(500);
      }
   }
   return(false);
}

//+------------------------------------------------------------------+
//| Map a filter name/number string to the enum                      |
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
//| Read optional filter config from <Common>\Files\xau_filter.cfg   |
//|  Lines: filter=ADX  runtag=adx  adxperiod=14  adxthreshold=20    |
//|         donchian=20  linreg=20  sarstep=0.02  sarmax=0.2         |
//+------------------------------------------------------------------+
void ReadFilterConfig(void)
{
   int h = FileOpen("xau_filter.cfg", FILE_READ | FILE_COMMON | FILE_ANSI | FILE_TXT);
   if(h == INVALID_HANDLE)
      return;   // no config file -> keep input defaults

   int loaded = 0;
   while(!FileIsEnding(h))
   {
      string line = FileReadString(h);
      StringTrimLeft(line); StringTrimRight(line);
      if(StringLen(line) == 0) continue;
      if(StringGetCharacter(line, 0) == '#') continue;   // comment
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
//| Expert initialization                                            |
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

   //--- runtime filter params: defaults from inputs, then file override ---
   g_trendFilter    = InpTrendFilter;
   g_adxPeriod      = InpAdxPeriod;
   g_adxThreshold   = InpAdxThreshold;
   g_donchianPeriod = InpDonchianPeriod;
   g_linRegPeriod   = InpLinRegPeriod;
   g_sarStep        = InpSarStep;
   g_sarMax         = InpSarMax;
   g_runTag         = InpRunTag;
   ReadFilterConfig();

   //--- trend indicator handles ---
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

   datetime curD1 = iTime(_Symbol, PERIOD_D1, 0);

   if(InpEnterOnAttachSameDay)
   {
      g_lastD1BarTime = 0;
      g_tradedToday   = false;
   }
   else
   {
      g_lastD1BarTime = curD1;
      g_tradedToday   = true;
   }
   g_entryEquity   = AccountInfoDouble(ACCOUNT_EQUITY);
   g_firstTickDone = false;

   PrintFormat("OnInit OK: magic=%I64d equity=%.2f filter=%s runtag=%s",
               InpMagic, g_entryEquity, EnumToString(g_trendFilter), g_runTag);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_adxHandle != INVALID_HANDLE) IndicatorRelease(g_adxHandle);
   if(g_sarHandle != INVALID_HANDLE) IndicatorRelease(g_sarHandle);
}

//+------------------------------------------------------------------+
//| Trend filter: ADX (ADX>=thr AND +DI>-DI on last completed bar)   |
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

//+------------------------------------------------------------------+
//| Trend filter: Donchian (close[1] > highest high of prior N days) |
//+------------------------------------------------------------------+
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

//+------------------------------------------------------------------+
//| Trend filter: linear-regression slope>0 over last N daily closes |
//+------------------------------------------------------------------+
bool UpByLinReg(void)
{
   int n = g_linRegPeriod;
   if(n < 3) n = 3;
   double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0;
   for(int i = 0; i < n; i++)
   {
      int shift = n - i;                 // i=0 -> oldest, i=n-1 -> newest (shift 1)
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

//+------------------------------------------------------------------+
//| Trend filter: Parabolic SAR below close[1]                       |
//+------------------------------------------------------------------+
bool UpBySar(void)
{
   if(g_sarHandle == INVALID_HANDLE) return(true);
   double sar[];
   if(CopyBuffer(g_sarHandle, 0, 1, 1, sar) < 1) return(false);
   double close1 = iClose(_Symbol, PERIOD_D1, 1);
   return(sar[0] < close1);
}

//+------------------------------------------------------------------+
//| Dispatch the selected trend filter                               |
//+------------------------------------------------------------------+
bool IsUpTrend(void)
{
   switch(g_trendFilter)
   {
      case TREND_ADX:      return(UpByAdx());
      case TREND_DONCHIAN: return(UpByDonchian());
      case TREND_LINREG:   return(UpByLinReg());
      case TREND_SAR:      return(UpBySar());
      default:             return(true);   // TREND_NONE
   }
}

//+------------------------------------------------------------------+
//| Main tick handler                                                |
//+------------------------------------------------------------------+
void OnTick(void)
{
   if(!IsTradingAllowed()) return;

   datetime curD1 = iTime(_Symbol, PERIOD_D1, 0);
   if(curD1 == 0) return;

   //-----------------------------------------------------------------
   // 1) NEW D1 BAR detection
   //-----------------------------------------------------------------
   if(curD1 != g_lastD1BarTime)
   {
      if(g_lastD1BarTime == 0 && !InpEnterOnAttachSameDay && !g_firstTickDone)
      {
         g_lastD1BarTime = curD1;
      }
      else
      {
         ulong tk = 0;
         if(HasOpenPosition(tk))
            ClosePosition(tk, "End-of-day (new D1 bar)");

         g_tradedToday      = false;
         g_entryEquity      = AccountInfoDouble(ACCOUNT_EQUITY);
         g_lastD1BarTime    = curD1;
         g_lastEntryAttempt = 0;
         g_todayUpTrend     = IsUpTrend();
         PrintFormat("New D1 bar %s | entryEquity=%.2f | filter=%s trend=%s",
                     TimeToString(curD1, TIME_DATE), g_entryEquity,
                     EnumToString(g_trendFilter),
                     (g_todayUpTrend ? "UP" : "down(NO-TRADE)"));
      }
      g_firstTickDone = true;
   }

   //-----------------------------------------------------------------
   // 2) WEEKEND GUARD
   //-----------------------------------------------------------------
   if(InpCloseBeforeWeekend)
   {
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      if(dt.day_of_week == 5 && dt.hour >= InpFridayCloseHour)
      {
         ulong tk = 0;
         if(HasOpenPosition(tk))
            ClosePosition(tk,
               StringFormat("Weekend guard (Friday %02d:00)", dt.hour));
         g_tradedToday = true;
      }
   }

   //-----------------------------------------------------------------
   // 3) PROFIT TARGET (price move in $)
   //-----------------------------------------------------------------
   ulong tk = 0;
   if(HasOpenPosition(tk) && PositionSelectByTicket(tk))
   {
      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double bid       = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double priceMove = bid - openPrice;
      double target    = InpProfitTargetPrice;
      if(target > 0.0 && priceMove >= target)
      {
         ClosePosition(tk,
            StringFormat("Profit target hit (price +%.*f >= %.*f)",
                         _Digits, priceMove, _Digits, target));
         g_tradedToday = true;
      }
   }

   //-----------------------------------------------------------------
   // 4) ENTRY (one per day, only when trend filter says UP)
   //-----------------------------------------------------------------
   tk = 0;
   if(g_todayUpTrend && !HasOpenPosition(tk) && !g_tradedToday && g_entryEquity > 0.0)
   {
      datetime now = TimeCurrent();
      if(g_lastEntryAttempt == 0 || now - g_lastEntryAttempt >= InpEntryCooldownSec)
      {
         if(TryOpenLong())
            g_tradedToday = true;
         g_lastEntryAttempt = now;
      }
   }
}

//+------------------------------------------------------------------+
//| Tester: write results to <Common>\Files\results_<runtag>.csv     |
//+------------------------------------------------------------------+
double OnTester(void)
{
   double netProfit = TesterStatistics(STAT_PROFIT);
   double pf        = TesterStatistics(STAT_PROFIT_FACTOR);
   double ddPct     = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   long   trades    = (long)TesterStatistics(STAT_TRADES);
   long   winT      = (long)TesterStatistics(STAT_PROFIT_TRADES);
   long   lossT     = (long)TesterStatistics(STAT_LOSS_TRADES);
   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);

   string fn = StringFormat("results_%s.csv", g_runTag);
   int h = FileOpen(fn, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(h != INVALID_HANDLE)
   {
      FileWrite(h, "tag","filter","netProfit","profitFactor","maxDDpct","trades","wins","losses","finalBalance");
      FileWrite(h, g_runTag, EnumToString(g_trendFilter),
                DoubleToString(netProfit,2), DoubleToString(pf,3),
                DoubleToString(ddPct,2), trades, winT, lossT,
                DoubleToString(balance,2));
      FileClose(h);
      PrintFormat("OnTester CSV written: %s | net=%.2f PF=%.3f DD=%.2f%% trades=%I64d",
                  fn, netProfit, pf, ddPct, trades);
   }
   return(netProfit);
}
//+------------------------------------------------------------------+
