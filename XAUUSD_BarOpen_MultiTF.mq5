//+------------------------------------------------------------------+
//|                                        XAUUSD_BarOpen_MultiTF.mq5 |
//|                                                                  |
//|=== STRATEGY (generalized from XAUUSD_DayOpen_Long_Robust) =======|
//|  - Exactly ONE market trade per bar of the SELECTED timeframe    |
//|    (D1 / H12 / H8 / H4 ...), opened near the bar open.           |
//|  - Direction mode:                                               |
//|      LONG_ONLY  : trend filter UP -> BUY, else skip (v2 behavior)|
//|      SHORT_ONLY : trend filter DOWN -> SELL, else skip           |
//|      BOTH       : filter UP -> BUY, DOWN -> SELL, none -> skip   |
//|  - Profit target / disaster SL as % of entry equity (money-based |
//|    net of swap+commission), broker-side orders attached at entry.|
//|  - Position sizing: 0.01 lot per $100 of equity (mandate).       |
//|  - Position never held across the bar (end-of-bar close) and is  |
//|    force-closed before the weekend.                              |
//|  - All robustness machinery from v2.12 retained: spread gate,    |
//|    entry delay, attempt cap, entry window, circuit breaker,      |
//|    margin-fit lot, Sunday skip.                                  |
//|                                                                  |
//|=== COST WARNING FOR INTRADAY TIMEFRAMES =========================|
//|  Spread is paid PER TRADE. H4 = up to 6 trades/day = up to ~6x   |
//|  the daily cost drag of D1 (~0.3% of equity per trade at 0.01    |
//|  lot/$100). Intraday TFs need either a stricter filter or a      |
//|  larger TP to survive costs. Validate with grid_backtest.py on   |
//|  M1 data before running this on anything but D1.                 |
//+------------------------------------------------------------------+
#property copyright "2026"
#property version   "3.00"
#property description "One trade per bar on a selectable timeframe, long/short/both, money-based TP/SL, robust risk controls."

#include <Trade\Trade.mqh>

enum ENUM_TREND_FILTER
{
   TREND_NONE     = 0, // None (always trade the long side)
   TREND_ADX      = 1, // ADX: ADX>=thr, +DI/-DI decides direction
   TREND_DONCHIAN = 2, // Donchian breakout up/down of prior N bars
   TREND_LINREG   = 3, // Linear-regression slope sign over N bars
   TREND_SAR      = 4  // Parabolic SAR side of close[1]
};

enum ENUM_DIRECTION_MODE
{
   DIR_LONG_ONLY  = 0, // Long only (uptrend bars)
   DIR_SHORT_ONLY = 1, // Short only (downtrend bars)
   DIR_BOTH       = 2  // Both (filter picks the side per bar)
};

//--- inputs ---------------------------------------------------------
input group "=== Strategy ==="
input ENUM_TIMEFRAMES InpTradeTF        = PERIOD_D1; // Trading timeframe (one trade per bar)
input ENUM_DIRECTION_MODE InpDirection  = DIR_LONG_ONLY; // Direction mode
input double  InpProfitPctPerBar        = 1.0;      // Profit target per bar (% of entry equity)
input double  InpCapitalPer001Lot       = 100.0;    // $ equity per 0.01 lot (position sizing)
input bool    InpCloseBeforeWeekend     = true;     // Force-close before weekend
input int     InpFridayCloseHour        = 22;       // Friday server hour to force-close (0-23)

input group "=== Risk control ==="
input double  InpMaxLossPctPerBar       = 9.0;      // Disaster SL (% of entry equity, 0=off - NOT recommended)
input double  InpMaxTotalDDPct          = 30.0;     // Circuit breaker: halt below this % drawdown from peak (0=off)
input int     InpMaxSpreadPoints        = 80;       // Max spread (points) allowed at entry (0=off)
input int     InpEntryDelayMinutes      = 3;        // Wait after bar open before entering
input int     InpEntryWindowPct         = 50;       // Entry window: % of bar duration (no late entries after)
input int     InpMaxEntryAttempts       = 20;       // Max order attempts per bar
input bool    InpSkipSunday             = true;     // Skip Sunday bars/entries
input double  InpMarginUseFraction      = 0.8;      // Max fraction of free margin the position may use

input group "=== Trend Filter ==="
input ENUM_TREND_FILTER InpTrendFilter    = TREND_ADX; // Trend filter (direction source)
input int     InpAdxPeriod                = 14;     // ADX period
input double  InpAdxThreshold             = 20.0;   // ADX min strength to count as trending
input int     InpDonchianPeriod           = 20;     // Donchian lookback (bars)
input int     InpLinRegPeriod             = 20;     // Linear-regression lookback (bars)
input double  InpSarStep                  = 0.02;   // Parabolic SAR step
input double  InpSarMax                   = 0.2;    // Parabolic SAR max
input int     InpEntryCooldownSec         = 30;     // Cooldown (sec) between entry retries
input string  InpRunTag                   = "mtf";  // Results-file tag (for batch backtests)

input group "=== Execution ==="
input long    InpMagic                = 20260705; // Magic number (EA id) - differs from v2.12
input ulong   InpMaxSlippagePoints    = 50;       // Max slippage (points)
input string  InpTradeComment         = "XAU_BarOpen_MTF"; // Trade comment

//--- globals --------------------------------------------------------
CTrade        trade;
datetime      g_lastBarTime      = 0;
bool          g_tradedThisBar    = false;
double        g_entryEquity      = 0.0;
double        g_targetMoney      = 0.0;
double        g_maxLossMoney     = 0.0;
int           g_adxHandle        = INVALID_HANDLE;
int           g_sarHandle        = INVALID_HANDLE;
int           g_barDirection     = 0;     // +1 long, -1 short, 0 no-trade
bool          g_dirEvaluated     = false;
datetime      g_lastEntryAttempt = 0;
int           g_entryAttempts    = 0;
bool          g_haltedByDD       = false;
string        g_peakGvName       = "";

// runtime params (inputs; overridable via xau_filter.cfg)
ENUM_TREND_FILTER g_trendFilter = TREND_NONE;
int     g_adxPeriod      = 14;
double  g_adxThreshold   = 20.0;
int     g_donchianPeriod = 20;
int     g_linRegPeriod   = 20;
double  g_sarStep        = 0.02;
double  g_sarMax         = 0.2;
string  g_runTag         = "mtf";
double  g_profitPct      = 1.0;
double  g_maxLossPct     = 9.0;

//+------------------------------------------------------------------+
bool IsTradingAllowed(void)
{
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))   return(false);
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))             return(false);
   if((ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE)
        == SYMBOL_TRADE_MODE_DISABLED)                return(false);
   return(true);
}

ENUM_ORDER_TYPE_FILLING DetectFilling(void)
{
   int filling = (int)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_IOC) != 0) return(ORDER_FILLING_IOC);
   if((filling & SYMBOL_FILLING_FOK) != 0) return(ORDER_FILLING_FOK);
   return(ORDER_FILLING_RETURN);
}

int VolumeStepDigits(double step)
{
   if(step >= 1.0)   return(0);
   if(step >= 0.1)   return(1);
   if(step >= 0.01)  return(2);
   return(3);
}

double MoneyPerUnitPerLot(void)
{
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickValue <= 0.0 || tickSize <= 0.0)
      return(SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE));
   return(tickValue / tickSize);
}

//+------------------------------------------------------------------+
//| Mandate sizing, shrunk to fit free margin                         |
//+------------------------------------------------------------------+
double CalcLot(double equity, double price, ENUM_ORDER_TYPE type)
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

   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE) * InpMarginUseFraction;
   for(int i = 0; i < 200 && lot >= vmin; i++)
   {
      double need = 0.0;
      if(!OrderCalcMargin(type, _Symbol, lot, price, need))
         return(0.0);
      if(need <= freeMargin)
         return(lot);
      lot = NormalizeDouble(lot - step, VolumeStepDigits(step));
   }
   return(0.0);
}

bool HasOpenPosition(ulong &ticketOut)
{
   ticketOut = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
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

double PositionNetProfit(ulong ticket)
{
   if(!PositionSelectByTicket(ticket)) return(0.0);
   return(PositionGetDouble(POSITION_PROFIT)
        + PositionGetDouble(POSITION_SWAP)
        + GetPositionCommission(ticket));
}

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
//| Peak-equity high-water mark / circuit breaker                     |
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
   if(equity > LoadPeakEquity(equity))
      GlobalVariableSet(g_peakGvName, equity);
}

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

bool SpreadOk(void)
{
   if(InpMaxSpreadPoints <= 0) return(true);
   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   return(spread > 0 && spread <= InpMaxSpreadPoints);
}

//+------------------------------------------------------------------+
//| Open the bar's trade (dir=+1 long / -1 short) with SL/TP attached |
//+------------------------------------------------------------------+
bool TryOpen(int dir)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double ask    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0) return(false);

   ENUM_ORDER_TYPE type = (dir > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   double entry = (dir > 0 ? ask : bid);

   double lot = CalcLot(equity, entry, type);
   if(lot <= 0.0)
   {
      PrintFormat("WARN: minimum lot does not fit free margin (equity=%.2f) - skipping this bar.", equity);
      g_tradedThisBar = true;
      return(false);
   }

   double mpu = MoneyPerUnitPerLot();
   if(mpu <= 0.0) return(false);

   double targetMoney  = equity * g_profitPct  / 100.0;
   double maxLossMoney = equity * g_maxLossPct / 100.0;
   double tpMove = targetMoney / (lot * mpu);
   double slMove = (g_maxLossPct > 0.0 ? maxLossMoney / (lot * mpu) : 0.0);

   double point   = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double minStop = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   double sprd    = ask - bid;

   tpMove = MathMax(tpMove, minStop);
   if(slMove > 0.0) slMove = MathMax(slMove, minStop + sprd);

   double tpPrice, slPrice = 0.0;
   if(dir > 0)
   {
      tpPrice = NormalizeDouble(entry + tpMove, _Digits);
      if(slMove > 0.0) slPrice = NormalizeDouble(MathMax(entry - slMove, 0.0), _Digits);
   }
   else
   {
      tpPrice = NormalizeDouble(entry - tpMove, _Digits);
      if(slMove > 0.0) slPrice = NormalizeDouble(entry + slMove, _Digits);
      if(tpPrice <= 0.0) return(false);
   }

   for(int attempt = 0; attempt < 2; attempt++)
   {
      bool ok = (dir > 0)
         ? trade.Buy (lot, _Symbol, entry, slPrice, tpPrice, InpTradeComment)
         : trade.Sell(lot, _Symbol, entry, slPrice, tpPrice, InpTradeComment);
      if(ok)
      {
         g_entryEquity  = equity;
         g_targetMoney  = targetMoney;
         g_maxLossMoney = maxLossMoney;
         PrintFormat("%s opened lot=%.2f @ %.*f | TP=%.*f (+$%.2f=%.2f%%) | SL=%.*f (-$%.2f=%.2f%%) | ticket=%I64u",
                     (dir > 0 ? "BUY" : "SELL"), lot, _Digits, entry,
                     _Digits, tpPrice, targetMoney, g_profitPct,
                     _Digits, slPrice, maxLossMoney, g_maxLossPct,
                     trade.ResultOrder());
         return(true);
      }
      PrintFormat("Open failed (attempt %d): retcode=%u err=%d (%s)",
                  attempt + 1, trade.ResultRetcode(), GetLastError(),
                  trade.ResultRetcodeDescription());
      if(attempt == 0) Sleep(500);
   }
   return(false);
}

//+------------------------------------------------------------------+
//| Config file (shared format with the v2 EA) + tf/dir extras        |
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

void ReadFilterConfig(void)
{
   int h = FileOpen("xau_filter.cfg", FILE_READ | FILE_COMMON | FILE_ANSI | FILE_TXT);
   if(h == INVALID_HANDLE) return;
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
      if(key == "filter")           { g_trendFilter    = FilterFromString(val); loaded++; }
      else if(key == "runtag")      { g_runTag         = val; loaded++; }
      else if(key == "adxperiod")   { g_adxPeriod      = (int)StringToInteger(val); loaded++; }
      else if(key == "adxthreshold"){ g_adxThreshold   = StringToDouble(val); loaded++; }
      else if(key == "donchian")    { g_donchianPeriod = (int)StringToInteger(val); loaded++; }
      else if(key == "linreg")      { g_linRegPeriod   = (int)StringToInteger(val); loaded++; }
      else if(key == "sarstep")     { g_sarStep        = StringToDouble(val); loaded++; }
      else if(key == "sarmax")      { g_sarMax         = StringToDouble(val); loaded++; }
      else if(key == "tppct")       { g_profitPct      = StringToDouble(val); loaded++; }
      else if(key == "slpct")       { g_maxLossPct     = StringToDouble(val); loaded++; }
   }
   FileClose(h);
   if(loaded > 0)
      PrintFormat("Filter config loaded (%d fields): filter=%s runtag=%s tp=%.2f sl=%.2f",
                  loaded, EnumToString(g_trendFilter), g_runTag, g_profitPct, g_maxLossPct);
}

//+------------------------------------------------------------------+
int OnInit()
{
   if(!IsTradingAllowed())
   {
      Print("OnInit: trading is NOT allowed.");
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
   g_profitPct      = InpProfitPctPerBar;
   g_maxLossPct     = InpMaxLossPctPerBar;
   ReadFilterConfig();

   if(g_trendFilter == TREND_ADX)
   {
      g_adxHandle = iADX(_Symbol, InpTradeTF, g_adxPeriod);
      if(g_adxHandle == INVALID_HANDLE)
         PrintFormat("WARN: iADX handle failed err=%d", GetLastError());
   }
   if(g_trendFilter == TREND_SAR)
   {
      g_sarHandle = iSAR(_Symbol, InpTradeTF, g_sarStep, g_sarMax);
      if(g_sarHandle == INVALID_HANDLE)
         PrintFormat("WARN: iSAR handle failed err=%d", GetLastError());
   }

   g_peakGvName    = StringFormat("XAU_MTF_peak_%I64d", InpMagic);
   g_lastBarTime   = iTime(_Symbol, InpTradeTF, 0);
   g_tradedThisBar = true;      // wait for the next full bar
   g_dirEvaluated  = false;
   g_entryAttempts = 0;
   g_haltedByDD    = false;
   UpdatePeakEquity(AccountInfoDouble(ACCOUNT_EQUITY));

   PrintFormat("OnInit OK v3.00: tf=%s dir=%s magic=%I64d filter=%s tp=%.2f%% sl=%.2f%% maxDD=%.1f%%",
               EnumToString(InpTradeTF), EnumToString(InpDirection), InpMagic,
               EnumToString(g_trendFilter), g_profitPct, g_maxLossPct, InpMaxTotalDDPct);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(g_adxHandle != INVALID_HANDLE) IndicatorRelease(g_adxHandle);
   if(g_sarHandle != INVALID_HANDLE) IndicatorRelease(g_sarHandle);
}

//+------------------------------------------------------------------+
//| Trend verdict on the trading TF: +1 up, -1 down, 0 no-trade       |
//+------------------------------------------------------------------+
int TrendByAdx(void)
{
   if(g_adxHandle == INVALID_HANDLE) return(0);
   double adx[], plus[], minus[];
   if(CopyBuffer(g_adxHandle, 0, 1, 1, adx)   < 1) return(0);
   if(CopyBuffer(g_adxHandle, 1, 1, 1, plus)  < 1) return(0);
   if(CopyBuffer(g_adxHandle, 2, 1, 1, minus) < 1) return(0);
   if(adx[0] < g_adxThreshold) return(0);
   return(plus[0] > minus[0] ? 1 : -1);
}

int TrendByDonchian(void)
{
   int n = MathMax(g_donchianPeriod, 2);
   double maxHigh = -DBL_MAX, minLow = DBL_MAX;
   for(int s = 2; s <= n + 1; s++)
   {
      maxHigh = MathMax(maxHigh, iHigh(_Symbol, InpTradeTF, s));
      minLow  = MathMin(minLow,  iLow(_Symbol, InpTradeTF, s));
   }
   double close1 = iClose(_Symbol, InpTradeTF, 1);
   if(close1 > maxHigh) return(1);
   if(close1 < minLow)  return(-1);
   return(0);
}

int TrendByLinReg(void)
{
   int n = MathMax(g_linRegPeriod, 3);
   double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
   for(int i = 0; i < n; i++)
   {
      double y = iClose(_Symbol, InpTradeTF, n - i);
      if(y <= 0.0) return(0);
      double x = (double)i;
      sumX += x; sumY += y; sumXY += x * y; sumX2 += x * x;
   }
   double denom = (double)n * sumX2 - sumX * sumX;
   if(denom == 0.0) return(0);
   double slope = ((double)n * sumXY - sumX * sumY) / denom;
   if(slope > 0.0) return(1);
   if(slope < 0.0) return(-1);
   return(0);
}

int TrendBySar(void)
{
   if(g_sarHandle == INVALID_HANDLE) return(0);
   double sar[];
   if(CopyBuffer(g_sarHandle, 0, 1, 1, sar) < 1) return(0);
   double close1 = iClose(_Symbol, InpTradeTF, 1);
   return(sar[0] < close1 ? 1 : -1);
}

int TrendVerdict(void)
{
   switch(g_trendFilter)
   {
      case TREND_ADX:      return(TrendByAdx());
      case TREND_DONCHIAN: return(TrendByDonchian());
      case TREND_LINREG:   return(TrendByLinReg());
      case TREND_SAR:      return(TrendBySar());
      default:             return(1);   // TREND_NONE: long side
   }
}

//+------------------------------------------------------------------+
//| Verdict -> tradable direction under the selected mode             |
//+------------------------------------------------------------------+
int BarDirection(void)
{
   int v = TrendVerdict();
   switch(InpDirection)
   {
      case DIR_LONG_ONLY:  return(v > 0 ?  1 : 0);
      case DIR_SHORT_ONLY: return(v < 0 ? -1 : 0);
      case DIR_BOTH:       return(v);
   }
   return(0);
}

//+------------------------------------------------------------------+
void OnTick(void)
{
   if(!IsTradingAllowed()) return;

   datetime curBar = iTime(_Symbol, InpTradeTF, 0);
   if(curBar == 0) return;

   // 1) new bar: close leftover, reset bar state
   if(curBar != g_lastBarTime)
   {
      ulong tk = 0;
      if(HasOpenPosition(tk))
         ClosePosition(tk, "End-of-bar (new bar)");
      g_tradedThisBar    = false;
      g_lastBarTime      = curBar;
      g_lastEntryAttempt = 0;
      g_entryAttempts    = 0;
      g_dirEvaluated     = false;
   }

   if(!g_dirEvaluated)
   {
      g_barDirection = BarDirection();
      g_dirEvaluated = true;
      PrintFormat("Bar %s | equity=%.2f | filter=%s dir=%s",
                  TimeToString(curBar), AccountInfoDouble(ACCOUNT_EQUITY),
                  EnumToString(g_trendFilter),
                  (g_barDirection > 0 ? "LONG" : g_barDirection < 0 ? "SHORT" : "none(NO-TRADE)"));
   }

   // 2) weekend guard
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(InpCloseBeforeWeekend && dt.day_of_week == 5 && dt.hour >= InpFridayCloseHour)
   {
      ulong tk = 0;
      if(HasOpenPosition(tk))
         ClosePosition(tk, StringFormat("Weekend guard (Friday %02d:00)", dt.hour));
      g_tradedThisBar = true;
   }

   // 3) money-based target / emergency loss cap (backs up broker SL/TP)
   ulong tk = 0;
   if(HasOpenPosition(tk))
   {
      double net = PositionNetProfit(tk);
      if(g_targetMoney > 0.0 && net >= g_targetMoney)
      {
         ClosePosition(tk, StringFormat("Profit target hit (net $%.2f >= $%.2f = %.2f%%)",
                                        net, g_targetMoney, g_profitPct));
         g_tradedThisBar = true;
      }
      else if(g_maxLossMoney > 0.0 && net <= -g_maxLossMoney)
      {
         ClosePosition(tk, StringFormat("EMERGENCY loss cap (net $%.2f <= -$%.2f = %.2f%%)",
                                        net, g_maxLossMoney, g_maxLossPct));
         g_tradedThisBar = true;
      }
   }

   // 4) entry, gated by every robustness check
   tk = 0;
   if(g_tradedThisBar || HasOpenPosition(tk) || g_barDirection == 0) return;
   if(CircuitBreakerTripped())                                       return;
   if(InpSkipSunday && dt.day_of_week == 0)                          return;

   datetime now = TimeCurrent();
   long barSecs = (long)PeriodSeconds(InpTradeTF);
   long into    = (long)(now - curBar);
   if(into < (long)InpEntryDelayMinutes * 60)                        return;
   if(InpEntryWindowPct > 0 && into > barSecs * InpEntryWindowPct / 100)
   {
      g_tradedThisBar = true;   // window closed
      return;
   }
   if(g_entryAttempts >= InpMaxEntryAttempts)
   {
      g_tradedThisBar = true;
      PrintFormat("Entry attempts exhausted (%d) - giving up for this bar.", g_entryAttempts);
      return;
   }
   if(!SpreadOk())                                                   return;
   if(g_lastEntryAttempt != 0 && now - g_lastEntryAttempt < InpEntryCooldownSec)
      return;

   g_entryAttempts++;
   g_lastEntryAttempt = now;
   if(TryOpen(g_barDirection))
      g_tradedThisBar = true;
}

//+------------------------------------------------------------------+
//| Tester: DD-penalized score + per-pass CSV incl. tf/dir            |
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

   double score = netProfit * (100.0 - MathMin(ddPct, 100.0)) / 100.0;

   string fn = StringFormat("results_%s_%s_%s_tp%s_sl%s.csv", g_runTag,
                            EnumToString(InpTradeTF), EnumToString(InpDirection),
                            DoubleToString(g_profitPct, 2), DoubleToString(g_maxLossPct, 2));
   int h = FileOpen(fn, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(h != INVALID_HANDLE)
   {
      FileWrite(h, "tag","tf","dir","filter","tpPct","slPct","netProfit","profitFactor",
                   "maxDDpct","sharpe","score","trades","wins","losses","finalBalance");
      FileWrite(h, g_runTag, EnumToString(InpTradeTF), EnumToString(InpDirection),
                EnumToString(g_trendFilter),
                DoubleToString(g_profitPct,2), DoubleToString(g_maxLossPct,2),
                DoubleToString(netProfit,2), DoubleToString(pf,3),
                DoubleToString(ddPct,2), DoubleToString(sharpe,3),
                DoubleToString(score,2),
                trades, winT, lossT, DoubleToString(balance,2));
      FileClose(h);
   }
   return(score);
}
//+------------------------------------------------------------------+
