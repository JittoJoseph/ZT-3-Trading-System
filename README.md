# **ZT-3 Trading System: Updated Project Plan**

## **1. Project Overview**

**Objective**: Develop a Python-based algorithmic trading system that implements the Gaussian Channel Strategy for Indian equities using the Upstox API.

**Core Strategy**: Gaussian Channel Strategy with Stochastic RSI and Volume filters

- 5-minute candlestick data

- Gaussian Channel indicators for trend identification

- Stochastic RSI(14,14,3) for signal confirmation

- Fixed take-profit at 4.0x ATR

- Exit on close crossing below Gaussian filter line

**Target Markets**:

- Liquid Indian equities in affordable price range (₹50-₹150)

- Initial focus on PNB with capability to expand to multiple stocks

**Capital Allocation**:

- Initial capital: ₹2,000-₹3,000

- Using approximately 95% of capital per trade to maximize efficiency and minimize impact of transaction costs

- Focus on quality trades rather than quantity to reduce brokerage fees

## **2. System Architecture**

### **2.1 Component Overview**

├── config/           # Configuration files

├── data/             # Market data module

├── strategy/         # Signal detection and strategy logic (isolated)

├── broker/           # Order execution engine

├── paper_trading/    # Paper trading simulation module

├── interface/        # CLI interface

├── utils/            # Utilities, logging, notifications

├── backtest/         # Backtesting framework

└── main.py           # Application entry point

### **2.2 Detailed Component Design**

#### **2.2.1 Configuration Module (config/)**

- YAML or JSON configuration files

- Parameters for:

  - API credentials

  - Target symbols list

  - Strategy parameters

  - Risk management settings

  - Notification preferences

#### **2.2.2 Market Data Module (data/)**

- Connects to Upstox WebSocket API

- Implements tick data buffering

- Aggregates ticks into 5-minute OHLC candles

- Handles market data subscriptions for multiple symbols

- Ensures data integrity and handles connection issues

#### **2.2.3 Strategy Module (strategy/)**

- Completely isolated from other system components

- Implements the Gaussian Channel strategy logic

- Calculates technical indicators

- Detects entry and exit conditions and generates signals

- Designed with a standard interface to allow easy strategy swapping

- Emits structured signals (buy/sell, symbol, size, price)

#### **2.2.4 Execution Module (broker/)**

- Connects to Upstox REST API

- Places market/limit orders with defined parameters

- Manages open positions and tracks P\&L

- Implements position sizing based on capital allocation parameters

- Handles order status updates and rejections

- Supports trading across multiple symbols

#### **2.2.5 Paper Trading Module (paper_trading/)**

- Simulates order execution using real market data

- Maintains virtual portfolio and positions

- Calculates hypothetical P\&L

- Imposes realistic slippage and execution delay

- Stores paper trading history locally

- Provides performance analysis tools

#### **2.2.6 Interface Module (interface/)**

- Command-line interface for starting/stopping the bot

- Configuration management and parameter adjustments

- Real-time status reporting and command processing

- Manual override capabilities

- Switching between paper trading and live trading

#### **2.2.7 Utilities Module (utils/)**

- Logging framework with rotating file logs

- Discord webhook integration for notifications

- Error handling and reporting utilities

- Performance measurement tools

#### **2.2.8 Backtesting Module (backtest/)**

- Historical data retrieval from Upstox or external sources

- Strategy simulation on historical data

- Performance metrics calculation

- Parameter optimization capabilities

- Visual reporting of backtest results

### **2.3 Data Flow**

1. Config module loads parameters and initializes all components

2. Market data module establishes connection and begins streaming data

3. For each symbol, 5-minute candles are constructed and passed to strategy module

4. Strategy module calculates indicators and checks for signal conditions

5. When signal conditions are met, execution module places appropriate orders (or paper trading module records simulated orders)

6. Position tracking and P\&L are continuously updated

7. Trade notifications are sent via Discord

8. Logs are maintained for all system activities

## **3. Trading Strategy**

### **3.1 Gaussian Channel Strategy**

The trading system implements the Gaussian Channel Strategy with Stochastic RSI and Volume filters as defined below:

//@version=5

// --- START OF STRATEGY HEADER ---

// Strategy Name: Demo GPT - Gaussian Channel Strategy v3.13 + Stoch RSI(14,14,3) + Vol + TP (GC Mult 1.2)

// Strategy Title: Demo GPT - Gaussian Channel Strategy v3.13 + Stoch RSI(14,14,3) + Vol + TP (GC Mult 1.2)

// Integration and Modification: Per user request + Enhancements v3.13

// Description: TSL disabled. TP 4.0x ATR. GC Period 144. Stoch RSI(14,14,3). GC Multiplier changed to 1.2.

//              Enters long: GC green, close > hband(mult=1.2), Stoch RSI(14,14,3) > 80, volume > MA.

//              Exits long: Take Profit hit OR close crosses below GC filter line.

//              Long-only, approximated realistic commissions.

// --- END OF STRATEGY HEADER ---

// === STRATEGY SETTINGS ===

strategy(title="Demo GPT - Gaussian Channel Strategy v3.13 + Stoch RSI(14,14,3) + Vol + TP (GC Mult 1.2)", // Updated Title

overlay=true,

calc_on_every_tick=false,

initial_capital=2500,

default_qty_type=strategy.percent_of_equity,

default_qty_value=100,

commission_type=strategy.commission.percent,

commission_value=0.03,

slippage=0,

fill_orders_on_standard_ohlc=true)

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

// Date Filtering Inputs

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

startDate = input.time(timestamp("1 January 2018 00:00 +0000"), "Start Date", group="Date Range")

endDate = input.time(timestamp("31 Dec 2069 23:59 +0000"), "End Date", group="Date Range")

timeCondition = time >= startDate and time <= endDate

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

// Gaussian Channel Indicator - courtesy of @DonovanWall & @e2e4mfck

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

gc_group = "Gaussian Channel Settings"

// Filter function (f_filt9x and f_pole remain unchanged)

f_filt9x (\_a, \_s, \_i) =>

int \_m2 = 0, int \_m3 = 0, int \_m4 = 0, int \_m5 = 0, int \_m6 = 0,

int \_m7 = 0, int \_m8 = 0, int \_m9 = 0, float \_f = .0, \_x = (1 - \_a)

\_m2 := \_i == 9 ? 36  : \_i == 8 ? 28 : \_i == 7 ? 21 : \_i == 6 ? 15 : \_i == 5 ? 10 : \_i == 4 ? 6 : \_i == 3 ? 3 : \_i == 2 ? 1 : 0

\_m3 := \_i == 9 ? 84  : \_i == 8 ? 56 : \_i == 7 ? 35 : \_i == 6 ? 20 : \_i == 5 ? 10 : \_i == 4 ? 4 : \_i == 3 ? 1 : 0

\_m4 := \_i == 9 ? 126 : \_i == 8 ? 70 : \_i == 7 ? 35 : \_i == 6 ? 15 : \_i == 5 ? 5  : \_i == 4 ? 1 : 0

\_m5 := \_i == 9 ? 126 : \_i == 8 ? 56 : \_i == 7 ? 21 : \_i == 6 ? 6  : \_i == 5 ? 1  : 0

\_m6 := \_i == 9 ? 84  : \_i == 8 ? 28 : \_i == 7 ? 7  : \_i == 6 ? 1  : 0

\_m7 := \_i == 9 ? 36  : \_i == 8 ? 8  : \_i == 7 ? 1  : 0

\_m8 := \_i == 9 ? 9   : \_i == 8 ? 1  : 0

\_m9 := \_i == 9 ? 1   : 0

\_f :=   math.pow(\_a, \_i) \* nz(\_s) +

\_i  \*     \_x      \* nz(\_f\[1])      - (\_i >= 2 ?

\_m2 \* math.pow(\_x, 2)  \* nz(\_f\[2]) : 0) + (\_i >= 3 ?

\_m3 \* math.pow(\_x, 3)  \* nz(\_f\[3]) : 0) - (\_i >= 4 ?

\_m4 \* math.pow(\_x, 4)  \* nz(\_f\[4]) : 0) + (\_i >= 5 ?

\_m5 \* math.pow(\_x, 5)  \* nz(\_f\[5]) : 0) - (\_i >= 6 ?

\_m6 \* math.pow(\_x, 6)  \* nz(\_f\[6]) : 0) + (\_i >= 7 ?

\_m7 \* math.pow(\_x, 7)  \* nz(\_f\[7]) : 0) - (\_i >= 8 ?

\_m8 \* math.pow(\_x, 8)  \* nz(\_f\[8]) : 0) + (\_i == 9 ?

\_m9 \* math.pow(\_x, 9)  \* nz(\_f\[9]) : 0)

\_f // return \_f implicitly

f_pole (\_a, \_s, \_i) =>

\_f1 =            f_filt9x(\_a, \_s, 1),      \_f2 = (\_i >= 2 ? f_filt9x(\_a, \_s, 2) : 0), \_f3 = (\_i >= 3 ? f_filt9x(\_a, \_s, 3) : 0)

\_f4 = (\_i >= 4 ? f_filt9x(\_a, \_s, 4) : 0), \_f5 = (\_i >= 5 ? f_filt9x(\_a, \_s, 5) : 0), \_f6 = (\_i >= 6 ? f_filt9x(\_a, \_s, 6) : 0)

\_f7 = (\_i >= 7 ? f_filt9x(\_a, \_s, 7) : 0), \_f8 = (\_i >= 8 ? f_filt9x(\_a, \_s, 8) : 0), \_f9 = (\_i == 9 ? f_filt9x(\_a, \_s, 9) : 0)

\_fn = \_i == 1 ? \_f1 : \_i == 2 ? \_f2 : \_i == 3 ? \_f3 :

\_i == 4     ? \_f4 : \_i == 5 ? \_f5 : \_i == 6 ? \_f6 :

\_i == 7     ? \_f7 : \_i == 8 ? \_f8 : \_i == 9 ? \_f9 : na

\[\_fn, \_f1]

// Gaussian Channel Inputs

src_gc = input(defval=hlc3, title="GC Source", group=gc_group)

int N = input.int(defval=4, title="GC Poles", minval=1, maxval=9, group=gc_group)

int per = input.int(defval=144, title="GC Sampling Period", minval=2, group=gc_group) // Kept at 144

float mult = input.float(defval=1.2, title="GC Filtered True Range Multiplier", minval=0, group=gc_group) // <<< CHANGED to 1.2

bool modeLag  = input.bool(defval=false, title="GC Reduced Lag Mode", group=gc_group)

bool modeFast = input.bool(defval=false, title="GC Fast Response Mode", group=gc_group)

// Gaussian Channel Definitions (will now use mult=1.2)

beta  = (1 - math.cos(4\*math.asin(1)/per)) / (math.pow(1.414, 2/N) - 1)

alpha = - beta + math.sqrt(math.pow(beta, 2) + 2\*beta)

lag = (per - 1)/(2\*N)

srcdata = modeLag ? src_gc + (src_gc - src_gc\[lag]) : src_gc

trdata  = modeLag ? ta.tr(true) + (ta.tr(true) - ta.tr(true)\[lag]) : ta.tr(true)

\[filtn, filt1]     = f_pole(alpha, srcdata, N)

\[filtntr, filt1tr] = f_pole(alpha, trdata,  N)

filt   = modeFast ? (filtn + filt1)/2 : filtn

filttr = modeFast ? (filtntr + filt1tr)/2 : filtntr

hband = filt + filttr\*mult // Uses mult=1.2

lband = filt - filttr\*mult // Uses mult=1.2

// Gaussian Channel Colors and Plots

color1   = #0aff68

color2   = #00752d

color3   = #ff0a5a

color4   = #990032

fcolor   = filt > filt\[1] ? color1 : filt < filt\[1] ? color3 : #cccccc

barcolor_gc = (src_gc > src_gc\[1]) and (src_gc > filt) and (src_gc < hband) ? color1 : (src_gc > src_gc\[1]) and (src_gc >= hband) ? #0aff1b : (src_gc <= src_gc\[1]) and (src_gc > filt) ? color2 :

(src_gc < src_gc\[1]) and (src_gc < filt) and (src_gc > lband) ? color3 : (src_gc < src_gc\[1]) and (src_gc <= lband) ? #ff0a11 : (src_gc >= src_gc\[1]) and (src_gc < filt) ? color4 : #cccccc

filtplot = plot(filt, title="Filter", color=fcolor, linewidth=3)

hbandplot = plot(hband, title="Filtered True Range High Band", color=fcolor)

lbandplot = plot(lband, title="Filtered True Range Low Band", color=fcolor)

fill(hbandplot, lbandplot, title="Channel Fill", color=color.new(fcolor, 80))

barcolor(barcolor_gc)

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

// Stochastic RSI Calculation

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

stoch_group = "Stochastic RSI Settings"

smoothK = input.int(3, "Stoch K Smooth", minval=1, group=stoch_group) // <<< REVERTED to 3

smoothD = input.int(3, "Stoch D Smooth", minval=1, group=stoch_group)

lengthRSI = input.int(14, "RSI Length", minval=1, group=stoch_group) // Kept at 14

lengthStoch = input.int(14, "Stochastic Length", minval=1, group=stoch_group) // Kept at 14

src_stochrsi = input(close, title="Stoch RSI Source", group=stoch_group)

rsi1 = ta.rsi(src_stochrsi, lengthRSI)

k = ta.sma(ta.stoch(rsi1, rsi1, rsi1, lengthStoch), smoothK) // K line uses smoothK=3

stochUpperBand = 80.0

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

// Additional Filters

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

filter_group = "Additional Filters"

useVolumeFilter = input.bool(true, "Use Volume Filter", group=filter_group)

volMALength = input.int(20, "Volume MA Length", minval=1, group=filter_group)

volMA = ta.sma(volume, volMALength)

volumeCondition = not useVolumeFilter or (volume > volMA)

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

// Take Profit Settings and Calculation

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

exit_group = "Exit Settings"

// TSL (Disabled)

useTrailingStop = input.bool(false, "Use ATR Trailing Stop", group=exit_group)

atrLength = input.int(14, "ATR Length (for TSL & TP)", minval=1, group=exit_group)

atrMultiplierTSL = input.float(2.0, "ATR Multiplier (Trailing Stop)", minval=0.1, step=0.1, group=exit_group)

// TP

useTakeProfit = input.bool(true, "Use ATR Take Profit", group=exit_group)

atrMultiplierTP = input.float(4.0, "ATR Multiplier (Take Profit)", minval=0.1, step=0.1, group=exit_group) // Kept at 4.0x

atrValue = ta.atr(atrLength)

// Variables to store exit levels

var float trailStopPrice = na

var float takeProfitLevel = na

// Update TSL only if in a position AND enabled

if strategy.position_size > 0 and useTrailingStop

newStop = filt - atrValue \* atrMultiplierTSL

trailStopPrice := na(trailStopPrice\[1]) or strategy.position_size\[1] == 0 ? newStop : math.max(newStop, trailStopPrice\[1])

else

trailStopPrice := na

// Plot TSL only if enabled

plot(strategy.position_size > 0 and useTrailingStop ? trailStopPrice : na, "Trail Stop", color.orange, style=plot.style_linebr)

// Plot TP

plot(strategy.position_size > 0 and useTakeProfit ? takeProfitLevel : na, "Take Profit", color.green, style=plot.style_linebr)

\

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

// Trading Logic (Improved based on requirements V3.13)

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

// Entry Condition (GC hband uses mult=1.2)

longEntryCondition = (filt > filt\[1]) and (close > hband) and (k > stochUpperBand) and volumeCondition and timeCondition

// Calculate potential Take Profit Level dynamically

if strategy.position_size > 0 and na(takeProfitLevel) // Only set it once per trade

atrEntry = na(atrValue\[1]) ? atrValue : atrValue\[1]

takeProfitLevel := strategy.position_avg_price + atrEntry \* atrMultiplierTP

// Exit Conditions (Filter cross uses GC period 144)

takeProfitExit = useTakeProfit and strategy.position_size > 0 and not na(takeProfitLevel) and high >= takeProfitLevel and timeCondition

trailStopExit = useTrailingStop and strategy.position_size > 0 and not na(trailStopPrice) and low <= trailStopPrice and timeCondition // Inactive

filterCrossExit = ta.crossunder(close, filt) and timeCondition

// Strategy Execution

if (longEntryCondition and strategy.position_size == 0)

strategy.entry("Long", strategy.long)

// Reset levels

trailStopPrice := na

takeProfitLevel := na

// Exit Execution

if strategy.position_size > 0

if takeProfitExit

strategy.close("Long", comment = "TP Exit")

trailStopPrice := na

takeProfitLevel := na

else if trailStopExit // Inactive path

strategy.close("Long", comment = "TSL Exit")

trailStopPrice := na

takeProfitLevel := na

else if filterCrossExit // Active exit path

strategy.close("Long", comment="Filt Cross Exit")

trailStopPrice := na

takeProfitLevel := na

// Reset levels if position closed

if strategy.position_size == 0 and strategy.position_size\[1] > 0

takeProfitLevel := na

trailStopPrice := na

// --- END OF SCRIPT ---

### **3.2 Strategy Summary**

The Gaussian Channel Strategy is a trend-following strategy with the following characteristics:

**Entry Conditions:**

- Gaussian filter line is rising (filt > filt\[1])

- Price closes above the high band (close > hband)

- Stochastic RSI(14,14,3) K-line is above 80

- Volume is above its 20-period moving average

**Exit Conditions:**

- Take profit at 4.0x ATR from entry price

- OR when price closes below the Gaussian filter line

**Key Parameters:**

- Gaussian Channel Period: 144

- Gaussian Channel Multiplier: 1.2

- Stochastic RSI: 14,14,3

- No trailing stop-loss (disabled)

- 5-minute chart timeframe

- Long-only strategy

## **4. Position Sizing and Risk Management**

### **4.1 Capital Allocation**

- Using approximately 95% of available capital per trade

- This approach maximizes capital efficiency while minimizing the impact of transaction costs

- For example, with ₹3,000 capital, around ₹2,850 would be allocated per trade

### **4.2 Position Sizing Logic**

def calculate_position_size(capital, capital_percent, entry_price):

"""

Calculate number of shares based on available capital and percentage allocation

Args:

capital (float): Total available capital

capital_percent (float): Percentage of capital to use (e.g., 95.0)

entry_price (float): Entry price per share

Returns:

int: Number of shares to buy (rounded down)

"""

allocation = capital \* (capital_percent / 100)

shares = int(allocation / entry_price)

return shares

### **4.3 Risk Control Measures**

- Maximum daily loss limit (stop trading if breached)

- Maximum number of trades per day

- Automatic end-of-day closing of all positions

- Circuit breaker mechanism for unusual market conditions

## **5. Discord Notification Structure**

### **5.1 Channel Organization**

#### **5.1.1 Trade Alerts Channel**

- **Purpose**: Real-time alerts of actual trades

- **Content**: Entry/exit signals, prices, position sizes, P\&L per trade

- **Format**: Concise, actionable messages with colored embeds for buy/sell

- **Example**:

```

🟢 BUY SIGNAL - PNB

Entry: ₹82.35

Quantity: 34 shares

Take Profit: ₹83.60

Time: 10:45 AM

Capital Used: ₹2,799.90 (93.3% of capital)
```

#### **5.1.2 Performance Monitoring Channel**

- **Purpose**: Track overall system performance

- **Content**: Daily P\&L summaries, win/loss ratios, drawdown metrics

- **Format**: Daily digest with performance metrics

- **Example**:

```
📊 DAILY SUMMARY - Apr 11, 2025

Trades: 5 (3 wins, 2 losses)

Win Rate: 60.0%

Net P\&L: +₹152.75 (+5.1%)

Max Drawdown: ₹55.40 (1.8%)

Best Trade: PNB +₹86.70

Worst Trade: SBIN -₹39.45
```

#### **5.1.3 Signal Alerts Channel**

- **Purpose**: All strategy signals (even those not traded)

- **Content**: Gaussian Channel status, Stochastic RSI conditions, potential entry/exit points

- **Format**: Brief technical alerts

- **Example**:

```

📝 SIGNAL DETECTED - PNB

GC Filter: Rising

Price above High Band

Stoch RSI: 82.5 (above threshold)

Volume: Above MA

Current Price: ₹82.35

Time: 10:43 AM
```

#### **5.1.4 System Status Channel**

- **Purpose**: System health monitoring

- **Content**: Connection status, API issues, error reports, operational status

- **Format**: Status updates and warnings

- **Example**:

```

🟢 SYSTEM ONLINE - 09:17 AM

Connected to Upstox API

Monitoring 3 symbols

Paper trading mode active
```

```

🔴 MARKET CLOSED - 03:30 PM

Trading session ended

Today's P\&L: +₹152.75

System shutting down
```

### **5.2 Notification Implementation**

- Discord webhook integration for real-time alerts

- Configurable notification levels by channel

- Color-coded messages for quick visual identification

## **6. Technology Stack**

- **Language**: Python 3.10+

- **Key Libraries**:

  - `upstox-api`: Upstox API wrapper

  - `pandas`: Data manipulation and indicator calculation

  - `numpy`: Numerical operations

  - `click`: Command-line interface

  - `pyyaml`: Configuration management

  - `websocket-client`: WebSocket connectivity

  - `requests`: HTTP communications and webhooks

  - `logging`: Logging framework

- **Trading API**: Upstox API

- **Notification**: Discord Webhooks

- **Version Control**: Git + GitHub

- **Development Environment**: Local machine

- **Future Deployment**: AWS EC2

## **7. Testing Framework**

### **7.1 Unit Testing**

- Test individual components in isolation

- Mock external dependencies

- Verify indicator calculations

- Validate signal generation logic

### **7.2 Integration Testing**

- Test interoperation between components

- Verify data flow through the system

- Ensure proper error handling and recovery

### **7.3 Backtesting**

- Test strategy on historical data

- Calculate performance metrics

- Optimize strategy parameters

- Stress test with different market conditions

### **7.4 Paper Trading**

- Live test with real market data but simulated orders

- Compare performance against backtest results

- Identify execution issues before risking capital

- Store paper trading history locally for analysis

## **8. Configuration Schema (Example)**

api:

api_key: "your_upstox_api_key"

api_secret: "your_upstox_api_secret"

symbols:

- ticker: "PNB"

exchange: "NSE"

- ticker: "SBIN"

exchange: "NSE"

- ticker: "BANKBARODA"

exchange: "NSE"

strategy:

name: "GaussianChannel"

params:

gc_period: 144

gc_multiplier: 1.2

stoch_rsi_length: 14

stoch_length: 14

stoch_k_smooth: 3

stoch_upper_band: 80.0

atr_length: 14

atr_tp_multiplier: 4.0

risk:

capital_percent: 95.0

max_daily_loss: 5.0

max_daily_trades: 5

paper_trading:

enabled: true

starting_capital: 3000.0

simulate_slippage: true

slippage_percent: 0.1

execution:

order_type: "LIMIT"

slippage_tolerance: 0.1

retry_attempts: 3

retry_delay: 2

notifications:

discord_webhooks:

trade_alerts: "webhook_url_for_trade_alerts"

performance: "webhook_url_for_performance"

signals: "webhook_url_for_signals"

system_status: "webhook_url_for_system_status"

notification_levels:

trade_alerts: true

performance: true

signals: true

system_status: true

## **9. Implementation Considerations**

### **9.1 Multi-Symbol Management**

- Dynamic watchlist from configuration

- Priority queue for simultaneous signals

- Resource allocation based on symbol liquidity and volatility

- Capital allocation for multi-symbol trading

### **9.2 Performance Optimization**

- Buffered processing of tick data

- Efficient indicator calculation (rolling window)

- Parallel processing for multiple symbols

- Optimized WebSocket handling

- Minimized API calls through batching

### **9.3 Robustness Features**

- Connection loss recovery

- Data validation and sanity checks

- Order execution retry logic

- Heartbeat monitoring

- Extensive logging for troubleshooting

- Graceful shutdown procedure
