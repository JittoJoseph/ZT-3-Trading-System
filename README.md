# ZT-3 Trading System

An algorithmic trading system implementing the Gaussian Channel Strategy for Indian equities using the Upstox API.

## Overview

ZT-3 is a Python-based trading system that uses the Gaussian Channel Strategy with Stochastic RSI and Volume filters to identify trading opportunities in liquid Indian equities.

### Core Strategy

- 5-minute candlestick data analysis
- Gaussian Channel indicators (period: 144, multiplier: 1.2)
- Stochastic RSI(14,14,3) for signal confirmation
- Take-profit at 4.0x ATR
- Exit on close crossing below Gaussian filter line

### Target Markets

- Liquid Indian equities in affordable price range (₹50-₹150)
- Initial focus on PNB with capability to expand to multiple stocks

## System Architecture

```
├── config/           # Configuration files
├── data/             # Market data module
├── strategy/         # Signal detection and strategy logic
├── broker/           # Order execution engine
├── paper_trading/    # Paper trading simulation module
├── interface/        # CLI interface
├── utils/            # Utilities, logging, notifications
├── backtest/         # Backtesting framework
└── main.py           # Application entry point
```

## Key Features

- Fully configurable strategy parameters
- Paper trading simulation with realistic slippage
- Live trading through Upstox API
- Discord notifications for trade alerts and system status
- Comprehensive backtesting framework
- Multi-symbol trading capability

## Setup and Configuration

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up configuration in config/settings.yaml:
   - API credentials
   - Target symbols
   - Strategy parameters
   - Risk management settings
   - Notification preferences

## Running the System

Start the trading system:
```
python main.py --mode paper  # For paper trading mode
python main.py --mode live   # For live trading mode
```

## Risk Management

- Uses approximately 95% of available capital per trade
- Maximum daily loss limit and maximum trades per day
- Automatic end-of-day position closing
- Circuit breaker for unusual market conditions

## Technology Stack

- Python 3.10+
- Upstox API for market data and order execution
- Discord webhooks for notifications
