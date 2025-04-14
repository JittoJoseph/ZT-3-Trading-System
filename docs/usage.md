# ZT-3 Trading System: Usage Guide

## Table of Contents

- [Command Line Usage](#command-line-usage)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [Trading Modes](#trading-modes)
- [Backtesting](#backtesting)
- [Discord Notifications](#discord-notifications)
- [Common Operations](#common-operations)

## Command Line Usage

The ZT-3 Trading System has a simple command line interface that can be used to control all aspects of its operation.

### Basic Usage

```bash
# Start the trading system with default configuration
python main.py

# Start with specific config file
python main.py --config custom_config.yaml

# Start in paper trading mode (default)
python main.py --paper

# Start in live trading mode (real money)
python main.py --live

# Start with specific logging level
python main.py --log-level DEBUG
```

### Available Command Line Arguments

| Argument      | Short | Default               | Description                                              |
| ------------- | ----- | --------------------- | -------------------------------------------------------- |
| `--config`    | `-c`  | `default_config.yaml` | Configuration file name within config directory          |
| `--paper`     | `-p`  | Enabled by default    | Use paper trading mode (no real money)                   |
| `--live`      | `-l`  | Disabled by default   | Use live trading mode (uses real money)                  |
| `--log-level` | `-d`  | `INFO`                | Set logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL |

### Using the CLI Interface

Once the application is running, you can also use the CLI interface to control it:

```bash
# Start the trading system
python -m interface.cli start

# Stop the trading system
python -m interface.cli stop

# Check current status
python -m interface.cli status

# Validate a configuration file
python -m interface.cli validate path/to/config.yaml
```

## Configuration

The system uses YAML configuration files located in the `config/` directory. The default configuration file is `default_config.yaml`.

### Configuration Sections

- **api**: API credentials for connecting to Upstox
- **symbols**: List of trading symbols to monitor
- **strategy**: Strategy parameters (Gaussian Channel settings)
- **risk**: Risk management settings
- **paper_trading**: Paper trading settings
- **execution**: Order execution parameters
- **notifications**: Discord webhook settings
- **logging**: Log file configuration

### Sample Configuration

```yaml
# API Credentials
api:
  api_key: "${UPSTOX_API_KEY}" # From environment variable
  api_secret: "${UPSTOX_API_SECRET}" # From environment variable
  redirect_uri: "http://localhost:3000/callback"

# Trading Symbols
symbols:
  - ticker: "PNB"
    exchange: "NSE"
    name: "Punjab National Bank"
    min_qty: 1
    tick_size: 0.05
    lot_size: 1
  - ticker: "SBIN"
    exchange: "NSE"
    name: "State Bank of India"

# Strategy Configuration
strategy:
  name: "GaussianChannel"
  params:
    gc_period: 144
    gc_multiplier: 1.2
    stoch_rsi_length: 14
    stoch_length: 14
    stoch_k_smooth: 3
    stoch_upper_band: 80.0
    volume_ma_length: 20
    atr_length: 14
    atr_tp_multiplier: 4.0

# Risk Management Settings
risk:
  max_daily_loss: 5.0 # Percentage of capital
  max_daily_trades: 5
  capital_percent: 95.0 # Percentage per trade
  circuit_breaker_enabled: true
  end_of_day_closure: true
  end_of_day_time: "15:15" # HH:MM in 24-hour format
```

### Environment Variables

Sensitive information like API keys should be stored in environment variables rather than directly in the configuration file. The system will automatically substitute environment variables using the `${VARIABLE_NAME}` syntax.

## Authentication

The system uses OAuth2 for authentication with Upstox.

### Initial Authentication

1. When you first run the system, it will open a browser window to authenticate with Upstox
2. Log in with your Upstox credentials and authorize the application
3. You will be redirected to the callback URL with an authorization code
4. The system will use this code to obtain an access token
5. The token will be saved for future sessions

### Token Management

Access tokens are stored in `broker/upstox_token.json` and are automatically refreshed when needed. Tokens typically expire at 6:00 AM IST the next day.

## Trading Modes

### Paper Trading Mode

Paper trading mode allows you to test the system with real market data but without risking real money.

```bash
python main.py --paper
```

Features:

- Simulates order execution using real market data
- Calculates hypothetical P&L
- Applies realistic slippage and commissions
- Stores trading history for analysis

### Live Trading Mode

Live trading mode connects to Upstox and places real orders with real money.

```bash
python main.py --live
```

⚠️ **WARNING**: Live trading mode uses real money and executes actual trades on your Upstox account.

## Backtesting

The ZT-3 Trading System includes a powerful backtesting engine that allows you to test your strategies against historical data before risking real money.

### Running a Backtest

You can run a backtest using the `run_backtest.py` script:

```bash
python run_backtest.py --config default_config.yaml --start-date 2023-01-01 --end-date 2023-12-31
```

### Available Backtest Arguments

| Argument        | Short | Default               | Description                                             |
| --------------- | ----- | --------------------- | ------------------------------------------------------- |
| `--config`      | `-c`  | `default_config.yaml` | Configuration file name within config directory         |
| `--start-date`  | `-s`  | `2020-01-01`          | Start date for backtest in YYYY-MM-DD format            |
| `--end-date`    | `-e`  | Current date          | End date for backtest in YYYY-MM-DD format              |
| `--data-source` |       | `api`                 | Source of historical data: `api` or `csv`               |
| `--csv-path`    |       | None                  | Path to CSV file(s) when data source is "csv"           |
| `--output-dir`  | `-o`  | `results`             | Output directory for backtest results                   |
| `--log-level`   | `-d`  | `INFO`                | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL    |
| `--params`      | `-p`  | None                  | JSON string with strategy parameters to override config |

### Using CSV Data

You can run backtests against your own CSV files:

```bash
# Using a single CSV file for all symbols
python run_backtest.py --data-source csv --csv-path data/historical/nse_stocks.csv

# Using different CSV files for different symbols
python run_backtest.py --data-source csv --csv-path "NSE:SBIN:data/sbin.csv,NSE:PNB:data/pnb.csv"
```

The CSV files should have at least these columns: timestamp, open, high, low, close, volume.

### Overriding Strategy Parameters

You can override strategy parameters from the command line:

```bash
python run_backtest.py --params '{"gc_period": 120, "gc_multiplier": 1.5, "stoch_rsi_length": 10}'
```

### Backtest Results

After running a backtest, the system will:

1. Print a summary of results to the console
2. Generate detailed HTML and JSON reports in the output directory
3. Optionally send results to Discord if configured

The results include:

- Performance metrics (total return, annual return, Sharpe ratio)
- Drawdown analysis
- Trade statistics (win rate, profit factor, average trade)
- Equity curve and other visualizations

### Example Backtest Command

```bash
python run_backtest.py \
  --config gaussian_channel_config.yaml \
  --start-date 2022-06-01 \
  --end-date 2023-06-01 \
  --output-dir results/june_backtest \
  --params '{"gc_multiplier": 1.3, "stoch_upper_band": 75.0}'
```

## Discord Notifications

The system sends real-time notifications to Discord through webhooks for different types of events.

### Discord Webhook Configuration

In your config file, set up webhooks for each notification channel:

```yaml
notifications:
  discord:
    enabled: true
    webhooks:
      trade_alerts: "${DISCORD_WEBHOOK_TRADE_ALERTS}"
      performance: "${DISCORD_WEBHOOK_PERFORMANCE}"
      signals: "${DISCORD_WEBHOOK_SIGNALS}"
      system_status: "${DISCORD_WEBHOOK_SYSTEM}"
      backtest_results: "${DISCORD_WEBHOOK_BACKTEST}" # For detailed backtest results
  notification_levels:
    trade_alerts: true
    performance: true
    signals: true
    system_status: true
```

### Notification Types

1. **Trade Alerts**: Real-time alerts when trades are executed

   - Buy/sell signals with prices and position sizes
   - Styled with rich embeds (green for buys, purple for sells)

2. **Performance Monitoring**: Daily performance summaries

   - P&L summaries, win/loss ratios, and metrics
   - Presented as daily digest with charts

3. **Signal Alerts**: Real-time strategy signals

   - Technical condition details
   - Can include signals that didn't result in trades

4. **System Status**: System health monitoring
   - Connection status, API issues, error reports
   - Startup/shutdown notifications

## Common Operations

### Starting the System

```bash
python main.py
```

### Checking System Status

```bash
python -m interface.cli status
```

### Stopping the System

```bash
python -m interface.cli stop
```

### Viewing Logs

Logs are stored in the `logs/` directory:

- `logs/zt3.log`: General system logs
- `logs/trades.log`: Trade-specific logs in CSV format

### Managing Multiple Configurations

Create different config files in the `config/` directory:

```bash
python main.py --config weekend_config.yaml
```

### End-of-Day Operations

The system will automatically:

1. Close all positions before market close (configurable time)
2. Generate daily performance summary
3. Send end-of-day notification

### Handling Errors

1. All errors are logged to `logs/zt3.log`
2. Critical errors will trigger Discord notifications
3. Check the logs for detailed information about any issues

### Backups

It's recommended to regularly back up:

- Configuration files in `config/`
- Authentication tokens in `broker/`
- Trading logs in `logs/`
