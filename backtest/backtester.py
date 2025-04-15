"""
Backtesting Module for ZT-3 Trading System.

This module provides backtesting functionality to test
the ZT-3 Trading System strategies on historical data.
"""

import logging
import time
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import requests
import sys
from dotenv import load_dotenv

# Import strategy and indicator modules
from strategy import SignalType, ExitReason, Signal
from strategy.gaussian_channel import GaussianChannelStrategy

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class Trade:
    """
    Represents a trade during backtesting.
    """
    
    def __init__(self, 
                symbol: str, 
                entry_price: float, 
                entry_time: datetime,
                quantity: int,
                take_profit: Optional[float] = None,
                stop_loss: Optional[float] = None):
        """
        Initialize a new trade.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            entry_time: Entry timestamp
            quantity: Number of shares/contracts
            take_profit: Optional take profit level
            stop_loss: Optional stop loss level
        """
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.quantity = quantity
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        
        # Exit information (to be filled when trade is closed)
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_percent = 0.0
        self.is_open = True
        self.duration = None  # To be filled with timedelta
    
    def close(self, 
             exit_price: float, 
             exit_time: datetime, 
             exit_reason: str) -> float:
        """
        Close the trade.
        
        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Reason for exit
            
        Returns:
            Realized PnL
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        self.is_open = False
        self.duration = exit_time - self.entry_time
        
        # Calculate PnL
        self.pnl = (exit_price - self.entry_price) * self.quantity
        self.pnl_percent = ((exit_price - self.entry_price) / self.entry_price) * 100
        
        return self.pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trade to dictionary for serialization.
        
        Returns:
            Dictionary representation of the trade
        """
        result = {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "quantity": self.quantity,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "is_open": self.is_open,
        }
        
        if not self.is_open:
            result.update({
                "exit_price": self.exit_price,
                "exit_time": self.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_reason": self.exit_reason,
                "pnl": self.pnl,
                "pnl_percent": self.pnl_percent,
                "duration_minutes": self.duration.total_seconds() / 60 if self.duration else None,
            })
        
        return result


class Backtester:
    """
    Backtester for the ZT-3 Trading System.
    
    Tests trading strategies on historical data and generates
    performance metrics and visualizations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backtester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.strategy_name = config.get('strategy', {}).get('name', 'GaussianChannelStrategy')
        self.symbols = [f"{s['exchange']}:{s['ticker']}" for s in config.get('symbols', [])]
        
        # Trading settings
        self.starting_capital = float(config.get('backtest', {}).get('starting_capital', 3000.0))
        self.commission_percent = float(config.get('backtest', {}).get('commission_percent', 0.03))
        self.slippage_percent = float(config.get('backtest', {}).get('slippage_percent', 0.1))
        
        # Store historical data and trades
        self.data = {}  # {symbol: DataFrame}
        self.trades = []  # List of Trade objects
        self.equity_curve = None  # To be created during backtest
        
        # Performance metrics (to be calculated after backtest)
        self.metrics = {}
        
        # Strategy instance
        self.strategy = self._load_strategy()
        
        # API client for accessing historical data
        self.api_key = os.environ.get('UPSTOX_API_KEY') or config.get('api', {}).get('api_key')
        self.api_secret = os.environ.get('UPSTOX_API_SECRET') or config.get('api', {}).get('api_secret')
        self.access_token = os.environ.get('UPSTOX_ACCESS_TOKEN', '')
        
        # API base URLs
        self.base_url = "https://api.upstox.com/v2"
        self.historical_url = f"{self.base_url}/historical-candle"
        
        # Initialize notification manager for discord alerts
        try:
            from utils.notifications import NotificationManager
            self.notification_manager = NotificationManager(config)
        except ImportError:
            logger.warning("NotificationManager not available. Discord notifications will be disabled.")
            self.notification_manager = None
        
        logger.info(f"Backtester initialized for {self.strategy_name} with {len(self.symbols)} symbols")
    
    def _load_strategy(self):
        """
        Load the strategy class based on the configuration.
        
        Returns:
            Strategy instance
        """
        try:
            # Currently we only support GaussianChannelStrategy
            # In the future this could be more dynamic based on strategy_name
            return GaussianChannelStrategy(self.config)
            
        except Exception as e:
            logger.error(f"Failed to load strategy {self.strategy_name}: {e}")
            raise
    
    def load_data(self, 
                symbol: str, 
                start_date: str, 
                end_date: str, 
                source: str = 'api',
                interval: str = '30minute',  # Will be forced to 30minute
                csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical data for a symbol.
        
        Args:
            symbol: Trading symbol (format: exchange:ticker)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            source: Data source ('api' or 'csv')
            interval: Timeframe interval (always '30minute' for Upstox API)
            csv_path: Path to CSV file if source is 'csv'
            
        Returns:
            DataFrame with historical data
        """
        if source == 'api':
            # Parse exchange and ticker from symbol
            parts = symbol.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid symbol format: {symbol}. Expected format: 'exchange:ticker'")
            
            exchange, ticker = parts
            
            # Always use 30minute interval for Upstox API
            interval = '30minute'
            
            # Maximum number of retry attempts
            max_retries = 3
            retry_delay = 5  # seconds
            retry_attempt = 0
            
            while retry_attempt < max_retries:
                try:
                    # Check if we have access token
                    if not self.access_token:
                        error_msg = "No access token available for API request. Please run utils/get_token.py first."
                        logger.error(error_msg)
                        if self.notification_manager:
                            self.notification_manager.send_system_notification(
                                "Authentication Error", 
                                error_msg, 
                                "error"
                            )
                        raise ValueError(error_msg)
                    
                    # Fetch historical data from API
                    logger.info(f"Attempt {retry_attempt+1}/{max_retries}: Fetching historical data for {symbol} from {start_date} to {end_date}")
                    df = self._fetch_historical_data(ticker, exchange, interval, start_date, end_date)
                    
                    if df.empty:
                        logger.warning(f"No data returned from API for {symbol}")
                        retry_attempt += 1
                        if retry_attempt < max_retries:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            error_msg = f"No historical data available for {symbol} from {start_date} to {end_date} after {max_retries} attempts"
                            logger.error(error_msg)
                            if self.notification_manager:
                                self.notification_manager.send_system_notification(
                                    "Data Fetch Error", 
                                    error_msg, 
                                    "error"
                                )
                            raise ValueError(error_msg)
                    
                    # Successfully fetched data
                    break
                    
                except Exception as e:
                    logger.error(f"Error fetching data from API (attempt {retry_attempt+1}/{max_retries}): {e}")
                    retry_attempt += 1
                    
                    if retry_attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        error_msg = f"Failed to fetch historical data for {symbol} after {max_retries} attempts: {str(e)}"
                        logger.error(error_msg)
                        if self.notification_manager:
                            self.notification_manager.send_system_notification(
                                "Data Fetch Failed", 
                                error_msg, 
                                "error"
                            )
                        raise ValueError(error_msg)
        
        elif source == 'csv':
            if not csv_path:
                raise ValueError("csv_path must be provided when source is 'csv'")
            
            # Load data from CSV
            df = pd.read_csv(csv_path, parse_dates=['timestamp'])
            
            # Filter by date range
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
            # Set timestamp as index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
        
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Add technical indicators required by the strategy
        df = self.strategy.prepare_data(df)
        
        # Store the data
        self.data[symbol] = df
        
        logger.info(f"Loaded {len(df)} data points for {symbol} from {start_date} to {end_date}")
        return df
    
    def _fetch_historical_data(self, ticker: str, exchange: str, interval: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetch historical data from Upstox API.
        
        According to the Upstox API docs, the historical candle URL format is:
        https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}
        
        Args:
            ticker: Trading symbol
            exchange: Exchange code
            interval: Timeframe interval
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical data
        """
        # Look up the ISIN for this ticker from our mapping or use the instrument key format
        instrument_key = self._get_instrument_key(ticker, exchange)
        
        # Set up headers as required by the API documentation
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        
        # URL encode the instrument key
        encoded_instrument_key = requests.utils.quote(instrument_key)
        
        # Always use 30minute interval as it's known to work well with the API
        interval = '30minute'
        
        # Construct URL according to the API docs format
        # Example: https://api.upstox.com/v2/historical-candle/NSE_EQ%7CINE848E01016/30minute/2023-11-13/2023-11-12
        url = f"{self.base_url}/historical-candle/{encoded_instrument_key}/{interval}/{to_date}/{from_date}"
        
        logger.info(f"Fetching historical data from: {url}")
        
        try:
            # Make the API request
            response = requests.get(url, headers=headers)
            
            # Check for successful response
            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return pd.DataFrame()
            
            # Parse the response JSON
            data = response.json()
            
            # Check if data was returned successfully
            if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
                candles = data['data']['candles']
                
                if not candles:
                    logger.warning(f"No candles returned for {exchange}:{ticker}")
                    return pd.DataFrame()
                
                # Create DataFrame from candles data
                # According to the documentation, the candle data format is:
                # [timestamp, open, high, low, close, volume, oi]
                df = pd.DataFrame(candles, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'
                ])
                
                # Convert timestamp to datetime and ensure timezone-naive
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                
                # Drop open interest column if it exists and is not needed
                if 'oi' in df.columns:
                    df.drop(columns=['oi'], inplace=True)
                
                logger.info(f"Retrieved {len(df)} historical candles for {exchange}:{ticker}")
                return df
            else:
                logger.error(f"Invalid response format: {data}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def _get_instrument_key(self, ticker: str, exchange: str) -> str:
        """
        Get the correct instrument key for the Upstox API.
        
        According to the field pattern documentation, the format is:
        NSE_EQ|INE160A01022 (exchange_segment|ISIN)
        
        Args:
            ticker: Trading symbol (like PNB)
            exchange: Exchange code (like NSE)
            
        Returns:
            Properly formatted instrument key
        """
        # Hardcoded mapping of ticker to ISIN for common symbols
        ticker_to_isin = {
            'PNB': 'INE160A01022',        # Punjab National Bank
            'SBIN': 'INE062A01020',       # State Bank of India
            'TATASTEEL': 'INE081A01020',  # Tata Steel
            'RELIANCE': 'INE002A01018',   # Reliance Industries
            'INFY': 'INE009A01021',       # Infosys
            'BANKBARODA': 'INE028A01039', # Bank of Baroda
            'CANBK': 'INE476A01022',      # Canara Bank
            'ITC': 'INE154A01025',        # ITC Limited
            'HDFCBANK': 'INE040A01034',   # HDFC Bank
            'TCS': 'INE467B01029',        # Tata Consultancy Services
            'NHPC': 'INE848E01016',       # NHPC Limited
            'RVNL': 'INE415G01027',       # Rail Vikas Nigam Limited
            'YESBANK': 'INE528G01035'     # Yes Bank Limited
        }
        
        # Get ISIN if available
        isin = ticker_to_isin.get(ticker, '')
        
        # Create instrument key according to the format in the API documentation
        if isin:
            # Use ISIN format: NSE_EQ|INE160A01022
            instrument_key = f"{exchange}_EQ|{isin}"
        else:
            # Fallback to using ticker in case ISIN is not available
            # Not ideal, but might work for some symbols
            instrument_key = f"{exchange}_EQ|{ticker}"
        
        return instrument_key

    def _save_working_api_config(self, config: Dict[str, Any]) -> None:
        """
        Save a working API configuration for future use.
        
        Args:
            config: Working API configuration
        """
        try:
            # Create a directory for storing API configs
            config_dir = Path('data/api_configs')
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the config to a file
            config_file = config_dir / 'historical_data_config.json'
            
            # If the file exists, load it first
            if config_file.exists():
                with open(config_file, 'r') as f:
                    existing_configs = json.load(f)
            else:
                existing_configs = []
            
            # Add this config if it doesn't exist
            if config not in existing_configs:
                existing_configs.append(config)
                
                # Save the updated configs
                with open(config_file, 'w') as f:
                    json.dump(existing_configs, f, indent=2)
                    
                logger.info(f"Saved working API config to {config_file}")
        except Exception as e:
            logger.warning(f"Failed to save API config: {e}")

    def run_backtest(self, 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest on loaded data.
        
        Args:
            start_date: Optional start date to filter data (YYYY-MM-DD)
            end_date: Optional end date to filter data (YYYY-MM-DD)
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize backtest variables
        self.trades = []
        open_trades = {}  # symbol -> Trade
        cash = self.starting_capital
        equity = cash
        equity_curve = []
        daily_returns = []
        
        # Convert date strings to timestamps if provided
        start_timestamp = pd.Timestamp(start_date) if start_date else None
        end_timestamp = pd.Timestamp(end_date) if end_date else None
        
        # Process each symbol
        for symbol in self.symbols:
            if symbol not in self.data:
                logger.warning(f"No data loaded for {symbol}, skipping")
                continue
            
            df = self.data[symbol]
            
            # Filter by date range if provided
            if start_timestamp:
                # Convert index to timezone-naive for correct comparison
                if df.index.tz is not None:
                    df_index_naive = df.index.tz_localize(None)
                    df = df[df_index_naive >= start_timestamp]
                else:
                    df = df[df.index >= start_timestamp]
                    
            if end_timestamp:
                # Convert index to timezone-naive for correct comparison
                if df.index.tz is not None:
                    df_index_naive = df.index.tz_localize(None)
                    df = df[df_index_naive <= end_timestamp]
                else:
                    df = df[df.index <= end_timestamp]
            
            # Generate signals using the strategy
            signals = self.strategy.generate_signals(df, symbol)
            
            # Process signals in chronological order
            for signal in sorted(signals, key=lambda s: s.timestamp):
                timestamp = signal.timestamp
                
                if signal.signal_type == SignalType.ENTRY and symbol not in open_trades:
                    # Entry signal
                    entry_price = self._apply_slippage(signal.price, True)
                    
                    # Calculate position size (fixed percentage of capital)
                    capital_percent = self.config.get('risk', {}).get('capital_percent', 95.0) / 100.0
                    position_size = int((cash * capital_percent) / entry_price)
                    if position_size > 0:
                        # Calculate commission
                        commission = self._calculate_commission(entry_price, position_size)
                        
                        # Create trade
                        trade = Trade(
                            symbol=symbol,
                            entry_price=entry_price,
                            entry_time=timestamp,
                            quantity=position_size,
                            take_profit=signal.take_profit_level,
                            stop_loss=signal.stop_loss_level
                        )
                        
                        # Update cash and add to open trades
                        cash -= (entry_price * position_size + commission)
                        open_trades[symbol] = trade
                        
                        logger.debug(f"Backtest: Entry for {symbol} at {entry_price:.2f}, Qty: {position_size}")
                        
                elif signal.signal_type == SignalType.EXIT and symbol in open_trades:
                    # Exit signal
                    exit_price = self._apply_slippage(signal.price, False)
                    trade = open_trades[symbol]
                    exit_reason = signal.exit_reason.value if signal.exit_reason else "SIGNAL"
                    
                    # Close trade
                    trade_pnl = trade.close(exit_price, timestamp, exit_reason)
                    
                    # Update cash and remove from open trades
                    cash += trade.quantity * exit_price - self._calculate_commission(exit_price, trade.quantity)
                    self.trades.append(trade)
                    del open_trades[symbol]
                    
                    logger.debug(f"Backtest: Exit for {symbol} at {exit_price:.2f}, PnL: {trade_pnl:.2f}")
                
                # Calculate equity at this point
                current_equity = cash
                for sym, trade in open_trades.items():
                    # For open trades, use price at current timestamp
                    current_price = df.loc[df.index <= timestamp, 'close'][-1] if sym == symbol else \
                                   self.data[sym].loc[self.data[sym].index <= timestamp, 'close'][-1]
                    
                    # Add unrealized P&L to equity
                    position_value = trade.quantity * current_price
                    current_equity += position_value
                
                # Record equity for equity curve
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': current_equity
                })
                
                # Calculate daily return if this is end of day
                current_date = timestamp.date()
                next_day = False
                if len(equity_curve) > 1:
                    last_date = equity_curve[-2]['timestamp'].date()
                    next_day = current_date != last_date
                
                if next_day:
                    # Find previous day's equity
                    prev_day_equity = next(
                        (point['equity'] for point in reversed(equity_curve[:-1]) 
                         if point['timestamp'].date() < current_date), 
                        self.starting_capital
                    )
                    
                    daily_return = (current_equity - prev_day_equity) / prev_day_equity
                    daily_returns.append({
                        'date': current_date,
                        'return': daily_return
                    })
        
        # Close any remaining open trades at the end of the backtest
        for symbol, trade in list(open_trades.items()):
            last_price = self.data[symbol]['close'][-1]
            exit_price = self._apply_slippage(last_price, False)
            trade_pnl = trade.close(exit_price, self.data[symbol].index[-1], "END_OF_BACKTEST")
            
            # Update cash
            cash += trade.quantity * exit_price - self._calculate_commission(exit_price, trade.quantity)
            self.trades.append(trade)
            
            logger.info(f"Backtest: Closing remaining trade for {symbol} at {exit_price:.2f}, PnL: {trade_pnl:.2f}")
        
        # Convert equity curve and daily returns to DataFrames
        if equity_curve:
            self.equity_curve = pd.DataFrame(equity_curve).set_index('timestamp')
            
            # Calculate performance metrics
            daily_returns_df = pd.DataFrame(daily_returns).set_index('date') if daily_returns else \
                              pd.DataFrame(columns=['return']).set_index(pd.Index([], name='date'))
            
            self.metrics = self._calculate_metrics(self.equity_curve, daily_returns_df)
            
            logger.info(f"Backtest completed: {self.metrics['total_trades']} trades, Final equity: {self.metrics['final_equity']:.2f}")
        else:
            # No trades were executed
            self.equity_curve = pd.DataFrame(columns=['equity'])
            self.metrics = {
                'start_date': start_date or 'N/A',
                'end_date': end_date or 'N/A',
                'duration_days': 0,
                'starting_equity': self.starting_capital,
                'final_equity': self.starting_capital,
                'total_return': 0,
                'total_return_percent': 0,
                'annual_return': 0,
                'annual_return_percent': 0,
                'max_drawdown': 0,
                'max_drawdown_percent': 0,
                'sharpe_ratio': 0,
                'total_trades': 0,
                'win_trades': 0,
                'loss_trades': 0,
                'win_rate': 0,
                'win_rate_percent': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'avg_trade': 0,
                'avg_duration_minutes': 0
            }
            logger.warning("Backtest completed with no trades")
        
        return self.metrics    
    
    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Apply slippage to price.
        
        Args:
            price: Base price
            is_buy: True for buy orders, False for sell orders
            
        Returns:
            Price with slippage applied
        """
        if self.slippage_percent == 0:
            return price
        
        # Calculate slippage amount
        slippage_amount = price * self.slippage_percent / 100.0
        
        # Apply slippage (higher price for buys, lower for sells)
        if is_buy:
            return price + slippage_amount
        else:
            return price - slippage_amount    
    
    def _calculate_commission(self, price: float, quantity: int) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            price: Trade price
            quantity: Number of shares
            
        Returns:
            Commission amount
        """
        trade_value = price * quantity
        return trade_value * self.commission_percent / 100.0
    
    def _calculate_metrics(self, equity_curve: pd.DataFrame, daily_returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            equity_curve: DataFrame with equity curve
            daily_returns: DataFrame with daily returns
            
        Returns:
            Dictionary with performance metrics
        """
        # Extract basic metrics
        start_equity = self.starting_capital
        final_equity = equity_curve['equity'].iloc[-1]
        
        # Calculate returns
        total_return = (final_equity - start_equity) / start_equity
        annual_return = total_return * (252 / len(daily_returns)) if len(daily_returns) > 0 else 0
        
        # Calculate drawdown
        rolling_max = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Calculate Sharpe Ratio (annualized)
        if len(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns['return'].mean() / daily_returns['return'].std() \
                if daily_returns['return'].std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate trade metrics
        win_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        loss_trades = sum(1 for trade in self.trades if trade.pnl <= 0)
        total_trades = len(self.trades)
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average profit and loss
        avg_profit = np.mean([trade.pnl for trade in self.trades if trade.pnl > 0]) \
            if win_trades > 0 else 0
        avg_loss = np.mean([trade.pnl for trade in self.trades if trade.pnl <= 0]) \
            if loss_trades > 0 else 0
        
        # Calculate profit factor
        total_profit = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        total_loss = sum(abs(trade.pnl) for trade in self.trades if trade.pnl <= 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate expectancy and average trade
        expectancy = (win_rate * avg_profit) - ((1 - win_rate) * abs(avg_loss)) \
            if total_trades > 0 else 0
        avg_trade = sum(trade.pnl for trade in self.trades) / total_trades \
            if total_trades > 0 else 0
        
        # Calculate average trade duration
        durations = [trade.duration.total_seconds() / 60 for trade in self.trades if trade.duration]
        avg_duration = np.mean(durations) if durations else 0
        
        return {
            'start_date': equity_curve.index[0].strftime('%Y-%m-%d'),
            'end_date': equity_curve.index[-1].strftime('%Y-%m-%d'),
            'duration_days': (equity_curve.index[-1] - equity_curve.index[0]).days,
            'starting_equity': start_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_percent': total_return * 100,
            'annual_return': annual_return,
            'annual_return_percent': annual_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'win_rate_percent': win_rate * 100,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_trade': avg_trade,
            'avg_duration_minutes': avg_duration
        }
    
    def save_results(self, output_dir: str) -> Dict[str, str]:
        """
        Save backtest results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary with saved file paths
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trades to CSV
        trades_file = output_path / f"trades_{timestamp}.csv"
        trades_data = [trade.to_dict() for trade in self.trades]
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(trades_file, index=False)
        
        # Save equity curve to CSV
        equity_file = output_path / f"equity_{timestamp}.csv"
        self.equity_curve.to_csv(equity_file)
        
        # Save metrics to JSON
        metrics_file = output_path / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Generate plots
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Equity curve plot
        equity_plot_file = plots_dir / f"equity_curve_{timestamp}.png"
        self._plot_equity_curve(equity_plot_file)
        
        # Drawdown plot
        drawdown_plot_file = plots_dir / f"drawdown_{timestamp}.png"
        self._plot_drawdown(drawdown_plot_file)
        
        # Generate HTML report
        report_file = output_path / f"report_{timestamp}.html"
        self._generate_html_report(report_file)
        
        logger.info(f"Backtest results saved to {output_path}")
        
        return {
            'trades': str(trades_file),
            'equity': str(equity_file),
            'metrics': str(metrics_file),
            'equity_plot': str(equity_plot_file),
            'drawdown_plot': str(drawdown_plot_file),
            'report': str(report_file)
        }
    
    def _plot_equity_curve(self, filename: str) -> None:
        """
        Generate equity curve plot.
        
        Args:
            filename: File to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def _plot_drawdown(self, filename: str) -> None:
        """
        Generate drawdown plot.
        
        Args:
            filename: File to save the plot
        """
        rolling_max = self.equity_curve['equity'].cummax()
        drawdown = (self.equity_curve['equity'] - rolling_max) / rolling_max
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, drawdown * 100)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown %')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    @property
    def strategy_params(self) -> Dict[str, Any]:
        """Get the strategy parameters for reporting."""
        if isinstance(self.strategy, GaussianChannelStrategy):
            return {
                'gc_period': self.strategy.gc_period,
                'gc_multiplier': self.strategy.gc_multiplier,
                'gc_poles': self.strategy.gc_poles,
                'stoch_rsi_length': self.strategy.stoch_rsi_length,
                'stoch_length': self.strategy.stoch_length,
                'stoch_k_smooth': self.strategy.stoch_k_smooth,
                'stoch_upper_band': self.strategy.stoch_upper_band,
                'use_volume_filter': self.strategy.use_volume_filter,
                'vol_ma_length': self.strategy.vol_ma_length,
                'use_take_profit': self.strategy.use_take_profit,
                'atr_tp_multiplier': self.strategy.atr_tp_multiplier
            }
        return {}
        
    def _generate_html_report(self, filename: str) -> None:
        """
        Generate HTML backtest report.
        
        Args:
            filename: File to save the report
        """
        # Format metrics for display
        metrics_html = ""
        for key, value in self.metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            metrics_html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"
        
        # Format trades for display
        trades_html = ""
        for i, trade in enumerate(self.trades[:100]):  # Limit to first 100 trades
            trade_dict = trade.to_dict()
            trades_html += "<tr>"
            trades_html += f"<td>{i+1}</td>"
            trades_html += f"<td>{trade_dict['symbol']}</td>"
            trades_html += f"<td>{trade_dict['entry_time']}</td>"
            trades_html += f"<td>{trade_dict['entry_price']:.2f}</td>"
            if not trade.is_open:
                trades_html += f"<td>{trade_dict['exit_time']}</td>"
                trades_html += f"<td>{trade_dict['exit_price']:.2f}</td>"
                trades_html += f"<td>{trade_dict['exit_reason']}</td>"
                # Color code PnL
                pnl_class = "positive" if trade.pnl > 0 else "negative"
                trades_html += f"<td class='{pnl_class}'>{trade_dict['pnl']:.2f}</td>"
                trades_html += f"<td class='{pnl_class}'>{trade_dict['pnl_percent']:.2f}%</td>"
            else:
                trades_html += "<td colspan='5'>Trade still open</td>"
            trades_html += "</tr>"
        
        # Create HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ZT-3 Trading System Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2a3f5f; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .report-section {{ margin-bottom: 30px; }}
                .images {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                .images img {{ width: 48%; }}
            </style>
        </head>
        <body>
            <div class="report-section">
                <h1>ZT-3 Trading System Backtest Report</h1>
                <p>Strategy: {self.strategy_name}</p>
                <p>Period: {self.metrics['start_date']} to {self.metrics['end_date']} ({self.metrics['duration_days']} days)</p>
                <p>Symbols: {', '.join(self.symbols)}</p>
            </div>
            
            <div class="report-section">
                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {metrics_html}
                </table>
            </div>
            
            <div class="report-section">
                <h2>Performance Charts</h2>
                <div class="images">
                    <img src="plots/equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" alt="Equity Curve">
                    <img src="plots/drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" alt="Drawdown">
                </div>
            </div>
            
            <div class="report-section">
                <h2>Trade List</h2>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Symbol</th>
                        <th>Entry Time</th>
                        <th>Entry Price</th>
                        <th>Exit Time</th>
                        <th>Exit Price</th>
                        <th>Exit Reason</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                    </tr>
                    {trades_html}
                </table>
                {f"<p>Showing first 100 out of {len(self.trades)} trades</p>" if len(self.trades) > 100 else ""}
            </div>
            
            <div class="report-section">
                <h2>Strategy Parameters</h2>
                <pre>{json.dumps(self.strategy_params, indent=4)}</pre>
            </div>
            
            <footer>
                <p>Generated by ZT-3 Backtesting Module on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(filename, 'w') as f:
            f.write(html)