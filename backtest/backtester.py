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
from strategy.swing_pro import SwingProStrategy

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
        
        # Add TP levels specific to SwingPro
        self.take_profit1 = None # For partial exit
        self.take_profit2 = None # For full exit

        # Exit information (to be filled when trade is closed)
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_percent = 0.0
        self.is_open = True
        self.duration = None  # To be filled with timedelta
        self.initial_quantity = quantity # Store initial quantity for partial exit logic
    
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
        
        # Calculate PnL based on the *original* quantity for full close metrics
        # PnL calculation might need adjustment if partial closes happened before.
        # Let's calculate PnL based on the remaining quantity being closed.
        # The total PnL will be accumulated in the backtester.
        pnl_on_close = (exit_price - self.entry_price) * self.quantity
        self.pnl += pnl_on_close # Accumulate PnL if partial close happened

        # PnL percent is tricky after partial closes. Calculate based on initial investment?
        initial_value = self.entry_price * self.initial_quantity
        # Final value needs careful calculation considering partial exits.
        # For simplicity, let's report overall PnL % in metrics, not per trade after partials.
        self.pnl_percent = (self.pnl / initial_value) * 100 if initial_value else 0

        return pnl_on_close # Return PnL for this closing transaction
    
    def partial_close(self,
                      exit_price: float,
                      exit_time: datetime,
                      exit_reason: str,
                      close_quantity: int) -> float:
        """
        Partially close the trade.

        Args:
            exit_price: Exit price for the partial close.
            exit_time: Exit timestamp.
            exit_reason: Reason for partial exit.
            close_quantity: Quantity to close.

        Returns:
            Realized PnL for this partial close.
        """
        if close_quantity <= 0 or close_quantity > self.quantity:
            raise ValueError("Invalid quantity for partial close")

        partial_pnl = (exit_price - self.entry_price) * close_quantity
        self.pnl += partial_pnl # Accumulate PnL
        self.quantity -= close_quantity # Reduce open quantity

        logger.info(f"Partially closed {close_quantity}/{self.initial_quantity} of {self.symbol} at {exit_price:.2f}. Remaining: {self.quantity}. PnL: {partial_pnl:.2f}")

        # Update duration? Or only set on full close? Let's keep duration for full close.
        # Update exit reason? Maybe store a list of exit events? Keep it simple for now.
        # self.exit_reason = f"Partial: {exit_reason}" # Overwrites previous reasons

        return partial_pnl
    
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
            "quantity": self.quantity, # Current open quantity
            "initial_quantity": self.initial_quantity, # Added
            "take_profit": self.take_profit, # Keep for compatibility? Or remove? Let's keep.
            "take_profit1": self.take_profit1, # Added
            "take_profit2": self.take_profit2, # Added
            "stop_loss": self.stop_loss,
            "is_open": self.is_open,
        }
        
        if not self.is_open:
            result.update({
                "exit_price": self.exit_price,
                "exit_time": self.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_reason": self.exit_reason,
                "pnl": self.pnl, # Accumulated PnL
                "pnl_percent": self.pnl_percent, # Overall PnL %
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
        self.strategy_name = config.get('strategy', {}).get('name', 'SwingProStrategy')
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
        # Reset strategy state at the beginning of each backtest run
        self.strategy.reset_position_tracking()
    
    def _load_strategy(self):
        """
        Load the strategy class based on the configuration.
        
        Returns:
            Strategy instance
        """
        try:
            # Currently we only support SwingProStrategy
            # In the future this could be more dynamic based on strategy_name
            if self.strategy_name == 'SwingProStrategy':
                 return SwingProStrategy(self.config)
            else:
                 # Fallback or error for unknown strategy
                 # For now, assume SwingPro
                 logger.warning(f"Unknown strategy '{self.strategy_name}', loading SwingProStrategy.")
                 return SwingProStrategy(self.config)

        except Exception as e:
            logger.error(f"Failed to load strategy {self.strategy_name}: {e}")
            raise

    def load_data(self,
                symbol: str,
                start_date: str,
                end_date: str,
                source: str = 'api',
                # interval: str = 'day', # Removed interval parameter
                csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical data for a symbol. Always uses '1hour' interval via V3 API.

        Args:
            symbol: Trading symbol (format: exchange:ticker)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            source: Data source ('api' or 'csv')
            # interval: Timeframe interval (e.g., 'day', '30minute') # Removed
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

            # Define desired interval details for V3 API
            unit = 'hours'
            interval_value = '1' # For 1-hour candles

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

                    # Fetch historical data from API using V3 parameters
                    logger.info(f"Attempt {retry_attempt+1}/{max_retries}: Fetching historical data (V3 API) for {symbol} ({interval_value}{unit}) from {start_date} to {end_date}")
                    df = self._fetch_historical_data_v3(ticker, exchange, unit, interval_value, start_date, end_date)

                    if df.empty:
                        logger.warning(f"No data returned from V3 API for {symbol}")
                        retry_attempt += 1
                        if retry_attempt < max_retries:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            error_msg = f"No historical data available for {symbol} from {start_date} to {end_date} after {max_retries} attempts (V3 API)"
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
                    logger.error(f"Error fetching data from V3 API (attempt {retry_attempt+1}/{max_retries}): {e}")
                    retry_attempt += 1

                    if retry_attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        error_msg = f"Failed to fetch historical data for {symbol} after {max_retries} attempts (V3 API): {str(e)}"
                        logger.error(error_msg)
                        if self.notification_manager:
                            self.notification_manager.send_system_notification(
                                "Data Fetch Failed",
                                error_msg,
                                "error"
                            )
                        raise ValueError(error_msg)

        elif source == 'csv':
            # ... existing CSV loading code ...
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
        # Ensure data is sorted by time before calculating indicators
        df.sort_index(inplace=True)
        df = self.strategy.prepare_data(df) # Assumes 1-hour data

        # Store the data
        self.data[symbol] = df

        logger.info(f"Loaded {len(df)} data points for {symbol} from {start_date} to {end_date}")
        return df

    # Rename the old fetch method to avoid conflicts
    def _fetch_historical_data_v2(self, ticker: str, exchange: str, interval: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetch historical data from Upstox API V2 (DEPRECATED in favor of V3).
        """
        # ... (Keep the old V2 implementation here for reference or potential fallback) ...
        # Look up the ISIN for this ticker from our mapping or use the instrument key format
        instrument_key = self._get_instrument_key(ticker, exchange)
        # ... rest of V2 implementation ...
        logger.warning("Using deprecated V2 API fetch method.")
        # Construct URL according to the V2 API docs format
        # Example: https://api.upstox.com/v2/historical-candle/NSE_EQ%7CINE848E01016/day/2023-11-13/2023-11-01
        # Using self.base_url which is already set to the V2 base.
        # Map internal interval to V2 API interval string (e.g., '30minute' -> '30minute')
        # V2 uses simple strings like 'day', 'week', 'month', '1minute', '30minute'
        api_interval_map_v2 = {
            '30minute': '30minute',
            'day': 'day', 'week': 'week', 'month': 'month'
        }
        api_interval = api_interval_map_v2.get(interval)
        if not api_interval:
            logger.error(f"Interval '{interval}' not mappable to Upstox V2 API path.")
            return pd.DataFrame()

        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        encoded_instrument_key = requests.utils.quote(instrument_key)
        url = f"{self.base_url}/historical-candle/{encoded_instrument_key}/{api_interval}/{to_date}/{from_date}"
        # ... rest of V2 request logic ...
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                logger.error(f"V2 API request failed with status {response.status_code}: {response.text}")
                return pd.DataFrame()
            data = response.json()
            if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
                candles = data['data']['candles']
                if not candles: return pd.DataFrame()
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                df.set_index('timestamp', inplace=True)
                if 'oi' in df.columns: df.drop(columns=['oi'], inplace=True)
                return df
            else:
                logger.error(f"Invalid V2 response format: {data}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching V2 historical data: {e}")
            return pd.DataFrame()


    def _fetch_historical_data_v3(self, ticker: str, exchange: str, unit: str, interval_value: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetch historical data from Upstox API V3. Handles quarterly limits for hourly data.

        Uses the V3 endpoint: /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}

        Args:
            ticker: Trading symbol
            exchange: Exchange code
            unit: Timeframe unit ('minutes', 'hours', 'days', 'weeks', 'months')
            interval_value: Interval value (e.g., '1', '5', '60')
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with historical data
        """
        # Log the received dates immediately
        logger.debug(f"Entering _fetch_historical_data_v3 for {ticker} ({exchange})")
        logger.debug(f"Received from_date: '{from_date}' (type: {type(from_date)}), to_date: '{to_date}' (type: {type(to_date)})")

        # Validate date format before proceeding
        try:
            start_dt = datetime.strptime(from_date, '%Y-%m-%d')
            end_dt = datetime.strptime(to_date, '%Y-%m-%d')
            logger.debug("Date formats appear valid (YYYY-MM-DD).")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid date format or type detected! from_date='{from_date}', to_date='{to_date}'. Error: {e}")
            return pd.DataFrame()

        instrument_key = self._get_instrument_key(ticker, exchange)
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        encoded_instrument_key = requests.utils.quote(instrument_key)
        v3_base_url = "https://api.upstox.com"

        # Define max duration for hourly data (approx 1 quarter = 90 days)
        max_duration_days = 90
        chunk_delay = 0.5 # Small delay between chunk requests

        all_data_chunks = []

        # Check if chunking is needed (only for 'hours' unit and duration > max_duration_days)
        if unit == 'hours' and (end_dt - start_dt).days > max_duration_days:
            logger.info(f"Hourly data request exceeds {max_duration_days} days. Fetching in chunks...")
            current_to_date = end_dt
            while current_to_date >= start_dt:
                # Calculate start date for this chunk (max 90 days back, but not before original from_date)
                current_from_date = max(start_dt, current_to_date - timedelta(days=max_duration_days - 1)) # -1 to make range inclusive? Check API behavior. Let's try 90 days span.
                current_from_date_str = current_from_date.strftime('%Y-%m-%d')
                current_to_date_str = current_to_date.strftime('%Y-%m-%d')

                logger.debug(f"Fetching chunk: {current_from_date_str} to {current_to_date_str}")

                url = f"{v3_base_url}/v3/historical-candle/{encoded_instrument_key}/{unit}/{interval_value}/{current_to_date_str}/{current_from_date_str}"
                logger.debug(f"Chunk URL: {url}")

                try:
                    response = requests.get(url, headers=headers)
                    if response.status_code != 200:
                        logger.error(f"V3 API chunk request failed ({current_from_date_str} to {current_to_date_str}) with status {response.status_code}: {response.text}")
                        # Attempt to parse error details if JSON
                        try:
                            error_data = response.json()
                            logger.error(f"V3 API Error Details: {error_data}")
                        except json.JSONDecodeError:
                            pass
                        # Decide how to handle chunk failure: stop or skip? Let's stop for now.
                        return pd.DataFrame() # Return empty if any chunk fails

                    data = response.json()
                    if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
                        candles = data['data']['candles']
                        if candles:
                            df_chunk = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                            all_data_chunks.insert(0, df_chunk) # Prepend to maintain order
                            logger.debug(f"Retrieved {len(df_chunk)} candles for chunk.")
                        else:
                            logger.debug(f"No candles returned for chunk {current_from_date_str} to {current_to_date_str}.")
                    else:
                        logger.error(f"Invalid V3 response format for chunk: {data}")
                        return pd.DataFrame() # Stop if format is invalid

                except Exception as e:
                    logger.error(f"Error fetching V3 historical data chunk: {e}")
                    return pd.DataFrame() # Stop on exception

                # Move to the previous chunk
                current_to_date = current_from_date - timedelta(days=1)
                time.sleep(chunk_delay) # Be nice to the API

            # Concatenate all chunks if any were fetched
            if all_data_chunks:
                final_df = pd.concat(all_data_chunks, ignore_index=True)
                # Convert timestamp, set index, drop 'oi'
                final_df['timestamp'] = pd.to_datetime(final_df['timestamp']).dt.tz_localize(None)
                final_df.set_index('timestamp', inplace=True)
                if 'oi' in final_df.columns:
                    final_df.drop(columns=['oi'], inplace=True)
                # Remove potential duplicates from overlapping chunks (if any)
                final_df = final_df[~final_df.index.duplicated(keep='first')]
                final_df.sort_index(inplace=True) # Ensure final sort
                logger.info(f"Successfully retrieved {len(final_df)} total historical candles (V3) in chunks for {exchange}:{ticker}")
                return final_df
            else:
                logger.warning(f"No data retrieved after chunking for {exchange}:{ticker}")
                return pd.DataFrame()

        else:
            # Original logic for single API call (if duration is within limits or unit is not 'hours')
            logger.debug("Request duration within limits or unit is not 'hours'. Making single API call.")
            url = f"{v3_base_url}/v3/historical-candle/{encoded_instrument_key}/{unit}/{interval_value}/{to_date}/{from_date}"
            logger.debug(f"Constructed V3 URL: {url}")

            try:
                response = requests.get(url, headers=headers)

                if response.status_code != 200:
                    logger.error(f"V3 API request failed with status {response.status_code}: {response.text}")
                    try:
                        error_data = response.json()
                        logger.error(f"V3 API Error Details: {error_data}")
                    except json.JSONDecodeError:
                        pass
                    return pd.DataFrame()

                data = response.json()

                if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
                    candles = data['data']['candles']
                    if not candles:
                        logger.warning(f"No candles returned from V3 API for {exchange}:{ticker}")
                        return pd.DataFrame()

                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                    df.set_index('timestamp', inplace=True)
                    if 'oi' in df.columns:
                        df.drop(columns=['oi'], inplace=True)
                    df.sort_index(inplace=True) # Ensure sorted
                    logger.info(f"Retrieved {len(df)} historical candles (V3) for {exchange}:{ticker}")
                    return df
                else:
                    logger.error(f"Invalid V3 response format: {data}")
                    return pd.DataFrame()

            except Exception as e:
                logger.error(f"Error fetching V3 historical data: {e}")
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
        bar_returns = [] # Renamed from daily_returns

        # Reset strategy state before iterating data
        self.strategy.reset_position_tracking()

        # Combine and sort all data by timestamp for chronological processing
        all_data = pd.concat(self.data.values(), keys=self.symbols, names=['symbol', 'timestamp'])
        all_data.reset_index(inplace=True)
        all_data.set_index('timestamp', inplace=True)
        all_data.sort_index(inplace=True)

        # Convert date strings to timestamps if provided
        start_timestamp = pd.Timestamp(start_date) if start_date else None
        end_timestamp = pd.Timestamp(end_date) if end_date else None

        # Filter combined data by date range if provided
        if start_timestamp:
            all_data = all_data[all_data.index.normalize() >= start_timestamp.normalize()]
        if end_timestamp:
            all_data = all_data[all_data.index.normalize() <= end_timestamp.normalize()]

        logger.info(f"Processing {len(all_data)} total data points across {len(self.symbols)} symbols.")

        # Iterate through each timestamp in the combined data
        processed_data_slices = {symbol: pd.DataFrame(columns=self.data[symbol].columns) for symbol in self.symbols}

        for timestamp, group in all_data.groupby(level=0):
            # Process signals for each symbol present at this timestamp
            # Corrected loop: idx is the timestamp index, candle is the row Series
            for idx, candle in group.iterrows():
                # Extract the actual symbol string from the 'symbol' column
                actual_symbol = candle['symbol']

                # Ensure the symbol exists in processed_data_slices (should always be true)
                if actual_symbol not in processed_data_slices:
                    logger.error(f"Logic Error: Symbol '{actual_symbol}' not found in processed_data_slices at {timestamp}")
                    continue # Skip this candle

                # Append current candle to the processed slice for this symbol
                candle_df_row = candle.to_frame().T
                candle_df_row.index = pd.DatetimeIndex([timestamp])
                # Ensure columns match using the correct symbol key
                candle_df_row = candle_df_row.reindex(columns=processed_data_slices[actual_symbol].columns, fill_value=np.nan)

                # Use concat instead of append, handling the initial empty DataFrame case
                if processed_data_slices[actual_symbol].empty:
                    processed_data_slices[actual_symbol] = candle_df_row
                else:
                    processed_data_slices[actual_symbol] = pd.concat([processed_data_slices[actual_symbol], candle_df_row])

                # Generate signal using data up to current candle for this symbol
                # Pass the correct symbol string
                signal = self.strategy.process_candle(processed_data_slices[actual_symbol], actual_symbol)

                if signal:
                    logger.debug(f"Signal received at {timestamp} for {actual_symbol}: {signal}")
                    # --- Handle Entry Signal ---
                    # Use actual_symbol for checking open_trades
                    if signal.signal_type == SignalType.ENTRY and actual_symbol not in open_trades:
                        entry_price = self._apply_slippage(signal.price, True)

                        # Calculate position size based on strategy logic and equity
                        # Use actual_symbol when accessing processed_data_slices
                        equity_at_entry = cash + sum(t.quantity * processed_data_slices[t.symbol]['close'].iloc[-1] for s, t in open_trades.items() if s != actual_symbol and not processed_data_slices[t.symbol].empty) # Approx equity
                        capital_percent = self.strategy.calculate_entry_quantity_percent(processed_data_slices[actual_symbol])
                        position_value = equity_at_entry * capital_percent
                        position_size = int(position_value / entry_price) if entry_price > 0 else 0

                        if position_size > 0:
                            commission = self._calculate_commission(entry_price, position_size)
                            cost = entry_price * position_size + commission

                            if cost <= cash: # Check affordability
                                # Get SL/TP levels from strategy's internal state for this entry
                                # Use actual_symbol
                                pos_details = self.strategy.open_positions.get(actual_symbol, {})
                                stop_loss_level = pos_details.get('stop_loss', signal.stop_loss_level) # Use signal's SL as fallback
                                take_profit1_level = pos_details.get('take_profit1')
                                take_profit2_level = pos_details.get('take_profit2')

                                trade = Trade(
                                    symbol=actual_symbol, # Use actual_symbol
                                    entry_price=entry_price,
                                    entry_time=timestamp,
                                    quantity=position_size,
                                    stop_loss=stop_loss_level
                                )
                                trade.take_profit1 = take_profit1_level
                                trade.take_profit2 = take_profit2_level

                                cash -= cost
                                open_trades[actual_symbol] = trade # Use actual_symbol

                                # Update strategy's internal tracking AFTER trade is confirmed
                                # Use actual_symbol
                                self.strategy.update_open_position(actual_symbol, {
                                    'entry_price': trade.entry_price,
                                    'quantity': trade.quantity,
                                    'initial_quantity': trade.initial_quantity,
                                    'stop_loss': trade.stop_loss,
                                    'take_profit1': trade.take_profit1,
                                    'take_profit2': trade.take_profit2
                                })

                                logger.info(f"{timestamp} - ENTRY: {actual_symbol} Qty: {position_size} @ {entry_price:.2f}, Cost: {cost:.2f}, Cash: {cash:.2f}")
                            else:
                                logger.warning(f"{timestamp} - ENTRY Skipped (Insufficient cash): {actual_symbol} Qty: {position_size} @ {entry_price:.2f}, Cost: {cost:.2f}, Cash: {cash:.2f}")
                                # Remove placeholder from strategy tracking if entry failed
                                # Use actual_symbol
                                self.strategy.update_open_position(actual_symbol, None)
                        else:
                            logger.warning(f"{timestamp} - ENTRY Skipped (Zero quantity): {actual_symbol} @ {entry_price:.2f}")
                            # Remove placeholder from strategy tracking if entry failed
                            # Use actual_symbol
                            self.strategy.update_open_position(actual_symbol, None)


                    # --- Handle Exit Signal ---
                    # Use actual_symbol
                    elif signal.signal_type == SignalType.EXIT and actual_symbol in open_trades:
                        trade = open_trades[actual_symbol] # Use actual_symbol
                        exit_price = self._apply_slippage(signal.price, False)
                        exit_reason = signal.exit_reason.value if signal.exit_reason else "SIGNAL"

                        # Handle Partial Exit
                        if signal.exit_reason == ExitReason.PARTIAL_TAKE_PROFIT:
                            # Close 50% of the initial quantity
                            close_quantity = trade.initial_quantity // 2
                            if close_quantity > 0 and trade.quantity >= close_quantity:
                                commission = self._calculate_commission(exit_price, close_quantity)
                                proceeds = exit_price * close_quantity - commission
                                partial_pnl = trade.partial_close(exit_price, timestamp, exit_reason, close_quantity)
                                cash += proceeds
                                logger.info(f"{timestamp} - PARTIAL EXIT: {actual_symbol} Qty: {close_quantity} @ {exit_price:.2f}, PnL: {partial_pnl:.2f}, Proceeds: {proceeds:.2f}, Cash: {cash:.2f}")
                                # Update strategy tracking with remaining quantity
                                # Use actual_symbol
                                self.strategy.update_open_position(actual_symbol, {
                                    'entry_price': trade.entry_price,
                                    'quantity': trade.quantity, # Remaining quantity
                                    'initial_quantity': trade.initial_quantity,
                                    'stop_loss': trade.stop_loss,
                                    'take_profit1': trade.take_profit1, # Keep levels
                                    'take_profit2': trade.take_profit2
                                })
                            else:
                                logger.warning(f"{timestamp} - Partial exit skipped for {actual_symbol}: Invalid quantity ({close_quantity}) or already partially closed?")

                        # Handle Full Exit
                        else:
                            close_quantity = trade.quantity # Close remaining quantity
                            commission = self._calculate_commission(exit_price, close_quantity)
                            proceeds = exit_price * close_quantity - commission
                            trade_pnl = trade.close(exit_price, timestamp, exit_reason) # Closes remaining quantity
                            cash += proceeds
                            self.trades.append(trade)
                            del open_trades[actual_symbol] # Use actual_symbol
                            # Update strategy tracking (position closed)
                            # Use actual_symbol
                            self.strategy.update_open_position(actual_symbol, None)
                            logger.info(f"{timestamp} - FULL EXIT: {actual_symbol} Qty: {close_quantity} @ {exit_price:.2f}, Reason: {exit_reason}, PnL: {trade_pnl:.2f}, Proceeds: {proceeds:.2f}, Cash: {cash:.2f}")


            # --- Update Equity Curve ---
            # Calculate equity at the end of the timestamp after processing all symbols
            current_equity = cash
            for sym, trade in open_trades.items():
                # Use the last known close price for the symbol up to this timestamp
                if sym in processed_data_slices and not processed_data_slices[sym].empty:
                    current_price = processed_data_slices[sym]['close'].iloc[-1]
                    position_value = trade.quantity * current_price
                    current_equity += position_value
                else:
                    # Handle case where symbol data might not be present at this exact timestamp (unlikely with hourly)
                    logger.warning(f"Missing price data for open trade {sym} at {timestamp}")


            equity_curve.append({'timestamp': timestamp, 'equity': current_equity})

            # Calculate return per bar
            if len(equity_curve) > 1:
                prev_equity = equity_curve[-2]['equity']
                bar_ret = (current_equity - prev_equity) / prev_equity if prev_equity != 0 else 0
                # Store with timestamp for potential analysis, date might be less relevant now
                bar_returns.append({'timestamp': timestamp, 'return': bar_ret})


        # Close any remaining open trades at the end of the backtest
        if not all_data.empty:
            last_timestamp = all_data.index[-1]
            for symbol, trade in list(open_trades.items()):
                 if symbol in processed_data_slices and not processed_data_slices[symbol].empty:
                    last_price = processed_data_slices[symbol]['close'].iloc[-1]
                    exit_price = self._apply_slippage(last_price, False)
                    close_quantity = trade.quantity
                    commission = self._calculate_commission(exit_price, close_quantity)
                    proceeds = exit_price * close_quantity - commission
                    trade_pnl = trade.close(exit_price, last_timestamp, "END_OF_BACKTEST")
                    cash += proceeds
                    self.trades.append(trade)
                    logger.info(f"End of Backtest - Closing {symbol} Qty: {close_quantity} @ {exit_price:.2f}, PnL: {trade_pnl:.2f}, Cash: {cash:.2f}")
                 else:
                     logger.warning(f"Could not close remaining trade for {symbol} at end of backtest - missing data.")


        # Convert equity curve and bar returns to DataFrames
        if equity_curve:
            self.equity_curve = pd.DataFrame(equity_curve).set_index('timestamp')
            
            # Calculate performance metrics
            # Use timestamp index for returns calculation
            bar_returns_df = pd.DataFrame(bar_returns).set_index('timestamp') if bar_returns else \
                              pd.DataFrame(columns=['return']).set_index(pd.Index([], name='timestamp'))
            
            self.metrics = self._calculate_metrics(self.equity_curve, bar_returns_df)
            
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
    
    def _calculate_metrics(self, equity_curve: pd.DataFrame, bar_returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            equity_curve: DataFrame with equity curve
            bar_returns: DataFrame with returns per bar (e.g., hourly)
            
        Returns:
            Dictionary with performance metrics
        """
        # Extract basic metrics
        start_equity = self.starting_capital
        final_equity = equity_curve['equity'].iloc[-1]
        
        # Calculate returns
        total_return = (final_equity - start_equity) / start_equity
        # Annualization needs careful consideration for hourly data.
        # Sticking to daily annualization factor (252) for simplicity.
        trading_days_in_period = len(bar_returns.index.normalize().unique()) if not bar_returns.empty else 0
        annual_return = total_return * (252 / trading_days_in_period) if trading_days_in_period > 0 else 0
        
        # Calculate drawdown
        rolling_max = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Calculate Sharpe Ratio (annualized)
        # Annualization factor for Sharpe should match return annualization. Using sqrt(252) for daily-based annualization.
        if not bar_returns.empty:
            # Calculate daily returns first for Sharpe calculation based on daily volatility
            daily_returns_grouped = bar_returns['return'].resample('D').sum() # Approximate daily return from hourly bars
            sharpe_ratio = np.sqrt(252) * daily_returns_grouped.mean() / daily_returns_grouped.std() \
                if daily_returns_grouped.std() > 0 else 0
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
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trades to CSV
        trades_file = output_path / f"trades_{timestamp_str}.csv"
        trades_data = [trade.to_dict() for trade in self.trades]
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(trades_file, index=False)
        
        # Save equity curve to CSV
        equity_file = output_path / f"equity_{timestamp_str}.csv"
        self.equity_curve.to_csv(equity_file)
        
        # Save metrics to JSON
        metrics_file = output_path / f"metrics_{timestamp_str}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Generate plots
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Equity curve plot
        equity_plot_file = plots_dir / f"equity_curve_{timestamp_str}.png"
        self._plot_equity_curve(equity_plot_file)
        
        # Drawdown plot
        drawdown_plot_file = plots_dir / f"drawdown_{timestamp_str}.png"
        self._plot_drawdown(drawdown_plot_file)
        
        # Generate HTML report
        report_file = output_path / f"report_{timestamp_str}.html"
        self._generate_html_report(report_file, timestamp_str) # Pass timestamp for plot paths
        
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
        if isinstance(self.strategy, SwingProStrategy):
            return {
                'risk_mult_atr_sl': self.strategy.risk_mult_atr_sl,
                'risk_mult_atr_tp1': self.strategy.risk_mult_atr_tp1,
                'risk_mult_atr_tp2': self.strategy.risk_mult_atr_tp2,
                'trail_ema_len': self.strategy.trail_ema_len,
                'ema_50_len': self.strategy.ema_50_len,
                'ema_200_len': self.strategy.ema_200_len,
                'rsi_len': self.strategy.rsi_len,
                'rsi_ob_level': self.strategy.rsi_ob_level,
                'macd_fast': self.strategy.macd_fast,
                'macd_slow': self.strategy.macd_slow,
                'macd_signal': self.strategy.macd_signal,
                'atr_len': self.strategy.atr_len,
                'use_mtf_trend': self.strategy.use_mtf_trend,
                'daily_ema_len': self.strategy.daily_ema_len
            }
        return {}
        
    def _generate_html_report(self, filename: str, timestamp_str: str) -> None:
        """
        Generate HTML backtest report.
        
        Args:
            filename: File to save the report
            timestamp_str: Timestamp string used for plot filenames
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
            trades_html += f"<td>{trade_dict['initial_quantity']}</td>" # Show initial quantity
            if not trade.is_open:
                trades_html += f"<td>{trade_dict['exit_time']}</td>"
                trades_html += f"<td>{trade_dict['exit_price']:.2f}</td>"
                trades_html += f"<td>{trade_dict['exit_reason']}</td>"
                # Color code PnL
                pnl_class = "positive" if trade.pnl > 0 else ("negative" if trade.pnl < 0 else "neutral")
                trades_html += f"<td class='{pnl_class}'>{trade_dict['pnl']:.2f}</td>"
                trades_html += f"<td class='{pnl_class}'>{trade_dict['pnl_percent']:.2f}%</td>"
            else:
                # Should not happen if backtest closes all trades
                trades_html += "<td colspan='5'>Trade still open?</td>"
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
                .neutral {{ color: black; }}
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
                    <img src="plots/equity_curve_{timestamp_str}.png" alt="Equity Curve">
                    <img src="plots/drawdown_{timestamp_str}.png" alt="Drawdown">
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
                        <th>Initial Qty</th>
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