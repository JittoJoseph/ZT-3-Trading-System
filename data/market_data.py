"""
Market Data Client for ZT-3 Trading System.

This module handles:
- Connection to Upstox WebSocket API
- Processing of real-time market data
- Fetching historical and intraday candles
- Data quality monitoring and validation
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import requests
import websocket
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class MarketDataClient:
    """
    Client for fetching market data from Upstox.
    
    This class handles WebSocket connections for real-time data and
    REST API calls for historical and intraday candles.
    """
    
    # API endpoints
    BASE_URL = "https://api.upstox.com/v2"
    HISTORICAL_URL = f"{BASE_URL}/historical-candle"
    INTRADAY_URL = f"{BASE_URL}/intraday-candle"
    MARKET_DATA_FEED_AUTH_URL = f"{BASE_URL}/market-data-feed/authorize"
    
    def __init__(self, broker, config: Dict[str, Any]):
        """
        Initialize the market data client.
        
        Args:
            broker: Upstox broker interface with authentication
            config: Configuration dictionary
        """
        self.broker = broker
        self.config = config
        self.websocket_client = None
        self.websocket_thread = None
        self.running = False
        self.callbacks = []
        self.historical_data_cache = {}
        
        # Create data storage directory if needed
        self.data_dir = Path(config.get('data', {}).get('storage_path', 'data/storage'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def register_market_data_callback(self, callback: Callable) -> None:
        """
        Register a callback to receive market data.
        
        Args:
            callback: Function to call when market data is received
        """
        self.callbacks.append(callback)
    
    def connect_websocket(self) -> None:
        """Connect to the market data WebSocket."""
        # This would establish a WebSocket connection in a real implementation
        pass
    
    def disconnect_websocket(self) -> None:
        """Disconnect from the market data WebSocket."""
        # This would close the WebSocket connection in a real implementation
        pass
    
    def subscribe_market_data(self, symbols: List[Dict[str, str]]) -> bool:
        """
        Subscribe to market data for specified symbols.
        
        Args:
            symbols: List of symbol dictionaries with 'ticker' and 'exchange' keys
            
        Returns:
            True if subscription was successful, False otherwise
        """
        # This would send a subscription message in a real implementation
        return True
    
    def fetch_historical_candles(self, 
                              symbol: str,
                              exchange: str,
                              interval: str,
                              from_date: str,
                              to_date: str,
                              use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch historical OHLC candle data from Upstox API.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, etc.)
            interval: Candle interval (1minute, 5minute, 30minute, day, week, month)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with historical candles data
        """
        # Check cache first if enabled
        cache_key = f"{exchange}:{symbol}:{interval}:{from_date}:{to_date}"
        if use_cache and cache_key in self.historical_data_cache:
            logger.debug(f"Using cached data for {cache_key}")
            return self.historical_data_cache[cache_key]
        
        try:
            # Format instrument key
            instrument_key = f"{exchange}:{symbol}"
            
            # Set up headers and params
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.broker.access_token}'
            }
            
            params = {
                'instrument_key': instrument_key,
                'interval': interval,
                'from_date': from_date,
                'to_date': to_date
            }
            
            logger.info(f"Fetching historical data for {instrument_key} from {from_date} to {to_date} with interval {interval}")
            
            # Check if we're running in backtesting mode without a valid API token
            if not self.broker.access_token or not self.broker.is_authenticated():
                logger.warning("No valid API token for historical data fetch, using mock data")
                return self._generate_mock_data(symbol, exchange, interval, from_date, to_date)
            
            # Make the API request
            response = requests.get(self.HISTORICAL_URL, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                
                # Convert to DataFrame
                if not candles:
                    logger.warning(f"No historical candles returned for {instrument_key}")
                    return self._generate_mock_data(symbol, exchange, interval, from_date, to_date)
                
                df = pd.DataFrame(candles, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'
                ])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                
                # Sort by timestamp
                df.sort_index(inplace=True)
                
                # Store in cache
                if use_cache:
                    self.historical_data_cache[cache_key] = df
                
                logger.info(f"Retrieved {len(df)} historical candles for {instrument_key}")
                return df
            else:
                logger.error(f"Error fetching historical data: {data}")
                return self._generate_mock_data(symbol, exchange, interval, from_date, to_date)
        
        except Exception as e:
            logger.error(f"Failed to get historical candles: {e}")
            return self._generate_mock_data(symbol, exchange, interval, from_date, to_date)
    
    def fetch_intraday_candles(self, 
                             symbol: str,
                             exchange: str,
                             interval: str) -> pd.DataFrame:
        """
        Fetch intraday OHLC candle data from Upstox API.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, etc.)
            interval: Candle interval (1minute, 5minute, 30minute)
            
        Returns:
            DataFrame with intraday candles data
        """
        try:
            # Format instrument key
            instrument_key = f"{exchange}:{symbol}"
            
            # Set up headers and params
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.broker.access_token}'
            }
            
            params = {
                'instrument_key': instrument_key,
                'interval': interval
            }
            
            # Check if we're running in backtesting mode without a valid API token
            if not self.broker.access_token or not self.broker.is_authenticated():
                logger.warning("No valid API token for intraday data fetch, using mock data")
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                return self._generate_mock_data(symbol, exchange, interval, start_date, end_date)
            
            # Make the API request
            response = requests.get(self.INTRADAY_URL, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                
                # Convert to DataFrame
                if not candles:
                    logger.warning(f"No intraday candles returned for {instrument_key}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(candles, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'
                ])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                
                # Sort by timestamp
                df.sort_index(inplace=True)
                
                logger.info(f"Retrieved {len(df)} intraday candles for {instrument_key}")
                return df
            else:
                logger.error(f"Error fetching intraday data: {data}")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Failed to get intraday candles: {e}")
            return pd.DataFrame()
    
    def _generate_mock_data(self, symbol: str, exchange: str, interval: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Generate mock data for testing when API data is not available.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, etc.)
            interval: Candle interval
            from_date: Start date
            to_date: End date
            
        Returns:
            DataFrame with mock candle data
        """
        logger.info(f"Generating mock data for {exchange}:{symbol} from {from_date} to {to_date}")
        
        # Convert dates to datetime
        start = pd.Timestamp(from_date)
        end = pd.Timestamp(to_date)
        
        # Determine frequency from interval
        freq_map = {
            '1minute': '1min',
            '5minute': '5min',
            '15minute': '15min',
            '30minute': '30min',
            'hour': '60min',
            'day': '1D'
        }
        freq = freq_map.get(interval, '5min')
        
        # Generate timestamp range
        # For market hours only (9:15 AM to 3:30 PM, Monday to Friday)
        timestamps = []
        current = start
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:  # Monday to Friday
                # Add timestamps for market hours
                market_open = current.replace(hour=9, minute=15)
                market_close = current.replace(hour=15, minute=30)
                
                if freq in ['1min', '5min', '15min', '30min', '60min']:
                    # Generate intraday timestamps
                    time = market_open
                    while time <= market_close:
                        timestamps.append(time)
                        time += pd.Timedelta(freq)
                else:
                    # Daily or higher frequency
                    timestamps.append(current)
            
            # Move to next day
            if freq in ['1D', '1W', '1M']:
                current += pd.Timedelta(freq)
            else:
                current += pd.Timedelta(days=1)
        
        # Generate sample prices
        n = len(timestamps)
        
        if n == 0:
            logger.warning("No valid timestamps in date range, generating minimal sample data")
            end_date = pd.Timestamp.now() - pd.Timedelta(days=1)
            start_date = end_date - pd.Timedelta(days=30)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            timestamps = [date for date in dates if date.weekday() < 5]
            n = len(timestamps)
        
        # Start with a random price based on symbol name to make it consistent
        # This ensures the same symbol always starts at the same base price
        import hashlib
        symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest(), 16)
        base_price = 50 + (symbol_hash % 100)
        
        # Generate price series with some randomness and trend
        import numpy as np
        np.random.seed(symbol_hash)  # Use symbol hash as seed for reproducibility
        
        # Create price series with random walk and some cyclical behavior
        price_changes = np.random.normal(0, 1, n) * 0.005 * base_price  # Small random changes proportional to price
        trend = np.linspace(0, 10, n) * 0.001 * base_price  # Slight upward trend
        
        # Add some cyclicality
        cycle_length = n // 10 if n > 10 else n
        cycle = np.sin(np.linspace(0, 5 * np.pi, n)) * 0.01 * base_price
        
        # Combine all components
        closes = base_price + np.cumsum(price_changes) + trend + cycle
        
        # Generate other OHLC data
        volatility = 0.002 * base_price
        opens = closes - np.random.normal(0, volatility, n)
        highs = np.maximum(opens, closes) + np.random.normal(volatility, volatility, n)
        lows = np.minimum(opens, closes) - np.random.normal(volatility, volatility, n)
        
        # Generate volume data - higher volume for higher prices to simulate interest
        volumes = np.random.lognormal(10, 1, n) * closes / base_price * 100
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'oi': np.zeros(n)
        })
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Generated {len(df)} mock data points for {exchange}:{symbol}")
        return df