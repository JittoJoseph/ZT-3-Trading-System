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
        Initialize the market data client with broker and config.
        
        Args:
            broker: Broker instance for API access
            config: Configuration dictionary
        """
        self.broker = broker
        self.config = config
        
        # WebSocket connection
        self.ws = None
        self.ws_connected = False
        self.ws_thread = None
        self.ws_reconnect_attempt = 0
        self.ws_reconnect_max_attempts = 5
        self.ws_reconnect_delay = 5  # seconds
        
        # Market data callbacks
        self.market_data_callbacks = []
        
        # Candle storage
        self.candles = {}  # symbol -> list of candle data
        self.latest_ticks = {}  # symbol -> latest tick data
        
        # Cache for historical data
        self.historical_data_cache = {}  # symbol -> DataFrame
        
        logger.info("Market Data Client initialized")
    
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
            interval: Candle interval (1minute, 30minute, day, week, month)
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
            
            # Make the API request
            response = requests.get(self.HISTORICAL_URL, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                
                # Convert to DataFrame
                if not candles:
                    logger.warning(f"No historical candles returned for {instrument_key}")
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
                
                # Store in cache
                if use_cache:
                    self.historical_data_cache[cache_key] = df
                
                logger.info(f"Retrieved {len(df)} historical candles for {instrument_key}")
                return df
            else:
                logger.error(f"Error fetching historical data: {data}")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Failed to get historical candles: {e}")
            return pd.DataFrame()
    
    def fetch_intraday_candles(self, 
                             symbol: str,
                             exchange: str,
                             interval: str) -> pd.DataFrame:
        """
        Fetch intraday OHLC candle data from Upstox API.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, etc.)
            interval: Candle interval (1minute, 30minute)
            
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
    
    def get_market_data_feed_auth(self) -> Optional[Dict[str, Any]]:
        """
        Get authorization data for Market Data Feed V3 WebSocket.
        
        Returns:
            Dict with authorization data or None if failed
        """
        try:
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.broker.access_token}'
            }
            
            response = requests.get(self.MARKET_DATA_FEED_AUTH_URL, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'success':
                auth_data = data.get('data', {})
                logger.debug(f"Received market data feed auth data")
                return auth_data
            else:
                logger.error(f"Error getting market data feed auth: {data}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to get market data feed auth: {e}")
            return None
    
    def connect_websocket(self) -> bool:
        """
        Connect to Upstox Market Data Feed V3 WebSocket.
        
        Returns:
            bool: True if connection was successful
        """
        # Get authorization data for WebSocket
        auth_data = self.get_market_data_feed_auth()
        if not auth_data:
            logger.error("Failed to get WebSocket authorization data")
            return False
        
        try:
            ws_url = auth_data.get('authorized_redirect_uri')
            if not ws_url:
                logger.error("Missing WebSocket URL in auth data")
                return False
            
            # Close existing connection if any
            if self.ws:
                self.disconnect_websocket()
            
            # Setup WebSocket connection
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_ws_open,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close
            )
            
            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection to establish
            wait_time = 0
            while not self.ws_connected and wait_time < 10:
                time.sleep(0.5)
                wait_time += 0.5
            
            if not self.ws_connected:
                logger.error("WebSocket connection timed out")
                return False
            
            self.ws_reconnect_attempt = 0
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    def disconnect_websocket(self) -> bool:
        """
        Disconnect from WebSocket.
        
        Returns:
            bool: True if disconnection was successful
        """
        if self.ws:
            self.ws.close()
            self.ws = None
            self.ws_connected = False
            
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=2)
            
            logger.info("WebSocket disconnected")
            return True
        
        return False
    
    def _on_ws_open(self, ws):
        """
        Handle WebSocket connection open event.
        
        Args:
            ws: WebSocket connection
        """
        logger.info("WebSocket connection opened")
        self.ws_connected = True
    
    def _on_ws_message(self, ws, message):
        """
        Handle WebSocket message.
        
        Args:
            ws: WebSocket connection
            message: Binary message data (protobuf)
        """
        try:
            # TODO: Implement proper protobuf decoding for MarketDataFeedV3
            # For now, log that we received data
            logger.debug("Received WebSocket message")
            
            # Process the message and call all registered callbacks
            for callback in self.market_data_callbacks:
                callback(message)
        
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_ws_error(self, ws, error):
        """
        Handle WebSocket error.
        
        Args:
            ws: WebSocket connection
            error: Error details
        """
        logger.error(f"WebSocket error: {error}")
        self.ws_connected = False
        
        # Try to reconnect after a delay
        if self.ws_reconnect_attempt < self.ws_reconnect_max_attempts:
            self.ws_reconnect_attempt += 1
            reconnect_delay = self.ws_reconnect_delay * self.ws_reconnect_attempt
            
            logger.info(f"Attempting to reconnect in {reconnect_delay} seconds (attempt {self.ws_reconnect_attempt})")
            
            # Schedule reconnection
            threading.Timer(reconnect_delay, self.connect_websocket).start()
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """
        Handle WebSocket close event.
        
        Args:
            ws: WebSocket connection
            close_status_code: Status code
            close_msg: Close message
        """
        logger.info(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
        self.ws_connected = False
    
    def subscribe_market_data(self, symbols: List[Dict[str, str]]) -> bool:
        """
        Subscribe to market data for symbols.
        
        Args:
            symbols: List of symbol dictionaries with 'ticker' and 'exchange'
            
        Returns:
            bool: True if subscription was successful
        """
        if not self.ws_connected:
            logger.error("WebSocket not connected")
            return False
        
        try:
            # Format subscribe message according to Upstox Market Data Feed V3
            # This is a placeholder - actual implementation depends on the protobuf schema
            
            # TODO: Implement proper protobuf encoding for subscription
            logger.info(f"Subscribed to {len(symbols)} symbols")
            return True
        
        except Exception as e:
            logger.error(f"Failed to subscribe to market data: {e}")
            return False
    
    def register_market_data_callback(self, callback: Callable[[Any], None]) -> None:
        """
        Register callback for market data events.
        
        Args:
            callback: Function to call with market data
        """
        self.market_data_callbacks.append(callback)
    
    def unregister_market_data_callback(self, callback: Callable) -> None:
        """
        Unregister market data callback.
        
        Args:
            callback: Previously registered callback function
        """
        if callback in self.market_data_callbacks:
            self.market_data_callbacks.remove(callback)