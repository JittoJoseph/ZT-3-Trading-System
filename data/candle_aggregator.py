"""
Candle Aggregator for ZT-3 Trading System.

This module handles:
- Aggregation of tick data into OHLC candles
- Management of multiple candle timeframes
- Data quality monitoring and validation
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class CandleAggregator:
    """
    Aggregates tick data into OHLC candles.
    
    This class handles the aggregation of real-time tick data into
    OHLC candles of various timeframes, with a focus on 5-minute candles
    for the Gaussian Channel Strategy.
    """
    
    def __init__(self, market_data_client):
        """
        Initialize the candle aggregator with market data client.
        
        Args:
            market_data_client: Market data client instance
        """
        self.market_data_client = market_data_client
        
        # Register callback with market data client
        self.market_data_client.register_market_data_callback(self._on_market_data)
        
        # Candle storage by symbol and timeframe
        # symbol -> timeframe -> list of candles
        self.candles = defaultdict(lambda: defaultdict(list))
        
        # Current candle being built for each symbol and timeframe
        # symbol -> timeframe -> current candle dict
        self.current_candles = defaultdict(lambda: defaultdict(dict))
        
        # Candle completion callbacks
        # timeframe -> list of callbacks
        self.candle_callbacks = defaultdict(list)
        
        # Supported timeframes in minutes
        self.timeframes = [1, 5, 15, 30, 60]
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Candle Aggregator initialized")
    
    def register_candle_callback(self, timeframe: int, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Register a callback for completed candles of specific timeframe.
        
        Args:
            timeframe: Timeframe in minutes
            callback: Function to call with symbol and completed candle
        """
        with self.lock:
            if timeframe not in self.timeframes:
                logger.warning(f"Unsupported timeframe: {timeframe}")
                return
            
            self.candle_callbacks[timeframe].append(callback)
    
    def unregister_candle_callback(self, timeframe: int, callback: Callable) -> None:
        """
        Unregister a candle callback.
        
        Args:
            timeframe: Timeframe in minutes
            callback: Previously registered callback function
        """
        with self.lock:
            if callback in self.candle_callbacks[timeframe]:
                self.candle_callbacks[timeframe].remove(callback)
    
    def get_candles(self, symbol: str, timeframe: int, count: int = 100) -> pd.DataFrame:
        """
        Get historical candles for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe in minutes
            count: Number of candles to return
            
        Returns:
            DataFrame with historical candles
        """
        with self.lock:
            symbol_candles = self.candles[symbol][timeframe]
            
            if not symbol_candles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(symbol_candles[-count:])
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set timestamp as index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            
            return df
    
    def _on_market_data(self, message: Any) -> None:
        """
        Process market data message and update candles.
        
        Args:
            message: Market data message from WebSocket
        """
        try:
            # TODO: Implement proper processing of Market Data Feed V3 protobuf messages
            # For now, this is a placeholder for the tick processing logic
            
            # Extract relevant data from message
            # For example:
            # symbol = message.symbol
            # timestamp = message.timestamp
            # price = message.last_price
            # volume = message.volume
            
            # Mock data for development
            symbol = "NSE:PNB"  # Example symbol
            timestamp = datetime.now()
            price = 100.0  # Example price
            volume = 100  # Example volume
            
            # Update candles for all timeframes
            self._update_candles(symbol, timestamp, price, volume)
            
        except Exception as e:
            logger.error(f"Error processing market data for candles: {e}")
    
    def _update_candles(self, symbol: str, timestamp: datetime, price: float, volume: int) -> None:
        """
        Update candles for all timeframes based on new tick data.
        
        Args:
            symbol: Trading symbol
            timestamp: Tick timestamp
            price: Tick price
            volume: Tick volume
        """
        with self.lock:
            for timeframe in self.timeframes:
                self._update_timeframe_candle(symbol, timeframe, timestamp, price, volume)
    
    def _update_timeframe_candle(self, 
                               symbol: str, 
                               timeframe: int, 
                               timestamp: datetime, 
                               price: float, 
                               volume: int) -> None:
        """
        Update candle for a specific timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe in minutes
            timestamp: Tick timestamp
            price: Tick price
            volume: Tick volume
        """
        # Calculate candle start time by rounding down to the nearest timeframe boundary
        minutes_since_epoch = int(timestamp.timestamp()) // 60
        candle_start_minutes = (minutes_since_epoch // timeframe) * timeframe
        candle_start = datetime.fromtimestamp(candle_start_minutes * 60)
        candle_end = candle_start + timedelta(minutes=timeframe)
        
        current_candle = self.current_candles[symbol][timeframe]
        
        # If we don't have a current candle or this tick belongs to a new candle
        if not current_candle or timestamp >= candle_end:
            # If we have a current candle, finalize it before creating a new one
            if current_candle:
                # Complete the current candle
                self._complete_candle(symbol, timeframe, current_candle)
            
            # Create a new candle
            new_candle = {
                'timestamp': candle_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
            
            self.current_candles[symbol][timeframe] = new_candle
        else:
            # Update the current candle
            current_candle['high'] = max(current_candle['high'], price)
            current_candle['low'] = min(current_candle['low'], price)
            current_candle['close'] = price
            current_candle['volume'] += volume
    
    def _complete_candle(self, symbol: str, timeframe: int, candle: Dict[str, Any]) -> None:
        """
        Complete a candle and notify callbacks.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe in minutes
            candle: Candle data
        """
        # Make a copy of the candle
        completed_candle = candle.copy()
        
        # Add to the historical candles
        self.candles[symbol][timeframe].append(completed_candle)
        
        # Limit the number of stored candles to prevent memory issues
        max_candles = 1000
        if len(self.candles[symbol][timeframe]) > max_candles:
            self.candles[symbol][timeframe] = self.candles[symbol][timeframe][-max_candles:]
        
        # Notify callbacks
        for callback in self.candle_callbacks[timeframe]:
            try:
                callback(symbol, completed_candle)
            except Exception as e:
                logger.error(f"Error in candle callback: {e}")
    
    def load_historical_candles(self, 
                              symbol: str, 
                              exchange: str,
                              timeframe: int,
                              days_back: int = 10) -> bool:
        """
        Load historical candles to initialize the candle history.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, etc.)
            timeframe: Timeframe in minutes
            days_back: Number of days to load
            
        Returns:
            bool: True if loading was successful
        """
        try:
            # Convert timeframe to Upstox format
            if timeframe == 1:
                interval = "1minute"
            elif timeframe == 5:
                interval = "5minute"  # Note: This is not in the API docs, might be "5minute" or unsupported
            elif timeframe == 15:
                interval = "15minute"  # Note: This is not in the API docs, might be "15minute" or unsupported
            elif timeframe == 30:
                interval = "30minute"
            elif timeframe == 60:
                interval = "1hour"
            else:
                logger.error(f"Unsupported timeframe for historical data: {timeframe}")
                return False
            
            # For intraday timeframes <= 30 min, use intraday API for today's data
            today = datetime.now().strftime('%Y-%m-%d')
            
            if timeframe <= 30:
                # Get today's data from intraday API
                df = self.market_data_client.fetch_intraday_candles(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval
                )
                
                if not df.empty:
                    # Convert to list of dicts
                    candles = df.reset_index().to_dict('records')
                    
                    # Store candles
                    with self.lock:
                        full_symbol = f"{exchange}:{symbol}"
                        self.candles[full_symbol][timeframe].extend(candles)
                        
                        # Set current candle to the most recent one
                        if candles:
                            last_candle = candles[-1]
                            self.current_candles[full_symbol][timeframe] = last_candle.copy()
            
            # For historical data beyond today
            if days_back > 1:
                end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                # Get historical data
                df = self.market_data_client.fetch_historical_candles(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    from_date=start_date,
                    to_date=end_date
                )
                
                if not df.empty:
                    # Convert to list of dicts
                    candles = df.reset_index().to_dict('records')
                    
                    # Store candles
                    with self.lock:
                        full_symbol = f"{exchange}:{symbol}"
                        
                        # Add historical candles before today's candles
                        current_candles = self.candles[full_symbol][timeframe]
                        self.candles[full_symbol][timeframe] = candles + current_candles
            
            logger.info(f"Loaded historical candles for {symbol} ({timeframe} min)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load historical candles: {e}")
            return False