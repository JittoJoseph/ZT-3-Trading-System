"""
Candle Aggregator for ZT-3 Trading System.

This module handles aggregation of tick data into OHLC candles.
"""

import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)

class CandleAggregator:
    """
    Aggregates tick data into OHLC candles of specified intervals.
    
    This class processes incoming tick data and generates OHLC candles
    for different timeframes, with proper time alignment.
    """
    
    def __init__(self, symbols: List[str], intervals: List[str] = None):
        """
        Initialize the candle aggregator.
        
        Args:
            symbols: List of symbols to aggregate candles for
            intervals: List of interval strings ('1min', '5min', '15min', '30min', '1h', '1d')
                      Default: ['5min'] if None is provided
        """
        self.symbols = symbols
        self.intervals = intervals or ['5min']
        
        # Validate intervals
        valid_intervals = ['1min', '5min', '15min', '30min', '1h', '1d']
        for interval in self.intervals:
            if interval not in valid_intervals:
                logger.warning(f"Unsupported interval: {interval} - will be ignored")
        
        # Dictionary to store tick data for each symbol
        self.ticks = {symbol: [] for symbol in symbols}
        
        # Dictionary to store partial candles for each symbol and interval
        # {symbol: {interval: partial_candle}}
        self.partial_candles = {}
        
        # Dictionary to store completed candles for each symbol and interval
        # {symbol: {interval: [candles]}}
        self.candles = {}
        
        # Callbacks for completed candles
        # {symbol: {interval: [callbacks]}}
        self.callbacks = defaultdict(lambda: defaultdict(list))
        
        # Initialize data structures
        self._initialize_data_structures()
        
        logger.info(f"Initialized candle aggregator for {len(symbols)} symbols with intervals: {intervals}")
    
    def _initialize_data_structures(self) -> None:
        """Initialize internal data structures for all symbols and intervals."""
        for symbol in self.symbols:
            self.partial_candles[symbol] = {}
            self.candles[symbol] = {}
            
            for interval in self.intervals:
                self.partial_candles[symbol][interval] = None
                self.candles[symbol][interval] = []
    
    def _get_interval_seconds(self, interval: str) -> int:
        """
        Convert interval string to seconds.
        
        Args:
            interval: Interval string ('1min', '5min', etc.)
            
        Returns:
            Interval duration in seconds
        """
        if interval == '1min':
            return 60
        elif interval == '5min':
            return 300
        elif interval == '15min':
            return 900
        elif interval == '30min':
            return 1800
        elif interval == '1h':
            return 3600
        elif interval == '1d':
            return 86400
        else:
            logger.warning(f"Unsupported interval: {interval} - defaulting to 5min (300s)")
            return 300
    
    def _get_candle_start_time(self, timestamp: datetime, interval: str) -> datetime:
        """
        Get the properly aligned start time for a candle.
        
        Args:
            timestamp: Timestamp to align
            interval: Candle interval
            
        Returns:
            Aligned start time for the candle
        """
        interval_seconds = self._get_interval_seconds(interval)
        
        if interval == '1min':
            return timestamp.replace(second=0, microsecond=0)
        elif interval in ['5min', '15min', '30min']:
            minutes = (timestamp.minute // (interval_seconds // 60)) * (interval_seconds // 60)
            return timestamp.replace(minute=minutes, second=0, microsecond=0)
        elif interval == '1h':
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif interval == '1d':
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Default to 5min
            minutes = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minutes, second=0, microsecond=0)
    
    def add_tick(self, symbol: str, tick: Dict[str, Any]) -> None:
        """
        Add a tick to be processed.
        
        Args:
            symbol: Symbol for the tick
            tick: Tick data dictionary (must contain 'timestamp' and 'price')
        """
        if symbol not in self.symbols:
            logger.warning(f"Tick received for unregistered symbol: {symbol}")
            return
        
        if 'timestamp' not in tick or 'price' not in tick:
            logger.warning(f"Invalid tick data for {symbol}: {tick}")
            return
        
        # Store the tick
        self.ticks[symbol].append(tick)
        
        # Process ticks for all intervals
        for interval in self.intervals:
            self._process_tick(symbol, tick, interval)
    
    def _process_tick(self, symbol: str, tick: Dict[str, Any], interval: str) -> None:
        """
        Process a tick for a specific interval.
        
        Args:
            symbol: Symbol for the tick
            tick: Tick data dictionary
            interval: Candle interval
        """
        # Get timestamp and price from the tick
        timestamp = tick['timestamp']
        price = float(tick['price'])
        volume = float(tick.get('volume', 0))
        
        # Get the candle start time
        candle_start = self._get_candle_start_time(timestamp, interval)
        
        # Check if we need to create a new candle
        if (self.partial_candles[symbol][interval] is None or 
                candle_start > self.partial_candles[symbol][interval]['timestamp']):
            
            # If there was a previous candle, complete it and move to history
            if self.partial_candles[symbol][interval] is not None:
                completed_candle = self.partial_candles[symbol][interval].copy()
                self.candles[symbol][interval].append(completed_candle)
                
                # Limit stored history to avoid memory issues (keep last 1000 candles)
                if len(self.candles[symbol][interval]) > 1000:
                    self.candles[symbol][interval] = self.candles[symbol][interval][-1000:]
                
                # Notify callbacks
                for callback in self.callbacks[symbol][interval]:
                    try:
                        callback(symbol, completed_candle)
                    except Exception as e:
                        logger.error(f"Error in candle callback: {e}")
            
            # Create new candle
            self.partial_candles[symbol][interval] = {
                'timestamp': candle_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
        else:
            # Update existing candle
            candle = self.partial_candles[symbol][interval]
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            candle['volume'] += volume
    
    def get_last_candle(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """
        Get the last completed candle for a symbol and interval.
        
        Args:
            symbol: Symbol to get candle for
            interval: Candle interval
            
        Returns:
            Last completed candle data or None if no candle is available
        """
        if (symbol not in self.candles or 
                interval not in self.candles[symbol] or 
                not self.candles[symbol][interval]):
            return None
        
        return self.candles[symbol][interval][-1]
    
    def get_current_candle(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """
        Get the current partial candle for a symbol and interval.
        
        Args:
            symbol: Symbol to get candle for
            interval: Candle interval
            
        Returns:
            Current partial candle data or None if no candle is available
        """
        if (symbol not in self.partial_candles or 
                interval not in self.partial_candles[symbol]):
            return None
        
        return self.partial_candles[symbol][interval]
    
    def get_candles(self, symbol: str, interval: str, count: int = None) -> List[Dict[str, Any]]:
        """
        Get completed candles for a symbol and interval.
        
        Args:
            symbol: Symbol to get candles for
            interval: Candle interval
            count: Number of candles to return (None for all available)
            
        Returns:
            List of completed candles
        """
        if (symbol not in self.candles or 
                interval not in self.candles[symbol] or 
                not self.candles[symbol][interval]):
            return []
        
        if count is None:
            return self.candles[symbol][interval]
        else:
            return self.candles[symbol][interval][-count:]
    
    def register_candle_callback(self, symbol: str, interval: str, 
                               callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Register a callback for completed candles.
        
        Args:
            symbol: Symbol to register callback for
            interval: Candle interval
            callback: Function to call when a candle is completed
        """
        self.callbacks[symbol][interval].append(callback)
    
    def unregister_candle_callback(self, symbol: str, interval: str, 
                                 callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Unregister a candle callback.
        
        Args:
            symbol: Symbol the callback was registered for
            interval: Candle interval
            callback: Callback function to remove
        """
        if symbol in self.callbacks and interval in self.callbacks[symbol]:
            if callback in self.callbacks[symbol][interval]:
                self.callbacks[symbol][interval].remove(callback)
    
    def flush(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> None:
        """
        Flush partial candles and complete them.
        
        This is useful for end of day processing or shutdown.
        
        Args:
            symbol: Specific symbol to flush (None for all)
            interval: Specific interval to flush (None for all)
        """
        symbols_to_flush = [symbol] if symbol else self.symbols
        
        for sym in symbols_to_flush:
            intervals_to_flush = [interval] if interval else self.intervals
            
            for intv in intervals_to_flush:
                if (sym in self.partial_candles and 
                        intv in self.partial_candles[sym] and 
                        self.partial_candles[sym][intv] is not None):
                    
                    # Complete the partial candle
                    completed_candle = self.partial_candles[sym][intv].copy()
                    self.candles[sym][intv].append(completed_candle)
                    
                    # Reset partial candle
                    self.partial_candles[sym][intv] = None
                    
                    # Notify callbacks
                    for callback in self.callbacks[sym][intv]:
                        try:
                            callback(sym, completed_candle)
                        except Exception as e:
                            logger.error(f"Error in candle callback during flush: {e}")
    
    def clear_data(self, symbol: Optional[str] = None) -> None:
        """
        Clear stored tick and candle data.
        
        Args:
            symbol: Specific symbol to clear data for (None for all)
        """
        symbols_to_clear = [symbol] if symbol else self.symbols
        
        for sym in symbols_to_clear:
            self.ticks[sym] = []
            
            for interval in self.intervals:
                self.partial_candles[sym][interval] = None
                self.candles[sym][interval] = []
        
        logger.info(f"Cleared data for {len(symbols_to_clear)} symbols")
    
    def to_dataframe(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Convert candle data to a pandas DataFrame.
        
        Args:
            symbol: Symbol to get candles for
            interval: Candle interval
            
        Returns:
            DataFrame with OHLCV data
        """
        if (symbol not in self.candles or 
                interval not in self.candles[symbol] or 
                not self.candles[symbol][interval]):
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert to DataFrame
        df = pd.DataFrame(self.candles[symbol][interval])
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df