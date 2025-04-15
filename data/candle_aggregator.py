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
    Aggregates real-time tick data into OHLC candles.
    
    Supports multiple symbols and timeframes simultaneously.
    """
    
    def __init__(self, symbols: List[str], timeframes: List[str]):
        """
        Initialize candle aggregator.
        
        Args:
            symbols: List of symbols to track
            timeframes: List of timeframe intervals (e.g., '1min', '5min')
        """
        self.symbols = symbols
        self.timeframes = timeframes
        
        # Data structures
        self.candles = defaultdict(lambda: defaultdict(list))  # {symbol: {timeframe: [candles]}}
        self.partial_candles = defaultdict(lambda: defaultdict(dict))  # {symbol: {timeframe: partial_candle}}
        self.callbacks = defaultdict(lambda: defaultdict(list))  # {symbol: {timeframe: [callbacks]}}
    
    def register_candle_callback(self, symbol: str, timeframe: str, callback: Callable) -> None:
        """
        Register callback for completed candles.
        
        Args:
            symbol: Symbol to monitor
            timeframe: Candle timeframe
            callback: Function to call when candle completes
        """
        self.callbacks[symbol][timeframe].append(callback)
        logger.debug(f"Registered candle callback for {symbol} {timeframe}")
    
    def add_tick(self, symbol: str, tick: Dict[str, Any]) -> None:
        """
        Add a new price tick to be aggregated.
        
        Args:
            symbol: Symbol the tick is for
            tick: Tick data with 'price', 'volume', and 'timestamp' fields
        """
        price = tick.get('price')
        volume = tick.get('volume', 0)
        timestamp = tick.get('timestamp')
        
        if not price or not timestamp:
            logger.warning(f"Incomplete tick data for {symbol}: {tick}")
            return
            
        # Process for each timeframe
        for interval in self.timeframes:
            self._process_tick(symbol, tick, interval)
    
    def add_bar(self, symbol: str, bar: Dict[str, Any], timeframe: str) -> None:
        """
        Add a pre-formed bar/candle directly.
        
        Args:
            symbol: Symbol the bar is for
            bar: Bar data with OHLCV fields
            timeframe: Timeframe of the bar
        """
        self.candles[symbol][timeframe].append(bar.copy())
        
        # Limit stored history
        if len(self.candles[symbol][timeframe]) > 1000:
            self.candles[symbol][timeframe] = self.candles[symbol][timeframe][-1000:]
        
        # Notify callbacks
        for callback in self.callbacks[symbol][timeframe]:
            try:
                callback(symbol, bar)
            except Exception as e:
                logger.error(f"Error in candle callback: {e}")
    
    def _process_tick(self, symbol: str, tick: Dict[str, Any], interval: str) -> None:
        """
        Process tick for a specific timeframe.
        
        Args:
            symbol: Symbol
            tick: Tick data
            interval: Candle interval
        """
        price = tick['price']
        volume = tick.get('volume', 0)
        timestamp = tick['timestamp']
        
        # Convert interval to timedelta
        if interval == '1min':
            td = timedelta(minutes=1)
        elif interval == '5min':
            td = timedelta(minutes=5)
        elif interval == '15min':
            td = timedelta(minutes=15)
        elif interval == '30min':
            td = timedelta(minutes=30)
        elif interval == '1h':
            td = timedelta(hours=1)
        else:
            logger.warning(f"Unsupported interval: {interval}")
            return
        
        # Determine candle start time
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        
        minutes = timestamp.minute
        seconds = timestamp.second
        microseconds = timestamp.microsecond
        
        if interval == '1min':
            candle_start = timestamp.replace(second=0, microsecond=0)
        elif interval == '5min':
            candle_start = timestamp.replace(minute=(minutes // 5) * 5, second=0, microsecond=0)
        elif interval == '15min':
            candle_start = timestamp.replace(minute=(minutes // 15) * 15, second=0, microsecond=0)
        elif interval == '30min':
            candle_start = timestamp.replace(minute=(minutes // 30) * 30, second=0, microsecond=0)
        elif interval == '1h':
            candle_start = timestamp.replace(minute=0, second=0, microsecond=0)
        
        # Check if we have a partial candle for this symbol and timeframe
        if symbol in self.partial_candles and interval in self.partial_candles[symbol]:
            partial = self.partial_candles[symbol][interval]
            
            # Check if tick belongs to current candle
            if candle_start == partial['timestamp']:
                # Update partial candle
                partial['high'] = max(partial['high'], price)
                partial['low'] = min(partial['low'], price)
                partial['close'] = price
                partial['volume'] += volume
            else:
                # Candle is complete, finalize it
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
            # First tick for this symbol and timeframe
            self.partial_candles[symbol][interval] = {
                'timestamp': candle_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
    
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
    
    def get_latest_candle(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest complete candle for a symbol and timeframe.
        
        Args:
            symbol: Symbol
            interval: Candle interval
            
        Returns:
            Latest candle or None if no candles exist
        """
        if (symbol not in self.candles or 
                interval not in self.candles[symbol] or 
                not self.candles[symbol][interval]):
            return None
            
        return self.candles[symbol][interval][-1]
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Symbol
            
        Returns:
            Latest price or None if no data exists
        """
        for interval in self.timeframes:
            # Check partial candles first
            if symbol in self.partial_candles and interval in self.partial_candles[symbol]:
                return self.partial_candles[symbol][interval]['close']
            
            # Check completed candles
            if symbol in self.candles and interval in self.candles[symbol]:
                if self.candles[symbol][interval]:
                    return self.candles[symbol][interval][-1]['close']
                    
        return None