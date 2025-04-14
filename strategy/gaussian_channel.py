"""
Gaussian Channel Strategy Implementation for ZT-3 Trading System.

This module implements the Gaussian Channel strategy as defined in the Pine script,
adapted for the ZT-3 trading system.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from strategy import Strategy, Signal, SignalType, ExitReason
from strategy.indicators import Indicators

logger = logging.getLogger(__name__)

class GaussianChannelStrategy(Strategy):
    """
    Gaussian Channel Strategy implementation.
    
    This strategy implements the Gaussian Channel strategy with the following features:
    - Gaussian Channel indicator for trend detection
    - Stochastic RSI filter for entry confirmation
    - Volume filter for additional confirmation
    - Take profit at 4.0x ATR
    - Exit on close crossing below Gaussian filter line
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Gaussian Channel Strategy.
        
        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config)
        logger.info("Initializing Gaussian Channel Strategy")
        
        # Extract strategy parameters
        self.gc_period = self.params.get('gc_period', 144)
        self.gc_multiplier = self.params.get('gc_multiplier', 1.2)
        self.stoch_upper_band = self.params.get('stoch_upper_band', 80.0)
        self.use_volume_filter = self.params.get('use_volume_filter', True)
        self.use_take_profit = self.params.get('use_take_profit', True)
        self.atr_tp_multiplier = self.params.get('atr_tp_multiplier', 4.0)
        
        logger.info(f"Strategy parameters: GC Period={self.gc_period}, GC Mult={self.gc_multiplier}, "
                   f"Stoch Upper Band={self.stoch_upper_band}")
    
    def process_candle(self, candle_data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Process new candle data and generate signals for the Gaussian Channel strategy.
        
        Entry Conditions:
        - Gaussian filter line is rising (filt > filt[1])
        - Price closes above the high band (close > hband)
        - Stochastic RSI(14,14,3) K-line is above 80
        - Volume is above its 20-period moving average
        
        Exit Conditions:
        - Take profit at 4.0x ATR from entry price
        - OR when price closes below the Gaussian filter line
        
        Args:
            candle_data: DataFrame containing candle data with indicators
            symbol: Trading symbol
            
        Returns:
            Signal if conditions are met, None otherwise
        """
        if len(candle_data) < 2:
            logger.warning(f"Insufficient data for {symbol} to generate signals")
            return None
        
        # Get the latest candle
        current = candle_data.iloc[-1]
        previous = candle_data.iloc[-2]
        
        timestamp = pd.Timestamp(current.name) if isinstance(current.name, pd.Timestamp) else pd.Timestamp.now()
        
        # Check if we have all required indicators
        required_columns = ['gc_filter', 'gc_upper', 'stoch_rsi_k', 'close']
        if self.use_volume_filter:
            required_columns.append('volume')
            required_columns.append('volume_ma')
        
        missing_columns = [col for col in required_columns if col not in candle_data.columns]
        if missing_columns:
            logger.warning(f"Missing required indicators for {symbol}: {missing_columns}")
            return None
            
        # Check for exit signal first (if we have an open position)
        # In a real implementation, we'd check the position manager here
        # For now, we'll just generate the exit signal based on the conditions
        
        # Exit condition 1: Close crossed below the Gaussian filter line
        filter_cross_exit = (current['close'] < current['gc_filter'] and 
                             previous['close'] >= previous['gc_filter'])
        
        if filter_cross_exit:
            # Don't check for duplicates on exits - we want to exit as soon as possible
            exit_signal = Signal(
                signal_type=SignalType.EXIT,
                symbol=symbol,
                price=current['close'],
                timestamp=timestamp,
                exit_reason=ExitReason.FILTER_CROSS
            )
            logger.info(f"Generated exit signal for {symbol}: {exit_signal}")
            return exit_signal
            
        # Entry conditions:
        # 1. Gaussian filter line is rising
        filter_rising = current['gc_filter'] > previous['gc_filter']
        
        # 2. Close above high band
        close_above_high_band = current['close'] > current['gc_upper']
        
        # 3. Stochastic RSI K-line is above upper band (80)
        stoch_rsi_condition = current['stoch_rsi_k'] > self.stoch_upper_band
        
        # 4. Volume filter (if enabled)
        volume_condition = True
        if self.use_volume_filter:
            volume_condition = current['volume'] > current['volume_ma']
        
        # Check all entry conditions
        if (filter_rising and close_above_high_band and stoch_rsi_condition and volume_condition):
            # Check for duplicate signals
            if self.is_duplicate_signal(SignalType.ENTRY, symbol, timestamp):
                logger.debug(f"Duplicate entry signal detected for {symbol}, ignoring")
                return None
                
            # Calculate take profit level using ATR
            take_profit_level = None
            if self.use_take_profit and 'atr' in candle_data.columns:
                take_profit_level = self.calculate_take_profit(current['close'], candle_data)
                
            # Generate entry signal
            entry_signal = Signal(
                signal_type=SignalType.ENTRY,
                symbol=symbol,
                price=current['close'],
                timestamp=timestamp,
                take_profit_level=take_profit_level
            )
            
            logger.info(f"Generated entry signal for {symbol}: {entry_signal}")
            return entry_signal
            
        return None
        
    def calculate_take_profit(self, entry_price: float, candle_data: pd.DataFrame) -> float:
        """
        Calculate take profit level based on ATR.
        
        Args:
            entry_price: Entry price
            candle_data: DataFrame containing candle data with indicators
            
        Returns:
            Take profit price level
        """
        atr = candle_data['atr'].iloc[-1]
        return entry_price + (atr * self.atr_tp_multiplier)