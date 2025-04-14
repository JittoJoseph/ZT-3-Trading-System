"""
Gaussian Channel Strategy Module for ZT-3 Trading System.

This module implements the Gaussian Channel trading strategy, which uses
a Gaussian filter for generating entry and exit signals.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime
from enum import Enum

from strategy.indicators import Indicators
from strategy import SignalType, ExitReason, Signal

logger = logging.getLogger(__name__)

class GaussianChannelStrategy:
    """
    Gaussian Channel trading strategy.
    
    This strategy uses a Gaussian filter for market trend direction 
    in combination with StochasticRSI and volume filters.
    
    Entry conditions:
    - Gaussian filter line is rising (filt > filt[1])
    - Price closes above the high band (close > hband)
    - Stochastic RSI K-line is above 80
    - Volume is above its 20-period moving average
    
    Exit conditions:
    - Take profit at 4.0x ATR from entry price
    - OR when price closes below the Gaussian filter line
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Gaussian Channel strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.strategy_config = config.get('strategy', {})
        self.params = self.strategy_config.get('params', {})
        
        # Extract channel parameters
        self.gc_period = self.params.get('gc_period', 144)
        self.gc_multiplier = self.params.get('gc_multiplier', 1.2)
        
        # Extract StochasticRSI parameters
        self.stoch_rsi_length = self.params.get('stoch_rsi_length', 14)
        self.stoch_length = self.params.get('stoch_length', 14)
        self.stoch_k_smooth = self.params.get('stoch_k_smooth', 3)
        self.stoch_upper_band = self.params.get('stoch_upper_band', 80.0)
        
        # Extract volume filter parameters
        self.vol_ma_length = self.params.get('volume_ma_length', 20)
        
        # Extract take profit parameters
        self.atr_length = self.params.get('atr_length', 14)
        self.atr_tp_multiplier = self.params.get('atr_tp_multiplier', 4.0)
        
        # For position tracking in real-time signal generation
        self.open_positions = {}  # symbol -> position info
        
        logger.info(f"Initialized Gaussian Channel Strategy with period={self.gc_period}, multiplier={self.gc_multiplier}")
        
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        Generate trading signals based on the Gaussian Channel strategy.
        
        Args:
            df: DataFrame with price data (should contain OHLCV data)
            symbol: Trading symbol
            
        Returns:
            List of Signal objects
        """
        if len(df) < self.gc_period + 10:  # Need enough data for reliable signals
            logger.warning(f"Not enough data for {symbol} to generate reliable signals (need at least {self.gc_period + 10} candles)")
            return []
            
        # Calculate indicators if they aren't already in the dataframe
        if not all(col in df.columns for col in ['gc_filter', 'gc_upper', 'gc_lower', 'stoch_rsi_k', 'volume_ma', 'atr']):
            # Calculate Gaussian Channel indicators
            df = Indicators.gaussian_channel(
                df, 
                source_col='close', 
                period=self.gc_period, 
                multiplier=self.gc_multiplier
            )
            
            # Calculate Stochastic RSI
            df = Indicators.stochastic_rsi(
                df,
                close_col='close',
                rsi_period=self.stoch_rsi_length,
                stoch_period=self.stoch_length,
                k_smoothing=self.stoch_k_smooth
            )
            
            # Calculate Volume MA
            df = Indicators.volume_analysis(
                df,
                volume_col='volume',
                ma_period=self.vol_ma_length
            )
            
            # Calculate ATR for take profit
            df = Indicators.atr(
                df,
                period=self.atr_length
            )
            
        # Generate signals
        signals = []
        in_position = False
        entry_price = 0.0
        take_profit_level = 0.0
        
        # Process each candle
        for i in range(1, len(df)):
            # Skip rows with NaN values in any required indicator
            if any(pd.isna(df.iloc[i][col]) for col in ['gc_filter', 'gc_upper', 'stoch_rsi_k', 'volume_ma', 'atr']):
                continue
                
            timestamp = df.index[i]
            current_close = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            
            # Check for exits first (if in a position)
            if in_position:
                # Check take profit
                if current_high >= take_profit_level:
                    # Exit at take profit
                    exit_signal = Signal(
                        signal_type=SignalType.EXIT,
                        symbol=symbol,
                        price=take_profit_level,  # Assume we exit at TP level exactly
                        timestamp=timestamp,
                        exit_reason=ExitReason.TAKE_PROFIT
                    )
                    signals.append(exit_signal)
                    in_position = False
                    logger.debug(f"Generated exit signal at take profit: {exit_signal}")
                    
                # Check filter cross
                elif current_close < df.iloc[i]['gc_filter']:
                    # Exit when close crosses below filter
                    exit_signal = Signal(
                        signal_type=SignalType.EXIT,
                        symbol=symbol,
                        price=current_close,
                        timestamp=timestamp,
                        exit_reason=ExitReason.FILTER_CROSS
                    )
                    signals.append(exit_signal)
                    in_position = False
                    logger.debug(f"Generated exit signal on filter cross: {exit_signal}")
            
            # Check for entries (if not in a position)
            if not in_position:
                # Entry conditions:
                # 1. Filter line is rising
                filter_rising = df.iloc[i]['gc_filter'] > df.iloc[i-1]['gc_filter']
                
                # 2. Close above high band
                close_above_hband = current_close > df.iloc[i]['gc_upper']
                
                # 3. Stochastic RSI K-line above threshold
                stoch_k_above_threshold = df.iloc[i]['stoch_rsi_k'] > self.stoch_upper_band
                
                # 4. Volume above MA
                volume_above_ma = df.iloc[i]['volume'] > df.iloc[i]['volume_ma']
                
                # Combined entry condition
                if filter_rising and close_above_hband and stoch_k_above_threshold and volume_above_ma:
                    # Calculate take profit level based on ATR
                    atr_value = df.iloc[i]['atr']
                    entry_price = current_close
                    take_profit_level = entry_price + (atr_value * self.atr_tp_multiplier)
                    
                    # Create entry signal
                    entry_signal = Signal(
                        signal_type=SignalType.ENTRY,
                        symbol=symbol,
                        price=entry_price,
                        timestamp=timestamp,
                        take_profit_level=take_profit_level
                    )
                    signals.append(entry_signal)
                    in_position = True
                    logger.debug(f"Generated entry signal: {entry_signal}")
        
        return signals
        
    def process_candle(self, candle_data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Process a new candle and generate a signal if conditions are met.
        
        Args:
            candle_data: DataFrame containing the latest candle data with indicators
            symbol: Trading symbol
            
        Returns:
            Signal if conditions are met, None otherwise
        """
        if candle_data.empty or len(candle_data) < 2:
            logger.warning(f"Insufficient candle data for {symbol}")
            return None
            
        # Extract the latest candle
        latest_candle = candle_data.iloc[-1]
        prev_candle = candle_data.iloc[-2]
        
        timestamp = latest_candle.name if hasattr(latest_candle, 'name') else datetime.now()
        current_close = latest_candle['close']
        current_high = latest_candle['high']
        
        # Check if we have an open position for this symbol
        if symbol in self.open_positions:
            position_info = self.open_positions[symbol]
            
            # Check take profit
            if current_high >= position_info['take_profit_level']:
                # Exit at take profit
                exit_signal = Signal(
                    signal_type=SignalType.EXIT,
                    symbol=symbol,
                    price=position_info['take_profit_level'],
                    timestamp=timestamp,
                    exit_reason=ExitReason.TAKE_PROFIT
                )
                # Remove position from tracking
                self.open_positions.pop(symbol, None)
                return exit_signal
                
            # Check filter cross
            elif current_close < latest_candle['gc_filter']:
                # Exit when close crosses below filter
                exit_signal = Signal(
                    signal_type=SignalType.EXIT,
                    symbol=symbol,
                    price=current_close,
                    timestamp=timestamp,
                    exit_reason=ExitReason.FILTER_CROSS
                )
                # Remove position from tracking
                self.open_positions.pop(symbol, None)
                return exit_signal
        else:
            # Check entry conditions
            # 1. Filter line is rising
            filter_rising = latest_candle['gc_filter'] > prev_candle['gc_filter']
            
            # 2. Close above high band
            close_above_hband = current_close > latest_candle['gc_upper']
            
            # 3. Stochastic RSI K-line above threshold
            stoch_k_above_threshold = latest_candle['stoch_rsi_k'] > self.stoch_upper_band
            
            # 4. Volume above MA
            volume_above_ma = latest_candle['volume'] > latest_candle['volume_ma']
            
            # Combined entry condition
            if filter_rising and close_above_hband and stoch_k_above_threshold and volume_above_ma:
                # Calculate take profit level based on ATR
                atr_value = latest_candle['atr']
                entry_price = current_close
                take_profit_level = entry_price + (atr_value * self.atr_tp_multiplier)
                
                # Create entry signal
                entry_signal = Signal(
                    signal_type=SignalType.ENTRY,
                    symbol=symbol,
                    price=entry_price,
                    timestamp=timestamp,
                    take_profit_level=take_profit_level
                )
                
                # Track this position
                self.open_positions[symbol] = {
                    'entry_price': entry_price,
                    'entry_time': timestamp,
                    'take_profit_level': take_profit_level
                }
                
                return entry_signal
                
        # No signal conditions met
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
        if 'atr' not in candle_data.columns or candle_data.empty:
            logger.warning("ATR not available for calculating take profit")
            # Default to a percentage-based take profit
            return entry_price * 1.02
            
        atr_value = candle_data['atr'].iloc[-1]
        return entry_price + (atr_value * self.atr_tp_multiplier)
        
    def reset_position_tracking(self):
        """Reset internal position tracking."""
        self.open_positions = {}