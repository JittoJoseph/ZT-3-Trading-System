"""
Gaussian Channel Strategy implementation for ZT-3 Trading System.

This strategy is based on the following conditions:
- Entry: Gaussian filter rising, price closes above high band, 
         Stochastic RSI(14,14,3) > 80, volume > 20MA
- Exit: Take profit at 4.0x ATR from entry OR price closes below Gaussian filter
"""

import numpy as np
import pandas as pd
import math
import logging
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals that can be generated."""
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    NEUTRAL = "NEUTRAL"

class ExitReason(Enum):
    """Reasons for exit signals."""
    TAKE_PROFIT = "TAKE_PROFIT"
    FILTER_CROSS = "FILTER_CROSS"
    TRAILING_STOP = "TRAILING_STOP"
    NONE = "NONE"

class GaussianChannelStrategy:
    """
    Implements the Gaussian Channel Strategy with Stochastic RSI filter.
    
    This class calculates all technical indicators and generates trading signals
    based on the strategy rules defined in the Pine script.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy with configuration parameters.
        
        Args:
            config: Configuration dictionary containing strategy parameters.
        """
        self.params = config['strategy']['params']
        
        # Extract parameters from config with defaults if missing
        self.gc_source = self.params.get('gc_source', 'hlc3')
        self.gc_poles = int(self.params.get('gc_poles', 4))
        self.gc_period = int(self.params.get('gc_period', 144))
        self.gc_multiplier = float(self.params.get('gc_multiplier', 1.2))
        self.gc_reduced_lag_mode = self.params.get('gc_reduced_lag_mode', False)
        self.gc_fast_response_mode = self.params.get('gc_fast_response_mode', False)
        
        self.stoch_rsi_length = int(self.params.get('stoch_rsi_length', 14))
        self.stoch_length = int(self.params.get('stoch_length', 14))
        self.stoch_k_smooth = int(self.params.get('stoch_k_smooth', 3))
        self.stoch_d_smooth = int(self.params.get('stoch_d_smooth', 3))
        self.stoch_upper_band = float(self.params.get('stoch_upper_band', 80.0))
        
        self.use_volume_filter = self.params.get('use_volume_filter', True)
        self.volume_ma_length = int(self.params.get('volume_ma_length', 20))
        
        self.use_trailing_stop = self.params.get('use_trailing_stop', False)
        self.atr_length = int(self.params.get('atr_length', 14))
        self.atr_multiplier_tsl = float(self.params.get('atr_multiplier_tsl', 2.0))
        self.use_take_profit = self.params.get('use_take_profit', True)
        self.atr_multiplier_tp = float(self.params.get('atr_multiplier_tp', 4.0))
        
        # Trade state tracking
        self.in_position = False
        self.entry_price = 0.0
        self.take_profit_level = None
        self.trail_stop_level = None
        
        logger.info(f"Initialized Gaussian Channel Strategy with period={self.gc_period}, multiplier={self.gc_multiplier}")
    
    def calculate_gaussian_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Gaussian Channel filter and bands.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with added Gaussian Channel columns.
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate source based on config
        if self.gc_source == 'hlc3':
            result['src'] = (result['high'] + result['low'] + result['close']) / 3
        elif self.gc_source == 'ohlc4':
            result['src'] = (result['open'] + result['high'] + result['low'] + result['close']) / 4
        elif self.gc_source == 'close':
            result['src'] = result['close']
        else:
            result['src'] = (result['high'] + result['low'] + result['close']) / 3  # Default to hlc3
            
        # Calculate True Range
        result['tr'] = np.maximum(
            result['high'] - result['low'],
            np.maximum(
                np.abs(result['high'] - result['close'].shift(1)),
                np.abs(result['low'] - result['close'].shift(1))
            )
        )
        
        # Calculate beta and alpha parameters for Gaussian filter
        beta = (1 - math.cos(4 * math.asin(1) / self.gc_period)) / (math.pow(1.414, 2 / self.gc_poles) - 1)
        alpha = -beta + math.sqrt(math.pow(beta, 2) + 2 * beta)
        
        # Apply reduced lag mode if enabled
        if self.gc_reduced_lag_mode:
            lag = (self.gc_period - 1) / (2 * self.gc_poles)
            lag = int(lag)
            result['srcdata'] = result['src'] + (result['src'] - result['src'].shift(lag))
            result['trdata'] = result['tr'] + (result['tr'] - result['tr'].shift(lag))
        else:
            result['srcdata'] = result['src']
            result['trdata'] = result['tr']
        
        # Calculate Gaussian filter
        # This is a recursive calculation that simulates the pole functions from Pine
        result['filt'] = self._calculate_pole_filter(result['srcdata'].values, alpha, self.gc_poles)
        result['filttr'] = self._calculate_pole_filter(result['trdata'].values, alpha, self.gc_poles)
        
        # Apply fast response mode if enabled
        if self.gc_fast_response_mode:
            filt1 = self._calculate_pole_filter(result['srcdata'].values, alpha, 1)
            filt1tr = self._calculate_pole_filter(result['trdata'].values, alpha, 1)
            result['filt'] = (result['filt'] + filt1) / 2
            result['filttr'] = (result['filttr'] + filt1tr) / 2
            
        # Calculate bands
        result['hband'] = result['filt'] + result['filttr'] * self.gc_multiplier
        result['lband'] = result['filt'] - result['filttr'] * self.gc_multiplier
        
        return result
    
    def _calculate_pole_filter(self, data: np.ndarray, alpha: float, poles: int) -> np.ndarray:
        """
        Calculate the Gaussian filter pole function.
        
        Args:
            data: Input data array
            alpha: Alpha parameter
            poles: Number of poles
            
        Returns:
            Filtered data array
        """
        # Initialize result array
        length = len(data)
        result = np.zeros(length)
        
        # Apply filter for each pole recursively
        for p in range(poles):
            pole_result = np.zeros(length)
            
            # First value initialization
            if not np.isnan(data[0]):
                pole_result[0] = data[0]
            
            # Apply filter formula
            for i in range(1, length):
                if np.isnan(data[i]):
                    pole_result[i] = pole_result[i-1]
                else:
                    pole_result[i] = (1 - alpha) * pole_result[i-1] + alpha * data[i]
            
            # Set this pole's result as input for next pole
            data = pole_result
        
        return data
    
    def calculate_stochastic_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic RSI indicator.
        
        Args:
            df: DataFrame with price data.
            
        Returns:
            DataFrame with added Stochastic RSI columns.
        """
        result = df.copy()
        
        # Calculate RSI
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=self.stoch_rsi_length).mean()
        avg_loss = loss.rolling(window=self.stoch_rsi_length).mean()
        
        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic from RSI
        result['rsi_min'] = result['rsi'].rolling(window=self.stoch_length).min()
        result['rsi_max'] = result['rsi'].rolling(window=self.stoch_length).max()
        
        # Calculate Stochastic RSI
        result['stoch_rsi'] = ((result['rsi'] - result['rsi_min']) / 
                           (result['rsi_max'] - result['rsi_min'] + 1e-9)) * 100
                           
        # Apply smoothing
        result['stoch_k'] = result['stoch_rsi'].rolling(window=self.stoch_k_smooth).mean()
        result['stoch_d'] = result['stoch_k'].rolling(window=self.stoch_d_smooth).mean()
        
        return result
    
    def calculate_volume_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume moving average.
        
        Args:
            df: DataFrame with volume data.
            
        Returns:
            DataFrame with added volume MA column.
        """
        result = df.copy()
        result['volume_ma'] = result['volume'].rolling(window=self.volume_ma_length).mean()
        return result
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            DataFrame with added ATR column.
        """
        result = df.copy()
        result['atr'] = result['tr'].rolling(window=self.atr_length).mean()
        return result
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process market data through all indicators.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with all indicators and signals.
        """
        # Ensure df has the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate all indicators
        result = self.calculate_gaussian_filter(df)
        result = self.calculate_stochastic_rsi(result)
        result = self.calculate_volume_ma(result)
        result = self.calculate_atr(result)
        
        # Generate signals based on indicators
        result['signal_type'] = SignalType.NEUTRAL.value
        result['exit_reason'] = ExitReason.NONE.value
        
        # Process each candle for signals
        for i in range(1, len(result)):
            self._process_candle(result, i)
            
        return result
    
    def _process_candle(self, df: pd.DataFrame, idx: int) -> None:
        """
        Process a single candle for entry/exit signals.
        
        Args:
            df: DataFrame with indicators.
            idx: Index of the candle to process.
        """
        # Skip the first candle as we need previous values
        if idx <= 1:
            return
            
        # Get current and previous values
        current = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        # Check entry conditions
        filter_rising = current['filt'] > prev['filt']
        close_above_hband = current['close'] > current['hband']
        stoch_above_threshold = current['stoch_k'] > self.stoch_upper_band
        volume_condition = not self.use_volume_filter or (current['volume'] > current['volume_ma'])
        
        entry_condition = (
            filter_rising and 
            close_above_hband and 
            stoch_above_threshold and 
            volume_condition and 
            not self.in_position
        )
        
        # Check exit conditions
        take_profit_hit = False
        filter_cross_exit = False
        trail_stop_hit = False
        
        # Only calculate exit if in a position
        if self.in_position:
            # Take profit condition
            if self.use_take_profit and self.take_profit_level is not None:
                take_profit_hit = current['high'] >= self.take_profit_level
            
            # Filter cross condition
            filter_cross_exit = current['close'] < current['filt'] and prev['close'] >= prev['filt']
            
            # Trail stop condition
            if self.use_trailing_stop and self.trail_stop_level is not None:
                trail_stop_hit = current['low'] <= self.trail_stop_level
                
                # Update trailing stop if price moved higher
                new_stop = current['filt'] - current['atr'] * self.atr_multiplier_tsl
                self.trail_stop_level = max(new_stop, self.trail_stop_level)
        
        # Apply entry logic
        if entry_condition:
            df.at[df.index[idx], 'signal_type'] = SignalType.ENTRY.value
            
            # Update state
            self.in_position = True
            self.entry_price = current['close']
            
            # Calculate take profit level
            if self.use_take_profit:
                self.take_profit_level = self.entry_price + current['atr'] * self.atr_multiplier_tp
            
            # Calculate initial trailing stop if enabled
            if self.use_trailing_stop:
                self.trail_stop_level = current['filt'] - current['atr'] * self.atr_multiplier_tsl
                
            logger.info(f"ENTRY signal at price: {self.entry_price:.2f}, TP: {self.take_profit_level:.2f}")
        
        # Apply exit logic
        elif self.in_position and (take_profit_hit or filter_cross_exit or trail_stop_hit):
            df.at[df.index[idx], 'signal_type'] = SignalType.EXIT.value
            
            # Set exit reason
            if take_profit_hit:
                df.at[df.index[idx], 'exit_reason'] = ExitReason.TAKE_PROFIT.value
                logger.info(f"EXIT signal (Take Profit) at price: {current['high']:.2f}")
            elif trail_stop_hit:
                df.at[df.index[idx], 'exit_reason'] = ExitReason.TRAILING_STOP.value
                logger.info(f"EXIT signal (Trail Stop) at price: {current['low']:.2f}")
            else:
                df.at[df.index[idx], 'exit_reason'] = ExitReason.FILTER_CROSS.value
                logger.info(f"EXIT signal (Filter Cross) at price: {current['close']:.2f}")
            
            # Reset state
            self.in_position = False
            self.entry_price = 0.0
            self.take_profit_level = None
            self.trail_stop_level = None
    
    def generate_signal(self, candle: Dict[str, Any], position_size: int = 0) -> Tuple[SignalType, ExitReason, Optional[Dict[str, Any]]]:
        """
        Generate a trading signal based on the latest candle.
        
        Args:
            candle: Latest price candle with indicators.
            position_size: Current position size (positive for long, negative for short, 0 for flat).
            
        Returns:
            Tuple containing:
            - Signal type (ENTRY, EXIT, NEUTRAL)
            - Exit reason if applicable
            - Dictionary with additional signal information
        """
        # Set up the position state based on position_size
        self.in_position = position_size > 0
        
        # Create a single-row DataFrame to process
        df = pd.DataFrame([candle])
        
        # Process the data (this handles just a single candle in this case)
        result = self.process_data(df)
        
        # Extract the signal
        signal_type = SignalType(result['signal_type'].iloc[0])
        exit_reason = ExitReason(result['exit_reason'].iloc[0]) if signal_type == SignalType.EXIT else ExitReason.NONE
        
        # Create additional information for the signal
        signal_info = {
            'filter_value': result['filt'].iloc[0],
            'high_band': result['hband'].iloc[0],
            'low_band': result['lband'].iloc[0],
            'stoch_k': result['stoch_k'].iloc[0],
            'volume_ma': result['volume_ma'].iloc[0],
            'atr': result['atr'].iloc[0],
            'take_profit_level': self.take_profit_level,
            'trail_stop_level': self.trail_stop_level
        }
        
        return signal_type, exit_reason, signal_info