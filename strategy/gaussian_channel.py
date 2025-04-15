"""
Gaussian Channel Strategy Module for ZT-3 Trading System.

This module implements the Gaussian Channel trading strategy as defined
in the PineScript, translated to Python for consistent behavior across all modes.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime
from enum import Enum

from strategy import SignalType, ExitReason, Signal, Strategy

logger = logging.getLogger(__name__)

class GaussianChannelStrategy(Strategy):
    """
    Gaussian Channel trading strategy.
    
    Implementation of the PineScript Gaussian Channel Strategy v3.13 with:
    - Stochastic RSI(14,14,3)
    - Volume filter
    - Take Profit at 4.0x ATR
    - Exit on filter cross
    
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
        Initialize the Gaussian Channel strategy with hardcoded parameters 
        to match the PineScript implementation.
        
        Args:
            config: System configuration (not used for strategy parameters)
        """
        super().__init__(config)
        self.name = "GaussianChannelStrategy"
        
        # Hardcoded strategy parameters from PineScript
        # Gaussian Channel parameters
        self.gc_period = 144
        self.gc_multiplier = 1.2
        self.gc_poles = 4
        self.gc_mode_lag = False
        self.gc_mode_fast = False
        self.gc_source = 'hlc3'  # (high + low + close) / 3
        
        # Stochastic RSI parameters
        self.stoch_rsi_length = 14
        self.stoch_length = 14
        self.stoch_k_smooth = 3
        self.stoch_d_smooth = 3
        self.stoch_upper_band = 80.0
        self.stoch_source = 'close'
        
        # Volume filter parameters
        self.use_volume_filter = True
        self.vol_ma_length = 20
        
        # Take profit parameters
        self.use_take_profit = True
        self.atr_length = 14
        self.atr_tp_multiplier = 4.0
        
        # Trailing stop parameters (disabled in PineScript)
        self.use_trailing_stop = False
        self.atr_tsl_multiplier = 2.0
        
        # For position tracking
        self.open_positions = {}  # symbol -> position info
        
        logger.info(f"Initialized Gaussian Channel Strategy with hardcoded parameters: "
                   f"period={self.gc_period}, multiplier={self.gc_multiplier}")
    
    def _calculate_hlc3(self, df: pd.DataFrame) -> pd.Series:
        """Calculate HLC3 price source like in PineScript."""
        return (df['high'] + df['low'] + df['close']) / 3
    
    def _f_filt9x(self, a: float, s: float, i: int, prev_filters: List[float]) -> float:
        """
        Implement the PineScript f_filt9x function.
        
        This is a direct implementation of the filter function in PineScript.
        
        Args:
            a: Alpha parameter
            s: Source value
            i: Filter order
            prev_filters: List of previous filter values
            
        Returns:
            Filter result
        """
        x = 1 - a
        
        # Map i to m values exactly as in PineScript
        if i == 9:
            m2, m3, m4, m5, m6, m7, m8, m9 = 36, 84, 126, 126, 84, 36, 9, 1
        elif i == 8:
            m2, m3, m4, m5, m6, m7, m8, m9 = 28, 56, 70, 56, 28, 8, 1, 0
        elif i == 7:
            m2, m3, m4, m5, m6, m7, m8, m9 = 21, 35, 35, 21, 7, 1, 0, 0
        elif i == 6:
            m2, m3, m4, m5, m6, m7, m8, m9 = 15, 20, 15, 6, 1, 0, 0, 0
        elif i == 5:
            m2, m3, m4, m5, m6, m7, m8, m9 = 10, 10, 5, 1, 0, 0, 0, 0
        elif i == 4:
            m2, m3, m4, m5, m6, m7, m8, m9 = 6, 4, 1, 0, 0, 0, 0, 0
        elif i == 3:
            m2, m3, m4, m5, m6, m7, m8, m9 = 3, 1, 0, 0, 0, 0, 0, 0
        elif i == 2:
            m2, m3, m4, m5, m6, m7, m8, m9 = 1, 0, 0, 0, 0, 0, 0, 0
        else:  # i == 1 or other
            m2, m3, m4, m5, m6, m7, m8, m9 = 0, 0, 0, 0, 0, 0, 0, 0
        
        # Get previous filter values or use 0
        f1 = prev_filters[0] if len(prev_filters) > 0 else 0
        f2 = prev_filters[1] if len(prev_filters) > 1 else 0
        f3 = prev_filters[2] if len(prev_filters) > 2 else 0
        f4 = prev_filters[3] if len(prev_filters) > 3 else 0
        f5 = prev_filters[4] if len(prev_filters) > 4 else 0
        f6 = prev_filters[5] if len(prev_filters) > 5 else 0
        f7 = prev_filters[6] if len(prev_filters) > 6 else 0
        f8 = prev_filters[7] if len(prev_filters) > 7 else 0
        f9 = prev_filters[8] if len(prev_filters) > 8 else 0
        
        # Calculate f as in PineScript
        f = (a ** i) * s + \
            i * x * f1 - \
            (m2 * (x ** 2) * f2 if i >= 2 else 0) + \
            (m3 * (x ** 3) * f3 if i >= 3 else 0) - \
            (m4 * (x ** 4) * f4 if i >= 4 else 0) + \
            (m5 * (x ** 5) * f5 if i >= 5 else 0) - \
            (m6 * (x ** 6) * f6 if i >= 6 else 0) + \
            (m7 * (x ** 7) * f7 if i >= 7 else 0) - \
            (m8 * (x ** 8) * f8 if i >= 8 else 0) + \
            (m9 * (x ** 9) * f9 if i == 9 else 0)
            
        return f
    
    def _f_pole(self, a: float, s: float, i: int, prev_filters: List[List[float]]) -> Tuple[float, float]:
        """
        Implement the PineScript f_pole function.
        
        Args:
            a: Alpha parameter
            s: Source value
            i: Number of poles
            prev_filters: List of previous filter values for each pole
            
        Returns:
            Tuple of (final filter value, first filter value)
        """
        f1 = self._f_filt9x(a, s, 1, prev_filters[0] if len(prev_filters) > 0 else [])
        
        if i >= 2:
            f2 = self._f_filt9x(a, s, 2, prev_filters[1] if len(prev_filters) > 1 else [])
        else:
            f2 = 0
            
        if i >= 3:
            f3 = self._f_filt9x(a, s, 3, prev_filters[2] if len(prev_filters) > 2 else [])
        else:
            f3 = 0
            
        if i >= 4:
            f4 = self._f_filt9x(a, s, 4, prev_filters[3] if len(prev_filters) > 3 else [])
        else:
            f4 = 0
            
        if i >= 5:
            f5 = self._f_filt9x(a, s, 5, prev_filters[4] if len(prev_filters) > 4 else [])
        else:
            f5 = 0
            
        if i >= 6:
            f6 = self._f_filt9x(a, s, 6, prev_filters[5] if len(prev_filters) > 5 else [])
        else:
            f6 = 0
            
        if i >= 7:
            f7 = self._f_filt9x(a, s, 7, prev_filters[6] if len(prev_filters) > 6 else [])
        else:
            f7 = 0
            
        if i >= 8:
            f8 = self._f_filt9x(a, s, 8, prev_filters[7] if len(prev_filters) > 7 else [])
        else:
            f8 = 0
            
        if i == 9:
            f9 = self._f_filt9x(a, s, 9, prev_filters[8] if len(prev_filters) > 8 else [])
        else:
            f9 = 0
            
        # Select the fn based on i
        if i == 1:
            fn = f1
        elif i == 2:
            fn = f2
        elif i == 3:
            fn = f3
        elif i == 4:
            fn = f4
        elif i == 5:
            fn = f5
        elif i == 6:
            fn = f6
        elif i == 7:
            fn = f7
        elif i == 8:
            fn = f8
        elif i == 9:
            fn = f9
        else:
            fn = float('nan')
            
        return fn, f1
    
    def _calculate_gaussian_channel(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gaussian Channel following the exact algorithm from PineScript.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Gaussian Channel indicators added
        """
        if len(df) < self.gc_period:
            logger.warning(f"Not enough data for Gaussian Channel calculation. Need at least {self.gc_period} data points.")
            return df.copy()
        
        # Create a copy to avoid modifying original
        result = df.copy()
        
        # Calculate HLC3
        if 'hlc3' not in result.columns:
            result['hlc3'] = self._calculate_hlc3(result)
        
        # Calculate Gaussian Channel parameters
        beta = (1 - np.cos(4 * np.pi / self.gc_period)) / (np.power(1.414, 2/self.gc_poles) - 1)
        alpha = -beta + np.sqrt(beta**2 + 2*beta)
        lag = (self.gc_period - 1) / (2 * self.gc_poles)
        lag = int(lag)
        
        # Prepare columns
        result['filt'] = np.nan
        result['filttr'] = np.nan
        result['hband'] = np.nan
        result['lband'] = np.nan
        
        # Initialize filter history
        filter_history = [[] for _ in range(self.gc_poles)]
        filter_tr_history = [[] for _ in range(self.gc_poles)]
        
        # Calculate true range for each row
        tr_values = []
        for i in range(len(result)):
            if i == 0:
                tr = result['high'].iloc[i] - result['low'].iloc[i]
            else:
                high_low = result['high'].iloc[i] - result['low'].iloc[i]
                high_close = abs(result['high'].iloc[i] - result['close'].iloc[i-1])
                low_close = abs(result['low'].iloc[i] - result['close'].iloc[i-1])
                tr = max(high_low, high_close, low_close)
            tr_values.append(tr)
        
        result['tr'] = tr_values
        
        # Apply Gaussian filter
        for i in range(len(result)):
            # Get source data
            src_gc = result['hlc3'].iloc[i]
            tr_data = result['tr'].iloc[i]
            
            # Apply lag mode if enabled
            if self.gc_mode_lag and i >= lag:
                src_gc = src_gc + (src_gc - result['hlc3'].iloc[i-lag])
                tr_data = tr_data + (tr_data - result['tr'].iloc[i-lag])
            
            # Calculate filter for source
            filtn, filt1 = self._f_pole(alpha, src_gc, self.gc_poles, filter_history)
            # Update filter history
            for j in range(self.gc_poles):
                if j == 0:
                    filter_history[j].insert(0, filt1)
                else:
                    if j < len(filter_history) and i > j:
                        filter_history[j].insert(0, self._f_filt9x(alpha, src_gc, j+1, filter_history[j-1]))
                # Keep history to required length
                if len(filter_history[j]) > self.gc_poles:
                    filter_history[j] = filter_history[j][:self.gc_poles]
            
            # Calculate filter for true range
            filtntr, filt1tr = self._f_pole(alpha, tr_data, self.gc_poles, filter_tr_history)
            # Update filter TR history
            for j in range(self.gc_poles):
                if j == 0:
                    filter_tr_history[j].insert(0, filt1tr)
                else:
                    if j < len(filter_tr_history) and i > j:
                        filter_tr_history[j].insert(0, self._f_filt9x(alpha, tr_data, j+1, filter_tr_history[j-1]))
                # Keep history to required length
                if len(filter_tr_history[j]) > self.gc_poles:
                    filter_tr_history[j] = filter_tr_history[j][:self.gc_poles]
            
            # Apply fast mode if enabled
            filt = (filtn + filt1) / 2 if self.gc_mode_fast else filtn
            filttr = (filtntr + filt1tr) / 2 if self.gc_mode_fast else filtntr
            
            # Calculate bands
            hband = filt + filttr * self.gc_multiplier
            lband = filt - filttr * self.gc_multiplier
            
            # Store values
            result.loc[result.index[i], 'filt'] = filt
            result.loc[result.index[i], 'filttr'] = filttr
            result.loc[result.index[i], 'hband'] = hband
            result.loc[result.index[i], 'lband'] = lband
        
        return result
    
    def _calculate_stochastic_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic RSI exactly as in PineScript.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Stochastic RSI indicators added
        """
        result = df.copy()
        
        # Calculate RSI
        delta = result['close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        
        avg_up = up.rolling(window=self.stoch_rsi_length).mean()
        avg_down = down.rolling(window=self.stoch_rsi_length).mean()
        
        rs = avg_up / avg_down
        rsi = 100 - (100 / (1 + rs))
        result['rsi'] = rsi
        
        # Calculate Stochastic RSI
        min_rsi = rsi.rolling(window=self.stoch_length).min()
        max_rsi = rsi.rolling(window=self.stoch_length).max()
        
        # Avoid division by zero
        rsi_range = max_rsi - min_rsi
        rsi_range = np.where(rsi_range == 0, 1, rsi_range)
        
        stoch = 100 * (rsi - min_rsi) / rsi_range
        
        # Apply smoothing
        k = stoch.rolling(window=self.stoch_k_smooth).mean()
        d = k.rolling(window=self.stoch_d_smooth).mean()
        
        result['k'] = k
        result['d'] = d
        
        return result
    
    def _calculate_volume_condition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume condition as in PineScript.
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            DataFrame with volume MA added
        """
        result = df.copy()
        
        # Calculate volume MA
        volume_ma = result['volume'].rolling(window=self.vol_ma_length).mean()
        result['volume_ma'] = volume_ma
        
        # Calculate volume condition
        result['volume_condition'] = (~self.use_volume_filter) | (result['volume'] > result['volume_ma'])
        
        return result
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR as in PineScript.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with ATR added
        """
        result = df.copy()
        
        # Ensure true range exists
        if 'tr' not in result.columns:
            tr_values = []
            for i in range(len(result)):
                if i == 0:
                    tr = result['high'].iloc[i] - result['low'].iloc[i]
                else:
                    high_low = result['high'].iloc[i] - result['low'].iloc[i]
                    high_close = abs(result['high'].iloc[i] - result['close'].iloc[i-1])
                    low_close = abs(result['low'].iloc[i] - result['close'].iloc[i-1])
                    tr = max(high_low, high_close, low_close)
                tr_values.append(tr)
            result['tr'] = tr_values
        
        # Calculate ATR using EMA like in PineScript
        result['atr'] = result['tr'].ewm(span=self.atr_length, adjust=False).mean()
        
        return result
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare all indicators needed for the strategy.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        if len(df) < self.gc_period + 10:  # Need enough data
            logger.warning(f"Not enough data for indicators. Need at least {self.gc_period + 10} candles.")
            return df
        
        # Calculate HLC3
        df['hlc3'] = self._calculate_hlc3(df)
        
        # Calculate Gaussian Channel
        df = self._calculate_gaussian_channel(df)
        
        # Calculate Stochastic RSI
        df = self._calculate_stochastic_rsi(df)
        
        # Calculate Volume condition
        df = self._calculate_volume_condition(df)
        
        # Calculate ATR
        df = self._calculate_atr(df)
        
        return df
    
    def process_candle(self, candle_data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Process a new candle and generate a signal if conditions are met.
        
        Args:
            candle_data: DataFrame containing candle data
            symbol: Trading symbol
            
        Returns:
            Signal if conditions are met, None otherwise
        """
        if len(candle_data) < self.gc_period + 10:  # Need enough data
            logger.warning(f"Not enough data for {symbol} to generate reliable signals")
            return None
            
        # Prepare data with all indicators
        data = self.prepare_data(candle_data)
        
        # Skip if we don't have enough processed data
        if len(data) < 2:
            return None
            
        # Get the last two candles for signal generation
        current = data.iloc[-1]
        previous = data.iloc[-2]
        timestamp = current.name
        
        # Check if we have an open position for this symbol
        if symbol in self.open_positions:
            position_info = self.open_positions[symbol]
            
            # Check take profit
            if self.use_take_profit and current['high'] >= position_info['take_profit_level']:
                # Exit at take profit
                exit_signal = Signal(
                    signal_type=SignalType.EXIT,
                    symbol=symbol,
                    price=position_info['take_profit_level'],  # Exit at TP level
                    timestamp=timestamp,
                    exit_reason=ExitReason.TAKE_PROFIT
                )
                # Remove position from tracking
                self.open_positions.pop(symbol, None)
                return exit_signal
                
            # Check filter crossunder
            elif current['close'] < current['filt'] and not (previous['close'] < previous['filt']):
                # Exit when price closes below filter
                exit_signal = Signal(
                    signal_type=SignalType.EXIT,
                    symbol=symbol,
                    price=current['close'],
                    timestamp=timestamp,
                    exit_reason=ExitReason.FILTER_CROSS
                )
                # Remove position from tracking
                self.open_positions.pop(symbol, None)
                return exit_signal
        else:
            # Entry conditions exactly as in PineScript
            # 1. Filter line is rising
            filter_rising = current['filt'] > previous['filt']
            
            # 2. Close above high band
            close_above_hband = current['close'] > current['hband']
            
            # 3. Stochastic RSI K-line above threshold
            stoch_k_above_threshold = current['k'] > self.stoch_upper_band
            
            # 4. Volume condition
            volume_condition = True
            if self.use_volume_filter:
                volume_condition = current['volume'] > current['volume_ma']
            
            # Combined entry condition
            if filter_rising and close_above_hband and stoch_k_above_threshold and volume_condition:
                # Calculate take profit level
                entry_price = current['close']
                atr_value = current['atr']
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
                
        # No signal generated
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
        data = candle_data
        if 'atr' not in data.columns:
            data = self._calculate_atr(candle_data)
            
        if data.empty:
            logger.warning("No data available for calculating take profit")
            return entry_price * 1.02  # Default 2% take profit
        
        atr_value = data['atr'].iloc[-1]
        return entry_price + (atr_value * self.atr_tp_multiplier)
    
    def reset_position_tracking(self):
        """Reset internal position tracking."""
        self.open_positions = {}