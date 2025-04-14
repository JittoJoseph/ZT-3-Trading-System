"""
Technical Indicators Module for ZT-3 Trading System.

This module provides implementations of technical indicators used
for the Gaussian Channel strategy and other technical analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class Indicators:
    """
    Technical indicators for the ZT-3 Trading System.
    
    Provides implementations of indicators used in the Gaussian Channel strategy
    and other technical analysis methods.
    """
    
    @staticmethod
    def gaussian_channel(df: pd.DataFrame,
                        source_col: str = 'hlc3',
                        period: int = 144,
                        multiplier: float = 1.2,
                        poles: int = 4,
                        reduced_lag: bool = False,
                        fast_response: bool = False) -> pd.DataFrame:
        """
        Calculate the Gaussian Channel indicator.
        
        Args:
            df: DataFrame with price data
            source_col: Source column to use for calculations ('hlc3', 'close', etc.)
            period: Gaussian Channel sampling period
            multiplier: True range multiplier for bands
            poles: Number of poles (1-9)
            reduced_lag: Use reduced lag mode
            fast_response: Use fast response mode
            
        Returns:
            DataFrame with Gaussian Channel indicators added
        """
        # Create a copy of the dataframe to avoid modifying the original
        result = df.copy()
        
        # Ensure source column exists
        if source_col not in result.columns:
            if source_col == 'hlc3' and all(col in result.columns for col in ['high', 'low', 'close']):
                # Create HLC3 column if it doesn't exist
                result[source_col] = (result['high'] + result['low'] + result['close']) / 3
            else:
                logger.error(f"Source column '{source_col}' not found in dataframe")
                return result
        
        # Calculate Gaussian Channel parameters
        beta = (1 - np.cos(4 * np.pi / period)) / ((np.power(1.414, 2 / poles)) - 1)
        alpha = -beta + np.sqrt(beta * beta + 2 * beta)
        lag = (period - 1) / (2 * poles)
        
        # Prepare source data
        srcdata = result[source_col].values
        if reduced_lag:
            # Create lagged version for reduced lag mode
            srcdata_lag = np.roll(srcdata, int(lag))
            srcdata_lag[:int(lag)] = srcdata[:int(lag)]  # Avoid lookforward bias
            srcdata = srcdata + (srcdata - srcdata_lag)
        
        # Calculate true range for bands
        high_vals = result['high'].values
        low_vals = result['low'].values
        close_vals = result['close'].values
        prev_close = np.roll(close_vals, 1)
        prev_close[0] = close_vals[0]  # Avoid lookforward bias
        
        tr = np.maximum(high_vals - low_vals, 
                        np.maximum(np.abs(high_vals - prev_close), 
                                   np.abs(low_vals - prev_close)))
        
        if reduced_lag:
            # Create lagged version for reduced lag mode
            tr_lag = np.roll(tr, int(lag))
            tr_lag[:int(lag)] = tr[:int(lag)]  # Avoid lookforward bias
            trdata = tr + (tr - tr_lag)
        else:
            trdata = tr
        
        # Implement the filter calculations
        filtn = Indicators._gaussian_filter(srcdata, alpha, poles)
        filt1 = Indicators._gaussian_filter(srcdata, alpha, 1)
        
        filtntr = Indicators._gaussian_filter(trdata, alpha, poles)
        filt1tr = Indicators._gaussian_filter(trdata, alpha, 1)
        
        # Apply fast response mode if requested
        if fast_response:
            filt = (filtn + filt1) / 2
            filttr = (filtntr + filt1tr) / 2
        else:
            filt = filtn
            filttr = filtntr
        
        # Calculate bands
        hband = filt + filttr * multiplier
        lband = filt - filttr * multiplier
        
        # Add calculated values to the result dataframe
        result['gc_filter'] = filt
        result['gc_upper'] = hband
        result['gc_lower'] = lband
        
        return result
    
    @staticmethod
    def _gaussian_filter(data: np.ndarray, alpha: float, poles: int) -> np.ndarray:
        """
        Internal method to calculate the Gaussian filter.
        
        Args:
            data: Input data array
            alpha: Alpha parameter
            poles: Number of poles (1-9)
            
        Returns:
            Filtered data
        """
        if poles < 1 or poles > 9:
            logger.warning("Poles must be between 1 and 9. Using 4 as default.")
            poles = 4
        
        result = np.zeros_like(data, dtype=float)
        
        if poles == 1:
            # Single pole implementation (simpler case)
            a = alpha
            b = 1 - a
            prev_val = data[0]
            
            for i in range(len(data)):
                curr_val = a * data[i] + b * prev_val
                result[i] = curr_val
                prev_val = curr_val
                
            return result
        
        # Multi-pole implementation
        filtered = []
        
        # First calculate each pole's filtered series
        for pole in range(1, poles + 1):
            pole_result = Indicators._filter_pole(data, alpha, pole)
            filtered.append(pole_result)
        
        # Use the requested number of poles
        return filtered[poles - 1]
    
    @staticmethod
    def _filter_pole(data: np.ndarray, alpha: float, pole: int) -> np.ndarray:
        """
        Calculate a single pole of the Gaussian filter.
        
        Args:
            data: Input data array
            alpha: Alpha parameter
            pole: Pole number (1-9)
            
        Returns:
            Filtered data for this pole
        """
        # Initialize coefficients based on pole number
        coef = {}
        
        # m2 to m9 coefficients based on pole number
        if pole >= 2:
            coef['m2'] = 36 if pole == 9 else \
                        28 if pole == 8 else \
                        21 if pole == 7 else \
                        15 if pole == 6 else \
                        10 if pole == 5 else \
                         6 if pole == 4 else \
                         3 if pole == 3 else \
                         1 if pole == 2 else 0
        
        if pole >= 3:
            coef['m3'] = 84 if pole == 9 else \
                        56 if pole == 8 else \
                        35 if pole == 7 else \
                        20 if pole == 6 else \
                        10 if pole == 5 else \
                         4 if pole == 4 else \
                         1 if pole == 3 else 0
        
        if pole >= 4:
            coef['m4'] = 126 if pole == 9 else \
                         70 if pole == 8 else \
                         35 if pole == 7 else \
                         15 if pole == 6 else \
                          5 if pole == 5 else \
                          1 if pole == 4 else 0
        
        if pole >= 5:
            coef['m5'] = 126 if pole == 9 else \
                         56 if pole == 8 else \
                         21 if pole == 7 else \
                          6 if pole == 6 else \
                          1 if pole == 5 else 0
        
        if pole >= 6:
            coef['m6'] = 84 if pole == 9 else \
                         28 if pole == 8 else \
                          7 if pole == 7 else \
                          1 if pole == 6 else 0
        
        if pole >= 7:
            coef['m7'] = 36 if pole == 9 else \
                          8 if pole == 8 else \
                          1 if pole == 7 else 0
        
        if pole >= 8:
            coef['m8'] = 9 if pole == 9 else \
                         1 if pole == 8 else 0
        
        if pole >= 9:
            coef['m9'] = 1 if pole == 9 else 0
        
        # Calculate the filtered series
        result = np.zeros_like(data, dtype=float)
        beta = 1 - alpha
        
        # Previous values storage for up to 9 poles
        prev_vals = [0.0] * 10
        
        for i in range(len(data)):
            # Update previous values
            for j in range(9, 0, -1):
                prev_vals[j] = prev_vals[j-1]
            
            # Calculate new value
            val = alpha ** pole * data[i]
            val += pole * beta * prev_vals[1]
            
            if pole >= 2:
                val -= coef['m2'] * (beta ** 2) * prev_vals[2]
            if pole >= 3:
                val += coef['m3'] * (beta ** 3) * prev_vals[3]
            if pole >= 4:
                val -= coef['m4'] * (beta ** 4) * prev_vals[4]
            if pole >= 5:
                val += coef['m5'] * (beta ** 5) * prev_vals[5]
            if pole >= 6:
                val -= coef['m6'] * (beta ** 6) * prev_vals[6]
            if pole >= 7:
                val += coef['m7'] * (beta ** 7) * prev_vals[7]
            if pole >= 8:
                val -= coef['m8'] * (beta ** 8) * prev_vals[8]
            if pole >= 9:
                val += coef['m9'] * (beta ** 9) * prev_vals[9]
            
            result[i] = val
            prev_vals[0] = val
        
        return result
    
    @staticmethod
    def stochastic_rsi(df: pd.DataFrame,
                      rsi_length: int = 14,
                      stoch_length: int = 14, 
                      k_smoothing: int = 3,
                      d_smoothing: int = 3,
                      source_col: str = 'close') -> pd.DataFrame:
        """
        Calculate Stochastic RSI indicator.
        
        Args:
            df: DataFrame with price data
            rsi_length: RSI lookback period
            stoch_length: Stochastic lookback period
            k_smoothing: K line smoothing period
            d_smoothing: D line smoothing period
            source_col: Source column for calculations
            
        Returns:
            DataFrame with Stochastic RSI added
        """
        # Create a copy of the dataframe to avoid modifying the original
        result = df.copy()
        
        # Ensure source column exists
        if source_col not in result.columns:
            logger.error(f"Source column '{source_col}' not found in dataframe")
            return result
        
        # Step 1: Calculate RSI
        delta = result[source_col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_length).mean()
        avg_loss = loss.rolling(window=rsi_length).mean()
        
        # Handle division by zero
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        # Step 2: Calculate Stochastic of RSI
        rsi_min = rsi.rolling(window=stoch_length).min()
        rsi_max = rsi.rolling(window=stoch_length).max()
        
        # Handle division by zero
        denominator = rsi_max - rsi_min
        denominator = denominator.replace(0, np.finfo(float).eps)
        
        stoch = 100 * (rsi - rsi_min) / denominator
        
        # Step 3: Apply smoothing to K line
        k = stoch.rolling(window=k_smoothing).mean()
        
        # Step 4: Calculate D line (moving average of K)
        d = k.rolling(window=d_smoothing).mean()
        
        # Add calculated values to the result dataframe
        result['rsi'] = rsi
        result['stoch_rsi_k'] = k
        result['stoch_rsi_d'] = d
        
        return result
    
    @staticmethod
    def volume_ma(df: pd.DataFrame, 
                 length: int = 20, 
                 source_col: str = 'volume') -> pd.DataFrame:
        """
        Calculate Volume Moving Average.
        
        Args:
            df: DataFrame with price and volume data
            length: Moving average period
            source_col: Source column for calculations
            
        Returns:
            DataFrame with Volume MA added
        """
        # Create a copy of the dataframe to avoid modifying the original
        result = df.copy()
        
        # Ensure source column exists
        if source_col not in result.columns:
            logger.error(f"Source column '{source_col}' not found in dataframe")
            return result
        
        # Calculate simple moving average of volume
        result['volume_ma'] = result[source_col].rolling(window=length).mean()
        
        return result
    
    @staticmethod
    def atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: DataFrame with price data (must have 'high', 'low', 'close')
            length: ATR period
            
        Returns:
            DataFrame with ATR added
        """
        # Create a copy of the dataframe to avoid modifying the original
        result = df.copy()
        
        # Check required columns
        if not all(col in result.columns for col in ['high', 'low', 'close']):
            logger.error("ATR calculation requires high, low, and close columns")
            return result
        
        # Calculate true range
        high_low = result['high'] - result['low']
        high_close_prev = np.abs(result['high'] - result['close'].shift(1))
        low_close_prev = np.abs(result['low'] - result['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR (simple moving average of true range)
        result['atr'] = tr.rolling(window=length).mean()
        
        return result
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate all indicators needed for the Gaussian Channel strategy.
        
        Args:
            df: DataFrame with price and volume data
            config: Configuration dictionary with indicator parameters
            
        Returns:
            DataFrame with all indicators added
        """
        # Extract indicator parameters from config
        strategy_params = config.get('strategy', {}).get('params', {})
        
        # Gaussian Channel parameters
        gc_period = strategy_params.get('gc_period', 144)
        gc_multiplier = strategy_params.get('gc_multiplier', 1.2)
        gc_poles = strategy_params.get('gc_poles', 4)
        
        # Stochastic RSI parameters
        rsi_length = strategy_params.get('stoch_rsi_length', 14)
        stoch_length = strategy_params.get('stoch_length', 14)
        k_smooth = strategy_params.get('stoch_k_smooth', 3)
        
        # Volume MA parameters
        vol_ma_length = strategy_params.get('vol_ma_length', 20)
        
        # ATR parameters
        atr_length = strategy_params.get('atr_length', 14)
        
        # Create a copy of the dataframe to avoid modifying the original
        result = df.copy()
        
        # Calculate HLC3 (typical price) if not already present
        if 'hlc3' not in result.columns:
            result['hlc3'] = (result['high'] + result['low'] + result['close']) / 3
        
        # Calculate each indicator
        logger.info("Calculating indicators")
        
        # Step 1: ATR
        result = Indicators.atr(result, atr_length)
        
        # Step 2: Gaussian Channel
        result = Indicators.gaussian_channel(
            result, 
            source_col='hlc3',
            period=gc_period,
            multiplier=gc_multiplier,
            poles=gc_poles
        )
        
        # Step 3: Stochastic RSI
        result = Indicators.stochastic_rsi(
            result,
            rsi_length=rsi_length,
            stoch_length=stoch_length,
            k_smoothing=k_smooth
        )
        
        # Step 4: Volume MA
        if 'volume' in result.columns:
            result = Indicators.volume_ma(
                result,
                length=vol_ma_length
            )
        else:
            logger.warning("Volume data not available, skipping Volume MA calculation")
        
        logger.info("Indicator calculation complete")
        return result