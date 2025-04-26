"""
Technical Indicators Module for ZT-3 Trading System.

This module provides implementations of various technical indicators
used by the trading strategies.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple

logger = logging.getLogger(__name__)

class Indicators:
    """
    Technical indicators for trading strategies.
    
    This class implements various technical indicators that may be used
    by different trading strategies.
    """
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14, 
           high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).
        Used by SwingProStrategy.
        
        Args:
            df: DataFrame with price data
            period: Period for ATR calculation
            high_col: Column name with high prices
            low_col: Column name with low prices
            close_col: Column name with close prices
            
        Returns:
            DataFrame with added column: atr
        """
        if len(df) < period + 1:
            logger.warning(f"Not enough data for ATR calculation. Need at least {period + 1} data points.")
            return df.copy()
            
        # Create a copy of the input DataFrame to avoid modifying it
        result = df.copy()
        
        # Calculate True Range
        result['tr0'] = abs(result[high_col] - result[low_col])
        result['tr1'] = abs(result[high_col] - result[close_col].shift())
        result['tr2'] = abs(result[low_col] - result[close_col].shift())
        result['tr'] = result[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Calculate ATR
        result['atr'] = result['tr'].ewm(span=period, adjust=False).mean()
        
        # Drop temporary columns
        result.drop(columns=['tr0', 'tr1', 'tr2', 'tr'], inplace=True)
        
        return result

    @staticmethod
    def rsi(df: pd.DataFrame, close_col: str = 'close', period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        Used by SwingProStrategy.

        Args:
            df: DataFrame with price data
            close_col: Column name with close prices
            period: Period for RSI calculation

        Returns:
            DataFrame with added column: rsi
        """
        if len(df) < period + 1:
            logger.warning(f"Not enough data for RSI calculation. Need at least {period + 1} data points.")
            return df.copy()

        result = df.copy()

        # Calculate price differences
        close_delta = result[close_col].diff()

        # Make two series: one for gains and one for losses
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        # Calculate EMAs for gains and losses
        # Use Wilder's smoothing method (equivalent to SMA for the first value, then EMA)
        # Pandas ewm with com=period-1 approximates this well for large datasets
        up_ema = up.ewm(com=period - 1, adjust=False).mean()
        down_ema = down.ewm(com=period - 1, adjust=False).mean()

        # Calculate RS and RSI
        rs = up_ema / down_ema
        rsi_series = 100.0 - (100.0 / (1.0 + rs))

        # Handle potential NaN/inf values at the beginning or due to zero division
        rsi_series = rsi_series.bfill() # Backfill initial NaNs (replaces fillna with method)
        rsi_series = rsi_series.replace([np.inf, -np.inf], 100.0) # Replace inf with 100 (or 0 if needed)

        result['rsi'] = rsi_series # Assign the modified series back to the DataFrame

        return result

    @staticmethod
    def macd(df: pd.DataFrame, close_col: str = 'close',
             fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        Used by SwingProStrategy.

        Args:
            df: DataFrame with price data
            close_col: Column name with close prices
            fast_period: Period for the fast EMA
            slow_period: Period for the slow EMA
            signal_period: Period for the signal line EMA

        Returns:
            DataFrame with added columns: macd_line, macd_signal, macd_hist
        """
        if len(df) < slow_period + signal_period:
            logger.warning("Not enough data for MACD calculation")
            return df.copy()

        result = df.copy()

        # Calculate Fast and Slow EMAs
        ema_fast = result[close_col].ewm(span=fast_period, adjust=False).mean()
        ema_slow = result[close_col].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD Line
        result['macd_line'] = ema_fast - ema_slow

        # Calculate Signal Line (EMA of MACD Line)
        result['macd_signal'] = result['macd_line'].ewm(span=signal_period, adjust=False).mean()

        # Calculate MACD Histogram
        result['macd_hist'] = result['macd_line'] - result['macd_signal']

        return result

    @staticmethod
    def ema(df: pd.DataFrame, source_col: str = 'close', period: int = 20) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average (EMA).
        Used by SwingProStrategy.

        Args:
            df: DataFrame with price data
            source_col: Column name to use as source data
            period: Period for the EMA

        Returns:
            DataFrame with added column: ema_<period>
        """
        if len(df) < period:
            logger.warning(f"Not enough data for EMA({period}) calculation")
            return df.copy()

        result = df.copy()
        ema_col_name = f'ema_{period}'
        result[ema_col_name] = result[source_col].ewm(span=period, adjust=False).mean()
        return result

    @staticmethod
    def sma(df: pd.DataFrame, source_col: str = 'close', period: int = 20) -> pd.DataFrame:
        """
        Calculate Simple Moving Average (SMA).

        Args:
            df: DataFrame with price data
            source_col: Column name to use as source data
            period: Period for the SMA

        Returns:
            DataFrame with added column: sma_<source>_<period>
        """
        if len(df) < period:
            logger.warning(f"Not enough data for SMA({period}) on {source_col} calculation")
            return df.copy()

        result = df.copy()
        sma_col_name = f'sma_{source_col}_{period}' # Include source in name for clarity
        result[sma_col_name] = result[source_col].rolling(window=period).mean()
        return result

    @staticmethod
    def normalize_macd_hist(df: pd.DataFrame, hist_col: str = 'macd_hist', period: int = 50) -> pd.DataFrame:
        """
        Calculates a normalization factor for the MACD histogram based on its
        rolling maximum absolute value over a specified period.

        Args:
            df: DataFrame with MACD histogram calculated (must contain hist_col).
            hist_col: Column name of the MACD histogram.
            period: The lookback period for finding the max absolute value.

        Returns:
            DataFrame with added column: macd_hist_norm_factor
        """
        if hist_col not in df.columns:
            logger.error(f"MACD Histogram column '{hist_col}' not found for normalization.")
            return df.copy()
        if len(df) < period:
            logger.warning(f"Not enough data for MACD Hist normalization ({len(df)} < {period}). Factor will be NaN initially.")
            # Avoid returning early, let rolling handle initial NaNs

        result = df.copy()
        norm_factor_col = f'{hist_col}_norm_factor'

        # Calculate the rolling maximum of the *absolute* histogram value
        rolling_max_abs_hist = result[hist_col].abs().rolling(window=period, min_periods=1).max()

        # Set the normalization factor. Replace 0 with 1 to avoid division by zero.
        # Use a small epsilon instead of 1? Let's stick with 1 for simplicity.
        result[norm_factor_col] = rolling_max_abs_hist.replace(0, 1)

        # Forward fill initial NaNs that rolling max might produce if min_periods > 1 was used
        # result[norm_factor_col] = result[norm_factor_col].ffill() # Not needed with min_periods=1

        logger.debug(f"Calculated MACD Hist normalization factor over {period} periods.")
        return result