"""
Technical Indicators Module for ZT-3 Trading System.

This module provides implementations of various technical indicators
used by the trading strategies, including the Gaussian Channel indicator.

Note: The GaussianChannelStrategy now implements its own indicators directly
to match PineScript exactly. This module is kept for other strategies or utilities.
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
    def gaussian_channel(df: pd.DataFrame, source_col: str = 'close', period: int = 144, 
                        multiplier: float = 1.2, std_dev_period: int = 144) -> pd.DataFrame:
        """
        Calculate the Gaussian Channel indicator.
        
        Args:
            df: DataFrame with price data
            source_col: Column name to use as source data (default: 'close')
            period: Period for the Gaussian filter
            multiplier: Multiplier for the bands
            std_dev_period: Period for the standard deviation calculation
            
        Returns:
            DataFrame with added columns: gc_filter, gc_upper, gc_lower
        """
        if len(df) < period:
            logger.warning(f"Not enough data for Gaussian Channel calculation. Need at least {period} data points.")
            return df.copy()
            
        # Create a copy of the input DataFrame to avoid modifying it
        result = df.copy()
        
        # Calculate Gaussian filter
        # This is a simplified implementation using a weighted moving average with normal distribution weights
        x = np.arange(-3, 3, 6/period)
        weights = np.exp(-(x**2)/2) / (np.sqrt(2*np.pi))
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Apply Gaussian filter to source data
        source_data = result[source_col].values
        
        # Use convolution for smoothing
        filter_line = np.convolve(source_data, weights, 'valid')
        
        # Pad the beginning to match the original length
        padding_length = len(source_data) - len(filter_line)
        filter_line = np.append(np.full(padding_length, np.nan), filter_line)
        
        result['gc_filter'] = filter_line
        
        # Calculate standard deviation over specified period
        result['rolling_std'] = result[source_col].rolling(window=std_dev_period).std()
        
        # Calculate upper and lower bands
        result['gc_upper'] = result['gc_filter'] + (result['rolling_std'] * multiplier)
        result['gc_lower'] = result['gc_filter'] - (result['rolling_std'] * multiplier)
        
        # Drop the temporary column
        result.drop(columns=['rolling_std'], inplace=True)
        
        return result
        
    @staticmethod
    def stochastic_rsi(df: pd.DataFrame, close_col: str = 'close', 
                      rsi_period: int = 14, stoch_period: int = 14, k_smoothing: int = 3, 
                      d_smoothing: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic RSI indicator.
        
        Args:
            df: DataFrame with price data
            close_col: Column name with close prices
            rsi_period: Period for RSI calculation
            stoch_period: Period for Stochastic calculation
            k_smoothing: Smoothing period for K line
            d_smoothing: Smoothing period for D line
            
        Returns:
            DataFrame with added columns: rsi, stoch_rsi_k, stoch_rsi_d
        """
        if len(df) < max(rsi_period, stoch_period) + k_smoothing + d_smoothing:
            logger.warning("Not enough data for Stochastic RSI calculation")
            return df.copy()
        
        # Create a copy of the input DataFrame to avoid modifying it
        result = df.copy()
        
        # Calculate RSI
        close_delta = result[close_col].diff()
        
        # Make two series: one for gains and one for losses
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
        
        # Calculate EMAs
        up_ema = up.ewm(com=rsi_period-1, adjust=False).mean()
        down_ema = down.ewm(com=rsi_period-1, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = up_ema / down_ema
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic RSI
        min_rsi = result['rsi'].rolling(window=stoch_period).min()
        max_rsi = result['rsi'].rolling(window=stoch_period).max()
        
        # Handle division by zero
        rsi_range = max_rsi - min_rsi
        rsi_range = np.where(rsi_range == 0, 1, rsi_range)  # Replace 0 with 1 to avoid division by zero
        
        # Calculate raw K values
        stoch_rsi_k_raw = 100 * ((result['rsi'] - min_rsi) / rsi_range)
        
        # Apply smoothing to K values
        result['stoch_rsi_k'] = pd.Series(stoch_rsi_k_raw).rolling(window=k_smoothing).mean().values
        
        # Calculate D values (simple moving average of K)
        result['stoch_rsi_d'] = result['stoch_rsi_k'].rolling(window=d_smoothing).mean()
        
        return result
    
    @staticmethod
    def volume_analysis(df: pd.DataFrame, volume_col: str = 'volume', 
                       ma_period: int = 20) -> pd.DataFrame:
        """
        Calculate volume indicators.
        
        Args:
            df: DataFrame with price and volume data
            volume_col: Column name with volume data
            ma_period: Period for the volume moving average
            
        Returns:
            DataFrame with added column: volume_ma
        """
        if len(df) < ma_period:
            logger.warning(f"Not enough data for volume analysis. Need at least {ma_period} data points.")
            return df.copy()
            
        # Create a copy of the input DataFrame to avoid modifying it
        result = df.copy()
        
        # Calculate volume moving average
        result['volume_ma'] = result[volume_col].rolling(window=ma_period).mean()
        
        return result
        
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14, 
           high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).
        
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