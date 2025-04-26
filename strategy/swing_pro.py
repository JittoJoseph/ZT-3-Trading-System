"""
SwingPro Strategy Module for ZT-3 Trading System.

This module implements a swing trading strategy based on MACD, RSI,
EMAs, and ATR, inspired by the provided PineScript.
It operates on daily candles.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime

from strategy import SignalType, ExitReason, Signal, Strategy
from strategy.indicators import Indicators # Assuming indicators are in this class

logger = logging.getLogger(__name__)

class SwingProStrategy(Strategy):
    """
    SwingPro trading strategy for daily intervals.

    Based on PineScript: "SwingPro v7 – Adaptive Signal Sizer"

    Entry: Long when MACD is positive, RSI < 70. (MTF filter disabled for now)
    Exit: ATR-based SL/TP (TP1 partial, TP2 full), or EMA(20) crossunder.
    Sizing: Dynamic based on signal strength (MACD hist + RSI).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SwingPro strategy.

        Args:
            config: System configuration dictionary
        """
        super().__init__(config)
        self.name = "SwingProStrategy"

        # Strategy Parameters (from PineScript inputs)
        self.risk_mult_atr_sl = 1.5
        self.risk_mult_atr_tp1 = 3.0
        self.risk_mult_atr_tp2 = 4.0
        self.trail_ema_len = 20
        self.ema_50_len = 50
        self.ema_200_len = 200
        self.rsi_len = 14
        self.rsi_ob_level = 70.0
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.atr_len = 14
        self.use_mtf_trend = False # Disabled for now
        self.daily_ema_len = 50 # Relevant if use_mtf_trend is True

        # Position tracking (stores entry price, quantity, TP/SL levels)
        # Structure: {symbol: {'entry_price': float, 'quantity': int, 'initial_quantity': int,
        #                      'stop_loss': float, 'take_profit1': float, 'take_profit2': float}}
        self.open_positions = {}

        logger.info(f"Initialized {self.name} Strategy")

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by calculating all necessary indicators.

        Args:
            df: DataFrame with raw OHLCV data (expected daily)

        Returns:
            DataFrame with all indicators added
        """
        if df.empty:
            return df

        # Calculate indicators using the Indicators class or direct pandas ops
        df = Indicators.ema(df, period=self.trail_ema_len)
        df = Indicators.ema(df, period=self.ema_50_len)
        df = Indicators.ema(df, period=self.ema_200_len)
        df = Indicators.rsi(df, rsi_period=self.rsi_len)
        df = Indicators.macd(df, fast_period=self.macd_fast, slow_period=self.macd_slow, signal_period=self.macd_signal)
        df = Indicators.atr(df, period=self.atr_len)

        # Add MTF Daily EMA if enabled (requires more complex data handling)
        if self.use_mtf_trend:
            # This part needs implementation if MTF is required.
            # It would involve requesting daily data if the primary timeframe isn't daily,
            # or ensuring the input df is daily.
            # For now, we assume df is daily and calculate EMA on it.
            df = Indicators.ema(df, period=self.daily_ema_len, source_col='close')
            df.rename(columns={f'ema_{self.daily_ema_len}': 'daily_ema'}, inplace=True)
            df['in_uptrend_mtf'] = df['close'] > df['daily_ema']
        else:
            df['in_uptrend_mtf'] = True # Always true if filter is disabled

        # Calculate signal strength components
        df['macd_positive'] = df['macd_line'] > df['macd_signal']
        df['rsi_not_overbought'] = df['rsi'] < self.rsi_ob_level

        # Calculate signal strength (0 to 1 range approx)
        # Normalize MACD hist? PineScript uses raw value. Let's try normalizing.
        # Normalize RSI strength (0 to 1, higher when further below 70)
        macd_hist_abs_norm = df['macd_hist'].abs() / df['macd_hist'].abs().rolling(window=50, min_periods=10).max().fillna(1) # Normalize over rolling window
        rsi_strength = (self.rsi_ob_level - df['rsi'].clip(upper=self.rsi_ob_level)) / self.rsi_ob_level
        df['signal_strength'] = (macd_hist_abs_norm.fillna(0) + rsi_strength.fillna(0)) / 2 # Average normalized strengths
        df['signal_strength'] = df['signal_strength'].clip(0, 1) # Ensure 0-1 range

        # Calculate position size multiplier (0.8 to 1.0)
        df['position_percent_mult'] = 0.8 + (df['signal_strength'] * 0.2)

        return df

    def calculate_entry_quantity_percent(self, candle_data: pd.DataFrame) -> float:
        """
        Calculate the percentage of equity to use for an entry based on signal strength.

        Args:
            candle_data: DataFrame ending with the current candle.

        Returns:
            Percentage of equity (e.g., 0.95 for 95%).
        """
        if candle_data.empty or 'position_percent_mult' not in candle_data.columns:
            return 0.95 # Default if calculation fails

        # Use the multiplier from the latest candle data
        multiplier = candle_data['position_percent_mult'].iloc[-1]

        # Ensure multiplier is within expected bounds
        return max(0.8, min(1.0, multiplier if pd.notna(multiplier) else 0.95))

    def calculate_stop_loss(self, entry_price: float, candle_data: pd.DataFrame) -> float:
        """Calculate stop loss level."""
        if candle_data.empty or 'atr' not in candle_data.columns:
            return entry_price * (1 - 0.05) # Default 5% SL if ATR fails
        atr_value = candle_data['atr'].iloc[-1]
        return entry_price - (atr_value * self.risk_mult_atr_sl)

    def calculate_take_profit_levels(self, entry_price: float, candle_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate take profit levels."""
        if candle_data.empty or 'atr' not in candle_data.columns:
            tp1 = entry_price * (1 + 0.10) # Default 10% TP1
            tp2 = entry_price * (1 + 0.15) # Default 15% TP2
            return tp1, tp2
        atr_value = candle_data['atr'].iloc[-1]
        tp1 = entry_price + (atr_value * self.risk_mult_atr_tp1)
        tp2 = entry_price + (atr_value * self.risk_mult_atr_tp2)
        return tp1, tp2

    def process_candle(self, candle_data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Process new candle data and generate signals if conditions are met.
        Operates on daily data.

        Args:
            candle_data: DataFrame containing candle data with indicators, ending with the current candle.
            symbol: Trading symbol

        Returns:
            Signal if conditions are met, None otherwise
        """
        min_required_data = max(self.ema_200_len, self.macd_slow + self.macd_signal) + 5
        if len(candle_data) < min_required_data:
            # logger.debug(f"Not enough data for {symbol} ({len(candle_data)} < {min_required_data})")
            return None # Not enough data

        # Ensure data is prepared (indicators calculated)
        # Assuming prepare_data was called beforehand or is implicitly handled by the caller (backtester)
        # For safety, call it, but this might be redundant if backtester already does it.
        # data = self.prepare_data(candle_data) # Potentially redundant
        data = candle_data # Assume data is prepared

        # Check for NaN values in critical indicators for the latest candle
        current = data.iloc[-1]
        required_cols = ['close', 'high', 'low', f'ema_{self.trail_ema_len}', 'macd_positive', 'rsi_not_overbought', 'in_uptrend_mtf', 'atr']
        if current[required_cols].isnull().any():
            # logger.debug(f"NaN values found in indicators for {symbol} at {current.name}")
            return None # Wait for indicators to be calculated

        timestamp = current.name

        # --- Check Exit Conditions First ---
        if symbol in self.open_positions:
            position = self.open_positions[symbol]
            exit_signal = None
            exit_price = current['close'] # Default exit price

            # 1. Stop Loss
            if current['low'] <= position['stop_loss']:
                exit_price = position['stop_loss'] # Exit at SL level
                exit_signal = Signal(SignalType.EXIT, symbol, exit_price, timestamp, exit_reason=ExitReason.STOP_LOSS)
                logger.debug(f"{timestamp} - {symbol}: Stop Loss triggered at {exit_price:.2f}")

            # 2. Take Profit 2 (Full Exit)
            elif current['high'] >= position['take_profit2']:
                exit_price = position['take_profit2'] # Exit at TP2 level
                exit_signal = Signal(SignalType.EXIT, symbol, exit_price, timestamp, exit_reason=ExitReason.TAKE_PROFIT)
                logger.debug(f"{timestamp} - {symbol}: Take Profit 2 triggered at {exit_price:.2f}")

            # 3. Take Profit 1 (Partial Exit - 50%)
            elif current['high'] >= position['take_profit1'] and position['quantity'] == position['initial_quantity']: # Only trigger TP1 once
                exit_price = position['take_profit1'] # Exit at TP1 level
                # Create a partial exit signal - backtester needs to handle this
                exit_signal = Signal(SignalType.EXIT, symbol, exit_price, timestamp, exit_reason=ExitReason.PARTIAL_TAKE_PROFIT)
                logger.debug(f"{timestamp} - {symbol}: Partial Take Profit 1 triggered at {exit_price:.2f}")
                # Note: We don't remove the position here, backtester handles partial close

            # 4. EMA Trail Exit
            elif current['close'] < current[f'ema_{self.trail_ema_len}']:
                exit_signal = Signal(SignalType.EXIT, symbol, exit_price, timestamp, exit_reason=ExitReason.EMA_CROSS)
                logger.debug(f"{timestamp} - {symbol}: EMA Trail Exit triggered at {exit_price:.2f}")

            # If a full exit signal was generated, remove the position
            if exit_signal and exit_signal.exit_reason != ExitReason.PARTIAL_TAKE_PROFIT:
                self.open_positions.pop(symbol, None)

            if exit_signal:
                 # Avoid duplicate exit signals for the same candle
                if not self.is_duplicate_signal(SignalType.EXIT, symbol, timestamp):
                    return exit_signal
                else:
                    logger.debug(f"Duplicate EXIT signal ignored for {symbol} at {timestamp}")
                    return None


        # --- Check Entry Conditions ---
        elif symbol not in self.open_positions: # Only enter if no position is open
            # Entry Conditions from PineScript
            entry_condition = current['macd_positive'] and \
                              current['rsi_not_overbought'] and \
                              current['in_uptrend_mtf'] # MTF filter applied here

            if entry_condition:
                 # Avoid duplicate entry signals for the same candle
                if not self.is_duplicate_signal(SignalType.ENTRY, symbol, timestamp):
                    entry_price = current['close']
                    # Calculate SL/TP levels based on current ATR
                    stop_loss_level = self.calculate_stop_loss(entry_price, data)
                    take_profit1_level, take_profit2_level = self.calculate_take_profit_levels(entry_price, data)

                    entry_signal = Signal(
                        signal_type=SignalType.ENTRY,
                        symbol=symbol,
                        price=entry_price,
                        timestamp=timestamp,
                        stop_loss_level=stop_loss_level,
                        # Store TP levels in the signal? Or just in open_positions? Let's store here for consistency.
                        # Backtester will use these to populate the Trade object.
                        # Using custom fields or a dict might be better if Signal class is fixed.
                        # For now, let's assume backtester looks up TP levels when creating Trade.
                    )

                    # Store position details - Backtester will use this signal to create the Trade object
                    # and then populate self.open_positions
                    # We pre-calculate levels here for clarity, backtester confirms/uses them.
                    self.open_positions[symbol] = {
                         # 'entry_price': entry_price, # Set by backtester
                         # 'quantity': 0, # Set by backtester
                         # 'initial_quantity': 0, # Set by backtester
                         'stop_loss': stop_loss_level,
                         'take_profit1': take_profit1_level,
                         'take_profit2': take_profit2_level
                    }
                    logger.debug(f"{timestamp} - {symbol}: Entry signal generated at {entry_price:.2f}")
                    return entry_signal
                else:
                    logger.debug(f"Duplicate ENTRY signal ignored for {symbol} at {timestamp}")


        return None # No signal generated

    def update_open_position(self, symbol: str, trade_info: Dict[str, Any]):
        """
        Called by the backtester after a trade is opened or partially closed.
        Updates the internal tracking.

        Args:
            symbol: The symbol traded.
            trade_info: Dictionary containing current trade state
                        (e.g., entry_price, quantity, initial_quantity, sl, tp1, tp2).
                        If trade_info is None, it means the position was fully closed.
        """
        if trade_info:
            self.open_positions[symbol] = trade_info
            logger.debug(f"Updated open position for {symbol}: Qty={trade_info.get('quantity')}")
        else:
            if symbol in self.open_positions:
                del self.open_positions[symbol]
                logger.debug(f"Removed closed position for {symbol}")

    def reset_position_tracking(self):
        """Reset internal position tracking."""
        self.open_positions = {}
        logger.info("Reset internal position tracking for SwingProStrategy.")

