"""
SwingPro Strategy Implementation for ZT-3 Trading System.

Based on the provided PineScript logic.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from . import Strategy, Signal, SignalType, ExitReason
from .indicators import Indicators

logger = logging.getLogger(__name__)

class SwingProStrategy(Strategy):
    """
    SwingPro Strategy based on EMA, RSI, MACD, and ATR.

    Uses 1-hour timeframe data and an optional daily EMA filter.
    """

    def __init__(self, config: Dict[str, Any], max_concurrent_positions: int):
        """
        Initialize the SwingProStrategy.

        Args:
            config: System configuration dictionary (used for general settings)
            max_concurrent_positions: Max concurrent positions allowed by the backtester/runner.
        """
        super().__init__(config) # Pass config to base Strategy class
        self.name = "SwingProStrategy"
        self.max_concurrent_positions = max(1, max_concurrent_positions) # Ensure at least 1

        # === Hardcoded Parameters (based on strategy.pine v7) ===
        self.risk_mult_atr_sl: float = 1.5
        self.risk_mult_atr_tp1: float = 3.0
        self.risk_mult_atr_tp2: float = 4.0
        self.trail_ema_len: int = 20 # Re-introduce EMA cross exit
        self.ema_50_len: int = 50
        self.ema_200_len: int = 200
        self.rsi_len: int = 14
        self.rsi_ob_level: int = 70 # Overbought level for entry condition
        self.macd_fast: int = 12
        self.macd_slow: int = 26
        self.macd_signal: int = 9
        self.atr_len: int = 14
        self.use_mtf_trend: bool = True
        self.daily_ema_len: int = 50
        # === Volume Filter Parameters ===
        self.volume_filter_enabled: bool = True
        self.volume_sma_len: int = 20

        # Calculate fixed allocation percentage (base allocation per slot)
        self.allocation_buffer_factor = 0.95 # Use 95% of the calculated slot size
        self.base_slot_allocation_percent = (1.0 / self.max_concurrent_positions) * self.allocation_buffer_factor

        # Internal state for tracking open positions and their details
        # Reverted: Removed trailing_stop_level
        self.open_positions: Dict[str, Dict[str, Any]] = {} # symbol -> {entry_price, quantity, initial_quantity}

        logger.info(f"{self.name} initialized with hardcoded parameters.")
        logger.info(f"Max Concurrent Positions: {self.max_concurrent_positions}, Fixed Allocation per Slot: {self.base_slot_allocation_percent:.2%}") # Updated log
        logger.debug(f"Parameters: ATR SL={self.risk_mult_atr_sl}, TP1={self.risk_mult_atr_tp1}, TP2={self.risk_mult_atr_tp2}, TrailEMA={self.trail_ema_len}, UseMTF={self.use_mtf_trend}, DailyEMA={self.daily_ema_len}")
        logger.debug(f"Volume Filter Enabled: {self.volume_filter_enabled}, SMA Period: {self.volume_sma_len}")

    def reset_position_tracking(self):
        """Resets the tracking of open positions."""
        self.open_positions = {}
        logger.debug(f"{self.name}: Position tracking reset.")

    def update_open_position(self, symbol: str, details: Optional[Dict[str, Any]]):
        """Updates or removes the tracking details for an open position."""
        if details:
            # Reverted: Removed trailing_stop_level check/handling
            if symbol in self.open_positions:
                 self.open_positions[symbol].update(details)
                 logger.debug(f"Updated open position tracking for {symbol}: {self.open_positions[symbol]}")
            else:
                 self.open_positions[symbol] = details
                 logger.debug(f"Confirmed and added position tracking for {symbol}: {details}")

        elif symbol in self.open_positions:
            # If details are None (full exit or failed entry), remove
            del self.open_positions[symbol]
            logger.debug(f"Removed open position tracking for {symbol}. Count: {len(self.open_positions)}")


    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by calculating all necessary indicators for SwingPro.

        Args:
            df: DataFrame with raw OHLCV data (assumed 1-hour)

        Returns:
            DataFrame with all indicators added
        """
        if df.empty:
            return df

        logger.debug(f"Preparing data for {self.name} - Initial rows: {len(df)}")

        # Calculate indicators using hardcoded periods
        df = Indicators.ema(df, period=self.trail_ema_len) # ema_20
        df = Indicators.ema(df, period=self.ema_50_len)    # ema_50
        df = Indicators.ema(df, period=self.ema_200_len)   # ema_200
        df = Indicators.rsi(df, period=self.rsi_len)       # rsi_14
        df = Indicators.macd(df, fast_period=self.macd_fast, slow_period=self.macd_slow, signal_period=self.macd_signal)
        df = Indicators.atr(df, period=self.atr_len)       # atr_14

        # Calculate Volume SMA if filter enabled
        if self.volume_filter_enabled:
            df = Indicators.sma(df, source_col='volume', period=self.volume_sma_len)

        # Calculate Daily EMA if needed (requires resampling)
        if self.use_mtf_trend:
            try:
                # Resample to daily, calculate EMA, then reindex back to original hourly index
                daily_close = df['close'].resample('D').last()
                daily_ema = daily_close.ewm(span=self.daily_ema_len, adjust=False).mean()
                # Forward fill the daily EMA onto the hourly index
                df[f'daily_ema_{self.daily_ema_len}'] = daily_ema.reindex(df.index, method='ffill')
                # Calculate the MTF trend condition
                df['in_uptrend_mtf'] = df['close'] > df[f'daily_ema_{self.daily_ema_len}']
            except Exception as e:
                logger.error(f"Error calculating daily EMA trend: {e}. Disabling MTF filter for this run.")
                df['in_uptrend_mtf'] = True # Default to true if calculation fails

        # Drop rows with NaN values created by indicator calculations
        # Adjust min_rows for volume SMA
        min_rows = max(self.ema_200_len, self.macd_slow + self.macd_signal, self.atr_len)
        if self.volume_filter_enabled:
            min_rows = max(min_rows, self.volume_sma_len) # Add volume SMA period
        if self.use_mtf_trend:
             # Daily EMA needs more data points initially due to resampling
             min_rows = max(min_rows, self.daily_ema_len * 2) # Heuristic, needs enough days

        initial_len = len(df)
        df.dropna(inplace=True)
        final_len = len(df)
        logger.debug(f"Data preparation complete. Dropped {initial_len - final_len} rows due to NaNs. Final rows: {final_len}")

        if final_len < 1:
             logger.warning("Not enough data remains after indicator calculation and NaN removal.")


        return df

    def calculate_entry_quantity_percent(self) -> float: # Removed symbol parameter
        """
        Returns a fixed percentage of capital to allocate per trade,
        based on the max_concurrent_positions setting.

        Returns:
            Fixed percentage of capital (e.g., 0.475 for max_concurrent=2).
        """
        # Return the pre-calculated fixed percentage
        logger.debug(f"Returning fixed allocation percent: {self.base_slot_allocation_percent:.3%}")
        return self.base_slot_allocation_percent


    def process_candle(self, candle_data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Process the latest candle data for the SwingPro strategy.

        Args:
            candle_data: DataFrame with OHLCV and calculated indicators.
                         Assumes the last row is the current candle to process.
            symbol: Trading symbol (e.g., 'NSE:PNB')

        Returns:
            Signal if conditions are met, None otherwise.
        """
        if candle_data.empty or len(candle_data) < 2:
            # Need at least 2 rows for some checks (like previous candle state)
            return None

        # Get the latest and previous candle data
        last = candle_data.iloc[-1]
        # prev = candle_data.iloc[-2] # No longer needed for EMA cross

        timestamp = last.name # Timestamp is the index
        current_price = last['close']
        current_atr = last['atr'] # Get ATR for the current candle

        # Check if a position is already open for this symbol
        is_position_open = symbol in self.open_positions
        open_pos_details = self.open_positions.get(symbol)

        # --- Check Exit Conditions First (if position is open) ---
        if is_position_open and open_pos_details:
            entry_price = open_pos_details['entry_price']
            initial_quantity = open_pos_details.get('initial_quantity', 0) # Get initial quantity safely
            current_quantity = open_pos_details.get('quantity', 0) # Get current quantity safely

            # Ensure initial_quantity is valid before using it for partial close calculation
            if initial_quantity <= 0:
                 logger.warning(f"Cannot process exits for {symbol} at {timestamp}: Initial quantity is {initial_quantity}.")
                 return None # Cannot determine partial close quantity

            # --- Recalculate TP and Fixed SL levels based on current ATR ---
            dynamic_take_profit1_level = entry_price + (current_atr * self.risk_mult_atr_tp1)
            dynamic_take_profit2_level = entry_price + (current_atr * self.risk_mult_atr_tp2)
            dynamic_stop_loss_level = entry_price - (current_atr * self.risk_mult_atr_sl) # Fixed SL

            # --- Update ATR Trailing Stop Level (REMOVED) ---
            # ... (ATR trail stop logic removed) ...

            # --- Check Exit Conditions (Reverted Order & Logic) ---

            # 1. Stop Loss Hit (Uses the dynamic fixed SL, check current_price)
            if current_price <= dynamic_stop_loss_level:
                 if not self.is_duplicate_signal(SignalType.EXIT, symbol, timestamp):
                    logger.info(f"{timestamp} - EXIT Signal (SL): {symbol} @ {current_price:.2f} (Fixed SL Level: {dynamic_stop_loss_level:.2f}, ATR: {current_atr:.2f})") # Updated log
                    return Signal(SignalType.EXIT, symbol, current_price, timestamp, exit_reason=ExitReason.STOP_LOSS)
                 else:
                    logger.debug(f"Duplicate EXIT signal (SL) ignored for {symbol} at {timestamp}")
                    return None

            # 1b. ATR Trailing Stop Hit (REMOVED)

            # 2. Take Profit 2 Hit (Uses dynamic TP2 level, check current_price)
            if dynamic_take_profit2_level and current_price >= dynamic_take_profit2_level:
                 if not self.is_duplicate_signal(SignalType.EXIT, symbol, timestamp):
                    logger.info(f"{timestamp} - EXIT Signal (TP2): {symbol} @ {current_price:.2f} (Dynamic TP2 Level: {dynamic_take_profit2_level:.2f}, ATR: {current_atr:.2f})") # Updated log
                    return Signal(SignalType.EXIT, symbol, current_price, timestamp, exit_reason=ExitReason.TAKE_PROFIT)
                 else:
                    logger.debug(f"Duplicate EXIT signal (TP2) ignored for {symbol} at {timestamp}")
                    return None

            # 3. Trailing EMA Cross Exit (Reverted from MACD Cross)
            if current_price < last[f'ema_{self.trail_ema_len}']:
                 if not self.is_duplicate_signal(SignalType.EXIT, symbol, timestamp):
                    logger.info(f"{timestamp} - EXIT Signal (EMA Cross): {symbol} @ {current_price:.2f} (EMA: {last[f'ema_{self.trail_ema_len}']:.2f})")
                    # Use ExitReason.EMA_CROSS
                    return Signal(SignalType.EXIT, symbol, current_price, timestamp, exit_reason=ExitReason.EMA_CROSS)
                 else:
                    logger.debug(f"Duplicate EXIT signal (EMA Cross) ignored for {symbol} at {timestamp}")
                    return None # Avoid processing TP1 if EMA cross duplicate

            # 4. Take Profit 1 Hit (Partial Exit - uses dynamic TP1 level, check current_price, only if not already partially closed)
            if dynamic_take_profit1_level and current_price >= dynamic_take_profit1_level and current_quantity == initial_quantity:
                 if not self.is_duplicate_signal(SignalType.EXIT, symbol, timestamp):
                    # Change partial close quantity from 60% to 50%
                    partial_close_qty = int(initial_quantity * 0.5) # Close 50%
                    if partial_close_qty > 0:
                        # Update log message to reflect 50%
                        logger.info(f"{timestamp} - PARTIAL EXIT Signal (TP1 - 50%): {symbol} @ {current_price:.2f} (Dynamic TP1 Level: {dynamic_take_profit1_level:.2f}, ATR: {current_atr:.2f}), Close Qty: {partial_close_qty}")
                        return Signal(
                            signal_type=SignalType.EXIT,
                            symbol=symbol,
                            price=current_price, # Use current price for exit
                            timestamp=timestamp,
                            exit_reason=ExitReason.PARTIAL_TAKE_PROFIT,
                            close_quantity=partial_close_qty
                        )
                    else: # Handle case where 50% rounds down to 0
                         logger.warning(f"{timestamp} - PARTIAL EXIT Signal (TP1) for {symbol} resulted in zero quantity ({initial_quantity} * 0.5). Skipping partial exit.")

                 else:
                    logger.debug(f"Duplicate PARTIAL EXIT signal (TP1) ignored for {symbol} at {timestamp}")


        # --- Check Entry Conditions (if no position is open) ---
        elif not is_position_open:
            # *** Check Position Limit FIRST ***
            if len(self.open_positions) >= self.max_concurrent_positions:
                logger.debug(f"{timestamp} - ENTRY Condition Met for {symbol}, but SKIPPED (Max concurrent positions limit reached: {self.max_concurrent_positions})")
                return None # Do not generate signal if limit is reached

            # Conditions from PineScript: macd_positive and rsi_not_overbought and trend_filter
            macd_positive = last['macd_line'] > last['macd_signal']
            rsi_not_overbought = last['rsi'] < self.rsi_ob_level
            trend_filter = last['in_uptrend_mtf'] if self.use_mtf_trend else True

            # Volume Confirmation
            volume_confirmation = True # Default to true if filter disabled or data missing
            if self.volume_filter_enabled:
                vol_sma_col = f'sma_volume_{self.volume_sma_len}'
                if vol_sma_col in last and not pd.isna(last[vol_sma_col]):
                    volume_confirmation = last['volume'] > last[vol_sma_col]
                else:
                    volume_confirmation = False # Fail confirmation if SMA is NaN
                    logger.debug(f"{timestamp} - {symbol}: Volume confirmation skipped (SMA Volume NaN)")

            entry_condition = macd_positive and rsi_not_overbought and trend_filter and volume_confirmation # Added volume confirmation

            if entry_condition:
                if not self.is_duplicate_signal(SignalType.ENTRY, symbol, timestamp):
                    entry_price = current_price # Use close price for signal
                    atr_value = last['atr'] # Use ATR of the signal bar for initial reference SL/TP calculation
                    rsi_at_signal = last['rsi'] # Get RSI value
                    volume_at_signal = last['volume']
                    volume_sma_at_signal = last.get(f'sma_volume_{self.volume_sma_len}', np.nan)

                    # Calculate initial reference SL and TP levels based on entry price and signal bar's ATR
                    initial_stop_loss_level = entry_price - (atr_value * self.risk_mult_atr_sl)
                    initial_take_profit1_level = entry_price + (atr_value * self.risk_mult_atr_tp1)
                    initial_take_profit2_level = entry_price + (atr_value * self.risk_mult_atr_tp2)
                    # initial_trail_stop = entry_price - (atr_value * self.atr_trail_mult) # REMOVED

                    logger.info(f"{timestamp} - ENTRY Signal: {symbol} @ {entry_price:.2f} (RSI: {rsi_at_signal:.2f}, Vol: {volume_at_signal:.0f}/{volume_sma_at_signal:.0f}, ATR: {atr_value:.2f}, SL: {initial_stop_loss_level:.2f}, TP1: {initial_take_profit1_level:.2f}, TP2: {initial_take_profit2_level:.2f})") # Updated log

                    # Store tentative details (Reverted: removed trail stop)
                    self.open_positions[symbol] = {
                        'entry_price': entry_price,
                        'quantity': 0,
                        'initial_quantity': 0,
                    }
                    logger.debug(f"Tentatively added {symbol} to open_positions. Count: {len(self.open_positions)}")

                    return Signal(
                        signal_type=SignalType.ENTRY,
                        symbol=symbol,
                        price=entry_price,
                        timestamp=timestamp,
                        stop_loss_level=initial_stop_loss_level,
                        rsi_value=rsi_at_signal,
                        # Removed initial_trail_stop
                    )
                else:
                    logger.debug(f"Duplicate ENTRY signal ignored for {symbol} at {timestamp}")

        # No signal generated
        return None

    # calculate_take_profit and calculate_stop_loss might not be needed if
    # levels are determined at entry and stored, as done above.
    # Keep stubs if the base class requires them.

    def calculate_take_profit(self, entry_price: float, candle_data: pd.DataFrame) -> float:
        """
        Calculate take profit level (Not directly used if TP levels are set at entry).
        """
        # This logic is now handled within process_candle at entry time.
        # Return a placeholder or raise NotImplementedError if required by base class structure.
        # For SwingPro, TP is ATR-based from entry. Let's return TP2 as an example.
        if candle_data.empty: return entry_price * 1.1 # Placeholder
        atr_value = candle_data['atr'].iloc[-1]
        return entry_price + (atr_value * self.risk_mult_atr_tp2)

    def calculate_stop_loss(self, entry_price: float, candle_data: pd.DataFrame) -> float:
        """
        Calculate stop loss level (Not directly used if SL level is set at entry).
        """
        # This logic is now handled within process_candle at entry time.
        if candle_data.empty: return entry_price * 0.9 # Placeholder
        atr_value = candle_data['atr'].iloc[-1]
        return entry_price - (atr_value * self.risk_mult_atr_sl)

