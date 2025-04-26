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

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SwingProStrategy.

        Args:
            config: System configuration dictionary (used for general settings, not strategy parameters)
        """
        super().__init__(config)
        self.name = "SwingProStrategy"

        # === Hardcoded Parameters (based on strategy.pine v7) ===
        self.risk_mult_atr_sl: float = 1.5
        self.risk_mult_atr_tp1: float = 3.0
        self.risk_mult_atr_tp2: float = 4.0
        self.trail_ema_len: int = 20
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
        # Position sizing parameters (from PineScript v7)
        self.base_position_percent: float = 0.80 # Base allocation (80%)
        self.scaling_position_percent: float = 0.20 # Scalable part (20%)

        # Internal state for tracking open positions and their details
        self.open_positions: Dict[str, Dict[str, Any]] = {} # symbol -> {entry_price, quantity, initial_quantity, stop_loss, take_profit1, take_profit2}

        logger.info(f"{self.name} initialized with hardcoded parameters.")
        logger.debug(f"Parameters: ATR SL={self.risk_mult_atr_sl}, TP1={self.risk_mult_atr_tp1}, TP2={self.risk_mult_atr_tp2}, TrailEMA={self.trail_ema_len}, UseMTF={self.use_mtf_trend}, DailyEMA={self.daily_ema_len}")

    def reset_position_tracking(self):
        """Resets the tracking of open positions."""
        self.open_positions = {}
        logger.debug(f"{self.name}: Position tracking reset.")

    def update_open_position(self, symbol: str, details: Optional[Dict[str, Any]]):
        """Updates or removes the tracking details for an open position."""
        if details:
            self.open_positions[symbol] = details
            logger.debug(f"Updated open position tracking for {symbol}: {details}")
        elif symbol in self.open_positions:
            del self.open_positions[symbol]
            logger.debug(f"Removed open position tracking for {symbol}.")


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
        # Calculate the minimum required rows based on the longest period used
        min_rows = max(self.ema_200_len, self.macd_slow + self.macd_signal, self.atr_len) # Use macd_slow here
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

    def calculate_entry_quantity_percent(self, candle_data: pd.DataFrame) -> float:
        """
        Calculate the percentage of capital to allocate for an entry,
        based on signal strength (MACD histogram and RSI).

        Args:
            candle_data: DataFrame with indicators up to the current candle.

        Returns:
            Percentage of capital (e.g., 0.8 to 1.0).
        """
        if candle_data.empty or len(candle_data) < 2:
            return self.base_position_percent # Default if not enough data

        last_candle = candle_data.iloc[-1]

        # Calculate signal strength components
        macd_hist_strength = abs(last_candle.get('macd_hist', 0))
        # Normalize RSI strength (0 to 1, higher when further from OB level)
        rsi_strength = max(0.0, self.rsi_ob_level - last_candle.get('rsi', self.rsi_ob_level)) / self.rsi_ob_level

        # Combine strengths (simple addition, capped at 1.0 - adjust logic if needed)
        # This needs refinement to match PineScript's scaling intent.
        # Let's try a simple scaling: assume max reasonable MACD hist is ~ some value, scale it 0-1.
        # This is tricky without knowing typical hist ranges. Let's use a simpler approach for now:
        # Average the normalized RSI strength and a capped/scaled MACD strength.
        # Example: Cap MACD hist contribution to 1.0
        scaled_macd_strength = min(1.0, macd_hist_strength / last_candle.get('atr', 1)) # Scale by ATR? Needs testing.
        # Let's stick closer to PineScript: macd_strength + rsi_strength, then scale result.
        raw_strength = scaled_macd_strength + rsi_strength # Combine (could exceed 1)
        # Scale the combined strength to influence the scaling_position_percent
        # Map raw_strength (e.g., 0 to 2 range?) to a 0-1 multiplier for the scaling part.
        strength_multiplier = min(1.0, raw_strength / 1.5) # Normalize combined strength (heuristic)

        position_percent = self.base_position_percent + (self.scaling_position_percent * strength_multiplier)

        # Ensure it's within reasonable bounds [0, 1]
        position_percent = max(0.0, min(1.0, position_percent))

        logger.debug(f"Calculated position percent: {position_percent:.3f} (RSI Strength: {rsi_strength:.3f}, MACD Strength: {scaled_macd_strength:.3f})")

        return position_percent


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
        # prev = candle_data.iloc[-2] # Not explicitly used in this version's logic, but might be useful

        timestamp = last.name # Timestamp is the index
        current_price = last['close']

        # Check if a position is already open for this symbol
        is_position_open = symbol in self.open_positions
        open_pos_details = self.open_positions.get(symbol)

        # --- Check Exit Conditions First (if position is open) ---
        if is_position_open and open_pos_details:
            entry_price = open_pos_details['entry_price']
            stop_loss_level = open_pos_details['stop_loss']
            take_profit1_level = open_pos_details['take_profit1']
            take_profit2_level = open_pos_details['take_profit2']
            initial_quantity = open_pos_details['initial_quantity']
            current_quantity = open_pos_details['quantity']

            # 1. Stop Loss Hit
            if current_price <= stop_loss_level:
                 if not self.is_duplicate_signal(SignalType.EXIT, symbol, timestamp):
                    logger.info(f"{timestamp} - EXIT Signal (SL): {symbol} @ {current_price:.2f} (SL Level: {stop_loss_level:.2f})")
                    # No need to update self.open_positions here, backtester handles it
                    return Signal(SignalType.EXIT, symbol, current_price, timestamp, exit_reason=ExitReason.STOP_LOSS)
                 else:
                    logger.debug(f"Duplicate EXIT signal (SL) ignored for {symbol} at {timestamp}")


            # 2. Take Profit 2 Hit (Full Exit)
            if take_profit2_level and current_price >= take_profit2_level:
                 if not self.is_duplicate_signal(SignalType.EXIT, symbol, timestamp):
                    logger.info(f"{timestamp} - EXIT Signal (TP2): {symbol} @ {current_price:.2f} (TP2 Level: {take_profit2_level:.2f})")
                    # No need to update self.open_positions here, backtester handles it
                    return Signal(SignalType.EXIT, symbol, current_price, timestamp, exit_reason=ExitReason.TAKE_PROFIT)
                 else:
                    logger.debug(f"Duplicate EXIT signal (TP2) ignored for {symbol} at {timestamp}")


            # 3. Trailing EMA Cross Exit (Full Exit)
            if current_price < last[f'ema_{self.trail_ema_len}']:
                 if not self.is_duplicate_signal(SignalType.EXIT, symbol, timestamp):
                    logger.info(f"{timestamp} - EXIT Signal (EMA Cross): {symbol} @ {current_price:.2f} (EMA: {last[f'ema_{self.trail_ema_len}']:.2f})")
                    # No need to update self.open_positions here, backtester handles it
                    return Signal(SignalType.EXIT, symbol, current_price, timestamp, exit_reason=ExitReason.EMA_CROSS)
                 else:
                    logger.debug(f"Duplicate EXIT signal (EMA Cross) ignored for {symbol} at {timestamp}")


            # 4. Take Profit 1 Hit (Partial Exit - only if not already partially closed)
            # Check if current quantity equals initial quantity (meaning no partial close yet)
            if take_profit1_level and current_price >= take_profit1_level and current_quantity == initial_quantity:
                 if not self.is_duplicate_signal(SignalType.EXIT, symbol, timestamp): # Use EXIT type for tracking partials too
                    logger.info(f"{timestamp} - PARTIAL EXIT Signal (TP1): {symbol} @ {current_price:.2f} (TP1 Level: {take_profit1_level:.2f})")
                    # Backtester needs to handle the partial close logic based on this reason
                    # Signal price is the current price
                    # No need to update self.open_positions here, backtester handles it
                    return Signal(SignalType.EXIT, symbol, current_price, timestamp, exit_reason=ExitReason.PARTIAL_TAKE_PROFIT)
                 else:
                    logger.debug(f"Duplicate PARTIAL EXIT signal (TP1) ignored for {symbol} at {timestamp}")


        # --- Check Entry Conditions (if no position is open) ---
        elif not is_position_open:
            # Conditions from PineScript: macd_positive and rsi_not_overbought and trend_filter
            macd_positive = last['macd_line'] > last['macd_signal']
            rsi_not_overbought = last['rsi'] < self.rsi_ob_level
            trend_filter = last['in_uptrend_mtf'] if self.use_mtf_trend else True

            entry_condition = macd_positive and rsi_not_overbought and trend_filter

            if entry_condition:
                if not self.is_duplicate_signal(SignalType.ENTRY, symbol, timestamp):
                    entry_price = current_price # Use close price for signal
                    atr_value = last['atr']

                    # Calculate SL and TP levels based on entry price and current ATR
                    stop_loss_level = entry_price - (atr_value * self.risk_mult_atr_sl)
                    take_profit1_level = entry_price + (atr_value * self.risk_mult_atr_tp1)
                    take_profit2_level = entry_price + (atr_value * self.risk_mult_atr_tp2)

                    logger.info(f"{timestamp} - ENTRY Signal: {symbol} @ {entry_price:.2f} (ATR: {atr_value:.2f}, SL: {stop_loss_level:.2f}, TP1: {take_profit1_level:.2f}, TP2: {take_profit2_level:.2f})")

                    # Store intended position details temporarily - Backtester will confirm and update
                    # This helps calculate_entry_quantity_percent if needed by backtester before trade confirmation
                    # Note: Backtester should overwrite/confirm this upon actual trade execution
                    self.open_positions[symbol] = {
                        'entry_price': entry_price, # Tentative
                        'quantity': 0, # Tentative
                        'initial_quantity': 0, # Tentative
                        'stop_loss': stop_loss_level,
                        'take_profit1': take_profit1_level,
                        'take_profit2': take_profit2_level
                    }


                    return Signal(
                        signal_type=SignalType.ENTRY,
                        symbol=symbol,
                        price=entry_price,
                        timestamp=timestamp,
                        stop_loss_level=stop_loss_level
                        # TP levels are stored internally for exit checks, not needed in entry signal itself
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

