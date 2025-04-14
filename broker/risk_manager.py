"""
Risk Management Module for ZT-3 Trading System.

This module handles risk management including:
- Daily loss limits
- Maximum trades per day
- Circuit breakers
- End of day position closure
"""
import logging
import time
import threading
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Any, Optional, Set

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Manages trading risk limits and safety mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any], position_manager=None):
        """
        Initialize the risk manager with configuration.
        
        Args:
            config: Configuration dictionary
            position_manager: Optional position manager instance
        """
        self.config = config
        self.position_manager = position_manager
        self.risk_config = config.get('risk', {})
        
        # Extract risk parameters
        self.max_daily_loss = float(self.risk_config.get('max_daily_loss', 5.0))
        self.max_daily_trades = int(self.risk_config.get('max_daily_trades', 5))
        self.circuit_breaker_enabled = bool(self.risk_config.get('circuit_breaker_enabled', True))
        self.end_of_day_closure = bool(self.risk_config.get('end_of_day_closure', True))
        self.max_position_size = int(self.risk_config.get('max_position_size', 0))
        
        # Starting capital
        self.starting_capital = float(config.get('paper_trading', {}).get('starting_capital', 3000.0))
        
        # Circuit breaker parameters
        self.circuit_breaker_activated = False
        self.rapid_loss_threshold_percent = float(self.risk_config.get('rapid_loss_threshold', 3.0))
        
        # Risk tracking
        self.daily_losses = 0.0
        self.daily_trades_count = 0
        
        # EOD handling
        self.eod_time_str = self.risk_config.get('end_of_day_time', '15:15')
        self.eod_warning_timer = None
        self.eod_closure_timer = None
        
        # Set up EOD timer
        if self.end_of_day_closure:
            self._setup_eod_timer()
        
        logger.info(f"Risk Manager initialized with max daily loss: {self.max_daily_loss}%, "
                   f"max daily trades: {self.max_daily_trades}")
    
    def check_position_size(self, symbol: str, quantity: int, price: float) -> bool:
        """
        Check if position size is within limits.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares to trade
            price: Price per share
            
        Returns:
            True if position size is acceptable, False otherwise
        """
        # If max position size is not set, allow any size
        if self.max_position_size <= 0:
            return True
            
        # Check if quantity exceeds max position size
        if quantity > self.max_position_size:
            logger.warning(f"Position size check failed for {symbol}: {quantity} > {self.max_position_size}")
            return False
            
        return True
    
    def check_capital_percent(self, price: float, quantity: int, available_capital: float) -> bool:
        """
        Check if trade uses acceptable percentage of capital.
        
        Args:
            price: Price per share
            quantity: Number of shares to trade
            available_capital: Available trading capital
            
        Returns:
            True if capital percentage is acceptable, False otherwise
        """
        # Get max capital percentage from config
        max_capital_percent = float(self.risk_config.get('capital_percent', 95.0))
        
        # Calculate trade value as percentage of capital
        trade_value = price * quantity
        capital_percent = (trade_value / available_capital) * 100
        
        if capital_percent > max_capital_percent:
            logger.warning(f"Capital percent check failed: {capital_percent:.1f}% > {max_capital_percent:.1f}%")
            return False
            
        return True
    
    def can_place_order(self, symbol: str, quantity: int, price: float, available_capital: float) -> bool:
        """
        Check if an order can be placed according to risk rules.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares to trade
            price: Price per share
            available_capital: Available capital for trading
            
        Returns:
            True if order can be placed, False otherwise
        """
        # Check if circuit breaker is activated
        if self.circuit_breaker_activated:
            logger.warning("Order rejected: Circuit breaker is active")
            return False
        
        # Check if market is open
        if not self._is_market_open():
            logger.warning("Order rejected: Market is closed")
            return False
        
        # Check daily trade count
        if self.daily_trades_count >= self.max_daily_trades:
            logger.warning(f"Order rejected: Maximum daily trades reached ({self.max_daily_trades})")
            return False
            
        # Check daily loss limit
        daily_loss_percent = (self.daily_losses / self.starting_capital) * 100
        if daily_loss_percent < -self.max_daily_loss:
            logger.warning(f"Order rejected: Daily loss limit reached ({daily_loss_percent:.1f}% < -{self.max_daily_loss}%)")
            return False
            
        # Check position size
        if not self.check_position_size(symbol, quantity, price):
            logger.warning(f"Order rejected: Position size check failed")
            return False
            
        # Check capital percentage
        if not self.check_capital_percent(price, quantity, available_capital):
            logger.warning(f"Order rejected: Capital percentage check failed")
            return False
            
        # All checks passed
        return True
    
    def register_loss(self, amount: float) -> None:
        """
        Register a trading loss and check if circuit breaker should be activated.
        
        Args:
            amount: Loss amount (negative value)
        """
        if amount >= 0:
            return  # Not a loss
            
        # Update daily losses
        self.daily_losses += amount
        
        # Calculate loss as percentage of capital
        loss_percent = (amount / self.starting_capital) * 100
        
        # Check if we should activate circuit breaker
        if self.circuit_breaker_enabled and loss_percent < -self.rapid_loss_threshold_percent:
            logger.warning(f"Rapid loss detected: {loss_percent:.1f}% (threshold: {self.rapid_loss_threshold_percent}%)")
            self._activate_circuit_breaker()
    
    def _activate_circuit_breaker(self) -> None:
        """Activate the circuit breaker to pause trading after rapid losses."""
        if self.circuit_breaker_activated:
            return  # Already activated
            
        self.circuit_breaker_activated = True
        logger.warning("CIRCUIT BREAKER ACTIVATED - Trading paused due to rapid losses")
        
        # Close all positions if position manager is available
        if self.position_manager:
            # Get all current prices - in a real implementation, you'd get current market prices
            current_prices = {}
            open_positions = self.position_manager.get_open_positions()
            
            for position in open_positions:
                current_prices[position.symbol] = position.last_price
                
            if current_prices:
                self.position_manager.close_all_positions(current_prices, reason="CIRCUIT_BREAKER")
                logger.info("Closed all positions due to circuit breaker activation")
    
    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker to allow trading again."""
        if not self.circuit_breaker_activated:
            return  # Not activated
            
        self.circuit_breaker_activated = False
        logger.info("Circuit breaker reset - Trading resumed")
    
    def _is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            True if market is open, False otherwise
        """
        now = datetime.now()
        
        # Check if it's a weekday (0=Monday, 6=Sunday)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
            
        # Check if time is between 9:15 AM and 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _setup_eod_timer(self) -> None:
        """Set up timers for end-of-day warnings and closure."""
        # Parse EOD time
        try:
            hour, minute = map(int, self.eod_time_str.split(':'))
            eod_time = dt_time(hour, minute, 0)
        except ValueError:
            logger.error(f"Invalid EOD time format: {self.eod_time_str}, using default 15:15")
            eod_time = dt_time(15, 15, 0)
        
        # Calculate warning time (5 minutes before EOD)
        warning_hour, warning_minute = hour, minute - 5
        if warning_minute < 0:
            warning_hour -= 1
            warning_minute += 60
        warning_time = dt_time(warning_hour, warning_minute, 0)
        
        # Calculate seconds until warning and EOD
        now = datetime.now()
        today = now.date()
        
        warning_datetime = datetime.combine(today, warning_time)
        eod_datetime = datetime.combine(today, eod_time)
        
        # If times are already passed for today, schedule for tomorrow
        if now > warning_datetime:
            warning_datetime += timedelta(days=1)
        if now > eod_datetime:
            eod_datetime += timedelta(days=1)
        
        # Calculate seconds until warning and EOD
        seconds_until_warning = (warning_datetime - now).total_seconds()
        seconds_until_eod = (eod_datetime - now).total_seconds()
        
        # Set up timers
        self.eod_warning_timer = threading.Timer(seconds_until_warning, self._market_close_warning)
        self.eod_closure_timer = threading.Timer(seconds_until_eod, self._end_of_day_closure)
        
        # Start timers
        self.eod_warning_timer.daemon = True
        self.eod_closure_timer.daemon = True
        self.eod_warning_timer.start()
        self.eod_closure_timer.start()
        
        logger.info(f"EOD timers set up - Warning at {warning_time}, Closure at {eod_time}")
    
    def _market_close_warning(self) -> None:
        """Send a warning about imminent market closure."""
        logger.warning("Market closing in 5 minutes - preparing for end-of-day closure")
        
        # TODO: Send notification to subscribers
    
    def _end_of_day_closure(self) -> None:
        """Perform end-of-day position closure."""
        if not self.position_manager or not self.end_of_day_closure:
            return
            
        logger.info("Executing end-of-day position closure")
        
        # Get all current prices - in a real implementation, you'd get current market prices
        current_prices = {}
        open_positions = self.position_manager.get_open_positions()
        
        for position in open_positions:
            current_prices[position.symbol] = position.last_price
            
        if current_prices:
            self.position_manager.close_all_positions(current_prices, reason="END_OF_DAY")
            logger.info("Closed all positions for end-of-day")
            
        # Reset EOD timer for the next day
        self._setup_eod_timer()
    
    def stop(self) -> None:
        """Stop risk manager timers."""
        # Cancel timers
        if self.eod_warning_timer:
            self.eod_warning_timer.cancel()
        if self.eod_closure_timer:
            self.eod_closure_timer.cancel()
            
        logger.info("Risk manager stopped")
    
    def start_new_trading_day(self) -> None:
        """
        Reset daily tracking for a new trading day.
        
        This should be called at the start of each trading day.
        """
        self.daily_losses = 0.0
        self.daily_trades_count = 0
        
        # Reset circuit breaker
        self.circuit_breaker_activated = False
        
        # Set up EOD timer for the new day
        if self.end_of_day_closure:
            self._setup_eod_timer()
            
        logger.info("Risk manager reset for new trading day")
    
    def set_position_manager(self, position_manager) -> None:
        """
        Set the position manager instance.
        
        Args:
            position_manager: Position manager instance
        """
        self.position_manager = position_manager