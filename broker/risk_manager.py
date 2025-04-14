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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import threading

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
        self.circuit_breaker_enabled = bool(self.risk_config.get('circuit_breaker', True))
        self.end_of_day_closure = bool(self.risk_config.get('end_of_day_closure', True))
        self.max_position_size = int(self.risk_config.get('max_position_size', 0))
        
        # Starting capital
        self.starting_capital = float(config.get('paper_trading', {}).get('starting_capital', 3000.0))
        
        # Circuit breaker parameters
        self.circuit_breaker_activated = False
        self.rapid_loss_threshold_percent = float(self.risk_config.get('rapid_loss_threshold', 3.0))
        self.rapid_loss_time_window = int(self.risk_config.get('rapid_loss_time_window', 300))  # seconds
        self.loss_events = []  # List of (timestamp, loss_percent) tuples
        
        # Trading allowed flag
        self.trading_allowed = True
        
        # End of day parameters
        self.market_close_time = self.risk_config.get('market_close_time', "15:30")
        self.closure_warning_minutes = int(self.risk_config.get('closure_warning_minutes', 15))
        self.eod_timer = None
        
        # Set up end of day timer if enabled
        if self.end_of_day_closure:
            self._setup_eod_timer()
        
        logger.info(f"Risk Manager initialized with max daily loss: {self.max_daily_loss}%, " +
                  f"max trades: {self.max_daily_trades}, circuit breaker: {self.circuit_breaker_enabled}")
    
    def check_position_size(self, symbol: str, quantity: int, price: float) -> bool:
        """
        Check if a position size is within limits.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Share price
            
        Returns:
            True if position size is allowed
        """
        # If max position size is 0, no limit is applied
        if self.max_position_size <= 0:
            return True
        
        if quantity > self.max_position_size:
            logger.warning(f"Rejecting {symbol} position: quantity {quantity} exceeds maximum {self.max_position_size}")
            return False
        
        return True
    
    def check_capital_percent(self, price: float, quantity: int, available_capital: float) -> bool:
        """
        Check if an order uses an acceptable percentage of capital.
        
        Args:
            price: Share price
            quantity: Number of shares
            available_capital: Available trading capital
            
        Returns:
            True if capital percentage is acceptable
        """
        order_value = price * quantity
        capital_percent = (order_value / available_capital) * 100
        
        # Get the maximum allowed capital percentage per trade
        max_capital_percent = float(self.risk_config.get('capital_percent', 95.0))
        
        if capital_percent > max_capital_percent:
            logger.warning(f"Rejecting order: uses {capital_percent:.2f}% of capital, exceeding limit of {max_capital_percent}%")
            return False
        
        return True
    
    def can_place_order(self, symbol: str, quantity: int, price: float, available_capital: float) -> bool:
        """
        Check if an order can be placed based on all risk criteria.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Share price
            available_capital: Available trading capital
            
        Returns:
            True if order is allowed
        """
        # Check if trading is generally allowed
        if not self.trading_allowed:
            logger.warning("Trading is currently disabled")
            return False
        
        # Check if circuit breaker is activated
        if self.circuit_breaker_activated:
            logger.warning("Circuit breaker is activated, no new trades allowed")
            return False
        
        # Check position size limits
        if not self.check_position_size(symbol, quantity, price):
            return False
        
        # Check capital percentage limits
        if not self.check_capital_percent(price, quantity, available_capital):
            return False
        
        # Check daily trading limits
        if self.position_manager and not self.position_manager.can_trade():
            return False
        
        # Check market hours
        if not self._is_market_open():
            logger.warning("Market is closed, trading not allowed")
            return False
        
        return True
    
    def register_loss(self, amount: float) -> None:
        """
        Register a loss event and check for circuit breaker activation.
        
        Args:
            amount: Loss amount
        """
        if not self.circuit_breaker_enabled:
            return
        
        # Calculate loss as percentage of starting capital
        loss_percent = (amount / self.starting_capital) * 100
        if loss_percent >= 0:  # Only register negative P&L
            return
            
        loss_percent = abs(loss_percent)  # Convert to positive for easier comparison
        current_time = time.time()
        
        # Register the loss event
        self.loss_events.append((current_time, loss_percent))
        
        # Clean up old events outside the time window
        cutoff_time = current_time - self.rapid_loss_time_window
        self.loss_events = [event for event in self.loss_events if event[0] >= cutoff_time]
        
        # Calculate total percentage loss in the time window
        total_loss_percent = sum(percent for _, percent in self.loss_events)
        
        # Check if circuit breaker should be activated
        if total_loss_percent >= self.rapid_loss_threshold_percent:
            self._activate_circuit_breaker()
    
    def _activate_circuit_breaker(self) -> None:
        """Activate the circuit breaker to halt trading."""
        if not self.circuit_breaker_activated:
            self.circuit_breaker_activated = True
            self.trading_allowed = False
            logger.warning("CIRCUIT BREAKER ACTIVATED - Trading halted due to rapid losses")
            
            # Notify about circuit breaker activation
            # TODO: Add notification mechanism
    
    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker (manual intervention)."""
        if self.circuit_breaker_activated:
            self.circuit_breaker_activated = False
            self.trading_allowed = True
            self.loss_events = []
            logger.info("Circuit breaker has been reset - Trading allowed")
            
            # Notify about circuit breaker reset
            # TODO: Add notification mechanism
    
    def _is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            True if market is open for trading
        """
        now = datetime.now()
        
        # Parse market open and close times
        market_open_time = self.risk_config.get('market_open_time', "09:15")
        market_close_time = self.market_close_time
        
        hour_open, minute_open = map(int, market_open_time.split(':'))
        hour_close, minute_close = map(int, market_close_time.split(':'))
        
        market_open = now.replace(hour=hour_open, minute=minute_open, second=0, microsecond=0)
        market_close = now.replace(hour=hour_close, minute=minute_close, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _setup_eod_timer(self) -> None:
        """Set up end of day closure timer."""
        now = datetime.now()
        
        # Parse market close time
        hour_close, minute_close = map(int, self.market_close_time.split(':'))
        market_close = now.replace(hour=hour_close, minute=minute_close, second=0, microsecond=0)
        
        # If market is already closed for today, do nothing
        if now > market_close:
            return
        
        # Calculate time until market close warning
        warning_time = market_close - timedelta(minutes=self.closure_warning_minutes)
        seconds_until_warning = max(0, (warning_time - now).total_seconds())
        
        # Schedule the warning
        if seconds_until_warning > 0:
            self.eod_timer = threading.Timer(seconds_until_warning, self._market_close_warning)
            self.eod_timer.daemon = True
            self.eod_timer.start()
            logger.info(f"Scheduled market close warning for {warning_time.strftime('%H:%M:%S')}")
        
        # Schedule the actual close
        seconds_until_close = max(0, (market_close - now).total_seconds())
        if seconds_until_close > 0:
            close_timer = threading.Timer(seconds_until_close, self._end_of_day_closure)
            close_timer.daemon = True
            close_timer.start()
            logger.info(f"Scheduled end of day closure for {market_close.strftime('%H:%M:%S')}")
    
    def _market_close_warning(self) -> None:
        """Send a warning that market close is approaching."""
        logger.warning(f"MARKET CLOSE WARNING - Market will close in {self.closure_warning_minutes} minutes")
        
        # Notify about market close warning
        # TODO: Add notification mechanism
    
    def _end_of_day_closure(self) -> None:
        """Perform end of day position closure."""
        if not self.end_of_day_closure:
            return
            
        logger.info("MARKET CLOSED - Executing end of day position closure")
        
        # Disable trading
        self.trading_allowed = False
        
        # Close all positions if position manager is available
        if self.position_manager:
            # In a real implementation, get current prices from market data module
            # For now, we'll use last prices stored in positions
            prices = {}
            for position in self.position_manager.get_open_positions():
                prices[position.symbol] = position.last_price
                
            self.position_manager.close_all_positions(prices, reason="EOD")
        
        # Reset circuit breaker and daily tracking
        self.reset_circuit_breaker()
        if self.position_manager:
            self.position_manager.reset_daily_tracking()
        
        # Notify about end of day closure
        # TODO: Add notification mechanism
    
    def stop(self) -> None:
        """Stop all timers and clean up resources."""
        if self.eod_timer and self.eod_timer.is_alive():
            self.eod_timer.cancel()
    
    def start_new_trading_day(self) -> None:
        """Initialize for a new trading day."""
        # Reset circuit breaker
        self.reset_circuit_breaker()
        
        # Enable trading
        self.trading_allowed = True
        
        # Clear loss events
        self.loss_events = []
        
        # Reset position manager daily tracking
        if self.position_manager:
            self.position_manager.reset_daily_tracking()
        
        # Set up end of day timer
        if self.end_of_day_closure:
            self._setup_eod_timer()
            
        logger.info("Risk manager initialized for new trading day")
    
    def set_position_manager(self, position_manager) -> None:
        """
        Set a reference to the position manager.
        
        Args:
            position_manager: Position manager instance
        """
        self.position_manager = position_manager