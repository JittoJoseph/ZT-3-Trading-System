"""
Position Management Module for ZT-3 Trading System.

This module handles tracking of positions, P&L calculations,
and position lifecycle management.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    """Enum for position status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    PENDING = "PENDING"

class Position:
    """
    Represents a trading position with tracking of P&L and states.
    """
    
    def __init__(self, 
                symbol: str, 
                entry_price: float, 
                quantity: int, 
                entry_time: datetime = None,
                take_profit_level: Optional[float] = None,
                stop_loss_level: Optional[float] = None,
                position_id: Optional[str] = None):
        """
        Initialize a new position.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price per share
            quantity: Number of shares
            entry_time: Time of entry (defaults to current time)
            take_profit_level: Optional take profit price
            stop_loss_level: Optional stop loss price
            position_id: Optional position ID (defaults to timestamp)
        """
        self.symbol = symbol
        self.entry_price = entry_price
        self.initial_quantity = quantity
        self.current_quantity = quantity
        self.entry_time = entry_time or datetime.now()
        self.exit_time = None
        self.exit_price = None
        self.take_profit_level = take_profit_level
        self.stop_loss_level = stop_loss_level
        self.status = PositionStatus.OPEN
        self.realized_pnl = 0.0
        self.position_id = position_id or f"{symbol}_{int(time.time())}"
        self.trades = []  # List of trades for this position
        self.exit_reason = None
        self.last_price = entry_price
        
        logger.info(f"Created position {self.position_id}: {symbol} x{quantity} @ {entry_price}")
    
    def update_price(self, current_price: float) -> None:
        """
        Update the current price for P&L calculations.
        
        Args:
            current_price: Current market price
        """
        self.last_price = current_price
    
    def get_unrealized_pnl(self) -> float:
        """
        Calculate unrealized P&L for the position.
        
        Returns:
            Unrealized P&L amount
        """
        if self.current_quantity > 0:
            return (self.last_price - self.entry_price) * self.current_quantity
        return 0.0
    
    def get_unrealized_pnl_percent(self) -> float:
        """
        Calculate unrealized P&L percentage.
        
        Returns:
            Unrealized P&L as percentage
        """
        if self.current_quantity > 0 and self.entry_price > 0:
            return ((self.last_price - self.entry_price) / self.entry_price) * 100
        return 0.0
    
    def get_realized_pnl(self) -> float:
        """
        Get realized P&L for closed or partially closed positions.
        
        Returns:
            Realized P&L amount
        """
        return self.realized_pnl
    
    def get_total_pnl(self) -> float:
        """
        Get total P&L (realized + unrealized).
        
        Returns:
            Total P&L amount
        """
        return self.realized_pnl + self.get_unrealized_pnl()
    
    def close_position(self, exit_price: float, exit_time: datetime = None, reason: str = None) -> float:
        """
        Close the position completely.
        
        Args:
            exit_price: Exit price per share
            exit_time: Time of exit (defaults to current time)
            reason: Reason for closing the position
            
        Returns:
            Realized P&L for this exit
        """
        if self.status == PositionStatus.CLOSED:
            logger.warning(f"Attempting to close already closed position {self.position_id}")
            return 0.0
        
        exit_time = exit_time or datetime.now()
        quantity_closed = self.current_quantity
        pnl = (exit_price - self.entry_price) * quantity_closed
        
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.realized_pnl += pnl
        self.status = PositionStatus.CLOSED
        self.current_quantity = 0
        self.exit_reason = reason
        
        # Record the trade
        trade = {
            "type": "EXIT",
            "price": exit_price,
            "quantity": quantity_closed,
            "time": exit_time,
            "pnl": pnl,
            "reason": reason
        }
        self.trades.append(trade)
        
        logger.info(f"Closed position {self.position_id}: {self.symbol} x{quantity_closed} @ {exit_price} | P&L: {pnl:.2f} | Reason: {reason}")
        return pnl
    
    def partial_close(self, quantity: int, exit_price: float, exit_time: datetime = None, reason: str = None) -> float:
        """
        Close part of the position.
        
        Args:
            quantity: Number of shares to close
            exit_price: Exit price per share
            exit_time: Time of exit (defaults to current time)
            reason: Reason for partial close
            
        Returns:
            Realized P&L for this partial exit
        """
        if self.status == PositionStatus.CLOSED:
            logger.warning(f"Attempting to partially close already closed position {self.position_id}")
            return 0.0
        
        if quantity > self.current_quantity:
            logger.warning(f"Attempting to close more shares than available in position {self.position_id}")
            quantity = self.current_quantity
        
        exit_time = exit_time or datetime.now()
        pnl = (exit_price - self.entry_price) * quantity
        
        self.current_quantity -= quantity
        self.realized_pnl += pnl
        
        # Update status
        if self.current_quantity == 0:
            self.status = PositionStatus.CLOSED
            self.exit_price = exit_price
            self.exit_time = exit_time
        else:
            self.status = PositionStatus.PARTIALLY_CLOSED
        
        # Record the trade
        trade = {
            "type": "PARTIAL_EXIT",
            "price": exit_price,
            "quantity": quantity,
            "time": exit_time,
            "pnl": pnl,
            "reason": reason
        }
        self.trades.append(trade)
        
        logger.info(f"Partially closed position {self.position_id}: {self.symbol} x{quantity}/{self.initial_quantity} @ {exit_price} | P&L: {pnl:.2f} | Reason: {reason}")
        return pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary for serialization.
        
        Returns:
            Dictionary representation of the position
        """
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "current_price": self.last_price,
            "initial_quantity": self.initial_quantity,
            "current_quantity": self.current_quantity,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "take_profit_level": self.take_profit_level,
            "stop_loss_level": self.stop_loss_level,
            "status": self.status.value,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.get_unrealized_pnl(),
            "total_pnl": self.get_total_pnl(),
            "pnl_percent": self.get_unrealized_pnl_percent(),
            "exit_reason": self.exit_reason,
            "trades": self.trades
        }


class PositionManager:
    """
    Manages all trading positions across multiple symbols.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the position manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.positions: Dict[str, Position] = {}  # position_id -> Position
        self.symbol_positions: Dict[str, Set[str]] = defaultdict(set)  # symbol -> set of position_ids
        self.open_positions: Set[str] = set()  # set of open position_ids
        self.closed_positions: Set[str] = set()  # set of closed position_ids
        
        # Daily tracking
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
        self.max_daily_loss = config.get('risk', {}).get('max_daily_loss', 5.0)
        self.max_daily_trades = config.get('risk', {}).get('max_daily_trades', 5)
        
        self.daily_positions: Dict[str, List[Position]] = {}  # date -> list of positions
        
        logger.info("Position manager initialized")
    
    def create_position(self, 
                       symbol: str, 
                       entry_price: float, 
                       quantity: int, 
                       take_profit_level: Optional[float] = None,
                       stop_loss_level: Optional[float] = None) -> str:
        """
        Create a new position.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price per share
            quantity: Number of shares
            take_profit_level: Optional take profit price
            stop_loss_level: Optional stop loss price
            
        Returns:
            Position ID of created position
        """
        # Create new position
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            take_profit_level=take_profit_level,
            stop_loss_level=stop_loss_level
        )
        
        position_id = position.position_id
        
        # Store position
        self.positions[position_id] = position
        self.symbol_positions[symbol].add(position_id)
        self.open_positions.add(position_id)
        
        # Update daily tracking
        self.daily_trades_count += 1
        
        # Update daily positions tracking
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_positions:
            self.daily_positions[today] = []
        self.daily_positions[today].append(position)
        
        return position_id
    
    def close_position(self, 
                      position_id: str, 
                      exit_price: float, 
                      reason: str = None) -> bool:
        """
        Close a position completely.
        
        Args:
            position_id: ID of position to close
            exit_price: Exit price per share
            reason: Reason for closing the position
            
        Returns:
            True if position was closed successfully
        """
        if position_id not in self.positions:
            logger.error(f"Position {position_id} not found")
            return False
        
        position = self.positions[position_id]
        
        if position.status == PositionStatus.CLOSED:
            logger.warning(f"Position {position_id} is already closed")
            return False
        
        # Close the position
        pnl = position.close_position(exit_price, reason=reason)
        
        # Update tracking
        self.daily_pnl += pnl
        if position_id in self.open_positions:
            self.open_positions.remove(position_id)
        self.closed_positions.add(position_id)
        
        return True
    
    def close_all_positions(self, prices: Dict[str, float], reason: str = "EOD") -> float:
        """
        Close all open positions.
        
        Args:
            prices: Dictionary of current prices for each symbol
            reason: Reason for closing positions
            
        Returns:
            Total realized P&L from closing positions
        """
        total_pnl = 0.0
        
        for position_id in list(self.open_positions):
            position = self.positions[position_id]
            symbol = position.symbol
            
            if symbol in prices:
                pnl = position.close_position(prices[symbol], reason=reason)
                total_pnl += pnl
                
                # Update tracking
                self.open_positions.remove(position_id)
                self.closed_positions.add(position_id)
            else:
                logger.warning(f"No price available to close position {position_id} for {symbol}")
        
        self.daily_pnl += total_pnl
        return total_pnl
    
    def close_positions_for_symbol(self, symbol: str, exit_price: float, reason: str = None) -> float:
        """
        Close all open positions for a specific symbol.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price per share
            reason: Reason for closing positions
            
        Returns:
            Total realized P&L from closing positions
        """
        if symbol not in self.symbol_positions:
            return 0.0
        
        total_pnl = 0.0
        
        for position_id in list(self.symbol_positions[symbol]):
            if position_id in self.open_positions:
                position = self.positions[position_id]
                pnl = position.close_position(exit_price, reason=reason)
                total_pnl += pnl
                
                # Update tracking
                self.open_positions.remove(position_id)
                self.closed_positions.add(position_id)
        
        self.daily_pnl += total_pnl
        return total_pnl
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Get a position by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position object or None if not found
        """
        return self.positions.get(position_id)
    
    def get_positions_for_symbol(self, symbol: str, only_open: bool = True) -> List[Position]:
        """
        Get positions for a specific symbol.
        
        Args:
            symbol: Trading symbol
            only_open: If True, return only open positions
            
        Returns:
            List of Position objects
        """
        if symbol not in self.symbol_positions:
            return []
        
        positions = []
        for position_id in self.symbol_positions[symbol]:
            position = self.positions[position_id]
            if not only_open or position.status in (PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED):
                positions.append(position)
        
        return positions
    
    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of open Position objects
        """
        return [self.positions[position_id] for position_id in self.open_positions]
    
    def get_position_count(self, only_open: bool = True) -> int:
        """
        Get the number of positions.
        
        Args:
            only_open: If True, count only open positions
            
        Returns:
            Number of positions
        """
        if only_open:
            return len(self.open_positions)
        return len(self.positions)
    
    def update_position_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for all positions.
        
        Args:
            prices: Dictionary of current prices for each symbol
        """
        for position_id in self.open_positions:
            position = self.positions[position_id]
            symbol = position.symbol
            
            if symbol in prices:
                position.update_price(prices[symbol])
    
    def get_total_pnl(self, realized_only: bool = False) -> float:
        """
        Get total P&L across all positions.
        
        Args:
            realized_only: If True, include only realized P&L
            
        Returns:
            Total P&L amount
        """
        total = 0.0
        
        for position in self.positions.values():
            if realized_only:
                total += position.get_realized_pnl()
            else:
                total += position.get_total_pnl()
        
        return total
    
    def get_daily_pnl_summary(self) -> Dict[str, float]:
        """
        Get P&L summary for today.
        
        Returns:
            Dictionary with daily P&L metrics
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in self.daily_positions:
            return {
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "total_pnl": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "total_count": 0,
                "win_rate": 0.0
            }
        
        today_positions = self.daily_positions[today]
        
        realized_pnl = sum(p.get_realized_pnl() for p in today_positions)
        unrealized_pnl = sum(p.get_unrealized_pnl() for p in today_positions)
        total_pnl = realized_pnl + unrealized_pnl
        
        win_count = sum(1 for p in today_positions if p.get_realized_pnl() > 0)
        loss_count = sum(1 for p in today_positions if p.get_realized_pnl() < 0)
        total_count = len(today_positions)
        
        win_rate = (win_count / total_count * 100) if total_count > 0 else 0.0
        
        return {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "win_count": win_count,
            "loss_count": loss_count,
            "total_count": total_count,
            "win_rate": win_rate
        }
    
    def reset_daily_tracking(self) -> None:
        """
        Reset daily tracking metrics.
        
        This should be called at the start of a new trading day.
        """
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
    
    def can_trade(self) -> bool:
        """
        Check if allowed to trade based on risk parameters.
        
        Returns:
            True if trading is allowed
        """
        # Check daily loss limit
        capital = self.config.get('paper_trading', {}).get('starting_capital', 3000.0)
        current_pnl_percent = (self.daily_pnl / capital) * 100
        
        if abs(current_pnl_percent) >= self.max_daily_loss and current_pnl_percent < 0:
            logger.warning(f"Daily loss limit reached: {current_pnl_percent:.2f}% (limit: {self.max_daily_loss:.2f}%)")
            return False
        
        # Check maximum trades per day
        if self.daily_trades_count >= self.max_daily_trades:
            logger.warning(f"Maximum daily trades reached: {self.daily_trades_count} (limit: {self.max_daily_trades})")
            return False
        
        return True
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all positions.
        
        Returns:
            Dictionary with position summary
        """
        open_count = len(self.open_positions)
        closed_count = len(self.closed_positions)
        symbols = len(self.symbol_positions)
        
        realized_pnl = sum(p.get_realized_pnl() for p in self.positions.values())
        unrealized_pnl = sum(p.get_unrealized_pnl() for p in self.positions.values())
        total_pnl = realized_pnl + unrealized_pnl
        
        return {
            "open_positions": open_count,
            "closed_positions": closed_count,
            "total_positions": open_count + closed_count,
            "symbols": symbols,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl
        }