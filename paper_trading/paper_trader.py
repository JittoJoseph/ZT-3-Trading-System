"""
Paper Trading Module for ZT-3 Trading System.

This module provides a simulated trading environment for testing
strategies without risking real money.
"""

import logging
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from broker.position_manager import PositionManager, Position, PositionStatus
from strategy import SignalType, ExitReason, Signal

logger = logging.getLogger(__name__)

class PaperTrader:
    """
    Paper trading simulation engine.
    
    Simulates trades with realistic execution, slippage, and fees
    for testing strategies in a risk-free environment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the paper trader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Extract paper trading configuration
        paper_config = self.config.get('paper_trading', {})
        self.starting_capital = float(paper_config.get('starting_capital', 3000.0))
        self.current_capital = self.starting_capital
        self.simulate_slippage = paper_config.get('simulate_slippage', True)
        self.slippage_percent = float(paper_config.get('slippage_percent', 0.1))
        self.commission_percent = float(paper_config.get('commission_percent', 0.03))
        
        # Initialize position manager
        self.position_manager = PositionManager(config)
        
        # Trade history
        self.trade_history = []
        
        # Latest prices for each symbol
        self.latest_prices = {}
        
        # Profit and loss tracking
        self.total_commissions = 0.0
        self.daily_pnl = 0.0
        self.overall_pnl = 0.0
        self.highest_capital = self.starting_capital
        self.max_drawdown = 0.0
        
        # Statistics
        self.win_trades = 0
        self.loss_trades = 0
        self.total_trades = 0
        
        logger.info(f"Paper trading initialized with ₹{self.starting_capital:.2f} capital")
    
    def update_price(self, symbol: str, price: Dict[str, Any]) -> None:
        """
        Update the latest price data for a symbol.
        
        Args:
            symbol: Trading symbol
            price: Dictionary with price data (OHLCV)
        """
        self.latest_prices[symbol] = price
        
        # Update positions with new prices for P&L calculation
        if symbol in self.latest_prices:
            close_price = float(price['close'])
            self.position_manager.update_position_prices({symbol: close_price})
            
        # Recalculate current capital and drawdown
        self._update_capital_and_drawdown()
    
    def _update_capital_and_drawdown(self) -> None:
        """Update current capital and maximum drawdown."""
        # Calculate total position value
        total_position_value = 0.0
        for position in self.position_manager.get_open_positions():
            total_position_value += position.current_quantity * position.last_price
        
        # Update current capital (cash + positions)
        unrealized_pnl = self.position_manager.get_total_pnl(realized_only=False) - self.position_manager.get_total_pnl(realized_only=True)
        realized_pnl = self.position_manager.get_total_pnl(realized_only=True)
        
        # Current capital is starting capital + realized P&L + unrealized P&L - commissions
        self.current_capital = self.starting_capital + realized_pnl + unrealized_pnl - self.total_commissions
        
        # Update highest capital
        if self.current_capital > self.highest_capital:
            self.highest_capital = self.current_capital
        
        # Update maximum drawdown
        current_drawdown = self.highest_capital - self.current_capital
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Apply simulated slippage to a price.
        
        Args:
            price: Base price
            is_buy: True for buy orders, False for sell orders
            
        Returns:
            Price with slippage applied
        """
        if not self.simulate_slippage:
            return price
            
        # Generate random slippage within the configured percentage
        # Buy orders get worse prices (higher), sell orders get worse prices (lower)
        max_slippage = price * self.slippage_percent / 100.0
        slippage_amount = random.uniform(0, max_slippage)
        
        if is_buy:
            return price + slippage_amount
        else:
            return price - slippage_amount
    
    def _calculate_commission(self, price: float, quantity: int) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            price: Trade price
            quantity: Number of shares
            
        Returns:
            Commission amount
        """
        trade_value = price * quantity
        return trade_value * self.commission_percent / 100.0
    
    def _create_trade_record(self, 
                             symbol: str,
                             order_type: str, 
                             price: float, 
                             quantity: int,
                             commission: float,
                             slippage: float,
                             take_profit: Optional[float] = None,
                             stop_loss: Optional[float] = None,
                             pnl: Optional[float] = None,
                             exit_reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a trade record for history.
        
        Args:
            symbol: Trading symbol
            order_type: Type of order (BUY/SELL)
            price: Execution price
            quantity: Number of shares
            commission: Commission paid
            slippage: Slippage amount
            take_profit: Optional take profit level
            stop_loss: Optional stop loss level
            pnl: Optional P&L for exits
            exit_reason: Optional exit reason
            
        Returns:
            Trade record dictionary
        """
        return {
            "symbol": symbol,
            "type": order_type,
            "price": price,
            "quantity": quantity,
            "commission": commission,
            "slippage": slippage,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "pnl": pnl,
            "exit_reason": exit_reason,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_signal(self, signal: Signal) -> Dict[str, Any]:
        """
        Process a trading signal and execute a simulated trade.
        
        Args:
            signal: Trading signal
            
        Returns:
            Dictionary with trade result
        """
        symbol = signal.symbol
        signal_type = signal.signal_type
        
        # Ensure we have latest prices
        if symbol not in self.latest_prices:
            logger.warning(f"No price data available for {symbol}")
            return {"success": False, "message": f"No price data available for {symbol}"}
        
        latest_price = self.latest_prices[symbol]
        close_price = float(latest_price['close'])
        
        # Process entry signals
        if signal_type == SignalType.ENTRY:
            # Check if there's already an open position
            open_positions = self.position_manager.get_positions_for_symbol(symbol, only_open=True)
            if open_positions:
                logger.info(f"Position already exists for {symbol}, ignoring entry signal")
                return {"success": False, "message": f"Position already exists for {symbol}"}
            
            # Calculate position size based on available capital
            risk_params = self.config.get('risk', {})
            capital_percent = risk_params.get('capital_percent', 95.0)
            capital_to_use = self.current_capital * (capital_percent / 100.0)
            
            # Calculate maximum shares we can buy with the available capital
            price_with_slippage = self._apply_slippage(close_price, is_buy=True)
            max_shares = int(capital_to_use / price_with_slippage)
            
            # Ensure quantity is positive
            if max_shares <= 0:
                logger.warning(f"Insufficient capital to buy {symbol}")
                return {"success": False, "message": "Insufficient capital for trade"}
            
            # Apply commission to check if we can afford the trade
            commission = self._calculate_commission(price_with_slippage, max_shares)
            
            # Check if we can afford the trade with commission
            total_cost = (price_with_slippage * max_shares) + commission
            
            if total_cost > self.current_capital:
                # Reduce shares to fit capital
                max_shares = int((self.current_capital - commission) / price_with_slippage)
                
                if max_shares <= 0:
                    logger.warning(f"Insufficient capital to buy {symbol} after commission")
                    return {"success": False, "message": "Insufficient capital after commission"}
                    
                # Recalculate commission
                commission = self._calculate_commission(price_with_slippage, max_shares)
                total_cost = (price_with_slippage * max_shares) + commission
            
            # Create position
            take_profit_level = None
            if signal.take_profit_level:
                take_profit_level = signal.take_profit_level
            
            stop_loss_level = None
            if signal.stop_loss_level:
                stop_loss_level = signal.stop_loss_level
            
            # Create the position
            position_id = self.position_manager.create_position(
                symbol=symbol,
                entry_price=price_with_slippage,
                quantity=max_shares,
                take_profit_level=take_profit_level,
                stop_loss_level=stop_loss_level
            )
            
            # Update capital and commissions
            self.total_commissions += commission
            self._update_capital_and_drawdown()
            
            # Create trade record
            trade_record = self._create_trade_record(
                symbol=symbol,
                order_type="BUY",
                price=price_with_slippage,
                quantity=max_shares,
                commission=commission,
                slippage=price_with_slippage - close_price,
                take_profit=take_profit_level,
                stop_loss=stop_loss_level
            )
            
            self.trade_history.append(trade_record)
            
            logger.info(f"BUY {symbol}: {max_shares} shares @ ₹{price_with_slippage:.2f} | Commission: ₹{commission:.2f}")
            
            return {
                "success": True,
                "action": "BUY",
                "symbol": symbol,
                "quantity": max_shares,
                "price": price_with_slippage,
                "commission": commission,
                "position_id": position_id,
                "take_profit": take_profit_level,
                "stop_loss": stop_loss_level
            }
            
        # Process exit signals
        elif signal_type == SignalType.EXIT:
            # Check for open positions
            open_positions = self.position_manager.get_positions_for_symbol(symbol, only_open=True)
            
            if not open_positions:
                logger.info(f"No open positions for {symbol}, ignoring exit signal")
                return {"success": False, "message": f"No open positions for {symbol}"}
            
            # Get the position to close
            position = open_positions[0]
            price_with_slippage = self._apply_slippage(close_price, is_buy=False)
            
            # Calculate commission
            commission = self._calculate_commission(price_with_slippage, position.current_quantity)
            
            # Close the position
            exit_reason_str = signal.exit_reason.name if signal.exit_reason else "SIGNAL"
            pnl = self.position_manager.close_position(
                position_id=position.position_id,
                exit_price=price_with_slippage,
                reason=exit_reason_str
            )
            
            # Update statistics
            self.total_trades += 1
            if pnl > 0:
                self.win_trades += 1
            elif pnl < 0:
                self.loss_trades += 1
                
            # Update total commissions
            self.total_commissions += commission
            self.daily_pnl += pnl
            self.overall_pnl += pnl
            
            # Create trade record
            trade_record = self._create_trade_record(
                symbol=symbol,
                order_type="SELL",
                price=price_with_slippage,
                quantity=position.current_quantity,
                commission=commission,
                slippage=close_price - price_with_slippage,
                pnl=pnl,
                exit_reason=exit_reason_str
            )
            
            self.trade_history.append(trade_record)
            
            # Update capital and drawdown
            self._update_capital_and_drawdown()
            
            logger.info(f"SELL {symbol}: {position.current_quantity} shares @ ₹{price_with_slippage:.2f} | P&L: ₹{pnl:.2f} | Commission: ₹{commission:.2f}")
            
            return {
                "success": True,
                "action": "SELL",
                "symbol": symbol,
                "quantity": position.current_quantity,
                "price": price_with_slippage,
                "commission": commission,
                "pnl": pnl,
                "exit_reason": exit_reason_str
            }
        
        return {"success": False, "message": "Invalid signal type"}
    
    def check_exit_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Check for exit conditions based on price movement.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Trade result if an exit was triggered, None otherwise
        """
        if symbol not in self.latest_prices:
            return None
            
        latest_price = self.latest_prices[symbol]
        open_positions = self.position_manager.get_positions_for_symbol(symbol, only_open=True)
        
        if not open_positions:
            return None
            
        position = open_positions[0]
        
        # Check take profit
        if position.take_profit_level and float(latest_price['high']) >= position.take_profit_level:
            # Create exit signal
            exit_signal = Signal(
                signal_type=SignalType.EXIT,
                symbol=symbol,
                price=position.take_profit_level,
                timestamp=pd.Timestamp(latest_price.get('timestamp', datetime.now())),
                exit_reason=ExitReason.TAKE_PROFIT
            )
            
            # Process the exit signal
            result = self.process_signal(exit_signal)
            if result["success"]:
                logger.info(f"Take profit triggered for {symbol} at {position.take_profit_level:.2f}")
                return result
        
        # Check stop loss
        if position.stop_loss_level and float(latest_price['low']) <= position.stop_loss_level:
            # Create exit signal
            exit_signal = Signal(
                signal_type=SignalType.EXIT,
                symbol=symbol,
                price=position.stop_loss_level,
                timestamp=pd.Timestamp(latest_price.get('timestamp', datetime.now())),
                exit_reason=ExitReason.TRAILING_STOP
            )
            
            # Process the exit signal
            result = self.process_signal(exit_signal)
            if result["success"]:
                logger.info(f"Stop loss triggered for {symbol} at {position.stop_loss_level:.2f}")
                return result
                
        return None
    
    def close_all_positions(self, reason: str = "END_OF_DAY") -> Dict[str, Any]:
        """
        Close all open positions.
        
        Args:
            reason: Reason for closing positions
            
        Returns:
            Dictionary with results
        """
        positions = self.position_manager.get_open_positions()
        
        if not positions:
            return {"success": True, "message": "No open positions to close", "positions_closed": 0}
        
        # Close each position
        closed_positions = 0
        total_pnl = 0.0
        results = []
        
        for position in positions:
            symbol = position.symbol
            
            if symbol in self.latest_prices:
                latest_price = self.latest_prices[symbol]
                close_price = float(latest_price['close'])
                
                # Create and process exit signal
                exit_signal = Signal(
                    signal_type=SignalType.EXIT,
                    symbol=symbol,
                    price=close_price,
                    timestamp=pd.Timestamp(latest_price.get('timestamp', datetime.now())),
                    exit_reason=ExitReason.END_OF_DAY if reason == "END_OF_DAY" else ExitReason.MANUAL
                )
                
                result = self.process_signal(exit_signal)
                
                if result["success"]:
                    closed_positions += 1
                    total_pnl += result["pnl"]
                    results.append(result)
            else:
                logger.warning(f"No price data available to close position for {symbol}")
        
        return {
            "success": True,
            "message": f"Closed {closed_positions} positions with total P&L: ₹{total_pnl:.2f}",
            "positions_closed": closed_positions,
            "total_pnl": total_pnl,
            "results": results
        }
    
    def reset_daily_tracking(self) -> None:
        """Reset daily statistics."""
        self.daily_pnl = 0.0
        self.position_manager.reset_daily_tracking()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the paper trading session.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate win rate
        win_rate = 0.0
        if self.total_trades > 0:
            win_rate = (self.win_trades / self.total_trades) * 100.0
            
        # Calculate drawdown percent
        max_drawdown_percent = 0.0
        if self.highest_capital > 0:
            max_drawdown_percent = (self.max_drawdown / self.highest_capital) * 100.0
            
        # Calculate P&L percentage
        pnl_percent = ((self.current_capital - self.starting_capital) / self.starting_capital) * 100.0
        
        # Get best and worst trades
        best_trade = {}
        worst_trade = {}
        
        if self.trade_history:
            # Filter only sell trades
            sell_trades = [t for t in self.trade_history if t.get('type') == 'SELL' and t.get('pnl') is not None]
            
            if sell_trades:
                best_trade = max(sell_trades, key=lambda x: x.get('pnl', 0))
                worst_trade = min(sell_trades, key=lambda x: x.get('pnl', 0))
        
        return {
            "starting_capital": self.starting_capital,
            "current_capital": self.current_capital,
            "net_pnl": self.overall_pnl,
            "net_pnl_percent": pnl_percent,
            "daily_pnl": self.daily_pnl,
            "total_commissions": self.total_commissions,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_percent": max_drawdown_percent,
            "win_trades": self.win_trades,
            "loss_trades": self.loss_trades,
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "open_positions": len(self.position_manager.get_open_positions()),
            "best_trade": {
                "symbol": best_trade.get('symbol', ''),
                "pnl": best_trade.get('pnl', 0),
                "timestamp": best_trade.get('timestamp', '')
            } if best_trade else {},
            "worst_trade": {
                "symbol": worst_trade.get('symbol', ''),
                "pnl": worst_trade.get('pnl', 0),
                "timestamp": worst_trade.get('timestamp', '')
            } if worst_trade else {},
            "date": datetime.now().strftime("%Y-%m-%d")
        }
