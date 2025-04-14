"""
Backtesting Module for ZT-3 Trading System.

This module provides backtesting functionality to test
the ZT-3 Trading System strategies on historical data.
"""

import logging
import time
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Import strategy and indicator modules
from strategy import SignalType, ExitReason, Signal
from strategy.indicators import Indicators

logger = logging.getLogger(__name__)

class Trade:
    """
    Represents a trade during backtesting.
    """
    
    def __init__(self, 
                symbol: str, 
                entry_price: float, 
                entry_time: datetime,
                quantity: int,
                take_profit: Optional[float] = None,
                stop_loss: Optional[float] = None):
        """
        Initialize a new trade.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            entry_time: Entry timestamp
            quantity: Number of shares/contracts
            take_profit: Optional take profit level
            stop_loss: Optional stop loss level
        """
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.quantity = quantity
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        
        # Exit information (to be filled when trade is closed)
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_percent = 0.0
        self.is_open = True
        self.duration = None  # To be filled with timedelta
    
    def close(self, 
             exit_price: float, 
             exit_time: datetime, 
             exit_reason: str) -> float:
        """
        Close the trade.
        
        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Reason for exit
            
        Returns:
            Realized PnL
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        self.is_open = False
        self.duration = exit_time - self.entry_time
        
        # Calculate PnL
        self.pnl = (exit_price - self.entry_price) * self.quantity
        self.pnl_percent = ((exit_price - self.entry_price) / self.entry_price) * 100
        
        return self.pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trade to dictionary for serialization.
        
        Returns:
            Dictionary representation of the trade
        """
        result = {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "quantity": self.quantity,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "is_open": self.is_open,
        }
        
        if not self.is_open:
            result.update({
                "exit_price": self.exit_price,
                "exit_time": self.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_reason": self.exit_reason,
                "pnl": self.pnl,
                "pnl_percent": self.pnl_percent,
                "duration_minutes": self.duration.total_seconds() / 60 if self.duration else None,
            })
        
        return result


class Backtester:
    """
    Backtester for the ZT-3 Trading System.
    
    Tests trading strategies on historical data and generates
    performance metrics and visualizations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backtester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.strategy_name = config.get('strategy', {}).get('name', 'GaussianChannelStrategy')
        self.strategy_params = config.get('strategy', {}).get('params', {})
        self.symbols = [f"{s['exchange']}:{s['ticker']}" for s in config.get('symbols', [])]
        
        # Trading settings
        self.starting_capital = float(config.get('backtest', {}).get('starting_capital', 3000.0))
        self.commission_percent = float(config.get('backtest', {}).get('commission_percent', 0.03))
        self.slippage_percent = float(config.get('backtest', {}).get('slippage_percent', 0.1))
        
        # Store historical data and trades
        self.data = {}  # {symbol: DataFrame}
        self.trades = []  # List of Trade objects
        self.equity_curve = None  # To be created during backtest
        
        # Performance metrics (to be calculated after backtest)
        self.metrics = {}
        
        # Strategy instance
        self.strategy = self._load_strategy()
        
        logger.info(f"Backtester initialized for {self.strategy_name} with {len(self.symbols)} symbols")
    
    def _load_strategy(self):
        """
        Load the strategy class dynamically.
        
        Returns:
            Strategy instance
        """
        try:
            # Import the strategy module dynamically
            from strategy.gaussian_channel import GaussianChannelStrategy
            
            # Create and return the strategy instance
            return GaussianChannelStrategy(self.config)
            
        except Exception as e:
            logger.error(f"Failed to load strategy {self.strategy_name}: {e}")
            raise
    
    def load_data(self, 
                symbol: str, 
                start_date: str, 
                end_date: str, 
                source: str = 'api',
                interval: str = '5min',
                csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical data for a symbol.
        
        Args:
            symbol: Trading symbol (format: exchange:ticker)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            source: Data source ('api' or 'csv')
            interval: Timeframe interval
            csv_path: Path to CSV file if source is 'csv'
            
        Returns:
            DataFrame with historical data
        """
        if source == 'api':
            # This would connect to the Upstox API in a real implementation
            # For backtesting purposes, we'll use local data or mock data
            logger.warning("API data fetching not implemented yet, using sample data")
            df = self._generate_sample_data(symbol, start_date, end_date, interval)
        
        elif source == 'csv':
            if not csv_path:
                raise ValueError("csv_path must be provided when source is 'csv'")
            
            # Load data from CSV
            df = pd.read_csv(csv_path, parse_dates=['timestamp'])
            
            # Filter by date range
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
            # Set timestamp as index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
        
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Add technical indicators required by the strategy
        df = Indicators.add_all_indicators(df, self.config)
        
        # Store the data
        self.data[symbol] = df
        
        logger.info(f"Loaded {len(df)} data points for {symbol} from {start_date} to {end_date}")
        return df
    
    def _generate_sample_data(self, 
                            symbol: str, 
                            start_date: str, 
                            end_date: str,
                            interval: str) -> pd.DataFrame:
        """
        Generate sample data for testing when API data is not available.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Timeframe interval
            
        Returns:
            DataFrame with sample data
        """
        # Convert dates to datetime
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Determine frequency from interval
        freq_map = {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '60min',
            'day': '1D'
        }
        freq = freq_map.get(interval, '5min')
        
        # Generate timestamp range
        # For market hours only (9:15 AM to 3:30 PM, Monday to Friday)
        timestamps = []
        current = start
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:  # Monday to Friday
                # Add timestamps for market hours
                market_open = current.replace(hour=9, minute=15)
                market_close = current.replace(hour=15, minute=30)
                
                if freq in ['1min', '5min', '15min', '30min', '60min']:
                    # Generate intraday timestamps
                    time = market_open
                    while time <= market_close:
                        timestamps.append(time)
                        time += pd.Timedelta(freq)
                else:
                    # Daily or higher frequency
                    timestamps.append(current)
            
            # Move to next day
            if freq in ['1D', '1W', '1M']:
                current += pd.Timedelta(freq)
            else:
                current += pd.Timedelta(days=1)
        
        # Generate sample prices
        n = len(timestamps)
        
        # Start with a random price
        base_price = 100 + 50 * np.random.random()
        
        # Generate price series with some randomness and trend
        np.random.seed(42)  # For reproducibility
        
        # Create price series with random walk and some cyclical behavior
        price_changes = np.random.normal(0, 1, n) * 0.5  # Daily random changes
        trend = np.linspace(0, 10, n) * 0.1  # Slight upward trend
        cycle = 5 * np.sin(np.linspace(0, 5 * np.pi, n))  # Cyclical component
        
        # Combine all components
        closes = base_price + np.cumsum(price_changes) + trend + cycle
        
        # Generate other OHLC data
        daily_volatility = 0.02
        opens = closes - np.random.normal(0, daily_volatility, n) * closes
        highs = np.maximum(opens, closes) + np.random.normal(daily_volatility, daily_volatility, n) * closes
        lows = np.minimum(opens, closes) - np.random.normal(daily_volatility, daily_volatility, n) * closes
        
        # Generate volume data
        volumes = np.random.lognormal(10, 1, n) * 1000
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def run_backtest(self, 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest on loaded data.
        
        Args:
            start_date: Optional start date to filter data (YYYY-MM-DD)
            end_date: Optional end date to filter data (YYYY-MM-DD)
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize backtest variables
        self.trades = []
        open_trades = {}  # symbol -> Trade
        cash = self.starting_capital
        equity = cash
        equity_curve = []
        daily_returns = []
        
        # Process each symbol
        for symbol in self.symbols:
            if symbol not in self.data:
                logger.warning(f"No data loaded for {symbol}, skipping")
                continue
                
            df = self.data[symbol]
            
            # Filter by date range if provided
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]
            
            # Iterate through each candle
            for i in range(len(df)):
                # Skip the first few candles until we have enough data for indicators
                if i < max(self.strategy_params.get('gc_period', 144), 50):
                    continue
                
                # Get current candle and data up to current candle
                data_until_current = df.iloc[:i+1]
                current_candle = df.iloc[i]
                timestamp = current_candle.name  # Index is timestamp
                
                # Check for exit conditions first (for open trades)
                if symbol in open_trades:
                    trade = open_trades[symbol]
                    current_price = current_candle['close']
                    
                    # Check stop loss
                    if trade.stop_loss and current_candle['low'] <= trade.stop_loss:
                        # Stop loss hit
                        exit_price = self._apply_slippage(trade.stop_loss, False)
                        trade_pnl = trade.close(exit_price, timestamp, "STOP_LOSS")
                        
                        # Update cash and remove from open trades
                        cash += trade.entry_price * trade.quantity + trade_pnl - self._calculate_commission(exit_price, trade.quantity)
                        self.trades.append(trade)
                        del open_trades[symbol]
                        
                        logger.info(f"Backtest: Stop loss hit for {symbol} at {exit_price:.2f}, PnL: {trade_pnl:.2f}")
                        
                    # Check take profit
                    elif trade.take_profit and current_candle['high'] >= trade.take_profit:
                        # Take profit hit
                        exit_price = self._apply_slippage(trade.take_profit, False)
                        trade_pnl = trade.close(exit_price, timestamp, "TAKE_PROFIT")
                        
                        # Update cash and remove from open trades
                        cash += trade.entry_price * trade.quantity + trade_pnl - self._calculate_commission(exit_price, trade.quantity)
                        self.trades.append(trade)
                        del open_trades[symbol]
                        
                        logger.info(f"Backtest: Take profit hit for {symbol} at {exit_price:.2f}, PnL: {trade_pnl:.2f}")
                
                # Process with strategy
                try:
                    signal = self.strategy.process_candle(data_until_current, symbol)
                    
                    if signal:
                        if signal.signal_type == SignalType.ENTRY and symbol not in open_trades:
                            # Entry signal
                            entry_price = self._apply_slippage(signal.price, True)
                            
                            # Calculate position size (fixed percentage of capital)
                            capital_percent = self.config.get('risk', {}).get('capital_percent', 95.0) / 100.0
                            position_size = int((cash * capital_percent) / entry_price)
                            
                            if position_size > 0:
                                # Calculate commission
                                commission = self._calculate_commission(entry_price, position_size)
                                
                                # Create trade
                                trade = Trade(
                                    symbol=symbol,
                                    entry_price=entry_price,
                                    entry_time=timestamp,
                                    quantity=position_size,
                                    take_profit=signal.take_profit_level,
                                    stop_loss=signal.stop_loss_level
                                )
                                
                                # Update cash and add to open trades
                                cash -= (entry_price * position_size + commission)
                                open_trades[symbol] = trade
                                
                                logger.info(f"Backtest: Entry for {symbol} at {entry_price:.2f}, Qty: {position_size}")
                                
                        elif signal.signal_type == SignalType.EXIT and symbol in open_trades:
                            # Exit signal
                            exit_price = self._apply_slippage(signal.price, False)
                            trade = open_trades[symbol]
                            exit_reason = signal.exit_reason.value if signal.exit_reason else "SIGNAL"
                            
                            # Close trade
                            trade_pnl = trade.close(exit_price, timestamp, exit_reason)
                            
                            # Update cash and remove from open trades
                            cash += trade.entry_price * trade.quantity + trade_pnl - self._calculate_commission(exit_price, trade.quantity)
                            self.trades.append(trade)
                            del open_trades[symbol]
                            
                            logger.info(f"Backtest: Exit for {symbol} at {exit_price:.2f}, PnL: {trade_pnl:.2f}")
                
                except Exception as e:
                    logger.error(f"Error during backtest for {symbol} at {timestamp}: {e}")
                
                # Calculate equity at this point
                current_equity = cash
                for sym, trade in open_trades.items():
                    # For open trades, use current price
                    if sym == symbol:
                        current_price = current_candle['close']
                    else:
                        # For other symbols, use the most recent price
                        current_price = self.data[sym].loc[self.data[sym].index <= timestamp, 'close'][-1]
                    
                    # Add unrealized P&L to equity
                    current_equity += trade.quantity * current_price - trade.entry_price * trade.quantity
                
                # Record equity for equity curve
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': current_equity
                })
                
                # Calculate daily return if this is the last candle of the day
                if i == len(df) - 1 or df.index[i].date() != df.index[i+1].date():
                    if len(equity_curve) > 1:
                        prev_day_equity = next(
                            (point['equity'] for point in reversed(equity_curve[:-1]) 
                             if point['timestamp'].date() < timestamp.date()), 
                            self.starting_capital
                        )
                        
                        daily_return = (current_equity - prev_day_equity) / prev_day_equity
                        daily_returns.append({
                            'date': timestamp.date(),
                            'return': daily_return
                        })
        
        # Close any remaining open trades at the end of the backtest
        for symbol, trade in list(open_trades.items()):
            last_price = self.data[symbol]['close'][-1]
            exit_price = self._apply_slippage(last_price, False)
            trade_pnl = trade.close(exit_price, df.index[-1], "END_OF_BACKTEST")
            
            # Update cash
            cash += trade.entry_price * trade.quantity + trade_pnl - self._calculate_commission(exit_price, trade.quantity)
            self.trades.append(trade)
            
            logger.info(f"Backtest: Closing remaining trade for {symbol} at {exit_price:.2f}, PnL: {trade_pnl:.2f}")
        
        # Convert equity curve and daily returns to DataFrames
        self.equity_curve = pd.DataFrame(equity_curve).set_index('timestamp')
        daily_returns_df = pd.DataFrame(daily_returns).set_index('date')
        
        # Calculate performance metrics
        self.metrics = self._calculate_metrics(self.equity_curve, daily_returns_df)
        
        logger.info(f"Backtest completed: {self.metrics['total_trades']} trades, Final equity: {self.metrics['final_equity']:.2f}")
        
        return self.metrics
    
    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Apply slippage to price.
        
        Args:
            price: Base price
            is_buy: True for buy orders, False for sell orders
            
        Returns:
            Price with slippage applied
        """
        if self.slippage_percent == 0:
            return price
            
        # Calculate slippage amount
        slippage_amount = price * self.slippage_percent / 100.0
        
        # Apply slippage (higher price for buys, lower for sells)
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
    
    def _calculate_metrics(self, equity_curve: pd.DataFrame, daily_returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            equity_curve: DataFrame with equity curve
            daily_returns: DataFrame with daily returns
            
        Returns:
            Dictionary with performance metrics
        """
        # Extract basic metrics
        start_equity = self.starting_capital
        final_equity = equity_curve['equity'].iloc[-1]
        
        # Calculate returns
        total_return = (final_equity - start_equity) / start_equity
        annual_return = total_return * (252 / len(daily_returns)) if len(daily_returns) > 0 else 0
        
        # Calculate drawdown
        rolling_max = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Calculate Sharpe Ratio (annualized)
        if len(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns['return'].mean() / daily_returns['return'].std() \
                if daily_returns['return'].std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate trade metrics
        win_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        loss_trades = sum(1 for trade in self.trades if trade.pnl <= 0)
        total_trades = len(self.trades)
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average profit and loss
        avg_profit = np.mean([trade.pnl for trade in self.trades if trade.pnl > 0]) \
            if win_trades > 0 else 0
        avg_loss = np.mean([trade.pnl for trade in self.trades if trade.pnl <= 0]) \
            if loss_trades > 0 else 0
        
        # Calculate profit factor
        total_profit = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        total_loss = sum(abs(trade.pnl) for trade in self.trades if trade.pnl <= 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate expectancy and average trade
        expectancy = (win_rate * avg_profit) - ((1 - win_rate) * abs(avg_loss)) \
            if total_trades > 0 else 0
        avg_trade = sum(trade.pnl for trade in self.trades) / total_trades \
            if total_trades > 0 else 0
        
        # Calculate average trade duration
        durations = [trade.duration.total_seconds() / 60 for trade in self.trades if trade.duration]
        avg_duration = np.mean(durations) if durations else 0
        
        return {
            'start_date': equity_curve.index[0].strftime('%Y-%m-%d'),
            'end_date': equity_curve.index[-1].strftime('%Y-%m-%d'),
            'duration_days': (equity_curve.index[-1] - equity_curve.index[0]).days,
            'starting_equity': start_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_percent': total_return * 100,
            'annual_return': annual_return,
            'annual_return_percent': annual_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'win_rate_percent': win_rate * 100,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_trade': avg_trade,
            'avg_duration_minutes': avg_duration,
        }
    
    def save_results(self, output_dir: str) -> Dict[str, str]:
        """
        Save backtest results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary with saved file paths
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trades to CSV
        trades_file = output_path / f"trades_{timestamp}.csv"
        trades_data = [trade.to_dict() for trade in self.trades]
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(trades_file, index=False)
        
        # Save equity curve to CSV
        equity_file = output_path / f"equity_{timestamp}.csv"
        self.equity_curve.to_csv(equity_file)
        
        # Save metrics to JSON
        metrics_file = output_path / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Generate plots
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Equity curve plot
        equity_plot_file = plots_dir / f"equity_curve_{timestamp}.png"
        self._plot_equity_curve(equity_plot_file)
        
        # Drawdown plot
        drawdown_plot_file = plots_dir / f"drawdown_{timestamp}.png"
        self._plot_drawdown(drawdown_plot_file)
        
        # Generate HTML report
        report_file = output_path / f"report_{timestamp}.html"
        self._generate_html_report(report_file)
        
        logger.info(f"Backtest results saved to {output_path}")
        
        return {
            'trades': str(trades_file),
            'equity': str(equity_file),
            'metrics': str(metrics_file),
            'equity_plot': str(equity_plot_file),
            'drawdown_plot': str(drawdown_plot_file),
            'report': str(report_file)
        }
    
    def _plot_equity_curve(self, filename: str) -> None:
        """
        Generate equity curve plot.
        
        Args:
            filename: File to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def _plot_drawdown(self, filename: str) -> None:
        """
        Generate drawdown plot.
        
        Args:
            filename: File to save the plot
        """
        rolling_max = self.equity_curve['equity'].cummax()
        drawdown = (self.equity_curve['equity'] - rolling_max) / rolling_max
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, drawdown * 100)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown %')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def _generate_html_report(self, filename: str) -> None:
        """
        Generate HTML backtest report.
        
        Args:
            filename: File to save the report
        """
        # Format metrics for display
        metrics_html = ""
        for key, value in self.metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            
            metrics_html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"
        
        # Format trades for display
        trades_html = ""
        for i, trade in enumerate(self.trades[:100]):  # Limit to first 100 trades
            trade_dict = trade.to_dict()
            
            trades_html += "<tr>"
            trades_html += f"<td>{i+1}</td>"
            trades_html += f"<td>{trade_dict['symbol']}</td>"
            trades_html += f"<td>{trade_dict['entry_time']}</td>"
            trades_html += f"<td>{trade_dict['entry_price']:.2f}</td>"
            
            if not trade.is_open:
                trades_html += f"<td>{trade_dict['exit_time']}</td>"
                trades_html += f"<td>{trade_dict['exit_price']:.2f}</td>"
                trades_html += f"<td>{trade_dict['exit_reason']}</td>"
                
                # Color code PnL
                pnl_class = "positive" if trade.pnl > 0 else "negative"
                trades_html += f"<td class='{pnl_class}'>{trade_dict['pnl']:.2f}</td>"
                trades_html += f"<td class='{pnl_class}'>{trade_dict['pnl_percent']:.2f}%</td>"
            else:
                trades_html += "<td colspan='5'>Trade still open</td>"
            
            trades_html += "</tr>"
        
        # Create HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ZT-3 Trading System Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2a3f5f; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .report-section {{ margin-bottom: 30px; }}
                .images {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                .images img {{ width: 48%; }}
            </style>
        </head>
        <body>
            <div class="report-section">
                <h1>ZT-3 Trading System Backtest Report</h1>
                <p>Strategy: {self.strategy_name}</p>
                <p>Period: {self.metrics['start_date']} to {self.metrics['end_date']} ({self.metrics['duration_days']} days)</p>
                <p>Symbols: {', '.join(self.symbols)}</p>
            </div>
            
            <div class="report-section">
                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {metrics_html}
                </table>
            </div>
            
            <div class="report-section">
                <h2>Performance Charts</h2>
                <div class="images">
                    <img src="plots/equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" alt="Equity Curve">
                    <img src="plots/drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" alt="Drawdown">
                </div>
            </div>
            
            <div class="report-section">
                <h2>Trade List</h2>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Symbol</th>
                        <th>Entry Time</th>
                        <th>Entry Price</th>
                        <th>Exit Time</th>
                        <th>Exit Price</th>
                        <th>Exit Reason</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                    </tr>
                    {trades_html}
                </table>
                {f"<p>Showing first 100 out of {len(self.trades)} trades</p>" if len(self.trades) > 100 else ""}
            </div>
            
            <div class="report-section">
                <h2>Strategy Parameters</h2>
                <pre>{json.dumps(self.strategy_params, indent=4)}</pre>
            </div>
            
            <footer>
                <p>Generated by ZT-3 Backtesting Module on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(filename, 'w') as f:
            f.write(html)