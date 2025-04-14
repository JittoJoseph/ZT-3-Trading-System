#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ZT-3 Trading System
===================

An algorithmic trading system implementing the Gaussian Channel Strategy
for trading Indian equities using Upstox API.

This is the main entry point to the application.
"""

import sys
import os
import logging
import signal
import time
from pathlib import Path
import argparse
from typing import Dict, Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import internal modules
from config.loader import ConfigLoader
from utils.logger import setup_logging
from utils.notifications import NotificationManager
from data.market_data import MarketDataClient
from data.candle_aggregator import CandleAggregator
from broker.upstox import UpstoxBroker
from broker.position_manager import PositionManager
from broker.risk_manager import RiskManager
from strategy.gaussian_channel import GaussianChannelStrategy
from paper_trading.paper_trader import PaperTrader
from interface.cli import TradingCLI

# Global flags
running = True
trading_system = None

def signal_handler(sig, frame):
    """Handle interrupt signals."""
    global running
    logging.info("Shutdown signal received, stopping system...")
    running = False
    if trading_system:
        trading_system.stop()
    

class TradingSystem:
    """
    Main trading system class integrating all components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.paper_trading = config.get('paper_trading', {}).get('enabled', True)
        
        # Initialize components
        self.logger.info("Initializing trading system components...")
        self.notification_manager = NotificationManager(config)
        
        # Initialize broker
        self.broker = UpstoxBroker(config)
        
        # Initialize position and risk managers
        self.position_manager = PositionManager(config)
        self.risk_manager = RiskManager(config, self.position_manager)
        
        # Initialize market data handling
        self.market_data = MarketDataClient(self.broker, config)
        
        # Get symbols from config
        self.symbols = [
            f"{s.get('exchange')}:{s.get('ticker')}" 
            for s in config.get('symbols', [])
        ]
        
        # Initialize candle aggregator
        self.candle_aggregator = CandleAggregator(self.symbols, ['5min'])
        
        # Initialize strategy
        self.strategy = GaussianChannelStrategy(config)
        
        # Initialize paper trader or connect to live trading
        if self.paper_trading:
            self.paper_trader = PaperTrader(config)
            self.logger.info("Paper trading mode enabled")
            
            # Send notification
            self.notification_manager.send_system_notification(
                "System Starting", 
                "ZT-3 Trading System starting in PAPER TRADING mode",
                "info"
            )
        else:
            # Live trading setup would go here
            self.logger.info("Live trading mode enabled")
            
            # Send notification
            self.notification_manager.send_system_notification(
                "System Starting", 
                "ZT-3 Trading System starting in LIVE TRADING mode",
                "warning"
            )
        
        # Register callbacks
        self._setup_callbacks()
        
        self.logger.info("Trading system initialized")
    
    def _setup_callbacks(self):
        """Set up all callbacks between components."""
        # Register candle completion callback
        for symbol in self.symbols:
            self.candle_aggregator.register_candle_callback(
                symbol, 
                '5min', 
                self._on_candle_completed
            )
        
        # Register market data callback
        self.market_data.register_market_data_callback(self._on_market_data)
    
    def _on_market_data(self, market_data):
        """
        Handle incoming market data.
        
        Args:
            market_data: Market data from WebSocket
        """
        # Process market data through candle aggregator
        try:
            # Extract symbol and tick data
            # Note: This would need to be adjusted based on actual Upstox data format
            symbol = market_data.get('symbol')
            price = market_data.get('price')
            volume = market_data.get('volume', 0)
            timestamp = market_data.get('timestamp')
            
            if symbol and price:
                tick = {
                    'price': price,
                    'volume': volume,
                    'timestamp': timestamp
                }
                self.candle_aggregator.add_tick(symbol, tick)
                
                # Update paper trader with latest price
                if self.paper_trading:
                    self.paper_trader.update_price(symbol, {
                        'close': price,
                        'high': price,
                        'low': price,
                        'volume': volume
                    })
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}", exc_info=True)
            self.notification_manager.send_error_notification(
                "Market Data Processing Error",
                str(e)
            )
    
    def _on_candle_completed(self, symbol, candle):
        """
        Handle completed candle.
        
        Args:
            symbol: Symbol for the candle
            candle: The completed candle data
        """
        try:
            # Convert to pandas DataFrame
            import pandas as pd
            df = pd.DataFrame([candle])
            df.set_index('timestamp', inplace=True)
            
            # Get historical candles to have enough data for the strategy
            historical_df = self.candle_aggregator.to_dataframe(symbol, '5min')
            if len(historical_df) < self.config.get('strategy', {}).get('params', {}).get('gc_period', 144) + 20:
                self.logger.debug(f"Not enough historical data for {symbol} yet")
                return
                
            # Process candle with strategy
            signal = self.strategy.process_candle(historical_df, symbol)
            
            if signal:
                self.logger.info(f"Signal generated: {signal}")
                
                # Send notification
                self.notification_manager.send_signal_notification(signal)
                
                # Execute trade
                if self.paper_trading:
                    trade_result = self.paper_trader.process_signal(signal)
                    
                    if trade_result.get('success', False):
                        self.logger.info(f"Paper trade executed: {trade_result}")
                        self.notification_manager.send_trade_execution_notification(trade_result)
                    else:
                        self.logger.warning(f"Paper trade failed: {trade_result}")
                else:
                    # Live trading would go here
                    pass
        except Exception as e:
            self.logger.error(f"Error processing candle: {e}", exc_info=True)
            self.notification_manager.send_error_notification(
                "Candle Processing Error",
                str(e)
            )
    
    def start(self):
        """Start the trading system."""
        self.logger.info("Starting trading system...")
        
        # Connect to Upstox if needed
        if not self.broker.is_authenticated():
            self.logger.info("Authentication required, starting authentication flow...")
            self.broker.open_auth_url()
            # In a real scenario, we'd have a way to get the auth code from the user
            
        # Connect to market data
        self.market_data.connect_websocket()
        
        # Subscribe to market data for symbols
        symbols_list = [
            {"ticker": s.split(':')[1], "exchange": s.split(':')[0]} 
            for s in self.symbols
        ]
        self.market_data.subscribe_market_data(symbols_list)
        
        self.logger.info("Trading system started")
    
    def stop(self):
        """Stop the trading system."""
        self.logger.info("Stopping trading system...")
        
        # Close all positions in paper trader
        if self.paper_trading:
            close_results = self.paper_trader.close_all_positions("SYSTEM_SHUTDOWN")
            self.logger.info(f"Closed paper trading positions: {close_results}")
            
            # Get performance metrics
            metrics = self.paper_trader.get_performance_metrics()
            self.logger.info(f"Paper trading performance: {metrics}")
            
        # Disconnect from market data
        self.market_data.disconnect_websocket()
        
        # Send notification
        self.notification_manager.send_system_notification(
            "System Shutdown", 
            "ZT-3 Trading System has been stopped",
            "info"
        )
        
        self.logger.info("Trading system stopped")


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ZT-3 Trading System')
    parser.add_argument('--config', '-c', default='default_config.yaml', 
                        help='Configuration file name within config directory')
    parser.add_argument('--paper', '-p', action='store_true', default=True,
                        help='Use paper trading mode (default)')
    parser.add_argument('--live', '-l', action='store_false', dest='paper',
                        help='Use live trading mode (real money)')
    parser.add_argument('--log-level', '-d', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    args = parser.parse_args()
    
    # Set up logging
    setup_logging({
        'level': args.log_level,
        'file': 'logs/zt3.log'
    })
    logger = logging.getLogger(__name__)
    
    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info(f"Starting ZT-3 Trading System (Config: {args.config})")
        
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(args.config)
        if not config:
            logger.error(f"Failed to load configuration from {args.config}")
            return 1
            
        # Force paper trading mode based on command line
        if 'paper_trading' not in config:
            config['paper_trading'] = {}
        config['paper_trading']['enabled'] = args.paper
        
        # Initialize trading system
        global trading_system
        trading_system = TradingSystem(config)
        
        # Start trading system
        trading_system.start()
        
        # Run until interrupted
        while running:
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in main application: {e}", exc_info=True)
        return 1
        
    finally:
        # Ensure clean shutdown
        if trading_system:
            trading_system.stop()
            
    return 0

if __name__ == "__main__":
    sys.exit(main())