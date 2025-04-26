#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ZT-3 Trading System: Backtesting Script
======================================

This script allows you to run backtests for the ZT-3 Trading System
with different configurations and parameters.
"""

import sys
import os
import argparse
import logging
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import for loading environment variables
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import internal modules
from backtest.backtester import Backtester
from config.loader import ConfigLoader
from utils.logger import setup_logging
from utils.notifications import NotificationManager
# Import the new strategy class if needed for type hinting or direct use (though backtester loads it)
from strategy.swing_pro import SwingProStrategy

# Load environment variables from .env file
load_dotenv()

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='ZT-3 Trading System Backtester')
    
    # Configuration
    parser.add_argument(
        '--config', '-c', 
        default='default_config.yaml',
        help='Configuration file name within config directory'
    )
    
    # Date range - Adjust defaults for daily data (e.g., 2 years)
    parser.add_argument(
        '--start-date', '-s',
        default=(datetime.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d'),
        help='Start date for backtest (YYYY-MM-DD), default: 2 years ago'
    )
    
    parser.add_argument(
        '--end-date', '-e',
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date for backtest (YYYY-MM-DD), default: today'
    )
    
    # Data source
    parser.add_argument(
        '--data-source',
        choices=['api', 'csv'],
        default='api',
        help='Source of historical data ("api" or "csv")'
    )
    
    parser.add_argument(
        '--csv-path',
        default=None,
        help='Path to CSV file(s) when data source is "csv". Use symbol_name: path format for multiple files.'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='results',
        help='Output directory for backtest results'
    )
    
    # Logging
    parser.add_argument(
        '--log-level', '-d',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level for the log file and verbose console output' # Updated help text
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed trade-by-trade INFO/WARNING messages from the backtester on the console'
    )

    return parser.parse_args()

def main():
    """Main entry point for backtesting."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging (applies level to file handler primarily, console might be INFO by default)
    # Ensure setup_logging configures both a file handler (with args.log_level)
    # and a console handler (e.g., default INFO, but we'll override for backtester below)
    setup_logging({
        'level': args.log_level, # This level applies to the file handler
        'file': 'logs/backtest.log'
        # Assuming setup_logging also adds a console handler, possibly at INFO level by default
    })
    logger = logging.getLogger(__name__)

    # Adjust console logging level for backtester unless --verbose is used
    if not args.verbose:
        bt_logger = logging.getLogger('backtest.backtester')
        # Find the console handler (StreamHandler) and set its level higher (ERROR)
        # to suppress INFO and WARNING messages on the console by default.
        console_handler_found = False
        for handler in bt_logger.handlers:
            # Check if it's a console handler (StreamHandler) and not a FileHandler
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                logger.info(f"Setting console handler level for 'backtest.backtester' to ERROR (suppressing INFO/WARN). Use --verbose to see details.")
                handler.setLevel(logging.ERROR)
                console_handler_found = True
                break # Assume only one console handler per logger instance

        # If the specific logger didn't have a console handler (e.g., due to propagation),
        # find and adjust the root logger's console handler.
        if not console_handler_found:
             root_logger = logging.getLogger()
             for handler in root_logger.handlers:
                 if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                     logger.info(f"Setting root console handler level for 'backtest.backtester' messages to ERROR (suppressing INFO/WARN). Use --verbose to see details.")
                     # Note: This might affect other loggers if not handled carefully in setup_logging.
                     # A more robust solution uses Filters, but level setting is simpler here.
                     # We might need a filter specifically for backtest.backtester messages on this handler.
                     # For simplicity, let's try setting the level first.
                     handler.setLevel(logging.ERROR) # This might be too broad, consider filtering later if needed.
                     break


    try:
        logger.info(f"Starting ZT-3 Backtesting (Config: {args.config})")
        
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(args.config)
        if not config:
            logger.error(f"Failed to load configuration from {args.config}")
            return 1
        
        # Initialize backtester
        backtester = Backtester(config)
        
        # Ensure we have an access token
        if not backtester.access_token and args.data_source == 'api':
            logger.error("No Upstox API access token available. Please run utils/get_token.py first.")
            print("\nERROR: No Upstox API access token available.")
            print("Please run 'python utils/get_token.py' first to authenticate with Upstox.")
            
            if backtester.notification_manager:
                backtester.notification_manager.send_system_notification(
                    "Backtest Failed", 
                    "No Upstox API access token available. Please run utils/get_token.py first.", 
                    "error"
                )
            return 1
        
        # Load historical data for each symbol
        csv_paths = {}
        if args.csv_path:
            if ':' in args.csv_path:
                # Parse symbol:path format
                for pair in args.csv_path.split(','):
                    symbol, path = pair.strip().split(':', 1)
                    csv_paths[symbol] = path
            else:
                # Use same path for all symbols
                for symbol in backtester.symbols:
                    csv_paths[symbol] = args.csv_path
        
        logger.info(f"Using {args.data_source} as data source with 'day' interval")
        logger.info(f"Date range: {args.start_date} to {args.end_date}")
        
        # Setup notification for backtest start
        if backtester.notification_manager:
            try:
                backtester.notification_manager.send_system_notification(
                    "Backtest Started",
                    f"Starting backtest for {', '.join(backtester.symbols)} from {args.start_date} to {args.end_date}",
                    "info"
                )
            except Exception as e:
                logger.warning(f"Failed to send start notification: {e}")
        
        # Load data for each symbol
        for symbol in backtester.symbols:
            csv_path = csv_paths.get(symbol) if args.data_source == 'csv' else None
            try:
                backtester.load_data(
                    symbol=symbol,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    source=args.data_source,
                    csv_path=csv_path
                )
                logger.info(f"Loaded data for {symbol}")
            except ValueError as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
                print(f"\nERROR: {str(e)}")
                return 1
        
        # Run backtest
        metrics = backtester.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Print summary
        print("\n=== Backtest Results ===")
        print(f"Period: {metrics['start_date']} to {metrics['end_date']} ({metrics['duration_days']} days)")
        print(f"Starting Capital: {metrics['starting_equity']:.2f}")
        print(f"Final Equity: {metrics['final_equity']:.2f}")
        print(f"Total Return: {metrics['total_return_percent']:.2f}%")
        print(f"Annual Return: {metrics['annual_return_percent']:.2f}%")
        print(f"Max Drawdown: {metrics['max_drawdown_percent']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate_percent']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Save results
        output_dir = Path(args.output_dir)
        results = backtester.save_results(output_dir)
        
        print(f"\nBacktest results saved to: {output_dir}")
        print(f"Report: {results['report']}")
        
        # Send results to Discord if webhook is configured
        if backtester.notification_manager:
            try:
                # Add strategy name and symbol to metrics for the notification
                metrics['strategy_name'] = config.get('strategy', {}).get('name', 'ZT-3 Strategy')
                metrics['symbol'] = ', '.join(backtester.symbols)
                
                # Send notification
                if backtester.notification_manager.send_backtest_results(metrics):
                    logger.info("Backtest results sent to Discord successfully")
                    print("Backtest results sent to Discord webhook")
                else:
                    logger.warning("Failed to send backtest results to Discord")
            except Exception as e:
                logger.warning(f"Error sending Discord notification: {e}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user.")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}", exc_info=True)
        
        # Try to send error notification
        try:
            if 'backtester' in locals() and backtester.notification_manager:
                backtester.notification_manager.send_system_notification(
                    "Backtest Error",
                    f"An error occurred during backtesting: {str(e)}",
                    "error"
                )
        except Exception:
            pass  # Suppress notification errors
            
        print(f"\nERROR: {str(e)}")
        return 1
        
if __name__ == "__main__":
    sys.exit(main())