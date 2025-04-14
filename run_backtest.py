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
    
    # Date range
    parser.add_argument(
        '--start-date', '-s',
        default=None,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date', '-e',
        default=None,
        help='End date for backtest (YYYY-MM-DD)'
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
        help='Logging level'
    )
    
    # Strategy parameters override
    parser.add_argument(
        '--params', '-p',
        default=None,
        help='JSON string with strategy parameters to override config'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for backtesting."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging({
        'level': args.log_level,
        'file': 'logs/backtest.log'
    })
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting ZT-3 Backtesting (Config: {args.config})")
        
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(args.config)
        if not config:
            logger.error(f"Failed to load configuration from {args.config}")
            return 1
        
        # Override strategy parameters if provided
        if args.params:
            try:
                params = json.loads(args.params)
                if not isinstance(params, dict):
                    raise ValueError("Parameters must be provided as a JSON object")
                config['strategy']['params'].update(params)
                logger.info(f"Strategy parameters overridden: {params}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in parameters: {e}")
                return 1
            except Exception as e:
                logger.error(f"Failed to apply parameters: {e}")
                return 1
        
        # Initialize backtester
        backtester = Backtester(config)
        
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
        
        for symbol in backtester.symbols:
            csv_path = csv_paths.get(symbol) if args.data_source == 'csv' else None
            try:
                backtester.load_data(
                    symbol=symbol,
                    start_date=args.start_date or '2020-01-01',
                    end_date=args.end_date or datetime.now().strftime('%Y-%m-%d'),
                    source=args.data_source,
                    interval=config.get('strategy', {}).get('params', {}).get('interval', '5min'),
                    csv_path=csv_path
                )
                logger.info(f"Loaded data for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
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
        try:
            # Setup notifications with environment variables
            notifications_config = {
                'notifications': {
                    'discord': {
                        'enabled': True,
                        'webhooks': {
                            'backtest_results': os.environ.get('DISCORD_WEBHOOK_BACKTEST')
                        }
                    },
                    'notification_levels': {
                        'backtest_results': True
                    }
                }
            }
            
            # Initialize NotificationManager with config
            notifier = NotificationManager(notifications_config)
            
            # Add strategy name and symbol to metrics for the notification
            metrics['strategy_name'] = config.get('strategy', {}).get('name', 'ZT-3 Strategy')
            metrics['symbol'] = ', '.join(backtester.symbols)
            
            # Send notification
            if notifier.send_backtest_results(metrics):
                logger.info("Backtest results sent to Discord successfully")
                print("Backtest results sent to Discord webhook")
            else:
                logger.warning("Failed to send backtest results to Discord")
        except Exception as e:
            logger.warning(f"Error sending Discord notification: {e}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user.")
        return 0
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}", exc_info=True)
        return 1
        
if __name__ == "__main__":
    sys.exit(main())