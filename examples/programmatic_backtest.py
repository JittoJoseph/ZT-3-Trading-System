#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example: Programmatic Backtesting
================================

This example demonstrates how to use the ZT-3 backtester
programmatically to run a backtest using the configuration file.
Strategy parameters are now hardcoded in the strategy class itself.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
import matplotlib.pyplot as plt
import logging

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parents[1]))

# Import for loading environment variables
from dotenv import load_dotenv

# Import internal modules
from backtest.backtester import Backtester
from config.loader import ConfigLoader
from utils.logger import setup_logging
# Import the strategy class to check its type if needed, but parameters are internal now
from strategy.swing_pro import SwingProStrategy

# Load environment variables
load_dotenv()

def run_single_backtest_example():
    """
    Run a single backtest using the default configuration
    and the hardcoded strategy parameters.
    """
    # Set up logging
    setup_logging({
        'level': 'INFO',
        'file': 'logs/programmatic_backtest.log'
    })
    logger = logging.getLogger(__name__)

    # Load base configuration
    config_loader = ConfigLoader()
    base_config = config_loader.load_config('default_config.yaml')
    if not base_config:
        logger.error("Failed to load default_config.yaml")
        return

    # Use reasonable date range for backtesting (e.g., 2 years for 1-hour)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)

    # Convert to string format
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    print(f"Running backtest for strategy '{base_config.get('strategy', {}).get('name', 'N/A')}'")
    print(f"Date range: {start_date_str} to {end_date_str}")

    # Initialize backtester
    try:
        backtester = Backtester(base_config)
    except Exception as e:
        logger.error(f"Failed to initialize backtester: {e}")
        print(f"Error initializing backtester: {e}")
        return

    # Get symbols from config
    symbols = [f"{s['exchange']}:{s['ticker']}" for s in base_config.get('symbols', [])]
    if not symbols:
        logger.error("No symbols defined in the configuration.")
        print("Error: No symbols defined in config.")
        return

    print(f"Symbols: {', '.join(symbols)}")

    # Display hardcoded strategy parameters (optional)
    if isinstance(backtester.strategy, SwingProStrategy):
        print("\nStrategy Parameters (Hardcoded):")
        print(f"  ATR SL Mult: {backtester.strategy.risk_mult_atr_sl}")
        print(f"  ATR TP1 Mult: {backtester.strategy.risk_mult_atr_tp1}")
        print(f"  ATR TP2 Mult: {backtester.strategy.risk_mult_atr_tp2}")
        print(f"  Trail EMA Len: {backtester.strategy.trail_ema_len}")
        print(f"  Use MTF Trend: {backtester.strategy.use_mtf_trend}")
        print(f"  Daily EMA Len: {backtester.strategy.daily_ema_len}")
        # Add other relevant params if needed
    else:
        print(f"\nLoaded strategy: {backtester.strategy_name}")


    # Load data for each symbol
    all_data_loaded = True
    for symbol in symbols:
        try:
            print(f"Loading data for {symbol}...")
            backtester.load_data(
                symbol=symbol,
                start_date=start_date_str,
                end_date=end_date_str,
                source='api',  # Use API to get real data
            )
            print(f"Data loaded successfully for {symbol}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            print(f"Error loading data for {symbol}: {e}")
            all_data_loaded = False
            # Decide whether to continue or stop if one symbol fails
            # break # Stop if any symbol fails to load

    if not all_data_loaded and not backtester.data:
         print("Failed to load data for any symbols. Aborting backtest.")
         return

    if not backtester.data:
        print("No data loaded for any symbols. Aborting backtest.")
        return

    # Run backtest
    print("\nRunning backtest...")
    try:
        metrics = backtester.run_backtest(start_date=start_date_str, end_date=end_date_str)
    except Exception as e:
        logger.error(f"Error during backtest execution: {e}", exc_info=True)
        print(f"Error running backtest: {e}")
        return

    # Format metrics for display
    print(f"\nBacktest Results:")
    if metrics:
        print(f"  Total Return: {metrics.get('total_return_percent', 'N/A'):.2f}%")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.2f}")
        print(f"  Win Rate: {metrics.get('win_rate_percent', 'N/A'):.2f}%")
        print(f"  Profit Factor: {metrics.get('profit_factor', 'N/A'):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown_percent', 'N/A'):.2f}%")
        print(f"  Total Trades: {metrics.get('total_trades', 'N/A')}")

        # Save results
        results_dir = Path('results/programmatic_backtest')
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics to JSON
        metrics_file = results_dir / f"metrics_{timestamp}.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"\nMetrics saved to: {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            print(f"Error saving metrics: {e}")

        # Save full results (including plots, trades CSV etc.) using backtester method
        try:
            saved_files = backtester.save_results(str(results_dir))
            print(f"Full report saved to: {saved_files.get('report', 'N/A')}")
        except Exception as e:
            logger.error(f"Failed to save full results: {e}")
            print(f"Error saving full results: {e}")

    else:
        print("Backtest did not produce results (e.g., no trades executed).")


def main():
    """Main entry point for the script."""
    run_single_backtest_example()

if __name__ == "__main__":
    main()