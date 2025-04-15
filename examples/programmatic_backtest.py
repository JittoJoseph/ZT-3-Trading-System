#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example: Programmatic Backtesting
================================

This example demonstrates how to use the ZT-3 backtester
programmatically to run multiple backtests with different
parameters, which is useful for parameter optimization.
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
from strategy.gaussian_channel import GaussianChannelStrategy

# Load environment variables
load_dotenv()

def run_parameter_optimization():
    """
    Run a parameter optimization by testing multiple combinations
    of strategy parameters.
    """
    # Set up logging
    setup_logging({
        'level': 'INFO',
        'file': 'logs/parameter_opt.log'
    })
    
    # Load base configuration
    config_loader = ConfigLoader()
    base_config = config_loader.load_config('default_config.yaml')
    
    # Use reasonable date range for backtesting
    # Default to past 6 months of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Convert to string format
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    print(f"Testing strategy on date range {start_date_str} to {end_date_str}")
    
    # Create a function to run a single backtest with specific parameters
    def run_single_backtest(symbols, backtest_id=1, total_tests=1):
        print(f"\nRunning test {backtest_id}/{total_tests}")
        
        # Initialize backtester
        backtester = Backtester(base_config)
        
        # Get the strategy instance directly
        strategy = backtester.strategy
        
        # Output the current parameter values
        print(f"Current Parameters:")
        print(f"GC Period: {strategy.gc_period}")
        print(f"GC Multiplier: {strategy.gc_multiplier}")
        print(f"Stoch Upper Band: {strategy.stoch_upper_band}")
        print(f"ATR TP Multiplier: {strategy.atr_tp_multiplier}")
        
        # Load data for each symbol
        for symbol in symbols:
            try:
                backtester.load_data(
                    symbol=symbol,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    source='api',  # Use API to get real data
                    interval='5minute'  # Upstox API format
                )
                print(f"Data loaded for {symbol}")
            except Exception as e:
                print(f"Error loading data for {symbol}: {e}")
                return None
        
        # Run backtest
        metrics = backtester.run_backtest()
        
        # Format metrics for display
        print(f"\nBacktest Results:")
        print(f"Total Return: {metrics['total_return_percent']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate_percent']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_percent']:.2f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        
        return metrics
    
    # Run a single backtest using the default parameters
    symbols = [f"{s['exchange']}:{s['ticker']}" for s in base_config.get('symbols', [])]
    
    results = run_single_backtest(symbols)
    
    if results:
        # Save results
        results_dir = Path('results/backtest')
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert metrics to DataFrame
        results_df = pd.DataFrame([results])
        results_df.to_csv(results_dir / f"backtest_results_{timestamp}.csv", index=False)
        
        print(f"\nResults saved to: {results_dir}/backtest_results_{timestamp}.csv")

def main():
    """Main entry point for the script."""
    run_parameter_optimization()

if __name__ == "__main__":
    main()