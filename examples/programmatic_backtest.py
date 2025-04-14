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
from datetime import datetime
import itertools
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parents[1]))

# Import internal modules
from backtest.backtester import Backtester
from config.loader import ConfigLoader
from utils.logger import setup_logging

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
    
    # Parameters to test (modify these according to your requirements)
    gc_periods = [112, 144, 176]
    gc_multipliers = [1.0, 1.2, 1.4]
    stoch_upper_bands = [70.0, 80.0, 90.0]
    atr_tp_multipliers = [3.0, 4.0, 5.0]
    
    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        gc_periods, gc_multipliers, stoch_upper_bands, atr_tp_multipliers
    ))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    # Results storage
    results = []
    
    # Run backtest for each parameter combination
    for i, (gc_period, gc_multiplier, stoch_upper_band, atr_tp_multiplier) in enumerate(param_combinations):
        print(f"\nRunning test {i+1}/{len(param_combinations)}")
        print(f"Parameters: GC Period={gc_period}, GC Mult={gc_multiplier}, Stoch Band={stoch_upper_band}, ATR TP={atr_tp_multiplier}")
        
        # Create config for this test
        config = base_config.copy()
        config['strategy']['params'].update({
            'gc_period': gc_period,
            'gc_multiplier': gc_multiplier,
            'stoch_upper_band': stoch_upper_band,
            'atr_tp_multiplier': atr_tp_multiplier
        })
        
        # Initialize backtester
        backtester = Backtester(config)
        
        # Load data
        try:
            for symbol in backtester.symbols:
                backtester.load_data(
                    symbol=symbol,
                    start_date='2023-01-01',
                    end_date='2023-12-31',
                    source='api'
                )
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
            
        # Run backtest
        metrics = backtester.run_backtest()
        
        # Store results
        results.append({
            'gc_period': gc_period,
            'gc_multiplier': gc_multiplier,
            'stoch_upper_band': stoch_upper_band,
            'atr_tp_multiplier': atr_tp_multiplier,
            'total_return': metrics['total_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'max_drawdown': metrics['max_drawdown'],
            'total_trades': metrics['total_trades']
        })
        
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Sort by various metrics
    print("\nTop 5 by Total Return:")
    print(results_df.sort_values('total_return', ascending=False).head(5))
    
    print("\nTop 5 by Sharpe Ratio:")
    print(results_df.sort_values('sharpe_ratio', ascending=False).head(5))
    
    print("\nTop 5 by Profit Factor:")
    print(results_df.sort_values('profit_factor', ascending=False).head(5))
    
    # Find best overall parameters (simple scoring)
    results_df['score'] = (
        results_df['sharpe_ratio'] / results_df['sharpe_ratio'].max() +
        results_df['total_return'] / results_df['total_return'].max() +
        results_df['profit_factor'] / results_df['profit_factor'].max() -
        results_df['max_drawdown'] / results_df['max_drawdown'].max()
    )
    
    print("\nTop 5 Overall (Combined Score):")
    top_params = results_df.sort_values('score', ascending=False).head(5)
    print(top_params)
    
    # Save results
    results_dir = Path('results/parameter_optimization')
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(results_dir / f"param_opt_results_{timestamp}.csv", index=False)
    
    print(f"\nResults saved to: {results_dir}/param_opt_results_{timestamp}.csv")
    
    # Optionally run a final backtest with the best parameters
    best_params = top_params.iloc[0]
    print("\nRunning final backtest with best parameters:")
    print(f"GC Period: {best_params['gc_period']}")
    print(f"GC Multiplier: {best_params['gc_multiplier']}")
    print(f"Stoch Upper Band: {best_params['stoch_upper_band']}")
    print(f"ATR TP Multiplier: {best_params['atr_tp_multiplier']}")
    
    best_config = base_config.copy()
    best_config['strategy']['params'].update({
        'gc_period': best_params['gc_period'],
        'gc_multiplier': best_params['gc_multiplier'],
        'stoch_upper_band': best_params['stoch_upper_band'],
        'atr_tp_multiplier': best_params['atr_tp_multiplier']
    })
    
    # Initialize backtester with best parameters
    best_backtester = Backtester(best_config)
    
    # Load data (longer period for final backtest)
    for symbol in best_backtester.symbols:
        best_backtester.load_data(
            symbol=symbol,
            start_date='2022-01-01',
            end_date='2023-12-31',
            source='api'
        )
    
    # Run final backtest
    final_metrics = best_backtester.run_backtest()
    
    # Save final results
    best_backtester.save_results(results_dir / "best_parameters")
    
    print("\nFinal backtest completed and results saved.")

if __name__ == "__main__":
    run_parameter_optimization()