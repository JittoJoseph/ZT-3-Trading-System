"""
Logging configuration for ZT-3 Trading System.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging for the ZT-3 Trading System.
    
    Args:
        config: Logging configuration dictionary
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Set log level from config
    log_level_str = config.get('level', 'INFO')
    log_level = getattr(logging, log_level_str)
    
    # Get log file paths
    log_file = config.get('file', 'logs/zt3.log')
    trade_log = config.get('trade_log', 'logs/trades.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    main_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(main_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Create file handler for main log file with rotation
    max_size_mb = config.get('max_size_mb', 10)
    backup_count = config.get('backup_count', 5)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count
    )
    file_handler.setFormatter(main_formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    # Create special file handler for trade logs
    trade_handler = RotatingFileHandler(
        trade_log,
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count
    )
    trade_handler.setFormatter(logging.Formatter('%(asctime)s,%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    
    # Create trade logger
    trade_logger = logging.getLogger('trades')
    trade_logger.setLevel(log_level)
    trade_logger.addHandler(trade_handler)
    trade_logger.propagate = False  # Don't propagate to root logger
    
    # Log setup completion
    root_logger.info("Logging initialized at %s level", log_level_str)
    root_logger.info("Main log file: %s", log_file)
    root_logger.info("Trade log file: %s", trade_log)

def log_trade(action: str, symbol: str, price: float, quantity: int, 
             reason: str = None, pnl: float = None) -> None:
    """
    Log a trade to the dedicated trade log file.
    
    Args:
        action: Trade action (BUY, SELL)
        symbol: Trading symbol
        price: Trade price
        quantity: Number of shares
        reason: Optional reason for the trade
        pnl: Optional profit/loss for the trade
    """
    trade_logger = logging.getLogger('trades')
    
    # Format trade message in pipe-separated format for easy parsing
    msg_parts = [action, symbol, f"{price:.2f}", str(quantity)]
    
    if reason:
        msg_parts.append(reason)
    else:
        msg_parts.append("N/A")
        
    if pnl is not None:
        msg_parts.append(f"{pnl:.2f}")
    else:
        msg_parts.append("N/A")
        
    trade_logger.info('|'.join(msg_parts))

def get_trade_logger():
    """
    Get the dedicated trade logger.
    
    Returns:
        Logger instance for trade logging
    """
    return logging.getLogger('trades')