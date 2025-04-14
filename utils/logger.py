"""
Logging setup for ZT-3 Trading System.

This module configures the Python logging system with appropriate
formatting and handlers for different log types.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union, Optional

def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure logging with console and file handlers.
    
    Args:
        config: Optional configuration dictionary with logging settings.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get logging configuration from config or use defaults
    log_level_name = "INFO"
    log_file = "logs/zt3.log"
    max_size_mb = 10
    backup_count = 5
    trade_log = "logs/trades.log"
    
    if config and 'logging' in config:
        log_config = config['logging']
        log_level_name = log_config.get('level', log_level_name)
        log_file = log_config.get('file', log_file)
        max_size_mb = log_config.get('max_size_mb', max_size_mb)
        backup_count = log_config.get('backup_count', backup_count)
        trade_log = log_config.get('trade_log', trade_log)
    
    # Map string log level to logging constants
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = log_level_map.get(log_level_name.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Create rotating file handler for main log
    main_file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_size_mb * 1024 * 1024,  # Convert MB to bytes
        backupCount=backup_count
    )
    main_file_handler.setFormatter(file_formatter)
    main_file_handler.setLevel(log_level)
    root_logger.addHandler(main_file_handler)
    
    # Create trade logger (for trade-specific logging)
    trade_logger = logging.getLogger('trades')
    trade_logger.propagate = False  # Don't propagate to parent logger
    trade_logger.setLevel(logging.INFO)
    
    # Create trade log formatter
    trade_formatter = logging.Formatter('%(asctime)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create trade log handler
    trade_file_handler = logging.handlers.RotatingFileHandler(
        trade_log,
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count
    )
    trade_file_handler.setFormatter(trade_formatter)
    trade_logger.addHandler(trade_file_handler)
    
    # Log startup message
    logging.info(f"Logging initialized at {log_level_name} level")
    logging.info(f"Main log file: {log_file}")
    logging.info(f"Trade log file: {trade_log}")

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