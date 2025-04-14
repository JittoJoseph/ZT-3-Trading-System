"""
Logging utility for ZT-3 Trading System.

This module handles setting up logging with appropriate handlers
for file and console output.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any

def setup_logging(config: Dict[str, Any] = None) -> None:
    """
    Setup logging for the application.
    
    Args:
        config: Configuration dictionary with logging settings.
               If not provided, default settings will be used.
    """
    if config is None:
        config = {
            'level': 'INFO',
            'file': 'logs/zt3.log',
            'max_size_mb': 10,
            'backup_count': 5
        }
    
    # Create logs directory if it doesn't exist
    log_file = Path(config.get('file', 'logs/zt3.log'))
    log_dir = log_file.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the log level
    level_name = config.get('level', 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Setup file handler
    max_size_bytes = config.get('max_size_mb', 10) * 1024 * 1024
    backup_count = config.get('backup_count', 5)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Log startup
    logging.info(f"Logging initialized at level {level_name}")