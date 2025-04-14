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
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import internal modules
from config.loader import ConfigLoader
from interface.cli import CLI
from utils.logger import setup_logging

def main():
    """Main entry point for the application."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info("Starting ZT-3 Trading System...")
        config = ConfigLoader().load_config()
        
        # Initialize CLI
        cli = CLI(config)
        
        # Run CLI
        cli.run()
        
    except Exception as e:
        logger.error(f"Error in main application: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())