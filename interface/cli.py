"""
Command Line Interface for ZT-3 Trading System.

This module provides a CLI for interacting with the trading system,
allowing users to start/stop trading, check status, and more.
"""

import os
import sys
import click
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from utils.logger import setup_logging, TradeLogger
from utils.notifications import DiscordNotifier, NotificationColor

logger = logging.getLogger(__name__)

class TradingCLI:
    """
    Command Line Interface for the ZT-3 Trading System.
    
    This class provides methods for handling CLI commands
    and interacting with the trading system.
    """
    
    def __init__(self):
        """Initialize the CLI with default settings."""
        self.config = None
        self.config_path = None
        self.trading_active = False
        self.paper_trading = True
        self.notifier = None
        self.symbols = []
        
        # Components (will be initialized when starting)
        self.broker = None
        self.market_data = None
        self.strategy = None
    
    def load_config(self, config_path: str) -> bool:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            True if configuration was loaded successfully, False otherwise
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.error(f"Configuration file not found: {config_path}")
                return False
            
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.config_path = config_path
            logger.info(f"Loaded configuration from {config_path}")
            
            # Extract symbols from config
            self.symbols = []
            for symbol_config in self.config.get('symbols', []):
                if 'ticker' in symbol_config and 'exchange' in symbol_config:
                    self.symbols.append(f"{symbol_config['exchange']}:{symbol_config['ticker']}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def setup_components(self) -> bool:
        """
        Set up all required components for trading.
        
        Returns:
            True if all components were initialized successfully, False otherwise
        """
        try:
            # Initialize Discord notifier
            if 'notifications' in self.config:
                self.notifier = DiscordNotifier(self.config['notifications'])
                
                # Send status notification
                if self.notifier:
                    mode = "Paper Trading" if self.paper_trading else "Live Trading"
                    symbols_str = ', '.join(self.symbols)
                    self.notifier.send_status_update(
                        status=f"SYSTEM STARTUP - {mode}",
                        details=f"Monitoring {len(self.symbols)} symbols: {symbols_str}",
                        color=NotificationColor.INFO
                    )
            
            # TODO: Initialize broker, market data, and strategy components
            # For now, just log that we would initialize them
            logger.info("Would initialize broker connection here")
            logger.info("Would initialize market data client here")
            logger.info("Would initialize strategy here")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            return False
    
    def start_trading(self, paper_trading: bool = True) -> bool:
        """
        Start the trading system.
        
        Args:
            paper_trading: Whether to use paper trading or live trading
            
        Returns:
            True if trading was started successfully, False otherwise
        """
        if self.trading_active:
            logger.warning("Trading is already active")
            return False
        
        if not self.config:
            logger.error("No configuration loaded")
            return False
        
        try:
            self.paper_trading = paper_trading
            mode = "paper trading" if paper_trading else "LIVE TRADING"
            logger.info(f"Starting {mode} with {len(self.symbols)} symbols")
            
            # Set up components
            if not self.setup_components():
                logger.error("Failed to set up components")
                return False
            
            # Mark trading as active
            self.trading_active = True
            
            # Log success
            logger.info(f"{mode.upper()} started successfully")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
            return False
    
    def stop_trading(self) -> bool:
        """
        Stop the trading system.
        
        Returns:
            True if trading was stopped successfully, False otherwise
        """
        if not self.trading_active:
            logger.warning("Trading is not active")
            return False
        
        try:
            logger.info("Stopping trading...")
            
            # TODO: Properly clean up components
            # For now, just log that we would clean up
            logger.info("Would clean up broker connection here")
            logger.info("Would clean up market data connection here")
            
            # Mark trading as inactive
            self.trading_active = False
            
            # Send status notification
            if self.notifier:
                mode = "Paper Trading" if self.paper_trading else "Live Trading"
                self.notifier.send_status_update(
                    status=f"SYSTEM SHUTDOWN - {mode}",
                    details="Trading system has been stopped.",
                    color=NotificationColor.WARNING
                )
            
            # Log success
            logger.info("Trading stopped successfully")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to stop trading: {e}")
            return False
    
    def display_status(self) -> Dict[str, Any]:
        """
        Get the current status of the trading system.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'active': self.trading_active,
            'mode': "Paper Trading" if self.paper_trading else "Live Trading",
            'symbols': self.symbols,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # TODO: Add more detailed status information when components are implemented
        
        return status
    
    def validate_config(self, config_path: str) -> bool:
        """
        Validate a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.error(f"Configuration file not found: {config_path}")
                return False
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['api', 'symbols', 'strategy']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required section in config: {section}")
                    return False
            
            # Validate API section
            api_config = config.get('api', {})
            if 'api_key' not in api_config or 'api_secret' not in api_config:
                logger.error("API section must contain api_key and api_secret")
                return False
            
            # Validate symbols section
            symbols = config.get('symbols', [])
            if not symbols:
                logger.error("No symbols defined in configuration")
                return False
            
            for symbol in symbols:
                if 'ticker' not in symbol or 'exchange' not in symbol:
                    logger.error("Each symbol must have ticker and exchange")
                    return False
            
            # Validate strategy section
            strategy = config.get('strategy', {})
            if 'name' not in strategy or 'params' not in strategy:
                logger.error("Strategy section must contain name and params")
                return False
            
            # All checks passed
            logger.info(f"Configuration is valid: {config_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to validate configuration: {e}")
            return False


# Click CLI setup
@click.group()
@click.option('--config', '-c', default='config/default_config.yaml', help='Configuration file path')
@click.option('--log-level', '-l', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
@click.pass_context
def cli(ctx, config, log_level):
    """ZT-3 Trading System Command Line Interface."""
    # Set up logging
    setup_logging({
        'level': log_level,
        'file': 'logs/zt3.log'
    })
    
    # Initialize CLI instance
    ctx.obj = TradingCLI()
    
    # Load configuration
    if not ctx.obj.load_config(config):
        click.echo(f"Error loading configuration from {config}")
        ctx.exit(1)


@cli.command()
@click.option('--paper', is_flag=True, default=True, help='Use paper trading (default)')
@click.option('--live', is_flag=True, default=False, help='Use live trading (CAUTION: Uses real money)')
@click.pass_obj
def start(cli_obj, paper, live):
    """Start the trading system."""
    # If live flag is set, use live trading, otherwise use paper trading
    paper_trading = not live
    
    # Double-check for live trading
    if live:
        if not click.confirm('⚠️  WARNING: You are about to start LIVE TRADING using REAL MONEY. Are you sure?'):
            click.echo("Live trading cancelled.")
            return
    
    mode = "paper trading" if paper_trading else "LIVE TRADING"
    click.echo(f"Starting {mode}...")
    
    if cli_obj.start_trading(paper_trading):
        click.echo(f"✅ {mode.upper()} started successfully")
    else:
        click.echo(f"❌ Failed to start {mode}")


@cli.command()
@click.pass_obj
def stop(cli_obj):
    """Stop the trading system."""
    click.echo("Stopping trading...")
    
    if cli_obj.stop_trading():
        click.echo("✅ Trading stopped successfully")
    else:
        click.echo("❌ Failed to stop trading")


@cli.command()
@click.pass_obj
def status(cli_obj):
    """Display the current status of the trading system."""
    status_info = cli_obj.display_status()
    
    click.echo("=== ZT-3 Trading System Status ===")
    click.echo(f"Status: {'Active' if status_info['active'] else 'Inactive'}")
    click.echo(f"Mode: {status_info['mode']}")
    click.echo(f"Symbols: {', '.join(status_info['symbols'])}")
    click.echo(f"Time: {status_info['time']}")
    
    # TODO: Add more detailed status information when components are implemented


@cli.command()
@click.argument('config-path')
def validate(config_path):
    """Validate a configuration file."""
    cli_obj = TradingCLI()
    
    if cli_obj.validate_config(config_path):
        click.echo(f"✅ Configuration is valid: {config_path}")
    else:
        click.echo(f"❌ Configuration has errors: {config_path}")


if __name__ == '__main__':
    # This allows the CLI to be run directly (python -m interface.cli)
    cli()