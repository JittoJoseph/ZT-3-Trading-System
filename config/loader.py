"""
Configuration loader for the ZT-3 Trading System.

This module handles loading and validation of all configuration parameters
from YAML files.
"""

import os
import logging
from pathlib import Path
import yaml
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Config loader class that handles loading and validation of configuration files.
    """
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent / 'default_config.yaml'
    USER_CONFIG_PATH = Path(__file__).parent / 'config.yaml'
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Optional path to configuration file.
                         If not provided, default paths will be used.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from files and validate it.
        
        Returns:
            Dict containing configuration parameters.
        
        Raises:
            FileNotFoundError: If no configuration file is found.
            ValueError: If configuration validation fails.
        """
        # First load default configuration
        if self.DEFAULT_CONFIG_PATH.exists():
            with open(self.DEFAULT_CONFIG_PATH, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded default configuration from {self.DEFAULT_CONFIG_PATH}")
        
        # Then load user configuration if it exists (overriding defaults)
        config_path = self.config_path or self.USER_CONFIG_PATH
        if config_path.exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self._deep_update(self.config, user_config)
            logger.info(f"Loaded user configuration from {config_path}")
        elif not self.config:
            raise FileNotFoundError(f"No configuration file found at {config_path}")
            
        # Validate the configuration
        self._validate_config()
        
        return self.config
    
    def _deep_update(self, original: Dict, update: Dict) -> None:
        """
        Recursively update a nested dictionary.
        
        Args:
            original: Original dictionary to update.
            update: Dictionary with values to update.
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def _validate_config(self) -> None:
        """
        Validate that all required configuration parameters are present.
        
        Raises:
            ValueError: If validation fails.
        """
        required_sections = ['api', 'symbols', 'strategy', 'risk', 'notifications']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate API configuration
        api_config = self.config['api']
        if 'api_key' not in api_config or 'api_secret' not in api_config:
            raise ValueError("API configuration must include 'api_key' and 'api_secret'")
        
        # Validate symbols configuration
        symbols = self.config['symbols']
        if not symbols or not isinstance(symbols, list):
            raise ValueError("At least one symbol must be configured")
        
        for symbol in symbols:
            if not isinstance(symbol, dict) or 'ticker' not in symbol or 'exchange' not in symbol:
                raise ValueError("Each symbol must include 'ticker' and 'exchange'")
        
        # Validate strategy parameters
        strategy = self.config['strategy']
        if 'name' not in strategy or 'params' not in strategy:
            raise ValueError("Strategy configuration must include 'name' and 'params'")
            
        # Add other validations as needed
        
        logger.info("Configuration validation successful")