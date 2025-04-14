"""
Configuration Loader for ZT-3 Trading System.

This module handles loading and validating configuration files.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Configuration loader for the trading system.
    
    This class handles loading and validating YAML configuration files,
    with support for environment variable substitution and validation.
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files.
                        If None, defaults to 'config' directory.
        """
        if config_dir is None:
            # Default to config directory relative to this file
            self.config_dir = Path(__file__).parent
        else:
            self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
    
    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a configuration file and process it.
        
        Args:
            config_name: Name of the configuration file (with or without extension)
            
        Returns:
            Loaded configuration dictionary or None if loading failed
        """
        # Ensure the file has a YAML extension
        if not config_name.endswith(('.yaml', '.yml')):
            config_name = f"{config_name}.yaml"
        
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Process environment variables
            config = self._substitute_env_vars(config)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration values.
        
        Args:
            config: Configuration value (dict, list, str, etc.)
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # Extract environment variable name
            env_var = config[2:-1]
            # Get value with optional default after colon
            if ':' in env_var:
                env_var, default = env_var.split(':', 1)
                return os.environ.get(env_var, default)
            else:
                return os.environ.get(env_var, config)
        else:
            return config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
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
            logger.info("Configuration validation successful")
            return True
        
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> bool:
        """
        Save a configuration dictionary to a YAML file.
        
        Args:
            config: Configuration dictionary to save
            config_name: Name of the configuration file (with or without extension)
            
        Returns:
            True if configuration was saved successfully, False otherwise
        """
        # Ensure the file has a YAML extension
        if not config_name.endswith(('.yaml', '.yml')):
            config_name = f"{config_name}.yaml"
        
        config_path = self.config_dir / config_name
        
        try:
            # Create parent directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Saved configuration to {config_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_available_configs(self) -> list:
        """
        Get a list of available configuration files.
        
        Returns:
            List of configuration file names (without extension)
        """
        try:
            # List all YAML files in the config directory
            configs = []
            for item in self.config_dir.glob('*.yaml'):
                if item.is_file():
                    configs.append(item.stem)
            
            return configs
        
        except Exception as e:
            logger.error(f"Failed to list configurations: {e}")
            return []