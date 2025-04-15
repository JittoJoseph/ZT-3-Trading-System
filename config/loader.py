"""
Configuration Loader for ZT-3 Trading System.

Handles loading and validating configuration files.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Configuration loader and validator.
    
    Loads YAML configuration files and applies environment variable substitutions.
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialize config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path(__file__).parent
        
    def load_config(self, config_file: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Configuration file name in config directory
            
        Returns:
            Configuration dictionary or None if loading failed
        """
        # Build config file path
        if os.path.isabs(config_file):
            config_path = config_file
        else:
            config_path = os.path.join(self.config_dir, config_file)
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Apply environment variable substitutions
            config_data = self._substitute_env_vars(config_data)
            
            # Validate configuration
            is_valid, errors = self._validate_config(config_data)
            if not is_valid:
                logger.error(f"Invalid configuration: {', '.join(errors)}")
                return None
            
            logger.info(f"Loaded configuration from {config_path}")
            return config_data
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None
    
    def _substitute_env_vars(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration values.
        
        Environment variables are referenced as ${ENV_VAR_NAME}.
        
        Args:
            config_data: Configuration data
            
        Returns:
            Configuration data with environment variables substituted
        """
        if isinstance(config_data, dict):
            return {k: self._substitute_env_vars(v) for k, v in config_data.items()}
        elif isinstance(config_data, list):
            return [self._substitute_env_vars(item) for item in config_data]
        elif isinstance(config_data, str) and config_data.startswith("${") and config_data.endswith("}"):
            # Extract environment variable name
            env_var = config_data[2:-1]
            # Get value with a default of empty string
            return os.environ.get(env_var, "")
        else:
            return config_data
    
    def _validate_config(self, config_data: Dict[str, Any]) -> tuple:
        """
        Validate configuration data.
        
        Args:
            config_data: Configuration data
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Minimum required sections
        required_sections = ['api', 'symbols', 'strategy']
        for section in required_sections:
            if section not in config_data:
                errors.append(f"Missing required section: {section}")
        
        # Validate API section if present
        if 'api' in config_data:
            api_config = config_data['api']
            if not api_config.get('api_key'):
                errors.append("Missing API key in api section")
            if not api_config.get('api_secret'):
                errors.append("Missing API secret in api section")
        
        # Validate symbols section if present
        if 'symbols' in config_data:
            symbols = config_data['symbols']
            if not isinstance(symbols, list) or len(symbols) == 0:
                errors.append("symbols section must be a non-empty list")
            else:
                for i, symbol in enumerate(symbols):
                    if not isinstance(symbol, dict):
                        errors.append(f"Symbol at index {i} must be a dictionary")
                    else:
                        if 'ticker' not in symbol:
                            errors.append(f"Symbol at index {i} missing 'ticker'")
                        if 'exchange' not in symbol:
                            errors.append(f"Symbol at index {i} missing 'exchange'")
        
        # Validate strategy section if present
        if 'strategy' in config_data:
            strategy = config_data['strategy']
            if not strategy.get('name'):
                errors.append("Missing strategy name in strategy section")
        
        return len(errors) == 0, errors
    
    def save_config(self, config_data: Dict[str, Any], config_file: str) -> bool:
        """
        Save configuration to YAML file.
        
        Args:
            config_data: Configuration data
            config_file: Configuration file name in config directory
            
        Returns:
            True if saving was successful, False otherwise
        """
        # Build config file path
        if os.path.isabs(config_file):
            config_path = config_file
        else:
            config_path = os.path.join(self.config_dir, config_file)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            logger.info(f"Saved configuration to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False