"""
Upstox Broker Interface for ZT-3 Trading System.

This module handles connection to Upstox API for placing orders,
managing positions, and authentication.
"""

import logging
import json
import requests
import time
from datetime import datetime, timedelta
import os
import webbrowser
from typing import Dict, List, Any, Optional
from pathlib import Path
import urllib.parse

logger = logging.getLogger(__name__)

class UpstoxBroker:
    """
    Interface for the Upstox broker API.
    
    This class handles authentication, order placement, and position management.
    """
    
    # API endpoints
    BASE_URL = "https://api.upstox.com/v2"
    TOKEN_URL = f"{BASE_URL}/login/authorization/token"
    AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
    PROFILE_URL = f"{BASE_URL}/user/profile"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Upstox broker interface.
        
        Args:
            config: Configuration dictionary with API credentials
        """
        self.config = config
        
        # API credentials
        api_config = config.get('api', {})
        self.api_key = os.environ.get('UPSTOX_API_KEY') or api_config.get('api_key')
        self.api_secret = os.environ.get('UPSTOX_API_SECRET') or api_config.get('api_secret')
        self.redirect_uri = api_config.get('redirect_uri', 'http://localhost:3000/callback')
        
        if not self.api_key or not self.api_secret:
            logger.error("Missing API credentials. Please set UPSTOX_API_KEY and UPSTOX_API_SECRET in .env file")
            raise ValueError("Missing API credentials")
        
        # Authentication state
        self.access_token = os.environ.get('UPSTOX_ACCESS_TOKEN')
        self.refresh_token = None
        self.token_expiry = None
        
        # Token file storage
        self.token_file = Path('data/upstox_token.json')
        self.token_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Try to load existing token
        self._load_tokens()
        
        logger.debug("Upstox broker interface initialized")
    
    def _load_tokens(self) -> bool:
        """
        Load authentication tokens from file.
        
        Returns:
            True if valid tokens were loaded, False otherwise
        """
        try:
            if not self.token_file.exists():
                logger.info("No token file found, need to authenticate")
                return False
                
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
                
            self.access_token = token_data.get('access_token')
            self.refresh_token = token_data.get('refresh_token')
            
            expiry_str = token_data.get('expiry')
            if expiry_str:
                self.token_expiry = datetime.fromisoformat(expiry_str)
                
            # Check if token is expired or will expire soon
            if self.token_expiry and self.token_expiry > datetime.now() + timedelta(minutes=10):
                logger.info("Loaded valid authentication tokens")
                return True
            else:
                logger.info("Loaded tokens are expired or will expire soon")
                # Try to refresh the token
                if self.refresh_token:
                    return self._refresh_token()
                    
                return False
                
        except Exception as e:
            logger.error(f"Error loading tokens: {e}")
            return False
    
    def _save_tokens(self) -> None:
        """Save authentication tokens to file."""
        try:
            token_data = {
                'access_token': self.access_token,
                'refresh_token': self.refresh_token,
                'expiry': self.token_expiry.isoformat() if self.token_expiry else None
            }
            
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
                
            logger.info("Tokens saved successfully")
            
            # Also update the environment variable
            os.environ['UPSTOX_ACCESS_TOKEN'] = self.access_token
            
        except Exception as e:
            logger.error(f"Error saving tokens: {e}")
    
    def _refresh_token(self) -> bool:
        """
        Refresh the access token using the refresh token.
        
        Returns:
            True if token was refreshed successfully, False otherwise
        """
        if not self.refresh_token:
            logger.warning("No refresh token available")
            return False
            
        try:
            # Form data for refresh token request
            form_data = {
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'refresh_token': self.refresh_token,
                'grant_type': 'refresh_token'
            }
            
            # Headers according to documentation
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = requests.post(self.TOKEN_URL, data=form_data, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Token refresh failed with status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
            data = response.json()
            
            # Handle both the standard success format and the direct token format
            access_token = None
            if "status" in data and data.get("status") == "success":
                # Standard format
                token_data = data.get('data', {})
                access_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 86400)  # Default to 24 hours
            else:
                # Direct token format (new API behavior)
                access_token = data.get('access_token')
                expires_in = 86400  # Assume 24 hours if not provided
            
            if access_token:
                self.access_token = access_token
                self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
                logger.info("Token refreshed successfully")
                self._save_tokens()
                return True
            else:
                logger.error(f"Token refresh failed: No access token in response")
                logger.error(f"Response: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return False
    
    def authenticate_with_code(self, auth_code: str) -> bool:
        """
        Authenticate using the authorization code.
        
        Args:
            auth_code: Authorization code from redirect URL
            
        Returns:
            True if authentication was successful, False otherwise
        """
        try:
            # Create form data for token exchange with exact parameters as required by Upstox API
            form_data = {
                'code': auth_code,
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'redirect_uri': self.redirect_uri,
                'grant_type': 'authorization_code'
            }
            
            # Headers according to documentation
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            logger.info("Requesting access token")
            response = requests.post(self.TOKEN_URL, data=form_data, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Token request failed with status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
            data = response.json()
            
            # Handle both the standard success format and the direct token format
            access_token = None
            refresh_token = None
            expires_in = 86400  # Default to 24 hours
            
            if "status" in data and data.get("status") == "success":
                # Standard format
                token_data = data.get('data', {})
                access_token = token_data.get('access_token')
                refresh_token = token_data.get('refresh_token')
                expires_in = token_data.get('expires_in', 86400)
            else:
                # Direct token format (new API behavior)
                access_token = data.get('access_token')
                refresh_token = data.get('refresh_token')
            
            if access_token:
                self.access_token = access_token
                self.refresh_token = refresh_token
                self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
                
                logger.info("Authentication successful")
                self._save_tokens()
                return True
            else:
                logger.error(f"Authentication failed: No access token in response")
                logger.error(f"Response: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """
        Check if we have a valid authentication token.
        
        Returns:
            True if authenticated, False otherwise
        """
        if not self.access_token:
            return False
            
        if self.token_expiry and self.token_expiry <= datetime.now():
            logger.warning("Access token has expired")
            return self._refresh_token()
            
        return True
    
    def open_auth_url(self) -> None:
        """Open the Upstox authentication URL in a web browser."""
        # Use exact parameter names as specified in the Upstox API documentation
        auth_params = {
            'response_type': 'code',
            'client_id': self.api_key,
            'redirect_uri': self.redirect_uri
        }
        
        # URL encode parameter values
        params_str = '&'.join([f"{k}={urllib.parse.quote(v)}" for k, v in auth_params.items()])
        auth_url = f"{self.AUTH_URL}?{params_str}"
        
        logger.info(f"Opening authorization URL: {auth_url}")
        webbrowser.open(auth_url)
        
        print("\n=======================")
        print("Upstox Authentication Required")
        print("=======================")
        print("A browser window has been opened for you to login to Upstox.")
        print("After logging in, you will be redirected back to the application.")
        print("\nIf the browser doesn't open automatically, please navigate to:")
        print(auth_url)
        
    # Additional broker methods would go here (order placement, position management, etc.)