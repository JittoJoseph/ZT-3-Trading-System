"""
Upstox broker implementation for ZT-3 Trading System.

This module handles all interactions with the Upstox API including:
- Authentication
- Market data streaming
- Order execution
- Position management
"""

import json
import logging
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import requests
from urllib.parse import urlencode, parse_qs, urlparse

logger = logging.getLogger(__name__)

class UpstoxBroker:
    """
    Broker implementation for Upstox API.
    
    This class handles all interactions with the Upstox API, including
    authentication, order execution, and market data streaming.
    """
    
    # API endpoints based on Upstox API v2 documentation
    BASE_URL = "https://api.upstox.com/v2"
    AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
    TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
    LOGOUT_URL = "https://api.upstox.com/v2/logout"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Upstox broker with configuration.
        
        Args:
            config: Configuration dictionary with API credentials and settings.
        """
        self.api_config = config['api']
        self.api_key = self.api_config['api_key']
        self.api_secret = self.api_config['api_secret']
        self.redirect_uri = self.api_config.get('redirect_uri')
        self.access_token = self.api_config.get('access_token')
        
        # Authentication state
        self.token_expiry = None
        
        # Session for HTTP requests
        self.session = requests.Session()
        
        # Default headers
        self.session.headers.update({
            'Accept': 'application/json'
        })
        
        # Load token from file if available
        self._load_token_from_file()
        
        logger.info("Initialized Upstox broker interface")

    def _load_token_from_file(self) -> bool:
        """
        Load access token from file if available.
        
        Returns:
            bool: True if token was loaded successfully
        """
        token_path = Path(__file__).parent / 'upstox_token.json'
        if not token_path.exists():
            logger.debug("No token file found")
            return False
            
        try:
            with open(token_path, 'r') as f:
                token_data = json.load(f)
                
            # Check if token is still valid
            expiry_time = datetime.fromtimestamp(int(token_data.get('expires_at', 0)) / 1000)
            if expiry_time <= datetime.now():
                logger.info("Saved token has expired")
                return False
                
            self.access_token = token_data.get('access_token')
            self.token_expiry = expiry_time
            
            logger.info(f"Loaded access token from file, valid until {expiry_time}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading token from file: {e}")
            return False
            
    def _save_token_to_file(self, token_data: Dict[str, Any]) -> None:
        """
        Save access token to file for reuse.
        
        Args:
            token_data: Token data to save
        """
        token_path = Path(__file__).parent / 'upstox_token.json'
        try:
            with open(token_path, 'w') as f:
                json.dump(token_data, f)
            logger.debug("Token saved to file")
        except Exception as e:
            logger.error(f"Error saving token to file: {e}")
            
    def generate_auth_url(self) -> str:
        """
        Generate the authorization URL for OAuth2 flow.
        
        Returns:
            str: Authorization URL to redirect user for authentication
        """
        if not self.redirect_uri:
            raise ValueError("Redirect URI is required for OAuth2 flow")
            
        params = {
            'client_id': self.api_key,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'state': f"zt3_{int(time.time())}"  # Add state for security
        }
        
        auth_url = f"{self.AUTH_URL}?{urlencode(params)}"
        logger.info(f"Generated authorization URL: {auth_url}")
        return auth_url
    
    def open_auth_url(self) -> None:
        """
        Open the authorization URL in default browser.
        
        User will need to manually authorize and copy the code from redirect URL.
        """
        auth_url = self.generate_auth_url()
        logger.info("Opening authorization URL in browser")
        webbrowser.open(auth_url)
        
    def exchange_code_for_token(self, auth_code: str) -> bool:
        """
        Exchange authorization code for access token.
        
        Args:
            auth_code: Authorization code received after user approval
            
        Returns:
            bool: True if token exchange was successful
        """
        if not self.redirect_uri:
            raise ValueError("Redirect URI is required for OAuth2 flow")
            
        try:
            # Set headers for token request
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            payload = {
                'code': auth_code,
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'redirect_uri': self.redirect_uri,
                'grant_type': 'authorization_code'
            }
            
            response = requests.post(self.TOKEN_URL, headers=headers, data=payload)
            response.raise_for_status()
            
            result = response.json()
            if result.get('status') != 'success':
                logger.error(f"Token exchange failed: {result}")
                return False
            
            token_data = result.get('data', {})            
            logger.debug(f"Token response: {token_data}")
            
            # Extract access token and user details
            self.access_token = token_data.get('access_token')
            
            # Calculate token expiry time (Upstox tokens expire at 6:00 AM IST the next day)
            if 'expires_in' in token_data:
                expiry_seconds = token_data.get('expires_in', 0)
                self.token_expiry = datetime.now() + timedelta(seconds=expiry_seconds)
            else:
                # Default expiry at 6:00 AM IST tomorrow
                tomorrow = datetime.now() + timedelta(days=1)
                self.token_expiry = datetime(
                    tomorrow.year, tomorrow.month, tomorrow.day, 6, 0, 0
                )
                
            logger.info(f"Authentication successful, token expires at {self.token_expiry}")
            
            # Save token for reuse
            self._save_token_to_file(token_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exchanging code for token: {e}")
            return False
            
    def handle_auth_redirect(self, redirect_url: str) -> bool:
        """
        Handle redirect URL from authorization flow.
        
        Args:
            redirect_url: Full redirect URL with authorization code
            
        Returns:
            bool: True if authorization was successful
        """
        try:
            # Parse URL to extract code
            parsed_url = urlparse(redirect_url)
            query_params = parse_qs(parsed_url.query)
            
            # Extract code
            if 'code' not in query_params:
                logger.error("No authorization code found in redirect URL")
                return False
                
            auth_code = query_params['code'][0]
            
            # Check state if included
            if 'state' in query_params and 'zt3_' not in query_params['state'][0]:
                logger.warning("State parameter doesn't match, possible CSRF attack")
                return False
                
            # Exchange code for token
            return self.exchange_code_for_token(auth_code)
            
        except Exception as e:
            logger.error(f"Error handling auth redirect: {e}")
            return False
            
    def logout(self) -> bool:
        """
        Log out and invalidate the current access token.
        
        Returns:
            bool: True if logout was successful
        """
        if not self.access_token:
            logger.warning("No access token to invalidate")
            return False
            
        try:
            headers = self._get_headers()
            response = requests.post(self.LOGOUT_URL, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            if result.get('status') == 'success' and result.get('data') is True:
                logger.info("Logout successful")
                
                # Clear token data
                self.access_token = None
                self.token_expiry = None
                
                # Remove token file
                token_path = Path(__file__).parent / 'upstox_token.json'
                if token_path.exists():
                    token_path.unlink()
                    
                return True
            else:
                logger.error(f"Logout failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """
        Check if we have a valid access token.
        
        Returns:
            bool: True if authenticated with a valid token
        """
        if not self.access_token:
            return False
            
        # Check if token is expired (if we know expiry time)
        if self.token_expiry and datetime.now() >= self.token_expiry:
            logger.warning("Access token has expired")
            return False
            
        return True
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests including authentication.
        
        Returns:
            Dict of HTTP headers.
        
        Raises:
            ValueError: If not authenticated
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Authenticate first.")
            
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        
    def _api_request(self, 
                    method: str, 
                    endpoint: str, 
                    params: Optional[Dict[str, Any]] = None,
                    data: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, str]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Make an API request to Upstox with proper error handling.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint relative to BASE_URL
            params: URL parameters
            data: Request body data
            headers: Custom headers (will be merged with auth headers)
            
        Returns:
            Tuple containing (success boolean, response data)
        """
        try:
            # Get authentication headers
            base_headers = self._get_headers()
            
            # Merge with custom headers if provided
            if headers:
                request_headers = {**base_headers, **headers}
            else:
                request_headers = base_headers
                
            # Format URL
            url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
            
            # Make request
            response = requests.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers
            )
            
            # Handle 4xx/5xx errors
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Check API status
            if result.get('status') == 'error':
                error_data = result.get('data', {})
                error_code = error_data.get('code', 'unknown')
                error_message = error_data.get('message', 'Unknown error')
                logger.error(f"API error {error_code}: {error_message}")
                return False, result
                
            return True, result
            
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors
            if e.response.status_code == 401:
                logger.error("Authentication failed (401): Token may have expired")
            elif e.response.status_code == 429:
                logger.error("Rate limit exceeded (429): Too many requests")
            logger.error(f"HTTP error: {e}")
            
            # Try to parse error response
            try:
                error_json = e.response.json()
                error_data = error_json.get('data', {})
                error_message = error_data.get('message', str(e))
                logger.error(f"API error response: {error_message}")
                return False, error_json
            except Exception:
                return False, {'status': 'error', 'data': {'message': str(e)}}
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            return False, {'status': 'error', 'data': {'message': str(e)}}
    
    def get_profile(self) -> Dict[str, Any]:
        """
        Get user profile information.
        
        Returns:
            Dict containing user profile data.
        """
        success, response = self._api_request('GET', 'user/profile')
        if success:
            return response.get('data', {})
        return {}
    
    def get_funds(self) -> Dict[str, Any]:
        """
        Get account fund and margin details.
        
        Returns:
            Dict containing fund information.
        """
        success, response = self._api_request('GET', 'user/funds-and-margin')
        if success:
            return response.get('data', {})
        return {}
    
    def get_brokerage_charges(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate brokerage charges for a potential order.
        
        Args:
            params: Dict with order parameters including:
                - symbol: Trading symbol with exchange prefix
                - quantity: Number of shares
                - product: Product type (I/D)
                - transaction_type: BUY/SELL
                - price: Order price
                
        Returns:
            Dict with calculated brokerage details.
        """
        success, response = self._api_request('GET', 'brokerage', params=params)
        if success:
            return response.get('data', {})
        return {}
        
    def get_margin_required(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate margin required for an order.
        
        Args:
            order_data: Dict with order details:
                - symbol: Trading symbol with exchange prefix
                - quantity: Number of shares
                - product: Product type (I/D)
                - transaction_type: BUY/SELL
                - price: Order price
                
        Returns:
            Dict with margin requirement details.
        """
        success, response = self._api_request('POST', 'margin-calculator', data=order_data)
        if success:
            return response.get('data', {})
        return {}
    
    def get_instruments(self, instrument_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available instruments.
        
        Args:
            instrument_type: Optional filter by instrument_type 
                            (EQ, FUT, OPT, etc.)
                            
        Returns:
            List of instrument details.
        """
        params = {}
        if instrument_type:
            params['instrument_type'] = instrument_type
            
        success, response = self._api_request('GET', 'instruments', params=params)
        if success:
            return response.get('data', [])
        return []
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List of position details.
        """
        success, response = self._api_request('GET', 'portfolio/positions')
        if success:
            return response.get('data', [])
        return []
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """
        Get holdings information.
        
        Returns:
            List of holdings details.
        """
        success, response = self._api_request('GET', 'portfolio/holdings')
        if success:
            return response.get('data', [])
        return []
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get all orders for the day.
        
        Returns:
            List of order details.
        """
        success, response = self._api_request('GET', 'orders')
        if success:
            return response.get('data', [])
        return []
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """
        Get all trades for the day.
        
        Returns:
            List of trade details.
        """
        success, response = self._api_request('GET', 'trades')
        if success:
            return response.get('data', [])
        return []
    
    def place_order(self, 
                  symbol: str,
                  exchange: str,
                  transaction_type: str,
                  quantity: int,
                  product: str = 'I',  # 'I' for intraday, 'D' for delivery
                  order_type: str = 'MARKET',
                  price: float = 0,
                  trigger_price: float = 0,
                  disclosed_quantity: int = 0,
                  validity: str = 'DAY',
                  tag: str = "ZT3") -> Dict[str, Any]:
        """
        Place an order with Upstox.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, etc.)
            transaction_type: BUY or SELL
            quantity: Number of shares
            product: Product type (I: intraday, D: delivery)
            order_type: MARKET, LIMIT, SL, SL-M
            price: Order price (for LIMIT, SL orders)
            trigger_price: Trigger price (for SL, SL-M orders)
            disclosed_quantity: Disclosed quantity
            validity: DAY, IOC
            tag: Order tag for identification
            
        Returns:
            Order details including order ID.
        """
        payload = {
            'symbol': f"{exchange}:{symbol}",
            'transaction_type': transaction_type,
            'quantity': quantity,
            'product': product,
            'order_type': order_type,
            'validity': validity,
            'tag': tag
        }
        
        # Add price and trigger_price if needed
        if order_type in ('LIMIT', 'SL'):
            payload['price'] = price
        
        if order_type in ('SL', 'SL-M'):
            payload['trigger_price'] = trigger_price
            
        if disclosed_quantity > 0:
            payload['disclosed_quantity'] = disclosed_quantity
        
        success, response = self._api_request('POST', 'order/place', data=payload)
        if success:
            order_data = response.get('data', {})
            logger.info(f"Order placed: {transaction_type} {quantity} {symbol} @ {order_type}")
            return order_data
        return {}
    
    def modify_order(self, 
                    order_id: str,
                    quantity: Optional[int] = None,
                    price: Optional[float] = None,
                    trigger_price: Optional[float] = None,
                    disclosed_quantity: Optional[int] = None,
                    validity: Optional[str] = None) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of the order to modify
            quantity: New quantity
            price: New price
            trigger_price: New trigger price
            disclosed_quantity: New disclosed quantity
            validity: New validity
            
        Returns:
            Updated order details.
        """
        payload = {}
        
        # Add only parameters that are provided
        if quantity is not None:
            payload['quantity'] = quantity
        if price is not None:
            payload['price'] = price
        if trigger_price is not None:
            payload['trigger_price'] = trigger_price
        if disclosed_quantity is not None:
            payload['disclosed_quantity'] = disclosed_quantity
        if validity is not None:
            payload['validity'] = validity
        
        success, response = self._api_request('PUT', f"order/modify/{order_id}", data=payload)
        if success:
            order_data = response.get('data', {})
            logger.info(f"Order {order_id} modified successfully")
            return order_data
        return {}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if cancellation was successful.
        """
        success, response = self._api_request('DELETE', f"order/cancel/{order_id}")
        if success:
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        return False