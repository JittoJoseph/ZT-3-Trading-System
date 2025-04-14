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
from typing import Dict, List, Any, Optional, Callable, Union
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
    ACCESS_TOKEN_REQUEST_URL = "https://api.upstox.com/v2/login/authorization/initiate"
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
        self.notifier_url = self.api_config.get('notifier_url')
        
        # Authentication state
        self.token_expiry = None
        
        # Session for HTTP requests
        self.session = requests.Session()
        
        # Default headers
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
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
            token_data = response.json()
            
            if 'data' in token_data and token_data.get('status') == 'success':
                token_data = token_data['data']
            
            logger.debug(f"Token response: {token_data}")
            
            # Extract access token and user details
            self.access_token = token_data.get('access_token')
            
            # Calculate token expiry time (Upstox tokens expire at 3:30 AM IST the next day)
            if 'expires_in' in token_data:
                expiry_seconds = token_data.get('expires_in', 0)
                self.token_expiry = datetime.now() + timedelta(seconds=expiry_seconds)
            else:
                # Default expiry at 3:30 AM IST tomorrow
                tomorrow = datetime.now() + timedelta(days=1)
                self.token_expiry = datetime(
                    tomorrow.year, tomorrow.month, tomorrow.day, 3, 30, 0
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
            
    def request_access_token(self) -> bool:
        """
        Request access token using the Access Token Request flow.
        
        This is an alternative to the OAuth2 flow where Upstox will send
        the access token to the notifier URL once user approves.
        
        Returns:
            bool: True if request was submitted successfully
        """
        if not self.notifier_url:
            logger.error("Notifier URL is required for Access Token Request flow")
            return False
            
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            payload = {
                'client_id': self.api_key,
                'client_secret': self.api_secret
            }
            
            response = requests.post(
                self.ACCESS_TOKEN_REQUEST_URL, 
                headers=headers, 
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('status') != 'success':
                logger.error(f"Failed to request access token: {result}")
                return False
                
            data = result.get('data', {})
            expiry_time = datetime.fromtimestamp(int(data.get('authorization_expiry', 0)) / 1000)
            notifier_url = data.get('notifier_url')
            
            logger.info(f"Access token request sent successfully")
            logger.info(f"Authorization will expire at: {expiry_time}")
            logger.info(f"Token will be sent to: {notifier_url}")
            logger.info("Please approve the request in Upstox app")
            
            return True
            
        except Exception as e:
            logger.error(f"Error requesting access token: {e}")
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
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }
            
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
    
    def get_profile(self) -> Dict[str, Any]:
        """
        Get user profile information.
        
        Returns:
            Dict containing user profile data.
        """
        try:
            headers = self._get_headers()
            response = requests.get(f"{self.BASE_URL}/user/profile", headers=headers)
            response.raise_for_status()
            return response.json().get('data', {})
            
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            return {}
    
    def get_funds(self) -> Dict[str, Any]:
        """
        Get account fund details.
        
        Returns:
            Dict containing fund information.
        """
        try:
            headers = self._get_headers()
            response = requests.get(f"{self.BASE_URL}/user/funds-and-margin", headers=headers)
            response.raise_for_status()
            return response.json().get('data', {})
            
        except Exception as e:
            logger.error(f"Failed to get funds information: {e}")
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List of position details.
        """
        try:
            headers = self._get_headers()
            response = requests.get(f"{self.BASE_URL}/portfolio/positions", headers=headers)
            response.raise_for_status()
            return response.json().get('data', [])
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """
        Get holdings information.
        
        Returns:
            List of holdings details.
        """
        try:
            headers = self._get_headers()
            response = requests.get(f"{self.BASE_URL}/portfolio/holdings", headers=headers)
            response.raise_for_status()
            return response.json().get('data', [])
            
        except Exception as e:
            logger.error(f"Failed to get holdings: {e}")
            return []
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get all orders for the day.
        
        Returns:
            List of order details.
        """
        try:
            headers = self._get_headers()
            response = requests.get(f"{self.BASE_URL}/orders", headers=headers)
            response.raise_for_status()
            return response.json().get('data', [])
            
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
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
        try:
            headers = self._get_headers()
            payload = {
                'symbol': symbol,
                'exchange': exchange,
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
            
            response = requests.post(f"{self.BASE_URL}/order", headers=headers, json=payload)
            response.raise_for_status()
            order_data = response.json().get('data', {})
            
            logger.info(f"Order placed: {transaction_type} {quantity} {symbol} @ {order_type}")
            return order_data
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
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
        try:
            headers = self._get_headers()
            payload = {
                'order_id': order_id
            }
            
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
            
            response = requests.put(f"{self.BASE_URL}/order/{order_id}", headers=headers, json=payload)
            response.raise_for_status()
            order_data = response.json().get('data', {})
            
            logger.info(f"Order {order_id} modified successfully")
            return order_data
            
        except Exception as e:
            logger.error(f"Failed to modify order: {e}")
            return {}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if cancellation was successful.
        """
        try:
            headers = self._get_headers()
            response = requests.delete(f"{self.BASE_URL}/order/{order_id}", headers=headers)
            response.raise_for_status()
            
            logger.info(f"Order {order_id} cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    def get_historical_candles(self, 
                              symbol: str,
                              exchange: str,
                              interval: str,
                              from_date: str,
                              to_date: str) -> pd.DataFrame:
        """
        Get historical OHLC candle data.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, etc.)
            interval: Candle interval (1minute, 5minute, hour, day, etc.)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with historical candles data.
        """
        try:
            headers = self._get_headers()
            params = {
                'symbol': f"{exchange}:{symbol}",
                'interval': interval,
                'from': from_date,
                'to': to_date
            }
            
            response = requests.get(f"{self.BASE_URL}/historical-candle", headers=headers, params=params)
            response.raise_for_status()
            candle_data = response.json().get('data', [])
            
            # Convert to DataFrame
            if not candle_data:
                return pd.DataFrame()
                
            df = pd.DataFrame(candle_data)
            
            # Convert timestamps to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical candles: {e}")
            return pd.DataFrame()
    
    def _on_ws_message(self, ws, message):
        """
        Handle WebSocket messages.
        
        Args:
            ws: WebSocket connection
            message: Received message
        """
        try:
            data = json.loads(message)
            
            if 'type' not in data:
                return
                
            if data['type'] == 'connection':
                if data.get('status') == 'connected':
                    logger.info("WebSocket connected successfully")
                    self.ws_connected = True
                    
            elif data['type'] == 'order':
                # Handle order updates
                order_data = data.get('data', {})
                logger.info(f"Order update received: {order_data.get('order_id')}")
                
            elif data['type'] == 'trade':
                # Handle trade updates
                trade_data = data.get('data', {})
                logger.info(f"Trade update received: {trade_data.get('order_id')}")
                
            elif data['type'] == 'quote':
                # Handle market quotes
                quote_data = data.get('data', {})
                symbol = quote_data.get('symbol', '')
                
                if symbol:
                    self.latest_quotes[symbol] = quote_data
                    
                    # Process candle data if it's a full quote
                    if all(k in quote_data for k in ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
                        candle = {
                            'timestamp': quote_data['timestamp'],
                            'open': quote_data['open'],
                            'high': quote_data['high'],
                            'low': quote_data['low'],
                            'close': quote_data['close'],
                            'volume': quote_data['volume']
                        }
                        
                        if symbol not in self.candles:
                            self.candles[symbol] = []
                        self.candles[symbol].append(candle)
                        
                        # Call all callbacks
                        for callback in self.market_data_callbacks:
                            callback(symbol, candle)
            
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_ws_error(self, ws, error):
        """
        Handle WebSocket errors.
        
        Args:
            ws: WebSocket connection
            error: Error details
        """
        logger.error(f"WebSocket error: {error}")
        self.ws_connected = False
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """
        Handle WebSocket connection close.
        
        Args:
            ws: WebSocket connection
            close_status_code: Status code
            close_msg: Close message
        """
        logger.info(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
        self.ws_connected = False
    
    def _on_ws_open(self, ws):
        """
        Handle WebSocket connection open.
        
        Args:
            ws: WebSocket connection
        """
        logger.info("WebSocket connection opened")
        
        # Send authentication message
        auth_message = {
            'type': 'auth',
            'api_key': self.api_key,
            'access_token': self.access_token
        }
        ws.send(json.dumps(auth_message))
    
    def connect_websocket(self) -> bool:
        """
        Connect to Upstox WebSocket API for market data and order updates.
        
        Returns:
            bool: True if connection was successful.
        """
        if not self.access_token:
            logger.error("Not authenticated. Call authenticate() first.")
            return False
            
        try:
            # WebSocket URL for market data
            ws_url = "wss://api.upstox.com/v2/feed"
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(ws_url,
                                           on_open=self._on_ws_open,
                                           on_message=self._on_ws_message,
                                           on_error=self._on_ws_error,
                                           on_close=self._on_ws_close)
            
            # Start WebSocket connection in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection to establish
            wait_time = 0
            while not self.ws_connected and wait_time < 30:
                time.sleep(0.5)
                wait_time += 0.5
            
            return self.ws_connected
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    def subscribe_market_data(self, symbols: List[Dict[str, str]]) -> bool:
        """
        Subscribe to market data feed.
        
        Args:
            symbols: List of symbol dictionaries with 'ticker' and 'exchange'
            
        Returns:
            bool: True if subscription was successful.
        """
        if not self.ws_connected:
            logger.error("WebSocket not connected. Call connect_websocket() first.")
            return False
            
        try:
            # Format symbols
            formatted_symbols = [f"{s['exchange']}:{s['ticker']}" for s in symbols]
            
            # Create subscription message
            subscribe_message = {
                'type': 'subscribe',
                'symbols': formatted_symbols
            }
            
            # Send subscription message
            self.ws.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to {len(symbols)} symbols")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to market data: {e}")
            return False
    
    def register_market_data_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Register a callback for market data updates.
        
        Args:
            callback: Function to call with symbol and candle data.
        """
        self.market_data_callbacks.append(callback)
        
    def unregister_market_data_callback(self, callback: Callable) -> None:
        """
        Unregister a previously registered callback.
        
        Args:
            callback: Previously registered callback function.
        """
        if callback in self.market_data_callbacks:
            self.market_data_callbacks.remove(callback)
    
    def disconnect_websocket(self) -> bool:
        """
        Disconnect from WebSocket.
        
        Returns:
            bool: True if disconnection was successful.
        """
        if self.ws:
            self.ws.close()
            self.ws = None
            self.ws_connected = False
            
            if self.ws_thread and self.ws_thread.is_alive():
                # Wait for thread to finish
                self.ws_thread.join(timeout=5)
                
            logger.info("WebSocket disconnected")
            return True
            
        return False
    
    def get_market_quote(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get latest market quote for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, etc.)
            
        Returns:
            Dict with market quote data.
        """
        try:
            headers = self._get_headers()
            params = {
                'symbol': f"{exchange}:{symbol}"
            }
            
            response = requests.get(f"{self.BASE_URL}/market-quote", headers=headers, params=params)
            response.raise_for_status()
            
            quote_data = response.json().get('data', {})
            return quote_data
            
        except Exception as e:
            logger.error(f"Failed to get market quote: {e}")
            return {}