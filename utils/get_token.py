#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Upstox Authentication Token Helper

This script helps you obtain an access token for the Upstox API
to use with the backtester and trading system. It opens the Upstox 
authentication page and then captures the authorization code from 
the callback URL.
"""

import sys
import os
from pathlib import Path
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import time
import urllib.parse
import requests
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parents[1]))

# Import for loading environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables
auth_code = None
server_running = True

class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler to capture the authorization code from callback."""
    
    def do_GET(self):
        """Handle GET request with authorization code."""
        global auth_code, server_running
        
        # Parse the query parameters
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        
        # Check if code parameter exists
        if 'code' in params:
            auth_code = params['code'][0]
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # HTML response
            html = """
            <html>
            <head>
                <title>Authentication Successful</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        text-align: center;
                        padding: 50px;
                        background-color: #f5f5f5;
                    }
                    .container {
                        background-color: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        max-width: 600px;
                        margin: 0 auto;
                    }
                    h1 { color: #2ecc71; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Authentication Successful!</h1>
                    <p>You can now close this window and return to the terminal.</p>
                    <p>Your Upstox API access token has been saved.</p>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
            # Signal server to stop
            server_running = False
        else:
            # Error response
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <html>
            <head>
                <title>Authentication Error</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        text-align: center;
                        padding: 50px;
                        background-color: #f5f5f5;
                    }
                    .container {
                        background-color: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        max-width: 600px;
                        margin: 0 auto;
                    }
                    h1 { color: #e74c3c; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Authentication Error</h1>
                    <p>No authorization code found in the callback.</p>
                    <p>Please try again.</p>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
    
    def log_message(self, format, *args):
        """Override to suppress server logs."""
        return

def start_callback_server(port=3000):
    """Start a local server to capture the callback."""
    global server_running
    
    server = HTTPServer(('localhost', port), CallbackHandler)
    
    print(f"Starting callback server on http://localhost:{port}")
    
    # Run server until auth code is received or timeout
    while server_running:
        server.handle_request()

def authenticate_with_upstox():
    """
    Authenticate with Upstox API and get access token.
    
    Returns:
        Dict with authentication results or None if failed
    """
    # Load API credentials from environment variables
    api_key = os.environ.get('UPSTOX_API_KEY')
    api_secret = os.environ.get('UPSTOX_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Error: API credentials not found in environment variables.")
        print("Please ensure UPSTOX_API_KEY and UPSTOX_API_SECRET are set in your .env file.")
        return None
    
    print("🔑 API credentials loaded successfully.")
    
    # Set up the redirect URI and URLs according to Upstox documentation
    redirect_uri = "http://localhost:3000/callback"
    auth_url = "https://api.upstox.com/v2/login/authorization/dialog"
    token_url = "https://api.upstox.com/v2/login/authorization/token"
    
    # Start callback server in a thread
    server_thread = threading.Thread(target=start_callback_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Construct auth URL with exact parameters as specified in the documentation
    auth_params = {
        'response_type': 'code',                # Must be 'code'
        'client_id': api_key,                   # This is the API key
        'redirect_uri': redirect_uri            # Must match registered redirect URI
    }
    
    params_str = '&'.join([f"{k}={urllib.parse.quote(v)}" for k, v in auth_params.items()])
    full_auth_url = f"{auth_url}?{params_str}"
    
    print(f"🌐 Opening authorization URL in your browser...")
    print(f"URL: {full_auth_url}")
    webbrowser.open(full_auth_url)
    
    print("\n⌛ Waiting for authentication... (login to Upstox when the browser opens)")
    
    # Wait for auth_code to be set by callback handler
    start_time = time.time()
    timeout = 300  # 5 minutes timeout
    
    while auth_code is None:
        time.sleep(0.5)
        if time.time() - start_time > timeout:
            print("❌ Authentication timed out after 5 minutes.")
            return None
    
    print("\n✅ Authorization code received!")
    print("🔄 Exchanging code for access token...")
    
    # Exchange auth code for token using form data and correct parameter names
    try:
        # Prepare the form data exactly as specified in the documentation
        form_data = {
            'code': auth_code,
            'client_id': api_key,              # API Key
            'client_secret': api_secret,        # API Secret
            'redirect_uri': redirect_uri,       # Must match the URI used for authorization
            'grant_type': 'authorization_code'  # Must be 'authorization_code'
        }
        
        # Make the token request as application/x-www-form-urlencoded
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        # Show the exact request being made (for debugging)
        print(f"Sending token request to: {token_url}")
        print(f"Form data: {form_data}")
        
        response = requests.post(token_url, data=form_data, headers=headers)
        
        if response.status_code != 200:
            print(f"❌ Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        token_data = response.json()
        
        # Handle both the standard success format and the direct token format
        # The API might return either {"status": "success", "data": {...}} format
        # or directly return the token data
        access_token = None
        if "status" in token_data and token_data.get("status") == "success":
            # Standard format
            access_token = token_data.get('data', {}).get('access_token')
            refresh_token = token_data.get('data', {}).get('refresh_token')
            expires_in = token_data.get('data', {}).get('expires_in', 86400)  # Default to 24 hours
        else:
            # Direct token format (new API behavior)
            access_token = token_data.get('access_token')
            refresh_token = token_data.get('refresh_token', None)  # May not be present
            expires_in = 86400  # Assume 24 hours if not provided
            
        if access_token:
            print(f"✅ Access token received: {access_token[:10]}...{access_token[-5:]}")
            
            # Calculate expiry time
            expiry_time = datetime.now() + timedelta(seconds=expires_in)
            expiry_time_str = expiry_time.isoformat()
            
            # Save tokens to file
            token_file = Path('data/upstox_token.json')
            token_file.parent.mkdir(exist_ok=True, parents=True)
            
            token_info = {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'expiry': expiry_time_str
            }
            
            with open(token_file, 'w') as f:
                json.dump(token_info, f, indent=2)
                
            # Update .env file with access token
            env_path = Path('.env')
            
            if env_path.exists():
                with open(env_path, 'r') as f:
                    env_content = f.read()
                
                # Check if UPSTOX_ACCESS_TOKEN already exists
                if 'UPSTOX_ACCESS_TOKEN=' in env_content:
                    # Update existing token
                    env_lines = env_content.splitlines()
                    updated_lines = []
                    
                    for line in env_lines:
                        if line.startswith('UPSTOX_ACCESS_TOKEN='):
                            updated_lines.append(f'UPSTOX_ACCESS_TOKEN="{access_token}"')
                        else:
                            updated_lines.append(line)
                    
                    with open(env_path, 'w') as f:
                        f.write('\n'.join(updated_lines))
                else:
                    # Add new token
                    with open(env_path, 'a') as f:
                        f.write(f'\nUPSTOX_ACCESS_TOKEN="{access_token}"\n')
                
                print(f"✅ Access token added to .env file")
            else:
                print(f"⚠️ Note: .env file not found, token not saved to environment variables")
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'expiry': expiry_time_str,
                'token_file': str(token_file)
            }
        else:
            print(f"❌ Authentication failed: No access token in response")
            print(f"Response: {token_data}")
            return None
            
    except Exception as e:
        print(f"❌ Error exchanging code for token: {e}")
        return None

def main():
    """Main entry point for the script."""
    print("=" * 60)
    print("        Upstox API Authentication Token Helper")
    print("=" * 60)
    print("\nThis script will help you obtain an access token for the Upstox API.")
    print("The token will be used for backtesting and live trading.")
    print("\n1. A browser window will open for you to login to Upstox.")
    print("2. After logging in, authorize the application.")
    print("3. You'll be redirected back to this application.")
    print("4. The access token will be saved for future use.")
    
    # Check if we already have a valid token
    token_file = Path('data/upstox_token.json')
    if token_file.exists():
        try:
            with open(token_file, 'r') as f:
                token_data = json.load(f)
                
            expiry = datetime.fromisoformat(token_data.get('expiry', '2000-01-01T00:00:00'))
            
            if expiry > datetime.now():
                time_left = expiry - datetime.now()
                hours = time_left.seconds // 3600
                minutes = (time_left.seconds % 3600) // 60
                
                print(f"\n🔒 You already have a valid token that expires in {hours} hours and {minutes} minutes.")
                print(f"   Token: {token_data.get('access_token')[:10]}...{token_data.get('access_token')[-5:]}")
                
                choice = input("\nDo you want to generate a new token anyway? (y/n): ").strip().lower()
                if choice != 'y':
                    print("\n✅ Using existing token. No action required.")
                    return 0
                
                print("\n🔄 Proceeding to generate new token...")
        except Exception as e:
            print(f"Error reading existing token: {e}")
            print("Proceeding to generate new token...")
    
    input("\nPress Enter to continue...")
    
    # Start authentication process
    result = authenticate_with_upstox()
    
    if result:
        print("\n✅ Authentication successful!")
        print(f"Access token: {result['access_token'][:10]}...{result['access_token'][-5:]}")
        print(f"Token saved to: {result['token_file']}")
        print(f"Token expires: {result['expiry']}")
        
        print("\n🎉 You're all set! You can now use the ZT-3 system.")
        print("   Run 'python run_backtest.py' for backtesting")
        print("   Run 'python main.py' for paper/live trading")
        return 0
    else:
        print("\n❌ Authentication failed. Please try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
