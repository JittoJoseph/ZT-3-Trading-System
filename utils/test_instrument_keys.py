#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Instrument Keys for Upstox API

This script tests fetching historical data using the instrument key format
required by the Upstox API.
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Dictionary mapping common tickers to their ISINs
TICKER_TO_ISIN = {
    'PNB': 'INE160A01022',        # Punjab National Bank
    'SBIN': 'INE062A01020',       # State Bank of India
    'TATASTEEL': 'INE081A01020',  # Tata Steel
    'RELIANCE': 'INE002A01018',   # Reliance Industries
    'INFY': 'INE009A01021',       # Infosys
    'BANKBARODA': 'INE028A01039', # Bank of Baroda
    'CANBK': 'INE476A01022',      # Canara Bank
    'ITC': 'INE154A01025',        # ITC Limited
    'HDFCBANK': 'INE040A01034',   # HDFC Bank
    'TCS': 'INE467B01029',        # Tata Consultancy Services
    'NHPC': 'INE848E01016',       # NHPC Limited
    'RVNL': 'INE415G01027',       # Rail Vikas Nigam Limited
    'YESBANK': 'INE528G01035'     # Yes Bank Limited
}

def test_historical_candle_api():
    """Test the Upstox historical candle API with the correct instrument key format."""
    # Get access token
    access_token = os.environ.get('UPSTOX_ACCESS_TOKEN')
    if not access_token:
        print("Error: No access token found. Please set UPSTOX_ACCESS_TOKEN in .env file.")
        return
    
    # Set up headers
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    
    # Calculate date range (30 days back)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Test each ticker in our dictionary
    success_count = 0
    failed_count = 0
    results = []
    
    for ticker, isin in TICKER_TO_ISIN.items():
        exchange = "NSE"  # We're using NSE for all examples
        instrument_key = f"{exchange}_EQ|{isin}"
        encoded_instrument_key = requests.utils.quote(instrument_key)
        
        url = f"https://api.upstox.com/v2/historical-candle/{encoded_instrument_key}/30minute/{to_date}/{from_date}"
        
        print(f"\nTesting {ticker} (ISIN: {isin}):")
        print(f"URL: {url}")
        
        try:
            # Request historical data
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
                    candles = data['data']['candles']
                    print(f"✅ Success! Got {len(candles)} candles")
                    
                    if candles:
                        # Show first candle as sample
                        print(f"Sample candle: {candles[0]}")
                        
                    success_count += 1
                    results.append({
                        'ticker': ticker,
                        'isin': isin,
                        'instrument_key': instrument_key,
                        'status': 'success',
                        'candles_count': len(candles)
                    })
                else:
                    print(f"❌ Failed: Invalid response format")
                    failed_count += 1
                    results.append({
                        'ticker': ticker,
                        'isin': isin,
                        'instrument_key': instrument_key,
                        'status': 'failed',
                        'error': 'Invalid response format'
                    })
            else:
                print(f"❌ Failed: HTTP {response.status_code} - {response.text}")
                failed_count += 1
                results.append({
                    'ticker': ticker,
                    'isin': isin,
                    'instrument_key': instrument_key,
                    'status': 'failed',
                    'error': f"HTTP {response.status_code}"
                })
        
        except Exception as e:
            print(f"❌ Error: {e}")
            failed_count += 1
            results.append({
                'ticker': ticker,
                'isin': isin,
                'instrument_key': instrument_key,
                'status': 'error',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "-" * 50)
    print(f"Test completed: {success_count} successful, {failed_count} failed")
    print("-" * 50)
    
    # Save results to JSON file
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "instrument_key_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}/instrument_key_test_results.json")

if __name__ == "__main__":
    test_historical_candle_api()
