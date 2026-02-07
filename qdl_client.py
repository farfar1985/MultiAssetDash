"""
QDT Data Lake Client for Nexus
==============================
Standalone client to fetch data from Quantum Data Lake (QDL).

API Format: https://qdl.ai/get_data/{symbol},{provider}/{start_date}/{end_date}
Auth: qml-api-key header
Date format: YYYYMMDDHHMM

Based on: DTNIQ_CONNECTION_README.md
"""

import os
import logging
import requests
import pandas as pd
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from functools import wraps
import time
import json

log = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

QDL_URL = os.getenv('QDL_URL', 'https://qdl.ai')
QML_API_KEY = os.getenv('QML_API_KEY', 'Fp6c_bw7qkFdb_LhYVJmGA')

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# ============================================================================
# Asset Mapping: Nexus → QDL (symbol, provider)
# ============================================================================

NEXUS_TO_QDL_MAP: Dict[str, Tuple[str, str]] = {
    # US Indices (DTNIQ futures)
    'SP500': ('@ES#C', 'DTNIQ'),           # E-mini S&P 500
    'NASDAQ': ('@NQ#C', 'DTNIQ'),          # E-mini NASDAQ 100
    'Dow_Jones': ('@YM#C', 'DTNIQ'),       # E-mini Dow
    'Russell_2000': ('@RTY#C', 'DTNIQ'),   # E-mini Russell 2000
    
    # Commodities (DTNIQ futures)
    'Crude_Oil': ('QCL#', 'DTNIQ'),        # WTI Crude Oil
    'Brent_Oil': ('QBZ#', 'DTNIQ'),        # Brent Crude
    'GOLD': ('QGC#', 'DTNIQ'),             # Gold
    'MCX_Copper': ('QHG#', 'DTNIQ'),       # Copper
    'Natural_Gas': ('QNG#', 'DTNIQ'),      # Natural Gas
    
    # Crypto
    'Bitcoin': ('@BTC#C', 'DTNIQ'),        # Bitcoin futures
    
    # FX
    'US_DOLLAR': ('@DX#C', 'DTNIQ'),       # US Dollar Index
    'USD_INR': ('USDINR', 'NSE'),          # USD/INR (NSE provider)
    
    # International
    'Nikkei_225': ('@NKD#C', 'DTNIQ'),     # Nikkei futures
    
    # India (NSE provider)
    'Nifty_50': ('NIFTY', 'NSE'),          # Nifty 50 Index
    'Nifty_Bank': ('BANKNIFTY', 'NSE'),    # Bank Nifty Index
    
    # China
    'China_ETF': ('FXI', 'BARCHART'),      # iShares China Large-Cap ETF
}

# Provider-specific symbol formats
PROVIDER_CONFIGS = {
    'DTNIQ': {
        'date_format': '%Y%m%d%H%M',
        'requires_encoding': True,
    },
    'cme': {
        'date_format': '%Y%m%d%H%M%S',
        'requires_encoding': True,
    },
}

# ============================================================================
# Retry Decorator
# ============================================================================

def retry(exceptions, tries=3, delay=5, logger=None):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < tries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if logger:
                        logger.warning(f"Attempt {attempt}/{tries} failed: {e}")
                    if attempt < tries:
                        time.sleep(delay * attempt)
                    else:
                        raise
        return wrapper
    return decorator


# ============================================================================
# Core API Functions
# ============================================================================

@retry(Exception, tries=MAX_RETRIES, delay=RETRY_DELAY, logger=log)
def qdl_get_data(
    symbol: str,
    provider: str,
    from_time: datetime,
    to_time: datetime,
    session: Optional[requests.Session] = None
) -> pd.DataFrame:
    """
    Fetch data from QDT Data Lake.
    
    API: GET https://qdl.ai/get_data/{symbol},{provider}/{start}/{end}
    
    Args:
        symbol: Trading symbol (e.g., '@ES#C', 'QCL#')
        provider: Data provider (e.g., 'DTNIQ')
        from_time: Start datetime
        to_time: End datetime
        session: Optional requests session for connection pooling
        
    Returns:
        DataFrame with columns: symbol, time, open, high, low, close, volume
    """
    # Format dates
    start_date = from_time.strftime('%Y%m%d%H%M')
    end_date = to_time.strftime('%Y%m%d%H%M')
    
    # URL encode symbol,provider
    symbol_provider = f"{symbol},{provider}"
    encoded = urllib.parse.quote_plus(symbol_provider)
    
    url = f"{QDL_URL}/get_data/{encoded}/{start_date}/{end_date}"
    headers = {'qml-api-key': QML_API_KEY}
    
    log.debug(f"QDL request: {symbol}/{provider} from {from_time} to {to_time}")
    log.debug(f"URL: {url}")
    
    req = session or requests
    resp = req.get(url, headers=headers, timeout=60)
    
    if resp.status_code != 200:
        log.error(f"QDL API error {resp.status_code}: {resp.text[:200]}")
        return pd.DataFrame()
    
    data = resp.json()
    if not data:
        log.warning(f"QDL returned empty data for {symbol}/{provider}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Process time column
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    log.info(f"QDL fetched {len(df)} rows for {symbol}/{provider}")
    return df


def fetch_nexus_asset(
    asset_name: str,
    years_back: int = 5,
    end_date: Optional[datetime] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch data for a Nexus asset from QDL.
    
    Args:
        asset_name: Nexus asset name (e.g., 'SP500', 'Crude_Oil')
        years_back: How many years of history to fetch
        end_date: End date (defaults to today)
        
    Returns:
        DataFrame with OHLCV data, or None if asset not mapped/available
    """
    if asset_name not in NEXUS_TO_QDL_MAP:
        log.warning(f"Asset {asset_name} not mapped to QDL symbol")
        return None
    
    symbol, provider = NEXUS_TO_QDL_MAP[asset_name]
    
    if end_date is None:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=365 * years_back)
    
    df = qdl_get_data(symbol, provider, start_date, end_date)
    
    if df.empty:
        return None
    
    # Add nexus asset name for reference
    df['nexus_asset'] = asset_name
    
    return df


def fetch_all_nexus_assets(
    years_back: int = 5,
    save_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for all mapped Nexus assets from QDL.
    
    Args:
        years_back: How many years of history
        save_dir: Optional directory to save CSVs
        
    Returns:
        Dict mapping asset_name -> DataFrame
    """
    results = {}
    failed = []
    
    session = requests.Session()
    
    for asset_name, (symbol, provider) in NEXUS_TO_QDL_MAP.items():
        try:
            log.info(f"Fetching {asset_name} ({symbol}/{provider})...")
            df = fetch_nexus_asset(asset_name, years_back)
            
            if df is not None and not df.empty:
                results[asset_name] = df
                log.info(f"  OK: {len(df)} rows, {df.index.min()} to {df.index.max()}")
                
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    csv_path = os.path.join(save_dir, f"{asset_name}_qdl.csv")
                    df.to_csv(csv_path)
                    log.info(f"  Saved to {csv_path}")
            else:
                failed.append(asset_name)
                log.warning(f"  FAIL: no data")
                
        except Exception as e:
            failed.append(asset_name)
            log.error(f"  ERROR: {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    session.close()
    
    log.info(f"\nQDL fetch complete: {len(results)} success, {len(failed)} failed")
    if failed:
        log.info(f"Failed assets: {failed}")
    
    return results


def list_available_providers() -> List[str]:
    """List known data providers."""
    providers = set()
    for _, (symbol, provider) in NEXUS_TO_QDL_MAP.items():
        providers.add(provider)
    return sorted(providers)


# ============================================================================
# CLI / Test
# ============================================================================

def test_connection():
    """Test QDL connection with E-mini S&P 500."""
    print("=" * 60)
    print("QDT Data Lake Connection Test")
    print("=" * 60)
    print(f"URL: {QDL_URL}")
    print(f"API Key: {QML_API_KEY[:8]}...{QML_API_KEY[-4:]}")
    print()
    
    # Test with E-mini S&P 500
    symbol = '@ES#C'
    provider = 'DTNIQ'
    
    print(f"Fetching {symbol} ({provider})...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        df = qdl_get_data(symbol, provider, start_date, end_date)
        
        if df.empty:
            print("FAIL: No data returned")
            return False
        
        print(f"SUCCESS: Got {len(df)} rows")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        print()
        print("Sample data (last 5 rows):")
        print(df.tail())
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_all_assets():
    """Test fetching all mapped assets."""
    print("=" * 60)
    print("Testing All Mapped Assets")
    print("=" * 60)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    results = []
    
    for asset_name, (symbol, provider) in NEXUS_TO_QDL_MAP.items():
        print(f"\n{asset_name} ({symbol}/{provider})...", end=" ")
        
        try:
            df = qdl_get_data(symbol, provider, start_date, end_date)
            
            if df.empty:
                print("NO DATA")
                results.append((asset_name, 'NO DATA', 0))
            else:
                print(f"OK ({len(df)} rows)")
                results.append((asset_name, 'OK', len(df)))
                
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((asset_name, 'ERROR', 0))
        
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    ok = sum(1 for _, status, _ in results if status == 'OK')
    total = len(results)
    print(f"Success: {ok}/{total}")
    
    for asset, status, rows in results:
        print(f"  {asset}: {status} ({rows} rows)")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        test_all_assets()
    else:
        test_connection()
