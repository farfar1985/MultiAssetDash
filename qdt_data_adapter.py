"""
QDT Data Lake Adapter for Nexus v2
===================================
Bridges Nexus ensemble system with QDT's quantum_ml Data Lake.

This adapter provides a clean interface for Nexus to consume data from
QDT's production infrastructure (200K+ financial datasets).

Author: AmiraB
Date: 2026-02-05
Status: POC - Integration Phase 1

Usage:
    from qdt_data_adapter import QDTDataAdapter
    
    adapter = QDTDataAdapter()
    df = adapter.get_asset_data("crude_oil", days=365)
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

log = logging.getLogger(__name__)

# QDT Data Lake Configuration
QDL_URL = os.environ.get('QDL_URL', 'https://qdl.qdt.ai')
QML_API_KEY = os.environ.get('QML_API_KEY', 'Fp6c_bw7qkFdb_LhYVJmGA')
QML_API_TAG = os.environ.get('QML_API_TAG', 'nexus-v2')

# Asset mapping: Nexus asset names → QDT Data Lake symbols/providers
ASSET_MAPPING = {
    # Commodities
    "crude_oil": {"symbol": "CL", "provider": "cme"},
    "gold": {"symbol": "GC", "provider": "cme"},
    "natural_gas": {"symbol": "NG", "provider": "cme"},
    "copper": {"symbol": "HG", "provider": "cme"},
    "silver": {"symbol": "SI", "provider": "cme"},
    "platinum": {"symbol": "PL", "provider": "cme"},
    
    # Indices
    "sp500": {"symbol": "ES", "provider": "cme"},
    "nasdaq": {"symbol": "NQ", "provider": "cme"},
    "dow_jones": {"symbol": "YM", "provider": "cme"},
    "russell2000": {"symbol": "RTY", "provider": "cme"},
    
    # Crypto
    "bitcoin": {"symbol": "BTC", "provider": "cme"},
    "ethereum": {"symbol": "ETH", "provider": "cme"},
    
    # Forex
    "eurusd": {"symbol": "6E", "provider": "cme"},
    "gbpusd": {"symbol": "6B", "provider": "cme"},
    "usdjpy": {"symbol": "6J", "provider": "cme"},
}


class QDTDataAdapter:
    """
    Adapter for fetching data from QDT's Data Lake.
    
    Provides a simple interface for Nexus to consume production-grade
    financial data without needing to understand the underlying infrastructure.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize the adapter.
        
        Args:
            api_key: QML API key (defaults to env var or hardcoded fallback)
            base_url: Data Lake URL (defaults to env var or production URL)
        """
        self.api_key = api_key or QML_API_KEY
        self.base_url = base_url or QDL_URL
        self.session = requests.Session()
        self._cache: Dict[str, pd.DataFrame] = {}
        
    def get_asset_data(
        self,
        asset: str,
        days: int = 365,
        frequency: str = 'daily',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch asset data from QDT Data Lake.
        
        Args:
            asset: Nexus asset name (e.g., "crude_oil", "bitcoin")
            days: Number of days of history to fetch
            frequency: 'daily' or 'hourly'
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: datetime
        """
        cache_key = f"{asset}_{days}_{frequency}"
        
        if use_cache and cache_key in self._cache:
            log.info(f"Using cached data for {asset}")
            return self._cache[cache_key]
        
        # Map Nexus asset name to QDT symbol/provider
        if asset not in ASSET_MAPPING:
            log.warning(f"Unknown asset {asset}, falling back to CSV")
            return self._fallback_to_csv(asset)
        
        mapping = ASSET_MAPPING[asset]
        symbol = mapping["symbol"]
        provider = mapping["provider"]
        
        # Calculate time range
        to_time = datetime.now()
        from_time = to_time - timedelta(days=days)
        
        try:
            df = self._fetch_from_datalake(
                symbol=symbol,
                provider=provider,
                from_time=from_time,
                to_time=to_time,
                frequency=frequency
            )
            
            if not df.empty:
                self._cache[cache_key] = df
                log.info(f"Fetched {len(df)} rows for {asset} from Data Lake")
                return df
                
        except Exception as e:
            log.error(f"Data Lake fetch failed for {asset}: {e}")
        
        # Fallback to local CSV
        return self._fallback_to_csv(asset)
    
    def _fetch_from_datalake(
        self,
        symbol: str,
        provider: str,
        from_time: datetime,
        to_time: datetime,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Make actual API call to QDT Data Lake.
        
        This mirrors the quantum_ml.datalake.datalake_get_symbol() function
        but works standalone without the full quantum_ml infrastructure.
        """
        api_endpoint = f"{self.base_url}/qml/v1/get_symbol"
        
        params = {
            'key': self.api_key,
            'tag': QML_API_TAG,
            'symbol': symbol,
            'provider': provider,
            'bartype': 'day' if frequency == 'daily' else 'hour',
            't1': from_time.strftime('%Y%m%d%H%M%S'),
            't2': to_time.strftime('%Y%m%d%H%M%S'),
        }
        
        log.debug(f"Fetching from Data Lake: {symbol} ({provider})")
        
        resp = self.session.get(api_endpoint, params=params, timeout=30)
        
        if resp.status_code != 200:
            raise Exception(f"Data Lake API error: {resp.status_code}")
        
        df = pd.DataFrame(resp.json())
        
        if df.empty:
            return df
        
        # Process response (same as quantum_ml)
        df['time'] = pd.to_datetime(df['time'])
        if frequency == 'daily':
            df['time'] = df['time'].dt.strftime('%Y-%m-%d')
            df['time'] = pd.to_datetime(df['time'])
        
        df.set_index('time', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _fallback_to_csv(self, asset: str) -> pd.DataFrame:
        """
        Fallback to local CSV files when Data Lake is unavailable.
        
        This ensures Nexus continues to work during development/testing
        even without Data Lake connectivity.
        """
        # Try to find local CSV
        csv_paths = [
            f"data/{asset}/predictions.csv",
            f"data/{asset}_predictions.csv",
            f"data/{asset}.csv",
        ]
        
        for path in csv_paths:
            if os.path.exists(path):
                log.info(f"Using fallback CSV: {path}")
                return pd.read_csv(path, parse_dates=['date'], index_col='date')
        
        log.error(f"No data source found for {asset}")
        return pd.DataFrame()
    
    def list_available_assets(self) -> list:
        """List all assets available through the adapter."""
        return list(ASSET_MAPPING.keys())
    
    def get_asset_info(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get mapping info for an asset."""
        return ASSET_MAPPING.get(asset)
    
    def test_connection(self) -> bool:
        """
        Test Data Lake connectivity.
        
        Returns True if connection is successful, False otherwise.
        """
        try:
            # Try a minimal request
            resp = self.session.get(
                f"{self.base_url}/health",
                timeout=10
            )
            return resp.status_code == 200
        except Exception as e:
            log.error(f"Connection test failed: {e}")
            return False


# Quick test when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("QDT Data Lake Adapter - Connection Test")
    print("=" * 50)
    
    adapter = QDTDataAdapter()
    
    print(f"API Key: {adapter.api_key[:10]}...")
    print(f"Base URL: {adapter.base_url}")
    print(f"Available assets: {len(adapter.list_available_assets())}")
    print()
    
    # Test connection
    print("Testing connection...")
    if adapter.test_connection():
        print("✅ Connection successful!")
        
        # Try to fetch one asset
        print("\nFetching crude_oil data (30 days)...")
        df = adapter.get_asset_data("crude_oil", days=30)
        
        if not df.empty:
            print(f"✅ Got {len(df)} rows")
            print(df.tail())
        else:
            print("⚠️ No data returned (may need credentials)")
    else:
        print("❌ Connection failed (expected without VPN/credentials)")
        print("   Adapter will fallback to local CSV files")
