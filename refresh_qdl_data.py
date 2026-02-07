#!/usr/bin/env python3
"""
Refresh QDL data for all Nexus assets.
Fetches latest data from QDT Data Lake and updates local CSVs.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from qdl_client import qdl_get_data, NEXUS_TO_QDL_MAP

DATA_DIR = Path(__file__).parent / "data" / "qdl_history"

def refresh_all(days: int = 7):
    """Refresh all assets with recent data."""
    end = datetime.now()
    start = end - timedelta(days=days)
    
    print(f"Refreshing data from {start.date()} to {end.date()}")
    print("=" * 60)
    
    for asset_key, (symbol, provider) in NEXUS_TO_QDL_MAP.items():
        try:
            df = qdl_get_data(symbol, provider, start, end)
            if df is None or df.empty:
                print(f"  {asset_key}: no new data from QDL")
                continue
            
            # Reset index to make 'time' a column
            df = df.reset_index()
            
            csv_path = DATA_DIR / f"{asset_key}.csv"
            if csv_path.exists():
                existing = pd.read_csv(csv_path, parse_dates=['time'])
                df = pd.concat([existing, df]).drop_duplicates(subset=['time']).sort_values('time')
            
            df.to_csv(csv_path, index=False)
            latest_date = df.iloc[-1]['time']
            print(f"  {asset_key}: {len(df)} total rows, latest: {latest_date}")
            
        except Exception as e:
            print(f"  {asset_key}: ERROR - {e}")
    
    print("=" * 60)
    print("Refresh complete!")

if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    refresh_all(days)
