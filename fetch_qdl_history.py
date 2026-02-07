"""
Fetch multi-year historical data from QDT Data Lake for all assets.
Saves to data/qdl_history/ directory.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from qdl_client import qdl_get_data, NEXUS_TO_QDL_MAP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Only DTNIQ assets work reliably
DTNIQ_ASSETS = {
    'SP500': ('@ES#C', 'DTNIQ'),
    'NASDAQ': ('@NQ#C', 'DTNIQ'),
    'Dow_Jones': ('@YM#C', 'DTNIQ'),
    'Russell_2000': ('@RTY#C', 'DTNIQ'),
    'Crude_Oil': ('QCL#', 'DTNIQ'),
    'Brent_Oil': ('QBZ#', 'DTNIQ'),
    'GOLD': ('QGC#', 'DTNIQ'),
    'MCX_Copper': ('QHG#', 'DTNIQ'),
    'Natural_Gas': ('QNG#', 'DTNIQ'),
    'Bitcoin': ('@BTC#C', 'DTNIQ'),
    'US_DOLLAR': ('@DX#C', 'DTNIQ'),
    'Nikkei_225': ('@NKD#C', 'DTNIQ'),
}

def fetch_all_history(years: int = 5, output_dir: str = "data/qdl_history"):
    """Fetch multi-year history for all DTNIQ assets."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    print("=" * 60)
    print(f"QDT DATA LAKE - Fetching {years} years of history")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("=" * 60)
    
    results = []
    
    for asset_name, (symbol, provider) in DTNIQ_ASSETS.items():
        print(f"\n{asset_name} ({symbol})...", end=" ", flush=True)
        
        try:
            df = qdl_get_data(symbol, provider, start_date, end_date)
            
            if df.empty:
                print("NO DATA")
                results.append((asset_name, 'NO DATA', 0, None))
                continue
            
            # Save to CSV
            csv_path = os.path.join(output_dir, f"{asset_name}.csv")
            df.to_csv(csv_path)
            
            date_range = f"{df.index.min().date()} to {df.index.max().date()}"
            print(f"OK - {len(df)} rows ({date_range})")
            results.append((asset_name, 'OK', len(df), date_range))
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((asset_name, 'ERROR', 0, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_rows = 0
    ok_count = 0
    
    for asset, status, rows, info in results:
        if status == 'OK':
            ok_count += 1
            total_rows += rows
        print(f"  {asset}: {status} - {rows} rows" + (f" ({info})" if info and status == 'OK' else ""))
    
    print(f"\nTotal: {ok_count}/{len(results)} assets, {total_rows:,} rows")
    print(f"Output: {os.path.abspath(output_dir)}")
    
    return results


if __name__ == "__main__":
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    fetch_all_history(years=years)
