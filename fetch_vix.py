"""Fetch VIX data from QDL."""
from qdl_client import qdl_get_data
from datetime import datetime, timedelta
import os

end = datetime.now()
start = end - timedelta(days=365*5)

print('Fetching VIX from QDL...')
df = qdl_get_data('@VX#C', 'DTNIQ', start, end)
if not df.empty:
    os.makedirs('data/qdl_history', exist_ok=True)
    df.to_csv('data/qdl_history/VIX.csv')
    print(f'Saved {len(df)} rows to VIX.csv')
    print(f'Latest VIX: {df["close"].iloc[-1]:.2f}')
    print(f'Date range: {df.index[0]} to {df.index[-1]}')
else:
    print('No VIX data returned')
