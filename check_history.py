"""Check how much historical data is available in parquet files."""
import pandas as pd
from pathlib import Path

data_dir = Path("data")

for asset_dir in sorted(data_dir.iterdir()):
    if not asset_dir.is_dir():
        continue
    
    parquet = asset_dir / "training_data.parquet"
    if not parquet.exists():
        parquet = asset_dir / "master_dataset.parquet"
    
    if parquet.exists():
        try:
            df = pd.read_parquet(parquet)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                min_date = df['time'].min()
                max_date = df['time'].max()
                days = (max_date - min_date).days
                years = days / 365.25
                print(f"{asset_dir.name}: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({years:.1f} years, {len(df)} rows)")
        except Exception as e:
            print(f"{asset_dir.name}: Error - {e}")
