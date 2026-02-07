"""Check what data is available locally."""
import pandas as pd
from pathlib import Path

data_dir = Path("data")

print("LOCAL DATA INVENTORY")
print("=" * 60)

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
                
                # Get rows per year
                df['year'] = df['time'].dt.year
                year_counts = df.groupby('year').size()
                
                print(f"\n{asset_dir.name}")
                print(f"  Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                print(f"  History: {years:.1f} years ({len(df):,} rows)")
                print(f"  Years: {list(year_counts.index)}")
        except Exception as e:
            print(f"\n{asset_dir.name}: Error - {e}")
    else:
        # Check if there's CSV data
        csv_files = list(asset_dir.glob("*.csv"))
        if csv_files:
            try:
                df = pd.read_csv(csv_files[0])
                print(f"\n{asset_dir.name}")
                print(f"  CSV only: {len(df)} rows (no parquet)")
            except:
                print(f"\n{asset_dir.name}: CSV exists but couldn't read")
        else:
            print(f"\n{asset_dir.name}: NO DATA")
