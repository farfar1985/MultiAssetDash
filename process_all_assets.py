"""
Process All Assets
==================
Fetches data and generates horizon files for all available assets.

Usage: python process_all_assets.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config_sandbox as cfg

# Asset configurations (ID, Name)
ASSETS = [
    ("1625", "SP500"),
    ("1860", "Bitcoin"),
    ("1866", "Crude_Oil"),
    ("1859", "Brent_Oil"),
    ("1861", "Gold"),
    ("269", "NASDAQ"),
    ("336", "DOW_JONES_Mini"),
    ("655", "US_DOLLAR_Index"),
    ("1518", "RUSSEL"),
    ("358", "Nikkei_225"),
    ("1387", "Nifty_Bank"),
    ("1398", "Nifty_50"),
    ("256", "USD_INR"),
    ("477", "GOLD"),  # Different gold ticker?
    ("291", "SPDR_China_ETF"),
    ("1435", "MCX_Copper"),
]

def process_asset(project_id, project_name):
    """Fetch and process horizon data for a single asset."""
    print(f"\n{'='*60}")
    print(f"Processing: {project_name} (ID: {project_id})")
    print(f"{'='*60}")
    
    # Set project config
    cfg.set_project(project_id, project_name)
    
    # Check if we need to fetch data
    if not os.path.exists(cfg.MASTER_DATASET_PATH):
        print(f"  Fetching data from API...")
        import fetch_all_children
        fetch_all_children.fetch_all_models_bulk()
    else:
        print(f"  Data already exists at {cfg.MASTER_DATASET_PATH}")
    
    # Check if we need to generate horizons
    horizons_dir = os.path.join(cfg.DATA_DIR, 'horizons_wide')
    existing_horizons = len([f for f in os.listdir(horizons_dir) if f.endswith('.joblib')]) if os.path.exists(horizons_dir) else 0
    
    if existing_horizons == 0:
        print(f"  Generating horizon data...")
        import golden_engine
        golden_engine.prepare_horizon_data()
    else:
        print(f"  Horizon data exists ({existing_horizons} horizons)")
    
    return True


def main():
    print("=" * 60)
    print("BATCH ASSET PROCESSOR")
    print("=" * 60)
    
    # Check API key
    if not cfg.API_KEY:
        print("ERROR: QML_API_KEY not configured!")
        print("Set it in .env file or environment variable")
        return
    
    results = {}
    for project_id, project_name in ASSETS:
        try:
            success = process_asset(project_id, project_name)
            results[project_name] = "OK" if success else "FAILED"
        except Exception as e:
            print(f"  ERROR: {e}")
            results[project_name] = f"ERROR: {str(e)[:50]}"
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
