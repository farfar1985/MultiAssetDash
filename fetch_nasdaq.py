"""Quick script to fetch NASDAQ data"""
import sys
sys.path.insert(0, '.')

import config_sandbox as cfg

# Set to NASDAQ
cfg.set_project("269", "NASDAQ")

# Fetch
import fetch_all_children
fetch_all_children.fetch_all_models_bulk()

# Generate horizons
import golden_engine
golden_engine.prepare_horizon_data()

print("\nNASDAQ data fetched and processed!")
