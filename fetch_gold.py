"""Quick script to fetch Gold data"""
import sys
sys.path.insert(0, '.')

import config_sandbox as cfg

# Set to Gold
cfg.set_project("1861", "Gold")

# Fetch
import fetch_all_children
fetch_all_children.fetch_all_models_bulk()

# Generate horizons
import golden_engine
golden_engine.prepare_horizon_data()

print("\nGold data fetched and processed!")
