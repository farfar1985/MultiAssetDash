# q_ensemble_sandbox/golden_engine.py
# SANDBOXED - Prepares horizon data for ensemble (dynamic horizon discovery)
import pandas as pd
import numpy as np
import os
import sys
import joblib
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config_sandbox as cfg

def prepare_horizon_data():
    """
    Splits the master Parquet file into separate 'wide-format' files (one per horizon).
    DYNAMICALLY discovers all available horizons from n_predict column.
    Each file will have Index=Date, Columns=Model_IDs.
    """
    print(f"--- Sandbox Engine: Preparing Data for {cfg.PROJECT_NAME} ---")
    print(f"Loading master dataset: {cfg.MASTER_DATASET_PATH}")
    
    if not os.path.exists(cfg.MASTER_DATASET_PATH):
        print("ERROR: Master dataset not found. Run fetch_all_children.py first.")
        return

    df = pd.read_parquet(cfg.MASTER_DATASET_PATH)
    
    # Convert time to datetime and normalize
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    
    # Filter for Common Start Date to avoid massive NaNs
    START_DATE = '2025-01-01'
    print(f"Filtering data to start from {START_DATE}...")
    df = df[df['time'] >= pd.to_datetime(START_DATE)]
    
    # DYNAMIC HORIZON DISCOVERY - find all unique n_predict values
    available_horizons = sorted(df['n_predict'].dropna().unique().astype(int).tolist())
    print(f"[INFO] Discovered {len(available_horizons)} horizons: {available_horizons}")
    
    processed_dir = os.path.join(cfg.DATA_DIR, 'horizons_wide')
    os.makedirs(processed_dir, exist_ok=True)
    
    for h in available_horizons:
        print(f"Processing Horizon D+{h}...")
        
        # Filter for this horizon
        sub_df = df[df['n_predict'] == h]
        
        if sub_df.empty:
            print(f"  -> No data for D+{h}. Skipping.")
            continue
            
        # Extract Targets (Actuals)
        targets = sub_df.groupby('time')['target_var_price'].first()
        
        # Pivot Predictions
        predictions = sub_df.pivot(index='time', columns='symbol', values='close_predict')
        
        X = predictions.sort_index()
        y_current = targets.reindex(X.index)
        
        # Shift y to represent FUTURE price
        y_target = y_current.shift(-h)
        
        # Save
        out_file = os.path.join(processed_dir, f"horizon_{h}.joblib")
        
        data_package = {
            'X': X,
            'y': y_target,
            'horizon': h
        }
        
        joblib.dump(data_package, out_file)
        print(f"  -> Saved {X.shape[1]} models, {X.shape[0]} dates to {out_file}")

    print("--- Data Prep Complete ---")

if __name__ == "__main__":
    prepare_horizon_data()

