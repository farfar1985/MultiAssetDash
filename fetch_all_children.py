# q_ensemble_sandbox/fetch_all_children.py
# SANDBOXED - Fetches data for Bitcoin project and extracts price cache
import os
import sys
import json
import requests
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config_sandbox as cfg

def get_headers():
    return {
        'qml-api-key': cfg.API_KEY,
        'Content-Type': 'application/json'
    }

def fetch_all_models_bulk():
    """
    Fetches ALL models (Parents + Children) for the project.
    Also extracts price history from target_var_price for visualization.
    """
    print(f"--- SANDBOX BULK FETCH: Downloading All Models for {cfg.PROJECT_NAME} (Project {cfg.PROJECT_ID}) ---")
    url = f"{cfg.API_BASE_URL}/get_qml_models/{cfg.PROJECT_ID}"
    
    try:
        print(f"Requesting: {url}")
        resp = requests.get(url, headers=get_headers(), timeout=300)
        
        if resp.status_code != 200:
            print(f"!!! FAILED: Status {resp.status_code}")
            print(resp.text[:500])
            return None

        print("Download complete. Parsing JSON...")
        data = resp.json()
        df = pd.DataFrame(data)
        
        print(f"--- SUCCESS! ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        if not df.empty:
            print("\nSample Rows:")
            print(df.head())
            
            # Save the Raw Data
            print(f"\nSaving to {cfg.MASTER_DATASET_PATH}...")
            df.to_parquet(cfg.MASTER_DATASET_PATH)
            print("Saved.")
            
            # Save unique model IDs
            if 'symbol' in df.columns:
                ids = df['symbol'].unique()
                print(f"\nUnique Model IDs found: {len(ids)}")
                with open(os.path.join(cfg.DATA_DIR, 'model_ids.json'), 'w') as f:
                    json.dump([str(x) for x in ids], f, indent=2)
            
            # --- EXTRACT PRICE HISTORY ---
            print("\n--- Extracting Price History from target_var_price ---")
            extract_price_cache(df)
                
        return df

    except Exception as e:
        print(f"!!! ERROR during bulk fetch: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_price_cache(df):
    """
    Extracts unique (date, price) pairs from target_var_price field.
    Saves as price_history_cache.json for visualization.
    """
    try:
        # Get unique date-price pairs (target_var_price is the actual price at that time)
        df_copy = df.copy()
        df_copy['time'] = pd.to_datetime(df_copy['time']).dt.tz_localize(None)
        
        # Group by time, take first target_var_price (they should be identical for same date)
        prices = df_copy.groupby('time').agg({
            'target_var_price': 'first'
        }).reset_index()
        
        prices.columns = ['date', 'close']
        prices = prices.sort_values('date').dropna()
        
        # Add OHLV columns (using close as proxy since we only have close)
        prices['open'] = prices['close']
        prices['high'] = prices['close']
        prices['low'] = prices['close']
        prices['volume'] = 0
        
        # Format for JSON
        prices['date'] = prices['date'].dt.strftime('%Y-%m-%dT%H:%M:%S+0000')
        
        # Save
        prices[['date', 'open', 'high', 'low', 'close', 'volume']].to_json(
            cfg.PRICE_CACHE_PATH, orient='records', indent=4
        )
        
        print(f"Price cache saved: {len(prices)} records to {cfg.PRICE_CACHE_PATH}")
        print(f"Date range: {prices['date'].iloc[0]} to {prices['date'].iloc[-1]}")
        
    except Exception as e:
        print(f"Warning: Could not extract price cache: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_all_models_bulk()

