"""
Asset-Specific Ensemble Framework
=================================
For each asset:
1. Discover available horizons
2. Test all valid horizon combinations (pairs, triplets)
3. Find optimal pairwise slopes config
4. Store results

Author: AmiraB
Date: 2026-02-04
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import json

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def discover_assets() -> List[Dict]:
    """Find all assets with horizons_wide data."""
    assets = []
    for asset_dir in DATA_DIR.iterdir():
        if not asset_dir.is_dir():
            continue
        hw_dir = asset_dir / "horizons_wide"
        if hw_dir.exists():
            horizon_files = list(hw_dir.glob("horizon_*.joblib"))
            if horizon_files:
                # Extract asset ID and name from folder name (e.g., "1866_Crude_Oil")
                parts = asset_dir.name.split("_", 1)
                asset_id = parts[0]
                asset_name = parts[1] if len(parts) > 1 else asset_dir.name
                
                # Extract horizon numbers
                horizons = []
                for f in horizon_files:
                    try:
                        h = int(f.stem.replace("horizon_", ""))
                        horizons.append(h)
                    except:
                        pass
                
                assets.append({
                    'id': asset_id,
                    'name': asset_name,
                    'path': str(asset_dir),
                    'horizons': sorted(horizons),
                    'n_horizons': len(horizons)
                })
    
    return sorted(assets, key=lambda x: x['name'])


def load_horizon(asset_path: str, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load horizon data for an asset."""
    filepath = Path(asset_path) / "horizons_wide" / f"horizon_{horizon}.joblib"
    if not filepath.exists():
        return None, None
    
    data = joblib.load(filepath)
    X = data['X'].values if hasattr(data['X'], 'values') else data['X']
    y = data['y'].values if hasattr(data['y'], 'values') else data['y']
    
    # Remove NaN
    valid = ~np.isnan(y)
    return X[valid], y[valid]


def pairwise_slopes_signal(horizons_data: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                           threshold: float = 0.1) -> np.ndarray:
    """
    Compute pairwise slopes signal from multiple horizons.
    
    Returns signal array: +1 bullish, -1 bearish, 0 no signal
    """
    if len(horizons_data) < 2:
        return np.zeros(1)
    
    # Get common length
    min_len = min(X.shape[0] for X, y in horizons_data.values())
    
    # Mean prediction per horizon
    horizon_preds = {}
    for h, (X, y) in horizons_data.items():
        horizon_preds[h] = X[:min_len].mean(axis=1)
    
    # Get current prices (use any horizon's y)
    current_prices = list(horizons_data.values())[0][1][:min_len]
    
    # Compute pairwise slopes (normalized by price)
    slopes = []
    horizon_list = sorted(horizons_data.keys())
    
    for i, h1 in enumerate(horizon_list):
        for h2 in horizon_list[i+1:]:
            diff = (horizon_preds[h2] - horizon_preds[h1]) / current_prices * 100
            slope = diff / (h2 - h1)  # Normalize by horizon gap
            slopes.append(slope)
    
    if not slopes:
        return np.zeros(min_len)
    
    # Aggregate
    agg = np.mean(slopes, axis=0)
    
    # Apply threshold
    signal = np.where(np.abs(agg) > threshold, np.sign(agg), 0)
    return signal


def backtest(signal: np.ndarray, prices: np.ndarray) -> Dict:
    """Backtest a signal and return metrics."""
    returns = np.diff(prices) / prices[:-1]
    
    min_len = min(len(signal), len(returns))
    signal = signal[:min_len]
    returns = returns[:min_len]
    
    active = signal != 0
    if not active.any():
        return {'Sharpe': 0, 'Return': 0, 'WinRate': 0, 'DA': 0, 'Trades': 0}
    
    strat_ret = signal[active] * returns[active]
    
    sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)
    total_ret = np.prod(1 + strat_ret) - 1
    win_rate = np.mean(strat_ret > 0)
    da = np.mean(signal[active] == np.sign(returns[active]))
    
    return {
        'Sharpe': round(sharpe, 3),
        'Return': round(total_ret * 100, 2),
        'WinRate': round(win_rate * 100, 1),
        'DA': round(da * 100, 1),
        'Trades': int(active.sum())
    }


def test_asset(asset: Dict, max_horizons: int = 4, 
               thresholds: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5]) -> pd.DataFrame:
    """
    Test all horizon combinations for a single asset.
    Returns DataFrame of results.
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {asset['name']} (ID: {asset['id']})")
    print(f"Available horizons: {asset['horizons']}")
    print('='*60)
    
    # Load all horizon data
    horizons_data = {}
    for h in asset['horizons']:
        X, y = load_horizon(asset['path'], h)
        if X is not None and len(X) > 50:  # Need enough data
            horizons_data[h] = (X, y)
            print(f"  Loaded D+{h}: {X.shape[1]} models, {X.shape[0]} days")
    
    if len(horizons_data) < 2:
        print("  ERROR: Need at least 2 horizons for pairwise slopes")
        return pd.DataFrame()
    
    # Get prices for backtesting (use shortest horizon's y)
    base_h = min(horizons_data.keys())
    prices = horizons_data[base_h][1]
    
    # Split train/test
    n = len(prices)
    train_end = int(n * 0.6)
    test_prices = prices[train_end:]
    
    results = []
    horizon_list = sorted(horizons_data.keys())
    
    # Test all combinations of 2, 3, and 4 horizons
    for n_horizons in range(2, min(max_horizons + 1, len(horizon_list) + 1)):
        for combo in combinations(horizon_list, n_horizons):
            # Build subset for this combo
            combo_data = {h: horizons_data[h] for h in combo}
            
            for thresh in thresholds:
                # Compute signal
                signal = pairwise_slopes_signal(combo_data, threshold=thresh)
                
                # Use test period only
                test_signal = signal[train_end:train_end + len(test_prices)]
                
                # Backtest
                metrics = backtest(test_signal, test_prices)
                
                if metrics['Trades'] > 0:  # Only record if we traded
                    results.append({
                        'Asset': asset['name'],
                        'AssetID': asset['id'],
                        'Horizons': '+'.join(f"D{h}" for h in combo),
                        'N_Horizons': n_horizons,
                        'Threshold': thresh,
                        **metrics
                    })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        df = df.sort_values('Sharpe', ascending=False)
        print(f"\n  Top 5 configs for {asset['name']}:")
        print(df.head().to_string(index=False))
    
    return df


def run_full_framework():
    """Run the full asset-specific ensemble framework."""
    print("=" * 70)
    print("ASSET-SPECIFIC ENSEMBLE FRAMEWORK")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Discover assets
    assets = discover_assets()
    print(f"\nFound {len(assets)} assets with horizons_wide data:")
    for a in assets:
        print(f"  - {a['name']}: {a['n_horizons']} horizons {a['horizons']}")
    
    # Test each asset
    all_results = []
    best_configs = {}
    
    for asset in assets:
        if asset['n_horizons'] >= 2:  # Need at least 2 for pairwise
            df = test_asset(asset)
            if len(df) > 0:
                all_results.append(df)
                
                # Store best config
                best = df.iloc[0]
                best_configs[asset['name']] = {
                    'asset_id': asset['id'],
                    'horizons': best['Horizons'],
                    'threshold': best['Threshold'],
                    'sharpe': best['Sharpe'],
                    'return': best['Return'],
                    'win_rate': best['WinRate'],
                    'trades': best['Trades']
                }
    
    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        
        # Save full results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined.to_csv(RESULTS_DIR / f'asset_ensemble_results_{timestamp}.csv', index=False)
        
        # Save best configs
        with open(RESULTS_DIR / 'best_configs_per_asset.json', 'w') as f:
            json.dump(best_configs, f, indent=2)
        
        # Summary table
        print("\n" + "=" * 70)
        print("BEST CONFIG PER ASSET")
        print("=" * 70)
        
        summary = []
        for asset_name, config in best_configs.items():
            summary.append({
                'Asset': asset_name,
                'Best Horizons': config['horizons'],
                'Threshold': config['threshold'],
                'Sharpe': config['sharpe'],
                'Return': f"{config['return']}%",
                'WinRate': f"{config['win_rate']}%"
            })
        
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        
        print(f"\n\nFull results saved to: results/asset_ensemble_results_{timestamp}.csv")
        print(f"Best configs saved to: results/best_configs_per_asset.json")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return best_configs


if __name__ == '__main__':
    run_full_framework()
