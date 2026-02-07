"""
Per-Asset Ensemble Optimizer
============================
Finds the optimal ensemble configuration for each asset independently.

This is THE key differentiator for CME - asset-specific intelligence.

Created: 2026-02-05
Author: AmiraB
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Import from master_ensemble
from master_ensemble import calculate_pairwise_slope_signal


def load_asset_data(asset_dir: str):
    """Load horizon data and current prices for an asset."""
    data_dir = Path(asset_dir) / "horizons_wide"
    
    horizons = {}
    dates_index = None
    
    # Load all available horizons
    for h in range(1, 11):
        f = data_dir / f"horizon_{h}.joblib"
        if f.exists():
            data = joblib.load(f)
            X = data['X'].values if hasattr(data['X'], 'values') else data['X']
            y = data['y'].values if hasattr(data['y'], 'values') else data['y']
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)
            horizons[h] = {'X': np.array(X), 'y': np.array(y), 'horizon': h}
            
            if hasattr(data['X'], 'index'):
                dates_index = data['X'].index
    
    if not horizons:
        return None, None
    
    # Load current prices
    current_prices = None
    parquet_files = ['training_data.parquet', 'master_dataset.parquet']
    
    for pf in parquet_files:
        parquet_path = Path(asset_dir) / pf
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
                
                # Filter to match horizon data dates
                # Use all available history for better model training
                START_DATE = '2020-01-01'  # Extended from 2025-01-01 to use full history
                df = df[df['time'] >= pd.to_datetime(START_DATE)]
                
                prices_series = df.groupby('time')['target_var_price'].first().sort_index()
                
                if dates_index is not None:
                    prices_series = prices_series.reindex(dates_index)
                
                current_prices = prices_series.values
                break
            except Exception as e:
                print(f"  Warning: Could not load prices from {pf}: {e}")
    
    return horizons, current_prices


def backtest_config(horizons: dict, horizon_subset: list, 
                    current_prices: np.ndarray, threshold: float = 0.0,
                    aggregation: str = 'mean') -> dict:
    """
    Backtest a specific configuration.
    Returns metrics dict.
    """
    horizon_subset = [h for h in horizon_subset if h in horizons]
    if len(horizon_subset) < 2:
        return {'sharpe': -999, 'win_rate': 0, 'da': 0, 'total_return': 0}
    
    min_len = min(horizons[h]['X'].shape[0] for h in horizon_subset)
    train_end = int(min_len * 0.7)
    
    eval_horizon = min(horizon_subset)
    y_future = horizons[eval_horizon]['y']
    
    if current_prices is not None and len(current_prices) >= min_len:
        y_current = current_prices[:min_len]
    else:
        # Fallback
        y_current = np.zeros_like(y_future)
        h = eval_horizon
        y_current[h:] = y_future[:-h] if h > 0 else y_future
        y_current[:h] = y_future[:h]
    
    returns = []
    correct_directions = 0
    total_trades = 0
    
    for t in range(train_end, min_len - eval_horizon):
        sig_info = calculate_pairwise_slope_signal(horizons, t, horizon_subset, aggregation)
        net_prob = sig_info['net_prob']
        
        if abs(net_prob) < threshold:
            signal = 0
        else:
            signal = 1 if net_prob > 0 else -1
        
        if y_current[t] != 0:
            actual_return = (y_future[t] - y_current[t]) / y_current[t]
        else:
            actual_return = 0
        
        strategy_return = signal * actual_return
        returns.append(strategy_return)
        
        if signal != 0:
            total_trades += 1
            if np.sign(actual_return) == signal:
                correct_directions += 1
    
    returns = np.array(returns)
    
    if len(returns) == 0 or returns.std() == 0:
        return {'sharpe': -999, 'win_rate': 0, 'da': 0, 'total_return': 0}
    
    periods_per_year = 252 / eval_horizon
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(periods_per_year)
    
    trades = returns[returns != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0
    da = correct_directions / total_trades if total_trades > 0 else 0
    total_return = returns.sum()
    
    return {
        'sharpe': round(sharpe, 3),
        'win_rate': round(win_rate * 100, 2),
        'da': round(da * 100, 2),
        'total_return': round(total_return * 100, 2),
        'n_trades': total_trades
    }


def generate_horizon_combinations(available_horizons: list, min_size: int = 2, max_size: int = 5):
    """Generate all horizon combinations to test."""
    combos = []
    for size in range(min_size, min(max_size + 1, len(available_horizons) + 1)):
        for combo in combinations(available_horizons, size):
            combos.append(list(combo))
    return combos


def optimize_asset(asset_name: str, asset_dir: str) -> dict:
    """
    Find optimal ensemble configuration for a single asset.
    """
    print(f"\n{'='*60}")
    print(f"Optimizing: {asset_name}")
    print(f"{'='*60}")
    
    horizons, prices = load_asset_data(asset_dir)
    
    if horizons is None:
        print(f"  ERROR: No horizon data found for {asset_name}")
        return None
    
    available_horizons = sorted(horizons.keys())
    print(f"  Available horizons: {available_horizons}")
    print(f"  Data points: {horizons[available_horizons[0]]['X'].shape[0]}")
    print(f"  Current prices loaded: {prices is not None}")
    
    # Generate all combinations to test
    combos = generate_horizon_combinations(available_horizons)
    print(f"  Testing {len(combos)} horizon combinations...")
    
    # Test different configurations
    results = []
    aggregations = ['mean', 'median']
    thresholds = [0.0, 0.1, 0.2, 0.3]
    
    for combo in combos:
        for agg in aggregations:
            for thresh in thresholds:
                metrics = backtest_config(horizons, combo, prices, thresh, agg)
                results.append({
                    'horizons': combo,
                    'aggregation': agg,
                    'threshold': thresh,
                    **metrics
                })
    
    # Convert to DataFrame and find best
    df = pd.DataFrame(results)
    
    # Filter out failed configs
    df = df[df['sharpe'] > -900]
    
    if len(df) == 0:
        print(f"  ERROR: No valid configurations found")
        return None
    
    # Find best by Sharpe
    best_sharpe = df.loc[df['sharpe'].idxmax()]
    
    # Find best by Win Rate (among positive Sharpe)
    positive_sharpe = df[df['sharpe'] > 0]
    best_winrate = positive_sharpe.loc[positive_sharpe['win_rate'].idxmax()] if len(positive_sharpe) > 0 else None
    
    # Find best by DA (among positive Sharpe)
    best_da = positive_sharpe.loc[positive_sharpe['da'].idxmax()] if len(positive_sharpe) > 0 else None
    
    print(f"\n  BEST BY SHARPE:")
    print(f"    Horizons: {best_sharpe['horizons']}")
    print(f"    Aggregation: {best_sharpe['aggregation']}")
    print(f"    Threshold: {best_sharpe['threshold']}")
    print(f"    Sharpe: {best_sharpe['sharpe']}")
    print(f"    Win Rate: {best_sharpe['win_rate']}%")
    print(f"    DA: {best_sharpe['da']}%")
    
    if best_winrate is not None:
        print(f"\n  BEST BY WIN RATE (positive Sharpe):")
        print(f"    Horizons: {best_winrate['horizons']}")
        print(f"    Win Rate: {best_winrate['win_rate']}%")
        print(f"    Sharpe: {best_winrate['sharpe']}")
    
    # Build optimal config
    optimal_config = {
        'asset': asset_name,
        'optimized_at': datetime.now().isoformat(),
        'available_horizons': available_horizons,
        'best_config': {
            'horizons': best_sharpe['horizons'],
            'aggregation': best_sharpe['aggregation'],
            'threshold': best_sharpe['threshold'],
            'metrics': {
                'sharpe': best_sharpe['sharpe'],
                'win_rate': best_sharpe['win_rate'],
                'da': best_sharpe['da'],
                'total_return': best_sharpe['total_return'],
                'n_trades': int(best_sharpe['n_trades'])
            }
        },
        'all_results_summary': {
            'total_configs_tested': len(df),
            'positive_sharpe_configs': len(df[df['sharpe'] > 0]),
            'best_sharpe': float(df['sharpe'].max()),
            'best_win_rate': float(df['win_rate'].max()),
            'best_da': float(df['da'].max())
        }
    }
    
    return optimal_config


def run_all_assets():
    """Run optimization for all assets with data."""
    data_dir = Path("data")
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Find assets with horizon data
    assets = []
    for d in data_dir.iterdir():
        if d.is_dir() and (d / "horizons_wide").exists():
            horizon_files = list((d / "horizons_wide").glob("*.joblib"))
            if horizon_files:
                # Extract asset name from directory name (e.g., "1866_Crude_Oil" -> "crude_oil")
                parts = d.name.split('_', 1)
                asset_name = parts[1].lower().replace(' ', '_') if len(parts) > 1 else d.name.lower()
                assets.append((asset_name, str(d)))
    
    if not assets:
        print("ERROR: No assets with horizon data found!")
        return
    
    print(f"Found {len(assets)} assets with data: {[a[0] for a in assets]}")
    
    all_results = {}
    
    for asset_name, asset_dir in assets:
        config = optimize_asset(asset_name, asset_dir)
        
        if config:
            all_results[asset_name] = config
            
            # Save individual config
            config_file = configs_dir / f"optimal_{asset_name}.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  Saved: {config_file}")
    
    # Save summary
    summary = {
        'optimized_at': datetime.now().isoformat(),
        'assets': list(all_results.keys()),
        'results': {
            name: {
                'best_horizons': cfg['best_config']['horizons'],
                'sharpe': cfg['best_config']['metrics']['sharpe'],
                'win_rate': cfg['best_config']['metrics']['win_rate']
            }
            for name, cfg in all_results.items()
        }
    }
    
    summary_file = configs_dir / "optimization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nSummary saved to: {summary_file}")
    
    print("\nRESULTS SUMMARY:")
    for name, cfg in all_results.items():
        m = cfg['best_config']['metrics']
        h = cfg['best_config']['horizons']
        print(f"  {name.upper()}: Horizons={h}, Sharpe={m['sharpe']}, WR={m['win_rate']}%")
    
    return all_results


if __name__ == "__main__":
    results = run_all_assets()
