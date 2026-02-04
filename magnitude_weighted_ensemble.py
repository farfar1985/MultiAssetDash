# magnitude_weighted_ensemble.py
# Improved Multi-Horizon Ensemble with Magnitude Weighting
# Key insight: Weight horizon signals by their forecast magnitude

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from ensemble_methods import EnsembleMethods

DATA_DIR = r"C:\Users\William Dennis\projects\nexus\data"
OUTPUT_DIR = r"C:\Users\William Dennis\projects\nexus\results"


def load_horizon_data(asset_id: int, asset_name: str, horizon: int) -> tuple:
    """Load predictions and actuals for a specific horizon."""
    folder = f"{asset_id}_{asset_name}"
    path = os.path.join(DATA_DIR, folder, 'horizons_wide', f'horizon_{horizon}.joblib')
    if not os.path.exists(path):
        return None, None
    data = joblib.load(path)
    return data['X'], data['y']


def compute_magnitude_weighted_ensemble(asset_name: str, asset_id: int, 
                                         horizons: list, min_threshold: float) -> dict:
    """
    Build multi-horizon ensemble where each horizon's signal is weighted
    by its forecast magnitude. Bigger predicted moves get more weight.
    """
    
    # Load all horizon data
    all_data = {}
    for h in horizons:
        X, y = load_horizon_data(asset_id, asset_name, h)
        if X is not None and len(X) > 60:
            all_data[h] = {'X': X, 'y': y}
    
    if len(all_data) < 2:
        return None
    
    # Find common dates
    common_dates = None
    for h, data in all_data.items():
        dates = set(data['X'].index)
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates.intersection(dates)
    
    common_dates = sorted(list(common_dates))
    if len(common_dates) < 90:  # Need at least 90 days
        return None
    
    # Split: 70% train, 30% test
    split_idx = int(len(common_dates) * 0.7)
    train_dates = common_dates[:split_idx]
    test_dates = common_dates[split_idx:]
    
    if len(test_dates) < 30:
        return None
    
    # Get ensemble weights for each horizon using training data
    ensemble = EnsembleMethods(lookback_window=60)
    horizon_results = {}
    
    for h, data in all_data.items():
        X_train = data['X'].loc[train_dates]
        y_train = data['y'].loc[train_dates]
        X_test = data['X'].loc[test_dates]
        
        try:
            # Get top-k weights
            weights = ensemble.top_k_by_sharpe(X_train, y_train, top_pct=0.1)
            
            # Generate ensemble predictions
            train_pred = (X_train * weights).sum(axis=1)
            test_pred = (X_test * weights).sum(axis=1)
            
            # Calculate forecast changes (magnitude)
            train_changes = train_pred.diff().abs()
            test_changes = test_pred.diff().abs()
            
            # Calculate direction signals
            train_signals = np.sign(train_pred.diff())
            test_signals = np.sign(test_pred.diff())
            
            horizon_results[h] = {
                'train_signals': train_signals,
                'train_magnitudes': train_changes,
                'test_signals': test_signals,
                'test_magnitudes': test_changes,
                'test_predictions': test_pred
            }
        except Exception as e:
            print(f"    Error on D+{h}: {e}", flush=True)
            continue
    
    if len(horizon_results) < 2:
        return None
    
    # === MAGNITUDE-WEIGHTED VOTING ===
    
    # Method 1: Simple magnitude weighting
    test_signal_df = pd.DataFrame({h: r['test_signals'] for h, r in horizon_results.items()})
    test_mag_df = pd.DataFrame({h: r['test_magnitudes'] for h, r in horizon_results.items()})
    
    # Weight each signal by its magnitude
    weighted_votes = (test_signal_df * test_mag_df)
    
    # Sum weighted votes and normalize
    total_weighted = weighted_votes.sum(axis=1)
    total_magnitude = test_mag_df.sum(axis=1)
    
    # Final signal: weighted consensus
    consensus_raw = total_weighted / total_magnitude.replace(0, np.nan)
    consensus_signal = consensus_raw.apply(lambda x: 1 if x > 0.2 else (-1 if x < -0.2 else 0))
    
    # Average magnitude-weighted prediction
    pred_df = pd.DataFrame({h: r['test_predictions'] for h, r in horizon_results.items()})
    avg_forecast = (pred_df * test_mag_df).sum(axis=1) / total_magnitude.replace(0, np.nan)
    
    # Get actuals
    first_h = list(horizon_results.keys())[0]
    actuals = all_data[first_h]['y'].loc[test_dates]
    
    # === COMPUTE METRICS ===
    
    price_changes = actuals.diff().dropna()
    sig_aligned = consensus_signal.loc[price_changes.index]
    fcst_aligned = avg_forecast.loc[price_changes.index]
    
    # Basic returns
    returns = sig_aligned.shift(1) * price_changes.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns) < 10:
        return None
    
    # Sharpe
    sharpe = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252) if returns.std() > 0 else 0
    
    # Directional accuracy
    actual_dir = np.sign(price_changes)
    pred_dir = sig_aligned.shift(1)
    da = (pred_dir == actual_dir).dropna().mean() * 100
    
    # Forecast magnitude
    forecast_changes = fcst_aligned.diff().abs()
    avg_forecast_mag = forecast_changes.mean()
    
    # Significant moves (> threshold)
    sig_mask = price_changes.abs() > min_threshold
    if sig_mask.sum() > 5:
        sig_da = (pred_dir.shift(1)[sig_mask] == actual_dir[sig_mask]).mean() * 100
        n_significant = sig_mask.sum()
    else:
        sig_da = np.nan
        n_significant = 0
    
    # Practical Sharpe (only on significant forecasts)
    large_fcst_mask = forecast_changes > min_threshold
    large_returns = returns[large_fcst_mask.reindex(returns.index, fill_value=False)]
    if len(large_returns) > 10 and large_returns.std() > 0:
        practical_sharpe = (large_returns.mean() / large_returns.std(ddof=1)) * np.sqrt(252)
    else:
        practical_sharpe = np.nan
    
    # Total return
    cumulative = (1 + returns).cumprod()
    total_return = (cumulative.iloc[-1] - 1) * 100
    
    # Max drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min() * 100
    
    # Utility score
    utility = 0
    if not np.isnan(practical_sharpe):
        utility += 0.4 * min(practical_sharpe / 2, 1)
    if not np.isnan(sig_da):
        utility += 0.3 * (sig_da / 100)
    if avg_forecast_mag > 0:
        utility += 0.3 * min(avg_forecast_mag / (min_threshold * 3), 1)
    
    return {
        'asset': asset_name,
        'horizons': str(list(horizon_results.keys())),
        'n_horizons': len(horizon_results),
        'sharpe': round(sharpe, 3),
        'practical_sharpe': round(practical_sharpe, 3) if not np.isnan(practical_sharpe) else None,
        'da': round(da, 1),
        'significant_da': round(sig_da, 1) if not np.isnan(sig_da) else None,
        'n_significant': n_significant,
        'avg_forecast_mag': round(avg_forecast_mag, 2),
        'total_return': round(total_return, 1),
        'max_dd': round(max_dd, 1),
        'utility_score': round(utility, 3),
        'n_trades': len(returns)
    }


def main():
    print("=" * 70, flush=True)
    print("  MAGNITUDE-WEIGHTED MULTI-HORIZON ENSEMBLE", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)
    
    ASSETS = {
        'Crude_Oil': {'id': 1866, 'horizons': [5, 6, 7, 8, 9, 10], 'threshold': 0.50},
        'SP500': {'id': 1625, 'horizons': [5, 8, 13, 21, 34], 'threshold': 10.0},
        'Bitcoin': {'id': 1860, 'horizons': [5, 8, 10, 13], 'threshold': 500.0},
    }
    
    all_results = []
    
    for asset_name, info in ASSETS.items():
        print(f"\n{'='*70}", flush=True)
        print(f"  {asset_name}", flush=True)
        print(f"{'='*70}", flush=True)
        
        # Test different horizon combinations
        horizons = info['horizons']
        threshold = info['threshold']
        
        # All horizons
        print(f"\n  Testing ALL horizons: {horizons}", flush=True)
        result = compute_magnitude_weighted_ensemble(
            asset_name, info['id'], horizons, threshold
        )
        if result:
            result['config'] = 'all_horizons'
            all_results.append(result)
            print(f"    Sharpe: {result['sharpe']:.2f} | Practical: {result['practical_sharpe']} | Utility: {result['utility_score']:.3f}", flush=True)
        
        # Mid-to-long only (skip shortest)
        mid_long = horizons[1:]
        print(f"\n  Testing MID-LONG horizons: {mid_long}", flush=True)
        result = compute_magnitude_weighted_ensemble(
            asset_name, info['id'], mid_long, threshold
        )
        if result:
            result['config'] = 'mid_long'
            all_results.append(result)
            print(f"    Sharpe: {result['sharpe']:.2f} | Practical: {result['practical_sharpe']} | Utility: {result['utility_score']:.3f}", flush=True)
        
        # Long only (last 3)
        long_only = horizons[-3:] if len(horizons) >= 3 else horizons
        print(f"\n  Testing LONG horizons: {long_only}", flush=True)
        result = compute_magnitude_weighted_ensemble(
            asset_name, info['id'], long_only, threshold
        )
        if result:
            result['config'] = 'long_only'
            all_results.append(result)
            print(f"    Sharpe: {result['sharpe']:.2f} | Practical: {result['practical_sharpe']} | Utility: {result['utility_score']:.3f}", flush=True)
        
        # Best pairs from practical test
        if asset_name == 'Crude_Oil':
            pairs = [[7, 8], [8, 9], [8, 10]]
        elif asset_name == 'SP500':
            pairs = [[8, 13], [13, 21], [8, 21]]
        else:
            pairs = [[5, 10], [8, 10]]
        
        for pair in pairs:
            print(f"\n  Testing pair: D+{pair[0]} + D+{pair[1]}", flush=True)
            result = compute_magnitude_weighted_ensemble(
                asset_name, info['id'], pair, threshold
            )
            if result:
                result['config'] = f'pair_{pair[0]}_{pair[1]}'
                all_results.append(result)
                print(f"    Sharpe: {result['sharpe']:.2f} | Practical: {result['practical_sharpe']} | Utility: {result['utility_score']:.3f}", flush=True)
    
    # Save results
    df = pd.DataFrame(all_results)
    output_path = os.path.join(OUTPUT_DIR, f'magnitude_weighted_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}", flush=True)
    print(f"  MAGNITUDE-WEIGHTED ENSEMBLE COMPLETE", flush=True)
    print(f"  Results: {output_path}", flush=True)
    print(f"{'='*70}", flush=True)
    
    # Best by utility
    print("\n  TOP CONFIGURATIONS BY UTILITY:", flush=True)
    print("-" * 70, flush=True)
    
    for asset in ['Crude_Oil', 'SP500', 'Bitcoin']:
        asset_df = df[df['asset'] == asset]
        if len(asset_df) > 0:
            best = asset_df.loc[asset_df['utility_score'].idxmax()]
            print(f"\n  {asset}:", flush=True)
            print(f"    Config: {best['config']} ({best['horizons']})", flush=True)
            print(f"    Utility: {best['utility_score']:.3f} | Sharpe: {best['sharpe']:.2f} | Practical: {best['practical_sharpe']}", flush=True)
            print(f"    Sig DA: {best['significant_da']}% | Forecast Mag: {best['avg_forecast_mag']}", flush=True)


if __name__ == "__main__":
    main()
