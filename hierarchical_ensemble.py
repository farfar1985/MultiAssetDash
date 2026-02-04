# hierarchical_ensemble.py
# Hierarchical Ensemble: Series -> Horizon -> Final
# Per Artemis's direction: leverage full data structure

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
    folder = f"{asset_id}_{asset_name}"
    path = os.path.join(DATA_DIR, folder, 'horizons_wide', f'horizon_{horizon}.joblib')
    if not os.path.exists(path):
        return None, None
    data = joblib.load(path)
    return data['X'], data['y']


def parse_series_hierarchy(columns):
    """Parse model columns into parent series groups."""
    series_groups = {}
    for col in columns:
        col_str = str(col)
        if '_' in col_str:
            parent = col_str.split('_')[0]
        else:
            parent = col_str
        
        if parent not in series_groups:
            series_groups[parent] = []
        series_groups[parent].append(col)
    
    return series_groups


def hierarchical_series_ensemble(X: pd.DataFrame, y: pd.Series, 
                                  method: str = 'accuracy_weighted') -> tuple:
    """
    Level 1: Combine models within each series.
    Returns: DataFrame with one column per series (series-level predictions).
    """
    series_groups = parse_series_hierarchy(X.columns)
    ensemble = EnsembleMethods(lookback_window=60)
    
    series_predictions = {}
    series_weights = {}
    
    for parent, model_cols in series_groups.items():
        if len(model_cols) < 2:
            # Single model, use directly
            series_predictions[parent] = X[model_cols[0]]
            series_weights[parent] = pd.Series({model_cols[0]: 1.0})
            continue
        
        X_series = X[model_cols]
        
        try:
            if method == 'accuracy_weighted':
                weights = ensemble.accuracy_weighted(X_series, y)
            elif method == 'top_k':
                weights = ensemble.top_k_by_sharpe(X_series, y, top_pct=0.2)
            elif method == 'equal':
                weights = pd.Series(1.0 / len(model_cols), index=model_cols)
            else:
                weights = ensemble.accuracy_weighted(X_series, y)
            
            # Weighted combination for this series
            series_pred = (X_series * weights).sum(axis=1)
            series_predictions[parent] = series_pred
            series_weights[parent] = weights
            
        except Exception as e:
            # Fallback to simple mean
            series_predictions[parent] = X_series.mean(axis=1)
            series_weights[parent] = pd.Series(1.0 / len(model_cols), index=model_cols)
    
    return pd.DataFrame(series_predictions), series_weights


def hierarchical_horizon_ensemble(horizon_series: dict, y_ref: pd.Series,
                                   method: str = 'magnitude_weighted') -> tuple:
    """
    Level 2: Combine series-level predictions across horizons.
    Returns: Final ensemble signal and forecast.
    """
    # Find common dates
    common_dates = None
    for h, df in horizon_series.items():
        dates = set(df.index)
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates.intersection(dates)
    
    common_dates = sorted(list(common_dates))
    if len(common_dates) < 30:
        return None, None
    
    # Combine all series from all horizons
    all_series = {}
    for h, df in horizon_series.items():
        for col in df.columns:
            all_series[f"H{h}_{col}"] = df[col].loc[common_dates]
    
    combined_df = pd.DataFrame(all_series)
    y_aligned = y_ref.loc[common_dates]
    
    if method == 'magnitude_weighted':
        # Weight by forecast magnitude
        changes = combined_df.diff().abs()
        signals = np.sign(combined_df.diff())
        
        weighted_signal = (signals * changes).sum(axis=1) / changes.sum(axis=1).replace(0, np.nan)
        final_signal = weighted_signal.apply(lambda x: 1 if x > 0.15 else (-1 if x < -0.15 else 0))
        
        # Average forecast
        final_forecast = combined_df.mean(axis=1)
        
    elif method == 'voting':
        # Simple majority voting
        signals = np.sign(combined_df.diff())
        vote = signals.mean(axis=1)
        final_signal = vote.apply(lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0))
        final_forecast = combined_df.mean(axis=1)
        
    else:  # consensus
        # Require strong agreement
        signals = np.sign(combined_df.diff())
        agreement = signals.apply(lambda row: (row == row.mode().iloc[0]).mean() if len(row.mode()) > 0 else 0, axis=1)
        mode_signal = signals.mode(axis=1).iloc[:, 0] if len(signals.mode(axis=1).columns) > 0 else signals.mean(axis=1)
        
        # Only signal when >70% agreement
        final_signal = mode_signal.where(agreement > 0.7, 0)
        final_forecast = combined_df.mean(axis=1)
    
    return final_signal, final_forecast


def run_hierarchical_ensemble(asset_name: str, asset_id: int, horizons: list,
                               min_threshold: float) -> dict:
    """
    Full hierarchical ensemble:
    1. Load all horizons
    2. For each horizon: ensemble at series level
    3. Combine all series-level predictions across horizons
    """
    
    # Level 1: Series-level ensembling per horizon
    horizon_series = {}
    horizon_y = {}
    
    for h in horizons:
        X, y = load_horizon_data(asset_id, asset_name, h)
        if X is None:
            continue
        
        series_df, _ = hierarchical_series_ensemble(X, y, method='accuracy_weighted')
        horizon_series[h] = series_df
        horizon_y[h] = y
    
    if len(horizon_series) < 2:
        return None
    
    # Reference y (use longest horizon)
    ref_h = max(horizon_y.keys())
    y_ref = horizon_y[ref_h]
    
    # Level 2: Cross-horizon ensembling
    final_signal, final_forecast = hierarchical_horizon_ensemble(
        horizon_series, y_ref, method='magnitude_weighted'
    )
    
    if final_signal is None:
        return None
    
    # Split for evaluation
    split_idx = int(len(final_signal) * 0.7)
    test_signal = final_signal.iloc[split_idx:]
    test_forecast = final_forecast.iloc[split_idx:]
    test_y = y_ref.iloc[split_idx:]
    
    # Compute metrics
    price_changes = test_y.diff().dropna()
    sig_aligned = test_signal.loc[price_changes.index]
    fcst_aligned = test_forecast.loc[price_changes.index]
    
    returns = sig_aligned.shift(1) * price_changes.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns) < 10:
        return None
    
    sharpe = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252) if returns.std() > 0 else 0
    
    actual_dir = np.sign(price_changes)
    pred_dir = sig_aligned.shift(1)
    da = (pred_dir == actual_dir).dropna().mean() * 100
    
    forecast_mag = fcst_aligned.diff().abs().mean()
    
    # Significant moves
    sig_mask = price_changes.abs() > min_threshold
    sig_da = (pred_dir.shift(1)[sig_mask] == actual_dir[sig_mask]).mean() * 100 if sig_mask.sum() > 5 else np.nan
    
    # Practical Sharpe
    large_fcst = fcst_aligned.diff().abs() > min_threshold
    large_returns = returns[large_fcst.reindex(returns.index, fill_value=False)]
    practical_sharpe = (large_returns.mean() / large_returns.std(ddof=1)) * np.sqrt(252) if len(large_returns) > 10 and large_returns.std() > 0 else np.nan
    
    # Utility
    utility = 0
    if not np.isnan(practical_sharpe):
        utility += 0.4 * min(practical_sharpe / 2, 1)
    if not np.isnan(sig_da):
        utility += 0.3 * (sig_da / 100)
    if forecast_mag > 0:
        utility += 0.3 * min(forecast_mag / (min_threshold * 3), 1)
    
    return {
        'asset': asset_name,
        'horizons': str(horizons),
        'n_horizons': len(horizons),
        'n_series_per_horizon': len(horizon_series[list(horizon_series.keys())[0]].columns),
        'sharpe': round(sharpe, 3),
        'practical_sharpe': round(practical_sharpe, 3) if not np.isnan(practical_sharpe) else None,
        'da': round(da, 1),
        'significant_da': round(sig_da, 1) if not np.isnan(sig_da) else None,
        'forecast_mag': round(forecast_mag, 2),
        'utility': round(utility, 3),
        'n_trades': len(returns)
    }


def main():
    print("=" * 70, flush=True)
    print("  HIERARCHICAL ENSEMBLE: Series -> Horizon -> Final", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)
    
    ASSETS = {
        'Crude_Oil': {
            'id': 1866, 
            'all_horizons': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'threshold': 0.50
        },
        'SP500': {
            'id': 1625, 
            'all_horizons': [1, 3, 5, 8, 13, 21, 34, 55, 89],
            'threshold': 10.0
        },
        'Bitcoin': {
            'id': 1860, 
            'all_horizons': [1, 3, 5, 8, 10, 13],
            'threshold': 500.0
        },
    }
    
    all_results = []
    
    for asset_name, info in ASSETS.items():
        print(f"\n{'='*70}", flush=True)
        print(f"  {asset_name}", flush=True)
        print(f"{'='*70}", flush=True)
        
        horizons = info['all_horizons']
        threshold = info['threshold']
        
        # Test ALL horizons (full spectrum)
        print(f"\n  Testing ALL horizons: {horizons}", flush=True)
        result = run_hierarchical_ensemble(asset_name, info['id'], horizons, threshold)
        if result:
            result['config'] = 'full_spectrum'
            all_results.append(result)
            print(f"    Series/horizon: {result['n_series_per_horizon']}", flush=True)
            print(f"    Sharpe: {result['sharpe']:.2f} | Practical: {result['practical_sharpe']} | Utility: {result['utility']:.3f}", flush=True)
        
        # Short-term only (D+1 to D+3)
        short = [h for h in horizons if h <= 3]
        if len(short) >= 2:
            print(f"\n  Testing SHORT horizons: {short}", flush=True)
            result = run_hierarchical_ensemble(asset_name, info['id'], short, threshold)
            if result:
                result['config'] = 'short_term'
                all_results.append(result)
                print(f"    Sharpe: {result['sharpe']:.2f} | Practical: {result['practical_sharpe']} | Utility: {result['utility']:.3f}", flush=True)
        
        # Medium-term (D+5 to D+10)
        medium = [h for h in horizons if 5 <= h <= 13]
        if len(medium) >= 2:
            print(f"\n  Testing MEDIUM horizons: {medium}", flush=True)
            result = run_hierarchical_ensemble(asset_name, info['id'], medium, threshold)
            if result:
                result['config'] = 'medium_term'
                all_results.append(result)
                print(f"    Sharpe: {result['sharpe']:.2f} | Practical: {result['practical_sharpe']} | Utility: {result['utility']:.3f}", flush=True)
        
        # Long-term (D+13+)
        long = [h for h in horizons if h >= 13]
        if len(long) >= 2:
            print(f"\n  Testing LONG horizons: {long}", flush=True)
            result = run_hierarchical_ensemble(asset_name, info['id'], long, threshold)
            if result:
                result['config'] = 'long_term'
                all_results.append(result)
                print(f"    Sharpe: {result['sharpe']:.2f} | Practical: {result['practical_sharpe']} | Utility: {result['utility']:.3f}", flush=True)
    
    # Save
    df = pd.DataFrame(all_results)
    output_path = os.path.join(OUTPUT_DIR, f'hierarchical_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}", flush=True)
    print(f"  HIERARCHICAL ENSEMBLE COMPLETE", flush=True)
    print(f"  Results: {output_path}", flush=True)
    print(f"{'='*70}", flush=True)
    
    # Best by utility
    print("\n  TOP CONFIGURATIONS BY UTILITY:", flush=True)
    for asset in ['Crude_Oil', 'SP500', 'Bitcoin']:
        asset_df = df[df['asset'] == asset]
        if len(asset_df) > 0:
            best = asset_df.loc[asset_df['utility'].idxmax()]
            print(f"\n  {asset}: {best['config']}", flush=True)
            print(f"    Horizons: {best['horizons']}", flush=True)
            print(f"    Utility: {best['utility']:.3f} | Sharpe: {best['sharpe']:.2f} | Practical: {best['practical_sharpe']}", flush=True)


if __name__ == "__main__":
    main()
