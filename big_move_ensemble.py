"""
BIG MOVE Ensemble Optimizer
============================
Find ensemble configurations that PREDICT BIG MOVES.
Traders need $1+ moves on Crude, not $0.30 noise.

Focus on:
1. Signal magnitude (how big is the predicted move?)
2. Accuracy on big moves (when we predict big, are we right?)
3. Profit on big moves (actual $ made on large signals)
4. Filter out noise - only trade when confident of big move

Created: 2026-02-03
Author: AmiraB
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def load_all_horizons():
    """Load all Crude Oil horizon data."""
    data_dir = Path("data/1866_Crude_Oil/horizons_wide")
    
    horizons = {}
    for h in range(1, 11):
        f = data_dir / f"horizon_{h}.joblib"
        if f.exists():
            data = joblib.load(f)
            X = data['X'].values if hasattr(data['X'], 'values') else data['X']
            y = data['y'].values if hasattr(data['y'], 'values') else data['y']
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)
            horizons[h] = {'X': np.array(X), 'y': np.array(y)}
    
    return horizons


def compute_big_move_metrics(predictions: np.ndarray, actuals: np.ndarray,
                             big_move_threshold: float = 1.0,
                             train_end: int = None) -> dict:
    """
    Compute metrics focused on BIG MOVES.
    
    Args:
        predictions: Predicted prices
        actuals: Actual prices
        big_move_threshold: Minimum $ move to consider "big" (default $1)
        train_end: Start of OOS period
    """
    if train_end is None:
        train_end = int(len(actuals) * 0.7)
    
    # Out-of-sample only
    oos_pred = predictions[train_end:]
    oos_actual = actuals[train_end:]
    
    if len(oos_pred) < 10:
        return {'sharpe': -999, 'big_move_accuracy': 0, 'big_move_profit': 0}
    
    # Predicted move (from current actual to prediction)
    pred_move = oos_pred[1:] - oos_actual[:-1]  # What we predict
    actual_move = oos_actual[1:] - oos_actual[:-1]  # What actually happened
    
    # Identify BIG predicted moves (signal strength)
    big_signal_mask = np.abs(pred_move) >= big_move_threshold
    n_big_signals = big_signal_mask.sum()
    
    # Identify BIG actual moves
    big_actual_mask = np.abs(actual_move) >= big_move_threshold
    n_big_actual = big_actual_mask.sum()
    
    if n_big_signals == 0:
        return {
            'sharpe': 0,
            'big_move_accuracy': 0,
            'big_move_profit': 0,
            'n_big_signals': 0,
            'avg_signal_size': 0,
            'total_profit': 0,
            'profit_per_trade': 0,
            'big_move_capture': 0,
        }
    
    # Direction accuracy on BIG signals
    pred_dir_big = np.sign(pred_move[big_signal_mask])
    actual_dir_big = np.sign(actual_move[big_signal_mask])
    big_move_accuracy = (pred_dir_big == actual_dir_big).mean()
    
    # Profit when trading big signals only
    # Profit = direction we bet * actual move
    big_signal_profit = pred_dir_big * actual_move[big_signal_mask]
    total_big_profit = big_signal_profit.sum()
    avg_profit_per_big_trade = big_signal_profit.mean()
    
    # Average signal size
    avg_signal_size = np.abs(pred_move[big_signal_mask]).mean()
    
    # Big move capture rate - when actual big move happens, did we signal it?
    if n_big_actual > 0:
        # Check if we had a big signal when big actual happened
        correctly_signaled_big = (big_signal_mask & big_actual_mask).sum()
        big_move_capture = correctly_signaled_big / n_big_actual
    else:
        big_move_capture = 0
    
    # Sharpe on big trades only
    if len(big_signal_profit) > 1:
        sharpe = big_signal_profit.mean() / (big_signal_profit.std() + 1e-8) * np.sqrt(252)
    else:
        sharpe = 0
    
    # ALL trades metrics for comparison
    all_profit = np.sign(pred_move) * actual_move
    total_all_profit = all_profit.sum()
    
    return {
        'sharpe': round(sharpe, 4),
        'big_move_accuracy': round(big_move_accuracy * 100, 2),
        'big_move_profit': round(total_big_profit, 2),
        'n_big_signals': int(n_big_signals),
        'avg_signal_size': round(avg_signal_size, 2),
        'profit_per_trade': round(avg_profit_per_big_trade, 2),
        'big_move_capture': round(big_move_capture * 100, 2),
        'total_all_profit': round(total_all_profit, 2),
        'signal_rate': round(n_big_signals / len(pred_move) * 100, 1),
    }


def get_horizon_signal(horizons: dict, h: int, method: str = 'equal') -> np.ndarray:
    """Get aggregated signal from a single horizon."""
    X = horizons[h]['X']
    
    if method == 'equal':
        return X.mean(axis=1)
    elif method == 'median':
        return np.median(X, axis=1)
    elif method == 'spread':
        # Use spread between bullish and bearish models
        high = np.percentile(X, 90, axis=1)
        low = np.percentile(X, 10, axis=1)
        mid = X.mean(axis=1)
        # Return mid, but with spread info available
        return mid
    else:
        return X.mean(axis=1)


def combine_horizons_for_big_moves(signals: dict, method: str) -> np.ndarray:
    """Combine horizon signals with focus on big moves."""
    horizons = sorted(signals.keys())
    n_samples = len(signals[horizons[0]])
    
    if method == 'equal':
        combined = np.mean([signals[h] for h in horizons], axis=0)
        
    elif method == 'magnitude_weighted':
        # Weight by signal magnitude - stronger signals count more
        magnitudes = np.abs(np.array([signals[h] for h in horizons]))
        weights = magnitudes / (magnitudes.sum(axis=0, keepdims=True) + 1e-8)
        combined = (np.array([signals[h] for h in horizons]) * weights).sum(axis=0)
        
    elif method == 'agreement_amplified':
        # When horizons agree, amplify the signal
        combined = np.zeros(n_samples)
        for i in range(n_samples):
            vals = [signals[h][i] for h in horizons]
            signs = [np.sign(v) for v in vals]
            
            # If all agree on direction, use average
            if all(s == signs[0] for s in signs) and signs[0] != 0:
                combined[i] = np.mean(vals) * (1 + 0.2 * (len(horizons) - 1))  # Amplify
            else:
                combined[i] = np.mean(vals) * 0.5  # Dampen
        
    elif method == 'max_move':
        # Use the horizon predicting the biggest move
        combined = np.zeros(n_samples)
        for i in range(n_samples):
            vals = [signals[h][i] for h in horizons]
            max_idx = np.argmax(np.abs(vals))
            combined[i] = vals[max_idx]
            
    elif method == 'consensus_only':
        # Only signal when multiple horizons agree on direction
        combined = np.zeros(n_samples)
        for i in range(n_samples):
            vals = [signals[h][i] for h in horizons]
            signs = [np.sign(v) for v in vals]
            bullish = sum(1 for s in signs if s > 0)
            bearish = sum(1 for s in signs if s < 0)
            
            # Need 70%+ agreement
            if bullish >= 0.7 * len(horizons):
                combined[i] = np.mean([v for v, s in zip(vals, signs) if s > 0])
            elif bearish >= 0.7 * len(horizons):
                combined[i] = np.mean([v for v, s in zip(vals, signs) if s < 0])
            else:
                combined[i] = 0  # No trade
                
    elif method == 'long_horizon_priority':
        # Prioritize longer horizons (bigger picture)
        weights = {h: h for h in horizons}
        total_w = sum(weights.values())
        combined = sum(signals[h] * (weights[h] / total_w) for h in horizons)
        
    else:
        combined = np.mean([signals[h] for h in horizons], axis=0)
    
    return combined


def test_big_move_configurations(horizons: dict, thresholds: list = [0.5, 1.0, 1.5, 2.0]):
    """Test all configurations for big move performance."""
    results = []
    
    min_len = min(h['X'].shape[0] for h in horizons.values())
    train_end = int(min_len * 0.7)
    y = horizons[1]['y'][-min_len:]
    
    # Pre-compute signals
    horizon_signals = {}
    for h in horizons.keys():
        X = horizons[h]['X'][-min_len:]
        horizon_signals[h] = X.mean(axis=1)
    
    combination_methods = [
        'equal', 
        'magnitude_weighted', 
        'agreement_amplified',
        'max_move',
        'consensus_only',
        'long_horizon_priority'
    ]
    
    print("\nTesting for BIG MOVE performance...")
    
    # Best performing combinations from previous test
    best_horizon_combos = [
        [1, 2, 4],  # The winner
        [1, 3, 8, 10],
        [1, 6, 8, 9],
        [2, 4, 5],
        [1, 5, 8],
        [2, 8],
        [2, 4],
        [1, 4],
        list(range(1, 11)),  # All
        [8, 9, 10],  # Long only
        [1, 2, 3],  # Short only
        [5, 7, 10],  # Spread
    ]
    
    for threshold in thresholds:
        print(f"\n  Big move threshold: ${threshold}")
        
        for combo in best_horizon_combos:
            signals = {h: horizon_signals[h] for h in combo if h in horizon_signals}
            
            for method in combination_methods:
                combined = combine_horizons_for_big_moves(signals, method)
                
                metrics = compute_big_move_metrics(combined, y, 
                                                   big_move_threshold=threshold,
                                                   train_end=train_end)
                
                results.append({
                    'horizons': '+'.join([f'D+{h}' for h in combo]),
                    'n_horizons': len(combo),
                    'method': method,
                    'threshold': threshold,
                    **metrics
                })
    
    return results


def find_optimal_threshold(horizons: dict, combo: list, method: str) -> dict:
    """Find the optimal threshold for a given configuration."""
    min_len = min(h['X'].shape[0] for h in horizons.values())
    train_end = int(min_len * 0.7)
    y = horizons[1]['y'][-min_len:]
    
    horizon_signals = {h: horizons[h]['X'][-min_len:].mean(axis=1) for h in combo}
    combined = combine_horizons_for_big_moves(horizon_signals, method)
    
    best_threshold = 0.5
    best_profit = -999
    
    for threshold in np.arange(0.3, 3.0, 0.1):
        metrics = compute_big_move_metrics(combined, y, 
                                           big_move_threshold=threshold,
                                           train_end=train_end)
        
        # Optimize for profit with reasonable trade frequency
        if metrics['n_big_signals'] >= 10:  # At least 10 trades
            score = metrics['big_move_profit']
            if score > best_profit:
                best_profit = score
                best_threshold = threshold
    
    return {'optimal_threshold': best_threshold, 'optimal_profit': best_profit}


def main():
    print("=" * 70)
    print("BIG MOVE ENSEMBLE OPTIMIZER")
    print("Finding configurations that TRADERS can use to MAKE MONEY")
    print("=" * 70)
    
    horizons = load_all_horizons()
    
    if not horizons:
        print("ERROR: No data loaded")
        return
    
    print(f"Loaded {len(horizons)} horizons")
    
    # Test big move configurations
    results = test_big_move_configurations(horizons, thresholds=[0.5, 1.0, 1.5, 2.0])
    
    df = pd.DataFrame(results)
    
    # Filter for configs with at least 10 big signals
    df_valid = df[df['n_big_signals'] >= 10]
    
    print("\n" + "=" * 70)
    print("TOP 20 BY BIG MOVE PROFIT")
    print("=" * 70)
    
    top20_profit = df_valid.nlargest(20, 'big_move_profit')[[
        'horizons', 'method', 'threshold', 'big_move_profit', 
        'big_move_accuracy', 'n_big_signals', 'profit_per_trade'
    ]]
    print(top20_profit.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("TOP 15 BY BIG MOVE ACCURACY (>60%)")
    print("=" * 70)
    
    accurate = df_valid[df_valid['big_move_accuracy'] >= 50]
    if len(accurate) > 0:
        top15_acc = accurate.nlargest(15, 'big_move_accuracy')[[
            'horizons', 'method', 'threshold', 'big_move_accuracy',
            'big_move_profit', 'n_big_signals', 'profit_per_trade'
        ]]
        print(top15_acc.to_string(index=False))
    else:
        print("No configurations with >50% big move accuracy")
    
    print("\n" + "=" * 70)
    print("BEST PROFIT/TRADE (Efficiency)")
    print("=" * 70)
    
    top_efficient = df_valid.nlargest(10, 'profit_per_trade')[[
        'horizons', 'method', 'threshold', 'profit_per_trade',
        'big_move_accuracy', 'n_big_signals', 'big_move_profit'
    ]]
    print(top_efficient.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("BEST BY THRESHOLD")
    print("=" * 70)
    
    for thresh in [0.5, 1.0, 1.5, 2.0]:
        subset = df_valid[df_valid['threshold'] == thresh]
        if len(subset) > 0:
            best = subset.loc[subset['big_move_profit'].idxmax()]
            print(f"\n  ${thresh} threshold:")
            print(f"    Config:      {best['horizons']} | {best['method']}")
            print(f"    Profit:      ${best['big_move_profit']:.2f}")
            print(f"    Accuracy:    {best['big_move_accuracy']:.1f}%")
            print(f"    # Trades:    {best['n_big_signals']}")
            print(f"    $/Trade:     ${best['profit_per_trade']:.2f}")
    
    # Find THE BEST for trading
    if len(df_valid) > 0:
        best = df_valid.loc[df_valid['big_move_profit'].idxmax()]
        
        print("\n" + "=" * 70)
        print(">>> THE BEST TRADING CONFIGURATION <<<")
        print("=" * 70)
        print(f"Horizons:        {best['horizons']}")
        print(f"Method:          {best['method']}")
        print(f"Threshold:       ${best['threshold']:.2f} (only trade when predicted move > this)")
        print(f"")
        print(f"Big Move Profit: ${best['big_move_profit']:.2f}")
        print(f"Accuracy:        {best['big_move_accuracy']:.1f}%")
        print(f"# Big Trades:    {best['n_big_signals']} trades")
        print(f"Profit/Trade:    ${best['profit_per_trade']:.2f}")
        print(f"Signal Rate:     {best['signal_rate']:.1f}% of days")
        print(f"")
        print(f">>> TRADING RULE: Only enter when signal > ${best['threshold']:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path("results") / f"big_move_test_{timestamp}.csv"
    output_file.parent.mkdir(exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    results = main()
