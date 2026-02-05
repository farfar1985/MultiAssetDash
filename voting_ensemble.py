"""
VOTING ENSEMBLE: Simple Consensus of Multiple Methods
=====================================================
Instead of complex regime detection, just average signals from all methods.
The idea: When multiple methods agree, confidence is higher.

Created: 2026-02-05
Author: AmiraB
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from per_asset_optimizer import load_asset_data
from master_ensemble import calculate_pairwise_slope_signal


def get_magnitude_signal(horizons: dict, t: int, horizon_subset: list) -> float:
    """Magnitude-weighted voting signal."""
    votes_up = 0
    votes_down = 0
    
    for h in horizon_subset:
        if h not in horizons or t >= len(horizons[h]['X']):
            continue
        
        X = horizons[h]['X']
        y = horizons[h]['y']
        
        pred = X[t].mean()
        current = y[t - h] if t >= h else y[0]
        
        pct_change = (pred - current) / (current + 1e-8)
        magnitude = abs(pct_change)
        
        if pct_change > 0:
            votes_up += magnitude
        else:
            votes_down += magnitude
    
    total = votes_up + votes_down
    return (votes_up - votes_down) / total if total > 0 else 0


def get_median_signal(horizons: dict, t: int, horizon_subset: list) -> float:
    """Median prediction signal."""
    preds = []
    current = None
    
    min_h = min(horizon_subset)
    
    for h in horizon_subset:
        if h not in horizons or t >= len(horizons[h]['X']):
            continue
        
        X = horizons[h]['X']
        preds.append(X[t].mean())
        
        if current is None and t >= h:
            current = horizons[h]['y'][t - h]
    
    if not preds or current is None:
        return 0
    
    median_pred = np.median(preds)
    direction = np.sign(median_pred - current)
    return direction


def get_consensus_signal(horizons: dict, t: int, horizon_subset: list) -> dict:
    """
    Get consensus signal by averaging multiple methods.
    Returns signal strength based on agreement.
    """
    # Method 1: Pairwise slopes
    ps = calculate_pairwise_slope_signal(horizons, t, horizon_subset, 'mean')
    ps_signal = np.sign(ps['net_prob']) if ps['net_prob'] != 0 else 0
    
    # Method 2: Magnitude-weighted
    mag_prob = get_magnitude_signal(horizons, t, horizon_subset)
    mag_signal = np.sign(mag_prob) if mag_prob != 0 else 0
    
    # Method 3: Median
    med_signal = get_median_signal(horizons, t, horizon_subset)
    
    # Count votes
    signals = [ps_signal, mag_signal, med_signal]
    bullish = sum(1 for s in signals if s > 0)
    bearish = sum(1 for s in signals if s < 0)
    
    # Consensus
    if bullish >= 2:
        consensus = 1
        confidence = bullish / 3
    elif bearish >= 2:
        consensus = -1
        confidence = bearish / 3
    else:
        consensus = 0
        confidence = 0
    
    return {
        'signal': consensus,
        'confidence': confidence,
        'bullish_votes': bullish,
        'bearish_votes': bearish,
        'methods': {
            'pairwise_slopes': ps_signal,
            'magnitude': mag_signal,
            'median': med_signal
        }
    }


def backtest_voting(asset_dir: str, horizon_subset: list = None) -> dict:
    """Backtest voting consensus on an asset."""
    horizons, prices = load_asset_data(asset_dir)
    
    if horizons is None:
        return None
    
    available = sorted(horizons.keys())
    if horizon_subset is None:
        horizon_subset = available
    horizon_subset = [h for h in horizon_subset if h in available]
    
    min_len = min(horizons[h]['X'].shape[0] for h in horizon_subset)
    train_end = int(min_len * 0.7)
    
    min_h = min(horizon_subset)
    y_future = horizons[min_h]['y']
    
    returns = []
    high_conf_returns = []  # Only when all methods agree
    
    for t in range(train_end, min_len - min_h):
        result = get_consensus_signal(horizons, t, horizon_subset)
        signal = result['signal']
        conf = result['confidence']
        
        if prices[t] != 0:
            actual_return = (y_future[t] - prices[t]) / prices[t]
        else:
            actual_return = 0
        
        strategy_return = signal * actual_return
        returns.append(strategy_return)
        
        # Track high-confidence trades (all methods agree)
        if conf == 1.0:
            high_conf_returns.append(strategy_return)
    
    returns = np.array(returns)
    high_conf_returns = np.array(high_conf_returns) if high_conf_returns else np.array([0])
    
    periods = 252 / min_h
    
    # Overall metrics
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(periods)
    trades = returns[returns != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0
    
    # High-confidence metrics
    hc_sharpe = high_conf_returns.mean() / (high_conf_returns.std() + 1e-8) * np.sqrt(periods) if len(high_conf_returns) > 1 else 0
    hc_trades = high_conf_returns[high_conf_returns != 0]
    hc_win_rate = (hc_trades > 0).mean() if len(hc_trades) > 0 else 0
    
    return {
        'overall': {
            'sharpe': round(sharpe, 3),
            'win_rate': round(win_rate * 100, 2),
            'total_return': round(returns.sum() * 100, 2),
            'n_periods': len(returns)
        },
        'high_confidence': {
            'sharpe': round(hc_sharpe, 3),
            'win_rate': round(hc_win_rate * 100, 2),
            'total_return': round(high_conf_returns.sum() * 100, 2),
            'n_periods': len(high_conf_returns)
        }
    }


def run_voting_comparison():
    """Compare voting ensemble vs simple optimal."""
    import json
    
    # Load optimal configs
    with open('configs/optimization_summary.json') as f:
        opt = json.load(f)
    
    dirs = {
        'sp500': 'data/1625_SP500',
        'bitcoin': 'data/1860_Bitcoin', 
        'crude_oil': 'data/1866_Crude_Oil'
    }
    
    print("VOTING ENSEMBLE COMPARISON")
    print("=" * 60)
    print("Idea: Only trade when multiple methods AGREE\n")
    
    results = {}
    
    for asset in ['sp500', 'bitcoin', 'crude_oil']:
        best_h = opt['results'][asset]['best_horizons']
        
        # Voting backtest
        r = backtest_voting(dirs[asset], best_h)
        
        if r:
            results[asset] = r
            
            print(f"{asset.upper()} (horizons={best_h}):")
            print(f"  Simple Optimal:    Sharpe={opt['results'][asset]['sharpe']}, WR={opt['results'][asset]['win_rate']}%")
            print(f"  Voting Consensus:  Sharpe={r['overall']['sharpe']}, WR={r['overall']['win_rate']}%")
            print(f"  High-Conf Only:    Sharpe={r['high_confidence']['sharpe']}, WR={r['high_confidence']['win_rate']}% ({r['high_confidence']['n_periods']} trades)")
            print()
    
    # Save results
    output = {
        'generated_at': datetime.now().isoformat(),
        'method': 'voting_consensus',
        'results': results
    }
    
    with open('configs/voting_ensemble_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return results


if __name__ == "__main__":
    run_voting_comparison()
