"""
WALK-FORWARD VALIDATION
=======================
Proper out-of-sample testing for the 2pm meeting.

Addresses Artemis's concerns:
1. In-sample vs out-of-sample split
2. Transaction costs
3. Time period robustness (multiple windows)

Created: 2026-02-05
Author: AmiraB
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from per_asset_optimizer import load_asset_data
from master_ensemble import calculate_pairwise_slope_signal


def walk_forward_backtest(asset_name: str, asset_dir: str, 
                           horizons_config: list,
                           train_pct: float = 0.6,
                           val_pct: float = 0.2,
                           transaction_cost_bps: float = 5.0) -> dict:
    """
    Walk-forward validation with transaction costs.
    
    Split:
    - Train: First 60% (in-sample)
    - Validation: Next 20% (out-of-sample tuning)
    - Test: Final 20% (true out-of-sample)
    """
    print(f"\n{'='*50}")
    print(f"WALK-FORWARD: {asset_name}")
    print(f"{'='*50}")
    
    horizons, prices = load_asset_data(asset_dir)
    if horizons is None:
        return None
    
    available = sorted(horizons.keys())
    target_horizons = [h for h in horizons_config if h in available]
    
    if len(target_horizons) < 2:
        target_horizons = available[:3]
    
    min_len = min(horizons[h]['X'].shape[0] for h in target_horizons)
    
    # Split points
    train_end = int(min_len * train_pct)
    val_end = int(min_len * (train_pct + val_pct))
    
    print(f"  Total periods: {min_len}")
    print(f"  Train: 0-{train_end} ({train_pct*100:.0f}%)")
    print(f"  Validation: {train_end}-{val_end} ({val_pct*100:.0f}%)")
    print(f"  Test: {val_end}-{min_len} ({(1-train_pct-val_pct)*100:.0f}%)")
    print(f"  Horizons: {target_horizons}")
    print(f"  Transaction cost: {transaction_cost_bps} bps")
    
    min_h = min(target_horizons)
    y_future = horizons[min_h]['y']
    
    results = {'train': [], 'validation': [], 'test': []}
    trades = {'train': 0, 'validation': 0, 'test': 0}
    prev_signal = {'train': 0, 'validation': 0, 'test': 0}
    
    def get_phase(t):
        if t < train_end:
            return 'train'
        elif t < val_end:
            return 'validation'
        else:
            return 'test'
    
    # Run through all periods
    for t in range(50, min_len - min_h):
        phase = get_phase(t)
        
        # Skip train phase for signals (just for context)
        if phase == 'train':
            continue
        
        # Get signal
        sig_info = calculate_pairwise_slope_signal(horizons, t, target_horizons)
        net_prob = sig_info.get('net_prob', 0)
        signal = 1 if net_prob > 0 else (-1 if net_prob < 0 else 0)
        
        # Calculate return
        if prices[t] != 0:
            actual_return = (y_future[t] - prices[t]) / prices[t]
        else:
            actual_return = 0
        
        # Strategy return
        strategy_return = signal * actual_return
        
        # Apply transaction cost if signal changed
        if signal != prev_signal[phase] and signal != 0:
            cost = transaction_cost_bps / 10000 * 2  # Round-trip
            strategy_return -= cost
            trades[phase] += 1
        
        prev_signal[phase] = signal
        results[phase].append(strategy_return)
    
    # Calculate metrics for each phase
    metrics = {}
    
    for phase in ['validation', 'test']:
        returns = np.array(results[phase])
        
        if len(returns) == 0:
            continue
        
        periods_per_year = 252 / min_h
        
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(periods_per_year)
        
        # Gross Sharpe (no transaction costs)
        gross_returns = returns + (trades[phase] * transaction_cost_bps / 10000 * 2 / len(returns))
        gross_sharpe = gross_returns.mean() / (gross_returns.std() + 1e-8) * np.sqrt(periods_per_year)
        
        trade_returns = returns[returns != 0]
        win_rate = (trade_returns > 0).mean() * 100 if len(trade_returns) > 0 else 0
        
        cum = np.cumsum(returns)
        max_dd = (cum - np.maximum.accumulate(cum)).min() * 100
        
        total_return = returns.sum() * 100
        
        metrics[phase] = {
            'sharpe_net': round(sharpe, 3),
            'sharpe_gross': round(gross_sharpe, 3),
            'win_rate': round(win_rate, 1),
            'max_drawdown': round(max_dd, 2),
            'total_return': round(total_return, 2),
            'n_periods': len(returns),
            'n_trades': trades[phase],
            'avg_trade_return': round(returns.mean() * 100, 4)
        }
    
    print(f"\n  RESULTS:")
    print(f"  {'-'*45}")
    print(f"  {'Metric':<20} {'Validation':>12} {'Test (OOS)':>12}")
    print(f"  {'-'*45}")
    
    for metric in ['sharpe_net', 'sharpe_gross', 'win_rate', 'max_drawdown', 'total_return']:
        val = metrics.get('validation', {}).get(metric, 'N/A')
        test = metrics.get('test', {}).get(metric, 'N/A')
        print(f"  {metric:<20} {str(val):>12} {str(test):>12}")
    
    return {
        'asset': asset_name,
        'horizons': target_horizons,
        'transaction_cost_bps': transaction_cost_bps,
        'validation': metrics.get('validation'),
        'test': metrics.get('test')
    }


def run_robustness_test(asset_name: str, asset_dir: str,
                        horizons_config: list,
                        n_windows: int = 3) -> dict:
    """
    Test robustness across multiple time windows.
    """
    print(f"\n{'='*50}")
    print(f"ROBUSTNESS TEST: {asset_name}")
    print(f"{'='*50}")
    
    horizons, prices = load_asset_data(asset_dir)
    if horizons is None:
        return None
    
    available = sorted(horizons.keys())
    target_horizons = [h for h in horizons_config if h in available]
    
    if len(target_horizons) < 2:
        target_horizons = available[:3]
    
    min_len = min(horizons[h]['X'].shape[0] for h in target_horizons)
    min_h = min(target_horizons)
    y_future = horizons[min_h]['y']
    
    window_size = (min_len - 50) // n_windows
    
    window_results = []
    
    for w in range(n_windows):
        start = 50 + w * window_size
        end = start + window_size
        
        returns = []
        
        for t in range(start, min(end, min_len - min_h)):
            sig_info = calculate_pairwise_slope_signal(horizons, t, target_horizons)
            net_prob = sig_info.get('net_prob', 0)
            signal = 1 if net_prob > 0 else (-1 if net_prob < 0 else 0)
            
            if prices[t] != 0:
                actual_return = (y_future[t] - prices[t]) / prices[t]
            else:
                actual_return = 0
            
            returns.append(signal * actual_return)
        
        returns = np.array(returns)
        periods = 252 / min_h
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(periods)
        
        window_results.append({
            'window': w + 1,
            'start_idx': start,
            'end_idx': end,
            'sharpe': round(sharpe, 3),
            'total_return': round(returns.sum() * 100, 2),
            'n_periods': len(returns)
        })
        
        print(f"  Window {w+1} (t={start}-{end}): Sharpe={sharpe:.3f}, Return={returns.sum()*100:.1f}%")
    
    # Check consistency
    sharpes = [w['sharpe'] for w in window_results]
    consistent = all(s > 0 for s in sharpes)
    
    print(f"\n  Consistency: {'[OK] All windows positive' if consistent else '[X] Some windows negative'}")
    print(f"  Sharpe range: {min(sharpes):.3f} to {max(sharpes):.3f}")
    print(f"  Sharpe std: {np.std(sharpes):.3f}")
    
    return {
        'asset': asset_name,
        'n_windows': n_windows,
        'windows': window_results,
        'all_positive': consistent,
        'sharpe_min': round(min(sharpes), 3),
        'sharpe_max': round(max(sharpes), 3),
        'sharpe_std': round(np.std(sharpes), 3)
    }


def run_full_validation():
    """Run walk-forward validation for all assets."""
    print("=" * 60)
    print("WALK-FORWARD VALIDATION FOR 2PM MEETING")
    print("=" * 60)
    print("\nAddressing Artemis's concerns:")
    print("  1. In-sample vs out-of-sample")
    print("  2. Transaction costs (5 bps)")
    print("  3. Time period robustness")
    
    configs = {
        'SP500': {'dir': 'data/1625_SP500', 'horizons': [3, 8]},
        'Bitcoin': {'dir': 'data/1860_Bitcoin', 'horizons': [3, 5]},
        'Crude_Oil': {'dir': 'data/1866_Crude_Oil', 'horizons': [9, 10]}
    }
    
    all_results = {}
    
    for asset_name, config in configs.items():
        # Walk-forward
        wf_result = walk_forward_backtest(
            asset_name, config['dir'], config['horizons'],
            transaction_cost_bps=5.0
        )
        
        # Robustness
        rob_result = run_robustness_test(
            asset_name, config['dir'], config['horizons'],
            n_windows=3
        )
        
        all_results[asset_name] = {
            'walk_forward': wf_result,
            'robustness': rob_result
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY FOR PRESENTATION")
    print("=" * 60)
    
    print("\nOUT-OF-SAMPLE PERFORMANCE (Test Set, After Transaction Costs)")
    print("-" * 50)
    
    for asset, data in all_results.items():
        wf = data['walk_forward']
        if wf and wf.get('test'):
            test = wf['test']
            print(f"\n{asset}:")
            print(f"  Sharpe (net): {test['sharpe_net']}")
            print(f"  Sharpe (gross): {test['sharpe_gross']}")
            print(f"  Win Rate: {test['win_rate']}%")
            print(f"  Max DD: {test['max_drawdown']}%")
    
    print("\nROBUSTNESS (Across 3 Time Windows)")
    print("-" * 50)
    
    for asset, data in all_results.items():
        rob = data['robustness']
        if rob:
            status = "[OK] Consistent" if rob['all_positive'] else "[X] Inconsistent"
            print(f"\n{asset}: {status}")
            print(f"  Sharpe range: {rob['sharpe_min']} to {rob['sharpe_max']}")
    
    # Save
    output = {
        'generated_at': datetime.now().isoformat(),
        'purpose': 'walk_forward_validation_for_2pm_meeting',
        'transaction_cost_bps': 5.0,
        'results': all_results
    }
    
    output_path = Path('configs/walk_forward_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    results = run_full_validation()
