"""
QUANTUM REGIME-ADAPTIVE STRATEGIES
==================================
Combines quantum regime detection with optimal ensemble selection.
Key innovation: Switch strategy based on detected volatility regime.

This is where quantum gives us REAL edge.

Created: 2026-02-05
Author: AmiraB
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from quantum_volatility_detector import EnhancedQuantumRegimeDetector
from per_asset_optimizer import load_asset_data
from master_ensemble import calculate_pairwise_slope_signal


class RegimeAdaptiveStrategy:
    """
    Adapts trading strategy based on quantum-detected regime.
    
    Key insight: Different regimes need different strategies:
    - LOW_VOL: Aggressive positioning, use longer horizons
    - NORMAL: Standard pairwise slopes
    - ELEVATED: Reduce position size, tighter stops
    - CRISIS: Inverse signals or stay flat
    """
    
    def __init__(self, asset_name: str):
        self.asset_name = asset_name
        self.qrd = EnhancedQuantumRegimeDetector(lookback=20)
        
        # Regime-specific strategy parameters (can be optimized)
        self.regime_params = {
            'LOW_VOL': {
                'horizons': [5, 7, 10],      # Longer horizons in calm markets
                'position_size': 1.0,         # Full position
                'threshold': 0.1,             # Lower threshold = more trades
                'use_signal': True
            },
            'NORMAL': {
                'horizons': [3, 5, 7],        # Medium horizons
                'position_size': 0.8,         # Slightly reduced
                'threshold': 0.2,
                'use_signal': True
            },
            'ELEVATED': {
                'horizons': [1, 3, 5],        # Shorter horizons in volatile markets
                'position_size': 0.5,         # Half position
                'threshold': 0.3,             # Higher threshold = fewer trades
                'use_signal': True
            },
            'CRISIS': {
                'horizons': [1, 3],           # Very short horizons
                'position_size': 0.25,        # Quarter position
                'threshold': 0.5,             # Very selective
                'use_signal': False           # Option: stay flat in crisis
            }
        }
    
    def get_signal(self, horizons: dict, prices: np.ndarray, t: int) -> dict:
        """
        Get regime-adapted trading signal.
        """
        # Detect current regime
        regime_info = self.qrd.detect_regime(prices, t)
        regime = regime_info['regime']
        confidence = regime_info['confidence']
        
        # Get regime-specific parameters
        params = self.regime_params.get(regime, self.regime_params['NORMAL'])
        
        if not params['use_signal']:
            return {
                'signal': 0,
                'regime': regime,
                'regime_confidence': confidence,
                'reason': 'FLAT - Crisis regime',
                'position_size': 0,
                'net_prob': 0,
                'horizons_used': [],
                'features': regime_info['features'],
                'transition': regime_info.get('transition')
            }
        
        # Get available horizons that match our preference
        available = sorted(horizons.keys())
        target_horizons = [h for h in params['horizons'] if h in available]
        
        if len(target_horizons) < 2:
            target_horizons = available[:3]  # Fallback
        
        # Calculate pairwise slope signal
        sig_info = calculate_pairwise_slope_signal(horizons, t, target_horizons)
        net_prob = sig_info.get('net_prob', 0)
        
        # Apply threshold
        if abs(net_prob) < params['threshold']:
            signal = 0
            reason = 'FLAT - Below threshold'
        else:
            signal = 1 if net_prob > 0 else -1
            reason = 'BULLISH' if signal > 0 else 'BEARISH'
        
        # Adjust for regime-based position sizing
        position_size = params['position_size'] * min(confidence + 0.3, 1.0)
        
        return {
            'signal': signal,
            'net_prob': net_prob,
            'regime': regime,
            'regime_confidence': confidence,
            'reason': reason,
            'position_size': position_size,
            'horizons_used': target_horizons,
            'features': regime_info['features'],
            'transition': regime_info.get('transition')
        }


class EarlyWarningSystem:
    """
    Predicts regime transitions BEFORE they happen.
    Uses quantum interference patterns to detect regime instability.
    """
    
    def __init__(self, lookback: int = 5):
        self.lookback = lookback
        self.transition_history = []
        
    def calculate_regime_stability(self, regime_info: dict, history: list) -> dict:
        """
        Calculate how stable the current regime is.
        Unstable = transition likely soon.
        """
        if len(history) < self.lookback:
            return {'stability': 1.0, 'warning': None}
        
        recent = history[-self.lookback:]
        
        # Check regime consistency
        regimes = [h['regime'] for h in recent]
        current_regime = regime_info['regime']
        consistency = regimes.count(current_regime) / len(regimes)
        
        # Check transition probabilities
        trans_probs = [h.get('transition', {}).get('transition_probability', 0) 
                       for h in recent if h.get('transition')]
        avg_trans_prob = np.mean(trans_probs) if trans_probs else 0
        
        # Check volatility trend
        vols = [h['features']['realized_vol'] for h in recent]
        vol_trend = (vols[-1] - vols[0]) / (vols[0] + 1e-8)
        
        # Stability score (0 = unstable, 1 = stable)
        stability = (consistency * 0.4 + 
                    (1 - avg_trans_prob) * 0.3 + 
                    (1 - min(abs(vol_trend), 1)) * 0.3)
        
        # Generate warning
        warning = None
        if stability < 0.5:
            if vol_trend > 0.2:
                warning = 'VOLATILITY SPIKE INCOMING'
            elif vol_trend < -0.2:
                warning = 'VOLATILITY COMPRESSION'
            else:
                warning = 'REGIME INSTABILITY'
        
        return {
            'stability': round(stability, 3),
            'consistency': round(consistency, 3),
            'avg_transition_prob': round(avg_trans_prob, 3),
            'vol_trend': round(vol_trend, 3),
            'warning': warning
        }


def backtest_regime_adaptive(asset_dir: str, asset_name: str) -> dict:
    """
    Backtest the regime-adaptive strategy vs static strategy.
    """
    print(f"\n{'='*60}")
    print(f"REGIME-ADAPTIVE BACKTEST: {asset_name.upper()}")
    print(f"{'='*60}")
    
    horizons, prices = load_asset_data(asset_dir)
    
    if horizons is None:
        return None
    
    available = sorted(horizons.keys())
    min_len = min(horizons[h]['X'].shape[0] for h in available)
    train_end = int(min_len * 0.7)
    
    # Initialize strategies
    adaptive = RegimeAdaptiveStrategy(asset_name)
    ews = EarlyWarningSystem()
    
    # Track results
    adaptive_returns = []
    static_returns = []  # Using fixed [5,7,10] for comparison
    
    regime_history = []
    warnings_issued = []
    
    min_h = min(available)
    y_future = horizons[min_h]['y']
    
    print(f"  Backtesting from t={train_end} to t={min_len - min_h}...")
    
    for t in range(train_end, min_len - min_h):
        if t % 50 == 0:
            print(f"    t={t}")
        
        # Get regime-adaptive signal
        sig = adaptive.get_signal(horizons, prices, t)
        regime_history.append(sig)
        
        # Check early warning
        stability = ews.calculate_regime_stability(sig, regime_history)
        if stability['warning']:
            warnings_issued.append({
                't': t,
                'warning': stability['warning'],
                'stability': stability['stability']
            })
        
        # Calculate actual return
        if prices[t] != 0:
            actual_return = (y_future[t] - prices[t]) / prices[t]
        else:
            actual_return = 0
        
        # Adaptive strategy return (with position sizing)
        adaptive_return = sig['signal'] * sig['position_size'] * actual_return
        adaptive_returns.append(adaptive_return)
        
        # Static strategy return (for comparison)
        static_horizons = [h for h in [5, 7, 10] if h in available]
        if len(static_horizons) >= 2:
            static_sig = calculate_pairwise_slope_signal(horizons, t, static_horizons)
            static_signal = np.sign(static_sig.get('net_prob', 0))
        else:
            static_signal = 0
        static_returns.append(static_signal * actual_return)
    
    # Calculate metrics
    adaptive_returns = np.array(adaptive_returns)
    static_returns = np.array(static_returns)
    
    periods = 252 / min_h
    
    adaptive_sharpe = adaptive_returns.mean() / (adaptive_returns.std() + 1e-8) * np.sqrt(periods)
    static_sharpe = static_returns.mean() / (static_returns.std() + 1e-8) * np.sqrt(periods)
    
    adaptive_trades = adaptive_returns[adaptive_returns != 0]
    static_trades = static_returns[static_returns != 0]
    
    adaptive_wr = (adaptive_trades > 0).mean() * 100 if len(adaptive_trades) > 0 else 0
    static_wr = (static_trades > 0).mean() * 100 if len(static_trades) > 0 else 0
    
    # Max drawdown
    adaptive_cum = np.cumsum(adaptive_returns)
    static_cum = np.cumsum(static_returns)
    
    adaptive_dd = (adaptive_cum - np.maximum.accumulate(adaptive_cum)).min() * 100
    static_dd = (static_cum - np.maximum.accumulate(static_cum)).min() * 100
    
    # Regime breakdown
    regime_counts = pd.Series([h['regime'] for h in regime_history]).value_counts()
    
    print(f"\n  RESULTS:")
    print(f"    {'Metric':<20} {'Adaptive':<15} {'Static':<15} {'Improvement':<15}")
    print(f"    {'-'*60}")
    print(f"    {'Sharpe':<20} {adaptive_sharpe:>12.3f}   {static_sharpe:>12.3f}   {(adaptive_sharpe - static_sharpe):>+12.3f}")
    print(f"    {'Win Rate':<20} {adaptive_wr:>11.1f}%   {static_wr:>11.1f}%   {(adaptive_wr - static_wr):>+11.1f}%")
    print(f"    {'Max Drawdown':<20} {adaptive_dd:>11.1f}%   {static_dd:>11.1f}%   {(static_dd - adaptive_dd):>+11.1f}%")
    print(f"    {'Total Return':<20} {adaptive_returns.sum()*100:>11.1f}%   {static_returns.sum()*100:>11.1f}%")
    
    print(f"\n  REGIME DISTRIBUTION:")
    for regime, count in regime_counts.items():
        pct = count / len(regime_history) * 100
        print(f"    {regime}: {count} ({pct:.1f}%)")
    
    print(f"\n  EARLY WARNINGS ISSUED: {len(warnings_issued)}")
    if warnings_issued:
        for w in warnings_issued[-3:]:
            print(f"    t={w['t']}: {w['warning']} (stability={w['stability']:.2f})")
    
    return {
        'asset': asset_name,
        'adaptive': {
            'sharpe': round(adaptive_sharpe, 3),
            'win_rate': round(adaptive_wr, 1),
            'max_drawdown': round(adaptive_dd, 2),
            'total_return': round(adaptive_returns.sum() * 100, 2)
        },
        'static': {
            'sharpe': round(static_sharpe, 3),
            'win_rate': round(static_wr, 1),
            'max_drawdown': round(static_dd, 2),
            'total_return': round(static_returns.sum() * 100, 2)
        },
        'improvement': {
            'sharpe': round(adaptive_sharpe - static_sharpe, 3),
            'win_rate': round(adaptive_wr - static_wr, 1),
            'drawdown_reduction': round(static_dd - adaptive_dd, 2)
        },
        'regime_distribution': regime_counts.to_dict(),
        'warnings': len(warnings_issued)
    }


def run_adaptive_comparison():
    """Compare adaptive vs static strategies across assets."""
    dirs = {
        'sp500': 'data/1625_SP500',
        'bitcoin': 'data/1860_Bitcoin',
        'crude_oil': 'data/1866_Crude_Oil'
    }
    
    print("=" * 60)
    print("QUANTUM REGIME-ADAPTIVE vs STATIC STRATEGY")
    print("=" * 60)
    
    results = {}
    
    for asset_name, asset_dir in dirs.items():
        result = backtest_regime_adaptive(asset_dir, asset_name)
        if result:
            results[asset_name] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Asset':<12} {'Adaptive Sharpe':<18} {'Static Sharpe':<15} {'Improvement':<12}")
    print("-" * 60)
    
    for asset, r in results.items():
        print(f"{asset.upper():<12} {r['adaptive']['sharpe']:>15.3f}   {r['static']['sharpe']:>12.3f}   {r['improvement']['sharpe']:>+10.3f}")
    
    # Save
    output = {
        'generated_at': datetime.now().isoformat(),
        'method': 'quantum_regime_adaptive_strategy',
        'results': results
    }
    
    output_path = Path('configs/regime_adaptive_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_adaptive_comparison()
