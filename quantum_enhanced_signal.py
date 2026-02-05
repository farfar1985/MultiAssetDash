"""
QUANTUM-ENHANCED SIGNAL GENERATOR
=================================
Combines per-asset optimal ensembles with quantum regime detection
for the ultimate trading signal.

Signal = Per-Asset Optimal Ensemble + Regime-Based Position Sizing + Contagion Risk Adjustment

This is Nexus v2's core signal engine.

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

from per_asset_optimizer import load_asset_data
from master_ensemble import calculate_pairwise_slope_signal
from quantum_volatility_detector import EnhancedQuantumRegimeDetector
from quantum_contagion_detector import ContagionDetector


class QuantumEnhancedSignalGenerator:
    """
    The complete Nexus v2 signal engine.
    
    Combines:
    1. Per-asset optimal horizon selection
    2. Pairwise slopes signal generation
    3. Quantum regime-based position sizing
    4. Cross-asset contagion risk adjustment
    """
    
    def __init__(self):
        self.regime_detectors = {}
        self.contagion_detector = ContagionDetector(lookback=20)
        
        # Load optimal configs
        self.optimal_configs = self._load_optimal_configs()
        
        # Regime-based position multipliers
        self.regime_multipliers = {
            'LOW_VOL': 1.0,      # Full position
            'NORMAL': 0.8,       # Slightly reduced
            'ELEVATED': 0.5,     # Half position
            'CRISIS': 0.25       # Quarter position
        }
        
        # Contagion-based adjustments
        self.contagion_adjustments = {
            'LOW': 1.0,          # No adjustment
            'MODERATE': 0.9,     # 10% reduction
            'HIGH': 0.7,         # 30% reduction
            'CRITICAL': 0.5      # 50% reduction
        }
    
    def _load_optimal_configs(self) -> dict:
        """Load per-asset optimal configurations."""
        configs = {}
        config_dir = Path('configs')
        
        for config_file in config_dir.glob('optimal_*.json'):
            asset_name = config_file.stem.replace('optimal_', '')
            with open(config_file) as f:
                configs[asset_name] = json.load(f)
        
        return configs
    
    def get_regime_detector(self, asset_name: str) -> EnhancedQuantumRegimeDetector:
        """Get or create regime detector for an asset."""
        if asset_name not in self.regime_detectors:
            self.regime_detectors[asset_name] = EnhancedQuantumRegimeDetector(lookback=20)
        return self.regime_detectors[asset_name]
    
    def generate_signal(self, asset_name: str, horizons: dict, 
                        prices: np.ndarray, t: int,
                        contagion_info: dict = None) -> dict:
        """
        Generate quantum-enhanced trading signal for an asset.
        
        Returns:
            dict with signal, confidence, position_size, and all components
        """
        # 1. Get optimal config for this asset
        config = self.optimal_configs.get(asset_name.lower())
        if not config:
            # Fallback to default horizons
            optimal_horizons = [3, 5, 7]
            threshold = 0.0
            aggregation = 'mean'
        else:
            optimal_horizons = config['best_config']['horizons']
            threshold = config['best_config'].get('threshold', 0.0)
            aggregation = config['best_config'].get('aggregation', 'mean')
        
        # Filter to available horizons
        available = sorted(horizons.keys())
        optimal_horizons = [h for h in optimal_horizons if h in available]
        
        if len(optimal_horizons) < 2:
            optimal_horizons = available[:3]
        
        # 2. Calculate base signal using pairwise slopes
        sig_info = calculate_pairwise_slope_signal(horizons, t, optimal_horizons, aggregation)
        net_prob = sig_info.get('net_prob', 0)
        
        # Apply threshold
        if abs(net_prob) < threshold:
            base_signal = 0
            signal_strength = 'FLAT'
        else:
            base_signal = 1 if net_prob > 0 else -1
            signal_strength = 'STRONG' if abs(net_prob) > 0.3 else 'MODERATE'
        
        # 3. Quantum regime detection
        detector = self.get_regime_detector(asset_name)
        regime_info = detector.detect_regime(prices, t)
        regime = regime_info['regime']
        regime_confidence = regime_info['confidence']
        
        # 4. Calculate position size based on regime
        regime_multiplier = self.regime_multipliers.get(regime, 0.5)
        
        # Adjust for regime confidence
        confidence_adjusted = regime_multiplier * (0.5 + regime_confidence * 0.5)
        
        # 5. Apply contagion adjustment if available
        if contagion_info:
            contagion_level = contagion_info.get('alert_level', 'LOW')
            contagion_adj = self.contagion_adjustments.get(contagion_level, 1.0)
        else:
            contagion_adj = 1.0
            contagion_level = 'UNKNOWN'
        
        # 6. Final position size
        position_size = confidence_adjusted * contagion_adj
        
        # 7. Generate recommendation
        if base_signal == 0:
            recommendation = 'FLAT - No clear signal'
        elif regime == 'CRISIS':
            recommendation = f'CAUTION - Crisis regime detected, {base_signal > 0 and "LONG" or "SHORT"} with minimal size'
        elif position_size < 0.3:
            recommendation = f'REDUCED - High risk environment, small {base_signal > 0 and "LONG" or "SHORT"}'
        else:
            recommendation = f'{base_signal > 0 and "LONG" or "SHORT"} - {signal_strength} signal in {regime} regime'
        
        return {
            'asset': asset_name,
            'timestamp_idx': t,
            
            # Core signal
            'signal': base_signal,
            'net_probability': round(net_prob, 4),
            'signal_strength': signal_strength,
            
            # Position sizing
            'position_size': round(position_size, 3),
            'regime_multiplier': regime_multiplier,
            'contagion_adjustment': contagion_adj,
            
            # Regime info
            'regime': regime,
            'regime_confidence': round(regime_confidence, 3),
            'regime_probabilities': regime_info['probabilities'],
            
            # Volatility features
            'volatility': regime_info['features'],
            
            # Contagion
            'contagion_level': contagion_level,
            
            # Config used
            'horizons_used': optimal_horizons,
            'aggregation': aggregation,
            
            # Recommendation
            'recommendation': recommendation
        }


def run_enhanced_backtest():
    """Backtest the quantum-enhanced signal generator."""
    print("=" * 60)
    print("QUANTUM-ENHANCED SIGNAL GENERATOR BACKTEST")
    print("=" * 60)
    
    # Load all assets
    dirs = {
        'SP500': ('data/1625_SP500', 1625),
        'Bitcoin': ('data/1860_Bitcoin', 1860),
        'Crude_Oil': ('data/1866_Crude_Oil', 1866)
    }
    
    all_data = {}
    min_len = float('inf')
    
    for asset_name, (asset_dir, asset_id) in dirs.items():
        horizons, prices = load_asset_data(asset_dir)
        if horizons and prices is not None:
            all_data[asset_name] = {
                'horizons': horizons,
                'prices': prices,
                'id': asset_id
            }
            min_len = min(min_len, len(prices))
            print(f"  Loaded {asset_name}: {len(prices)} periods")
    
    # Initialize
    generator = QuantumEnhancedSignalGenerator()
    contagion_detector = ContagionDetector(lookback=20)
    
    # Truncate to common length
    for asset in all_data:
        all_data[asset]['prices'] = all_data[asset]['prices'][:int(min_len)]
    
    train_end = int(min_len * 0.7)
    
    print(f"\n  Common length: {int(min_len)}, train_end: {train_end}")
    print("\n  Running enhanced backtest...")
    
    # Track results per asset
    results = {asset: {'returns': [], 'signals': []} for asset in all_data}
    
    for t in range(train_end, int(min_len) - 10):
        if t % 50 == 0:
            print(f"    t={t}")
        
        # Get contagion info
        asset_prices = {name: data['prices'] for name, data in all_data.items()}
        contagion_info = contagion_detector.detect_contagion(asset_prices, t)
        
        # Generate signals for each asset
        for asset_name, data in all_data.items():
            sig = generator.generate_signal(
                asset_name,
                data['horizons'],
                data['prices'],
                t,
                contagion_info
            )
            
            results[asset_name]['signals'].append(sig)
            
            # Calculate return
            min_h = min(data['horizons'].keys())
            y_future = data['horizons'][min_h]['y']
            
            if data['prices'][t] != 0 and t < len(y_future):
                actual_return = (y_future[t] - data['prices'][t]) / data['prices'][t]
                strategy_return = sig['signal'] * sig['position_size'] * actual_return
                results[asset_name]['returns'].append(strategy_return)
    
    # Calculate metrics
    print(f"\n  RESULTS:")
    print(f"  {'='*55}")
    print(f"  {'Asset':<12} {'Sharpe':>10} {'Win Rate':>12} {'Max DD':>10} {'Total Ret':>12}")
    print(f"  {'-'*55}")
    
    summary = {}
    
    for asset_name in all_data:
        returns = np.array(results[asset_name]['returns'])
        
        if len(returns) == 0:
            continue
        
        # Metrics
        min_h = min(all_data[asset_name]['horizons'].keys())
        periods = 252 / min_h
        
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(periods)
        
        trades = returns[returns != 0]
        win_rate = (trades > 0).mean() * 100 if len(trades) > 0 else 0
        
        cum_returns = np.cumsum(returns)
        max_dd = (cum_returns - np.maximum.accumulate(cum_returns)).min() * 100
        
        total_ret = returns.sum() * 100
        
        print(f"  {asset_name:<12} {sharpe:>10.3f} {win_rate:>11.1f}% {max_dd:>9.1f}% {total_ret:>11.1f}%")
        
        summary[asset_name] = {
            'sharpe': round(sharpe, 3),
            'win_rate': round(win_rate, 1),
            'max_drawdown': round(max_dd, 2),
            'total_return': round(total_ret, 2),
            'n_signals': len(returns)
        }
        
        # Regime breakdown
        signals = results[asset_name]['signals']
        regime_returns = {}
        for sig, ret in zip(signals, returns):
            regime = sig['regime']
            if regime not in regime_returns:
                regime_returns[regime] = []
            regime_returns[regime].append(ret)
        
        print(f"\n    Regime breakdown for {asset_name}:")
        for regime, rets in regime_returns.items():
            rets = np.array(rets)
            r_sharpe = rets.mean() / (rets.std() + 1e-8) * np.sqrt(periods) if len(rets) > 1 else 0
            r_wr = (rets[rets != 0] > 0).mean() * 100 if len(rets[rets != 0]) > 0 else 0
            print(f"      {regime}: n={len(rets)}, Sharpe={r_sharpe:.2f}, WR={r_wr:.1f}%")
    
    # Save results
    output = {
        'generated_at': datetime.now().isoformat(),
        'method': 'quantum_enhanced_signal',
        'components': [
            'per_asset_optimal_horizons',
            'pairwise_slopes',
            'quantum_regime_detection',
            'contagion_adjustment'
        ],
        'summary': summary
    }
    
    output_path = Path('configs/quantum_enhanced_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to: {output_path}")
    
    return summary


if __name__ == "__main__":
    results = run_enhanced_backtest()
