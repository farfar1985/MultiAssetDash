"""
QUANTUM-INSPIRED REGIME DETECTOR
================================
Uses quantum simulation concepts for superior regime detection:
- Superposition: Consider multiple regime hypotheses simultaneously
- Interference: Detect regime transitions via wave interference patterns
- Entanglement: Cross-asset correlation regime detection

Then selects the optimal ensemble strategy per detected regime.

This is THE differentiator for CME.

Created: 2026-02-05
Author: AmiraB (Bill's quantum insight)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import expm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from per_asset_optimizer import load_asset_data
from master_ensemble import calculate_pairwise_slope_signal


class QuantumRegimeDetector:
    """
    Quantum-inspired regime detection using:
    1. Quantum state representation of market conditions
    2. Hamiltonian evolution for regime dynamics
    3. Measurement collapse for regime classification
    """
    
    def __init__(self, n_regimes: int = 4, lookback: int = 20):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.regime_names = ['MOMENTUM', 'MEAN_REVERT', 'HIGH_VOL', 'LOW_VOL']
        
        # Quantum state vector (amplitudes for each regime)
        self.state = np.ones(n_regimes) / np.sqrt(n_regimes)  # Equal superposition
        
        # Hamiltonian for regime transitions (energy landscape)
        self.H = self._build_hamiltonian()
        
        # Trained regime centroids
        self.scaler = StandardScaler()
        self.gmm = None
        self.regime_ensemble_map = {}
        
    def _build_hamiltonian(self) -> np.ndarray:
        """
        Build the Hamiltonian matrix for regime dynamics.
        Off-diagonal elements represent transition probabilities.
        Diagonal elements represent regime stability.
        """
        H = np.array([
            [-1.0, 0.3, 0.2, 0.1],   # MOMENTUM: stable, can transition to others
            [0.3, -1.0, 0.4, 0.2],   # MEAN_REVERT: moderate transitions
            [0.2, 0.4, -1.5, 0.5],   # HIGH_VOL: less stable
            [0.1, 0.2, 0.5, -0.5],   # LOW_VOL: most stable
        ])
        # Make Hermitian
        H = (H + H.T) / 2
        return H
    
    def extract_features(self, prices: np.ndarray, t: int) -> np.ndarray:
        """Extract quantum-relevant features from price data."""
        if t < self.lookback:
            return np.zeros(6)
        
        window = prices[t - self.lookback:t]
        returns = np.diff(window) / window[:-1]
        
        # Feature 1: Momentum (trend strength)
        momentum = (window[-1] - window[0]) / window[0]
        
        # Feature 2: Volatility (quantum uncertainty)
        volatility = returns.std()
        
        # Feature 3: Autocorrelation (mean reversion indicator)
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0
        else:
            autocorr = 0
        
        # Feature 4: Skewness (asymmetry in returns)
        skew = stats.skew(returns) if len(returns) > 2 else 0
        
        # Feature 5: Kurtosis (tail risk / "quantum tunneling" events)
        kurt = stats.kurtosis(returns) if len(returns) > 3 else 0
        
        # Feature 6: Hurst exponent approximation (persistence)
        # H > 0.5 = trending, H < 0.5 = mean reverting
        hurst = self._estimate_hurst(returns)
        
        return np.array([momentum, volatility, autocorr, skew, kurt, hurst])
    
    def _estimate_hurst(self, returns: np.ndarray, max_lag: int = 10) -> float:
        """Estimate Hurst exponent using R/S analysis."""
        if len(returns) < max_lag * 2:
            return 0.5
        
        lags = range(2, min(max_lag, len(returns) // 2))
        rs = []
        
        for lag in lags:
            chunks = [returns[i:i+lag] for i in range(0, len(returns) - lag, lag)]
            if not chunks:
                continue
            
            rs_values = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean_adj = chunk - chunk.mean()
                cum_dev = np.cumsum(mean_adj)
                R = cum_dev.max() - cum_dev.min()
                S = chunk.std()
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                rs.append((lag, np.mean(rs_values)))
        
        if len(rs) < 2:
            return 0.5
        
        # Fit log-log regression
        x = np.log([r[0] for r in rs])
        y = np.log([r[1] for r in rs])
        
        try:
            slope, _ = np.polyfit(x, y, 1)
            return np.clip(slope, 0, 1)
        except:
            return 0.5
    
    def quantum_evolve(self, features: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Evolve quantum state based on market features.
        Uses Schrodinger-like evolution with feature-modified Hamiltonian.
        """
        # Modify Hamiltonian based on features
        momentum, volatility, autocorr, skew, kurt, hurst = features
        
        # Create feature-dependent perturbation
        V = np.zeros((self.n_regimes, self.n_regimes))
        
        # Momentum favors MOMENTUM regime
        V[0, 0] = -momentum * 2
        
        # Negative autocorr favors MEAN_REVERT
        V[1, 1] = autocorr * 2  # More negative autocorr = lower energy
        
        # High volatility favors HIGH_VOL
        V[2, 2] = -volatility * 10
        
        # Low volatility favors LOW_VOL
        V[3, 3] = volatility * 10 - 0.5
        
        # Total Hamiltonian
        H_total = self.H + V
        
        # Quantum evolution: |psi(t+dt)> = exp(-i*H*dt) |psi(t)>
        # For real-valued simulation, use exp(-H*dt)
        U = expm(-H_total * dt)
        
        new_state = U @ self.state
        
        # Normalize (maintain probability conservation)
        new_state = new_state / np.linalg.norm(new_state)
        
        return new_state
    
    def measure(self, state: np.ndarray = None) -> tuple:
        """
        Quantum measurement: Collapse state to a regime.
        Returns (regime_index, regime_name, confidence)
        """
        if state is None:
            state = self.state
        
        # Probabilities are |amplitude|^2
        probabilities = np.abs(state) ** 2
        probabilities = probabilities / probabilities.sum()  # Normalize
        
        # Most likely regime (max probability)
        regime_idx = np.argmax(probabilities)
        confidence = probabilities[regime_idx]
        
        return regime_idx, self.regime_names[regime_idx], confidence, probabilities
    
    def detect_regime(self, prices: np.ndarray, t: int) -> dict:
        """
        Full quantum regime detection pipeline.
        """
        # Extract features
        features = self.extract_features(prices, t)
        
        # Quantum evolution
        self.state = self.quantum_evolve(features)
        
        # Measurement
        regime_idx, regime_name, confidence, probs = self.measure()
        
        return {
            'regime': regime_name,
            'regime_idx': regime_idx,
            'confidence': confidence,
            'probabilities': {
                name: float(probs[i]) 
                for i, name in enumerate(self.regime_names)
            },
            'features': {
                'momentum': features[0],
                'volatility': features[1],
                'autocorr': features[2],
                'hurst': features[5]
            }
        }
    
    def train_ensemble_mapping(self, horizons: dict, prices: np.ndarray, 
                                horizon_subset: list, train_end: int):
        """
        Learn which ensemble method works best in each regime.
        """
        from meta_ensemble import (
            PairwiseSlopesEnsemble, 
            AccuracyWeightedEnsemble,
            MagnitudeWeightedEnsemble
        )
        
        methods = [
            PairwiseSlopesEnsemble(),
            MagnitudeWeightedEnsemble(),
        ]
        
        # Track performance by regime
        regime_returns = {
            regime: {m.name: [] for m in methods}
            for regime in self.regime_names
        }
        
        min_h = min(horizon_subset)
        
        # Iterate through training period
        for t in range(self.lookback + 10, train_end):
            if t >= len(prices) or t + min_h >= len(horizons[min_h]['y']):
                continue
            
            # Detect regime
            result = self.detect_regime(prices, t)
            regime = result['regime']
            
            # Calculate actual return
            actual_return = (horizons[min_h]['y'][t] - prices[t]) / prices[t]
            
            # Test each method
            for method in methods:
                sig = method.get_signal(horizons, t, horizon_subset)
                signal = sig.get('signal', np.sign(sig.get('net_prob', 0)))
                strategy_return = signal * actual_return
                regime_returns[regime][method.name].append(strategy_return)
        
        # Find best method per regime
        for regime in self.regime_names:
            best_method = 'pairwise_slopes'
            best_sharpe = -999
            
            for method_name, returns in regime_returns[regime].items():
                if len(returns) > 5:
                    returns = np.array(returns)
                    sharpe = returns.mean() / (returns.std() + 1e-8)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_method = method_name
            
            self.regime_ensemble_map[regime] = {
                'method': best_method,
                'train_sharpe': round(best_sharpe, 3)
            }
        
        return self.regime_ensemble_map


class QuantumEnsembleStrategy:
    """
    Full quantum-powered trading strategy:
    1. Quantum regime detection
    2. Regime-optimal ensemble selection
    3. Signal generation
    """
    
    def __init__(self):
        self.qrd = QuantumRegimeDetector()
        self.trained = False
    
    def train(self, horizons: dict, prices: np.ndarray, 
              horizon_subset: list, train_end: int):
        """Train the quantum strategy."""
        print("  Training quantum regime detector...")
        mapping = self.qrd.train_ensemble_mapping(
            horizons, prices, horizon_subset, train_end
        )
        
        print("  Regime -> Ensemble mapping:")
        for regime, info in mapping.items():
            print(f"    {regime}: {info['method']} (train Sharpe: {info['train_sharpe']})")
        
        self.trained = True
        return mapping
    
    def get_signal(self, horizons: dict, t: int, horizon_subset: list,
                   prices: np.ndarray) -> dict:
        """Get trading signal using quantum regime detection."""
        from meta_ensemble import PairwiseSlopesEnsemble, MagnitudeWeightedEnsemble
        
        # Detect regime
        regime_info = self.qrd.detect_regime(prices, t)
        regime = regime_info['regime']
        
        # Get best method for this regime
        best_method = self.qrd.regime_ensemble_map.get(
            regime, {'method': 'pairwise_slopes'}
        )['method']
        
        # Get signal from selected method
        if best_method == 'magnitude_weighted':
            method = MagnitudeWeightedEnsemble()
        else:
            method = PairwiseSlopesEnsemble()
        
        sig = method.get_signal(horizons, t, horizon_subset)
        
        return {
            'signal': sig.get('signal', np.sign(sig.get('net_prob', 0))),
            'net_prob': sig.get('net_prob', 0),
            'regime': regime,
            'regime_confidence': regime_info['confidence'],
            'selected_method': best_method,
            'regime_probabilities': regime_info['probabilities']
        }


def backtest_quantum_strategy(asset_dir: str, horizon_subset: list = None):
    """Backtest the quantum regime-based strategy."""
    horizons, prices = load_asset_data(asset_dir)
    
    if horizons is None:
        return None
    
    available = sorted(horizons.keys())
    if horizon_subset is None:
        horizon_subset = available
    horizon_subset = [h for h in horizon_subset if h in available]
    
    print(f"  Horizons: {horizon_subset}")
    
    min_len = min(horizons[h]['X'].shape[0] for h in horizon_subset)
    train_end = int(min_len * 0.7)
    
    # Create and train quantum strategy
    strategy = QuantumEnsembleStrategy()
    strategy.train(horizons, prices, horizon_subset, train_end)
    
    # Backtest
    min_h = min(horizon_subset)
    y_future = horizons[min_h]['y']
    
    returns = []
    regime_returns = {r: [] for r in strategy.qrd.regime_names}
    
    print("  Backtesting quantum strategy...")
    
    for t in range(train_end, min_len - min_h):
        sig = strategy.get_signal(horizons, t, horizon_subset, prices)
        signal = sig['signal']
        regime = sig['regime']
        
        if prices[t] != 0:
            actual_return = (y_future[t] - prices[t]) / prices[t]
        else:
            actual_return = 0
        
        strategy_return = signal * actual_return
        returns.append(strategy_return)
        regime_returns[regime].append(strategy_return)
    
    returns = np.array(returns)
    
    # Metrics
    periods = 252 / min_h
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(periods)
    trades = returns[returns != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0
    
    # Per-regime metrics
    regime_metrics = {}
    for regime, rets in regime_returns.items():
        if len(rets) > 3:
            rets = np.array(rets)
            regime_metrics[regime] = {
                'sharpe': round(rets.mean() / (rets.std() + 1e-8) * np.sqrt(periods), 3),
                'win_rate': round((rets[rets != 0] > 0).mean() * 100, 1) if len(rets[rets != 0]) > 0 else 0,
                'n_periods': len(rets)
            }
    
    return {
        'sharpe': round(sharpe, 3),
        'win_rate': round(win_rate * 100, 2),
        'total_return': round(returns.sum() * 100, 2),
        'regime_ensemble_map': strategy.qrd.regime_ensemble_map,
        'regime_metrics': regime_metrics
    }


def run_quantum_comparison():
    """Compare quantum strategy vs simple optimal."""
    with open('configs/optimization_summary.json') as f:
        opt = json.load(f)
    
    dirs = {
        'sp500': 'data/1625_SP500',
        'bitcoin': 'data/1860_Bitcoin',
        'crude_oil': 'data/1866_Crude_Oil'
    }
    
    print("=" * 60)
    print("QUANTUM REGIME DETECTOR vs SIMPLE OPTIMAL")
    print("=" * 60)
    
    results = {}
    
    for asset in ['sp500', 'bitcoin', 'crude_oil']:
        print(f"\n{asset.upper()}")
        print("-" * 40)
        
        best_h = opt['results'][asset]['best_horizons']
        
        result = backtest_quantum_strategy(dirs[asset], best_h)
        
        if result:
            results[asset] = result
            
            print(f"\n  RESULTS:")
            print(f"    Simple Optimal: Sharpe={opt['results'][asset]['sharpe']}, WR={opt['results'][asset]['win_rate']}%")
            print(f"    Quantum Regime: Sharpe={result['sharpe']}, WR={result['win_rate']}%")
            
            print(f"\n  REGIME PERFORMANCE:")
            for regime, metrics in result['regime_metrics'].items():
                method = result['regime_ensemble_map'].get(regime, {}).get('method', '?')
                print(f"    {regime}: Sharpe={metrics['sharpe']}, WR={metrics['win_rate']}%, n={metrics['n_periods']} (using {method})")
    
    # Save results
    output = {
        'generated_at': datetime.now().isoformat(),
        'method': 'quantum_regime_detection',
        'results': results
    }
    
    with open('configs/quantum_strategy_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("Results saved to configs/quantum_strategy_results.json")
    
    return results


if __name__ == "__main__":
    results = run_quantum_comparison()
