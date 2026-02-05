"""
QUANTUM CROSS-ASSET CONTAGION DETECTOR
======================================
Uses quantum entanglement simulation to detect when volatility
spreads from one asset to another.

Key insight: During market stress, correlations spike and assets
that normally move independently become entangled.

This is critical for CME hedging clients who need to know when
their diversification breaks down.

Created: 2026-02-05
Author: AmiraB
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace, entropy

from per_asset_optimizer import load_asset_data
from quantum_volatility_detector import EnhancedQuantumRegimeDetector


class QuantumEntanglementMeasure:
    """
    Measures quantum entanglement between asset volatility states.
    High entanglement = assets moving together (contagion risk).
    """
    
    def __init__(self):
        self.simulator = AerSimulator(method='statevector')
    
    def encode_two_assets(self, features_a: np.ndarray, features_b: np.ndarray,
                          correlation: float) -> QuantumCircuit:
        """
        Encode two assets into entangled quantum state.
        Correlation strength determines entanglement degree.
        """
        qc = QuantumCircuit(4)  # 2 qubits per asset
        
        # Asset A: encode volatility and trend
        vol_a = np.clip(features_a[0], 0, 1) * np.pi
        trend_a = np.clip((features_a[1] + 1) / 2, 0, 1) * np.pi
        qc.ry(vol_a, 0)
        qc.ry(trend_a, 1)
        
        # Asset B: encode volatility and trend
        vol_b = np.clip(features_b[0], 0, 1) * np.pi
        trend_b = np.clip((features_b[1] + 1) / 2, 0, 1) * np.pi
        qc.ry(vol_b, 2)
        qc.ry(trend_b, 3)
        
        # Entangle based on correlation
        # Higher correlation = stronger entanglement
        entangle_angle = abs(correlation) * np.pi / 2
        
        # CNOT + rotation creates partial entanglement
        qc.cx(0, 2)  # Volatility entanglement
        qc.rz(entangle_angle, 2)
        qc.cx(0, 2)
        
        qc.cx(1, 3)  # Trend entanglement
        qc.rz(entangle_angle, 3)
        qc.cx(1, 3)
        
        return qc
    
    def measure_entanglement(self, qc: QuantumCircuit) -> dict:
        """
        Measure entanglement using von Neumann entropy.
        """
        sv = Statevector.from_instruction(qc)
        
        # Partial trace over asset B to get reduced density matrix of A
        rho_a = partial_trace(sv, [2, 3])
        
        # Von Neumann entropy of reduced state
        # S = 0 means pure state (no entanglement)
        # S = 1 means maximally entangled
        s_a = entropy(rho_a, base=2)
        
        # Also measure B
        rho_b = partial_trace(sv, [0, 1])
        s_b = entropy(rho_b, base=2)
        
        # Average entropy as entanglement measure
        entanglement = (s_a + s_b) / 2
        
        return {
            'entanglement': float(entanglement),
            'entropy_a': float(s_a),
            'entropy_b': float(s_b),
            'max_entanglement': 2.0,  # Max for 2 qubits
            'normalized': float(entanglement / 2.0)
        }


class ContagionDetector:
    """
    Detects volatility contagion between multiple assets.
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.qem = QuantumEntanglementMeasure()
        self.regime_detectors = {}
        
        # Contagion thresholds
        self.thresholds = {
            'LOW': 0.3,
            'MODERATE': 0.5,
            'HIGH': 0.7,
            'CRITICAL': 0.85
        }
    
    def calculate_rolling_correlation(self, returns_a: np.ndarray, 
                                       returns_b: np.ndarray) -> float:
        """Calculate rolling correlation between two return series."""
        if len(returns_a) < 5 or len(returns_b) < 5:
            return 0.0
        
        min_len = min(len(returns_a), len(returns_b))
        corr = np.corrcoef(returns_a[-min_len:], returns_b[-min_len:])[0, 1]
        
        return 0.0 if np.isnan(corr) else corr
    
    def extract_features(self, prices: np.ndarray, t: int) -> np.ndarray:
        """Extract features for quantum encoding."""
        if t < self.lookback:
            return np.zeros(4)
        
        window = prices[t - self.lookback:t]
        returns = np.diff(window) / (window[:-1] + 1e-10)
        
        # Realized volatility (normalized)
        vol = returns.std() * np.sqrt(252)
        vol_norm = np.clip(vol / 0.5, 0, 1)
        
        # Momentum
        momentum = (window[-1] - window[0]) / (window[0] + 1e-10)
        
        # Volatility trend
        half = len(returns) // 2
        vol_first = returns[:half].std() if half > 1 else 0
        vol_second = returns[half:].std() if half > 1 else 0
        vol_trend = (vol_second - vol_first) / (vol_first + 1e-10)
        
        # Tail risk (kurtosis proxy)
        tail_risk = np.clip(abs(returns).max() / (returns.std() + 1e-10) / 5, 0, 1)
        
        return np.array([vol_norm, momentum, vol_trend, tail_risk])
    
    def detect_contagion(self, asset_prices: dict, t: int) -> dict:
        """
        Detect contagion between all asset pairs.
        """
        assets = list(asset_prices.keys())
        n_assets = len(assets)
        
        # Extract features and returns for all assets
        features = {}
        returns = {}
        
        for asset, prices in asset_prices.items():
            features[asset] = self.extract_features(prices, t)
            if t >= self.lookback:
                window = prices[t - self.lookback:t]
                returns[asset] = np.diff(window) / (window[:-1] + 1e-10)
            else:
                returns[asset] = np.array([0])
        
        # Calculate pairwise contagion
        contagion_matrix = np.zeros((n_assets, n_assets))
        pair_details = {}
        
        for i, asset_a in enumerate(assets):
            for j, asset_b in enumerate(assets):
                if i >= j:
                    continue
                
                # Classical correlation
                corr = self.calculate_rolling_correlation(
                    returns[asset_a], returns[asset_b]
                )
                
                # Quantum entanglement
                qc = self.qem.encode_two_assets(
                    features[asset_a], features[asset_b], corr
                )
                entanglement = self.qem.measure_entanglement(qc)
                
                # Combined contagion score
                # Weight entanglement higher when correlation is high
                contagion_score = (
                    abs(corr) * 0.4 + 
                    entanglement['normalized'] * 0.4 +
                    (features[asset_a][0] + features[asset_b][0]) / 2 * 0.2  # Vol level
                )
                
                contagion_matrix[i, j] = contagion_score
                contagion_matrix[j, i] = contagion_score
                
                pair_details[f"{asset_a}-{asset_b}"] = {
                    'correlation': round(corr, 3),
                    'entanglement': round(entanglement['normalized'], 3),
                    'contagion_score': round(contagion_score, 3),
                    'vol_a': round(features[asset_a][0] * 0.5, 3),
                    'vol_b': round(features[asset_b][0] * 0.5, 3)
                }
        
        # System-wide contagion (average of all pairs)
        upper_tri = contagion_matrix[np.triu_indices(n_assets, k=1)]
        system_contagion = upper_tri.mean() if len(upper_tri) > 0 else 0
        max_contagion = upper_tri.max() if len(upper_tri) > 0 else 0
        
        # Determine alert level
        if max_contagion >= self.thresholds['CRITICAL']:
            alert = 'CRITICAL'
        elif max_contagion >= self.thresholds['HIGH']:
            alert = 'HIGH'
        elif max_contagion >= self.thresholds['MODERATE']:
            alert = 'MODERATE'
        else:
            alert = 'LOW'
        
        # Find highest contagion pair
        max_pair = max(pair_details, key=lambda k: pair_details[k]['contagion_score'])
        
        return {
            'timestamp_idx': t,
            'system_contagion': round(system_contagion, 3),
            'max_contagion': round(max_contagion, 3),
            'alert_level': alert,
            'highest_contagion_pair': max_pair,
            'pair_details': pair_details,
            'contagion_matrix': contagion_matrix.tolist()
        }


class ContagionEarlyWarning:
    """
    Predicts contagion events before they fully materialize.
    """
    
    def __init__(self, history_length: int = 10):
        self.history_length = history_length
        self.contagion_history = []
    
    def update(self, contagion_result: dict):
        """Add new contagion measurement to history."""
        self.contagion_history.append(contagion_result)
        if len(self.contagion_history) > self.history_length:
            self.contagion_history = self.contagion_history[-self.history_length:]
    
    def predict_contagion_spike(self) -> dict:
        """
        Predict if contagion is likely to spike.
        Uses rate of change in contagion scores.
        """
        if len(self.contagion_history) < 3:
            return {'warning': None, 'probability': 0}
        
        # Get recent system contagion values
        recent = [h['system_contagion'] for h in self.contagion_history[-5:]]
        
        # Calculate trend
        if len(recent) >= 2:
            trend = (recent[-1] - recent[0]) / (len(recent) - 1)
        else:
            trend = 0
        
        # Calculate acceleration
        if len(recent) >= 3:
            first_diff = recent[-2] - recent[-3]
            second_diff = recent[-1] - recent[-2]
            acceleration = second_diff - first_diff
        else:
            acceleration = 0
        
        # Volatility of contagion (instability)
        contagion_vol = np.std(recent) if len(recent) > 1 else 0
        
        # Predict probability of spike
        spike_probability = (
            np.clip(trend * 2, 0, 0.4) +
            np.clip(acceleration * 3, 0, 0.3) +
            np.clip(contagion_vol * 2, 0, 0.3)
        )
        
        # Generate warning
        warning = None
        if spike_probability > 0.6:
            warning = 'CONTAGION SPIKE IMMINENT'
        elif spike_probability > 0.4:
            warning = 'ELEVATED CONTAGION RISK'
        elif trend > 0.05:
            warning = 'RISING CORRELATION'
        
        return {
            'warning': warning,
            'probability': round(spike_probability, 3),
            'trend': round(trend, 4),
            'acceleration': round(acceleration, 4),
            'current_level': recent[-1] if recent else 0
        }


def backtest_contagion_detection():
    """Run contagion detection across all assets."""
    print("=" * 60)
    print("QUANTUM CONTAGION DETECTION BACKTEST")
    print("=" * 60)
    
    # Load all assets
    dirs = {
        'SP500': 'data/1625_SP500',
        'Bitcoin': 'data/1860_Bitcoin',
        'Crude_Oil': 'data/1866_Crude_Oil'
    }
    
    asset_prices = {}
    min_len = float('inf')
    
    for asset_name, asset_dir in dirs.items():
        horizons, prices = load_asset_data(asset_dir)
        if prices is not None:
            asset_prices[asset_name] = prices
            min_len = min(min_len, len(prices))
            print(f"  Loaded {asset_name}: {len(prices)} periods")
    
    if len(asset_prices) < 2:
        print("ERROR: Need at least 2 assets")
        return None
    
    # Truncate to common length
    for asset in asset_prices:
        asset_prices[asset] = asset_prices[asset][:int(min_len)]
    
    print(f"\n  Common length: {int(min_len)} periods")
    
    # Initialize detector
    detector = ContagionDetector(lookback=20)
    ews = ContagionEarlyWarning()
    
    # Run detection
    results = []
    alerts = []
    
    print("\n  Running contagion detection...")
    
    for t in range(25, int(min_len) - 5):
        if t % 50 == 0:
            print(f"    t={t}")
        
        result = detector.detect_contagion(asset_prices, t)
        results.append(result)
        
        # Update early warning
        ews.update(result)
        prediction = ews.predict_contagion_spike()
        
        if prediction['warning']:
            alerts.append({
                't': t,
                'warning': prediction['warning'],
                'probability': prediction['probability'],
                'contagion_level': result['system_contagion']
            })
    
    # Analyze results
    print(f"\n  RESULTS:")
    print(f"  {'='*50}")
    
    # Contagion distribution
    contagion_levels = [r['system_contagion'] for r in results]
    alert_levels = [r['alert_level'] for r in results]
    
    print(f"\n  System Contagion Statistics:")
    print(f"    Mean: {np.mean(contagion_levels):.3f}")
    print(f"    Max:  {np.max(contagion_levels):.3f}")
    print(f"    Min:  {np.min(contagion_levels):.3f}")
    print(f"    Std:  {np.std(contagion_levels):.3f}")
    
    print(f"\n  Alert Level Distribution:")
    for level in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']:
        count = alert_levels.count(level)
        pct = count / len(alert_levels) * 100
        print(f"    {level}: {count} ({pct:.1f}%)")
    
    print(f"\n  Early Warnings Issued: {len(alerts)}")
    if alerts:
        print("    Recent warnings:")
        for a in alerts[-5:]:
            print(f"      t={a['t']}: {a['warning']} (prob={a['probability']:.2f}, level={a['contagion_level']:.2f})")
    
    # Pair analysis
    print(f"\n  Pair Contagion Summary:")
    all_pairs = {}
    for r in results:
        for pair, details in r['pair_details'].items():
            if pair not in all_pairs:
                all_pairs[pair] = []
            all_pairs[pair].append(details['contagion_score'])
    
    for pair, scores in all_pairs.items():
        print(f"    {pair}: Mean={np.mean(scores):.3f}, Max={np.max(scores):.3f}")
    
    # Save results
    output = {
        'generated_at': datetime.now().isoformat(),
        'method': 'quantum_contagion_detection',
        'assets': list(asset_prices.keys()),
        'n_periods': len(results),
        'statistics': {
            'mean_contagion': round(np.mean(contagion_levels), 3),
            'max_contagion': round(np.max(contagion_levels), 3),
            'std_contagion': round(np.std(contagion_levels), 3)
        },
        'alert_distribution': {
            level: alert_levels.count(level) 
            for level in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        },
        'pair_summary': {
            pair: {
                'mean': round(np.mean(scores), 3),
                'max': round(np.max(scores), 3)
            }
            for pair, scores in all_pairs.items()
        },
        'warnings': alerts[-20:],  # Last 20 warnings
        'sample_results': results[-10:]  # Last 10 detailed results
    }
    
    output_path = Path('configs/contagion_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    results = backtest_contagion_detection()
