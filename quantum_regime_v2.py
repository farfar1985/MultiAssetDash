"""
QUANTUM REGIME DETECTION v2
===========================
Advanced improvements to regime change detection:

1. Multi-Scale Analysis - Detect regimes at different time horizons
2. Quantum Phase Transitions - Model regime changes as phase transitions
3. Transition Prediction - Predict changes BEFORE they happen
4. Adaptive Boundaries - Learn optimal thresholds from data
5. Entanglement Entropy - Measure market complexity/disorder

Created: 2026-02-05
Author: AmiraB
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace, entropy, DensityMatrix
from qiskit.circuit.library import RealAmplitudes

from per_asset_optimizer import load_asset_data


class MultiScaleRegimeDetector:
    """
    Detects regimes at multiple time scales simultaneously.
    
    Key insight: A regime change at the daily scale might not be
    visible at the weekly scale, and vice versa. By combining
    multiple scales, we get more robust detection.
    """
    
    def __init__(self, scales: list = [5, 10, 20, 40]):
        self.scales = scales
        self.n_qubits = len(scales)
        
    def extract_scale_features(self, prices: np.ndarray, t: int, scale: int) -> dict:
        """Extract features at a specific time scale."""
        if t < scale:
            return {'vol': 0, 'trend': 0, 'mean_rev': 0}
        
        window = prices[t - scale:t]
        returns = np.diff(window) / (window[:-1] + 1e-10)
        
        # Volatility at this scale
        vol = returns.std() * np.sqrt(252 / scale)
        
        # Trend strength
        trend = (window[-1] - window[0]) / (window[0] + 1e-10)
        
        # Mean reversion indicator (Hurst proxy)
        if len(returns) > 2:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            mean_rev = -autocorr if not np.isnan(autocorr) else 0
        else:
            mean_rev = 0
        
        return {
            'vol': np.clip(vol, 0, 1),
            'trend': np.clip(trend, -1, 1),
            'mean_rev': np.clip(mean_rev, -1, 1)
        }
    
    def encode_multiscale(self, prices: np.ndarray, t: int) -> QuantumCircuit:
        """Encode multi-scale features into quantum state."""
        qc = QuantumCircuit(self.n_qubits)
        
        for i, scale in enumerate(self.scales):
            features = self.extract_scale_features(prices, t, scale)
            
            # Encode volatility as Y rotation
            vol_angle = features['vol'] * np.pi
            qc.ry(vol_angle, i)
            
            # Encode trend as Z rotation
            trend_angle = (features['trend'] + 1) / 2 * np.pi
            qc.rz(trend_angle, i)
        
        # Entangle scales (correlation between time scales)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        
        return qc
    
    def detect_multiscale_regime(self, prices: np.ndarray, t: int) -> dict:
        """Detect regime using multi-scale analysis."""
        qc = self.encode_multiscale(prices, t)
        sv = Statevector.from_instruction(qc)
        
        # Measure entanglement between scales (market coherence)
        dm = DensityMatrix(sv)
        
        # Entanglement entropy between short and long scales
        short_scales = list(range(self.n_qubits // 2))
        long_scales = list(range(self.n_qubits // 2, self.n_qubits))
        
        rho_short = partial_trace(sv, long_scales)
        coherence = entropy(rho_short, base=2)
        
        # Extract per-scale features
        scale_features = {}
        for scale in self.scales:
            scale_features[f'scale_{scale}'] = self.extract_scale_features(prices, t, scale)
        
        # Aggregate volatility across scales
        vols = [f['vol'] for f in scale_features.values()]
        avg_vol = np.mean(vols)
        vol_dispersion = np.std(vols)  # High dispersion = regime uncertainty
        
        # Determine regime
        if avg_vol < 0.15:
            regime = 'LOW_VOL'
        elif avg_vol < 0.30:
            regime = 'NORMAL'
        elif avg_vol < 0.50:
            regime = 'ELEVATED'
        else:
            regime = 'CRISIS'
        
        return {
            'regime': regime,
            'avg_volatility': round(avg_vol, 4),
            'vol_dispersion': round(vol_dispersion, 4),
            'scale_coherence': round(coherence, 4),
            'scale_features': scale_features,
            'confidence': round(1 - vol_dispersion, 3)
        }


class QuantumPhaseTransitionDetector:
    """
    Models regime changes as quantum phase transitions.
    
    Key insight: Just like quantum systems undergo phase transitions
    at critical points, markets transition between regimes at
    critical volatility/correlation thresholds.
    
    Uses order parameter (like magnetization in physics) to detect
    when the market is near a phase transition.
    """
    
    def __init__(self, n_qubits: int = 6):
        self.n_qubits = n_qubits
        self.critical_points = []
        
    def calculate_order_parameter(self, prices: np.ndarray, t: int, 
                                   lookback: int = 20) -> float:
        """
        Calculate order parameter (market "magnetization").
        
        High order parameter = Market in ordered state (trending)
        Low order parameter = Market in disordered state (mean-reverting)
        """
        if t < lookback:
            return 0.5
        
        window = prices[t - lookback:t]
        returns = np.diff(window) / (window[:-1] + 1e-10)
        
        # Order parameter: ratio of directional moves to total moves
        up_moves = (returns > 0).sum()
        total_moves = len(returns)
        
        # Normalize to [-1, 1]
        order_param = (2 * up_moves / total_moves) - 1
        
        return order_param
    
    def calculate_susceptibility(self, prices: np.ndarray, t: int,
                                  lookback: int = 40) -> float:
        """
        Calculate susceptibility (sensitivity to perturbations).
        
        High susceptibility near phase transitions = market very
        sensitive to news/shocks.
        """
        if t < lookback:
            return 0
        
        # Rolling order parameter
        order_params = []
        for i in range(lookback - 20, 0, -1):
            op = self.calculate_order_parameter(prices, t - i, 20)
            order_params.append(op)
        
        # Susceptibility = variance of order parameter
        # High variance = near critical point
        susceptibility = np.var(order_params) if order_params else 0
        
        return susceptibility
    
    def encode_phase_state(self, order_param: float, susceptibility: float,
                           vol: float) -> QuantumCircuit:
        """Encode market phase into quantum state."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Order parameter determines superposition
        theta = (order_param + 1) / 2 * np.pi
        
        # Apply to all qubits (collective state)
        for i in range(self.n_qubits):
            qc.ry(theta, i)
        
        # Susceptibility determines entanglement strength
        ent_angle = susceptibility * np.pi * 2
        for i in range(self.n_qubits - 1):
            qc.crz(ent_angle, i, i + 1)
        
        # Volatility adds disorder
        for i in range(self.n_qubits):
            qc.rx(vol * np.pi, i)
        
        return qc
    
    def detect_phase_transition(self, prices: np.ndarray, t: int) -> dict:
        """Detect if market is near a phase transition."""
        # Calculate phase indicators
        order_param = self.calculate_order_parameter(prices, t)
        susceptibility = self.calculate_susceptibility(prices, t)
        
        # Calculate volatility
        if t >= 20:
            window = prices[t-20:t]
            returns = np.diff(window) / (window[:-1] + 1e-10)
            vol = returns.std() * np.sqrt(252)
        else:
            vol = 0.2
        
        vol_normalized = np.clip(vol, 0, 1)
        
        # Encode into quantum state
        qc = self.encode_phase_state(order_param, susceptibility, vol_normalized)
        sv = Statevector.from_instruction(qc)
        
        # Measure quantum coherence (indicates phase stability)
        dm = DensityMatrix(sv)
        coherence = entropy(dm, base=2)
        
        # Near phase transition when:
        # 1. High susceptibility
        # 2. Order parameter near 0 (between ordered and disordered)
        # 3. High quantum coherence loss
        
        transition_score = (
            susceptibility * 0.4 +
            (1 - abs(order_param)) * 0.3 +
            (coherence / self.n_qubits) * 0.3
        )
        
        # Classify phase
        if abs(order_param) > 0.5:
            phase = 'TRENDING'
        elif susceptibility > 0.1:
            phase = 'CRITICAL'  # Near transition!
        else:
            phase = 'MEAN_REVERTING'
        
        # Is transition imminent?
        transition_imminent = transition_score > 0.5 and susceptibility > 0.08
        
        if transition_imminent:
            self.critical_points.append(t)
        
        return {
            'phase': phase,
            'order_parameter': round(order_param, 4),
            'susceptibility': round(susceptibility, 4),
            'transition_score': round(transition_score, 4),
            'transition_imminent': transition_imminent,
            'volatility': round(vol, 4),
            'quantum_coherence': round(coherence, 4)
        }


class AdaptiveRegimeBoundaries:
    """
    Learns optimal regime boundaries from data using quantum-inspired
    optimization.
    """
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.boundaries = None
        self.regime_names = ['LOW_VOL', 'NORMAL', 'ELEVATED', 'CRISIS']
        
    def extract_vol_series(self, prices: np.ndarray, lookback: int = 20) -> np.ndarray:
        """Extract rolling volatility series."""
        vols = []
        for t in range(lookback, len(prices)):
            returns = np.diff(prices[t-lookback:t]) / (prices[t-lookback:t-1] + 1e-10)
            vol = returns.std() * np.sqrt(252)
            vols.append(vol)
        return np.array(vols)
    
    def fit_boundaries(self, prices: np.ndarray) -> dict:
        """Learn optimal boundaries using Gaussian Mixture Model."""
        vols = self.extract_vol_series(prices)
        
        if len(vols) < 50:
            # Default boundaries
            self.boundaries = [0.15, 0.25, 0.40]
            return {'boundaries': self.boundaries, 'method': 'default'}
        
        # Fit GMM to find natural clusters
        gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
        gmm.fit(vols.reshape(-1, 1))
        
        # Get cluster centers and sort
        centers = gmm.means_.flatten()
        centers = np.sort(centers)
        
        # Boundaries are midpoints between centers
        self.boundaries = []
        for i in range(len(centers) - 1):
            boundary = (centers[i] + centers[i + 1]) / 2
            self.boundaries.append(boundary)
        
        return {
            'boundaries': [round(b, 4) for b in self.boundaries],
            'centers': [round(c, 4) for c in centers],
            'method': 'gmm_fitted'
        }
    
    def classify_regime(self, vol: float) -> tuple:
        """Classify regime based on learned boundaries."""
        if self.boundaries is None:
            self.boundaries = [0.15, 0.25, 0.40]
        
        for i, boundary in enumerate(self.boundaries):
            if vol < boundary:
                return i, self.regime_names[i]
        
        return len(self.boundaries), self.regime_names[-1]


class TransitionPredictor:
    """
    Predicts regime transitions before they happen using quantum
    interference patterns.
    """
    
    def __init__(self, history_length: int = 10):
        self.history_length = history_length
        self.history = []
        
    def update(self, regime_info: dict):
        """Add new observation to history."""
        self.history.append(regime_info)
        if len(self.history) > self.history_length:
            self.history = self.history[-self.history_length:]
    
    def calculate_transition_probability(self) -> dict:
        """
        Calculate probability of regime transition in next period.
        
        Uses:
        1. Regime stability (how long in current regime)
        2. Volatility trend
        3. Susceptibility trend
        """
        if len(self.history) < 3:
            return {'probability': 0, 'direction': 'STABLE'}
        
        # Current regime and stability
        current_regime = self.history[-1].get('regime') or self.history[-1].get('phase')
        regime_duration = 1
        for i in range(len(self.history) - 2, -1, -1):
            prev_regime = self.history[i].get('regime') or self.history[i].get('phase')
            if prev_regime == current_regime:
                regime_duration += 1
            else:
                break
        
        # Volatility trend
        vols = [h.get('volatility', h.get('avg_volatility', 0.2)) for h in self.history]
        if len(vols) >= 2:
            vol_trend = (vols[-1] - vols[0]) / (len(vols) - 1)
        else:
            vol_trend = 0
        
        # Susceptibility/transition score trend
        scores = [h.get('transition_score', h.get('susceptibility', 0)) for h in self.history]
        if len(scores) >= 2:
            score_trend = (scores[-1] - scores[0]) / (len(scores) - 1)
        else:
            score_trend = 0
        
        # Calculate transition probability
        # Higher when: short duration, rising vol, rising susceptibility
        duration_factor = np.exp(-regime_duration / 5)  # Decays with duration
        vol_factor = np.clip(vol_trend * 5, 0, 0.5)
        score_factor = np.clip(score_trend * 3, 0, 0.3)
        
        prob = duration_factor * 0.4 + vol_factor + score_factor
        prob = np.clip(prob, 0, 1)
        
        # Predict direction
        if vol_trend > 0.02:
            direction = 'RISK_ON'  # Transitioning to higher vol regime
        elif vol_trend < -0.02:
            direction = 'RISK_OFF'  # Transitioning to lower vol regime
        else:
            direction = 'STABLE'
        
        return {
            'probability': round(prob, 3),
            'direction': direction,
            'regime_duration': regime_duration,
            'vol_trend': round(vol_trend, 4),
            'current_regime': current_regime
        }


class QuantumRegimeDetectorV2:
    """
    Complete v2 quantum regime detector combining all improvements.
    """
    
    def __init__(self):
        self.multiscale = MultiScaleRegimeDetector()
        self.phase = QuantumPhaseTransitionDetector()
        self.boundaries = AdaptiveRegimeBoundaries()
        self.predictor = TransitionPredictor()
        self.trained = False
        
    def train(self, prices: np.ndarray):
        """Train adaptive components."""
        boundary_info = self.boundaries.fit_boundaries(prices)
        self.trained = True
        return boundary_info
    
    def detect(self, prices: np.ndarray, t: int) -> dict:
        """
        Complete regime detection with all v2 features.
        """
        # Multi-scale analysis
        multiscale_result = self.multiscale.detect_multiscale_regime(prices, t)
        
        # Phase transition analysis
        phase_result = self.phase.detect_phase_transition(prices, t)
        
        # Adaptive regime classification
        vol = phase_result['volatility']
        regime_idx, regime_name = self.boundaries.classify_regime(vol)
        
        # Combine results
        combined = {
            'regime': regime_name,
            'regime_idx': regime_idx,
            'volatility': vol,
            
            # Multi-scale
            'scale_coherence': multiscale_result['scale_coherence'],
            'vol_dispersion': multiscale_result['vol_dispersion'],
            
            # Phase transition
            'phase': phase_result['phase'],
            'order_parameter': phase_result['order_parameter'],
            'susceptibility': phase_result['susceptibility'],
            'transition_imminent': phase_result['transition_imminent'],
            
            # Confidence
            'confidence': (multiscale_result['confidence'] + 
                          (1 - phase_result['transition_score'])) / 2
        }
        
        # Update predictor
        self.predictor.update(combined)
        
        # Get transition prediction
        prediction = self.predictor.calculate_transition_probability()
        combined['transition_prediction'] = prediction
        
        return combined


def backtest_v2_detector():
    """Backtest the v2 quantum regime detector."""
    print("=" * 60)
    print("QUANTUM REGIME DETECTOR V2 BACKTEST")
    print("=" * 60)
    
    dirs = {
        'SP500': 'data/1625_SP500',
        'Bitcoin': 'data/1860_Bitcoin',
        'Crude_Oil': 'data/1866_Crude_Oil'
    }
    
    all_results = {}
    
    for asset_name, asset_dir in dirs.items():
        print(f"\n{'-'*40}")
        print(f"Testing: {asset_name}")
        print(f"{'-'*40}")
        
        horizons, prices = load_asset_data(asset_dir)
        if prices is None:
            continue
        
        # Initialize and train detector
        detector = QuantumRegimeDetectorV2()
        boundary_info = detector.train(prices)
        print(f"  Learned boundaries: {boundary_info}")
        
        # Run detection
        results = []
        transitions_predicted = 0
        transitions_correct = 0
        
        for t in range(50, len(prices) - 5):
            if t % 50 == 0:
                print(f"  Processing t={t}/{len(prices)}")
            
            result = detector.detect(prices, t)
            results.append(result)
            
            # Track prediction accuracy
            if result['transition_prediction']['probability'] > 0.5:
                transitions_predicted += 1
                
                # Check if transition actually happened
                if t + 5 < len(prices):
                    future_result = detector.detect(prices, t + 5)
                    if future_result['regime'] != result['regime']:
                        transitions_correct += 1
        
        # Analyze results
        regimes = [r['regime'] for r in results]
        phases = [r['phase'] for r in results]
        imminent = [r['transition_imminent'] for r in results]
        
        regime_dist = pd.Series(regimes).value_counts()
        phase_dist = pd.Series(phases).value_counts()
        
        print(f"\n  REGIME DISTRIBUTION:")
        for regime, count in regime_dist.items():
            print(f"    {regime}: {count} ({count/len(regimes)*100:.1f}%)")
        
        print(f"\n  PHASE DISTRIBUTION:")
        for phase, count in phase_dist.items():
            print(f"    {phase}: {count} ({count/len(phases)*100:.1f}%)")
        
        print(f"\n  TRANSITION DETECTION:")
        print(f"    Transitions flagged: {sum(imminent)}")
        print(f"    Predictions made: {transitions_predicted}")
        if transitions_predicted > 0:
            print(f"    Prediction accuracy: {transitions_correct/transitions_predicted*100:.1f}%")
        
        # Average metrics
        avg_coherence = np.mean([r['scale_coherence'] for r in results])
        avg_susceptibility = np.mean([r['susceptibility'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"\n  AVERAGE METRICS:")
        print(f"    Scale coherence: {avg_coherence:.3f}")
        print(f"    Susceptibility: {avg_susceptibility:.3f}")
        print(f"    Confidence: {avg_confidence:.3f}")
        
        all_results[asset_name] = {
            'regime_distribution': regime_dist.to_dict(),
            'phase_distribution': phase_dist.to_dict(),
            'transitions_flagged': sum(imminent),
            'predictions_made': transitions_predicted,
            'prediction_accuracy': transitions_correct / transitions_predicted if transitions_predicted > 0 else 0,
            'avg_coherence': round(avg_coherence, 3),
            'avg_susceptibility': round(avg_susceptibility, 3),
            'avg_confidence': round(avg_confidence, 3),
            'learned_boundaries': boundary_info['boundaries']
        }
    
    # Save results
    output = {
        'generated_at': datetime.now().isoformat(),
        'version': 'v2',
        'improvements': [
            'multi_scale_analysis',
            'phase_transition_detection',
            'adaptive_boundaries',
            'transition_prediction'
        ],
        'results': all_results
    }
    
    output_path = Path('configs/quantum_regime_v2_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    results = backtest_v2_detector()
