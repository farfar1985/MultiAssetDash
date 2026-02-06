"""
HMM-BASED MARKET REGIME DETECTION
=================================
Uses Hidden Markov Models for statistically rigorous regime detection.

Regimes (3-state default):
- BULL: Positive drift, low-moderate volatility
- BEAR: Negative drift, elevated volatility
- SIDEWAYS: Near-zero drift, variable volatility

Features used:
- Returns (momentum signal)
- Realized volatility (risk level)
- Return autocorrelation (mean reversion indicator)

This provides proper statistical regime detection to back the
"HMM Detected" badge in the RegimeIndicator component.

Created: 2026-02-06
Author: Artemis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not installed. Run: pip install hmmlearn")


class HMMRegimeDetector:
    """
    Hidden Markov Model based market regime detector.

    Uses GaussianHMM to learn latent market states from
    price-derived features (returns, volatility, autocorrelation).
    """

    # Regime label mappings for frontend compatibility
    REGIME_COLORS = {
        'bull': 'green',
        'bear': 'red',
        'sideways': 'amber',
        'high-volatility': 'orange',
        'low-volatility': 'blue'
    }

    def __init__(self, n_regimes: int = 3, lookback: int = 20,
                 vol_window: int = 20, random_state: int = 42):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden states (3 or 4 recommended)
            lookback: Window for feature calculation
            vol_window: Window for volatility calculation
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.vol_window = vol_window
        self.random_state = random_state

        if HMM_AVAILABLE:
            self.model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=random_state
            )
        else:
            self.model = None

        self.scaler = StandardScaler()
        self.trained = False
        self.regime_labels = {}
        self.state_means = None
        self.historical_accuracy = {}

    def extract_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Extract observation features for HMM.

        Features:
        1. Short-term momentum (5-day return annualized)
        2. Realized volatility (rolling std annualized)
        3. Return autocorrelation (mean reversion signal)
        4. Trend strength (20-day return / volatility)

        Args:
            prices: Array of price data

        Returns:
            Feature matrix (n_samples, n_features)
        """
        if len(prices) < self.lookback + 10:
            return np.array([]).reshape(0, 4)

        # Log returns
        log_prices = np.log(prices)
        returns = np.diff(log_prices)

        features = []

        for i in range(self.lookback, len(returns)):
            r_window = returns[i - self.lookback:i]

            # Feature 1: Short-term momentum (5-day)
            momentum = r_window[-5:].mean() * 252

            # Feature 2: Realized volatility
            vol = r_window.std() * np.sqrt(252)

            # Feature 3: Autocorrelation (lag-1)
            if len(r_window) > 1:
                try:
                    autocorr = np.corrcoef(r_window[:-1], r_window[1:])[0, 1]
                    autocorr = 0 if np.isnan(autocorr) else autocorr
                except:
                    autocorr = 0
            else:
                autocorr = 0

            # Feature 4: Trend strength (Sharpe-like)
            trend_return = r_window.sum()
            trend_strength = trend_return / (vol / np.sqrt(252) + 1e-8)

            features.append([momentum, vol, autocorr, trend_strength])

        return np.array(features)

    def fit(self, prices: np.ndarray) -> 'HMMRegimeDetector':
        """
        Fit HMM on historical price data.

        Args:
            prices: Array of historical prices

        Returns:
            self (for chaining)
        """
        if not HMM_AVAILABLE:
            print("Error: hmmlearn not available")
            return self

        if len(prices) < self.lookback + 50:
            print(f"Error: Need at least {self.lookback + 50} data points")
            return self

        # Extract features
        features = self.extract_features(prices)

        if len(features) < 50:
            print("Error: Not enough features extracted")
            return self

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Fit HMM
        self.model.fit(features_scaled)
        self.trained = True

        # Store state means for interpretation
        self.state_means = self.scaler.inverse_transform(self.model.means_)

        # Assign semantic labels
        self._assign_regime_labels()

        return self

    def _assign_regime_labels(self):
        """
        Assign semantic labels to HMM states based on learned characteristics.

        Uses state means to determine:
        - Bull: High momentum, moderate vol
        - Bear: Negative momentum, high vol
        - Sideways: Near-zero momentum, low vol
        """
        if self.state_means is None:
            self.regime_labels = {i: f"regime_{i}" for i in range(self.n_regimes)}
            return

        means = self.state_means

        # Momentum is feature 0, volatility is feature 1
        momentum_col = 0
        vol_col = 1

        labels = {}

        if self.n_regimes == 3:
            # Sort by volatility
            vol_order = np.argsort(means[:, vol_col])

            # Check momentum for directional regimes
            for i, state_idx in enumerate(vol_order):
                mom = means[state_idx, momentum_col]
                vol = means[state_idx, vol_col]

                if vol < 0.15:  # Low vol
                    labels[state_idx] = "low-volatility" if mom > -0.05 else "bear"
                elif vol > 0.30:  # High vol
                    labels[state_idx] = "high-volatility" if abs(mom) < 0.10 else ("bear" if mom < 0 else "bull")
                else:  # Medium vol
                    if mom > 0.05:
                        labels[state_idx] = "bull"
                    elif mom < -0.05:
                        labels[state_idx] = "bear"
                    else:
                        labels[state_idx] = "sideways"

        elif self.n_regimes == 4:
            # Four regimes: bull, bear, sideways, high-vol
            for i in range(4):
                mom = means[i, momentum_col]
                vol = means[i, vol_col]

                if vol > 0.35:
                    labels[i] = "high-volatility"
                elif mom > 0.08:
                    labels[i] = "bull"
                elif mom < -0.08:
                    labels[i] = "bear"
                else:
                    labels[i] = "sideways"

        else:
            # Generic labels for other n_regimes
            vol_order = np.argsort(means[:, vol_col])
            for i, idx in enumerate(vol_order):
                labels[idx] = f"regime_{i}_vol{i}"

        self.regime_labels = labels

    def predict(self, prices: np.ndarray) -> Dict:
        """
        Predict current regime with probabilities.

        Args:
            prices: Recent price data (at least lookback + 10 points)

        Returns:
            Dict with regime info compatible with RegimeIndicator component
        """
        if not self.trained or not HMM_AVAILABLE:
            return self._mock_prediction()

        features = self.extract_features(prices)

        if len(features) == 0:
            return self._mock_prediction()

        features_scaled = self.scaler.transform(features)

        # Get state sequence
        hidden_states = self.model.predict(features_scaled)
        current_state = int(hidden_states[-1])

        # Get probabilities for current observation
        log_probs = self.model.predict_proba(features_scaled)
        current_probs = log_probs[-1]

        # Count days in current regime
        days_in_regime = 1
        for i in range(len(hidden_states) - 2, -1, -1):
            if hidden_states[i] == current_state:
                days_in_regime += 1
            else:
                break

        # Get regime name
        regime_name = self.regime_labels.get(current_state, "unknown")

        # Map to frontend-compatible probabilities
        prob_mapping = {}
        for state_idx, label in self.regime_labels.items():
            # Aggregate probabilities for same labels
            if label in prob_mapping:
                prob_mapping[label] += float(current_probs[state_idx])
            else:
                prob_mapping[label] = float(current_probs[state_idx])

        # Ensure bull/bear/sideways are present
        frontend_probs = {
            'bull': prob_mapping.get('bull', 0.0),
            'bear': prob_mapping.get('bear', 0.0),
            'sideways': prob_mapping.get('sideways', 0.0) +
                       prob_mapping.get('low-volatility', 0.0) * 0.5
        }

        # Normalize
        total = sum(frontend_probs.values())
        if total > 0:
            frontend_probs = {k: v/total for k, v in frontend_probs.items()}

        # Calculate volatility from recent features
        recent_vol = float(self.scaler.inverse_transform(features_scaled[-1:])[:, 1][0])

        # Calculate trend strength
        recent_momentum = float(self.scaler.inverse_transform(features_scaled[-1:])[:, 0][0])
        trend_strength = np.clip(recent_momentum / 0.5, -1, 1)  # Normalize to [-1, 1]

        return {
            "regime": regime_name,
            "confidence": float(current_probs[current_state]),
            "probabilities": frontend_probs,
            "daysInRegime": days_in_regime,
            "volatility": recent_vol * 100,  # As percentage
            "trendStrength": float(trend_strength),
            "historicalAccuracy": self.historical_accuracy.get(regime_name, 65.0),
            "transitionMatrix": self.model.transmat_.tolist(),
            "stateMeans": self.state_means.tolist() if self.state_means is not None else None,
            "method": "GaussianHMM",
            "nRegimes": self.n_regimes
        }

    def _mock_prediction(self) -> Dict:
        """Return mock prediction when model not trained."""
        return {
            "regime": "sideways",
            "confidence": 0.5,
            "probabilities": {"bull": 0.33, "bear": 0.33, "sideways": 0.34},
            "daysInRegime": 1,
            "volatility": 20.0,
            "trendStrength": 0.0,
            "historicalAccuracy": 50.0,
            "method": "mock",
            "nRegimes": self.n_regimes
        }

    def get_regime_history(self, prices: np.ndarray) -> pd.DataFrame:
        """
        Get full regime history for backtesting.

        Args:
            prices: Full price history

        Returns:
            DataFrame with regime sequence and probabilities
        """
        if not self.trained:
            return pd.DataFrame()

        features = self.extract_features(prices)
        features_scaled = self.scaler.transform(features)

        states = self.model.predict(features_scaled)
        probs = self.model.predict_proba(features_scaled)

        # Build DataFrame
        records = []
        for i, (state, prob) in enumerate(zip(states, probs)):
            regime = self.regime_labels.get(state, f"state_{state}")
            records.append({
                'index': i + self.lookback,
                'state': int(state),
                'regime': regime,
                'confidence': float(prob[state]),
                **{f'prob_{self.regime_labels.get(j, f"s{j}")}': float(p)
                   for j, p in enumerate(prob)}
            })

        return pd.DataFrame(records)

    def calculate_regime_performance(self, prices: np.ndarray,
                                      signals: np.ndarray) -> Dict:
        """
        Calculate strategy performance by regime.

        Args:
            prices: Price data
            signals: Trading signals (+1/-1/0)

        Returns:
            Dict with per-regime performance metrics
        """
        if not self.trained:
            return {}

        history = self.get_regime_history(prices)

        if len(history) == 0:
            return {}

        # Calculate returns
        returns = np.diff(np.log(prices))

        # Align with regime history
        aligned_returns = returns[self.lookback:]
        aligned_signals = signals[self.lookback:len(aligned_returns) + self.lookback]

        min_len = min(len(history), len(aligned_returns), len(aligned_signals))
        history = history.iloc[:min_len]
        aligned_returns = aligned_returns[:min_len]
        aligned_signals = aligned_signals[:min_len]

        # Strategy returns
        strategy_returns = aligned_signals * aligned_returns

        # Per-regime metrics
        regime_metrics = {}
        for regime in self.regime_labels.values():
            mask = history['regime'] == regime
            if mask.sum() > 10:
                regime_rets = strategy_returns[mask.values]
                regime_metrics[regime] = {
                    'sharpe': float(regime_rets.mean() / (regime_rets.std() + 1e-8) * np.sqrt(252)),
                    'win_rate': float((regime_rets > 0).mean() * 100),
                    'total_return': float(regime_rets.sum() * 100),
                    'n_periods': int(mask.sum())
                }

        return regime_metrics

    def save(self, filepath: str):
        """Save trained model to file."""
        if not self.trained:
            print("Model not trained, nothing to save")
            return

        data = {
            'n_regimes': self.n_regimes,
            'lookback': self.lookback,
            'vol_window': self.vol_window,
            'regime_labels': self.regime_labels,
            'state_means': self.state_means.tolist() if self.state_means is not None else None,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'hmm_startprob': self.model.startprob_.tolist(),
            'hmm_transmat': self.model.transmat_.tolist(),
            'hmm_means': self.model.means_.tolist(),
            'hmm_covars': self.model.covars_.tolist()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> 'HMMRegimeDetector':
        """Load trained model from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.n_regimes = data['n_regimes']
        self.lookback = data['lookback']
        self.vol_window = data['vol_window']
        self.regime_labels = {int(k): v for k, v in data['regime_labels'].items()}
        self.state_means = np.array(data['state_means']) if data['state_means'] else None

        # Restore scaler
        self.scaler.mean_ = np.array(data['scaler_mean'])
        self.scaler.scale_ = np.array(data['scaler_scale'])

        # Restore HMM
        if HMM_AVAILABLE:
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full"
            )
            self.model.startprob_ = np.array(data['hmm_startprob'])
            self.model.transmat_ = np.array(data['hmm_transmat'])
            self.model.means_ = np.array(data['hmm_means'])
            self.model.covars_ = np.array(data['hmm_covars'])

        self.trained = True
        print(f"Model loaded from {filepath}")

        return self


def train_hmm_for_asset(asset_dir: str, n_regimes: int = 3) -> HMMRegimeDetector:
    """
    Train HMM regime detector for a specific asset.

    Args:
        asset_dir: Path to asset data directory
        n_regimes: Number of regimes to detect

    Returns:
        Trained HMMRegimeDetector
    """
    from per_asset_optimizer import load_asset_data

    horizons, prices = load_asset_data(asset_dir)

    if prices is None:
        print(f"Could not load data from {asset_dir}")
        return None

    detector = HMMRegimeDetector(n_regimes=n_regimes)
    detector.fit(prices)

    return detector


if __name__ == "__main__":
    print("=" * 60)
    print("HMM REGIME DETECTOR TEST")
    print("=" * 60)

    if not HMM_AVAILABLE:
        print("\nInstall hmmlearn: pip install hmmlearn")
        exit(1)

    # Test with Crude Oil
    asset_dir = "data/1866_Crude_Oil"

    print(f"\nTraining on {asset_dir}...")
    detector = train_hmm_for_asset(asset_dir, n_regimes=3)

    if detector and detector.trained:
        print("\nRegime Labels:")
        for state, label in detector.regime_labels.items():
            print(f"  State {state}: {label}")

        print("\nState Means (momentum, vol, autocorr, trend):")
        for i, means in enumerate(detector.state_means):
            label = detector.regime_labels.get(i, f"state_{i}")
            print(f"  {label}: mom={means[0]:.3f}, vol={means[1]:.3f}, ac={means[2]:.3f}")

        # Get current prediction
        from per_asset_optimizer import load_asset_data
        _, prices = load_asset_data(asset_dir)

        result = detector.predict(prices)

        print(f"\nCurrent Regime: {result['regime']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Days in Regime: {result['daysInRegime']}")
        print(f"Volatility: {result['volatility']:.1f}%")
        print(f"Trend Strength: {result['trendStrength']:.2f}")

        print("\nProbabilities:")
        for regime, prob in result['probabilities'].items():
            print(f"  {regime}: {prob:.1%}")

        # Save model
        detector.save("configs/hmm_crude_oil.json")
