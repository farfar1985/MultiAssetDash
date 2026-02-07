"""
Tier 2 Ensemble Methods for Pairwise Horizon Predictions
=========================================================

This module implements the advanced (Tier 2) ensemble methods from
ENSEMBLE_METHODS_PLAN.md. These methods build on Tier 1 by incorporating:

1. BayesianModelAveraging - Weight models by posterior probability
2. RegimeAdaptiveEnsemble - Adjust weights based on current HMM regime
3. ConformalPredictionInterval - Provide calibrated uncertainty bounds

All methods integrate with the existing Nexus infrastructure including
the HMM regime detector and standardized metrics.

Created: 2026-02-06
Author: AmiraB
Reference: docs/ENSEMBLE_METHODS_PLAN.md Section 2.5-2.7
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

# Use standardized metrics
from utils.metrics import calculate_sharpe_ratio_daily, calculate_sharpe_ratio

# Import HMM regime detector for regime-adaptive ensemble
try:
    from hmm_regime_detector import HMMRegimeDetector
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("HMMRegimeDetector not available. RegimeAdaptiveEnsemble will use fallback.")

# Import sklearn for meta-learning
try:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Some features will be limited.")


# =============================================================================
# RESULT CLASSES
# =============================================================================

@dataclass
class Tier2EnsembleResult:
    """Result from a Tier 2 ensemble prediction."""
    signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    net_probability: float  # -1.0 to 1.0
    weights: Dict[Tuple[int, int], float]  # Pair weights used
    uncertainty: Optional[float] = None  # From BMA or Conformal
    interval_lower: Optional[float] = None  # Prediction interval lower bound
    interval_upper: Optional[float] = None  # Prediction interval upper bound
    regime: Optional[str] = None  # Current detected regime
    metadata: Dict = field(default_factory=dict)


@dataclass
class ConformalInterval:
    """Conformal prediction interval with coverage guarantee."""
    lower: float
    point: float
    upper: float
    coverage: float  # Nominal coverage level
    actual_coverage: Optional[float] = None  # Empirical coverage from calibration


# =============================================================================
# BASE CLASS
# =============================================================================

class BaseTier2Ensemble(ABC):
    """Abstract base class for Tier 2 ensemble methods."""

    def __init__(self, lookback_window: int = 60, threshold: float = 0.3):
        """
        Initialize Tier 2 ensemble method.

        Parameters
        ----------
        lookback_window : int
            Number of historical observations for weight calculation.
        threshold : float
            Signal threshold: |net_prob| > threshold => BULLISH/BEARISH
        """
        self.lookback_window = lookback_window
        self.threshold = threshold
        self._weights: Optional[Dict[Tuple[int, int], float]] = None
        self._is_fitted = False

    @abstractmethod
    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'BaseTier2Ensemble':
        """Fit the ensemble based on historical data."""
        pass

    @abstractmethod
    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: List[int],
                       **kwargs) -> Tier2EnsembleResult:
        """Generate ensemble prediction for a single timestamp."""
        pass

    def predict(self,
                forecasts: pd.DataFrame,
                horizons: List[int],
                **kwargs) -> pd.Series:
        """
        Generate ensemble predictions for all timestamps.

        Returns pd.Series of Tier2EnsembleResult objects.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict()")

        results = []
        for date in forecasts.index:
            result = self.predict_single(forecasts.loc[date], horizons, **kwargs)
            results.append(result)

        return pd.Series(results, index=forecasts.index)

    def _signal_from_probability(self, net_prob: float) -> str:
        """Convert net probability to signal."""
        if net_prob > self.threshold:
            return 'BULLISH'
        elif net_prob < -self.threshold:
            return 'BEARISH'
        else:
            return 'NEUTRAL'


# =============================================================================
# 1. BAYESIAN MODEL AVERAGING (BMA)
# =============================================================================

class BayesianModelAveraging(BaseTier2Ensemble):
    """
    Bayesian Model Averaging Ensemble

    Weights models/pairs by their posterior probability given observed data.
    Uses rolling likelihood estimation to update beliefs about model quality.

    From ENSEMBLE_METHODS_PLAN.md Section 2.5:
    - Posterior proportional to Likelihood x Prior
    - Provides uncertainty quantification via weighted variance
    - Expected improvement: +10-25%

    Parameters
    ----------
    lookback_window : int
        Window for likelihood estimation.
    threshold : float
        Signal threshold.
    prior : str
        Prior distribution type: 'uniform' or 'accuracy'
    likelihood_decay : float
        Exponential decay for recent observations (0-1, higher = more decay)
    min_likelihood : float
        Minimum likelihood to prevent zero weights.
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 prior: str = 'uniform',
                 likelihood_decay: float = 0.95,
                 min_likelihood: float = 0.01):
        super().__init__(lookback_window, threshold)
        self.prior = prior
        self.likelihood_decay = likelihood_decay
        self.min_likelihood = min_likelihood

        self._likelihoods: Dict[Tuple[int, int], float] = {}
        self._posteriors: Dict[Tuple[int, int], float] = {}
        self._prior_weights: Dict[Tuple[int, int], float] = {}

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'BayesianModelAveraging':
        """
        Fit BMA by computing likelihoods and posteriors.

        The likelihood for each pair is based on how well it predicted
        the actual direction, with exponential decay weighting recent
        observations more heavily.
        """
        # Align data
        common_idx = forecasts.index.intersection(actuals.index)
        forecasts = forecasts.loc[common_idx]
        actuals = actuals.loc[common_idx]

        # Use lookback window
        if len(common_idx) > self.lookback_window:
            eval_idx = common_idx[-self.lookback_window:]
            forecasts = forecasts.loc[eval_idx]
            actuals = actuals.loc[eval_idx]

        # Calculate actual returns
        actual_returns = actuals.pct_change().dropna()

        # Build pair list
        pairs = []
        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                col1 = f'd{h1}' if f'd{h1}' in forecasts.columns else str(h1)
                col2 = f'd{h2}' if f'd{h2}' in forecasts.columns else str(h2)
                if col1 in forecasts.columns and col2 in forecasts.columns:
                    pairs.append((h1, h2, col1, col2))

        # Set prior weights
        n_pairs = len(pairs)
        if self.prior == 'uniform':
            self._prior_weights = {(h1, h2): 1.0 / n_pairs for h1, h2, _, _ in pairs}
        else:
            # Accuracy-based prior would require accuracy cache
            self._prior_weights = {(h1, h2): 1.0 / n_pairs for h1, h2, _, _ in pairs}

        # Calculate likelihoods using exponential weighting
        n_obs = len(forecasts)
        decay_weights = np.array([self.likelihood_decay ** (n_obs - 1 - i) for i in range(n_obs)])
        decay_weights = decay_weights / decay_weights.sum()  # Normalize

        for h1, h2, col1, col2 in pairs:
            # Pair drift direction
            pair_drift = forecasts[col2] - forecasts[col1]
            pair_direction = np.sign(pair_drift)

            # Actual direction
            actual_direction = np.sign(actual_returns)

            # Align
            common = pair_direction.index.intersection(actual_direction.index)
            if len(common) < 5:
                self._likelihoods[(h1, h2)] = self.min_likelihood
                continue

            pred_dir = pair_direction.loc[common].values
            act_dir = actual_direction.loc[common].values
            weights_aligned = decay_weights[-len(common):]

            # Likelihood: weighted accuracy
            correct = (pred_dir == act_dir).astype(float)
            likelihood = np.sum(correct * weights_aligned)

            # Ensure minimum likelihood
            self._likelihoods[(h1, h2)] = max(likelihood, self.min_likelihood)

        # Calculate posteriors: P(model|data) ∝ P(data|model) * P(model)
        raw_posteriors = {}
        for pair, likelihood in self._likelihoods.items():
            prior = self._prior_weights.get(pair, 1.0 / n_pairs)
            raw_posteriors[pair] = likelihood * prior

        # Normalize posteriors
        total = sum(raw_posteriors.values())
        if total > 0:
            self._posteriors = {k: v / total for k, v in raw_posteriors.items()}
        else:
            self._posteriors = {k: 1.0 / len(raw_posteriors) for k in raw_posteriors}

        self._weights = self._posteriors.copy()
        self._horizons = horizons
        self._is_fitted = True

        return self

    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: List[int],
                       **kwargs) -> Tier2EnsembleResult:
        """
        Generate BMA prediction with uncertainty quantification.

        Returns weighted average prediction plus weighted variance
        as a measure of model disagreement/uncertainty.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict_single()")

        predictions = {}  # (h1, h2) -> predicted probability
        weighted_sum = 0.0
        total_weight = 0.0

        for (h1, h2), weight in self._posteriors.items():
            col1 = f'd{h1}' if f'd{h1}' in forecast_row.index else str(h1)
            col2 = f'd{h2}' if f'd{h2}' in forecast_row.index else str(h2)

            if col1 not in forecast_row.index or col2 not in forecast_row.index:
                continue

            f1 = forecast_row[col1]
            f2 = forecast_row[col2]

            if pd.isna(f1) or pd.isna(f2):
                continue

            drift = f2 - f1
            direction = 1.0 if drift > 0 else -1.0

            predictions[(h1, h2)] = direction
            weighted_sum += direction * weight
            total_weight += weight

        # Calculate weighted mean (net probability)
        if total_weight > 0:
            net_prob = weighted_sum / total_weight
        else:
            net_prob = 0.0

        # Calculate uncertainty: weighted variance of predictions
        if total_weight > 0 and len(predictions) > 1:
            variance = sum(
                self._posteriors.get(pair, 0) * (pred - net_prob) ** 2
                for pair, pred in predictions.items()
            )
            uncertainty = np.sqrt(variance)
        else:
            uncertainty = 0.5  # Maximum uncertainty when no data

        signal = self._signal_from_probability(net_prob)
        confidence = min(1.0, abs(net_prob) / self.threshold) if self.threshold > 0 else abs(net_prob)

        return Tier2EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=net_prob,
            weights=self._posteriors.copy(),
            uncertainty=uncertainty,
            metadata={
                'method': 'BayesianModelAveraging',
                'n_pairs': len(predictions),
                'likelihoods': self._likelihoods.copy(),
                'posteriors': self._posteriors.copy()
            }
        )

    def get_posteriors(self) -> Dict[Tuple[int, int], float]:
        """Return posterior probabilities for each pair."""
        return self._posteriors.copy()

    def get_likelihoods(self) -> Dict[Tuple[int, int], float]:
        """Return likelihood estimates for each pair."""
        return self._likelihoods.copy()


# =============================================================================
# 2. REGIME-ADAPTIVE ENSEMBLE
# =============================================================================

class RegimeAdaptiveEnsemble(BaseTier2Ensemble):
    """
    Regime-Adaptive Ensemble

    Learns different optimal weights for different market regimes.
    Uses HMM to detect regime, then applies regime-specific weights.

    From ENSEMBLE_METHODS_PLAN.md Section 2.6:
    - Fit HMM on returns to detect regimes
    - Train stacking model per regime
    - Apply regime-specific weights at prediction time
    - Expected improvement: +10-25%

    Parameters
    ----------
    lookback_window : int
        Window for weight estimation per regime.
    threshold : float
        Signal threshold.
    n_regimes : int
        Number of HMM regimes (passed to HMMRegimeDetector)
    min_regime_samples : int
        Minimum samples required to train per-regime weights.
    hmm_lookback : int
        Lookback window for HMM feature extraction.
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 n_regimes: int = 3,
                 min_regime_samples: int = 30,
                 hmm_lookback: int = 20):
        super().__init__(lookback_window, threshold)
        self.n_regimes = n_regimes
        self.min_regime_samples = min_regime_samples
        self.hmm_lookback = hmm_lookback

        # Initialize HMM detector
        if HMM_AVAILABLE:
            self._hmm = HMMRegimeDetector(
                n_regimes=n_regimes,
                lookback=hmm_lookback
            )
        else:
            self._hmm = None

        # Per-regime weights: regime_name -> {(h1, h2): weight}
        self._regime_weights: Dict[str, Dict[Tuple[int, int], float]] = {}

        # Default fallback weights
        self._default_weights: Dict[Tuple[int, int], float] = {}

        # Current regime info
        self._current_regime: Optional[str] = None
        self._regime_confidence: float = 0.0

        # Store prices for regime detection
        self._prices: Optional[np.ndarray] = None

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'RegimeAdaptiveEnsemble':
        """
        Fit regime-specific weights using HMM and per-regime optimization.
        """
        # Align data
        common_idx = forecasts.index.intersection(actuals.index)
        forecasts = forecasts.loc[common_idx]
        actuals = actuals.loc[common_idx]

        # Store prices for regime detection
        self._prices = actuals.values.astype(float)
        self._horizons = horizons

        # Build pair list
        pairs = []
        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                col1 = f'd{h1}' if f'd{h1}' in forecasts.columns else str(h1)
                col2 = f'd{h2}' if f'd{h2}' in forecasts.columns else str(h2)
                if col1 in forecasts.columns and col2 in forecasts.columns:
                    pairs.append((h1, h2, col1, col2))

        # Default equal weights
        n_pairs = len(pairs)
        self._default_weights = {(h1, h2): 1.0 / n_pairs for h1, h2, _, _ in pairs}

        if not HMM_AVAILABLE or len(self._prices) < self.hmm_lookback + 50:
            # Fallback: use equal weights for all regimes
            self._regime_weights = {
                'bull': self._default_weights.copy(),
                'bear': self._default_weights.copy(),
                'sideways': self._default_weights.copy()
            }
            self._weights = self._default_weights.copy()
            self._is_fitted = True
            return self

        # Fit HMM
        self._hmm.fit(self._prices)

        # Get regime history
        regime_history = self._hmm.get_regime_history(self._prices)

        if len(regime_history) == 0:
            self._regime_weights = {'default': self._default_weights.copy()}
            self._weights = self._default_weights.copy()
            self._is_fitted = True
            return self

        # Calculate actual returns
        actual_returns = actuals.pct_change()
        actual_direction = np.sign(actual_returns)

        # Build pair direction matrix
        pair_directions = {}
        for h1, h2, col1, col2 in pairs:
            drift = forecasts[col2] - forecasts[col1]
            pair_directions[(h1, h2)] = np.sign(drift)

        # Learn weights per regime
        for regime_name in regime_history['regime'].unique():
            # Get indices for this regime
            regime_mask = regime_history['regime'] == regime_name
            regime_indices = regime_history.loc[regime_mask, 'index'].values

            if len(regime_indices) < self.min_regime_samples:
                self._regime_weights[regime_name] = self._default_weights.copy()
                continue

            # Calculate accuracy for each pair in this regime
            pair_accuracies = {}
            for (h1, h2), directions in pair_directions.items():
                # Align with regime indices
                valid_idx = [i for i in regime_indices if i < len(directions)]
                if len(valid_idx) < 10:
                    pair_accuracies[(h1, h2)] = 0.5
                    continue

                pred_dir = directions.iloc[valid_idx].values
                act_dir = actual_direction.iloc[valid_idx].values

                # Remove NaN
                mask = ~(np.isnan(pred_dir) | np.isnan(act_dir))
                if mask.sum() < 5:
                    pair_accuracies[(h1, h2)] = 0.5
                    continue

                accuracy = (pred_dir[mask] == act_dir[mask]).mean()
                pair_accuracies[(h1, h2)] = accuracy

            # Convert accuracies to weights (squared for amplification)
            raw_weights = {k: max(0.01, v) ** 2 for k, v in pair_accuracies.items()}
            total = sum(raw_weights.values())
            if total > 0:
                self._regime_weights[regime_name] = {k: v / total for k, v in raw_weights.items()}
            else:
                self._regime_weights[regime_name] = self._default_weights.copy()

        # Set current weights to default (will be updated at prediction time)
        self._weights = self._default_weights.copy()
        self._is_fitted = True

        return self

    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: List[int],
                       prices: Optional[np.ndarray] = None,
                       **kwargs) -> Tier2EnsembleResult:
        """
        Generate regime-adaptive prediction.

        Detects current regime and applies regime-specific weights.

        Parameters
        ----------
        forecast_row : pd.Series
            Single row of forecasts.
        horizons : List[int]
            Horizon list.
        prices : np.ndarray, optional
            Recent prices for regime detection. If not provided,
            uses stored prices from fit().
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict_single()")

        # Detect current regime
        prices_for_detection = prices if prices is not None else self._prices

        if prices_for_detection is not None and HMM_AVAILABLE and self._hmm is not None and self._hmm.trained:
            regime_info = self._hmm.predict(prices_for_detection)
            current_regime = regime_info.get('regime', 'sideways')
            regime_confidence = regime_info.get('confidence', 0.5)
        else:
            current_regime = 'sideways'
            regime_confidence = 0.5

        self._current_regime = current_regime
        self._regime_confidence = regime_confidence

        # Get regime-specific weights
        active_weights = self._regime_weights.get(
            current_regime,
            self._default_weights
        )

        # Calculate weighted signal
        weighted_bull = 0.0
        weighted_bear = 0.0

        for (h1, h2), weight in active_weights.items():
            col1 = f'd{h1}' if f'd{h1}' in forecast_row.index else str(h1)
            col2 = f'd{h2}' if f'd{h2}' in forecast_row.index else str(h2)

            if col1 not in forecast_row.index or col2 not in forecast_row.index:
                continue

            f1 = forecast_row[col1]
            f2 = forecast_row[col2]

            if pd.isna(f1) or pd.isna(f2):
                continue

            drift = f2 - f1

            if drift > 0:
                weighted_bull += weight
            else:
                weighted_bear += weight

        total_weight = weighted_bull + weighted_bear
        if total_weight > 0:
            net_prob = (weighted_bull - weighted_bear) / total_weight
        else:
            net_prob = 0.0

        signal = self._signal_from_probability(net_prob)
        confidence = min(1.0, abs(net_prob) / self.threshold) if self.threshold > 0 else abs(net_prob)

        return Tier2EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=net_prob,
            weights=active_weights.copy(),
            regime=current_regime,
            metadata={
                'method': 'RegimeAdaptiveEnsemble',
                'regime_confidence': regime_confidence,
                'n_regimes': self.n_regimes,
                'available_regimes': list(self._regime_weights.keys())
            }
        )

    def get_regime_weights(self) -> Dict[str, Dict[Tuple[int, int], float]]:
        """Return all per-regime weight dictionaries."""
        return {k: v.copy() for k, v in self._regime_weights.items()}

    def get_current_regime(self) -> Tuple[str, float]:
        """Return current regime and confidence."""
        return (self._current_regime or 'unknown', self._regime_confidence)

    def update_prices(self, prices: np.ndarray):
        """Update stored prices for regime detection."""
        self._prices = prices


# =============================================================================
# 3. CONFORMAL PREDICTION INTERVALS
# =============================================================================

class ConformalPredictionInterval(BaseTier2Ensemble):
    """
    Conformal Prediction Intervals

    Provides distribution-free prediction intervals with guaranteed coverage.
    Uses nonconformity scores from calibration set to construct intervals.

    From ENSEMBLE_METHODS_PLAN.md Section 2.7:
    - Calibrate on held-out data to get nonconformity scores
    - Use quantile of scores for interval width
    - Provides statistically valid uncertainty bounds
    - Coverage guarantee holds regardless of data distribution

    Parameters
    ----------
    lookback_window : int
        Window for base prediction.
    threshold : float
        Signal threshold.
    coverage : float
        Desired coverage level (e.g., 0.90 for 90% intervals)
    calibration_pct : float
        Percentage of data to use for calibration (rest for training)
    base_ensemble : str
        Base ensemble method: 'accuracy', 'magnitude', or 'equal'
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 coverage: float = 0.90,
                 calibration_pct: float = 0.3,
                 base_ensemble: str = 'accuracy'):
        super().__init__(lookback_window, threshold)
        self.coverage = coverage
        self.calibration_pct = calibration_pct
        self.base_ensemble = base_ensemble

        # Calibration data
        self._calibration_scores: List[float] = []
        self._interval_margin: float = 0.5  # Default margin before calibration
        self._actual_coverage: Optional[float] = None

        # Base weights (from simple accuracy weighting)
        self._base_weights: Dict[Tuple[int, int], float] = {}

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'ConformalPredictionInterval':
        """
        Fit conformal predictor with calibration.

        1. Split data into training and calibration sets
        2. Compute base weights on training set
        3. Calculate nonconformity scores on calibration set
        4. Determine interval margin from score quantile
        """
        # Align data
        common_idx = forecasts.index.intersection(actuals.index)
        forecasts = forecasts.loc[common_idx]
        actuals = actuals.loc[common_idx]

        n = len(forecasts)
        if n < 50:
            # Not enough data, use defaults
            self._setup_default_weights(horizons, forecasts)
            self._is_fitted = True
            return self

        # Split: training first, calibration last
        cal_size = int(n * self.calibration_pct)
        train_end = n - cal_size

        train_forecasts = forecasts.iloc[:train_end]
        train_actuals = actuals.iloc[:train_end]
        cal_forecasts = forecasts.iloc[train_end:]
        cal_actuals = actuals.iloc[train_end:]

        # Compute base weights on training data
        self._compute_base_weights(train_forecasts, train_actuals, horizons)

        # Calculate predictions on calibration set
        cal_returns = cal_actuals.pct_change().dropna()

        self._calibration_scores = []

        for i, date in enumerate(cal_forecasts.index):
            if date not in cal_returns.index:
                continue

            forecast_row = cal_forecasts.loc[date]
            actual_return = cal_returns.loc[date]

            # Predict using base weights
            net_prob = self._predict_net_prob(forecast_row, horizons)

            # Nonconformity score: |prediction - actual|
            # Map actual return direction to [-1, 1]
            actual_direction = 1.0 if actual_return > 0 else -1.0

            score = abs(net_prob - actual_direction)
            self._calibration_scores.append(score)

        # Calculate interval margin from quantile
        if len(self._calibration_scores) > 5:
            self._calibration_scores.sort()
            n_cal = len(self._calibration_scores)

            # Quantile index for coverage level
            q_level = np.ceil((n_cal + 1) * self.coverage) / n_cal
            q_idx = min(int(q_level * n_cal), n_cal - 1)

            self._interval_margin = self._calibration_scores[q_idx]

            # Calculate actual coverage
            in_interval = sum(1 for s in self._calibration_scores if s <= self._interval_margin)
            self._actual_coverage = in_interval / n_cal
        else:
            self._interval_margin = 1.0  # Maximum uncertainty
            self._actual_coverage = None

        self._weights = self._base_weights.copy()
        self._horizons = horizons
        self._is_fitted = True

        return self

    def _setup_default_weights(self, horizons: List[int], forecasts: pd.DataFrame):
        """Set up default equal weights."""
        pairs = []
        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                col1 = f'd{h1}' if f'd{h1}' in forecasts.columns else str(h1)
                col2 = f'd{h2}' if f'd{h2}' in forecasts.columns else str(h2)
                if col1 in forecasts.columns and col2 in forecasts.columns:
                    pairs.append((h1, h2))

        n_pairs = len(pairs)
        self._base_weights = {p: 1.0 / n_pairs for p in pairs}
        self._weights = self._base_weights.copy()
        self._horizons = horizons

    def _compute_base_weights(self,
                               forecasts: pd.DataFrame,
                               actuals: pd.Series,
                               horizons: List[int]):
        """Compute base weights using accuracy-weighted approach."""
        actual_returns = actuals.pct_change()
        actual_direction = np.sign(actual_returns)

        weights = {}

        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                col1 = f'd{h1}' if f'd{h1}' in forecasts.columns else str(h1)
                col2 = f'd{h2}' if f'd{h2}' in forecasts.columns else str(h2)

                if col1 not in forecasts.columns or col2 not in forecasts.columns:
                    continue

                pair_drift = forecasts[col2] - forecasts[col1]
                pair_direction = np.sign(pair_drift)

                # Calculate accuracy
                common = pair_direction.index.intersection(actual_direction.index)
                if len(common) > 5:
                    accuracy = (pair_direction.loc[common] == actual_direction.loc[common]).mean()
                else:
                    accuracy = 0.5

                weights[(h1, h2)] = accuracy ** 2

        # Normalize
        total = sum(weights.values())
        if total > 0:
            self._base_weights = {k: v / total for k, v in weights.items()}
        else:
            n = len(weights)
            self._base_weights = {k: 1.0 / n for k in weights}

    def _predict_net_prob(self,
                          forecast_row: pd.Series,
                          horizons: List[int]) -> float:
        """Calculate base net probability from weights."""
        weighted_bull = 0.0
        weighted_bear = 0.0

        for (h1, h2), weight in self._base_weights.items():
            col1 = f'd{h1}' if f'd{h1}' in forecast_row.index else str(h1)
            col2 = f'd{h2}' if f'd{h2}' in forecast_row.index else str(h2)

            if col1 not in forecast_row.index or col2 not in forecast_row.index:
                continue

            f1 = forecast_row[col1]
            f2 = forecast_row[col2]

            if pd.isna(f1) or pd.isna(f2):
                continue

            drift = f2 - f1

            if drift > 0:
                weighted_bull += weight
            else:
                weighted_bear += weight

        total = weighted_bull + weighted_bear
        if total > 0:
            return (weighted_bull - weighted_bear) / total
        return 0.0

    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: List[int],
                       **kwargs) -> Tier2EnsembleResult:
        """
        Generate prediction with conformal interval.

        Returns point prediction plus calibrated prediction interval
        with guaranteed coverage.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict_single()")

        net_prob = self._predict_net_prob(forecast_row, horizons)

        # Construct prediction interval
        lower = net_prob - self._interval_margin
        upper = net_prob + self._interval_margin

        # Clip to valid range
        lower = max(-1.0, lower)
        upper = min(1.0, upper)

        signal = self._signal_from_probability(net_prob)
        confidence = min(1.0, abs(net_prob) / self.threshold) if self.threshold > 0 else abs(net_prob)

        return Tier2EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=net_prob,
            weights=self._base_weights.copy(),
            interval_lower=lower,
            interval_upper=upper,
            uncertainty=self._interval_margin,
            metadata={
                'method': 'ConformalPredictionInterval',
                'coverage': self.coverage,
                'actual_coverage': self._actual_coverage,
                'interval_margin': self._interval_margin,
                'n_calibration_samples': len(self._calibration_scores)
            }
        )

    def get_interval(self, net_prob: float) -> ConformalInterval:
        """
        Get conformal interval for a point prediction.

        Parameters
        ----------
        net_prob : float
            Point prediction (net probability).

        Returns
        -------
        ConformalInterval with lower, point, upper bounds.
        """
        return ConformalInterval(
            lower=max(-1.0, net_prob - self._interval_margin),
            point=net_prob,
            upper=min(1.0, net_prob + self._interval_margin),
            coverage=self.coverage,
            actual_coverage=self._actual_coverage
        )

    def get_calibration_scores(self) -> List[float]:
        """Return the calibration nonconformity scores."""
        return self._calibration_scores.copy()


# =============================================================================
# COMBINED TIER 2 ENSEMBLE
# =============================================================================

class CombinedTier2Ensemble:
    """
    Combined Tier 2 Ensemble

    Combines all three Tier 2 methods into a meta-ensemble:
    1. Bayesian Model Averaging (BMA)
    2. Regime-Adaptive Ensemble
    3. Conformal Prediction Intervals

    The final signal uses regime-adaptive weights with BMA posteriors,
    plus conformal intervals for uncertainty quantification.
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 coverage: float = 0.90,
                 n_regimes: int = 3):
        """
        Initialize combined Tier 2 ensemble.
        """
        self.lookback_window = lookback_window
        self.threshold = threshold
        self.coverage = coverage
        self.n_regimes = n_regimes

        # Initialize component ensembles
        self.bma = BayesianModelAveraging(lookback_window, threshold)
        self.regime_adaptive = RegimeAdaptiveEnsemble(
            lookback_window, threshold, n_regimes
        )
        self.conformal = ConformalPredictionInterval(
            lookback_window, threshold, coverage
        )

        self._is_fitted = False

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'CombinedTier2Ensemble':
        """Fit all component ensembles."""
        self.bma.fit(forecasts, actuals, horizons)
        self.regime_adaptive.fit(forecasts, actuals, horizons)
        self.conformal.fit(forecasts, actuals, horizons)
        self._horizons = horizons
        self._is_fitted = True
        return self

    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: Optional[List[int]] = None,
                       prices: Optional[np.ndarray] = None) -> Tier2EnsembleResult:
        """
        Generate combined Tier 2 prediction.

        Combines:
        - BMA for posterior-weighted signal
        - Regime-adaptive for regime-specific adjustments
        - Conformal for calibrated uncertainty intervals
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() first")

        horizons = horizons or self._horizons

        # Get predictions from each method
        bma_result = self.bma.predict_single(forecast_row, horizons)
        regime_result = self.regime_adaptive.predict_single(forecast_row, horizons, prices=prices)
        conformal_result = self.conformal.predict_single(forecast_row, horizons)

        # Combine: use regime-adaptive as base, blend with BMA
        # Weight: 60% regime-adaptive, 40% BMA
        combined_prob = (
            0.6 * regime_result.net_probability +
            0.4 * bma_result.net_probability
        )

        # Apply conformal interval to combined prediction
        interval = self.conformal.get_interval(combined_prob)

        signal = 'BULLISH' if combined_prob > self.threshold else \
                 'BEARISH' if combined_prob < -self.threshold else 'NEUTRAL'

        confidence = min(1.0, abs(combined_prob) / self.threshold) if self.threshold > 0 else abs(combined_prob)

        return Tier2EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=combined_prob,
            weights={},  # Combined from multiple sources
            uncertainty=interval.upper - interval.lower,
            interval_lower=interval.lower,
            interval_upper=interval.upper,
            regime=regime_result.regime,
            metadata={
                'method': 'CombinedTier2',
                'component_signals': {
                    'bma': bma_result.signal,
                    'regime_adaptive': regime_result.signal,
                    'conformal': conformal_result.signal
                },
                'component_probs': {
                    'bma': bma_result.net_probability,
                    'regime_adaptive': regime_result.net_probability,
                    'conformal': conformal_result.net_probability
                },
                'bma_uncertainty': bma_result.uncertainty,
                'conformal_coverage': self.coverage
            }
        )

    def predict(self,
                forecasts: pd.DataFrame,
                horizons: Optional[List[int]] = None,
                prices: Optional[np.ndarray] = None) -> pd.Series:
        """Generate predictions for all timestamps."""
        if not self._is_fitted:
            raise ValueError("Must call fit() first")

        horizons = horizons or self._horizons
        results = []

        for date in forecasts.index:
            result = self.predict_single(forecasts.loc[date], horizons, prices)
            results.append(result)

        return pd.Series(results, index=forecasts.index)


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_tier2_ensemble(
    ensemble: BaseTier2Ensemble,
    forecasts: pd.DataFrame,
    actuals: pd.Series,
    horizons: List[int],
    train_pct: float = 0.7
) -> Dict:
    """
    Evaluate a Tier 2 ensemble method with train/test split.
    """
    # Split data
    n = len(forecasts)
    split_idx = int(n * train_pct)

    train_forecasts = forecasts.iloc[:split_idx]
    train_actuals = actuals.iloc[:split_idx]

    test_forecasts = forecasts.iloc[split_idx:]
    test_actuals = actuals.iloc[split_idx:]

    # Fit on training data
    ensemble.fit(train_forecasts, train_actuals, horizons)

    # Predict on test data
    predictions = ensemble.predict(test_forecasts, horizons)

    # Calculate metrics
    test_returns = test_actuals.pct_change().dropna()

    # Extract signals
    signals = []
    for result in predictions:
        if result.signal == 'BULLISH':
            signals.append(1)
        elif result.signal == 'BEARISH':
            signals.append(-1)
        else:
            signals.append(0)

    signal_series = pd.Series(signals, index=predictions.index)

    # Align
    common_idx = signal_series.index.intersection(test_returns.index)
    if len(common_idx) < 2:
        return {'error': 'Insufficient aligned data'}

    aligned_signals = signal_series.loc[common_idx]
    aligned_returns = test_returns.loc[common_idx]

    # Strategy returns
    strategy_returns = (aligned_signals.shift(1) * aligned_returns).dropna()

    if len(strategy_returns) < 2:
        return {'error': 'Insufficient strategy returns'}

    sharpe = calculate_sharpe_ratio_daily(strategy_returns.values)

    # Directional accuracy
    pred_dir = aligned_signals.shift(1).dropna()
    actual_dir = np.sign(aligned_returns)

    common = pred_dir.index.intersection(actual_dir.index)
    if len(common) > 0:
        pred_dir = pred_dir.loc[common]
        actual_dir = actual_dir.loc[common]
        non_neutral = pred_dir != 0
        if non_neutral.sum() > 0:
            accuracy = ((pred_dir[non_neutral] == actual_dir[non_neutral]).mean()) * 100
        else:
            accuracy = 50.0
    else:
        accuracy = 50.0

    total_return = strategy_returns.sum() * 100
    win_rate = (strategy_returns > 0).mean() * 100

    # Additional Tier 2 metrics
    result_dict = {
        'method': ensemble.__class__.__name__,
        'sharpe': round(sharpe, 3),
        'directional_accuracy': round(accuracy, 2),
        'total_return': round(total_return, 2),
        'win_rate': round(win_rate, 2),
        'n_predictions': len(strategy_returns)
    }

    # Add method-specific metrics
    if hasattr(ensemble, 'get_interval'):
        # Conformal: check coverage
        result_dict['coverage'] = getattr(ensemble, 'coverage', None)
        result_dict['actual_coverage'] = getattr(ensemble, '_actual_coverage', None)

    if hasattr(ensemble, 'get_current_regime'):
        regime, conf = ensemble.get_current_regime()
        result_dict['last_regime'] = regime
        result_dict['regime_confidence'] = round(conf, 3)

    return result_dict


def compare_tier2_methods(
    forecasts: pd.DataFrame,
    actuals: pd.Series,
    horizons: List[int],
    train_pct: float = 0.7
) -> pd.DataFrame:
    """
    Compare all Tier 2 methods on the same data.
    """
    results = []

    methods = [
        ('BMA', BayesianModelAveraging()),
        ('Regime-Adaptive', RegimeAdaptiveEnsemble()),
        ('Conformal-90%', ConformalPredictionInterval(coverage=0.90)),
        ('Combined-Tier2', CombinedTier2Ensemble()),
    ]

    for name, ensemble in methods:
        try:
            if isinstance(ensemble, CombinedTier2Ensemble):
                # Manual evaluation for combined
                n = len(forecasts)
                split_idx = int(n * train_pct)

                train_forecasts = forecasts.iloc[:split_idx]
                train_actuals = actuals.iloc[:split_idx]
                test_forecasts = forecasts.iloc[split_idx:]
                test_actuals = actuals.iloc[split_idx:]

                ensemble.fit(train_forecasts, train_actuals, horizons)
                predictions = ensemble.predict(test_forecasts, horizons)

                test_returns = test_actuals.pct_change().dropna()
                signals = []
                for res in predictions:
                    if res.signal == 'BULLISH':
                        signals.append(1)
                    elif res.signal == 'BEARISH':
                        signals.append(-1)
                    else:
                        signals.append(0)

                signal_series = pd.Series(signals, index=predictions.index)
                common_idx = signal_series.index.intersection(test_returns.index)

                if len(common_idx) > 2:
                    strategy_returns = (signal_series.loc[common_idx].shift(1) *
                                       test_returns.loc[common_idx]).dropna()

                    result = {
                        'method': name,
                        'sharpe': round(calculate_sharpe_ratio_daily(strategy_returns.values), 3),
                        'total_return': round(strategy_returns.sum() * 100, 2),
                        'win_rate': round((strategy_returns > 0).mean() * 100, 2),
                        'n_predictions': len(strategy_returns)
                    }
                else:
                    result = {'method': name, 'error': 'Insufficient data'}
            else:
                result = evaluate_tier2_ensemble(ensemble, forecasts, actuals, horizons, train_pct)
                result['method'] = name

            results.append(result)
            print(f"  [OK] {name}: Sharpe={result.get('sharpe', 'N/A')}")
        except Exception as e:
            print(f"  [ERR] {name}: {str(e)}")
            results.append({'method': name, 'error': str(e)})

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TIER 2 ENSEMBLE METHODS - Advanced Implementations")
    print("=" * 70)
    print("\nAvailable methods:")
    print("  1. BayesianModelAveraging - Weight by posterior probability")
    print("  2. RegimeAdaptiveEnsemble - Adjust weights based on HMM regime")
    print("  3. ConformalPredictionInterval - Calibrated uncertainty bounds")
    print("  4. CombinedTier2Ensemble - Meta-ensemble of all three")
    print("\nUsage:")
    print("  from backend.ensemble_tier2 import BayesianModelAveraging")
    print("  ensemble = BayesianModelAveraging()")
    print("  ensemble.fit(forecasts_df, actuals_series, [5, 10, 20])")
    print("  result = ensemble.predict_single(forecast_row, [5, 10, 20])")
    print("\n  # For regime-adaptive, can provide current prices:")
    print("  from backend.ensemble_tier2 import RegimeAdaptiveEnsemble")
    print("  ensemble = RegimeAdaptiveEnsemble(n_regimes=3)")
    print("  result = ensemble.predict_single(forecast_row, horizons, prices=recent_prices)")
    print("\nEvaluation:")
    print("  from backend.ensemble_tier2 import compare_tier2_methods")
    print("  results_df = compare_tier2_methods(forecasts, actuals, horizons)")
    print("\nDependencies:")
    print(f"  HMM available: {HMM_AVAILABLE}")
    print(f"  sklearn available: {SKLEARN_AVAILABLE}")
