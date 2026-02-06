"""
Tier 1 Ensemble Methods for Pairwise Horizon Predictions
=========================================================

This module implements the essential (Tier 1) ensemble methods from
ENSEMBLE_METHODS_PLAN.md. These methods improve upon the simple pairwise
slopes voting mechanism by incorporating:

1. AccuracyWeightedEnsemble - Weight models by historical accuracy
2. MagnitudeWeightedVoting - Weight by signal magnitude (stronger = more confident)
3. ErrorCorrelationWeighting - Downweight models with correlated errors

All methods are designed to work with pairwise horizon predictions
and integrate with the existing Nexus ensemble infrastructure.

Created: 2026-02-06
Author: AmiraB
Reference: docs/ENSEMBLE_METHODS_PLAN.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Use standardized metrics
from utils.metrics import calculate_sharpe_ratio_daily, calculate_sharpe_ratio


# =============================================================================
# BASE CLASS
# =============================================================================

@dataclass
class EnsembleResult:
    """Result from an ensemble prediction."""
    signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    net_probability: float  # -1.0 to 1.0
    weights: Dict[Tuple[int, int], float]  # Pair weights used
    metadata: Dict = field(default_factory=dict)


class BaseTier1Ensemble(ABC):
    """Abstract base class for Tier 1 ensemble methods."""

    def __init__(self, lookback_window: int = 60, threshold: float = 0.3):
        """
        Initialize ensemble method.

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
            horizons: List[int]) -> 'BaseTier1Ensemble':
        """Fit the ensemble weights based on historical data."""
        pass

    @abstractmethod
    def _calculate_pair_weights(self,
                                forecasts: pd.DataFrame,
                                actuals: pd.Series,
                                horizons: List[int]) -> Dict[Tuple[int, int], float]:
        """Calculate weights for each horizon pair."""
        pass

    def predict(self,
                forecasts: pd.DataFrame,
                horizons: List[int]) -> pd.Series:
        """
        Generate ensemble predictions for all timestamps.

        Parameters
        ----------
        forecasts : pd.DataFrame
            DataFrame with columns for each horizon (e.g., 'd5', 'd10', etc.)
        horizons : List[int]
            List of horizon values to use.

        Returns
        -------
        pd.Series
            Series of EnsembleResult objects indexed by timestamp.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict()")

        results = []

        for date in forecasts.index:
            result = self.predict_single(forecasts.loc[date], horizons)
            results.append(result)

        return pd.Series(results, index=forecasts.index)

    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: List[int]) -> EnsembleResult:
        """
        Generate ensemble prediction for a single timestamp.

        Parameters
        ----------
        forecast_row : pd.Series
            Single row of forecasts with horizon columns.
        horizons : List[int]
            List of horizon values to use.

        Returns
        -------
        EnsembleResult
            Prediction result with signal, confidence, and weights.
        """
        if not self._is_fitted or self._weights is None:
            raise ValueError("Must call fit() before predict_single()")

        weighted_bull = 0.0
        weighted_bear = 0.0

        for (h1, h2), weight in self._weights.items():
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

        # Convert to signal
        if net_prob > self.threshold:
            signal = 'BULLISH'
        elif net_prob < -self.threshold:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        # Confidence is how far from neutral
        confidence = min(1.0, abs(net_prob) / self.threshold) if self.threshold > 0 else abs(net_prob)

        return EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=net_prob,
            weights=self._weights.copy(),
            metadata={'method': self.__class__.__name__}
        )

    def get_weights(self) -> Dict[Tuple[int, int], float]:
        """Return the current pair weights."""
        if not self._is_fitted:
            raise ValueError("Must call fit() first")
        return self._weights.copy()


# =============================================================================
# 1. ACCURACY-WEIGHTED ENSEMBLE
# =============================================================================

class AccuracyWeightedEnsemble(BaseTier1Ensemble):
    """
    Accuracy-Weighted Ensemble

    Weights each horizon pair by its historical directional accuracy.
    Better-performing pairs get more influence on the final signal.

    From ENSEMBLE_METHODS_PLAN.md Section 2.1:
    - Track rolling accuracy for each (h1, h2) pair
    - Weight = accuracy^2 (amplifies differences)
    - Expected improvement: +5-15%

    Parameters
    ----------
    lookback_window : int
        Number of historical periods for accuracy calculation.
    threshold : float
        Signal threshold for BULLISH/BEARISH determination.
    accuracy_power : float
        Power to raise accuracy to (2.0 = square, amplifies differences).
    min_accuracy : float
        Minimum accuracy threshold; pairs below this get zero weight.
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 accuracy_power: float = 2.0,
                 min_accuracy: float = 0.45):
        super().__init__(lookback_window, threshold)
        self.accuracy_power = accuracy_power
        self.min_accuracy = min_accuracy
        self._pair_accuracies: Dict[Tuple[int, int], float] = {}

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'AccuracyWeightedEnsemble':
        """
        Fit accuracy weights based on historical performance.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Historical forecasts with horizon columns.
        actuals : pd.Series
            Actual prices aligned with forecasts.
        horizons : List[int]
            List of horizons to consider.

        Returns
        -------
        self
        """
        self._weights = self._calculate_pair_weights(forecasts, actuals, horizons)
        self._is_fitted = True
        return self

    def _calculate_pair_weights(self,
                                forecasts: pd.DataFrame,
                                actuals: pd.Series,
                                horizons: List[int]) -> Dict[Tuple[int, int], float]:
        """Calculate accuracy-based weights for each horizon pair."""
        weights = {}
        self._pair_accuracies = {}

        # Align data
        common_idx = forecasts.index.intersection(actuals.index)
        forecasts = forecasts.loc[common_idx]
        actuals = actuals.loc[common_idx]

        # Calculate actual direction (for accuracy measurement)
        actual_returns = actuals.pct_change()
        actual_direction = np.sign(actual_returns)

        # Use recent data based on lookback window
        if len(common_idx) > self.lookback_window:
            eval_idx = common_idx[-self.lookback_window:]
            forecasts = forecasts.loc[eval_idx]
            actual_direction = actual_direction.loc[eval_idx]

        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                col1 = f'd{h1}' if f'd{h1}' in forecasts.columns else str(h1)
                col2 = f'd{h2}' if f'd{h2}' in forecasts.columns else str(h2)

                if col1 not in forecasts.columns or col2 not in forecasts.columns:
                    continue

                # Calculate pair drift direction
                pair_drift = forecasts[col2] - forecasts[col1]
                pair_direction = np.sign(pair_drift)

                # Calculate accuracy (how often pair predicts correct direction)
                matches = (pair_direction == actual_direction).dropna()

                if len(matches) > 0:
                    accuracy = matches.mean()
                else:
                    accuracy = 0.5  # Default to random

                self._pair_accuracies[(h1, h2)] = accuracy

                # Apply minimum threshold
                if accuracy < self.min_accuracy:
                    weights[(h1, h2)] = 0.0
                else:
                    # Weight = accuracy^power (amplifies differences)
                    weights[(h1, h2)] = accuracy ** self.accuracy_power

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Fallback to equal weights
            n_pairs = len(weights)
            if n_pairs > 0:
                weights = {k: 1.0 / n_pairs for k in weights}

        return weights

    def get_pair_accuracies(self) -> Dict[Tuple[int, int], float]:
        """Return the calculated accuracy for each pair."""
        return self._pair_accuracies.copy()


# =============================================================================
# 2. MAGNITUDE-WEIGHTED VOTING
# =============================================================================

class MagnitudeWeightedVoting(BaseTier1Ensemble):
    """
    Magnitude-Weighted Voting Ensemble

    Weights votes by the magnitude of the predicted price drift.
    Larger predicted moves indicate stronger conviction.

    From ENSEMBLE_METHODS_PLAN.md Section 2.2:
    - Magnitude = |drift| / base_price (as percentage)
    - Separation weight = log1p(h2 - h1) (longer spans = more independent)
    - Final weight = magnitude * separation_weight
    - Expected improvement: +5-10%

    Parameters
    ----------
    lookback_window : int
        Not used directly; kept for API consistency.
    threshold : float
        Signal threshold for BULLISH/BEARISH determination.
    use_separation_weight : bool
        Whether to include horizon separation in weighting.
    magnitude_cap : float
        Maximum magnitude weight to prevent outlier dominance.
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 use_separation_weight: bool = True,
                 magnitude_cap: float = 0.1):
        super().__init__(lookback_window, threshold)
        self.use_separation_weight = use_separation_weight
        self.magnitude_cap = magnitude_cap
        self._base_price: Optional[float] = None

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'MagnitudeWeightedVoting':
        """
        Fit magnitude weights (precomputes base price).

        For magnitude voting, the weights are computed dynamically
        at prediction time based on the actual forecast values.
        This fit() method stores the base price for normalization.
        """
        # Store the most recent actual price as base
        if len(actuals) > 0:
            self._base_price = float(actuals.iloc[-1])
        else:
            self._base_price = 1.0  # Fallback

        # Pre-calculate static separation weights
        self._separation_weights = {}
        max_horizon = max(horizons) if horizons else 1

        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                if self.use_separation_weight:
                    sep_weight = np.log1p(h2 - h1) / np.log1p(max_horizon)
                else:
                    sep_weight = 1.0
                self._separation_weights[(h1, h2)] = sep_weight

        # Store horizons for later use
        self._horizons = horizons

        # For API consistency, create placeholder weights
        self._weights = {k: 1.0 / len(self._separation_weights)
                        for k in self._separation_weights}
        self._is_fitted = True
        return self

    def _calculate_pair_weights(self,
                                forecasts: pd.DataFrame,
                                actuals: pd.Series,
                                horizons: List[int]) -> Dict[Tuple[int, int], float]:
        """Not used directly; weights are computed dynamically."""
        return self._separation_weights.copy()

    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: List[int]) -> EnsembleResult:
        """
        Generate magnitude-weighted prediction for a single timestamp.

        Override base class to compute magnitude weights dynamically.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict_single()")

        weighted_sum = 0.0
        total_weight = 0.0
        dynamic_weights = {}

        for (h1, h2), sep_weight in self._separation_weights.items():
            col1 = f'd{h1}' if f'd{h1}' in forecast_row.index else str(h1)
            col2 = f'd{h2}' if f'd{h2}' in forecast_row.index else str(h2)

            if col1 not in forecast_row.index or col2 not in forecast_row.index:
                continue

            f1 = forecast_row[col1]
            f2 = forecast_row[col2]

            if pd.isna(f1) or pd.isna(f2):
                continue

            drift = f2 - f1

            # Magnitude as percentage of base price
            if self._base_price and self._base_price > 0:
                magnitude = min(abs(drift) / self._base_price, self.magnitude_cap)
            else:
                magnitude = min(abs(drift), self.magnitude_cap)

            # Final weight = magnitude * separation
            weight = magnitude * sep_weight
            dynamic_weights[(h1, h2)] = weight

            # Accumulate weighted vote
            direction = np.sign(drift)
            weighted_sum += direction * weight
            total_weight += weight

        # Calculate net probability
        if total_weight > 0:
            net_prob = weighted_sum / total_weight
        else:
            net_prob = 0.0

        # Convert to signal
        if net_prob > self.threshold:
            signal = 'BULLISH'
        elif net_prob < -self.threshold:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        confidence = min(1.0, abs(net_prob) / self.threshold) if self.threshold > 0 else abs(net_prob)

        return EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=net_prob,
            weights=dynamic_weights,
            metadata={
                'method': 'MagnitudeWeightedVoting',
                'base_price': self._base_price,
                'total_weight': total_weight
            }
        )


# =============================================================================
# 3. ERROR CORRELATION WEIGHTING
# =============================================================================

class ErrorCorrelationWeighting(BaseTier1Ensemble):
    """
    Error Correlation Weighting Ensemble

    Downweights models/pairs whose errors are highly correlated with others.
    The goal is to reduce redundancy and favor diverse signals.

    From ENSEMBLE_METHODS_PLAN.md Section 2.4:
    - Compute error correlation matrix across pairs
    - Average correlation of each pair with all others
    - Weight inversely: less correlated = higher weight
    - Expected improvement: +10-20%

    Parameters
    ----------
    lookback_window : int
        Number of historical periods for correlation calculation.
    threshold : float
        Signal threshold for BULLISH/BEARISH determination.
    epsilon : float
        Small value added to correlation for numerical stability.
    correlation_power : float
        Power for inverse correlation weighting.
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 epsilon: float = 0.1,
                 correlation_power: float = 1.0):
        super().__init__(lookback_window, threshold)
        self.epsilon = epsilon
        self.correlation_power = correlation_power
        self._error_correlations: Optional[pd.DataFrame] = None
        self._avg_correlations: Dict[Tuple[int, int], float] = {}

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'ErrorCorrelationWeighting':
        """
        Fit error correlation weights.

        Computes prediction errors for each horizon pair and calculates
        the correlation matrix to determine which pairs are redundant.
        """
        self._weights = self._calculate_pair_weights(forecasts, actuals, horizons)
        self._is_fitted = True
        return self

    def _calculate_pair_weights(self,
                                forecasts: pd.DataFrame,
                                actuals: pd.Series,
                                horizons: List[int]) -> Dict[Tuple[int, int], float]:
        """Calculate inverse-correlation weights for each horizon pair."""
        # Align data
        common_idx = forecasts.index.intersection(actuals.index)
        forecasts = forecasts.loc[common_idx]
        actuals = actuals.loc[common_idx]

        # Use lookback window
        if len(common_idx) > self.lookback_window:
            eval_idx = common_idx[-self.lookback_window:]
            forecasts = forecasts.loc[eval_idx]
            actuals = actuals.loc[eval_idx]

        # Calculate actual returns for error computation
        actual_returns = actuals.pct_change().dropna()

        # Build error matrix: each column is a pair's prediction error
        error_data = {}
        pair_keys = []

        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                col1 = f'd{h1}' if f'd{h1}' in forecasts.columns else str(h1)
                col2 = f'd{h2}' if f'd{h2}' in forecasts.columns else str(h2)

                if col1 not in forecasts.columns or col2 not in forecasts.columns:
                    continue

                # Pair drift predicts direction
                pair_drift = forecasts[col2] - forecasts[col1]
                predicted_direction = np.sign(pair_drift)

                # Error: 1 if wrong, 0 if correct
                actual_direction = np.sign(actual_returns)

                # Align indices
                common = predicted_direction.index.intersection(actual_direction.index)
                if len(common) < 10:
                    continue

                pred_dir = predicted_direction.loc[common]
                act_dir = actual_direction.loc[common]

                # Error series (binary: wrong prediction = 1)
                errors = (pred_dir != act_dir).astype(float)

                pair_key = (h1, h2)
                error_data[pair_key] = errors.values
                pair_keys.append(pair_key)

        if len(pair_keys) < 2:
            # Not enough pairs for correlation, use equal weights
            if pair_keys:
                return {k: 1.0 / len(pair_keys) for k in pair_keys}
            return {}

        # Build error DataFrame
        min_len = min(len(v) for v in error_data.values())
        error_df = pd.DataFrame({
            str(k): v[:min_len] for k, v in error_data.items()
        })

        # Calculate correlation matrix
        self._error_correlations = error_df.corr()

        # Average correlation for each pair (with all others)
        self._avg_correlations = {}
        weights = {}

        for pair_key in pair_keys:
            key_str = str(pair_key)
            if key_str in self._error_correlations.columns:
                # Average correlation with all OTHER pairs
                corr_with_others = self._error_correlations[key_str].drop(key_str)
                avg_corr = corr_with_others.abs().mean()  # Use absolute correlation
            else:
                avg_corr = 0.5  # Default

            self._avg_correlations[pair_key] = avg_corr

            # Inverse correlation weighting
            # Lower correlation = higher weight (more unique information)
            raw_weight = 1.0 / (avg_corr + self.epsilon)
            weights[pair_key] = raw_weight ** self.correlation_power

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_error_correlations(self) -> Optional[pd.DataFrame]:
        """Return the error correlation matrix."""
        return self._error_correlations.copy() if self._error_correlations is not None else None

    def get_avg_correlations(self) -> Dict[Tuple[int, int], float]:
        """Return average correlation for each pair."""
        return self._avg_correlations.copy()


# =============================================================================
# COMBINED TIER 1 ENSEMBLE
# =============================================================================

class CombinedTier1Ensemble:
    """
    Combined Tier 1 Ensemble

    Combines all three Tier 1 methods into a meta-ensemble:
    1. Accuracy-Weighted
    2. Magnitude-Weighted
    3. Error-Correlation-Weighted

    The final signal is determined by weighted voting across methods.
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 method_weights: Optional[Dict[str, float]] = None):
        """
        Initialize combined ensemble.

        Parameters
        ----------
        lookback_window : int
            Lookback window for all methods.
        threshold : float
            Signal threshold.
        method_weights : Dict[str, float], optional
            Weights for each method. Keys: 'accuracy', 'magnitude', 'correlation'.
            Defaults to equal weights.
        """
        self.lookback_window = lookback_window
        self.threshold = threshold

        # Default to equal weights
        self.method_weights = method_weights or {
            'accuracy': 1.0 / 3,
            'magnitude': 1.0 / 3,
            'correlation': 1.0 / 3
        }

        # Initialize component ensembles
        self.accuracy_ensemble = AccuracyWeightedEnsemble(lookback_window, threshold)
        self.magnitude_ensemble = MagnitudeWeightedVoting(lookback_window, threshold)
        self.correlation_ensemble = ErrorCorrelationWeighting(lookback_window, threshold)

        self._is_fitted = False

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'CombinedTier1Ensemble':
        """Fit all component ensembles."""
        self.accuracy_ensemble.fit(forecasts, actuals, horizons)
        self.magnitude_ensemble.fit(forecasts, actuals, horizons)
        self.correlation_ensemble.fit(forecasts, actuals, horizons)
        self._horizons = horizons
        self._is_fitted = True
        return self

    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: Optional[List[int]] = None) -> EnsembleResult:
        """
        Generate combined prediction from all methods.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() first")

        horizons = horizons or self._horizons

        # Get predictions from each method
        acc_result = self.accuracy_ensemble.predict_single(forecast_row, horizons)
        mag_result = self.magnitude_ensemble.predict_single(forecast_row, horizons)
        corr_result = self.correlation_ensemble.predict_single(forecast_row, horizons)

        # Weighted combination of net probabilities
        combined_prob = (
            self.method_weights['accuracy'] * acc_result.net_probability +
            self.method_weights['magnitude'] * mag_result.net_probability +
            self.method_weights['correlation'] * corr_result.net_probability
        )

        # Convert to signal
        if combined_prob > self.threshold:
            signal = 'BULLISH'
        elif combined_prob < -self.threshold:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        confidence = min(1.0, abs(combined_prob) / self.threshold) if self.threshold > 0 else abs(combined_prob)

        return EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=combined_prob,
            weights={},  # Combined weights are method-level
            metadata={
                'method': 'CombinedTier1',
                'component_signals': {
                    'accuracy': acc_result.signal,
                    'magnitude': mag_result.signal,
                    'correlation': corr_result.signal
                },
                'component_probs': {
                    'accuracy': acc_result.net_probability,
                    'magnitude': mag_result.net_probability,
                    'correlation': corr_result.net_probability
                }
            }
        )

    def predict(self,
                forecasts: pd.DataFrame,
                horizons: Optional[List[int]] = None) -> pd.Series:
        """Generate predictions for all timestamps."""
        if not self._is_fitted:
            raise ValueError("Must call fit() first")

        horizons = horizons or self._horizons
        results = []

        for date in forecasts.index:
            result = self.predict_single(forecasts.loc[date], horizons)
            results.append(result)

        return pd.Series(results, index=forecasts.index)


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_ensemble(
    ensemble: BaseTier1Ensemble,
    forecasts: pd.DataFrame,
    actuals: pd.Series,
    horizons: List[int],
    train_pct: float = 0.7
) -> Dict:
    """
    Evaluate an ensemble method with train/test split.

    Parameters
    ----------
    ensemble : BaseTier1Ensemble
        The ensemble method to evaluate.
    forecasts : pd.DataFrame
        Full forecast data.
    actuals : pd.Series
        Full actual prices.
    horizons : List[int]
        Horizons to use.
    train_pct : float
        Percentage of data for training.

    Returns
    -------
    Dict with evaluation metrics.
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

    # Extract signals and align with returns
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

    # Strategy returns (signal applied to next day's return)
    strategy_returns = (aligned_signals.shift(1) * aligned_returns).dropna()

    # Metrics
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
        # Only count non-neutral predictions
        non_neutral = pred_dir != 0
        if non_neutral.sum() > 0:
            accuracy = ((pred_dir[non_neutral] == actual_dir[non_neutral]).mean()) * 100
        else:
            accuracy = 50.0
    else:
        accuracy = 50.0

    total_return = strategy_returns.sum() * 100
    win_rate = (strategy_returns > 0).mean() * 100

    return {
        'method': ensemble.__class__.__name__,
        'sharpe': round(sharpe, 3),
        'directional_accuracy': round(accuracy, 2),
        'total_return': round(total_return, 2),
        'win_rate': round(win_rate, 2),
        'n_predictions': len(strategy_returns)
    }


def compare_tier1_methods(
    forecasts: pd.DataFrame,
    actuals: pd.Series,
    horizons: List[int],
    train_pct: float = 0.7
) -> pd.DataFrame:
    """
    Compare all Tier 1 methods on the same data.

    Returns DataFrame with metrics for each method.
    """
    results = []

    methods = [
        ('Accuracy-Weighted', AccuracyWeightedEnsemble()),
        ('Magnitude-Weighted', MagnitudeWeightedVoting()),
        ('Error-Correlation', ErrorCorrelationWeighting()),
        ('Combined-Tier1', CombinedTier1Ensemble()),
    ]

    for name, ensemble in methods:
        try:
            if isinstance(ensemble, CombinedTier1Ensemble):
                # Split and evaluate manually for combined
                n = len(forecasts)
                split_idx = int(n * train_pct)

                train_forecasts = forecasts.iloc[:split_idx]
                train_actuals = actuals.iloc[:split_idx]
                test_forecasts = forecasts.iloc[split_idx:]
                test_actuals = actuals.iloc[split_idx:]

                ensemble.fit(train_forecasts, train_actuals, horizons)
                predictions = ensemble.predict(test_forecasts, horizons)

                # Calculate metrics
                test_returns = test_actuals.pct_change().dropna()
                signals = []
                for result in predictions:
                    if result.signal == 'BULLISH':
                        signals.append(1)
                    elif result.signal == 'BEARISH':
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
                result = evaluate_ensemble(ensemble, forecasts, actuals, horizons, train_pct)
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
    print("TIER 1 ENSEMBLE METHODS - Backend Implementation")
    print("=" * 70)
    print("\nAvailable methods:")
    print("  1. AccuracyWeightedEnsemble - Weight by historical accuracy")
    print("  2. MagnitudeWeightedVoting - Weight by signal magnitude")
    print("  3. ErrorCorrelationWeighting - Downweight correlated errors")
    print("  4. CombinedTier1Ensemble - Meta-ensemble of all three")
    print("\nUsage:")
    print("  from backend.ensemble_tier1 import AccuracyWeightedEnsemble")
    print("  ensemble = AccuracyWeightedEnsemble()")
    print("  ensemble.fit(forecasts_df, actuals_series, [5, 10, 20])")
    print("  result = ensemble.predict_single(forecast_row, [5, 10, 20])")
    print("\nEvaluation:")
    print("  from backend.ensemble_tier1 import compare_tier1_methods")
    print("  results_df = compare_tier1_methods(forecasts, actuals, horizons)")
