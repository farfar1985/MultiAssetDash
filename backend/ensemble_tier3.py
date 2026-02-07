"""
Tier 3 Ensemble Methods for Pairwise Horizon Predictions
=========================================================

This module implements cutting-edge research (Tier 3) ensemble methods from
ENSEMBLE_METHODS_PLAN.md. These methods explore advanced techniques:

1. ThompsonSamplingEnsemble - Online learning with multi-armed bandit approach
2. AttentionBasedEnsemble - Transformer-style attention over model predictions
3. QuantileRegressionForest - Non-parametric prediction intervals

THEORETICAL FOUNDATIONS
-----------------------

1. THOMPSON SAMPLING (Multi-Armed Bandit)
   Reference: Thompson, W. (1933). "On the Likelihood that One Unknown
              Probability Exceeds Another"

   The Thompson Sampling approach treats model selection as a multi-armed
   bandit problem where each model/pair is an "arm" with unknown reward
   probability. The algorithm maintains Beta distributions for each arm:

   - Beta(alpha, beta) where alpha = successes + 1, beta = failures + 1
   - At each step, sample from each arm's posterior and select the best
   - Update beliefs based on observed outcomes (correct/incorrect prediction)

   Key advantages:
   - Balances exploration (trying uncertain models) vs exploitation (using best)
   - Adapts online without retraining from scratch
   - Converges to optimal model weights over time
   - Theoretically optimal regret bounds

2. ATTENTION-BASED ENSEMBLE (Transformer-style)
   Reference: Vaswani et al. (2017). "Attention Is All You Need"

   Applies the attention mechanism from transformers to ensemble weighting:

   Q = query = f(market context features)
   K = keys = g(model predictions)
   V = values = h(model predictions)

   Attention weights: softmax(QK^T / sqrt(d_k))
   Output: Weighted combination of values

   Key advantages:
   - Context-dependent weighting (adapts to current market conditions)
   - Can learn complex nonlinear relationships
   - Interpretable attention weights show which models are trusted
   - Scalable to large numbers of base models

3. QUANTILE REGRESSION FOREST
   Reference: Meinshausen, N. (2006). "Quantile Regression Forests"

   Extends random forests to predict entire conditional distributions:

   - Each tree's leaves store all training values (not just mean)
   - Predictions aggregate values across leaves from all trees
   - Quantiles computed directly from aggregated empirical distribution

   Key advantages:
   - Non-parametric: no distributional assumptions
   - Robust prediction intervals
   - Handles heteroscedasticity naturally
   - Provides full predictive distribution, not just point estimate

Created: 2026-02-06
Author: AmiraB
Reference: docs/ENSEMBLE_METHODS_PLAN.md Section 2.8-2.10
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict

# Use standardized metrics
from utils.metrics import calculate_sharpe_ratio_daily, calculate_sharpe_ratio

# Optional imports for advanced features
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. AttentionBasedEnsemble will use fallback.")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. QuantileRegressionForest will use fallback.")


# =============================================================================
# RESULT CLASSES
# =============================================================================

@dataclass
class Tier3EnsembleResult:
    """Result from a Tier 3 ensemble prediction."""
    signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    net_probability: float  # -1.0 to 1.0
    weights: Dict[Tuple[int, int], float]  # Pair weights used
    uncertainty: Optional[float] = None  # Model disagreement/variance
    interval_lower: Optional[float] = None  # Prediction interval lower bound
    interval_upper: Optional[float] = None  # Prediction interval upper bound
    quantiles: Optional[Dict[float, float]] = None  # Multiple quantile predictions
    attention_weights: Optional[Dict[str, float]] = None  # For attention ensemble
    exploration_bonus: Optional[float] = None  # For Thompson sampling
    metadata: Dict = field(default_factory=dict)


@dataclass
class BetaDistribution:
    """
    Beta distribution for Thompson Sampling.

    Represents our belief about the success probability of a model/pair.
    Beta(alpha, beta) has mean = alpha/(alpha+beta)
    """
    alpha: float = 1.0  # Successes + prior
    beta: float = 1.0   # Failures + prior

    def sample(self) -> float:
        """Draw a random sample from the posterior."""
        return np.random.beta(self.alpha, self.beta)

    def mean(self) -> float:
        """Expected success probability."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Variance of the distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def update(self, reward: float):
        """
        Update beliefs based on observed outcome.

        Parameters
        ----------
        reward : float
            Reward signal, typically in [0, 1].
            1.0 = full success, 0.0 = full failure
        """
        # Probabilistic update for continuous rewards
        self.alpha += reward
        self.beta += (1 - reward)


# =============================================================================
# BASE CLASS
# =============================================================================

class BaseTier3Ensemble(ABC):
    """Abstract base class for Tier 3 ensemble methods."""

    def __init__(self, lookback_window: int = 60, threshold: float = 0.3):
        """
        Initialize Tier 3 ensemble method.

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
        self._horizons: List[int] = []

    @abstractmethod
    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'BaseTier3Ensemble':
        """Fit the ensemble based on historical data."""
        pass

    @abstractmethod
    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: List[int],
                       **kwargs) -> Tier3EnsembleResult:
        """Generate ensemble prediction for a single timestamp."""
        pass

    def predict(self,
                forecasts: pd.DataFrame,
                horizons: List[int],
                **kwargs) -> pd.Series:
        """
        Generate ensemble predictions for all timestamps.

        Returns pd.Series of Tier3EnsembleResult objects.
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

    def _get_pair_columns(self, h1: int, h2: int, df: pd.DataFrame) -> Tuple[str, str]:
        """Get column names for horizon pair."""
        col1 = f'd{h1}' if f'd{h1}' in df.columns else str(h1)
        col2 = f'd{h2}' if f'd{h2}' in df.columns else str(h2)
        return col1, col2


# =============================================================================
# 1. THOMPSON SAMPLING ENSEMBLE
# =============================================================================

class ThompsonSamplingEnsemble(BaseTier3Ensemble):
    """
    Thompson Sampling Ensemble (Multi-Armed Bandit)

    Treats each horizon pair as an arm in a multi-armed bandit problem.
    Maintains Beta distribution beliefs about each pair's accuracy and
    uses Thompson Sampling to balance exploration vs exploitation.

    THEORETICAL BASIS
    -----------------
    Thompson Sampling is a Bayesian approach to the exploration-exploitation
    tradeoff. For each pair (model), we maintain a posterior distribution
    over its true success probability:

        P(theta_i | data) ~ Beta(alpha_i, beta_i)

    At each decision point:
    1. Sample theta_i ~ Beta(alpha_i, beta_i) for each pair
    2. Select pairs with highest sampled values
    3. Weight ensemble by sampled probabilities
    4. After observing outcome, update posteriors

    The algorithm is proven to achieve optimal regret bounds and naturally
    balances trying uncertain pairs (exploration) with using known-good
    pairs (exploitation).

    Parameters
    ----------
    lookback_window : int
        Initial warm-up period for prior estimation.
    threshold : float
        Signal threshold.
    prior_alpha : float
        Prior successes (higher = more optimistic prior).
    prior_beta : float
        Prior failures (higher = more pessimistic prior).
    decay_rate : float
        Rate at which old observations are forgotten (0-1, lower = more memory).
    exploration_bonus : float
        Bonus weight for uncertain (high variance) pairs.
    n_samples : int
        Number of Thompson samples to average for stable weights.
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 prior_alpha: float = 1.0,
                 prior_beta: float = 1.0,
                 decay_rate: float = 0.99,
                 exploration_bonus: float = 0.1,
                 n_samples: int = 100):
        super().__init__(lookback_window, threshold)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.decay_rate = decay_rate
        self.exploration_bonus = exploration_bonus
        self.n_samples = n_samples

        # Beta distributions for each pair
        self._distributions: Dict[Tuple[int, int], BetaDistribution] = {}

        # Tracking for online updates
        self._prediction_history: List[Dict] = []
        self._total_updates: int = 0

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'ThompsonSamplingEnsemble':
        """
        Initialize Thompson Sampling distributions from historical data.

        Uses historical performance to set informed priors for each pair.
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
        actual_direction = np.sign(actual_returns)

        # Initialize distributions for each pair
        self._distributions = {}

        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                col1, col2 = self._get_pair_columns(h1, h2, forecasts)

                if col1 not in forecasts.columns or col2 not in forecasts.columns:
                    continue

                # Calculate pair's historical accuracy
                pair_drift = forecasts[col2] - forecasts[col1]
                pair_direction = np.sign(pair_drift)

                # Align indices
                common = pair_direction.index.intersection(actual_direction.index)
                if len(common) < 5:
                    # Use uninformative prior
                    self._distributions[(h1, h2)] = BetaDistribution(
                        self.prior_alpha,
                        self.prior_beta
                    )
                    continue

                pred_dir = pair_direction.loc[common].values
                act_dir = actual_direction.loc[common].values

                # Count successes and failures
                correct = (pred_dir == act_dir)
                successes = correct.sum()
                failures = len(correct) - successes

                # Initialize Beta distribution with observed counts + prior
                self._distributions[(h1, h2)] = BetaDistribution(
                    alpha=self.prior_alpha + successes,
                    beta=self.prior_beta + failures
                )

        self._horizons = horizons
        self._is_fitted = True
        self._weights = self._sample_weights()

        return self

    def _sample_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Generate weights by averaging multiple Thompson samples.

        This provides more stable weights while preserving exploration.
        """
        if not self._distributions:
            return {}

        # Accumulate samples
        weight_sums = defaultdict(float)

        for _ in range(self.n_samples):
            samples = {
                pair: dist.sample()
                for pair, dist in self._distributions.items()
            }

            # Normalize samples to weights
            total = sum(samples.values())
            if total > 0:
                for pair, sample in samples.items():
                    weight_sums[pair] += sample / total

        # Average across samples
        weights = {
            pair: weight_sum / self.n_samples
            for pair, weight_sum in weight_sums.items()
        }

        return weights

    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: List[int],
                       **kwargs) -> Tier3EnsembleResult:
        """
        Generate Thompson Sampling prediction.

        Samples from posterior distributions to determine weights,
        providing natural exploration-exploitation balance.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict_single()")

        # Sample fresh weights for exploration
        sampled_weights = self._sample_weights()

        # Add exploration bonus for high-variance pairs
        if self.exploration_bonus > 0:
            variances = {
                pair: dist.variance()
                for pair, dist in self._distributions.items()
            }
            max_var = max(variances.values()) if variances else 1.0

            for pair in sampled_weights:
                if pair in variances and max_var > 0:
                    # Normalize variance and add bonus
                    bonus = self.exploration_bonus * (variances[pair] / max_var)
                    sampled_weights[pair] = sampled_weights[pair] * (1 + bonus)

            # Renormalize
            total = sum(sampled_weights.values())
            if total > 0:
                sampled_weights = {k: v / total for k, v in sampled_weights.items()}

        # Calculate weighted signal
        weighted_bull = 0.0
        weighted_bear = 0.0
        exploration_total = 0.0

        for (h1, h2), weight in sampled_weights.items():
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

            # Track exploration (uncertainty in weights)
            if (h1, h2) in self._distributions:
                exploration_total += self._distributions[(h1, h2)].variance()

        total_weight = weighted_bull + weighted_bear
        if total_weight > 0:
            net_prob = (weighted_bull - weighted_bear) / total_weight
        else:
            net_prob = 0.0

        signal = self._signal_from_probability(net_prob)
        confidence = min(1.0, abs(net_prob) / self.threshold) if self.threshold > 0 else abs(net_prob)

        # Average uncertainty from distributions
        n_pairs = len(self._distributions)
        avg_exploration = exploration_total / n_pairs if n_pairs > 0 else 0.5

        return Tier3EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=net_prob,
            weights=sampled_weights.copy(),
            uncertainty=avg_exploration,
            exploration_bonus=avg_exploration,
            metadata={
                'method': 'ThompsonSamplingEnsemble',
                'n_samples': self.n_samples,
                'total_updates': self._total_updates,
                'pair_means': {str(k): v.mean() for k, v in self._distributions.items()},
                'pair_variances': {str(k): v.variance() for k, v in self._distributions.items()}
            }
        )

    def update(self,
               forecast_row: pd.Series,
               actual_return: float,
               horizons: Optional[List[int]] = None):
        """
        Online update of posterior distributions based on observed outcome.

        This is the key online learning step that allows the ensemble to
        adapt over time without retraining from scratch.

        Parameters
        ----------
        forecast_row : pd.Series
            The forecasts that were used to make the prediction.
        actual_return : float
            The actual return that was observed.
        horizons : List[int], optional
            Horizons to update. Defaults to all fitted horizons.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before update()")

        horizons = horizons or self._horizons
        actual_direction = 1.0 if actual_return > 0 else -1.0

        # Decay old observations (forgetfulness)
        if self.decay_rate < 1.0:
            for dist in self._distributions.values():
                # Shrink towards prior by decaying counts
                excess_alpha = dist.alpha - self.prior_alpha
                excess_beta = dist.beta - self.prior_beta
                dist.alpha = self.prior_alpha + excess_alpha * self.decay_rate
                dist.beta = self.prior_beta + excess_beta * self.decay_rate

        # Update each pair based on whether it predicted correctly
        for (h1, h2), dist in self._distributions.items():
            col1 = f'd{h1}' if f'd{h1}' in forecast_row.index else str(h1)
            col2 = f'd{h2}' if f'd{h2}' in forecast_row.index else str(h2)

            if col1 not in forecast_row.index or col2 not in forecast_row.index:
                continue

            f1 = forecast_row[col1]
            f2 = forecast_row[col2]

            if pd.isna(f1) or pd.isna(f2):
                continue

            drift = f2 - f1
            predicted_direction = 1.0 if drift > 0 else -1.0

            # Reward: 1 if correct, 0 if incorrect
            reward = 1.0 if predicted_direction == actual_direction else 0.0

            # Update Beta distribution
            dist.update(reward)

        self._total_updates += 1
        self._weights = self._sample_weights()

    def get_distributions(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Return the current Beta distribution parameters for each pair."""
        return {
            pair: {
                'alpha': dist.alpha,
                'beta': dist.beta,
                'mean': dist.mean(),
                'variance': dist.variance()
            }
            for pair, dist in self._distributions.items()
        }

    def get_exploration_scores(self) -> Dict[Tuple[int, int], float]:
        """Return exploration scores (variance) for each pair."""
        return {
            pair: dist.variance()
            for pair, dist in self._distributions.items()
        }


# =============================================================================
# 2. ATTENTION-BASED ENSEMBLE
# =============================================================================

class AttentionBasedEnsemble(BaseTier3Ensemble):
    """
    Attention-Based Ensemble (Transformer-style)

    Uses the attention mechanism from transformers to learn context-dependent
    model weights. The "context" can include market features like volatility,
    trend strength, regime indicators, etc.

    THEORETICAL BASIS
    -----------------
    The attention mechanism computes:

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Where:
    - Q (Query): Derived from market context features
    - K (Keys): Derived from model predictions
    - V (Values): Model predictions to be aggregated

    The softmax creates a probability distribution over models, with higher
    weights for models whose "keys" align well with the current "query"
    (context). This allows the ensemble to dynamically attend to different
    models based on current market conditions.

    When PyTorch is not available, falls back to a simplified attention
    mechanism using learned linear weights.

    Parameters
    ----------
    lookback_window : int
        Training window.
    threshold : float
        Signal threshold.
    context_dim : int
        Dimension of market context features.
    hidden_dim : int
        Hidden dimension for attention computation.
    learning_rate : float
        Learning rate for training.
    n_epochs : int
        Number of training epochs.
    use_market_context : bool
        Whether to use market features as context.
    context_features : List[str]
        Which context features to use: 'volatility', 'trend', 'momentum'
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 context_dim: int = 5,
                 hidden_dim: int = 32,
                 learning_rate: float = 0.001,
                 n_epochs: int = 50,
                 use_market_context: bool = True,
                 context_features: Optional[List[str]] = None):
        super().__init__(lookback_window, threshold)
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.use_market_context = use_market_context
        self.context_features = context_features or ['volatility', 'trend', 'momentum']

        # Model components (initialized in fit)
        self._attention_model = None
        self._pair_indices: Dict[Tuple[int, int], int] = {}
        self._n_pairs: int = 0

        # Fallback weights if torch not available
        self._fallback_weights: Dict[Tuple[int, int], float] = {}

        # Context cache for prediction
        self._last_context: Optional[np.ndarray] = None
        self._prices_cache: Optional[np.ndarray] = None

    def _build_attention_model(self, n_pairs: int):
        """Build PyTorch attention model."""
        if not TORCH_AVAILABLE:
            return None

        class AttentionModule(nn.Module):
            def __init__(self, context_dim: int, n_pairs: int, hidden_dim: int):
                super().__init__()
                self.n_pairs = n_pairs

                # Query from context
                self.query_net = nn.Sequential(
                    nn.Linear(context_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

                # Key and value from model predictions
                self.key_net = nn.Linear(1, hidden_dim)
                self.value_net = nn.Linear(1, hidden_dim)

                # Output projection
                self.output_net = nn.Linear(hidden_dim, 1)

                # Scale factor for attention
                self.scale = hidden_dim ** 0.5

            def forward(self, context: torch.Tensor, predictions: torch.Tensor):
                """
                context: (batch_size, context_dim)
                predictions: (batch_size, n_pairs)
                Returns: (batch_size, 1), (batch_size, n_pairs) attention weights
                """
                batch_size = context.shape[0]

                # Query from context: (batch_size, hidden_dim)
                Q = self.query_net(context)

                # Keys and values from predictions: (batch_size, n_pairs, hidden_dim)
                pred_expanded = predictions.unsqueeze(-1)  # (batch, n_pairs, 1)
                K = self.key_net(pred_expanded)  # (batch, n_pairs, hidden)
                V = self.value_net(pred_expanded)  # (batch, n_pairs, hidden)

                # Attention scores: (batch, n_pairs)
                Q_expanded = Q.unsqueeze(1)  # (batch, 1, hidden)
                scores = torch.bmm(Q_expanded, K.transpose(1, 2)).squeeze(1) / self.scale

                # Softmax weights
                weights = torch.softmax(scores, dim=-1)  # (batch, n_pairs)

                # Weighted value combination
                weighted_V = torch.bmm(weights.unsqueeze(1), V).squeeze(1)  # (batch, hidden)

                # Output
                output = self.output_net(weighted_V)  # (batch, 1)

                return output, weights

        return AttentionModule(self.context_dim, n_pairs, self.hidden_dim)

    def _compute_context_features(self, prices: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Compute market context features from price series.

        Returns array of shape (n_samples, context_dim) with features:
        - Volatility (rolling std of returns)
        - Trend strength (price vs MA ratio)
        - Momentum (return over window)
        - Volatility change
        - Return skewness
        """
        if len(prices) < window + 5:
            # Return neutral context
            return np.zeros((1, self.context_dim))

        returns = np.diff(prices) / prices[:-1]

        n_samples = len(returns) - window + 1
        features = np.zeros((n_samples, self.context_dim))

        for i in range(n_samples):
            end_idx = window + i
            window_returns = returns[i:end_idx]
            window_prices = prices[i:end_idx+1]

            # Volatility (normalized)
            vol = np.std(window_returns)
            features[i, 0] = np.clip(vol / 0.02, 0, 5)  # Normalize by typical volatility

            # Trend: price vs moving average
            ma = np.mean(window_prices)
            trend = (window_prices[-1] / ma - 1) * 10
            features[i, 1] = np.clip(trend, -3, 3)

            # Momentum
            momentum = (window_prices[-1] / window_prices[0] - 1) * 5
            features[i, 2] = np.clip(momentum, -3, 3)

            # Volatility change
            if i > 0:
                prev_vol = features[i-1, 0] * 0.02
                vol_change = (vol - prev_vol) / (prev_vol + 1e-8)
                features[i, 3] = np.clip(vol_change, -2, 2)

            # Return skewness
            if len(window_returns) > 2:
                from scipy.stats import skew
                try:
                    sk = skew(window_returns)
                    features[i, 4] = np.clip(sk, -2, 2)
                except:
                    features[i, 4] = 0.0

        return features

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'AttentionBasedEnsemble':
        """
        Train the attention model on historical data.

        Learns to map market context to optimal model weights.
        """
        # Align data
        common_idx = forecasts.index.intersection(actuals.index)
        forecasts = forecasts.loc[common_idx]
        actuals = actuals.loc[common_idx]

        # Store prices for context computation
        self._prices_cache = actuals.values.astype(float)

        # Build pair list
        pairs = []
        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                col1, col2 = self._get_pair_columns(h1, h2, forecasts)
                if col1 in forecasts.columns and col2 in forecasts.columns:
                    pairs.append((h1, h2, col1, col2))

        self._n_pairs = len(pairs)
        self._pair_indices = {(h1, h2): idx for idx, (h1, h2, _, _) in enumerate(pairs)}

        if self._n_pairs == 0:
            self._is_fitted = True
            self._horizons = horizons
            return self

        # Fallback: use accuracy-based weights if no torch
        if not TORCH_AVAILABLE:
            self._compute_fallback_weights(forecasts, actuals, pairs)
            self._is_fitted = True
            self._horizons = horizons
            return self

        # Build attention model
        self._attention_model = self._build_attention_model(self._n_pairs)

        # Prepare training data
        actual_returns = actuals.pct_change().dropna()
        n_samples = len(actual_returns)

        if n_samples < 50:
            self._compute_fallback_weights(forecasts, actuals, pairs)
            self._is_fitted = True
            self._horizons = horizons
            return self

        # Compute context features
        context_features = self._compute_context_features(self._prices_cache)

        # Align context with returns
        min_len = min(len(context_features), n_samples)
        context_features = context_features[-min_len:]
        aligned_returns = actual_returns.iloc[-min_len:]
        aligned_forecasts = forecasts.iloc[-min_len:]

        # Build prediction matrix: (n_samples, n_pairs)
        predictions = np.zeros((min_len, self._n_pairs))
        for h1, h2, col1, col2 in pairs:
            idx = self._pair_indices[(h1, h2)]
            drift = aligned_forecasts[col2] - aligned_forecasts[col1]
            predictions[:, idx] = np.sign(drift.values)

        # Convert to tensors
        X_context = torch.FloatTensor(context_features)
        X_pred = torch.FloatTensor(predictions)
        y = torch.FloatTensor(np.sign(aligned_returns.values)).unsqueeze(1)

        # Train the model
        optimizer = optim.Adam(self._attention_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(self.n_epochs):
            self._attention_model.train()
            optimizer.zero_grad()

            output, _ = self._attention_model(X_context, X_pred)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

        self._attention_model.eval()
        self._is_fitted = True
        self._horizons = horizons

        return self

    def _compute_fallback_weights(self,
                                  forecasts: pd.DataFrame,
                                  actuals: pd.Series,
                                  pairs: List[Tuple]):
        """Compute simple accuracy-based weights as fallback."""
        actual_returns = actuals.pct_change().dropna()
        actual_direction = np.sign(actual_returns)

        weights = {}

        for h1, h2, col1, col2 in pairs:
            pair_drift = forecasts[col2] - forecasts[col1]
            pair_direction = np.sign(pair_drift)

            common = pair_direction.index.intersection(actual_direction.index)
            if len(common) > 5:
                accuracy = (pair_direction.loc[common] == actual_direction.loc[common]).mean()
            else:
                accuracy = 0.5

            weights[(h1, h2)] = accuracy ** 2

        # Normalize
        total = sum(weights.values())
        if total > 0:
            self._fallback_weights = {k: v / total for k, v in weights.items()}
        else:
            n = len(weights)
            self._fallback_weights = {k: 1.0 / n for k in weights} if n > 0 else {}

    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: List[int],
                       prices: Optional[np.ndarray] = None,
                       **kwargs) -> Tier3EnsembleResult:
        """
        Generate attention-weighted prediction.

        Uses learned attention to weight models based on current context.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict_single()")

        # Use fallback if no torch or model
        if not TORCH_AVAILABLE or self._attention_model is None:
            return self._predict_fallback(forecast_row, horizons)

        # Compute current context
        prices_for_context = prices if prices is not None else self._prices_cache
        if prices_for_context is not None and len(prices_for_context) > 20:
            context = self._compute_context_features(prices_for_context)[-1:]
        else:
            context = np.zeros((1, self.context_dim))

        # Build prediction vector
        predictions = np.zeros((1, self._n_pairs))
        for (h1, h2), idx in self._pair_indices.items():
            col1 = f'd{h1}' if f'd{h1}' in forecast_row.index else str(h1)
            col2 = f'd{h2}' if f'd{h2}' in forecast_row.index else str(h2)

            if col1 in forecast_row.index and col2 in forecast_row.index:
                f1 = forecast_row[col1]
                f2 = forecast_row[col2]
                if not (pd.isna(f1) or pd.isna(f2)):
                    predictions[0, idx] = np.sign(f2 - f1)

        # Get attention output
        X_context = torch.FloatTensor(context)
        X_pred = torch.FloatTensor(predictions)

        with torch.no_grad():
            output, attention_weights = self._attention_model(X_context, X_pred)

        net_prob = float(output[0, 0])
        attention_weights_np = attention_weights[0].numpy()

        # Map attention weights back to pairs
        pair_weights = {}
        attention_dict = {}
        for (h1, h2), idx in self._pair_indices.items():
            pair_weights[(h1, h2)] = float(attention_weights_np[idx])
            attention_dict[f"({h1},{h2})"] = float(attention_weights_np[idx])

        # Clip net_prob to valid range
        net_prob = np.clip(net_prob, -1.0, 1.0)

        signal = self._signal_from_probability(net_prob)
        confidence = min(1.0, abs(net_prob) / self.threshold) if self.threshold > 0 else abs(net_prob)

        return Tier3EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=net_prob,
            weights=pair_weights,
            attention_weights=attention_dict,
            metadata={
                'method': 'AttentionBasedEnsemble',
                'context_features': context[0].tolist() if len(context) > 0 else [],
                'using_torch': True,
                'hidden_dim': self.hidden_dim
            }
        )

    def _predict_fallback(self,
                          forecast_row: pd.Series,
                          horizons: List[int]) -> Tier3EnsembleResult:
        """Fallback prediction using simple accuracy weights."""
        weighted_bull = 0.0
        weighted_bear = 0.0

        for (h1, h2), weight in self._fallback_weights.items():
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
            net_prob = (weighted_bull - weighted_bear) / total
        else:
            net_prob = 0.0

        signal = self._signal_from_probability(net_prob)
        confidence = min(1.0, abs(net_prob) / self.threshold) if self.threshold > 0 else abs(net_prob)

        return Tier3EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=net_prob,
            weights=self._fallback_weights.copy(),
            attention_weights={str(k): v for k, v in self._fallback_weights.items()},
            metadata={
                'method': 'AttentionBasedEnsemble',
                'using_torch': False,
                'fallback': 'accuracy_weighted'
            }
        )

    def get_attention_weights(self,
                              context: np.ndarray,
                              predictions: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Get attention weights for given context and predictions."""
        if not TORCH_AVAILABLE or self._attention_model is None:
            return self._fallback_weights.copy()

        X_context = torch.FloatTensor(context.reshape(1, -1))
        X_pred = torch.FloatTensor(predictions.reshape(1, -1))

        with torch.no_grad():
            _, attention_weights = self._attention_model(X_context, X_pred)

        weights_np = attention_weights[0].numpy()

        return {
            pair: float(weights_np[idx])
            for pair, idx in self._pair_indices.items()
        }


# =============================================================================
# 3. QUANTILE REGRESSION FOREST
# =============================================================================

class QuantileRegressionForest(BaseTier3Ensemble):
    """
    Quantile Regression Forest Ensemble

    Uses random forests to predict the full conditional distribution of
    returns, enabling robust non-parametric prediction intervals.

    THEORETICAL BASIS
    -----------------
    Standard random forests predict the conditional mean E[Y|X]. Quantile
    Regression Forests instead estimate any quantile Q_tau(Y|X) by:

    1. For each tree, find the leaf node containing X
    2. Collect all training Y values that fell into that leaf
    3. Aggregate values across all trees
    4. Compute empirical quantile from the aggregated distribution

    This provides:
    - Non-parametric prediction intervals (no distributional assumptions)
    - Heteroscedastic uncertainty (wider intervals when more uncertain)
    - Full predictive distribution for risk analysis
    - Robust to outliers

    The ensemble signal is derived from whether key quantiles (e.g., 10th
    and 90th percentile) are above or below zero.

    Parameters
    ----------
    lookback_window : int
        Training window size.
    threshold : float
        Signal threshold.
    n_estimators : int
        Number of trees in the forest.
    min_samples_leaf : int
        Minimum samples per leaf (affects smoothness of quantiles).
    quantiles : List[float]
        Quantiles to compute (default: 10%, 25%, 50%, 75%, 90%)
    max_depth : int
        Maximum tree depth.
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 n_estimators: int = 100,
                 min_samples_leaf: int = 5,
                 quantiles: Optional[List[float]] = None,
                 max_depth: Optional[int] = None):
        super().__init__(lookback_window, threshold)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.quantiles = quantiles or [0.10, 0.25, 0.50, 0.75, 0.90]
        self.max_depth = max_depth

        # Forest and leaf storage
        self._forest: Optional[RandomForestRegressor] = None
        self._leaf_values: Dict[Tuple[int, int], np.ndarray] = {}  # (tree_idx, leaf_id) -> y values
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

        # Feature names for interpretability
        self._feature_names: List[str] = []

        # Fallback weights if sklearn not available
        self._fallback_weights: Dict[Tuple[int, int], float] = {}

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'QuantileRegressionForest':
        """
        Fit the Quantile Regression Forest.

        Features are derived from horizon pair drifts and market features.
        Target is the actual return direction/magnitude.
        """
        # Align data
        common_idx = forecasts.index.intersection(actuals.index)
        forecasts = forecasts.loc[common_idx]
        actuals = actuals.loc[common_idx]

        # Build pair list
        pairs = []
        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                col1, col2 = self._get_pair_columns(h1, h2, forecasts)
                if col1 in forecasts.columns and col2 in forecasts.columns:
                    pairs.append((h1, h2, col1, col2))

        if len(pairs) == 0:
            self._is_fitted = True
            self._horizons = horizons
            return self

        # Build feature matrix: pair drifts as features
        self._feature_names = [f"drift_{h1}_{h2}" for h1, h2, _, _ in pairs]

        X_data = np.zeros((len(forecasts), len(pairs)))
        for idx, (h1, h2, col1, col2) in enumerate(pairs):
            drift = forecasts[col2] - forecasts[col1]
            # Normalize drift by base value
            base = forecasts[col1].replace(0, np.nan)
            normalized_drift = (drift / base).fillna(0)
            X_data[:, idx] = normalized_drift.values

        # Target: actual return direction (or magnitude)
        actual_returns = actuals.pct_change()
        y_data = np.sign(actual_returns.values)

        # Remove NaN rows
        valid_mask = ~(np.isnan(X_data).any(axis=1) | np.isnan(y_data))
        X_data = X_data[valid_mask]
        y_data = y_data[valid_mask]

        if len(X_data) < 30:
            # Fallback to simple weights
            self._compute_fallback_weights(forecasts, actuals, pairs)
            self._is_fitted = True
            self._horizons = horizons
            return self

        if not SKLEARN_AVAILABLE:
            self._compute_fallback_weights(forecasts, actuals, pairs)
            self._is_fitted = True
            self._horizons = horizons
            return self

        # Fit Random Forest
        self._forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        self._forest.fit(X_data, y_data)

        # Store training data for quantile computation
        self._X_train = X_data
        self._y_train = y_data

        # Collect leaf assignments for all training samples
        self._leaf_values = {}
        for tree_idx, tree in enumerate(self._forest.estimators_):
            # Get leaf IDs for all training samples
            leaf_ids = tree.apply(X_data)

            for leaf_id in np.unique(leaf_ids):
                mask = leaf_ids == leaf_id
                self._leaf_values[(tree_idx, leaf_id)] = y_data[mask]

        # Store pair indices for prediction
        self._pair_indices = {(h1, h2): idx for idx, (h1, h2, _, _) in enumerate(pairs)}

        self._is_fitted = True
        self._horizons = horizons

        return self

    def _compute_fallback_weights(self,
                                  forecasts: pd.DataFrame,
                                  actuals: pd.Series,
                                  pairs: List[Tuple]):
        """Compute accuracy-based weights as fallback."""
        actual_returns = actuals.pct_change().dropna()
        actual_direction = np.sign(actual_returns)

        weights = {}

        for h1, h2, col1, col2 in pairs:
            pair_drift = forecasts[col2] - forecasts[col1]
            pair_direction = np.sign(pair_drift)

            common = pair_direction.index.intersection(actual_direction.index)
            if len(common) > 5:
                accuracy = (pair_direction.loc[common] == actual_direction.loc[common]).mean()
            else:
                accuracy = 0.5

            weights[(h1, h2)] = accuracy ** 2

        total = sum(weights.values())
        if total > 0:
            self._fallback_weights = {k: v / total for k, v in weights.items()}
        else:
            n = len(weights)
            self._fallback_weights = {k: 1.0 / n for k in weights} if n > 0 else {}

    def _get_quantile_prediction(self, x: np.ndarray) -> Dict[float, float]:
        """
        Get quantile predictions for a single sample.

        Aggregates training values from all leaves across all trees.
        """
        if self._forest is None:
            return {q: 0.0 for q in self.quantiles}

        x = x.reshape(1, -1)

        # Collect values from all trees
        all_values = []

        for tree_idx, tree in enumerate(self._forest.estimators_):
            leaf_id = tree.apply(x)[0]
            key = (tree_idx, leaf_id)

            if key in self._leaf_values:
                all_values.extend(self._leaf_values[key])

        if len(all_values) == 0:
            return {q: 0.0 for q in self.quantiles}

        all_values = np.array(all_values)

        # Compute quantiles
        return {
            q: float(np.percentile(all_values, q * 100))
            for q in self.quantiles
        }

    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: List[int],
                       **kwargs) -> Tier3EnsembleResult:
        """
        Generate quantile forest prediction with uncertainty intervals.

        Returns point prediction (median) plus quantile-based intervals.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict_single()")

        # Fallback if no forest
        if self._forest is None or not SKLEARN_AVAILABLE:
            return self._predict_fallback(forecast_row, horizons)

        # Build feature vector
        x = np.zeros(len(self._pair_indices))

        for (h1, h2), idx in self._pair_indices.items():
            col1 = f'd{h1}' if f'd{h1}' in forecast_row.index else str(h1)
            col2 = f'd{h2}' if f'd{h2}' in forecast_row.index else str(h2)

            if col1 in forecast_row.index and col2 in forecast_row.index:
                f1 = forecast_row[col1]
                f2 = forecast_row[col2]

                if not (pd.isna(f1) or pd.isna(f2)) and f1 != 0:
                    x[idx] = (f2 - f1) / f1

        # Get quantile predictions
        quantile_preds = self._get_quantile_prediction(x)

        # Point prediction: median (50th percentile)
        net_prob = quantile_preds.get(0.50, 0.0)

        # Prediction interval: 10th to 90th percentile
        interval_lower = quantile_preds.get(0.10, -1.0)
        interval_upper = quantile_preds.get(0.90, 1.0)

        # Uncertainty: width of interval
        uncertainty = interval_upper - interval_lower

        # Signal based on quantile analysis
        # Strong signal if even extreme quantiles agree on direction
        if quantile_preds.get(0.10, 0) > 0:  # Even 10th percentile is positive
            signal = 'BULLISH'
            confidence = min(1.0, quantile_preds.get(0.10, 0) / 0.5)
        elif quantile_preds.get(0.90, 0) < 0:  # Even 90th percentile is negative
            signal = 'BEARISH'
            confidence = min(1.0, abs(quantile_preds.get(0.90, 0)) / 0.5)
        elif net_prob > self.threshold:
            signal = 'BULLISH'
            confidence = min(1.0, abs(net_prob) / self.threshold)
        elif net_prob < -self.threshold:
            signal = 'BEARISH'
            confidence = min(1.0, abs(net_prob) / self.threshold)
        else:
            signal = 'NEUTRAL'
            confidence = 1.0 - abs(net_prob) / self.threshold if self.threshold > 0 else 0.5

        # Feature importances as weights
        pair_weights = {}
        if hasattr(self._forest, 'feature_importances_'):
            importances = self._forest.feature_importances_
            for (h1, h2), idx in self._pair_indices.items():
                pair_weights[(h1, h2)] = float(importances[idx])

            # Normalize
            total = sum(pair_weights.values())
            if total > 0:
                pair_weights = {k: v / total for k, v in pair_weights.items()}

        return Tier3EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=net_prob,
            weights=pair_weights,
            uncertainty=uncertainty,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            quantiles=quantile_preds,
            metadata={
                'method': 'QuantileRegressionForest',
                'n_estimators': self.n_estimators,
                'quantiles_computed': self.quantiles,
                'feature_importances': pair_weights
            }
        )

    def _predict_fallback(self,
                          forecast_row: pd.Series,
                          horizons: List[int]) -> Tier3EnsembleResult:
        """Fallback prediction using simple accuracy weights."""
        weighted_bull = 0.0
        weighted_bear = 0.0

        for (h1, h2), weight in self._fallback_weights.items():
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
            net_prob = (weighted_bull - weighted_bear) / total
        else:
            net_prob = 0.0

        signal = self._signal_from_probability(net_prob)
        confidence = min(1.0, abs(net_prob) / self.threshold) if self.threshold > 0 else abs(net_prob)

        return Tier3EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=net_prob,
            weights=self._fallback_weights.copy(),
            uncertainty=1.0,  # Maximum uncertainty in fallback
            interval_lower=-1.0,
            interval_upper=1.0,
            quantiles={q: net_prob for q in self.quantiles},
            metadata={
                'method': 'QuantileRegressionForest',
                'fallback': 'accuracy_weighted',
                'sklearn_available': SKLEARN_AVAILABLE
            }
        )

    def get_feature_importances(self) -> Dict[Tuple[int, int], float]:
        """Get feature importances from the forest."""
        if self._forest is None or not hasattr(self._forest, 'feature_importances_'):
            return {}

        importances = self._forest.feature_importances_
        return {
            pair: float(importances[idx])
            for pair, idx in self._pair_indices.items()
        }

    def predict_quantile(self, x: np.ndarray, quantile: float) -> float:
        """Predict a specific quantile."""
        quantiles = self._get_quantile_prediction(x)
        return quantiles.get(quantile, 0.0)


# =============================================================================
# COMBINED TIER 3 ENSEMBLE
# =============================================================================

class CombinedTier3Ensemble:
    """
    Combined Tier 3 Ensemble

    Combines all three Tier 3 research methods:
    1. Thompson Sampling - for exploration-aware online learning
    2. Attention-Based - for context-dependent weighting
    3. Quantile Forest - for robust uncertainty quantification

    The combination leverages:
    - Thompson Sampling's exploration bonus for uncertain markets
    - Attention's context-awareness for regime adaptation
    - Quantile Forest's distribution for robust intervals
    """

    def __init__(self,
                 lookback_window: int = 60,
                 threshold: float = 0.3,
                 thompson_weight: float = 0.35,
                 attention_weight: float = 0.35,
                 qrf_weight: float = 0.30):
        """
        Initialize combined Tier 3 ensemble.

        Parameters
        ----------
        lookback_window : int
            Training window.
        threshold : float
            Signal threshold.
        thompson_weight : float
            Weight for Thompson Sampling component.
        attention_weight : float
            Weight for Attention-Based component.
        qrf_weight : float
            Weight for Quantile Regression Forest component.
        """
        self.lookback_window = lookback_window
        self.threshold = threshold

        self.component_weights = {
            'thompson': thompson_weight,
            'attention': attention_weight,
            'qrf': qrf_weight
        }

        # Initialize components
        self.thompson = ThompsonSamplingEnsemble(lookback_window, threshold)
        self.attention = AttentionBasedEnsemble(lookback_window, threshold)
        self.qrf = QuantileRegressionForest(lookback_window, threshold)

        self._is_fitted = False
        self._horizons: List[int] = []

    def fit(self,
            forecasts: pd.DataFrame,
            actuals: pd.Series,
            horizons: List[int]) -> 'CombinedTier3Ensemble':
        """Fit all component ensembles."""
        self.thompson.fit(forecasts, actuals, horizons)
        self.attention.fit(forecasts, actuals, horizons)
        self.qrf.fit(forecasts, actuals, horizons)

        self._horizons = horizons
        self._is_fitted = True

        return self

    def predict_single(self,
                       forecast_row: pd.Series,
                       horizons: Optional[List[int]] = None,
                       prices: Optional[np.ndarray] = None) -> Tier3EnsembleResult:
        """
        Generate combined Tier 3 prediction.

        Combines signals from all components with specified weights,
        uses QRF for uncertainty intervals.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() first")

        horizons = horizons or self._horizons

        # Get predictions from each component
        thompson_result = self.thompson.predict_single(forecast_row, horizons)
        attention_result = self.attention.predict_single(forecast_row, horizons, prices=prices)
        qrf_result = self.qrf.predict_single(forecast_row, horizons)

        # Weighted combination of net probabilities
        combined_prob = (
            self.component_weights['thompson'] * thompson_result.net_probability +
            self.component_weights['attention'] * attention_result.net_probability +
            self.component_weights['qrf'] * qrf_result.net_probability
        )

        # Use QRF intervals (most robust)
        interval_lower = qrf_result.interval_lower
        interval_upper = qrf_result.interval_upper

        # Uncertainty: combine exploration bonus with QRF uncertainty
        uncertainty = (
            0.5 * (thompson_result.exploration_bonus or 0.5) +
            0.5 * (qrf_result.uncertainty or 1.0)
        )

        signal = 'BULLISH' if combined_prob > self.threshold else \
                 'BEARISH' if combined_prob < -self.threshold else 'NEUTRAL'

        confidence = min(1.0, abs(combined_prob) / self.threshold) if self.threshold > 0 else abs(combined_prob)

        return Tier3EnsembleResult(
            signal=signal,
            confidence=confidence,
            net_probability=combined_prob,
            weights={},  # Combined from multiple sources
            uncertainty=uncertainty,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            quantiles=qrf_result.quantiles,
            attention_weights=attention_result.attention_weights,
            exploration_bonus=thompson_result.exploration_bonus,
            metadata={
                'method': 'CombinedTier3',
                'component_signals': {
                    'thompson': thompson_result.signal,
                    'attention': attention_result.signal,
                    'qrf': qrf_result.signal
                },
                'component_probs': {
                    'thompson': thompson_result.net_probability,
                    'attention': attention_result.net_probability,
                    'qrf': qrf_result.net_probability
                },
                'component_weights': self.component_weights
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

    def update_thompson(self,
                        forecast_row: pd.Series,
                        actual_return: float):
        """Update Thompson Sampling component with new observation."""
        self.thompson.update(forecast_row, actual_return)


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_tier3_ensemble(
    ensemble: BaseTier3Ensemble,
    forecasts: pd.DataFrame,
    actuals: pd.Series,
    horizons: List[int],
    train_pct: float = 0.7
) -> Dict:
    """
    Evaluate a Tier 3 ensemble method with train/test split.

    Returns comprehensive metrics including uncertainty calibration.
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
    uncertainties = []
    intervals_contain_actual = []

    for i, result in enumerate(predictions):
        if result.signal == 'BULLISH':
            signals.append(1)
        elif result.signal == 'BEARISH':
            signals.append(-1)
        else:
            signals.append(0)

        uncertainties.append(result.uncertainty or 0.5)

        # Check interval coverage
        if result.interval_lower is not None and result.interval_upper is not None:
            idx = predictions.index[i]
            if idx in test_returns.index:
                actual = np.sign(test_returns.loc[idx])
                in_interval = result.interval_lower <= actual <= result.interval_upper
                intervals_contain_actual.append(in_interval)

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

    result_dict = {
        'method': ensemble.__class__.__name__,
        'sharpe': round(sharpe, 3),
        'directional_accuracy': round(accuracy, 2),
        'total_return': round(total_return, 2),
        'win_rate': round(win_rate, 2),
        'n_predictions': len(strategy_returns),
        'avg_uncertainty': round(np.mean(uncertainties), 3)
    }

    # Interval coverage (for QRF)
    if intervals_contain_actual:
        result_dict['interval_coverage'] = round(np.mean(intervals_contain_actual) * 100, 2)

    # Method-specific metrics
    if hasattr(ensemble, 'get_distributions'):
        dists = ensemble.get_distributions()
        result_dict['n_pairs_tracked'] = len(dists)

    if hasattr(ensemble, 'get_feature_importances'):
        importances = ensemble.get_feature_importances()
        if importances:
            top_pair = max(importances, key=importances.get)
            result_dict['top_feature'] = str(top_pair)

    return result_dict


def compare_tier3_methods(
    forecasts: pd.DataFrame,
    actuals: pd.Series,
    horizons: List[int],
    train_pct: float = 0.7
) -> pd.DataFrame:
    """
    Compare all Tier 3 methods on the same data.

    Returns DataFrame with metrics for each method.
    """
    results = []

    methods = [
        ('Thompson-Sampling', ThompsonSamplingEnsemble()),
        ('Attention-Based', AttentionBasedEnsemble()),
        ('Quantile-Forest', QuantileRegressionForest()),
        ('Combined-Tier3', CombinedTier3Ensemble()),
    ]

    for name, ensemble in methods:
        try:
            if isinstance(ensemble, CombinedTier3Ensemble):
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
                result = evaluate_tier3_ensemble(ensemble, forecasts, actuals, horizons, train_pct)
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
    print("TIER 3 ENSEMBLE METHODS - Research Implementations")
    print("=" * 70)
    print("\nAvailable methods:")
    print("  1. ThompsonSamplingEnsemble - Online learning with multi-armed bandit")
    print("  2. AttentionBasedEnsemble - Transformer-style attention over predictions")
    print("  3. QuantileRegressionForest - Non-parametric prediction intervals")
    print("  4. CombinedTier3Ensemble - Meta-ensemble of all three")
    print("\nTheoretical Foundations:")
    print("  - Thompson Sampling: Bayesian exploration-exploitation (Thompson, 1933)")
    print("  - Attention: Scaled dot-product attention (Vaswani et al., 2017)")
    print("  - Quantile RF: Conditional distribution forests (Meinshausen, 2006)")
    print("\nUsage:")
    print("  from backend.ensemble_tier3 import ThompsonSamplingEnsemble")
    print("  ensemble = ThompsonSamplingEnsemble()")
    print("  ensemble.fit(forecasts_df, actuals_series, [5, 10, 20])")
    print("  result = ensemble.predict_single(forecast_row, [5, 10, 20])")
    print("\n  # Online update for Thompson Sampling:")
    print("  ensemble.update(forecast_row, actual_return)")
    print("\n  # Get quantile predictions:")
    print("  from backend.ensemble_tier3 import QuantileRegressionForest")
    print("  qrf = QuantileRegressionForest()")
    print("  result = qrf.predict_single(forecast_row, horizons)")
    print("  print(result.quantiles)  # {0.1: ..., 0.5: ..., 0.9: ...}")
    print("\nEvaluation:")
    print("  from backend.ensemble_tier3 import compare_tier3_methods")
    print("  results_df = compare_tier3_methods(forecasts, actuals, horizons)")
    print("\nDependencies:")
    print(f"  PyTorch available: {TORCH_AVAILABLE}")
    print(f"  sklearn available: {SKLEARN_AVAILABLE}")
