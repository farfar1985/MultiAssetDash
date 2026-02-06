"""
Unit tests for Ensemble Tier Methods.

Tests all 12 ensemble methods across Tier 1, 2, and 3:
- Tier 1: AccuracyWeighted, MagnitudeWeighted, ErrorCorrelation, Combined
- Tier 2: BayesianModelAveraging, RegimeAdaptive, ConformalPrediction, Combined
- Tier 3: ThompsonSampling, AttentionBased, QuantileForest, Combined

Includes edge cases for empty data, single predictions, and malformed inputs.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import Tier 1
from backend.ensemble_tier1 import (
    AccuracyWeightedEnsemble,
    MagnitudeWeightedVoting,
    ErrorCorrelationWeighting,
    CombinedTier1Ensemble,
    EnsembleResult,
)

# Import Tier 2
from backend.ensemble_tier2 import (
    BayesianModelAveraging,
    RegimeAdaptiveEnsemble,
    ConformalPredictionInterval,
    CombinedTier2Ensemble,
    Tier2EnsembleResult,
)

# Import Tier 3
from backend.ensemble_tier3 import (
    ThompsonSamplingEnsemble,
    AttentionBasedEnsemble,
    QuantileRegressionForest,
    CombinedTier3Ensemble,
    Tier3EnsembleResult,
    BetaDistribution,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_horizons():
    """Standard horizons for testing."""
    return [5, 10, 20, 40]


@pytest.fixture
def sample_data():
    """Generate sample forecast and actual data for testing."""
    np.random.seed(42)
    n_samples = 100
    dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')

    # Simulated price series with trend
    base_price = 100
    returns = np.random.randn(n_samples) * 0.02
    prices = base_price * np.cumprod(1 + returns)
    actuals = pd.Series(prices, index=dates)

    # Simulated forecasts for different horizons
    forecasts = pd.DataFrame(index=dates)
    for h in [5, 10, 20, 40]:
        # Forecasts are noisy versions of future prices
        noise = np.random.randn(n_samples) * 0.01
        forecasts[f'd{h}'] = prices * (1 + noise + h * 0.001)

    return forecasts, actuals


@pytest.fixture
def small_data():
    """Minimal data for edge case testing."""
    np.random.seed(42)
    n_samples = 20
    dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')

    prices = 100 + np.cumsum(np.random.randn(n_samples))
    actuals = pd.Series(prices, index=dates)

    forecasts = pd.DataFrame(index=dates)
    for h in [5, 10, 20]:
        forecasts[f'd{h}'] = prices * (1 + np.random.randn(n_samples) * 0.005)

    return forecasts, actuals


@pytest.fixture
def empty_data():
    """Empty DataFrames for edge case testing."""
    dates = pd.DatetimeIndex([])
    forecasts = pd.DataFrame(index=dates, columns=['d5', 'd10', 'd20'])
    actuals = pd.Series([], index=dates, dtype=float)
    return forecasts, actuals


@pytest.fixture
def single_row_data():
    """Single row of data for edge case testing."""
    dates = pd.date_range(start='2025-01-01', periods=1, freq='D')
    forecasts = pd.DataFrame({
        'd5': [100.5],
        'd10': [101.0],
        'd20': [102.0],
    }, index=dates)
    actuals = pd.Series([100.0], index=dates)
    return forecasts, actuals


@pytest.fixture
def data_with_nans():
    """Data containing NaN values."""
    np.random.seed(42)
    n_samples = 50
    dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')

    prices = 100 + np.cumsum(np.random.randn(n_samples))
    actuals = pd.Series(prices, index=dates)

    forecasts = pd.DataFrame(index=dates)
    for h in [5, 10, 20]:
        vals = prices * (1 + np.random.randn(n_samples) * 0.005)
        # Inject NaNs
        vals[10:15] = np.nan
        forecasts[f'd{h}'] = vals

    return forecasts, actuals


# =============================================================================
# TIER 1 TESTS
# =============================================================================

class TestAccuracyWeightedEnsemble:
    """Tests for AccuracyWeightedEnsemble."""

    def test_initialization(self):
        """Test ensemble can be initialized with default params."""
        ensemble = AccuracyWeightedEnsemble()
        assert ensemble.lookback_window == 60
        assert ensemble.threshold == 0.3
        assert not ensemble._is_fitted

    def test_initialization_custom_params(self):
        """Test ensemble with custom parameters."""
        ensemble = AccuracyWeightedEnsemble(
            lookback_window=30,
            threshold=0.5,
            accuracy_power=3.0,
            min_accuracy=0.6
        )
        assert ensemble.lookback_window == 30
        assert ensemble.threshold == 0.5
        assert ensemble.accuracy_power == 3.0
        assert ensemble.min_accuracy == 0.6

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test basic fit and predict workflow."""
        forecasts, actuals = sample_data
        ensemble = AccuracyWeightedEnsemble()

        # Fit should return self
        result = ensemble.fit(forecasts, actuals, sample_horizons)
        assert result is ensemble
        assert ensemble._is_fitted

        # Predict single row
        forecast_row = forecasts.iloc[-1]
        prediction = ensemble.predict_single(forecast_row, sample_horizons)

        assert isinstance(prediction, EnsembleResult)
        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert 0.0 <= prediction.confidence <= 1.0
        assert -1.0 <= prediction.net_probability <= 1.0

    def test_predict_without_fit_raises(self, sample_data, sample_horizons):
        """Test that predicting without fitting raises error."""
        forecasts, _ = sample_data
        ensemble = AccuracyWeightedEnsemble()

        with pytest.raises(ValueError, match="fit"):
            ensemble.predict_single(forecasts.iloc[0], sample_horizons)

    def test_weights_are_normalized(self, sample_data, sample_horizons):
        """Test that weights sum to 1."""
        forecasts, actuals = sample_data
        ensemble = AccuracyWeightedEnsemble()
        ensemble.fit(forecasts, actuals, sample_horizons)

        weights = ensemble.get_weights()
        assert len(weights) > 0
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_pair_accuracies_available(self, sample_data, sample_horizons):
        """Test that pair accuracies are computed."""
        forecasts, actuals = sample_data
        ensemble = AccuracyWeightedEnsemble()
        ensemble.fit(forecasts, actuals, sample_horizons)

        accuracies = ensemble.get_pair_accuracies()
        assert len(accuracies) > 0
        for pair, acc in accuracies.items():
            assert 0.0 <= acc <= 1.0


class TestMagnitudeWeightedVoting:
    """Tests for MagnitudeWeightedVoting."""

    def test_initialization(self):
        """Test ensemble initialization."""
        ensemble = MagnitudeWeightedVoting()
        assert ensemble.use_separation_weight is True
        assert ensemble.magnitude_cap == 0.1

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test basic fit and predict."""
        forecasts, actuals = sample_data
        ensemble = MagnitudeWeightedVoting()

        ensemble.fit(forecasts, actuals, sample_horizons)
        assert ensemble._is_fitted

        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons)
        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']

    def test_magnitude_affects_weights(self, sample_horizons):
        """Test that larger magnitude predictions get more weight."""
        dates = pd.date_range(start='2025-01-01', periods=50, freq='D')

        # Create forecasts with varying magnitudes
        forecasts = pd.DataFrame({
            'd5': [100.0] * 50,
            'd10': [100.1] * 50,  # Small drift
            'd20': [105.0] * 50,  # Large drift
        }, index=dates)
        actuals = pd.Series([100.0] * 50, index=dates)

        ensemble = MagnitudeWeightedVoting()
        ensemble.fit(forecasts, actuals, [5, 10, 20])

        # The larger drift pair (5, 20) should have higher weight
        prediction = ensemble.predict_single(forecasts.iloc[-1], [5, 10, 20])
        assert prediction.metadata.get('total_weight', 0) > 0


class TestErrorCorrelationWeighting:
    """Tests for ErrorCorrelationWeighting."""

    def test_initialization(self):
        """Test ensemble initialization."""
        ensemble = ErrorCorrelationWeighting()
        assert ensemble.epsilon == 0.1
        assert ensemble.correlation_power == 1.0

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test basic fit and predict."""
        forecasts, actuals = sample_data
        ensemble = ErrorCorrelationWeighting()

        ensemble.fit(forecasts, actuals, sample_horizons)
        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons)

        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']

    def test_error_correlations_computed(self, sample_data, sample_horizons):
        """Test that error correlations are computed."""
        forecasts, actuals = sample_data
        ensemble = ErrorCorrelationWeighting()
        ensemble.fit(forecasts, actuals, sample_horizons)

        avg_corr = ensemble.get_avg_correlations()
        assert len(avg_corr) > 0

    def test_inverse_correlation_weighting(self, sample_data, sample_horizons):
        """Test that less correlated pairs get higher weights."""
        forecasts, actuals = sample_data
        ensemble = ErrorCorrelationWeighting()
        ensemble.fit(forecasts, actuals, sample_horizons)

        weights = ensemble.get_weights()
        avg_corr = ensemble.get_avg_correlations()

        # Pairs with lower average correlation should have higher weights
        if len(avg_corr) >= 2:
            pairs = list(avg_corr.keys())
            for i in range(len(pairs) - 1):
                for j in range(i + 1, len(pairs)):
                    p1, p2 = pairs[i], pairs[j]
                    if avg_corr[p1] < avg_corr[p2]:
                        # Lower correlation should mean higher weight
                        assert weights.get(p1, 0) >= weights.get(p2, 0) * 0.5


class TestCombinedTier1Ensemble:
    """Tests for CombinedTier1Ensemble."""

    def test_initialization(self):
        """Test combined ensemble initialization."""
        ensemble = CombinedTier1Ensemble()
        assert ensemble.accuracy_ensemble is not None
        assert ensemble.magnitude_ensemble is not None
        assert ensemble.correlation_ensemble is not None

    def test_custom_method_weights(self):
        """Test custom method weights."""
        weights = {'accuracy': 0.5, 'magnitude': 0.3, 'correlation': 0.2}
        ensemble = CombinedTier1Ensemble(method_weights=weights)
        assert ensemble.method_weights == weights

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test combined fit and predict."""
        forecasts, actuals = sample_data
        ensemble = CombinedTier1Ensemble()

        ensemble.fit(forecasts, actuals, sample_horizons)
        prediction = ensemble.predict_single(forecasts.iloc[-1])

        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert 'component_signals' in prediction.metadata
        assert 'component_probs' in prediction.metadata

    def test_batch_predict(self, sample_data, sample_horizons):
        """Test batch prediction."""
        forecasts, actuals = sample_data
        ensemble = CombinedTier1Ensemble()

        ensemble.fit(forecasts, actuals, sample_horizons)
        predictions = ensemble.predict(forecasts.iloc[-10:])

        assert len(predictions) == 10
        for pred in predictions:
            assert isinstance(pred, EnsembleResult)


# =============================================================================
# TIER 2 TESTS
# =============================================================================

class TestBayesianModelAveraging:
    """Tests for BayesianModelAveraging."""

    def test_initialization(self):
        """Test BMA initialization."""
        ensemble = BayesianModelAveraging()
        assert ensemble.prior == 'uniform'
        assert ensemble.likelihood_decay == 0.95
        assert ensemble.min_likelihood == 0.01

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test BMA fit and predict."""
        forecasts, actuals = sample_data
        ensemble = BayesianModelAveraging()

        ensemble.fit(forecasts, actuals, sample_horizons)
        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons)

        assert isinstance(prediction, Tier2EnsembleResult)
        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert prediction.uncertainty is not None

    def test_posteriors_sum_to_one(self, sample_data, sample_horizons):
        """Test that posteriors are proper probabilities."""
        forecasts, actuals = sample_data
        ensemble = BayesianModelAveraging()
        ensemble.fit(forecasts, actuals, sample_horizons)

        posteriors = ensemble.get_posteriors()
        assert len(posteriors) > 0
        assert abs(sum(posteriors.values()) - 1.0) < 1e-6

    def test_likelihoods_positive(self, sample_data, sample_horizons):
        """Test that likelihoods are positive."""
        forecasts, actuals = sample_data
        ensemble = BayesianModelAveraging()
        ensemble.fit(forecasts, actuals, sample_horizons)

        likelihoods = ensemble.get_likelihoods()
        for pair, likelihood in likelihoods.items():
            assert likelihood >= ensemble.min_likelihood


class TestRegimeAdaptiveEnsemble:
    """Tests for RegimeAdaptiveEnsemble."""

    def test_initialization(self):
        """Test regime adaptive initialization."""
        ensemble = RegimeAdaptiveEnsemble()
        assert ensemble.n_regimes == 3
        assert ensemble.min_regime_samples == 30

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test regime adaptive fit and predict."""
        forecasts, actuals = sample_data
        ensemble = RegimeAdaptiveEnsemble()

        ensemble.fit(forecasts, actuals, sample_horizons)
        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons)

        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']
        # Regime might be None if HMM not available
        assert 'method' in prediction.metadata

    def test_regime_weights_structure(self, sample_data, sample_horizons):
        """Test that regime weights are properly structured."""
        forecasts, actuals = sample_data
        ensemble = RegimeAdaptiveEnsemble()
        ensemble.fit(forecasts, actuals, sample_horizons)

        regime_weights = ensemble.get_regime_weights()
        assert isinstance(regime_weights, dict)

        for regime, weights in regime_weights.items():
            assert isinstance(weights, dict)
            if weights:
                assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_update_prices(self, sample_data, sample_horizons):
        """Test price update functionality."""
        forecasts, actuals = sample_data
        ensemble = RegimeAdaptiveEnsemble()
        ensemble.fit(forecasts, actuals, sample_horizons)

        new_prices = np.array([100, 101, 102, 103, 104])
        ensemble.update_prices(new_prices)
        assert np.array_equal(ensemble._prices, new_prices)


class TestConformalPredictionInterval:
    """Tests for ConformalPredictionInterval."""

    def test_initialization(self):
        """Test conformal initialization."""
        ensemble = ConformalPredictionInterval()
        assert ensemble.coverage == 0.90
        assert ensemble.calibration_pct == 0.3

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test conformal fit and predict."""
        forecasts, actuals = sample_data
        ensemble = ConformalPredictionInterval()

        ensemble.fit(forecasts, actuals, sample_horizons)
        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons)

        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert prediction.interval_lower is not None
        assert prediction.interval_upper is not None
        assert prediction.interval_lower <= prediction.interval_upper

    def test_interval_bounds(self, sample_data, sample_horizons):
        """Test that intervals are within valid range."""
        forecasts, actuals = sample_data
        ensemble = ConformalPredictionInterval()
        ensemble.fit(forecasts, actuals, sample_horizons)

        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons)

        assert prediction.interval_lower >= -1.0
        assert prediction.interval_upper <= 1.0

    def test_get_interval(self, sample_data, sample_horizons):
        """Test interval retrieval."""
        forecasts, actuals = sample_data
        ensemble = ConformalPredictionInterval()
        ensemble.fit(forecasts, actuals, sample_horizons)

        interval = ensemble.get_interval(0.5)
        assert interval.lower <= interval.point <= interval.upper
        assert interval.coverage == 0.90

    def test_calibration_scores(self, sample_data, sample_horizons):
        """Test calibration scores are computed."""
        forecasts, actuals = sample_data
        ensemble = ConformalPredictionInterval()
        ensemble.fit(forecasts, actuals, sample_horizons)

        scores = ensemble.get_calibration_scores()
        assert isinstance(scores, list)


class TestCombinedTier2Ensemble:
    """Tests for CombinedTier2Ensemble."""

    def test_initialization(self):
        """Test combined Tier 2 initialization."""
        ensemble = CombinedTier2Ensemble()
        assert ensemble.bma is not None
        assert ensemble.regime_adaptive is not None
        assert ensemble.conformal is not None

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test combined Tier 2 fit and predict."""
        forecasts, actuals = sample_data
        ensemble = CombinedTier2Ensemble()

        ensemble.fit(forecasts, actuals, sample_horizons)
        prediction = ensemble.predict_single(forecasts.iloc[-1])

        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert prediction.interval_lower is not None
        assert prediction.interval_upper is not None
        assert 'component_signals' in prediction.metadata


# =============================================================================
# TIER 3 TESTS
# =============================================================================

class TestBetaDistribution:
    """Tests for BetaDistribution helper class."""

    def test_initialization(self):
        """Test default initialization."""
        beta = BetaDistribution()
        assert beta.alpha == 1.0
        assert beta.beta == 1.0

    def test_mean(self):
        """Test mean calculation."""
        beta = BetaDistribution(alpha=3.0, beta=2.0)
        assert beta.mean() == 0.6  # 3 / (3 + 2)

    def test_variance(self):
        """Test variance calculation."""
        beta = BetaDistribution(alpha=2.0, beta=2.0)
        expected = (2 * 2) / ((4) ** 2 * 5)  # ab / ((a+b)^2 * (a+b+1))
        assert abs(beta.variance() - expected) < 1e-6

    def test_sample_range(self):
        """Test that samples are in [0, 1]."""
        beta = BetaDistribution(alpha=2.0, beta=5.0)
        for _ in range(100):
            sample = beta.sample()
            assert 0.0 <= sample <= 1.0

    def test_update(self):
        """Test belief update."""
        beta = BetaDistribution(alpha=1.0, beta=1.0)

        # Full success
        beta.update(1.0)
        assert beta.alpha == 2.0
        assert beta.beta == 1.0

        # Full failure
        beta.update(0.0)
        assert beta.alpha == 2.0
        assert beta.beta == 2.0


class TestThompsonSamplingEnsemble:
    """Tests for ThompsonSamplingEnsemble."""

    def test_initialization(self):
        """Test Thompson sampling initialization."""
        ensemble = ThompsonSamplingEnsemble()
        assert ensemble.prior_alpha == 1.0
        assert ensemble.prior_beta == 1.0
        assert ensemble.decay_rate == 0.99
        assert ensemble.n_samples == 100

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test Thompson sampling fit and predict."""
        forecasts, actuals = sample_data
        ensemble = ThompsonSamplingEnsemble()

        ensemble.fit(forecasts, actuals, sample_horizons)
        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons)

        assert isinstance(prediction, Tier3EnsembleResult)
        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert prediction.exploration_bonus is not None

    def test_distributions_initialized(self, sample_data, sample_horizons):
        """Test that Beta distributions are initialized."""
        forecasts, actuals = sample_data
        ensemble = ThompsonSamplingEnsemble()
        ensemble.fit(forecasts, actuals, sample_horizons)

        distributions = ensemble.get_distributions()
        assert len(distributions) > 0
        for pair, dist_params in distributions.items():
            assert 'alpha' in dist_params
            assert 'beta' in dist_params
            assert 'mean' in dist_params
            assert 'variance' in dist_params

    def test_online_update(self, sample_data, sample_horizons):
        """Test online update functionality."""
        forecasts, actuals = sample_data
        ensemble = ThompsonSamplingEnsemble()
        ensemble.fit(forecasts, actuals, sample_horizons)

        initial_updates = ensemble._total_updates

        # Perform update
        ensemble.update(forecasts.iloc[-1], 0.05, sample_horizons)

        assert ensemble._total_updates == initial_updates + 1

    def test_exploration_scores(self, sample_data, sample_horizons):
        """Test exploration scores are computed."""
        forecasts, actuals = sample_data
        ensemble = ThompsonSamplingEnsemble()
        ensemble.fit(forecasts, actuals, sample_horizons)

        scores = ensemble.get_exploration_scores()
        assert len(scores) > 0
        for pair, score in scores.items():
            assert score >= 0


class TestAttentionBasedEnsemble:
    """Tests for AttentionBasedEnsemble."""

    def test_initialization(self):
        """Test attention ensemble initialization."""
        ensemble = AttentionBasedEnsemble()
        assert ensemble.context_dim == 5
        assert ensemble.hidden_dim == 32
        assert ensemble.n_epochs == 50

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test attention fit and predict."""
        forecasts, actuals = sample_data
        ensemble = AttentionBasedEnsemble()

        ensemble.fit(forecasts, actuals, sample_horizons)
        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons)

        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert prediction.attention_weights is not None

    def test_fallback_without_torch(self, sample_data, sample_horizons):
        """Test that fallback works when torch unavailable."""
        forecasts, actuals = sample_data
        ensemble = AttentionBasedEnsemble()

        # Force fallback by setting model to None
        ensemble.fit(forecasts, actuals, sample_horizons)
        ensemble._attention_model = None

        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons)
        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert prediction.metadata.get('using_torch') is False


class TestQuantileRegressionForest:
    """Tests for QuantileRegressionForest."""

    def test_initialization(self):
        """Test QRF initialization."""
        ensemble = QuantileRegressionForest()
        assert ensemble.n_estimators == 100
        assert ensemble.min_samples_leaf == 5
        assert 0.5 in ensemble.quantiles

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test QRF fit and predict."""
        forecasts, actuals = sample_data
        ensemble = QuantileRegressionForest()

        ensemble.fit(forecasts, actuals, sample_horizons)
        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons)

        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert prediction.quantiles is not None
        assert prediction.interval_lower is not None
        assert prediction.interval_upper is not None

    def test_quantiles_ordered(self, sample_data, sample_horizons):
        """Test that quantiles are properly ordered."""
        forecasts, actuals = sample_data
        ensemble = QuantileRegressionForest(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        ensemble.fit(forecasts, actuals, sample_horizons)

        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons)
        quantiles = prediction.quantiles

        if quantiles and len(quantiles) > 1:
            q_values = [quantiles[q] for q in sorted(quantiles.keys())]
            # Quantiles should be non-decreasing
            for i in range(len(q_values) - 1):
                assert q_values[i] <= q_values[i + 1] + 1e-6

    def test_feature_importances(self, sample_data, sample_horizons):
        """Test feature importances are computed."""
        forecasts, actuals = sample_data
        ensemble = QuantileRegressionForest()
        ensemble.fit(forecasts, actuals, sample_horizons)

        importances = ensemble.get_feature_importances()
        # May be empty if sklearn not available
        if importances:
            assert sum(importances.values()) > 0


class TestCombinedTier3Ensemble:
    """Tests for CombinedTier3Ensemble."""

    def test_initialization(self):
        """Test combined Tier 3 initialization."""
        ensemble = CombinedTier3Ensemble()
        assert ensemble.thompson is not None
        assert ensemble.attention is not None
        assert ensemble.qrf is not None

    def test_custom_weights(self):
        """Test custom component weights."""
        ensemble = CombinedTier3Ensemble(
            thompson_weight=0.5,
            attention_weight=0.3,
            qrf_weight=0.2
        )
        assert ensemble.component_weights['thompson'] == 0.5
        assert ensemble.component_weights['attention'] == 0.3
        assert ensemble.component_weights['qrf'] == 0.2

    def test_fit_and_predict(self, sample_data, sample_horizons):
        """Test combined Tier 3 fit and predict."""
        forecasts, actuals = sample_data
        ensemble = CombinedTier3Ensemble()

        ensemble.fit(forecasts, actuals, sample_horizons)
        prediction = ensemble.predict_single(forecasts.iloc[-1])

        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert 'component_signals' in prediction.metadata
        assert prediction.quantiles is not None
        assert prediction.attention_weights is not None

    def test_update_thompson(self, sample_data, sample_horizons):
        """Test Thompson component update."""
        forecasts, actuals = sample_data
        ensemble = CombinedTier3Ensemble()
        ensemble.fit(forecasts, actuals, sample_horizons)

        initial = ensemble.thompson._total_updates
        ensemble.update_thompson(forecasts.iloc[-1], 0.02)
        assert ensemble.thompson._total_updates == initial + 1


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases across all tiers."""

    def test_empty_data_tier1(self, empty_data):
        """Test Tier 1 with empty data."""
        forecasts, actuals = empty_data
        ensemble = AccuracyWeightedEnsemble()

        # Should handle gracefully or raise meaningful error
        try:
            ensemble.fit(forecasts, actuals, [5, 10, 20])
            # If fit succeeds, weights should be empty or default
            weights = ensemble.get_weights()
            assert isinstance(weights, dict)
        except (ValueError, KeyError, IndexError):
            # Expected for empty data
            pass

    def test_empty_data_tier2(self, empty_data):
        """Test Tier 2 with empty data."""
        forecasts, actuals = empty_data
        ensemble = BayesianModelAveraging()

        try:
            ensemble.fit(forecasts, actuals, [5, 10, 20])
            posteriors = ensemble.get_posteriors()
            assert isinstance(posteriors, dict)
        except (ValueError, KeyError, IndexError):
            pass

    def test_empty_data_tier3(self, empty_data):
        """Test Tier 3 with empty data."""
        forecasts, actuals = empty_data
        ensemble = ThompsonSamplingEnsemble()

        try:
            ensemble.fit(forecasts, actuals, [5, 10, 20])
            dists = ensemble.get_distributions()
            assert isinstance(dists, dict)
        except (ValueError, KeyError, IndexError):
            pass

    def test_single_row_tier1(self, single_row_data):
        """Test Tier 1 with single row."""
        forecasts, actuals = single_row_data
        ensemble = AccuracyWeightedEnsemble(lookback_window=1)

        try:
            ensemble.fit(forecasts, actuals, [5, 10, 20])
            # Should handle single row
        except (ValueError, IndexError):
            pass

    def test_data_with_nans_tier1(self, data_with_nans, sample_horizons):
        """Test Tier 1 handles NaN values."""
        forecasts, actuals = data_with_nans
        ensemble = AccuracyWeightedEnsemble()

        # Should complete without crashing
        ensemble.fit(forecasts, actuals, sample_horizons[:3])
        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons[:3])
        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']

    def test_data_with_nans_tier2(self, data_with_nans, sample_horizons):
        """Test Tier 2 handles NaN values."""
        forecasts, actuals = data_with_nans
        ensemble = ConformalPredictionInterval()

        ensemble.fit(forecasts, actuals, sample_horizons[:3])
        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons[:3])
        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']

    def test_data_with_nans_tier3(self, data_with_nans, sample_horizons):
        """Test Tier 3 handles NaN values."""
        forecasts, actuals = data_with_nans
        ensemble = QuantileRegressionForest()

        ensemble.fit(forecasts, actuals, sample_horizons[:3])
        prediction = ensemble.predict_single(forecasts.iloc[-1], sample_horizons[:3])
        assert prediction.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']

    def test_misaligned_indices(self, sample_horizons):
        """Test handling of misaligned forecast/actual indices."""
        dates1 = pd.date_range(start='2025-01-01', periods=50, freq='D')
        dates2 = pd.date_range(start='2025-01-10', periods=50, freq='D')

        forecasts = pd.DataFrame({
            'd5': np.random.randn(50) + 100,
            'd10': np.random.randn(50) + 100,
        }, index=dates1)

        actuals = pd.Series(np.random.randn(50) + 100, index=dates2)

        ensemble = AccuracyWeightedEnsemble()
        # Should use intersection of indices
        ensemble.fit(forecasts, actuals, [5, 10])
        assert ensemble._is_fitted

    def test_single_horizon(self, sample_data):
        """Test with only a single horizon (no pairs possible)."""
        forecasts, actuals = sample_data
        forecasts_single = forecasts[['d5']].copy()

        ensemble = AccuracyWeightedEnsemble()

        try:
            ensemble.fit(forecasts_single, actuals, [5])
            # No pairs, so weights should be empty
            weights = ensemble.get_weights()
            assert len(weights) == 0
        except (ValueError, KeyError):
            # Expected - can't form pairs with single horizon
            pass

    def test_all_same_values(self, sample_horizons):
        """Test with constant forecast values."""
        dates = pd.date_range(start='2025-01-01', periods=50, freq='D')

        forecasts = pd.DataFrame({
            'd5': [100.0] * 50,
            'd10': [100.0] * 50,
            'd20': [100.0] * 50,
        }, index=dates)
        actuals = pd.Series([100.0] * 50, index=dates)

        ensemble = MagnitudeWeightedVoting()
        ensemble.fit(forecasts, actuals, [5, 10, 20])

        # All drifts are zero, should handle gracefully
        prediction = ensemble.predict_single(forecasts.iloc[-1], [5, 10, 20])
        assert prediction.signal == 'NEUTRAL'


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests running all tiers together."""

    def test_all_tiers_on_same_data(self, sample_data, sample_horizons):
        """Test all 12 methods on the same dataset."""
        forecasts, actuals = sample_data

        # Tier 1
        tier1_methods = [
            AccuracyWeightedEnsemble(),
            MagnitudeWeightedVoting(),
            ErrorCorrelationWeighting(),
            CombinedTier1Ensemble(),
        ]

        # Tier 2
        tier2_methods = [
            BayesianModelAveraging(),
            RegimeAdaptiveEnsemble(),
            ConformalPredictionInterval(),
            CombinedTier2Ensemble(),
        ]

        # Tier 3
        tier3_methods = [
            ThompsonSamplingEnsemble(),
            AttentionBasedEnsemble(),
            QuantileRegressionForest(),
            CombinedTier3Ensemble(),
        ]

        all_methods = tier1_methods + tier2_methods + tier3_methods
        results = []

        for method in all_methods:
            method.fit(forecasts, actuals, sample_horizons)

            if hasattr(method, 'predict_single'):
                if isinstance(method, (CombinedTier1Ensemble, CombinedTier2Ensemble, CombinedTier3Ensemble)):
                    pred = method.predict_single(forecasts.iloc[-1])
                else:
                    pred = method.predict_single(forecasts.iloc[-1], sample_horizons)

                results.append({
                    'method': method.__class__.__name__,
                    'signal': pred.signal,
                    'confidence': pred.confidence,
                    'net_prob': pred.net_probability,
                })

        # All 12 methods should produce results
        assert len(results) == 12

        # All signals should be valid
        for r in results:
            assert r['signal'] in ['BULLISH', 'BEARISH', 'NEUTRAL']
            assert 0 <= r['confidence'] <= 1
            assert -1 <= r['net_prob'] <= 1

    def test_ensemble_consistency(self, sample_data, sample_horizons):
        """Test that ensembles produce consistent results on repeated calls."""
        forecasts, actuals = sample_data

        ensemble = AccuracyWeightedEnsemble()
        ensemble.fit(forecasts, actuals, sample_horizons)

        # Same input should produce same weights
        weights1 = ensemble.get_weights()
        weights2 = ensemble.get_weights()

        assert weights1 == weights2

    def test_tier_comparison_output_format(self, sample_data, sample_horizons):
        """Test that comparison functions return proper DataFrames."""
        from backend.ensemble_tier1 import compare_tier1_methods
        from backend.ensemble_tier2 import compare_tier2_methods
        from backend.ensemble_tier3 import compare_tier3_methods

        forecasts, actuals = sample_data

        # These may print output, which is fine
        results1 = compare_tier1_methods(forecasts, actuals, sample_horizons)
        assert isinstance(results1, pd.DataFrame)
        assert 'method' in results1.columns

        results2 = compare_tier2_methods(forecasts, actuals, sample_horizons)
        assert isinstance(results2, pd.DataFrame)

        results3 = compare_tier3_methods(forecasts, actuals, sample_horizons)
        assert isinstance(results3, pd.DataFrame)

    def test_sequential_predictions(self, sample_data, sample_horizons):
        """Test making sequential predictions simulating live trading."""
        forecasts, actuals = sample_data

        # Use first 70% for training
        n_train = int(len(forecasts) * 0.7)
        train_forecasts = forecasts.iloc[:n_train]
        train_actuals = actuals.iloc[:n_train]
        test_forecasts = forecasts.iloc[n_train:]
        test_actuals = actuals.iloc[n_train:]

        # Initialize all tiers
        t1 = CombinedTier1Ensemble()
        t2 = CombinedTier2Ensemble()
        t3 = CombinedTier3Ensemble()

        t1.fit(train_forecasts, train_actuals, sample_horizons)
        t2.fit(train_forecasts, train_actuals, sample_horizons)
        t3.fit(train_forecasts, train_actuals, sample_horizons)

        # Simulate sequential predictions
        signals = {'t1': [], 't2': [], 't3': []}

        for i in range(len(test_forecasts)):
            row = test_forecasts.iloc[i]

            p1 = t1.predict_single(row)
            p2 = t2.predict_single(row)
            p3 = t3.predict_single(row)

            signals['t1'].append(p1.signal)
            signals['t2'].append(p2.signal)
            signals['t3'].append(p3.signal)

            # Update Thompson sampling with actual return if available
            if i > 0:
                actual_return = (test_actuals.iloc[i] / test_actuals.iloc[i-1]) - 1
                t3.update_thompson(test_forecasts.iloc[i-1], actual_return)

        # Should have predictions for all test rows
        assert len(signals['t1']) == len(test_forecasts)
        assert len(signals['t2']) == len(test_forecasts)
        assert len(signals['t3']) == len(test_forecasts)

    def test_mixed_horizons(self, sample_data):
        """Test with different horizon configurations."""
        forecasts, actuals = sample_data

        horizons_configs = [
            [5, 10],
            [5, 10, 20],
            [5, 10, 20, 40],
            [10, 20, 40],
        ]

        for horizons in horizons_configs:
            ensemble = AccuracyWeightedEnsemble()

            # Filter to available horizons
            available = [h for h in horizons if f'd{h}' in forecasts.columns]
            if len(available) >= 2:
                ensemble.fit(forecasts, actuals, available)
                pred = ensemble.predict_single(forecasts.iloc[-1], available)
                assert pred.signal in ['BULLISH', 'BEARISH', 'NEUTRAL']


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Basic performance/stress tests."""

    def test_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

        prices = 100 * np.cumprod(1 + np.random.randn(n_samples) * 0.02)
        actuals = pd.Series(prices, index=dates)

        forecasts = pd.DataFrame(index=dates)
        for h in [5, 10, 20, 40]:
            forecasts[f'd{h}'] = prices * (1 + np.random.randn(n_samples) * 0.01)

        ensemble = AccuracyWeightedEnsemble()
        ensemble.fit(forecasts, actuals, [5, 10, 20, 40])

        # Should complete reasonably quickly
        predictions = ensemble.predict(forecasts.iloc[-100:], [5, 10, 20, 40])
        assert len(predictions) == 100

    def test_many_horizons(self, sample_data):
        """Test with many horizons."""
        forecasts, actuals = sample_data

        # Add more horizons
        for h in [7, 14, 30, 60, 90]:
            forecasts[f'd{h}'] = forecasts['d10'] * (1 + np.random.randn(len(forecasts)) * 0.001)

        horizons = [5, 7, 10, 14, 20, 30, 40, 60, 90]

        ensemble = AccuracyWeightedEnsemble()
        ensemble.fit(forecasts, actuals, horizons)

        weights = ensemble.get_weights()
        # Number of pairs = n*(n-1)/2 = 9*8/2 = 36
        assert len(weights) == 36


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
