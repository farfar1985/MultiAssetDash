"""
Unit tests for Backtesting Framework.

Tests for:
1. TransactionCostModel - fixed, percentage, spread, market impact costs
2. WalkForwardValidator - fold splitting, metrics calculation
3. Cost-adjusted returns
4. Integration tests with walk-forward validation

Created: 2026-02-06
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import asdict
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import backtesting modules
from backend.backtesting.costs import (
    TransactionCostModel,
    CostConfig,
    CostBreakdown,
    apply_costs_to_returns,
)

from backend.backtesting.walk_forward import (
    WalkForwardValidator,
    FoldResult,
    MethodComparison,
)

from backend.ensemble_tier1 import EnsembleResult


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_cost_config():
    """Sample cost configuration."""
    return CostConfig(
        fixed_cost=5.0,
        pct_cost=0.001,
        spread_cost=0.0002,
        market_impact_coef=0.1,
    )


@pytest.fixture
def sample_forecasts():
    """Generate sample forecast data."""
    np.random.seed(42)
    n_samples = 200
    dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')

    # Simulated price series
    base_price = 100
    returns = np.random.randn(n_samples) * 0.02
    prices = base_price * np.cumprod(1 + returns)

    # Forecasts for different horizons
    forecasts = pd.DataFrame(index=dates)
    for h in [5, 10, 20]:
        noise = np.random.randn(n_samples) * 0.01
        forecasts[f'd{h}'] = prices * (1 + noise + h * 0.001)

    actuals = pd.Series(prices, index=dates)

    return forecasts, actuals, [5, 10, 20]


@pytest.fixture
def small_forecasts():
    """Small dataset for quick tests."""
    np.random.seed(42)
    n_samples = 50
    dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')

    prices = 100 + np.cumsum(np.random.randn(n_samples))
    actuals = pd.Series(prices, index=dates)

    forecasts = pd.DataFrame(index=dates)
    for h in [5, 10]:
        forecasts[f'd{h}'] = prices * (1 + np.random.randn(n_samples) * 0.005)

    return forecasts, actuals, [5, 10]


# =============================================================================
# COST CONFIG TESTS
# =============================================================================

class TestCostConfig:
    """Tests for CostConfig dataclass."""

    def test_default_initialization(self):
        """Test default config values."""
        config = CostConfig()
        assert config.fixed_cost == 0.0
        assert config.pct_cost == 0.0
        assert config.spread_cost == 0.0
        assert config.market_impact_coef == 0.0
        assert config.market_impact_exp == 0.5
        assert config.avg_daily_volume == 1e9

    def test_custom_initialization(self):
        """Test custom config values."""
        config = CostConfig(
            fixed_cost=10.0,
            pct_cost=0.002,
            spread_cost=0.0005,
            market_impact_coef=0.15,
            market_impact_exp=0.6,
        )
        assert config.fixed_cost == 10.0
        assert config.pct_cost == 0.002
        assert config.spread_cost == 0.0005
        assert config.market_impact_coef == 0.15
        assert config.market_impact_exp == 0.6

    def test_to_dict(self, sample_cost_config):
        """Test conversion to dictionary."""
        d = sample_cost_config.to_dict()
        assert isinstance(d, dict)
        assert 'fixed_cost' in d
        assert 'pct_cost' in d
        assert d['fixed_cost'] == 5.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {'fixed_cost': 7.0, 'pct_cost': 0.0015}
        config = CostConfig.from_dict(d)
        assert config.fixed_cost == 7.0
        assert config.pct_cost == 0.0015

    def test_from_string_basic(self):
        """Test parsing from CLI string."""
        config = CostConfig.from_string("fixed=5 pct=0.001")
        assert config.fixed_cost == 5.0
        assert config.pct_cost == 0.001

    def test_from_string_complex(self):
        """Test parsing complex CLI string."""
        config = CostConfig.from_string("fixed=10 pct=0.002 spread=0.0003 impact=0.1 exp=0.4")
        assert config.fixed_cost == 10.0
        assert config.pct_cost == 0.002
        assert config.spread_cost == 0.0003
        assert config.market_impact_coef == 0.1
        assert config.market_impact_exp == 0.4

    def test_from_string_empty(self):
        """Test parsing empty string."""
        config = CostConfig.from_string("")
        assert config.fixed_cost == 0.0
        assert config.pct_cost == 0.0

    def test_from_string_slippage_alias(self):
        """Test slippage alias in string parsing."""
        config = CostConfig.from_string("slip=0.0005")
        assert config.slippage == 0.0005

        config2 = CostConfig.from_string("slippage=0.0003")
        assert config2.slippage == 0.0003


# =============================================================================
# COST BREAKDOWN TESTS
# =============================================================================

class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    def test_default_values(self):
        """Test default breakdown values."""
        breakdown = CostBreakdown()
        assert breakdown.fixed == 0.0
        assert breakdown.percentage == 0.0
        assert breakdown.spread == 0.0
        assert breakdown.market_impact == 0.0
        assert breakdown.total == 0.0
        assert breakdown.total_bps == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        breakdown = CostBreakdown(
            fixed=5.0,
            percentage=10.0,
            spread=2.0,
            market_impact=3.0,
            total=20.0,
            total_bps=20.0,
        )
        d = breakdown.to_dict()
        assert d['fixed'] == 5.0
        assert d['total'] == 20.0


# =============================================================================
# TRANSACTION COST MODEL TESTS
# =============================================================================

class TestTransactionCostModel:
    """Tests for TransactionCostModel."""

    def test_default_initialization(self):
        """Test default model has zero costs."""
        model = TransactionCostModel()
        breakdown = model.calculate_cost(price=100, trade_value=10000)
        assert breakdown.total == 0.0

    def test_fixed_cost_only(self):
        """Test model with only fixed cost."""
        model = TransactionCostModel(fixed_cost=5.0)

        # Round-trip should double the fixed cost
        breakdown = model.calculate_cost(price=100, trade_value=10000, is_round_trip=True)
        assert breakdown.fixed == 10.0  # 5 * 2
        assert breakdown.total == 10.0

        # Single leg should be just the fixed cost
        breakdown_single = model.calculate_cost(price=100, trade_value=10000, is_round_trip=False)
        assert breakdown_single.fixed == 5.0
        assert breakdown_single.total == 5.0

    def test_percentage_cost_only(self):
        """Test model with only percentage cost."""
        model = TransactionCostModel(pct_cost=0.001)  # 10 bps

        breakdown = model.calculate_cost(price=100, trade_value=10000, is_round_trip=True)
        # 10000 * 0.001 * 2 = 20
        assert breakdown.percentage == 20.0
        assert breakdown.total == 20.0
        assert breakdown.total_bps == 20.0  # 20 / 10000 * 10000

    def test_spread_cost_only(self):
        """Test model with only spread cost."""
        model = TransactionCostModel(spread_cost=0.0002)  # 2 bps half-spread

        breakdown = model.calculate_cost(price=100, trade_value=10000, is_round_trip=True)
        # 10000 * 0.0002 * 2 = 4
        assert breakdown.spread == 4.0
        assert breakdown.total == 4.0

    def test_combined_costs(self):
        """Test model with multiple cost components."""
        model = TransactionCostModel(
            fixed_cost=5.0,
            pct_cost=0.001,
            spread_cost=0.0002,
        )

        breakdown = model.calculate_cost(price=100, trade_value=10000, is_round_trip=True)

        assert breakdown.fixed == 10.0  # 5 * 2
        assert breakdown.percentage == 20.0  # 10000 * 0.001 * 2
        assert breakdown.spread == 4.0  # 10000 * 0.0002 * 2
        assert breakdown.total == 34.0
        assert breakdown.total_bps == 34.0

    def test_market_impact(self):
        """Test market impact calculation."""
        model = TransactionCostModel(market_impact_coef=0.1)

        # With volume, should have market impact
        breakdown = model.calculate_cost(
            price=100,
            trade_value=100000,
            volume=1000000,
            is_round_trip=True,
        )

        # Impact = 0.1 * (100000/1000000)^0.5 * 100000 * 2
        # = 0.1 * 0.316 * 100000 * 2 = 6324 approx
        assert breakdown.market_impact > 0

        # Without volume, no market impact
        breakdown_no_vol = model.calculate_cost(
            price=100,
            trade_value=100000,
            is_round_trip=True,
        )
        assert breakdown_no_vol.market_impact == 0.0

    def test_from_preset_zero(self):
        """Test zero cost preset."""
        model = TransactionCostModel.from_preset('zero')
        breakdown = model.calculate_cost(price=100, trade_value=10000)
        assert breakdown.total == 0.0

    def test_from_preset_low(self):
        """Test low cost preset."""
        model = TransactionCostModel.from_preset('low')
        breakdown = model.calculate_cost(price=100, trade_value=10000, is_round_trip=True)
        # low = 5 bps pct + 1 bp spread = 6 bps * 2 = 12 bps
        assert breakdown.total > 0

    def test_from_preset_invalid(self):
        """Test invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            TransactionCostModel.from_preset('invalid_preset')

    def test_from_bps(self):
        """Test creating model from basis points."""
        model = TransactionCostModel.from_bps(10.0)  # 10 bps

        breakdown = model.calculate_cost(price=100, trade_value=10000, is_round_trip=True)
        # 10 bps = 0.001, * 10000 * 2 = 20
        assert breakdown.total == 20.0
        assert breakdown.total_bps == 20.0

    def test_from_string(self):
        """Test creating model from CLI string."""
        model = TransactionCostModel.from_string("fixed=5 pct=0.001")

        breakdown = model.calculate_cost(price=100, trade_value=10000, is_round_trip=True)
        assert breakdown.fixed == 10.0
        assert breakdown.percentage == 20.0
        assert breakdown.total == 30.0

    def test_cost_as_pct(self):
        """Test cost_as_pct method."""
        model = TransactionCostModel(pct_cost=0.001)

        pct = model.cost_as_pct(price=100, trade_value=10000, is_round_trip=True)
        # 20 / 10000 * 100 = 0.2%
        assert abs(pct - 0.2) < 0.001

    def test_cost_as_bps(self):
        """Test cost_as_bps method."""
        model = TransactionCostModel(pct_cost=0.001)

        bps = model.cost_as_bps(price=100, trade_value=10000, is_round_trip=True)
        assert abs(bps - 20.0) < 0.1

    def test_summary(self):
        """Test human-readable summary."""
        model = TransactionCostModel(fixed_cost=5.0, pct_cost=0.001)
        summary = model.summary()

        assert "$5.00 fixed" in summary
        assert "10.0 bps commission" in summary

    def test_summary_no_costs(self):
        """Test summary with no costs."""
        model = TransactionCostModel()
        assert model.summary() == "No transaction costs"

    def test_repr(self):
        """Test string representation."""
        model = TransactionCostModel(pct_cost=0.001)
        repr_str = repr(model)
        assert "TransactionCostModel" in repr_str

    def test_minimum_cost(self):
        """Test minimum cost enforcement."""
        config = CostConfig(
            min_cost=10.0,
            pct_cost=0.0001,  # Very small percentage cost
        )
        model = TransactionCostModel(config=config)

        # Small trade value with minimum cost
        breakdown = model.calculate_cost(price=100, trade_value=100, is_round_trip=True)
        # pct = 100 * 0.0001 * 2 = 0.02, but min = 10 * 2 = 20
        assert breakdown.total >= 20.0

    def test_estimate_breakeven_edge(self):
        """Test breakeven edge estimation."""
        model = TransactionCostModel(pct_cost=0.001)
        edge = model.estimate_breakeven_edge()

        # Should return cost percentage for typical trade
        assert edge > 0


class TestApplyCostsToReturns:
    """Tests for apply_costs_to_returns function."""

    def test_basic_cost_application(self):
        """Test applying costs to returns."""
        returns = np.array([2.0, 1.5, -0.5, 3.0])
        prices = np.array([100, 101, 102, 103])
        trade_values = np.array([10000, 10000, 10000, 10000])

        model = TransactionCostModel(pct_cost=0.001)  # 10 bps

        adjusted = apply_costs_to_returns(returns, prices, trade_values, model)

        # Each return should be reduced by cost
        # Cost = 0.001 * 10000 * 2 = 20, as pct = 0.2%
        assert len(adjusted) == len(returns)
        for i in range(len(returns)):
            assert adjusted[i] < returns[i]

    def test_zero_cost_model(self):
        """Test that zero cost model doesn't change returns."""
        returns = np.array([2.0, 1.5, -0.5])
        prices = np.array([100, 101, 102])
        trade_values = np.array([10000, 10000, 10000])

        model = TransactionCostModel()  # Zero costs

        adjusted = apply_costs_to_returns(returns, prices, trade_values, model)

        np.testing.assert_array_almost_equal(adjusted, returns)


# =============================================================================
# WALK-FORWARD VALIDATOR TESTS
# =============================================================================

class TestWalkForwardValidator:
    """Tests for WalkForwardValidator."""

    def test_default_initialization(self):
        """Test default validator initialization."""
        validator = WalkForwardValidator()
        assert validator.n_folds == 5
        assert validator.train_pct == 0.7
        assert validator.min_train_samples == 100
        assert validator.holding_period == 5

    def test_custom_initialization(self):
        """Test custom validator initialization."""
        cost_model = TransactionCostModel.from_bps(10)
        validator = WalkForwardValidator(
            n_folds=3,
            train_pct=0.8,
            min_train_samples=50,
            transaction_cost_bps=10.0,
            holding_period=10,
            cost_model=cost_model,
        )
        assert validator.n_folds == 3
        assert validator.train_pct == 0.8
        assert validator.min_train_samples == 50
        assert validator.holding_period == 10
        assert validator.cost_model is cost_model

    def test_load_from_dataframe(self, sample_forecasts):
        """Test loading data from DataFrame."""
        forecasts, actuals, horizons = sample_forecasts

        validator = WalkForwardValidator(verbose=False)
        validator.load_from_dataframe(forecasts, actuals, horizons)

        assert validator._forecasts is not None
        assert validator._actuals is not None
        assert validator._horizons == sorted(horizons)
        assert len(validator._forecasts) == len(forecasts)

    def test_load_from_dataframe_with_regimes(self, sample_forecasts):
        """Test loading with regime data."""
        forecasts, actuals, horizons = sample_forecasts

        # Create regime series
        regimes = pd.Series(['bull'] * 100 + ['bear'] * 100, index=actuals.index)

        validator = WalkForwardValidator(verbose=False)
        validator.load_from_dataframe(forecasts, actuals, horizons, regimes=regimes)

        assert validator._regimes is not None
        assert len(validator._regimes) == len(regimes)

    def test_create_folds(self, sample_forecasts):
        """Test fold creation."""
        forecasts, actuals, horizons = sample_forecasts

        validator = WalkForwardValidator(
            n_folds=3,
            min_train_samples=50,
            verbose=False,
        )
        validator.load_from_dataframe(forecasts, actuals, horizons)

        folds = validator._create_folds()

        assert len(folds) > 0
        for train_idx, test_idx in folds:
            assert len(train_idx) >= validator.min_train_samples
            assert len(test_idx) > 0
            # Train should come before test
            assert train_idx[-1] < test_idx[0]

    def test_calculate_max_drawdown(self, sample_forecasts):
        """Test max drawdown calculation."""
        forecasts, actuals, horizons = sample_forecasts

        validator = WalkForwardValidator(verbose=False)

        # Positive returns: no drawdown
        returns = [1.0, 2.0, 3.0, 4.0]
        dd = validator._calculate_max_drawdown(returns)
        assert dd == 0.0

        # Negative returns create drawdown
        returns = [5.0, -2.0, -3.0, 1.0]
        dd = validator._calculate_max_drawdown(returns)
        assert dd > 0.0
        assert dd == 5.0  # Lost 5% from peak

        # Empty returns
        dd = validator._calculate_max_drawdown([])
        assert dd == 0.0

    def test_calculate_accuracy(self, sample_forecasts):
        """Test accuracy calculation."""
        forecasts, actuals, horizons = sample_forecasts

        validator = WalkForwardValidator(
            holding_period=5,
            verbose=False,
        )
        validator.load_from_dataframe(forecasts, actuals, horizons)

        # Create mock signals
        signals = pd.Series([
            EnsembleResult(
                signal='BULLISH',
                confidence=0.8,
                net_probability=0.5,
                weights={},
            )
            for _ in range(len(actuals))
        ], index=actuals.index)

        acc = validator._calculate_accuracy(signals, actuals, horizon=5)
        # Accuracy should be between 0 and 100
        assert 0 <= acc <= 100

    def test_available_methods(self):
        """Test that all expected methods are available."""
        expected_methods = [
            'tier1_accuracy', 'tier1_magnitude', 'tier1_correlation', 'tier1_combined',
            'tier2_bma', 'tier2_regime', 'tier2_conformal', 'tier2_combined',
            'tier3_thompson', 'tier3_attention', 'tier3_quantile', 'tier3_combined',
        ]

        for method in expected_methods:
            assert method in WalkForwardValidator.AVAILABLE_METHODS

    def test_invalid_method_raises(self, sample_forecasts):
        """Test that invalid method raises error."""
        forecasts, actuals, horizons = sample_forecasts

        validator = WalkForwardValidator(verbose=False)
        validator.load_from_dataframe(forecasts, actuals, horizons)

        with pytest.raises(ValueError, match="Unknown method"):
            validator.run_validation(methods=['invalid_method'])

    def test_run_validation_without_data_raises(self):
        """Test that running without data raises error."""
        validator = WalkForwardValidator(verbose=False)

        with pytest.raises(ValueError, match="Must load data"):
            validator.run_validation()


class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_default_values(self):
        """Test FoldResult with required fields."""
        result = FoldResult(
            fold_id=0,
            train_start='2025-01-01',
            train_end='2025-03-01',
            test_start='2025-03-01',
            test_end='2025-04-01',
            n_train=60,
            n_test=30,
            accuracy=55.0,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=5.0,
            win_rate=52.0,
            total_return=10.0,
            n_trades=15,
            avg_trade_return=0.67,
            avg_holding_days=5.0,
            n_bullish=8,
            n_bearish=5,
            n_neutral=2,
        )

        assert result.fold_id == 0
        assert result.accuracy == 55.0
        assert result.sharpe_ratio == 1.2
        assert result.total_costs == 0.0  # Default

    def test_with_cost_metrics(self):
        """Test FoldResult with cost metrics."""
        result = FoldResult(
            fold_id=1,
            train_start='2025-01-01',
            train_end='2025-03-01',
            test_start='2025-03-01',
            test_end='2025-04-01',
            n_train=60,
            n_test=30,
            accuracy=55.0,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=5.0,
            win_rate=52.0,
            total_return=10.0,
            n_trades=15,
            avg_trade_return=0.67,
            avg_holding_days=5.0,
            n_bullish=8,
            n_bearish=5,
            n_neutral=2,
            total_costs=150.0,
            avg_cost_per_trade=10.0,
            avg_cost_bps=10.0,
            cost_drag_pct=1.5,
        )

        assert result.total_costs == 150.0
        assert result.avg_cost_per_trade == 10.0
        assert result.avg_cost_bps == 10.0
        assert result.cost_drag_pct == 1.5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the backtesting framework."""

    def test_single_method_validation(self, small_forecasts):
        """Test running validation with a single method."""
        forecasts, actuals, horizons = small_forecasts

        validator = WalkForwardValidator(
            n_folds=2,
            min_train_samples=20,
            transaction_cost_bps=5.0,
            verbose=False,
        )
        validator.load_from_dataframe(forecasts, actuals, horizons)

        # Run with single tier 1 method
        results = validator.run_validation(
            methods=['tier1_accuracy'],
            asset_id='test',
            asset_name='Test Asset',
        )

        assert isinstance(results, MethodComparison)
        assert 'tier1_accuracy' in results.method_results
        assert len(results.method_results['tier1_accuracy']) > 0

    def test_multiple_methods_validation(self, small_forecasts):
        """Test running validation with multiple methods."""
        forecasts, actuals, horizons = small_forecasts

        validator = WalkForwardValidator(
            n_folds=2,
            min_train_samples=20,
            verbose=False,
        )
        validator.load_from_dataframe(forecasts, actuals, horizons)

        methods = ['tier1_accuracy', 'tier1_magnitude']
        results = validator.run_validation(methods=methods)

        for method in methods:
            assert method in results.method_results

    def test_cost_model_integration(self, small_forecasts):
        """Test that cost model is properly integrated."""
        forecasts, actuals, horizons = small_forecasts

        # Create custom cost model
        cost_model = TransactionCostModel(
            fixed_cost=5.0,
            pct_cost=0.001,
        )

        validator = WalkForwardValidator(
            n_folds=2,
            min_train_samples=20,
            cost_model=cost_model,
            verbose=False,
        )
        validator.load_from_dataframe(forecasts, actuals, horizons)

        results = validator.run_validation(methods=['tier1_accuracy'])

        # Check that cost metrics are populated
        fold_results = results.method_results['tier1_accuracy']
        for fold in fold_results:
            # If there were trades, should have costs
            if fold.n_trades > 0:
                assert fold.avg_cost_bps >= 0

    def test_walk_forward_with_presets(self, small_forecasts):
        """Test walk-forward with different cost presets."""
        forecasts, actuals, horizons = small_forecasts

        presets = ['zero', 'low', 'medium']
        results_by_preset = {}

        for preset in presets:
            cost_model = TransactionCostModel.from_preset(preset)
            validator = WalkForwardValidator(
                n_folds=2,
                min_train_samples=20,
                cost_model=cost_model,
                verbose=False,
            )
            validator.load_from_dataframe(forecasts, actuals, horizons)

            results = validator.run_validation(methods=['tier1_accuracy'])
            results_by_preset[preset] = results

        # All presets should produce results
        for preset, results in results_by_preset.items():
            assert 'tier1_accuracy' in results.method_results

    def test_fold_metrics_are_calculated(self, small_forecasts):
        """Test that all fold metrics are properly calculated."""
        forecasts, actuals, horizons = small_forecasts

        validator = WalkForwardValidator(
            n_folds=2,
            min_train_samples=20,
            verbose=False,
        )
        validator.load_from_dataframe(forecasts, actuals, horizons)

        results = validator.run_validation(methods=['tier1_accuracy'])

        for fold in results.method_results['tier1_accuracy']:
            # Check required metrics are present
            assert hasattr(fold, 'accuracy')
            assert hasattr(fold, 'sharpe_ratio')
            assert hasattr(fold, 'max_drawdown')
            assert hasattr(fold, 'win_rate')
            assert hasattr(fold, 'total_return')
            assert hasattr(fold, 'n_trades')

            # Check date ranges
            assert fold.train_start is not None
            assert fold.train_end is not None
            assert fold.test_start is not None
            assert fold.test_end is not None

    def test_regime_performance(self, sample_forecasts):
        """Test regime-specific performance tracking."""
        forecasts, actuals, horizons = sample_forecasts

        # Create regime labels
        n = len(actuals)
        regimes = pd.Series(
            ['bull'] * (n // 2) + ['bear'] * (n - n // 2),
            index=actuals.index
        )

        validator = WalkForwardValidator(
            n_folds=2,
            min_train_samples=50,
            verbose=False,
        )
        validator.load_from_dataframe(
            forecasts, actuals, horizons,
            regimes=regimes
        )

        results = validator.run_validation(methods=['tier1_accuracy'])

        # Check regime performance is captured
        for fold in results.method_results['tier1_accuracy']:
            # regime_performance should be a dict (may be empty if no signals in regime)
            assert isinstance(fold.regime_performance, dict)


class TestEdgeCases:
    """Edge case tests for backtesting."""

    def test_zero_trade_value(self):
        """Test cost calculation with zero trade value."""
        model = TransactionCostModel(pct_cost=0.001)

        breakdown = model.calculate_cost(price=100, trade_value=0)
        assert breakdown.total == 0.0
        assert breakdown.total_bps == 0.0

    def test_very_large_trade(self):
        """Test cost calculation with very large trade."""
        model = TransactionCostModel(
            fixed_cost=5.0,
            pct_cost=0.001,
            market_impact_coef=0.1,
        )

        # $100M trade
        breakdown = model.calculate_cost(
            price=100,
            trade_value=100_000_000,
            volume=50_000_000,
            is_round_trip=True,
        )

        assert breakdown.total > 0
        assert breakdown.market_impact > 0

    def test_empty_returns_drawdown(self):
        """Test drawdown with empty returns."""
        validator = WalkForwardValidator(verbose=False)
        dd = validator._calculate_max_drawdown([])
        assert dd == 0.0

    def test_single_return_drawdown(self):
        """Test drawdown with single return."""
        validator = WalkForwardValidator(verbose=False)

        dd_pos = validator._calculate_max_drawdown([5.0])
        assert dd_pos == 0.0  # No drawdown from single positive

        dd_neg = validator._calculate_max_drawdown([-5.0])
        assert dd_neg >= 0.0

    def test_all_neutral_signals(self, small_forecasts):
        """Test handling of all neutral signals."""
        forecasts, actuals, horizons = small_forecasts

        validator = WalkForwardValidator(verbose=False)
        validator.load_from_dataframe(forecasts, actuals, horizons)

        # Create all neutral signals
        signals = pd.Series([
            EnsembleResult(
                signal='NEUTRAL',
                confidence=0.5,
                net_probability=0.0,
                weights={},
            )
            for _ in range(len(actuals))
        ], index=actuals.index)

        acc = validator._calculate_accuracy(signals, actuals, horizon=5)
        # With all neutral signals, accuracy calculation should handle gracefully
        assert acc == 0.0  # No non-neutral signals to evaluate

    def test_minimum_fold_size(self, small_forecasts):
        """Test with minimum possible data for folds."""
        forecasts, actuals, horizons = small_forecasts

        # Use very aggressive settings
        validator = WalkForwardValidator(
            n_folds=10,  # Many folds
            min_train_samples=10,  # Low minimum
            verbose=False,
        )
        validator.load_from_dataframe(forecasts, actuals, horizons)

        folds = validator._create_folds()
        # Should create at least some folds
        assert len(folds) >= 1


class TestCostPresets:
    """Test all available cost presets."""

    @pytest.mark.parametrize("preset", [
        'zero', 'low', 'medium', 'high', 'futures', 'forex', 'crypto', 'commodities'
    ])
    def test_preset_creates_valid_model(self, preset):
        """Test each preset creates a valid model."""
        model = TransactionCostModel.from_preset(preset)
        assert isinstance(model, TransactionCostModel)

        # Should be able to calculate costs
        breakdown = model.calculate_cost(price=100, trade_value=10000)
        assert isinstance(breakdown, CostBreakdown)
        assert breakdown.total >= 0

    def test_preset_costs_ordering(self):
        """Test that preset costs are ordered as expected."""
        zero = TransactionCostModel.from_preset('zero')
        low = TransactionCostModel.from_preset('low')
        medium = TransactionCostModel.from_preset('medium')
        high = TransactionCostModel.from_preset('high')

        trade_value = 10000
        price = 100

        cost_zero = zero.calculate_cost(price, trade_value).total
        cost_low = low.calculate_cost(price, trade_value).total
        cost_medium = medium.calculate_cost(price, trade_value).total
        cost_high = high.calculate_cost(price, trade_value).total

        assert cost_zero <= cost_low
        assert cost_low <= cost_medium
        assert cost_medium <= cost_high


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
