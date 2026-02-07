"""
Walk-Forward Validation Framework for Ensemble Methods
=======================================================

This module provides a comprehensive walk-forward validation framework
for comparing Tier 1, 2, and 3 ensemble methods on commodity forecasts.

Features:
- Rolling train/test window splits
- Per-fold metrics: accuracy, Sharpe, max drawdown, win rate
- Cross-method comparison on identical data splits
- Per-regime performance analysis
- Bootstrap significance tests for method comparison

Usage:
    python -m backend.backtesting.walk_forward --asset crude_oil --folds 5

Created: 2026-02-06
Author: AmiraB
Reference: docs/ENSEMBLE_METHODS_PLAN.md
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
)

from backend.backtesting.costs import (
    TransactionCostModel,
    CostConfig,
    CostBreakdown,
)

# Import ensemble methods
from backend.ensemble_tier1 import (
    AccuracyWeightedEnsemble,
    MagnitudeWeightedVoting,
    ErrorCorrelationWeighting,
    CombinedTier1Ensemble,
    EnsembleResult,
)
from backend.ensemble_tier2 import (
    BayesianModelAveraging,
    RegimeAdaptiveEnsemble,
    ConformalPredictionInterval,
    CombinedTier2Ensemble,
)
from backend.ensemble_tier3 import (
    ThompsonSamplingEnsemble,
    AttentionBasedEnsemble,
    QuantileRegressionForest,
    CombinedTier3Ensemble,
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int

    # Core metrics
    accuracy: float  # Directional accuracy (%)
    sharpe_ratio: float  # Annualized Sharpe
    sortino_ratio: float  # Annualized Sortino
    max_drawdown: float  # Maximum drawdown (%)
    win_rate: float  # Percentage of winning trades
    total_return: float  # Total return (%)

    # Trade statistics
    n_trades: int
    avg_trade_return: float
    avg_holding_days: float

    # Signal breakdown
    n_bullish: int
    n_bearish: int
    n_neutral: int

    # Transaction cost summary
    total_costs: float = 0.0            # Total costs paid ($)
    avg_cost_per_trade: float = 0.0     # Average cost per trade ($)
    avg_cost_bps: float = 0.0           # Average cost in basis points
    cost_drag_pct: float = 0.0          # Total cost as % of returns

    # Per-regime performance (if available)
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Raw returns for significance testing
    returns: List[float] = field(default_factory=list)


@dataclass
class MethodComparison:
    """Comparison results across all methods."""
    asset_id: str
    asset_name: str
    n_folds: int
    timestamp: str

    # Method -> List[FoldResult]
    method_results: Dict[str, List[FoldResult]] = field(default_factory=dict)

    # Aggregated metrics per method
    summary_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Significance tests
    significance_tests: Dict[str, 'SignificanceTest'] = field(default_factory=dict)

    # Best method rankings
    rankings: Dict[str, int] = field(default_factory=dict)


@dataclass
class SignificanceTest:
    """Bootstrap significance test results."""
    method_a: str
    method_b: str
    metric: str
    mean_diff: float  # method_a - method_b
    std_diff: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    p_value: float
    is_significant: bool  # p < 0.05
    n_bootstrap: int


# =============================================================================
# WALK-FORWARD VALIDATOR
# =============================================================================

class WalkForwardValidator:
    """
    Walk-forward validation framework for ensemble methods.

    Splits data into rolling train/test windows and evaluates each
    ensemble method on identical data splits for fair comparison.

    Parameters
    ----------
    n_folds : int
        Number of walk-forward folds (default: 5)
    train_pct : float
        Percentage of each fold used for training (default: 0.7)
    min_train_samples : int
        Minimum training samples required (default: 100)
    transaction_cost_bps : float
        Transaction cost in basis points (default: 5.0)
    holding_period : int
        Default holding period in days for trades (default: 5)
    """

    AVAILABLE_METHODS = {
        # Tier 1
        'tier1_accuracy': AccuracyWeightedEnsemble,
        'tier1_magnitude': MagnitudeWeightedVoting,
        'tier1_correlation': ErrorCorrelationWeighting,
        'tier1_combined': CombinedTier1Ensemble,
        # Tier 2
        'tier2_bma': BayesianModelAveraging,
        'tier2_regime': RegimeAdaptiveEnsemble,
        'tier2_conformal': ConformalPredictionInterval,
        'tier2_combined': CombinedTier2Ensemble,
        # Tier 3
        'tier3_thompson': ThompsonSamplingEnsemble,
        'tier3_attention': AttentionBasedEnsemble,
        'tier3_quantile': QuantileRegressionForest,
        'tier3_combined': CombinedTier3Ensemble,
    }

    def __init__(
        self,
        n_folds: int = 5,
        train_pct: float = 0.7,
        min_train_samples: int = 100,
        transaction_cost_bps: float = 5.0,
        holding_period: int = 5,
        verbose: bool = True,
        cost_model: Optional[TransactionCostModel] = None,
    ):
        self.n_folds = n_folds
        self.train_pct = train_pct
        self.min_train_samples = min_train_samples
        self.transaction_cost_bps = transaction_cost_bps
        self.holding_period = holding_period
        self.verbose = verbose

        # Use provided cost model or create from legacy bps parameter
        if cost_model is not None:
            self.cost_model = cost_model
        else:
            # Convert legacy bps to cost model for backward compatibility
            self.cost_model = TransactionCostModel.from_bps(transaction_cost_bps)

        self._forecasts: Optional[pd.DataFrame] = None
        self._actuals: Optional[pd.Series] = None
        self._prices: Optional[pd.Series] = None
        self._horizons: Optional[List[int]] = None
        self._regimes: Optional[pd.Series] = None
        self._volumes: Optional[pd.Series] = None  # For market impact

    def load_data(
        self,
        asset_id: int,
        asset_name: str,
        data_dir: str = "data",
    ) -> 'WalkForwardValidator':
        """
        Load forecast and actual data for an asset.

        Supports two data formats:
        1. CSV files: forecast_d{horizon}.csv with date,prediction,actual columns
        2. Joblib files: horizons_wide/horizon_{h}.joblib with X,y dict

        Parameters
        ----------
        asset_id : int
            Numeric asset ID (e.g., 1866 for Crude Oil)
        asset_name : str
            Asset name (e.g., "Crude_Oil")
        data_dir : str
            Base data directory

        Returns
        -------
        self
        """
        asset_dir = Path(data_dir) / f"{asset_id}_{asset_name}"

        if not asset_dir.exists():
            raise FileNotFoundError(f"Asset directory not found: {asset_dir}")

        # Try CSV format first (forecast_d{horizon}.csv)
        csv_files = sorted(asset_dir.glob("forecast_d*.csv"))

        if csv_files:
            return self._load_from_csv(asset_dir, asset_name)

        # Try joblib format (horizons_wide/horizon_*.joblib)
        horizons_dir = asset_dir / "horizons_wide"
        if horizons_dir.exists():
            return self._load_from_joblib(horizons_dir, asset_name)

        raise FileNotFoundError(
            f"No forecast data found in {asset_dir}. "
            f"Expected either forecast_d*.csv or horizons_wide/horizon_*.joblib"
        )

    def _load_from_csv(
        self,
        asset_dir: Path,
        asset_name: str,
    ) -> 'WalkForwardValidator':
        """Load data from CSV files (forecast_d{horizon}.csv format)."""
        csv_files = sorted(asset_dir.glob("forecast_d*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No forecast CSV files found in {asset_dir}")

        all_forecasts = []
        actuals = None
        horizons = []

        for csv_file in csv_files:
            # Extract horizon from filename (forecast_d5.csv -> 5)
            h = int(csv_file.stem.replace('forecast_d', ''))
            horizons.append(h)

            # Read CSV
            df = pd.read_csv(csv_file)

            # Parse date column
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # Get forecast column (prediction)
            forecast_col = f'd{h}'
            df[forecast_col] = df['prediction']
            all_forecasts.append(df[[forecast_col]])

            # Get actuals (should be same across all horizons for a given date)
            if actuals is None and 'actual' in df.columns:
                actuals = df['actual'].copy()

        # Combine forecasts
        if all_forecasts:
            self._forecasts = pd.concat(all_forecasts, axis=1)
            self._forecasts = self._forecasts.sort_index()
            # Drop rows with any NaN
            self._forecasts = self._forecasts.dropna()

        self._actuals = actuals.sort_index() if actuals is not None else None
        self._prices = self._actuals.copy()  # Use actuals as price series
        self._horizons = sorted(horizons)

        # Align indices
        if self._actuals is not None:
            common_idx = self._forecasts.index.intersection(self._actuals.index)
            self._forecasts = self._forecasts.loc[common_idx]
            self._actuals = self._actuals.loc[common_idx]
            self._prices = self._prices.loc[common_idx]

        if self.verbose:
            print(f"Loaded {len(self._forecasts)} samples for {asset_name} (CSV format)")
            print(f"Horizons: {self._horizons}")
            print(f"Date range: {self._forecasts.index[0]} to {self._forecasts.index[-1]}")

        return self

    def _load_from_joblib(
        self,
        horizons_dir: Path,
        asset_name: str,
    ) -> 'WalkForwardValidator':
        """Load data from joblib files (horizons_wide format)."""
        horizon_files = sorted(horizons_dir.glob("horizon_*.joblib"))

        if not horizon_files:
            raise FileNotFoundError(f"No horizon files found in {horizons_dir}")

        all_forecasts = []
        actuals = None
        horizons = []

        for hf in horizon_files:
            h = int(hf.stem.split('_')[1])
            horizons.append(h)

            data = joblib.load(hf)
            X = data['X']
            y = data['y']

            # Rename columns to d{h} format
            if isinstance(X, pd.DataFrame):
                X = X.rename(columns=lambda c: f'd{h}' if c == 'forecast' else c)
                if f'd{h}' not in X.columns:
                    X[f'd{h}'] = X.iloc[:, 0]
                all_forecasts.append(X[[f'd{h}']])

            if actuals is None:
                actuals = y.copy()

        # Combine forecasts
        if all_forecasts:
            self._forecasts = pd.concat(all_forecasts, axis=1)
            self._forecasts = self._forecasts.sort_index()

        self._actuals = actuals.sort_index() if actuals is not None else None
        self._prices = self._actuals.copy()
        self._horizons = sorted(horizons)

        # Align indices
        common_idx = self._forecasts.index.intersection(self._actuals.index)
        self._forecasts = self._forecasts.loc[common_idx]
        self._actuals = self._actuals.loc[common_idx]
        self._prices = self._prices.loc[common_idx]

        if self.verbose:
            print(f"Loaded {len(self._forecasts)} samples for {asset_name} (joblib format)")
            print(f"Horizons: {self._horizons}")
            print(f"Date range: {self._forecasts.index[0]} to {self._forecasts.index[-1]}")

        return self

    def load_from_dataframe(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.Series,
        horizons: List[int],
        prices: Optional[pd.Series] = None,
        regimes: Optional[pd.Series] = None,
    ) -> 'WalkForwardValidator':
        """
        Load data directly from DataFrames.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecast data with columns for each horizon
        actuals : pd.Series
            Actual values
        horizons : List[int]
            List of horizon values
        prices : pd.Series, optional
            Price series for P&L calculation
        regimes : pd.Series, optional
            Market regime labels for regime-specific analysis

        Returns
        -------
        self
        """
        self._forecasts = forecasts.copy()
        self._actuals = actuals.copy()
        self._horizons = sorted(horizons)
        self._prices = prices.copy() if prices is not None else actuals.copy()
        self._regimes = regimes.copy() if regimes is not None else None

        return self

    def _create_folds(self) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Create walk-forward fold indices.

        Returns list of (train_idx, test_idx) tuples.
        """
        n_samples = len(self._forecasts)
        fold_size = n_samples // self.n_folds

        folds = []

        for i in range(self.n_folds):
            # Each fold: train on cumulative data, test on next segment
            train_end = self.min_train_samples + (i * fold_size)
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)

            if test_end <= test_start:
                break

            train_idx = self._forecasts.index[:train_end]
            test_idx = self._forecasts.index[test_start:test_end]

            if len(train_idx) >= self.min_train_samples and len(test_idx) > 0:
                folds.append((train_idx, test_idx))

        return folds

    def _calculate_returns(
        self,
        signals: pd.Series,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[int], List[CostBreakdown]]:
        """
        Calculate strategy returns from signals with detailed cost tracking.

        Parameters
        ----------
        signals : pd.Series
            Series of EnsembleResult objects
        prices : pd.Series
            Price series aligned with signals
        volumes : pd.Series, optional
            Volume series for market impact calculation

        Returns
        -------
        returns : List[float]
            Percentage returns per trade (after costs)
        holding_days : List[int]
            Holding period for each trade
        cost_breakdowns : List[CostBreakdown]
            Detailed cost breakdown for each trade
        """
        returns = []
        holding_days = []
        cost_breakdowns = []
        prev_signal = None
        position_start_idx = None

        for i, (date, result) in enumerate(signals.items()):
            if not isinstance(result, EnsembleResult):
                continue

            signal_dir = 1 if result.signal == 'BULLISH' else (-1 if result.signal == 'BEARISH' else 0)

            if signal_dir != 0 and signal_dir != prev_signal:
                # New position
                if position_start_idx is not None and prev_signal is not None:
                    # Close previous position
                    start_price = prices.iloc[position_start_idx]
                    end_price = prices.loc[date] if date in prices.index else start_price

                    # Raw return before costs
                    pct_return = ((end_price / start_price) - 1) * 100 * prev_signal

                    # Calculate transaction costs using the cost model
                    # Assume $100k notional per trade for cost calculation
                    trade_value = 100000.0
                    vol = None
                    if volumes is not None and date in volumes.index:
                        vol = volumes.loc[date]

                    cost_breakdown = self.cost_model.calculate_cost(
                        price=start_price,
                        trade_value=trade_value,
                        volume=vol,
                        is_round_trip=True,
                    )

                    # Subtract cost as percentage
                    cost_pct = cost_breakdown.total / trade_value * 100
                    pct_return -= cost_pct

                    returns.append(pct_return)
                    cost_breakdowns.append(cost_breakdown)

                    # Calculate holding days
                    hold = i - position_start_idx
                    holding_days.append(max(1, hold))

                position_start_idx = i
                prev_signal = signal_dir

        return returns, holding_days, cost_breakdowns

    def _calculate_accuracy(
        self,
        signals: pd.Series,
        actuals: pd.Series,
        horizon: int = 5,
    ) -> float:
        """
        Calculate directional accuracy.

        Compares signal direction to actual price movement.
        """
        correct = 0
        total = 0

        for date, result in signals.items():
            if not isinstance(result, EnsembleResult):
                continue

            if result.signal == 'NEUTRAL':
                continue

            # Get future return
            date_idx = actuals.index.get_loc(date)
            if date_idx + horizon >= len(actuals):
                continue

            future_price = actuals.iloc[date_idx + horizon]
            current_price = actuals.loc[date]

            actual_dir = 1 if future_price > current_price else -1
            signal_dir = 1 if result.signal == 'BULLISH' else -1

            if actual_dir == signal_dir:
                correct += 1
            total += 1

        return (correct / total * 100) if total > 0 else 0.0

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0.0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative

        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _evaluate_fold(
        self,
        method_name: str,
        method_class: type,
        train_forecasts: pd.DataFrame,
        train_actuals: pd.Series,
        test_forecasts: pd.DataFrame,
        test_actuals: pd.Series,
        fold_id: int,
    ) -> FoldResult:
        """
        Evaluate a method on a single fold.

        Parameters
        ----------
        method_name : str
            Name of the method
        method_class : type
            Ensemble class to instantiate
        train_forecasts, train_actuals : pd.DataFrame, pd.Series
            Training data
        test_forecasts, test_actuals : pd.DataFrame, pd.Series
            Test data
        fold_id : int
            Fold identifier

        Returns
        -------
        FoldResult
        """
        # Instantiate and fit ensemble
        try:
            ensemble = method_class()
            ensemble.fit(train_forecasts, train_actuals, self._horizons)
        except Exception as e:
            if self.verbose:
                print(f"  Warning: {method_name} fit failed: {e}")
            # Return empty result
            return FoldResult(
                fold_id=fold_id,
                train_start=str(train_forecasts.index[0]),
                train_end=str(train_forecasts.index[-1]),
                test_start=str(test_forecasts.index[0]),
                test_end=str(test_forecasts.index[-1]),
                n_train=len(train_forecasts),
                n_test=len(test_forecasts),
                accuracy=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_return=0.0,
                n_trades=0,
                avg_trade_return=0.0,
                avg_holding_days=0.0,
                n_bullish=0,
                n_bearish=0,
                n_neutral=0,
            )

        # Generate predictions
        try:
            signals = ensemble.predict(test_forecasts, self._horizons)
        except Exception as e:
            if self.verbose:
                print(f"  Warning: {method_name} predict failed: {e}")
            return FoldResult(
                fold_id=fold_id,
                train_start=str(train_forecasts.index[0]),
                train_end=str(train_forecasts.index[-1]),
                test_start=str(test_forecasts.index[0]),
                test_end=str(test_forecasts.index[-1]),
                n_train=len(train_forecasts),
                n_test=len(test_forecasts),
                accuracy=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_return=0.0,
                n_trades=0,
                avg_trade_return=0.0,
                avg_holding_days=0.0,
                n_bullish=0,
                n_bearish=0,
                n_neutral=0,
            )

        # Calculate returns with cost tracking
        test_prices = self._prices.loc[test_forecasts.index]
        test_volumes = self._volumes.loc[test_forecasts.index] if self._volumes is not None else None
        returns, holding_days, cost_breakdowns = self._calculate_returns(
            signals, test_prices, test_volumes
        )

        # Calculate cost metrics
        total_costs = sum(cb.total for cb in cost_breakdowns) if cost_breakdowns else 0.0
        avg_cost_per_trade = total_costs / len(cost_breakdowns) if cost_breakdowns else 0.0
        avg_cost_bps = np.mean([cb.total_bps for cb in cost_breakdowns]) if cost_breakdowns else 0.0

        # Calculate metrics
        accuracy = self._calculate_accuracy(signals, test_actuals, horizon=self.holding_period)

        sharpe = calculate_sharpe_ratio(
            returns,
            holding_days=holding_days,
            default_holding_days=self.holding_period,
        ) if returns else 0.0

        sortino = calculate_sortino_ratio(
            returns,
            holding_days=holding_days,
            default_holding_days=self.holding_period,
        ) if returns else 0.0

        max_dd = self._calculate_max_drawdown(returns)

        win_rate = (sum(1 for r in returns if r > 0) / len(returns) * 100) if returns else 0.0

        total_return = sum(returns) if returns else 0.0

        # Cost drag: what percentage of gross returns went to costs
        gross_return = total_return + (total_costs / 1000)  # Approximate gross
        cost_drag_pct = (total_costs / 1000 / gross_return * 100) if gross_return > 0 else 0.0

        # Count signals
        n_bullish = sum(1 for r in signals if isinstance(r, EnsembleResult) and r.signal == 'BULLISH')
        n_bearish = sum(1 for r in signals if isinstance(r, EnsembleResult) and r.signal == 'BEARISH')
        n_neutral = sum(1 for r in signals if isinstance(r, EnsembleResult) and r.signal == 'NEUTRAL')

        # Per-regime performance
        regime_perf = {}
        if self._regimes is not None:
            test_regimes = self._regimes.loc[test_forecasts.index]
            for regime in test_regimes.unique():
                regime_mask = test_regimes == regime
                regime_signals = signals.loc[regime_mask]
                regime_actuals = test_actuals.loc[regime_mask]
                regime_prices = test_prices.loc[regime_mask]

                if len(regime_signals) > 0:
                    regime_returns, regime_holding, _ = self._calculate_returns(regime_signals, regime_prices)
                    regime_acc = self._calculate_accuracy(regime_signals, regime_actuals)

                    regime_perf[str(regime)] = {
                        'accuracy': regime_acc,
                        'n_signals': len(regime_signals),
                        'total_return': sum(regime_returns) if regime_returns else 0.0,
                        'win_rate': (sum(1 for r in regime_returns if r > 0) / len(regime_returns) * 100) if regime_returns else 0.0,
                    }

        return FoldResult(
            fold_id=fold_id,
            train_start=str(train_forecasts.index[0]),
            train_end=str(train_forecasts.index[-1]),
            test_start=str(test_forecasts.index[0]),
            test_end=str(test_forecasts.index[-1]),
            n_train=len(train_forecasts),
            n_test=len(test_forecasts),
            accuracy=accuracy,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_return=total_return,
            n_trades=len(returns),
            avg_trade_return=np.mean(returns) if returns else 0.0,
            avg_holding_days=np.mean(holding_days) if holding_days else self.holding_period,
            n_bullish=n_bullish,
            n_bearish=n_bearish,
            n_neutral=n_neutral,
            total_costs=total_costs,
            avg_cost_per_trade=avg_cost_per_trade,
            avg_cost_bps=avg_cost_bps,
            cost_drag_pct=cost_drag_pct,
            regime_performance=regime_perf,
            returns=returns,
        )

    def run_validation(
        self,
        methods: Optional[List[str]] = None,
        asset_id: Optional[str] = None,
        asset_name: Optional[str] = None,
    ) -> MethodComparison:
        """
        Run walk-forward validation for all specified methods.

        Parameters
        ----------
        methods : List[str], optional
            Methods to evaluate. If None, uses all available methods.
        asset_id : str, optional
            Asset ID for results
        asset_name : str, optional
            Asset name for results

        Returns
        -------
        MethodComparison
            Comparison results across all methods and folds.
        """
        if self._forecasts is None or self._actuals is None:
            raise ValueError("Must load data before running validation")

        if methods is None:
            methods = list(self.AVAILABLE_METHODS.keys())

        # Validate methods
        for m in methods:
            if m not in self.AVAILABLE_METHODS:
                raise ValueError(f"Unknown method: {m}. Available: {list(self.AVAILABLE_METHODS.keys())}")

        # Create folds
        folds = self._create_folds()

        if self.verbose:
            print(f"\nRunning walk-forward validation with {len(folds)} folds")
            print(f"Methods to evaluate: {methods}")

        # Results storage
        method_results: Dict[str, List[FoldResult]] = {m: [] for m in methods}

        # Evaluate each method on each fold
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            if self.verbose:
                print(f"\nFold {fold_id + 1}/{len(folds)}: "
                      f"Train {train_idx[0]} to {train_idx[-1]}, "
                      f"Test {test_idx[0]} to {test_idx[-1]}")

            train_forecasts = self._forecasts.loc[train_idx]
            train_actuals = self._actuals.loc[train_idx]
            test_forecasts = self._forecasts.loc[test_idx]
            test_actuals = self._actuals.loc[test_idx]

            for method_name in methods:
                if self.verbose:
                    print(f"  Evaluating {method_name}...")

                method_class = self.AVAILABLE_METHODS[method_name]

                result = self._evaluate_fold(
                    method_name,
                    method_class,
                    train_forecasts,
                    train_actuals,
                    test_forecasts,
                    test_actuals,
                    fold_id,
                )

                method_results[method_name].append(result)

                if self.verbose:
                    print(f"    Accuracy: {result.accuracy:.1f}%, "
                          f"Sharpe: {result.sharpe_ratio:.2f}, "
                          f"Win Rate: {result.win_rate:.1f}%")

        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(method_results)

        # Run significance tests
        significance_tests = self._run_significance_tests(method_results)

        # Calculate rankings
        rankings = self._calculate_rankings(summary_metrics)

        return MethodComparison(
            asset_id=asset_id or "unknown",
            asset_name=asset_name or "Unknown",
            n_folds=len(folds),
            timestamp=datetime.now().isoformat(),
            method_results=method_results,
            summary_metrics=summary_metrics,
            significance_tests=significance_tests,
            rankings=rankings,
        )

    def _calculate_summary_metrics(
        self,
        method_results: Dict[str, List[FoldResult]],
    ) -> Dict[str, Dict[str, float]]:
        """Calculate aggregated metrics across folds for each method."""
        summary = {}

        for method_name, fold_results in method_results.items():
            if not fold_results:
                continue

            accuracies = [r.accuracy for r in fold_results]
            sharpes = [r.sharpe_ratio for r in fold_results]
            sortinos = [r.sortino_ratio for r in fold_results]
            max_dds = [r.max_drawdown for r in fold_results]
            win_rates = [r.win_rate for r in fold_results]
            total_returns = [r.total_return for r in fold_results]

            # Cost metrics
            total_costs = [r.total_costs for r in fold_results]
            avg_cost_bps = [r.avg_cost_bps for r in fold_results]

            summary[method_name] = {
                # Mean metrics
                'mean_accuracy': np.mean(accuracies),
                'mean_sharpe': np.mean(sharpes),
                'mean_sortino': np.mean(sortinos),
                'mean_max_drawdown': np.mean(max_dds),
                'mean_win_rate': np.mean(win_rates),
                'mean_return': np.mean(total_returns),

                # Std metrics
                'std_accuracy': np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0,
                'std_sharpe': np.std(sharpes, ddof=1) if len(sharpes) > 1 else 0.0,
                'std_return': np.std(total_returns, ddof=1) if len(total_returns) > 1 else 0.0,

                # Best/worst
                'best_sharpe': max(sharpes),
                'worst_sharpe': min(sharpes),
                'cumulative_return': sum(total_returns),

                # Trade stats
                'total_trades': sum(r.n_trades for r in fold_results),
                'avg_trades_per_fold': np.mean([r.n_trades for r in fold_results]),

                # Cost metrics
                'total_costs': sum(total_costs),
                'mean_cost_bps': np.mean(avg_cost_bps) if avg_cost_bps else 0.0,
            }

        return summary

    def _run_significance_tests(
        self,
        method_results: Dict[str, List[FoldResult]],
        n_bootstrap: int = 10000,
        alpha: float = 0.05,
    ) -> Dict[str, SignificanceTest]:
        """
        Run bootstrap significance tests comparing methods.

        Tests whether differences in Sharpe ratios are statistically significant.
        """
        methods = list(method_results.keys())
        significance_tests = {}

        # Find the best method by mean Sharpe
        best_method = max(methods, key=lambda m: np.mean([r.sharpe_ratio for r in method_results[m]]))

        # Compare best method to all others
        for method_name in methods:
            if method_name == best_method:
                continue

            key = f"{best_method}_vs_{method_name}"

            # Get pooled returns
            returns_a = []
            returns_b = []

            for fold_a, fold_b in zip(method_results[best_method], method_results[method_name]):
                returns_a.extend(fold_a.returns)
                returns_b.extend(fold_b.returns)

            if len(returns_a) < 10 or len(returns_b) < 10:
                continue

            # Bootstrap test
            test_result = self._bootstrap_sharpe_test(
                np.array(returns_a),
                np.array(returns_b),
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                method_a=best_method,
                method_b=method_name,
            )

            significance_tests[key] = test_result

        return significance_tests

    def _bootstrap_sharpe_test(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        n_bootstrap: int,
        alpha: float,
        method_a: str,
        method_b: str,
    ) -> SignificanceTest:
        """
        Bootstrap test for difference in Sharpe ratios.

        Uses paired bootstrap to account for correlation between methods
        evaluated on the same data.
        """
        n = min(len(returns_a), len(returns_b))
        returns_a = returns_a[:n]
        returns_b = returns_b[:n]

        # Calculate observed difference
        sharpe_a = calculate_sharpe_ratio(returns_a, default_holding_days=self.holding_period)
        sharpe_b = calculate_sharpe_ratio(returns_b, default_holding_days=self.holding_period)
        observed_diff = sharpe_a - sharpe_b

        # Bootstrap
        boot_diffs = []
        rng = np.random.default_rng(42)  # Reproducible

        for _ in range(n_bootstrap):
            # Paired bootstrap: sample same indices for both
            idx = rng.integers(0, n, size=n)
            boot_a = returns_a[idx]
            boot_b = returns_b[idx]

            boot_sharpe_a = calculate_sharpe_ratio(boot_a, default_holding_days=self.holding_period)
            boot_sharpe_b = calculate_sharpe_ratio(boot_b, default_holding_days=self.holding_period)

            boot_diffs.append(boot_sharpe_a - boot_sharpe_b)

        boot_diffs = np.array(boot_diffs)

        # Calculate confidence interval
        ci_lower = np.percentile(boot_diffs, alpha / 2 * 100)
        ci_upper = np.percentile(boot_diffs, (1 - alpha / 2) * 100)

        # Calculate p-value (two-tailed test of H0: diff = 0)
        # Count how many bootstrap samples have opposite sign
        if observed_diff > 0:
            p_value = np.mean(boot_diffs <= 0) * 2
        else:
            p_value = np.mean(boot_diffs >= 0) * 2
        p_value = min(p_value, 1.0)

        return SignificanceTest(
            method_a=method_a,
            method_b=method_b,
            metric='sharpe_ratio',
            mean_diff=observed_diff,
            std_diff=np.std(boot_diffs),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            is_significant=p_value < alpha,
            n_bootstrap=n_bootstrap,
        )

    def _calculate_rankings(
        self,
        summary_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, int]:
        """Rank methods by mean Sharpe ratio."""
        if not summary_metrics:
            return {}

        # Sort by mean Sharpe (higher is better)
        sorted_methods = sorted(
            summary_metrics.keys(),
            key=lambda m: summary_metrics[m].get('mean_sharpe', 0.0),
            reverse=True,
        )

        return {method: rank + 1 for rank, method in enumerate(sorted_methods)}

    def print_results(self, comparison: MethodComparison) -> None:
        """Print formatted results to console."""
        print("\n" + "=" * 90)
        print(f"WALK-FORWARD VALIDATION RESULTS")
        print(f"Asset: {comparison.asset_name} ({comparison.asset_id})")
        print(f"Folds: {comparison.n_folds}")
        print(f"Cost Model: {self.cost_model.summary()}")
        print(f"Timestamp: {comparison.timestamp}")
        print("=" * 90)

        # Summary table
        print("\n{:<22} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>5}".format(
            "Method", "Acc%", "Sharpe", "Sortino", "MaxDD%", "Win%", "Cost(bps)", "Rank"
        ))
        print("-" * 90)

        for method in sorted(comparison.summary_metrics.keys(),
                            key=lambda m: comparison.rankings.get(m, 99)):
            metrics = comparison.summary_metrics[method]
            rank = comparison.rankings.get(method, 0)

            print("{:<22} {:>8.1f} {:>8.2f} {:>8.2f} {:>8.1f} {:>8.1f} {:>8.1f} {:>5}".format(
                method,
                metrics['mean_accuracy'],
                metrics['mean_sharpe'],
                metrics['mean_sortino'],
                metrics['mean_max_drawdown'],
                metrics['mean_win_rate'],
                metrics.get('mean_cost_bps', 0.0),
                rank,
            ))

        # Cost impact summary
        print("\n" + "-" * 90)
        print("TRANSACTION COST IMPACT")
        print("-" * 90)

        total_costs_all = sum(
            metrics.get('total_costs', 0.0)
            for metrics in comparison.summary_metrics.values()
        )
        total_trades_all = sum(
            metrics.get('total_trades', 0)
            for metrics in comparison.summary_metrics.values()
        )
        if total_trades_all > 0:
            avg_cost_per_trade = total_costs_all / total_trades_all
            print(f"Total costs across all methods: ${total_costs_all:,.2f}")
            print(f"Average cost per trade: ${avg_cost_per_trade:.2f}")
            print(f"Breakeven edge required: {self.cost_model.estimate_breakeven_edge():.2f}% per trade")

        # Significance tests
        if comparison.significance_tests:
            print("\n" + "-" * 90)
            print("SIGNIFICANCE TESTS (vs best method)")
            print("-" * 90)

            for key, test in comparison.significance_tests.items():
                sig_marker = "*" if test.is_significant else ""
                print(f"{test.method_a} vs {test.method_b}: "
                      f"diff={test.mean_diff:+.3f} "
                      f"95% CI=[{test.ci_lower:.3f}, {test.ci_upper:.3f}] "
                      f"p={test.p_value:.4f}{sig_marker}")

        # Per-tier summary
        print("\n" + "-" * 90)
        print("PER-TIER SUMMARY")
        print("-" * 90)

        for tier in ['tier1', 'tier2', 'tier3']:
            tier_methods = [m for m in comparison.summary_metrics.keys() if m.startswith(tier)]
            if tier_methods:
                tier_sharpes = [comparison.summary_metrics[m]['mean_sharpe'] for m in tier_methods]
                tier_best = tier_methods[np.argmax(tier_sharpes)]
                print(f"{tier.upper()}: Best={tier_best} (Sharpe={max(tier_sharpes):.2f})")

    def save_results(
        self,
        comparison: MethodComparison,
        output_path: str,
    ) -> None:
        """Save results to JSON file."""
        # Convert to serializable format
        output = {
            'asset_id': comparison.asset_id,
            'asset_name': comparison.asset_name,
            'n_folds': comparison.n_folds,
            'timestamp': comparison.timestamp,
            'cost_model': {
                'description': self.cost_model.summary(),
                'config': self.cost_model.config.to_dict(),
            },
            'summary_metrics': comparison.summary_metrics,
            'rankings': comparison.rankings,
            'significance_tests': {
                k: asdict(v) for k, v in comparison.significance_tests.items()
            },
            'fold_details': {
                method: [asdict(r) for r in results]
                for method, results in comparison.method_results.items()
            },
        }

        # Remove raw returns from fold details (too large)
        for method, folds in output['fold_details'].items():
            for fold in folds:
                fold.pop('returns', None)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for ensemble methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation with 5 folds
    python -m backend.backtesting.walk_forward --asset crude_oil --folds 5

    # With custom transaction costs
    python -m backend.backtesting.walk_forward --asset crude_oil --costs "fixed=5 pct=0.001 spread=0.0002"

    # Using a cost preset
    python -m backend.backtesting.walk_forward --asset crude_oil --cost-preset commodities

    # Compare specific methods with output
    python -m backend.backtesting.walk_forward --asset gold \\
        --methods tier1_combined tier2_combined tier3_combined \\
        --costs "pct=0.0005" --output results.json

Cost model parameters:
    fixed=X     - Fixed cost per trade in $ (e.g., fixed=5)
    pct=X       - Percentage commission (e.g., pct=0.001 for 10 bps)
    spread=X    - Half bid-ask spread (e.g., spread=0.0002 for 2 bps)
    slip=X      - Slippage (e.g., slip=0.0001)
    impact=X    - Market impact coefficient (e.g., impact=0.1)

Cost presets: zero, low, medium, high, futures, forex, crypto, commodities
        """,
    )

    parser.add_argument(
        '--asset',
        type=str,
        required=True,
        help="Asset to validate (e.g., 'crude_oil', '1866_Crude_Oil')",
    )

    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help="Number of walk-forward folds (default: 5)",
    )

    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=None,
        help="Methods to evaluate (default: all). Options: " +
             ", ".join(WalkForwardValidator.AVAILABLE_METHODS.keys()),
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output JSON file path (optional)",
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help="Data directory path (default: data)",
    )

    parser.add_argument(
        '--transaction-cost',
        type=float,
        default=5.0,
        help="[DEPRECATED] Use --costs instead. Transaction cost in bps (default: 5.0)",
    )

    parser.add_argument(
        '--costs',
        type=str,
        default=None,
        help="Transaction cost specification (e.g., 'fixed=5 pct=0.001 spread=0.0002')",
    )

    parser.add_argument(
        '--cost-preset',
        type=str,
        default=None,
        choices=['zero', 'low', 'medium', 'high', 'futures', 'forex', 'crypto', 'commodities'],
        help="Use a predefined cost model preset",
    )

    parser.add_argument(
        '--holding-period',
        type=int,
        default=5,
        help="Default holding period in days (default: 5)",
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress progress output",
    )

    return parser.parse_args()


def resolve_asset(asset_str: str, data_dir: str) -> Tuple[int, str]:
    """
    Resolve asset string to (asset_id, asset_name).

    Supports formats:
    - "crude_oil" -> finds matching directory
    - "1866_Crude_Oil" -> direct match
    - "1866" -> finds by ID
    """
    data_path = Path(data_dir)

    # Try direct match first
    if '_' in asset_str:
        parts = asset_str.split('_', 1)
        try:
            asset_id = int(parts[0])
            asset_name = parts[1]
            if (data_path / asset_str).exists():
                return asset_id, asset_name
        except ValueError:
            pass

    # Search for matching directory
    asset_lower = asset_str.lower().replace('_', '')

    for d in data_path.iterdir():
        if d.is_dir():
            dir_lower = d.name.lower().replace('_', '')
            if asset_lower in dir_lower:
                parts = d.name.split('_', 1)
                try:
                    asset_id = int(parts[0])
                    asset_name = parts[1] if len(parts) > 1 else d.name
                    return asset_id, asset_name
                except ValueError:
                    continue

    raise ValueError(f"Could not find asset matching: {asset_str}")


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Resolve asset
        asset_id, asset_name = resolve_asset(args.asset, args.data_dir)

        if not args.quiet:
            print(f"Resolved asset: {asset_id}_{asset_name}")

        # Create cost model
        cost_model = None
        if args.cost_preset:
            # Use preset
            cost_model = TransactionCostModel.from_preset(args.cost_preset)
            if not args.quiet:
                print(f"Using cost preset: {args.cost_preset}")
        elif args.costs:
            # Parse cost string
            cost_model = TransactionCostModel.from_string(args.costs)
            if not args.quiet:
                print(f"Cost model: {cost_model.summary()}")
        else:
            # Fall back to legacy bps parameter
            cost_model = TransactionCostModel.from_bps(args.transaction_cost)
            if not args.quiet:
                print(f"Using legacy cost: {args.transaction_cost} bps")

        # Create validator
        validator = WalkForwardValidator(
            n_folds=args.folds,
            transaction_cost_bps=args.transaction_cost,
            holding_period=args.holding_period,
            verbose=not args.quiet,
            cost_model=cost_model,
        )

        # Load data
        validator.load_data(
            asset_id=asset_id,
            asset_name=asset_name,
            data_dir=args.data_dir,
        )

        # Run validation
        comparison = validator.run_validation(
            methods=args.methods,
            asset_id=str(asset_id),
            asset_name=asset_name,
        )

        # Print results
        validator.print_results(comparison)

        # Save if output specified
        if args.output:
            validator.save_results(comparison, args.output)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
