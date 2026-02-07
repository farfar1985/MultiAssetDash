"""
Backtesting framework for QDT Nexus ensemble methods.

This package provides walk-forward validation and backtesting tools
for comparing ensemble methods across different market regimes.
"""

from .walk_forward import (
    WalkForwardValidator,
    FoldResult,
    MethodComparison,
    SignificanceTest,
)

from .costs import (
    TransactionCostModel,
    CostConfig,
    CostBreakdown,
    apply_costs_to_returns,
    compare_cost_impact,
)

__all__ = [
    # Walk-forward validation
    'WalkForwardValidator',
    'FoldResult',
    'MethodComparison',
    'SignificanceTest',
    # Transaction costs
    'TransactionCostModel',
    'CostConfig',
    'CostBreakdown',
    'apply_costs_to_returns',
    'compare_cost_impact',
]
