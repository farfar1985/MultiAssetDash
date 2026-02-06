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

__all__ = [
    'WalkForwardValidator',
    'FoldResult',
    'MethodComparison',
    'SignificanceTest',
]
