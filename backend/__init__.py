"""
Backend Module for QDT Nexus

This module contains server-side ensemble methods and utilities.
"""

from backend.ensemble_tier1 import (
    # Base classes
    BaseTier1Ensemble,
    EnsembleResult,

    # Tier 1 Methods
    AccuracyWeightedEnsemble,
    MagnitudeWeightedVoting,
    ErrorCorrelationWeighting,
    CombinedTier1Ensemble,

    # Evaluation utilities
    evaluate_ensemble,
    compare_tier1_methods,
)

__all__ = [
    'BaseTier1Ensemble',
    'EnsembleResult',
    'AccuracyWeightedEnsemble',
    'MagnitudeWeightedVoting',
    'ErrorCorrelationWeighting',
    'CombinedTier1Ensemble',
    'evaluate_ensemble',
    'compare_tier1_methods',
]
