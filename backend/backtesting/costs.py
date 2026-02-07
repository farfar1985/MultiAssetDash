"""
Transaction Cost Modeling for Backtesting
==========================================

This module provides realistic transaction cost modeling including:
- Fixed costs per trade (minimum commissions)
- Percentage-based costs (commissions, slippage)
- Market impact model (square-root of volume)
- Bid-ask spread modeling

Accurate cost modeling is critical for realistic Sharpe ratio calculations
and proper strategy evaluation.

Usage:
    from backend.backtesting.costs import TransactionCostModel

    # Create model with defaults
    model = TransactionCostModel()

    # Or with custom parameters
    model = TransactionCostModel(
        fixed_cost=5.0,           # $5 per trade
        pct_cost=0.001,           # 0.1% (10 bps)
        spread_cost=0.0002,       # 0.02% (2 bps) half-spread
        market_impact_coef=0.1,   # Market impact coefficient
    )

    # Calculate cost for a trade
    cost = model.calculate_cost(
        price=100.0,
        trade_value=10000.0,
        volume=1000000.0,  # Optional: for market impact
    )

Created: 2026-02-06
Author: AmiraB
Reference: docs/ENSEMBLE_METHODS_PLAN.md
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Union, Any
from dataclasses import dataclass, field, asdict


@dataclass
class CostBreakdown:
    """Detailed breakdown of transaction costs."""
    fixed: float = 0.0          # Fixed cost component
    percentage: float = 0.0      # Percentage-based cost
    spread: float = 0.0          # Bid-ask spread cost
    market_impact: float = 0.0   # Market impact cost
    total: float = 0.0           # Total cost

    # As percentage of trade value
    total_bps: float = 0.0       # Total cost in basis points

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CostConfig:
    """Configuration for transaction cost model."""
    # Fixed costs
    fixed_cost: float = 0.0           # Fixed cost per trade ($)
    min_cost: float = 0.0             # Minimum cost per trade ($)

    # Percentage costs (as decimals, e.g., 0.001 = 0.1%)
    pct_cost: float = 0.0             # Commission as % of trade value
    slippage: float = 0.0             # Expected slippage as % of price

    # Spread costs (as decimals)
    spread_cost: float = 0.0          # Half bid-ask spread as % of price

    # Market impact (Almgren-Chriss style)
    market_impact_coef: float = 0.0   # Market impact coefficient
    market_impact_exp: float = 0.5    # Exponent (typically 0.5 = sqrt)

    # Volume-based adjustments
    avg_daily_volume: float = 1e9     # Average daily volume for scaling

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CostConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_string(cls, cost_str: str) -> 'CostConfig':
        """
        Parse cost configuration from CLI string.

        Format: "fixed=5 pct=0.001 spread=0.0002 impact=0.1"

        Supported keys:
            fixed   - Fixed cost per trade ($)
            min     - Minimum cost per trade ($)
            pct     - Percentage commission (decimal, e.g., 0.001 = 10 bps)
            slip    - Slippage (decimal)
            spread  - Half spread (decimal)
            impact  - Market impact coefficient
            exp     - Market impact exponent (default 0.5)

        Examples:
            "fixed=5 pct=0.001"           -> $5 fixed + 10 bps
            "pct=0.0005 spread=0.0002"    -> 5 bps + 2 bps spread
            "impact=0.1 exp=0.5"          -> Market impact model
        """
        config = cls()

        if not cost_str or cost_str.strip() == '':
            return config

        # Parse key=value pairs
        for part in cost_str.split():
            if '=' not in part:
                continue

            key, value = part.split('=', 1)
            key = key.strip().lower()
            try:
                val = float(value.strip())
            except ValueError:
                continue

            if key == 'fixed':
                config.fixed_cost = val
            elif key == 'min':
                config.min_cost = val
            elif key == 'pct':
                config.pct_cost = val
            elif key == 'slip' or key == 'slippage':
                config.slippage = val
            elif key == 'spread':
                config.spread_cost = val
            elif key == 'impact':
                config.market_impact_coef = val
            elif key == 'exp':
                config.market_impact_exp = val
            elif key == 'volume' or key == 'adv':
                config.avg_daily_volume = val

        return config


class TransactionCostModel:
    """
    Comprehensive transaction cost model for backtesting.

    Combines multiple cost components:

    1. Fixed Costs
       - Minimum commission per trade
       - Platform/exchange fees

    2. Percentage Costs
       - Commission as % of trade value
       - Expected slippage

    3. Spread Costs
       - Bid-ask spread (pay half-spread on entry and exit)

    4. Market Impact
       - Price impact from trading (Almgren-Chriss model)
       - Scales with sqrt(trade_size / volume)

    Total Cost Formula:
        cost = max(min_cost, fixed + pct * value + spread * value + impact)

        where impact = coef * value * (trade_size / adv) ^ exp

    Parameters
    ----------
    fixed_cost : float
        Fixed cost per trade in dollars (default: 0.0)
    pct_cost : float
        Percentage cost as decimal (0.001 = 10 bps, default: 0.0)
    spread_cost : float
        Half-spread as decimal (default: 0.0)
    market_impact_coef : float
        Market impact coefficient (default: 0.0)
    config : CostConfig, optional
        Full configuration object (overrides individual parameters)

    Examples
    --------
    >>> # Simple 5 bps all-in cost
    >>> model = TransactionCostModel(pct_cost=0.0005)
    >>> model.calculate_cost(price=100, trade_value=10000)
    CostBreakdown(total=5.0, total_bps=5.0, ...)

    >>> # Complex model with market impact
    >>> model = TransactionCostModel(
    ...     fixed_cost=5.0,
    ...     pct_cost=0.0002,
    ...     spread_cost=0.0001,
    ...     market_impact_coef=0.1,
    ... )
    >>> model.calculate_cost(price=100, trade_value=100000, volume=1e6)
    CostBreakdown(total=..., ...)
    """

    # Common presets for different asset classes
    PRESETS = {
        'zero': CostConfig(),  # No costs (for comparison)
        'low': CostConfig(pct_cost=0.0005, spread_cost=0.0001),  # 5 + 1 bps
        'medium': CostConfig(pct_cost=0.001, spread_cost=0.0002),  # 10 + 2 bps
        'high': CostConfig(pct_cost=0.002, spread_cost=0.0005),  # 20 + 5 bps
        'futures': CostConfig(fixed_cost=2.5, pct_cost=0.00005),  # Futures-like
        'forex': CostConfig(spread_cost=0.00005),  # FX spread only
        'crypto': CostConfig(pct_cost=0.001, spread_cost=0.001),  # Crypto markets
        'commodities': CostConfig(pct_cost=0.0005, spread_cost=0.0002, fixed_cost=5.0),
    }

    def __init__(
        self,
        fixed_cost: float = 0.0,
        pct_cost: float = 0.0,
        spread_cost: float = 0.0,
        market_impact_coef: float = 0.0,
        config: Optional[CostConfig] = None,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = CostConfig(
                fixed_cost=fixed_cost,
                pct_cost=pct_cost,
                spread_cost=spread_cost,
                market_impact_coef=market_impact_coef,
            )

    @classmethod
    def from_preset(cls, preset: str) -> 'TransactionCostModel':
        """
        Create model from a preset configuration.

        Available presets:
            'zero'        - No costs (baseline comparison)
            'low'         - 6 bps total (institutional)
            'medium'      - 12 bps total (retail broker)
            'high'        - 25 bps total (illiquid markets)
            'futures'     - $2.50 fixed + 0.5 bps
            'forex'       - 0.5 bps spread only
            'crypto'      - 20 bps total
            'commodities' - $5 fixed + 7 bps

        Parameters
        ----------
        preset : str
            Name of the preset to use

        Returns
        -------
        TransactionCostModel
        """
        if preset not in cls.PRESETS:
            available = ', '.join(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

        return cls(config=cls.PRESETS[preset])

    @classmethod
    def from_bps(cls, total_bps: float) -> 'TransactionCostModel':
        """
        Create simple model from total cost in basis points.

        Parameters
        ----------
        total_bps : float
            Total round-trip cost in basis points

        Returns
        -------
        TransactionCostModel
        """
        return cls(pct_cost=total_bps / 10000)

    @classmethod
    def from_string(cls, cost_str: str) -> 'TransactionCostModel':
        """
        Create model from CLI-style string.

        Parameters
        ----------
        cost_str : str
            Cost specification string (e.g., "fixed=5 pct=0.001 spread=0.0002")

        Returns
        -------
        TransactionCostModel
        """
        config = CostConfig.from_string(cost_str)
        return cls(config=config)

    def calculate_cost(
        self,
        price: float,
        trade_value: float,
        volume: Optional[float] = None,
        is_round_trip: bool = True,
    ) -> CostBreakdown:
        """
        Calculate transaction cost for a trade.

        Parameters
        ----------
        price : float
            Current asset price
        trade_value : float
            Total value of the trade ($)
        volume : float, optional
            Daily volume for market impact calculation
        is_round_trip : bool
            If True, include both entry and exit costs (default: True)

        Returns
        -------
        CostBreakdown
            Detailed cost breakdown
        """
        multiplier = 2.0 if is_round_trip else 1.0

        # Fixed cost
        fixed = self.config.fixed_cost * multiplier

        # Percentage cost (commission + slippage)
        pct_rate = self.config.pct_cost + self.config.slippage
        percentage = trade_value * pct_rate * multiplier

        # Spread cost (pay half-spread on each leg)
        spread = trade_value * self.config.spread_cost * multiplier

        # Market impact (Almgren-Chriss style)
        market_impact = 0.0
        if self.config.market_impact_coef > 0 and volume is not None and volume > 0:
            # Participation rate
            participation = trade_value / volume

            # Impact: coef * value * (participation) ^ exp
            impact_pct = self.config.market_impact_coef * (
                participation ** self.config.market_impact_exp
            )
            market_impact = trade_value * impact_pct * multiplier

        # Total with minimum
        total = fixed + percentage + spread + market_impact
        total = max(total, self.config.min_cost * multiplier)

        # Convert to basis points
        total_bps = (total / trade_value * 10000) if trade_value > 0 else 0.0

        return CostBreakdown(
            fixed=fixed,
            percentage=percentage,
            spread=spread,
            market_impact=market_impact,
            total=total,
            total_bps=total_bps,
        )

    def cost_as_pct(
        self,
        price: float,
        trade_value: float,
        volume: Optional[float] = None,
        is_round_trip: bool = True,
    ) -> float:
        """
        Calculate cost as a percentage of trade value.

        Convenience method for direct use in return calculations.

        Parameters
        ----------
        price : float
            Current asset price
        trade_value : float
            Total value of the trade ($)
        volume : float, optional
            Daily volume for market impact calculation
        is_round_trip : bool
            If True, include both entry and exit costs (default: True)

        Returns
        -------
        float
            Cost as percentage (e.g., 0.12 for 12 bps)
        """
        breakdown = self.calculate_cost(price, trade_value, volume, is_round_trip)
        return breakdown.total / trade_value * 100 if trade_value > 0 else 0.0

    def cost_as_bps(
        self,
        price: float,
        trade_value: float,
        volume: Optional[float] = None,
        is_round_trip: bool = True,
    ) -> float:
        """
        Calculate cost in basis points.

        Parameters
        ----------
        price : float
            Current asset price
        trade_value : float
            Total value of the trade ($)
        volume : float, optional
            Daily volume for market impact calculation
        is_round_trip : bool
            If True, include both entry and exit costs (default: True)

        Returns
        -------
        float
            Cost in basis points
        """
        breakdown = self.calculate_cost(price, trade_value, volume, is_round_trip)
        return breakdown.total_bps

    def estimate_breakeven_edge(self, trades_per_year: int = 50) -> float:
        """
        Estimate the minimum edge needed to break even after costs.

        Assumes a typical trade value and calculates the per-trade
        return needed to cover transaction costs.

        Parameters
        ----------
        trades_per_year : int
            Expected number of trades per year

        Returns
        -------
        float
            Required edge per trade as percentage
        """
        # Use $100k as typical trade value
        typical_trade = 100000
        cost_pct = self.cost_as_pct(100, typical_trade)

        return cost_pct

    def summary(self) -> str:
        """Return a human-readable summary of the cost model."""
        parts = []

        if self.config.fixed_cost > 0:
            parts.append(f"${self.config.fixed_cost:.2f} fixed")

        if self.config.pct_cost > 0:
            bps = self.config.pct_cost * 10000
            parts.append(f"{bps:.1f} bps commission")

        if self.config.slippage > 0:
            bps = self.config.slippage * 10000
            parts.append(f"{bps:.1f} bps slippage")

        if self.config.spread_cost > 0:
            bps = self.config.spread_cost * 10000
            parts.append(f"{bps:.1f} bps spread")

        if self.config.market_impact_coef > 0:
            parts.append(f"impact={self.config.market_impact_coef:.2f}")

        if not parts:
            return "No transaction costs"

        return " + ".join(parts)

    def __repr__(self) -> str:
        return f"TransactionCostModel({self.summary()})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_costs_to_returns(
    returns: np.ndarray,
    prices: np.ndarray,
    trade_values: np.ndarray,
    cost_model: TransactionCostModel,
    volumes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply transaction costs to a series of returns.

    Parameters
    ----------
    returns : np.ndarray
        Raw returns (as percentages, e.g., 2.5 for 2.5%)
    prices : np.ndarray
        Prices at each trade
    trade_values : np.ndarray
        Trade values at each trade
    cost_model : TransactionCostModel
        Cost model to apply
    volumes : np.ndarray, optional
        Volumes at each trade (for market impact)

    Returns
    -------
    np.ndarray
        Cost-adjusted returns
    """
    adjusted = np.zeros_like(returns)

    for i in range(len(returns)):
        vol = volumes[i] if volumes is not None else None
        cost_pct = cost_model.cost_as_pct(
            prices[i],
            trade_values[i],
            vol,
            is_round_trip=True,
        )
        adjusted[i] = returns[i] - cost_pct

    return adjusted


def compare_cost_impact(
    returns: np.ndarray,
    cost_models: Dict[str, TransactionCostModel],
    trade_value: float = 100000,
    price: float = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Compare impact of different cost models on returns.

    Parameters
    ----------
    returns : np.ndarray
        Raw returns (as percentages)
    cost_models : Dict[str, TransactionCostModel]
        Named cost models to compare
    trade_value : float
        Typical trade value
    price : float
        Typical price

    Returns
    -------
    Dict[str, Dict[str, float]]
        Comparison metrics for each cost model
    """
    from utils.metrics import calculate_sharpe_ratio

    results = {}

    for name, model in cost_models.items():
        cost_pct = model.cost_as_pct(price, trade_value)
        adjusted = returns - cost_pct

        results[name] = {
            'cost_per_trade_pct': cost_pct,
            'cost_per_trade_bps': cost_pct * 100,
            'mean_return': float(np.mean(adjusted)),
            'total_return': float(np.sum(adjusted)),
            'sharpe_ratio': calculate_sharpe_ratio(adjusted),
            'break_even_trades': int(np.sum(adjusted > 0)),
        }

    return results
