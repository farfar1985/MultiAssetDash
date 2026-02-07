"""
Standardized Performance Metrics for QDT Nexus

This module provides the CANONICAL implementations of performance metrics
used across the codebase. All ensemble code should use these functions
instead of inline calculations to ensure consistency.

Sharpe Ratio Standard:
- Uses percentage returns per trade (not dollar returns)
- Uses sample standard deviation (ddof=1) for unbiased estimation
- Annualization factor based on actual trades per year

References:
- docs/SHARPE_DISCREPANCY_ANALYSIS.md
- find_diverse_combo.py (Pattern B - correct implementation)
"""

from typing import Union, Optional, List
import numpy as np
from numpy.typing import ArrayLike


# Trading days per year (standard for annualization)
TRADING_DAYS_PER_YEAR = 252


def calculate_sharpe_ratio(
    returns: ArrayLike,
    holding_days: Optional[ArrayLike] = None,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
    default_holding_days: float = 5.0,
) -> float:
    """
    Calculate the Sharpe ratio using the standardized QDT methodology.

    This is the CANONICAL Sharpe calculation for QDT Nexus. Use this function
    instead of inline calculations to ensure consistency across the codebase.

    Parameters
    ----------
    returns : ArrayLike
        Array of percentage returns per trade (e.g., [2.5, -1.2, 0.8]).
        IMPORTANT: These should be percentage returns, not dollar returns.

    holding_days : ArrayLike, optional
        Array of holding periods in days for each trade.
        If not provided, uses default_holding_days for annualization.

    risk_free_rate : float, default=0.0
        Annual risk-free rate for excess returns calculation.
        Set to 0.0 for basic Sharpe ratio.

    annualize : bool, default=True
        Whether to annualize the Sharpe ratio based on trade frequency.

    default_holding_days : float, default=5.0
        Default holding period to use if holding_days is not provided.
        This affects the annualization factor.

    Returns
    -------
    float
        The Sharpe ratio. Returns 0.0 if insufficient data or zero std deviation.

    Examples
    --------
    >>> returns = [2.5, -1.2, 0.8, 3.1, -0.5]
    >>> calculate_sharpe_ratio(returns)
    1.234  # (example output)

    >>> # With explicit holding periods
    >>> returns = [2.5, -1.2, 0.8]
    >>> holding_days = [5, 3, 7]
    >>> calculate_sharpe_ratio(returns, holding_days)
    1.456  # (example output)

    Notes
    -----
    The annualization formula is:
        sharpe = (mean_return / std_return) * sqrt(trades_per_year)

    where trades_per_year = TRADING_DAYS_PER_YEAR / avg_holding_days

    This correctly accounts for the actual trading frequency rather than
    assuming daily returns (which would use sqrt(252) directly).
    """
    returns = np.asarray(returns, dtype=np.float64)

    # Need at least 2 observations for std calculation
    if len(returns) < 2:
        return 0.0

    # Handle holding days
    if holding_days is not None:
        holding_days = np.asarray(holding_days, dtype=np.float64)
        avg_hold_days = float(np.mean(holding_days))
    else:
        avg_hold_days = default_holding_days

    # Ensure positive holding days
    avg_hold_days = max(1.0, avg_hold_days)

    # Calculate trades per year for annualization
    trades_per_year = TRADING_DAYS_PER_YEAR / avg_hold_days

    # Calculate mean and std with sample correction (ddof=1)
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1))  # Sample std, not population

    # Avoid division by zero
    if std_return <= 0:
        return 0.0

    # Calculate base Sharpe (mean / std)
    sharpe = mean_return / std_return

    # Annualize if requested
    if annualize:
        sharpe *= np.sqrt(trades_per_year)

    return float(sharpe)


def calculate_sharpe_ratio_daily(
    daily_returns: ArrayLike,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Sharpe ratio from daily returns.

    Use this function when you have daily portfolio returns (e.g., from a
    time series). For per-trade returns, use calculate_sharpe_ratio() instead.

    Parameters
    ----------
    daily_returns : ArrayLike
        Array of daily percentage returns.

    risk_free_rate : float, default=0.0
        Annual risk-free rate.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    daily_returns = np.asarray(daily_returns, dtype=np.float64)

    if len(daily_returns) < 2:
        return 0.0

    # Adjust for risk-free rate (convert annual to daily)
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = daily_returns - daily_rf

    mean_return = float(np.mean(excess_returns))
    std_return = float(np.std(excess_returns, ddof=1))

    if std_return <= 0:
        return 0.0

    # Annualize: multiply by sqrt(252) for daily returns
    sharpe = (mean_return / std_return) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return float(sharpe)


def calculate_sortino_ratio(
    returns: ArrayLike,
    holding_days: Optional[ArrayLike] = None,
    target_return: float = 0.0,
    annualize: bool = True,
    default_holding_days: float = 5.0,
) -> float:
    """
    Calculate the Sortino ratio (downside risk-adjusted return).

    Similar to Sharpe ratio but uses downside deviation instead of
    standard deviation, penalizing only negative volatility.

    Parameters
    ----------
    returns : ArrayLike
        Array of percentage returns per trade.

    holding_days : ArrayLike, optional
        Array of holding periods in days for each trade.

    target_return : float, default=0.0
        Minimum acceptable return threshold for downside calculation.

    annualize : bool, default=True
        Whether to annualize the ratio.

    default_holding_days : float, default=5.0
        Default holding period if holding_days not provided.

    Returns
    -------
    float
        The Sortino ratio.
    """
    returns = np.asarray(returns, dtype=np.float64)

    if len(returns) < 2:
        return 0.0

    # Handle holding days
    if holding_days is not None:
        holding_days = np.asarray(holding_days, dtype=np.float64)
        avg_hold_days = float(np.mean(holding_days))
    else:
        avg_hold_days = default_holding_days

    avg_hold_days = max(1.0, avg_hold_days)
    trades_per_year = TRADING_DAYS_PER_YEAR / avg_hold_days

    mean_return = float(np.mean(returns))

    # Calculate downside deviation (only negative deviations from target)
    downside_returns = returns - target_return
    downside_returns = downside_returns[downside_returns < 0]

    if len(downside_returns) == 0:
        # No negative returns - infinite Sortino (return a high value)
        return 10.0

    downside_std = float(np.std(downside_returns, ddof=1))

    if downside_std <= 0:
        return 0.0

    sortino = mean_return / downside_std

    if annualize:
        sortino *= np.sqrt(trades_per_year)

    return float(sortino)


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown: float,
    years: float = 1.0,
) -> float:
    """
    Calculate the Calmar ratio (return / max drawdown).

    Parameters
    ----------
    total_return : float
        Total percentage return over the period.

    max_drawdown : float
        Maximum drawdown as a positive percentage (e.g., 15.0 for 15%).

    years : float, default=1.0
        Number of years for the period (for annualization).

    Returns
    -------
    float
        The Calmar ratio.
    """
    if max_drawdown <= 0 or years <= 0:
        return 0.0

    annualized_return = total_return / years
    calmar = annualized_return / max_drawdown

    return float(calmar)


def calculate_information_ratio(
    returns: ArrayLike,
    benchmark_returns: ArrayLike,
    annualize: bool = True,
) -> float:
    """
    Calculate the Information Ratio (excess returns / tracking error).

    Parameters
    ----------
    returns : ArrayLike
        Strategy returns.

    benchmark_returns : ArrayLike
        Benchmark returns (same length as returns).

    annualize : bool, default=True
        Whether to annualize the ratio.

    Returns
    -------
    float
        The Information Ratio.
    """
    returns = np.asarray(returns, dtype=np.float64)
    benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)

    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0

    # Active returns (excess over benchmark)
    active_returns = returns - benchmark_returns

    mean_active = float(np.mean(active_returns))
    tracking_error = float(np.std(active_returns, ddof=1))

    if tracking_error <= 0:
        return 0.0

    ir = mean_active / tracking_error

    if annualize:
        ir *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return float(ir)


# Convenience aliases for backward compatibility
sharpe_ratio = calculate_sharpe_ratio
sortino_ratio = calculate_sortino_ratio
calmar_ratio = calculate_calmar_ratio
