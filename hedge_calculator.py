"""
Optimal Hedge Calculator — Hedging Desk Intelligence
=====================================================
Calculates optimal hedge ratios, basis forecasts, and hedge effectiveness.
Designed for CME hedging desks and procurement teams.

Key Features:
- Minimum variance hedge ratio
- Basis risk estimation
- Roll cost forecasting
- Hedge effectiveness metrics

Author: AmiraB
Created: 2026-02-08
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path("data/qdl_history")


@dataclass
class HedgeRecommendation:
    """Hedge recommendation for a position"""
    asset: str
    position_type: str  # LONG or SHORT (what you're hedging)
    notional_value: float
    
    # Optimal hedge
    optimal_hedge_ratio: float
    hedge_notional: float
    hedge_instrument: str
    
    # Costs
    estimated_basis_risk: float  # bps
    estimated_roll_cost: float   # % per roll
    total_hedge_cost: float      # bps annualized
    
    # Effectiveness
    expected_variance_reduction: float  # %
    r_squared: float
    tracking_error: float  # bps
    
    # Timing
    urgency: str  # IMMEDIATE, TODAY, THIS_WEEK, MONITOR
    entry_recommendation: str  # MARKET, LIMIT, TWAP
    
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "asset": self.asset,
            "position": self.position_type,
            "notional": self.notional_value,
            "hedge": {
                "optimal_ratio": round(self.optimal_hedge_ratio, 3),
                "notional": round(self.hedge_notional, 2),
                "instrument": self.hedge_instrument
            },
            "costs": {
                "basis_risk_bps": round(self.estimated_basis_risk, 1),
                "roll_cost_pct": round(self.estimated_roll_cost, 3),
                "total_annualized_bps": round(self.total_hedge_cost, 1)
            },
            "effectiveness": {
                "variance_reduction_pct": round(self.expected_variance_reduction, 1),
                "r_squared": round(self.r_squared, 3),
                "tracking_error_bps": round(self.tracking_error, 1)
            },
            "execution": {
                "urgency": self.urgency,
                "entry": self.entry_recommendation
            },
            "reasoning": self.reasoning
        }


@dataclass
class HedgeEffectivenessReport:
    """Hedge effectiveness analysis"""
    asset: str
    period: str
    
    # Performance
    unhedged_pnl: float
    hedged_pnl: float
    hedge_pnl: float
    net_benefit: float
    
    # Effectiveness metrics
    effectiveness_ratio: float  # Hedge P&L / Spot P&L
    variance_reduction: float
    correlation: float
    
    # Basis analysis
    basis_mean: float
    basis_std: float
    basis_range: Tuple[float, float]
    
    def to_dict(self) -> Dict:
        return {
            "asset": self.asset,
            "period": self.period,
            "pnl": {
                "unhedged": round(self.unhedged_pnl, 2),
                "hedged": round(self.hedged_pnl, 2),
                "hedge_instrument": round(self.hedge_pnl, 2),
                "net_benefit": round(self.net_benefit, 2)
            },
            "effectiveness": {
                "ratio": round(self.effectiveness_ratio, 3),
                "variance_reduction": round(self.variance_reduction, 1),
                "correlation": round(self.correlation, 3)
            },
            "basis": {
                "mean": round(self.basis_mean, 3),
                "std": round(self.basis_std, 3),
                "range": [round(self.basis_range[0], 3), round(self.basis_range[1], 3)]
            }
        }


# Asset to futures mapping
HEDGE_INSTRUMENTS = {
    "CRUDE": {"futures": "@CL#C", "exchange": "NYMEX", "contract_size": 1000, "unit": "barrels"},
    "GOLD": {"futures": "@GC#C", "exchange": "COMEX", "contract_size": 100, "unit": "oz"},
    "SP500": {"futures": "@ES#C", "exchange": "CME", "contract_size": 50, "unit": "index"},
    "NASDAQ": {"futures": "@NQ#C", "exchange": "CME", "contract_size": 20, "unit": "index"},
    "BITCOIN": {"futures": "BTC", "exchange": "CME", "contract_size": 5, "unit": "BTC"},
}

# Historical basis and roll statistics (from research)
HEDGE_STATS = {
    "CRUDE": {"avg_basis": 0.15, "basis_vol": 0.8, "roll_cost": 0.12, "r_squared": 0.95},
    "GOLD": {"avg_basis": 0.08, "basis_vol": 0.3, "roll_cost": 0.05, "r_squared": 0.98},
    "SP500": {"avg_basis": 0.02, "basis_vol": 0.1, "roll_cost": 0.02, "r_squared": 0.99},
    "NASDAQ": {"avg_basis": 0.03, "basis_vol": 0.12, "roll_cost": 0.02, "r_squared": 0.99},
    "BITCOIN": {"avg_basis": 0.5, "basis_vol": 2.0, "roll_cost": 0.25, "r_squared": 0.92},
}


def load_price_data(asset: str) -> Optional[pd.DataFrame]:
    """Load price data for hedge calculations"""
    file_map = {
        "CRUDE": "Crude_Oil.csv",
        "GOLD": "GOLD.csv",
        "SP500": "SP500.csv",
        "NASDAQ": "NASDAQ.csv",
        "BITCOIN": "Bitcoin.csv",
    }
    
    file_name = file_map.get(asset)
    if not file_name:
        return None
    
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        date_col = "date" if "date" in df.columns else "time"
        df["date"] = pd.to_datetime(df[date_col])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


def calculate_optimal_hedge_ratio(spot_returns: np.ndarray, futures_returns: np.ndarray) -> Tuple[float, float]:
    """
    Calculate minimum variance hedge ratio using OLS regression.
    h* = Cov(S,F) / Var(F)
    """
    if len(spot_returns) != len(futures_returns) or len(spot_returns) < 30:
        return 1.0, 0.0
    
    covariance = np.cov(spot_returns, futures_returns)[0, 1]
    futures_variance = np.var(futures_returns)
    
    if futures_variance == 0:
        return 1.0, 0.0
    
    hedge_ratio = covariance / futures_variance
    
    # Calculate R-squared
    correlation = np.corrcoef(spot_returns, futures_returns)[0, 1]
    r_squared = correlation ** 2
    
    return hedge_ratio, r_squared


def calculate_hedge_effectiveness(
    spot_returns: np.ndarray, 
    futures_returns: np.ndarray,
    hedge_ratio: float
) -> float:
    """Calculate variance reduction from hedging"""
    unhedged_var = np.var(spot_returns)
    hedged_returns = spot_returns - hedge_ratio * futures_returns
    hedged_var = np.var(hedged_returns)
    
    if unhedged_var == 0:
        return 0.0
    
    variance_reduction = (1 - hedged_var / unhedged_var) * 100
    return max(0, variance_reduction)


def generate_hedge_recommendation(
    asset: str,
    position_type: str,  # "LONG" or "SHORT"
    notional_value: float,
    current_volatility: Optional[float] = None
) -> Optional[HedgeRecommendation]:
    """Generate optimal hedge recommendation"""
    
    if asset not in HEDGE_INSTRUMENTS:
        return None
    
    # Load price data
    df = load_price_data(asset)
    if df is None or len(df) < 60:
        # Use default stats
        stats = HEDGE_STATS.get(asset, HEDGE_STATS["SP500"])
        hedge_ratio = 1.0
        r_squared = stats["r_squared"]
    else:
        # Calculate from data
        prices = df["close"].values
        returns = np.diff(np.log(prices))
        
        # Assume futures track spot closely (use same returns with slight noise)
        # In production, would load actual futures data
        futures_returns = returns * (1 + np.random.randn(len(returns)) * 0.01)
        
        hedge_ratio, r_squared = calculate_optimal_hedge_ratio(returns[-252:], futures_returns[-252:])
        current_volatility = np.std(returns[-20:]) * np.sqrt(252) * 100
    
    # Get instrument info
    instrument = HEDGE_INSTRUMENTS[asset]
    stats = HEDGE_STATS.get(asset, HEDGE_STATS["SP500"])
    
    # Calculate hedge size
    hedge_notional = notional_value * hedge_ratio
    
    # Estimate costs
    basis_risk = stats["basis_vol"] * 100  # Convert to bps
    roll_cost = stats["roll_cost"]
    
    # Assume 4 rolls per year for most commodities
    rolls_per_year = 4 if asset in ["CRUDE", "GOLD"] else 12
    annual_roll_cost = roll_cost * rolls_per_year
    
    total_cost = basis_risk + annual_roll_cost * 100  # bps
    
    # Calculate effectiveness
    variance_reduction = r_squared * 100  # Approximate
    tracking_error = stats["basis_vol"] * 100
    
    # Determine urgency based on volatility
    if current_volatility and current_volatility > 30:
        urgency = "IMMEDIATE"
        entry = "MARKET"
    elif current_volatility and current_volatility > 20:
        urgency = "TODAY"
        entry = "TWAP"
    else:
        urgency = "THIS_WEEK"
        entry = "LIMIT"
    
    # Build reasoning
    reasoning = []
    reasoning.append(f"Optimal hedge ratio: {hedge_ratio:.3f} (R² = {r_squared:.2f})")
    reasoning.append(f"Expected variance reduction: {variance_reduction:.0f}%")
    reasoning.append(f"Estimated annual cost: {total_cost:.0f}bps (basis {basis_risk:.0f}bps + rolls {annual_roll_cost*100:.0f}bps)")
    
    if hedge_ratio > 1.1:
        reasoning.append("⚠️ Hedge ratio > 1.0 suggests futures more volatile than spot")
    elif hedge_ratio < 0.9:
        reasoning.append("📊 Under-hedging may be optimal due to basis correlation")
    
    if position_type == "LONG":
        reasoning.append(f"Sell {instrument['futures']} to hedge long {asset} exposure")
    else:
        reasoning.append(f"Buy {instrument['futures']} to hedge short {asset} exposure")
    
    return HedgeRecommendation(
        asset=asset,
        position_type=position_type,
        notional_value=notional_value,
        optimal_hedge_ratio=hedge_ratio,
        hedge_notional=hedge_notional,
        hedge_instrument=instrument["futures"],
        estimated_basis_risk=basis_risk,
        estimated_roll_cost=roll_cost,
        total_hedge_cost=total_cost,
        expected_variance_reduction=variance_reduction,
        r_squared=r_squared,
        tracking_error=tracking_error,
        urgency=urgency,
        entry_recommendation=entry,
        reasoning=reasoning
    )


def calculate_hedge_effectiveness_report(
    asset: str,
    hedge_ratio: float,
    period_days: int = 30
) -> Optional[HedgeEffectivenessReport]:
    """Calculate hedge effectiveness over a period"""
    
    df = load_price_data(asset)
    if df is None or len(df) < period_days:
        return None
    
    prices = df["close"].values[-period_days:]
    
    # Simulate futures (in production, use actual futures data)
    stats = HEDGE_STATS.get(asset, HEDGE_STATS["SP500"])
    basis = np.random.randn(len(prices)) * stats["basis_vol"] / 100
    futures_prices = prices * (1 + basis)
    
    # Calculate P&L
    spot_return = (prices[-1] / prices[0] - 1) * 100
    futures_return = (futures_prices[-1] / futures_prices[0] - 1) * 100
    
    # Assume long spot, short futures hedge
    unhedged_pnl = spot_return
    hedge_pnl = -futures_return * hedge_ratio
    hedged_pnl = unhedged_pnl + hedge_pnl
    
    # Effectiveness
    if abs(spot_return) > 0.01:
        effectiveness = -hedge_pnl / spot_return
    else:
        effectiveness = 0
    
    # Correlation and variance
    spot_returns = np.diff(np.log(prices))
    futures_returns = np.diff(np.log(futures_prices))
    correlation = np.corrcoef(spot_returns, futures_returns)[0, 1]
    
    hedged_returns = spot_returns - hedge_ratio * futures_returns
    variance_reduction = (1 - np.var(hedged_returns) / np.var(spot_returns)) * 100
    
    # Basis analysis
    basis_series = (prices - futures_prices) / prices * 100
    
    return HedgeEffectivenessReport(
        asset=asset,
        period=f"{period_days} days",
        unhedged_pnl=unhedged_pnl,
        hedged_pnl=hedged_pnl,
        hedge_pnl=hedge_pnl,
        net_benefit=unhedged_pnl - hedged_pnl,
        effectiveness_ratio=effectiveness,
        variance_reduction=variance_reduction,
        correlation=correlation,
        basis_mean=np.mean(basis_series),
        basis_std=np.std(basis_series),
        basis_range=(np.min(basis_series), np.max(basis_series))
    )


def get_hedge_for_api(asset: str, position: str = "LONG", notional: float = 1000000) -> Dict:
    """Get hedge recommendation for API"""
    rec = generate_hedge_recommendation(asset, position, notional)
    if rec is None:
        return {"error": f"Hedging not available for {asset}"}
    return rec.to_dict()


def get_effectiveness_for_api(asset: str, days: int = 30) -> Dict:
    """Get hedge effectiveness for API"""
    report = calculate_hedge_effectiveness_report(asset, 1.0, days)
    if report is None:
        return {"error": f"Effectiveness data not available for {asset}"}
    return report.to_dict()


if __name__ == "__main__":
    print("=" * 70)
    print("OPTIMAL HEDGE CALCULATOR")
    print("=" * 70)
    
    for asset in ["CRUDE", "GOLD", "SP500"]:
        rec = generate_hedge_recommendation(asset, "LONG", 1000000)
        if rec:
            print(f"\n{'-' * 50}")
            print(f"[{asset}] Hedging $1,000,000 LONG position")
            print(f"Optimal Ratio: {rec.optimal_hedge_ratio:.3f}")
            print(f"Hedge Size: ${rec.hedge_notional:,.0f}")
            print(f"Instrument: {rec.hedge_instrument}")
            print(f"Variance Reduction: {rec.expected_variance_reduction:.0f}%")
            print(f"Total Cost: {rec.total_hedge_cost:.0f}bps/year")
            print(f"Urgency: {rec.urgency}")
            print(f"\nReasoning:")
            for r in rec.reasoning:
                print(f"  - {r}")
