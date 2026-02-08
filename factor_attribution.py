"""
Factor Attribution Panel — Institutional Intelligence
======================================================
Decomposes portfolio returns into factor exposures.
Essential for institutional compliance and performance attribution.

Factors:
- Market (Beta)
- Rates (Duration)
- Oil/Commodities
- USD/Currency
- Volatility (VIX)
- Credit

Author: AmiraB
Created: 2026-02-08
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats

DATA_DIR = Path("data/qdl_history")


@dataclass
class FactorExposure:
    """Single factor exposure"""
    name: str
    beta: float           # Sensitivity to factor
    contribution: float   # % of return explained
    t_stat: float         # Statistical significance
    current_level: str    # HIGH, NORMAL, LOW
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "beta": round(self.beta, 3),
            "contribution_pct": round(self.contribution, 1),
            "t_stat": round(self.t_stat, 2),
            "significance": "***" if abs(self.t_stat) > 2.58 else "**" if abs(self.t_stat) > 1.96 else "*" if abs(self.t_stat) > 1.65 else "",
            "level": self.current_level
        }


@dataclass
class FactorAttributionResult:
    """Complete factor attribution analysis"""
    timestamp: datetime
    asset: str
    period: str
    
    # Total explained
    r_squared: float
    total_return: float
    factor_return: float
    alpha: float  # Unexplained return (skill or luck)
    
    # Risk decomposition
    factor_risk_pct: float    # % of risk from factors
    specific_risk_pct: float  # % of risk idiosyncratic
    
    # Individual factors (with default)
    factors: List[FactorExposure] = field(default_factory=list)
    
    # Key insights (with defaults)
    dominant_factor: str = ""
    risk_warning: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "period": self.period,
            "summary": {
                "r_squared": round(self.r_squared, 3),
                "total_return": round(self.total_return, 2),
                "factor_return": round(self.factor_return, 2),
                "alpha": round(self.alpha, 2)
            },
            "factors": [f.to_dict() for f in self.factors],
            "risk_decomposition": {
                "factor_risk_pct": round(self.factor_risk_pct, 1),
                "specific_risk_pct": round(self.specific_risk_pct, 1)
            },
            "insights": {
                "dominant_factor": self.dominant_factor,
                "risk_warning": self.risk_warning
            }
        }


def load_factor_data() -> Dict[str, pd.Series]:
    """Load data for each factor"""
    factors = {}
    
    # Market factor (SP500)
    sp500_file = DATA_DIR / "SP500.csv"
    if sp500_file.exists():
        df = pd.read_csv(sp500_file)
        date_col = "date" if "date" in df.columns else "time"
        df["date"] = pd.to_datetime(df[date_col])
        df = df.sort_values("date")
        factors["Market"] = df.set_index("date")["close"].pct_change()
    
    # Oil factor
    oil_file = DATA_DIR / "Crude_Oil.csv"
    if oil_file.exists():
        df = pd.read_csv(oil_file)
        date_col = "date" if "date" in df.columns else "time"
        df["date"] = pd.to_datetime(df[date_col])
        df = df.sort_values("date")
        factors["Oil"] = df.set_index("date")["close"].pct_change()
    
    # Gold as rates proxy (inverse relationship)
    gold_file = DATA_DIR / "GOLD.csv"
    if gold_file.exists():
        df = pd.read_csv(gold_file)
        date_col = "date" if "date" in df.columns else "time"
        df["date"] = pd.to_datetime(df[date_col])
        df = df.sort_values("date")
        factors["Rates"] = -df.set_index("date")["close"].pct_change()  # Inverted
    
    # Crypto as risk appetite
    btc_file = DATA_DIR / "Bitcoin.csv"
    if btc_file.exists():
        df = pd.read_csv(btc_file)
        date_col = "date" if "date" in df.columns else "time"
        df["date"] = pd.to_datetime(df[date_col])
        df = df.sort_values("date")
        factors["RiskAppetite"] = df.set_index("date")["close"].pct_change()
    
    return factors


def run_factor_regression(
    asset_returns: pd.Series,
    factor_returns: Dict[str, pd.Series],
    lookback: int = 252
) -> Tuple[Dict[str, Tuple[float, float]], float, float]:
    """
    Run multivariate regression of asset on factors.
    Returns: (factor_betas, r_squared, alpha)
    """
    # Align all series
    data = {"asset": asset_returns}
    for name, series in factor_returns.items():
        data[name] = series
    
    df = pd.DataFrame(data).dropna().tail(lookback)
    
    if len(df) < 60:
        return {}, 0.0, 0.0
    
    y = df["asset"].values
    X = df.drop("asset", axis=1).values
    
    # Add constant for alpha
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    try:
        # OLS regression
        beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
        
        alpha = beta[0] * 252  # Annualized
        factor_betas = {name: beta[i+1] for i, name in enumerate(factor_returns.keys())}
        
        # Calculate R-squared
        y_pred = X_with_const @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate t-stats
        n = len(y)
        k = X.shape[1] + 1
        mse = ss_res / (n - k) if n > k else 1
        var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
        se_beta = np.sqrt(var_beta)
        t_stats = beta / se_beta
        
        factor_t_stats = {name: t_stats[i+1] for i, name in enumerate(factor_returns.keys())}
        
        return factor_betas, factor_t_stats, r_squared, alpha
        
    except Exception:
        return {}, {}, 0.0, 0.0


def analyze_factor_attribution(asset: str, period_days: int = 252) -> Optional[FactorAttributionResult]:
    """Run full factor attribution analysis"""
    
    # Load asset data
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
        df = df.sort_values("date")
        asset_returns = df.set_index("date")["close"].pct_change()
    except Exception:
        return None
    
    # Load factors
    factor_data = load_factor_data()
    
    # Remove self from factors (if asset is a factor)
    if asset == "SP500" and "Market" in factor_data:
        del factor_data["Market"]
    if asset == "CRUDE" and "Oil" in factor_data:
        del factor_data["Oil"]
    if asset == "GOLD" and "Rates" in factor_data:
        del factor_data["Rates"]
    if asset == "BITCOIN" and "RiskAppetite" in factor_data:
        del factor_data["RiskAppetite"]
    
    if not factor_data:
        return None
    
    # Run regression
    betas, t_stats, r_squared, alpha = run_factor_regression(asset_returns, factor_data, period_days)
    
    if not betas:
        return None
    
    # Calculate total return
    total_return = (1 + asset_returns.tail(period_days)).prod() - 1
    total_return_pct = total_return * 100
    
    # Build factor exposures
    factors = []
    total_contribution = 0
    
    for name, beta in betas.items():
        t_stat = t_stats.get(name, 0)
        
        # Calculate contribution (approximate)
        factor_return = factor_data[name].tail(period_days).mean() * period_days
        contribution = beta * factor_return / total_return * 100 if total_return != 0 else 0
        total_contribution += contribution
        
        # Determine level
        if abs(beta) > 1.5:
            level = "HIGH"
        elif abs(beta) < 0.5:
            level = "LOW"
        else:
            level = "NORMAL"
        
        factors.append(FactorExposure(
            name=name,
            beta=beta,
            contribution=contribution,
            t_stat=t_stat,
            current_level=level
        ))
    
    # Sort by absolute contribution
    factors.sort(key=lambda x: abs(x.contribution), reverse=True)
    
    # Calculate factor vs specific risk
    factor_risk_pct = r_squared * 100
    specific_risk_pct = (1 - r_squared) * 100
    
    # Identify dominant factor
    dominant = factors[0] if factors else None
    dominant_factor = f"{dominant.name} (beta={dominant.beta:.2f})" if dominant else "None"
    
    # Risk warning
    risk_warning = ""
    high_exposure = [f for f in factors if f.current_level == "HIGH"]
    if high_exposure:
        risk_warning = f"High exposure to: {', '.join(f.name for f in high_exposure)}"
    
    factor_return = total_return_pct * r_squared
    
    return FactorAttributionResult(
        timestamp=datetime.now(),
        asset=asset,
        period=f"{period_days} days",
        r_squared=r_squared,
        total_return=total_return_pct,
        factor_return=factor_return,
        alpha=alpha * 100,
        factors=factors,
        factor_risk_pct=factor_risk_pct,
        specific_risk_pct=specific_risk_pct,
        dominant_factor=dominant_factor,
        risk_warning=risk_warning
    )


def get_attribution_for_api(asset: str, days: int = 252) -> Dict:
    """Get factor attribution for API"""
    result = analyze_factor_attribution(asset, days)
    if result is None:
        return {"error": f"Factor attribution not available for {asset}"}
    return result.to_dict()


if __name__ == "__main__":
    print("=" * 70)
    print("FACTOR ATTRIBUTION ANALYSIS")
    print("=" * 70)
    
    for asset in ["SP500", "CRUDE", "GOLD", "BITCOIN", "NASDAQ"]:
        result = analyze_factor_attribution(asset)
        if result:
            print(f"\n{'-' * 50}")
            print(f"[{asset}] Factor Attribution (252 days)")
            print(f"R-squared: {result.r_squared:.1%}")
            print(f"Total Return: {result.total_return:.1f}%")
            print(f"Factor Return: {result.factor_return:.1f}%")
            print(f"Alpha: {result.alpha:.2f}%")
            print(f"\nFactor Exposures:")
            for f in result.factors:
                sig = "***" if abs(f.t_stat) > 2.58 else "**" if abs(f.t_stat) > 1.96 else "*" if abs(f.t_stat) > 1.65 else ""
                print(f"  {f.name}: beta={f.beta:.3f}, contrib={f.contribution:.1f}%, t={f.t_stat:.2f}{sig}")
            print(f"\nRisk: {result.factor_risk_pct:.0f}% factor, {result.specific_risk_pct:.0f}% specific")
            if result.risk_warning:
                print(f"⚠️ {result.risk_warning}")
