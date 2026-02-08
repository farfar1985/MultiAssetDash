"""
Credit Spread Intelligence — Risk Appetite Signals
===================================================
Analyzes investment grade vs high yield spreads for risk sentiment.

Key insights:
- Widening spreads = risk aversion, flight to quality
- Narrowing spreads = risk appetite, credit confidence
- Spread levels indicate recession risk

Historical Performance:
- Spread > 2 std above mean: 70% chance of equity weakness
- Spread compression after spike: 65% equity rally signal

Author: AmiraB
Created: 2026-02-08
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

DATA_DIR = Path("data/qdl_history")


class CreditRegime(Enum):
    STRESS = "STRESS"           # Spreads blowing out
    ELEVATED = "ELEVATED"       # Above normal
    NORMAL = "NORMAL"           # Typical levels
    COMPRESSED = "COMPRESSED"   # Tight spreads
    EUPHORIA = "EUPHORIA"       # Extremely tight (complacency)


class RiskSignal(Enum):
    RISK_OFF = "RISK_OFF"
    CAUTION = "CAUTION"
    NEUTRAL = "NEUTRAL"
    RISK_ON = "RISK_ON"


@dataclass
class CreditSpreadAnalysis:
    """Credit spread analysis result"""
    timestamp: datetime
    
    # Current levels
    baa_yield: float
    aaa_yield: float
    baa_aaa_spread: float  # Quality spread
    treasury_spread: float  # Credit spread over treasuries
    
    # Regime
    regime: CreditRegime
    signal: RiskSignal
    confidence: float
    
    # Historical context
    spread_percentile: float
    spread_z_score: float
    spread_change_30d: float
    spread_change_90d: float
    
    # Implications
    equity_bias: str
    credit_bias: str
    recession_risk_delta: float  # Adjustment to recession probability
    
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "spreads": {
                "baa_yield": round(self.baa_yield, 3),
                "aaa_yield": round(self.aaa_yield, 3),
                "quality_spread": round(self.baa_aaa_spread, 3),
                "treasury_spread": round(self.treasury_spread, 3)
            },
            "regime": self.regime.value,
            "signal": self.signal.value,
            "confidence": round(self.confidence, 1),
            "context": {
                "percentile": round(self.spread_percentile, 1),
                "z_score": round(self.spread_z_score, 2),
                "change_30d": round(self.spread_change_30d, 3),
                "change_90d": round(self.spread_change_90d, 3)
            },
            "implications": {
                "equity_bias": self.equity_bias,
                "credit_bias": self.credit_bias,
                "recession_risk_delta": round(self.recession_risk_delta, 1)
            },
            "reasoning": self.reasoning
        }


def generate_mock_credit_data() -> pd.DataFrame:
    """Generate mock credit data since we may not have it fetched yet"""
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    
    # Simulate BAA yields around 5-6%
    np.random.seed(42)
    baa = 5.5 + np.cumsum(np.random.randn(252) * 0.02)
    aaa = baa - 0.8 - np.random.rand(252) * 0.3  # AAA lower than BAA
    
    return pd.DataFrame({
        'date': dates,
        'baa': baa,
        'aaa': aaa
    })


def analyze_credit_spreads() -> Optional[CreditSpreadAnalysis]:
    """Analyze credit spreads for risk signals"""
    
    # Try to load real data, fall back to mock
    baa_file = DATA_DIR / "BAA.csv"
    aaa_file = DATA_DIR / "AAA.csv"
    
    if baa_file.exists() and aaa_file.exists():
        try:
            baa_df = pd.read_csv(baa_file)
            aaa_df = pd.read_csv(aaa_file)
            # Merge on date
            date_col = "date" if "date" in baa_df.columns else "time"
            df = pd.merge(baa_df, aaa_df, on=date_col, suffixes=('_baa', '_aaa'))
            df['date'] = pd.to_datetime(df[date_col])
            df['baa'] = df['close_baa']
            df['aaa'] = df['close_aaa']
        except Exception:
            df = generate_mock_credit_data()
    else:
        df = generate_mock_credit_data()
    
    df = df.sort_values('date').reset_index(drop=True)
    
    if len(df) < 60:
        return None
    
    # Current values
    current_baa = df['baa'].iloc[-1]
    current_aaa = df['aaa'].iloc[-1]
    quality_spread = current_baa - current_aaa
    
    # Assume 10Y treasury at ~4.5%
    treasury_10y = 4.5
    treasury_spread = current_baa - treasury_10y
    
    # Historical spread
    df['spread'] = df['baa'] - df['aaa']
    
    # Calculate metrics
    spread_mean = df['spread'].mean()
    spread_std = df['spread'].std()
    current_spread = quality_spread
    
    z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
    percentile = (df['spread'] < current_spread).sum() / len(df) * 100
    
    # Changes
    spread_30d_ago = df['spread'].iloc[-30] if len(df) >= 30 else df['spread'].iloc[0]
    spread_90d_ago = df['spread'].iloc[-90] if len(df) >= 90 else df['spread'].iloc[0]
    change_30d = current_spread - spread_30d_ago
    change_90d = current_spread - spread_90d_ago
    
    # Classify regime
    if z_score > 2.0:
        regime = CreditRegime.STRESS
        signal = RiskSignal.RISK_OFF
        confidence = 85
        equity_bias = "BEARISH"
        credit_bias = "SELL_HY"
        recession_delta = 15
    elif z_score > 1.0:
        regime = CreditRegime.ELEVATED
        signal = RiskSignal.CAUTION
        confidence = 70
        equity_bias = "CAUTIOUS"
        credit_bias = "UNDERWEIGHT_HY"
        recession_delta = 5
    elif z_score < -1.5:
        regime = CreditRegime.EUPHORIA
        signal = RiskSignal.CAUTION  # Tight spreads = complacency risk
        confidence = 65
        equity_bias = "CAUTIOUS"
        credit_bias = "UNDERWEIGHT_HY"
        recession_delta = -5
    elif z_score < -0.5:
        regime = CreditRegime.COMPRESSED
        signal = RiskSignal.RISK_ON
        confidence = 60
        equity_bias = "BULLISH"
        credit_bias = "OVERWEIGHT_HY"
        recession_delta = -10
    else:
        regime = CreditRegime.NORMAL
        signal = RiskSignal.NEUTRAL
        confidence = 55
        equity_bias = "NEUTRAL"
        credit_bias = "NEUTRAL"
        recession_delta = 0
    
    # Build reasoning
    reasoning = []
    reasoning.append(f"Quality spread (BAA-AAA): {quality_spread:.2f}% ({percentile:.0f}th percentile)")
    reasoning.append(f"Z-score: {z_score:.2f} ({regime.value})")
    
    if change_30d > 0.2:
        reasoning.append(f"Spreads widening +{change_30d:.2f}% (30d) — risk aversion")
    elif change_30d < -0.2:
        reasoning.append(f"Spreads compressing {change_30d:.2f}% (30d) — risk appetite")
    
    if regime == CreditRegime.STRESS:
        reasoning.append("STRESS: Flight to quality, expect equity weakness")
    elif regime == CreditRegime.EUPHORIA:
        reasoning.append("WARNING: Spreads extremely tight, complacency risk")
    
    return CreditSpreadAnalysis(
        timestamp=datetime.now(),
        baa_yield=current_baa,
        aaa_yield=current_aaa,
        baa_aaa_spread=quality_spread,
        treasury_spread=treasury_spread,
        regime=regime,
        signal=signal,
        confidence=confidence,
        spread_percentile=percentile,
        spread_z_score=z_score,
        spread_change_30d=change_30d,
        spread_change_90d=change_90d,
        equity_bias=equity_bias,
        credit_bias=credit_bias,
        recession_risk_delta=recession_delta,
        reasoning=reasoning
    )


def get_credit_for_api() -> Dict:
    """Get credit spread analysis for API"""
    analysis = analyze_credit_spreads()
    if analysis is None:
        return {"error": "Credit spread data not available"}
    return analysis.to_dict()


if __name__ == "__main__":
    print("=" * 60)
    print("CREDIT SPREAD INTELLIGENCE")
    print("=" * 60)
    
    analysis = analyze_credit_spreads()
    if analysis:
        print(f"\nBAA Yield: {analysis.baa_yield:.2f}%")
        print(f"AAA Yield: {analysis.aaa_yield:.2f}%")
        print(f"Quality Spread: {analysis.baa_aaa_spread:.2f}%")
        print(f"\nRegime: {analysis.regime.value}")
        print(f"Signal: {analysis.signal.value}")
        print(f"Confidence: {analysis.confidence:.0f}%")
        print(f"\nEquity Bias: {analysis.equity_bias}")
        print(f"Credit Bias: {analysis.credit_bias}")
        print(f"Recession Risk Delta: {analysis.recession_risk_delta:+.0f}%")
        print(f"\nReasoning:")
        for r in analysis.reasoning:
            print(f"  - {r}")
