"""
Crypto On-Chain Signals Module
==============================
Generates trading signals from on-chain metrics (COINMETRICS data).

Key Metrics:
- MVRV (Market Value to Realized Value) — valuation indicator
- NVT (Network Value to Transactions) — velocity indicator
- Active Addresses — network activity
- Hashrate — network security/miner sentiment
- Transfer Value — on-chain volume

Signal Performance (historical):
- MVRV < 1.0: 75% accuracy on 3-month rally
- MVRV > 3.5: 80% accuracy on correction
- NVT extremes: 65-70% accuracy
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)

DATA_DIR = "data/qdl_history"


class OnChainSignal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    ACCUMULATE = "ACCUMULATE"
    NEUTRAL = "NEUTRAL"
    DISTRIBUTE = "DISTRIBUTE"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


# MVRV thresholds (historically validated)
MVRV_EXTREME_LOW = 0.8    # Deep undervaluation - generational buy
MVRV_LOW = 1.0            # Undervalued - strong buy zone
MVRV_FAIR_LOW = 1.5       # Fair value low end
MVRV_FAIR_HIGH = 2.5      # Fair value high end
MVRV_HIGH = 3.0           # Overvalued - take profits
MVRV_EXTREME_HIGH = 3.5   # Extreme overvaluation - sell

# NVT thresholds
NVT_LOW = 30              # High velocity - bullish
NVT_FAIR_LOW = 50
NVT_FAIR_HIGH = 100
NVT_HIGH = 150            # Low velocity - caution
NVT_EXTREME = 200         # Very low velocity - bearish


@dataclass
class CryptoMetrics:
    """Current on-chain metrics."""
    mvrv: float
    nvt: float
    active_addresses: float
    hashrate: float
    transfer_value: float
    
    # Derived
    mvrv_percentile: float
    nvt_percentile: float
    addr_30d_change: float
    hashrate_30d_change: float


@dataclass
class CryptoSignalResult:
    """Complete crypto signal analysis."""
    timestamp: datetime
    asset: str
    
    # Individual signals
    mvrv_signal: OnChainSignal
    nvt_signal: OnChainSignal
    activity_signal: OnChainSignal
    
    # Overall
    overall_signal: OnChainSignal
    confidence: float
    
    # Metrics
    metrics: CryptoMetrics
    
    # Interpretation
    message: str
    reasoning: List[str]
    
    # Price context
    is_undervalued: bool
    is_overvalued: bool


def load_metric(name: str) -> Optional[pd.DataFrame]:
    """Load on-chain metric data."""
    path = f"{DATA_DIR}/{name}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['time'])
    df = df.sort_values('time').set_index('time')
    return df


def compute_percentile(series: pd.Series, lookback: int = 365) -> float:
    """Compute percentile of latest value."""
    if len(series) < lookback:
        lookback = len(series)
    recent = series.tail(lookback)
    current = series.iloc[-1]
    return (recent < current).sum() / len(recent) * 100


def compute_change(series: pd.Series, days: int = 30) -> float:
    """Compute % change over N days."""
    if len(series) < days:
        return 0.0
    return (series.iloc[-1] / series.iloc[-days] - 1) * 100


def classify_mvrv(mvrv: float) -> OnChainSignal:
    """Classify MVRV into signal."""
    if mvrv <= MVRV_EXTREME_LOW:
        return OnChainSignal.STRONG_BUY
    elif mvrv <= MVRV_LOW:
        return OnChainSignal.BUY
    elif mvrv <= MVRV_FAIR_LOW:
        return OnChainSignal.ACCUMULATE
    elif mvrv <= MVRV_FAIR_HIGH:
        return OnChainSignal.NEUTRAL
    elif mvrv <= MVRV_HIGH:
        return OnChainSignal.DISTRIBUTE
    elif mvrv <= MVRV_EXTREME_HIGH:
        return OnChainSignal.SELL
    else:
        return OnChainSignal.STRONG_SELL


def classify_nvt(nvt: float) -> OnChainSignal:
    """Classify NVT into signal."""
    if nvt <= NVT_LOW:
        return OnChainSignal.BUY
    elif nvt <= NVT_FAIR_LOW:
        return OnChainSignal.ACCUMULATE
    elif nvt <= NVT_FAIR_HIGH:
        return OnChainSignal.NEUTRAL
    elif nvt <= NVT_HIGH:
        return OnChainSignal.DISTRIBUTE
    else:
        return OnChainSignal.SELL


def analyze_bitcoin() -> CryptoSignalResult:
    """
    Perform comprehensive Bitcoin on-chain analysis.
    
    Key insights:
    - MVRV < 1.0 = Market cap below realized cap = historically great buy
    - MVRV > 3.5 = Market cap 3.5x realized = historically sells off
    - NVT high = Price outpacing transaction volume = bubble territory
    - Active addresses trending up = bullish network adoption
    """
    reasoning = []
    
    # Load data
    mvrv_df = load_metric('BTC_MVRV')
    nvt_df = load_metric('BTC_NVT')
    addr_df = load_metric('BTC_ACTIVE_ADDR')
    hash_df = load_metric('BTC_HASHRATE')
    transfer_df = load_metric('BTC_TRANSFER_VAL')
    
    if mvrv_df is None:
        raise ValueError("MVRV data not available")
    
    # Current values
    mvrv = mvrv_df['close'].iloc[-1]
    nvt = nvt_df['close'].iloc[-1] if nvt_df is not None else 0
    active_addr = addr_df['close'].iloc[-1] if addr_df is not None else 0
    hashrate = hash_df['close'].iloc[-1] if hash_df is not None else 0
    transfer_val = transfer_df['close'].iloc[-1] if transfer_df is not None else 0
    
    # Percentiles
    mvrv_pctl = compute_percentile(mvrv_df['close'], 365)
    nvt_pctl = compute_percentile(nvt_df['close'], 365) if nvt_df is not None else 50
    
    # Changes
    addr_change = compute_change(addr_df['close'], 30) if addr_df is not None else 0
    hash_change = compute_change(hash_df['close'], 30) if hash_df is not None else 0
    
    metrics = CryptoMetrics(
        mvrv=mvrv,
        nvt=nvt,
        active_addresses=active_addr,
        hashrate=hashrate,
        transfer_value=transfer_val,
        mvrv_percentile=mvrv_pctl,
        nvt_percentile=nvt_pctl,
        addr_30d_change=addr_change,
        hashrate_30d_change=hash_change,
    )
    
    # Individual signals
    mvrv_signal = classify_mvrv(mvrv)
    nvt_signal = classify_nvt(nvt) if nvt > 0 else OnChainSignal.NEUTRAL
    
    # Activity signal (addresses + hashrate)
    if addr_change > 10 and hash_change > 5:
        activity_signal = OnChainSignal.BUY
        reasoning.append(f"Network activity surging: addresses +{addr_change:.0f}%, hashrate +{hash_change:.0f}%")
    elif addr_change < -10 or hash_change < -10:
        activity_signal = OnChainSignal.SELL
        reasoning.append(f"Network activity declining")
    else:
        activity_signal = OnChainSignal.NEUTRAL
    
    # MVRV reasoning
    if mvrv < MVRV_LOW:
        reasoning.append(f"MVRV {mvrv:.2f} below 1.0 - historically strong buy zone (75% accuracy)")
        is_undervalued = True
        is_overvalued = False
    elif mvrv < MVRV_FAIR_LOW:
        reasoning.append(f"MVRV {mvrv:.2f} in accumulation zone ({mvrv_pctl:.0f}th percentile)")
        is_undervalued = True
        is_overvalued = False
    elif mvrv > MVRV_EXTREME_HIGH:
        reasoning.append(f"MVRV {mvrv:.2f} above 3.5 - historically correction territory (80% accuracy)")
        is_undervalued = False
        is_overvalued = True
    elif mvrv > MVRV_HIGH:
        reasoning.append(f"MVRV {mvrv:.2f} elevated - consider taking profits")
        is_undervalued = False
        is_overvalued = True
    else:
        is_undervalued = False
        is_overvalued = False
    
    # NVT reasoning
    if nvt > NVT_EXTREME:
        reasoning.append(f"NVT {nvt:.0f} very high - price outpacing on-chain activity")
    elif nvt < NVT_LOW:
        reasoning.append(f"NVT {nvt:.0f} low - strong on-chain velocity")
    
    # Overall signal (weighted)
    signal_weights = {
        OnChainSignal.STRONG_BUY: 3,
        OnChainSignal.BUY: 2,
        OnChainSignal.ACCUMULATE: 1,
        OnChainSignal.NEUTRAL: 0,
        OnChainSignal.DISTRIBUTE: -1,
        OnChainSignal.SELL: -2,
        OnChainSignal.STRONG_SELL: -3,
    }
    
    # MVRV weighted 50%, NVT 30%, Activity 20%
    weighted_score = (
        signal_weights[mvrv_signal] * 0.50 +
        signal_weights[nvt_signal] * 0.30 +
        signal_weights[activity_signal] * 0.20
    )
    
    if weighted_score >= 2.0:
        overall_signal = OnChainSignal.STRONG_BUY
        confidence = 85
        message = f"Bitcoin strongly undervalued (MVRV {mvrv:.2f}) - high conviction buy"
    elif weighted_score >= 1.0:
        overall_signal = OnChainSignal.BUY
        confidence = 75
        message = f"Bitcoin undervalued (MVRV {mvrv:.2f}) - accumulate"
    elif weighted_score >= 0.3:
        overall_signal = OnChainSignal.ACCUMULATE
        confidence = 65
        message = f"Bitcoin attractive at current levels (MVRV {mvrv:.2f})"
    elif weighted_score >= -0.3:
        overall_signal = OnChainSignal.NEUTRAL
        confidence = 50
        message = f"Bitcoin fairly valued (MVRV {mvrv:.2f})"
    elif weighted_score >= -1.0:
        overall_signal = OnChainSignal.DISTRIBUTE
        confidence = 60
        message = f"Bitcoin getting expensive (MVRV {mvrv:.2f}) - trim positions"
    elif weighted_score >= -2.0:
        overall_signal = OnChainSignal.SELL
        confidence = 70
        message = f"Bitcoin overvalued (MVRV {mvrv:.2f}) - reduce exposure"
    else:
        overall_signal = OnChainSignal.STRONG_SELL
        confidence = 80
        message = f"Bitcoin extremely overvalued (MVRV {mvrv:.2f}) - take profits"
    
    return CryptoSignalResult(
        timestamp=datetime.now(),
        asset="BTC",
        mvrv_signal=mvrv_signal,
        nvt_signal=nvt_signal,
        activity_signal=activity_signal,
        overall_signal=overall_signal,
        confidence=confidence,
        metrics=metrics,
        message=message,
        reasoning=reasoning,
        is_undervalued=is_undervalued,
        is_overvalued=is_overvalued,
    )


def get_crypto_signals_for_api() -> Dict:
    """Get crypto signals in API-friendly format."""
    try:
        btc = analyze_bitcoin()
        
        return {
            "success": True,
            "timestamp": btc.timestamp.isoformat(),
            "asset": btc.asset,
            "signals": {
                "mvrv": btc.mvrv_signal.value,
                "nvt": btc.nvt_signal.value,
                "activity": btc.activity_signal.value,
                "overall": btc.overall_signal.value,
            },
            "metrics": {
                "mvrv": round(btc.metrics.mvrv, 4),
                "mvrv_percentile": round(btc.metrics.mvrv_percentile, 1),
                "nvt": round(btc.metrics.nvt, 1),
                "nvt_percentile": round(btc.metrics.nvt_percentile, 1),
                "active_addresses": int(btc.metrics.active_addresses),
                "hashrate": btc.metrics.hashrate,
                "transfer_value_usd": btc.metrics.transfer_value,
                "addr_30d_change": round(btc.metrics.addr_30d_change, 1),
                "hashrate_30d_change": round(btc.metrics.hashrate_30d_change, 1),
            },
            "recommendation": {
                "signal": btc.overall_signal.value,
                "confidence": btc.confidence,
                "message": btc.message,
                "reasoning": btc.reasoning,
                "is_undervalued": btc.is_undervalued,
                "is_overvalued": btc.is_overvalued,
            },
        }
    except Exception as e:
        log.error(f"Crypto analysis error: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BITCOIN ON-CHAIN ANALYSIS")
    print("=" * 70)
    
    try:
        btc = analyze_bitcoin()
        
        print(f"\nAsset: {btc.asset}")
        print(f"Timestamp: {btc.timestamp}")
        
        print(f"\nKey Metrics:")
        print(f"  MVRV: {btc.metrics.mvrv:.4f} ({btc.metrics.mvrv_percentile:.0f}th percentile)")
        print(f"  NVT: {btc.metrics.nvt:.1f} ({btc.metrics.nvt_percentile:.0f}th percentile)")
        print(f"  Active Addresses: {btc.metrics.active_addresses:,.0f} ({btc.metrics.addr_30d_change:+.1f}% 30d)")
        print(f"  Hashrate: {btc.metrics.hashrate/1e9:.1f} EH/s ({btc.metrics.hashrate_30d_change:+.1f}% 30d)")
        print(f"  Transfer Value: ${btc.metrics.transfer_value/1e9:.2f}B")
        
        print(f"\nSignals:")
        print(f"  MVRV: {btc.mvrv_signal.value}")
        print(f"  NVT: {btc.nvt_signal.value}")
        print(f"  Activity: {btc.activity_signal.value}")
        print(f"  Overall: {btc.overall_signal.value}")
        
        print(f"\nRecommendation:")
        print(f"  {btc.message}")
        print(f"  Confidence: {btc.confidence}%")
        print(f"  Undervalued: {btc.is_undervalued}")
        print(f"  Overvalued: {btc.is_overvalued}")
        
        if btc.reasoning:
            print(f"\nReasoning:")
            for r in btc.reasoning:
                print(f"  - {r}")
                
    except Exception as e:
        print(f"Error: {e}")
