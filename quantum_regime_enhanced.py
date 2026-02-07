"""
Enhanced Quantum Regime Detection using QDT Data Lake
======================================================
Multi-factor regime detection using:
- Price data (momentum, volatility)
- VIX (market fear index)
- Cross-asset correlations (risk-on/risk-off)
- Regime persistence and transition probabilities

This provides a richer regime classification than single-asset analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from qdl_client import qdl_get_data

log = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = "data/qdl_history"

# VIX thresholds (historical percentiles)
VIX_LOW = 15      # Below = Complacency
VIX_NORMAL = 20   # Normal range
VIX_ELEVATED = 25 # Elevated fear
VIX_HIGH = 30     # High fear
VIX_EXTREME = 40  # Extreme fear / crisis

# Enhanced regime states
REGIME_STATES = {
    0: 'STRONG_BULL',     # Strong uptrend + low VIX
    1: 'BULL',            # Uptrend
    2: 'WEAK_BULL',       # Uptrend but elevated VIX
    3: 'SIDEWAYS',        # Consolidation
    4: 'WEAK_BEAR',       # Downtrend but VIX normalizing
    5: 'BEAR',            # Downtrend
    6: 'STRONG_BEAR',     # Strong downtrend + high VIX
    7: 'CRISIS',          # Extreme volatility / capitulation
    8: 'RECOVERY',        # Coming out of crisis
}

# Asset symbols for cross-correlation
REGIME_ASSETS = {
    'SP500': ('@ES#C', 'DTNIQ'),
    'VIX': ('@VX#C', 'DTNIQ'),
    'GOLD': ('QGC#', 'DTNIQ'),
    'US_DOLLAR': ('@DX#C', 'DTNIQ'),
    'Treasury_10Y': ('@TN#C', 'DTNIQ'),
    'Crude_Oil': ('QCL#', 'DTNIQ'),
    'Bitcoin': ('@BTC#C', 'DTNIQ'),
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RegimeState:
    """Enhanced regime state with multi-factor context."""
    regime_id: int
    regime_name: str
    confidence: float          # 0-100 confidence in classification
    vix_level: float
    vix_regime: str            # 'low', 'normal', 'elevated', 'high', 'extreme'
    momentum_20d: float
    momentum_60d: float
    volatility_20d: float
    volatility_60d: float
    risk_appetite: float       # -1 (risk-off) to +1 (risk-on)
    regime_duration: int       # Days in current regime
    transition_prob: Dict[str, float]  # Probabilities to other regimes


@dataclass
class MarketContext:
    """Overall market context from multiple assets."""
    timestamp: datetime
    sp500_regime: RegimeState
    cross_correlations: Dict[str, float]
    risk_on_score: float       # -100 to +100
    leading_indicator: str     # Which asset is leading
    divergences: List[str]     # Any notable divergences


# ============================================================================
# Data Loading
# ============================================================================

def load_asset_data(asset_name: str, years: int = 5) -> Optional[pd.DataFrame]:
    """Load asset data from CSV or fetch from QDL."""
    csv_path = f"{DATA_DIR}/{asset_name}.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['time'])
        df = df.sort_values('time').set_index('time')
        return df
    
    # Try to fetch from QDL
    if asset_name in REGIME_ASSETS:
        symbol, provider = REGIME_ASSETS[asset_name]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        try:
            df = qdl_get_data(symbol, provider, start_date, end_date)
            if not df.empty:
                return df
        except Exception as e:
            log.warning(f"Failed to fetch {asset_name}: {e}")
    
    return None


def load_vix_data(years: int = 5) -> Optional[pd.DataFrame]:
    """Load VIX data, trying CSV first then QDL."""
    csv_path = f"{DATA_DIR}/VIX.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['time'])
        df = df.sort_values('time').set_index('time')
        return df
    
    # Fetch from QDL
    symbol, provider = REGIME_ASSETS['VIX']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    try:
        df = qdl_get_data(symbol, provider, start_date, end_date)
        if not df.empty:
            # Save for future use
            os.makedirs(DATA_DIR, exist_ok=True)
            df.to_csv(csv_path)
            return df
    except Exception as e:
        log.warning(f"Failed to fetch VIX: {e}")
    
    return None


# ============================================================================
# Feature Engineering
# ============================================================================

def compute_enhanced_features(
    price_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute enhanced features for regime detection.
    
    Features:
    - Multi-timeframe momentum (5d, 20d, 60d)
    - Multi-timeframe volatility
    - VIX levels and changes
    - Price position relative to moving averages
    - Trend strength (ADX-like)
    """
    df = price_df.copy()
    features = pd.DataFrame(index=df.index)
    
    # Returns at multiple timeframes
    features['returns_1d'] = df['close'].pct_change()
    features['returns_5d'] = df['close'].pct_change(5)
    features['returns_20d'] = df['close'].pct_change(20)
    features['returns_60d'] = df['close'].pct_change(60)
    
    # Volatility at multiple timeframes
    features['vol_5d'] = features['returns_1d'].rolling(5).std() * np.sqrt(252)
    features['vol_20d'] = features['returns_1d'].rolling(20).std() * np.sqrt(252)
    features['vol_60d'] = features['returns_1d'].rolling(60).std() * np.sqrt(252)
    
    # Volatility trend (increasing or decreasing)
    features['vol_trend'] = features['vol_20d'] / features['vol_60d'] - 1
    
    # Moving averages
    features['ma_20'] = df['close'].rolling(20).mean()
    features['ma_50'] = df['close'].rolling(50).mean()
    features['ma_200'] = df['close'].rolling(200).mean()
    
    # Price relative to MAs
    features['price_vs_ma20'] = df['close'] / features['ma_20'] - 1
    features['price_vs_ma50'] = df['close'] / features['ma_50'] - 1
    features['price_vs_ma200'] = df['close'] / features['ma_200'] - 1
    
    # MA alignment (trend strength)
    features['ma_alignment'] = (
        (features['ma_20'] > features['ma_50']).astype(int) +
        (features['ma_50'] > features['ma_200']).astype(int)
    ) / 2  # 0 = bearish, 0.5 = mixed, 1 = bullish
    
    # Trend strength (simplified ADX concept)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr_20'] = true_range.rolling(20).mean()
    features['atr_pct'] = features['atr_20'] / df['close']
    
    # Directional movement
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    features['trend_strength'] = abs(
        plus_dm.rolling(20).sum() - minus_dm.rolling(20).sum()
    ) / (plus_dm.rolling(20).sum() + minus_dm.rolling(20).sum() + 1e-8)
    
    # Add VIX features if available
    if vix_df is not None and 'close' in vix_df.columns:
        # Align VIX to price index
        vix_aligned = vix_df['close'].reindex(df.index, method='ffill')
        features['vix'] = vix_aligned
        features['vix_ma20'] = vix_aligned.rolling(20).mean()
        features['vix_change_5d'] = vix_aligned.pct_change(5)
        features['vix_percentile'] = vix_aligned.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )
    
    return features.dropna()


def classify_vix_regime(vix: float) -> str:
    """Classify VIX level into regime."""
    if vix < VIX_LOW:
        return 'low'
    elif vix < VIX_NORMAL:
        return 'normal'
    elif vix < VIX_ELEVATED:
        return 'elevated'
    elif vix < VIX_HIGH:
        return 'high'
    else:
        return 'extreme'


# ============================================================================
# Regime Classification
# ============================================================================

def classify_regime(features: pd.Series, history: pd.DataFrame = None) -> RegimeState:
    """
    Classify market regime using multi-factor analysis.
    
    Considers:
    - Price momentum (multiple timeframes)
    - Volatility level and trend
    - VIX level (if available)
    - MA alignment
    """
    momentum_20d = features.get('returns_20d', 0)
    momentum_60d = features.get('returns_60d', 0)
    vol_20d = features.get('vol_20d', 0.15)
    vol_60d = features.get('vol_60d', 0.15)
    vix = features.get('vix', 20)
    ma_alignment = features.get('ma_alignment', 0.5)
    trend_strength = features.get('trend_strength', 0.5)
    
    # Classify VIX regime
    vix_regime = classify_vix_regime(vix) if not pd.isna(vix) else 'normal'
    
    # Determine base regime from momentum
    if momentum_20d > 0.08:  # Strong up
        base_regime = 0  # STRONG_BULL
    elif momentum_20d > 0.03:
        base_regime = 1  # BULL
    elif momentum_20d > 0:
        base_regime = 2  # WEAK_BULL
    elif momentum_20d > -0.03:
        base_regime = 3  # SIDEWAYS
    elif momentum_20d > -0.08:
        base_regime = 5  # BEAR
    else:
        base_regime = 6  # STRONG_BEAR
    
    # Adjust for VIX
    if vix_regime == 'extreme':
        if momentum_20d < -0.05:
            base_regime = 7  # CRISIS
        else:
            base_regime = 8  # RECOVERY
    elif vix_regime == 'high' and momentum_20d > 0:
        base_regime = 2  # WEAK_BULL (elevated risk)
    elif vix_regime == 'low' and momentum_20d > 0.03:
        base_regime = 0  # STRONG_BULL (complacency bull)
    
    # Confidence based on alignment of signals
    signals_aligned = 0
    if (momentum_20d > 0) == (momentum_60d > 0):
        signals_aligned += 1
    if ma_alignment > 0.75 or ma_alignment < 0.25:
        signals_aligned += 1
    if trend_strength > 0.6:
        signals_aligned += 1
    
    confidence = 50 + signals_aligned * 15
    
    # Risk appetite score
    risk_appetite = 0.0
    if vix_regime in ['low', 'normal']:
        risk_appetite += 0.3
    elif vix_regime in ['high', 'extreme']:
        risk_appetite -= 0.5
    if momentum_20d > 0:
        risk_appetite += min(momentum_20d * 5, 0.5)
    else:
        risk_appetite += max(momentum_20d * 5, -0.5)
    
    risk_appetite = max(-1, min(1, risk_appetite))
    
    # Calculate regime duration (simplified - would need history)
    regime_duration = 1
    
    # Transition probabilities (simplified - would compute from history)
    transition_prob = {
        REGIME_STATES[i]: 0.1 for i in range(9)
    }
    transition_prob[REGIME_STATES[base_regime]] = 0.5
    
    return RegimeState(
        regime_id=base_regime,
        regime_name=REGIME_STATES[base_regime],
        confidence=confidence,
        vix_level=vix if not pd.isna(vix) else 0,
        vix_regime=vix_regime,
        momentum_20d=momentum_20d,
        momentum_60d=momentum_60d,
        volatility_20d=vol_20d,
        volatility_60d=vol_60d,
        risk_appetite=risk_appetite,
        regime_duration=regime_duration,
        transition_prob=transition_prob,
    )


# ============================================================================
# Cross-Asset Analysis
# ============================================================================

def compute_cross_correlations(
    assets: Dict[str, pd.DataFrame],
    window: int = 20
) -> Dict[str, float]:
    """
    Compute rolling correlations between assets.
    
    Key relationships:
    - SP500 vs VIX (should be negative)
    - SP500 vs Gold (risk indicator)
    - Gold vs USD (typically negative)
    - Oil vs SP500 (demand indicator)
    """
    correlations = {}
    
    # Get returns for each asset
    returns = {}
    for name, df in assets.items():
        if df is not None and 'close' in df.columns:
            returns[name] = df['close'].pct_change()
    
    # Compute key correlations
    pairs = [
        ('SP500', 'VIX'),
        ('SP500', 'GOLD'),
        ('GOLD', 'US_DOLLAR'),
        ('SP500', 'Crude_Oil'),
        ('SP500', 'Bitcoin'),
        ('VIX', 'GOLD'),
    ]
    
    for asset1, asset2 in pairs:
        if asset1 in returns and asset2 in returns:
            # Align the series
            combined = pd.concat([returns[asset1], returns[asset2]], axis=1).dropna()
            if len(combined) >= window:
                corr = combined.iloc[-window:].corr().iloc[0, 1]
                correlations[f"{asset1}_vs_{asset2}"] = corr
    
    return correlations


def compute_risk_on_score(
    sp500_features: pd.Series,
    correlations: Dict[str, float],
    vix_level: float
) -> float:
    """
    Compute overall risk-on/risk-off score.
    
    Positive = Risk-on (bullish)
    Negative = Risk-off (defensive)
    """
    score = 0.0
    
    # VIX contribution (-30 to +20)
    if vix_level < VIX_LOW:
        score += 20  # Complacency = risk-on
    elif vix_level < VIX_NORMAL:
        score += 10
    elif vix_level < VIX_ELEVATED:
        score += 0
    elif vix_level < VIX_HIGH:
        score -= 15
    else:
        score -= 30  # Fear = risk-off
    
    # SP500 momentum contribution (-25 to +25)
    momentum = sp500_features.get('returns_20d', 0)
    score += min(max(momentum * 250, -25), 25)
    
    # Correlation contributions
    # Normal: SP500 vs VIX should be negative
    spx_vix = correlations.get('SP500_vs_VIX', -0.5)
    if spx_vix > 0:  # Unusual - both rising = potential issue
        score -= 10
    
    # Gold correlation (safe haven demand)
    spx_gold = correlations.get('SP500_vs_GOLD', 0)
    if spx_gold < -0.3:  # Negative = flight to safety
        score -= 10
    
    # MA alignment contribution (-20 to +20)
    ma_align = sp500_features.get('ma_alignment', 0.5)
    score += (ma_align - 0.5) * 40
    
    return max(-100, min(100, score))


def detect_divergences(
    assets: Dict[str, pd.DataFrame],
    window: int = 20
) -> List[str]:
    """Detect notable divergences between assets."""
    divergences = []
    
    # Get recent returns
    returns = {}
    for name, df in assets.items():
        if df is not None and 'close' in df.columns and len(df) >= window:
            returns[name] = df['close'].iloc[-1] / df['close'].iloc[-window] - 1
    
    # Check for divergences
    if 'SP500' in returns and 'VIX' in returns:
        # Both rising = unusual
        if returns['SP500'] > 0.02 and returns['VIX'] > 0.1:
            divergences.append("SP500 rising but VIX rising too - caution")
    
    if 'GOLD' in returns and 'US_DOLLAR' in returns:
        # Both rising = unusual (typically inverse)
        if returns['GOLD'] > 0.02 and returns['US_DOLLAR'] > 0.01:
            divergences.append("Gold and USD both rising - safe haven demand")
    
    if 'SP500' in returns and 'Bitcoin' in returns:
        # Big divergence
        diff = abs(returns['SP500'] - returns['Bitcoin'])
        if diff > 0.1:
            leader = 'Bitcoin' if returns['Bitcoin'] > returns['SP500'] else 'SP500'
            divergences.append(f"Large SP500/BTC divergence - {leader} leading")
    
    return divergences


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_market_regime(years: int = 5) -> MarketContext:
    """
    Perform comprehensive market regime analysis.
    
    Returns MarketContext with:
    - Current regime classification
    - Cross-asset correlations
    - Risk-on/risk-off score
    - Divergences and warnings
    """
    print("=" * 70)
    print("ENHANCED QUANTUM REGIME ANALYSIS")
    print("=" * 70)
    
    # Load all assets
    assets = {}
    for name in REGIME_ASSETS:
        print(f"Loading {name}...", end=" ", flush=True)
        df = load_asset_data(name, years)
        if df is not None:
            assets[name] = df
            print(f"OK ({len(df)} rows)")
        else:
            print("MISSING")
    
    # Load VIX specifically
    vix_df = assets.get('VIX')
    
    # Get SP500 data for primary analysis
    sp500_df = assets.get('SP500')
    if sp500_df is None:
        raise ValueError("SP500 data required for regime analysis")
    
    # Compute features
    print("\nComputing features...")
    features = compute_enhanced_features(sp500_df, vix_df)
    
    # Classify current regime
    current_features = features.iloc[-1]
    sp500_regime = classify_regime(current_features)
    
    # Cross-asset correlations
    print("Computing cross-correlations...")
    correlations = compute_cross_correlations(assets)
    
    # Risk score
    vix_level = current_features.get('vix', 20)
    risk_score = compute_risk_on_score(current_features, correlations, vix_level)
    
    # Divergences
    divergences = detect_divergences(assets)
    
    # Determine leading indicator
    leading = "SP500"
    if 'VIX' in assets and vix_level > VIX_ELEVATED:
        leading = "VIX"
    elif 'Bitcoin' in assets:
        btc_mom = assets['Bitcoin']['close'].pct_change(5).iloc[-1]
        sp_mom = sp500_df['close'].pct_change(5).iloc[-1]
        if abs(btc_mom) > abs(sp_mom) * 2:
            leading = "Bitcoin"
    
    context = MarketContext(
        timestamp=datetime.now(),
        sp500_regime=sp500_regime,
        cross_correlations=correlations,
        risk_on_score=risk_score,
        leading_indicator=leading,
        divergences=divergences,
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("CURRENT MARKET REGIME")
    print("=" * 70)
    print(f"  Regime:        {sp500_regime.regime_name}")
    print(f"  Confidence:    {sp500_regime.confidence:.0f}%")
    print(f"  VIX Level:     {sp500_regime.vix_level:.1f} ({sp500_regime.vix_regime})")
    print(f"  Momentum 20d:  {sp500_regime.momentum_20d:+.1%}")
    print(f"  Momentum 60d:  {sp500_regime.momentum_60d:+.1%}")
    print(f"  Volatility:    {sp500_regime.volatility_20d:.1%}")
    print(f"  Risk Appetite: {sp500_regime.risk_appetite:+.2f}")
    
    print(f"\n  Risk Score:    {risk_score:+.0f} ({'RISK-ON' if risk_score > 20 else 'RISK-OFF' if risk_score < -20 else 'NEUTRAL'})")
    print(f"  Leading:       {leading}")
    
    print("\nCROSS-CORRELATIONS (20d rolling):")
    for pair, corr in correlations.items():
        print(f"  {pair}: {corr:+.2f}")
    
    if divergences:
        print("\n[!] DIVERGENCES DETECTED:")
        for div in divergences:
            print(f"  - {div}")
    
    return context


# ============================================================================
# API for Dashboard
# ============================================================================

def get_current_regime() -> Dict:
    """Get current regime state for API/dashboard."""
    try:
        context = analyze_market_regime(years=2)
        regime = context.sp500_regime
        
        return {
            "success": True,
            "regime": {
                "id": regime.regime_id,
                "name": regime.regime_name,
                "confidence": regime.confidence,
                "vix_level": regime.vix_level,
                "vix_regime": regime.vix_regime,
                "momentum_20d": regime.momentum_20d,
                "momentum_60d": regime.momentum_60d,
                "volatility_20d": regime.volatility_20d,
                "risk_appetite": regime.risk_appetite,
            },
            "context": {
                "risk_score": context.risk_on_score,
                "risk_label": 'RISK-ON' if context.risk_on_score > 20 else 'RISK-OFF' if context.risk_on_score < -20 else 'NEUTRAL',
                "leading_indicator": context.leading_indicator,
                "correlations": context.cross_correlations,
                "divergences": context.divergences,
            },
            "timestamp": context.timestamp.isoformat(),
        }
    except Exception as e:
        log.error(f"Error computing regime: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    context = analyze_market_regime(years)
