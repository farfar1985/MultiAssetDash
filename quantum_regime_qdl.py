"""
Quantum Regime Detection using QDT Data Lake
=============================================
Combines QDL price data with quantum-inspired regime detection.

This is the integration layer between:
- qdl_client.py (Data Lake access)
- quantum_regime_detector.py (regime detection algorithms)
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple

from qdl_client import qdl_get_data, NEXUS_TO_QDL_MAP

log = logging.getLogger(__name__)

# ============================================================================
# Quantum-Inspired Regime States
# ============================================================================

REGIME_STATES = {
    0: 'BULL',       # Strong uptrend
    1: 'BEAR',       # Strong downtrend
    2: 'SIDEWAYS',   # Low volatility consolidation
    3: 'HIGH_VOL',   # High volatility (uncertain)
}

# ============================================================================
# Feature Engineering for Regime Detection
# ============================================================================

def compute_regime_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute features for regime detection from OHLCV data.
    
    Features:
    - Returns (log returns)
    - Volatility (rolling std of returns)
    - Momentum (rate of change)
    - Volume trend
    - Price position (relative to recent range)
    """
    features = pd.DataFrame(index=df.index)
    
    # Log returns
    features['returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Rolling volatility
    features['volatility'] = features['returns'].rolling(lookback).std() * np.sqrt(252)
    
    # Momentum (20-day return)
    features['momentum'] = df['close'].pct_change(lookback)
    
    # Volume trend (vs 20-day average)
    if 'volume' in df.columns:
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(lookback).mean()
    else:
        features['volume_ratio'] = 1.0
    
    # Price position (0-1 range relative to lookback high/low)
    rolling_high = df['high'].rolling(lookback).max()
    rolling_low = df['low'].rolling(lookback).min()
    features['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
    
    # ATR (Average True Range) for volatility context
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr'] = true_range.rolling(lookback).mean() / df['close']
    
    return features.dropna()


def classify_regime(features: pd.Series, vol_threshold: float = 0.25) -> int:
    """
    Classify current market regime based on features.
    
    Uses a rule-based approach inspired by quantum state superposition:
    - High momentum + low vol = BULL or BEAR
    - Low momentum + low vol = SIDEWAYS
    - High vol = HIGH_VOL (uncertainty state)
    """
    vol = features['volatility']
    mom = features['momentum']
    pos = features['price_position']
    
    # High volatility overrides everything
    if vol > vol_threshold:
        return 3  # HIGH_VOL
    
    # Strong momentum determines bull/bear
    if mom > 0.05:  # 5% momentum threshold
        return 0  # BULL
    elif mom < -0.05:
        return 1  # BEAR
    else:
        return 2  # SIDEWAYS


def detect_regimes(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detect market regimes for entire price history.
    
    Returns DataFrame with regime classifications.
    """
    features = compute_regime_features(df, lookback)
    
    regimes = []
    for idx in features.index:
        regime_id = classify_regime(features.loc[idx])
        regimes.append({
            'time': idx,
            'regime_id': regime_id,
            'regime': REGIME_STATES[regime_id],
            'volatility': features.loc[idx, 'volatility'],
            'momentum': features.loc[idx, 'momentum'],
            'price_position': features.loc[idx, 'price_position'],
        })
    
    return pd.DataFrame(regimes).set_index('time')


# ============================================================================
# Multi-Asset Regime Analysis
# ============================================================================

def analyze_all_assets_from_qdl(
    years_back: int = 5,
    output_dir: str = "data/regime_analysis"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data from QDL and perform regime analysis on all assets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    
    results = {}
    
    # Only use DTNIQ assets (confirmed working)
    DTNIQ_ASSETS = {
        'SP500': ('@ES#C', 'DTNIQ'),
        'NASDAQ': ('@NQ#C', 'DTNIQ'),
        'Dow_Jones': ('@YM#C', 'DTNIQ'),
        'Russell_2000': ('@RTY#C', 'DTNIQ'),
        'Crude_Oil': ('QCL#', 'DTNIQ'),
        'Brent_Oil': ('QBZ#', 'DTNIQ'),
        'GOLD': ('QGC#', 'DTNIQ'),
        'MCX_Copper': ('QHG#', 'DTNIQ'),
        'Natural_Gas': ('QNG#', 'DTNIQ'),
        'Bitcoin': ('@BTC#C', 'DTNIQ'),
        'US_DOLLAR': ('@DX#C', 'DTNIQ'),
        'Nikkei_225': ('@NKD#C', 'DTNIQ'),
    }
    
    print("=" * 60)
    print("QUANTUM REGIME ANALYSIS - QDT Data Lake")
    print("=" * 60)
    
    for asset_name, (symbol, provider) in DTNIQ_ASSETS.items():
        print(f"\n{asset_name}...", end=" ", flush=True)
        
        try:
            # Fetch from QDL
            df = qdl_get_data(symbol, provider, start_date, end_date)
            
            if df.empty:
                print("NO DATA")
                continue
            
            # Detect regimes
            regimes = detect_regimes(df)
            
            # Current regime
            current = regimes.iloc[-1]
            
            print(f"{current['regime']} (vol={current['volatility']:.1%}, mom={current['momentum']:.1%})")
            
            # Save results
            regimes.to_csv(os.path.join(output_dir, f"{asset_name}_regimes.csv"))
            results[asset_name] = regimes
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("CURRENT REGIME SUMMARY")
    print("=" * 60)
    
    for asset_name, regimes in results.items():
        current = regimes.iloc[-1]
        print(f"  {asset_name:15} {current['regime']:10} vol={current['volatility']:6.1%} mom={current['momentum']:+6.1%}")
    
    return results


def compute_regime_transitions(regimes: pd.DataFrame) -> Dict:
    """
    Compute regime transition probabilities.
    """
    transitions = {}
    
    for i in range(len(regimes) - 1):
        from_regime = regimes.iloc[i]['regime']
        to_regime = regimes.iloc[i + 1]['regime']
        
        key = f"{from_regime}→{to_regime}"
        transitions[key] = transitions.get(key, 0) + 1
    
    # Normalize
    total = sum(transitions.values())
    return {k: v / total for k, v in transitions.items()}


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
    results = analyze_all_assets_from_qdl(years_back=years)
    
    print(f"\nAnalysis complete. Results saved to data/regime_analysis/")
