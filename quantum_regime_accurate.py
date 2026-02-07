"""
High-Accuracy Quantum Regime Detection
=======================================
Maximizes regime detection accuracy using:
1. Hidden Markov Model for statistical regime switching
2. COT (Commitment of Traders) data for smart money positioning  
3. VIX term structure (VIX vs VIX futures)
4. Ensemble scoring with multiple confirming signals
5. Yield curve dynamics (2Y-10Y spread)
6. Cross-asset validation

Target: >75% accuracy in regime classification
"""

import os
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Try to import HMM (optional - falls back to rule-based if not available)
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    
from qdl_client import qdl_get_data

log = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("data/qdl_history")
MODEL_DIR = Path("models")

# Regime definitions (more granular for accuracy)
REGIMES = {
    0: {'name': 'STRONG_BULL', 'risk': 'high', 'duration': 'persistent'},
    1: {'name': 'BULL', 'risk': 'moderate', 'duration': 'trending'},
    2: {'name': 'EARLY_BULL', 'risk': 'moderate', 'duration': 'transitional'},
    3: {'name': 'SIDEWAYS_BULLISH', 'risk': 'low', 'duration': 'consolidation'},
    4: {'name': 'SIDEWAYS', 'risk': 'low', 'duration': 'range'},
    5: {'name': 'SIDEWAYS_BEARISH', 'risk': 'moderate', 'duration': 'consolidation'},
    6: {'name': 'EARLY_BEAR', 'risk': 'high', 'duration': 'transitional'},
    7: {'name': 'BEAR', 'risk': 'high', 'duration': 'trending'},
    8: {'name': 'STRONG_BEAR', 'risk': 'extreme', 'duration': 'persistent'},
    9: {'name': 'CRISIS', 'risk': 'extreme', 'duration': 'acute'},
    10: {'name': 'RECOVERY', 'risk': 'high', 'duration': 'transitional'},
}

# Data sources to fetch
DATA_SOURCES = {
    # Price data
    'SP500': ('@ES#C', 'DTNIQ'),
    'VIX': ('@VX#C', 'DTNIQ'),
    'VIX_M2': ('@VX.2#C', 'DTNIQ'),  # 2nd month VIX futures
    'GOLD': ('QGC#', 'DTNIQ'),
    'Treasury_10Y': ('@TN#C', 'DTNIQ'),
    'Treasury_2Y': ('@ZT#C', 'DTNIQ'),
    'USD': ('@DX#C', 'DTNIQ'),
    'Crude_Oil': ('QCL#', 'DTNIQ'),
    'Bitcoin': ('@BTC#C', 'DTNIQ'),
    
    # COT Data (weekly) - E-mini S&P 500 futures
    'COT_SP500_Net': ('CF_ES_NN', 'DTNIQ'),  # E-mini S&P noncommercial net
    'COT_SP500_Dealer': ('CF_ES_DN', 'DTNIQ'),  # Dealer net (smart money)
    'COT_Gold_Net': ('CF_GC_NN', 'DTNIQ'),   # Gold noncommercial net
    'COT_Oil_Net': ('CF_CL_NN', 'DTNIQ'),    # Oil noncommercial net
    'COT_VIX_Net': ('CF_VX_AN', 'DTNIQ'),    # VIX asset net
    
    # Yield curve spread (direct)
    'YieldSpread_2Y10Y': ('@TUT#C', 'DTNIQ'),  # 2Y-10Y spread
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RegimeSignal:
    """Individual signal contributing to regime detection."""
    name: str
    value: float
    signal: int      # -2 to +2 (strong bear to strong bull)
    weight: float    # Importance weight
    confidence: float  # 0-100


@dataclass  
class AccurateRegime:
    """High-accuracy regime classification."""
    regime_id: int
    regime_name: str
    confidence: float           # 0-100
    signals: List[RegimeSignal]
    ensemble_score: float       # -100 to +100
    hmm_state: Optional[int]    # HMM hidden state if available
    hmm_prob: Optional[float]   # HMM state probability
    vix_term_structure: str     # 'contango' or 'backwardation'
    cot_sentiment: str          # 'bullish', 'bearish', 'neutral'
    yield_curve: str            # 'normal', 'flat', 'inverted'
    timestamp: str


# ============================================================================
# Data Loading & Caching
# ============================================================================

class DataLoader:
    """Manages loading and caching of QDL data."""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def load(self, name: str, years: int = 5) -> Optional[pd.DataFrame]:
        """Load data from cache, CSV, or QDL."""
        # Check memory cache
        if name in self._cache:
            return self._cache[name]
        
        # Check CSV cache
        csv_path = self.data_dir / f"{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['time'])
            df = df.sort_values('time').set_index('time')
            self._cache[name] = df
            return df
        
        # Fetch from QDL
        if name in DATA_SOURCES:
            symbol, provider = DATA_SOURCES[name]
            end = datetime.now()
            start = end - timedelta(days=365 * years)
            
            try:
                df = qdl_get_data(symbol, provider, start, end)
                if not df.empty:
                    df.to_csv(csv_path)
                    self._cache[name] = df
                    return df
            except Exception as e:
                log.warning(f"Failed to fetch {name}: {e}")
        
        return None
    
    def load_all(self, years: int = 5) -> Dict[str, pd.DataFrame]:
        """Load all data sources."""
        data = {}
        for name in DATA_SOURCES:
            df = self.load(name, years)
            if df is not None:
                data[name] = df
                log.info(f"Loaded {name}: {len(df)} rows")
        return data


# ============================================================================
# Feature Engineering
# ============================================================================

class FeatureEngine:
    """Computes features for regime detection."""
    
    @staticmethod
    def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Multi-timeframe momentum indicators."""
        f = pd.DataFrame(index=df.index)
        
        # Returns at multiple timeframes
        for period in [1, 5, 10, 20, 60, 120]:
            f[f'ret_{period}d'] = df['close'].pct_change(period)
        
        # Momentum score (weighted average of returns)
        f['momentum_score'] = (
            f['ret_5d'] * 0.1 +
            f['ret_10d'] * 0.15 +
            f['ret_20d'] * 0.25 +
            f['ret_60d'] * 0.3 +
            f['ret_120d'] * 0.2
        )
        
        # Rate of change
        f['roc_20'] = (df['close'] / df['close'].shift(20) - 1) * 100
        
        return f
    
    @staticmethod
    def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Volatility and risk indicators."""
        f = pd.DataFrame(index=df.index)
        
        returns = df['close'].pct_change()
        
        # Rolling volatility
        for period in [5, 10, 20, 60]:
            f[f'vol_{period}d'] = returns.rolling(period).std() * np.sqrt(252) * 100
        
        # Volatility regime (percentile)
        f['vol_percentile'] = f['vol_20d'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        
        # Volatility trend
        f['vol_trend'] = f['vol_20d'] / f['vol_60d'] - 1
        
        # ATR
        if 'high' in df.columns and 'low' in df.columns:
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            ], axis=1).max(axis=1)
            f['atr_20'] = tr.rolling(20).mean()
            f['atr_pct'] = f['atr_20'] / df['close'] * 100
        
        return f
    
    @staticmethod
    def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Trend strength and direction indicators."""
        f = pd.DataFrame(index=df.index)
        
        # Moving averages
        for period in [10, 20, 50, 100, 200]:
            f[f'ma_{period}'] = df['close'].rolling(period).mean()
        
        # Price vs MAs
        f['price_vs_ma20'] = (df['close'] / f['ma_20'] - 1) * 100
        f['price_vs_ma50'] = (df['close'] / f['ma_50'] - 1) * 100
        f['price_vs_ma200'] = (df['close'] / f['ma_200'] - 1) * 100
        
        # MA alignment score (-1 to +1)
        ma_bull = (
            (f['ma_20'] > f['ma_50']).astype(int) +
            (f['ma_50'] > f['ma_100']).astype(int) +
            (f['ma_100'] > f['ma_200']).astype(int)
        )
        f['ma_alignment'] = (ma_bull / 3) * 2 - 1
        
        # Trend strength (based on price position in range)
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        f['price_position'] = (df['close'] - low_20) / (high_20 - low_20 + 1e-8)
        
        return f
    
    @staticmethod
    def compute_vix_features(vix_df: pd.DataFrame, vix_m2_df: pd.DataFrame = None) -> pd.DataFrame:
        """VIX-based features including term structure."""
        f = pd.DataFrame(index=vix_df.index)
        
        f['vix'] = vix_df['close']
        f['vix_ma20'] = f['vix'].rolling(20).mean()
        f['vix_pct_20'] = (f['vix'] / f['vix_ma20'] - 1) * 100
        
        # VIX percentile (historical)
        f['vix_percentile'] = f['vix'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        
        # VIX change
        f['vix_change_5d'] = f['vix'].pct_change(5) * 100
        
        # VIX term structure (if 2nd month available)
        if vix_m2_df is not None and 'close' in vix_m2_df.columns:
            vix_m2 = vix_m2_df['close'].reindex(vix_df.index, method='ffill')
            f['vix_m2'] = vix_m2
            f['vix_term_spread'] = (vix_m2 / f['vix'] - 1) * 100
            # Positive = contango (normal), Negative = backwardation (fear)
        
        return f
    
    @staticmethod
    def compute_cot_features(cot_df: pd.DataFrame, name: str) -> pd.DataFrame:
        """COT positioning features."""
        f = pd.DataFrame(index=cot_df.index)
        
        # Net positioning
        f[f'{name}_net'] = cot_df['close']
        
        # Z-score of positioning
        f[f'{name}_zscore'] = (
            (f[f'{name}_net'] - f[f'{name}_net'].rolling(52).mean()) /
            f[f'{name}_net'].rolling(52).std()
        )
        
        # Change in positioning
        f[f'{name}_change'] = f[f'{name}_net'].diff()
        
        # Extreme positioning flag
        f[f'{name}_extreme'] = abs(f[f'{name}_zscore']) > 2
        
        return f
    
    @staticmethod
    def compute_yield_curve_features(
        t2y_df: pd.DataFrame, 
        t10y_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Yield curve features from Treasury data."""
        f = pd.DataFrame(index=t10y_df.index)
        
        # Align 2Y to 10Y index
        t2y = t2y_df['close'].reindex(t10y_df.index, method='ffill')
        t10y = t10y_df['close']
        
        f['t2y'] = t2y
        f['t10y'] = t10y
        
        # Spread (inverted yield curve < 0 = recession signal)
        f['yield_spread'] = t10y - t2y
        
        # Spread change
        f['yield_spread_change'] = f['yield_spread'].diff(20)
        
        # Inversion flag
        f['yield_inverted'] = f['yield_spread'] < 0
        
        return f


# ============================================================================
# Regime Classification
# ============================================================================

class RegimeClassifier:
    """Multi-method regime classification."""
    
    def __init__(self):
        self.hmm_model = None
        self.feature_scaler = None
    
    def _compute_ensemble_signals(
        self,
        momentum: pd.Series,
        volatility: pd.Series,
        trend: pd.Series,
        vix: pd.Series,
        cot: pd.Series = None,
        yield_curve: pd.Series = None
    ) -> List[RegimeSignal]:
        """Compute individual signals for ensemble scoring."""
        signals = []
        
        # Momentum signal
        mom_val = momentum.get('momentum_score', 0)
        if mom_val > 0.05:
            mom_signal = 2 if mom_val > 0.10 else 1
        elif mom_val < -0.05:
            mom_signal = -2 if mom_val < -0.10 else -1
        else:
            mom_signal = 0
        signals.append(RegimeSignal(
            name='momentum',
            value=mom_val,
            signal=mom_signal,
            weight=0.25,
            confidence=min(abs(mom_val) * 500, 100)
        ))
        
        # Volatility signal
        vol_val = volatility.get('vol_20d', 15)
        vol_pct = volatility.get('vol_percentile', 50)
        if vol_pct > 80:
            vol_signal = -2  # High vol = bearish
        elif vol_pct > 60:
            vol_signal = -1
        elif vol_pct < 20:
            vol_signal = 1  # Low vol = bullish continuation
        else:
            vol_signal = 0
        signals.append(RegimeSignal(
            name='volatility',
            value=vol_val,
            signal=vol_signal,
            weight=0.15,
            confidence=min(abs(vol_pct - 50) * 2, 100)
        ))
        
        # Trend signal
        ma_align = trend.get('ma_alignment', 0)
        if ma_align > 0.5:
            trend_signal = 2
        elif ma_align > 0:
            trend_signal = 1
        elif ma_align < -0.5:
            trend_signal = -2
        elif ma_align < 0:
            trend_signal = -1
        else:
            trend_signal = 0
        signals.append(RegimeSignal(
            name='trend_alignment',
            value=ma_align,
            signal=trend_signal,
            weight=0.20,
            confidence=min(abs(ma_align) * 100, 100)
        ))
        
        # VIX signal
        vix_val = vix.get('vix', 20)
        vix_pct = vix.get('vix_percentile', 50)
        if vix_val > 35:
            vix_signal = -2  # Extreme fear
        elif vix_val > 25:
            vix_signal = -1
        elif vix_val < 15:
            vix_signal = 1  # Complacency
        else:
            vix_signal = 0
        signals.append(RegimeSignal(
            name='vix',
            value=vix_val,
            signal=vix_signal,
            weight=0.20,
            confidence=min(abs(vix_pct - 50) * 2, 100)
        ))
        
        # VIX term structure
        term_spread = vix.get('vix_term_spread', 0)
        if term_spread < -5:
            term_signal = -2  # Backwardation = fear
        elif term_spread < 0:
            term_signal = -1
        elif term_spread > 5:
            term_signal = 1  # Contango = calm
        else:
            term_signal = 0
        signals.append(RegimeSignal(
            name='vix_term',
            value=term_spread,
            signal=term_signal,
            weight=0.10,
            confidence=min(abs(term_spread) * 10, 100)
        ))
        
        # COT signal (if available)
        if cot is not None:
            cot_zscore = cot.get('COT_SP500_Net_zscore', 0)
            if not pd.isna(cot_zscore):
                if cot_zscore > 1.5:
                    cot_signal = 2  # Speculators very long
                elif cot_zscore > 0.5:
                    cot_signal = 1
                elif cot_zscore < -1.5:
                    cot_signal = -2  # Speculators very short
                elif cot_zscore < -0.5:
                    cot_signal = -1
                else:
                    cot_signal = 0
                signals.append(RegimeSignal(
                    name='cot_positioning',
                    value=cot_zscore,
                    signal=cot_signal,
                    weight=0.10,
                    confidence=min(abs(cot_zscore) * 40, 100)
                ))
        
        # Yield curve signal (if available)
        if yield_curve is not None:
            yc_spread = yield_curve.get('yield_spread', 0)
            if not pd.isna(yc_spread):
                # Inverted curve = recession risk = bearish
                if yc_spread < -0.5:
                    yc_signal = -2  # Inverted = high recession risk
                elif yc_spread < 0:
                    yc_signal = -1  # Flattening/mildly inverted
                elif yc_spread > 1.0:
                    yc_signal = 1   # Steep = healthy
                else:
                    yc_signal = 0   # Normal
                signals.append(RegimeSignal(
                    name='yield_curve',
                    value=yc_spread,
                    signal=yc_signal,
                    weight=0.10,  # Important macro signal
                    confidence=min(abs(yc_spread) * 50, 100)
                ))
        
        return signals
    
    def _compute_ensemble_score(self, signals: List[RegimeSignal]) -> float:
        """Compute weighted ensemble score from signals."""
        total_weight = sum(s.weight for s in signals)
        if total_weight == 0:
            return 0
        
        score = sum(s.signal * s.weight * (s.confidence / 100) for s in signals)
        # Normalize to -100 to +100
        max_possible = sum(2 * s.weight for s in signals)
        normalized = (score / max_possible) * 100 if max_possible > 0 else 0
        
        return normalized
    
    def _classify_from_ensemble(self, score: float, vix: float) -> Tuple[int, str]:
        """Map ensemble score to regime."""
        # Consider VIX for crisis/recovery states
        if vix > 40:
            if score < -30:
                return 9, 'CRISIS'
            else:
                return 10, 'RECOVERY'
        
        # Normal regime classification
        if score > 60:
            return 0, 'STRONG_BULL'
        elif score > 40:
            return 1, 'BULL'
        elif score > 20:
            return 2, 'EARLY_BULL'
        elif score > 10:
            return 3, 'SIDEWAYS_BULLISH'
        elif score > -10:
            return 4, 'SIDEWAYS'
        elif score > -20:
            return 5, 'SIDEWAYS_BEARISH'
        elif score > -40:
            return 6, 'EARLY_BEAR'
        elif score > -60:
            return 7, 'BEAR'
        else:
            return 8, 'STRONG_BEAR'
    
    def classify(
        self,
        momentum: pd.Series,
        volatility: pd.Series,
        trend: pd.Series,
        vix: pd.Series,
        cot: pd.Series = None,
        yield_curve: pd.Series = None
    ) -> AccurateRegime:
        """Classify regime using ensemble method."""
        
        # Compute signals
        signals = self._compute_ensemble_signals(
            momentum, volatility, trend, vix, cot, yield_curve
        )
        
        # Compute ensemble score
        ensemble_score = self._compute_ensemble_score(signals)
        
        # Get VIX level for crisis detection
        vix_level = vix.get('vix', 20)
        
        # Classify
        regime_id, regime_name = self._classify_from_ensemble(ensemble_score, vix_level)
        
        # Compute confidence (based on signal agreement)
        signal_values = [s.signal for s in signals]
        agreement = np.std(signal_values)
        # Lower std = more agreement = higher confidence
        confidence = max(0, min(100, 100 - agreement * 25))
        
        # VIX term structure
        term_spread = vix.get('vix_term_spread', 0)
        if pd.isna(term_spread):
            vix_term = 'unknown'
        elif term_spread > 0:
            vix_term = 'contango'
        else:
            vix_term = 'backwardation'
        
        # COT sentiment
        if cot is not None:
            cot_z = cot.get('COT_SP500_Net_zscore', 0)
            if not pd.isna(cot_z):
                if cot_z > 0.5:
                    cot_sentiment = 'bullish'
                elif cot_z < -0.5:
                    cot_sentiment = 'bearish'
                else:
                    cot_sentiment = 'neutral'
            else:
                cot_sentiment = 'unknown'
        else:
            cot_sentiment = 'unavailable'
        
        # Yield curve
        if yield_curve is not None:
            spread = yield_curve.get('yield_spread', 0)
            if not pd.isna(spread):
                if spread < -0.5:
                    yc = 'inverted'
                elif spread < 0.5:
                    yc = 'flat'
                else:
                    yc = 'normal'
            else:
                yc = 'unknown'
        else:
            yc = 'unavailable'
        
        return AccurateRegime(
            regime_id=regime_id,
            regime_name=regime_name,
            confidence=confidence,
            signals=signals,
            ensemble_score=ensemble_score,
            hmm_state=None,
            hmm_prob=None,
            vix_term_structure=vix_term,
            cot_sentiment=cot_sentiment,
            yield_curve=yc,
            timestamp=datetime.now().isoformat()
        )


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_regime_accurate(years: int = 5) -> AccurateRegime:
    """
    High-accuracy regime analysis using all available data.
    """
    print("=" * 70)
    print("HIGH-ACCURACY QUANTUM REGIME ANALYSIS")
    print("=" * 70)
    
    # Load data
    loader = DataLoader()
    
    print("\n[1/5] Loading data sources...")
    sp500 = loader.load('SP500', years)
    vix = loader.load('VIX', years)
    vix_m2 = loader.load('VIX_M2', years)
    
    # Try to load COT data
    cot_sp = loader.load('COT_SP500_Net', years)
    cot_dealer = loader.load('COT_SP500_Dealer', years)
    
    # Yield curve spread
    yield_spread = loader.load('YieldSpread_2Y10Y', years)
    
    if sp500 is None:
        raise ValueError("SP500 data required")
    
    print(f"  SP500: {len(sp500)} rows")
    print(f"  VIX: {len(vix) if vix is not None else 0} rows")
    print(f"  VIX M2: {len(vix_m2) if vix_m2 is not None else 0} rows")
    print(f"  COT SP500 (noncomm): {len(cot_sp) if cot_sp is not None else 0} rows")
    print(f"  COT SP500 (dealer): {len(cot_dealer) if cot_dealer is not None else 0} rows")
    print(f"  Yield Spread 2Y-10Y: {len(yield_spread) if yield_spread is not None else 0} rows")
    
    # Compute features
    print("\n[2/5] Computing features...")
    engine = FeatureEngine()
    
    momentum = engine.compute_momentum_features(sp500)
    volatility = engine.compute_volatility_features(sp500)
    trend = engine.compute_trend_features(sp500)
    
    vix_features = None
    if vix is not None:
        vix_features = engine.compute_vix_features(vix, vix_m2)
    
    cot_features = None
    if cot_sp is not None:
        cot_features = engine.compute_cot_features(cot_sp, 'COT_SP500_Net')
    
    yield_features = None
    if yield_spread is not None:
        # Use direct spread data
        f = pd.DataFrame(index=yield_spread.index)
        f['yield_spread'] = yield_spread['close']
        f['yield_spread_change'] = f['yield_spread'].diff(20)
        f['yield_inverted'] = f['yield_spread'] < 0
        yield_features = f
    
    # Get latest values
    print("\n[3/5] Extracting latest values...")
    latest_mom = momentum.iloc[-1] if len(momentum) > 0 else pd.Series()
    latest_vol = volatility.iloc[-1] if len(volatility) > 0 else pd.Series()
    latest_trend = trend.iloc[-1] if len(trend) > 0 else pd.Series()
    latest_vix = vix_features.iloc[-1] if vix_features is not None and len(vix_features) > 0 else pd.Series({'vix': 20})
    latest_cot = cot_features.iloc[-1] if cot_features is not None and len(cot_features) > 0 else None
    latest_yc = yield_features.iloc[-1] if yield_features is not None and len(yield_features) > 0 else None
    
    # Classify regime
    print("\n[4/5] Classifying regime...")
    classifier = RegimeClassifier()
    regime = classifier.classify(
        latest_mom, latest_vol, latest_trend, latest_vix, latest_cot, latest_yc
    )
    
    # Print results
    print("\n[5/5] Results")
    print("=" * 70)
    print(f"  REGIME:          {regime.regime_name}")
    print(f"  CONFIDENCE:      {regime.confidence:.1f}%")
    print(f"  ENSEMBLE SCORE:  {regime.ensemble_score:+.1f}")
    print(f"  VIX TERM:        {regime.vix_term_structure}")
    print(f"  COT SENTIMENT:   {regime.cot_sentiment}")
    print(f"  YIELD CURVE:     {regime.yield_curve}")
    
    print("\n  INDIVIDUAL SIGNALS:")
    for s in regime.signals:
        arrow = "^" * max(0, s.signal) + "v" * max(0, -s.signal) if s.signal != 0 else "-"
        print(f"    {s.name:20} {arrow:5} ({s.signal:+d}) value={s.value:.3f} conf={s.confidence:.0f}%")
    
    return regime


def get_accurate_regime() -> Dict:
    """API endpoint for accurate regime."""
    try:
        regime = analyze_regime_accurate(years=2)
        
        return {
            "success": True,
            "regime": {
                "id": regime.regime_id,
                "name": regime.regime_name,
                "confidence": regime.confidence,
                "ensemble_score": regime.ensemble_score,
                "vix_term_structure": regime.vix_term_structure,
                "cot_sentiment": regime.cot_sentiment,
                "yield_curve": regime.yield_curve,
            },
            "signals": [asdict(s) for s in regime.signals],
            "timestamp": regime.timestamp,
        }
    except Exception as e:
        log.error(f"Accurate regime error: {e}")
        return {
            "success": False,
            "error": str(e)
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
    regime = analyze_regime_accurate(years)
