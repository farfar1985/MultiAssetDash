"""
ENHANCED QUANTUM REGIME DETECTOR
================================
Incorporates additional data sources for improved regime detection:

1. Implied Volatility (VIX-like forward-looking)
2. Trader Positioning (CFTC COT sentiment)
3. News Sentiment (if available)
4. Cross-Asset Signals (contagion-aware)
5. Macro Indicators (interest rates, credit spreads)

Data Sources Available in quantum_ml:
- ingest_cftc_cot.py - Commitments of Traders
- ingest_cme_cvol.py - CME Volatility Index
- ingest_cboe_indices.py - VIX and related
- ingest_fred.py - Federal Reserve data
- ingest_gdelt.py - News sentiment

Created: 2026-02-05
Author: AmiraB
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, entropy, DensityMatrix

from per_asset_optimizer import load_asset_data
from quantum_regime_v2 import QuantumRegimeDetectorV2


class DataSourceManager:
    """
    Manages loading and caching of external data sources.
    
    Currently uses price-derived proxies. Can be extended to load
    real data from quantum_ml ingestion pipeline.
    """
    
    def __init__(self, asset_dir: str):
        self.asset_dir = Path(asset_dir)
        self.cache = {}
        
    def load_price_data(self) -> np.ndarray:
        """Load price data from parquet or cache."""
        if 'prices' not in self.cache:
            parquet_path = self.asset_dir / 'training_data.parquet'
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
                df = df[df['time'] >= '2025-01-01']
                prices = df.groupby('time')['target_var_price'].first().sort_index().values
                self.cache['prices'] = prices
            else:
                self.cache['prices'] = None
        return self.cache['prices']
    
    def calculate_implied_vol_proxy(self, prices: np.ndarray, t: int, 
                                     lookback: int = 20) -> float:
        """
        Calculate implied volatility proxy.
        
        Real implementation would use:
        - CME CVOL data (ingest_cme_cvol.py)
        - VIX for equity indices (ingest_cboe_indices.py)
        - Crypto vol indices for BTC
        
        Proxy: Uses realized vol with forward-looking adjustment based on
        recent vol trend (if vol is rising, implied > realized).
        """
        if t < lookback:
            return 0.2
        
        window = prices[t - lookback:t]
        returns = np.diff(window) / (window[:-1] + 1e-10)
        realized_vol = returns.std() * np.sqrt(252)
        
        # Vol trend adjustment (proxy for implied > realized during stress)
        if t >= lookback * 2:
            early_vol = np.diff(prices[t-lookback*2:t-lookback]) / (prices[t-lookback*2:t-lookback-1] + 1e-10)
            early_vol = early_vol.std() * np.sqrt(252)
            vol_trend = (realized_vol - early_vol) / (early_vol + 1e-10)
            
            # Implied vol tends to be higher when realized vol is rising
            implied_premium = max(0, vol_trend * 0.2)
            implied_vol = realized_vol * (1 + implied_premium)
        else:
            implied_vol = realized_vol * 1.1  # Default 10% premium
        
        return implied_vol
    
    def calculate_positioning_proxy(self, prices: np.ndarray, t: int,
                                     lookback: int = 20) -> float:
        """
        Calculate trader positioning proxy.
        
        Real implementation would use:
        - CFTC COT data (ingest_cftc_cot.py)
        - Long/short ratios from futures markets
        
        Proxy: Uses price momentum and mean reversion signals.
        Extreme momentum = crowded positioning.
        """
        if t < lookback:
            return 0
        
        window = prices[t - lookback:t]
        
        # Momentum as positioning proxy
        momentum = (window[-1] - window[0]) / (window[0] + 1e-10)
        
        # Normalize to [-1, 1] (extreme long = 1, extreme short = -1)
        positioning = np.tanh(momentum * 5)
        
        return positioning
    
    def calculate_sentiment_proxy(self, prices: np.ndarray, t: int,
                                   lookback: int = 5) -> float:
        """
        Calculate news/market sentiment proxy.
        
        Real implementation would use:
        - GDELT news sentiment (ingest_gdelt.py)
        - Social media sentiment
        - RSS news analysis (ingest_rss_news.py)
        
        Proxy: Short-term momentum direction and magnitude.
        """
        if t < lookback:
            return 0
        
        window = prices[t - lookback:t]
        
        # Short-term trend as sentiment proxy
        up_days = sum(np.diff(window) > 0)
        sentiment = (up_days / (lookback - 1)) * 2 - 1  # Normalize to [-1, 1]
        
        return sentiment
    
    def calculate_macro_proxy(self, prices: np.ndarray, t: int,
                               lookback: int = 60) -> dict:
        """
        Calculate macro environment proxy.
        
        Real implementation would use:
        - FRED data (ingest_fred.py): interest rates, credit spreads
        - Treasury yields (ingest_us_treasury.py)
        
        Proxy: Long-term trend and volatility regime.
        """
        if t < lookback:
            return {'trend': 0, 'stability': 0.5}
        
        window = prices[t - lookback:t]
        
        # Long-term trend
        trend = (window[-1] - window[0]) / (window[0] + 1e-10)
        
        # Stability (inverse of long-term vol)
        returns = np.diff(window) / (window[:-1] + 1e-10)
        long_vol = returns.std()
        stability = 1 / (1 + long_vol * 10)
        
        return {
            'trend': np.tanh(trend * 3),
            'stability': stability
        }


class EnhancedQuantumRegimeDetector:
    """
    Enhanced regime detector incorporating multiple data sources.
    
    Quantum encoding:
    - Qubit 0-1: Price/volatility features
    - Qubit 2: Implied volatility
    - Qubit 3: Positioning/sentiment
    - Qubit 4: Macro environment
    - Qubit 5: Cross-asset contagion
    """
    
    def __init__(self, n_qubits: int = 6):
        self.n_qubits = n_qubits
        self.base_detector = QuantumRegimeDetectorV2()
        self.data_manager = None
        
    def initialize(self, asset_dir: str, prices: np.ndarray):
        """Initialize with asset data."""
        self.data_manager = DataSourceManager(asset_dir)
        self.base_detector.train(prices)
        
    def encode_enhanced_state(self, prices: np.ndarray, t: int,
                               contagion_score: float = 0) -> QuantumCircuit:
        """Encode all data sources into quantum state."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Feature extraction
        implied_vol = self.data_manager.calculate_implied_vol_proxy(prices, t)
        positioning = self.data_manager.calculate_positioning_proxy(prices, t)
        sentiment = self.data_manager.calculate_sentiment_proxy(prices, t)
        macro = self.data_manager.calculate_macro_proxy(prices, t)
        
        # Calculate realized vol
        if t >= 20:
            window = prices[t-20:t]
            returns = np.diff(window) / (window[:-1] + 1e-10)
            realized_vol = returns.std() * np.sqrt(252)
        else:
            realized_vol = 0.2
        
        # Qubit 0-1: Volatility state
        vol_angle = np.clip(realized_vol, 0, 1) * np.pi
        qc.ry(vol_angle, 0)
        qc.ry(vol_angle * 0.8, 1)
        qc.cx(0, 1)
        
        # Qubit 2: Implied volatility (forward-looking)
        iv_angle = np.clip(implied_vol, 0, 1) * np.pi
        qc.ry(iv_angle, 2)
        
        # Qubit 3: Positioning/Sentiment
        pos_angle = (positioning + 1) / 2 * np.pi
        sent_angle = (sentiment + 1) / 2 * np.pi
        qc.ry(pos_angle, 3)
        qc.rz(sent_angle, 3)
        
        # Qubit 4: Macro environment
        macro_angle = (macro['trend'] + 1) / 2 * np.pi
        stability_angle = macro['stability'] * np.pi
        qc.ry(macro_angle, 4)
        qc.rz(stability_angle, 4)
        
        # Qubit 5: Contagion
        cont_angle = np.clip(contagion_score, 0, 1) * np.pi
        qc.ry(cont_angle, 5)
        
        # Entangle all sources (market interconnection)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(self.n_qubits - 1, 0)  # Circular
        
        return qc
    
    def detect_enhanced(self, prices: np.ndarray, t: int,
                        contagion_score: float = 0) -> dict:
        """
        Detect regime with enhanced data sources.
        """
        # Base detection
        base_result = self.base_detector.detect(prices, t)
        
        # Enhanced quantum encoding
        qc = self.encode_enhanced_state(prices, t, contagion_score)
        sv = Statevector.from_instruction(qc)
        
        # Calculate quantum metrics
        dm = DensityMatrix(sv)
        total_entropy = entropy(dm, base=2)
        
        # Measure entanglement between price and external factors
        from qiskit.quantum_info import partial_trace
        
        # Price qubits entropy
        rho_price = partial_trace(sv, [2, 3, 4, 5])
        price_entropy = entropy(rho_price, base=2)
        
        # External factors entropy
        rho_external = partial_trace(sv, [0, 1])
        external_entropy = entropy(rho_external, base=2)
        
        # Calculate additional features
        implied_vol = self.data_manager.calculate_implied_vol_proxy(prices, t)
        positioning = self.data_manager.calculate_positioning_proxy(prices, t)
        sentiment = self.data_manager.calculate_sentiment_proxy(prices, t)
        macro = self.data_manager.calculate_macro_proxy(prices, t)
        
        # Regime confidence adjustment based on external factors
        # High implied vol premium = regime stress
        if t >= 20:
            window2 = prices[t-20:t]
            returns2 = np.diff(window2) / (window2[:-1] + 1e-10)
            realized_vol = returns2.std() * np.sqrt(252)
        else:
            realized_vol = 0.2
        
        vol_premium = (implied_vol - realized_vol) / (realized_vol + 1e-10)
        
        # Crowded positioning warning
        crowded = abs(positioning) > 0.7
        
        # Enhanced regime assessment
        enhanced_result = {
            **base_result,
            
            # Enhanced features
            'implied_volatility': round(implied_vol, 4),
            'vol_premium': round(vol_premium, 4),
            'positioning': round(positioning, 4),
            'sentiment': round(sentiment, 4),
            'macro_trend': round(macro['trend'], 4),
            'macro_stability': round(macro['stability'], 4),
            'contagion_score': round(contagion_score, 4),
            
            # Quantum metrics
            'total_entropy': round(total_entropy, 4),
            'price_entropy': round(price_entropy, 4),
            'external_entropy': round(external_entropy, 4),
            
            # Warnings
            'crowded_positioning': crowded,
            'stress_indicator': vol_premium > 0.2,
            
            # Enhanced confidence
            'enhanced_confidence': round(
                base_result['confidence'] * (1 - abs(vol_premium) * 0.3) * 
                (1 - contagion_score * 0.2),
                3
            )
        }
        
        return enhanced_result


def backtest_enhanced_detector():
    """Backtest the enhanced quantum regime detector."""
    print("=" * 60)
    print("ENHANCED QUANTUM REGIME DETECTOR BACKTEST")
    print("=" * 60)
    
    dirs = {
        'SP500': 'data/1625_SP500',
        'Bitcoin': 'data/1860_Bitcoin',
        'Crude_Oil': 'data/1866_Crude_Oil'
    }
    
    all_results = {}
    
    for asset_name, asset_dir in dirs.items():
        print(f"\n{'-'*40}")
        print(f"Testing: {asset_name}")
        print(f"{'-'*40}")
        
        horizons, prices = load_asset_data(asset_dir)
        if prices is None:
            continue
        
        # Initialize detector
        detector = EnhancedQuantumRegimeDetector()
        detector.initialize(asset_dir, prices)
        
        # Run detection
        results = []
        
        for t in range(60, len(prices) - 5):
            if t % 50 == 0:
                print(f"  Processing t={t}/{len(prices)}")
            
            # Simulate contagion score (would come from contagion detector in production)
            contagion = np.random.uniform(0.1, 0.4)  # Placeholder
            
            result = detector.detect_enhanced(prices, t, contagion)
            results.append(result)
        
        # Analyze
        regimes = [r['regime'] for r in results]
        vol_premiums = [r['vol_premium'] for r in results]
        crowded = [r['crowded_positioning'] for r in results]
        stress = [r['stress_indicator'] for r in results]
        
        regime_dist = pd.Series(regimes).value_counts()
        
        print(f"\n  REGIME DISTRIBUTION:")
        for regime, count in regime_dist.items():
            print(f"    {regime}: {count} ({count/len(regimes)*100:.1f}%)")
        
        print(f"\n  ENHANCED METRICS:")
        print(f"    Avg Vol Premium: {np.mean(vol_premiums)*100:.1f}%")
        print(f"    Crowded Positioning: {sum(crowded)} periods ({sum(crowded)/len(crowded)*100:.1f}%)")
        print(f"    Stress Periods: {sum(stress)} ({sum(stress)/len(stress)*100:.1f}%)")
        # Use price_entropy (subsystem entropy) which shows entanglement
        # total_entropy is ~0 for pure states, but subsystem entropy reveals correlations
        avg_price_entropy = np.mean([r['price_entropy'] for r in results])
        avg_external_entropy = np.mean([r['external_entropy'] for r in results])
        print(f"    Avg Price Entropy: {avg_price_entropy:.3f}")
        print(f"    Avg External Entropy: {avg_external_entropy:.3f}")
        
        all_results[asset_name] = {
            'regime_distribution': regime_dist.to_dict(),
            'avg_vol_premium': round(np.mean(vol_premiums), 4),
            'crowded_pct': round(sum(crowded)/len(crowded)*100, 1),
            'stress_pct': round(sum(stress)/len(stress)*100, 1),
            'avg_price_entropy': round(avg_price_entropy, 3),
            'avg_external_entropy': round(avg_external_entropy, 3),
            'avg_enhanced_confidence': round(np.mean([r['enhanced_confidence'] for r in results]), 3)
        }
    
    # Save
    output = {
        'generated_at': datetime.now().isoformat(),
        'version': 'enhanced',
        'data_sources': [
            'price_volatility',
            'implied_vol_proxy',
            'positioning_proxy',
            'sentiment_proxy',
            'macro_proxy',
            'contagion_score'
        ],
        'potential_real_sources': [
            'ingest_cme_cvol.py',
            'ingest_cftc_cot.py',
            'ingest_cboe_indices.py',
            'ingest_fred.py',
            'ingest_gdelt.py'
        ],
        'results': all_results
    }
    
    output_path = Path('configs/quantum_regime_enhanced_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    results = backtest_enhanced_detector()
