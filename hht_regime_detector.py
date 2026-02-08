"""
Hilbert-Huang Transform (HHT) Regime Detector
==============================================
Advanced regime detection using Empirical Mode Decomposition (EMD)
and Hilbert Transform, based on arXiv:2601.08571 research.

Why HHT > HMM:
- HHT is adaptive to non-stationary data (markets are non-stationary)
- No assumption of Gaussian returns
- Captures time-varying volatility and frequency
- Better at detecting regime transitions in real-time

Key Metrics:
- Instantaneous frequency: Rate of change in market dynamics
- Instantaneous amplitude: Volatility regime
- IMF energy distribution: Which timescales dominate

Author: AmiraB
Created: 2026-02-07
Based on: arXiv:2601.08571 "Regime Discovery and Intra-Regime Return Dynamics"
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from scipy.signal import hilbert
from scipy.interpolate import CubicSpline

DATA_DIR = Path("data/qdl_history")


class MarketRegime(Enum):
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    SIDEWAYS = "SIDEWAYS"
    BEAR = "BEAR"
    CRISIS = "CRISIS"
    TRANSITION = "TRANSITION"


class RegimePhase(Enum):
    EARLY = "EARLY"       # Just entered regime
    MIDDLE = "MIDDLE"     # Established regime
    LATE = "LATE"         # Regime may be ending
    TRANSITION = "TRANSITION"  # Between regimes


@dataclass
class IMFAnalysis:
    """Intrinsic Mode Function analysis"""
    imf_number: int
    energy: float           # % of total signal energy
    avg_period: float       # Average period in days
    volatility: float       # Amplitude volatility
    trend: float            # Recent trend direction
    
    def to_dict(self) -> Dict:
        return {
            "imf": self.imf_number,
            "energy_pct": round(self.energy, 2),
            "avg_period_days": round(self.avg_period, 1),
            "volatility": round(self.volatility, 4),
            "trend": round(self.trend, 4)
        }


@dataclass 
class HHTRegimeResult:
    """Complete HHT regime analysis"""
    timestamp: datetime
    asset: str
    
    # Current regime
    regime: MarketRegime
    regime_confidence: float
    regime_phase: RegimePhase
    regime_duration_days: int
    
    # HHT metrics
    instantaneous_frequency: float  # Current rate of dynamics
    instantaneous_amplitude: float  # Current volatility
    frequency_trend: float          # Is frequency accelerating?
    amplitude_trend: float          # Is volatility expanding?
    
    # IMF breakdown
    imf_analysis: List[IMFAnalysis] = field(default_factory=list)
    dominant_timescale: str = ""    # Which IMF dominates
    
    # Transition signals
    transition_probability: float = 0.0
    likely_next_regime: Optional[MarketRegime] = None
    
    # Historical context
    entropy: float = 0.0            # Signal complexity (higher = more uncertain)
    
    # Actionable signals
    trend_strength: float = 0.0     # -100 to +100
    volatility_regime: str = "NORMAL"  # LOW, NORMAL, HIGH, EXTREME
    mean_reversion_signal: float = 0.0  # -1 to +1
    
    # Reasoning
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "regime": {
                "current": self.regime.value,
                "confidence": round(self.regime_confidence, 1),
                "phase": self.regime_phase.value,
                "duration_days": self.regime_duration_days
            },
            "hht_metrics": {
                "instantaneous_frequency": round(self.instantaneous_frequency, 4),
                "instantaneous_amplitude": round(self.instantaneous_amplitude, 4),
                "frequency_trend": round(self.frequency_trend, 4),
                "amplitude_trend": round(self.amplitude_trend, 4),
                "entropy": round(self.entropy, 3)
            },
            "imf_breakdown": [imf.to_dict() for imf in self.imf_analysis],
            "dominant_timescale": self.dominant_timescale,
            "transition": {
                "probability": round(self.transition_probability, 1),
                "likely_next": self.likely_next_regime.value if self.likely_next_regime else None
            },
            "signals": {
                "trend_strength": round(self.trend_strength, 1),
                "volatility_regime": self.volatility_regime,
                "mean_reversion": round(self.mean_reversion_signal, 3)
            },
            "reasoning": self.reasoning
        }


def emd(signal: np.ndarray, max_imfs: int = 5, max_siftings: int = 100) -> List[np.ndarray]:
    """
    Empirical Mode Decomposition
    Decomposes signal into Intrinsic Mode Functions (IMFs)
    """
    imfs = []
    residue = signal.copy()
    
    for _ in range(max_imfs):
        h = residue.copy()
        
        for _ in range(max_siftings):
            # Find local extrema
            maxima_idx = []
            minima_idx = []
            
            for i in range(1, len(h) - 1):
                if h[i] > h[i-1] and h[i] > h[i+1]:
                    maxima_idx.append(i)
                elif h[i] < h[i-1] and h[i] < h[i+1]:
                    minima_idx.append(i)
            
            if len(maxima_idx) < 3 or len(minima_idx) < 3:
                break
            
            # Extend extrema to boundaries
            maxima_idx = [0] + maxima_idx + [len(h) - 1]
            minima_idx = [0] + minima_idx + [len(h) - 1]
            
            maxima_vals = h[maxima_idx]
            minima_vals = h[minima_idx]
            
            # Create envelopes using cubic spline
            try:
                upper_spline = CubicSpline(maxima_idx, maxima_vals)
                lower_spline = CubicSpline(minima_idx, minima_vals)
                
                x = np.arange(len(h))
                upper_env = upper_spline(x)
                lower_env = lower_spline(x)
                
                mean_env = (upper_env + lower_env) / 2
                h = h - mean_env
                
                # Check if IMF conditions are met (simplified)
                if np.std(mean_env) < 0.01 * np.std(h):
                    break
            except Exception:
                break
        
        imfs.append(h)
        residue = residue - h
        
        # Stop if residue is monotonic
        if len(np.where(np.diff(np.sign(np.diff(residue))))[0]) < 2:
            break
    
    return imfs


def hilbert_transform(imf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Hilbert Transform to get instantaneous amplitude and frequency
    """
    analytic_signal = hilbert(imf)
    amplitude = np.abs(analytic_signal)
    
    # Instantaneous phase
    phase = np.unwrap(np.angle(analytic_signal))
    
    # Instantaneous frequency (derivative of phase)
    frequency = np.diff(phase) / (2 * np.pi)
    frequency = np.append(frequency, frequency[-1])  # Pad to same length
    
    # Ensure positive frequencies
    frequency = np.abs(frequency)
    
    return amplitude, frequency


def compute_imf_energy(imf: np.ndarray) -> float:
    """Compute energy (variance) of an IMF"""
    return np.var(imf)


def estimate_period(frequency: np.ndarray) -> float:
    """Estimate average period from instantaneous frequency"""
    mean_freq = np.mean(frequency[frequency > 0.001])  # Filter near-zero
    if mean_freq > 0:
        return 1.0 / mean_freq
    return 0.0


def compute_entropy(imfs: List[np.ndarray]) -> float:
    """Compute signal entropy from IMF energy distribution"""
    energies = np.array([compute_imf_energy(imf) for imf in imfs])
    total_energy = np.sum(energies)
    
    if total_energy == 0:
        return 0.0
    
    # Normalized energy distribution
    p = energies / total_energy
    p = p[p > 0]  # Remove zeros for log
    
    # Shannon entropy (normalized)
    entropy = -np.sum(p * np.log2(p)) / np.log2(len(p)) if len(p) > 1 else 0
    
    return entropy


def classify_regime(
    trend: float,
    volatility: float,
    entropy: float,
    frequency_trend: float
) -> Tuple[MarketRegime, float]:
    """Classify market regime based on HHT metrics"""
    
    confidence = 70.0
    
    # Strong trends with low volatility
    if trend > 15 and volatility < 0.015:
        regime = MarketRegime.STRONG_BULL
        confidence = min(95, 75 + abs(trend))
    
    elif trend < -15 and volatility > 0.025:
        regime = MarketRegime.CRISIS
        confidence = min(95, 75 + abs(trend))
    
    elif trend > 5:
        regime = MarketRegime.BULL
        confidence = 65 + abs(trend) * 0.5
    
    elif trend < -5:
        regime = MarketRegime.BEAR
        confidence = 65 + abs(trend) * 0.5
    
    # High entropy = uncertain, transitional
    elif entropy > 0.8:
        regime = MarketRegime.TRANSITION
        confidence = 50
    
    else:
        regime = MarketRegime.SIDEWAYS
        confidence = 55
    
    return regime, min(95, confidence)


def classify_phase(regime_duration: int, transition_prob: float) -> RegimePhase:
    """Classify regime phase"""
    
    if transition_prob > 60:
        return RegimePhase.TRANSITION
    
    if regime_duration < 10:
        return RegimePhase.EARLY
    elif regime_duration > 60:
        return RegimePhase.LATE
    else:
        return RegimePhase.MIDDLE


def classify_volatility(amplitude: float, historical_amplitudes: np.ndarray) -> str:
    """Classify volatility regime"""
    percentile = (historical_amplitudes < amplitude).sum() / len(historical_amplitudes) * 100
    
    if percentile > 90:
        return "EXTREME"
    elif percentile > 75:
        return "HIGH"
    elif percentile < 25:
        return "LOW"
    else:
        return "NORMAL"


def analyze_asset(asset: str) -> Optional[HHTRegimeResult]:
    """Run full HHT regime analysis on an asset"""
    
    # Asset file mapping
    file_map = {
        "SP500": "SP500.csv",
        "NASDAQ": "NASDAQ.csv",
        "GOLD": "GOLD.csv",
        "CRUDE": "Crude_Oil.csv",
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
    except Exception:
        return None
    
    if len(df) < 200:
        return None
    
    # Use log returns for stationarity
    prices = df["close"].values
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    
    # EMD decomposition
    imfs = emd(returns[-252:])  # Last year of data
    
    if len(imfs) < 2:
        return None
    
    # Analyze each IMF
    imf_analyses = []
    total_energy = sum(compute_imf_energy(imf) for imf in imfs)
    
    for i, imf in enumerate(imfs):
        amplitude, frequency = hilbert_transform(imf)
        energy = compute_imf_energy(imf)
        period = estimate_period(frequency)
        
        imf_analyses.append(IMFAnalysis(
            imf_number=i + 1,
            energy=energy / total_energy * 100 if total_energy > 0 else 0,
            avg_period=period,
            volatility=np.std(amplitude),
            trend=np.mean(imf[-20:]) - np.mean(imf[-60:-20]) if len(imf) > 60 else 0
        ))
    
    # Find dominant timescale
    dominant_idx = np.argmax([imf.energy for imf in imf_analyses])
    dominant = imf_analyses[dominant_idx]
    
    if dominant.avg_period < 5:
        dominant_timescale = "SHORT-TERM (< 1 week)"
    elif dominant.avg_period < 22:
        dominant_timescale = "MEDIUM-TERM (1-4 weeks)"
    elif dominant.avg_period < 66:
        dominant_timescale = "INTERMEDIATE (1-3 months)"
    else:
        dominant_timescale = "LONG-TERM (> 3 months)"
    
    # Aggregate HHT metrics from first IMF (highest frequency)
    amplitude_1, frequency_1 = hilbert_transform(imfs[0])
    inst_amplitude = amplitude_1[-1]
    inst_frequency = frequency_1[-1]
    
    # Trends
    amplitude_trend = np.mean(amplitude_1[-10:]) - np.mean(amplitude_1[-30:-10])
    frequency_trend = np.mean(frequency_1[-10:]) - np.mean(frequency_1[-30:-10])
    
    # Price trend (60-day return)
    price_trend = (prices[-1] / prices[-60] - 1) * 100 if len(prices) > 60 else 0
    
    # Entropy
    entropy = compute_entropy(imfs)
    
    # Volatility
    volatility = np.std(returns[-20:]) * np.sqrt(252)
    volatility_regime = classify_volatility(inst_amplitude, amplitude_1)
    
    # Regime classification
    regime, confidence = classify_regime(
        price_trend, 
        volatility, 
        entropy, 
        frequency_trend
    )
    
    # Transition probability (based on entropy and phase)
    transition_prob = min(90, entropy * 100 * 0.7 + abs(frequency_trend) * 1000)
    
    # Likely next regime
    likely_next = None
    if transition_prob > 50:
        if regime in [MarketRegime.BULL, MarketRegime.STRONG_BULL]:
            if amplitude_trend > 0:  # Volatility expanding
                likely_next = MarketRegime.SIDEWAYS
        elif regime == MarketRegime.BEAR:
            if amplitude_trend < 0:  # Volatility contracting
                likely_next = MarketRegime.SIDEWAYS
        elif regime == MarketRegime.CRISIS:
            likely_next = MarketRegime.BEAR
        elif regime == MarketRegime.SIDEWAYS:
            if price_trend > 0:
                likely_next = MarketRegime.BULL
            else:
                likely_next = MarketRegime.BEAR
    
    # Regime duration (simplified - count days in similar regime)
    regime_duration = 30  # Placeholder - would need historical regime tracking
    
    # Phase
    phase = classify_phase(regime_duration, transition_prob)
    
    # Trend strength (-100 to +100)
    trend_strength = np.clip(price_trend * 3, -100, 100)
    
    # Mean reversion signal
    # If price deviates significantly from recent mean
    recent_mean = np.mean(prices[-20:])
    deviation = (prices[-1] - recent_mean) / recent_mean
    mean_reversion = -np.clip(deviation * 10, -1, 1)  # Opposite of deviation
    
    # Build reasoning
    reasoning = []
    reasoning.append(f"Regime: {regime.value} with {confidence:.0f}% confidence")
    reasoning.append(f"Dominant timescale: {dominant_timescale} ({dominant.energy:.1f}% of signal energy)")
    reasoning.append(f"Signal entropy: {entropy:.2f} ({'high uncertainty' if entropy > 0.7 else 'structured'})")
    
    if amplitude_trend > 0:
        reasoning.append(f"Volatility expanding ({volatility_regime} regime)")
    else:
        reasoning.append(f"Volatility contracting ({volatility_regime} regime)")
    
    if transition_prob > 50:
        reasoning.append(f"Elevated transition probability: {transition_prob:.0f}%")
        if likely_next:
            reasoning.append(f"Likely transitioning to: {likely_next.value}")
    
    return HHTRegimeResult(
        timestamp=datetime.now(),
        asset=asset,
        regime=regime,
        regime_confidence=confidence,
        regime_phase=phase,
        regime_duration_days=regime_duration,
        instantaneous_frequency=inst_frequency,
        instantaneous_amplitude=inst_amplitude,
        frequency_trend=frequency_trend,
        amplitude_trend=amplitude_trend,
        imf_analysis=imf_analyses,
        dominant_timescale=dominant_timescale,
        transition_probability=transition_prob,
        likely_next_regime=likely_next,
        entropy=entropy,
        trend_strength=trend_strength,
        volatility_regime=volatility_regime,
        mean_reversion_signal=mean_reversion,
        reasoning=reasoning
    )


def get_hht_for_api(asset: str) -> Dict:
    """Get HHT analysis formatted for API"""
    result = analyze_asset(asset)
    
    if result is None:
        return {"error": f"Unable to analyze {asset}"}
    
    return result.to_dict()


def analyze_all_assets() -> Dict[str, Dict]:
    """Analyze all assets"""
    assets = ["SP500", "GOLD", "CRUDE", "BITCOIN", "NASDAQ"]
    results = {}
    
    for asset in assets:
        result = analyze_asset(asset)
        if result:
            results[asset] = result.to_dict()
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("HHT (HILBERT-HUANG TRANSFORM) REGIME ANALYSIS")
    print("Based on arXiv:2601.08571")
    print("=" * 70)
    
    for asset in ["SP500", "GOLD", "CRUDE", "BITCOIN"]:
        result = analyze_asset(asset)
        
        if result:
            print(f"\n{'-' * 60}")
            print(f"[{asset}]")
            print(f"Regime: {result.regime.value} ({result.regime_confidence:.0f}% conf)")
            print(f"Phase: {result.regime_phase.value}")
            print(f"Volatility: {result.volatility_regime}")
            print(f"Trend Strength: {result.trend_strength:+.0f}")
            print(f"Entropy: {result.entropy:.2f}")
            print(f"Transition Prob: {result.transition_probability:.0f}%")
            if result.likely_next_regime:
                print(f"Likely Next: {result.likely_next_regime.value}")
            print(f"Dominant Timescale: {result.dominant_timescale}")
            print(f"Mean Reversion Signal: {result.mean_reversion_signal:+.2f}")
