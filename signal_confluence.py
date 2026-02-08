"""
Signal Confluence Engine — The Brain of QDT Trading Intelligence

Aggregates ALL signals (VIX, COT, Regime, Energy, Crypto, Technicals) into a
unified conviction score with multi-timeframe alignment detection.

This is what separates QDT from Bloomberg terminals and TradingView charts.
We don't just show indicators — we synthesize intelligence.

Author: AmiraB
Created: 2026-02-07
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Import our signal modules
try:
    from smart_money_signals import generate_cot_signal, COTSignal
    HAS_SMART_MONEY = True
except ImportError:
    HAS_SMART_MONEY = False
    generate_cot_signal = None
    COTSignal = None

try:
    from vix_intelligence import analyze_vix
    HAS_VIX = True
except ImportError:
    HAS_VIX = False
    analyze_vix = None

try:
    from crypto_onchain_signals import analyze_bitcoin
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    analyze_bitcoin = None


class SignalStrength(Enum):
    """Signal conviction levels"""
    STRONG_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    STRONG_BEARISH = -2


class Timeframe(Enum):
    """Analysis timeframes"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class Signal:
    """Individual signal with metadata"""
    name: str
    source: str
    value: float
    direction: SignalStrength
    confidence: float  # 0-100
    timeframe: Timeframe
    historical_win_rate: float  # Historical accuracy
    last_updated: datetime
    description: str
    
    def to_dict(self):
        return {
            "name": self.name,
            "source": self.source,
            "value": self.value,
            "direction": self.direction.name,
            "direction_value": self.direction.value,
            "confidence": self.confidence,
            "timeframe": self.timeframe.value,
            "historical_win_rate": self.historical_win_rate,
            "last_updated": self.last_updated.isoformat(),
            "description": self.description
        }


@dataclass
class ConfluenceResult:
    """Unified confluence analysis result"""
    asset: str
    timestamp: datetime
    
    # Core metrics (with defaults for initialization)
    conviction_score: float = 0.0  # -100 to +100
    conviction_label: str = "NEUTRAL"  # "STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"
    confidence: float = 0.0  # 0-100, based on signal agreement
    
    # Signal breakdown
    signals: list[Signal] = field(default_factory=list)
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    
    # Multi-timeframe alignment
    daily_bias: float = 0.0
    weekly_bias: float = 0.0
    monthly_bias: float = 0.0
    timeframe_alignment: float = 0.0  # 0-100, how aligned are timeframes
    
    # Historical context
    similar_setups_count: int = 0
    historical_avg_return: float = 0.0
    historical_win_rate: float = 0.0
    
    # Risk metrics
    risk_score: float = 0.0  # 0-100
    suggested_position_size: float = 0.0  # % of portfolio
    
    # Narrative
    headline: str = ""
    key_drivers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "asset": self.asset,
            "timestamp": self.timestamp.isoformat(),
            "conviction_score": round(self.conviction_score, 1),
            "conviction_label": self.conviction_label,
            "confidence": round(self.confidence, 1),
            "signals": [s.to_dict() for s in self.signals],
            "signal_counts": {
                "bullish": self.bullish_count,
                "bearish": self.bearish_count,
                "neutral": self.neutral_count,
                "total": len(self.signals)
            },
            "timeframe_analysis": {
                "daily_bias": round(self.daily_bias, 1),
                "weekly_bias": round(self.weekly_bias, 1),
                "monthly_bias": round(self.monthly_bias, 1),
                "alignment": round(self.timeframe_alignment, 1)
            },
            "historical_context": {
                "similar_setups": self.similar_setups_count,
                "avg_return": round(self.historical_avg_return, 2),
                "win_rate": round(self.historical_win_rate, 1)
            },
            "risk": {
                "score": round(self.risk_score, 1),
                "suggested_position_pct": round(self.suggested_position_size, 2)
            },
            "narrative": {
                "headline": self.headline,
                "key_drivers": self.key_drivers,
                "risks": self.risks
            }
        }


class SignalConfluenceEngine:
    """
    The core intelligence engine that synthesizes all trading signals
    into actionable, contextual intelligence.
    """
    
    # Signal weights by source (higher = more important)
    SIGNAL_WEIGHTS = {
        "cot": 1.5,           # COT is highly predictive
        "vix": 1.3,           # VIX structure matters
        "regime": 1.2,        # Regime context is crucial
        "onchain": 1.2,       # On-chain for crypto
        "energy": 1.1,        # Energy fundamentals
        "technical": 1.0,     # Standard technicals
        "sentiment": 0.8,     # Sentiment is noisy
    }
    
    # Historical win rates by signal type (from backtests)
    HISTORICAL_WIN_RATES = {
        "cot_extreme_long": 0.72,
        "cot_extreme_short": 0.68,
        "vix_spike": 0.65,
        "vix_contango": 0.62,
        "regime_bull": 0.70,
        "regime_bear": 0.68,
        "mvrv_undervalued": 0.75,
        "mvrv_overvalued": 0.80,
        "nvt_low": 0.65,
        "confluence_5plus": 0.78,  # 5+ signals aligned
        "confluence_6plus": 0.82,  # 6+ signals aligned
    }
    
    # Asset name mapping to data files
    ASSET_FILE_MAP = {
        "SP500": "SP500.csv",
        "NASDAQ": "NASDAQ.csv",
        "GOLD": "GOLD.csv",
        "CRUDE": "Crude_Oil.csv",
        "BITCOIN": "Bitcoin.csv",
        "DOW": "Dow_Jones.csv",
        "RUSSELL": "Russell_2000.csv",
    }
    
    def __init__(self, data_dir: str = "data/qdl_history"):
        self.data_dir = Path(data_dir)
        
    def analyze(self, asset: str) -> ConfluenceResult:
        """
        Run full confluence analysis for an asset.
        Returns unified conviction score with all context.
        """
        result = ConfluenceResult(
            asset=asset,
            timestamp=datetime.now()
        )
        
        # Collect all signals
        signals = []
        
        # 1. COT / Smart Money signals
        cot_signals = self._get_cot_signals(asset)
        signals.extend(cot_signals)
        
        # 2. VIX signals (for equity-related assets)
        if asset in ["SP500", "NASDAQ", "DOW", "RUSSELL"]:
            vix_signals = self._get_vix_signals()
            signals.extend(vix_signals)
        
        # 3. Crypto on-chain (for crypto assets)
        if asset in ["BITCOIN", "BTC"]:
            crypto_sigs = self._get_crypto_signals()
            signals.extend(crypto_sigs)
        
        # 4. Technical signals
        tech_signals = self._get_technical_signals(asset)
        signals.extend(tech_signals)
        
        # 5. Regime signals
        regime_signals = self._get_regime_signals(asset)
        signals.extend(regime_signals)
        
        result.signals = signals
        
        # Calculate conviction score
        self._calculate_conviction(result)
        
        # Analyze timeframe alignment
        self._analyze_timeframes(result)
        
        # Generate narrative
        self._generate_narrative(result)
        
        # Calculate risk and position sizing
        self._calculate_risk(result)
        
        return result
    
    def _get_cot_signals(self, asset: str) -> list[Signal]:
        """Get COT-based signals"""
        signals = []
        
        if not HAS_SMART_MONEY:
            return signals
        
        # Map asset to COT symbol
        cot_map = {
            "GOLD": "GC",
            "CRUDE": "CL",
            "SP500": "SP500",
        }
        
        cot_name = cot_map.get(asset.upper())
        if not cot_name:
            return signals
            
        try:
            cot_signal = generate_cot_signal(cot_name)
            if cot_signal:
                z_score = cot_signal.z_score
                percentile = cot_signal.percentile
                
                # Determine signal direction
                if z_score < -2.0:
                    direction = SignalStrength.STRONG_BULLISH  # Contrarian
                    win_rate = self.HISTORICAL_WIN_RATES["cot_extreme_long"]
                    desc = f"Commercials extremely short (z={z_score:.2f}) — contrarian BUY"
                elif z_score < -1.0:
                    direction = SignalStrength.BULLISH
                    win_rate = 0.65
                    desc = f"Commercials moderately short — leaning bullish"
                elif z_score > 2.0:
                    direction = SignalStrength.STRONG_BEARISH  # Contrarian
                    win_rate = self.HISTORICAL_WIN_RATES["cot_extreme_short"]
                    desc = f"Commercials extremely long (z={z_score:.2f}) — contrarian SELL"
                elif z_score > 1.0:
                    direction = SignalStrength.BEARISH
                    win_rate = 0.62
                    desc = f"Commercials moderately long — leaning bearish"
                else:
                    direction = SignalStrength.NEUTRAL
                    win_rate = 0.50
                    desc = f"COT positioning neutral (z={z_score:.2f})"
                
                signals.append(Signal(
                    name="COT Positioning",
                    source="cot",
                    value=z_score,
                    direction=direction,
                    confidence=min(abs(z_score) * 40, 95),
                    timeframe=Timeframe.WEEKLY,
                    historical_win_rate=win_rate,
                    last_updated=datetime.now(),
                    description=desc
                ))
        except Exception as e:
            pass  # Silently handle missing data
            
        return signals
    
    def _get_vix_signals(self) -> list[Signal]:
        """Get VIX-based signals"""
        signals = []
        
        if not HAS_VIX:
            return signals
        
        try:
            vix_analysis = analyze_vix()
            if vix_analysis:
                level = vix_analysis.current_level
                percentile = vix_analysis.percentile
                term_structure = vix_analysis.term_structure or "normal"
                
                # VIX level signal
                if level > 30:
                    direction = SignalStrength.BULLISH  # High VIX = fear = buy opportunity
                    win_rate = self.HISTORICAL_WIN_RATES["vix_spike"]
                    desc = f"VIX elevated at {level:.1f} — fear spike, potential buy"
                elif level < 15:
                    direction = SignalStrength.BEARISH  # Complacency
                    win_rate = 0.58
                    desc = f"VIX low at {level:.1f} — complacency warning"
                else:
                    direction = SignalStrength.NEUTRAL
                    win_rate = 0.50
                    desc = f"VIX normal at {level:.1f}"
                
                signals.append(Signal(
                    name="VIX Level",
                    source="vix",
                    value=level,
                    direction=direction,
                    confidence=60,
                    timeframe=Timeframe.DAILY,
                    historical_win_rate=win_rate,
                    last_updated=datetime.now(),
                    description=desc
                ))
                
                # Term structure signal
                if term_structure == "backwardation":
                    signals.append(Signal(
                        name="VIX Term Structure",
                        source="vix",
                        value=1,
                        direction=SignalStrength.BULLISH,
                        confidence=70,
                        timeframe=Timeframe.DAILY,
                        historical_win_rate=self.HISTORICAL_WIN_RATES["vix_contango"],
                        last_updated=datetime.now(),
                        description="VIX in backwardation — near-term fear, often marks bottoms"
                    ))
                    
        except Exception:
            pass
            
        return signals
    
    def _get_crypto_signals(self) -> list[Signal]:
        """Get crypto on-chain signals"""
        signals = []
        
        if not HAS_CRYPTO:
            return signals
        
        try:
            analysis = analyze_bitcoin()
            if analysis:
                mvrv_data = {"value": analysis.mvrv.value if analysis.mvrv else 1.0}
                nvt_data = {"value": analysis.nvt.value if analysis.nvt else 50.0}
                
                # MVRV signal
                mvrv_value = mvrv_data.get("value", 1.0)
                if mvrv_value < 1.0:
                    direction = SignalStrength.STRONG_BULLISH
                    win_rate = self.HISTORICAL_WIN_RATES["mvrv_undervalued"]
                    desc = f"MVRV {mvrv_value:.2f} — market below realized value, ACCUMULATE"
                elif mvrv_value > 3.5:
                    direction = SignalStrength.STRONG_BEARISH
                    win_rate = self.HISTORICAL_WIN_RATES["mvrv_overvalued"]
                    desc = f"MVRV {mvrv_value:.2f} — severely overvalued, DISTRIBUTE"
                elif mvrv_value > 2.5:
                    direction = SignalStrength.BEARISH
                    win_rate = 0.70
                    desc = f"MVRV {mvrv_value:.2f} — getting stretched"
                else:
                    direction = SignalStrength.NEUTRAL
                    win_rate = 0.50
                    desc = f"MVRV {mvrv_value:.2f} — fair value range"
                
                signals.append(Signal(
                    name="MVRV Ratio",
                    source="onchain",
                    value=mvrv_value,
                    direction=direction,
                    confidence=80,
                    timeframe=Timeframe.WEEKLY,
                    historical_win_rate=win_rate,
                    last_updated=datetime.now(),
                    description=desc
                ))
                
        except Exception:
            pass
            
        return signals
    
    def _get_technical_signals(self, asset: str) -> list[Signal]:
        """Get technical analysis signals"""
        signals = []
        
        # Load price data using asset mapping
        file_name = self.ASSET_FILE_MAP.get(asset.upper(), f"{asset}.csv")
        price_file = self.data_dir / file_name
        
        if not price_file.exists():
            # Try alternate names
            for alt in [f"{asset.upper()}.csv", f"{asset.lower()}.csv", f"{asset}.csv"]:
                alt_file = self.data_dir / alt
                if alt_file.exists():
                    price_file = alt_file
                    break
        
        if not price_file.exists():
            return signals
            
        try:
            df = pd.read_csv(price_file)
            # Handle different date column names
            date_col = "date" if "date" in df.columns else "time"
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            if len(df) < 200:
                return signals
            
            close = df["close"].values
            
            # Moving average signals
            ma50 = np.mean(close[-50:])
            ma200 = np.mean(close[-200:])
            current = close[-1]
            
            # Trend signal
            if current > ma50 > ma200:
                direction = SignalStrength.BULLISH
                desc = f"Price above rising MAs — uptrend confirmed"
            elif current < ma50 < ma200:
                direction = SignalStrength.BEARISH
                desc = f"Price below falling MAs — downtrend confirmed"
            elif current > ma200:
                direction = SignalStrength.BULLISH
                desc = f"Price above 200MA — long-term bullish"
            else:
                direction = SignalStrength.BEARISH
                desc = f"Price below 200MA — long-term bearish"
            
            signals.append(Signal(
                name="Trend (MA)",
                source="technical",
                value=(current - ma200) / ma200 * 100,
                direction=direction,
                confidence=65,
                timeframe=Timeframe.DAILY,
                historical_win_rate=0.60,
                last_updated=datetime.now(),
                description=desc
            ))
            
            # RSI signal
            rsi = self._calculate_rsi(close)
            if rsi < 30:
                direction = SignalStrength.BULLISH
                desc = f"RSI {rsi:.0f} — oversold, potential bounce"
            elif rsi > 70:
                direction = SignalStrength.BEARISH
                desc = f"RSI {rsi:.0f} — overbought, potential pullback"
            else:
                direction = SignalStrength.NEUTRAL
                desc = f"RSI {rsi:.0f} — neutral"
            
            signals.append(Signal(
                name="RSI",
                source="technical",
                value=rsi,
                direction=direction,
                confidence=55,
                timeframe=Timeframe.DAILY,
                historical_win_rate=0.58,
                last_updated=datetime.now(),
                description=desc
            ))
            
            # Momentum signal (rate of change)
            roc_20 = (current - close[-21]) / close[-21] * 100 if len(close) > 20 else 0
            if roc_20 > 10:
                direction = SignalStrength.BULLISH
                desc = f"Strong momentum +{roc_20:.1f}% in 20 days"
            elif roc_20 < -10:
                direction = SignalStrength.BEARISH
                desc = f"Weak momentum {roc_20:.1f}% in 20 days"
            else:
                direction = SignalStrength.NEUTRAL
                desc = f"Moderate momentum {roc_20:+.1f}% in 20 days"
            
            signals.append(Signal(
                name="Momentum (20d)",
                source="technical",
                value=roc_20,
                direction=direction,
                confidence=50,
                timeframe=Timeframe.DAILY,
                historical_win_rate=0.55,
                last_updated=datetime.now(),
                description=desc
            ))
            
        except Exception as e:
            pass
            
        return signals
    
    def _get_regime_signals(self, asset: str) -> list[Signal]:
        """Get market regime signals"""
        signals = []
        
        # Load price data using asset mapping
        file_name = self.ASSET_FILE_MAP.get(asset.upper(), f"{asset}.csv")
        price_file = self.data_dir / file_name
        
        if not price_file.exists():
            for alt in [f"{asset.upper()}.csv", f"{asset.lower()}.csv", "SP500.csv"]:
                alt_file = self.data_dir / alt
                if alt_file.exists():
                    price_file = alt_file
                    break
        
        if not price_file.exists():
            return signals
            
        try:
            df = pd.read_csv(price_file)
            # Handle different date column names
            date_col = "date" if "date" in df.columns else "time"
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            if len(df) < 60:
                return signals
            
            close = df["close"].values
            
            # Simple regime detection based on volatility and trend
            returns = np.diff(np.log(close))
            volatility = np.std(returns[-20:]) * np.sqrt(252) * 100
            trend = (close[-1] - close[-60]) / close[-60] * 100
            
            # Determine regime
            if trend > 10 and volatility < 20:
                regime = "STRONG_BULL"
                direction = SignalStrength.STRONG_BULLISH
                desc = f"Strong bull market — trend +{trend:.1f}%, low vol"
            elif trend > 5:
                regime = "BULL"
                direction = SignalStrength.BULLISH
                desc = f"Bull market — trend +{trend:.1f}%"
            elif trend < -10 and volatility > 25:
                regime = "CRISIS"
                direction = SignalStrength.STRONG_BEARISH
                desc = f"Crisis mode — trend {trend:.1f}%, high vol {volatility:.0f}%"
            elif trend < -5:
                regime = "BEAR"
                direction = SignalStrength.BEARISH
                desc = f"Bear market — trend {trend:.1f}%"
            elif volatility > 25:
                regime = "HIGH_VOL"
                direction = SignalStrength.NEUTRAL
                desc = f"High volatility regime — {volatility:.0f}% annualized"
            else:
                regime = "SIDEWAYS"
                direction = SignalStrength.NEUTRAL
                desc = f"Sideways consolidation — low conviction"
            
            signals.append(Signal(
                name="Market Regime",
                source="regime",
                value=trend,
                direction=direction,
                confidence=70,
                timeframe=Timeframe.MONTHLY,
                historical_win_rate=self.HISTORICAL_WIN_RATES.get(f"regime_{regime.lower()[:4]}", 0.60),
                last_updated=datetime.now(),
                description=desc
            ))
            
        except Exception:
            pass
            
        return signals
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_conviction(self, result: ConfluenceResult):
        """Calculate unified conviction score from all signals"""
        if not result.signals:
            result.conviction_score = 0
            result.conviction_label = "NO DATA"
            result.confidence = 0
            return
        
        # Weighted score calculation
        total_weight = 0
        weighted_sum = 0
        
        for signal in result.signals:
            weight = self.SIGNAL_WEIGHTS.get(signal.source, 1.0)
            confidence_factor = signal.confidence / 100
            
            weighted_sum += signal.direction.value * weight * confidence_factor
            total_weight += weight * confidence_factor
            
            # Count directions
            if signal.direction.value > 0:
                result.bullish_count += 1
            elif signal.direction.value < 0:
                result.bearish_count += 1
            else:
                result.neutral_count += 1
        
        # Normalize to -100 to +100
        if total_weight > 0:
            raw_score = (weighted_sum / total_weight) * 50  # Scale to -100 to +100
        else:
            raw_score = 0
        
        result.conviction_score = max(-100, min(100, raw_score))
        
        # Calculate confidence based on signal agreement
        total_signals = len(result.signals)
        max_direction = max(result.bullish_count, result.bearish_count, result.neutral_count)
        agreement = max_direction / total_signals if total_signals > 0 else 0
        
        # Boost confidence for strong confluence
        if result.bullish_count >= 5 or result.bearish_count >= 5:
            agreement = min(agreement * 1.2, 1.0)
        
        result.confidence = agreement * 100
        
        # Set conviction label
        score = result.conviction_score
        if score >= 60:
            result.conviction_label = "STRONG BUY"
        elif score >= 30:
            result.conviction_label = "BUY"
        elif score <= -60:
            result.conviction_label = "STRONG SELL"
        elif score <= -30:
            result.conviction_label = "SELL"
        else:
            result.conviction_label = "NEUTRAL"
    
    def _analyze_timeframes(self, result: ConfluenceResult):
        """Analyze multi-timeframe alignment"""
        daily_signals = [s for s in result.signals if s.timeframe == Timeframe.DAILY]
        weekly_signals = [s for s in result.signals if s.timeframe == Timeframe.WEEKLY]
        monthly_signals = [s for s in result.signals if s.timeframe == Timeframe.MONTHLY]
        
        def avg_direction(signals):
            if not signals:
                return 0
            return sum(s.direction.value for s in signals) / len(signals) * 50
        
        result.daily_bias = avg_direction(daily_signals)
        result.weekly_bias = avg_direction(weekly_signals)
        result.monthly_bias = avg_direction(monthly_signals)
        
        # Calculate alignment (0-100)
        biases = [result.daily_bias, result.weekly_bias, result.monthly_bias]
        non_zero = [b for b in biases if b != 0]
        
        if len(non_zero) >= 2:
            # Check if all same sign
            all_positive = all(b > 0 for b in non_zero)
            all_negative = all(b < 0 for b in non_zero)
            
            if all_positive or all_negative:
                # High alignment
                min_abs = min(abs(b) for b in non_zero)
                result.timeframe_alignment = min(min_abs * 2, 100)
            else:
                # Conflicting timeframes
                result.timeframe_alignment = 20
        else:
            result.timeframe_alignment = 50  # Not enough data
    
    def _generate_narrative(self, result: ConfluenceResult):
        """Generate human-readable narrative"""
        # Headline
        score = result.conviction_score
        asset = result.asset
        
        if abs(score) >= 60:
            strength = "STRONG"
        elif abs(score) >= 30:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        if score > 0:
            direction = "BULLISH"
        elif score < 0:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        result.headline = f"{strength} {direction} on {asset} — {result.bullish_count} bullish vs {result.bearish_count} bearish signals"
        
        # Key drivers (top 3 signals by confidence)
        sorted_signals = sorted(result.signals, key=lambda s: s.confidence, reverse=True)
        result.key_drivers = [s.description for s in sorted_signals[:3]]
        
        # Risks
        result.risks = []
        
        if result.timeframe_alignment < 50:
            result.risks.append("Timeframes conflicting — increased uncertainty")
        
        if result.confidence < 50:
            result.risks.append("Low signal agreement — wait for clarity")
        
        contrary_signals = [s for s in result.signals if s.direction.value * result.conviction_score < 0]
        if contrary_signals:
            result.risks.append(f"{len(contrary_signals)} contrary signal(s): {contrary_signals[0].name}")
    
    def _calculate_risk(self, result: ConfluenceResult):
        """Calculate risk metrics and suggested position size"""
        # Base risk on regime and volatility
        regime_signals = [s for s in result.signals if s.source == "regime"]
        
        if regime_signals:
            regime_value = regime_signals[0].value
            if abs(regime_value) > 10:
                result.risk_score = min(abs(regime_value) * 3, 100)
            else:
                result.risk_score = 30
        else:
            result.risk_score = 50
        
        # Position size based on conviction and risk
        base_size = 5.0  # 5% base position
        
        # Adjust for conviction
        conviction_mult = 1 + (abs(result.conviction_score) / 100) * 0.5
        
        # Adjust for confidence
        confidence_mult = result.confidence / 100
        
        # Adjust for risk
        risk_mult = max(0.3, 1 - result.risk_score / 150)
        
        result.suggested_position_size = base_size * conviction_mult * confidence_mult * risk_mult


def analyze_all_assets():
    """Run confluence analysis on all available assets"""
    engine = SignalConfluenceEngine()
    
    assets = ["SP500", "GOLD", "CRUDE", "BITCOIN", "NASDAQ"]
    results = {}
    
    for asset in assets:
        try:
            result = engine.analyze(asset)
            results[asset] = result.to_dict()
            print(f"\n{'='*60}")
            print(f"[ANALYSIS] {asset} CONFLUENCE ANALYSIS")
            print(f"{'='*60}")
            print(f"Conviction: {result.conviction_score:+.0f} ({result.conviction_label})")
            print(f"Confidence: {result.confidence:.0f}%")
            print(f"Signals: {result.bullish_count} bullish, {result.bearish_count} bearish, {result.neutral_count} neutral")
            print(f"\nHeadline: {result.headline}")
            print(f"\nKey Drivers:")
            for driver in result.key_drivers:
                print(f"  - {driver}")
            if result.risks:
                print(f"\nRisks:")
                for risk in result.risks:
                    print(f"  [!] {risk}")
            print(f"\nSuggested Position: {result.suggested_position_size:.1f}%")
        except Exception as e:
            print(f"Error analyzing {asset}: {e}")
    
    return results


if __name__ == "__main__":
    results = analyze_all_assets()
    
    # Save results
    output_file = Path("data/confluence_analysis.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {output_file}")
