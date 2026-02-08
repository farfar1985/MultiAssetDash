"""
Historical Analog Finder — Pattern Match Current Setup to History

Finds similar market conditions in history and shows what happened next.
This is what traders pay big money for — "the current setup looks like X,
and here's what happened in the next 30 days."

Author: AmiraB
Created: 2026-02-07
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class HistoricalAnalog:
    """A historical period that matches current conditions"""
    start_date: str
    end_date: str
    similarity_score: float  # 0-100%
    
    # Market conditions at the time
    price_at_match: float
    volatility: float
    trend_20d: float
    rsi: float
    
    # What happened AFTER
    return_7d: float
    return_14d: float
    return_30d: float
    return_60d: float
    max_drawdown_30d: float
    max_gain_30d: float
    
    # Context
    event_context: str  # What was happening (if notable)
    regime: str
    
    def to_dict(self):
        return {
            "period": {
                "start": self.start_date,
                "end": self.end_date
            },
            "similarity": round(self.similarity_score, 1),
            "conditions_at_match": {
                "price": round(self.price_at_match, 2),
                "volatility": round(self.volatility, 1),
                "trend_20d": round(self.trend_20d, 2),
                "rsi": round(self.rsi, 1)
            },
            "forward_returns": {
                "7d": round(self.return_7d, 2),
                "14d": round(self.return_14d, 2),
                "30d": round(self.return_30d, 2),
                "60d": round(self.return_60d, 2)
            },
            "risk_profile": {
                "max_drawdown_30d": round(self.max_drawdown_30d, 2),
                "max_gain_30d": round(self.max_gain_30d, 2)
            },
            "context": {
                "event": self.event_context,
                "regime": self.regime
            }
        }


@dataclass
class AnalogAnalysis:
    """Complete analog analysis result"""
    asset: str
    analysis_date: str
    
    # Current conditions (what we're matching)
    current_price: float
    current_volatility: float
    current_trend_20d: float
    current_rsi: float
    current_regime: str
    
    # Top analogs
    analogs: list[HistoricalAnalog] = field(default_factory=list)
    
    # Aggregate statistics from analogs
    avg_return_7d: float = 0.0
    avg_return_14d: float = 0.0
    avg_return_30d: float = 0.0
    avg_return_60d: float = 0.0
    win_rate_30d: float = 0.0  # % of analogs with positive 30d return
    
    median_return_30d: float = 0.0
    best_case_30d: float = 0.0
    worst_case_30d: float = 0.0
    
    # Forecast
    forecast_direction: str = ""  # "BULLISH", "BEARISH", "NEUTRAL"
    forecast_confidence: float = 0.0
    forecast_narrative: str = ""
    
    def to_dict(self):
        return {
            "asset": self.asset,
            "analysis_date": self.analysis_date,
            "current_conditions": {
                "price": round(self.current_price, 2),
                "volatility": round(self.current_volatility, 1),
                "trend_20d": round(self.current_trend_20d, 2),
                "rsi": round(self.current_rsi, 1),
                "regime": self.current_regime
            },
            "analogs": [a.to_dict() for a in self.analogs],
            "statistics": {
                "count": len(self.analogs),
                "avg_return_7d": round(self.avg_return_7d, 2),
                "avg_return_14d": round(self.avg_return_14d, 2),
                "avg_return_30d": round(self.avg_return_30d, 2),
                "avg_return_60d": round(self.avg_return_60d, 2),
                "win_rate_30d": round(self.win_rate_30d, 1),
                "median_return_30d": round(self.median_return_30d, 2),
                "best_case_30d": round(self.best_case_30d, 2),
                "worst_case_30d": round(self.worst_case_30d, 2)
            },
            "forecast": {
                "direction": self.forecast_direction,
                "confidence": round(self.forecast_confidence, 1),
                "narrative": self.forecast_narrative
            }
        }


class HistoricalAnalogFinder:
    """
    Finds similar historical periods and projects forward returns.
    Uses multiple features for matching:
    - Price momentum (trend)
    - Volatility regime
    - RSI levels
    - Market regime
    """
    
    # Known historical events for context
    HISTORICAL_EVENTS = {
        "2020-03": "COVID-19 Crash",
        "2020-04": "COVID-19 Recovery",
        "2022-01": "Fed Pivot / Rate Hikes Begin",
        "2022-06": "Crypto Winter",
        "2022-10": "UK Gilt Crisis",
        "2023-03": "SVB Bank Crisis",
        "2023-10": "Israel-Gaza Conflict",
        "2024-08": "Yen Carry Trade Unwind",
        "2025-01": "Trump 2.0 Inauguration",
    }
    
    def __init__(self, data_dir: str = "data/qdl_history"):
        self.data_dir = Path(data_dir)
        
    def find_analogs(
        self, 
        asset: str, 
        lookback_years: int = 5,
        n_analogs: int = 10,
        min_similarity: float = 70.0
    ) -> AnalogAnalysis:
        """
        Find historical periods similar to current conditions.
        
        Args:
            asset: Asset to analyze
            lookback_years: How many years of history to search
            n_analogs: Maximum number of analogs to return
            min_similarity: Minimum similarity score (0-100)
        """
        # Load price data
        df = self._load_data(asset)
        if df is None or len(df) < 252:  # Need at least 1 year
            return self._empty_analysis(asset)
        
        # Calculate features for all periods
        df = self._calculate_features(df)
        
        # Get current conditions
        current = df.iloc[-1]
        
        # Create analysis result
        analysis = AnalogAnalysis(
            asset=asset,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            current_price=current["close"],
            current_volatility=current["volatility"],
            current_trend_20d=current["trend_20d"],
            current_rsi=current["rsi"],
            current_regime=self._classify_regime(current)
        )
        
        # Find similar periods (exclude last 60 days to avoid lookahead)
        lookback_cutoff = len(df) - 60
        min_date = df.iloc[-lookback_years * 252]["date"] if len(df) > lookback_years * 252 else df.iloc[0]["date"]
        
        analogs = []
        
        for i in range(100, lookback_cutoff):  # Start from day 100 to have enough history
            row = df.iloc[i]
            
            if row["date"] < min_date:
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(current, row)
            
            if similarity >= min_similarity:
                # Calculate forward returns
                analog = self._create_analog(df, i, row, similarity)
                if analog:
                    analogs.append(analog)
        
        # Sort by similarity and take top N
        analogs.sort(key=lambda x: x.similarity_score, reverse=True)
        analysis.analogs = analogs[:n_analogs]
        
        # Calculate aggregate statistics
        self._calculate_statistics(analysis)
        
        # Generate forecast
        self._generate_forecast(analysis)
        
        return analysis
    
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
    
    def _load_data(self, asset: str) -> Optional[pd.DataFrame]:
        """Load price data for asset"""
        # Try mapped name first
        mapped_name = self.ASSET_FILE_MAP.get(asset.upper())
        patterns = [mapped_name] if mapped_name else []
        patterns.extend([
            f"{asset}.csv",
            f"{asset.upper()}.csv",
            f"{asset.lower()}.csv",
        ])
        
        for pattern in patterns:
            if pattern is None:
                continue
            file_path = self.data_dir / pattern
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Handle different date column names
                date_col = "date" if "date" in df.columns else "time"
                df["date"] = pd.to_datetime(df[date_col])
                df = df.sort_values("date").reset_index(drop=True)
                return df
        
        return None
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for similarity matching"""
        close = df["close"].values
        
        # Volatility (20-day rolling)
        returns = np.diff(np.log(close))
        df["volatility"] = np.nan
        for i in range(20, len(df)):
            df.loc[df.index[i], "volatility"] = np.std(returns[i-20:i]) * np.sqrt(252) * 100
        
        # Trend (20-day return)
        df["trend_20d"] = df["close"].pct_change(20) * 100
        
        # Trend (60-day return for regime)
        df["trend_60d"] = df["close"].pct_change(60) * 100
        
        # RSI
        df["rsi"] = self._calculate_rsi_series(close)
        
        # Z-score of price relative to 200-day MA
        df["ma200"] = df["close"].rolling(200).mean()
        df["price_zscore"] = (df["close"] - df["ma200"]) / df["close"].rolling(200).std()
        
        return df.dropna()
    
    def _calculate_rsi_series(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI for entire series"""
        rsi = np.full(len(prices), np.nan)
        
        for i in range(period + 1, len(prices)):
            deltas = np.diff(prices[i-period:i+1])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _classify_regime(self, row) -> str:
        """Classify market regime"""
        trend = row.get("trend_60d", row.get("trend_20d", 0))
        vol = row.get("volatility", 15)
        
        if trend > 15 and vol < 20:
            return "STRONG_BULL"
        elif trend > 5:
            return "BULL"
        elif trend < -15 and vol > 25:
            return "CRISIS"
        elif trend < -5:
            return "BEAR"
        elif vol > 25:
            return "HIGH_VOL"
        else:
            return "SIDEWAYS"
    
    def _calculate_similarity(self, current, historical) -> float:
        """
        Calculate similarity score between current and historical conditions.
        Returns 0-100 score.
        """
        scores = []
        
        # Volatility similarity (weight: 30%)
        vol_diff = abs(current["volatility"] - historical["volatility"])
        vol_score = max(0, 100 - vol_diff * 3)  # 3 points per 1% vol difference
        scores.append(vol_score * 0.30)
        
        # Trend similarity (weight: 30%)
        trend_diff = abs(current["trend_20d"] - historical["trend_20d"])
        trend_score = max(0, 100 - trend_diff * 4)  # 4 points per 1% trend difference
        scores.append(trend_score * 0.30)
        
        # RSI similarity (weight: 20%)
        rsi_diff = abs(current["rsi"] - historical["rsi"])
        rsi_score = max(0, 100 - rsi_diff * 2)  # 2 points per 1 RSI point difference
        scores.append(rsi_score * 0.20)
        
        # Price z-score similarity (weight: 20%)
        zscore_diff = abs(current["price_zscore"] - historical["price_zscore"])
        zscore_score = max(0, 100 - zscore_diff * 25)  # 25 points per 1 z-score difference
        scores.append(zscore_score * 0.20)
        
        return sum(scores)
    
    def _create_analog(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        row, 
        similarity: float
    ) -> Optional[HistoricalAnalog]:
        """Create a HistoricalAnalog from a matched period"""
        try:
            # Get forward returns
            forward_60 = min(idx + 60, len(df) - 1)
            
            prices_forward = df.iloc[idx:forward_60 + 1]["close"].values
            if len(prices_forward) < 30:
                return None
            
            base_price = prices_forward[0]
            
            # Calculate returns
            return_7d = (prices_forward[min(7, len(prices_forward)-1)] / base_price - 1) * 100
            return_14d = (prices_forward[min(14, len(prices_forward)-1)] / base_price - 1) * 100
            return_30d = (prices_forward[min(30, len(prices_forward)-1)] / base_price - 1) * 100
            return_60d = (prices_forward[-1] / base_price - 1) * 100 if len(prices_forward) >= 60 else return_30d
            
            # Max drawdown and gain in 30 days
            prices_30d = prices_forward[:min(31, len(prices_forward))]
            running_max = np.maximum.accumulate(prices_30d)
            drawdowns = (prices_30d - running_max) / running_max * 100
            max_drawdown = drawdowns.min()
            
            running_min = np.minimum.accumulate(prices_30d)
            gains = (prices_30d - running_min) / running_min * 100
            max_gain = gains.max()
            
            # Event context
            date_str = row["date"].strftime("%Y-%m")
            event = self.HISTORICAL_EVENTS.get(date_str, "")
            
            return HistoricalAnalog(
                start_date=row["date"].strftime("%Y-%m-%d"),
                end_date=df.iloc[min(idx + 30, len(df) - 1)]["date"].strftime("%Y-%m-%d"),
                similarity_score=similarity,
                price_at_match=row["close"],
                volatility=row["volatility"],
                trend_20d=row["trend_20d"],
                rsi=row["rsi"],
                return_7d=return_7d,
                return_14d=return_14d,
                return_30d=return_30d,
                return_60d=return_60d,
                max_drawdown_30d=max_drawdown,
                max_gain_30d=max_gain,
                event_context=event,
                regime=self._classify_regime(row)
            )
            
        except Exception as e:
            return None
    
    def _calculate_statistics(self, analysis: AnalogAnalysis):
        """Calculate aggregate statistics from analogs"""
        if not analysis.analogs:
            return
        
        returns_7d = [a.return_7d for a in analysis.analogs]
        returns_14d = [a.return_14d for a in analysis.analogs]
        returns_30d = [a.return_30d for a in analysis.analogs]
        returns_60d = [a.return_60d for a in analysis.analogs]
        
        analysis.avg_return_7d = np.mean(returns_7d)
        analysis.avg_return_14d = np.mean(returns_14d)
        analysis.avg_return_30d = np.mean(returns_30d)
        analysis.avg_return_60d = np.mean(returns_60d)
        
        analysis.median_return_30d = np.median(returns_30d)
        analysis.best_case_30d = np.max(returns_30d)
        analysis.worst_case_30d = np.min(returns_30d)
        
        # Win rate (positive return)
        wins = sum(1 for r in returns_30d if r > 0)
        analysis.win_rate_30d = (wins / len(returns_30d)) * 100
    
    def _generate_forecast(self, analysis: AnalogAnalysis):
        """Generate forecast based on analog analysis"""
        if not analysis.analogs:
            analysis.forecast_direction = "NEUTRAL"
            analysis.forecast_confidence = 0
            analysis.forecast_narrative = "Insufficient historical data for analog analysis."
            return
        
        # Direction based on average and median
        avg = analysis.avg_return_30d
        median = analysis.median_return_30d
        win_rate = analysis.win_rate_30d
        
        # Combined score
        if avg > 3 and median > 2 and win_rate > 60:
            analysis.forecast_direction = "BULLISH"
            analysis.forecast_confidence = min(95, win_rate + abs(avg) * 2)
        elif avg < -3 and median < -2 and win_rate < 40:
            analysis.forecast_direction = "BEARISH"
            analysis.forecast_confidence = min(95, (100 - win_rate) + abs(avg) * 2)
        else:
            analysis.forecast_direction = "NEUTRAL"
            analysis.forecast_confidence = 50
        
        # Narrative
        n = len(analysis.analogs)
        best = analysis.best_case_30d
        worst = analysis.worst_case_30d
        
        analysis.forecast_narrative = (
            f"Based on {n} similar historical setups, the next 30 days have shown "
            f"an average return of {avg:+.1f}% with a {win_rate:.0f}% win rate. "
            f"Range: {worst:+.1f}% to {best:+.1f}%. "
        )
        
        if analysis.forecast_direction == "BULLISH":
            analysis.forecast_narrative += "Historical bias is bullish."
        elif analysis.forecast_direction == "BEARISH":
            analysis.forecast_narrative += "Historical bias is bearish."
        else:
            analysis.forecast_narrative += "Mixed signals suggest caution."
        
        # Add event context if notable
        notable_events = [a.event_context for a in analysis.analogs if a.event_context]
        if notable_events:
            events = list(set(notable_events))[:3]
            analysis.forecast_narrative += f" Notable analogs include: {', '.join(events)}."
    
    def _empty_analysis(self, asset: str) -> AnalogAnalysis:
        """Return empty analysis when data is insufficient"""
        return AnalogAnalysis(
            asset=asset,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            current_price=0,
            current_volatility=0,
            current_trend_20d=0,
            current_rsi=50,
            current_regime="UNKNOWN"
        )


def analyze_all_assets():
    """Run analog analysis on all available assets"""
    finder = HistoricalAnalogFinder()
    
    assets = ["SP500", "GOLD", "CRUDE", "BITCOIN", "NASDAQ"]
    results = {}
    
    for asset in assets:
        try:
            analysis = finder.find_analogs(asset)
            results[asset] = analysis.to_dict()
            
            print(f"\n{'='*70}")
            print(f"[ANALYSIS] {asset} HISTORICAL ANALOG ANALYSIS")
            print(f"{'='*70}")
            print(f"\nCurrent Conditions:")
            print(f"  Price: ${analysis.current_price:,.2f}")
            print(f"  Volatility: {analysis.current_volatility:.1f}%")
            print(f"  20-day Trend: {analysis.current_trend_20d:+.1f}%")
            print(f"  RSI: {analysis.current_rsi:.0f}")
            print(f"  Regime: {analysis.current_regime}")
            
            print(f"\n[FOUND] {len(analysis.analogs)} Similar Historical Periods")
            
            if analysis.analogs:
                print(f"\nTop 3 Analogs:")
                for i, analog in enumerate(analysis.analogs[:3], 1):
                    print(f"\n  {i}. {analog.start_date} ({analog.similarity_score:.0f}% similar)")
                    print(f"     Conditions: Vol {analog.volatility:.0f}%, Trend {analog.trend_20d:+.1f}%, RSI {analog.rsi:.0f}")
                    print(f"     Next 30d: {analog.return_30d:+.1f}% (DD: {analog.max_drawdown_30d:.1f}%, Max: +{analog.max_gain_30d:.1f}%)")
                    if analog.event_context:
                        print(f"     Context: {analog.event_context}")
                
                print(f"\n[STATS] Aggregate Statistics (from {len(analysis.analogs)} analogs):")
                print(f"  Avg 7-day:  {analysis.avg_return_7d:+.2f}%")
                print(f"  Avg 14-day: {analysis.avg_return_14d:+.2f}%")
                print(f"  Avg 30-day: {analysis.avg_return_30d:+.2f}%")
                print(f"  Avg 60-day: {analysis.avg_return_60d:+.2f}%")
                print(f"  30-day Win Rate: {analysis.win_rate_30d:.0f}%")
                print(f"  Range: {analysis.worst_case_30d:+.1f}% to {analysis.best_case_30d:+.1f}%")
                
                print(f"\n[FORECAST] {analysis.forecast_direction} ({analysis.forecast_confidence:.0f}% confidence)")
                print(f"   {analysis.forecast_narrative}")
            
        except Exception as e:
            print(f"Error analyzing {asset}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


if __name__ == "__main__":
    results = analyze_all_assets()
    
    # Save results
    output_file = Path("data/analog_analysis.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {output_file}")
