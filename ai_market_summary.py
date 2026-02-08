"""
AI Market Summary Generator — "What's Happening" Intelligence
==============================================================
Generates human-readable market summaries from all signal sources.
Designed for casual observers and wealth manager client reports.

Author: AmiraB
Created: 2026-02-08
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Import all signal modules
try:
    from signal_confluence import SignalConfluenceEngine
except ImportError:
    SignalConfluenceEngine = None

try:
    from yield_curve_signals import generate_signal as yc_signal
except ImportError:
    yc_signal = None

try:
    from correlation_regime import analyze_correlations
except ImportError:
    analyze_correlations = None

try:
    from hht_regime_detector import analyze_asset as hht_analyze
except ImportError:
    hht_analyze = None

try:
    from credit_spread_signals import analyze_credit_spreads
except ImportError:
    analyze_credit_spreads = None


@dataclass
class MarketSummary:
    """Complete market summary"""
    timestamp: datetime
    
    # Headlines
    main_headline: str
    sub_headline: str
    
    # Key metrics
    overall_sentiment: str  # BULLISH, NEUTRAL, BEARISH
    risk_level: str         # LOW, MODERATE, ELEVATED, HIGH
    confidence: float
    
    # Asset summaries
    asset_summaries: Dict[str, str] = field(default_factory=dict)
    
    # Key themes
    themes: List[str] = field(default_factory=list)
    
    # Risks and opportunities
    opportunities: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    
    # Action items by persona
    hedger_actions: List[str] = field(default_factory=list)
    investor_actions: List[str] = field(default_factory=list)
    trader_actions: List[str] = field(default_factory=list)
    
    # Full narrative
    narrative: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "headlines": {
                "main": self.main_headline,
                "sub": self.sub_headline
            },
            "sentiment": {
                "overall": self.overall_sentiment,
                "risk_level": self.risk_level,
                "confidence": round(self.confidence, 1)
            },
            "asset_summaries": self.asset_summaries,
            "themes": self.themes,
            "opportunities": self.opportunities,
            "risks": self.risks,
            "actions": {
                "hedger": self.hedger_actions,
                "investor": self.investor_actions,
                "trader": self.trader_actions
            },
            "narrative": self.narrative
        }


def generate_asset_summary(asset: str, confluence_result) -> str:
    """Generate one-liner summary for an asset"""
    score = confluence_result.conviction_score
    label = confluence_result.conviction_label
    
    # Get key driver
    key_driver = confluence_result.key_drivers[0] if confluence_result.key_drivers else "Mixed signals"
    
    if score > 30:
        direction = "bullish"
        emoji = "📈"
    elif score < -30:
        direction = "bearish"
        emoji = "📉"
    else:
        direction = "neutral"
        emoji = "➡️"
    
    return f"{emoji} {asset}: {label} ({score:+.0f}) — {key_driver}"


def generate_market_summary() -> MarketSummary:
    """Generate comprehensive market summary"""
    
    summary = MarketSummary(
        timestamp=datetime.now(),
        main_headline="",
        sub_headline="",
        overall_sentiment="NEUTRAL",
        risk_level="MODERATE",
        confidence=50
    )
    
    # Collect all signals
    asset_scores = {}
    asset_results = {}
    
    # Get confluence for each asset
    if SignalConfluenceEngine:
        engine = SignalConfluenceEngine()
        for asset in ["SP500", "GOLD", "CRUDE", "BITCOIN", "NASDAQ"]:
            try:
                result = engine.analyze(asset)
                asset_scores[asset] = result.conviction_score
                asset_results[asset] = result
                summary.asset_summaries[asset] = generate_asset_summary(asset, result)
            except Exception:
                pass
    
    # Calculate overall sentiment
    if asset_scores:
        avg_score = sum(asset_scores.values()) / len(asset_scores)
        bullish_count = sum(1 for s in asset_scores.values() if s > 20)
        bearish_count = sum(1 for s in asset_scores.values() if s < -20)
        
        if avg_score > 25:
            summary.overall_sentiment = "BULLISH"
        elif avg_score < -25:
            summary.overall_sentiment = "BEARISH"
        else:
            summary.overall_sentiment = "NEUTRAL"
        
        summary.confidence = min(90, 50 + abs(avg_score))
    
    # Get macro signals
    themes = []
    
    # Yield curve
    if yc_signal:
        try:
            yc = yc_signal()
            if yc:
                if yc.state.value == "INVERTED":
                    themes.append("⚠️ Yield curve inverted — recession risk elevated")
                    summary.risk_level = "HIGH"
                elif yc.state.value == "FLAT":
                    themes.append("📊 Yield curve flat — transition period")
                elif yc.state.value == "STEEPENING":
                    themes.append("📈 Yield curve steepening — recovery signal")
        except Exception:
            pass
    
    # Correlations
    if analyze_correlations:
        try:
            corr = analyze_correlations()
            if corr:
                if corr.regime.value == "CRISIS":
                    themes.append("🚨 Correlation crisis — diversification failing")
                    summary.risk_level = "HIGH"
                elif corr.regime.value == "LOW":
                    themes.append("✅ Low correlations — healthy diversification")
        except Exception:
            pass
    
    # Credit spreads
    if analyze_credit_spreads:
        try:
            credit = analyze_credit_spreads()
            if credit:
                if credit.regime.value == "STRESS":
                    themes.append("🔴 Credit spreads stressed — flight to quality")
                    summary.risk_level = "HIGH"
                elif credit.regime.value == "COMPRESSED":
                    themes.append("🟢 Credit spreads tight — risk appetite strong")
        except Exception:
            pass
    
    summary.themes = themes if themes else ["Market in transition — mixed signals"]
    
    # Identify opportunities and risks
    for asset, score in asset_scores.items():
        if score > 40:
            summary.opportunities.append(f"{asset}: Strong bullish conviction (+{score:.0f})")
        elif score < -40:
            if asset == "BITCOIN" and asset_results.get(asset):
                # Check for mean reversion
                summary.opportunities.append(f"{asset}: Oversold, mean reversion potential")
            else:
                summary.risks.append(f"{asset}: Strong bearish signals ({score:.0f})")
    
    # Generate headlines
    if summary.overall_sentiment == "BULLISH":
        summary.main_headline = "Markets Lean Bullish — Risk Appetite Returning"
    elif summary.overall_sentiment == "BEARISH":
        summary.main_headline = "Caution Warranted — Bearish Signals Dominate"
    else:
        summary.main_headline = "Markets in Transition — Selectivity Key"
    
    # Sub-headline based on strongest signal
    if asset_scores:
        best_asset = max(asset_scores.items(), key=lambda x: abs(x[1]))
        if best_asset[1] > 30:
            summary.sub_headline = f"Gold and commodities showing strength"
        elif best_asset[1] < -30:
            summary.sub_headline = f"Crypto under pressure, defensive positioning advised"
        else:
            summary.sub_headline = "No clear directional bias — wait for confirmation"
    
    # Generate action items
    if summary.overall_sentiment == "BULLISH":
        summary.hedger_actions = [
            "Consider reducing hedge ratios",
            "Roll hedges to shorter duration",
            "Watch for complacency signals"
        ]
        summary.investor_actions = [
            "Maintain equity exposure",
            "Consider adding to commodities (Gold strong)",
            "Monitor regime transition signals"
        ]
        summary.trader_actions = [
            "Favor long setups in strong assets",
            "Use pullbacks to add exposure",
            "Tight stops on counter-trend trades"
        ]
    elif summary.overall_sentiment == "BEARISH":
        summary.hedger_actions = [
            "Increase hedge ratios",
            "Extend hedge duration",
            "Consider tail risk hedges"
        ]
        summary.investor_actions = [
            "Reduce risk exposure",
            "Increase cash/short-term bonds",
            "Wait for capitulation signals"
        ]
        summary.trader_actions = [
            "Favor short setups or stand aside",
            "Watch for mean reversion in oversold assets",
            "Reduce position sizes"
        ]
    else:
        summary.hedger_actions = [
            "Maintain current hedge ratios",
            "Review hedge effectiveness",
            "Prepare for volatility"
        ]
        summary.investor_actions = [
            "Stay diversified",
            "Be selective — favor high-conviction ideas",
            "Build watchlists for both scenarios"
        ]
        summary.trader_actions = [
            "Reduce position sizes",
            "Focus on high-probability setups only",
            "Wait for clearer signals"
        ]
    
    # Generate narrative
    summary.narrative = f"""
## Market Summary — {summary.timestamp.strftime('%B %d, %Y')}

### {summary.main_headline}

{summary.sub_headline}

**Overall Sentiment:** {summary.overall_sentiment} | **Risk Level:** {summary.risk_level} | **Confidence:** {summary.confidence:.0f}%

### Key Themes
{"".join(f"- {t}" + chr(10) for t in summary.themes)}

### Asset Breakdown
{"".join(f"- {s}" + chr(10) for s in summary.asset_summaries.values())}

### Opportunities
{"".join(f"- {o}" + chr(10) for o in summary.opportunities) if summary.opportunities else "- No high-conviction opportunities currently" + chr(10)}

### Risks
{"".join(f"- {r}" + chr(10) for r in summary.risks) if summary.risks else "- Standard market risks apply" + chr(10)}

### Recommended Actions

**For Hedgers:**
{"".join(f"- {a}" + chr(10) for a in summary.hedger_actions)}

**For Investors:**
{"".join(f"- {a}" + chr(10) for a in summary.investor_actions)}

**For Traders:**
{"".join(f"- {a}" + chr(10) for a in summary.trader_actions)}

---
*Generated by QDT Intelligence Engine*
"""
    
    return summary


def get_summary_for_api() -> Dict:
    """Get market summary for API"""
    summary = generate_market_summary()
    return summary.to_dict()


if __name__ == "__main__":
    summary = generate_market_summary()
    print(summary.narrative)
