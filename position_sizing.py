"""
Intelligent Position Sizing Engine
===================================
Combines regime detection, COT positioning, and signal confidence
to compute optimal position sizes for each persona type.

Key Principles:
1. Never risk more than drawdown budget allows
2. Scale up in favorable regimes, down in unfavorable
3. Higher conviction = larger position (but capped)
4. Kelly criterion for edge sizing
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum
import numpy as np

log = logging.getLogger(__name__)


class RiskTolerance(Enum):
    CONSERVATIVE = "conservative"  # Max 2% drawdown per trade
    MODERATE = "moderate"          # Max 5% drawdown per trade
    AGGRESSIVE = "aggressive"      # Max 10% drawdown per trade


class Persona(Enum):
    HEDGING = "hedging"
    INSTITUTIONAL = "institutional"
    WEALTH_MANAGER = "wealth_manager"
    HEDGE_FUND = "hedge_fund"
    RETAIL = "retail"
    CASUAL = "casual"


@dataclass
class PositionRecommendation:
    """Complete position sizing recommendation."""
    asset: str
    direction: str  # 'long' or 'short'
    
    # Position sizing
    base_size_pct: float        # Base % of portfolio
    regime_multiplier: float    # 0.3 to 1.5
    confidence_multiplier: float # 0.5 to 1.3
    cot_adjustment: float       # -0.2 to +0.2
    final_size_pct: float       # Actual recommended size
    
    # Risk management
    stop_loss_pct: float        # Distance to stop
    profit_target_pct: float    # Distance to target
    risk_reward_ratio: float    # Target / Stop
    max_loss_dollars: float     # Max $ loss at position size
    
    # Timing
    urgency: str                # 'immediate', 'today', 'this_week', 'monitor'
    entry_strategy: str         # 'market', 'limit', 'scale_in'
    
    # Reasoning
    reasoning: list[str]
    warnings: list[str]


# Regime multipliers (how much to scale position by regime)
REGIME_MULTIPLIERS = {
    'STRONG_BULL': 1.3,
    'BULL': 1.1,
    'WEAK_BULL': 0.9,
    'SIDEWAYS': 0.7,
    'WEAK_BEAR': 0.6,
    'BEAR': 0.5,
    'STRONG_BEAR': 0.4,
    'CRISIS': 0.2,
    'RECOVERY': 0.8,
}

# Persona risk profiles
PERSONA_PROFILES = {
    Persona.HEDGING: {
        'max_position': 0.20,      # 20% max
        'base_position': 0.10,     # 10% base
        'risk_tolerance': RiskTolerance.CONSERVATIVE,
        'prefer_regime_neutral': True,  # Hedge regardless of regime
        'stop_loss_default': 0.03,
    },
    Persona.INSTITUTIONAL: {
        'max_position': 0.15,
        'base_position': 0.08,
        'risk_tolerance': RiskTolerance.MODERATE,
        'prefer_regime_neutral': False,
        'stop_loss_default': 0.05,
    },
    Persona.WEALTH_MANAGER: {
        'max_position': 0.10,
        'base_position': 0.05,
        'risk_tolerance': RiskTolerance.CONSERVATIVE,
        'prefer_regime_neutral': False,
        'stop_loss_default': 0.04,
    },
    Persona.HEDGE_FUND: {
        'max_position': 0.25,
        'base_position': 0.12,
        'risk_tolerance': RiskTolerance.AGGRESSIVE,
        'prefer_regime_neutral': False,
        'stop_loss_default': 0.07,
    },
    Persona.RETAIL: {
        'max_position': 0.10,
        'base_position': 0.05,
        'risk_tolerance': RiskTolerance.MODERATE,
        'prefer_regime_neutral': False,
        'stop_loss_default': 0.05,
    },
    Persona.CASUAL: {
        'max_position': 0.05,
        'base_position': 0.02,
        'risk_tolerance': RiskTolerance.CONSERVATIVE,
        'prefer_regime_neutral': True,
        'stop_loss_default': 0.03,
    },
}


def compute_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Kelly Criterion for optimal bet sizing.
    Returns fraction of bankroll to risk.
    
    Formula: f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = win/loss ratio
    """
    if avg_loss == 0:
        return 0.0
    
    p = win_rate
    q = 1 - win_rate
    b = avg_win / avg_loss
    
    kelly = (p * b - q) / b
    
    # Cap at 25% (half-Kelly is common in practice)
    return max(0, min(0.25, kelly))


def get_regime_context() -> Tuple[str, float, float]:
    """Get current regime from regime detector."""
    try:
        from quantum_regime_enhanced import get_current_regime
        result = get_current_regime()
        
        if result.get('success'):
            regime = result['regime']['name']
            confidence = result['regime']['confidence']
            vix = result['regime']['vix_level']
            return regime, confidence, vix
    except Exception as e:
        log.warning(f"Could not get regime: {e}")
    
    return 'SIDEWAYS', 50.0, 20.0  # Default


def get_cot_context(asset: str) -> Tuple[str, float]:
    """Get COT signal for asset."""
    try:
        from smart_money_signals import generate_cot_signal
        signal = generate_cot_signal(asset.upper())
        
        if signal:
            return signal.signal.value, signal.z_score
    except Exception as e:
        log.warning(f"Could not get COT for {asset}: {e}")
    
    return 'NEUTRAL', 0.0


def compute_position_size(
    asset: str,
    direction: str,  # 'long' or 'short'
    signal_confidence: float,  # 0-100
    win_rate: float,  # 0-100
    expected_move_pct: float,  # Expected % move
    portfolio_value: float,
    persona: Persona,
    override_regime: str = None,
) -> PositionRecommendation:
    """
    Compute optimal position size with all factors considered.
    
    Inputs:
    - asset: Asset name (crude-oil, gold, bitcoin, etc.)
    - direction: long or short
    - signal_confidence: Model confidence (0-100)
    - win_rate: Historical accuracy (0-100)
    - expected_move_pct: Predicted move magnitude (e.g., 2.5 for 2.5%)
    - portfolio_value: Total portfolio $ value
    - persona: User persona type
    - override_regime: Force a specific regime (for testing)
    
    Returns:
    - PositionRecommendation with complete sizing guidance
    """
    profile = PERSONA_PROFILES[persona]
    reasoning = []
    warnings = []
    
    # Get context
    regime, regime_confidence, vix = get_regime_context()
    if override_regime:
        regime = override_regime
    
    cot_signal, cot_z = get_cot_context(asset)
    
    # =========================================================================
    # BASE POSITION SIZE (from confidence)
    # =========================================================================
    
    # Map confidence to position size (50% conf -> 0.5x base, 100% -> 1.0x base)
    confidence_factor = max(0.5, min(1.3, signal_confidence / 75))
    base_size = profile['base_position'] * confidence_factor
    
    reasoning.append(f"Base size {base_size:.1%} (confidence: {signal_confidence:.0f}%)")
    
    # =========================================================================
    # REGIME ADJUSTMENT
    # =========================================================================
    
    if profile['prefer_regime_neutral']:
        regime_mult = 1.0
        reasoning.append("Regime-neutral positioning (hedging persona)")
    else:
        regime_mult = REGIME_MULTIPLIERS.get(regime, 1.0)
        
        # Adjust for direction vs regime
        is_bullish_regime = regime in ['STRONG_BULL', 'BULL', 'WEAK_BULL', 'RECOVERY']
        is_bearish_regime = regime in ['STRONG_BEAR', 'BEAR', 'WEAK_BEAR', 'CRISIS']
        
        if direction == 'long' and is_bearish_regime:
            regime_mult *= 0.5
            warnings.append(f"Long position in {regime} regime - reduced size")
        elif direction == 'short' and is_bullish_regime:
            regime_mult *= 0.5
            warnings.append(f"Short position in {regime} regime - reduced size")
        
        reasoning.append(f"Regime: {regime} (mult: {regime_mult:.2f})")
    
    # =========================================================================
    # VIX ADJUSTMENT
    # =========================================================================
    
    if vix > 30:
        vix_mult = 0.7
        warnings.append(f"High VIX ({vix:.1f}) - reducing size by 30%")
    elif vix > 25:
        vix_mult = 0.85
        reasoning.append(f"Elevated VIX ({vix:.1f}) - slightly reduced size")
    elif vix < 15:
        vix_mult = 1.1
        reasoning.append(f"Low VIX ({vix:.1f}) - favorable conditions")
    else:
        vix_mult = 1.0
    
    # =========================================================================
    # COT ADJUSTMENT
    # =========================================================================
    
    cot_adjustment = 0.0
    
    if cot_signal in ['STRONG_BUY', 'BUY'] and direction == 'long':
        cot_adjustment = 0.15
        reasoning.append(f"COT supports long (specs short, z={cot_z:.1f})")
    elif cot_signal in ['STRONG_SELL', 'SELL'] and direction == 'short':
        cot_adjustment = 0.15
        reasoning.append(f"COT supports short (specs long, z={cot_z:.1f})")
    elif cot_signal in ['STRONG_BUY', 'BUY'] and direction == 'short':
        cot_adjustment = -0.15
        warnings.append(f"COT conflicts with short - specs already short")
    elif cot_signal in ['STRONG_SELL', 'SELL'] and direction == 'long':
        cot_adjustment = -0.15
        warnings.append(f"COT conflicts with long - specs already long")
    
    # =========================================================================
    # KELLY CRITERION
    # =========================================================================
    
    avg_win = expected_move_pct / 100
    avg_loss = avg_win * 0.7  # Assume 0.7 loss ratio
    kelly = compute_kelly_fraction(win_rate / 100, avg_win, avg_loss)
    
    if kelly > 0.15:
        reasoning.append(f"Kelly suggests {kelly:.1%} - strong edge")
    elif kelly > 0.05:
        reasoning.append(f"Kelly suggests {kelly:.1%} - moderate edge")
    elif kelly <= 0:
        warnings.append("Kelly is negative - no statistical edge")
        base_size *= 0.5  # Reduce if no edge
    
    # =========================================================================
    # FINAL CALCULATION
    # =========================================================================
    
    final_size = base_size * regime_mult * vix_mult * (1 + cot_adjustment)
    
    # Cap at max position
    if final_size > profile['max_position']:
        final_size = profile['max_position']
        warnings.append(f"Capped at max position {profile['max_position']:.1%}")
    
    # Minimum position (don't bother with tiny positions)
    if final_size < 0.01:
        final_size = 0.0
        warnings.append("Position too small to be meaningful")
    
    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================
    
    stop_loss = profile['stop_loss_default']
    
    # Widen stop in high vol, tighten in low vol
    if vix > 25:
        stop_loss *= 1.3
    elif vix < 15:
        stop_loss *= 0.8
    
    # Profit target based on expected move
    profit_target = expected_move_pct / 100
    risk_reward = profit_target / stop_loss if stop_loss > 0 else 0
    
    position_value = portfolio_value * final_size
    max_loss = position_value * stop_loss
    
    # =========================================================================
    # TIMING
    # =========================================================================
    
    if signal_confidence > 80 and regime_confidence > 70:
        urgency = 'immediate'
        entry_strategy = 'market'
    elif signal_confidence > 65:
        urgency = 'today'
        entry_strategy = 'limit'
    elif signal_confidence > 50:
        urgency = 'this_week'
        entry_strategy = 'scale_in'
    else:
        urgency = 'monitor'
        entry_strategy = 'wait'
    
    return PositionRecommendation(
        asset=asset,
        direction=direction,
        base_size_pct=base_size,
        regime_multiplier=regime_mult * vix_mult,
        confidence_multiplier=confidence_factor,
        cot_adjustment=cot_adjustment,
        final_size_pct=final_size,
        stop_loss_pct=stop_loss,
        profit_target_pct=profit_target,
        risk_reward_ratio=risk_reward,
        max_loss_dollars=max_loss,
        urgency=urgency,
        entry_strategy=entry_strategy,
        reasoning=reasoning,
        warnings=warnings,
    )


def get_position_for_api(
    asset: str,
    direction: str,
    confidence: float,
    win_rate: float,
    expected_move: float,
    portfolio_value: float,
    persona: str = 'retail',
) -> Dict:
    """Get position recommendation in API-friendly format."""
    try:
        persona_enum = Persona(persona.lower())
    except ValueError:
        persona_enum = Persona.RETAIL
    
    rec = compute_position_size(
        asset=asset,
        direction=direction,
        signal_confidence=confidence,
        win_rate=win_rate,
        expected_move_pct=expected_move,
        portfolio_value=portfolio_value,
        persona=persona_enum,
    )
    
    return {
        'asset': rec.asset,
        'direction': rec.direction,
        'position_size_pct': round(rec.final_size_pct * 100, 2),
        'position_value': round(portfolio_value * rec.final_size_pct, 2),
        'risk_management': {
            'stop_loss_pct': round(rec.stop_loss_pct * 100, 2),
            'profit_target_pct': round(rec.profit_target_pct * 100, 2),
            'risk_reward': round(rec.risk_reward_ratio, 2),
            'max_loss': round(rec.max_loss_dollars, 2),
        },
        'timing': {
            'urgency': rec.urgency,
            'entry_strategy': rec.entry_strategy,
        },
        'factors': {
            'base_size': round(rec.base_size_pct * 100, 2),
            'regime_mult': round(rec.regime_multiplier, 2),
            'confidence_mult': round(rec.confidence_multiplier, 2),
            'cot_adjustment': round(rec.cot_adjustment * 100, 1),
        },
        'reasoning': rec.reasoning,
        'warnings': rec.warnings,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("POSITION SIZING ENGINE TEST")
    print("=" * 70)
    
    # Example: Crude oil long signal
    result = get_position_for_api(
        asset='crude-oil',
        direction='long',
        confidence=72,
        win_rate=68,
        expected_move=2.5,
        portfolio_value=100000,
        persona='hedge_fund',
    )
    
    print(f"\nAsset: {result['asset']}")
    print(f"Direction: {result['direction']}")
    print(f"Position Size: {result['position_size_pct']}% (${result['position_value']:,.0f})")
    print(f"\nRisk Management:")
    print(f"  Stop Loss: {result['risk_management']['stop_loss_pct']}%")
    print(f"  Profit Target: {result['risk_management']['profit_target_pct']}%")
    print(f"  Risk/Reward: {result['risk_management']['risk_reward']:.1f}x")
    print(f"  Max Loss: ${result['risk_management']['max_loss']:,.0f}")
    print(f"\nTiming:")
    print(f"  Urgency: {result['timing']['urgency']}")
    print(f"  Entry: {result['timing']['entry_strategy']}")
    print(f"\nReasoning:")
    for r in result['reasoning']:
        print(f"  - {r}")
    if result['warnings']:
        print(f"\nWarnings:")
        for w in result['warnings']:
            print(f"  ! {w}")
