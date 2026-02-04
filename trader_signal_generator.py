"""
PRACTICAL TRADER SIGNAL GENERATOR
==================================
Generate actionable trading signals that show:
1. DIRECTION: Bullish or Bearish consensus
2. CONFIDENCE: How many models/horizons agree
3. TARGETS: Price targets with ranges
4. CONVICTION LEVEL: When to trade vs when to wait

This is for the DASHBOARD - showing traders what to do.

Created: 2026-02-03
Author: AmiraB
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def load_all_horizons():
    """Load all Crude Oil horizon data."""
    data_dir = Path("data/1866_Crude_Oil/horizons_wide")
    
    horizons = {}
    for h in range(1, 11):
        f = data_dir / f"horizon_{h}.joblib"
        if f.exists():
            data = joblib.load(f)
            X = data['X'].values if hasattr(data['X'], 'values') else data['X']
            y = data['y'].values if hasattr(data['y'], 'values') else data['y']
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)
            horizons[h] = {'X': np.array(X), 'y': np.array(y)}
    
    return horizons


def generate_trading_signal(horizons: dict) -> dict:
    """
    Generate a complete trading signal for the CURRENT moment.
    This is what goes on the dashboard.
    """
    # Use latest data point
    t = -1  # Last available
    
    # Get current price estimate (most recent actual)
    current_price = horizons[1]['y'][t]
    
    # Analyze each horizon
    horizon_analysis = {}
    all_bullish = 0
    all_bearish = 0
    all_neutral = 0
    
    target_prices = []
    model_spreads = []
    
    for h, data in horizons.items():
        X_t = data['X'][t, :]  # All model predictions for this horizon
        
        # Statistics
        pred_mean = X_t.mean()
        pred_median = np.median(X_t)
        pred_std = X_t.std()
        pred_low = np.percentile(X_t, 10)
        pred_high = np.percentile(X_t, 90)
        
        # Direction consensus
        pred_moves = X_t - current_price
        bullish_pct = (pred_moves > 0.5).mean() * 100  # > $0.50 move up
        bearish_pct = (pred_moves < -0.5).mean() * 100  # > $0.50 move down
        
        # Expected move
        expected_move = pred_median - current_price
        
        # Classify
        if bullish_pct > 60:
            direction = 'BULLISH'
            all_bullish += 1
        elif bearish_pct > 60:
            direction = 'BEARISH'
            all_bearish += 1
        else:
            direction = 'NEUTRAL'
            all_neutral += 1
        
        horizon_analysis[f'D+{h}'] = {
            'direction': direction,
            'target_price': round(pred_median, 2),
            'target_range': [round(pred_low, 2), round(pred_high, 2)],
            'expected_move': round(expected_move, 2),
            'expected_move_pct': round(expected_move / current_price * 100, 2),
            'model_spread': round(pred_std, 2),
            'bullish_models': round(bullish_pct, 1),
            'bearish_models': round(bearish_pct, 1),
            'n_models': len(X_t),
        }
        
        target_prices.append(pred_median)
        model_spreads.append(pred_std)
    
    # Overall signal
    total_horizons = all_bullish + all_bearish + all_neutral
    
    if all_bullish >= 7:  # 7+ of 10 horizons bullish
        overall_signal = 'STRONG_BUY'
        signal_color = '#22c55e'
    elif all_bullish >= 5:
        overall_signal = 'BUY'
        signal_color = '#86efac'
    elif all_bearish >= 7:
        overall_signal = 'STRONG_SELL'
        signal_color = '#ef4444'
    elif all_bearish >= 5:
        overall_signal = 'SELL'
        signal_color = '#fca5a5'
    else:
        overall_signal = 'HOLD'
        signal_color = '#f59e0b'
    
    # Conviction score (0-100)
    horizon_agreement = max(all_bullish, all_bearish) / total_horizons
    avg_spread = np.mean(model_spreads)
    spread_score = max(0, 1 - avg_spread / 2)  # Lower spread = higher score
    conviction = round((horizon_agreement * 0.7 + spread_score * 0.3) * 100, 1)
    
    # Price targets
    avg_target = np.mean(target_prices)
    min_target = min(target_prices)
    max_target = max(target_prices)
    
    # Expected move summary
    avg_move = avg_target - current_price
    
    return {
        'timestamp': datetime.now().isoformat(),
        'asset': 'Crude Oil',
        'current_price': round(current_price, 2),
        
        'signal': {
            'direction': overall_signal,
            'color': signal_color,
            'conviction': conviction,
            'bullish_horizons': all_bullish,
            'bearish_horizons': all_bearish,
            'neutral_horizons': all_neutral,
        },
        
        'targets': {
            'average': round(avg_target, 2),
            'conservative': round(min_target, 2),
            'aggressive': round(max_target, 2),
            'expected_move': round(avg_move, 2),
            'expected_move_pct': round(avg_move / current_price * 100, 2),
        },
        
        'horizons': horizon_analysis,
        
        'trading_recommendation': generate_trading_recommendation(
            overall_signal, conviction, avg_move, current_price
        ),
    }


def generate_trading_recommendation(signal: str, conviction: float, 
                                    expected_move: float, current_price: float) -> dict:
    """Generate human-readable trading recommendation."""
    
    if signal in ['STRONG_BUY', 'BUY']:
        action = 'LONG'
        direction = 'bullish'
    elif signal in ['STRONG_SELL', 'SELL']:
        action = 'SHORT'
        direction = 'bearish'
    else:
        action = 'WAIT'
        direction = 'neutral'
    
    # Position sizing based on conviction
    if conviction >= 80:
        size = 'Full position'
        risk = 'Normal'
    elif conviction >= 60:
        size = 'Half position'
        risk = 'Reduced'
    elif conviction >= 40:
        size = 'Quarter position'
        risk = 'Conservative'
    else:
        size = 'No position'
        risk = 'N/A'
    
    # Stop loss / take profit
    if action in ['LONG', 'SHORT']:
        stop_distance = abs(expected_move) * 0.5  # 50% of expected move
        tp_distance = abs(expected_move) * 1.5  # 150% of expected move
        
        if action == 'LONG':
            stop = current_price - stop_distance
            tp = current_price + tp_distance
        else:
            stop = current_price + stop_distance
            tp = current_price - tp_distance
    else:
        stop = None
        tp = None
    
    recommendation = {
        'action': action,
        'direction': direction,
        'position_size': size,
        'risk_level': risk,
        'entry_price': current_price,
        'stop_loss': round(stop, 2) if stop else None,
        'take_profit': round(tp, 2) if tp else None,
        'conviction_level': 'HIGH' if conviction >= 70 else ('MEDIUM' if conviction >= 50 else 'LOW'),
    }
    
    # Human readable summary
    if action == 'WAIT':
        recommendation['summary'] = (
            f"No clear signal. Markets are {direction}. "
            f"Wait for conviction to increase above 50% before entering."
        )
    else:
        recommendation['summary'] = (
            f"Go {action} at ${current_price:.2f}. "
            f"Target: ${tp:.2f} ({abs(expected_move):.2f} move). "
            f"Stop: ${stop:.2f}. "
            f"Conviction: {conviction:.0f}% ({recommendation['conviction_level']}). "
            f"Position: {size}."
        )
    
    return recommendation


def generate_target_ladder(horizons: dict) -> list:
    """
    Generate price target ladder across all horizons.
    This shows traders the expected price at each timeframe.
    """
    current_price = horizons[1]['y'][-1]
    
    ladder = []
    
    for h in sorted(horizons.keys()):
        X_t = horizons[h]['X'][-1, :]
        
        target = np.median(X_t)
        low = np.percentile(X_t, 20)
        high = np.percentile(X_t, 80)
        
        move = target - current_price
        move_pct = move / current_price * 100
        
        # Agreement strength
        bullish = (X_t > current_price + 0.5).mean()
        bearish = (X_t < current_price - 0.5).mean()
        
        ladder.append({
            'horizon': f'D+{h}',
            'days': h,
            'target': round(target, 2),
            'range': [round(low, 2), round(high, 2)],
            'move': round(move, 2),
            'move_pct': round(move_pct, 2),
            'direction': 'UP' if move > 0.5 else ('DOWN' if move < -0.5 else 'FLAT'),
            'strength': round(max(bullish, bearish) * 100, 1),
        })
    
    return ladder


def main():
    print("=" * 70)
    print("TRADER SIGNAL GENERATOR")
    print("Generating actionable signals for the dashboard")
    print("=" * 70)
    
    horizons = load_all_horizons()
    
    if not horizons:
        print("ERROR: No data loaded")
        return
    
    # Generate trading signal
    signal = generate_trading_signal(horizons)
    
    print(f"\n{'='*70}")
    print("CURRENT TRADING SIGNAL - CRUDE OIL")
    print(f"{'='*70}")
    
    print(f"\nCurrent Price: ${signal['current_price']}")
    print(f"\n>>> {signal['signal']['direction']} <<<")
    print(f"Conviction: {signal['signal']['conviction']}%")
    print(f"Bullish Horizons: {signal['signal']['bullish_horizons']}/10")
    print(f"Bearish Horizons: {signal['signal']['bearish_horizons']}/10")
    
    print(f"\nPRICE TARGETS:")
    print(f"  Conservative: ${signal['targets']['conservative']}")
    print(f"  Average:      ${signal['targets']['average']}")
    print(f"  Aggressive:   ${signal['targets']['aggressive']}")
    print(f"  Expected Move: ${signal['targets']['expected_move']} ({signal['targets']['expected_move_pct']}%)")
    
    print(f"\nTRADING RECOMMENDATION:")
    rec = signal['trading_recommendation']
    print(f"  Action: {rec['action']}")
    print(f"  Position Size: {rec['position_size']}")
    if rec['stop_loss']:
        print(f"  Stop Loss: ${rec['stop_loss']}")
        print(f"  Take Profit: ${rec['take_profit']}")
    print(f"\n  {rec['summary']}")
    
    # Target ladder
    print(f"\n{'='*70}")
    print("TARGET LADDER (Price Targets by Horizon)")
    print(f"{'='*70}")
    
    ladder = generate_target_ladder(horizons)
    print(f"\n{'Horizon':<10} {'Target':>10} {'Range':>20} {'Move':>10} {'Dir':>8} {'Strength':>10}")
    print("-" * 70)
    for rung in ladder:
        range_str = f"${rung['range'][0]} - ${rung['range'][1]}"
        print(f"{rung['horizon']:<10} ${rung['target']:>8.2f} {range_str:>20} "
              f"${rung['move']:>+7.2f} {rung['direction']:>8} {rung['strength']:>9.1f}%")
    
    # Save to JSON for dashboard
    output = {
        'signal': signal,
        'target_ladder': ladder,
    }
    
    output_file = Path("results") / "current_trading_signal.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSignal saved to: {output_file}")
    
    # Also save horizon breakdown
    print(f"\n{'='*70}")
    print("HORIZON BREAKDOWN")
    print(f"{'='*70}")
    
    for h, analysis in signal['horizons'].items():
        print(f"\n{h}:")
        print(f"  Direction: {analysis['direction']}")
        print(f"  Target: ${analysis['target_price']} (${analysis['target_range'][0]} - ${analysis['target_range'][1]})")
        print(f"  Expected Move: ${analysis['expected_move']} ({analysis['expected_move_pct']}%)")
        print(f"  Models: {analysis['n_models']} ({analysis['bullish_models']:.0f}% bullish, {analysis['bearish_models']:.0f}% bearish)")
    
    return output


if __name__ == "__main__":
    output = main()
