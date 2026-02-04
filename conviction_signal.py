"""
CONVICTION-BASED Signal Generator
=================================
Instead of just averaging models, look at:
1. MODEL SPREAD - tight spread = high conviction
2. HORIZON AGREEMENT - when multiple horizons agree
3. SIGNAL STRENGTH - filter for strong consensus signals

For traders: Only trade when conviction is HIGH

Created: 2026-02-03
Author: AmiraB
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
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


def compute_conviction_signal(horizons: dict, t: int) -> dict:
    """
    Compute a conviction-based trading signal for time t.
    
    Returns:
        signal: +1 (bullish), -1 (bearish), 0 (no trade)
        conviction: 0-100 (how confident)
        expected_move: predicted $ move
        spread: model disagreement
    """
    # Get current price (use actual from first horizon)
    current_price = horizons[1]['y'][t] if t < len(horizons[1]['y']) else 0
    
    bullish_count = 0
    bearish_count = 0
    total_horizons = 0
    
    all_predicted_moves = []
    all_spreads = []
    
    for h, data in horizons.items():
        if t >= len(data['X']):
            continue
            
        X_t = data['X'][t, :]  # All model predictions for horizon h at time t
        
        # Model spread (disagreement)
        spread = X_t.std()
        all_spreads.append(spread)
        
        # Predicted move from current price
        pred_moves = X_t - current_price
        avg_pred_move = pred_moves.mean()
        all_predicted_moves.append(avg_pred_move)
        
        # Count bullish vs bearish models
        bullish_models = (pred_moves > 0).sum()
        bearish_models = (pred_moves < 0).sum()
        
        total_models = len(pred_moves)
        if bullish_models > bearish_models:
            bullish_count += 1
        elif bearish_models > bullish_models:
            bearish_count += 1
            
        total_horizons += 1
    
    if total_horizons == 0:
        return {'signal': 0, 'conviction': 0, 'expected_move': 0, 'spread': 0}
    
    # Horizon agreement
    horizon_bullish_pct = bullish_count / total_horizons
    horizon_bearish_pct = bearish_count / total_horizons
    
    # Average expected move
    avg_move = np.mean(all_predicted_moves) if all_predicted_moves else 0
    
    # Average spread (lower = more agreement)
    avg_spread = np.mean(all_spreads) if all_spreads else 999
    
    # Compute conviction score (0-100)
    # High conviction = strong horizon agreement + low model spread
    horizon_agreement = max(horizon_bullish_pct, horizon_bearish_pct)
    spread_score = max(0, 1 - avg_spread / 2)  # Normalize, lower spread = higher score
    
    conviction = (horizon_agreement * 0.6 + spread_score * 0.4) * 100
    
    # Signal
    if horizon_bullish_pct >= 0.7:  # 70%+ horizons bullish
        signal = 1
    elif horizon_bearish_pct >= 0.7:  # 70%+ horizons bearish
        signal = -1
    else:
        signal = 0  # No clear consensus
    
    return {
        'signal': signal,
        'conviction': round(conviction, 1),
        'expected_move': round(avg_move, 2),
        'spread': round(avg_spread, 2),
        'horizon_agreement': round(horizon_agreement * 100, 1),
        'bullish_horizons': bullish_count,
        'bearish_horizons': bearish_count,
    }


def backtest_conviction_strategy(horizons: dict, 
                                  min_conviction: float = 50,
                                  min_expected_move: float = 0.5) -> dict:
    """
    Backtest a conviction-based trading strategy.
    
    Args:
        min_conviction: Only trade when conviction >= this (0-100)
        min_expected_move: Only trade when expected move >= this ($)
    """
    min_len = min(h['X'].shape[0] for h in horizons.values())
    train_end = int(min_len * 0.7)
    
    y = horizons[1]['y']
    
    trades = []
    oos_start = train_end
    
    for t in range(oos_start, min_len - 1):
        signal_info = compute_conviction_signal(horizons, t)
        
        # Filter by conviction and expected move
        if (signal_info['conviction'] >= min_conviction and 
            abs(signal_info['expected_move']) >= min_expected_move and
            signal_info['signal'] != 0):
            
            # Execute trade
            actual_move = y[t + 1] - y[t]
            profit = signal_info['signal'] * actual_move
            correct = (signal_info['signal'] == np.sign(actual_move))
            
            trades.append({
                'time': t,
                'signal': signal_info['signal'],
                'conviction': signal_info['conviction'],
                'expected_move': signal_info['expected_move'],
                'actual_move': actual_move,
                'profit': profit,
                'correct': correct,
            })
    
    if not trades:
        return {
            'n_trades': 0,
            'total_profit': 0,
            'accuracy': 0,
            'avg_profit': 0,
            'win_rate': 0,
            'sharpe': 0,
        }
    
    df_trades = pd.DataFrame(trades)
    
    n_trades = len(df_trades)
    total_profit = df_trades['profit'].sum()
    accuracy = df_trades['correct'].mean()
    avg_profit = df_trades['profit'].mean()
    win_rate = (df_trades['profit'] > 0).mean()
    
    profits = df_trades['profit'].values
    sharpe = profits.mean() / (profits.std() + 1e-8) * np.sqrt(252)
    
    return {
        'n_trades': n_trades,
        'total_profit': round(total_profit, 2),
        'accuracy': round(accuracy * 100, 1),
        'avg_profit': round(avg_profit, 2),
        'win_rate': round(win_rate * 100, 1),
        'sharpe': round(sharpe, 2),
        'trades': df_trades,
    }


def optimize_conviction_parameters(horizons: dict) -> dict:
    """Find optimal conviction parameters."""
    
    results = []
    
    for min_conv in [30, 40, 50, 60, 70, 80]:
        for min_move in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            bt = backtest_conviction_strategy(horizons, min_conv, min_move)
            
            if bt['n_trades'] >= 5:  # At least 5 trades
                results.append({
                    'min_conviction': min_conv,
                    'min_move': min_move,
                    **{k: v for k, v in bt.items() if k != 'trades'}
                })
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("CONVICTION-BASED TRADING SIGNAL OPTIMIZER")
    print("For traders who need HIGH CONFIDENCE signals")
    print("=" * 70)
    
    horizons = load_all_horizons()
    
    if not horizons:
        print("ERROR: No data loaded")
        return
    
    print(f"\nLoaded {len(horizons)} horizons")
    
    # Optimize parameters
    print("\nOptimizing conviction parameters...")
    results = optimize_conviction_parameters(horizons)
    
    if len(results) == 0:
        print("No valid parameter combinations found")
        return
    
    print("\n" + "=" * 70)
    print("TOP 20 PARAMETER COMBINATIONS BY TOTAL PROFIT")
    print("=" * 70)
    
    top_profit = results.nlargest(20, 'total_profit')[[
        'min_conviction', 'min_move', 'total_profit', 'accuracy', 
        'n_trades', 'avg_profit', 'sharpe'
    ]]
    print(top_profit.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("TOP 10 BY ACCURACY (> 50%)")
    print("=" * 70)
    
    accurate = results[results['accuracy'] > 50]
    if len(accurate) > 0:
        top_acc = accurate.nlargest(10, 'accuracy')[[
            'min_conviction', 'min_move', 'accuracy', 'total_profit',
            'n_trades', 'win_rate', 'sharpe'
        ]]
        print(top_acc.to_string(index=False))
    else:
        print("No combinations with >50% accuracy")
    
    print("\n" + "=" * 70)
    print("TOP 10 BY SHARPE RATIO")
    print("=" * 70)
    
    top_sharpe = results.nlargest(10, 'sharpe')[[
        'min_conviction', 'min_move', 'sharpe', 'total_profit',
        'accuracy', 'n_trades', 'avg_profit'
    ]]
    print(top_sharpe.to_string(index=False))
    
    # Find best overall
    best = results.loc[results['total_profit'].idxmax()]
    
    print("\n" + "=" * 70)
    print(">>> BEST TRADING CONFIGURATION <<<")
    print("=" * 70)
    print(f"Min Conviction:  {best['min_conviction']}%")
    print(f"Min Exp Move:    ${best['min_move']}")
    print(f"")
    print(f"Total Profit:    ${best['total_profit']:.2f}")
    print(f"Accuracy:        {best['accuracy']:.1f}%")
    print(f"# Trades:        {best['n_trades']}")
    print(f"Avg Profit:      ${best['avg_profit']:.2f}/trade")
    print(f"Win Rate:        {best['win_rate']:.1f}%")
    print(f"Sharpe:          {best['sharpe']:.2f}")
    
    # Run detailed backtest on best params
    print("\n" + "=" * 70)
    print("DETAILED BACKTEST - BEST CONFIGURATION")
    print("=" * 70)
    
    bt = backtest_conviction_strategy(horizons, best['min_conviction'], best['min_move'])
    
    if 'trades' in bt and len(bt['trades']) > 0:
        df_trades = bt['trades']
        
        # Show recent trades
        print("\nLast 10 Trades:")
        print("-" * 60)
        
        for _, trade in df_trades.tail(10).iterrows():
            direction = "LONG" if trade['signal'] > 0 else "SHORT"
            result = "WIN" if trade['correct'] else "LOSS"
            print(f"  {direction:5s} | Conv: {trade['conviction']:.0f}% | "
                  f"Expected: ${trade['expected_move']:+.2f} | "
                  f"Actual: ${trade['actual_move']:+.2f} | "
                  f"P/L: ${trade['profit']:+.2f} | {result}")
        
        # Summary stats
        print("\nWinning Trades:")
        wins = df_trades[df_trades['profit'] > 0]
        if len(wins) > 0:
            print(f"  Count: {len(wins)}")
            print(f"  Avg Win: ${wins['profit'].mean():.2f}")
            print(f"  Max Win: ${wins['profit'].max():.2f}")
        
        print("\nLosing Trades:")
        losses = df_trades[df_trades['profit'] <= 0]
        if len(losses) > 0:
            print(f"  Count: {len(losses)}")
            print(f"  Avg Loss: ${losses['profit'].mean():.2f}")
            print(f"  Max Loss: ${losses['profit'].min():.2f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path("results") / f"conviction_optimization_{timestamp}.csv"
    output_file.parent.mkdir(exist_ok=True)
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = main()
