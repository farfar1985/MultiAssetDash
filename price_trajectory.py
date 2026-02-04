"""
PRICE TRAJECTORY VISUALIZATION
==============================
Show the probable price path from D+1 to D+10 in a compelling way.

Features:
1. Multi-horizon price trajectory (the "path")
2. Confidence cone (widens over time)
3. Model consensus strength at each point
4. Pairwise slope indicators (trend momentum)
5. Actionable trading zones

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


def generate_price_trajectory(horizons: dict) -> dict:
    """
    Generate a complete price trajectory visualization.
    
    This creates the "probable path" from today through D+10,
    with confidence bands and momentum indicators.
    """
    # Get current price (last valid actual)
    y = horizons[1]['y']
    valid_prices = y[y > 0]
    current_price = valid_prices[-1] if len(valid_prices) > 0 else 62.0
    
    today = datetime.now()
    
    trajectory = {
        'current': {
            'date': today.strftime('%Y-%m-%d'),
            'price': round(current_price, 2),
            'type': 'actual'
        },
        'path': [],
        'summary': {},
        'trading_signal': {},
    }
    
    # Build the trajectory for each horizon
    prev_price = current_price
    cumulative_bullish = 0
    cumulative_bearish = 0
    
    for h in range(1, 11):
        if h not in horizons:
            continue
        
        X = horizons[h]['X']
        latest_preds = X[-1, :]  # All model predictions for latest time
        
        # Statistics
        pred_mean = latest_preds.mean()
        pred_median = np.median(latest_preds)
        pred_std = latest_preds.std()
        
        # Use TOP 10% most confident (lowest variance historically)
        if X.shape[0] > 10:
            model_vars = X[-30:].var(axis=0)  # Last 30 days variance
            top_k = max(1, int(len(latest_preds) * 0.1))
            top_idx = np.argsort(model_vars)[:top_k]
            top_10_mean = latest_preds[top_idx].mean()
            top_10_std = latest_preds[top_idx].std()
        else:
            top_10_mean = pred_mean
            top_10_std = pred_std
        
        # Confidence intervals
        conf_90_low = np.percentile(latest_preds, 5)
        conf_90_high = np.percentile(latest_preds, 95)
        conf_50_low = np.percentile(latest_preds, 25)
        conf_50_high = np.percentile(latest_preds, 75)
        
        # Model consensus
        bullish_models = (latest_preds > current_price + 0.5).sum()
        bearish_models = (latest_preds < current_price - 0.5).sum()
        neutral_models = len(latest_preds) - bullish_models - bearish_models
        
        bullish_pct = bullish_models / len(latest_preds) * 100
        bearish_pct = bearish_models / len(latest_preds) * 100
        
        # Trend momentum (slope from previous horizon)
        move_from_current = top_10_mean - current_price
        move_from_prev = top_10_mean - prev_price
        
        # Cumulative direction
        if move_from_prev > 0.1:
            cumulative_bullish += 1
        elif move_from_prev < -0.1:
            cumulative_bearish += 1
        
        # Determine signal strength
        if bullish_pct > 70:
            signal = 'STRONG_BULLISH'
            signal_color = '#22c55e'
        elif bullish_pct > 55:
            signal = 'BULLISH'
            signal_color = '#86efac'
        elif bearish_pct > 70:
            signal = 'STRONG_BEARISH'
            signal_color = '#ef4444'
        elif bearish_pct > 55:
            signal = 'BEARISH'
            signal_color = '#fca5a5'
        else:
            signal = 'NEUTRAL'
            signal_color = '#f59e0b'
        
        forecast_date = today + timedelta(days=h)
        
        point = {
            'horizon': h,
            'date': forecast_date.strftime('%Y-%m-%d'),
            'days_out': h,
            
            # Price predictions
            'price_forecast': round(top_10_mean, 2),
            'price_all_models': round(pred_mean, 2),
            
            # Confidence bands
            'conf_90': [round(conf_90_low, 2), round(conf_90_high, 2)],
            'conf_50': [round(conf_50_low, 2), round(conf_50_high, 2)],
            'uncertainty': round(top_10_std, 2),
            
            # Move from current
            'move_from_current': round(move_from_current, 2),
            'move_pct': round(move_from_current / current_price * 100, 2),
            
            # Trend
            'trend_slope': round(move_from_prev, 2),
            'trend_direction': 'UP' if move_from_prev > 0.1 else ('DOWN' if move_from_prev < -0.1 else 'FLAT'),
            
            # Consensus
            'bullish_pct': round(bullish_pct, 1),
            'bearish_pct': round(bearish_pct, 1),
            'consensus': signal,
            'consensus_color': signal_color,
            
            # Model info
            'n_models': len(latest_preds),
            'model_spread': round(pred_std, 2),
        }
        
        trajectory['path'].append(point)
        prev_price = top_10_mean
    
    # Summary statistics
    path = trajectory['path']
    final_point = path[-1] if path else None
    
    if final_point:
        # Overall trajectory direction
        total_move = final_point['price_forecast'] - current_price
        
        # Count bullish vs bearish horizons
        bullish_horizons = sum(1 for p in path if p['bullish_pct'] > 55)
        bearish_horizons = sum(1 for p in path if p['bearish_pct'] > 55)
        
        # Trend consistency
        up_slopes = sum(1 for p in path if p['trend_slope'] > 0.1)
        down_slopes = sum(1 for p in path if p['trend_slope'] < -0.1)
        
        trajectory['summary'] = {
            'current_price': round(current_price, 2),
            'final_target': final_point['price_forecast'],
            'total_move': round(total_move, 2),
            'total_move_pct': round(total_move / current_price * 100, 2),
            'bullish_horizons': bullish_horizons,
            'bearish_horizons': bearish_horizons,
            'trend_consistency': round((max(up_slopes, down_slopes) / len(path)) * 100, 1),
            'avg_uncertainty': round(np.mean([p['uncertainty'] for p in path]), 2),
        }
        
        # Trading signal based on trajectory
        if bullish_horizons >= 7 and total_move > 1:
            overall_signal = 'STRONG_BUY'
            confidence = 'HIGH'
        elif bullish_horizons >= 5 and total_move > 0.5:
            overall_signal = 'BUY'
            confidence = 'MEDIUM'
        elif bearish_horizons >= 7 and total_move < -1:
            overall_signal = 'STRONG_SELL'
            confidence = 'HIGH'
        elif bearish_horizons >= 5 and total_move < -0.5:
            overall_signal = 'SELL'
            confidence = 'MEDIUM'
        else:
            overall_signal = 'HOLD'
            confidence = 'LOW'
        
        trajectory['trading_signal'] = {
            'signal': overall_signal,
            'confidence': confidence,
            'target_price': final_point['price_forecast'],
            'stop_loss': round(current_price - abs(total_move) * 0.5, 2) if total_move > 0 else round(current_price + abs(total_move) * 0.5, 2),
            'take_profit': round(current_price + total_move * 1.5, 2),
            'risk_reward': round(abs(total_move * 1.5) / (abs(total_move) * 0.5), 2) if total_move != 0 else 0,
        }
    
    return trajectory


def generate_trajectory_chart_data(trajectory: dict) -> dict:
    """
    Generate data formatted for charting libraries (ECharts, etc.)
    """
    path = trajectory['path']
    current = trajectory['current']
    
    # X-axis: dates
    dates = [current['date']] + [p['date'] for p in path]
    
    # Main forecast line
    forecast_line = [current['price']] + [p['price_forecast'] for p in path]
    
    # All-models line (comparison)
    all_models_line = [current['price']] + [p['price_all_models'] for p in path]
    
    # Confidence bands (for area chart)
    conf_90_upper = [current['price']] + [p['conf_90'][1] for p in path]
    conf_90_lower = [current['price']] + [p['conf_90'][0] for p in path]
    conf_50_upper = [current['price']] + [p['conf_50'][1] for p in path]
    conf_50_lower = [current['price']] + [p['conf_50'][0] for p in path]
    
    # Consensus colors for each point
    consensus_colors = ['#3b82f6'] + [p['consensus_color'] for p in path]
    
    # Trend arrows
    trend_indicators = [
        {
            'horizon': p['horizon'],
            'direction': p['trend_direction'],
            'slope': p['trend_slope']
        }
        for p in path
    ]
    
    return {
        'dates': dates,
        'series': {
            'forecast': forecast_line,
            'all_models': all_models_line,
            'conf_90_upper': conf_90_upper,
            'conf_90_lower': conf_90_lower,
            'conf_50_upper': conf_50_upper,
            'conf_50_lower': conf_50_lower,
        },
        'consensus_colors': consensus_colors,
        'trend_indicators': trend_indicators,
        'annotations': {
            'current': {
                'x': dates[0],
                'y': forecast_line[0],
                'label': f'NOW: ${forecast_line[0]:.2f}'
            },
            'target': {
                'x': dates[-1],
                'y': forecast_line[-1],
                'label': f'D+10: ${forecast_line[-1]:.2f}'
            }
        }
    }


def print_trajectory_visual(trajectory: dict):
    """Print an ASCII visualization of the trajectory."""
    path = trajectory['path']
    current = trajectory['current']
    
    print("\n" + "=" * 70)
    print("CRUDE OIL PRICE TRAJECTORY")
    print("=" * 70)
    
    print(f"\nCurrent Price: ${current['price']:.2f}")
    print(f"Date: {current['date']}")
    
    print("\n" + "-" * 70)
    print(f"{'Horizon':<8} {'Target':>10} {'Move':>10} {'Trend':>8} {'Consensus':>15} {'Models'}")
    print("-" * 70)
    
    for p in path:
        move_str = f"${p['move_from_current']:+.2f}" if p['move_from_current'] != 0 else "$0.00"
        trend_arrow = "^" if p['trend_direction'] == 'UP' else ("v" if p['trend_direction'] == 'DOWN' else "-")
        
        print(f"D+{p['horizon']:<6} ${p['price_forecast']:>8.2f} {move_str:>10} "
              f"{trend_arrow:>8} {p['consensus']:>15} {p['bullish_pct']:.0f}%B/{p['bearish_pct']:.0f}%S")
    
    print("-" * 70)
    
    summary = trajectory['summary']
    signal = trajectory['trading_signal']
    
    print(f"\nTOTAL PROJECTED MOVE: ${summary['total_move']:+.2f} ({summary['total_move_pct']:+.1f}%)")
    print(f"Bullish Horizons: {summary['bullish_horizons']}/10")
    print(f"Bearish Horizons: {summary['bearish_horizons']}/10")
    print(f"Trend Consistency: {summary['trend_consistency']:.1f}%")
    
    print(f"\n>>> TRADING SIGNAL: {signal['signal']} <<<")
    print(f"Confidence: {signal['confidence']}")
    print(f"Target: ${signal['target_price']:.2f}")
    print(f"Stop Loss: ${signal['stop_loss']:.2f}")
    print(f"Take Profit: ${signal['take_profit']:.2f}")
    print(f"Risk/Reward: {signal['risk_reward']:.1f}:1")


def main():
    print("=" * 70)
    print("PRICE TRAJECTORY GENERATOR")
    print("Showing the probable path from NOW to D+10")
    print("=" * 70)
    
    horizons = load_all_horizons()
    
    if not horizons:
        print("ERROR: No data loaded")
        return
    
    print(f"Loaded {len(horizons)} horizons")
    
    # Generate trajectory
    trajectory = generate_price_trajectory(horizons)
    
    # Print visual
    print_trajectory_visual(trajectory)
    
    # Generate chart data
    chart_data = generate_trajectory_chart_data(trajectory)
    
    # Save outputs
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Full trajectory
    with open(output_dir / "price_trajectory.json", 'w') as f:
        json.dump(trajectory, f, indent=2)
    print(f"\nTrajectory saved to: {output_dir / 'price_trajectory.json'}")
    
    # Chart data
    with open(output_dir / "trajectory_chart_data.json", 'w') as f:
        json.dump(chart_data, f, indent=2)
    print(f"Chart data saved to: {output_dir / 'trajectory_chart_data.json'}")
    
    # Print chart data structure for frontend
    print("\n" + "=" * 70)
    print("CHART DATA STRUCTURE (for ECharts)")
    print("=" * 70)
    print(f"\nDates: {chart_data['dates']}")
    print(f"\nForecast line: {chart_data['series']['forecast']}")
    print(f"\n90% Confidence Upper: {chart_data['series']['conf_90_upper']}")
    print(f"90% Confidence Lower: {chart_data['series']['conf_90_lower']}")
    
    return trajectory, chart_data


if __name__ == "__main__":
    trajectory, chart_data = main()
