"""
Generate Beautiful Chart Data for Nexus Dashboard
==================================================
Creates visualization-ready JSON for:
1. Historical price + forecast overlay
2. Multi-horizon forecast fan chart
3. Confidence band visualization
4. Model agreement heatmap data
5. Ensemble contribution breakdown

Designed for Apache ECharts / Lightweight Charts / D3.js

Created: 2026-02-03
Author: AmiraB
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import joblib


def load_crude_oil_data() -> Tuple[Dict, np.ndarray]:
    """Load all Crude Oil horizon data."""
    data_dir = Path("data/1866_Crude_Oil/horizons_wide")
    
    horizon_data = {}
    for h in range(1, 11):
        f = data_dir / f"horizon_{h}.joblib"
        if f.exists():
            data = joblib.load(f)
            # Ensure numpy arrays
            X = data['X'].values if hasattr(data['X'], 'values') else data['X']
            y = data['y'].values if hasattr(data['y'], 'values') else data['y']
            horizon_data[h] = {
                'X': np.array(X),  # (n_times, n_models)
                'y': np.array(y),  # (n_times,) actuals
            }
            print(f"Loaded D+{h}: {horizon_data[h]['X'].shape[0]} times, {horizon_data[h]['X'].shape[1]} models")
    
    return horizon_data


def generate_historical_chart(horizon_data: Dict, 
                              days: int = 90,
                              ensemble_method: str = 'equal') -> List[Dict]:
    """
    Generate historical forecast vs actual data for charting.
    
    Returns a list of points with:
    - date, actual, forecast, error, signal, confidence
    """
    # Use horizon 1 (D+1) for historical comparison
    if 1 not in horizon_data:
        h = min(horizon_data.keys())
    else:
        h = 1
    
    X = horizon_data[h]['X']
    y = horizon_data[h]['y']
    
    # Use last N days
    X = X[-days:]
    y = y[-days:]
    
    # Generate dates (ending today)
    end_date = datetime.now()
    dates = [end_date - timedelta(days=days-i-1) for i in range(days)]
    
    # Ensemble forecast (equal weight for simplicity)
    if ensemble_method == 'equal':
        forecast = X.mean(axis=1)
    elif ensemble_method == 'best_10pct':
        # Use top 10% by variance (proxy for confidence)
        var = X.var(axis=0)
        top_k = max(1, int(X.shape[1] * 0.1))
        top_idx = np.argsort(var)[:top_k]  # Lowest variance = most confident
        forecast = X[:, top_idx].mean(axis=1)
    else:
        forecast = X.mean(axis=1)
    
    # Model agreement (what % of models agree on direction)
    model_directions = np.sign(np.diff(X, axis=0, prepend=X[:1]))
    agreement = np.abs(model_directions.mean(axis=1))
    
    # Confidence from model agreement (scaled 0-100)
    confidence = agreement * 100
    
    # Signal (1 = bullish, -1 = bearish, 0 = neutral)
    consensus_direction = np.sign(model_directions.mean(axis=1))
    
    chart_data = []
    for i in range(len(dates)):
        point = {
            'date': dates[i].strftime('%Y-%m-%d'),
            'actual': round(float(y[i]), 2),
            'forecast': round(float(forecast[i]), 2),
            'error': round(float(forecast[i] - y[i]), 2),
            'errorPct': round(float((forecast[i] - y[i]) / y[i] * 100), 2),
            'signal': int(consensus_direction[i]),
            'confidence': round(float(confidence[i]), 1),
            'modelAgreement': round(float(agreement[i]), 3),
        }
        chart_data.append(point)
    
    return chart_data


def generate_forecast_fan(horizon_data: Dict) -> List[Dict]:
    """
    Generate multi-horizon forecast fan chart data.
    
    Shows forecasts for D+1 through D+10 with confidence bands,
    creating a "fan" shape as uncertainty grows with horizon.
    """
    # Get latest actual price
    ref_horizon = min(horizon_data.keys())
    last_actual = horizon_data[ref_horizon]['y'][-1]
    last_date = datetime.now()
    
    fan_data = []
    
    # Add anchor point (today's price)
    fan_data.append({
        'date': last_date.strftime('%Y-%m-%d'),
        'horizon': 0,
        'forecast': round(float(last_actual), 2),
        'lower90': round(float(last_actual), 2),
        'upper90': round(float(last_actual), 2),
        'lower50': round(float(last_actual), 2),
        'upper50': round(float(last_actual), 2),
        'isActual': True,
        'modelCount': 0,
        'signalStrength': 0,
    })
    
    # Add each horizon forecast
    for h in sorted(horizon_data.keys()):
        X = horizon_data[h]['X']
        
        # Latest predictions from all models
        latest_preds = X[-1, :]
        
        # Central tendency
        forecast = np.median(latest_preds)  # Median more robust than mean
        
        # Confidence intervals from model spread
        lower90 = np.percentile(latest_preds, 5)
        upper90 = np.percentile(latest_preds, 95)
        lower50 = np.percentile(latest_preds, 25)
        upper50 = np.percentile(latest_preds, 75)
        
        # Model agreement on direction
        directions = np.sign(latest_preds - last_actual)
        bullish_pct = (directions > 0).mean()
        bearish_pct = (directions < 0).mean()
        signal_strength = abs(bullish_pct - bearish_pct)  # 0 = split, 1 = unanimous
        
        forecast_date = last_date + timedelta(days=h)
        
        fan_data.append({
            'date': forecast_date.strftime('%Y-%m-%d'),
            'horizon': h,
            'forecast': round(float(forecast), 2),
            'lower90': round(float(lower90), 2),
            'upper90': round(float(upper90), 2),
            'lower50': round(float(lower50), 2),
            'upper50': round(float(upper50), 2),
            'isActual': False,
            'modelCount': X.shape[1],
            'signalStrength': round(signal_strength, 3),
            'bullishPct': round(bullish_pct * 100, 1),
            'bearishPct': round(bearish_pct * 100, 1),
            'priceChange': round(float(forecast - last_actual), 2),
            'priceChangePct': round(float((forecast - last_actual) / last_actual * 100), 2),
        })
    
    return fan_data


def generate_heatmap_data(horizon_data: Dict, days: int = 30) -> Dict:
    """
    Generate model agreement heatmap data.
    
    Returns a matrix of (date x horizon) showing signal strength.
    Perfect for visualizing when/where models agree.
    """
    horizons = sorted(horizon_data.keys())
    
    # Generate dates
    end_date = datetime.now()
    dates = [(end_date - timedelta(days=days-i-1)).strftime('%Y-%m-%d') 
             for i in range(days)]
    
    # Build heatmap matrix
    heatmap = []
    
    for d_idx, date in enumerate(dates):
        row = {'date': date}
        
        for h in horizons:
            X = horizon_data[h]['X']
            if d_idx < len(X):
                # Model agreement on direction at this time
                models_at_time = X[-(days-d_idx), :]
                
                # Compare to previous (signal direction)
                if d_idx > 0:
                    prev = X[-(days-d_idx+1), :]
                    directions = np.sign(models_at_time - prev)
                    agreement = np.mean(directions)  # -1 to +1
                else:
                    agreement = 0
                
                row[f'h{h}'] = round(agreement, 3)
                row[f'h{h}_strength'] = round(abs(agreement), 3)
            else:
                row[f'h{h}'] = None
                row[f'h{h}_strength'] = None
        
        heatmap.append(row)
    
    return {
        'dates': dates,
        'horizons': [f'D+{h}' for h in horizons],
        'data': heatmap
    }


def generate_ensemble_breakdown(horizon_data: Dict) -> Dict:
    """
    Generate ensemble contribution breakdown for visualization.
    
    Shows how different model groups contribute to the final signal.
    """
    # Analyze horizon 5 (middle of range)
    h = 5 if 5 in horizon_data else min(horizon_data.keys())
    X = horizon_data[h]['X']
    y = horizon_data[h]['y']
    
    n_models = X.shape[1]
    
    # Group models by performance
    # Calculate rolling accuracy for each model
    accuracies = []
    for m in range(n_models):
        pred_dir = np.sign(np.diff(X[-100:, m]))
        actual_dir = np.sign(np.diff(y[-100:]))
        acc = (pred_dir == actual_dir).mean()
        accuracies.append(acc)
    
    accuracies = np.array(accuracies)
    
    # Create performance tiers
    p25, p50, p75 = np.percentile(accuracies, [25, 50, 75])
    
    tiers = {
        'elite': (accuracies >= p75).sum(),
        'above_avg': ((accuracies >= p50) & (accuracies < p75)).sum(),
        'below_avg': ((accuracies >= p25) & (accuracies < p50)).sum(),
        'weak': (accuracies < p25).sum(),
    }
    
    # Current signal by tier
    latest = X[-1, :]
    prev = X[-2, :]
    directions = np.sign(latest - prev)
    
    tier_signals = {}
    tier_masks = {
        'elite': accuracies >= p75,
        'above_avg': (accuracies >= p50) & (accuracies < p75),
        'below_avg': (accuracies >= p25) & (accuracies < p50),
        'weak': accuracies < p25,
    }
    
    for tier, mask in tier_masks.items():
        tier_dirs = directions[mask]
        if len(tier_dirs) > 0:
            bullish = (tier_dirs > 0).mean()
            bearish = (tier_dirs < 0).mean()
            tier_signals[tier] = {
                'bullish': round(bullish * 100, 1),
                'bearish': round(bearish * 100, 1),
                'net': round((bullish - bearish) * 100, 1),
            }
    
    return {
        'totalModels': n_models,
        'tierCounts': tiers,
        'tierSignals': tier_signals,
        'accuracyStats': {
            'min': round(float(accuracies.min()), 3),
            'max': round(float(accuracies.max()), 3),
            'mean': round(float(accuracies.mean()), 3),
            'median': round(float(np.median(accuracies)), 3),
        }
    }


def generate_complete_chart_package(asset_id: int = 1866, 
                                     asset_name: str = "Crude Oil") -> Dict:
    """
    Generate complete chart data package for dashboard.
    
    This is the main entry point - creates everything needed for
    a stunning, Bloomberg-grade visualization.
    """
    print(f"\nGenerating chart package for {asset_name} ({asset_id})...")
    
    # Load data
    horizon_data = load_crude_oil_data()
    if not horizon_data:
        return None
    
    # Get metadata
    ref_h = min(horizon_data.keys())
    n_days = len(horizon_data[ref_h]['y'])
    latest_price = horizon_data[ref_h]['y'][-1]
    
    # Generate all chart components
    print("  - Historical chart data (90 days)...")
    historical = generate_historical_chart(horizon_data, days=90)
    
    print("  - Forecast fan chart...")
    fan = generate_forecast_fan(horizon_data)
    
    print("  - Heatmap data (30 days)...")
    heatmap = generate_heatmap_data(horizon_data, days=30)
    
    print("  - Ensemble breakdown...")
    breakdown = generate_ensemble_breakdown(horizon_data)
    
    # Compute summary metrics
    recent_forecasts = [p['forecast'] for p in historical[-10:]]
    recent_actuals = [p['actual'] for p in historical[-10:]]
    recent_errors = [abs(f - a) for f, a in zip(recent_forecasts, recent_actuals)]
    
    # Current signal from fan chart
    fan_h5 = next((p for p in fan if p['horizon'] == 5), fan[-1])
    
    package = {
        'meta': {
            'assetId': asset_id,
            'assetName': asset_name,
            'generatedAt': datetime.now().isoformat(),
            'dataPoints': n_days,
            'horizonsAvailable': sorted(horizon_data.keys()),
            'latestPrice': round(float(latest_price), 2),
        },
        'summary': {
            'currentSignal': 'BULLISH' if fan_h5['bullishPct'] > 60 else ('BEARISH' if fan_h5['bearishPct'] > 60 else 'NEUTRAL'),
            'signalStrength': round(fan_h5['signalStrength'] * 100, 1),
            'priceTarget5D': fan_h5['forecast'],
            'expectedMove': fan_h5['priceChangePct'],
            'confidence90': [fan_h5['lower90'], fan_h5['upper90']],
            'recentMAE': round(np.mean(recent_errors), 2),
            'totalModels': breakdown['totalModels'],
        },
        'charts': {
            'historical': historical,
            'forecastFan': fan,
            'heatmap': heatmap,
            'ensembleBreakdown': breakdown,
        },
        'chartConfig': {
            'colors': {
                'bullish': '#22c55e',
                'bearish': '#ef4444',
                'neutral': '#f59e0b',
                'confidence90': 'rgba(59, 130, 246, 0.15)',
                'confidence50': 'rgba(59, 130, 246, 0.3)',
                'actual': '#f8fafc',
                'forecast': '#3b82f6',
            },
            'historicalDays': 90,
            'forecastDays': 10,
        }
    }
    
    return package


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_chart_package(package: Dict, output_dir: str = "results"):
    """Save chart package to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    asset_id = package['meta']['assetId']
    asset_name = package['meta']['assetName'].lower().replace(' ', '_')
    
    # Full package
    full_file = output_path / f"chart_package_{asset_name}.json"
    with open(full_file, 'w') as f:
        json.dump(package, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved full package: {full_file}")
    
    # Individual chart files (for lazy loading)
    for chart_name, chart_data in package['charts'].items():
        chart_file = output_path / f"chart_{asset_name}_{chart_name}.json"
        with open(chart_file, 'w') as f:
            json.dump(chart_data, f, indent=2, cls=NumpyEncoder)
        print(f"Saved: {chart_file}")
    
    # Summary for quick dashboard loading
    summary_file = output_path / f"summary_{asset_name}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'meta': package['meta'],
            'summary': package['summary'],
            'chartConfig': package['chartConfig'],
        }, f, indent=2, cls=NumpyEncoder)
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS CHART DATA GENERATOR")
    print("=" * 60)
    
    # Generate for Crude Oil
    package = generate_complete_chart_package(asset_id=1866, asset_name="Crude Oil")
    
    if package:
        save_chart_package(package)
        
        # Print summary
        print("\n" + "=" * 60)
        print("CHART PACKAGE SUMMARY")
        print("=" * 60)
        print(f"\nAsset: {package['meta']['assetName']}")
        print(f"Latest Price: ${package['meta']['latestPrice']}")
        print(f"\nCurrent Signal: {package['summary']['currentSignal']}")
        print(f"Signal Strength: {package['summary']['signalStrength']}%")
        print(f"5-Day Target: ${package['summary']['priceTarget5D']}")
        print(f"Expected Move: {package['summary']['expectedMove']:+.1f}%")
        print(f"90% Confidence: ${package['summary']['confidence90'][0]} - ${package['summary']['confidence90'][1]}")
        print(f"\nTotal Models: {package['summary']['totalModels']}")
        print(f"Recent MAE: ${package['summary']['recentMAE']}")
        
        # Show sample forecast fan
        print("\n" + "-" * 40)
        print("FORECAST FAN (Next 10 Days)")
        print("-" * 40)
        for p in package['charts']['forecastFan']:
            if p['isActual']:
                print(f"TODAY: ${p['forecast']:.2f}")
            else:
                signal = "^" if p['bullishPct'] > 50 else "v" if p['bearishPct'] > 50 else "-"
                print(f"D+{p['horizon']:2d}: ${p['forecast']:.2f} ({p['priceChangePct']:+.1f}%) "
                      f"[{p['lower90']:.2f} - {p['upper90']:.2f}] "
                      f"{signal} {p['signalStrength']*100:.0f}% agreement")
    
    print("\n" + "=" * 60)
    print("CHART DATA GENERATION COMPLETE!")
    print("=" * 60)
