# api_ensemble.py
# Nexus Ensemble API Server
# Implements API contract for frontend integration

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)
CORS(app)

DATA_DIR = r"C:\Users\William Dennis\projects\nexus\data"
RESULTS_DIR = r"C:\Users\William Dennis\projects\nexus\results"

# Asset configuration - updated with per-asset optimized ensembles (2026-02-05)
CONFIGS_DIR = r"C:\Users\William Dennis\projects\nexus\configs"

def load_optimal_config(asset_name: str) -> dict:
    """Load optimal config from JSON if available."""
    config_path = os.path.join(CONFIGS_DIR, f"optimal_{asset_name.lower()}.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

ASSETS = {
    1866: {
        'name': 'Crude_Oil',
        'display_name': 'Crude Oil (WTI)',
        'category': 'Commodities',
        'horizons': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'threshold': 0.0,  # Updated from optimization
        'best_ensemble': {'method': 'pairwise_slopes', 'horizons': [9, 10], 'aggregation': 'mean'}  # Sharpe 1.727, WR 68.5%
    },
    1625: {
        'name': 'SP500',
        'display_name': 'S&P 500',
        'category': 'Indices',
        'horizons': [1, 3, 5, 8, 9],
        'threshold': 0.0,  # Updated from optimization
        'best_ensemble': {'method': 'pairwise_slopes', 'horizons': [3, 8], 'aggregation': 'mean'}  # Sharpe 1.184, WR 57.4%
    },
    1860: {
        'name': 'Bitcoin',
        'display_name': 'Bitcoin (BTC)',
        'category': 'Crypto',
        'horizons': [1, 3, 5, 8, 10],
        'threshold': 0.0,  # Updated from optimization
        'best_ensemble': {'method': 'pairwise_slopes', 'horizons': [3, 5], 'aggregation': 'median'}  # Sharpe 0.359, WR 55.6%
    },
    1861: {
        'name': 'Gold',
        'display_name': 'Gold',
        'category': 'Commodities',
        'horizons': [],
        'threshold': 0.0,
        'best_ensemble': {'method': 'pending', 'horizons': []}
    }
}


def load_horizon_data(asset_id: int, horizon: int):
    """Load predictions and actuals for a specific horizon."""
    asset = ASSETS.get(asset_id)
    if not asset:
        return None, None
    
    folder = f"{asset_id}_{asset['name']}"
    path = os.path.join(DATA_DIR, folder, 'horizons_wide', f'horizon_{horizon}.joblib')
    
    if not os.path.exists(path):
        return None, None
    
    data = joblib.load(path)
    return data['X'], data['y']


def compute_ensemble_signal(asset_id: int):
    """Compute the current ensemble signal for an asset."""
    from ensemble_methods import EnsembleMethods
    
    asset = ASSETS.get(asset_id)
    if not asset or not asset['best_ensemble']['horizons']:
        return None
    
    ensemble = EnsembleMethods(lookback_window=60)
    horizons = asset['best_ensemble']['horizons']
    
    horizon_signals = []
    horizon_forecasts = []
    horizon_data = []
    
    for h in horizons:
        X, y = load_horizon_data(asset_id, h)
        if X is None:
            continue
        
        try:
            weights = ensemble.top_k_by_sharpe(X, y, top_pct=0.1)
            ensemble_pred = (X * weights).sum(axis=1)
            
            # Latest signal
            signal = np.sign(ensemble_pred.diff().iloc[-1])
            forecast = ensemble_pred.iloc[-1]
            
            horizon_signals.append(signal)
            horizon_forecasts.append(forecast)
            horizon_data.append({
                'horizon': h,
                'signal': 'BULLISH' if signal > 0 else ('BEARISH' if signal < 0 else 'NEUTRAL'),
                'forecast': float(forecast),
                'models_used': int((weights > 0.001).sum())
            })
        except Exception as e:
            continue
    
    if not horizon_signals:
        return None
    
    # Magnitude-weighted consensus
    magnitudes = [abs(f - horizon_forecasts[0]) for f in horizon_forecasts]
    if sum(magnitudes) > 0:
        weighted_signal = sum(s * m for s, m in zip(horizon_signals, magnitudes)) / sum(magnitudes)
    else:
        weighted_signal = np.mean(horizon_signals)
    
    direction = 'BULLISH' if weighted_signal > 0.2 else ('BEARISH' if weighted_signal < -0.2 else 'NEUTRAL')
    probability = min(abs(weighted_signal) + 0.5, 1.0)
    
    agreeing = sum(1 for s in horizon_signals if (s > 0) == (weighted_signal > 0))
    
    # Get current price from latest actual
    _, y = load_horizon_data(asset_id, horizons[0])
    current_price = float(y.iloc[-1]) if y is not None else None
    
    return {
        'direction': direction,
        'probability': round(probability, 2),
        'confidence': 'HIGH' if agreeing >= len(horizon_signals) * 0.8 else ('MEDIUM' if agreeing >= len(horizon_signals) * 0.5 else 'LOW'),
        'horizons_agreeing': agreeing,
        'total_horizons': len(horizon_signals),
        'current_price': current_price,
        'avg_forecast': np.mean(horizon_forecasts),
        'horizon_details': horizon_data
    }


# ============ API ROUTES ============

@app.route('/api/v1/assets', methods=['GET'])
def list_assets():
    """List all available assets."""
    assets = []
    for asset_id, info in ASSETS.items():
        # Count models
        models_count = 0
        for h in info['horizons'][:1]:  # Just check first horizon
            X, _ = load_horizon_data(asset_id, h)
            if X is not None:
                models_count = X.shape[1]
                break
        
        assets.append({
            'id': asset_id,
            'name': info['name'],
            'display_name': info['display_name'],
            'category': info['category'],
            'horizons_available': info['horizons'],
            'models_count': models_count,
            'has_data': len(info['horizons']) > 0
        })
    
    return jsonify({'assets': assets})


@app.route('/api/v1/signal/<int:asset_id>', methods=['GET'])
def get_signal(asset_id):
    """Get current trading signal for an asset."""
    if asset_id not in ASSETS:
        return jsonify({'error': True, 'code': 'ASSET_NOT_FOUND', 'message': f'Asset {asset_id} not found'}), 404
    
    asset = ASSETS[asset_id]
    signal_data = compute_ensemble_signal(asset_id)
    
    if not signal_data:
        return jsonify({'error': True, 'code': 'INSUFFICIENT_DATA', 'message': 'Not enough data for signal'}), 400
    
    # Compute targets
    current = signal_data['current_price']
    avg_forecast = signal_data['avg_forecast']
    
    if current and avg_forecast:
        expected_move = avg_forecast - current
        expected_move_pct = (expected_move / current) * 100
        
        forecast = {
            'current_price': round(current, 2),
            'target_1': round(current + expected_move * 0.5, 2),
            'target_2': round(avg_forecast, 2),
            'stop_loss': round(current - abs(expected_move) * 0.5, 2),
            'expected_move_pct': round(expected_move_pct, 2),
            'expected_move_usd': round(expected_move, 2)
        }
    else:
        forecast = None
    
    return jsonify({
        'asset_id': asset_id,
        'asset_name': asset['name'],
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'signal': {
            'direction': signal_data['direction'],
            'probability': signal_data['probability'],
            'confidence': signal_data['confidence'],
            'horizons_agreeing': signal_data['horizons_agreeing'],
            'total_horizons': signal_data['total_horizons']
        },
        'forecast': forecast,
        'ensemble_config': {
            'method': asset['best_ensemble']['method'],
            'horizons_used': asset['best_ensemble']['horizons']
        }
    })


@app.route('/api/v1/horizons/<int:asset_id>', methods=['GET'])
def get_horizons(asset_id):
    """Get per-horizon breakdown."""
    if asset_id not in ASSETS:
        return jsonify({'error': True, 'code': 'ASSET_NOT_FOUND'}), 404
    
    asset = ASSETS[asset_id]
    signal_data = compute_ensemble_signal(asset_id)
    
    if not signal_data:
        return jsonify({'error': True, 'code': 'INSUFFICIENT_DATA'}), 400
    
    return jsonify({
        'asset_id': asset_id,
        'asset_name': asset['name'],
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'horizons': signal_data['horizon_details']
    })


@app.route('/api/v1/ensemble-config/<int:asset_id>', methods=['GET'])
def get_ensemble_config(asset_id):
    """Get ensemble configuration for an asset."""
    if asset_id not in ASSETS:
        return jsonify({'error': True, 'code': 'ASSET_NOT_FOUND'}), 404
    
    asset = ASSETS[asset_id]
    
    return jsonify({
        'asset_id': asset_id,
        'method': asset['best_ensemble']['method'],
        'horizons': {
            'selected': asset['best_ensemble']['horizons'],
            'available': asset['horizons']
        },
        'parameters': {
            'lookback_window': 60,
            'min_forecast_threshold': asset['threshold'],
            'signal_threshold': 0.2
        }
    })


@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'version': '1.0.0'
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  Nexus Ensemble API Server")
    print("  Starting on http://localhost:5001")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5001, debug=True)
