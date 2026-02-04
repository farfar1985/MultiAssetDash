"""
QDTNexus API Server
===================
REST API for accessing QDT Ensemble data with API key authentication.

Endpoints:
- GET /api/v1/ohlcv/{asset} - Historical OHLCV data
- GET /api/v1/signals/{asset} - Historical signals (snake chart data)
- GET /api/v1/forecast/{asset} - Live forecast
- GET /api/v1/confidence/{asset} - Historical confidence values
- GET /api/v1/equity/{asset} - Equity curve data
- GET /api/v1/metrics/{asset} - Quant details/performance metrics
- GET /api/v1/assets - List assets user has access to

Authentication:
- API key via header: X-API-Key: your_api_key_here
- Or query parameter: ?api_key=your_api_key_here

Usage:
    python api_server.py
    # Server runs on http://localhost:5000 by default
    # Or set PORT environment variable
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
import pandas as pd
from datetime import datetime
from functools import wraps
import sys
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add sandbox root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import data loading functions from existing modules
from build_qdt_dashboard import (
    load_price_data, 
    load_forecast_data,
    load_live_forecast,
    calculate_signals,
    ASSETS as DASHBOARD_ASSETS
)
from precalculate_metrics import (
    load_forecast_data as load_forecast_for_metrics,
    calculate_signals_pairwise_slopes,
    calculate_trading_performance,
    calculate_metrics,
    ASSETS as METRICS_ASSETS
)

app = Flask(__name__)

# CORS Configuration - explicit allowed origins
# Set CORS_ALLOWED_ORIGINS env var as comma-separated list, or use defaults
CORS_ALLOWED_ORIGINS = os.environ.get('CORS_ALLOWED_ORIGINS', '').strip()
if CORS_ALLOWED_ORIGINS:
    allowed_origins = [origin.strip() for origin in CORS_ALLOWED_ORIGINS.split(',')]
else:
    # Default to localhost origins for development
    allowed_origins = [
        'http://localhost:3000',
        'http://localhost:5000',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:5000',
    ]

CORS(app, origins=allowed_origins, supports_credentials=True)


# Security Headers Middleware
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    # Content Security Policy - restrict resource loading
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"

    # HTTP Strict Transport Security - force HTTPS (1 year)
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'

    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'

    # XSS Protection (legacy browsers)
    response.headers['X-XSS-Protection'] = '1; mode=block'

    # Referrer Policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

    # Permissions Policy - disable unnecessary features
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'

    # Remove server identification
    response.headers.pop('Server', None)

    return response

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
CONFIGS_DIR = os.path.join(SCRIPT_DIR, 'configs')
API_KEYS_FILE = os.path.join(SCRIPT_DIR, 'api_keys.json')

# No default API keys - keys must be created via manage_api_keys.py or migrate_api_keys.py
# Never ship hardcoded credentials in source code
DEFAULT_API_KEYS = {}

def load_api_keys():
    """Load API keys from JSON file, create default if doesn't exist."""
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    else:
        # Create default file
        with open(API_KEYS_FILE, 'w') as f:
            json.dump({"api_keys": DEFAULT_API_KEYS}, f, indent=2)
        return {"api_keys": DEFAULT_API_KEYS}

def save_api_keys(keys_data):
    """Save API keys to JSON file."""
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(keys_data, f, indent=2)

def get_api_key():
    """Extract API key from request header or query parameter."""
    # Check header first
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        # Check query parameter
        api_key = request.args.get('api_key')
    return api_key

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = get_api_key()
        if not api_key:
            return jsonify({
                "success": False,
                "error": "API key required. Provide via X-API-Key header or ?api_key= parameter"
            }), 401
        
        keys_data = load_api_keys()
        key_info = keys_data.get("api_keys", {}).get(api_key)
        
        if not key_info:
            return jsonify({
                "success": False,
                "error": "Invalid API key"
            }), 401
        
        # Attach key info to request for use in route
        request.api_key_info = key_info
        return f(*args, **kwargs)
    
    return decorated_function

def check_asset_access(asset_name):
    """Check if API key has access to this asset."""
    if not hasattr(request, 'api_key_info'):
        return False
    
    allowed_assets = request.api_key_info.get('assets', [])
    # If assets list is empty or contains '*', allow all
    if not allowed_assets or '*' in allowed_assets:
        return True
    
    return asset_name in allowed_assets

def get_asset_data_dir(asset_name):
    """Get data directory for an asset."""
    asset_config = DASHBOARD_ASSETS.get(asset_name)
    if not asset_config:
        return None
    
    asset_id = asset_config.get('id', '')
    return os.path.join(DATA_DIR, f'{asset_id}_{asset_name}')

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint (no auth required)."""
    return jsonify({
        "success": True,
        "service": "QDTNexus API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/v1/assets', methods=['GET'])
@require_api_key
def list_assets():
    """List all assets the API key has access to."""
    key_info = request.api_key_info
    allowed_assets = key_info.get('assets', [])
    
    if not allowed_assets or '*' in allowed_assets:
        # Return all assets
        assets_list = list(DASHBOARD_ASSETS.keys())
    else:
        assets_list = allowed_assets
    
    assets_data = []
    for asset_name in assets_list:
        asset_config = DASHBOARD_ASSETS.get(asset_name)
        if asset_config:
            assets_data.append({
                "name": asset_name,
                "id": asset_config.get('id', ''),
                "display_name": asset_config.get('name', asset_name)
            })
    
    return jsonify({
        "success": True,
        "assets": assets_data,
        "count": len(assets_data),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/v1/ohlcv/<asset_name>', methods=['GET'])
@require_api_key
def get_ohlcv(asset_name):
    """Get historical OHLCV data for an asset."""
    if not check_asset_access(asset_name):
        return jsonify({
            "success": False,
            "error": f"Access denied for asset: {asset_name}"
        }), 403
    
    data_dir = get_asset_data_dir(asset_name)
    if not data_dir or not os.path.exists(data_dir):
        return jsonify({
            "success": False,
            "error": f"Asset not found: {asset_name}"
        }), 404
    
    # Get date range from query params
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Load price data
    price_path = os.path.join(data_dir, 'price_history_cache.json')
    if not os.path.exists(price_path):
        return jsonify({
            "success": False,
            "error": "Price data not found"
        }), 404
    
    with open(price_path, 'r') as f:
        price_data = json.load(f)
    
    # Convert to DataFrame for filtering
    df = pd.DataFrame(price_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter by date range if provided
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    # Sort by date
    df = df.sort_values('date')
    
    # Convert back to list of dicts
    ohlcv_data = df.to_dict('records')
    
    # Format dates as strings
    for record in ohlcv_data:
        if isinstance(record['date'], pd.Timestamp):
            record['date'] = record['date'].strftime('%Y-%m-%d')
    
    return jsonify({
        "success": True,
        "asset": asset_name,
        "data": ohlcv_data,
        "count": len(ohlcv_data),
        "start_date": start_date or (ohlcv_data[0]['date'] if ohlcv_data else None),
        "end_date": end_date or (ohlcv_data[-1]['date'] if ohlcv_data else None),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/v1/signals/<asset_name>', methods=['GET'])
@require_api_key
def get_signals(asset_name):
    """Get historical signals (snake chart data) for an asset."""
    if not check_asset_access(asset_name):
        return jsonify({
            "success": False,
            "error": f"Access denied for asset: {asset_name}"
        }), 403
    
    data_dir = get_asset_data_dir(asset_name)
    if not data_dir or not os.path.exists(data_dir):
        return jsonify({
            "success": False,
            "error": f"Asset not found: {asset_name}"
        }), 404
    
    asset_config = DASHBOARD_ASSETS.get(asset_name, {})
    threshold = asset_config.get('threshold', 0.1)
    
    # Load forecast data
    forecast_df, available_horizons = load_forecast_data(data_dir)
    if forecast_df is None:
        return jsonify({
            "success": False,
            "error": "Forecast data not found"
        }), 404
    
    # Calculate signals
    signals_df = calculate_signals(forecast_df, available_horizons, threshold)
    
    # Get date range from query params
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Filter by date range
    if start_date:
        signals_df = signals_df[signals_df.index >= pd.to_datetime(start_date)]
    if end_date:
        signals_df = signals_df[signals_df.index <= pd.to_datetime(end_date)]
    
    # Convert to list
    signals_data = []
    for date, row in signals_df.iterrows():
        signals_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "signal": row.get('signal', 'NEUTRAL'),
            "net_prob": float(row.get('net_prob', 0.0)),
            "strength": float(row.get('strength', 0.0))
        })
    
    return jsonify({
        "success": True,
        "asset": asset_name,
        "data": signals_data,
        "count": len(signals_data),
        "start_date": start_date or (signals_data[0]['date'] if signals_data else None),
        "end_date": end_date or (signals_data[-1]['date'] if signals_data else None),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/v1/forecast/<asset_name>', methods=['GET'])
@require_api_key
def get_forecast(asset_name):
    """Get live forecast for an asset."""
    if not check_asset_access(asset_name):
        return jsonify({
            "success": False,
            "error": f"Access denied for asset: {asset_name}"
        }), 403
    
    data_dir = get_asset_data_dir(asset_name)
    if not data_dir or not os.path.exists(data_dir):
        return jsonify({
            "success": False,
            "error": f"Asset not found: {asset_name}"
        }), 404
    
    # Load live forecast
    live_forecast = load_live_forecast(data_dir)
    if not live_forecast:
        return jsonify({
            "success": False,
            "error": "Live forecast not found"
        }), 404
    
    # Load optimized forecast for additional metadata
    opt_path = os.path.join(data_dir, 'optimized_forecast.json')
    optimized_forecast = {}
    if os.path.exists(opt_path):
        with open(opt_path, 'r') as f:
            optimized_forecast = json.load(f)
    
    return jsonify({
        "success": True,
        "asset": asset_name,
        "forecasts": live_forecast.get('predictions', []),
        "signal": optimized_forecast.get('signal', 'N/A'),
        "confidence": optimized_forecast.get('confidence', 0),
        "viable_horizons": optimized_forecast.get('viable_horizons', []),
        "timestamp": optimized_forecast.get('timestamp', datetime.now().isoformat())
    })

@app.route('/api/v1/confidence/<asset_name>', methods=['GET'])
@require_api_key
def get_confidence(asset_name):
    """Get historical confidence values for an asset."""
    if not check_asset_access(asset_name):
        return jsonify({
            "success": False,
            "error": f"Access denied for asset: {asset_name}"
        }), 403
    
    data_dir = get_asset_data_dir(asset_name)
    if not data_dir or not os.path.exists(data_dir):
        return jsonify({
            "success": False,
            "error": f"Asset not found: {asset_name}"
        }), 404
    
    # Load confidence stats
    conf_path = os.path.join(data_dir, 'confidence_stats.json')
    if not os.path.exists(conf_path):
        return jsonify({
            "success": False,
            "error": "Confidence data not found"
        }), 404
    
    with open(conf_path, 'r') as f:
        confidence_stats = json.load(f)
    
    return jsonify({
        "success": True,
        "asset": asset_name,
        "stats_by_horizon": confidence_stats.get('stats_by_horizon', {}),
        "overall_stats": confidence_stats.get('overall_stats', {}),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/v1/equity/<asset_name>', methods=['GET'])
@require_api_key
def get_equity(asset_name):
    """Get equity curve data for an asset."""
    if not check_asset_access(asset_name):
        return jsonify({
            "success": False,
            "error": f"Access denied for asset: {asset_name}"
        }), 403
    
    data_dir = get_asset_data_dir(asset_name)
    if not data_dir or not os.path.exists(data_dir):
        return jsonify({
            "success": False,
            "error": f"Asset not found: {asset_name}"
        }), 404
    
    # Load config to get optimal horizons
    config_file = os.path.join(CONFIGS_DIR, f'optimal_{asset_name.lower()}.json')
    if not os.path.exists(config_file):
        return jsonify({
            "success": False,
            "error": "Config file not found"
        }), 404
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    optimal_horizons = config.get('viable_horizons', [])
    threshold = config.get('threshold', 0.1)
    
    # Load forecast data and calculate trades
    forecast_df, prices, available_horizons = load_forecast_for_metrics(data_dir)
    if forecast_df is None:
        return jsonify({
            "success": False,
            "error": "Forecast data not found"
        }), 404
    
    # Calculate signals and trades
    signals, net_probs = calculate_signals_pairwise_slopes(
        forecast_df, optimal_horizons, threshold
    )
    trades = calculate_trading_performance(signals, prices)
    
    # Build equity curve
    equity = 100.0
    equity_curve = [{"date": signals.index[0].strftime('%Y-%m-%d'), "equity": equity}]
    
    for trade in trades:
        equity *= (1 + trade['pnl'] / 100)
        equity_curve.append({
            "date": trade['exit_date'].strftime('%Y-%m-%d'),
            "equity": round(equity, 2),
            "trade_pnl": trade['pnl']
        })
    
    return jsonify({
        "success": True,
        "asset": asset_name,
        "equity_curve": equity_curve,
        "final_equity": equity,
        "total_return": round((equity - 100), 2),
        "count": len(equity_curve),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/v1/metrics/<asset_name>', methods=['GET'])
@require_api_key
def get_metrics(asset_name):
    """Get quant details/performance metrics for an asset."""
    if not check_asset_access(asset_name):
        return jsonify({
            "success": False,
            "error": f"Access denied for asset: {asset_name}"
        }), 403
    
    # Load config file
    config_file = os.path.join(CONFIGS_DIR, f'optimal_{asset_name.lower()}.json')
    if not os.path.exists(config_file):
        return jsonify({
            "success": False,
            "error": "Config file not found"
        }), 404
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract metrics
    metrics = {
        "asset": asset_name,
        "optimized_metrics": {
            "total_return": config.get('optimized_total_return', 0),
            "sharpe_ratio": config.get('optimized_sharpe', 0),
            "win_rate": config.get('optimized_win_rate', 0),
            "profit_factor": config.get('optimized_profit_factor', 0),
            "max_drawdown": config.get('optimized_max_drawdown', 0),
            "total_trades": config.get('optimized_total_trades', 0)
        },
        "raw_metrics": {
            "total_return": config.get('raw_total_return', 0),
            "sharpe_ratio": config.get('raw_sharpe', 0),
            "win_rate": config.get('raw_win_rate', 0),
            "profit_factor": config.get('raw_profit_factor', 0),
            "max_drawdown": config.get('raw_max_drawdown', 0),
            "total_trades": config.get('raw_total_trades', 0)
        },
        "configuration": {
            "viable_horizons": config.get('viable_horizons', []),
            "threshold": config.get('threshold', 0.1),
            "avg_accuracy": config.get('avg_accuracy', 0),
            "health_score": config.get('health_score', 0)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify({
        "success": True,
        **metrics
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors - don't leak details."""
    return jsonify({
        "success": False,
        "error": "Bad request"
    }), 400


@app.errorhandler(401)
def unauthorized(error):
    """Handle unauthorized errors."""
    return jsonify({
        "success": False,
        "error": "Unauthorized"
    }), 401


@app.errorhandler(403)
def forbidden(error):
    """Handle forbidden errors."""
    return jsonify({
        "success": False,
        "error": "Forbidden"
    }), 403


@app.errorhandler(404)
def not_found(error):
    """Handle not found errors - don't reveal internal paths."""
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle method not allowed errors."""
    return jsonify({
        "success": False,
        "error": "Method not allowed"
    }), 405


@app.errorhandler(429)
def rate_limited(error):
    """Handle rate limit errors."""
    return jsonify({
        "success": False,
        "error": "Too many requests"
    }), 429


@app.errorhandler(500)
def internal_error(error):
    """Handle internal errors - log details but don't expose to client."""
    # Log the actual error for debugging (server-side only)
    logger.error(f"Internal server error: {error}")
    logger.error(traceback.format_exc())

    # Return generic message to client - never expose stack traces
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


@app.errorhandler(Exception)
def handle_exception(error):
    """Catch-all exception handler - prevent any stack trace leakage."""
    # Log the actual error for debugging (server-side only)
    logger.error(f"Unhandled exception: {type(error).__name__}: {error}")
    logger.error(traceback.format_exc())

    # Return generic message to client - never expose internals
    return jsonify({
        "success": False,
        "error": "An unexpected error occurred"
    }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Load API keys on startup
    load_api_keys()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("="*70)
    print("  QDTNexus API Server")
    print("="*70)
    print(f"  Starting server on {host}:{port}")
    print(f"  API Keys file: {API_KEYS_FILE}")
    print(f"  Data directory: {DATA_DIR}")
    print("="*70)
    print("\n  Endpoints:")
    print("    GET /api/v1/health - Health check")
    print("    GET /api/v1/assets - List accessible assets")
    print("    GET /api/v1/ohlcv/<asset> - Historical OHLCV")
    print("    GET /api/v1/signals/<asset> - Historical signals")
    print("    GET /api/v1/forecast/<asset> - Live forecast")
    print("    GET /api/v1/confidence/<asset> - Confidence stats")
    print("    GET /api/v1/equity/<asset> - Equity curve")
    print("    GET /api/v1/metrics/<asset> - Performance metrics")
    print("\n  Authentication:")
    print("    Header: X-API-Key: your_key")
    print("    Query: ?api_key=your_key")
    print("="*70)
    
    app.run(host=host, port=port, debug=False)

