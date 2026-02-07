"""
QDTNexus API v2 - Real Data from QDT Data Lake
===============================================
REST API serving real market data from QDL instead of mock/simulated data.

Endpoints:
- GET /api/v2/assets - All assets with current prices
- GET /api/v2/assets/{asset_id} - Single asset details
- GET /api/v2/signals/{asset_id} - Trading signals for all horizons
- GET /api/v2/chart/{asset_id} - OHLCV chart data
- GET /api/v2/refresh/{asset_id} - Refresh data from QDL API

Usage:
    # Import and register with main app
    from api_v2 import register_v2_routes
    register_v2_routes(app)

    # Or run standalone on port 5002
    python api_v2.py
"""

from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
from dataclasses import asdict
import logging

from qdl_data_service import get_service

log = logging.getLogger(__name__)

# Create blueprint for v2 routes
v2_bp = Blueprint('api_v2', __name__, url_prefix='/api/v2')


# ============================================================================
# V2 ENDPOINTS
# ============================================================================

@v2_bp.route('/health', methods=['GET'])
def health():
    """Health check for v2 API."""
    return jsonify({
        "status": "healthy",
        "version": "2.0.0",
        "data_source": "QDT Data Lake"
    })


@v2_bp.route('/assets', methods=['GET'])
def get_assets():
    """Get all assets with current prices."""
    try:
        service = get_service()
        assets = service.get_all_assets()

        return jsonify({
            "success": True,
            "count": len(assets),
            "assets": [asdict(a) for a in assets]
        })
    except Exception as e:
        log.error(f"Error fetching assets: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@v2_bp.route('/assets/<asset_id>', methods=['GET'])
def get_asset(asset_id: str):
    """Get single asset details."""
    try:
        service = get_service()
        asset = service.get_asset_data(asset_id)

        if not asset:
            return jsonify({
                "success": False,
                "error": f"Asset not found: {asset_id}"
            }), 404

        return jsonify({
            "success": True,
            "asset": asdict(asset)
        })
    except Exception as e:
        log.error(f"Error fetching asset {asset_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@v2_bp.route('/signals/<asset_id>', methods=['GET'])
def get_signals(asset_id: str):
    """Get trading signals for all horizons."""
    try:
        service = get_service()
        signals = service.get_signals(asset_id)

        if not signals:
            return jsonify({
                "success": False,
                "error": f"No signals available for: {asset_id}"
            }), 404

        return jsonify({
            "success": True,
            "assetId": asset_id,
            "signals": {k: asdict(v) for k, v in signals.items()}
        })
    except Exception as e:
        log.error(f"Error fetching signals for {asset_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@v2_bp.route('/chart/<asset_id>', methods=['GET'])
def get_chart(asset_id: str):
    """Get OHLCV chart data."""
    try:
        days = request.args.get('days', 365, type=int)
        days = min(days, 1825)  # Max 5 years

        service = get_service()
        chart_data = service.get_chart_data(asset_id, days=days)

        if not chart_data:
            return jsonify({
                "success": False,
                "error": f"No chart data for: {asset_id}"
            }), 404

        return jsonify({
            "success": True,
            "assetId": asset_id,
            "days": len(chart_data),
            "data": [asdict(p) for p in chart_data]
        })
    except Exception as e:
        log.error(f"Error fetching chart for {asset_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@v2_bp.route('/refresh/<asset_id>', methods=['POST'])
def refresh_asset(asset_id: str):
    """Refresh asset data from QDL API."""
    try:
        days = request.args.get('days', 30, type=int)

        service = get_service()
        success = service.refresh_from_qdl(asset_id, days=days)

        if success:
            return jsonify({
                "success": True,
                "message": f"Refreshed {asset_id} with {days} days of data"
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Failed to refresh {asset_id}"
            }), 500
    except Exception as e:
        log.error(f"Error refreshing {asset_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@v2_bp.route('/summary', methods=['GET'])
def get_summary():
    """Get market summary with all assets and top signals."""
    try:
        service = get_service()
        assets = service.get_all_assets()

        # Get signals for each asset
        all_signals = []
        for asset in assets:
            signals = service.get_signals(asset.id)
            if signals and "D+5" in signals:
                sig = signals["D+5"]
                all_signals.append({
                    "assetId": asset.id,
                    "assetName": asset.name,
                    "price": asset.currentPrice,
                    "change24h": asset.changePercent24h,
                    "direction": sig.direction,
                    "confidence": sig.confidence,
                    "sharpe": sig.sharpeRatio
                })

        # Sort by confidence
        all_signals.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            "success": True,
            "assetCount": len(assets),
            "topSignals": all_signals[:5],
            "allSignals": all_signals
        })
    except Exception as e:
        log.error(f"Error generating summary: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# REGISTRATION HELPER
# ============================================================================

def register_v2_routes(app):
    """Register v2 blueprint with an existing Flask app."""
    app.register_blueprint(v2_bp)
    log.info("Registered API v2 routes")


# ============================================================================
# STANDALONE SERVER
# ============================================================================

if __name__ == '__main__':
    import os

    logging.basicConfig(level=logging.INFO)

    app = Flask(__name__)
    CORS(app)
    register_v2_routes(app)

    port = int(os.environ.get('PORT', 5002))

    print("=" * 60)
    print("  QDTNexus API v2 - Real Data Server")
    print("=" * 60)
    print(f"  Running on http://localhost:{port}")
    print()
    print("  Endpoints:")
    print("    GET /api/v2/health - Health check")
    print("    GET /api/v2/assets - All assets with prices")
    print("    GET /api/v2/assets/<id> - Single asset")
    print("    GET /api/v2/signals/<id> - Trading signals")
    print("    GET /api/v2/chart/<id>?days=365 - OHLCV data")
    print("    GET /api/v2/summary - Market summary")
    print("    POST /api/v2/refresh/<id>?days=30 - Refresh from QDL")
    print("=" * 60)

    app.run(host='0.0.0.0', port=port, debug=True)
