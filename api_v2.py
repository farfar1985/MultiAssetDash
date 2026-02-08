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
from datetime import datetime
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
# REGIME ENDPOINT
# ============================================================================

@v2_bp.route('/regime', methods=['GET'])
def get_regime():
    """Get current market regime analysis."""
    try:
        from quantum_regime_enhanced import get_current_regime
        result = get_current_regime()
        return jsonify(result)
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Regime analysis module not available"
        }), 500
    except Exception as e:
        log.error(f"Regime analysis error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/regime/accurate', methods=['GET'])
def get_regime_accurate():
    """Get high-accuracy regime analysis with COT and yield curve data."""
    try:
        from quantum_regime_accurate import get_accurate_regime
        result = get_accurate_regime()
        return jsonify(result)
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Accurate regime module not available"
        }), 500
    except Exception as e:
        log.error(f"Accurate regime error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# COT / SMART MONEY SIGNALS
# ============================================================================

@v2_bp.route('/energy', methods=['GET'])
def get_energy_signals():
    """Get crude oil energy market signals."""
    try:
        from energy_signals import get_energy_signals_for_api
        result = get_energy_signals_for_api()
        return jsonify(result)
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Energy signals module not available"
        }), 500
    except Exception as e:
        log.error(f"Energy signals error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/crypto', methods=['GET'])
def get_crypto_signals():
    """Get Bitcoin on-chain signals (MVRV, NVT, etc.)."""
    try:
        from crypto_onchain_signals import get_crypto_signals_for_api
        result = get_crypto_signals_for_api()
        return jsonify(result)
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Crypto signals module not available"
        }), 500
    except Exception as e:
        log.error(f"Crypto signals error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/vix', methods=['GET'])
def get_vix_analysis():
    """Get comprehensive VIX intelligence analysis."""
    try:
        from vix_intelligence import get_vix_for_api
        result = get_vix_for_api()
        return jsonify(result)
    except ImportError:
        return jsonify({
            "success": False,
            "error": "VIX intelligence module not available"
        }), 500
    except Exception as e:
        log.error(f"VIX analysis error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/cot', methods=['GET'])
def get_cot_signals():
    """Get Commitment of Traders (COT) positioning signals."""
    try:
        from smart_money_signals import get_cot_signals_for_api
        result = get_cot_signals_for_api()
        return jsonify({
            "success": True,
            **result
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Smart money signals module not available"
        }), 500
    except Exception as e:
        log.error(f"COT signals error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/position', methods=['POST'])
def get_position_sizing():
    """Compute optimal position size for a trade."""
    try:
        from position_sizing import get_position_for_api
        
        data = request.get_json() or {}
        
        # Required fields
        asset = data.get('asset', 'crude-oil')
        direction = data.get('direction', 'long')
        confidence = float(data.get('confidence', 70))
        win_rate = float(data.get('winRate', 65))
        expected_move = float(data.get('expectedMove', 2.0))
        portfolio_value = float(data.get('portfolioValue', 100000))
        persona = data.get('persona', 'retail')
        
        result = get_position_for_api(
            asset=asset,
            direction=direction,
            confidence=confidence,
            win_rate=win_rate,
            expected_move=expected_move,
            portfolio_value=portfolio_value,
            persona=persona,
        )
        
        return jsonify({
            "success": True,
            **result
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Position sizing module not available"
        }), 500
    except Exception as e:
        log.error(f"Position sizing error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/cot/<asset>', methods=['GET'])
def get_cot_signal_for_asset(asset: str):
    """Get COT signal for a specific asset (GC, CL, ES, etc.)."""
    try:
        from smart_money_signals import generate_cot_signal
        signal = generate_cot_signal(asset)
        
        if not signal:
            return jsonify({
                "success": False,
                "error": f"No COT data available for: {asset}"
            }), 404
        
        return jsonify({
            "success": True,
            "asset": asset,
            "signal": signal.signal.value,
            "confidence": signal.confidence,
            "z_score": signal.z_score,
            "percentile": signal.percentile,
            "current_net": signal.current_net,
            "message": signal.message,
            "reasoning": signal.reasoning,
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Smart money signals module not available"
        }), 500
    except Exception as e:
        log.error(f"COT signal error for {asset}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# CONFLUENCE & ANALOG ENDPOINTS
# ============================================================================

@v2_bp.route('/confluence/<asset>', methods=['GET'])
def get_confluence(asset: str):
    """Get signal confluence analysis for an asset."""
    try:
        from signal_confluence import SignalConfluenceEngine
        
        engine = SignalConfluenceEngine()
        result = engine.analyze(asset)
        
        return jsonify({
            "success": True,
            **result.to_dict()
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Signal confluence module not available"
        }), 500
    except Exception as e:
        log.error(f"Confluence error for {asset}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/analogs/<asset>', methods=['GET'])
def get_analogs(asset: str):
    """Get historical analog analysis for an asset."""
    try:
        from historical_analogs import HistoricalAnalogFinder
        
        lookback = request.args.get('years', 5, type=int)
        n_analogs = request.args.get('n', 10, type=int)
        min_similarity = request.args.get('min_sim', 70.0, type=float)
        
        finder = HistoricalAnalogFinder()
        result = finder.find_analogs(
            asset=asset,
            lookback_years=lookback,
            n_analogs=n_analogs,
            min_similarity=min_similarity
        )
        
        return jsonify({
            "success": True,
            **result.to_dict()
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Historical analogs module not available"
        }), 500
    except Exception as e:
        log.error(f"Analog error for {asset}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/market-pulse', methods=['GET'])
def get_market_pulse():
    """Get quick market pulse overview for all assets."""
    try:
        from signal_confluence import SignalConfluenceEngine
        
        engine = SignalConfluenceEngine()
        assets = ["SP500", "NASDAQ", "GOLD", "CRUDE", "BITCOIN"]
        
        pulse = []
        for asset in assets:
            try:
                result = engine.analyze(asset)
                pulse.append({
                    "asset": asset,
                    "conviction_score": result.conviction_score,
                    "conviction_label": result.conviction_label,
                    "confidence": result.confidence,
                    "bullish": result.bullish_count,
                    "bearish": result.bearish_count,
                    "headline": result.headline,
                    "top_driver": result.key_drivers[0] if result.key_drivers else None
                })
            except Exception as e:
                pulse.append({
                    "asset": asset,
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "assets": pulse
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Signal confluence module not available"
        }), 500
    except Exception as e:
        log.error(f"Market pulse error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# YIELD CURVE, CORRELATION, HHT ENDPOINTS
# ============================================================================

@v2_bp.route('/yield-curve', methods=['GET'])
def get_yield_curve():
    """Get yield curve analysis and recession probability."""
    try:
        from yield_curve_signals import get_yield_curve_for_api
        return jsonify({
            "success": True,
            **get_yield_curve_for_api()
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Yield curve module not available"
        }), 500
    except Exception as e:
        log.error(f"Yield curve error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/correlations', methods=['GET'])
def get_correlations():
    """Get cross-asset correlation regime analysis."""
    try:
        from correlation_regime import get_correlation_for_api
        return jsonify({
            "success": True,
            **get_correlation_for_api()
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Correlation module not available"
        }), 500
    except Exception as e:
        log.error(f"Correlation error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/hht/<asset>', methods=['GET'])
def get_hht_regime(asset: str):
    """Get HHT-based regime analysis for an asset."""
    try:
        from hht_regime_detector import get_hht_for_api
        return jsonify({
            "success": True,
            **get_hht_for_api(asset)
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "HHT regime module not available"
        }), 500
    except Exception as e:
        log.error(f"HHT error for {asset}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/regime-summary', methods=['GET'])
def get_regime_summary():
    """Get comprehensive regime summary across all assets."""
    try:
        from hht_regime_detector import analyze_all_assets as hht_all
        from yield_curve_signals import generate_signal as yc_signal
        from correlation_regime import analyze_correlations as corr_analysis
        
        # HHT regimes
        hht_results = hht_all()
        
        # Yield curve
        yc = yc_signal()
        yc_data = yc.to_dict() if yc else None
        
        # Correlations
        corr = corr_analysis()
        corr_data = corr.to_dict() if corr else None
        
        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "hht_regimes": hht_results,
            "yield_curve": yc_data,
            "correlations": corr_data
        })
    except Exception as e:
        log.error(f"Regime summary error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# CREDIT SPREAD ENDPOINTS
# ============================================================================

@v2_bp.route('/credit-spreads', methods=['GET'])
def get_credit_spreads():
    """Get credit spread analysis for risk sentiment."""
    try:
        from credit_spread_signals import get_credit_for_api
        return jsonify({
            "success": True,
            **get_credit_for_api()
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Credit spread module not available"
        }), 500
    except Exception as e:
        log.error(f"Credit spread error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# AI SUMMARY ENDPOINT
# ============================================================================

@v2_bp.route('/ai-summary', methods=['GET'])
def get_ai_summary():
    """Get AI-generated market summary."""
    try:
        from ai_market_summary import get_summary_for_api
        return jsonify({
            "success": True,
            **get_summary_for_api()
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "AI summary module not available"
        }), 500
    except Exception as e:
        log.error(f"AI summary error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# HEDGE CALCULATOR ENDPOINTS
# ============================================================================

@v2_bp.route('/hedge/<asset>', methods=['GET'])
def get_hedge_recommendation(asset: str):
    """Get optimal hedge recommendation for an asset."""
    try:
        from hedge_calculator import get_hedge_for_api
        
        position = request.args.get('position', 'LONG').upper()
        notional = request.args.get('notional', 1000000, type=float)
        
        return jsonify({
            "success": True,
            **get_hedge_for_api(asset, position, notional)
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Hedge calculator module not available"
        }), 500
    except Exception as e:
        log.error(f"Hedge calculation error for {asset}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/hedge-effectiveness/<asset>', methods=['GET'])
def get_hedge_effectiveness(asset: str):
    """Get hedge effectiveness report for an asset."""
    try:
        from hedge_calculator import get_effectiveness_for_api
        
        days = request.args.get('days', 30, type=int)
        
        return jsonify({
            "success": True,
            **get_effectiveness_for_api(asset, days)
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Hedge calculator module not available"
        }), 500
    except Exception as e:
        log.error(f"Hedge effectiveness error for {asset}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# FACTOR ATTRIBUTION ENDPOINTS
# ============================================================================

@v2_bp.route('/factor-attribution/<asset>', methods=['GET'])
def get_factor_attribution(asset: str):
    """Get factor attribution analysis for an asset."""
    try:
        from factor_attribution import get_attribution_for_api
        
        days = request.args.get('days', 252, type=int)
        
        return jsonify({
            "success": True,
            **get_attribution_for_api(asset, days)
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Factor attribution module not available"
        }), 500
    except Exception as e:
        log.error(f"Factor attribution error for {asset}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# CROWDED TRADE DETECTION ENDPOINTS
# ============================================================================

@v2_bp.route('/crowding/<asset>', methods=['GET'])
def get_crowding(asset: str):
    """Get crowded trade analysis for an asset."""
    try:
        from crowded_trade_detector import get_crowding_for_api
        return jsonify({
            "success": True,
            **get_crowding_for_api(asset)
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Crowded trade module not available"
        }), 500
    except Exception as e:
        log.error(f"Crowding error for {asset}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@v2_bp.route('/crowding', methods=['GET'])
def get_all_crowding():
    """Get crowded trade analysis for all assets."""
    try:
        from crowded_trade_detector import get_all_crowding_for_api
        return jsonify({
            "success": True,
            **get_all_crowding_for_api()
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Crowded trade module not available"
        }), 500
    except Exception as e:
        log.error(f"All crowding error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# REPORT GENERATION ENDPOINT
# ============================================================================

@v2_bp.route('/report', methods=['POST'])
def generate_report():
    """Generate a market intelligence report."""
    try:
        from pdf_report_generator import get_report_for_api
        
        data = request.get_json() or {}
        client_name = data.get('clientName', 'Valued Client')
        
        return jsonify(get_report_for_api(client_name))
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Report generator module not available"
        }), 500
    except Exception as e:
        log.error(f"Report generation error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
