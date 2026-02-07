"""
Early Warning API Endpoints
===========================
Flask routes for the Early Warning System.

Add to main api_server.py or run standalone on port 5002.

Endpoints:
  GET  /api/v1/early-warning/status      - Overall risk status
  GET  /api/v1/early-warning/alerts      - Active alerts
  GET  /api/v1/early-warning/asset/<id>  - Single asset details
  GET  /api/v1/early-warning/contagion   - Contagion risk by group
  GET  /api/v1/early-warning/dashboard   - Full dashboard data
  POST /api/v1/early-warning/refresh     - Force refresh scan

Author: AmiraB
Date: 2026-02-06
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from early_warning_system import EarlyWarningSystem
import os

app = Flask(__name__)
CORS(app)

# Initialize the early warning system
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ews = EarlyWarningSystem("regime_models")


@app.route('/api/v1/early-warning/status', methods=['GET'])
def get_status():
    """Get overall risk status."""
    stability = ews.get_stability_report()
    overall_risk = ews._calculate_overall_risk(stability)
    
    return jsonify({
        'status': 'ok',
        'overall_risk': overall_risk,
        'summary': stability['summary'],
        'total_assets': stability['total_assets'],
        'avg_p_change': stability['avg_p_change'],
        'timestamp': stability['timestamp']
    })


@app.route('/api/v1/early-warning/alerts', methods=['GET'])
def get_alerts():
    """Get active alerts."""
    min_level = request.args.get('level', 'watch')
    alerts = ews.get_alerts(min_level)
    
    return jsonify({
        'status': 'ok',
        'alert_count': len(alerts),
        'min_level': min_level,
        'alerts': alerts
    })


@app.route('/api/v1/early-warning/asset/<asset_id>', methods=['GET'])
def get_asset(asset_id):
    """Get transition probability for a single asset."""
    result = ews.get_transition_probability(asset_id)
    
    if 'error' in result:
        return jsonify({'status': 'error', 'message': result['error']}), 404
    
    return jsonify({
        'status': 'ok',
        'data': result
    })


@app.route('/api/v1/early-warning/contagion', methods=['GET'])
def get_contagion():
    """Get contagion risk by asset group."""
    contagion = ews.detect_contagion_risk()
    
    return jsonify({
        'status': 'ok',
        'data': contagion
    })


@app.route('/api/v1/early-warning/dashboard', methods=['GET'])
def get_dashboard():
    """Get full dashboard data."""
    dashboard = ews.generate_dashboard_data()
    
    return jsonify({
        'status': 'ok',
        'data': dashboard
    })


@app.route('/api/v1/early-warning/refresh', methods=['POST'])
def refresh():
    """Force refresh all models and rescan."""
    global ews
    ews = EarlyWarningSystem("regime_models")
    dashboard = ews.generate_dashboard_data()
    
    return jsonify({
        'status': 'ok',
        'message': 'Early warning system refreshed',
        'overall_risk': dashboard['overall_risk']
    })


@app.route('/api/v1/early-warning/assets', methods=['GET'])
def get_all_assets():
    """Get transition probabilities for all assets."""
    all_probs = ews.scan_all_assets()
    
    return jsonify({
        'status': 'ok',
        'asset_count': len(all_probs),
        'assets': all_probs
    })


if __name__ == '__main__':
    print("Early Warning API starting on port 5002...")
    print("Endpoints:")
    print("  GET  /api/v1/early-warning/status")
    print("  GET  /api/v1/early-warning/alerts")
    print("  GET  /api/v1/early-warning/asset/<id>")
    print("  GET  /api/v1/early-warning/contagion")
    print("  GET  /api/v1/early-warning/dashboard")
    print("  POST /api/v1/early-warning/refresh")
    app.run(host='0.0.0.0', port=5002, debug=False)
