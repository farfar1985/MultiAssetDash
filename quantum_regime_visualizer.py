"""
QUANTUM REGIME VISUALIZER
=========================
Creates asset-specific visualizations for quantum regime detection.

Features:
1. Regime timeline chart
2. Quantum state probability distribution
3. Phase space diagram
4. Multi-scale coherence view
5. Transition prediction gauge

Created: 2026-02-05
Author: AmiraB
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from per_asset_optimizer import load_asset_data
from quantum_regime_v2 import QuantumRegimeDetectorV2


def generate_regime_data(asset_name: str, asset_dir: str) -> dict:
    """Generate regime detection data for visualization."""
    print(f"Generating regime data for {asset_name}...")
    
    horizons, prices = load_asset_data(asset_dir)
    if prices is None:
        return None
    
    detector = QuantumRegimeDetectorV2()
    detector.train(prices)
    
    results = []
    for t in range(50, len(prices)):
        result = detector.detect(prices, t)
        result['t'] = t
        result['price'] = float(prices[t])
        results.append(result)
    
    return {
        'asset': asset_name,
        'results': results,
        'boundaries': detector.boundaries.boundaries,
        'n_periods': len(results)
    }


def create_asset_dashboard_html(asset_name: str, data: dict) -> str:
    """Create HTML dashboard for asset-specific regime visualization."""
    
    results = data['results']
    boundaries = data['boundaries']
    
    # Prepare data for charts
    timestamps = list(range(len(results)))
    prices = [r['price'] for r in results]
    regimes = [r['regime'] for r in results]
    volatilities = [r['volatility'] for r in results]
    confidences = [r['confidence'] for r in results]
    coherences = [r['scale_coherence'] for r in results]
    order_params = [r['order_parameter'] for r in results]
    susceptibilities = [r['susceptibility'] for r in results]
    
    # Regime colors
    regime_colors = {
        'LOW_VOL': '#00ff88',
        'NORMAL': '#00d4ff',
        'ELEVATED': '#ffaa00',
        'CRISIS': '#ff4444'
    }
    
    # Count regimes
    regime_counts = pd.Series(regimes).value_counts().to_dict()
    
    # Latest state
    latest = results[-1]
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Regime Analysis: {asset_name} | Nexus v2</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid rgba(100, 200, 255, 0.2);
            margin-bottom: 20px;
        }}
        .header h1 {{
            font-size: 2em;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        .card {{
            background: rgba(30, 30, 60, 0.8);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(100, 200, 255, 0.1);
        }}
        .card h2 {{
            color: #00d4ff;
            font-size: 1.1em;
            margin-bottom: 15px;
        }}
        .full-width {{ grid-column: 1 / -1; }}
        
        .regime-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.2em;
        }}
        .regime-LOW_VOL {{ background: rgba(0, 255, 136, 0.2); color: #00ff88; }}
        .regime-NORMAL {{ background: rgba(0, 212, 255, 0.2); color: #00d4ff; }}
        .regime-ELEVATED {{ background: rgba(255, 170, 0, 0.2); color: #ffaa00; }}
        .regime-CRISIS {{ background: rgba(255, 68, 68, 0.2); color: #ff4444; }}
        
        .metrics-row {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 15px;
        }}
        .metric {{
            flex: 1;
            min-width: 120px;
            text-align: center;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #00d4ff;
        }}
        .metric-label {{
            font-size: 0.8em;
            color: #888;
            margin-top: 5px;
        }}
        
        .chart-container {{
            position: relative;
            height: 250px;
        }}
        
        .boundary-list {{
            display: flex;
            gap: 15px;
            margin-top: 10px;
        }}
        .boundary {{
            padding: 10px 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            font-size: 0.9em;
        }}
        
        .regime-bar {{
            display: flex;
            height: 30px;
            border-radius: 8px;
            overflow: hidden;
            margin-top: 15px;
        }}
        .regime-segment {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
            font-weight: bold;
        }}
        
        .phase-indicator {{
            text-align: center;
            padding: 20px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin-top: 15px;
        }}
        .phase-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #7c3aed;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚛️ Quantum Regime Analysis: {asset_name}</h1>
        <p style="color: #888; margin-top: 10px;">Real-time volatility regime detection using quantum simulation</p>
    </div>

    <div class="grid">
        <!-- Current State -->
        <div class="card">
            <h2>📊 Current Regime State</h2>
            <div style="text-align: center; padding: 20px;">
                <span class="regime-badge regime-{latest['regime']}">{latest['regime']}</span>
                <p style="margin-top: 15px; color: #888;">Confidence: {latest['confidence']*100:.1f}%</p>
            </div>
            
            <div class="metrics-row">
                <div class="metric">
                    <div class="metric-value">{latest['volatility']*100:.1f}%</div>
                    <div class="metric-label">Volatility</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{latest['scale_coherence']:.3f}</div>
                    <div class="metric-label">Scale Coherence</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{latest['susceptibility']:.4f}</div>
                    <div class="metric-label">Susceptibility</div>
                </div>
            </div>
        </div>

        <!-- Phase State -->
        <div class="card">
            <h2>🌀 Phase Transition Analysis</h2>
            <div class="phase-indicator">
                <div class="phase-name">{latest['phase']}</div>
                <p style="color: #888; margin-top: 10px;">Order Parameter: {latest['order_parameter']:.3f}</p>
            </div>
            
            <div class="metrics-row">
                <div class="metric">
                    <div class="metric-value" style="color: {'#ff4444' if latest['transition_imminent'] else '#00ff88'}">
                        {'⚠️ YES' if latest['transition_imminent'] else '✓ NO'}
                    </div>
                    <div class="metric-label">Transition Imminent</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{latest['transition_prediction']['probability']*100:.0f}%</div>
                    <div class="metric-label">Transition Prob</div>
                </div>
            </div>
        </div>

        <!-- Learned Boundaries -->
        <div class="card">
            <h2>🎯 Adaptive Boundaries (Learned)</h2>
            <p style="color: #888; font-size: 0.9em;">Thresholds learned from {asset_name}'s historical data</p>
            
            <div class="boundary-list">
                <div class="boundary">
                    <span style="color: #00ff88;">LOW_VOL</span> &lt; {boundaries[0]*100:.1f}%
                </div>
                <div class="boundary">
                    <span style="color: #00d4ff;">NORMAL</span> &lt; {boundaries[1]*100:.1f}%
                </div>
                <div class="boundary">
                    <span style="color: #ffaa00;">ELEVATED</span> &lt; {boundaries[2]*100:.1f}%
                </div>
                <div class="boundary">
                    <span style="color: #ff4444;">CRISIS</span> ≥ {boundaries[2]*100:.1f}%
                </div>
            </div>
        </div>

        <!-- Regime Distribution -->
        <div class="card">
            <h2>📈 Regime Distribution</h2>
            <div class="regime-bar">
                {''.join([f'<div class="regime-segment" style="width: {regime_counts.get(r, 0)/len(regimes)*100}%; background: {regime_colors[r]}40; color: {regime_colors[r]};">{r.replace("_", " ")}<br>{regime_counts.get(r, 0)/len(regimes)*100:.0f}%</div>' for r in ['LOW_VOL', 'NORMAL', 'ELEVATED', 'CRISIS'] if regime_counts.get(r, 0) > 0])}
            </div>
        </div>

        <!-- Volatility Timeline -->
        <div class="card full-width">
            <h2>📉 Volatility & Regime Timeline</h2>
            <div class="chart-container">
                <canvas id="volChart"></canvas>
            </div>
        </div>

        <!-- Order Parameter Chart -->
        <div class="card">
            <h2>🧲 Order Parameter (Market Direction)</h2>
            <div class="chart-container">
                <canvas id="orderChart"></canvas>
            </div>
        </div>

        <!-- Scale Coherence Chart -->
        <div class="card">
            <h2>🔗 Multi-Scale Coherence</h2>
            <div class="chart-container">
                <canvas id="coherenceChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        // Volatility Timeline with regime background
        const volCtx = document.getElementById('volChart').getContext('2d');
        new Chart(volCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(timestamps[-200:])},
                datasets: [{{
                    label: 'Volatility',
                    data: {json.dumps([round(v*100, 2) for v in volatilities[-200:]])},
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#888', maxTicksLimit: 10 }}
                    }}
                }}
            }}
        }});

        // Order Parameter Chart
        const orderCtx = document.getElementById('orderChart').getContext('2d');
        new Chart(orderCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(timestamps[-100:])},
                datasets: [{{
                    label: 'Order Parameter',
                    data: {json.dumps([round(o, 3) for o in order_params[-100:]])},
                    borderColor: '#7c3aed',
                    backgroundColor: 'rgba(124, 58, 237, 0.1)',
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{
                        min: -1, max: 1,
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ display: false }}
                    }}
                }}
            }}
        }});

        // Scale Coherence Chart
        const cohCtx = document.getElementById('coherenceChart').getContext('2d');
        new Chart(cohCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(timestamps[-100:])},
                datasets: [{{
                    label: 'Coherence',
                    data: {json.dumps([round(c, 3) for c in coherences[-100:]])},
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ display: false }}
                    }}
                }}
            }}
        }});
    </script>

    <div style="text-align: center; padding: 30px; color: #666; font-size: 0.9em;">
        <p>Nexus v2 Quantum Regime Detection | QDT | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
</body>
</html>'''
    
    return html


def generate_all_visualizations():
    """Generate regime visualizations for all assets."""
    dirs = {
        'SP500': 'data/1625_SP500',
        'Bitcoin': 'data/1860_Bitcoin',
        'Crude_Oil': 'data/1866_Crude_Oil'
    }
    
    print("=" * 60)
    print("GENERATING QUANTUM REGIME VISUALIZATIONS")
    print("=" * 60)
    
    for asset_name, asset_dir in dirs.items():
        print(f"\nProcessing {asset_name}...")
        
        data = generate_regime_data(asset_name, asset_dir)
        if data is None:
            continue
        
        html = create_asset_dashboard_html(asset_name, data)
        
        output_path = Path(f'quantum_regime_{asset_name.lower()}.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"  Saved: {output_path}")
    
    print("\n" + "=" * 60)
    print("Visualizations complete!")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_visualizations()
