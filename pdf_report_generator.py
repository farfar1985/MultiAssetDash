"""
PDF Report Generator — Wealth Manager Client Reports
=====================================================
Generates professional PDF reports from market intelligence.
Designed for wealth managers to share with clients.

Features:
- Executive summary
- Market overview with visuals
- Asset-by-asset breakdown
- Risk analysis
- Recommended actions

Author: AmiraB
Created: 2026-02-08
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

# Try to import PDF libraries
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.platypus import Image as RLImage
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Import our signal modules
try:
    from signal_confluence import SignalConfluenceEngine
except ImportError:
    SignalConfluenceEngine = None

try:
    from ai_market_summary import generate_market_summary
except ImportError:
    generate_market_summary = None

try:
    from yield_curve_signals import generate_signal as yc_signal
except ImportError:
    yc_signal = None

try:
    from correlation_regime import analyze_correlations
except ImportError:
    analyze_correlations = None


OUTPUT_DIR = Path("reports")


@dataclass
class ReportConfig:
    """Report configuration"""
    client_name: str = "Valued Client"
    advisor_name: str = "QDT Advisory"
    include_technicals: bool = True
    include_macro: bool = True
    include_recommendations: bool = True
    theme: str = "professional"  # professional, modern, minimal


def generate_html_report(config: Optional[ReportConfig] = None) -> str:
    """Generate HTML report (works without reportlab)"""
    
    if config is None:
        config = ReportConfig()
    
    # Gather data
    asset_data = {}
    if SignalConfluenceEngine:
        engine = SignalConfluenceEngine()
        for asset in ["SP500", "GOLD", "CRUDE", "BITCOIN", "NASDAQ"]:
            try:
                result = engine.analyze(asset)
                asset_data[asset] = result
            except Exception:
                pass
    
    # Get market summary
    summary = None
    if generate_market_summary:
        try:
            summary = generate_market_summary()
        except Exception:
            pass
    
    # Get macro data
    yc_data = None
    if yc_signal:
        try:
            yc_data = yc_signal()
        except Exception:
            pass
    
    corr_data = None
    if analyze_correlations:
        try:
            corr_data = analyze_correlations()
        except Exception:
            pass
    
    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Market Intelligence Report - {datetime.now().strftime('%B %d, %Y')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px;
            color: #333;
            background: #fff;
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #1a365d;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #1a365d;
            margin: 0;
            font-size: 28px;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #1a365d;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            font-size: 20px;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #1a365d 0%, #2d4a7c 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 25px;
        }}
        .summary-box h3 {{
            margin-top: 0;
            font-size: 22px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #1a365d;
        }}
        .metric-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #1a365d;
        }}
        .metric-card .label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .asset-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .asset-table th, .asset-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .asset-table th {{
            background: #1a365d;
            color: white;
            font-weight: 500;
        }}
        .asset-table tr:hover {{
            background: #f5f5f5;
        }}
        .signal-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .signal-buy {{ background: #22c55e; color: white; }}
        .signal-sell {{ background: #ef4444; color: white; }}
        .signal-neutral {{ background: #f59e0b; color: white; }}
        .action-list {{
            background: #f0fdf4;
            border-left: 4px solid #22c55e;
            padding: 15px 20px;
            margin: 15px 0;
        }}
        .risk-list {{
            background: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 15px 20px;
            margin: 15px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
        @media print {{
            body {{ padding: 20px; }}
            .summary-box {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Market Intelligence Report</h1>
        <div class="subtitle">
            Prepared for {config.client_name} | {datetime.now().strftime('%B %d, %Y')}
        </div>
    </div>
"""
    
    # Executive Summary
    if summary:
        sentiment_emoji = "📈" if summary.overall_sentiment == "BULLISH" else "📉" if summary.overall_sentiment == "BEARISH" else "➡️"
        html += f"""
    <div class="summary-box">
        <h3>{sentiment_emoji} {summary.main_headline}</h3>
        <p>{summary.sub_headline}</p>
        <div class="metrics-grid" style="margin-top: 20px;">
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: bold;">{summary.overall_sentiment}</div>
                <div style="font-size: 12px; opacity: 0.8;">MARKET SENTIMENT</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: bold;">{summary.risk_level}</div>
                <div style="font-size: 12px; opacity: 0.8;">RISK LEVEL</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: bold;">{summary.confidence:.0f}%</div>
                <div style="font-size: 12px; opacity: 0.8;">CONFIDENCE</div>
            </div>
        </div>
    </div>
"""
    
    # Asset Overview
    html += """
    <div class="section">
        <h2>Asset Overview</h2>
        <table class="asset-table">
            <thead>
                <tr>
                    <th>Asset</th>
                    <th>Conviction</th>
                    <th>Signal</th>
                    <th>Key Driver</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for asset, data in asset_data.items():
        score = data.conviction_score
        label = data.conviction_label
        
        if "BUY" in label:
            badge_class = "signal-buy"
        elif "SELL" in label:
            badge_class = "signal-sell"
        else:
            badge_class = "signal-neutral"
        
        key_driver = data.key_drivers[0] if data.key_drivers else "Mixed signals"
        
        html += f"""
                <tr>
                    <td><strong>{asset}</strong></td>
                    <td>{score:+.0f}</td>
                    <td><span class="signal-badge {badge_class}">{label}</span></td>
                    <td>{key_driver}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
"""
    
    # Macro Environment
    if config.include_macro:
        html += """
    <div class="section">
        <h2>Macro Environment</h2>
        <div class="metrics-grid">
"""
        if yc_data:
            html += f"""
            <div class="metric-card">
                <div class="value">{yc_data.spread_2y10y:.0f}bps</div>
                <div class="label">Yield Curve (2Y-10Y)</div>
            </div>
            <div class="metric-card">
                <div class="value">{yc_data.recession_probability:.0f}%</div>
                <div class="label">Recession Probability</div>
            </div>
"""
        if corr_data:
            html += f"""
            <div class="metric-card">
                <div class="value">{corr_data.diversification_score:.0f}/100</div>
                <div class="label">Diversification Score</div>
            </div>
"""
        html += """
        </div>
    </div>
"""
    
    # Recommendations
    if config.include_recommendations and summary:
        html += """
    <div class="section">
        <h2>Recommended Actions</h2>
"""
        if summary.opportunities:
            html += """
        <div class="action-list">
            <strong>✅ Opportunities</strong>
            <ul>
"""
            for opp in summary.opportunities:
                html += f"                <li>{opp}</li>\n"
            html += """
            </ul>
        </div>
"""
        
        if summary.risks:
            html += """
        <div class="risk-list">
            <strong>⚠️ Risks to Monitor</strong>
            <ul>
"""
            for risk in summary.risks:
                html += f"                <li>{risk}</li>\n"
            html += """
            </ul>
        </div>
"""
        
        html += """
        <h3>Investor Actions</h3>
        <ul>
"""
        for action in summary.investor_actions:
            html += f"            <li>{action}</li>\n"
        html += """
        </ul>
    </div>
"""
    
    # Footer
    html += f"""
    <div class="footer">
        <p>Generated by QDT Intelligence Engine | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p>This report is for informational purposes only and does not constitute investment advice.</p>
    </div>
</body>
</html>
"""
    
    return html


def save_report(config: Optional[ReportConfig] = None, output_path: Optional[str] = None) -> str:
    """Generate and save HTML report"""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"market_report_{timestamp}.html"
    else:
        output_path = Path(output_path)
    
    html = generate_html_report(config)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return str(output_path)


def get_report_for_api(client_name: str = "Valued Client") -> Dict:
    """Generate report and return path"""
    config = ReportConfig(client_name=client_name)
    
    try:
        path = save_report(config)
        return {
            "success": True,
            "path": path,
            "message": f"Report generated: {path}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("PDF/HTML REPORT GENERATOR")
    print("=" * 60)
    
    config = ReportConfig(
        client_name="Demo Client",
        advisor_name="QDT Advisory"
    )
    
    path = save_report(config)
    print(f"\n✓ Report generated: {path}")
    print("Open in browser to view")
