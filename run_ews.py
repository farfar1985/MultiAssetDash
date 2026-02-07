"""Quick runner for Early Warning System."""
import sys
import json
sys.stdout.reconfigure(encoding='utf-8')

from early_warning_system import EarlyWarningSystem

ews = EarlyWarningSystem('regime_models')

# Get stability report
stability = ews.get_stability_report()
print("=" * 60)
print("EARLY WARNING SYSTEM - Regime Transition Monitor")
print("=" * 60)
print()
print("STABILITY REPORT")
print("-" * 40)
print(f"Total assets monitored: {stability['total_assets']}")
print(f"Average P(change): {stability['avg_p_change']:.1%}")
print()
print(f"  Stable (<10%):     {stability['summary']['stable']}")
print(f"  Moderate (10-30%): {stability['summary']['moderate']}")
print(f"  Elevated (30-50%): {stability['summary']['elevated']}")
print(f"  High Risk (>50%):  {stability['summary']['high_risk']}")

# Get alerts
print()
print("ACTIVE ALERTS")
print("-" * 40)
alerts = ews.get_alerts('watch')
if alerts:
    for alert in alerts:
        level = alert['alert_level'].upper()
        print(f"[{level}] {alert['asset']}")
        print(f"   Current: {alert['current_regime']} (confidence: {alert['confidence']:.1%})")
        print(f"   P(change): {alert['p_change']:.1%} -> likely {alert['most_likely_next']}")
        print(f"   Duration: {alert['regime_duration_days']} days")
        print()
else:
    print("No alerts - all assets stable")

# Contagion risk
print()
print("CONTAGION RISK BY GROUP")
print("-" * 40)
contagion = ews.detect_contagion_risk()
for group_name, data in contagion['groups'].items():
    print(f"[{data['contagion_risk']}] {group_name}")
    print(f"   Avg P(change): {data['avg_p_change']:.1%}")
    print(f"   Elevated: {data['elevated_count']}/{data['total']}")

# Save dashboard data
print()
print("=" * 60)
dashboard = ews.generate_dashboard_data()
print(f"Overall Risk: {dashboard['overall_risk']}")

# Save to JSON
with open('regime_models/early_warning_latest.json', 'w') as f:
    json.dump(dashboard, f, indent=2)
print("Saved to regime_models/early_warning_latest.json")
