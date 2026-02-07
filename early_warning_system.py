"""
Early Warning System for Regime Transitions
============================================
Detects regime changes BEFORE they happen by monitoring transition probabilities.

Key metrics:
- P(regime_change): Probability of leaving current state in next period
- Trend: Is transition probability rising over last N days?
- Alert levels: 30% (watch), 50% (warning), 70% (critical)

Author: AmiraB
Date: 2026-02-06
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EarlyWarningSystem:
    """Monitor regime transition probabilities and generate alerts."""
    
    ALERT_THRESHOLDS = {
        'watch': 0.30,      # 30% - elevated risk
        'warning': 0.50,    # 50% - high risk  
        'critical': 0.70    # 70% - imminent transition
    }
    
    def __init__(self, models_dir: str = "regime_models"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, object] = {}
        self.summary: Dict = {}
        self.alerts_history: List[Dict] = []
        
        # Load summary and models
        self._load_summary()
        self._load_models()
    
    def _load_summary(self):
        """Load regime summary with current states and transition matrices."""
        summary_path = self.models_dir / "regime_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
            print(f"Loaded summary for {len(self.summary)} assets")
    
    def _load_models(self):
        """Load all HMM models."""
        for asset_key, asset_data in self.summary.items():
            if 'error' in asset_data:
                continue
            model_path = self.models_dir / f"{asset_key}_hmm.joblib"
            if model_path.exists():
                try:
                    self.models[asset_key] = joblib.load(model_path)
                except Exception as e:
                    print(f"Failed to load {asset_key}: {e}")
        print(f"Loaded {len(self.models)} HMM models")
    
    def get_transition_probability(self, asset_key: str) -> Dict:
        """
        Calculate probability of regime change for an asset.
        
        Returns:
            Dict with current_regime, p_change, p_by_regime, alert_level
        """
        if asset_key not in self.summary:
            return {'error': f'Asset {asset_key} not found'}
        
        asset_data = self.summary[asset_key]
        if 'error' in asset_data:
            return {'error': asset_data['error']}
        
        current = asset_data.get('current_regime', {})
        if not current:
            return {'error': 'No current regime data'}
        
        # Get current state probabilities and transition matrix
        probs = current.get('probabilities', [])
        trans_matrix = np.array(current.get('transitionMatrix', []))
        current_regime = current.get('currentRegime', 'unknown')
        confidence = current.get('confidence', 0)
        duration = current.get('regimeDuration', 0)
        
        if len(trans_matrix) == 0 or len(probs) == 0:
            return {'error': 'Missing transition data'}
        
        # Find current state index
        current_state_idx = None
        for i, p in enumerate(probs):
            if p.get('regime') == current_regime:
                current_state_idx = i
                break
        
        if current_state_idx is None:
            # Fallback: use highest probability state
            current_state_idx = np.argmax([p['probability'] for p in probs])
        
        # P(stay in current state) = diagonal of transition matrix
        p_stay = trans_matrix[current_state_idx, current_state_idx]
        p_change = 1 - p_stay
        
        # Get transition probabilities to each other state
        p_to_other = {}
        for i, p in enumerate(probs):
            if i != current_state_idx:
                regime_name = p.get('regime', f'state_{i}')
                p_to_other[regime_name] = float(trans_matrix[current_state_idx, i])
        
        # Determine alert level
        alert_level = 'normal'
        for level, threshold in sorted(self.ALERT_THRESHOLDS.items(), 
                                       key=lambda x: x[1], reverse=True):
            if p_change >= threshold:
                alert_level = level
                break
        
        # Most likely next regime if transition occurs
        if p_to_other:
            most_likely_next = max(p_to_other.items(), key=lambda x: x[1])
        else:
            most_likely_next = (current_regime, 0)
        
        return {
            'asset': asset_key,
            'current_regime': current_regime,
            'confidence': round(confidence, 4),
            'regime_duration_days': duration,
            'p_stay': round(p_stay, 4),
            'p_change': round(p_change, 4),
            'p_to_regimes': {k: round(v, 4) for k, v in p_to_other.items()},
            'most_likely_next': most_likely_next[0],
            'p_most_likely_next': round(most_likely_next[1], 4),
            'alert_level': alert_level,
            'timestamp': datetime.now().isoformat()
        }
    
    def scan_all_assets(self) -> Dict[str, Dict]:
        """Scan all assets and return transition probabilities."""
        results = {}
        for asset_key in self.summary.keys():
            if 'error' not in self.summary[asset_key]:
                results[asset_key] = self.get_transition_probability(asset_key)
        return results
    
    def get_alerts(self, min_level: str = 'watch') -> List[Dict]:
        """
        Get all assets with alerts at or above specified level.
        
        Args:
            min_level: 'watch', 'warning', or 'critical'
        """
        level_order = {'normal': 0, 'watch': 1, 'warning': 2, 'critical': 3}
        min_order = level_order.get(min_level, 1)
        
        all_probs = self.scan_all_assets()
        alerts = []
        
        for asset_key, data in all_probs.items():
            if 'error' in data:
                continue
            alert_level = data.get('alert_level', 'normal')
            if level_order.get(alert_level, 0) >= min_order:
                alerts.append(data)
        
        # Sort by p_change descending
        alerts.sort(key=lambda x: x.get('p_change', 0), reverse=True)
        return alerts
    
    def get_stability_report(self) -> Dict:
        """
        Generate a stability report across all assets.
        
        Returns summary of regime stability across the portfolio.
        """
        all_probs = self.scan_all_assets()
        
        stable = []      # p_change < 10%
        moderate = []    # 10% <= p_change < 30%
        elevated = []    # 30% <= p_change < 50%
        high_risk = []   # p_change >= 50%
        
        for asset_key, data in all_probs.items():
            if 'error' in data:
                continue
            
            p_change = data.get('p_change', 0)
            entry = {
                'asset': asset_key,
                'regime': data.get('current_regime'),
                'p_change': p_change,
                'duration': data.get('regime_duration_days', 0)
            }
            
            if p_change < 0.10:
                stable.append(entry)
            elif p_change < 0.30:
                moderate.append(entry)
            elif p_change < 0.50:
                elevated.append(entry)
            else:
                high_risk.append(entry)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_assets': len(all_probs),
            'summary': {
                'stable': len(stable),
                'moderate': len(moderate),
                'elevated': len(elevated),
                'high_risk': len(high_risk)
            },
            'stable_assets': stable,
            'moderate_risk': moderate,
            'elevated_risk': elevated,
            'high_risk_assets': high_risk,
            'avg_p_change': round(np.mean([d.get('p_change', 0) for d in all_probs.values() if 'error' not in d]), 4)
        }
    
    def detect_contagion_risk(self) -> Dict:
        """
        Detect potential contagion - multiple correlated assets showing elevated transition risk.
        
        Groups assets by type and checks for coordinated regime stress.
        """
        # Asset groupings
        groups = {
            'US_Equities': ['1625_SP500', '269_NASDAQ', '1518_RUSSEL', '336_DOW_JONES_Mini'],
            'India': ['1387_Nifty_Bank', '1398_Nifty_50'],
            'Commodities': ['477_GOLD', '1435_MCX_Copper', '1859_Brent_Oil'],
            'FX': ['256_USD_INR', '655_US_DOLLAR_Index'],
            'International': ['291_SPDR_China_ETF', '358_Nikkei_225']
        }
        
        all_probs = self.scan_all_assets()
        
        contagion_report = {}
        for group_name, assets in groups.items():
            group_data = []
            for asset in assets:
                if asset in all_probs and 'error' not in all_probs[asset]:
                    group_data.append({
                        'asset': asset,
                        'p_change': all_probs[asset].get('p_change', 0),
                        'regime': all_probs[asset].get('current_regime'),
                        'alert_level': all_probs[asset].get('alert_level', 'normal')
                    })
            
            if group_data:
                avg_p_change = np.mean([d['p_change'] for d in group_data])
                elevated_count = sum(1 for d in group_data if d['p_change'] >= 0.30)
                
                contagion_report[group_name] = {
                    'assets': group_data,
                    'avg_p_change': round(avg_p_change, 4),
                    'elevated_count': elevated_count,
                    'total': len(group_data),
                    'contagion_risk': 'HIGH' if elevated_count >= 2 else 
                                      'MODERATE' if elevated_count == 1 else 'LOW'
                }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'groups': contagion_report
        }
    
    def generate_dashboard_data(self) -> Dict:
        """Generate data formatted for dashboard display."""
        stability = self.get_stability_report()
        alerts = self.get_alerts('watch')
        contagion = self.detect_contagion_risk()
        
        # Format for dashboard
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_risk': self._calculate_overall_risk(stability),
            'summary': stability['summary'],
            'alerts': alerts[:5],  # Top 5 alerts
            'contagion_groups': contagion['groups'],
            'asset_details': self.scan_all_assets()
        }
    
    def _calculate_overall_risk(self, stability: Dict) -> str:
        """Calculate overall portfolio risk level."""
        high = stability['summary']['high_risk']
        elevated = stability['summary']['elevated']
        total = stability['total_assets']
        
        if high >= 3 or (high + elevated) / max(total, 1) > 0.5:
            return 'CRITICAL'
        elif high >= 1 or elevated >= 3:
            return 'HIGH'
        elif elevated >= 1:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def save_snapshot(self, output_path: str = None):
        """Save current state to JSON for historical tracking."""
        if output_path is None:
            output_path = self.models_dir / f"early_warning_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        data = self.generate_dashboard_data()
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Snapshot saved to {output_path}")
        return output_path


def main():
    """Run early warning scan and print results."""
    print("=" * 60)
    print("EARLY WARNING SYSTEM - Regime Transition Monitor")
    print("=" * 60)
    print()
    
    # Initialize system
    ews = EarlyWarningSystem("regime_models")
    
    # Get stability report
    print("\n📊 STABILITY REPORT")
    print("-" * 40)
    stability = ews.get_stability_report()
    print(f"Total assets monitored: {stability['total_assets']}")
    print(f"Average P(change): {stability['avg_p_change']:.1%}")
    print()
    print(f"  🟢 Stable (<10%):     {stability['summary']['stable']}")
    print(f"  🟡 Moderate (10-30%): {stability['summary']['moderate']}")
    print(f"  🟠 Elevated (30-50%): {stability['summary']['elevated']}")
    print(f"  🔴 High Risk (>50%):  {stability['summary']['high_risk']}")
    
    # Get alerts
    print("\n⚠️  ACTIVE ALERTS")
    print("-" * 40)
    alerts = ews.get_alerts('watch')
    if alerts:
        for alert in alerts:
            level_emoji = {'watch': '🟡', 'warning': '🟠', 'critical': '🔴'}.get(alert['alert_level'], '⚪')
            print(f"{level_emoji} {alert['asset']}")
            print(f"   Current: {alert['current_regime']} (confidence: {alert['confidence']:.1%})")
            print(f"   P(change): {alert['p_change']:.1%} → likely {alert['most_likely_next']}")
            print(f"   Duration: {alert['regime_duration_days']} days")
            print()
    else:
        print("No alerts - all assets stable")
    
    # Contagion risk
    print("\n🌐 CONTAGION RISK BY GROUP")
    print("-" * 40)
    contagion = ews.detect_contagion_risk()
    for group_name, data in contagion['groups'].items():
        risk_emoji = {'HIGH': '🔴', 'MODERATE': '🟠', 'LOW': '🟢'}.get(data['contagion_risk'], '⚪')
        print(f"{risk_emoji} {group_name}: {data['contagion_risk']}")
        print(f"   Avg P(change): {data['avg_p_change']:.1%}")
        print(f"   Elevated: {data['elevated_count']}/{data['total']}")
    
    # Save snapshot
    print("\n" + "=" * 60)
    snapshot_path = ews.save_snapshot()
    print(f"Dashboard data saved to: {snapshot_path}")
    
    return ews


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    ews = main()
