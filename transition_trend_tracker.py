"""
Transition Probability Trend Tracker
====================================
Tracks how transition probabilities change over time to detect
building regime stress before it hits alert thresholds.

Key signals:
- Rising trend: P(change) increasing over last N observations
- Acceleration: Rate of increase is itself increasing  
- Duration stress: Long regime duration + rising P(change)

Author: AmiraB
Date: 2026-02-06
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from scipy import stats
from early_warning_system import EarlyWarningSystem


class TransitionTrendTracker:
    """Track trends in transition probabilities over time."""
    
    def __init__(self, 
                 models_dir: str = "regime_models",
                 history_file: str = "regime_models/transition_history.json"):
        self.models_dir = Path(models_dir)
        self.history_file = Path(history_file)
        self.ews = EarlyWarningSystem(models_dir)
        self.history: Dict[str, List[Dict]] = {}
        self._load_history()
    
    def _load_history(self):
        """Load historical transition probability data."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
            print(f"Loaded history with {len(self.history)} assets")
        else:
            self.history = {}
            print("No history file found - starting fresh")
    
    def _save_history(self):
        """Save history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record_snapshot(self):
        """Record current transition probabilities for all assets."""
        timestamp = datetime.now().isoformat()
        all_probs = self.ews.scan_all_assets()
        
        for asset_key, data in all_probs.items():
            if 'error' in data:
                continue
            
            if asset_key not in self.history:
                self.history[asset_key] = []
            
            # Keep last 100 observations to limit file size
            if len(self.history[asset_key]) >= 100:
                self.history[asset_key] = self.history[asset_key][-99:]
            
            self.history[asset_key].append({
                'timestamp': timestamp,
                'p_change': data['p_change'],
                'p_stay': data['p_stay'],
                'regime': data['current_regime'],
                'confidence': data['confidence'],
                'duration': data.get('regime_duration_days', 0)
            })
        
        self._save_history()
        print(f"Recorded snapshot at {timestamp}")
        return len(all_probs)
    
    def analyze_trend(self, asset_key: str, lookback: int = 10) -> Dict:
        """
        Analyze transition probability trend for an asset.
        
        Args:
            asset_key: Asset identifier
            lookback: Number of observations to analyze
            
        Returns:
            Dict with trend analysis
        """
        if asset_key not in self.history:
            return {'error': f'No history for {asset_key}'}
        
        history = self.history[asset_key]
        if len(history) < 3:
            return {
                'asset': asset_key,
                'status': 'insufficient_data',
                'observations': len(history)
            }
        
        # Get recent data
        recent = history[-lookback:] if len(history) >= lookback else history
        p_changes = [h['p_change'] for h in recent]
        timestamps = [h['timestamp'] for h in recent]
        
        # Current values
        current_p = p_changes[-1]
        current_regime = recent[-1]['regime']
        
        # Calculate trend (linear regression)
        x = np.arange(len(p_changes))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, p_changes)
        
        # Trend direction
        if slope > 0.005:
            trend_direction = 'RISING'
        elif slope < -0.005:
            trend_direction = 'FALLING'
        else:
            trend_direction = 'STABLE'
        
        # Calculate acceleration (second derivative - is the rate of change increasing?)
        if len(p_changes) >= 5:
            first_half = np.mean(np.diff(p_changes[:len(p_changes)//2]))
            second_half = np.mean(np.diff(p_changes[len(p_changes)//2:]))
            acceleration = second_half - first_half
        else:
            acceleration = 0
        
        # Volatility in P(change)
        p_volatility = np.std(p_changes)
        
        # Min/max over period
        p_min = min(p_changes)
        p_max = max(p_changes)
        
        # Signal strength
        signal = 'NONE'
        if trend_direction == 'RISING':
            if current_p > 0.20 and slope > 0.01:
                signal = 'STRONG_WARNING'
            elif current_p > 0.10 and slope > 0.005:
                signal = 'EARLY_WARNING'
            elif slope > 0.01:
                signal = 'WATCH'
        
        return {
            'asset': asset_key,
            'current_regime': current_regime,
            'current_p_change': round(current_p, 4),
            'observations': len(recent),
            'trend': {
                'direction': trend_direction,
                'slope': round(slope, 6),
                'r_squared': round(r_value**2, 4),
                'p_value': round(p_value, 4),
                'acceleration': round(acceleration, 6)
            },
            'range': {
                'min': round(p_min, 4),
                'max': round(p_max, 4),
                'volatility': round(p_volatility, 4)
            },
            'signal': signal,
            'first_timestamp': timestamps[0],
            'last_timestamp': timestamps[-1]
        }
    
    def get_rising_assets(self, min_slope: float = 0.005) -> List[Dict]:
        """Get all assets with rising transition probabilities."""
        rising = []
        
        for asset_key in self.history.keys():
            analysis = self.analyze_trend(asset_key)
            if 'error' in analysis or analysis.get('status') == 'insufficient_data':
                continue
            
            if analysis['trend']['slope'] >= min_slope:
                rising.append(analysis)
        
        # Sort by slope descending
        rising.sort(key=lambda x: x['trend']['slope'], reverse=True)
        return rising
    
    def get_early_warnings(self) -> List[Dict]:
        """
        Get assets showing early warning signals.
        
        These are assets that may not have hit alert thresholds yet,
        but are showing concerning trends.
        """
        warnings = []
        
        for asset_key in self.history.keys():
            analysis = self.analyze_trend(asset_key)
            if 'error' in analysis or analysis.get('status') == 'insufficient_data':
                continue
            
            if analysis['signal'] in ['WATCH', 'EARLY_WARNING', 'STRONG_WARNING']:
                warnings.append(analysis)
        
        # Sort by signal severity then slope
        severity = {'STRONG_WARNING': 3, 'EARLY_WARNING': 2, 'WATCH': 1}
        warnings.sort(key=lambda x: (severity.get(x['signal'], 0), x['trend']['slope']), reverse=True)
        return warnings
    
    def generate_trend_report(self) -> Dict:
        """Generate a comprehensive trend report."""
        all_trends = {}
        rising_count = 0
        falling_count = 0
        stable_count = 0
        
        for asset_key in self.history.keys():
            analysis = self.analyze_trend(asset_key)
            if 'error' in analysis or analysis.get('status') == 'insufficient_data':
                continue
            
            all_trends[asset_key] = analysis
            direction = analysis['trend']['direction']
            if direction == 'RISING':
                rising_count += 1
            elif direction == 'FALLING':
                falling_count += 1
            else:
                stable_count += 1
        
        early_warnings = self.get_early_warnings()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_assets': len(all_trends),
                'rising': rising_count,
                'falling': falling_count,
                'stable': stable_count
            },
            'early_warnings': early_warnings,
            'trends': all_trends
        }
    
    def print_report(self):
        """Print trend report to console."""
        print("=" * 60)
        print("TRANSITION TREND ANALYSIS")
        print("=" * 60)
        
        report = self.generate_trend_report()
        summary = report['summary']
        
        print(f"\nTotal assets: {summary['total_assets']}")
        print(f"  Rising:  {summary['rising']}")
        print(f"  Stable:  {summary['stable']}")
        print(f"  Falling: {summary['falling']}")
        
        print("\n" + "-" * 40)
        print("EARLY WARNING SIGNALS")
        print("-" * 40)
        
        warnings = report['early_warnings']
        if warnings:
            for w in warnings:
                print(f"\n[{w['signal']}] {w['asset']}")
                print(f"  Regime: {w['current_regime']}")
                print(f"  P(change): {w['current_p_change']:.1%}")
                print(f"  Trend slope: {w['trend']['slope']:.4f}")
                print(f"  Range: {w['range']['min']:.1%} - {w['range']['max']:.1%}")
        else:
            print("\nNo early warning signals detected.")
        
        print("\n" + "-" * 40)
        print("ALL ASSET TRENDS")
        print("-" * 40)
        
        for asset_key, analysis in report['trends'].items():
            direction = analysis['trend']['direction']
            arrow = {'RISING': '^', 'FALLING': 'v', 'STABLE': '-'}.get(direction, '?')
            print(f"{arrow} {asset_key}: {analysis['current_p_change']:.1%} ({direction})")


def main():
    """Run trend tracker and print report."""
    tracker = TransitionTrendTracker()
    
    # Record current snapshot
    print("Recording current transition probabilities...")
    tracker.record_snapshot()
    
    # Print report
    tracker.print_report()
    
    return tracker


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    tracker = main()
