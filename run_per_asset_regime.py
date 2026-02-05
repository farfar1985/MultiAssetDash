#!/usr/bin/env python3
"""
Run enhanced quantum regime detection on all available assets.
Per-asset analysis for Bill's research directive.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from quantum_regime_enhanced import EnhancedQuantumRegimeDetector
from per_asset_optimizer import load_asset_data

# Find all asset directories
data_dir = Path('data')
asset_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name != 'horizons_wide']

print(f"Found {len(asset_dirs)} assets to analyze")
print("=" * 60)

all_results = {}

for asset_dir in sorted(asset_dirs):
    asset_name = asset_dir.name
    print(f"\n[{asset_name}]")
    
    try:
        horizons, prices = load_asset_data(str(asset_dir))
        
        if prices is None or len(prices) < 100:
            print(f"  Skipping - insufficient data ({len(prices) if prices is not None else 0} points)")
            continue
        
        print(f"  Loaded {len(prices)} price points")
        
        # Initialize detector
        detector = EnhancedQuantumRegimeDetector()
        detector.initialize(str(asset_dir), prices)
        
        # Run detection on all valid time points
        results = []
        for t in range(60, len(prices) - 5):
            if t % 100 == 0:
                print(f"    Processing t={t}/{len(prices)}")
            
            contagion = np.random.uniform(0.1, 0.4)  # Placeholder
            result = detector.detect_enhanced(prices, t, contagion)
            results.append(result)
        
        # Analyze regimes
        import pandas as pd
        regimes = [r['regime'] for r in results]
        regime_dist = pd.Series(regimes).value_counts()
        
        # Metrics
        avg_price_entropy = np.mean([r['price_entropy'] for r in results])
        avg_confidence = np.mean([r['enhanced_confidence'] for r in results])
        avg_vol_premium = np.mean([r['vol_premium'] for r in results])
        crowded_pct = sum(r['crowded_positioning'] for r in results) / len(results) * 100
        
        print(f"  Regime Distribution:")
        for regime, count in regime_dist.items():
            print(f"    {regime}: {count} ({count/len(regimes)*100:.1f}%)")
        print(f"  Avg Price Entropy: {avg_price_entropy:.3f}")
        print(f"  Avg Confidence: {avg_confidence:.3f}")
        print(f"  Avg Vol Premium: {avg_vol_premium*100:.1f}%")
        
        all_results[asset_name] = {
            'n_samples': len(results),
            'regime_distribution': regime_dist.to_dict(),
            'avg_price_entropy': round(avg_price_entropy, 3),
            'avg_external_entropy': round(np.mean([r['external_entropy'] for r in results]), 3),
            'avg_confidence': round(avg_confidence, 3),
            'avg_vol_premium': round(avg_vol_premium, 4),
            'crowded_pct': round(crowded_pct, 1),
            'crisis_pct': round(regime_dist.get('CRISIS', 0) / len(regimes) * 100, 1),
            'low_vol_pct': round(regime_dist.get('LOW_VOL', 0) / len(regimes) * 100, 1)
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

# Save results
output = {
    'generated_at': datetime.now().isoformat(),
    'version': 'per_asset_v1',
    'n_assets': len(all_results),
    'results': all_results
}

output_path = Path('configs/per_asset_regime_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 60)
print(f"Saved results to {output_path}")

# Summary table
print("\nSUMMARY BY ASSET:")
print("-" * 80)
print(f"{'Asset':<25} {'LOW_VOL%':>10} {'CRISIS%':>10} {'Entropy':>10} {'Confidence':>12}")
print("-" * 80)
for name, data in sorted(all_results.items(), key=lambda x: x[1]['crisis_pct'], reverse=True):
    print(f"{name:<25} {data['low_vol_pct']:>10.1f} {data['crisis_pct']:>10.1f} {data['avg_price_entropy']:>10.3f} {data['avg_confidence']:>12.3f}")
