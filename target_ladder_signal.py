# target_ladder_signal.py
# Target-Based Signal Generator
# Key insight: n1-n10 are PRICE TARGETS, not day forecasts

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from ensemble_methods import EnsembleMethods

DATA_DIR = r"C:\Users\William Dennis\projects\nexus\data"


def load_horizon_data(asset_id: int, asset_name: str, horizon: int) -> tuple:
    folder = f"{asset_id}_{asset_name}"
    path = os.path.join(DATA_DIR, folder, 'horizons_wide', f'horizon_{horizon}.joblib')
    if not os.path.exists(path):
        return None, None
    data = joblib.load(path)
    return data['X'], data['y']


def identify_parent_child(columns):
    """
    Separate parent models from children.
    Parent: column without underscore (e.g., '4442.0')
    Children: columns with underscore (e.g., '4442.0_100001')
    """
    parents = []
    children = []
    
    for col in columns:
        col_str = str(col)
        if '_' in col_str:
            children.append(col)
        else:
            parents.append(col)
    
    return parents, children


def compute_child_weighted_signal(X: pd.DataFrame, y: pd.Series, 
                                   use_children_only: bool = True) -> tuple:
    """
    Compute signal using child models weighted by their individual accuracy.
    
    Returns:
        (signal_series, forecast_series, weights_used)
    """
    ensemble = EnsembleMethods(lookback_window=60)
    
    parents, children = identify_parent_child(X.columns)
    
    if use_children_only and len(children) > 10:
        X_use = X[children]
    else:
        X_use = X
    
    # Weight by accuracy
    weights = ensemble.accuracy_weighted(X_use, y)
    
    # Get top performers (children with weight > 1.5x average)
    avg_weight = weights.mean()
    top_children = weights[weights > avg_weight * 1.2].index.tolist()
    
    if len(top_children) < 5:
        top_children = weights.nlargest(min(50, len(weights))).index.tolist()
    
    # Ensemble prediction from top children
    X_top = X_use[top_children]
    weights_top = weights[top_children]
    weights_top = weights_top / weights_top.sum()  # Normalize
    
    ensemble_pred = (X_top * weights_top).sum(axis=1)
    
    return ensemble_pred, weights_top


def generate_target_ladder(asset_id: int, asset_name: str, 
                           horizons: list, min_move_threshold: float) -> dict:
    """
    Generate a target ladder signal based on multi-horizon consensus.
    
    Key concepts:
    - n1-n10 are price TARGETS, not time predictions
    - Look for consensus across targets
    - Only signal when magnitude is significant (not 20-30 cent moves)
    """
    
    # Load current price from most recent non-NaN actual
    _, y = load_horizon_data(asset_id, asset_name, horizons[0])
    if y is None:
        return None
    
    # Get most recent non-NaN value
    y_clean = y.dropna()
    if len(y_clean) == 0:
        return None
    
    current_price = float(y_clean.iloc[-1])
    current_date = y_clean.index[-1]
    
    # Get target predictions from each horizon
    targets = {}
    child_signals = {}
    
    for h in horizons:
        X, y_h = load_horizon_data(asset_id, asset_name, h)
        if X is None:
            continue
        
        # Use child-weighted ensemble
        ensemble_pred, weights = compute_child_weighted_signal(X, y_h, use_children_only=True)
        
        # Latest non-NaN prediction is the target
        ensemble_clean = ensemble_pred.dropna()
        if len(ensemble_clean) == 0:
            continue
        target_price = float(ensemble_clean.iloc[-1])
        target_move = target_price - current_price
        target_direction = 1 if target_move > 0 else (-1 if target_move < 0 else 0)
        
        targets[f'n{h}'] = {
            'price': round(target_price, 2),
            'move': round(target_move, 2),
            'move_pct': round((target_move / current_price) * 100, 2),
            'direction': 'BULLISH' if target_direction > 0 else ('BEARISH' if target_direction < 0 else 'NEUTRAL'),
            'children_used': len(weights)
        }
        child_signals[h] = target_direction
    
    if not targets:
        return None
    
    # Compute consensus
    directions = list(child_signals.values())
    bullish_count = sum(1 for d in directions if d > 0)
    bearish_count = sum(1 for d in directions if d < 0)
    total = len(directions)
    
    if bullish_count > bearish_count:
        consensus_direction = 'BULLISH'
        consensus_pct = (bullish_count / total) * 100
    elif bearish_count > bullish_count:
        consensus_direction = 'BEARISH'
        consensus_pct = (bearish_count / total) * 100
    else:
        consensus_direction = 'NEUTRAL'
        consensus_pct = 50.0
    
    # Compute target ladder (sorted by expected move)
    sorted_targets = sorted(targets.items(), 
                           key=lambda x: abs(x[1]['move']), 
                           reverse=False)
    
    target_ladder = []
    for name, data in sorted_targets:
        if data['direction'] == consensus_direction:
            target_ladder.append({
                'level': name,
                'price': data['price'],
                'move': data['move']
            })
    
    # Max expected move
    max_move = max(abs(t['move']) for t in targets.values())
    
    # Is this significant?
    is_significant = max_move >= min_move_threshold
    
    # Conviction level
    if consensus_pct >= 80 and is_significant:
        conviction = 'HIGH'
    elif consensus_pct >= 60 and is_significant:
        conviction = 'MEDIUM'
    else:
        conviction = 'LOW'
    
    return {
        'asset': asset_name,
        'current_price': current_price,
        'current_date': str(current_date),
        'signal': {
            'direction': consensus_direction,
            'consensus_pct': round(consensus_pct, 1),
            'targets_agreeing': bullish_count if consensus_direction == 'BULLISH' else bearish_count,
            'total_targets': total,
            'conviction': conviction
        },
        'magnitude': {
            'max_move': round(max_move, 2),
            'max_move_pct': round((max_move / current_price) * 100, 2),
            'is_significant': is_significant,
            'threshold': min_move_threshold
        },
        'target_ladder': target_ladder[:5],  # Top 5 targets in direction
        'all_targets': targets
    }


def main():
    print("=" * 70, flush=True)
    print("  TARGET LADDER SIGNAL GENERATOR", flush=True)
    print("  n1-n10 = Price Targets, not Day Forecasts", flush=True)
    print("=" * 70, flush=True)
    
    ASSETS = {
        'Crude_Oil': {'id': 1866, 'horizons': [1, 2, 3, 5, 7, 10], 'threshold': 0.75},
        'SP500': {'id': 1625, 'horizons': [1, 3, 5, 8, 13], 'threshold': 15.0},
        'Bitcoin': {'id': 1860, 'horizons': [1, 3, 5, 8, 10], 'threshold': 1000.0},
    }
    
    for asset_name, info in ASSETS.items():
        print(f"\n{'='*70}", flush=True)
        print(f"  {asset_name}", flush=True)
        print(f"{'='*70}", flush=True)
        
        result = generate_target_ladder(
            info['id'], asset_name, info['horizons'], info['threshold']
        )
        
        if not result:
            print("  No data available", flush=True)
            continue
        
        print(f"\n  Current Price: ${result['current_price']:.2f}", flush=True)
        print(f"  Date: {result['current_date']}", flush=True)
        
        sig = result['signal']
        print(f"\n  SIGNAL: {sig['direction']}", flush=True)
        print(f"  Consensus: {sig['consensus_pct']:.1f}% ({sig['targets_agreeing']}/{sig['total_targets']} targets agree)", flush=True)
        print(f"  Conviction: {sig['conviction']}", flush=True)
        
        mag = result['magnitude']
        print(f"\n  Max Expected Move: ${mag['max_move']:.2f} ({mag['max_move_pct']:.2f}%)", flush=True)
        print(f"  Significant (>= ${mag['threshold']}): {'YES' if mag['is_significant'] else 'NO'}", flush=True)
        
        if result['target_ladder']:
            print(f"\n  TARGET LADDER ({sig['direction']}):", flush=True)
            for t in result['target_ladder']:
                print(f"    {t['level']}: ${t['price']:.2f} (move: ${t['move']:+.2f})", flush=True)
    
    print(f"\n{'='*70}", flush=True)
    print("  Signal generation complete", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
