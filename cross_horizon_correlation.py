# cross_horizon_correlation.py
# Cross-Horizon Correlation Analysis
# Per Artemis: "When D+1 and D+10 agree, is confidence higher?"

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r"C:\Users\William Dennis\projects\nexus\data"
OUTPUT_DIR = r"C:\Users\William Dennis\projects\nexus\results"


def load_horizon_data(asset_id: int, asset_name: str, horizon: int) -> tuple:
    folder = f"{asset_id}_{asset_name}"
    path = os.path.join(DATA_DIR, folder, 'horizons_wide', f'horizon_{horizon}.joblib')
    if not os.path.exists(path):
        return None, None
    data = joblib.load(path)
    return data['X'], data['y']


def compute_horizon_signals(asset_id: int, asset_name: str, horizons: list) -> pd.DataFrame:
    """Compute ensemble signals for each horizon."""
    sys.path.insert(0, os.path.dirname(__file__))
    from ensemble_methods import EnsembleMethods
    
    ensemble = EnsembleMethods(lookback_window=60)
    signals = {}
    
    for h in horizons:
        X, y = load_horizon_data(asset_id, asset_name, h)
        if X is None:
            continue
        
        # Use top-k ensemble
        try:
            weights = ensemble.top_k_by_sharpe(X, y, top_pct=0.1)
            ensemble_pred = (X * weights).sum(axis=1)
            signals[f'D+{h}'] = np.sign(ensemble_pred.diff())
        except:
            continue
    
    return pd.DataFrame(signals)


def analyze_cross_horizon_correlation(signal_df: pd.DataFrame, actuals: pd.Series, 
                                       min_threshold: float) -> dict:
    """
    Analyze how horizon agreement affects accuracy.
    Key question: Does multi-horizon consensus improve predictions?
    """
    results = {
        'correlation_matrix': None,
        'agreement_accuracy': {},
        'disagreement_analysis': {},
        'pair_predictive_power': {}
    }
    
    # 1. Correlation matrix between horizons
    corr = signal_df.corr()
    results['correlation_matrix'] = corr
    
    # 2. Agreement accuracy: When N horizons agree, what's the accuracy?
    common_idx = signal_df.dropna().index.intersection(actuals.index)
    sig_aligned = signal_df.loc[common_idx]
    act_aligned = actuals.loc[common_idx]
    
    actual_dir = np.sign(act_aligned.diff())
    
    for n_agree in range(2, len(signal_df.columns) + 1):
        # Count how many horizons agree with majority per row
        mode_vals = sig_aligned.mode(axis=1)
        if mode_vals.empty or len(mode_vals.columns) == 0:
            continue
        mode_signal = mode_vals.iloc[:, 0]
        
        # Count agreement with mode
        agreement_count = sig_aligned.apply(lambda row: (row == mode_signal.loc[row.name]).sum(), axis=1)
        
        mask = agreement_count >= n_agree
        if mask.sum() < 10:
            continue
        
        # Get the consensus signal
        consensus = mode_signal.loc[mask]
        actual_at_consensus = actual_dir.reindex(consensus.index)
        
        accuracy = (consensus.shift(1) == actual_at_consensus).dropna().mean() * 100
        
        results['agreement_accuracy'][n_agree] = {
            'accuracy': round(accuracy, 1),
            'n_signals': int(mask.sum()),
            'pct_of_total': round(mask.mean() * 100, 1)
        }
    
    # 3. Pair-wise predictive power
    horizons = signal_df.columns.tolist()
    for i, h1 in enumerate(horizons[:-1]):
        for h2 in horizons[i+1:]:
            pair_df = signal_df[[h1, h2]].dropna()
            
            # When both agree
            agree_mask = pair_df[h1] == pair_df[h2]
            
            if agree_mask.sum() < 10:
                continue
            
            agree_signal = pair_df.loc[agree_mask, h1]
            agree_actual = actual_dir.reindex(agree_signal.index)
            
            agree_acc = (agree_signal.shift(1) == agree_actual).dropna().mean() * 100
            
            # When they disagree
            disagree_mask = ~agree_mask
            if disagree_mask.sum() >= 10:
                # Which horizon is right when they disagree?
                h1_right = (pair_df.loc[disagree_mask, h1].shift(1) == actual_dir.reindex(pair_df.loc[disagree_mask].index)).dropna().mean() * 100
                h2_right = (pair_df.loc[disagree_mask, h2].shift(1) == actual_dir.reindex(pair_df.loc[disagree_mask].index)).dropna().mean() * 100
            else:
                h1_right = h2_right = np.nan
            
            results['pair_predictive_power'][f'{h1}_{h2}'] = {
                'agreement_rate': round(agree_mask.mean() * 100, 1),
                'accuracy_when_agree': round(agree_acc, 1),
                f'{h1}_right_when_disagree': round(h1_right, 1) if not np.isnan(h1_right) else None,
                f'{h2}_right_when_disagree': round(h2_right, 1) if not np.isnan(h2_right) else None
            }
    
    return results


def main():
    print("=" * 70, flush=True)
    print("  CROSS-HORIZON CORRELATION ANALYSIS", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)
    
    ASSETS = {
        'Crude_Oil': {'id': 1866, 'horizons': [1, 2, 3, 5, 7, 10], 'threshold': 0.50},
        'SP500': {'id': 1625, 'horizons': [1, 3, 5, 8, 13, 21], 'threshold': 10.0},
        'Bitcoin': {'id': 1860, 'horizons': [1, 3, 5, 8, 10], 'threshold': 500.0},
    }
    
    all_results = {}
    
    for asset_name, info in ASSETS.items():
        print(f"\n{'='*70}", flush=True)
        print(f"  {asset_name}", flush=True)
        print(f"{'='*70}", flush=True)
        
        # Get signals for all horizons
        signal_df = compute_horizon_signals(info['id'], asset_name, info['horizons'])
        
        if len(signal_df.columns) < 2:
            print("  Not enough horizons with data", flush=True)
            continue
        
        print(f"  Horizons with data: {list(signal_df.columns)}", flush=True)
        
        # Get actuals
        _, y = load_horizon_data(info['id'], asset_name, info['horizons'][0])
        
        # Analyze
        results = analyze_cross_horizon_correlation(signal_df, y, info['threshold'])
        all_results[asset_name] = results
        
        # Print correlation matrix
        print("\n  CORRELATION MATRIX:", flush=True)
        print(results['correlation_matrix'].round(2).to_string(), flush=True)
        
        # Print agreement accuracy
        print("\n  AGREEMENT ACCURACY:", flush=True)
        print("  (When N horizons agree, what's the accuracy?)", flush=True)
        for n, data in results['agreement_accuracy'].items():
            print(f"    {n}+ horizons agree: {data['accuracy']:.1f}% accuracy ({data['n_signals']} signals, {data['pct_of_total']:.1f}% of total)", flush=True)
        
        # Print best pairs
        print("\n  HORIZON PAIR ANALYSIS:", flush=True)
        pair_data = results['pair_predictive_power']
        # Sort by accuracy when agree
        sorted_pairs = sorted(pair_data.items(), 
                              key=lambda x: x[1].get('accuracy_when_agree', 0), 
                              reverse=True)
        for pair, data in sorted_pairs[:5]:
            print(f"    {pair}: {data['agreement_rate']:.1f}% agree -> {data['accuracy_when_agree']:.1f}% accuracy", flush=True)
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, f'cross_horizon_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    # Convert to serializable format
    import json
    serializable = {}
    for asset, data in all_results.items():
        serializable[asset] = {
            'correlation_matrix': data['correlation_matrix'].to_dict() if data['correlation_matrix'] is not None else None,
            'agreement_accuracy': data['agreement_accuracy'],
            'pair_predictive_power': data['pair_predictive_power']
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\n{'='*70}", flush=True)
    print(f"  ANALYSIS COMPLETE", flush=True)
    print(f"  Results: {output_path}", flush=True)
    print(f"{'='*70}", flush=True)
    
    # Key insight summary
    print("\n  KEY INSIGHTS:", flush=True)
    for asset, data in all_results.items():
        if data['agreement_accuracy']:
            max_agree = max(data['agreement_accuracy'].keys())
            max_acc = data['agreement_accuracy'][max_agree]['accuracy']
            print(f"    {asset}: {max_agree}+ horizons agree -> {max_acc:.1f}% accuracy", flush=True)


if __name__ == "__main__":
    main()
