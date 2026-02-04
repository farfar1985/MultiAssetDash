"""
Meta-Ensemble: Stack multiple ensemble methods
==============================================
Layer 1: Run multiple ensemble strategies
Layer 2: Combine their outputs via majority vote / weighted average
"""

import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_horizons():
    """Load D+5, D+7, D+10 data"""
    data_dir = Path('data/1866_Crude_Oil/horizons_wide')
    horizons = {}
    for h in [5, 7, 10]:
        f = data_dir / f'horizon_{h}.joblib'
        data = joblib.load(f)
        X = data['X'].values if hasattr(data['X'], 'values') else np.array(data['X'])
        y = data['y'].values if hasattr(data['y'], 'values') else np.array(data['y'])
        horizons[h] = {'X': X, 'y': y}
        print(f'D+{h}: {X.shape}')
    return horizons

# =============================================================
# LAYER 1: Individual Ensemble Methods
# =============================================================

def pairwise_slopes(horizons, threshold=0.4):
    """Original winning method - compare horizon predictions"""
    signals = []
    for i in range(len(horizons[5]['y'])):
        slopes = []
        pairs = [(5,7), (5,10), (7,10)]
        for h1, h2 in pairs:
            p1 = horizons[h1]['X'][i].mean()
            p2 = horizons[h2]['X'][i].mean()
            slope = (p2 - p1) / (h2 - h1)
            slopes.append(slope)
        avg_slope = np.mean(slopes)
        if avg_slope > threshold:
            signals.append(1)
        elif avg_slope < -threshold:
            signals.append(-1)
        else:
            signals.append(0)
    return np.array(signals)

def top_k_mean(X, k_pct=0.1):
    """Top K% models by lowest variance (most confident)"""
    var = X.var(axis=0)
    k = max(1, int(X.shape[1] * k_pct))
    top_idx = np.argsort(var)[:k]
    return X[:, top_idx].mean(axis=1)

def inverse_variance_weighted(X):
    """Weight models by inverse variance"""
    var = X.var(axis=0) + 1e-8
    weights = 1 / var
    weights /= weights.sum()
    return X @ weights

def horizon_consensus(horizons, agreement_threshold=0.6):
    """Signal when horizons agree on direction"""
    signals = []
    for i in range(len(horizons[5]['y'])):
        base = horizons[5]['y'][i]
        directions = []
        for h in [5, 7, 10]:
            pred = horizons[h]['X'][i].mean()
            directions.append(1 if pred > base else -1)
        
        agreement = abs(sum(directions)) / 3
        if agreement >= agreement_threshold:
            signals.append(np.sign(sum(directions)))
        else:
            signals.append(0)
    return np.array(signals)

def magnitude_weighted(horizons):
    """Weight predictions by their magnitude (conviction)"""
    preds = []
    mags = []
    for h in [5, 7, 10]:
        pred = horizons[h]['X'].mean(axis=1)
        base = horizons[h]['y']
        mag = np.abs(pred - base) + 1e-8
        preds.append(pred)
        mags.append(mag)
    
    weighted = np.zeros(len(base))
    total_mag = np.zeros(len(base))
    for p, m in zip(preds, mags):
        weighted += p * m
        total_mag += m
    return weighted / total_mag

def to_signal(pred, threshold_pct=0.001):
    """Convert price predictions to directional signals"""
    pct_change = np.diff(pred) / (pred[:-1] + 1e-8)
    signals = np.zeros(len(pct_change))
    signals[pct_change > threshold_pct] = 1
    signals[pct_change < -threshold_pct] = -1
    return signals

# =============================================================
# LAYER 2: Meta-Ensemble Combiners
# =============================================================

def majority_vote(signals_list):
    """Majority vote across ensemble methods"""
    stacked = np.column_stack(signals_list)
    return np.sign(stacked.sum(axis=1))

def weighted_vote(signals_list, weights):
    """Weighted combination of signals"""
    stacked = np.column_stack(signals_list)
    weighted = stacked @ np.array(weights)
    return np.sign(weighted)

def agreement_filter(signals_list, min_agree=3):
    """Only trade when N methods agree"""
    stacked = np.column_stack(signals_list)
    agreement = np.abs(stacked.sum(axis=1))
    signals = np.sign(stacked.sum(axis=1))
    signals[agreement < min_agree] = 0
    return signals

def calc_metrics(signals, actual_returns):
    """Calculate trading metrics"""
    mask = signals != 0
    n_trades = mask.sum()
    
    if n_trades == 0:
        return {'sharpe': 0, 'return': 0, 'win_rate': 0, 'trades': 0, 'da': 0}
    
    returns = signals[mask] * actual_returns[mask]
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    total_ret = returns.sum()
    win_rate = (returns > 0).mean()
    
    # Directional accuracy on trades
    actual_dir = np.sign(actual_returns[mask])
    da = (signals[mask] == actual_dir).mean()
    
    return {
        'sharpe': sharpe,
        'return': total_ret,
        'win_rate': win_rate,
        'trades': n_trades,
        'da': da
    }

def main():
    print("=" * 70)
    print("META-ENSEMBLE: Stacking Multiple Ensemble Methods")
    print("=" * 70)
    
    # Load data
    horizons = load_horizons()
    y = horizons[5]['y']
    actual_returns = np.diff(y)
    n = len(y)
    
    print(f"\nTotal samples: {n}")
    print(f"Actual return range: {actual_returns.min():.2f} to {actual_returns.max():.2f}")
    
    # =============================================================
    # LAYER 1: Generate signals from each method
    # =============================================================
    print("\n" + "-" * 70)
    print("LAYER 1: Individual Ensemble Methods")
    print("-" * 70)
    
    methods = {}
    
    # Method 1: Pairwise Slopes (our winner)
    sig_ps = pairwise_slopes(horizons, threshold=0.4)
    methods['Pairwise_Slopes_0.4'] = sig_ps[:-1]
    
    # Method 2: Pairwise Slopes with different threshold
    sig_ps_02 = pairwise_slopes(horizons, threshold=0.2)
    methods['Pairwise_Slopes_0.2'] = sig_ps_02[:-1]
    
    # Method 3: Top 10% by variance
    pred_top10 = (top_k_mean(horizons[5]['X'], 0.1) + 
                  top_k_mean(horizons[7]['X'], 0.1) + 
                  top_k_mean(horizons[10]['X'], 0.1)) / 3
    methods['Top10_Average'] = to_signal(pred_top10)
    
    # Method 4: Inverse variance weighted
    pred_iv = (inverse_variance_weighted(horizons[5]['X']) +
               inverse_variance_weighted(horizons[7]['X']) +
               inverse_variance_weighted(horizons[10]['X'])) / 3
    methods['InvVariance'] = to_signal(pred_iv)
    
    # Method 5: Horizon consensus
    sig_cons = horizon_consensus(horizons, 0.6)
    methods['Horizon_Consensus'] = sig_cons[:-1]
    
    # Method 6: Magnitude weighted
    pred_mag = magnitude_weighted(horizons)
    methods['Magnitude_Weighted'] = to_signal(pred_mag)
    
    # Evaluate Layer 1
    print("\nLayer 1 Results:")
    layer1_results = []
    for name, sig in methods.items():
        m = calc_metrics(sig, actual_returns)
        layer1_results.append({
            'method': name,
            'layer': 1,
            **m
        })
        print(f"  {name:25s} | Sharpe: {m['sharpe']:6.2f} | Win: {m['win_rate']*100:5.1f}% | DA: {m['da']*100:5.1f}% | Trades: {m['trades']:3d}")
    
    # =============================================================
    # LAYER 2: Meta-Ensemble Combinations
    # =============================================================
    print("\n" + "-" * 70)
    print("LAYER 2: Meta-Ensemble Combinations")
    print("-" * 70)
    
    # Get top 4 methods by Sharpe for meta-ensemble
    layer1_sorted = sorted(layer1_results, key=lambda x: -x['sharpe'])
    top_methods = [r['method'] for r in layer1_sorted[:4]]
    print(f"\nTop 4 methods for meta-ensemble: {top_methods}")
    
    top_signals = [methods[m] for m in top_methods]
    
    meta_results = []
    
    # Meta 1: Simple majority vote
    meta_maj = majority_vote(top_signals)
    m = calc_metrics(meta_maj, actual_returns)
    meta_results.append({'method': 'META_MajorityVote', 'layer': 2, **m})
    print(f"\n  META_MajorityVote        | Sharpe: {m['sharpe']:6.2f} | Win: {m['win_rate']*100:5.1f}% | DA: {m['da']*100:5.1f}% | Trades: {m['trades']:3d}")
    
    # Meta 2: Weighted by Sharpe
    sharpes = [r['sharpe'] for r in layer1_sorted[:4]]
    weights = np.array(sharpes) / sum(sharpes)
    meta_wt = weighted_vote(top_signals, weights)
    m = calc_metrics(meta_wt, actual_returns)
    meta_results.append({'method': 'META_SharpeWeighted', 'layer': 2, **m})
    print(f"  META_SharpeWeighted      | Sharpe: {m['sharpe']:6.2f} | Win: {m['win_rate']*100:5.1f}% | DA: {m['da']*100:5.1f}% | Trades: {m['trades']:3d}")
    
    # Meta 3: Agreement filter (3/4 must agree)
    meta_agree3 = agreement_filter(top_signals, min_agree=3)
    m = calc_metrics(meta_agree3, actual_returns)
    meta_results.append({'method': 'META_Agree3of4', 'layer': 2, **m})
    print(f"  META_Agree3of4           | Sharpe: {m['sharpe']:6.2f} | Win: {m['win_rate']*100:5.1f}% | DA: {m['da']*100:5.1f}% | Trades: {m['trades']:3d}")
    
    # Meta 4: Agreement filter (4/4 must agree)
    meta_agree4 = agreement_filter(top_signals, min_agree=4)
    m = calc_metrics(meta_agree4, actual_returns)
    meta_results.append({'method': 'META_Agree4of4', 'layer': 2, **m})
    print(f"  META_Agree4of4           | Sharpe: {m['sharpe']:6.2f} | Win: {m['win_rate']*100:5.1f}% | DA: {m['da']*100:5.1f}% | Trades: {m['trades']:3d}")
    
    # Meta 5: Use ALL 6 methods
    all_signals = list(methods.values())
    meta_all = majority_vote(all_signals)
    m = calc_metrics(meta_all, actual_returns)
    meta_results.append({'method': 'META_All6_MajVote', 'layer': 2, **m})
    print(f"  META_All6_MajVote        | Sharpe: {m['sharpe']:6.2f} | Win: {m['win_rate']*100:5.1f}% | DA: {m['da']*100:5.1f}% | Trades: {m['trades']:3d}")
    
    # Meta 6: All 6 with high agreement (5/6)
    meta_all_agree = agreement_filter(all_signals, min_agree=5)
    m = calc_metrics(meta_all_agree, actual_returns)
    meta_results.append({'method': 'META_All6_Agree5', 'layer': 2, **m})
    print(f"  META_All6_Agree5         | Sharpe: {m['sharpe']:6.2f} | Win: {m['win_rate']*100:5.1f}% | DA: {m['da']*100:5.1f}% | Trades: {m['trades']:3d}")
    
    # =============================================================
    # FINAL RANKING
    # =============================================================
    print("\n" + "=" * 70)
    print("FINAL RANKING (All Methods)")
    print("=" * 70)
    
    all_results = layer1_results + meta_results
    all_results_sorted = sorted(all_results, key=lambda x: -x['sharpe'])
    
    print(f"\n{'Rank':<5} {'Method':<28} {'Layer':<6} {'Sharpe':>8} {'Win%':>7} {'DA%':>7} {'Trades':>7}")
    print("-" * 70)
    
    for i, r in enumerate(all_results_sorted, 1):
        print(f"{i:<5} {r['method']:<28} {r['layer']:<6} {r['sharpe']:>8.2f} {r['win_rate']*100:>6.1f}% {r['da']*100:>6.1f}% {r['trades']:>7}")
    
    # Save to CSV
    df = pd.DataFrame(all_results_sorted)
    outfile = f"results/meta_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(outfile, index=False)
    print(f"\nResults saved to: {outfile}")
    
    # Best result
    best = all_results_sorted[0]
    print(f"\n*** BEST: {best['method']} (Layer {best['layer']}) ***")
    print(f"    Sharpe: {best['sharpe']:.2f}")
    print(f"    Win Rate: {best['win_rate']*100:.1f}%")
    print(f"    DA: {best['da']*100:.1f}%")
    print(f"    Trades: {best['trades']}")

if __name__ == '__main__':
    main()
