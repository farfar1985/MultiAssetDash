"""
Hybrid Quantum-Classical Ensemble
=================================
Two-stage approach:
1. Classical pre-filter: Select top K models by performance
2. Quantum refinement: Apply quantum methods to find optimal weights

This makes quantum simulation tractable (20-50 models instead of 990).

Author: AmiraB
Date: 2026-02-03
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from quantum_simulator_ensemble import QuantumSimulatorEnsemble

# Configuration
DATA_DIR = Path("data/1866_Crude_Oil/horizons_wide")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_horizon(horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load horizon data."""
    filepath = DATA_DIR / f"horizon_{horizon}.joblib"
    if not filepath.exists():
        return None, None
    
    data = joblib.load(filepath)
    X = data['X']
    y = data['y']
    
    # Convert to numpy if DataFrame
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    return X, y


def classical_prefilter(predictions: np.ndarray, actuals: np.ndarray,
                        n_select: int = 30, method: str = 'sharpe') -> np.ndarray:
    """
    Classical pre-filtering to select top models.
    
    Returns indices of selected models.
    """
    n_models = predictions.shape[1]
    
    if method == 'sharpe':
        # Compute Sharpe ratio for each model
        returns = np.diff(actuals) / actuals[:-1]
        sharpes = []
        for i in range(n_models):
            pred = predictions[:-1, i]
            signal = np.sign(pred)
            strat_ret = signal * returns
            sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)
            sharpes.append(sharpe)
        sharpes = np.array(sharpes)
        selected = np.argsort(sharpes)[-n_select:]
        
    elif method == 'mse':
        # Select by lowest MSE
        mse = np.mean((predictions - actuals.reshape(-1, 1))**2, axis=0)
        selected = np.argsort(mse)[:n_select]
        
    elif method == 'da':
        # Select by directional accuracy
        returns = np.diff(actuals)
        pred_dir = np.sign(np.diff(predictions, axis=0))
        actual_dir = np.sign(returns).reshape(-1, 1)
        da = np.mean(pred_dir == actual_dir, axis=0)
        selected = np.argsort(da)[-n_select:]
        
    elif method == 'diverse':
        # Select diverse models (maximize variance in predictions)
        # Start with best by MSE, then add most different
        mse = np.mean((predictions - actuals.reshape(-1, 1))**2, axis=0)
        selected = [np.argmin(mse)]
        
        while len(selected) < n_select:
            remaining = [i for i in range(n_models) if i not in selected]
            best_add = None
            best_diversity = -np.inf
            
            for idx in remaining:
                # Average correlation with already selected
                corrs = []
                for sel_idx in selected:
                    corr = np.corrcoef(predictions[:, idx], predictions[:, sel_idx])[0, 1]
                    corrs.append(abs(corr))
                avg_corr = np.mean(corrs)
                diversity = 1 - avg_corr
                
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_add = idx
            
            if best_add is not None:
                selected.append(best_add)
        
        selected = np.array(selected)
    
    else:
        # Random selection
        selected = np.random.choice(n_models, n_select, replace=False)
    
    return selected


def pairwise_slopes_signal(horizons_data: Dict[int, Tuple], 
                           threshold: float = 0.4) -> np.ndarray:
    """Compute pairwise slopes baseline."""
    min_len = min(d[0].shape[0] for d in horizons_data.values())
    
    signals = []
    horizon_list = sorted(horizons_data.keys())
    
    for i, h1 in enumerate(horizon_list):
        for h2 in horizon_list[i+1:]:
            pred1 = horizons_data[h1][0][:min_len].mean(axis=1)
            pred2 = horizons_data[h2][0][:min_len].mean(axis=1)
            slope = (pred2 - pred1) / (h2 - h1)
            signals.append(slope)
    
    agg = np.mean(signals, axis=0) if signals else np.zeros(min_len)
    return np.where(np.abs(agg) > threshold, np.sign(agg), 0)


def backtest(signal: np.ndarray, prices: np.ndarray) -> Dict:
    """Backtest and compute metrics."""
    returns = np.diff(prices) / prices[:-1]
    signal = signal[:len(returns)]
    
    active = signal != 0
    if not active.any():
        return {'Sharpe': 0, 'Return': 0, 'WinRate': 0, 'DA': 0, 'Trades': 0}
    
    strat_ret = signal[active] * returns[active]
    
    sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)
    total_ret = np.prod(1 + strat_ret) - 1
    win_rate = np.mean(strat_ret > 0)
    da = np.mean(signal[active] == np.sign(returns[active]))
    
    return {
        'Sharpe': round(sharpe, 3),
        'Return': round(total_ret * 100, 2),
        'WinRate': round(win_rate * 100, 1),
        'DA': round(da * 100, 1),
        'Trades': int(active.sum())
    }


def hybrid_quantum_test():
    """Run hybrid quantum-classical ensemble test."""
    print("=" * 70)
    print("HYBRID QUANTUM-CLASSICAL ENSEMBLE TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load all horizons
    horizons = [5, 7, 10]
    horizons_data = {}
    
    for h in horizons:
        X, y = load_horizon(h)
        if X is not None:
            horizons_data[h] = (X, y)
            print(f"Loaded D+{h}: {X.shape[1]} models, {X.shape[0]} days")
    
    # Use D+5 for quantum tests
    X, y = horizons_data[5]
    n_total = X.shape[1]
    n_days = X.shape[0]
    
    # Pre-filter settings
    n_quantum = 30  # Manageable for quantum simulation
    
    print(f"\nPre-filtering {n_total} models -> {n_quantum} for quantum methods")
    print("-" * 70)
    
    # Split data
    train_size = int(n_days * 0.6)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    print(f"Train: {train_size} days, Test: {n_days - train_size} days")
    
    results = []
    
    # Test different pre-filter methods
    prefilter_methods = ['sharpe', 'mse', 'da', 'diverse']
    
    for pf_method in prefilter_methods:
        print(f"\n{'='*70}")
        print(f"PRE-FILTER: {pf_method.upper()}")
        print('='*70)
        
        # Pre-filter on training data
        selected = classical_prefilter(X_train, y_train, n_quantum, pf_method)
        
        X_sel_train = X_train[:, selected]
        X_sel_test = X_test[:, selected]
        
        print(f"Selected {len(selected)} models")
        
        # Initialize quantum ensemble
        qse = QuantumSimulatorEnsemble(n_quantum, random_state=42)
        
        # Test quantum methods
        quantum_methods = [
            ('VQE', lambda: qse.vqe_ensemble(X_sel_train, y_train, n_layers=3, n_iterations=50)),
            ('QAOA', lambda: qse.qaoa_ensemble(X_sel_train, y_train, n_models_to_select=10)),
            ('SQA', lambda: qse.quantum_annealing_ensemble(X_sel_train, y_train, n_sweeps=200, n_trotter=10)),
            ('Grover', lambda: qse.grover_search_ensemble(X_sel_train, y_train)),
            ('QBM', lambda: qse.quantum_boltzmann_ensemble(X_sel_train, y_train, n_hidden=5)),
            ('TensorNet', lambda: qse.tensor_network_ensemble(X_sel_train, y_train, bond_dim=4)),
            ('QWalk', lambda: qse.quantum_walk_ensemble(X_sel_train, y_train, n_steps=30)),
        ]
        
        for q_name, q_fn in quantum_methods:
            print(f"  Testing {q_name}...", end=' ', flush=True)
            try:
                import time
                start = time.time()
                weights = q_fn()
                elapsed = time.time() - start
                
                # Apply weights to test data
                ensemble_pred = X_sel_test @ weights
                signal = np.sign(ensemble_pred)
                
                metrics = backtest(signal, y_test)
                metrics['Prefilter'] = pf_method
                metrics['Quantum'] = q_name
                metrics['Time'] = f"{elapsed:.1f}s"
                metrics['Active'] = int((weights > 0.01).sum())
                
                results.append(metrics)
                print(f"Sharpe: {metrics['Sharpe']:.3f}")
                
            except Exception as e:
                print(f"ERROR: {str(e)[:40]}")
                results.append({
                    'Prefilter': pf_method,
                    'Quantum': q_name,
                    'Sharpe': 'ERR',
                    'Return': '-',
                    'WinRate': '-',
                    'DA': '-',
                    'Trades': '-',
                    'Time': '-',
                    'Active': '-'
                })
    
    # Add baselines
    print(f"\n{'='*70}")
    print("BASELINES")
    print('='*70)
    
    # Equal weight all models
    eq_weights = np.ones(n_total) / n_total
    eq_signal = np.sign(X_test @ eq_weights)
    eq_metrics = backtest(eq_signal, y_test)
    eq_metrics['Prefilter'] = 'none'
    eq_metrics['Quantum'] = 'Equal'
    eq_metrics['Time'] = '0s'
    eq_metrics['Active'] = n_total
    results.append(eq_metrics)
    print(f"Equal Weight (all {n_total}): Sharpe {eq_metrics['Sharpe']:.3f}")
    
    # Pairwise slopes
    ps_signal = pairwise_slopes_signal(horizons_data, 0.4)
    ps_signal_test = ps_signal[train_size:train_size + len(y_test)]
    ps_metrics = backtest(ps_signal_test, y_test)
    ps_metrics['Prefilter'] = 'none'
    ps_metrics['Quantum'] = 'PairSlopes*'
    ps_metrics['Time'] = '0.1s'
    ps_metrics['Active'] = 'N/A'
    results.append(ps_metrics)
    print(f"Pairwise Slopes (D+5+7+10): Sharpe {ps_metrics['Sharpe']:.3f}")
    
    # Results DataFrame
    df = pd.DataFrame(results)
    df = df[['Prefilter', 'Quantum', 'Sharpe', 'Return', 'WinRate', 'DA', 'Trades', 'Active', 'Time']]
    
    # Sort by Sharpe
    def safe_sharpe(x):
        try:
            return float(x)
        except:
            return -999
    
    df['_s'] = df['Sharpe'].apply(safe_sharpe)
    df = df.sort_values('_s', ascending=False).drop('_s', axis=1)
    
    print(f"\n{'='*70}")
    print("ALL RESULTS (sorted by Sharpe)")
    print('='*70)
    print(df.to_string(index=False))
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f'hybrid_quantum_results_{timestamp}.csv'
    df.to_csv(result_file, index=False)
    print(f"\nSaved to: {result_file}")
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print('='*70)
    
    numeric = [(r['Prefilter'], r['Quantum'], r['Sharpe']) 
               for r in results if isinstance(r['Sharpe'], (int, float))]
    
    if numeric:
        best = max(numeric, key=lambda x: x[2])
        print(f"Best: {best[0]}+{best[1]} (Sharpe: {best[2]:.3f})")
        
        ps_sharpe = ps_metrics.get('Sharpe', 0)
        if isinstance(ps_sharpe, (int, float)):
            better = [f"{p}+{q}" for p, q, s in numeric 
                     if s > ps_sharpe and q != 'PairSlopes*']
            if better:
                print(f"\n🎉 {len(better)} hybrid methods beat pairwise slopes!")
                for m in better[:5]:
                    print(f"   - {m}")
            else:
                print("\nPairwise slopes still wins.")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df


if __name__ == '__main__':
    hybrid_quantum_test()
