"""
Find the BEST Ensemble Configuration
=====================================
Comprehensive benchmark of ALL methods on Crude Oil data.
Goal: Find the absolute best-performing ensemble.

Created: 2026-02-03
Author: AmiraB
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our ensemble methods
try:
    from advanced_ensemble import MetaEnsemble, HedgeAlgorithm, MinTReconciliation
    from quantum_ensemble import QuantumAnnealingEnsemble, AttentionEnsemble, AdaptiveConformalEnsemble
    ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some advanced methods unavailable: {e}")
    ADVANCED_AVAILABLE = False


def load_all_horizons():
    """Load all Crude Oil horizon data."""
    data_dir = Path("data/1866_Crude_Oil/horizons_wide")
    
    horizons = {}
    for h in range(1, 11):
        f = data_dir / f"horizon_{h}.joblib"
        if f.exists():
            data = joblib.load(f)
            X = data['X'].values if hasattr(data['X'], 'values') else data['X']
            y = data['y'].values if hasattr(data['y'], 'values') else data['y']
            horizons[h] = {'X': np.array(X), 'y': np.array(y)}
    
    return horizons


def compute_metrics(predictions: np.ndarray, actuals: np.ndarray, 
                   train_end: int = None) -> dict:
    """Compute comprehensive performance metrics."""
    if train_end is None:
        train_end = int(len(actuals) * 0.7)
    
    # Out-of-sample only
    oos_pred = predictions[train_end:]
    oos_actual = actuals[train_end:]
    
    # Remove NaN
    valid = ~(np.isnan(oos_pred) | np.isnan(oos_actual))
    oos_pred = oos_pred[valid]
    oos_actual = oos_actual[valid]
    
    if len(oos_pred) < 10:
        return {'sharpe': -999, 'da': 0, 'mse': 999999, 'returns': 0}
    
    # Directional accuracy
    pred_dir = np.sign(np.diff(oos_pred))
    actual_dir = np.sign(np.diff(oos_actual))
    da = (pred_dir == actual_dir).mean()
    
    # MSE
    mse = np.mean((oos_pred - oos_actual) ** 2)
    
    # Sharpe-like metric (signal quality)
    # Treat predictions as signals: profit when signal matches direction
    returns = np.sign(oos_pred[:-1]) * actual_dir
    avg_ret = returns.mean()
    std_ret = returns.std() + 1e-8
    sharpe = avg_ret / std_ret * np.sqrt(252)
    
    # Total return
    total_return = returns.sum()
    
    return {
        'sharpe': round(sharpe, 3),
        'da': round(da * 100, 1),
        'mse': round(mse, 4),
        'returns': round(total_return, 2),
        'n_samples': len(oos_pred)
    }


def test_tier1_methods(X: np.ndarray, y: np.ndarray) -> list:
    """Test all Tier 1 ensemble methods."""
    results = []
    n_samples, n_models = X.shape
    train_end = int(n_samples * 0.7)
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    # Need historical accuracy for some methods
    # Compute simple model accuracy (directional)
    model_accuracies = []
    for m in range(n_models):
        pred_dir = np.sign(np.diff(X[:train_end, m]))
        actual_dir = np.sign(np.diff(y[:train_end]))
        acc = (pred_dir == actual_dir).mean()
        model_accuracies.append(acc)
    model_accuracies = np.array(model_accuracies)
    
    # Model Sharpe ratios
    model_sharpes = []
    for m in range(n_models):
        returns = np.sign(X[:train_end, m][:-1]) * np.sign(np.diff(y[:train_end]))
        sharpe = returns.mean() / (returns.std() + 1e-8)
        model_sharpes.append(sharpe)
    model_sharpes = np.array(model_sharpes)
    
    # 1. Equal Weight
    pred = X.mean(axis=1)
    metrics = compute_metrics(pred, y, train_end)
    results.append({'method': 'equal_weight', **metrics})
    
    # 2. Top-K by Accuracy (various K)
    for k_pct in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(n_models * k_pct))
        top_idx = np.argsort(model_accuracies)[-k:]
        pred = X[:, top_idx].mean(axis=1)
        metrics = compute_metrics(pred, y, train_end)
        results.append({'method': f'top_k_acc_{int(k_pct*100)}pct', **metrics})
    
    # 3. Top-K by Sharpe (various K)
    for k_pct in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(n_models * k_pct))
        top_idx = np.argsort(model_sharpes)[-k:]
        pred = X[:, top_idx].mean(axis=1)
        metrics = compute_metrics(pred, y, train_end)
        results.append({'method': f'top_k_sharpe_{int(k_pct*100)}pct', **metrics})
    
    # 4. Inverse Variance
    variances = X[:train_end].var(axis=0) + 1e-8
    weights = 1.0 / variances
    weights = weights / weights.sum()
    pred = X @ weights
    metrics = compute_metrics(pred, y, train_end)
    results.append({'method': 'inverse_variance', **metrics})
    
    # 5. Exponential Decay
    for decay in [0.9, 0.95, 0.99]:
        decay_weights = np.array([decay ** i for i in range(n_models)])[::-1]
        decay_weights = decay_weights / decay_weights.sum()
        pred = X @ decay_weights
        metrics = compute_metrics(pred, y, train_end)
        results.append({'method': f'exp_decay_{decay}', **metrics})
    
    # 6. Ridge Stack
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X[:train_end], y[:train_end])
    pred = ridge.predict(X)
    metrics = compute_metrics(pred, y, train_end)
    results.append({'method': 'ridge_stack', **metrics})
    
    # 7. Median (robust to outliers)
    pred = np.median(X, axis=1)
    metrics = compute_metrics(pred, y, train_end)
    results.append({'method': 'median', **metrics})
    
    # 8. Trimmed Mean (remove top/bottom 10%)
    pred = np.array([np.mean(np.sort(row)[int(n_models*0.1):int(n_models*0.9)]) for row in X])
    metrics = compute_metrics(pred, y, train_end)
    results.append({'method': 'trimmed_mean_10pct', **metrics})
    
    return results


def test_advanced_methods(X: np.ndarray, y: np.ndarray) -> list:
    """Test advanced ensemble methods."""
    if not ADVANCED_AVAILABLE:
        return []
    
    results = []
    n_samples, n_models = X.shape
    train_end = int(n_samples * 0.7)
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    # 1. Quantum Annealing (fast version)
    print("    Testing Quantum Annealing...")
    try:
        qa = QuantumAnnealingEnsemble(n_iterations=300)
        qa.fit(X[:train_end], y[:train_end])
        pred = qa.predict(X)
        metrics = compute_metrics(pred, y, train_end)
        results.append({'method': 'quantum_annealing', **metrics})
    except Exception as e:
        print(f"      Error: {e}")
        results.append({'method': 'quantum_annealing', 'sharpe': -999, 'da': 0, 'mse': 999999, 'returns': 0})
    
    # 2. Attention-Based
    print("    Testing Attention Ensemble...")
    try:
        attn = AttentionEnsemble(n_heads=4, context_window=20)
        attn.fit(X[:train_end], y[:train_end])
        pred = attn.predict(X)
        metrics = compute_metrics(pred, y, train_end)
        results.append({'method': 'attention_ensemble', **metrics})
    except Exception as e:
        print(f"      Error: {e}")
        results.append({'method': 'attention_ensemble', 'sharpe': -999, 'da': 0, 'mse': 999999, 'returns': 0})
    
    # 3. Hedge Algorithm
    print("    Testing Hedge Algorithm...")
    try:
        hedge = HedgeAlgorithm(n_experts=min(n_models, 100), learning_rate=0.1)
        # Train by updating on historical data
        for i in range(train_end):
            expert_preds = X[i, :min(n_models, 100)]
            hedge.update(expert_preds, y[i], loss_fn='squared')
        # Predict using final weights
        weights = hedge.get_weights()
        if len(weights) < n_models:
            full_weights = np.ones(n_models) / n_models
            full_weights[:len(weights)] = weights
            weights = full_weights / full_weights.sum()
        pred = X @ weights
        metrics = compute_metrics(pred, y, train_end)
        results.append({'method': 'hedge_algorithm', **metrics})
    except Exception as e:
        print(f"      Error: {e}")
        results.append({'method': 'hedge_algorithm', 'sharpe': -999, 'da': 0, 'mse': 999999, 'returns': 0})
    
    # 4. Meta-Ensemble
    print("    Testing Meta-Ensemble...")
    try:
        df = pd.DataFrame(X, columns=[f'm{i}' for i in range(n_models)])
        meta = MetaEnsemble(meta_method='hedge')
        meta.fit(df.iloc[:train_end], pd.Series(y[:train_end]))
        pred = meta.predict(df)
        metrics = compute_metrics(pred, y, train_end)
        results.append({'method': 'meta_ensemble_hedge', **metrics})
    except Exception as e:
        print(f"      Error: {e}")
        results.append({'method': 'meta_ensemble_hedge', 'sharpe': -999, 'da': 0, 'mse': 999999, 'returns': 0})
    
    return results


def test_cross_horizon_ensemble(horizons: dict) -> list:
    """Test cross-horizon ensemble strategies."""
    results = []
    
    # Get common time range
    min_len = min(h['X'].shape[0] for h in horizons.values())
    train_end = int(min_len * 0.7)
    
    # Use y from horizon 1 as actuals (or any horizon - they should be similar)
    y = horizons[1]['y'][-min_len:]
    
    print("    Testing Cross-Horizon Ensembles...")
    
    # 1. Simple average across all horizons
    all_preds = []
    for h, data in horizons.items():
        X = data['X'][-min_len:]
        all_preds.append(X.mean(axis=1))  # Average within horizon
    
    combined = np.column_stack(all_preds)
    pred = combined.mean(axis=1)  # Average across horizons
    metrics = compute_metrics(pred, y, train_end)
    results.append({'method': 'cross_horizon_equal', **metrics})
    
    # 2. Magnitude-weighted (weight by forecast magnitude)
    magnitudes = np.abs(combined).mean(axis=0)
    weights = magnitudes / magnitudes.sum()
    pred = combined @ weights
    metrics = compute_metrics(pred, y, train_end)
    results.append({'method': 'cross_horizon_magnitude', **metrics})
    
    # 3. Best individual horizon
    best_horizon_sharpe = -999
    best_horizon = None
    for h, data in horizons.items():
        X = data['X'][-min_len:]
        pred = X.mean(axis=1)
        metrics = compute_metrics(pred, y, train_end)
        if metrics['sharpe'] > best_horizon_sharpe:
            best_horizon_sharpe = metrics['sharpe']
            best_horizon = h
    
    results.append({'method': f'best_single_horizon_D+{best_horizon}', 
                   'sharpe': best_horizon_sharpe, 'da': 0, 'mse': 0, 'returns': 0})
    
    # 4. Long horizons only (D+5 to D+10) - Farzaneh's direction
    long_horizons = [h for h in horizons.keys() if h >= 5]
    if long_horizons:
        long_preds = []
        for h in long_horizons:
            X = horizons[h]['X'][-min_len:]
            long_preds.append(X.mean(axis=1))
        combined_long = np.column_stack(long_preds)
        pred = combined_long.mean(axis=1)
        metrics = compute_metrics(pred, y, train_end)
        results.append({'method': 'cross_horizon_long_only_D5-10', **metrics})
    
    # 5. Horizon pairs with high correlation agreement
    # Find horizon pairs that agree most often
    agreement_scores = {}
    for h1 in horizons.keys():
        for h2 in horizons.keys():
            if h2 > h1:
                pred1 = horizons[h1]['X'][-min_len:].mean(axis=1)
                pred2 = horizons[h2]['X'][-min_len:].mean(axis=1)
                dir1 = np.sign(np.diff(pred1))
                dir2 = np.sign(np.diff(pred2))
                agreement = (dir1 == dir2).mean()
                agreement_scores[(h1, h2)] = agreement
    
    # Best agreeing pair
    best_pair = max(agreement_scores, key=agreement_scores.get)
    h1, h2 = best_pair
    pred1 = horizons[h1]['X'][-min_len:].mean(axis=1)
    pred2 = horizons[h2]['X'][-min_len:].mean(axis=1)
    pred = (pred1 + pred2) / 2
    metrics = compute_metrics(pred, y, train_end)
    results.append({'method': f'best_agreeing_pair_D+{h1}_D+{h2}', **metrics,
                   'agreement': round(agreement_scores[best_pair], 3)})
    
    return results


def main():
    print("=" * 70)
    print("COMPREHENSIVE ENSEMBLE BENCHMARK")
    print("Finding the BEST ensemble configuration for Crude Oil")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    horizons = load_all_horizons()
    
    if not horizons:
        print("ERROR: No data loaded")
        return
    
    print(f"Loaded {len(horizons)} horizons")
    
    all_results = []
    
    # Test on each horizon individually
    for h in sorted(horizons.keys()):
        print(f"\n--- Horizon D+{h} ({horizons[h]['X'].shape[1]} models) ---")
        
        X = horizons[h]['X']
        y = horizons[h]['y']
        
        # Tier 1 methods
        print("  Tier 1 Methods...")
        tier1_results = test_tier1_methods(X, y)
        for r in tier1_results:
            r['horizon'] = f'D+{h}'
        all_results.extend(tier1_results)
        
        # Advanced methods (only on a few horizons to save time)
        if h in [1, 5, 10]:
            print("  Advanced Methods...")
            advanced_results = test_advanced_methods(X, y)
            for r in advanced_results:
                r['horizon'] = f'D+{h}'
            all_results.extend(advanced_results)
    
    # Cross-horizon ensembles
    print("\n--- Cross-Horizon Ensembles ---")
    cross_results = test_cross_horizon_ensemble(horizons)
    for r in cross_results:
        r['horizon'] = 'cross'
    all_results.extend(cross_results)
    
    # Convert to DataFrame and analyze
    df = pd.DataFrame(all_results)
    
    # Filter out failed methods
    df = df[df['sharpe'] > -900]
    
    print("\n" + "=" * 70)
    print("TOP 20 ENSEMBLE CONFIGURATIONS BY SHARPE")
    print("=" * 70)
    
    top20 = df.nlargest(20, 'sharpe')[['horizon', 'method', 'sharpe', 'da', 'returns']]
    print(top20.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("TOP 10 BY DIRECTIONAL ACCURACY")
    print("=" * 70)
    
    top10_da = df.nlargest(10, 'da')[['horizon', 'method', 'sharpe', 'da', 'returns']]
    print(top10_da.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("TOP 10 BY TOTAL RETURNS")
    print("=" * 70)
    
    top10_ret = df.nlargest(10, 'returns')[['horizon', 'method', 'sharpe', 'da', 'returns']]
    print(top10_ret.to_string(index=False))
    
    # Find THE BEST
    best = df.loc[df['sharpe'].idxmax()]
    print("\n" + "=" * 70)
    print(">>> THE BEST ENSEMBLE CONFIGURATION <<<")
    print("=" * 70)
    print(f"Horizon: {best['horizon']}")
    print(f"Method:  {best['method']}")
    print(f"Sharpe:  {best['sharpe']}")
    print(f"DA:      {best['da']}%")
    print(f"Returns: {best['returns']}")
    
    # Save results
    output_file = Path("results") / f"ensemble_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_file.parent.mkdir(exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")
    
    # Also save just the top results
    top_file = Path("results") / "best_ensemble_configs.csv"
    df.nlargest(50, 'sharpe').to_csv(top_file, index=False)
    print(f"Top 50 configs saved to: {top_file}")
    
    return df


if __name__ == "__main__":
    results = main()
