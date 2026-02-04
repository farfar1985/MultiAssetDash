"""
Benchmark Advanced Ensemble Methods on Crude Oil D+5 Data
=========================================================
Tests: QuantumAnnealingEnsemble, WassersteinEnsemble, 
       AdaptiveConformalEnsemble, AttentionEnsemble

Baseline: Simple Mean DA ~35%, Pairwise Slopes Sharpe = 1.757
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_ensemble import (
    QuantumAnnealingEnsemble,
    WassersteinEnsemble,
    AdaptiveConformalEnsemble,
    AttentionEnsemble,
    NeuralEnsemble
)


def compute_directional_accuracy(predictions, actuals):
    """Compute directional accuracy."""
    pred_dir = np.sign(np.diff(predictions))
    actual_dir = np.sign(np.diff(actuals))
    # Filter out NaN/Inf
    valid = np.isfinite(pred_dir) & np.isfinite(actual_dir)
    if valid.sum() < 1:
        return 0.0
    return (pred_dir[valid] == actual_dir[valid]).mean()


def compute_sharpe(predictions, actuals, annualize=True):
    """Compute Sharpe ratio of trading strategy."""
    pred_dir = np.sign(np.diff(predictions))
    actual_changes = np.diff(actuals)
    returns = pred_dir * actual_changes
    
    # Filter out NaN/Inf
    valid = np.isfinite(returns)
    if valid.sum() < 10:
        return 0.0
    returns = returns[valid]
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    if std_ret < 1e-10:
        return 0.0
    
    sharpe = mean_ret / std_ret
    if annualize:
        sharpe *= np.sqrt(252)
    
    return sharpe


def compute_total_return(predictions, actuals):
    """Compute total return percentage of strategy."""
    pred_dir = np.sign(np.diff(predictions))
    actual_changes = np.diff(actuals)
    returns = pred_dir * actual_changes
    
    # Simple cumulative return
    total = returns.sum()
    # As percentage of initial price
    return_pct = (total / abs(actuals[0])) * 100
    
    return return_pct


def benchmark_method(name, predictions, actuals, baseline_sharpe=1.757):
    """Compute all metrics for a method."""
    da = compute_directional_accuracy(predictions, actuals)
    sharpe = compute_sharpe(predictions, actuals)
    ret_pct = compute_total_return(predictions, actuals)
    vs_baseline = "BEATS" if sharpe > baseline_sharpe else "BELOW"
    
    return {
        'method': name,
        'directional_accuracy': round(da, 4),
        'sharpe': round(sharpe, 4),
        'return_pct': round(ret_pct, 2),
        'vs_baseline': vs_baseline
    }


def main():
    print("=" * 70)
    print("ADVANCED ENSEMBLE BENCHMARK - Crude Oil D+5")
    print("=" * 70)
    
    # Load data
    data_path = "data/1866_Crude_Oil/horizons_wide/horizon_5.joblib"
    print(f"\nLoading data from: {data_path}")
    
    data = joblib.load(data_path)
    
    # Handle different data formats
    if isinstance(data, dict):
        X = data.get('X', data.get('predictions'))
        y = data.get('y', data.get('actuals'))
    elif isinstance(data, tuple):
        X, y = data[0], data[1]
    else:
        raise ValueError(f"Unknown data format: {type(data)}")
    
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Remove rows with NaN in y
    nan_mask = np.isnan(y)
    if nan_mask.any():
        print(f"Removing {nan_mask.sum()} rows with NaN in y")
        valid_idx = ~nan_mask
        X = X[valid_idx]
        y = y[valid_idx]
        print(f"After cleaning: X shape={X.shape}, y shape={y.shape}")
    
    # Baselines
    print("\n" + "-" * 70)
    print("BASELINES")
    print("-" * 70)
    
    BASELINE_SHARPE = 1.757
    
    # Simple Mean baseline
    simple_mean = X.mean(axis=1)
    mean_result = benchmark_method("Simple_Mean", simple_mean, y, BASELINE_SHARPE)
    print(f"Simple Mean: DA={mean_result['directional_accuracy']:.2%}, Sharpe={mean_result['sharpe']:.3f}")
    
    results = [mean_result]
    
    # Pairwise Slopes (reference only - we don't have its predictions here)
    results.append({
        'method': 'Pairwise_Slopes_REFERENCE',
        'directional_accuracy': None,
        'sharpe': BASELINE_SHARPE,
        'return_pct': None,
        'vs_baseline': 'BASELINE'
    })
    
    # Test each advanced method
    print("\n" + "-" * 70)
    print("ADVANCED METHODS")
    print("-" * 70)
    
    # 1. Quantum Annealing Ensemble
    print("\n1. QUANTUM ANNEALING ENSEMBLE")
    try:
        qa = QuantumAnnealingEnsemble(n_iterations=500)
        qa.fit(X, y)
        qa_pred = qa.predict(X)
        
        if np.isfinite(qa_pred).all():
            qa_result = benchmark_method("QuantumAnnealing", qa_pred, y, BASELINE_SHARPE)
            print(f"   DA={qa_result['directional_accuracy']:.2%}, Sharpe={qa_result['sharpe']:.3f}, {qa_result['vs_baseline']}")
            results.append(qa_result)
        else:
            print("   ERROR: NaN in predictions")
            results.append({'method': 'QuantumAnnealing', 'directional_accuracy': None, 'sharpe': None, 'return_pct': None, 'vs_baseline': 'ERROR'})
    except Exception as e:
        print(f"   ERROR: {e}")
        results.append({'method': 'QuantumAnnealing', 'directional_accuracy': None, 'sharpe': None, 'return_pct': None, 'vs_baseline': 'ERROR'})
    
    # 2. Wasserstein Ensemble
    print("\n2. WASSERSTEIN ENSEMBLE")
    try:
        wass = WassersteinEnsemble(n_bins=100, n_iter=30)
        wass.fit(X, y)
        
        # Wasserstein gives a single distribution - use weighted model average instead
        # Get the barycenter weights and apply to original predictions
        if wass.barycenter_weights is not None:
            wass_pred = X @ wass.barycenter_weights
        else:
            # Use distribution center as offset
            wass_pred = simple_mean  # fallback
        
        wass_result = benchmark_method("Wasserstein", wass_pred, y, BASELINE_SHARPE)
        print(f"   DA={wass_result['directional_accuracy']:.2%}, Sharpe={wass_result['sharpe']:.3f}, {wass_result['vs_baseline']}")
        results.append(wass_result)
    except Exception as e:
        print(f"   ERROR: {e}")
        results.append({'method': 'Wasserstein', 'directional_accuracy': None, 'sharpe': None, 'return_pct': None, 'vs_baseline': 'ERROR'})
    
    # 3. Attention Ensemble
    print("\n3. ATTENTION ENSEMBLE")
    try:
        attn = AttentionEnsemble(n_heads=4, context_window=20)
        attn.fit(X, y)
        attn_pred = attn.predict(X)
        
        attn_result = benchmark_method("Attention", attn_pred, y, BASELINE_SHARPE)
        print(f"   DA={attn_result['directional_accuracy']:.2%}, Sharpe={attn_result['sharpe']:.3f}, {attn_result['vs_baseline']}")
        results.append(attn_result)
    except Exception as e:
        print(f"   ERROR: {e}")
        results.append({'method': 'Attention', 'directional_accuracy': None, 'sharpe': None, 'return_pct': None, 'vs_baseline': 'ERROR'})
    
    # 4. Conformal Ensemble (uses simple mean as base, adds intervals)
    print("\n4. ADAPTIVE CONFORMAL ENSEMBLE")
    try:
        conf = AdaptiveConformalEnsemble(coverage=0.90, window_size=50)
        conf.fit(simple_mean, y)
        
        # Conformal doesn't change point predictions, but we can report coverage
        coverage = conf.get_current_coverage()
        print(f"   (Uses Simple Mean for point prediction)")
        print(f"   Empirical coverage: {coverage:.2%}")
        
        # Use the simple mean result but note it's conformal
        conf_result = mean_result.copy()
        conf_result['method'] = 'Conformal'
        conf_result['vs_baseline'] = f"COVERAGE={coverage:.2%}"
        results.append(conf_result)
    except Exception as e:
        print(f"   ERROR: {e}")
        results.append({'method': 'Conformal', 'directional_accuracy': None, 'sharpe': None, 'return_pct': None, 'vs_baseline': 'ERROR'})
    
    # 5. Neural Ensemble
    print("\n5. NEURAL ENSEMBLE")
    try:
        neural = NeuralEnsemble(hidden_size=64, learning_rate=0.001, n_epochs=100)
        neural.fit(X, y)
        neural_pred = neural.predict(X)
        
        neural_result = benchmark_method("Neural", neural_pred, y, BASELINE_SHARPE)
        print(f"   DA={neural_result['directional_accuracy']:.2%}, Sharpe={neural_result['sharpe']:.3f}, {neural_result['vs_baseline']}")
        results.append(neural_result)
    except Exception as e:
        print(f"   ERROR: {e}")
        results.append({'method': 'Neural', 'directional_accuracy': None, 'sharpe': None, 'return_pct': None, 'vs_baseline': 'ERROR'})
    
    # Save results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save to CSV
    os.makedirs("results", exist_ok=True)
    output_path = "results/advanced_ensemble_benchmark.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Report winners
    print("\n" + "-" * 70)
    print("METHODS BEATING BASELINE (Sharpe > 1.757):")
    print("-" * 70)
    
    winners = [r for r in results if r.get('sharpe') is not None 
               and r['sharpe'] > BASELINE_SHARPE 
               and r['method'] != 'Pairwise_Slopes_REFERENCE']
    
    if winners:
        for w in sorted(winners, key=lambda x: x['sharpe'], reverse=True):
            print(f"  ✓ {w['method']}: Sharpe={w['sharpe']:.3f} (+{w['sharpe']-BASELINE_SHARPE:.3f})")
    else:
        print("  No methods beat the pairwise slopes baseline.")
    
    return results_df


if __name__ == "__main__":
    main()
