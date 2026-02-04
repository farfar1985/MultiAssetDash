# persona_ensemble_optimizer.py
# Optimize ensemble methods for each persona
# Include ALL models (parents + children), test everything

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from ensemble_methods import EnsembleMethods, Tier2EnsembleMethods

DATA_DIR = r"C:\Users\William Dennis\projects\nexus\data"
OUTPUT_DIR = r"C:\Users\William Dennis\projects\nexus\results"

# Persona optimization targets
PERSONAS = {
    'quant': {
        'name': 'Hardcore Quant',
        'optimize_for': 'sharpe',  # Max risk-adjusted return
        'min_trades': 50,
        'accept_complexity': True,
        'weight_consistency': 0.3
    },
    'hedging': {
        'name': 'Hedging Team',
        'optimize_for': 'min_drawdown',  # Minimize max drawdown
        'min_trades': 30,
        'accept_complexity': True,
        'weight_consistency': 0.5  # Prefer stable signals
    },
    'alpha_gen': {
        'name': 'Alpha Gen Pro',
        'optimize_for': 'return',  # Max total return
        'min_trades': 40,
        'accept_complexity': True,
        'weight_consistency': 0.2
    },
    'hedge_fund': {
        'name': 'Hedge Fund',
        'optimize_for': 'sharpe',  # Risk-adjusted
        'min_trades': 40,
        'accept_complexity': True,
        'weight_consistency': 0.4
    },
    'pro_retail': {
        'name': 'Pro Retail',
        'optimize_for': 'win_rate',  # High win rate, simpler
        'min_trades': 20,
        'accept_complexity': False,
        'weight_consistency': 0.3
    },
    'retail': {
        'name': 'Retail',
        'optimize_for': 'win_rate',  # High win rate
        'min_trades': 10,
        'accept_complexity': False,
        'weight_consistency': 0.5  # Very stable signals
    },
    'procurement': {
        'name': 'Procurement',
        'optimize_for': 'consistency',  # Stable, explainable
        'min_trades': 20,
        'accept_complexity': False,
        'weight_consistency': 0.7
    }
}


def load_horizon_data(asset_id: int, asset_name: str, horizon: int) -> tuple:
    folder = f"{asset_id}_{asset_name}"
    path = os.path.join(DATA_DIR, folder, 'horizons_wide', f'horizon_{horizon}.joblib')
    if not os.path.exists(path):
        return None, None
    data = joblib.load(path)
    return data['X'], data['y']


def compute_metrics(signals: pd.Series, actuals: pd.Series) -> dict:
    """Compute comprehensive metrics for persona optimization."""
    price_changes = actuals.diff().dropna()
    sig_aligned = signals.reindex(price_changes.index)
    
    returns = sig_aligned.shift(1) * price_changes.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns) < 5:
        return None
    
    # Sharpe
    sharpe = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252) if returns.std() > 0 else 0
    
    # Win rate
    win_rate = (returns > 0).mean() * 100
    
    # Total return
    cumulative = (1 + returns).cumprod()
    total_return = (cumulative.iloc[-1] - 1) * 100 if len(cumulative) > 0 else 0
    
    # Max drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min() * 100
    
    # Signal consistency (how often signal stays same)
    signal_changes = (sig_aligned.diff() != 0).sum()
    consistency = 1 - (signal_changes / len(sig_aligned)) if len(sig_aligned) > 0 else 0
    
    return {
        'sharpe': round(sharpe, 3),
        'win_rate': round(win_rate, 1),
        'total_return': round(total_return, 1),
        'max_dd': round(max_dd, 1),
        'consistency': round(consistency, 3),
        'n_trades': len(returns)
    }


def score_for_persona(metrics: dict, persona_config: dict) -> float:
    """Score an ensemble configuration for a specific persona."""
    if metrics is None:
        return -999
    
    if metrics['n_trades'] < persona_config['min_trades']:
        return -998
    
    optimize_for = persona_config['optimize_for']
    weight_consistency = persona_config['weight_consistency']
    
    # Base score from primary objective
    if optimize_for == 'sharpe':
        base_score = metrics['sharpe']
    elif optimize_for == 'return':
        base_score = metrics['total_return'] / 10  # Normalize
    elif optimize_for == 'win_rate':
        base_score = (metrics['win_rate'] - 50) / 10  # Center at 50%
    elif optimize_for == 'min_drawdown':
        base_score = -metrics['max_dd'] / 10  # Less drawdown = better
    elif optimize_for == 'consistency':
        base_score = metrics['consistency'] * 2
    else:
        base_score = metrics['sharpe']
    
    # Add consistency bonus
    consistency_bonus = metrics['consistency'] * weight_consistency
    
    return base_score + consistency_bonus


def test_ensemble_methods(X: pd.DataFrame, y: pd.Series) -> list:
    """Test all ensemble methods on the data."""
    results = []
    
    # Split data
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if len(X_test) < 30:
        return results
    
    tier1 = EnsembleMethods(lookback_window=60)
    tier2 = Tier2EnsembleMethods(lookback_window=60)
    
    # Tier 1 methods
    tier1_methods = [
        ('equal_weight', lambda X, y: pd.Series(1.0/X.shape[1], index=X.columns), {}),
        ('accuracy', tier1.accuracy_weighted, {}),
        ('exp_decay_90', tier1.exponential_decay_weighted, {'decay': 0.90}),
        ('exp_decay_95', tier1.exponential_decay_weighted, {'decay': 0.95}),
        ('top_k_5pct', tier1.top_k_by_sharpe, {'top_pct': 0.05}),
        ('top_k_10pct', tier1.top_k_by_sharpe, {'top_pct': 0.10}),
        ('top_k_20pct', tier1.top_k_by_sharpe, {'top_pct': 0.20}),
        ('inverse_var', tier1.inverse_variance_weighted, {}),
        ('ridge_01', tier1.ridge_stacking, {'alpha': 0.1}),
        ('ridge_1', tier1.ridge_stacking, {'alpha': 1.0}),
        ('ridge_10', tier1.ridge_stacking, {'alpha': 10.0}),
    ]
    
    for name, method, kwargs in tier1_methods:
        try:
            weights = method(X_train, y_train, **kwargs) if kwargs else method(X_train, y_train)
            ensemble_pred = (X_test * weights.reindex(X_test.columns).fillna(0)).sum(axis=1)
            signals = np.sign(ensemble_pred.diff())
            metrics = compute_metrics(signals, y_test)
            if metrics:
                results.append({'method': name, 'tier': 1, **metrics})
        except Exception as e:
            pass
    
    # Tier 2 methods (on reduced model set for speed)
    top_k_weights = tier1.top_k_by_sharpe(X_train, y_train, top_pct=0.15)
    top_models = top_k_weights[top_k_weights > 0.001].index.tolist()
    
    if len(top_models) >= 20:
        X_train_reduced = X_train[top_models]
        X_test_reduced = X_test[top_models]
        
        tier2_methods = [
            ('granger_constrained', tier2.granger_ramanathan, {'constrained': True}),
            ('granger_unconstrained', tier2.granger_ramanathan, {'constrained': False}),
            ('bma_em', tier2.bma_em, {}),
            ('quantile_50', tier2.quantile_averaging, {'quantile': 0.5}),
        ]
        
        for name, method, kwargs in tier2_methods:
            try:
                weights = method(X_train_reduced, y_train, **kwargs)
                ensemble_pred = (X_test_reduced * weights.reindex(X_test_reduced.columns).fillna(0)).sum(axis=1)
                signals = np.sign(ensemble_pred.diff())
                metrics = compute_metrics(signals, y_test)
                if metrics:
                    results.append({'method': name, 'tier': 2, **metrics})
            except Exception as e:
                pass
    
    return results


def main():
    print("=" * 70, flush=True)
    print("  PERSONA-OPTIMIZED ENSEMBLE TESTING", flush=True)
    print("  Include ALL models (parents + children)", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)
    
    ASSETS = {
        'Crude_Oil': {'id': 1866, 'horizon': 5},
        'SP500': {'id': 1625, 'horizon': 8},
        'Bitcoin': {'id': 1860, 'horizon': 5},
    }
    
    all_results = []
    
    for asset_name, info in ASSETS.items():
        print(f"\n{'='*70}", flush=True)
        print(f"  {asset_name} (horizon {info['horizon']})", flush=True)
        print(f"{'='*70}", flush=True)
        
        X, y = load_horizon_data(info['id'], asset_name, info['horizon'])
        if X is None:
            continue
        
        print(f"  Models: {X.shape[1]} (parents + children)", flush=True)
        
        # Test all methods
        method_results = test_ensemble_methods(X, y)
        
        if not method_results:
            continue
        
        # Score for each persona
        print(f"\n  BEST METHOD PER PERSONA:", flush=True)
        
        for persona_id, persona_config in PERSONAS.items():
            best_score = -999
            best_method = None
            best_metrics = None
            
            for result in method_results:
                score = score_for_persona(result, persona_config)
                if score > best_score:
                    best_score = score
                    best_method = result['method']
                    best_metrics = result
            
            if best_method:
                print(f"    {persona_config['name']:20} | {best_method:25} | Sharpe: {best_metrics['sharpe']:6.2f} | Win: {best_metrics['win_rate']:5.1f}%", flush=True)
                
                all_results.append({
                    'asset': asset_name,
                    'persona': persona_id,
                    'persona_name': persona_config['name'],
                    'best_method': best_method,
                    'score': round(best_score, 3),
                    **best_metrics
                })
    
    # Save results
    df = pd.DataFrame(all_results)
    output_path = os.path.join(OUTPUT_DIR, f'persona_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}", flush=True)
    print(f"  PERSONA OPTIMIZATION COMPLETE", flush=True)
    print(f"  Results: {output_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
