"""
META-ENSEMBLE: Ensemble of Ensembles with Regime Adaptation
============================================================
Bill's insight: Use the BEST of the BEST, optimized per market regime.

Architecture:
- Level 1: Multiple ensemble methods per asset
- Level 2: Meta-learner combines L1 outputs  
- Level 3: Regime detector selects optimal strategy

Created: 2026-02-05
Author: AmiraB (Bill's direction)
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from master_ensemble import calculate_pairwise_slope_signal


class RegimeDetector:
    """
    Detects market regime based on recent price action.
    
    Regimes:
    - TRENDING_UP: Strong upward momentum
    - TRENDING_DOWN: Strong downward momentum  
    - MEAN_REVERTING: Choppy, oscillating
    - HIGH_VOLATILITY: Large swings
    - LOW_VOLATILITY: Calm, steady
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def detect(self, prices: np.ndarray, t: int) -> str:
        """Detect regime at time t based on lookback window."""
        if t < self.lookback:
            return 'UNKNOWN'
        
        window = prices[t - self.lookback:t]
        
        # Calculate metrics
        returns = np.diff(window) / window[:-1]
        
        # Trend: cumulative return over window
        cum_return = (window[-1] - window[0]) / window[0]
        
        # Volatility: std of returns
        volatility = returns.std()
        
        # Mean reversion: autocorrelation of returns
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            autocorr = 0
        
        # Classify regime
        vol_threshold = 0.02  # 2% daily vol is high
        trend_threshold = 0.05  # 5% cumulative is trending
        
        if volatility > vol_threshold:
            return 'HIGH_VOLATILITY'
        elif volatility < vol_threshold / 2:
            return 'LOW_VOLATILITY'
        elif cum_return > trend_threshold:
            return 'TRENDING_UP'
        elif cum_return < -trend_threshold:
            return 'TRENDING_DOWN'
        elif autocorr < -0.2:  # Negative autocorr = mean reverting
            return 'MEAN_REVERTING'
        else:
            return 'NEUTRAL'
    
    def get_regime_features(self, prices: np.ndarray, t: int) -> dict:
        """Extract regime features for ML model."""
        if t < self.lookback:
            return {'trend': 0, 'volatility': 0, 'autocorr': 0, 'momentum': 0}
        
        window = prices[t - self.lookback:t]
        returns = np.diff(window) / window[:-1]
        
        cum_return = (window[-1] - window[0]) / window[0]
        volatility = returns.std()
        
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0
        else:
            autocorr = 0
        
        # Short-term momentum (last 5 days)
        short_window = min(5, len(window))
        momentum = (window[-1] - window[-short_window]) / window[-short_window] if short_window > 0 else 0
        
        return {
            'trend': cum_return,
            'volatility': volatility,
            'autocorr': autocorr,
            'momentum': momentum
        }


class EnsembleMethod:
    """Base class for ensemble methods."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_signal(self, horizons: dict, t: int, horizon_subset: list) -> dict:
        raise NotImplementedError


class PairwiseSlopesEnsemble(EnsembleMethod):
    """Original pairwise slopes method."""
    
    def __init__(self):
        super().__init__('pairwise_slopes')
    
    def get_signal(self, horizons: dict, t: int, horizon_subset: list) -> dict:
        return calculate_pairwise_slope_signal(horizons, t, horizon_subset, 'mean')


class AccuracyWeightedEnsemble(EnsembleMethod):
    """Weight models by their rolling accuracy."""
    
    def __init__(self, accuracy_window: int = 20):
        super().__init__('accuracy_weighted')
        self.accuracy_window = accuracy_window
        self.accuracy_cache = {}
    
    def get_signal(self, horizons: dict, t: int, horizon_subset: list) -> dict:
        if t < self.accuracy_window + 1:
            # Fall back to simple mean
            return calculate_pairwise_slope_signal(horizons, t, horizon_subset, 'mean')
        
        weighted_pred = 0
        total_weight = 0
        
        for h in horizon_subset:
            if h not in horizons or t >= len(horizons[h]['X']):
                continue
            
            X = horizons[h]['X']
            y = horizons[h]['y']
            
            # Calculate rolling accuracy
            correct = 0
            total = 0
            for i in range(t - self.accuracy_window, t):
                if i < h or i >= len(y):
                    continue
                pred_dir = np.sign(X[i].mean() - y[i - h]) if i >= h else 0
                actual_dir = np.sign(y[i] - y[i - h]) if i >= h else 0
                if pred_dir == actual_dir:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0.5
            weight = accuracy ** 2  # Square to emphasize good models
            
            pred = X[t].mean()
            weighted_pred += pred * weight
            total_weight += weight
        
        if total_weight == 0:
            return {'signal': 0, 'net_prob': 0}
        
        avg_pred = weighted_pred / total_weight
        
        # Compare to shortest horizon's current price proxy
        min_h = min(horizon_subset)
        current_proxy = horizons[min_h]['y'][t - min_h] if t >= min_h else horizons[min_h]['y'][0]
        
        direction = np.sign(avg_pred - current_proxy)
        confidence = abs(avg_pred - current_proxy) / (current_proxy + 1e-8)
        
        return {
            'signal': int(direction),
            'net_prob': direction * min(confidence * 10, 1.0),  # Scale to [-1, 1]
            'method': self.name
        }


class MagnitudeWeightedEnsemble(EnsembleMethod):
    """Weight by prediction magnitude (stronger predictions count more)."""
    
    def __init__(self):
        super().__init__('magnitude_weighted')
    
    def get_signal(self, horizons: dict, t: int, horizon_subset: list) -> dict:
        votes_up = 0
        votes_down = 0
        
        for h in horizon_subset:
            if h not in horizons or t >= len(horizons[h]['X']):
                continue
            
            X = horizons[h]['X']
            y = horizons[h]['y']
            
            pred = X[t].mean()
            current = y[t - h] if t >= h else y[0]
            
            pct_change = (pred - current) / (current + 1e-8)
            magnitude = abs(pct_change)
            
            if pct_change > 0:
                votes_up += magnitude
            else:
                votes_down += magnitude
        
        total = votes_up + votes_down
        if total == 0:
            return {'signal': 0, 'net_prob': 0, 'method': self.name}
        
        net_prob = (votes_up - votes_down) / total
        signal = 1 if net_prob > 0 else (-1 if net_prob < 0 else 0)
        
        return {
            'signal': signal,
            'net_prob': net_prob,
            'method': self.name
        }


class StackingEnsemble(EnsembleMethod):
    """Meta-learner that combines horizon predictions."""
    
    def __init__(self):
        super().__init__('stacking')
        self.model = Ridge(alpha=1.0)
        self.is_trained = False
        self.scaler = StandardScaler()
    
    def train(self, horizons: dict, horizon_subset: list, train_end: int):
        """Train the meta-learner on historical data."""
        X_train = []
        y_train = []
        
        min_h = min(horizon_subset)
        
        for t in range(max(horizon_subset) + 10, train_end):
            features = []
            for h in horizon_subset:
                if h in horizons and t < len(horizons[h]['X']):
                    features.append(horizons[h]['X'][t].mean())
                else:
                    features.append(0)
            
            if len(features) == len(horizon_subset):
                X_train.append(features)
                # Target: actual return
                if t < len(horizons[min_h]['y']) and t >= min_h:
                    y_train.append(horizons[min_h]['y'][t] - horizons[min_h]['y'][t - min_h])
        
        if len(X_train) > 10:
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_train = self.scaler.fit_transform(X_train)
            self.model.fit(X_train, y_train)
            self.is_trained = True
    
    def get_signal(self, horizons: dict, t: int, horizon_subset: list) -> dict:
        if not self.is_trained:
            return calculate_pairwise_slope_signal(horizons, t, horizon_subset, 'mean')
        
        features = []
        for h in horizon_subset:
            if h in horizons and t < len(horizons[h]['X']):
                features.append(horizons[h]['X'][t].mean())
            else:
                features.append(0)
        
        features = np.array(features).reshape(1, -1)
        features = self.scaler.transform(features)
        
        pred = self.model.predict(features)[0]
        signal = 1 if pred > 0 else (-1 if pred < 0 else 0)
        confidence = min(abs(pred) / 5, 1.0)  # Normalize
        
        return {
            'signal': signal,
            'net_prob': signal * confidence,
            'method': self.name
        }


class MetaEnsemble:
    """
    The SUPER ENSEMBLE that combines all ensemble methods.
    Uses regime detection to select the best strategy.
    """
    
    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.ensemble_methods = [
            PairwiseSlopesEnsemble(),
            AccuracyWeightedEnsemble(),
            MagnitudeWeightedEnsemble(),
            StackingEnsemble()
        ]
        
        # Track performance by regime
        self.regime_performance = {
            'TRENDING_UP': {m.name: [] for m in self.ensemble_methods},
            'TRENDING_DOWN': {m.name: [] for m in self.ensemble_methods},
            'MEAN_REVERTING': {m.name: [] for m in self.ensemble_methods},
            'HIGH_VOLATILITY': {m.name: [] for m in self.ensemble_methods},
            'LOW_VOLATILITY': {m.name: [] for m in self.ensemble_methods},
            'NEUTRAL': {m.name: [] for m in self.ensemble_methods},
            'UNKNOWN': {m.name: [] for m in self.ensemble_methods},
        }
        
        # Meta-learner for regime-based selection
        self.regime_selector = None
        self.is_trained = False
    
    def train(self, horizons: dict, current_prices: np.ndarray, 
              horizon_subset: list, train_end: int):
        """
        Train all ensemble methods and learn regime-based selection.
        """
        print("  Training stacking ensemble...")
        # Train stacking method
        for method in self.ensemble_methods:
            if isinstance(method, StackingEnsemble):
                method.train(horizons, horizon_subset, train_end)
        
        print("  Learning regime performance...")
        # Learn which method works best in each regime
        min_h = min(horizon_subset)
        
        for t in range(train_end // 2, train_end):
            if t >= len(current_prices) or t + min_h >= len(horizons[min_h]['y']):
                continue
            
            # Detect regime
            regime = self.regime_detector.detect(current_prices, t)
            
            # Get actual return
            actual_return = (horizons[min_h]['y'][t] - current_prices[t]) / current_prices[t]
            
            # Test each method
            for method in self.ensemble_methods:
                sig = method.get_signal(horizons, t, horizon_subset)
                signal = sig.get('signal', 0)
                
                # Strategy return
                strategy_return = signal * actual_return
                self.regime_performance[regime][method.name].append(strategy_return)
        
        # Calculate best method per regime
        self.best_method_per_regime = {}
        for regime in self.regime_performance:
            best_method = None
            best_sharpe = -999
            
            for method_name, returns in self.regime_performance[regime].items():
                if len(returns) > 5:
                    returns = np.array(returns)
                    sharpe = returns.mean() / (returns.std() + 1e-8)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_method = method_name
            
            self.best_method_per_regime[regime] = best_method or 'pairwise_slopes'
            print(f"    {regime}: Best method = {self.best_method_per_regime[regime]}")
        
        self.is_trained = True
    
    def get_signal(self, horizons: dict, t: int, horizon_subset: list,
                   current_prices: np.ndarray = None) -> dict:
        """
        Get signal using regime-adaptive method selection.
        """
        # Detect current regime
        regime = 'NEUTRAL'
        if current_prices is not None:
            regime = self.regime_detector.detect(current_prices, t)
        
        # Get best method for this regime
        best_method_name = self.best_method_per_regime.get(regime, 'pairwise_slopes')
        
        # Find and use that method
        for method in self.ensemble_methods:
            if method.name == best_method_name:
                sig = method.get_signal(horizons, t, horizon_subset)
                sig['regime'] = regime
                sig['selected_method'] = best_method_name
                return sig
        
        # Fallback
        return calculate_pairwise_slope_signal(horizons, t, horizon_subset, 'mean')
    
    def get_all_signals(self, horizons: dict, t: int, horizon_subset: list) -> dict:
        """Get signals from ALL methods for comparison."""
        results = {}
        for method in self.ensemble_methods:
            results[method.name] = method.get_signal(horizons, t, horizon_subset)
        return results


def backtest_meta_ensemble(asset_dir: str, horizon_subset: list = None):
    """
    Backtest the meta-ensemble on an asset.
    """
    from per_asset_optimizer import load_asset_data
    
    horizons, prices = load_asset_data(asset_dir)
    
    if horizons is None:
        print(f"ERROR: No data for {asset_dir}")
        return None
    
    available = sorted(horizons.keys())
    if horizon_subset is None:
        horizon_subset = available
    horizon_subset = [h for h in horizon_subset if h in available]
    
    print(f"  Horizons: {horizon_subset}")
    print(f"  Data points: {len(horizons[horizon_subset[0]]['X'])}")
    
    min_len = min(horizons[h]['X'].shape[0] for h in horizon_subset)
    train_end = int(min_len * 0.7)
    
    # Create and train meta-ensemble
    meta = MetaEnsemble()
    meta.train(horizons, prices, horizon_subset, train_end)
    
    # Backtest
    min_h = min(horizon_subset)
    y_future = horizons[min_h]['y']
    
    returns = []
    regime_returns = {r: [] for r in meta.regime_performance.keys()}
    
    print("  Backtesting...")
    for t in range(train_end, min_len - min_h):
        sig = meta.get_signal(horizons, t, horizon_subset, prices)
        signal = sig.get('signal', 0)
        regime = sig.get('regime', 'UNKNOWN')
        
        if prices[t] != 0:
            actual_return = (y_future[t] - prices[t]) / prices[t]
        else:
            actual_return = 0
        
        strategy_return = signal * actual_return
        returns.append(strategy_return)
        regime_returns[regime].append(strategy_return)
    
    returns = np.array(returns)
    
    # Metrics
    periods_per_year = 252 / min_h
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(periods_per_year)
    
    trades = returns[returns != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0
    total_return = returns.sum()
    
    # Per-regime metrics
    regime_metrics = {}
    for regime, rets in regime_returns.items():
        if len(rets) > 5:
            rets = np.array(rets)
            regime_metrics[regime] = {
                'sharpe': round(rets.mean() / (rets.std() + 1e-8) * np.sqrt(periods_per_year), 3),
                'win_rate': round((rets[rets != 0] > 0).mean() * 100, 1) if len(rets[rets != 0]) > 0 else 0,
                'count': len(rets)
            }
    
    return {
        'sharpe': round(sharpe, 3),
        'win_rate': round(win_rate * 100, 2),
        'total_return': round(total_return * 100, 2),
        'n_periods': len(returns),
        'best_method_per_regime': meta.best_method_per_regime,
        'regime_metrics': regime_metrics
    }


def run_all_meta_ensembles():
    """Run meta-ensemble backtest on all assets."""
    data_dir = Path("data")
    
    assets = []
    for d in data_dir.iterdir():
        if d.is_dir() and (d / "horizons_wide").exists():
            horizon_files = list((d / "horizons_wide").glob("*.joblib"))
            if horizon_files:
                parts = d.name.split('_', 1)
                name = parts[1].lower().replace(' ', '_') if len(parts) > 1 else d.name.lower()
                assets.append((name, str(d)))
    
    print(f"Running meta-ensemble on {len(assets)} assets...")
    print("=" * 60)
    
    all_results = {}
    
    for name, path in assets:
        print(f"\n{name.upper()}")
        print("-" * 40)
        result = backtest_meta_ensemble(path)
        
        if result:
            all_results[name] = result
            print(f"\n  RESULTS:")
            print(f"    Sharpe: {result['sharpe']}")
            print(f"    Win Rate: {result['win_rate']}%")
            print(f"    Total Return: {result['total_return']}%")
            print(f"\n  BEST METHOD PER REGIME:")
            for regime, method in result['best_method_per_regime'].items():
                print(f"    {regime}: {method}")
    
    # Summary
    print("\n" + "=" * 60)
    print("META-ENSEMBLE SUMMARY")
    print("=" * 60)
    
    for name, res in all_results.items():
        print(f"\n{name.upper()}: Sharpe={res['sharpe']}, WR={res['win_rate']}%")
    
    # Save results
    output = {
        'generated_at': datetime.now().isoformat(),
        'method': 'meta_ensemble_with_regime_adaptation',
        'results': all_results
    }
    
    with open('configs/meta_ensemble_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to configs/meta_ensemble_results.json")
    
    return all_results


if __name__ == "__main__":
    results = run_all_meta_ensembles()
