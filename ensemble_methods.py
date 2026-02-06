# ensemble_methods.py
# Tier 1 Ensemble Methods Implementation
# Created: 2026-02-03

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Use standardized metrics
from utils.metrics import calculate_sharpe_ratio_daily


class EnsembleMethods:
    """
    Collection of Tier 1 ensemble methods for model combination.
    All methods take predictions (DataFrame) and actuals (Series) and return weights.
    """
    
    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
    
    # ==================== METHOD 1: Accuracy-Weighted ====================
    def accuracy_weighted(self, predictions: pd.DataFrame, actuals: pd.Series) -> pd.Series:
        """
        Weight models by their directional accuracy.
        
        Args:
            predictions: DataFrame where columns are models, rows are dates
            actuals: Series of actual values
        
        Returns:
            Series of weights (sums to 1)
        """
        # Align data
        common_idx = predictions.index.intersection(actuals.index)
        preds = predictions.loc[common_idx]
        acts = actuals.loc[common_idx]
        
        # Calculate directional accuracy for each model
        prev_actuals = acts.shift(1)
        actual_direction = np.sign(acts - prev_actuals)
        
        accuracies = {}
        for col in preds.columns:
            pred_direction = np.sign(preds[col] - prev_actuals)
            matches = (pred_direction == actual_direction).dropna()
            accuracies[col] = matches.mean() if len(matches) > 0 else 0.5
        
        weights = pd.Series(accuracies)
        
        # Normalize to sum to 1
        weights = weights / weights.sum() if weights.sum() > 0 else weights * 0 + 1/len(weights)
        
        return weights
    
    # ==================== METHOD 2: Exponential Decay ====================
    def exponential_decay_weighted(self, predictions: pd.DataFrame, actuals: pd.Series, 
                                    decay: float = 0.95) -> pd.Series:
        """
        Weight models by accuracy with exponential decay (recent performance matters more).
        
        Args:
            predictions: DataFrame where columns are models, rows are dates
            actuals: Series of actual values
            decay: Decay factor (0.95 = 5% decay per period)
        
        Returns:
            Series of weights (sums to 1)
        """
        common_idx = predictions.index.intersection(actuals.index)
        preds = predictions.loc[common_idx]
        acts = actuals.loc[common_idx]
        
        prev_actuals = acts.shift(1)
        actual_direction = np.sign(acts - prev_actuals)
        
        n_periods = len(common_idx)
        time_weights = np.array([decay ** (n_periods - 1 - i) for i in range(n_periods)])
        time_weights = time_weights / time_weights.sum()
        
        scores = {}
        for col in preds.columns:
            pred_direction = np.sign(preds[col] - prev_actuals)
            matches = (pred_direction == actual_direction).fillna(0).astype(float)
            # Weighted accuracy - recent performance counts more
            scores[col] = (matches.values * time_weights).sum()
        
        weights = pd.Series(scores)
        weights = weights / weights.sum() if weights.sum() > 0 else weights * 0 + 1/len(weights)
        
        return weights
    
    # ==================== METHOD 3: Top-K by Sharpe ====================
    def top_k_by_sharpe(self, predictions: pd.DataFrame, actuals: pd.Series, 
                        k: int = None, top_pct: float = 0.1) -> pd.Series:
        """
        Select top K models by Sharpe ratio, equal weight among selected.
        
        Args:
            predictions: DataFrame where columns are models, rows are dates
            actuals: Series of actual values
            k: Number of top models (if None, uses top_pct)
            top_pct: Percentage of top models to use
        
        Returns:
            Series of weights (selected models get equal weight, others get 0)
        """
        common_idx = predictions.index.intersection(actuals.index)
        preds = predictions.loc[common_idx]
        acts = actuals.loc[common_idx]
        
        prev_actuals = acts.shift(1)
        returns = (acts - prev_actuals) / prev_actuals
        
        sharpes = {}
        for col in preds.columns:
            pred_direction = np.sign(preds[col] - prev_actuals)
            strategy_returns = pred_direction * returns
            strategy_returns = strategy_returns.dropna()

            # Use standardized Sharpe calculation
            sharpe = calculate_sharpe_ratio_daily(strategy_returns.values)
            sharpes[col] = sharpe
        
        sharpe_series = pd.Series(sharpes)
        
        # Select top K
        if k is None:
            k = max(1, int(len(sharpe_series) * top_pct))
        
        top_models = sharpe_series.nlargest(k).index
        
        weights = pd.Series(0.0, index=preds.columns)
        weights[top_models] = 1.0 / k
        
        return weights
    
    # ==================== METHOD 4: Ridge Stacking ====================
    def ridge_stacking(self, predictions: pd.DataFrame, actuals: pd.Series,
                       alpha: float = 1.0, train_pct: float = 0.7) -> pd.Series:
        """
        Use Ridge regression to learn optimal combination weights.
        
        Args:
            predictions: DataFrame where columns are models, rows are dates
            actuals: Series of actual values
            alpha: Regularization strength
            train_pct: Percentage of data for training (rest for validation)
        
        Returns:
            Series of weights (from Ridge coefficients)
        """
        common_idx = predictions.index.intersection(actuals.index)
        preds = predictions.loc[common_idx]
        acts = actuals.loc[common_idx]
        
        # Train/val split
        split_idx = int(len(common_idx) * train_pct)
        X_train = preds.iloc[:split_idx].values
        y_train = acts.iloc[:split_idx].values
        
        # Fit Ridge
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train, y_train)
        
        # Get weights from coefficients
        weights = pd.Series(model.coef_, index=preds.columns)
        
        # Normalize (make non-negative and sum to 1)
        weights = weights.clip(lower=0)
        weights = weights / weights.sum() if weights.sum() > 0 else weights * 0 + 1/len(weights)
        
        return weights
    
    # ==================== METHOD 5: Inverse Variance ====================
    def inverse_variance_weighted(self, predictions: pd.DataFrame, actuals: pd.Series) -> pd.Series:
        """
        Weight models by inverse of their prediction variance (error variance).
        Lower variance = higher weight.
        
        Args:
            predictions: DataFrame where columns are models, rows are dates
            actuals: Series of actual values
        
        Returns:
            Series of weights (sums to 1)
        """
        common_idx = predictions.index.intersection(actuals.index)
        preds = predictions.loc[common_idx]
        acts = actuals.loc[common_idx]
        
        # Calculate error variance for each model
        variances = {}
        for col in preds.columns:
            errors = preds[col] - acts
            var = errors.var()
            variances[col] = var if var > 0 else 1e-6  # Small epsilon to avoid division by zero
        
        var_series = pd.Series(variances)
        
        # Inverse variance weighting
        inv_var = 1.0 / var_series
        weights = inv_var / inv_var.sum()
        
        return weights
    
    # ==================== ENSEMBLE PREDICTION ====================
    def predict_ensemble(self, predictions: pd.DataFrame, weights: pd.Series) -> pd.Series:
        """
        Generate ensemble predictions using the given weights.
        
        Args:
            predictions: DataFrame where columns are models, rows are dates
            weights: Series of weights for each model
        
        Returns:
            Series of ensemble predictions
        """
        # Align weights with prediction columns
        aligned_weights = weights.reindex(predictions.columns).fillna(0)
        
        # Weighted combination
        ensemble_pred = (predictions * aligned_weights).sum(axis=1)
        
        return ensemble_pred
    
    # ==================== EVALUATION ====================
    def evaluate_method(self, predictions: pd.DataFrame, actuals: pd.Series,
                        method_name: str, **kwargs) -> Dict:
        """
        Evaluate an ensemble method on the data.
        
        Args:
            predictions: DataFrame where columns are models, rows are dates
            actuals: Series of actual values
            method_name: One of 'accuracy', 'decay', 'topk', 'ridge', 'inverse_var'
        
        Returns:
            Dict with performance metrics
        """
        # Get weights using specified method
        method_map = {
            'accuracy': self.accuracy_weighted,
            'decay': self.exponential_decay_weighted,
            'topk': self.top_k_by_sharpe,
            'ridge': self.ridge_stacking,
            'inverse_var': self.inverse_variance_weighted
        }
        
        if method_name not in method_map:
            raise ValueError(f"Unknown method: {method_name}")
        
        weights = method_map[method_name](predictions, actuals, **kwargs)
        ensemble_pred = self.predict_ensemble(predictions, weights)
        
        # Calculate metrics
        common_idx = ensemble_pred.index.intersection(actuals.index)
        ens = ensemble_pred.loc[common_idx]
        acts = actuals.loc[common_idx]
        
        # MSE
        mse = ((ens - acts) ** 2).mean()
        
        # Directional accuracy
        prev_acts = acts.shift(1)
        actual_dir = np.sign(acts - prev_acts)
        pred_dir = np.sign(ens - prev_acts)
        dir_acc = (actual_dir == pred_dir).dropna().mean() * 100
        
        # Sharpe (trading simulation) - using standardized calculation
        returns = (acts - prev_acts) / prev_acts
        strategy_returns = pred_dir * returns
        strategy_returns = strategy_returns.dropna()

        sharpe = calculate_sharpe_ratio_daily(strategy_returns.values)
        
        # Total return
        total_return = strategy_returns.sum() * 100
        
        return {
            'method': method_name,
            'mse': mse,
            'directional_accuracy': dir_acc,
            'sharpe': sharpe,
            'total_return': total_return,
            'n_models_used': (weights > 0.001).sum()
        }


# ==================== TIER 2 METHODS ====================

class Tier2EnsembleMethods:
    """
    Tier 2 Ensemble Methods: Statistical approaches.
    - Bayesian Model Averaging (BMA)
    - Granger-Ramanathan optimal combination
    - Quantile regression averaging
    """
    
    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
    
    def granger_ramanathan(self, predictions: pd.DataFrame, actuals: pd.Series,
                           constrained: bool = True) -> pd.Series:
        """
        Granger-Ramanathan optimal combination.
        OLS regression of actuals on predictions gives optimal weights.
        
        Args:
            predictions: DataFrame where columns are models, rows are dates
            actuals: Series of actual values
            constrained: If True, weights are non-negative and sum to 1
        
        Returns:
            Series of weights
        """
        from sklearn.linear_model import LinearRegression
        from scipy.optimize import minimize
        
        common_idx = predictions.index.intersection(actuals.index)
        X = predictions.loc[common_idx].values
        y = actuals.loc[common_idx].values
        
        # Remove NaNs
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if len(y) < 20:
            # Fallback to equal weights
            return pd.Series(1.0 / predictions.shape[1], index=predictions.columns)
        
        if constrained:
            # Constrained optimization: weights >= 0, sum to 1
            n_models = X.shape[1]
            
            def objective(w):
                pred = X @ w
                return np.sum((y - pred) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
            ]
            bounds = [(0, 1) for _ in range(n_models)]  # Non-negative
            
            # Initial guess
            w0 = np.ones(n_models) / n_models
            
            result = minimize(objective, w0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            weights = pd.Series(result.x, index=predictions.columns)
        else:
            # Unconstrained OLS
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            weights = pd.Series(model.coef_, index=predictions.columns)
        
        return weights
    
    def bma_em(self, predictions: pd.DataFrame, actuals: pd.Series,
               max_iter: int = 100, tol: float = 1e-6) -> pd.Series:
        """
        Bayesian Model Averaging using Expectation-Maximization.
        Approximates posterior model probabilities based on prediction errors.
        
        Args:
            predictions: DataFrame where columns are models, rows are dates
            actuals: Series of actual values
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
        
        Returns:
            Series of weights (posterior model probabilities)
        """
        common_idx = predictions.index.intersection(actuals.index)
        preds = predictions.loc[common_idx]
        acts = actuals.loc[common_idx]
        
        # Remove NaN rows
        mask = ~preds.isna().any(axis=1) & ~acts.isna()
        preds = preds.loc[mask]
        acts = acts.loc[mask]
        
        n_models = preds.shape[1]
        n_obs = len(acts)
        
        if n_obs < 20:
            return pd.Series(1.0 / n_models, index=preds.columns)
        
        # Initialize: equal weights
        weights = np.ones(n_models) / n_models
        
        # Compute squared errors for each model
        errors = preds.subtract(acts, axis=0)
        sq_errors = errors ** 2
        
        # EM iterations
        for iteration in range(max_iter):
            old_weights = weights.copy()
            
            # E-step: compute responsibilities
            # Using Gaussian likelihood approximation
            sigma_sq = sq_errors.mean(axis=0).values + 1e-8  # variance per model
            
            # Log-likelihood for each model at each observation
            log_likes = -0.5 * np.log(2 * np.pi * sigma_sq) - sq_errors.values / (2 * sigma_sq)
            
            # Weighted log-likelihood
            weighted_ll = log_likes + np.log(weights + 1e-10)
            
            # Normalize (softmax over models)
            max_ll = weighted_ll.max(axis=1, keepdims=True)
            exp_ll = np.exp(weighted_ll - max_ll)
            responsibilities = exp_ll / exp_ll.sum(axis=1, keepdims=True)
            
            # M-step: update weights
            weights = responsibilities.mean(axis=0)
            weights = weights / weights.sum()  # Normalize
            
            # Check convergence
            if np.abs(weights - old_weights).max() < tol:
                break
        
        return pd.Series(weights, index=preds.columns)
    
    def quantile_averaging(self, predictions: pd.DataFrame, actuals: pd.Series,
                          quantile: float = 0.5) -> pd.Series:
        """
        Quantile regression averaging.
        Weight models by their performance at a specific quantile.
        
        Args:
            predictions: DataFrame where columns are models, rows are dates
            actuals: Series of actual values
            quantile: Target quantile (0.5 = median)
        
        Returns:
            Series of weights
        """
        common_idx = predictions.index.intersection(actuals.index)
        preds = predictions.loc[common_idx]
        acts = actuals.loc[common_idx]
        
        # Compute quantile loss for each model
        errors = preds.subtract(acts, axis=0)
        
        def quantile_loss(e, q):
            return np.where(e >= 0, q * e, (q - 1) * e)
        
        q_losses = {}
        for col in preds.columns:
            q_loss = quantile_loss(errors[col].dropna(), quantile).mean()
            q_losses[col] = q_loss
        
        loss_series = pd.Series(q_losses)
        
        # Inverse loss weighting (lower loss = higher weight)
        inv_loss = 1.0 / (loss_series + 1e-8)
        weights = inv_loss / inv_loss.sum()
        
        return weights


def run_ensemble_comparison(predictions: pd.DataFrame, actuals: pd.Series,
                            lookback: int = 60) -> pd.DataFrame:
    """
    Compare all Tier 1 ensemble methods on the same data.
    
    Args:
        predictions: DataFrame of model predictions
        actuals: Series of actual values
        lookback: Lookback window for methods
    
    Returns:
        DataFrame comparing all methods
    """
    ensemble = EnsembleMethods(lookback_window=lookback)
    
    results = []
    
    # Test each method
    methods = [
        ('accuracy', {}),
        ('decay', {'decay': 0.95}),
        ('decay', {'decay': 0.99}),
        ('topk', {'top_pct': 0.05}),
        ('topk', {'top_pct': 0.10}),
        ('topk', {'top_pct': 0.20}),
        ('ridge', {'alpha': 0.1}),
        ('ridge', {'alpha': 1.0}),
        ('ridge', {'alpha': 10.0}),
        ('inverse_var', {}),
    ]
    
    for method_name, kwargs in methods:
        try:
            result = ensemble.evaluate_method(predictions, actuals, method_name, **kwargs)
            result['params'] = str(kwargs) if kwargs else 'default'
            results.append(result)
            print(f"  [OK] {method_name} ({result['params']}): Sharpe={result['sharpe']:.2f}, DA={result['directional_accuracy']:.1f}%")
        except Exception as e:
            print(f"  [ERR] {method_name}: {str(e)}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Ensemble Methods - Tier 1 Implementation")
    print("=" * 50)
    print("Available methods:")
    print("  1. accuracy_weighted - Weight by directional accuracy")
    print("  2. exponential_decay_weighted - Recent performance weighted more")
    print("  3. top_k_by_sharpe - Select top K by Sharpe ratio")
    print("  4. ridge_stacking - Ridge regression meta-learner")
    print("  5. inverse_variance_weighted - Weight by 1/error_variance")
    print("\nUse run_ensemble_comparison() to compare all methods.")
