# Ensemble Methods Research & Implementation Plan

**Version:** 1.0  
**Date:** 2026-02-03  
**Authors:** AmiraB  
**Status:** RESEARCH PLAN — For Review with Artemis

---

## Executive Summary

The current Nexus ensemble uses a simple "pairwise slopes" voting mechanism. This is statistically naïve and leaves significant accuracy gains on the table. We need to test multiple advanced ensemble techniques and implement the best performers.

**Goal:** Improve ensemble signal accuracy by 10-25% through advanced methods.

---

## Part 1: Current Method Analysis

### 1.1 How Pairwise Slopes Works

```python
# Current implementation (duplicated in 5+ files)
for each pair of horizons (D+i, D+j) where j > i:
    drift = forecast[j] - forecast[i]
    if drift > 0:
        bullish_count += 1
    else:
        bearish_count += 1

net_prob = (bullish_count - bearish_count) / total_pairs
signal = "BULLISH" if net_prob > threshold else "BEARISH" if net_prob < -threshold else "NEUTRAL"
```

### 1.2 Problems with Current Approach

| Problem | Impact | Solution Direction |
|---------|--------|-------------------|
| Equal weighting of all pairs | Treats (D+1,D+2) same as (D+1,D+100) | Weighted ensembles |
| Magnitude ignored | +$0.01 counts same as +$100 | Magnitude-weighted voting |
| No model metadata | Can't group by algorithm type | quantum_ml integration |
| No accuracy tracking | Don't know which models are reliable | Rolling accuracy weights |
| No uncertainty | Point estimate only | Conformal prediction, BMA |
| No regime adaptation | Same weights in all market conditions | Regime-adaptive ensemble |
| No error correlation | Redundant models over-counted | Correlation-based weighting |

### 1.3 Current Performance Baseline

From ENSEMBLE_AUDIT.md:
- Directional accuracy: ~55-65% (varies by asset)
- Sharpe ratio: 0.8-1.5 (before bug fixes)
- Max drawdown: 15-30%

**Target after improvements:**
- Directional accuracy: 65-75%
- Sharpe ratio: 1.5-2.5
- Max drawdown: <15%

---

## Part 2: Ensemble Methods to Test

### Tier 1: Essential Methods (Must Implement)

#### 2.1 Accuracy-Weighted Ensemble

**Concept:** Weight each model/horizon pair by its rolling historical accuracy.

```python
def accuracy_weighted_signal(forecasts, accuracy_cache, horizons, lookback=60):
    weighted_bull = 0.0
    weighted_bear = 0.0
    
    for i, h1 in enumerate(horizons):
        for h2 in horizons[i+1:]:
            drift = forecasts[h2] - forecasts[h1]
            
            # Get rolling accuracy for this pair
            pair_key = (h1, h2)
            accuracy = accuracy_cache.get(pair_key, 0.5)  # Default 50%
            
            # Weight by accuracy (better models count more)
            weight = accuracy ** 2  # Square to amplify differences
            
            if drift > 0:
                weighted_bull += weight
            else:
                weighted_bear += weight
    
    total = weighted_bull + weighted_bear
    return (weighted_bull - weighted_bear) / total if total > 0 else 0
```

**Expected improvement:** +5-15%
**Complexity:** Low
**Data requirements:** Historical predictions vs actuals

---

#### 2.2 Magnitude-Weighted Voting

**Concept:** Larger price moves indicate stronger conviction.

```python
def magnitude_weighted_signal(forecasts, horizons, base_price):
    weighted_sum = 0.0
    total_weight = 0.0
    
    for i, h1 in enumerate(horizons):
        for h2 in horizons[i+1:]:
            drift = forecasts[h2] - forecasts[h1]
            
            # Magnitude as percentage of base price
            magnitude = abs(drift) / base_price
            
            # Horizon separation (longer spans = more independent info)
            separation_weight = np.log1p(h2 - h1)
            
            weight = magnitude * separation_weight
            weighted_sum += np.sign(drift) * weight
            total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0
```

**Expected improvement:** +5-10%
**Complexity:** Low
**Data requirements:** Current forecasts only

---

#### 2.3 Stacking Meta-Learner

**Concept:** Train a second-level model on base model predictions.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

def train_stacking_ensemble(X_models, y_actual, n_splits=5):
    """
    X_models: DataFrame where each column is a model's predictions
    y_actual: Actual returns
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Use RidgeCV for automatic regularization tuning
    meta_model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
    
    # Walk-forward training
    oof_predictions = np.zeros(len(y_actual))
    
    for train_idx, val_idx in tscv.split(X_models):
        X_train, X_val = X_models.iloc[train_idx], X_models.iloc[val_idx]
        y_train = y_actual.iloc[train_idx]
        
        meta_model.fit(X_train, y_train)
        oof_predictions[val_idx] = meta_model.predict(X_val)
    
    # Final model on all data
    meta_model.fit(X_models, y_actual)
    
    return meta_model, meta_model.coef_
```

**Expected improvement:** +15-30%
**Complexity:** Medium
**Data requirements:** Multiple model predictions + actuals

---

#### 2.4 Error Correlation Weighting

**Concept:** Weight models inversely to their error correlation with others.

```python
def correlation_weighted_ensemble(model_errors):
    """
    model_errors: DataFrame where each column is a model's prediction errors
    Returns: weights for each model
    """
    # Compute error correlation matrix
    corr_matrix = model_errors.corr()
    
    # Average correlation of each model with all others
    avg_correlation = corr_matrix.mean()
    
    # Weight inversely (less correlated = higher weight)
    raw_weights = 1 / (avg_correlation + 0.1)  # Epsilon for stability
    
    # Normalize to sum to 1
    weights = raw_weights / raw_weights.sum()
    
    return weights
```

**Expected improvement:** +10-20%
**Complexity:** Medium
**Data requirements:** Historical errors for all models

---

### Tier 2: Advanced Methods (Should Implement)

#### 2.5 Bayesian Model Averaging (BMA)

**Concept:** Weight models by their posterior probability given observed data.

```python
def bayesian_model_averaging(predictions, likelihoods, prior='uniform'):
    """
    predictions: dict of model_name -> prediction
    likelihoods: dict of model_name -> P(data|model)
    """
    if prior == 'uniform':
        prior_weights = {m: 1/len(predictions) for m in predictions}
    
    # Posterior ∝ Likelihood × Prior
    posteriors = {}
    for model in predictions:
        posteriors[model] = likelihoods[model] * prior_weights[model]
    
    # Normalize
    total = sum(posteriors.values())
    posteriors = {m: p/total for m, p in posteriors.items()}
    
    # Weighted average prediction
    ensemble_pred = sum(predictions[m] * posteriors[m] for m in predictions)
    
    # Uncertainty: weighted variance
    mean_pred = ensemble_pred
    variance = sum(posteriors[m] * (predictions[m] - mean_pred)**2 for m in predictions)
    
    return ensemble_pred, np.sqrt(variance), posteriors
```

**Expected improvement:** +10-25%
**Complexity:** Medium-High
**Data requirements:** Model likelihoods (from quantum_ml)

---

#### 2.6 Regime-Adaptive Ensemble

**Concept:** Different ensemble weights for different market regimes.

```python
from hmmlearn import hmm

class RegimeAdaptiveEnsemble:
    def __init__(self, n_regimes=3):
        self.hmm = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full")
        self.regime_weights = {}  # regime_id -> model weights
    
    def fit(self, returns, model_predictions, y_actual):
        # Fit HMM on returns
        self.hmm.fit(returns.values.reshape(-1, 1))
        regimes = self.hmm.predict(returns.values.reshape(-1, 1))
        
        # Learn optimal weights per regime
        for regime in range(self.hmm.n_components):
            mask = regimes == regime
            if mask.sum() > 30:  # Minimum samples
                X_regime = model_predictions[mask]
                y_regime = y_actual[mask]
                
                # Train stacking model for this regime
                meta = RidgeCV().fit(X_regime, y_regime)
                self.regime_weights[regime] = meta.coef_
    
    def predict(self, returns_recent, model_predictions_today):
        # Detect current regime
        current_regime = self.hmm.predict(returns_recent.values.reshape(-1, 1))[-1]
        
        # Apply regime-specific weights
        weights = self.regime_weights.get(current_regime, np.ones(len(model_predictions_today)))
        weights = weights / weights.sum()
        
        return (model_predictions_today * weights).sum()
```

**Expected improvement:** +10-25%
**Complexity:** High
**Data requirements:** Historical returns + model predictions

---

#### 2.7 Conformal Prediction Intervals

**Concept:** Distribution-free prediction intervals with guaranteed coverage.

```python
class ConformalEnsemble:
    def __init__(self, coverage=0.90):
        self.coverage = coverage
        self.calibration_scores = []
    
    def calibrate(self, predictions, actuals):
        """Compute nonconformity scores on calibration set"""
        residuals = np.abs(predictions - actuals)
        self.calibration_scores = sorted(residuals)
    
    def predict_interval(self, point_prediction):
        """Return prediction interval with guaranteed coverage"""
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * self.coverage) / n
        q_idx = int(q_level * n) - 1
        
        margin = self.calibration_scores[min(q_idx, n-1)]
        
        return (
            point_prediction - margin,  # Lower bound
            point_prediction,            # Point estimate
            point_prediction + margin    # Upper bound
        )
```

**Expected improvement:** Better uncertainty quantification
**Complexity:** Medium
**Data requirements:** Calibration set of predictions vs actuals

---

### Tier 3: Cutting-Edge Methods (Research & Explore)

#### 2.8 Online Learning with Thompson Sampling

**Concept:** Treat model selection as a multi-armed bandit problem.

```python
class ThompsonSamplingEnsemble:
    def __init__(self, n_models):
        # Beta distributions for each model (successes, failures)
        self.alphas = np.ones(n_models)  # Successes + 1
        self.betas = np.ones(n_models)   # Failures + 1
    
    def select_model(self):
        """Sample from posterior and select best model"""
        samples = [np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]
        return np.argmax(samples)
    
    def update(self, model_idx, reward):
        """Update beliefs based on outcome"""
        if reward > 0:
            self.alphas[model_idx] += 1
        else:
            self.betas[model_idx] += 1
    
    def get_weights(self):
        """Convert to weights for ensemble"""
        means = self.alphas / (self.alphas + self.betas)
        return means / means.sum()
```

**Expected improvement:** Adaptive, explores new models
**Complexity:** High
**Data requirements:** Real-time feedback loop

---

#### 2.9 Attention-Based Ensemble (Transformer-style)

**Concept:** Learn which models to attend to based on context.

```python
import torch
import torch.nn as nn

class AttentionEnsemble(nn.Module):
    def __init__(self, n_models, context_dim, hidden_dim=64):
        super().__init__()
        self.query = nn.Linear(context_dim, hidden_dim)
        self.key = nn.Linear(1, hidden_dim)  # Each model prediction
        self.value = nn.Linear(1, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, context, model_predictions):
        """
        context: market features (volatility, regime, etc.)
        model_predictions: tensor of shape (n_models,)
        """
        Q = self.query(context)  # (hidden_dim,)
        K = self.key(model_predictions.unsqueeze(-1))  # (n_models, hidden_dim)
        V = self.value(model_predictions.unsqueeze(-1))  # (n_models, hidden_dim)
        
        # Attention scores
        scores = torch.matmul(K, Q) / np.sqrt(K.shape[-1])
        weights = torch.softmax(scores, dim=0)
        
        # Weighted combination
        attended = torch.matmul(weights, V)
        
        return self.output(attended), weights
```

**Expected improvement:** Unknown — experimental
**Complexity:** Very High
**Data requirements:** Large training set, context features

---

#### 2.10 Quantile Regression Forest Ensemble

**Concept:** Predict entire distribution, not just point estimate.

```python
from sklearn.ensemble import RandomForestRegressor

class QuantileForestEnsemble:
    def __init__(self, n_estimators=100):
        self.forest = RandomForestRegressor(n_estimators=n_estimators)
        self.leaf_predictions = None
    
    def fit(self, X, y):
        self.forest.fit(X, y)
        # Store all leaf predictions for quantile computation
        self.leaf_predictions = {}
        for i, tree in enumerate(self.forest.estimators_):
            leaves = tree.apply(X)
            for leaf_id in np.unique(leaves):
                mask = leaves == leaf_id
                self.leaf_predictions[(i, leaf_id)] = y[mask].values
    
    def predict_quantile(self, X, quantile=0.5):
        """Predict specific quantile"""
        predictions = []
        for idx in range(len(X)):
            all_leaf_values = []
            for i, tree in enumerate(self.forest.estimators_):
                leaf_id = tree.apply(X.iloc[[idx]])[0]
                all_leaf_values.extend(self.leaf_predictions.get((i, leaf_id), []))
            
            predictions.append(np.percentile(all_leaf_values, quantile * 100))
        
        return np.array(predictions)
```

**Expected improvement:** Full distribution, better risk estimates
**Complexity:** High
**Data requirements:** Large training set

---

## Part 3: Testing Framework

### 3.1 Walk-Forward Cross-Validation

```python
def walk_forward_test(ensemble_method, data, n_folds=5, train_size=180, test_size=30):
    """
    Rigorous walk-forward testing with multiple folds
    """
    results = []
    
    for fold in range(n_folds):
        test_end = len(data) - fold * test_size
        test_start = test_end - test_size
        train_end = test_start
        train_start = max(0, train_end - train_size)
        
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]
        
        # Train ensemble
        ensemble_method.fit(train_data)
        
        # Test
        predictions = ensemble_method.predict(test_data)
        actuals = test_data['actual_return']
        
        # Compute metrics
        fold_results = {
            'fold': fold,
            'accuracy': (np.sign(predictions) == np.sign(actuals)).mean(),
            'sharpe': compute_sharpe(predictions, actuals),
            'max_dd': compute_max_drawdown(predictions, actuals),
            'correlation': np.corrcoef(predictions, actuals)[0, 1]
        }
        results.append(fold_results)
    
    return pd.DataFrame(results)
```

### 3.2 Statistical Significance Testing

```python
def compare_ensembles(method_a, method_b, data, n_bootstrap=1000):
    """
    Bootstrap test for comparing two ensemble methods
    """
    diff_scores = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(len(data), size=len(data), replace=True)
        sample = data.iloc[idx]
        
        score_a = evaluate_ensemble(method_a, sample)
        score_b = evaluate_ensemble(method_b, sample)
        diff_scores.append(score_a - score_b)
    
    diff_scores = np.array(diff_scores)
    
    # Two-tailed p-value
    p_value = 2 * min(
        (diff_scores > 0).mean(),
        (diff_scores < 0).mean()
    )
    
    return {
        'mean_diff': diff_scores.mean(),
        'std_diff': diff_scores.std(),
        'ci_95': (np.percentile(diff_scores, 2.5), np.percentile(diff_scores, 97.5)),
        'p_value': p_value
    }
```

### 3.3 Transaction Cost Modeling

```python
def backtest_with_costs(signals, prices, cost_bps=5):
    """
    Realistic backtest including transaction costs
    """
    position = 0
    equity = [100]
    
    for i in range(1, len(signals)):
        new_position = np.sign(signals.iloc[i])
        
        if new_position != position:
            # Trade occurred
            trade_cost = cost_bps / 10000 * equity[-1]
            equity.append(equity[-1] - trade_cost)
            position = new_position
        
        # P&L from position
        return_pct = (prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]
        pnl = position * return_pct * equity[-1]
        equity.append(equity[-1] + pnl)
    
    return pd.Series(equity)
```

---

## Part 4: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task | Method | Priority | Owner |
|------|--------|----------|-------|
| Implement accuracy-weighted | 2.1 | P0 | AmiraB |
| Implement magnitude-weighted | 2.2 | P0 | AmiraB |
| Build testing framework | 3.1-3.3 | P0 | AmiraB |
| Baseline all methods on Crude Oil | — | P0 | AmiraB |

### Phase 2: Advanced Methods (Week 3-4)

| Task | Method | Priority | Owner |
|------|--------|----------|-------|
| Implement stacking | 2.3 | P0 | AmiraB |
| Implement error correlation | 2.4 | P0 | AmiraB |
| Implement BMA (if quantum_ml ready) | 2.5 | P1 | AmiraB |
| Implement regime-adaptive | 2.6 | P1 | AmiraB |

### Phase 3: Uncertainty Quantification (Week 5)

| Task | Method | Priority | Owner |
|------|--------|----------|-------|
| Implement conformal prediction | 2.7 | P0 | AmiraB |
| Integrate intervals into UI | — | P0 | Artemis |
| Validate coverage | — | P0 | AmiraB |

### Phase 4: Research Methods (Week 6+)

| Task | Method | Priority | Owner |
|------|--------|----------|-------|
| Thompson sampling prototype | 2.8 | P2 | AmiraB |
| Attention ensemble prototype | 2.9 | P3 | AmiraB |
| Quantile forest prototype | 2.10 | P2 | AmiraB |

---

## Part 5: Success Metrics

### Accuracy Targets

| Method | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| Current (pairwise slopes) | 58% | — | — |
| Accuracy-weighted | — | 63% | +5% |
| Magnitude-weighted | — | 61% | +3% |
| Stacking | — | 68% | +10% |
| Error correlation | — | 65% | +7% |
| BMA | — | 70% | +12% |
| Regime-adaptive | — | 72% | +14% |
| **Best ensemble** | — | **75%** | **+17%** |

### Risk Metrics Targets

| Metric | Current | Target |
|--------|---------|--------|
| Sharpe Ratio | 0.8-1.5 | 1.5-2.5 |
| Max Drawdown | 15-30% | <15% |
| Win Rate | 55-60% | 65-70% |
| Profit Factor | 1.2-1.5 | 1.8-2.5 |

---

## Part 6: Data Requirements

### What We Need from quantum_ml

| Data | Purpose | Status |
|------|---------|--------|
| Model predictions per date | All ensemble methods | Need repo access |
| Model metadata (algorithm, features) | BMA, grouping | Need repo access |
| Historical accuracy per model | Accuracy weighting | Via run_backtest |
| Feature importance | Explainability | Via compute_feature_importance |

### What We Can Compute Ourselves

| Data | Purpose | How |
|------|---------|-----|
| Prediction errors | Error correlation | predictions - actuals |
| Rolling accuracy | Accuracy weighting | Moving window hit rate |
| Regime indicators | Regime-adaptive | HMM on returns |
| Correlations | Diversification | Correlation matrix |

---

## Part 7: Integration with AI Agent

The ensemble methods feed directly into the AI Agent:

### Forecast Explainer
```
"The model is bullish because:
- 8 of 10 horizon pairs agree (pairwise signal)
- The accuracy-weighted signal is 72% confident
- The stacking meta-model predicts +1.2%
- We're in a LOW VOLATILITY regime where these models have 78% accuracy
- The 90% confidence interval is [$71.50, $74.20]"
```

### Position Advisor
```
"Given the ensemble signals:
- Point estimate: +1.8% over 5 days
- 90% CI: [+0.2%, +3.4%]
- Regime-adjusted Kelly: 8% position
- Recommended stop: $70.50 (1.5 ATR below)"
```

### What-If Scenarios
```
"If crude drops 3% tomorrow:
- Regime would likely shift to HIGH VOLATILITY
- Model weights would rebalance (momentum models down, mean-rev up)
- New signal would be: NEUTRAL (confidence drops to 52%)"
```

---

## Appendix: Literature References

1. **Stacking:** Wolpert, D. (1992). Stacked Generalization
2. **BMA:** Hoeting et al. (1999). Bayesian Model Averaging
3. **Conformal:** Vovk et al. (2005). Algorithmic Learning in a Random World
4. **Regime Switching:** Hamilton, J. (1989). A New Approach to Economic Analysis
5. **Thompson Sampling:** Thompson, W. (1933). On the Likelihood that One Unknown Probability Exceeds Another
6. **Attention:** Vaswani et al. (2017). Attention Is All You Need

---

*This document outlines the ensemble research plan. Methods will be tested rigorously and the best performers integrated into Nexus.*

**Created by:** AmiraB  
**Last updated:** 2026-02-03
