# Comprehensive Ensemble Methods Research
## Every Method We Should Consider for QDTnexus

**Author:** Artemis  
**Date:** 2026-02-03  
**Purpose:** Exhaustive survey of ensemble methods for financial forecasting

---

## Executive Summary

This document catalogs **ALL viable ensemble methods** for combining 10,000+ ML model predictions. Our goal: find the optimal ensemble strategy through systematic evaluation.

**Current state:** 10,179 models for Crude Oil alone, each producing directional predictions across multiple horizons.

**Challenge:** Most models are mediocre. The ensemble's job is to extract signal from noise.

---

## Part 1: Classical Ensemble Methods

### 1.1 Simple Averaging Methods

| Method | Description | Pros | Cons | Grid Search Params |
|--------|-------------|------|------|-------------------|
| **Simple Mean** | Equal weight to all models | Baseline, robust | Ignores model quality | None |
| **Trimmed Mean** | Drop top/bottom X% before averaging | Reduces outlier impact | Loses potentially valid signals | trim_pct: [5, 10, 15, 20, 25]% |
| **Winsorized Mean** | Cap extreme values at percentiles | Keeps all models, reduces outliers | May bias results | cap_pct: [1, 5, 10]% |
| **Median** | Middle value | Robust to outliers | Loses magnitude info | None |
| **Mode** | Most common prediction direction | Good for binary signals | Ignores confidence | None |

### 1.2 Weighted Averaging Methods

| Method | Description | Weight Formula | Grid Search Params |
|--------|-------------|----------------|-------------------|
| **Accuracy Weighting** | Weight by historical accuracy | w_i = acc_i / Σacc | lookback: [30, 60, 90, 180, 360] days |
| **Inverse Error Weighting** | Weight inversely by MSE/MAE | w_i = 1/err_i | lookback, error_metric: [MSE, MAE, MAPE] |
| **Sharpe Weighting** | Weight by risk-adjusted returns | w_i = sharpe_i / Σsharpe | lookback, annualization |
| **Information Ratio Weighting** | Weight by IR | w_i = IR_i / Σ|IR| | lookback, benchmark |
| **Recency Weighting** | Recent performance matters more | w_i = acc_i × decay^(days_ago) | decay: [0.9, 0.95, 0.99, 0.995] |
| **Exponential Decay** | Exponentially favor recent | w_i = e^(-λ × days_ago) | lambda: [0.001, 0.01, 0.05, 0.1] |

### 1.3 Rank-Based Methods

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Top-K Selection** | Only use top K models by accuracy | K: [10, 50, 100, 500, 1000] |
| **Percentile Cutoff** | Use models above X percentile | cutoff: [50, 75, 90, 95, 99]% |
| **Tournament Selection** | Bracket-style model competition | bracket_size, rounds |
| **Borda Count** | Rank aggregation voting | None |

---

## Part 2: Statistical Ensemble Methods

### 2.1 Bayesian Methods

| Method | Description | Complexity | Grid Search Params |
|--------|-------------|------------|-------------------|
| **Bayesian Model Averaging (BMA)** | Weight by posterior probability | Medium | prior: [uniform, accuracy-based], likelihood |
| **Bayesian Model Combination (BMC)** | Full Bayesian treatment | High | prior_type, MCMC_samples |
| **Spike-and-Slab** | Sparse Bayesian selection | High | sparsity_prior: [0.1, 0.3, 0.5] |
| **Bayesian Stacking** | Bayesian version of stacking | Medium | regularization, prior_scale |

### 2.2 Information-Theoretic Methods

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Minimum Description Length (MDL)** | Weight by compression efficiency | None |
| **AIC/BIC Weighting** | Weight by information criteria | criteria: [AIC, BIC, AICc] |
| **Mutual Information Weighting** | Weight by predictive info | bins, estimator |

### 2.3 Optimal Combination Theory

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Granger-Ramanathan** | OLS on predictions | constraint: [none, sum-to-1, positive] |
| **Constrained Least Squares** | OLS with constraints | constraints, regularization |
| **Quantile Regression Averaging** | Combine at different quantiles | quantiles: [0.1, 0.25, 0.5, 0.75, 0.9] |

---

## Part 3: Machine Learning Meta-Ensembles

### 3.1 Stacking (Meta-Learning)

| Method | Base | Meta-Learner | Grid Search Params |
|--------|------|--------------|-------------------|
| **Linear Stacking** | All models | Ridge/Lasso | alpha: [0.001, 0.01, 0.1, 1, 10] |
| **Elastic Net Stacking** | All models | Elastic Net | alpha, l1_ratio: [0.1, 0.3, 0.5, 0.7, 0.9] |
| **XGBoost Stacking** | All models | XGBoost | depth, n_estimators, learning_rate |
| **Neural Stacking** | All models | MLP | layers, units, dropout |
| **Multi-Level Stacking** | Grouped → combined | Multiple | hierarchy_depth: [2, 3] |

### 3.2 Dynamic Model Selection

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **META-DES** | Dynamic ensemble selection | competence_region_size, voting |
| **KNORA** | K-nearest oracles | K: [5, 10, 20, 50], distance_metric |
| **DCS-LA** | Local accuracy selection | neighborhood_size |
| **Oracle Selection** | Select best model per regime | regime_features |

### 3.3 Boosting Variants

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **AdaBoost (on residuals)** | Boost weak predictions | n_estimators, learning_rate |
| **Gradient Boost Meta** | GB on prediction errors | depth, min_samples, subsample |
| **CatBoost Ensemble** | Categorical-aware boosting | iterations, depth, l2_leaf_reg |

---

## Part 4: Correlation-Aware Methods

### 4.1 Diversification Methods

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Correlation Filtering** | Remove highly correlated models | threshold: [0.7, 0.8, 0.9, 0.95] |
| **Maximum Diversification** | Optimize for diversity | diversity_weight: [0.1, 0.3, 0.5] |
| **Negative Correlation Learning** | Encourage disagreement | lambda_ncl: [0.1, 0.5, 1.0] |
| **Clustering + Representatives** | Cluster models, pick representatives | n_clusters: [10, 50, 100, 500], method |

### 4.2 Portfolio-Inspired Methods

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Mean-Variance Optimization** | Markowitz on predictions | risk_aversion: [0.5, 1, 2, 5] |
| **Risk Parity** | Equal risk contribution | risk_measure: [var, CVaR, drawdown] |
| **Black-Litterman** | Incorporate prior views | tau, uncertainty_scaling |
| **Hierarchical Risk Parity** | Cluster-based allocation | linkage, n_clusters |

### 4.3 Error Correlation Methods

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Error Decorrelation** | Weight to minimize error correlation | lookback, regularization |
| **Optimal Error Weighting** | Minimize portfolio error variance | shrinkage: [0.1, 0.3, 0.5] |
| **Independent Component Selection** | Select models with independent errors | n_components |

---

## Part 5: Regime-Adaptive Methods

### 5.1 Market Regime Detection

| Regime Feature | Description | Implementation |
|----------------|-------------|----------------|
| **Volatility Regime** | High/Medium/Low vol | ATR percentile, GARCH |
| **Trend Regime** | Trending/Mean-reverting/Choppy | ADX, Hurst exponent |
| **Momentum Regime** | Strong/Weak momentum | RSI, ROC |
| **Correlation Regime** | Risk-on/Risk-off | Cross-asset correlations |

### 5.2 Regime-Switching Methods

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Hidden Markov Models** | Latent regime states | n_states: [2, 3, 4], covariance_type |
| **Threshold Models** | Hard regime boundaries | thresholds, hysteresis |
| **Markov-Switching** | Probabilistic transitions | n_regimes, transition_prior |
| **Online Regime Detection** | Real-time regime tracking | sensitivity, min_regime_length |

### 5.3 Conditional Ensembles

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Regime-Specific Weights** | Different weights per regime | n_regimes, weight_method |
| **Mixture of Experts** | Gating network selects models | n_experts, gating_type |
| **Conditional Model Averaging** | Condition on market features | features, conditioning_method |

---

## Part 6: Deep Learning Ensembles

### 6.1 Neural Combination Methods

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Attention-Based Weighting** | Learn attention over models | attention_heads, hidden_dim |
| **Transformer Ensemble** | Transformer on predictions | n_layers, d_model, n_heads |
| **LSTM Meta-Learner** | Sequential prediction combination | units, layers, dropout |
| **TCN (Temporal Conv)** | Dilated convolutions | filters, kernel_size, dilation |

### 6.2 Neural Architecture Search

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **AutoML Stacking** | Automated meta-learner search | search_space, budget |
| **Neural Ensemble Architecture** | Learn optimal combination structure | architecture_space |

---

## Part 7: Uncertainty Quantification Methods

### 7.1 Prediction Intervals

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Conformal Prediction** | Distribution-free intervals | coverage: [0.8, 0.9, 0.95] |
| **Quantile Ensembles** | Predict quantiles, not points | quantiles |
| **Bootstrap Intervals** | Resample for uncertainty | n_bootstrap: [100, 500, 1000] |
| **MC Dropout** | Dropout at inference | dropout_rate, n_samples |

### 7.2 Calibration Methods

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Platt Scaling** | Logistic calibration | None |
| **Isotonic Regression** | Non-parametric calibration | None |
| **Temperature Scaling** | Single parameter calibration | temperature |
| **Beta Calibration** | Beta distribution fit | None |

---

## Part 8: Online/Adaptive Methods

### 8.1 Online Learning

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **Exponential Weights** | Multiplicative weight updates | learning_rate: [0.01, 0.05, 0.1] |
| **Follow the Leader** | Use best historical model | None |
| **Follow Regularized Leader** | FTL with regularization | regularization |
| **Hedge Algorithm** | Optimal regret bounds | learning_rate |

### 8.2 Bandit Methods

| Method | Description | Grid Search Params |
|--------|-------------|-------------------|
| **UCB (Upper Confidence Bound)** | Exploration-exploitation | exploration_bonus |
| **Thompson Sampling** | Bayesian bandit | prior_params |
| **EXP3** | Adversarial bandits | gamma |

---

## Part 9: Novel/Cutting-Edge Methods

### 9.1 Recent Advances (2023-2026)

| Method | Source | Description |
|--------|--------|-------------|
| **Conformalized Quantile Regression** | Romano et al. | CQR for calibrated intervals |
| **Deep Ensemble Distillation** | Malinin et al. | Distill ensemble into single model |
| **Hypernetwork Ensembles** | Krueger et al. | Generate ensemble weights dynamically |
| **Neural Process Ensembles** | Garnelo et al. | Function-space uncertainty |
| **Evidential Deep Learning** | Sensoy et al. | Single-pass uncertainty |

### 9.2 Financial-Specific Methods

| Method | Description | Source |
|--------|-------------|--------|
| **Factor-Mimicking Ensembles** | Align predictions with known factors | Quant finance |
| **Cross-Asset Signal Propagation** | Use signals from correlated assets | Multi-asset desks |
| **Order Flow Integration** | Weight by market microstructure | HFT literature |
| **Sentiment-Adjusted Weights** | Incorporate sentiment signals | Alt data |

---

## Part 10: Recommended Grid Search Strategy

### Phase 1: Baseline Establishment (Day 1)

```python
baseline_methods = [
    'simple_mean',
    'median', 
    'accuracy_weighted',
    'top_100_models',
    'top_10_pct'
]
# Run on Crude Oil, all horizons
# Metric: Directional accuracy, Sharpe
```

### Phase 2: Classical Methods (Days 2-3)

```python
grid_search_params = {
    'accuracy_weighted': {
        'lookback': [30, 60, 90, 180, 360],
        'decay': [None, 0.95, 0.99]
    },
    'top_k': {
        'k': [10, 50, 100, 500, 1000, 2000]
    },
    'trimmed_mean': {
        'trim_pct': [5, 10, 15, 20]
    },
    'correlation_filtered': {
        'threshold': [0.7, 0.8, 0.9],
        'keep_method': ['best', 'random']
    }
}
```

### Phase 3: Statistical Methods (Days 4-5)

```python
statistical_methods = {
    'bayesian_model_averaging': {
        'prior': ['uniform', 'accuracy'],
        'likelihood': ['gaussian', 'laplace']
    },
    'granger_ramanathan': {
        'constraint': ['none', 'sum_to_1', 'positive']
    },
    'stacking_ridge': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    },
    'stacking_elastic': {
        'alpha': [0.01, 0.1, 1],
        'l1_ratio': [0.1, 0.5, 0.9]
    }
}
```

### Phase 4: Regime-Adaptive (Days 6-7)

```python
regime_methods = {
    'volatility_regime': {
        'vol_measure': ['ATR', 'realized_vol', 'GARCH'],
        'n_regimes': [2, 3],
        'weights_per_regime': 'optimize'
    },
    'hmm_regime': {
        'n_states': [2, 3, 4],
        'features': ['returns', 'vol', 'both']
    }
}
```

### Phase 5: Advanced Methods (Week 2)

```python
advanced_methods = {
    'xgboost_stacking': {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 500],
        'learning_rate': [0.01, 0.1]
    },
    'attention_ensemble': {
        'hidden_dim': [32, 64, 128],
        'n_heads': [2, 4, 8]
    },
    'mixture_of_experts': {
        'n_experts': [4, 8, 16],
        'gating': ['softmax', 'sparse']
    }
}
```

---

## Part 11: Evaluation Framework

### Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| **Directional Accuracy** | % correct direction | >55% |
| **Sharpe Ratio** | Risk-adjusted returns | >1.5 |
| **Information Ratio** | Alpha / Tracking Error | >0.5 |
| **Max Drawdown** | Worst peak-to-trough | <15% |
| **Win Rate** | % profitable trades | >52% |
| **Profit Factor** | Gross profit / Gross loss | >1.3 |
| **Calmar Ratio** | Return / Max DD | >1.0 |

### Statistical Validation

1. **Walk-Forward Validation** — Expanding window, no lookahead
2. **Bootstrap Confidence Intervals** — 1000 resamples
3. **Reality Check (White, 2000)** — Control for data snooping
4. **Step-M (Romano-Wolf)** — Multiple testing correction
5. **Model Confidence Set (Hansen)** — Identify statistically best set

### Cross-Validation Strategy

```python
cv_strategy = {
    'method': 'expanding_window',
    'initial_train': 180,  # days
    'step_size': 30,       # days
    'min_test_size': 30,   # days
    'purge_gap': 5,        # days between train/test
}
```

---

## Part 12: Implementation Priority

### Must Test (High Expected Value)

1. ⭐ **Accuracy-weighted with decay** — Simple, proven
2. ⭐ **Top-K selection** — Quick wins from best models
3. ⭐ **Ridge stacking** — Robust meta-learning
4. ⭐ **Regime-switching weights** — Market adaptation
5. ⭐ **Correlation filtering + accuracy** — Diversity + quality

### Should Test (Medium Expected Value)

6. **Bayesian Model Averaging** — Principled uncertainty
7. **XGBoost stacking** — Non-linear combination
8. **Conformal prediction** — Calibrated confidence
9. **Hierarchical Risk Parity** — Portfolio theory
10. **Online learning (Hedge)** — Adaptation

### Worth Exploring (Research Value)

11. **Attention-based weighting** — State-of-the-art
12. **Mixture of Experts** — Conditional combination
13. **Deep ensemble distillation** — Efficiency
14. **Factor-mimicking** — Finance-specific

---

## Part 13: Questions for Ale/quantum_ml Team

1. **What ensemble methods are already in quantum_ml?**
2. **Have they tested any of these? Results?**
3. **Computational budget per ensemble evaluation?**
4. **Any methods they've ruled out? Why?**
5. **Access to model metadata for smarter grouping?**

---

## Part 14: Next Steps

### Immediate (Today)
- [ ] Share this doc with Amira
- [ ] Get quantum_ml access (ping Ale again)
- [ ] Set up evaluation harness

### This Week
- [ ] Run Phase 1 baseline on Crude Oil
- [ ] Run Phase 2 classical methods
- [ ] Document results in ENSEMBLE_RESULTS.md

### Next Week
- [ ] Phase 3-5 advanced methods
- [ ] Statistical validation
- [ ] Select top 3 methods for production

---

*This document will be updated as we run experiments and discover what works.*
