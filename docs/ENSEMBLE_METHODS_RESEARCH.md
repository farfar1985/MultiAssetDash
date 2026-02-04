# Comprehensive Ensemble Methods Research for Nexus

**Created:** 2026-02-03  
**Authors:** AmiraB + Artemis  
**Purpose:** Catalog ALL viable ensemble methods for QDT Nexus forecasting platform  
**Status:** Living document — add methods as discovered

---

## Executive Summary

This document catalogs 70+ ensemble methods organized by category. Our goal: systematically test everything to find the absolute best-performing configuration for CME partnership.

**Current Implementation:** Pairwise slopes with quantile selection  
**Target:** Find methods that significantly outperform baseline

---

## 1. Classical Averaging Methods

### 1.1 Simple Averaging
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| Simple Mean | Equal weight to all models | Baseline, robust | Ignores model quality | LOW |
| Trimmed Mean | Remove top/bottom X% before averaging | Robust to outliers | Loses information | MEDIUM |
| Winsorized Mean | Cap extreme values at percentiles | Less aggressive than trimmed | Arbitrary thresholds | MEDIUM |
| Geometric Mean | Multiplicative averaging | Good for returns | Undefined for negatives | LOW |
| Harmonic Mean | Reciprocal averaging | Penalizes large errors | Sensitive to zeros | LOW |

### 1.2 Weighted Averaging
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **Accuracy-Weighted** | Weight by historical accuracy | Intuitive, rewards good models | Past != future | **HIGH** |
| **Inverse-Error** | Weight by 1/MSE or 1/MAE | Penalizes poor performers | Sensitive to outliers | **HIGH** |
| Brier Score Weighted | Weight by probability calibration | Good for directional | Needs probability outputs | MEDIUM |
| Rank-Based | Weight by performance rank | Robust to scale | Loses magnitude info | MEDIUM |
| **Exponential Decay** | Recent performance weighted more | Adapts to regime changes | Decay rate selection | **HIGH** |

### 1.3 Top-K Selection
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **Top-K by Accuracy** | Use only top K performing models | Simple, reduces noise | K selection arbitrary | **HIGH** |
| Top-K by Sharpe | Select by risk-adjusted return | Considers risk | Sharpe can be gamed | **HIGH** |
| Top-K by Information Ratio | Select by excess return / tracking error | Benchmark-relative | Needs benchmark | MEDIUM |
| **Dynamic K** | Adjust K based on agreement | Adaptive | Complex to tune | **HIGH** |

---

## 2. Statistical Methods

### 2.1 Bayesian Model Averaging (BMA)
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **BMA (Classic)** | Weight by posterior model probability | Principled uncertainty | Computationally expensive | **HIGH** |
| BMA-EM | Expectation-maximization for weights | Faster than MCMC | Local optima | **HIGH** |
| Bayesian Stacking | Stacking with Bayesian inference | Combines benefits | Complex implementation | MEDIUM |

### 2.2 Optimal Combination Theory
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **Granger-Ramanathan** | OLS on model predictions → actuals | Theoretically optimal | Overfitting risk | **HIGH** |
| Constrained Regression | GR with non-negative weights summing to 1 | Interpretable | May underperform unconstrained | **HIGH** |
| Ridge Combination | GR with L2 regularization | Reduces overfitting | Hyperparameter tuning | **HIGH** |
| LASSO Combination | GR with L1 regularization | Sparse selection | Can drop good models | MEDIUM |
| Elastic Net | L1 + L2 regularization | Best of both | Two hyperparameters | MEDIUM |

### 2.3 Variance-Based Methods
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| Inverse Variance | Weight by 1/σ² | Theoretically justified | Assumes independence | MEDIUM |
| **Minimum Variance** | Optimize for lowest portfolio variance | Risk-focused | Ignores expected return | **HIGH** |
| Mean-Variance | Portfolio optimization on forecasts | Balances risk/return | Estimation error | **HIGH** |

---

## 3. Machine Learning Meta-Ensembles

### 3.1 Stacking (Blending)
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **Linear Stacking** | Ridge/LASSO meta-learner | Interpretable weights | May underfit | **HIGH** |
| **XGBoost Stack** | Gradient boosting meta-learner | Captures non-linearities | Overfitting risk | **HIGH** |
| LightGBM Stack | Fast gradient boosting | Efficient | Similar to XGBoost | MEDIUM |
| Neural Stack | MLP meta-learner | Flexible | Needs more data | MEDIUM |
| **Random Forest Stack** | RF meta-learner | Robust, handles interactions | Slower | **HIGH** |

### 3.2 Dynamic Selection Methods
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **META-DES** | Dynamic ensemble selection | Adapts to input | Complex | **HIGH** |
| KNORA-E | K-nearest oracle ensemble | Local selection | Needs good distance metric | MEDIUM |
| KNORA-U | K-nearest oracle union | More models | May include noise | LOW |
| DES-P | Dynamic selection by performance | Recent-focused | Short-term bias | MEDIUM |
| **OLA** | Overall Local Accuracy | Competence-based | Sensitive to k | **HIGH** |

### 3.3 Boosting-Based
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| AdaBoost Ensemble | Boost on ensemble errors | Focuses on hard cases | Sensitive to noise | MEDIUM |
| Gradient Boost Ensemble | Gradient descent on residuals | State-of-the-art | Overfitting | MEDIUM |

---

## 4. Correlation-Aware Methods

### 4.1 Diversification Methods
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **Correlation Filtering** | Remove highly correlated models | Reduces redundancy | May lose signal | **HIGH** |
| **Maximum Diversification** | Maximize diversity ratio | Portfolio-inspired | Complex optimization | **HIGH** |
| Clustering + Selection | Cluster similar, pick best per cluster | Interpretable | Clustering arbitrary | MEDIUM |
| Principal Component | Use PC projections | Orthogonal by design | Loses interpretability | LOW |

### 4.2 Portfolio-Inspired Methods
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **MVO (Mean-Variance)** | Markowitz on forecast weights | Theoretically optimal | Estimation error | **HIGH** |
| Black-Litterman | Combine prior + views | Handles uncertainty | Complex | MEDIUM |
| Risk Parity | Equal risk contribution | Robust | Ignores returns | MEDIUM |
| **Hierarchical Risk Parity** | HRP clustering | Handles non-stationarity | Newer, less tested | **HIGH** |

---

## 5. Regime-Aware Methods

### 5.1 Regime Detection
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **HMM Switching** | Hidden Markov Model regimes | Principled | State selection | **HIGH** |
| Volatility Regimes | High/low vol switching | Simple | May miss regimes | **HIGH** |
| Trend Regimes | Trending/ranging detection | Intuitive | Lagging | MEDIUM |
| Correlation Regimes | Risk-on/risk-off | Market-aware | Hard to detect in real-time | MEDIUM |

### 5.2 Regime-Specific Ensembles
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **Regime-Specific Weights** | Different weights per regime | Adapts | Needs regime labels | **HIGH** |
| Mixture of Experts | Gating network selects models | Flexible | Training complexity | MEDIUM |
| Conditional Weighting | Weights depend on features | Context-aware | Feature engineering | MEDIUM |

---

## 6. Time-Series Specific Methods

### 6.1 Temporal Methods
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **Rolling Window** | Recalculate weights periodically | Adapts over time | Window size selection | **HIGH** |
| Expanding Window | Use all available history | More data | Slow to adapt | MEDIUM |
| **EWMA Weights** | Exponentially weighted combination | Smooth adaptation | Decay parameter | **HIGH** |

### 6.2 Forecast Combination Tests
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| Diebold-Mariano | Test if combination helps | Statistical rigor | Point-in-time | LOW |
| Model Confidence Set | Identify best models | Rigorous selection | Computationally heavy | LOW |
| Giacomini-White | Conditional predictive ability | Handles non-stationarity | Complex | LOW |

---

## 7. Uncertainty Quantification

### 7.1 Prediction Intervals
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **Conformal Prediction** | Distribution-free intervals | Guaranteed coverage | Conservative | **HIGH** |
| Quantile Regression | Direct quantile estimation | Flexible | Quantile crossing | MEDIUM |
| Bootstrap Intervals | Resample-based uncertainty | Non-parametric | Computationally expensive | MEDIUM |
| **Ensemble Disagreement** | Use model spread as uncertainty | Intuitive | Not calibrated | **HIGH** |

### 7.2 Calibration Methods
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| Platt Scaling | Logistic calibration | Simple | Assumes sigmoid | MEDIUM |
| Isotonic Regression | Non-parametric calibration | Flexible | Needs validation data | MEDIUM |
| Temperature Scaling | Single parameter calibration | Minimal overhead | May not generalize | LOW |

---

## 8. Novel/Emerging Methods (2023-2026)

### 8.1 Transformer-Based
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| Attention Ensemble | Attention weights on models | Learns importance | Needs lots of data | LOW |
| Cross-Model Transformer | Transformers on ensemble outputs | Captures interactions | Black box | LOW |

### 8.2 Hybrid Methods
| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **BMA + Stacking** | Bayesian prior + learned weights | Best of both | Complex | **HIGH** |
| **Regime-Aware Stacking** | Stacking with regime features | Adaptive | Feature engineering | **HIGH** |
| Dynamic BMA | Time-varying BMA weights | Adapts | MCMC overhead | MEDIUM |

---

## 9. Grid Search Dimensions

### Parameters to Optimize:
1. **Lookback windows:** 30, 60, 90, 120, 180, 252, 365 days
2. **Model selection thresholds:** Top 1%, 2%, 5%, 10%, 20%, 30%, 50%
3. **Weighting decay:** λ = 0.9, 0.95, 0.97, 0.99, 1.0
4. **Regularization strength:** α = 0.001, 0.01, 0.1, 1.0, 10.0
5. **Number of regimes:** 2, 3, 4
6. **Aggregation methods:** mean, median, trimmed mean (5%, 10%)
7. **Horizon-specific vs unified:** test both

### Cross-Validation Strategy:
- Walk-forward with minimum 5 OOS windows
- Each OOS window: 30+ days
- Purged cross-validation (gap between train/test)

---

## 10. NEWLY ADDED: Super-Advanced Methods (Feb 3, 2026)

### 10.1 Online Learning Methods
| Method | Description | Implementation |
|--------|-------------|----------------|
| **Hedge Algorithm** | Multiplicative weight update with regret bounds | `advanced_ensemble.py` ✅ |
| Follow-the-Regularized-Leader | Online optimization | NOT YET |
| Sleeping Experts | Models go dormant when unreliable | NOT YET |

### 10.2 Hierarchical Forecast Reconciliation  
| Method | Description | Implementation |
|--------|-------------|----------------|
| **MinT Reconciliation** | Minimum trace optimal reconciliation | `advanced_ensemble.py` ✅ |
| ERM Reconciliation | Empirical risk minimization | NOT YET |
| Game-Theoretic Reconciliation | Nash equilibrium approach | NOT YET |

### 10.3 Meta-Ensemble (Ensemble of Ensembles)
| Method | Description | Implementation |
|--------|-------------|----------------|
| **Meta-Ensemble (Hedge)** | Hedge algorithm over base methods | `advanced_ensemble.py` ✅ |
| **Meta-Ensemble (Ridge)** | Ridge regression over base methods | `advanced_ensemble.py` ✅ |
| **Meta-Ensemble (Best)** | Select best performing method | `advanced_ensemble.py` ✅ |

### 10.4 Information-Theoretic Methods
| Method | Description | Implementation |
|--------|-------------|----------------|
| Mutual Information Weighting | Weight by MI with outcome | NOT YET |
| Entropy-Weighted | Weight confident forecasts more | NOT YET |

### 10.5 Copula-Based Methods
| Method | Description | Implementation |
|--------|-------------|----------------|
| Copula Combination | Model joint distribution | NOT YET |
| Vine Copula | Hierarchical dependencies | NOT YET |

---

## 11. Implementation Priority Tiers

### Tier 1 (Week 1-2) — Quick Wins
1. Accuracy-weighted averaging
2. Exponential decay weighting
3. Top-K by Sharpe
4. Ridge stacking (linear)
5. Inverse variance weighting

### Tier 2 (Week 3-4) — Statistical
1. BMA (classic)
2. Granger-Ramanathan combination
3. Minimum variance optimization
4. Conformal prediction intervals

### Tier 3 (Week 5-6) — Advanced ML
1. XGBoost stacking
2. Random Forest stacking
3. META-DES dynamic selection
4. Hierarchical Risk Parity

### Tier 4 (Week 7-8) — Regime-Aware
1. HMM regime switching
2. Volatility regime weights
3. Regime-specific ensemble selection
4. BMA + Stacking hybrid

---

## 11. Computational Constraints

### Resources Available:
- **CPU:** 8-core workstation
- **RAM:** 32GB
- **GPU:** None (CPU only)
- **Time budget:** ~2 hours per full grid search

### Method Feasibility:
| Method | Time Complexity | Memory | Feasible? |
|--------|----------------|--------|-----------|
| Simple averaging | O(n) | Low | ✅ |
| Ridge stacking | O(n²) | Medium | ✅ |
| XGBoost stack | O(n log n) | Medium | ✅ |
| Full BMA MCMC | O(2^m) | High | ⚠️ (subset only) |
| Neural stack | O(epochs × n) | High | ⚠️ (small net) |
| META-DES | O(k × n × m) | Medium | ✅ |

---

## 12. Success Metrics

### Primary:
- Sharpe Ratio (annualized, out-of-sample)
- Directional Accuracy
- Maximum Drawdown

### Secondary:
- Profit Factor
- Win Rate
- Sortino Ratio
- Calmar Ratio

### Validation:
- All metrics must be OOS (walk-forward)
- Statistical significance via paired t-test
- Improvement must be >10% to be meaningful

---

## Appendix A: Python Libraries

```python
# Core
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Specialized
from hmmlearn import hmm  # Regime detection
from deslib.des import METADES, KNORAE  # Dynamic selection
from mapie.regression import MapieRegressor  # Conformal prediction

# Optimization
from scipy.optimize import minimize  # Portfolio optimization
from cvxpy import Variable, Problem, Minimize  # Convex optimization
```

---

## Appendix B: References

1. Timmermann, A. (2006). "Forecast Combinations" — Handbook of Economic Forecasting
2. Granger & Ramanathan (1984). "Improved Methods of Combining Forecasts"
3. Raftery et al. (1997). "Bayesian Model Averaging for Linear Regression Models"
4. Wolpert (1992). "Stacked Generalization"
5. Ko et al. (2008). "From Dynamic Classifier Selection to Dynamic Ensemble Selection"
6. Lopez de Prado (2018). "Advances in Financial Machine Learning" — HRP chapter

---

*This document will be updated as we discover and test new methods.*
