# Nexus Data Analysis & Ensemble Research Plan

**Date:** 2026-02-03  
**Purpose:** Understand data limitations before designing advanced ensemble methods

---

## 1. Data Sources Overview

### Upstream API: QuantumCloud.ai
- **Endpoint:** `https://quantumcloud.ai/get_qml_models/{PROJECT_ID}`
- **Auth:** `qml-api-key` header
- **Returns:** All model predictions for a given asset project

### What Each Model Provides

Per model, per date, per horizon:
```
{
  "symbol": "4419.0_100001",      # Model ID
  "time": "2025-03-01",           # Prediction date
  "n_predict": 5,                 # Horizon (D+5)
  "close_predict": 72.45,         # Predicted price
  "target_var_price": 71.20       # Actual price at prediction time
}
```

**That's it.** No feature importance, no model metadata, no confidence scores from individual models.

---

## 2. Current Data Inventory

### Model Counts by Asset

| Asset | Models | Date Range | Notes |
|-------|--------|------------|-------|
| Crude Oil | 10,179 | 2025-01-01 to 2026-01-04 | Most models |
| Bitcoin | 4,962 | 2025-01-01 to 2026-01-04 | |
| S&P 500 | 3,968 | 2025-01-01 to 2026-01-04 | |
| Nifty 50 | 3,947 | 2025-01-01 to 2026-01-04 | |
| Nifty Bank | 3,753 | 2025-01-01 to 2026-01-04 | |
| MCX Copper | 3,369 | 2025-01-01 to 2026-01-04 | |
| Russell 2000 | 3,398 | 2025-01-01 to 2026-01-04 | |
| Dow Jones | 3,099 | 2025-01-01 to 2026-01-04 | |
| Gold | 2,819 | 2025-01-01 to 2026-01-04 | |
| NASDAQ | 2,659 | 2025-01-01 to 2026-01-04 | |
| Nikkei 225 | 1,810 | 2025-01-01 to 2026-01-04 | |
| US Dollar Index | 1,770 | 2025-01-01 to 2026-01-04 | |
| USD/INR | 1,755 | 2025-01-01 to 2026-01-04 | |
| Brent Oil | 624 | 2025-01-01 to 2026-01-04 | Few models |
| SPDR China ETF | 9 | 2025-01-01 to 2026-01-04 | **Too few for ensemble!** |

### Horizons Available

Typically 5-10 horizons per asset:
- D+1, D+2, D+3, D+4, D+5, D+6, D+7, D+8, D+9, D+10
- Some assets have longer horizons (D+13, D+21, D+34, D+55, D+89, D+180)

### Data Shape Example (Crude Oil, D+1)

```
X matrix (predictions): 369 dates × 10,179 models
y vector (actuals):     369 dates

Each cell X[date, model] = that model's D+1 price prediction for that date
```

---

## 3. Data Limitations

### 🔴 Critical Limitations

1. **No Model Metadata**
   - We don't know what features each model uses
   - We don't know the algorithm (linear, tree, neural net)
   - We don't know training windows or hyperparameters
   - Can't do feature-based diversity selection

2. **No Model Confidence/Uncertainty**
   - Models return point predictions only
   - No prediction intervals
   - No model-reported confidence scores
   - Can't weight by model certainty

3. **No Model Version History**
   - Don't know when models were retrained
   - Can't detect model drift vs market regime change
   - No A/B testing of model versions

4. **Short History**
   - Only ~1 year of data (2025-01-01 to 2026-01-04)
   - 369 trading days
   - Limits statistical power for validation
   - Can't test through multiple market regimes

5. **Sparse Horizon Coverage**
   - Not all models provide all horizons
   - Some horizons have much fewer models
   - Creates missing data issues

### 🟡 Moderate Limitations

6. **No Real-Time Timestamps**
   - Daily predictions only
   - Don't know what time of day predictions were made
   - Can't account for intraday information advantage

7. **One Price Prediction Per Model**
   - No distributional forecasts
   - No scenario predictions (bull/bear/base)
   - Limits uncertainty quantification

8. **Model IDs Are Opaque**
   - `4419.0_100001` tells us nothing
   - Can't group models by family/approach
   - Can't identify redundant models

---

## 4. What We CAN Derive

Despite limitations, we can compute:

### From Prediction History
- **Per-model accuracy** (directional, MSE, MAE)
- **Per-model Sharpe** (if traded on signals)
- **Model correlation matrix** (which models agree)
- **Model error correlation** (which models fail together)
- **Regime-conditional accuracy** (high vol vs low vol)
- **Decay analysis** (do models degrade over time)

### From Ensemble Outputs
- **Consensus strength** (% agreement)
- **Prediction dispersion** (std across models)
- **Tail model behavior** (what do outlier models say)

### Derived Metrics Currently Computed
```
signal_history_detailed.csv contains:
- net_prob: (bullish_slopes - bearish_slopes) / total_slopes
- strength: abs(net_prob)
- agreement_ratio: how much models agree
- correct_1d/3d/5d/10d: was signal correct at each horizon
- actual_move_%: what actually happened
```

---

## 5. Advanced Ensemble Methods to Test

### Tier 1: Can Implement Now (No New Data Needed)

#### 1.1 Weighted Model Selection
**Current:** Select top-N% models by quantile, equal weight  
**Improvement:** Weight models by rolling performance

```python
# Exponentially weighted historical accuracy
weights[model] = accuracy[model] * decay^(days_since_prediction)
ensemble = sum(predictions * weights) / sum(weights)
```

#### 1.2 Prediction Diversity Weighting
**Current:** Ignore model correlations  
**Improvement:** Penalize correlated models

```python
# Inverse correlation weighting
corr_matrix = predictions.corr()
avg_corr = corr_matrix.mean(axis=1)
diversity_weight = 1 / (1 + avg_corr)
```

#### 1.3 Error Correlation Weighting
**Better than prediction correlation:** Models that make *different errors* add more value

```python
errors = predictions - actuals
error_corr = errors.corr()
# Select models with low error correlation
```

#### 1.4 Regime-Adaptive Ensemble
**Current:** Same ensemble in all conditions  
**Improvement:** Different weights in different regimes

```python
# Detect regime via rolling volatility
vol = returns.rolling(20).std()
regime = 'high_vol' if vol > threshold else 'low_vol'

# Use regime-specific model weights
weights = regime_weights[regime]
```

#### 1.5 Stacking Meta-Learner
**Current:** Simple aggregation (mean/median)  
**Improvement:** Train a model to combine models

```python
from sklearn.linear_model import RidgeCV

# Train: meta-learner predicts actuals from base predictions
meta = RidgeCV()
meta.fit(X_train_predictions, y_train_actuals)

# Predict: meta-learner weights base predictions
ensemble = meta.predict(X_test_predictions)
```

#### 1.6 Bayesian Model Averaging
**Current:** Discrete selection (top-N%)  
**Improvement:** Continuous weights proportional to posterior

```python
# Weight by model likelihood
log_likelihoods = -0.5 * ((predictions - actuals)**2 / sigma**2).sum()
weights = softmax(log_likelihoods)
ensemble = (predictions * weights).sum()
```

### Tier 2: Requires Computed Features (Derivable)

#### 2.1 Prediction Interval via Model Disagreement
Since individual models don't give confidence, use ensemble spread:

```python
# Confidence interval from model disagreement
lower = predictions.quantile(0.05)
median = predictions.quantile(0.50)
upper = predictions.quantile(0.95)
# Width of interval = uncertainty
```

#### 2.2 Conformal Prediction (Calibrated Intervals)
Use historical errors to calibrate prediction intervals:

```python
# Compute residuals on validation set
residuals = abs(prediction - actual)
calibration_quantile = np.percentile(residuals, 95)

# Prediction interval for new prediction
interval = [pred - calibration_quantile, pred + calibration_quantile]
# Guaranteed 95% coverage if distribution is stable
```

#### 2.3 Model Clustering
Group similar models to reduce redundancy:

```python
from sklearn.cluster import AgglomerativeClustering

# Cluster models by prediction correlation
clusters = AgglomerativeClustering(n_clusters=20).fit(predictions.T)

# Select representative from each cluster
for cluster_id in range(20):
    cluster_models = models[clusters.labels_ == cluster_id]
    best = cluster_models[accuracy[cluster_models].argmax()]
    selected.append(best)
```

### Tier 3: Would Require New Data

#### 3.1 Feature Importance Weighting
**Blocked:** We don't know what features each model uses

#### 3.2 Algorithm-Based Diversity
**Blocked:** We don't know model types (can't ensure "include one tree, one NN, one linear")

#### 3.3 Training Window Diversity
**Blocked:** We don't know when models were trained

---

## 6. Recommended Research Agenda

### Phase 1: Baseline Measurements (Week 1)

1. **Compute full model correlation matrix** for each asset
   - Identify highly correlated model clusters
   - Quantify effective number of independent models

2. **Compute error correlation matrix**
   - Find models with uncorrelated errors
   - These are the most valuable for ensemble

3. **Regime analysis**
   - Define vol regimes (rolling 20-day vol quantiles)
   - Measure per-model accuracy by regime
   - Which models work in which conditions?

4. **Decay analysis**
   - Does model accuracy degrade over time?
   - Half-life of model skill

### Phase 2: Ensemble Experiments (Week 2-3)

Test each method on held-out data (2025-10-01 to 2026-01-04):

| Method | Baseline OOS Sharpe | New OOS Sharpe | Improvement |
|--------|--------------------|-----------------| ------------|
| Current (pairwise slopes) | X.XX | - | - |
| Weighted by accuracy | - | X.XX | +Y% |
| Error correlation weighted | - | X.XX | +Y% |
| Stacking (Ridge) | - | X.XX | +Y% |
| Bayesian Model Averaging | - | X.XX | +Y% |
| Regime-adaptive | - | X.XX | +Y% |

### Phase 3: Statistical Validation (Week 3-4)

1. **Bootstrap confidence intervals** on all metrics
2. **Walk-forward cross-validation** (5+ folds)
3. **Statistical significance tests** (is improvement real?)
4. **Transaction cost sensitivity** (does it survive after costs?)

---

## 7. Questions for Bill

1. **Can we get more data from the QDT API?**
   - Model metadata (algorithm type, features used)?
   - Model training dates?
   - Model confidence scores?

2. **Can we extend the history?**
   - Is data available before 2025-01-01?
   - More history = more robust validation

3. **What's the latency requirement?**
   - Some methods (stacking, BMA) are more compute-intensive
   - Do we need real-time (sub-second) or can we batch daily?

4. **Asset priority?**
   - Crude Oil has most models (10K) — best for testing
   - SPDR China ETF has only 9 models — can't ensemble
   - Which assets matter most for CME?

5. **Risk constraints?**
   - Max acceptable drawdown?
   - Position sizing approach?
   - These affect which ensemble methods make sense

---

## 8. Immediate Next Steps

1. [ ] Run correlation analysis on Crude Oil (10K models)
2. [ ] Implement stacking meta-learner prototype
3. [ ] Build regime detection (vol-based)
4. [ ] Set up proper walk-forward backtesting framework
5. [ ] Create comparison dashboard for ensemble methods

---

*This document establishes the data foundation. Advanced ensemble methods should be designed around these constraints.*
