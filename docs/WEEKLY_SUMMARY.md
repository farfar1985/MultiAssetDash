# QDT Nexus - Weekly Summary

**Week Ending:** February 6, 2026
**Commits Today:** 31
**Lines Changed:** +24,416 / -2,261

---

## Dashboards (11 Routes)

| Route | Dashboard | Audience |
|-------|-----------|----------|
| `/dashboards` | Dashboard Index | All Users |
| `/dashboards/retail` | Retail | Beginners |
| `/dashboards/pro-retail` | Pro Retail | Learning Traders |
| `/dashboards/alpha-gen-pro` | Alpha Gen Pro | Active Traders |
| `/dashboards/hedge-fund` | Hedge Fund Portfolio | Portfolio Managers |
| `/dashboards/hedging` | Hedging | Risk Managers |
| `/dashboards/procurement` | Procurement | Supply Chain |
| `/dashboards/hardcore-quant` | Hardcore Quant | Quant Researchers |
| `/dashboards/regimes` | Market Regimes | All Users |
| `/dashboards/signals` | Unified Signals | All Users |
| `/dashboards/analytics` | Cross-Asset Analytics | Analysts |
| `/dashboards/backtest` | Walk-Forward Backtest | Quant Analysts |

---

## Ensemble Methods (12 Methods)

### Tier 1: Core Methods

| Method | Class | Description |
|--------|-------|-------------|
| Accuracy Weighted | `AccuracyWeightedEnsemble` | Weights pairs by historical directional accuracy |
| Magnitude Voting | `MagnitudeWeightedVoting` | Weights by signal magnitude (stronger = more confident) |
| Error Correlation | `ErrorCorrelationWeighting` | Downweights pairs with correlated prediction errors |
| Combined Tier 1 | `CombinedTier1Ensemble` | Meta-ensemble combining all Tier 1 methods |

**File:** `backend/ensemble_tier1.py` (971 lines)

### Tier 2: Advanced Methods

| Method | Class | Description |
|--------|-------|-------------|
| Bayesian Model Avg | `BayesianModelAveraging` | Posterior-weighted averaging with uncertainty quantification |
| Regime Adaptive | `RegimeAdaptiveEnsemble` | Learns different weights per HMM-detected regime |
| Conformal Prediction | `ConformalPredictionInterval` | Distribution-free calibrated prediction intervals |
| Combined Tier 2 | `CombinedTier2Ensemble` | Meta-ensemble combining all Tier 2 methods |

**File:** `backend/ensemble_tier2.py` (1,259 lines)

### Tier 3: Research Methods

| Method | Class | Description |
|--------|-------|-------------|
| Thompson Sampling | `ThompsonSamplingEnsemble` | Online learning via multi-armed bandit exploration |
| Attention Based | `AttentionBasedEnsemble` | Transformer-style context-dependent weighting |
| Quantile Forest | `QuantileRegressionForest` | Non-parametric prediction intervals via random forests |
| Combined Tier 3 | `CombinedTier3Ensemble` | Meta-ensemble combining all Tier 3 methods |

**File:** `backend/ensemble_tier3.py` (1,823 lines)

---

## API Endpoint Reference

**Base URL:** `http://localhost:5001/api/v1`

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check and version |
| GET | `/assets` | List available assets |
| GET | `/signals/{asset}` | Trading signals |
| GET | `/forecast/{asset}` | Live forecast data |
| GET | `/metrics/{asset}` | Performance metrics |
| GET | `/ohlcv/{asset}` | OHLCV price data |
| GET | `/equity/{asset}` | Equity curve |

### Ensemble Tier Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/ensemble/tier1/{asset_id}` | Tier 1 prediction |
| GET | `/ensemble/tier2/{asset_id}` | Tier 2 prediction |
| GET | `/ensemble/tier3/{asset_id}` | Tier 3 prediction |
| GET | `/ensemble/tiers/{asset_id}` | All tiers combined |

### Backtest Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/backtest/walk-forward/{asset_id}` | Walk-forward validation |
| GET | `/backtest/equity-curve/{asset_id}` | Strategy equity curve |
| GET | `/backtest/regime-performance/{asset_id}` | Per-regime breakdown |
| GET | `/backtest/methods` | Available methods list |

### Regime & Quantum

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/regime/{asset}` | HMM regime detection |
| GET | `/quantum/dashboard` | Regime status & contagion |

---

## Backtesting Framework

### Walk-Forward Validation

Rigorous out-of-sample testing with configurable folds, cost modeling, and regime analysis.

```python
from backend.backtesting.walk_forward import WalkForwardValidator

validator = WalkForwardValidator(
    n_folds=5,              # Walk-forward folds
    train_pct=0.7,          # Training split
    transaction_cost_bps=5, # Cost in basis points
)
results = validator.run_comparison(methods, asset_id, asset_name)
```

### Transaction Cost Model

Realistic cost estimation with fixed costs, percentage costs, spread, and market impact.

### Metrics Returned

| Metric | Description |
|--------|-------------|
| `accuracy` | Directional accuracy (%) |
| `sharpe_ratio` | Annualized Sharpe |
| `sortino_ratio` | Annualized Sortino |
| `max_drawdown` | Maximum drawdown (%) |
| `win_rate` | Winning trade % |
| `total_costs` | Transaction costs ($) |
| `regime_performance` | Per-regime breakdown |

---

## Today's Commits (31)

```
f706483 feat: Wire up BacktestDashboard to real API endpoints
f0df676 feat: Add backtest API endpoints for walk-forward validation
aab249d feat: Add walk-forward backtest dashboard with ensemble comparison
836d374 feat: Add transaction cost modeling to backtesting framework
9b91108 feat: Add walk-forward validation framework for ensemble methods
db5ee61 feat: Add individual tier ensemble hooks
c1e86b5 feat: Add TierComparisonCard component
136c76b feat: Add Tier 1/2/3 ensemble API endpoints
bc7b076 feat: Implement Tier 3 ensemble methods (1,823 lines)
a34c71f feat: Implement Tier 2 ensemble methods (1,259 lines)
d494511 feat: Implement Tier 1 ensemble methods (971 lines)
26d2f78 feat: Create /dashboards/analytics page
9f83a61 refactor: Migrate ensemble files to standardized Sharpe
0244884 feat: Standardize Sharpe calculation with shared utility
5cdd7f7 feat: Create unified /dashboards/signals page
74cc803 feat: Add early warning signals to RegimeIndicator
cd9f457 feat: Update dashboards to use API-connected ensembles
6ae98a7 feat: Add voting/intervals endpoint aliases
b238194 feat: Add MultiAssetRegimeOverview to dashboards
4bf5a0d feat: Add multi-asset HMM regime support
7879cd0 feat: Add API integration for all ensemble components
1e045b5 chore: Enable real API calls
455db3c feat: Update dashboards to use real HMM regime API
3e17521 feat: Add HMM regime detection API endpoint
455d61a feat: Add HMM-based market regime detection
134e480 docs: Add Sharpe ratio discrepancy analysis
4e10c40 feat: Integrate ensemble components into persona dashboards
1458222 feat: Integrate ensemble components into AlphaProDashboard
4ca6073 feat: Add quantum components and enhance persona dashboards
f045655 feat: Add ensemble visualization components
7fbad12 feat: Add routes for all 7 persona dashboards + index page
```

---

## Summary

| Category | Count |
|----------|-------|
| Dashboard Routes | 12 |
| Ensemble Methods | 12 |
| API Endpoints | 18 |
| Python Files Added | 3 |
| Total LOC Added | ~24,400 |
| Commits | 31 |
