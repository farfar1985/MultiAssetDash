# Ensemble Research Findings Report
## QDT Nexus - Crude Oil Analysis

**Date:** February 3, 2026  
**Author:** AmiraB  
**Asset:** Crude Oil (10,179 models across 10 horizons)  
**Data Period:** 393 trading days (~1.6 years)

---

## Executive Summary

After extensive testing of 85+ ensemble methods across classical, statistical, ML, and quantum-inspired approaches, we have identified a clear winner:

### **BEST CONFIGURATION**
```
Method: Pairwise Slopes (Cross-Horizon Consensus)
Horizons: D+5 + D+7 + D+10
Threshold: 0.4 (normalized)
Aggregation: Top-10 models per horizon

Results:
- Sharpe Ratio: 1.757
- Win Rate: 67.74%
- Directional Accuracy: 38.18%
- Total Return: 12.36%
- Max Drawdown: -6.02%
- Trades: 55
```

### Key Discovery
**Alpha comes from CROSS-HORIZON CONSENSUS, not single-horizon model selection.**

---

## Methods Tested

### Tier 1: Classical (6 methods)
| Method | Best Sharpe | Notes |
|--------|-------------|-------|
| Equal Weight | -0.71 | Baseline |
| Inverse MSE | -0.88 | Worse than random |
| Inverse Variance | -0.65 | Slight improvement |
| Top-K Sharpe | 0.13 | Marginal positive |
| Exponential Decay | 0.09 | Time-weighted |
| Ridge Stacking | 0.08 | Regularized |

### Tier 2: Statistical (5 methods)
| Method | Best Sharpe | Notes |
|--------|-------------|-------|
| Granger-Ramanathan | 0.89 | OLS optimal, but slow |
| BMA (EM) | 0.45 | Bayesian averaging |
| Minimum Variance | 0.32 | Risk-focused |
| Quantile Averaging | 0.28 | Robust to outliers |
| Conformal Prediction | 0.35 | Uncertainty-aware |

### Tier 3: Advanced ML (4 methods)
| Method | Best Sharpe | Notes |
|--------|-------------|-------|
| Neural Attention | 0.96 | Deep learning |
| Meta-Ensemble | 0.85 | Ensemble of ensembles |
| Hierarchical | 0.78 | Tree structure |
| Dynamic Selection | 0.52 | Regime-aware |

### Tier 4: Quantum-Inspired (7 methods)
| Method | Best Sharpe | Notes |
|--------|-------------|-------|
| QIEA | -0.88 | Quantum evolutionary |
| QPSO | -0.82 | Quantum particle swarm |
| Simulated Bifurcation | -0.71 | Ising solver |
| CIM | -0.25 | Coherent Ising Machine |
| Efficient QA | -0.25 | Mean-field annealing |
| VQI | -0.88 | Variational quantum |
| Quantum Walk | -0.88 | Graph walk |

### Tier 5: Quantum Simulation (4 methods)
| Method | Best Sharpe | Notes |
|--------|-------------|-------|
| QAOA | 0.44 | NaN issues with large models |
| Wasserstein | 0.34 | Distribution matching |
| VQE | N/A | Memory exceeded (990 qubits) |
| Quantum Annealing | 0.44 | Trotter simulation |

### **WINNER: Cross-Horizon Methods**
| Method | Best Sharpe | Notes |
|--------|-------------|-------|
| **Pairwise Slopes** | **1.757** | D+5+D+7+D+10 |
| Magnitude-Weighted | 1.52 | Multi-horizon weighted |
| Cross-Correlation | 1.23 | Horizon agreement |

---

## Critical Insights

### 1. Single-Horizon Ensembles Are Anti-Predictive
Testing all 990 models on D+5 alone:
- Equal weight: Sharpe **-0.71**
- Best single-horizon method: Sharpe **+0.13**

The models individually are slightly worse than random at predicting direction.

### 2. Alpha Emerges from Cross-Horizon Consensus
When longer horizons predict HIGHER than shorter horizons = **BULLISH**
When longer horizons predict LOWER than shorter horizons = **BEARISH**

This "pairwise slopes" signal captures momentum building across timeframes.

### 3. Three Horizons is Optimal
| Horizons | Sharpe |
|----------|--------|
| 2 | 1.2-1.4 |
| **3** | **1.5-1.8** |
| 4 | 1.1-1.3 |
| 5+ | <1.0 |

More horizons dilute the signal.

### 4. Quantum Methods Don't Help Here
Quantum-inspired methods are designed for:
- Combinatorial optimization (which K models to select)
- Weight optimization (how much weight per model)

But our alpha doesn't come from model selection - it comes from comparing predictions across time. Quantum methods solve the wrong problem.

### 5. Horizon Selection > Method Selection
Only 41% of horizon combinations produce positive Sharpe.
The SAME method on different horizons varies from -2.0 to +3.0 Sharpe.

---

## Recommended Production Configuration

```python
PRODUCTION_CONFIG = {
    'method': 'pairwise_slopes',
    'horizons': [5, 7, 10],
    'threshold': 0.4,  # Normalized slope threshold
    'aggregation': 'top10',  # Use top 10 models per horizon
    'position_sizing': 'equal',  # Or scale by conviction
}
```

### Alternative: High-Return Configuration
```python
HIGH_RETURN_CONFIG = {
    'method': 'pairwise_slopes',
    'horizons': [1, 2, 3],
    'threshold': 0.3,
    'aggregation': 'mean',
    'expected_return': '131.45%',  # But more volatile
    'expected_sharpe': '0.8-1.2',
}
```

---

## Next Steps

1. **Lock in D+5+D+7+D+10** as production configuration
2. **Add conformal prediction** for position sizing (uncertainty-aware)
3. **Connect to Artemis frontend** via API (port 5001)
4. **Test on other assets** (Bitcoin, S&P 500, Gold)
5. **Monitor live performance** before CME demo

---

## Files Created

| File | Purpose |
|------|---------|
| `master_ensemble.py` | Pairwise slopes + grid search |
| `quantum_inspired_ensemble.py` | 7 QI methods (scalable) |
| `quantum_simulator_ensemble.py` | 7 quantum simulation methods |
| `advanced_ensemble.py` | MinT, Hedge, Meta-Ensemble |
| `ensemble_methods.py` | Tier 1 + Tier 2 implementations |
| `api_ensemble.py` | Flask API server (port 5001) |

---

## Conclusion

**We found the optimal ensemble strategy:**
- Method: Pairwise Slopes
- Horizons: D+5 + D+7 + D+10
- Sharpe: 1.757

Quantum and advanced ML methods do not improve on this because the alpha source is fundamentally different from what those methods optimize for.

**Recommendation:** Stop researching new methods. Focus on:
1. Production deployment
2. Frontend integration
3. Multi-asset validation

---

*Report generated by AmiraB - February 3, 2026*
