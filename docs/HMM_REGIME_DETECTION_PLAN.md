# HMM Regime Detection Implementation Plan

**Date:** February 6, 2026
**Author:** Artemis
**Status:** Planning

---

## Current State Analysis

### Existing Regime Detection (Not HMM)

| File | Method | Regimes |
|------|--------|---------|
| `quantum_regime_detector.py` | Quantum-inspired Hamiltonian | MOMENTUM, MEAN_REVERT, HIGH_VOL, LOW_VOL |
| `quantum_volatility_detector.py` | Qiskit quantum circuits | LOW_VOL, NORMAL, ELEVATED, CRISIS |
| `quantum_regime_v2.py` | Qiskit Statevector | Multi-scale quantum |

### Frontend Component

`frontend/components/ensemble/RegimeIndicator.tsx`:
- Shows "HMM Detected" badge (line 207) **but backend doesn't use HMM**
- Expects regimes: `bull`, `bear`, `sideways`, `high-volatility`, `low-volatility`

### Planned but Not Implemented

`docs/ENSEMBLE_METHODS_PLAN.md:248-256` shows HMM code snippet:
```python
from hmmlearn import hmm
model = hmm.GaussianHMM(n_components=3, covariance_type="full")
```

---

## Implementation Plan

### Phase 1: Core HMM Regime Detector

Create `hmm_regime_detector.py`:

```python
"""
HMM-BASED MARKET REGIME DETECTION
=================================
Uses Hidden Markov Models for statistically rigorous regime detection.

Regimes:
- BULL: Positive drift, low-moderate volatility
- BEAR: Negative drift, elevated volatility
- SIDEWAYS: Near-zero drift, compressed volatility
- HIGH_VOL: Any drift, extreme volatility
- LOW_VOL: Low volatility, weak trend

Features:
- Returns (momentum signal)
- Realized volatility (risk level)
- Return autocorrelation (mean reversion indicator)
"""

from hmmlearn import hmm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class HMMRegimeDetector:
    def __init__(self, n_regimes: int = 3, lookback: int = 252):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.regime_labels = self._assign_regime_labels()

    def extract_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract HMM observation features."""
        returns = np.diff(np.log(prices))

        # Rolling features
        features = []
        window = 20

        for i in range(window, len(returns)):
            r_window = returns[i-window:i]

            # Feature 1: Recent returns (momentum)
            momentum = r_window[-5:].mean() * 252

            # Feature 2: Realized volatility
            vol = r_window.std() * np.sqrt(252)

            # Feature 3: Autocorrelation (mean reversion)
            if len(r_window) > 1:
                autocorr = np.corrcoef(r_window[:-1], r_window[1:])[0,1]
                autocorr = 0 if np.isnan(autocorr) else autocorr
            else:
                autocorr = 0

            features.append([momentum, vol, autocorr])

        return np.array(features)

    def fit(self, prices: np.ndarray):
        """Fit HMM on price data."""
        features = self.extract_features(prices)
        features_scaled = self.scaler.fit_transform(features)

        self.model.fit(features_scaled)
        self.trained = True

        # Assign semantic labels based on learned means
        self._assign_regime_labels()

        return self

    def _assign_regime_labels(self) -> dict:
        """Assign semantic labels to HMM states based on means."""
        if not self.trained:
            return {i: f"regime_{i}" for i in range(self.n_regimes)}

        means = self.model.means_

        # Sort by volatility (feature index 1)
        vol_order = np.argsort(means[:, 1])

        labels = {}
        if self.n_regimes == 3:
            labels[vol_order[0]] = "low-volatility"
            labels[vol_order[1]] = "sideways"
            labels[vol_order[2]] = "high-volatility"
        elif self.n_regimes == 4:
            labels[vol_order[0]] = "low-volatility"
            labels[vol_order[1]] = "sideways"
            labels[vol_order[2]] = "bull" if means[vol_order[2], 0] > 0 else "bear"
            labels[vol_order[3]] = "high-volatility"

        self.regime_labels = labels
        return labels

    def predict(self, prices: np.ndarray) -> dict:
        """Predict current regime with probabilities."""
        features = self.extract_features(prices)
        features_scaled = self.scaler.transform(features)

        # Get state sequence
        hidden_states = self.model.predict(features_scaled)
        current_state = hidden_states[-1]

        # Get probabilities
        log_probs = self.model.predict_proba(features_scaled)
        current_probs = log_probs[-1]

        # Days in current regime
        days_in_regime = 1
        for i in range(len(hidden_states) - 2, -1, -1):
            if hidden_states[i] == current_state:
                days_in_regime += 1
            else:
                break

        regime_name = self.regime_labels.get(current_state, "unknown")

        return {
            "regime": regime_name,
            "confidence": float(current_probs[current_state]),
            "probabilities": {
                self.regime_labels.get(i, f"state_{i}"): float(p)
                for i, p in enumerate(current_probs)
            },
            "daysInRegime": days_in_regime,
            "transitionMatrix": self.model.transmat_.tolist()
        }
```

### Phase 2: API Endpoint

Add to `api_ensemble.py`:

```python
from hmm_regime_detector import HMMRegimeDetector

# Cache trained models
hmm_models = {}

@app.route('/api/v1/hmm/regime/<int:asset_id>', methods=['GET'])
def get_hmm_regime(asset_id):
    """Get HMM-detected market regime."""
    asset_name = ASSET_MAP.get(asset_id)
    if not asset_name:
        return jsonify({"error": "Unknown asset"}), 404

    # Load or train model
    if asset_id not in hmm_models:
        prices = load_prices(asset_id)
        model = HMMRegimeDetector(n_regimes=3)
        model.fit(prices)
        hmm_models[asset_id] = model

    model = hmm_models[asset_id]
    prices = load_recent_prices(asset_id, lookback=252)

    result = model.predict(prices)
    return jsonify(result)
```

### Phase 3: Frontend Integration

Update `frontend/lib/api.ts`:

```typescript
export async function getHMMRegime(assetId: AssetId): Promise<RegimeData> {
  const response = await fetch(getApiUrl(`/hmm/regime/${assetId}`));
  if (!response.ok) {
    throw new Error(`Failed to fetch HMM regime: ${response.status}`);
  }
  return response.json();
}
```

### Phase 4: Regime Label Mapping

| HMM State Characteristic | Frontend Label | Color |
|--------------------------|----------------|-------|
| Low vol, weak trend | `low-volatility` | Blue |
| Moderate vol, near-zero drift | `sideways` | Amber |
| Moderate vol, positive drift | `bull` | Green |
| Moderate vol, negative drift | `bear` | Red |
| High vol, any drift | `high-volatility` | Orange |

---

## Dependencies

```bash
pip install hmmlearn scikit-learn
```

**hmmlearn** provides:
- `GaussianHMM` - Continuous observations
- `MultinomialHMM` - Discrete observations
- `GMMHMM` - Gaussian mixture emissions

---

## Comparison: HMM vs Current "Quantum" Approach

| Aspect | HMM | Quantum-Inspired |
|--------|-----|------------------|
| **Foundation** | Statistical (Baum-Welch) | Linear algebra (Hamiltonian) |
| **Transition Model** | Learned from data | Hand-crafted matrix |
| **Regime Assignment** | Unsupervised clustering | Rule-based |
| **Confidence** | Log-likelihood based | Amplitude squared |
| **Industry Standard** | Yes (widely used) | No (novel) |
| **Interpretability** | High | Medium |

### Recommendation

Use **both** approaches:
1. **HMM** as primary regime detector (statistically rigorous)
2. **Quantum** as auxiliary signal (captures non-linear patterns)
3. Ensemble when they agree, cautious when they disagree

---

## Testing Plan

1. **Unit tests** for HMMRegimeDetector class
2. **Backtest** regime-conditional strategy performance
3. **Compare** HMM vs quantum detector agreement rate
4. **Validate** regime persistence (should be sticky, not noisy)

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `hmm_regime_detector.py` | **CREATE** | Core HMM implementation |
| `api_ensemble.py` | MODIFY | Add `/api/v1/hmm/regime` endpoint |
| `frontend/lib/api.ts` | MODIFY | Add `getHMMRegime()` function |
| `frontend/components/ensemble/RegimeIndicator.tsx` | MODIFY | Use real HMM data |
| `tests/test_hmm_regime.py` | **CREATE** | Unit tests |

---

## Success Criteria

1. HMM correctly identifies bull/bear/sideways regimes on historical data
2. Regime transitions are smooth (not flickering daily)
3. Strategy Sharpe improves when adapting to regime
4. Frontend displays real HMM-detected regimes (not mock data)

---

*Plan created: February 6, 2026*
