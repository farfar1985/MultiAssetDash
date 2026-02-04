# Nexus Architecture

## System Overview

```
QML (Model Factory)                    Nexus (Dashboard)
┌─────────────────────┐               ┌──────────────────────────┐
│                     │               │                          │
│  Users / Agents     │               │  Persona Views           │
│       │             │               │  ├── Hedging Desk        │
│       ▼             │               │  ├── Institutional       │
│  Build ML Models    │    outputs    │  ├── Wealth Manager      │
│  ├── CatBoost       │──────────────▶│  ├── Hedge Fund          │
│  ├── GBR            │               │  ├── Retail              │
│  ├── XGBoost        │               │  └── Casual              │
│  ├── (any sklearn)  │               │                          │
│  └── Custom         │               │  Features                │
│       │             │               │  ├── Ensemble consensus  │
│       ▼             │               │  ├── Model comparison    │
│  Train & Validate   │               │  ├── Disagreement view   │
│  ├── Backtest       │   metadata    │  ├── Drill-down          │
│  ├── Cross-val      │──────────────▶│  ├── Historical accuracy │
│  ├── Out-of-sample  │               │  ├── Factor attribution  │
│  └── Feature imp.   │               │  └── Audit trail         │
│                     │               │                          │
└─────────────────────┘               └──────────────────────────┘
```

## Separation of Concerns

### QML Owns:
- Model training and validation
- Feature engineering
- Backtesting methodology
- Statistical rigor (cross-validation, out-of-sample, etc.)
- Model metadata (algorithm, parameters, training window, features used)
- Performance metrics (Sharpe, accuracy, precision, recall, feature importance)

### Nexus Owns:
- **Faithful representation** of QML outputs (no interpretation, no added noise)
- Ensemble aggregation (combining N models into consensus views)
- Persona-appropriate context and language
- Visualization and interaction design
- Drill-down from summary → individual model → methodology
- Audit trail and provenance display
- Export/reporting for institutional use

### Critical Rule
> Nexus never invents a number. Every value displayed traces back to a QML model output or a transparent aggregation of QML outputs.

## Data Flow

```
QML Model Output (per model):
{
  "model_id": "cat_aapl_90d_v3",
  "algorithm": "CatBoost",
  "target": "AAPL_30d_return",
  "forecast": 0.034,           // +3.4% expected return
  "direction": "long",
  "confidence": 0.78,          // model-reported probability
  "features_used": ["momentum_20d", "put_call_ratio", "earnings_surprise", ...],
  "feature_importance": {"momentum_20d": 0.31, "put_call_ratio": 0.22, ...},
  "backtest": {
    "sharpe": 1.4,
    "accuracy": 0.62,
    "max_drawdown": -0.08,
    "out_of_sample_r2": 0.14,
    "training_window": "2020-01-01 to 2025-06-30",
    "test_window": "2025-07-01 to 2025-12-31"
  },
  "timestamp": "2026-02-01T14:30:00Z"
}
```

```
Nexus Ensemble View (aggregated):
{
  "asset": "AAPL",
  "models_total": 1521,
  "models_bullish": 1247,
  "consensus": 0.82,
  "consensus_method": "sharpe_weighted",  // transparent
  "median_forecast": 0.028,
  "forecast_distribution": {...},          // full distribution, not just point est.
  "model_agreement": 0.67,                // how much models agree (low = uncertainty)
  "top_contributing_factors": [...],
  "regime_flag": "normal_vol",
  "last_updated": "2026-02-01T14:30:00Z"
}
```

## Model Ensemble Methods
Nexus must support (and clearly label) multiple aggregation methods:
- Equal weight (naive, baseline)
- Performance-weighted (trailing Sharpe, accuracy, etc.)
- Recency-weighted (favor recently accurate models)
- Diversity-weighted (penalize correlated models)
- Custom (user-defined weighting schemes)

The active method must always be visible. A quant should never wonder "how did you get this number?"

## Tech Stack (TBD)
- Frontend: Next.js
- API layer: TBD (needs to interface with QML outputs)
- Real-time: WebSocket for live model updates
- Auth: Role-based (persona = role)

---

*Updated as QML integration details become clearer.*
