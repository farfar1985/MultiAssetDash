# Early Warning System for Regime Transitions

**Author:** AmiraB  
**Date:** 2026-02-06  
**Status:** ✅ Complete

## Overview

The Early Warning System detects regime changes **before** they happen by monitoring transition probabilities from trained HMM models. Instead of reacting to regime changes after they occur, traders get advance notice when the probability of a transition is elevated.

## How It Works

### 1. Transition Probability
Each HMM model has a **transition matrix** that tells us the probability of moving from one regime to another:

```
        bull  bear  sideways
bull   [0.95, 0.03, 0.02]    <- 5% chance of leaving bull regime
bear   [0.04, 0.92, 0.04]    <- 8% chance of leaving bear regime
sideways[0.05, 0.05, 0.90]   <- 10% chance of leaving sideways
```

**P(regime_change) = 1 - P(stay_in_current_regime)**

### 2. Alert Thresholds

| Level | P(change) | Meaning |
|-------|-----------|---------|
| Normal | < 30% | Business as usual |
| Watch | 30-50% | Elevated risk, monitor closely |
| Warning | 50-70% | High risk, prepare for transition |
| Critical | > 70% | Imminent transition likely |

### 3. Trend Detection
We track P(change) over time to detect:
- **Rising trend**: Probability increasing over last N days
- **Acceleration**: Rate of increase itself increasing
- **Duration stress**: Long regime + rising probability

## Files Created

| File | Purpose |
|------|---------|
| `early_warning_system.py` | Core EWS class with all detection logic |
| `transition_trend_tracker.py` | Historical trend analysis |
| `api_early_warning.py` | Flask API endpoints (port 5002) |
| `run_ews.py` | Quick console runner |

## API Endpoints

Start API: `python api_early_warning.py`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/early-warning/status` | Overall risk status |
| GET | `/api/v1/early-warning/alerts` | Active alerts |
| GET | `/api/v1/early-warning/assets` | All asset probabilities |
| GET | `/api/v1/early-warning/asset/<id>` | Single asset detail |
| GET | `/api/v1/early-warning/contagion` | Contagion risk by group |
| GET | `/api/v1/early-warning/dashboard` | Full dashboard data |
| POST | `/api/v1/early-warning/refresh` | Force refresh |

## Output Example

```json
{
  "asset": "1625_SP500",
  "current_regime": "sideways",
  "confidence": 1.0,
  "regime_duration_days": 262,
  "p_stay": 1.0,
  "p_change": 0.0,
  "p_to_regimes": {
    "bear": 0.0,
    "bull": 0.0
  },
  "alert_level": "normal"
}
```

## Asset Groups (Contagion Detection)

The system groups assets to detect coordinated stress:

- **US_Equities**: S&P 500, NASDAQ, Russell, Dow Jones
- **India**: Nifty Bank, Nifty 50
- **Commodities**: Gold, Copper, Brent Oil
- **FX**: USD/INR, Dollar Index
- **International**: China ETF, Nikkei 225

When multiple assets in a group show elevated P(change), it signals potential contagion.

## Integration with Dashboard

The dashboard can poll `/api/v1/early-warning/dashboard` to get:
1. Overall risk level (LOW/MODERATE/HIGH/CRITICAL)
2. Active alerts with asset details
3. Contagion risk by group
4. Per-asset transition probabilities

## Trend Tracker Usage

```python
from transition_trend_tracker import TransitionTrendTracker

tracker = TransitionTrendTracker()

# Record current snapshot (run periodically)
tracker.record_snapshot()

# Get rising assets
rising = tracker.get_rising_assets()

# Get early warnings
warnings = tracker.get_early_warnings()

# Full report
report = tracker.generate_trend_report()
```

## Current Status (2026-02-06)

- **Total assets monitored**: 13 (2 failed: Bitcoin, Crude Oil)
- **Overall risk**: LOW
- **Avg P(change)**: 4.3%
- **All groups**: Low contagion risk

## Next Steps

1. Set up cron job to run `tracker.record_snapshot()` hourly
2. Build historical trend database (need more snapshots)
3. Integrate with Nexus dashboard
4. Add email/Telegram alerts for Warning/Critical levels
