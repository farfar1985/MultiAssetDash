# Nexus API Contract

**Version:** 1.0  
**Date:** 2026-02-03  
**Author:** AmiraB  
**For:** Artemis (Frontend Integration)

---

## Overview

This document defines the API contract between the Nexus backend (ensemble engine) and the frontend (Next.js dashboard). All endpoints return JSON.

---

## Base URL

```
http://localhost:5000/api/v1
```

---

## Endpoints

### 1. Get Current Signal

**`GET /signal/{asset_id}`**

Returns the current trading signal with confidence metrics.

**Response:**
```json
{
  "asset_id": 1866,
  "asset_name": "Crude_Oil",
  "timestamp": "2026-02-03T19:00:00Z",
  "signal": {
    "direction": "BULLISH",
    "probability": 0.72,
    "confidence": "HIGH",
    "horizons_agreeing": 5,
    "total_horizons": 6
  },
  "forecast": {
    "current_price": 78.45,
    "target_1": 79.02,
    "target_2": 79.85,
    "stop_loss": 77.88,
    "expected_move_pct": 0.73,
    "expected_move_usd": 0.57
  },
  "metrics": {
    "sharpe": 2.48,
    "practical_sharpe": 3.10,
    "utility_score": 0.639,
    "directional_accuracy": 41.9,
    "significant_move_accuracy": 41.9
  },
  "ensemble_config": {
    "method": "magnitude_weighted",
    "horizons_used": [8, 9, 10],
    "models_per_horizon": 165,
    "total_models": 495
  }
}
```

---

### 2. Get Horizon Breakdown

**`GET /horizons/{asset_id}`**

Returns signals and forecasts for each individual horizon.

**Response:**
```json
{
  "asset_id": 1866,
  "asset_name": "Crude_Oil",
  "timestamp": "2026-02-03T19:00:00Z",
  "horizons": [
    {
      "horizon": 1,
      "label": "D+1",
      "signal": "BULLISH",
      "probability": 0.65,
      "forecast_price": 78.72,
      "forecast_change_pct": 0.34,
      "models_used": 165,
      "model_agreement": 0.78
    },
    {
      "horizon": 3,
      "label": "D+3",
      "signal": "BULLISH",
      "probability": 0.71,
      "forecast_price": 79.15,
      "forecast_change_pct": 0.89,
      "models_used": 165,
      "model_agreement": 0.82
    }
    // ... more horizons
  ],
  "cross_horizon_correlation": {
    "D+1_D+3": 0.47,
    "D+5_D+10": 0.74,
    "D+7_D+10": 0.79
  }
}
```

---

### 3. Get Historical Performance

**`GET /performance/{asset_id}?days=90`**

Returns backtested performance metrics.

**Query Params:**
- `days` (optional, default 90): Lookback period

**Response:**
```json
{
  "asset_id": 1866,
  "asset_name": "Crude_Oil",
  "period": {
    "start": "2025-11-06",
    "end": "2026-02-03",
    "trading_days": 90
  },
  "performance": {
    "total_return_pct": 24.2,
    "sharpe_ratio": 2.09,
    "practical_sharpe": 3.10,
    "max_drawdown_pct": -8.5,
    "win_rate": 54.2,
    "profit_factor": 1.85
  },
  "signal_stats": {
    "total_signals": 89,
    "bullish_signals": 52,
    "bearish_signals": 37,
    "avg_hold_days": 3.2
  },
  "by_horizon": [
    {
      "horizon": 5,
      "sharpe": 0.72,
      "directional_accuracy": 37.8,
      "contribution_pct": 15
    }
    // ... more horizons
  ]
}
```

---

### 4. Get Model Heatmap Data

**`GET /heatmap/{asset_id}?days=30`**

Returns data for the model agreement heatmap visualization.

**Response:**
```json
{
  "asset_id": 1866,
  "dates": ["2026-01-04", "2026-01-05", ...],
  "horizons": ["D+1", "D+3", "D+5", "D+7", "D+10"],
  "data": [
    {
      "date": "2026-02-03",
      "values": {
        "D+1": 0.78,
        "D+3": 0.82,
        "D+5": 0.65,
        "D+7": 0.71,
        "D+10": 0.69
      }
    }
    // ... more dates
  ],
  "color_scale": {
    "min": -1,
    "max": 1,
    "neutral": 0
  }
}
```

---

### 5. Get Forecast Fan

**`GET /forecast-fan/{asset_id}`**

Returns multi-horizon forecast data for fan chart visualization.

**Response:**
```json
{
  "asset_id": 1866,
  "current_price": 78.45,
  "timestamp": "2026-02-03T19:00:00Z",
  "forecasts": [
    {
      "horizon": 1,
      "date": "2026-02-04",
      "forecast": 78.72,
      "confidence_interval": {
        "lower_5": 77.95,
        "lower_25": 78.32,
        "upper_75": 79.12,
        "upper_95": 79.49
      }
    },
    {
      "horizon": 5,
      "date": "2026-02-08",
      "forecast": 79.85,
      "confidence_interval": {
        "lower_5": 77.20,
        "lower_25": 78.50,
        "upper_75": 81.20,
        "upper_95": 82.50
      }
    }
    // ... more horizons
  ]
}
```

---

### 6. List Assets

**`GET /assets`**

Returns all available assets.

**Response:**
```json
{
  "assets": [
    {
      "id": 1866,
      "name": "Crude_Oil",
      "display_name": "Crude Oil (WTI)",
      "category": "Commodities",
      "horizons_available": [1, 2, 3, 5, 7, 10],
      "models_count": 990,
      "last_updated": "2026-02-03T19:00:00Z"
    },
    {
      "id": 1625,
      "name": "SP500",
      "display_name": "S&P 500",
      "category": "Indices",
      "horizons_available": [1, 3, 5, 8, 13, 21, 34, 55, 89],
      "models_count": 442,
      "last_updated": "2026-02-03T19:00:00Z"
    }
    // ... more assets
  ]
}
```

---

### 7. Get Ensemble Configuration

**`GET /ensemble-config/{asset_id}`**

Returns the current ensemble configuration for an asset.

**Response:**
```json
{
  "asset_id": 1866,
  "method": "magnitude_weighted",
  "method_description": "Weight horizon signals by forecast magnitude",
  "horizons": {
    "selected": [8, 9, 10],
    "available": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  },
  "parameters": {
    "lookback_window": 60,
    "min_forecast_threshold": 0.50,
    "signal_threshold": 0.2
  },
  "series_structure": {
    "parent_series": 6,
    "models_per_series": 165,
    "total_models": 990
  },
  "last_optimized": "2026-02-03T18:00:00Z"
}
```

---

## WebSocket (Future)

For real-time updates (not implemented yet):

```
ws://localhost:5000/ws/signals
```

---

## Error Responses

All errors return:
```json
{
  "error": true,
  "code": "ASSET_NOT_FOUND",
  "message": "Asset with ID 9999 not found",
  "timestamp": "2026-02-03T19:00:00Z"
}
```

Error codes:
- `ASSET_NOT_FOUND` - Invalid asset ID
- `INSUFFICIENT_DATA` - Not enough historical data
- `ENSEMBLE_ERROR` - Ensemble computation failed
- `RATE_LIMITED` - Too many requests

---

## Rate Limits

- 100 requests per minute per IP
- 1000 requests per hour per IP

---

## Notes for Frontend

1. **Polling interval:** Every 60 seconds for signals (daily data)
2. **Cache:** Heatmap and performance data can be cached for 5 min
3. **Error handling:** Always check for `error: true` in responses
4. **Timestamps:** All in ISO 8601 format (UTC)
5. **Null handling:** Missing data returns `null`, not undefined

---

## Next Steps

1. Implement Flask routes matching this contract
2. Add authentication (API keys)
3. Add WebSocket for real-time updates
4. Add batch endpoints for multiple assets

---

*Generated by AmiraB for Artemis integration*
