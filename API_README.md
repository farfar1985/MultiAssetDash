# QDTNexus API Documentation

REST API for accessing QDT Ensemble data with API key authentication.

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors
```

### 2. Start the API Server

```bash
python api_server.py
```

Server runs on `http://localhost:5000` by default.

### 3. Create an API Key

```bash
python manage_api_keys.py create --user-id user1 --assets Crude_Oil Bitcoin SP500
```

This will output an API key like: `qdt_abc123...`

### 4. Make API Requests

```bash
# Using header
curl -H "X-API-Key: your_api_key_here" http://localhost:5000/api/v1/assets

# Using query parameter
curl "http://localhost:5000/api/v1/assets?api_key=your_api_key_here"
```

---

## API Endpoints

### Health Check
```
GET /api/v1/health
```
No authentication required. Returns API status.

**Response:**
```json
{
  "success": true,
  "service": "QDTNexus API",
  "version": "1.0.0",
  "timestamp": "2025-02-02T10:00:00"
}
```

---

### List Assets
```
GET /api/v1/assets
```
Returns all assets the API key has access to.

**Response:**
```json
{
  "success": true,
  "assets": [
    {
      "name": "Crude_Oil",
      "id": "1866",
      "display_name": "Crude Oil"
    }
  ],
  "count": 1,
  "timestamp": "2025-02-02T10:00:00"
}
```

---

### Historical OHLCV Data
```
GET /api/v1/ohlcv/{asset}
```
Get historical Open, High, Low, Close, Volume data.

**Query Parameters:**
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)

**Example:**
```bash
curl -H "X-API-Key: your_key" \
  "http://localhost:5000/api/v1/ohlcv/Crude_Oil?start_date=2024-01-01&end_date=2024-12-31"
```

**Response:**
```json
{
  "success": true,
  "asset": "Crude_Oil",
  "data": [
    {
      "date": "2024-01-01",
      "open": 71.65,
      "high": 71.65,
      "low": 71.65,
      "close": 71.65,
      "volume": 0
    }
  ],
  "count": 365,
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "timestamp": "2025-02-02T10:00:00"
}
```

---

### Historical Signals (Snake Chart)
```
GET /api/v1/signals/{asset}
```
Get historical trading signals (BULLISH/BEARISH/NEUTRAL).

**Query Parameters:**
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)

**Example:**
```bash
curl -H "X-API-Key: your_key" \
  "http://localhost:5000/api/v1/signals/Crude_Oil"
```

**Response:**
```json
{
  "success": true,
  "asset": "Crude_Oil",
  "data": [
    {
      "date": "2024-01-15",
      "signal": "BULLISH",
      "net_prob": 0.75,
      "strength": 0.85
    }
  ],
  "count": 200,
  "timestamp": "2025-02-02T10:00:00"
}
```

---

### Live Forecast
```
GET /api/v1/forecast/{asset}
```
Get current live forecast predictions.

**Example:**
```bash
curl -H "X-API-Key: your_key" \
  "http://localhost:5000/api/v1/forecast/Crude_Oil"
```

**Response:**
```json
{
  "success": true,
  "asset": "Crude_Oil",
  "forecasts": [
    {
      "horizon_days": 1,
      "predicted_price": 78.50,
      "target_date": "2025-02-03"
    }
  ],
  "signal": "BULLISH",
  "confidence": 75,
  "viable_horizons": [1, 3, 5, 8],
  "timestamp": "2025-02-02T10:00:00"
}
```

---

### Historical Confidence Values
```
GET /api/v1/confidence/{asset}
```
Get historical confidence statistics by horizon.

**Example:**
```bash
curl -H "X-API-Key: your_key" \
  "http://localhost:5000/api/v1/confidence/Crude_Oil"
```

**Response:**
```json
{
  "success": true,
  "asset": "Crude_Oil",
  "stats_by_horizon": {
    "1": {
      "bullish_accuracy": 65.5,
      "bearish_accuracy": 58.2
    }
  },
  "overall_stats": {},
  "timestamp": "2025-02-02T10:00:00"
}
```

---

### Equity Curve
```
GET /api/v1/equity/{asset}
```
Get equity curve data from signal-following strategy.

**Example:**
```bash
curl -H "X-API-Key: your_key" \
  "http://localhost:5000/api/v1/equity/Crude_Oil"
```

**Response:**
```json
{
  "success": true,
  "asset": "Crude_Oil",
  "equity_curve": [
    {
      "date": "2024-01-01",
      "equity": 100.0
    },
    {
      "date": "2024-01-15",
      "equity": 105.5,
      "trade_pnl": 5.5
    }
  ],
  "final_equity": 184.6,
  "total_return": 84.6,
  "count": 86,
  "timestamp": "2025-02-02T10:00:00"
}
```

---

### Performance Metrics
```
GET /api/v1/metrics/{asset}
```
Get comprehensive quant details and performance metrics.

**Example:**
```bash
curl -H "X-API-Key: your_key" \
  "http://localhost:5000/api/v1/metrics/Crude_Oil"
```

**Response:**
```json
{
  "success": true,
  "asset": "Crude_Oil",
  "optimized_metrics": {
    "total_return": 84.6,
    "sharpe_ratio": 2.5,
    "win_rate": 57.0,
    "profit_factor": 2.26,
    "max_drawdown": -15.2,
    "total_trades": 86
  },
  "raw_metrics": {
    "total_return": 45.3,
    "sharpe_ratio": 1.8,
    "win_rate": 52.0,
    "profit_factor": 1.65,
    "max_drawdown": -22.1,
    "total_trades": 120
  },
  "configuration": {
    "viable_horizons": [1, 3, 5, 8],
    "threshold": 0.1,
    "avg_accuracy": 58.5,
    "health_score": 85
  },
  "timestamp": "2025-02-02T10:00:00"
}
```

---

## Authentication

All endpoints (except `/api/v1/health`) require API key authentication.

### Method 1: Header (Recommended)
```bash
curl -H "X-API-Key: your_api_key_here" http://localhost:5000/api/v1/assets
```

### Method 2: Query Parameter
```bash
curl "http://localhost:5000/api/v1/assets?api_key=your_api_key_here"
```

---

## API Key Management

### Create API Key
```bash
python manage_api_keys.py create --user-id user1 --assets Crude_Oil Bitcoin SP500
```

To give access to all assets:
```bash
python manage_api_keys.py create --user-id user1 --assets "*"
```

### List All Keys
```bash
python manage_api_keys.py list
```

### Delete API Key
```bash
python manage_api_keys.py delete --key qdt_abc123...
```

### Add Assets to Existing Key
```bash
python manage_api_keys.py add-assets --key qdt_abc123... --assets GOLD SP500
```

---

## Error Responses

### 401 Unauthorized
```json
{
  "success": false,
  "error": "API key required. Provide via X-API-Key header or ?api_key= parameter"
}
```

### 403 Forbidden
```json
{
  "success": false,
  "error": "Access denied for asset: GOLD"
}
```

### 404 Not Found
```json
{
  "success": false,
  "error": "Asset not found: Invalid_Asset"
}
```

---

## Production Deployment

### Using Gunicorn (Recommended)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

### Using systemd (Linux)

Create `/etc/systemd/system/qdt-api.service`:

```ini
[Unit]
Description=QDTNexus API Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/q_ensemble_sandbox
ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable qdt-api
sudo systemctl start qdt-api
```

---

## Rate Limiting

Each API key has a rate limit (default: 1000 requests/day). This can be configured when creating keys:

```bash
python manage_api_keys.py create --user-id user1 --assets "*" --rate-limit 5000
```

---

## Notes

- All dates are in `YYYY-MM-DD` format
- All timestamps are in ISO 8601 format
- Asset names are case-sensitive (use exact names from `/api/v1/assets`)
- The API reads data from the `data/` directory generated by the pipeline
- Make sure to run `run_complete_pipeline.py` regularly to keep data fresh

