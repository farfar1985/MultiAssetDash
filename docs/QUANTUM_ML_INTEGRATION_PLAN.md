# Quantum ML ↔ Nexus v2 Integration Feasibility & Plan

Date: 2026-02-05

## Executive Summary
Nexus v2 already integrates with QDT’s quantum_ml backend **via HTTP API** (`/get_qml_models/<project_id>` at `https://quantumcloud.ai`) and then builds its own **ensemble logic** (golden_engine → dynamic quantile ensembles → pairwise-slope signals → dashboard). This is a **good, decoupled architecture**, but the contract between Nexus and quantum_ml is implicit and fragile (field names, horizons, auth headers). 

**Recommendation:** Keep **API integration** as the primary integration path (Option B), formalize the API contract, and add 1–2 focused endpoints in quantum_ml to reduce data transfer and make Nexus’s pipeline more stable. Direct package import (Option A) is high-friction due to Python version mismatch and heavy dependencies (Cassandra/Redis), and shared DB access (Option C) increases coupling and security risk.

---

# 1) Architecture Comparison

## Nexus v2 (MultiAssetDash)
**Purpose:** Trading dashboard with ensemble forecasts, signals, metrics, and HTML output.

**Prediction/Data Flow (current):**
1. **Fetch QML model outputs** from QDT backend:
   - `fetch_all_children.py` calls: `GET {API_BASE_URL}/get_qml_models/<project_id>`
   - Auth header: `qml-api-key: <QML_API_KEY>`
   - Output columns used: `symbol, time, n_predict, target_var_price, close_predict, yn_p, yn_actual`
2. **Prepare horizon data:** `golden_engine.py` pivots by `n_predict` → creates `horizon_{h}.joblib`
3. **Per-horizon ensemble:** `run_dynamic_quantile.py`
   - Ranks child models by historical performance
   - Selects top quantiles (dynamic) and aggregates (mean/median)
   - Saves `forecast_d{h}.csv`
4. **Signal generation & metrics:**
   - `precalculate_metrics.py` uses **pairwise slopes** across horizons to generate signals (matrix drift)
   - `master_ensemble.py` explores horizon combos and thresholds
5. **Dashboard build:** `build_qdt_dashboard.py` generates static HTML
6. **Optional API** (`api_server.py`) exposes signals, forecasts, metrics via `X-API-Key`

**Core signal logic:**
- Pairwise slopes across horizons: if longer horizons > shorter → bullish; if lower → bearish.

**Key prediction components:**
- `golden_engine.py` (horizon slicing)
- `run_dynamic_quantile.py` (ensemble per horizon)
- `precalculate_metrics.py` (pairwise slope signals + metrics)
- `master_ensemble.py` (grid search for best horizon combos)

**Tech stack:**
- Python 3.8+ (README)
- pandas, numpy, joblib, requests, Flask
- Data persisted as parquet + CSV + JSON files in `data/{asset_id}_{asset}`

---

## quantum_ml (QDT Core ML Backend)
**Purpose:** Full ML platform (data ingestion, training, inference, storage, API service, dashboards).

**Architecture:**
- Flask app (`run.py app`) plus worker/controller/ingestion services
- Model pipeline modules: `qml_model.py`, `qen_model.py`, `qrd_model.py`, etc.
- Storage & infra: Cassandra + Redis + local/remote data lake
- API endpoints (in `app/views/views_api.py`), including:
  - `GET /get_qml_models/<project_id>` → raw model outputs (used by Nexus)
  - `GET /get_qml_forecast/<project_id>` → aggregated forecast + close prices
- Authorization via `qml-api-key` header or Basic auth token

**Tech stack:**
- Python **>=3.12** (pyproject)
- Flask + SQLAlchemy + JWT + Dash + Cassandra + Redis
- Extensive ML deps: xgboost, lightgbm, catboost, scikit-learn, etc.

**Key note:** quantum_ml is a *full platform* with heavy infra dependencies, not a light library.

---

# 2) Integration Points

## How Nexus v2 can consume quantum_ml models
### Current path (already working)
- `fetch_all_children.py` → `GET /get_qml_models/<project_id>`
- Nexus expects **row-level model outputs** (child predictions per horizon).

### Alternative within quantum_ml API
- `GET /get_qml_forecast/<project_id>` can deliver **aggregated forecasts** and close prices
- Could be used to bypass part of Nexus’s horizon aggregation (if aligned)

---

## Data Format Compatibility
**Expected by Nexus (from quantum_ml):**
- `symbol`: model ID
- `time`: timestamp (datetime)
- `n_predict`: horizon days (D+1, D+2, …)
- `target_var_price`: actuals
- `close_predict`: model prediction

Nexus pivots this into a matrix of shape: `time × model_id` per horizon.

**Potential mismatches to watch:**
- Missing horizons (handled dynamically by Nexus)
- Missing `target_var_price` or `close_predict` fields
- Timezone normalization (`time` must be consistent)

---

## API Structure Alignment
- quantum_ml endpoint used: `GET /get_qml_models/<project_id>`
- Auth: `qml-api-key` header (`QML_API_KEY` in `.env`)
- Nexus config: `API_BASE_URL` and `QML_API_KEY` in `config_sandbox.py`

---

## Authentication & Credential Handling
- quantum_ml API: header `qml-api-key` (or base64 basic auth token)
- Nexus uses `.env` (loaded via python-dotenv in `config_sandbox.py`)
- Nexus API server has its own `X-API-Key` system for downstream clients

**Integration decision:** keep QML API auth separate from Nexus client auth.

---

# 3) Integration Options

## Option A — Direct Import (quantum_ml as Python package)
**Description:** Import quantum_ml modules directly from Nexus, bypass API.

**Pros:**
- Lowest latency / no HTTP round-trips
- Direct access to model tables, internal utilities
- Potentially richer metadata access

**Cons:**
- **Python version mismatch** (Nexus 3.8+ vs quantum_ml 3.12+)
- **Heavy dependencies** (Cassandra/Redis, SQLAlchemy, Dash, etc.)
- Tight coupling → fragile updates
- Requires quantum_ml environment + configs in Nexus runtime

**Effort:** Large

---

## Option B — API Integration (quantum_ml as a service)
**Description:** Use quantum_ml’s REST API as a service (current pattern).

**Pros:**
- Already working (Nexus uses `/get_qml_models`)
- Loose coupling, clear interface
- Easier to scale / secure / update independently

**Cons:**
- Dependent on API uptime and latency
- Need stable schema contract + versioning

**Effort:** Small → Medium (mostly hardening & minor endpoint extensions)

---

## Option C — Shared Database/Data Lake
**Description:** Nexus reads directly from Cassandra/QDL or shared storage.

**Pros:**
- No API overhead
- Potentially faster bulk access

**Cons:**
- Tight coupling to internal schema
- Security + credential exposure
- Harder to change or migrate storage
- Complex infra requirements on Nexus side

**Effort:** Large

---

# 4) Migration Path

## Nexus v2 Components that would change
- **fetch_all_children.py**: could shift to a new endpoint or direct DB read
- **config_sandbox.py**: update base URL / auth strategy
- **golden_engine.py**: only changes if input schema changes
- **run_dynamic_quantile.py**: unchanged unless QML provides aggregated forecasts
- **precalculate_metrics.py / build_qdt_dashboard.py**: unchanged unless schema changes

## quantum_ml Components that would need exposure
- `views_api.py` endpoints:
  - `/get_qml_models/<project_id>` (existing)
  - `/get_qml_forecast/<project_id>` (existing)
  - **Recommended:** add a **“horizon-aggregated” endpoint** (optional)
- Auth: keep `qml-api-key` but document and version

---

## Effort & Risk by Option
| Option | Effort | Key Risks |
|---|---|---|
| A: Direct Import | **Large** | Python version mismatch, heavy infra deps, tight coupling |
| B: API Service | **Small–Medium** | API contract drift, uptime dependency |
| C: Shared DB | **Large** | Security, schema coupling, ops overhead |

---

# 5) Recommended Approach

## Best Strategy
**Option B (API Integration)** with **contract hardening** and **one new endpoint** for optimized data delivery.

### Why
- Minimal changes to Nexus (already API-based)
- Avoids dependency conflicts
- Keeps quantum_ml’s infra isolated
- Enables scaling & versioning

---

## Phased Implementation Plan

### Phase 0 — Stabilize Current Integration (Quick Win)
- **Document schema** for `/get_qml_models`:
  - Required fields: `symbol`, `time`, `n_predict`, `target_var_price`, `close_predict`
  - Ensure types and time format
- Add **version stamp** in API response headers or metadata
- Add Nexus-side schema validation (fail fast if missing columns)

### Phase 1 — API Contract & Performance Improvements
- Add optional params to reduce payload:
  - `?from_date=YYYY-MM-DD` to limit history
  - `?format=parquet|json` for efficiency
- Add endpoint for **horizon aggregates**:
  - e.g. `/get_qml_horizon_forecast/<project_id>`
  - Returns per-horizon ensemble series instead of raw model table
- Nexus could skip `golden_engine + run_dynamic_quantile` if desired

### Phase 2 — Integration Optimization
- Cache QML responses in Nexus (per asset/day)
- Add retry + backoff in `fetch_all_children.py`
- Add data freshness / watermark checks

### Phase 3 — Optional Deeper Integration (Long-term)
- If needed, migrate to shared storage or direct model service
- Introduce gRPC or streaming for high-frequency updates

---

## Quick Wins vs Long-Term
**Quick Wins (1–2 weeks):**
- Formal API contract documentation
- Add `from_date` / `format` params
- Schema validation in Nexus

**Long-Term (1–3 months):**
- Horizon-aggregated endpoint
- Model metadata endpoint for visibility & monitoring
- Optional API versioning / deprecation policy

---

# Appendix: Observed Endpoints & Auth

## quantum_ml (used by Nexus)
- `GET /get_qml_models/<project_id>`
  - Auth: `qml-api-key` header
  - Returns per-model predictions with horizons

## Nexus API Server (exposes to clients)
- `GET /api/v1/forecast/<asset>`
- `GET /api/v1/signals/<asset>`
- Auth: `X-API-Key`

---

# Final Recommendation
Maintain **API integration as the primary path**. It is already working, aligns with separation of concerns, and avoids heavy dependency conflicts. Formalize the data contract and add a “horizon-aggregated forecast” endpoint in quantum_ml to reduce Nexus’s preprocessing when desired. This yields minimal risk with strong long-term flexibility.
