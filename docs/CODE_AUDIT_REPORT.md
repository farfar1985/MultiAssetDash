# Nexus QML Repo – Code Audit Report
**Date:** 2026-02-04  
**Scope:** `golden_engine.py`, `master_ensemble.py`, `api_server.py`, `build_qdt_dashboard.py`, plus all other `.py` files in `C:\Users\William Dennis\projects\nexus\`.

---

## Summary
- **Python files scanned:** 64
- **Syntax errors (AST parse):** None detected
- **Findings:** 12 total  
  - **Critical:** 0  
  - **High:** 3  
  - **Medium:** 6  
  - **Low:** 3

---

## High Severity Findings

### 1) Misaligned backtest returns (look‑ahead / off‑by‑one)
- **File:** `master_ensemble.py`  
- **Lines:** 135–155  
- **Category:** Logic / ML-specific  
- **Issue:** `y` is already shifted forward (from `golden_engine.py`) to represent future prices. Backtest uses `actual_return = (y[t+1] - y[t]) / y[t]`, which computes return from *t+1 → t+2* while the signal at time *t* is based on forecasts for *t+1*. This introduces misalignment and can silently inflate/deflate performance.
- **Impact:** Backtest metrics (Sharpe, win rate, etc.) are not aligned to the signal timing.
- **Recommendation:** Use unshifted price series for returns, or shift signals to align with the same horizon; drop the last `h` rows to keep alignment clean.

### 2) Data leakage from backward fill in forecasts
- **File:** `build_qdt_dashboard.py`  
- **Lines:** 189–190  
- **Category:** ML-specific  
- **Issue:** `df = df.ffill().bfill()` backfills missing horizons using **future** data, which leaks information backward in time. Any signal/metric computed from this data will be contaminated.
- **Impact:** Inflated historical signal accuracy and misleading dashboard metrics.
- **Recommendation:** Avoid `bfill()` for time series features. Use forward-fill only, or leave missing values and handle them explicitly.

### 3) Path traversal risk via `strategy` query param
- **File:** `api_server.py`  
- **Lines:** 615–623  
- **Category:** Security  
- **Issue:** `strategy` is inserted directly into a filename without sanitization. A crafted value (e.g., `../secrets`) can escape `CONFIGS_DIR` and read unintended files.
- **Impact:** Potential unauthorized file read.
- **Recommendation:** Enforce a whitelist of allowed strategy names or sanitize to `[A-Za-z0-9_\-]` before building the path.

---

## Medium Severity Findings

### 4) Plaintext API key storage
- **File:** `api_server.py`  
- **Lines:** 120–134  
- **Category:** Security  
- **Issue:** API keys are stored in `api_keys.json` as plaintext.
- **Impact:** If the file is leaked, all keys are compromised.
- **Recommendation:** Hash API keys (e.g., SHA‑256) and compare hashes; optionally store in OS‑level secret storage.

### 5) API keys allowed in query string
- **File:** `api_server.py`  
- **Lines:** 136–143  
- **Category:** Security  
- **Issue:** Accepting `?api_key=...` exposes credentials in logs, referrers, and caches.
- **Impact:** Increased chance of key leakage.
- **Recommendation:** Require header-only auth or disable query-param fallback in production.

### 6) “top10” aggregation uses predictions instead of variance at t=0
- **File:** `master_ensemble.py`  
- **Lines:** 71–74  
- **Category:** Logic  
- **Issue:** For `t == 0`, `var = X_t` (predictions) is used where variance should be computed. This selects models with **smallest predictions**, not lowest variance.
- **Impact:** Incorrect model subset for the “top10” aggregation on the first step; inconsistent logic.
- **Recommendation:** Use a variance proxy even at `t=0` (e.g., zeros or NaN-safe fallback) and skip `top10` until enough history exists.

### 7) Unhandled missing horizons can crash grid search
- **File:** `master_ensemble.py`  
- **Line:** 132  
- **Category:** Error / Robustness  
- **Issue:** `min_len = min(horizons[h]['X'].shape[0] for h in horizon_subset)` assumes **all** horizons exist. If any horizon is missing, this raises `KeyError`.
- **Impact:** Grid search fails when horizon files are incomplete.
- **Recommendation:** Filter `horizon_subset` to available horizons before calculating `min_len`.

### 8) NaNs converted to zero in labels
- **File:** `master_ensemble.py`  
- **Lines:** 34–36, 153–156  
- **Category:** ML-specific  
- **Issue:** `np.nan_to_num(y, nan=0.0)` converts missing future labels to 0. Later, returns are computed with `if y[t] != 0`, effectively masking invalid periods.
- **Impact:** Distorts backtest metrics and may hide data quality issues.
- **Recommendation:** Drop tail rows created by shifting, or mask invalid indices before computing returns.

### 9) No out‑of‑sample split during grid search
- **File:** `master_ensemble.py`  
- **Lines:** 211–268  
- **Category:** ML-specific  
- **Issue:** Parameter grid search evaluates and selects the “best” config on the **same** data without a time‑based holdout.
- **Impact:** Overfitting risk; reported Sharpe/return likely overstated.
- **Recommendation:** Use time‑series split (e.g., walk‑forward or train/validation/test) for selection.

---

## Low Severity Findings

### 10) Open trades not closed at end of series
- **File:** `precalculate_metrics.py`  
- **Lines:** 127–173  
- **Category:** Logic  
- **Issue:** `calculate_trading_performance` never closes an open position at the end of the signal series.
- **Impact:** Under‑reports trades and return statistics.
- **Recommendation:** Force close at last price or explicitly document the behavior.

### 11) Signals/prices alignment assumed by position index
- **File:** `precalculate_metrics.py`  
- **Lines:** 127–130  
- **Category:** Logic / Code quality  
- **Issue:** Uses `signals.iloc[i]` with `prices.iloc[i]` without aligning by date. If indices drift, calculations are incorrect.
- **Impact:** Silent metric errors if the inputs are misaligned.
- **Recommendation:** Align with `signals, prices = signals.align(prices, join='inner')`.

### 12) Target selection assumes identical actuals across symbols
- **File:** `golden_engine.py`  
- **Lines:** 54–55  
- **Category:** Logic  
- **Issue:** `targets = sub_df.groupby('time')['target_var_price'].first()` assumes all symbols share identical target values per time. If they differ, the first row is arbitrary.
- **Impact:** Potential label noise.
- **Recommendation:** Validate consistency across symbols or aggregate (mean/median) explicitly.

---

## Cross‑Cutting Security Note (Unsafe Deserialization)
- **Files (examples):** `master_ensemble.py:32`, `build_qdt_dashboard.py:216`, `api_ensemble.py:73`, many others via `joblib.load()`  
- **Issue:** `joblib.load()`/pickle deserialization can execute arbitrary code if files are tampered with.
- **Recommendation:** Treat `.joblib` files as **trusted‑only** inputs, store them in protected directories, and validate file provenance.

---

## Positive Notes
- No hardcoded API keys in source; keys pulled from environment in `config_sandbox.py`.
- `api_server.py` includes robust error handlers and security headers.
- Network calls in `fetch_all_children.py` include timeouts.

---

## Suggested Next Steps
1. Fix the backtest alignment bug in `master_ensemble.py` and add a time‑based validation split.
2. Remove backward-fill in `build_qdt_dashboard.py` or isolate it strictly for visualization (not metrics).
3. Sanitize the `strategy` query param and enforce header‑only API keys.
4. Add unit tests for signal/return alignment and deserialization inputs.
