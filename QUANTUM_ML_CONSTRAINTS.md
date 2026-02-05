# quantum_ml Integration Constraints

**Date:** 2026-02-03  
**Source:** Email thread with Alessandro Savino (Ale)

## Critical Understanding

### Current API Structure

**Endpoint:** `/get_qml_models/<project_id>`

**Returns:** mdl_table with columns:
- symbol
- time
- n_predict
- target_var_price
- close_predict
- yn_p
- yn_actual

### Architecture Decision (MUST RESPECT)

**⚠️ CRITICAL:** Metrics are NOT stored in model output.

**All metrics are computed on-demand from mdl_table using:**
```python
quantum_ml.financial.run_backtest(mdl_table, strategy)
```

This was an intentional decision made ~1 year ago:
- Keeps source of truth lean
- Allows recomputation at any time
- Enables historical "rewind" capability

### Available Functions (DON'T DUPLICATE)

#### 1. run_backtest() - Main Entry Point
**Location:** `quantum_ml/financial.py`

Computes 20+ metrics including:
- Sharpe, Sortino, Information ratios
- Max drawdown
- Long/short profit accuracy  
- ROI (30/60/90/180/360 day)
- Win/loss day percentages

#### 2. batch_compute_accuracy_metrics()
**Location:** `quantum_ml/metrics.py`

Returns DataFrame with:
- `acc_predict` (overall directional accuracy)
- `acc_predict_up` (accuracy when predicting up)
- `acc_predict_down` (accuracy when predicting down)
- `acc_predict_abs` (absolute accuracy)

#### 3. compute_feature_importance()
**Location:** `quantum_ml/qml_model.py`

Computes feature importance per model/algorithm.

#### 4. Model Metadata
**Location:** `quantum_ml/dataset.py`

Available fields:
- `dataset.model_name` (algorithm: XGB, GBR, CAT)
- `dataset.n_predict` (prediction horizon)
- `dataset.n_train` (training window size)
- `dataset.n_forget` (forgetting factor)
- `dataset.strategy` (trading strategy)
- `dataset.code_version` (model version)

---

## The Killer Feature: Historical Rewind

**mdl_table is append-only by design.**

Every day adds one row. This creates a complete historical record.

### How to Rewind

```python
# Get metrics as they were on 2025-06-15
rewind_date = '2025-06-15'
mdl_table_historical = mdl_table[mdl_table['time'] <= rewind_date]
historical_metrics = run_backtest(mdl_table_historical, strategy)
```

### Use Cases for CME

1. **Audit past performance** - "What were the metrics on June 15?"
2. **Accuracy trajectories** - Show how reliability evolved over time
3. **Confidence weighting** - Weight models by rolling historical accuracy
4. **Build trust** - Show clients exactly what the model "knew" at any point

This is HUGE for institutional clients like CME's hedging desks.

---

## Package Status

**⚠️ BLOCKER:** quantum_ml is NOT pip-installable yet.

**Access:** Need direct GitHub repo access from Ale.

**Why not packaged?**
- Codebase still actively evolving
- New algorithms, ensemble methods, features being added
- Proper packaging on roadmap once core stabilizes

**Status:** Amira (Bill's AI) emailed Ale requesting repo access. Waiting on response.

---

## Two Integration Options

### Option 1: New API Endpoint
- Wrap run_backtest() in a new API endpoint
- Returns computed metrics alongside predictions
- **Pro:** Cleaner separation
- **Con:** Latency concerns, requires Ale's time

### Option 2: Nexus-Side Computation ✅ (RECOMMENDED)
- Import quantum_ml and call run_backtest() directly
- Full control over which metrics to compute and when
- **Pro:** No waiting on API changes, immediate start
- **Con:** Dependency on quantum_ml package

**Amira chose Option 2** - waiting on repo access.

---

## Implications for Our Action Plan

### BEFORE (Naive)
- "Fix bugs in metrics calculation"
- "Add historical accuracy display"
- "Improve performance metrics"

### AFTER (Informed)
- **Get quantum_ml repo access** (BLOCKER #1)
- **Replace custom calculations** with official quantum_ml functions
- **Don't duplicate logic** - use their run_backtest(), not our own
- **Implement historical rewind** for CME trajectory demos
- **Fix discrepancies** where Nexus calculates manually vs. using QML

### Likely Root Cause of "6 Backend Bugs"
Amira's audit probably found cases where:
- Nexus calculates Sharpe manually → disagrees with quantum_ml's version
- Directional accuracy computed wrong
- Metrics not updated when mdl_table changes
- Missing metrics that quantum_ml already provides

**Solution:** Use quantum_ml's functions everywhere. Stop reinventing the wheel.

---

## Action Items (UPDATED)

### IMMEDIATE (This Week)
1. ✅ Amira requested quantum_ml repo access (waiting on Ale)
2. ⏳ Once access granted: Clone repo, understand structure
3. ⏳ Map which Nexus functions need replacement with quantum_ml calls
4. ⏳ Document API contract between Nexus and quantum_ml

### PHASE 1 (Week 1-2)
- Integrate quantum_ml.run_backtest() into Nexus
- Replace all manual metric calculations
- Validate outputs match between old vs. new
- Add historical rewind capability

### PHASE 2 (Week 3-4)
- Build accuracy trajectory visualizations for CME
- Implement confidence weighting based on rolling accuracy
- Add feature importance display (if needed)

---

## Questions for Amira (When She Responds)

1. Which specific metrics in Nexus disagree with quantum_ml's calculations?
2. Do we have quantum_ml repo access yet? (I know you emailed Ale)
3. Should we pause bug fixes until we integrate quantum_ml properly?
4. Are any of the "6 bugs" fixable without quantum_ml access?

---

**Bottom Line:**

We can't just "fix bugs" in isolation. We need to integrate quantum_ml properly FIRST, then verify everything calculates correctly using their official functions.

The good news: Ale's team already built everything we need. We just need access.
