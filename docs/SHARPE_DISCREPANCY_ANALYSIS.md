# Sharpe Ratio Discrepancy Analysis: D+5/D+7/D+10 Pairwise Ensemble

**Date:** February 6, 2026
**Author:** Artemis
**Purpose:** Document and explain the Sharpe ratio discrepancy between Amira's initial pairwise slopes result (1.757) and subsequent validation (-0.82)

---

## Executive Summary

A significant discrepancy was identified between two calculations of the same D+5/D+7/D+10 pairwise slopes ensemble configuration:

| Source | Sharpe | Win Rate | Return |
|--------|--------|----------|--------|
| **Amira's initial result** | **+1.757** | 67.74% | Positive |
| **Deep dive validation** | **-0.82** | 42.6% | -26.2% |

This document explains the root causes and provides recommendations for standardizing calculations.

---

## The Discrepancy

### Source References

1. **Amira's 1.757 Sharpe**
   - File: `benchmark_advanced_ensembles.py:7`
   - Context: Used as baseline for advanced ensemble benchmarking
   - Quote: *"Baseline: Simple Mean DA ~35%, Pairwise Slopes Sharpe = 1.757"*

2. **Validation -0.82 Sharpe**
   - File: `AGENT_COLLABORATION_REPORT.md:333-338`
   - Context: Claude Code deep dive on February 4, 2026
   - Quote: *"The [5, 7, 10] combination was not just suboptimal — it was ANTI-PREDICTIVE."*

---

## Root Causes

### 1. Different Sharpe Calculation Methods

Three distinct implementations exist in the codebase:

#### Method A: Dollar-Based Returns (Amira's Approach)
**File:** `benchmark_advanced_ensembles.py:39-61`

```python
def compute_sharpe(predictions, actuals, annualize=True):
    pred_dir = np.sign(np.diff(predictions))
    actual_changes = np.diff(actuals)  # DOLLAR moves, not percentage
    returns = pred_dir * actual_changes

    mean_ret = returns.mean()
    std_ret = returns.std()  # Population std (no ddof)

    sharpe = mean_ret / std_ret
    if annualize:
        sharpe *= np.sqrt(252)  # Full 252-day annualization

    return sharpe
```

**Characteristics:**
- Uses **dollar price changes** (e.g., $0.50 move)
- Uses **population standard deviation** (no ddof correction)
- Annualizes by **sqrt(252)** regardless of trade frequency
- Does NOT account for holding period

#### Method B: Per-Trade Percentage Returns
**Files:** `comprehensive_diverse_experiments.py:447-450`, `find_diverse_combo.py:224-234`

```python
# Sharpe ratio
avg_hold_days = np.mean(holding_days) if holding_days else 1
trades_per_year = 252 / avg_hold_days if avg_hold_days > 0 else 50
sharpe = (avg_return_pct / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0
```

**Characteristics:**
- Uses **percentage returns** per trade
- Uses **sample standard deviation** (ddof=1)
- Annualizes by **sqrt(trades_per_year)** based on holding period
- Accounts for actual trading frequency

#### Method C: Daily Strategy Returns
**File:** `weighted_pairwise_slopes.py:263-264`

```python
strategy_returns = (position.shift(1) * returns).dropna()
sharpe = (strategy_returns.mean() / strategy_returns.std(ddof=1)) * np.sqrt(252)
```

**Characteristics:**
- Uses **daily percentage returns**
- Uses **sample standard deviation** (ddof=1)
- Annualizes by **sqrt(252)**
- Applies signal from previous day to next day's return

### Key Differences Summary

| Aspect | Method A (Amira) | Method B/C (Validation) |
|--------|------------------|-------------------------|
| Return type | Dollar moves | Percentage returns |
| Std deviation | Population (n) | Sample (n-1) |
| Annualization | sqrt(252) always | sqrt(trades_per_year) |
| Holding period | Not considered | Explicitly modeled |

**Impact:** Using dollar moves instead of percentage returns can dramatically inflate Sharpe ratios for higher-priced assets like crude oil (~$73/barrel).

---

### 2. Different Date Ranges / Train-Test Splits

#### Amira's Calculation
- **Dataset:** Full historical data (`data/1866_Crude_Oil/horizons_wide/horizon_5.joblib`)
- **Split:** None - uses entire dataset for both training and evaluation
- **Potential issue:** In-sample overfitting

#### Validation Calculation
From `find_diverse_combo.py:327-336`:
```python
split_date = forecast_df.index[-OOS_DAYS]  # OOS_DAYS = 30
train_df = forecast_df.loc[:split_date].tail(LOOKBACK_WINDOW)  # LOOKBACK_WINDOW = 60
test_df = forecast_df.loc[split_date:]
```

- **Training:** 60 days lookback
- **Testing:** 30 days out-of-sample
- **Total data:** 369 days available

**Impact:** In-sample results (1.757) vs out-of-sample results (-0.82) can differ dramatically due to overfitting.

---

### 3. Transaction Costs

From `AGENT_COLLABORATION_REPORT.md:613-626`:

| Cost Type | Typical Impact | Status |
|-----------|----------------|--------|
| Bid-ask spread | 0.02-0.10% per trade | **NOT modeled** |
| Commission fees | $0.50-$5 per contract | **NOT modeled** |
| Slippage | 0.01-0.05% per trade | **NOT modeled** |
| Funding costs | Variable | **NOT modeled** |

Quote: *"With 17-26 trades over the backtest period, transaction costs could reduce Sharpe ratios by 0.5-1.5 points depending on execution quality."*

**Impact:** Neither calculation includes transaction costs, so this is not a source of the discrepancy but is important context.

---

### 4. Position Sizing Approach

All implementations use identical **binary position sizing**:
- `+1` for BULLISH signal
- `-1` for BEARISH signal
- `0` for NEUTRAL signal

No leverage, volatility targeting, or Kelly criterion is applied.

**Impact:** Position sizing is NOT a source of the discrepancy.

---

## Why D+5/D+7/D+10 Specifically Fails

### Grid Search Evidence

From `data/1866_Crude_Oil/grid_search_top50.json`:
- **[5,7,10] is NOT in the top 50 configurations**
- Best performing 3-horizon combos include short-term horizons

### Top Performers vs [5,7,10]

| Horizons | OOS Sharpe | Win Rate | Notes |
|----------|------------|----------|-------|
| [5,6,7,8,10] | 5.55 | 80.0% | Top performer |
| [1,2,3] | 9.33 | 63.6% | Short-term only |
| [5,7,10] | **-0.82** | 42.6% | Anti-predictive |

### Theoretical Explanation

From `AGENT_COLLABORATION_REPORT.md:354-362`:

> *"Short horizon combos like [1,2,3] have great Sharpe but are USELESS for hedging. Why? A 1-3 day crude oil move is only ~$0.50. Hedging desks need to know where price is going over 5-10 days."*
>
> *"We need DIVERSE HORIZONS: short (D+1-3) for direction confidence + long (D+7-10) for profit target."*

The [5,7,10] combination **lacks short-term momentum confirmation** from D+1, D+2, D+3 horizons. Medium-term horizons alone produce conflicting signals during regime transitions.

---

## Recommended Sharpe Calculation Standard

Based on industry best practices and codebase analysis:

```python
def calculate_sharpe_ratio(trade_returns_pct, holding_days):
    """
    Standard Sharpe ratio calculation for trading strategies.

    Args:
        trade_returns_pct: List of percentage returns per trade
        holding_days: List of holding periods in days

    Returns:
        Annualized Sharpe ratio
    """
    if len(trade_returns_pct) < 2:
        return 0.0

    avg_return = np.mean(trade_returns_pct)
    std_return = np.std(trade_returns_pct, ddof=1)  # Sample std

    if std_return == 0:
        return 0.0

    # Annualize based on actual trading frequency
    avg_hold_days = np.mean(holding_days)
    trades_per_year = 252 / max(avg_hold_days, 1)

    sharpe = (avg_return / std_return) * np.sqrt(trades_per_year)

    return sharpe
```

**Key principles:**
1. Use **percentage returns**, not dollar moves
2. Use **sample standard deviation** (ddof=1)
3. Annualize by **actual trading frequency**, not fixed 252
4. Always use **out-of-sample** data for final metrics

---

## Files Containing Sharpe Calculations

| File | Lines | Method | Notes |
|------|-------|--------|-------|
| `ensemble_methods.py` | 272-280 | Per-trade % | Used in method evaluation |
| `weighted_pairwise_slopes.py` | 263-264 | Daily % | Rolling evaluation |
| `benchmark_advanced_ensembles.py` | 39-61 | Dollar-based | **Problematic** |
| `find_diverse_combo.py` | 224-234 | Per-trade % | Grid search |
| `comprehensive_diverse_experiments.py` | 447-450 | Per-trade % | Comprehensive testing |
| `ensemble_experiments.py` | 366-375 | Per-trade % | Experiment runner |

---

## Conclusions

### The 1.757 Figure is Incorrect

The Sharpe ratio of 1.757 for [5,7,10] was calculated using:
- Dollar-based returns (not percentage)
- Population standard deviation (not sample)
- Full 252-day annualization (not trade-frequency adjusted)
- In-sample data (not out-of-sample)

### The -0.82 Figure is More Accurate

The validation using proper methodology shows [5,7,10] is **anti-predictive** with:
- Negative returns (-26.2%)
- Sub-50% win rate (42.6%)
- Negative Sharpe (-0.82)

### Winning Configurations Avoid [5,7,10]

The documented winning strategies in `METHODOLOGY_COMPARISON.md`:

| Strategy | Horizons | Sharpe | Win Rate |
|----------|----------|--------|----------|
| Triple filter | [1,2,3,7,10] | **11.96** | 76.9% |
| Vol70_consec3 | [1,3,5,8,10] | **7.90** | 82.4% |

Both include **short-term horizons** (1-3) for momentum confirmation.

---

## Action Items

1. **Deprecate** the dollar-based Sharpe calculation in `benchmark_advanced_ensembles.py`
2. **Standardize** on percentage-based, sample-std Sharpe calculation across codebase
3. **Update** any documentation referencing the 1.757 figure
4. **Add unit tests** for Sharpe calculation consistency
5. **Consider** adding transaction cost modeling for more realistic metrics

---

*Analysis completed: February 6, 2026*
*Based on codebase review of 20+ files containing Sharpe calculations*
