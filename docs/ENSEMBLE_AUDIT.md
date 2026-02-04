# QDT Nexus Ensemble — Complete Logic Audit

**Date:** 2026-02-02  
**Auditor:** Automated deep-code review  
**Scope:** All ensemble logic, signal generation, statistical calculations, grid search, and dashboard metrics  
**Target:** CME Group partnership readiness  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Signal Generation (Pairwise Slopes)](#3-core-signal-generation-pairwise-slopes)
4. [Dynamic Quantile Ensemble (run_dynamic_quantile.py)](#4-dynamic-quantile-ensemble)
5. [Grid Search / Horizon Optimization (find_diverse_combo.py)](#5-grid-search--horizon-optimization)
6. [Statistical Metrics Audit](#6-statistical-metrics-audit)
7. [Accuracy Tracking](#7-accuracy-tracking)
8. [Price Targets & Exit Strategies](#8-price-targets--exit-strategies)
9. [Confidence Calculation](#9-confidence-calculation)
10. [Edge Cases & Robustness](#10-edge-cases--robustness)
11. [Critical Bugs & Errors](#11-critical-bugs--errors)
12. [Statistical Improvements](#12-statistical-improvements)
13. [Missing Techniques Worth Researching](#13-missing-techniques-worth-researching)
14. [Code Quality Issues](#14-code-quality-issues)
15. [Prioritized Action Plan](#15-prioritized-action-plan)

---

## 1. Executive Summary

The QDT Nexus platform implements a **Meta-Dynamic Quantile Ensemble** — a multi-model forecasting system that combines predictions across multiple time horizons (D+1 through D+200) using pairwise slope consensus to generate trading signals. The architecture is sound in concept but has several issues ranging from **critical statistical flaws** to **suboptimal methodological choices** that should be addressed before a CME partnership.

### Severity Summary

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 Critical | 6 | Data leakage, formula errors, incorrect annualization |
| 🟡 Moderate | 9 | Suboptimal methods, fragile assumptions, missing validation |
| 🟢 Minor | 8 | Code quality, magic numbers, naming inconsistencies |

---

## 2. Architecture Overview

### Data Flow
```
API (QuantumCloud) → fetch_all_children.py → training_data.parquet
    → golden_engine.py (prepare horizon data)
    → run_dynamic_quantile.py (walk-forward ensemble per horizon)
    → matrix_drift_analysis.py (pairwise slope signals)
    → find_diverse_combo.py (horizon selection optimization)
    → precalculate_metrics.py (compute trading metrics)
    → build_qdt_dashboard.py (HTML dashboard + JS analytics)
```

### Signal Generation Method
The core signal is the **"pairwise slopes" / "matrix drift"** method:
- For each pair of horizons (D+i, D+j where j > i), compute `drift = forecast[j] - forecast[i]`
- Count bullish (drift > 0) vs bearish (drift < 0) slopes
- `net_prob = (bullish_count - bearish_count) / total_pairs`
- Signal: BULLISH if net_prob > threshold, BEARISH if < -threshold, else NEUTRAL

This is implemented identically in 5+ locations (see Code Quality §14).

---

## 3. Core Signal Generation (Pairwise Slopes)

### Files: `build_qdt_dashboard.py:319-357`, `precalculate_metrics.py:79-110`, `find_diverse_combo.py:119-148`, `validate_calculations.py:51-72`, `matrix_drift_analysis.py:46-68`

### 3.1 Mathematical Correctness

**The pairwise slope method is conceptually sound but statistically naïve.**

The formula:
```python
net_prob = (bullish_count - bearish_count) / total_pairs
```

This produces a value in [-1, +1] representing the proportion of horizon pairs agreeing on direction. However:

🟡 **Issue: Equal weighting of all pairs is suboptimal.**
- A pair (D+1, D+2) has very different information content than (D+1, D+100).
- Short-horizon pairs will often agree (autocorrelation) while long-horizon pairs carry independent information.
- The method treats `(D+1, D+2)` drift the same as `(D+1, D+10)` drift — but the former is nearly noise while the latter reflects genuine trend.

🟡 **Issue: Only direction is used, magnitude is discarded.**
```python
slopes.append(drift)  # magnitude calculated but...
bullish = sum(1 for s in slopes if s > 0)  # ...only sign is used
```
A slope of +$0.01 counts the same as +$1000. The magnitude carries valuable information about conviction.

🟢 **Correct: The net_prob normalization is mathematically valid.** The range [-1, +1] is correct for this voting scheme.

### 3.2 Recommendation: Weighted Slope Aggregation

Replace simple vote counting with magnitude-weighted consensus:

```python
def calculate_signals_improved(forecast_df, horizons, threshold):
    for date in forecast_df.index:
        row = forecast_df.loc[date]
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i_idx, h1 in enumerate(horizons):
            for h2 in horizons[i_idx + 1:]:
                if pd.notna(row[h1]) and pd.notna(row[h2]):
                    drift = row[h2] - row[h1]
                    # Weight by horizon separation (longer spans = more informative)
                    weight = (h2 - h1) / max(horizons)
                    weighted_sum += np.sign(drift) * weight
                    total_weight += weight
        
        net_prob = weighted_sum / total_weight if total_weight > 0 else 0.0
```

---

## 4. Dynamic Quantile Ensemble

### File: `run_dynamic_quantile.py`

### 4.1 Walk-Forward Methodology

The walk-forward loop (line 173-217) is **correctly implemented** with proper temporal separation:

```python
window_start = common_dates[i - LOOKBACK_WINDOW]
window_end = common_dates[i-1]  # Previous day — no look-ahead
X_tr = X_train_all.loc[window_start:window_end]
```

✅ **No look-ahead bias in the walk-forward loop itself.**

### 4.2 🔴 CRITICAL: Ensemble Selection Data Leakage

**Lines 185-207 — The ensemble selection uses the SAME data for both training and validation.**

```python
for strat_name, weights in SCORING_STRATEGIES.items():
    metrics = calculate_metrics(X_tr, y_tr)      # Score models on training data
    ranked = score_models(metrics, weights)
    
    for agg in AGGREGATORS:
        for q in QUANTILE_CANDIDATES:
            top_n = max(1, int(len(ranked) * q))
            top_cols = ranked.head(top_n).index
            ens_hist = X_tr[top_cols].mean(axis=1)  # Ensemble ON training data
            score = evaluate_ensemble_quality(ens_hist, y_tr)  # EVALUATE on training data!
```

The `evaluate_ensemble_quality` function scores the ensemble **on the same data used to select models**. This is in-sample optimization:
- Models are ranked on window data
- The best quantile/aggregator is chosen by testing on the SAME window data
- This leads to **overfitting to the training window**

**Fix:** Split the lookback window into a train-validate split:
```python
split_point = int(len(X_tr) * 0.7)
X_fit, X_val = X_tr.iloc[:split_point], X_tr.iloc[split_point:]
y_fit, y_val = y_tr.iloc[:split_point], y_tr.iloc[split_point:]

# Score models on X_fit, y_fit
# Evaluate ensemble on X_val, y_val
```

### 4.3 🟡 Scoring Function Has Unstable Weighting

`evaluate_ensemble_quality` (line 74):
```python
score = (pnl * 10) + (hits * 10) + (dd * 5) - (mse * 1)
```

These weights are **magic numbers** with no theoretical basis. The P&L and accuracy terms can have wildly different magnitudes depending on the asset, making the scoring inconsistent across assets. The MSE penalty (weight=1) is negligible compared to PnL (weight=10).

**Fix:** Normalize each component to [0, 1] before weighting, or use rank-based scoring (which is already done in `score_models` — apply the same approach here).

### 4.4 Quantile Selection Range

```python
QUANTILE_CANDIDATES = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
```

This only considers the top 1-30% of models. If all models are mediocre, the "best" ensemble may still be poor. Consider adding higher quantiles (0.5, 0.75, 1.0) to allow the full ensemble when individual model quality is uniformly distributed.

---

## 5. Grid Search / Horizon Optimization

### File: `find_diverse_combo.py`

### 5.1 Train/Test Split

```python
LOOKBACK_WINDOW = 60
OOS_DAYS = 30

split_date = forecast_df.index[-OOS_DAYS]
train_df = forecast_df.loc[:split_date].tail(LOOKBACK_WINDOW)
test_df = forecast_df.loc[split_date:]
```

✅ **Proper temporal split — no data leakage in the horizon selection grid search.**

However:

🔴 **CRITICAL: The OOS window is only 30 days.** This is statistically insufficient for reliable evaluation. With typical trading frequencies (5-20 trades per month), the OOS sample often has < 10 trades. The `calculate_trading_performance` function explicitly requires `len(trades) >= 3`, but 3 trades is not statistically meaningful.

**Recommendation:** Use walk-forward validation with multiple OOS windows (e.g., 5 consecutive 30-day windows) and average the results. This provides more robust estimates:

```python
# Walk-forward cross-validation
n_folds = 5
fold_size = 30
results_per_combo = defaultdict(list)

for fold in range(n_folds):
    test_end = len(forecast_df) - fold * fold_size
    test_start = test_end - fold_size
    train_end = test_start
    train_start = max(0, train_end - LOOKBACK_WINDOW)
    # ... evaluate each combo on this fold
```

### 5.2 🔴 CRITICAL: Sharpe Ratio in Grid Search is Wrong

```python
# find_diverse_combo.py, line ~180
if trades_df['pnl_pct'].std() > 0:
    sharpe = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252 / max(1, len(trades_df)))
```

**The annualization factor `√(252 / N_trades)` is incorrect.** The standard annualized Sharpe ratio formula is:

```
Sharpe = (mean_return / std_return) × √(trades_per_year)
```

Where `trades_per_year` is estimated from **average holding period**, not from the number of trades in the sample. Using `252 / len(trades_df)` conflates sample size with trading frequency. If you have 5 trades in a 30-day OOS window, this formula computes `√(252/5) ≈ 7.1` — grossly inflating the Sharpe.

**This same error appears in `precalculate_metrics.py` line 139-142:**
```python
trades_per_year = 252 / avg_hold_days if avg_hold_days > 0 else 50
sharpe = (avg_return / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0
```

This version is **closer to correct** (uses avg holding days), but falls back to an arbitrary `50` trades per year. The dashboard JS version (line 7684) also uses `avg_hold_days`.

**Fix for find_diverse_combo.py:**
```python
avg_hold_days = trades_df.apply(
    lambda row: (row['exit_date'] - row['entry_date']).days, axis=1
).mean()
trades_per_year = 252 / max(1, avg_hold_days)
sharpe = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(trades_per_year)
```

### 5.3 Max Drawdown Calculation in Grid Search

```python
cumulative = trades_df['pnl_pct'].cumsum()
running_max = cumulative.cummax()
drawdown = cumulative - running_max
max_dd = drawdown.min()
```

🟡 **Issue: This uses additive (arithmetic) cumulative returns, not compounding.** For an equity curve, returns compound:

```python
equity = (1 + trades_df['pnl_pct'] / 100).cumprod()
running_max = equity.cummax()
drawdown = (equity - running_max) / running_max
max_dd = drawdown.min()
```

The difference is small for small returns but compounds significantly over many trades. The `precalculate_metrics.py` version (line 148-157) correctly uses multiplicative compounding.

### 5.4 Composite Score

```python
composite = (oos_return_score * 0.35 + oos_sharpe_score * 0.30 + 
            oos_winrate_score * 0.20 + oos_dd_score * 0.15)
```

🟢 This is reasonable — return-weighted with Sharpe and risk controls. The component normalization (clipping to [0, 1]) is appropriate.

---

## 6. Statistical Metrics Audit

### 6.1 Sharpe Ratio

**Locations:** `precalculate_metrics.py:139`, `build_qdt_dashboard.py:7684`, `find_diverse_combo.py:~180`

**Formula (precalculate_metrics.py, most correct version):**
```python
avg_return = np.mean(trade_returns)  # per-trade returns as decimals
std_return = np.std(trade_returns)   # sample std
trades_per_year = 252 / avg_hold_days
sharpe = (avg_return / std_return) * np.sqrt(trades_per_year)
```

🔴 **Issue: `np.std()` uses population standard deviation (ddof=0).** For Sharpe ratio, the sample standard deviation (ddof=1) should be used:

```python
std_return = np.std(trade_returns, ddof=1)
```

This particularly matters with small sample sizes (< 30 trades), where the bias is significant.

**Dashboard JS version (line 7684) correctly uses ddof=1:**
```javascript
Math.sqrt(tradeReturns.reduce((sum, r) => sum + Math.pow(r - avgTradeReturn, 2), 0) / (tradeReturns.length - 1))
```

🟡 **Issue: No risk-free rate subtraction.** The standard Sharpe formula subtracts the risk-free rate:
```
Sharpe = (R_p - R_f) / σ_p × √(annualization_factor)
```

For a CME-grade implementation, the risk-free rate (e.g., 3-month T-bill) should be incorporated, even if it's assumed to be ~5% annually for the current rate environment.

### 6.2 Sortino Ratio

**Location:** `build_qdt_dashboard.py:7717-7719`

```javascript
const negativeTrades = tradeReturns.filter(r => r < 0);
const downsideDeviation = negativeTrades.length > 0 
    ? Math.sqrt(negativeTrades.reduce((sum, r) => sum + Math.pow(r, 2), 0) / negativeTrades.length)
    : 0.0001;
const sortinoRatio = (avgTradeReturn / downsideDeviation) * Math.sqrt(tradesPerYear);
```

🟡 **Issue: The downside deviation should use ALL returns, squaring only the negative ones** (or returns below a target, typically 0). The current implementation filters to only negative returns first, then divides by the count of negative returns. The correct formula:

```javascript
// Correct Sortino downside deviation
const downsideSquared = tradeReturns.map(r => Math.pow(Math.min(r, 0), 2));
const downsideDeviation = Math.sqrt(downsideSquared.reduce((a, b) => a + b, 0) / tradeReturns.length);
```

The denominator should be **total N**, not just negative N. Using only negative N artificially reduces the downside deviation, inflating the Sortino ratio.

**Same issue appears in `calculateSignalFollowingMetrics` (line 5462-5465).**

### 6.3 VaR (Value at Risk)

**Location:** `build_qdt_dashboard.py:7731-7733`

```javascript
const sortedTradeReturns = [...tradeReturns].sort((a, b) => a - b);
const var95Index = Math.floor(sortedTradeReturns.length * 0.05);
const var95 = sortedTradeReturns[Math.max(0, var95Index)] * 100;
```

✅ **Correct: Historical VaR at 95% confidence.** This is the empirical quantile approach, which is appropriate for trade-level VaR. The `Math.max(0, var95Index)` prevents index underflow.

🟡 **Minor issue:** With small sample sizes (< 20 trades), historical VaR is unreliable. Consider a parametric fallback (assuming normal distribution) when N < 30:

```javascript
if (tradeReturns.length < 30) {
    // Parametric VaR (normal approximation)
    var95 = (avgTradeReturn - 1.645 * stdTradeReturn) * 100;
}
```

### 6.4 CVaR / Expected Shortfall

**Location:** `build_qdt_dashboard.py:7736-7737`

```javascript
const tailReturns = sortedTradeReturns.slice(0, Math.max(1, var95Index + 1));
const cvar = tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length * 100;
```

✅ **Correct: Average of returns in the left tail beyond VaR.** The `Math.max(1, ...)` prevents empty slice.

### 6.5 Kelly Criterion

**Location:** `build_qdt_dashboard.py:7764`

```javascript
const kellyPercent = avgWin > 0 && avgLoss > 0 
    ? ((winProb * avgWin) - (lossProb * avgLoss)) / avgWin * 100 
    : 0;
```

🔴 **CRITICAL: This is NOT the Kelly criterion formula.** The standard Kelly formula is:

```
Kelly% = W/A - (1-W)/B
```
Where W = win probability, A = average loss, B = average win. Or equivalently:
```
Kelly% = (W × B - (1-W) × A) / (A × B)
```

Or the simplified version for binary outcomes:
```
Kelly% = W - (1-W)/R  where R = avgWin/avgLoss
```

The current code computes:
```
((W × avgWin) - ((1-W) × avgLoss)) / avgWin
= W - (1-W) × avgLoss / avgWin
= W - (1-W) / R
```

**Wait — this is actually algebraically equivalent to the simplified Kelly formula!** Let me verify:
```
Kelly = W - (1-W)/R = W - (1-W) × (avgLoss/avgWin)
Code  = (W × avgWin - (1-W) × avgLoss) / avgWin = W - (1-W) × avgLoss/avgWin
```

✅ **Actually correct algebraically.** The formula is valid. However, full Kelly is aggressive — **half-Kelly or quarter-Kelly** should be recommended in the UI for risk management.

### 6.6 Profit Factor

**Locations:** `precalculate_metrics.py:132`, `build_qdt_dashboard.py:7672`, `find_diverse_combo.py:~175`

```python
profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
```

🟡 **Issue:** When `gross_loss == 0` and `gross_profit > 0`, the code returns `gross_profit` (a dollar amount) instead of infinity or a capped value. This produces inconsistent units. The dashboard JS correctly caps at 999:
```javascript
const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 999 : 0;
```

**Fix the Python version to match:**
```python
profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)
```

### 6.7 Skewness & Kurtosis

**Location:** `build_qdt_dashboard.py:7741-7750`

```javascript
const skewness = tradeReturns.length > 2 && stdTradeReturn > 0
    ? (tradeReturns.reduce((sum, r) => sum + Math.pow((r - avgTradeReturn) / stdTradeReturn, 3), 0) / tradeReturns.length)
    : 0;

const kurtosis = tradeReturns.length > 3 && stdTradeReturn > 0
    ? (tradeReturns.reduce((sum, r) => sum + Math.pow((r - avgTradeReturn) / stdTradeReturn, 4), 0) / tradeReturns.length) - 3
    : 0;
```

✅ **Correct: Sample skewness and excess kurtosis** (subtracting 3 for excess kurtosis). The guard conditions (n > 2 for skewness, n > 3 for kurtosis) are appropriate.

🟢 **Minor:** For small samples, these should use the bias-corrected formulas with `n(n-1)/(n-2)` and `(n-1)(n-2)(n-3)` adjustment factors, but this is cosmetic for the dashboard.

### 6.8 Ulcer Index

**Location:** `build_qdt_dashboard.py:7768-7776`

```javascript
let sumSquaredDD = 0;
peak = startingCapital;
for (const equity of ensembleEquity) {
    if (equity > peak) peak = equity;
    const dd = (peak - equity) / peak * 100;
    sumSquaredDD += dd * dd;
}
const ulcerIndex = Math.sqrt(sumSquaredDD / ensembleEquity.length);
```

✅ **Correct implementation of the Peter Martin Ulcer Index.**

### 6.9 Calmar Ratio

```javascript
const calmarRatio = maxDrawdown > 0 ? annualizedReturn / maxDrawdown : 0;
```

✅ **Correct formula.** Note: traditionally uses trailing 36-month data, but trade-level is acceptable for this context.

### 6.10 Omega Ratio

```javascript
const tradeGains = tradeReturns.filter(r => r > 0).reduce((a, b) => a + b, 0);
const tradeLosses = Math.abs(tradeReturns.filter(r => r < 0).reduce((a, b) => a + b, 0));
const omegaRatio = tradeLosses > 0 ? tradeGains / tradeLosses : tradeGains > 0 ? 999 : 0;
```

🟡 **Issue: This is actually the Gain-to-Pain ratio, not the Omega ratio.** The true Omega ratio is defined as:

```
Ω(threshold) = ∫[threshold,∞] (1 - F(r)) dr / ∫[-∞,threshold] F(r) dr
```

Where F(r) is the CDF. For threshold = 0, this simplifies to:
```
Ω(0) = (Sum of gains + N × threshold) / |Sum of losses - N × threshold| = 1 + E[R] / |E[R⁻]|
```

**The current implementation is missing the +1 offset.** The correct discrete approximation:
```javascript
const omegaRatio = tradeLosses > 0 ? 1 + (tradeGains - tradeLosses) / tradeLosses : ...;
// Or equivalently: tradeGains / tradeLosses (which is what you have)
```

Actually, for threshold = 0, Omega(0) = E[max(R,0)] / E[max(-R,0)], which IS `sum_gains / |sum_losses|`. So **the formula is correct, but the variable naming suggests Gain-to-Pain**. This is also separately computed as `gainToPain` (line 7757), making the two metrics identical. Remove one.

---

## 7. Accuracy Tracking

### 7.1 Dynamic Accuracy (T1 Hit Rate)

**Location:** `build_qdt_dashboard.py:5814-5867`

```javascript
function calculateDynamicAccuracy(data, signals) {
    const evalWindow = 15;
    // ...
    // Check if T1 target was hit (0.5% move in signal direction)
    const t1Target = signal === 'BULLISH' 
        ? entryPrice * 1.005 
        : entryPrice * 0.995;
```

🔴 **CRITICAL: The 0.5% T1 target is hardcoded and asset-agnostic.** Bitcoin moves 3-5% daily; crude oil moves 1-2%. A 0.5% move is almost guaranteed for BTC (biasing accuracy upward) but meaningful for USD/INR. This makes accuracy comparisons across assets meaningless.

**Fix:** Scale the T1 target by asset volatility (e.g., 0.5× daily ATR):
```javascript
const atrPercent = getATRPercent(data.prices, 14);
const currentATR = atrPercent[entryIdx] || 1.0;
const t1Target = signal === 'BULLISH' 
    ? entryPrice * (1 + currentATR / 200)  // Half of 1-day ATR
    : entryPrice * (1 - currentATR / 200);
```

### 7.2 Directional Accuracy (calculateAccuracy)

**Location:** `build_qdt_dashboard.py:5021-5049`

```javascript
function calculateAccuracy(data, signals) {
    for (let i = 0; i < signals.length - 1; i++) {
        const signal = signals[i];
        if (signal === 'NEUTRAL') continue;
        const currentPrice = data.prices[i];
        const nextPrice = data.prices[i + 1];
        const actualDirection = nextPrice > currentPrice ? 'UP' : 'DOWN';
        const predictedDirection = signal === 'BULLISH' ? 'UP' : 'DOWN';
```

🟡 **Issue: This measures next-day direction, but signals may have multi-day horizons.** If the ensemble uses D+3 through D+10, comparing against the next-day move is misaligned with the forecast horizon. The signal reflects a multi-day outlook but is tested against a 1-day move.

🟡 **Issue: Ties (nextPrice == currentPrice) are counted as 'DOWN'.** This should be excluded or treated separately.

### 7.3 Signal-Following Win Rate (precalculate_metrics.py)

The `calculate_directional_accuracy` function (line 175-191) has a different issue:

```python
for i in range(1, len(signals)):
    if signals.iloc[i] == 'NEUTRAL':
        continue
    pred_direction = 1 if signals.iloc[i] == 'BULLISH' else -1
    actual_direction = 1 if prices.iloc[i] > prices.iloc[i-1] else -1
```

Same problem: signal at time `i` is compared with the price change from `i-1` to `i`. But the signal at time `i` is a **forecast about the future**, not an explanation of the past move. This should compare signal at `i` with the price change from `i` to `i+1` (or the relevant holding period).

🔴 **This is a temporal alignment bug that inflates or deflates accuracy depending on autocorrelation.**

---

## 8. Price Targets & Exit Strategies

### 8.1 Price Target Derivation

**Location:** `build_qdt_dashboard.py:8441-8475`

Targets are derived from percentiles of forecast prices across enabled horizons:
- T1 (Conservative): 25th percentile
- T2 (Base Case): 50th percentile (median)
- T3 (Extended): 75th percentile

✅ **The percentile approach is statistically sound** for deriving point estimates from a forecast distribution.

🟡 **Issue: Signs are force-flipped to match signal direction** (lines 8475-8482):
```javascript
if (signal === 'BULLISH' && pctT1 < 0) {
    pctT1 = Math.abs(pctT1);
    pctT2 = Math.abs(pctT2);
    pctT3 = Math.abs(pctT3);
}
```

This is **dangerous misleading behavior**. If the forecasts disagree with the signal, the targets should NOT be sign-flipped. This masks model disagreement and presents false information. **If forecasts point down but the signal is bullish, the user should see this contradiction.**

🔴 **Recommendation: Remove sign-flipping. Show actual forecast values with a warning when forecasts contradict the signal.**

### 8.2 ATR Calculation

**Location:** `build_qdt_dashboard.py:4803-4836`

```javascript
function calculateATR(prices, period = 14) {
    // TR ≈ absolute daily change (simplified without high/low)
    // We multiply by 1.5 to approximate what H-L would be vs C-C
    const dailyChange = Math.abs(prices[i] - prices[i-1]);
    const estimatedTR = dailyChange * 1.5;  // Approximate H-L from C-C
```

🟡 **Issue: The 1.5× multiplier for estimating True Range from close-to-close is a rough approximation.** Academic literature suggests the ratio of H-L to |C-C| varies by asset and regime (typically 1.2-2.5). Using a fixed 1.5 can significantly over- or under-estimate ATR.

**Better approach:** If only close prices are available, use Parkinson's estimator:
```javascript
// Parkinson volatility estimator (better than C-C × constant)
const vol = Math.sqrt(1 / (4 * Math.LN2) * Math.pow(Math.log(prices[i] / prices[i-1]), 2));
```

Or better yet, **store OHLC data** (which you already have in `price_history_cache.json`) and compute proper ATR.

### 8.3 Holding Period Optimization

**Location:** `build_qdt_dashboard.py:556-670`

The holding period analysis tests fixed holding periods (1-30 days) and computes accuracy per period. This is sound methodology.

🟡 **Issue: The composite score formula** (line 674):
```python
composite_score = exp_factor * acc_factor * dd_factor * pf_factor * 100
```

Where `exp_factor = max(expectancy, 0)` — this means **any negative-expectancy holding period gets score 0**, preventing proper ranking of "least bad" options. Better to use:
```python
exp_factor = expectancy + 1  # Shift to make all values positive
```

### 8.4 Exit Strategy Expected Values

The expectancy calculation is correct:
```python
expectancy = (win_rate_decimal * avg_win) - ((1 - win_rate_decimal) * avg_loss)
```

✅ **Standard expectancy formula, mathematically sound.**

---

## 9. Confidence Calculation

### Location: `build_qdt_dashboard.py:5871-5986`

The confidence is derived from an expectancy-to-score mapping:

```javascript
if (expectancy <= 0) expectancyScore = Math.max(20, 40 + (expectancy * 10));
else if (expectancy < 1) expectancyScore = 40 + (expectancy * 15);
else if (expectancy < 2) expectancyScore = 55 + ((expectancy - 1) * 15);
else if (expectancy < 3) expectancyScore = 70 + ((expectancy - 2) * 15);
else expectancyScore = Math.min(95, 85 + ((expectancy - 3) * 3));
```

🟡 **Issue: The confidence score is based on in-sample expectancy.** The expectancy is calculated from ALL historical trades, not a recent out-of-sample window. Past expectancy doesn't guarantee future results. This number will be overly confident for assets that have had recent regime changes.

**Recommendation:** Weight recent performance more heavily (exponentially decaying weights on trades), or use a rolling window expectancy.

🟡 **Issue: The adjustment factors (±2-3 points each) are magic numbers.** RSI alignment, EMA alignment, Ichimoku alignment each add ±2 points. These should be calibrated against historical accuracy improvement from each filter.

---

## 10. Edge Cases & Robustness

### 10.1 Division by Zero

🟢 **Generally well-handled.** Most divisions are guarded:
```python
profit_factor = gross_profit / gross_loss if gross_loss > 0 else ...
sharpe = ... if std_return > 0 else 0
```

However:

🟡 **`precalculate_metrics.py:132`:**
```python
profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
```
When `gross_loss = 0` and `gross_profit > 0`, this returns `gross_profit` (could be 500.0) instead of a ratio.

### 10.2 Empty Data

🟡 **`golden_engine.py:36`:** The `y_target = y_current.shift(-h)` shift will produce NaN for the last `h` rows. These NaN targets propagate silently through the pipeline. An explicit `.dropna()` after the shift would be clearer.

🟡 **`build_qdt_dashboard.py:load_forecast_data`:** The `ffill().bfill()` fills missing forecasts with adjacent values, which is appropriate for small gaps but could mask data quality issues for large gaps.

### 10.3 Type Mismatches

🟡 **`precalculate_metrics.py:64`:** The prices Series index is set from `base_df['date']` but trades use `signals.index` (which is the forecast_matrix index). These should be aligned but could diverge if date formats differ.

### 10.4 Race Conditions in File I/O

🟡 **`run_dynamic_quantile.py:195-196`:** Checkpoint writes append to CSV without file locking:
```python
pd.DataFrame([row]).to_csv(out_file, mode='a', header=False, index=False)
```

If the pipeline is interrupted mid-write, the CSV could be corrupted. Use atomic writes (write to temp file, then rename).

---

## 11. Critical Bugs & Errors

| # | Severity | File | Line | Issue |
|---|----------|------|------|-------|
| 1 | 🔴 | `run_dynamic_quantile.py` | 185-207 | **Ensemble selection data leakage**: Models selected and validated on same window data. Leads to overfitting. |
| 2 | 🔴 | `find_diverse_combo.py` | ~180 | **Wrong Sharpe annualization**: Uses `√(252/N_trades)` instead of `√(trades_per_year)`. Inflates Sharpe for small OOS samples. |
| 3 | 🔴 | `precalculate_metrics.py` | 139 | **Population std instead of sample std**: `np.std()` without `ddof=1` biases Sharpe upward for small samples. |
| 4 | 🔴 | `build_qdt_dashboard.py` | 5834-5849 | **Hardcoded 0.5% T1 target**: Asset-agnostic threshold makes accuracy meaningless for volatile assets (BTC) and overly strict for low-vol assets (USD/INR). |
| 5 | 🔴 | `precalculate_metrics.py` | 180-188 | **Temporal misalignment in DA**: Signal at time `i` compared with price change `i-1` to `i` (backward-looking) instead of `i` to `i+1` (forward-looking). |
| 6 | 🔴 | `build_qdt_dashboard.py` | 8475-8482 | **Price target sign-flipping**: When forecasts contradict the signal, percentages are force-flipped to match signal direction, hiding model disagreement. |

---

## 12. Statistical Improvements

### 12.1 Replace Equal-Weight Slopes with Weighted Ensemble

**Current:** All pairwise slopes are voted equally.  
**Proposed:** Weight by:
- **Horizon separation** — pairs spanning more time carry more independent information
- **Recent accuracy** — pairs that were recently correct get higher weight
- **Magnitude** — larger drifts indicate stronger conviction

```python
def calculate_weighted_signal(forecast_df, horizons, threshold, accuracy_cache=None):
    for date in forecast_df.index:
        weighted_bull = 0.0
        weighted_bear = 0.0
        total_weight = 0.0
        
        for i_idx, h1 in enumerate(horizons):
            for h2 in horizons[i_idx + 1:]:
                drift = row[h2] - row[h1]
                
                # Weight components
                separation_weight = np.log1p(h2 - h1)  # Logarithmic separation
                magnitude_weight = abs(drift) / row[h1] if row[h1] > 0 else 0
                accuracy_weight = accuracy_cache.get((h1, h2), 1.0)
                
                weight = separation_weight * (1 + magnitude_weight) * accuracy_weight
                
                if drift > 0:
                    weighted_bull += weight
                elif drift < 0:
                    weighted_bear += weight
                total_weight += weight
        
        net_prob = (weighted_bull - weighted_bear) / total_weight if total_weight > 0 else 0
```

### 12.2 Bayesian Model Averaging

Instead of selecting top-N models via quantile thresholding, use **Bayesian Model Averaging (BMA)** where each model's weight is proportional to its posterior probability:

```python
# Simple BMA approximation using recent performance
model_log_likelihoods = -0.5 * ((X_tr - y_tr.values.reshape(-1, 1)) ** 2).sum(axis=0) / sigma_sq
model_weights = np.exp(model_log_likelihoods - model_log_likelihoods.max())  # numerically stable
model_weights /= model_weights.sum()  # normalize to probability

ensemble_prediction = (X.loc[today] * model_weights).sum()
```

### 12.3 Dynamic Weighting by Recent Performance

Track a rolling accuracy window per horizon pair and adjust weights in real-time:

```python
class AdaptiveEnsemble:
    def __init__(self, decay=0.95):
        self.pair_scores = {}  # (h1, h2) -> exponentially weighted score
        self.decay = decay
    
    def update(self, h1, h2, was_correct):
        key = (h1, h2)
        current = self.pair_scores.get(key, 0.5)
        self.pair_scores[key] = self.decay * current + (1 - self.decay) * float(was_correct)
```

### 12.4 Regime Detection

Add a regime detection layer to adapt thresholds and horizon weights:

```python
from hmmlearn import hmm

# Fit 2-3 state HMM on returns
model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.fit(returns.values.reshape(-1, 1))
current_regime = model.predict(returns.values.reshape(-1, 1))[-1]

# Adapt thresholds per regime
regime_thresholds = {0: 0.1, 1: 0.3, 2: 0.5}  # Low vol, normal, high vol
```

### 12.5 Correlation-Based Horizon Weighting

Currently, all horizons are treated independently. If two horizons are highly correlated, they contribute redundant information. Use **Inverse Correlation Weighting**:

```python
corr_matrix = forecast_df[horizons].corr()
# Weight inversely proportional to average correlation with other horizons
avg_corr = corr_matrix.mean()
weights = 1 / (avg_corr + 0.1)  # Small epsilon to prevent division by zero
weights /= weights.sum()
```

### 12.6 Shrinkage Estimators for Covariance

For the Sharpe ratio and risk calculations, use **Ledoit-Wolf shrinkage** instead of raw sample covariance:

```python
from sklearn.covariance import LedoitWolf
lw = LedoitWolf().fit(returns.values.reshape(-1, 1))
shrunk_var = lw.covariance_[0, 0]
```

---

## 13. Missing Techniques Worth Researching

### 13.1 Stacking / Meta-Learning

Instead of the current model selection (top-N by quantile), implement **stacking**:
1. Generate predictions from each base model
2. Train a meta-learner (e.g., Ridge regression) on base model predictions → actual values
3. The meta-learner learns optimal combination weights

```python
from sklearn.linear_model import RidgeCV
meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
meta_model.fit(X_train_predictions, y_train_actual)
ensemble_prediction = meta_model.predict(X_today_predictions)
```

### 13.2 Conformal Prediction

Add **calibrated prediction intervals** instead of point forecasts:
- Use conformal prediction to generate valid coverage intervals
- Report "80% prediction interval: [$X, $Y]" alongside point estimates
- This gives CME partners rigorous uncertainty quantification

### 13.3 Model Diversity Enforcement

Current diversity enforcement (find_diverse_combo.py) is purely horizon-based. Consider:
- **Prediction correlation diversity**: Select horizons whose predictions are least correlated
- **Error correlation diversity**: Select horizons whose errors are least correlated (better)
- **Information ratio diversity**: Select horizons that each add unique information

### 13.4 Transaction Cost Modeling

No transaction costs are modeled. For CME futures:
- Commission: ~$2-5 per contract per side
- Bid-ask spread: varies by contract
- Slippage: depends on execution speed

```python
TRANSACTION_COST_BPS = 5  # 0.05% round-trip
trade_pnl -= TRANSACTION_COST_BPS / 100 * abs(entry_price)
```

### 13.5 Walk-Forward Anchored vs Sliding Window

The current system uses a **sliding window** (last 60 days). Consider testing:
- **Expanding window**: Use all available history, growing over time
- **Anchored walk-forward**: Fixed start date, expanding end date
- **Adaptive window**: Size based on regime stability

### 13.6 Out-of-Sample Degradation Monitoring

Track the ratio of OOS performance to in-sample performance over time. If this ratio drops below 0.5, trigger an alert that the models may be degrading.

---

## 14. Code Quality Issues

### 14.1 🟡 DRY Violations: Signal Calculation Duplicated 5+ Times

The pairwise slope signal calculation is implemented independently in:
1. `build_qdt_dashboard.py:calculate_signals()` (Python, line 319)
2. `precalculate_metrics.py:calculate_signals_pairwise_slopes()` (Python, line 79)
3. `find_diverse_combo.py:calculate_signals_pairwise_slopes()` (Python, line 119)
4. `validate_calculations.py:calc_signals()` (Python, line 51)
5. `matrix_drift_analysis.py:calculate_matrix_drift()` (Python, line 46)
6. `build_qdt_dashboard.py` (JavaScript in HTML, line ~5100+, multiple JS functions)

Any formula change must be synchronized across all 5+ locations. **Extract to a shared module:**

```python
# signals.py
def calculate_pairwise_signals(forecast_df, horizons, threshold):
    """Single source of truth for signal generation."""
    ...
```

### 14.2 🟡 Inconsistent Trade Logic

The trade simulation (enter on signal, exit on flip/neutral) is also duplicated across:
- `precalculate_metrics.py:calculate_trading_performance()`
- `find_diverse_combo.py:calculate_trading_performance()`
- `validate_calculations.py:calc_trades()`
- `build_qdt_dashboard.py` (JS: at least 3 separate implementations)

These have **subtle differences**:
- `validate_calculations.py` exits on `prev_sig != curr_sig`, while `precalculate_metrics.py` exits when `current_signal != position`
- Some versions re-enter on signal flips, others don't
- The JS versions have different field names and slightly different logic

### 14.3 🟢 Magic Numbers

| Value | Location | Purpose |
|-------|----------|---------|
| `0.5%` | dashboard:5834 | T1 accuracy target |
| `1.5` | dashboard:4813 | ATR approximation multiplier |
| `15` | dashboard:5820 | Eval window for dynamic accuracy |
| `20` | dashboard:5826 | Max trades for accuracy sample |
| `60` | multiple | Lookback window |
| `30` | find_diverse_combo:67 | OOS days |
| `252` | multiple | Trading days per year |
| `0.0001` | multiple | Epsilon for zero-division prevention |
| `999` | multiple | Infinity substitute |
| `50` | multiple | Default trades_per_year fallback |

These should be centralized in a configuration file or constants module.

### 14.4 🟢 Asset Configuration Duplication

The asset configuration (id, name, threshold, etc.) is defined separately in:
- `build_qdt_dashboard.py:ASSETS`
- `precalculate_metrics.py:ASSETS`
- `find_diverse_combo.py:ASSETS`
- `run_complete_pipeline.py:ASSETS`
- `config_sandbox.py` (partially)

Each has slightly different structures and some have assets the others don't.

**Fix:** Single `assets.json` configuration file loaded by all modules.

### 14.5 🟡 No Logging Framework

All diagnostic output uses `print()`. For a production system, use Python's `logging` module with configurable log levels.

### 14.6 🟡 No Unit Tests

There are zero automated tests. For CME-grade code:
- Unit tests for each metric calculation (Sharpe, Sortino, VaR, etc.) against known values
- Integration tests for the signal pipeline
- Regression tests comparing outputs against known-good baselines

---

## 15. Prioritized Action Plan

### Phase 1: Critical Fixes (Before CME Review)

1. **Fix ensemble selection data leakage** (`run_dynamic_quantile.py`) — Split lookback window into train/validate
2. **Fix Sharpe annualization** in `find_diverse_combo.py` — Use trades_per_year based on avg holding days
3. **Fix `np.std` to use ddof=1** in `precalculate_metrics.py`
4. **Fix temporal alignment** in directional accuracy — Compare signal at `i` with price change from `i` to `i+1`
5. **Remove price target sign-flipping** — Show actual forecast values, add warning when contradictory
6. **Scale T1 accuracy target by ATR** — Use asset-specific volatility instead of fixed 0.5%

### Phase 2: Statistical Improvements (Partnership Enhancement)

7. **Implement weighted pairwise slopes** — Weight by horizon separation and magnitude
8. **Extend OOS validation** — Use walk-forward cross-validation with multiple folds
9. **Fix Sortino ratio** — Use total N as denominator, not just negative N
10. **Add transaction cost modeling** — Even a simple flat BPS cost per trade
11. **Fix max drawdown** in `find_diverse_combo.py` — Use multiplicative compounding
12. **Add risk-free rate** to Sharpe calculation

### Phase 3: Architecture Improvements (Production Readiness)

13. **Extract shared signal module** — Single source of truth for pairwise slopes
14. **Centralize asset configuration** — Single `assets.json` file
15. **Add unit test suite** — Especially for all metric calculations
16. **Add logging framework** — Replace print() with structured logging
17. **Implement regime detection** — HMM or simple volatility regimes
18. **Add conformal prediction intervals** — Calibrated uncertainty quantification

### Phase 4: Research Enhancements (Competitive Edge)

19. **Bayesian Model Averaging** — Replace quantile selection with BMA
20. **Stacking meta-learner** — Train Ridge/Lasso on base model predictions
21. **Adaptive ensemble weighting** — Exponentially decay pair weights by recent accuracy
22. **Correlation-based horizon selection** — Select horizons that contribute unique information
23. **Walk-forward degradation monitoring** — Alert when OOS/IS ratio drops

---

## Appendix A: Metric Calculation Summary

| Metric | Formula Used | Correct? | Notes |
|--------|-------------|----------|-------|
| Sharpe Ratio | `(mean/std) × √(252/avg_hold)` | 🟡 Mostly | Missing ddof=1 in Python, missing risk-free rate |
| Sortino Ratio | `(mean/downside_std) × √(annualize)` | 🔴 No | Denominator should use total N, not just negative N |
| VaR (95%) | Historical 5th percentile | ✅ Yes | |
| CVaR | Mean of tail beyond VaR | ✅ Yes | |
| Kelly % | `(W×B - L×A) / B` | ✅ Yes | Should recommend half-Kelly |
| Profit Factor | `gross_profit / gross_loss` | 🟡 Mostly | Inconsistent infinity handling between Python/JS |
| Max Drawdown | Multiplicative (metrics.py) / Additive (find_diverse) | 🟡 Mixed | find_diverse_combo.py uses additive — should be multiplicative |
| Expectancy | `(W × avg_win) - (L × avg_loss)` | ✅ Yes | |
| Skewness | Sample 3rd moment | ✅ Yes | |
| Kurtosis | Excess (4th moment - 3) | ✅ Yes | |
| Ulcer Index | `√(mean(DD²))` | ✅ Yes | |
| Calmar | Annualized return / max DD | ✅ Yes | |
| Omega | `sum_gains / |sum_losses|` | ✅ Yes | Identical to Gain-to-Pain (duplicate metric) |
| ATR | Close-to-close × 1.5 | 🟡 Approx | Should use proper OHLC when available |

---

## Appendix B: Files Reviewed

| File | Lines | Key Components |
|------|-------|----------------|
| `golden_engine.py` | 64 | Horizon data preparation, target shifting |
| `config_sandbox.py` | 52 | Project configuration, path management |
| `run_dynamic_quantile.py` | 267 | Walk-forward ensemble, model selection, live forecasting |
| `matrix_drift_analysis.py` | 96 | Pairwise slope signal generation |
| `find_diverse_combo.py` | 310 | Grid search for optimal horizon combos |
| `precalculate_metrics.py` | 298 | Trading simulation, metric calculation, config updates |
| `validate_calculations.py` | 150 | Cross-validation of signal calculations |
| `build_qdt_dashboard.py` | 9,359 | Dashboard generation, JS analytics, all display metrics |
| `api_server.py` | 314 | REST API for data access |
| `run_complete_pipeline.py` | 173 | Pipeline orchestration |
| `run_sandbox_pipeline.py` | 57 | Per-asset pipeline runner |

**Total lines reviewed:** ~11,140+

---

*This audit was conducted as a comprehensive code review. All line numbers reference the codebase as of 2026-02-02. Recommendations are prioritized by impact on statistical validity and partnership readiness.*
