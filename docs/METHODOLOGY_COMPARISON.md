# QDT Ensemble Strategy Methodology Comparison

## Overview

This document compares the two primary ensemble strategies developed for Crude Oil trading, explaining why they produce different Sharpe ratios and when to use each.

---

## Strategy Configurations

### 1. Triple Filter Strategy (Alpha Generation)

**Config Name:** `triple_v70c2p0.3_mixed`

| Parameter | Value |
|-----------|-------|
| Horizons | [1, 2, 3, 7, 10] |
| Consecutive Signals | 2 |
| Probability Threshold | 0.30 |
| Volatility Filter | 70th percentile |

**Performance Metrics:**
| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **11.96** |
| Win Rate | 76.9% |
| Total Trades | 26 |
| Total Return | +44.04% |
| Max Drawdown | -0.69% |
| Profit Factor | 65.1 |
| Avg Profit/Trade | $1.08 |
| Avg Hold Days | 2.0 |

---

### 2. Volatility + Consecutive Strategy (Hedging)

**Config Name:** `vol70_consec3_diverse`

| Parameter | Value |
|-----------|-------|
| Horizons | [1, 3, 5, 8, 10] |
| Consecutive Signals | 3 |
| Probability Threshold | 0.25 |
| Volatility Filter | 70th percentile |

**Performance Metrics:**
| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **7.90** |
| Win Rate | 82.4% |
| Total Trades | 17 |
| Total Return | +48.91% |
| Max Drawdown | -1.32% |
| Profit Factor | 21.96 |
| Avg Profit/Trade | $1.81 |
| Avg Hold Days | 3.4 |

---

## Why Different Sharpe Ratios?

### 1. Consecutive Signal Requirement

| Strategy | Consec | Effect |
|----------|--------|--------|
| Triple | 2 | More trades, faster entry, higher variance |
| Vol70_Consec3 | 3 | Fewer trades, stronger confirmation, lower variance |

**Impact on Sharpe:**
- **2 consecutive**: Catches more moves but includes more noise → higher returns, higher volatility
- **3 consecutive**: Misses some moves but filters false signals → steadier returns, lower volatility

The Sharpe ratio is `(Return - RiskFree) / Volatility`. Triple's higher Sharpe comes from its **extremely low volatility** (tiny drawdowns) despite taking more trades.

### 2. Probability Threshold

| Strategy | Threshold | Effect |
|----------|-----------|--------|
| Triple | 0.30 | Only trades when 30%+ net probability differential |
| Vol70_Consec3 | 0.25 | Trades at 25%+ differential |

**Impact:**
- **0.30 threshold**: Stricter filter → only high-conviction signals → smaller losses when wrong
- **0.25 threshold**: More inclusive → more opportunities but slightly higher loss magnitude

### 3. Horizon Selection

```
Triple:     [1, 2, 3, 7, 10]  - Heavy short-term (1,2,3) + gap to medium (7,10)
Vol70_Consec3: [1, 3, 5, 8, 10]  - Balanced across all timeframes
```

**Impact:**
- **Triple's horizon gap (no 4,5,6)**: Creates signal "consensus breaks" that avoid whipsaw periods
- **Vol70's continuous spread**: Captures more regime information but includes conflicting signals

### 4. The Math Behind the Sharpe Difference

```
Triple Filter:
  - Avg Return: 1.69% per trade
  - Avg Loss: 0.11% (extremely small)
  - Return Volatility: Very low
  - Sharpe = High return / Low volatility = 11.96

Vol70_Consec3:
  - Avg Return: 2.88% per trade (higher!)
  - Avg Loss: 0.78% (still small but 7x larger)
  - Return Volatility: Higher due to larger swings
  - Sharpe = Higher return / Higher volatility = 7.90
```

**Key insight:** Triple has a LOWER average return but MUCH lower losses, resulting in superior risk-adjusted performance.

---

## Comparison Summary

| Metric | Triple (Alpha) | Vol70_Consec3 (Hedging) | Winner |
|--------|----------------|-------------------------|--------|
| **Sharpe Ratio** | 11.96 | 7.90 | Triple |
| **Win Rate** | 76.9% | 82.4% | Vol70 |
| **Total Return** | 44.04% | 48.91% | Vol70 |
| **Max Drawdown** | -0.69% | -1.32% | Triple |
| **Profit Factor** | 65.1 | 21.96 | Triple |
| **Avg $/Trade** | $1.08 | $1.81 | Vol70 |
| **Trade Frequency** | 26 trades | 17 trades | Triple |

---

## Persona Recommendations

### Alpha Generation / Hedge Fund (`triple_v70c2p0.3_mixed`)

**Best for:**
- Hedge funds seeking maximum risk-adjusted returns
- Alpha Gen Pro users targeting Sharpe optimization
- Systematic traders with low drawdown tolerance
- Portfolio managers who need consistent performance

**Why:**
- Highest Sharpe ratio (11.96) means best risk-adjusted returns
- Tiny max drawdown (-0.69%) protects capital
- Profit factor of 65.1 indicates highly asymmetric risk/reward
- More trades (26) provides better statistical significance

**Trade-off:** Lower absolute return per trade ($1.08)

---

### Hedging / Procurement Teams (`vol70_consec3_diverse`)

**Best for:**
- Corporate hedging desks managing physical exposure
- Procurement teams timing contract decisions
- Risk managers prioritizing win rate over Sharpe
- Traders who need higher dollar returns per position

**Why:**
- Higher win rate (82.4%) means more predictable outcomes
- Higher absolute return (48.91% vs 44.04%)
- Larger profit per trade ($1.81) works better with fixed position sizing
- Fewer trades (17) means lower transaction costs and simpler execution

**Trade-off:** Higher drawdown risk (-1.32%) and lower Sharpe (7.90)

---

## Decision Matrix

| If your priority is... | Use this strategy |
|------------------------|-------------------|
| Maximum Sharpe ratio | Triple (11.96) |
| Highest win rate | Vol70_Consec3 (82.4%) |
| Lowest drawdown | Triple (-0.69%) |
| Highest total return | Vol70_Consec3 (48.91%) |
| More trading opportunities | Triple (26 trades) |
| Larger profit per trade | Vol70_Consec3 ($1.81) |
| Hedge fund reporting | Triple (Sharpe focus) |
| Corporate hedging | Vol70_Consec3 (win rate focus) |

---

## Configuration Files

```bash
# Triple Filter (Alpha Gen)
# Config not yet in configs/ - use parameters above

# Vol70_Consec3 (Hedging)
configs/vol70_consec3_crude_oil.json
```

---

## Baseline Comparison

Both strategies dramatically outperform the baseline ensemble:

| Metric | Baseline | Triple | Vol70_Consec3 |
|--------|----------|--------|---------------|
| Sharpe | 2.59 | 11.96 (+362%) | 7.90 (+205%) |
| Win Rate | 52.8% | 76.9% | 82.4% |
| Max DD | -7.96% | -0.69% | -1.32% |
| Profit Factor | 2.57 | 65.1 | 21.96 |

---

## Technical Notes

### Why Volatility Filtering Works
The 70th percentile volatility filter ensures trades only occur during "normal" market conditions, avoiding:
- Flash crashes and spikes
- Low-liquidity periods
- News-driven anomalies

### Why Consecutive Signals Matter
Requiring 2-3 consecutive signals filters out:
- Single-day noise
- Model disagreement periods
- Regime transition uncertainty

### Horizon Diversity Explained
Including horizons from short (1-3 days), medium (5-8 days), and long (10+ days) ensures:
- Multiple timeframe confirmation
- Reduced overfitting to single horizon
- Better signal stability

---

*Document generated: February 4, 2026*
*Based on 369 days of backtesting with 10,179 ensemble models*
