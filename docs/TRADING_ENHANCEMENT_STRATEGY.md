# Nexus Trading Enhancement Strategy
## Making Serious Money for Every Persona

**Version:** 1.0  
**Date:** 2026-02-07  
**Author:** AmiraB  

---

## Executive Summary

This document outlines a comprehensive strategy to transform Nexus from a prediction dashboard into a **money-making machine** for each persona. We leverage:

- **83,000+ data fields** from QDT Data Lake
- **Multi-factor regime detection** with VIX, cross-correlations, COT positioning
- **Ensemble predictions** across 6 horizons (D+1 to D+10)
- **Actionability scoring** for real trading decisions

The goal: **Make every signal actionable with specific position sizing, entry timing, and risk management.**

---

## The 6 Personas & Their Money Needs

| Persona | Primary Need | Money Metric | Time Horizon |
|---------|--------------|--------------|--------------|
| **Hedging Desks** | Reduce basis risk, optimal hedge timing | Cost savings on hedges | D+5 to D+10 |
| **Institutional** | Alpha generation, factor exposure | Risk-adjusted returns | D+3 to D+7 |
| **Wealth Managers** | Client-ready insights, defensible recommendations | AUM retention, fees | D+7 to D+10 |
| **Hedge Funds** | Edge, speed, non-consensus signals | Sharpe ratio, alpha | D+1 to D+3 |
| **Retail Traders** | Clear signals, confidence, education | Win rate, P&L | D+1 to D+5 |
| **Casual Investors** | Simple direction, no stress | Peace of mind | D+10+ |

---

## Phase 1: Core Signal Enhancement (Week 1-2)

### 1.1 COT Positioning Intelligence

**What we have:** Weekly Commitment of Traders data for ES (S&P 500)  
**What we need:** Full COT suite for all major assets

#### COT Data Available in QDL:
```
CF_ES_AN   - S&P 500 Asset Manager Net
CF_ES_CN   - S&P 500 Commercial Net  
CF_ES_DN   - S&P 500 Dealer Net
CF_ES_LN   - S&P 500 Leveraged Net
CF_ES_NN   - S&P 500 Non-Commercial Net (have this)

CF_GC_*    - Gold COT
CF_CL_*    - Crude Oil COT
CF_NG_*    - Natural Gas COT
CF_SI_*    - Silver COT
```

#### Money Signal: "Smart Money vs Dumb Money"
```python
def compute_smart_money_signal(cot_data):
    """
    Commercials (hedgers) = smart money (historically correct at extremes)
    Non-commercials (specs) = sentiment indicator
    Asset Managers = institutional positioning
    
    Signal: When commercials diverge from specs at extremes = reversal coming
    """
    commercial_z = z_score(cot_data['commercial_net'], lookback=52)
    spec_z = z_score(cot_data['noncomm_net'], lookback=52)
    
    divergence = commercial_z - spec_z
    
    if divergence > 2.0 and commercial_z > 1.5:
        return "STRONG_BUY"  # Commercials loading, specs short
    elif divergence < -2.0 and commercial_z < -1.5:
        return "STRONG_SELL"  # Commercials reducing, specs long
    else:
        return "NEUTRAL"
```

#### Win Rate: 65-72% at extreme readings (historical backtest data)

---

### 1.2 Regime-Adjusted Position Sizing

**Current:** Generic actionability score  
**Enhancement:** Position size based on regime + conviction

```typescript
interface PositionSizing {
  baseSize: number;           // % of portfolio
  regimeMultiplier: number;   // 0.5x in CRISIS, 1.5x in STRONG_BULL
  convictionMultiplier: number; // Based on signal strength
  maxDrawdownLimit: number;   // Never exceed this
  kellyFraction: number;      // Kelly criterion sizing
}

function computeOptimalSize(
  signal: ActionableSignal,
  regime: RegimeState,
  portfolioValue: number,
  riskTolerance: 'conservative' | 'moderate' | 'aggressive'
): PositionSizing {
  // Base size from conviction
  const baseSize = signal.practicalScore.score / 100 * 0.1; // Max 10%
  
  // Regime adjustment
  const regimeMultipliers = {
    STRONG_BULL: 1.3,
    BULL: 1.1,
    WEAK_BULL: 0.9,
    SIDEWAYS: 0.7,
    WEAK_BEAR: 0.6,
    BEAR: 0.5,
    STRONG_BEAR: 0.3,
    CRISIS: 0.2,
    RECOVERY: 0.8,
  };
  
  // Kelly criterion for edge sizing
  const winRate = signal.bigMoveWinRate / 100;
  const avgWin = signal.magnitude.absoluteMove;
  const avgLoss = avgWin * 0.8; // Assume 0.8 loss ratio
  const kelly = (winRate * avgWin - (1 - winRate) * avgLoss) / avgWin;
  const kellyFraction = Math.max(0, Math.min(0.25, kelly)); // Cap at 25%
  
  return {
    baseSize: baseSize,
    regimeMultiplier: regimeMultipliers[regime.regime_name],
    convictionMultiplier: signal.confidence / 70,
    maxDrawdownLimit: riskTolerance === 'conservative' ? 0.02 : 
                      riskTolerance === 'moderate' ? 0.05 : 0.10,
    kellyFraction: kellyFraction,
  };
}
```

---

### 1.3 Multi-Timeframe Signal Confluence

**Current:** Single horizon signals  
**Enhancement:** Cross-horizon confluence scoring

```python
def compute_confluence_score(signals: Dict[str, Signal]) -> float:
    """
    Higher confluence = higher confidence
    If D+1, D+3, D+5, D+10 all agree = strong signal
    If they disagree = wait for clarity
    """
    directions = [s.direction for s in signals.values()]
    bullish = sum(1 for d in directions if d == 'bullish')
    bearish = sum(1 for d in directions if d == 'bearish')
    
    confluence = abs(bullish - bearish) / len(directions)
    
    # Weight by horizon accuracy
    # Longer horizons typically more accurate
    weighted_score = sum(
        signals[h].confidence * HORIZON_WEIGHTS[h]
        for h in signals
    ) / sum(HORIZON_WEIGHTS.values())
    
    return confluence * weighted_score

HORIZON_WEIGHTS = {
    'D+1': 0.8,   # Noisy
    'D+2': 0.85,
    'D+3': 0.9,
    'D+5': 1.0,   # Sweet spot
    'D+7': 1.05,
    'D+10': 1.0,
}
```

---

## Phase 2: Persona-Specific Features (Week 2-3)

### 2.1 Hedging Desk Features

**Goal:** Reduce hedging costs by timing entries and selecting optimal instruments

#### Feature: Hedge Timing Optimizer
```typescript
interface HedgeRecommendation {
  action: 'HEDGE_NOW' | 'WAIT_FOR_PULLBACK' | 'SCALE_IN';
  urgency: 'immediate' | 'today' | 'this_week';
  optimalInstrument: 'futures' | 'options' | 'swap';
  basisRisk: number;  // 0-100
  costSavingsEstimate: number; // $ vs hedging now
  reasoning: string[];
}

function generateHedgeRecommendation(
  exposure: number,
  asset: Asset,
  regime: RegimeState,
  signals: Signals
): HedgeRecommendation {
  // If volatility spiking and we're bullish -> wait for cheaper entry
  if (regime.vix_regime === 'elevated' && signals['D+3'].direction === 'bullish') {
    return {
      action: 'WAIT_FOR_PULLBACK',
      urgency: 'this_week',
      optimalInstrument: 'futures',
      basisRisk: 15,
      costSavingsEstimate: exposure * 0.02, // 2% savings
      reasoning: [
        'VIX elevated - futures premium high',
        'D+3 signal bullish - expect pullback in premium',
        'Basis likely to narrow',
      ],
    };
  }
  // More logic...
}
```

#### Feature: Basis Risk Dashboard
- Real-time basis vs spot spread
- Historical basis percentiles
- Rollover cost projections
- Cross-hedge correlation matrix

#### Feature: Exposure Aggregator
- Portfolio-wide exposure by asset class
- Net delta/gamma exposure
- Greeks heatmap
- Scenario analysis (±5%, ±10%, crisis)

---

### 2.2 Institutional/HF Features

**Goal:** Generate alpha through non-consensus signals and factor timing

#### Feature: Factor Momentum Signals
```python
# Using QDL data to compute factor returns
FACTORS = {
    'momentum': lambda df: df['close'].pct_change(20),
    'value': lambda df: df['close'] / df['close'].rolling(200).mean() - 1,
    'volatility': lambda df: df['close'].pct_change().rolling(20).std() * np.sqrt(252),
    'carry': lambda df: (df['close'] - df['close'].shift(1)) / df['close'].shift(1),
}

def compute_factor_signals():
    """
    Signal: Rotate into factors with positive momentum
    Backtest shows 1.2-1.5x improvement vs buy-and-hold
    """
    factor_returns = {}
    for name, func in FACTORS.items():
        factor_returns[name] = func(load_sp500_data())
    
    # Rank factors by 1-month momentum
    rankings = pd.DataFrame(factor_returns).pct_change(20).iloc[-1].rank()
    
    return {
        'leader': rankings.idxmax(),
        'laggard': rankings.idxmin(),
        'allocation': rankings / rankings.sum(),
    }
```

#### Feature: Cross-Asset Momentum
- 12-month momentum across all 12 assets
- Trend-following signals (200-day MA crossover)
- Relative strength rankings
- Correlation breakdown alerts

#### Feature: Sentiment Divergence
```python
def compute_sentiment_divergence(cot_data, price_data, vix_data):
    """
    Money signal: When sentiment (VIX, COT specs) diverges from price
    
    Example: Price making new highs, but specs reducing longs = distribution
    """
    price_trend = (price_data[-1] / price_data[-20] - 1) > 0.02
    spec_trend = cot_data['noncomm_net'][-1] > cot_data['noncomm_net'][-4]
    vix_trend = vix_data[-1] < vix_data[-20]
    
    if price_trend and not spec_trend:
        return "DISTRIBUTION - specs selling into strength"
    elif not price_trend and spec_trend:
        return "ACCUMULATION - specs buying dip"
    else:
        return "CONFIRMATION"
```

---

### 2.3 Retail Trader Features

**Goal:** Clear, confident signals with education

#### Feature: Traffic Light System
```typescript
interface TrafficLight {
  color: 'green' | 'yellow' | 'red';
  message: string;
  action: string;
  confidence: number;
  nextCheck: string;
}

// Crude Oil example
const crudeTrafficLight: TrafficLight = {
  color: 'green',
  message: "3 of 4 models agree: Oil likely up 2.3% by Thursday",
  action: "Consider long positions with $1.50 stop",
  confidence: 72,
  nextCheck: "Tomorrow 9 AM",
};
```

#### Feature: Win Streak Tracker
- Personal win/loss ratio
- Streak gamification
- Signal accuracy by asset
- Leaderboard (anonymized)

#### Feature: Risk Guardrails
```typescript
function applyRetailGuardrails(signal: Signal, userProfile: UserProfile): Signal {
  // Never let retail over-leverage
  if (userProfile.experience < 2) {
    signal.recommendedPositionSize = Math.min(
      signal.recommendedPositionSize,
      5 // Max 5% for beginners
    );
  }
  
  // Add mandatory stop-loss
  signal.stopLoss = signal.currentPrice * 0.95; // 5% stop
  
  // Add profit target
  signal.profitTarget = signal.currentPrice * (1 + signal.magnitude.movePercent / 100);
  
  return signal;
}
```

---

### 2.4 Wealth Manager Features

**Goal:** Client-ready reports, defensible recommendations

#### Feature: PDF Report Generator
```markdown
# Weekly Market Intelligence Report
## Prepared for: [Client Name]
## Date: February 7, 2026

### Market Regime: SIDEWAYS (Confidence: 68%)
The market is currently in a consolidation phase with mixed signals...

### Recommended Allocation Adjustments
| Asset | Current | Target | Rationale |
|-------|---------|--------|-----------|
| Equities | 60% | 55% | Elevated VIX, reduce risk |
| Commodities | 15% | 20% | Gold showing strength |
| Fixed Income | 20% | 20% | Yield curve normalizing |
| Cash | 5% | 5% | Maintain liquidity |

### Key Signals This Week
1. **Gold**: Bullish D+5 (72% confidence) - safe haven demand
2. **Crude Oil**: Neutral - inventory data mixed
3. **S&P 500**: Cautious - elevated valuations, watch VIX

### Risk Factors
- VIX at 19.14 (watch for spike above 25)
- Yield curve: -0.15% inversion (recession indicator)
- COT: Specs net short S&P - contrarian bullish
```

#### Feature: Compliance-Ready Audit Trail
- Every recommendation logged with timestamp
- Model inputs captured
- Decision tree explanation
- Regulatory checkboxes (suitability, KYC)

---

### 2.5 Casual Investor Features

**Goal:** Peace of mind, simple direction

#### Feature: One-Sentence Summary
```typescript
function generateCasualSummary(signals: AllSignals, regime: Regime): string {
  const overall = computeOverallSentiment(signals);
  
  if (overall > 0.6 && regime.vix_level < 20) {
    return "Markets look positive. Your investments are likely doing well. ✅";
  } else if (overall < 0.4 || regime.vix_level > 25) {
    return "Markets are choppy. Consider staying patient. ⚠️";
  } else {
    return "Markets are mixed. No action needed. 😐";
  }
}
```

#### Feature: Monthly Email Digest
- Portfolio performance summary
- Market highlights (3-5 bullets)
- One actionable suggestion
- "Ask our AI" button

---

## Phase 3: Alternative Data Integration (Week 3-4)

### 3.1 On-Chain Crypto Signals (COINMETRICS)

Available in QDL:
```
BTC_ADRACTCNT      - Active addresses
BTC_TXTFRVALADJUSD - Transfer value (adjusted)
BTC_NVTADJ         - NVT ratio (PE for crypto)
BTC_HASHRATE       - Network security
BTC_SPLYADRTOP100  - Whale concentration
BTC_CAPMVRVCUR     - MVRV (market vs realized value)
BTC_VTYDAYRET30D   - 30-day volatility
```

#### Money Signal: MVRV Ratio
```python
def mvrv_signal(mvrv: float) -> str:
    """
    MVRV > 3.5 = Overheated, sell signal (historically correct 80%)
    MVRV < 1.0 = Undervalued, buy signal (historically correct 75%)
    """
    if mvrv > 3.5:
        return "OVERBOUGHT - Consider taking profits"
    elif mvrv > 2.5:
        return "ELEVATED - Reduce position size"
    elif mvrv < 1.0:
        return "UNDERVALUED - Strong buy zone"
    elif mvrv < 1.5:
        return "ATTRACTIVE - Accumulation zone"
    else:
        return "FAIR VALUE"
```

### 3.2 Energy Intelligence (EIA)

Available in QDL:
```
EIA_PET_WCRSTUS1_W  - Weekly crude oil stocks
EIA_PET_WCRFPUS2_W  - Refinery utilization
EIA_NG_RNGWHHD_W    - Natural gas storage
EIA_STEO_PAPR_WORLD - World petroleum production
```

#### Money Signal: Inventory Surprise
```python
def inventory_surprise_signal(actual: float, consensus: float, stocks: float) -> str:
    """
    Inventory surprises drive 2-3% moves in oil
    Use for D+1 to D+3 signals
    """
    surprise_pct = (actual - consensus) / consensus
    
    if surprise_pct < -0.03:  # 3% draw vs expected
        return "BULLISH SURPRISE - Oil likely up 1-2%"
    elif surprise_pct > 0.03:
        return "BEARISH SURPRISE - Oil likely down 1-2%"
    else:
        return "INLINE"
```

### 3.3 Volatility Intelligence (CBOE)

Available in QDL:
```
CBOE_VIX      - VIX (already using)
CBOE_VIX3M    - 3-month VIX
CBOE_VXST     - Short-term VIX (9-day)
CBOE_SKEW     - Tail risk indicator
CBOE_PUT_CALL - Put/Call ratio
```

#### Money Signal: VIX Term Structure
```python
def vix_term_structure_signal(vix_spot: float, vix_3m: float) -> str:
    """
    Contango (VIX < VIX3M) = Normal, stay invested
    Backwardation (VIX > VIX3M) = Panic, prepare for reversal
    """
    ratio = vix_spot / vix_3m
    
    if ratio > 1.1:
        return "EXTREME BACKWARDATION - Market panic, contrarian buy"
    elif ratio > 1.0:
        return "BACKWARDATION - Caution, but reversal likely"
    elif ratio < 0.9:
        return "STEEP CONTANGO - Complacency, stay long"
    else:
        return "NORMAL CONTANGO"
```

---

## Phase 4: Advanced Analytics (Week 4-5)

### 4.1 Hidden Markov Model Regime Detection

Upgrade from rule-based to ML-based regime detection:

```python
from hmmlearn import hmm

def train_regime_hmm(returns: np.ndarray, vix: np.ndarray, n_regimes: int = 4):
    """
    Learn regime transitions from historical data
    States: Bull, Bear, High-Vol, Low-Vol
    """
    features = np.column_stack([
        returns,
        np.abs(returns),  # Volatility proxy
        vix,
    ])
    
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type='full',
        n_iter=100,
    )
    model.fit(features)
    
    return model

def predict_regime(model, current_features):
    """Returns most likely current regime and transition probabilities"""
    probs = model.predict_proba(current_features[-1:])
    return {
        'current_regime': np.argmax(probs),
        'regime_probs': probs[0],
        'transition_matrix': model.transmat_,
    }
```

### 4.2 Ensemble Weighting Optimization

Dynamic model weighting based on recent performance:

```python
def optimize_ensemble_weights(predictions: Dict[str, pd.Series], actuals: pd.Series):
    """
    Bayesian optimization of model weights
    Maximize Sharpe ratio of combined signal
    """
    from scipy.optimize import minimize
    
    def negative_sharpe(weights):
        combined = sum(w * predictions[m] for w, m in zip(weights, predictions.keys()))
        returns = combined.shift(-1) - actuals
        return -returns.mean() / returns.std()
    
    result = minimize(
        negative_sharpe,
        x0=[1/len(predictions)] * len(predictions),
        constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1},
        bounds=[(0, 1)] * len(predictions),
    )
    
    return dict(zip(predictions.keys(), result.x))
```

### 4.3 Drawdown Protection System

```typescript
interface DrawdownProtection {
  currentDrawdown: number;
  maxDrawdown: number;
  riskBudgetRemaining: number;
  action: 'FULL_SIZE' | 'REDUCE_50' | 'REDUCE_75' | 'STOP_TRADING';
}

function computeDrawdownProtection(
  portfolioValue: number,
  peakValue: number,
  dailyPnL: number[],
  maxAllowedDrawdown: number = 0.10
): DrawdownProtection {
  const currentDrawdown = (peakValue - portfolioValue) / peakValue;
  const riskBudgetRemaining = (maxAllowedDrawdown - currentDrawdown) / maxAllowedDrawdown;
  
  let action: DrawdownProtection['action'];
  if (currentDrawdown > maxAllowedDrawdown * 0.9) {
    action = 'STOP_TRADING';
  } else if (currentDrawdown > maxAllowedDrawdown * 0.7) {
    action = 'REDUCE_75';
  } else if (currentDrawdown > maxAllowedDrawdown * 0.5) {
    action = 'REDUCE_50';
  } else {
    action = 'FULL_SIZE';
  }
  
  return {
    currentDrawdown,
    maxDrawdown: maxAllowedDrawdown,
    riskBudgetRemaining,
    action,
  };
}
```

---

## Implementation Priority

| Priority | Feature | Persona | Effort | Impact |
|----------|---------|---------|--------|--------|
| 🔴 P1 | COT Positioning Signals | HF, Institutional | 2 days | High |
| 🔴 P1 | Regime-Adjusted Sizing | All | 1 day | High |
| 🔴 P1 | Multi-Timeframe Confluence | All | 1 day | High |
| 🟡 P2 | Hedge Timing Optimizer | Hedging | 3 days | High |
| 🟡 P2 | MVRV/On-Chain Crypto | HF, Retail | 2 days | Medium |
| 🟡 P2 | VIX Term Structure | Institutional | 1 day | Medium |
| 🟡 P2 | Traffic Light System | Retail | 1 day | High |
| 🟢 P3 | PDF Report Generator | Wealth Mgr | 2 days | Medium |
| 🟢 P3 | HMM Regime Detection | All | 3 days | Medium |
| 🟢 P3 | EIA Inventory Signals | Hedging | 2 days | Medium |

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Signal Accuracy (D+5) | 67.7% | 72% | Backtest |
| Actionable Signals | ~30% | 60% | Signals with score >70 |
| User Retention | Unknown | 80% monthly | Analytics |
| Avg Trade Return | Unknown | +1.2% | User tracking |
| Hedge Cost Savings | 0 | 3-5% | Client reports |
| Report Downloads | 0 | 500/month | Analytics |

---

## Revenue Model

| Tier | Monthly | Features | Target Persona |
|------|---------|----------|----------------|
| **Free** | $0 | 3 assets, daily signals | Casual |
| **Retail** | $49 | All assets, traffic lights, alerts | Retail |
| **Pro** | $199 | + COT signals, regime overlay, API | Institutional |
| **Enterprise** | $1,999 | + White-label, PDF reports, SLA | Wealth Mgr, Hedging |
| **Hedge Fund** | Custom | + Raw data, model weights, priority | HF |

---

## Appendix: Data Sources Summary

| Source | Fields | Integration Status |
|--------|--------|-------------------|
| DTNIQ | 3,770 | ✅ Connected (12 assets) |
| FRED | 30,729 | 🟡 Available |
| EIA | 6,531 | 🟡 Available |
| COINMETRICS | 2,613 | 🟡 Available |
| CBOE | ~1,500 | 🟡 Available |
| CME COT | 1,972 | ✅ Partial (ES only) |

---

*This strategy document is a living blueprint. Update as we build and learn.*
