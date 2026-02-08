# Quant Methods Research — Jan-Feb 2026

**Compiled:** 2026-02-07  
**Sources:** arXiv, Bloomberg, SentimenTrader, Build Alpha, Barchart

---

## 1. Regime Detection — State of the Art

### Recent Paper: arXiv:2601.08571 (Jan 2026)
**"Regime Discovery and Intra-Regime Return Dynamics in Global Equity Markets"**

Key findings from this cutting-edge research:

**Method: Hilbert-Huang Transform (HHT)**
- Uses Empirical Mode Decomposition (EMD) to extract intrinsic mode functions
- Instantaneous energy from Hilbert spectrum separates regimes
- Three regime states: **Normal, High, Extreme**

**Key Insights:**
1. **Entropy peaks in High regimes** — maximum unpredictability during moderate-volatility periods (not extreme!)
2. **Developed markets normalize faster** — sharper volatility decline from Extreme to Normal
3. **Developing markets retain tail dependence** — residual downside persistence even in Normal regimes
4. **Variable-Length Markov Chains** — better than fixed-order HMM for transition dynamics

**Application for Nexus:**
- Our current 9-regime system aligns well (STRONG_BULL to CRISIS)
- Consider adding entropy-based signal (high entropy = reduce position size)
- Use VLMCs instead of HMMs for regime transitions

### Recommended Implementation
```python
# Regime detection with Hilbert-Huang Transform
from PyEMD import EMD
from scipy.signal import hilbert

def compute_hht_regime(price_series):
    """
    Hilbert-Huang Transform based regime detection.
    Uses instantaneous energy to classify Normal/High/Extreme.
    """
    emd = EMD()
    imfs = emd.emd(price_series.values)
    
    # Get instantaneous amplitude from first 3 IMFs
    energy = 0
    for imf in imfs[:3]:
        analytic = hilbert(imf)
        amplitude = np.abs(analytic)
        energy += amplitude ** 2
    
    # Classify based on energy percentiles
    pct = np.percentile(energy[-252:], [33, 66])
    current = energy[-1]
    
    if current > pct[1]:
        return "EXTREME"
    elif current > pct[0]:
        return "HIGH"
    else:
        return "NORMAL"
```

---

## 2. Alternative Data Signals

### 2.1 COT Positioning ✅ (Implemented)
- **Z-score contrarian signals** at extremes (|z| > 2.0)
- Historical win rate: 65-74%
- Currently tracking: Gold, Crude Oil, S&P 500

### 2.2 Options Flow
**Source:** Barchart, Unusual Whales, FlowAlgo

Key signals:
- **Large block trades** — Institutional positioning
- **Put/Call ratio extremes** — Sentiment indicator
- **Unusual options activity** — Smart money bets

**Example Signal:**
> "Options Flow Alert: Institutional Money Loading Up on Google Stock"
> Large call sweeps at above-ask prices = bullish institutional positioning

**Implementation Priority:** P2 (requires paid data source)

### 2.3 Sentiment Indicators
**Sources:** SentimenTrader, AAII, Fear & Greed Index

**SentimenTrader Features:**
- Smart Money / Dumb Money Confidence spread
- Sector fund flows
- Options premium ratios
- Insider trading patterns

**AAII Sentiment:**
- Weekly survey of individual investors
- Contrarian signal at extremes
- Bullish > 60% or Bearish > 60% = watch for reversal

### 2.4 Bloomberg Alternative Data
**Recent announcement (Jan 2026):**
> "Bloomberg Introduces Alternative Data Entitlements on the Terminal for Increased Alpha Discovery"

Available data:
- Satellite imagery (parking lots, shipping)
- Credit card transactions
- Web scraping (job postings, prices)
- Social sentiment

**Cost:** Enterprise pricing (likely $50K+/year)

---

## 3. Position Sizing Best Practices

### 3.1 Kelly Criterion ✅ (Implemented)
```python
f* = (p * b - q) / b
# where p = win probability, b = win/loss ratio
```

**Practical adjustments:**
- Use half-Kelly (f/2) for safety margin
- Cap at 25% max position
- Reduce during high VIX

### 3.2 Volatility Targeting
**Method:** Scale position inversely with volatility

```python
def volatility_target_size(base_size, current_vol, target_vol=0.15):
    """
    If target vol = 15% and current vol = 30%, halve position.
    """
    return base_size * (target_vol / current_vol)
```

### 3.3 Drawdown Control ✅ (Implemented)
- Track peak portfolio value
- Scale position by remaining risk budget
- Stop trading at 90% of max drawdown

---

## 4. Multi-Asset Signals

### 4.1 Cross-Asset Momentum
**Strategy:** Go long assets with positive 12-month momentum

**Historical performance:**
- Dual momentum (absolute + relative): ~15% CAGR, 0.8 Sharpe
- Time-series momentum: 60-65% hit rate

### 4.2 Risk Parity
**Method:** Allocate inversely to volatility

```python
def risk_parity_weights(volatilities):
    inv_vol = 1 / np.array(volatilities)
    return inv_vol / inv_vol.sum()
```

### 4.3 Factor Timing
**Factors to track:**
- Momentum (12-1 month)
- Value (P/E, P/B relative to history)
- Quality (ROE, debt ratios)
- Low Volatility

**Signal:** Rotate into factors with positive 1-month return

---

## 5. Crypto-Specific Signals (COINMETRICS)

### 5.1 MVRV Ratio
**Market Value to Realized Value**
- MVRV > 3.5 = Overheated, sell signal (80% accuracy)
- MVRV < 1.0 = Undervalued, buy signal (75% accuracy)

### 5.2 NVT Ratio
**Network Value to Transactions**
- NVT > 150 = Overvalued
- NVT < 20 = Undervalued

### 5.3 Active Addresses
- 30-day MA vs 200-day MA
- Golden cross = bullish
- Death cross = bearish

### 5.4 Whale Concentration
- Top 100 wallets % of supply
- Increasing concentration = accumulation
- Decreasing = distribution

---

## 6. Implementation Roadmap

| Priority | Signal | Data Source | Effort | Impact |
|----------|--------|-------------|--------|--------|
| ✅ Done | COT Positioning | QDL (CME COT) | — | High |
| ✅ Done | Regime Detection | QDL (prices) | — | High |
| ✅ Done | Position Sizing | Internal | — | High |
| P1 | HHT Regime Enhancement | PyEMD | 2 days | Medium |
| P1 | MVRV/NVT Crypto | QDL (COINMETRICS) | 1 day | High |
| P2 | VIX Term Structure | QDL (CBOE) | 1 day | Medium |
| P2 | Put/Call Ratio | QDL (CBOE) | 1 day | Medium |
| P3 | Options Flow | Barchart/FlowAlgo | 3 days | High |
| P3 | Sentiment Index | SentimenTrader | 2 days | Medium |

---

## 7. Key Takeaways

1. **Entropy matters** — High regimes (not Extreme) have max unpredictability. Reduce size in high entropy periods.

2. **Variable-Length Markov > HMM** — Better for regime transitions.

3. **COT works** — 68-74% win rate at extremes. Already implemented.

4. **MVRV is gold for crypto** — Simple, effective, 75-80% accuracy.

5. **Half-Kelly is practical** — Full Kelly too aggressive.

6. **Volatility targeting smooths returns** — Scale inversely with vol.

---

*Research compiled from web search, arXiv, and industry sources. Last updated 2026-02-07.*
