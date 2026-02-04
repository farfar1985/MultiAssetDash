# Nexus UI Design Specification

**Version:** 1.0  
**Date:** 2026-02-02  
**For:** Artemis (Design Collaboration)  
**From:** AmiraB (Backend/Architecture)  

---

## Executive Summary

Nexus needs a complete visual overhaul. The current dashboard is a 408KB monolithic HTML file with basic Plotly charts. We need:

1. **Visually stunning** — Bloomberg-grade institutional aesthetic
2. **7 distinct persona dashboards** — each with unique features and information density
3. **Best-in-class charting** — Bokeh, D3, or similar for rich interactivity
4. **Model visualization** — historical forecasts vs actuals, ensemble consensus, disagreement views
5. **CME Group ready** — this will be shown to hedging desks and procurement teams

---

## Design Principles

### Visual Identity
- **Dark mode default** (trading industry standard)
- **Light mode available** for procurement/corporate environments
- **Color palette**: Deep navy/charcoal backgrounds, accent colors for signals (green/red/amber)
- **Typography**: Monospace for numbers (alignment matters), clean sans-serif for labels
- **Information density**: Adjustable per persona — from "glanceable" to "wall of data"

### Charting Excellence
- **No basic charts** — every visualization should feel premium
- **Interactive by default** — zoom, pan, hover tooltips, click-to-drill-down
- **Responsive** — works on 4K monitors and laptops
- **Real-time capable** — WebSocket updates for live data
- **Export quality** — charts should be screenshot/PDF-ready for reports

### Trust & Transparency
- Every number must be traceable to its source
- Confidence intervals, not just point estimates
- Model disagreement should be visible, not hidden
- Historical accuracy prominently displayed

---

## The 7 Personas

### Overview Matrix

| Persona | Info Density | Primary Use Case | Key Differentiators |
|---------|-------------|------------------|---------------------|
| **Hardcore Quant** | Maximum | Model development, backtesting validation | Raw data, all metrics, statistical tests, code-level detail |
| **Procurement Team** | Low-Medium | Vendor evaluation, due diligence | Compliance focus, audit trails, methodology docs, SLAs |
| **Hedging Team** | High | Position management, exposure hedging | Greeks, correlation matrices, hedge ratios, basis risk |
| **Hedge Fund** | High | Alpha generation, portfolio construction | Factor attribution, regime indicators, risk decomposition |
| **Alpha Gen Pro** | Medium-High | Active trading, signal generation | Entry/exit signals, price targets, momentum indicators |
| **Pro Retail** | Medium | Informed individual trading | Simplified signals with educational context |
| **Retail** | Low | Casual market following | Plain English, visual signals, no jargon |

---

## Persona 1: Hardcore Quant

**User Profile:** Quantitative researcher, data scientist, model validator  
**Goal:** Understand every detail of how the ensemble works, validate statistical rigor

### Dashboard Layout
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER: Asset Selector | Date Range | Refresh | Export | Settings          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ MODEL PERFORMANCE MATRIX        │  │ ENSEMBLE WEIGHT EVOLUTION       │  │
│  │ (heatmap: model × horizon)      │  │ (stacked area: weights over time│  │
│  │ Color = accuracy, click=drill   │  │ Shows which models dominate when│  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ FORECAST VS ACTUAL (Bokeh interactive)                               │  │
│  │ - Multiple series: each horizon's forecast track                     │  │
│  │ - Actual price overlay                                               │  │
│  │ - Confidence bands (5th-95th percentile)                             │  │
│  │ - Click any point to see individual model forecasts                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ STATISTICAL METRICS             │  │ DISTRIBUTION ANALYSIS           │  │
│  │ Sharpe, Sortino, Calmar, VaR    │  │ Return distribution histogram   │  │
│  │ With confidence intervals!      │  │ QQ plot, normality tests        │  │
│  │ Bootstrap standard errors       │  │ Fat tail indicators             │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ WALK-FORWARD ANALYSIS                                                │  │
│  │ - OOS performance by period                                          │  │
│  │ - In-sample vs out-of-sample Sharpe ratio over time                  │  │
│  │ - Degradation detection alerts                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ MODEL CORRELATION MATRIX        │  │ RAW DATA INSPECTOR              │  │
│  │ (which models agree/disagree)   │  │ (searchable table of all data)  │  │
│  │ Hierarchical clustering viz     │  │ CSV export, API endpoint links  │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Unique Features
- **All 30+ quant metrics** with formulas visible on hover
- **Statistical significance tests** — are signals better than random?
- **Regime analysis** — performance breakdown by volatility regime
- **Correlation heatmaps** — model vs model, horizon vs horizon
- **Raw API access** — every chart has "Get API endpoint" button
- **Backtest parameters exposed** — lookback windows, thresholds, everything tunable
- **Code snippets** — "How was this calculated?" shows actual formula/code

### Charts Required
1. Model performance heatmap (Bokeh or D3)
2. Forecast fan chart with confidence bands
3. Walk-forward equity curve with IS/OOS split
4. Return distribution with fitted curves
5. QQ plot for normality assessment
6. Rolling Sharpe ratio over time
7. Model correlation dendrogram
8. Residual analysis plots

---

## Persona 2: Procurement Team

**User Profile:** Corporate buyer, vendor manager, compliance officer  
**Goal:** Evaluate the platform for enterprise adoption, check governance

### Dashboard Layout
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER: Company Logo | "QDT Ensemble Platform" | Help | Contact            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ PLATFORM OVERVIEW                                                    │  │
│  │ "15 Assets • 1000+ ML Models • 5+ Years Track Record"                │  │
│  │ Clean hero section with key stats                                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ METHODOLOGY DOCUMENTATION       │  │ DATA GOVERNANCE                 │  │
│  │ - How signals are generated     │  │ - Data sources & lineage        │  │
│  │ - Model validation process      │  │ - Update frequency              │  │
│  │ - Backtesting methodology       │  │ - Retention policies            │  │
│  │ [Download PDF]                  │  │ - SOC 2 status                  │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ HISTORICAL PERFORMANCE SUMMARY                                       │  │
│  │ Simple chart: strategy vs benchmark over time                        │  │
│  │ Key metrics: Annual return, max drawdown, Sharpe (simplified)        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ AUDIT TRAIL                     │  │ INTEGRATION OPTIONS             │  │
│  │ - Every signal logged with      │  │ - REST API documentation        │  │
│  │   timestamp and model versions  │  │ - Webhook support               │  │
│  │ - Exportable for compliance     │  │ - SSO / SAML integration        │  │
│  │ [Export Audit Log]              │  │ - Custom data feeds             │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ CONTACT & SUPPORT                                                    │  │
│  │ - SLA terms                                                          │  │
│  │ - Support channels                                                   │  │
│  │ - Request demo / Request custom report                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Unique Features
- **NO trading jargon** — use "risk-adjusted return" not "Sharpe ratio"
- **Downloadable documentation** — methodology, data governance, compliance
- **Audit trail viewer** — searchable log of all signals with timestamps
- **White-label ready** — customizable branding
- **Security certifications** — SOC 2 badge, data handling policies
- **Vendor questionnaire helper** — pre-filled answers to common procurement questions

### Visual Style
- **Light mode default** (corporate environments)
- Clean, professional, minimal
- Lots of white space
- Conservative color palette (blues, grays)
- Charts are simple and explanatory

---

## Persona 3: Hedging Team

**User Profile:** Corporate treasury, commodity hedger, risk manager  
**Goal:** Manage exposure, execute hedges, understand basis risk

### Dashboard Layout
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER: Asset | Position Entry | Current Exposure | P&L | Risk Limits      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ EXPOSURE OVERVIEW                                                    │  │
│  │ Current Position: [LONG 500 contracts] | Entry: $72.45 | MTM: +$12K │  │
│  │ Risk Metrics: VaR $45K | Delta: 0.85 | Hedge Ratio: 78%              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ PRICE FORECAST (HEDGER VIEW)   │  │ HEDGE RECOMMENDATION            │  │
│  │ - Forward curve overlay         │  │ Based on ensemble signal:       │  │
│  │ - Confidence bands              │  │ "Consider REDUCING hedge by 20%"│  │
│  │ - Seasonal patterns             │  │ - Current signal: BULLISH       │  │
│  │ - Contango/backwardation        │  │ - Confidence: 72%               │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ CORRELATION & BASIS ANALYSIS                                         │  │
│  │ - Spot vs futures correlation                                        │  │
│  │ - Basis risk indicators                                              │  │
│  │ - Cross-asset correlations (e.g., crude vs products)                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ SCENARIO ANALYSIS               │  │ HISTORICAL HEDGE PERFORMANCE    │  │
│  │ What-if calculator:             │  │ - Past signals vs actual moves  │  │
│  │ "If price moves +5%, P&L = ?"   │  │ - Hedge effectiveness tracking  │  │
│  │ Greeks exposure table           │  │ - Cost of carry analysis        │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ ROLL CALENDAR & ALERTS                                               │  │
│  │ - Upcoming contract expirations                                      │  │
│  │ - Roll recommendations                                               │  │
│  │ - Configurable alerts (price, volatility, position limits)           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Unique Features
- **Position tracking** — enter your position, see P&L in real-time
- **Hedge ratio calculator** — optimal hedge based on correlation
- **Basis analysis** — spot-futures spread, contango/backwardation
- **Scenario/stress testing** — what-if price moves X%
- **Roll calendar** — contract expiration alerts
- **Correlation matrix** — cross-asset correlations
- **Greeks display** — Delta, Gamma if options involved
- **Export to ETRM** — integration with energy trading systems

### Language
- Use hedging terminology: "exposure", "basis risk", "roll yield", "hedge ratio"
- Frame signals as hedging recommendations, not trading signals

---

## Persona 4: Hedge Fund

**User Profile:** Portfolio manager, systematic trader, risk officer  
**Goal:** Generate alpha, manage portfolio risk, attribute returns

### Dashboard Layout
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER: Portfolio View | Single Asset | Risk | Alpha | Settings            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ SIGNAL SUMMARY (All Assets)                                          │  │
│  │ ┌────────┬────────┬────────┬────────┬────────┐                       │  │
│  │ │ CRUDE  │ GOLD   │ BTC    │ SP500  │ NASDAQ │  ... (15 assets)      │  │
│  │ │ ▲ 78%  │ ▼ 65%  │ ▲ 82%  │ ─ 51%  │ ▲ 71%  │                       │  │
│  │ │ +2.1σ  │ -1.4σ  │ +2.8σ  │ +0.2σ  │ +1.6σ  │                       │  │
│  │ └────────┴────────┴────────┴────────┴────────┘                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ ALPHA SIGNAL STRENGTH           │  │ FACTOR ATTRIBUTION              │  │
│  │ Z-score visualization           │  │ - Momentum contribution         │  │
│  │ (how unusual is current signal) │  │ - Mean-reversion contribution   │  │
│  │ Historical percentile rank      │  │ - Sentiment contribution        │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ REGIME INDICATOR                                                     │  │
│  │ Current: [HIGH VOLATILITY REGIME] — signal thresholds adjusted       │  │
│  │ Regime history chart with overlay on price                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ RISK DECOMPOSITION              │  │ DRAWDOWN ANALYSIS               │  │
│  │ - Systematic vs idiosyncratic   │  │ - Current drawdown              │  │
│  │ - Factor exposures              │  │ - Historical drawdown dist.     │  │
│  │ - Tail risk metrics             │  │ - Recovery time estimates       │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ PORTFOLIO OPTIMIZER (Optional)                                       │  │
│  │ - Kelly-optimal position sizes                                       │  │
│  │ - Correlation-aware allocation                                       │  │
│  │ - Risk budget constraints                                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Unique Features
- **Multi-asset overview** — see all 15 assets at a glance
- **Z-score signals** — how unusual is the current signal vs history
- **Factor attribution** — what's driving the signal (momentum, mean-rev, etc.)
- **Regime detection** — automatic regime classification with adapted thresholds
- **Portfolio optimizer** — Kelly sizing, correlation-aware allocation
- **Risk decomposition** — systematic vs idiosyncratic risk
- **Tail risk metrics** — VaR, CVaR, expected shortfall
- **Integration with OMS** — ability to push signals to order management

### Visual Style
- Dense but clean
- Multiple small charts (small multiples pattern)
- Color-coded heatmaps for quick scanning
- Dark mode with high contrast

---

## Persona 5: Alpha Gen Pro

**User Profile:** Active trader, technical analyst, signal follower  
**Goal:** Get clear entry/exit signals with price targets

### Dashboard Layout
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER: Asset Selector | Timeframe | Alerts | Account                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          MAIN CHART                                  │  │
│  │  Full-width Bokeh chart with:                                        │  │
│  │  - Candlesticks / line (toggle)                                      │  │
│  │  - Signal markers (▲ buy, ▼ sell)                                    │  │
│  │  - Price targets (T1, T2, T3 horizontal lines)                       │  │
│  │  - Stop-loss level                                                   │  │
│  │  - Confidence bands                                                  │  │
│  │  - Volume bars                                                       │  │
│  │  Height: 400px minimum                                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ SIGNAL PANEL                                                         │  │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │  │
│  │ │  SIGNAL     │ │  ENTRY      │ │  TARGETS    │ │  STOP       │      │  │
│  │ │  ▲ BULLISH  │ │  $72.45     │ │  T1: $73.20 │ │  $71.10     │      │  │
│  │ │  78% conf   │ │  NOW        │ │  T2: $74.00 │ │  -1.9%      │      │  │
│  │ │             │ │             │ │  T3: $75.50 │ │  ATR-based  │      │  │
│  │ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ TECHNICAL FILTERS               │  │ HISTORICAL ACCURACY             │  │
│  │ ☑ RSI Filter (currently: 55)   │  │ T1 Hit Rate: 72% (last 20)      │  │
│  │ ☑ EMA 50 (price above ✓)       │  │ Avg hold time: 4.2 days         │  │
│  │ ☐ Ichimoku Cloud               │  │ Win rate: 68%                   │  │
│  │ [Accuracy updates dynamically]  │  │ Profit factor: 2.1             │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ TRADE HISTORY (Last 10)                                              │  │
│  │ Date     | Signal | Entry  | Exit   | P&L    | Targets Hit           │  │
│  │ 01/28    | BULL   | 71.20  | 73.45  | +3.2%  | T1 ✓ T2 ✓ T3 ✗        │  │
│  │ ...                                                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Unique Features
- **Actionable signals** — clear entry, targets, stops
- **Technical filter toggles** — RSI, EMA, Ichimoku with live accuracy updates
- **Trade history** — recent signals and outcomes
- **Alert configuration** — push/email when new signal fires
- **Position size calculator** — based on account size and risk tolerance
- **Risk/reward visualization** — potential gain vs stop distance
- **Multi-timeframe view** — toggle between daily/weekly signals

### Visual Style
- Trading-platform feel (TradingView-like)
- Large, prominent chart
- Clear signal indicators
- Green/red color coding
- Dark mode

---

## Persona 6: Pro Retail

**User Profile:** Informed individual investor, follows markets actively  
**Goal:** Get signals with educational context, understand the "why"

### Dashboard Layout
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER: Assets | My Watchlist | Learning | Settings                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ CURRENT SIGNAL                                                       │  │
│  │                                                                       │  │
│  │         ▲ BULLISH on Crude Oil                                       │  │
│  │           Confidence: 78%                                            │  │
│  │                                                                       │  │
│  │   "Our ensemble of ML models sees upward momentum.                   │  │
│  │    8 out of 10 time horizons point higher."                          │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ PRICE CHART (Interactive)                                            │  │
│  │ Clean chart with signal markers and simple trend indicators          │  │
│  │ "Learn" mode: click any element for explanation                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ WHY THIS SIGNAL?                │  │ SIGNAL TRACK RECORD             │  │
│  │ Expandable explanation:         │  │ Win Rate: 68%                   │  │
│  │ - Which models agree            │  │ "Out of the last 25 signals,    │  │
│  │ - Key factors driving signal    │  │  17 were profitable"            │  │
│  │ - What could change it          │  │                                 │  │
│  │ [Learn More →]                  │  │ [See full history]              │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ EDUCATIONAL SIDEBAR                                                  │  │
│  │ 💡 "What is an ensemble model?"                                      │  │
│  │    An ensemble combines many models to get a more reliable signal.   │  │
│  │    Think of it like asking 100 experts instead of just one.          │  │
│  │                                                                       │  │
│  │ 💡 "What does confidence mean?"                                      │  │
│  │    Higher confidence = more models agree. 78% means strong agreement.│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Unique Features
- **Plain language explanations** — no unexplained jargon
- **"Why this signal?"** — expandable explanation of what's driving it
- **Educational tooltips** — click anything to learn what it means
- **Watchlist** — save favorite assets
- **Learning center** — glossary, tutorials, methodology explainer
- **Simplified metrics** — win rate and track record, not Sharpe ratios
- **Risk warnings** — clear disclaimers about trading risks

### Visual Style
- Clean and approachable
- More white space than pro versions
- Friendly, educational tone
- Mobile-responsive

---

## Persona 7: Retail

**User Profile:** Casual investor, market curious, minimal trading experience  
**Goal:** Quick glance at market direction, simple yes/no guidance

### Dashboard Layout
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER: QDT Market Insights                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │                    [ASSET ICON: Oil Barrel]                          │  │
│  │                                                                       │  │
│  │                      Crude Oil                                       │  │
│  │                                                                       │  │
│  │              ┌─────────────────────────┐                             │  │
│  │              │     ▲ LOOKS GOOD        │                             │  │
│  │              │                         │                             │  │
│  │              │   Our AI models think   │                             │  │
│  │              │   oil prices will rise  │                             │  │
│  │              │                         │                             │  │
│  │              │   ●●●●○ Strong Signal   │                             │  │
│  │              └─────────────────────────┘                             │  │
│  │                                                                       │  │
│  │  ← Swipe for more assets →                                           │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ SIMPLE CHART                                                         │  │
│  │ Sparkline with green/red coloring, minimal axis labels               │  │
│  │ "Up 2.3% this week"                                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ WHAT THIS MEANS                                                      │  │
│  │ "Oil prices might continue rising over the next few days.            │  │
│  │  This could affect gas prices and energy stocks."                    │  │
│  │                                                                       │  │
│  │ ⚠️ This is not investment advice. Always do your own research.       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                  │
│  │  [Gold]       │  │  [Bitcoin]    │  │  [S&P 500]    │                  │
│  │  ▲ Good       │  │  ▲ Strong     │  │  ─ Neutral    │                  │
│  └───────────────┘  └───────────────┘  └───────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Unique Features
- **One-screen view** — everything visible without scrolling
- **Visual signal indicators** — green up arrow, red down arrow, gray neutral
- **Plain English** — "Looks Good" not "Bullish", "Strong Signal" not "78% confidence"
- **Swipe navigation** — mobile-first, card-based
- **Real-world context** — "This could affect gas prices"
- **Prominent disclaimers** — clear risk warnings
- **No numbers overload** — hide complexity

### Visual Style
- **Mobile-first design**
- Large touch targets
- Card-based UI
- Friendly illustrations/icons
- Minimal text
- Calming color palette (not aggressive red/green)

---

## Charting Library Recommendation

### Primary: Bokeh (Python) + BokehJS (Frontend)

**Why Bokeh:**
- **Publication quality** — charts look professional out of the box
- **Python-native** — integrates with existing Python backend
- **Interactive** — zoom, pan, hover, linked brushing
- **Server mode** — real-time updates via WebSocket
- **Embeddable** — outputs standalone HTML or integrates with Flask/Django
- **Large dataset handling** — WebGL backend for performance

### Alternative: Apache ECharts

**Why ECharts:**
- **Stunning visuals** — beautiful animations and effects
- **Massive chart library** — every chart type imaginable
- **Lightweight** — no Python dependency
- **Mobile optimized** — touch interactions built in
- **Huge community** — lots of examples and themes

### For Specific Needs:

| Need | Recommended |
|------|-------------|
| Complex financial charts | Bokeh or Lightweight Charts (TradingView's library) |
| Heatmaps/Matrices | D3.js or Plotly |
| Real-time streaming | Bokeh Server or Apache ECharts with WebSocket |
| Mobile-first | Apache ECharts or Chart.js |
| 3D visualizations | Three.js or Plotly |
| Network/relationship graphs | D3.js or Cytoscape.js |

---

## Technical Architecture Recommendation

### Frontend Stack
```
Next.js 14 (App Router)
├── TypeScript (strict mode)
├── Tailwind CSS (dark mode support)
├── Shadcn/ui (component library)
├── Bokeh or ECharts (charting)
├── TanStack Query (data fetching)
├── Zustand (state management)
└── WebSocket (real-time updates)
```

### Persona Switching
```typescript
// Persona context provider
type Persona = 
  | 'quant' 
  | 'procurement' 
  | 'hedging' 
  | 'hedge_fund' 
  | 'alpha_pro' 
  | 'pro_retail' 
  | 'retail';

// Each persona gets:
// - Different layout component
// - Different chart configurations
// - Different metric visibility
// - Different language/terminology
// - Different color theme (optional)
```

### URL Structure
```
/dashboard/quant/crude-oil
/dashboard/hedging/gold
/dashboard/retail/bitcoin
```

### API Layer
- REST endpoints for data fetching
- WebSocket for real-time signal updates
- GraphQL optional for complex queries

---

## Color System

### Dark Mode Palette (Primary)
```css
--bg-primary: #0f172a;      /* Deep navy */
--bg-secondary: #1e293b;    /* Lighter navy */
--bg-tertiary: #334155;     /* Card backgrounds */
--text-primary: #f8fafc;    /* White */
--text-secondary: #94a3b8;  /* Muted */
--accent-green: #22c55e;    /* Bullish */
--accent-red: #ef4444;      /* Bearish */
--accent-amber: #f59e0b;    /* Neutral/Warning */
--accent-blue: #3b82f6;     /* Links/Interactive */
```

### Light Mode Palette (Procurement/Corporate)
```css
--bg-primary: #ffffff;
--bg-secondary: #f8fafc;
--bg-tertiary: #f1f5f9;
--text-primary: #0f172a;
--text-secondary: #64748b;
/* Accents same as dark mode */
```

### Persona-Specific Accents (Optional)
- **Quant**: Purple accent (#8b5cf6) for "data science" feel
- **Hedge Fund**: Gold accent (#eab308) for "premium" feel
- **Retail**: Teal accent (#14b8a6) for "friendly" feel

---

## Next Steps for Artemis

1. **Pick a persona to prototype first** — recommend Alpha Gen Pro (most visual impact) or Retail (most different from current)

2. **Design the charting components** — we need reusable chart components that can be configured per persona

3. **Create a design system** — colors, typography, spacing, component library

4. **Wireframe the persona switcher** — how do users move between personas?

5. **Define the animation/interaction model** — loading states, transitions, hover effects

---

## Questions for Artemis

1. Should personas have completely separate dashboards (different URLs) or a unified dashboard with a persona toggle?

2. Do we want theme customization (let users pick colors) or strict per-persona themes?

3. Mobile priority — which personas need mobile support? (Retail definitely, Quant probably not)

4. Chart library preference — any experience with Bokeh vs ECharts vs others?

5. Animation level — minimal and fast, or rich and expressive?

---

*This document is a living spec. Update as design decisions are made.*

**— AmiraB, 2026-02-02**
