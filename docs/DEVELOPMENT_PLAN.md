# Nexus Development Plan

**Version:** 1.0  
**Date:** 2026-02-03  
**Status:** DRAFT — Awaiting Approval  
**Owner:** AmiraB + Artemis  
**Stakeholder:** Bill Dennis  

---

## Executive Summary

Nexus is a production ML ensemble platform for commodity/asset forecasting. CME Group partnership is active — they loved the concept but we need:

1. **Backend fixes** — 6 critical statistical bugs that affect credibility
2. **Spectacular visualizations** — Bloomberg-grade, not just functional charts
3. **7 persona dashboards** — each tailored to a specific user type

**Timeline:** Backend fixes this week. Visualization work begins after.

**Test Asset:** Crude Oil (10,179 models, most data for validation)

---

## Part 1: Backend Fixes

### 1.1 Critical Bugs (Must Fix)

| # | Bug | File | Line | Impact | Fix |
|---|-----|------|------|--------|-----|
| 1 | Data leakage in model selection | `run_dynamic_quantile.py` | 185-207 | Models selected AND validated on same data = overfitting | Split lookback window: 70% train, 30% validate |
| 2 | Wrong Sharpe annualization | `find_diverse_combo.py` | ~180 | Uses `√(252/N_trades)` instead of `√(trades_per_year)` — inflates Sharpe for small samples | Use `252/avg_hold_days` for annualization |
| 3 | Population std instead of sample std | `precalculate_metrics.py` | 139 | `np.std()` without `ddof=1` biases Sharpe upward | Add `ddof=1` to all `np.std()` calls |
| 4 | Hardcoded 0.5% T1 target | `build_qdt_dashboard.py` | 5834 | Asset-agnostic threshold — meaningless for BTC (moves 3-5%/day) vs USD/INR | Scale by ATR: use 0.5× daily ATR as T1 target |
| 5 | Temporal misalignment in accuracy | `precalculate_metrics.py` | 180-188 | Signal at time `i` compared with PAST price change, not future | Compare signal[i] with price change from i to i+1 |
| 6 | Price target sign-flipping | `build_qdt_dashboard.py` | 8475-8482 | When forecasts contradict signal, percentages are force-flipped — hides model disagreement | Remove flip, show actual values, add warning when contradictory |

### 1.2 Code Consolidation (Should Fix)

| Issue | Current State | Proposed Fix |
|-------|---------------|--------------|
| Signal calculation duplicated 5+ times | Same pairwise slope logic in 5 files | Extract to `nexus/lib/signals.py` — single source of truth |
| Trade simulation duplicated 4+ times | Subtle differences in each implementation | Extract to `nexus/lib/trades.py` — unified trade simulation |
| Asset config duplicated 5+ times | Each file has its own ASSETS dict | Centralize in `nexus/config/assets.json` |
| Magic numbers everywhere | 0.5%, 1.5×, 60 days, 30 days scattered | Move to `nexus/config/constants.py` |
| No logging framework | All diagnostics via `print()` | Add Python `logging` module |

### 1.3 Statistical Improvements (Nice to Have)

| Improvement | Benefit | Complexity |
|-------------|---------|------------|
| Weighted pairwise slopes | Better signal quality — weight by horizon separation and magnitude | Medium |
| Walk-forward cross-validation | More robust OOS estimates — multiple 30-day folds instead of one | Medium |
| Fix Sortino ratio denominator | Currently uses N_negative, should use N_total | Easy |
| Add transaction cost modeling | More realistic backtest — even 5bps round-trip matters | Easy |
| Add risk-free rate to Sharpe | Industry standard — subtract ~5% annual | Easy |
| Regime detection | Adapt thresholds per volatility regime | Hard |

### 1.4 Execution Sequence

```
Week 1: Backend Fixes
├── Day 1: Bugs #1-3 (data leakage, Sharpe, std)
├── Day 2: Bugs #4-6 (T1 target, temporal, sign-flip)
├── Day 3: Code consolidation (signals.py, trades.py, assets.json)
├── Day 4: Validation on Crude Oil (before/after comparison)
└── Day 5: Roll out to all 15 assets, regenerate metrics
```

---

## Part 2: Visualization Plan

### 2.1 Design Principles

**"Every pixel should earn its place."**

| Principle | What It Means |
|-----------|---------------|
| **Institutional aesthetic** | Bloomberg/Reuters feel — dark mode, dense but clean, monospace numbers |
| **Interactive everything** | Zoom, pan, hover tooltips, click-to-drill-down on every element |
| **Trust through transparency** | Every number traceable, model disagreement visible, not hidden |
| **Persona-appropriate density** | Quant sees 30+ metrics; Retail sees 3 simple indicators |
| **Mobile where it matters** | Retail/Pro Retail must work on phone; Quant can be desktop-only |

### 2.2 Charting Requirements

**The charts are the product. They must be spectacular.**

| Requirement | Details |
|-------------|---------|
| **Publication quality** | Screenshot-ready for reports, PDFs, presentations |
| **Interactive** | Not static images — zoom, pan, hover, select |
| **Real-time capable** | WebSocket updates for live data (future) |
| **Responsive** | Works on 4K monitors and laptops |
| **Consistent style** | Same visual language across all chart types |
| **Accessible** | Color-blind friendly palettes, clear labels |

### 2.3 Chart Library Decision

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Apache ECharts** | Stunning visuals, huge library, animations, mobile-optimized | JavaScript-only, no Python server mode | **RECOMMENDED for wow-factor** |
| **Bokeh** | Python-native, server mode for real-time, pub-quality | Steeper JS integration, less "wow" | Good for Quant persona |
| **Lightweight Charts** | TradingView's library, trading-specific, fast | Limited chart types | Good for price charts only |
| **D3.js** | Ultimate flexibility, any visualization possible | High dev effort, steep learning curve | For custom one-offs |
| **Plotly** | Current stack, familiar | Looks dated, limited customization | **REPLACE** |

**Recommendation:** ECharts as primary, Lightweight Charts for main price charts, D3 for custom visualizations (heatmaps, dendrograms).

### 2.4 Key Visualization Components

#### A. Main Price Chart (All Personas)
- Candlestick or line (user toggle)
- Signal markers (▲ buy, ▼ sell)
- Price targets (T1, T2, T3 horizontal lines)
- Stop-loss level
- Confidence bands (shaded area)
- Volume bars (bottom)

#### B. Forecast Fan Chart (Quant/HF)
- Multiple horizon forecast lines
- Actual price overlay
- Confidence intervals (5th-95th percentile shading)
- Click any point to see individual model forecasts

#### C. Model Agreement Heatmap (Quant/HF)
- Horizon × Date matrix
- Color intensity = signal strength
- Click cell to drill into contributing models

#### D. Signal Strength Gauge (All)
- Visual confidence meter (circular or linear)
- Current signal prominently displayed
- Historical percentile context

#### E. Equity Curve (Quant/Alpha/HF)
- Strategy performance vs benchmark
- Drawdown overlay (shaded)
- Walk-forward period markers
- In-sample vs out-of-sample splits visible

#### F. Correlation Matrix (Hedging/HF)
- Cross-asset correlations
- Hierarchical clustering dendrogram
- Interactive — click to see time series

#### G. Distribution Analysis (Quant)
- Return distribution histogram
- Fitted normal curve overlay
- QQ plot for normality assessment
- Skewness/kurtosis annotations

#### H. Trade History Table (Alpha/Pro Retail)
- Sortable, filterable
- Entry/exit/P&L/targets hit
- Click row to see chart at that trade

### 2.5 Non-Chart Visualizations

**The dashboard isn't just charts. Everything needs to look premium.**

| Element | Current | Proposed |
|---------|---------|----------|
| **Metric cards** | Plain text | Animated counters, sparklines, trend indicators |
| **Signal indicators** | Text "BULLISH" | Large colored badges with confidence rings |
| **Navigation** | Basic tabs | Sleek sidebar with persona switcher, asset tree |
| **Tables** | Basic HTML | Styled with alternating rows, hover effects, inline charts |
| **Loading states** | Spinner | Skeleton screens, shimmer effects |
| **Alerts/Toasts** | Browser alerts | Styled notifications with icons |
| **Forms/Inputs** | Basic HTML | Custom styled with validation feedback |
| **Tooltips** | Title attributes | Rich tooltips with charts/data |

### 2.6 Color System

#### Dark Mode (Primary)
```
Background Primary:    #0f172a (deep navy)
Background Secondary:  #1e293b (lighter navy)
Background Tertiary:   #334155 (card backgrounds)
Text Primary:          #f8fafc (white)
Text Secondary:        #94a3b8 (muted gray)
Accent Bullish:        #22c55e (green)
Accent Bearish:        #ef4444 (red)
Accent Neutral:        #f59e0b (amber)
Accent Interactive:    #3b82f6 (blue)
```

#### Light Mode (Procurement/Corporate)
```
Background Primary:    #ffffff
Background Secondary:  #f8fafc
Background Tertiary:   #f1f5f9
Text Primary:          #0f172a
Text Secondary:        #64748b
(Accents same as dark mode)
```

---

## Part 3: Persona Dashboards

### 3.1 The 7 Personas

| # | Persona | Primary User | Info Density | Key Focus |
|---|---------|--------------|--------------|-----------|
| 1 | **Hardcore Quant** | Data scientist, model validator | Maximum | All metrics, raw data, statistical tests |
| 2 | **Procurement Team** | Corporate buyer, compliance | Low-Medium | Methodology docs, audit trails, governance |
| 3 | **Hedging Team** | Treasury, commodity hedger | High | Position tracking, Greeks, correlations, basis risk |
| 4 | **Hedge Fund** | Portfolio manager, systematic trader | High | Multi-asset signals, factor attribution, regime detection |
| 5 | **Alpha Gen Pro** | Active trader, signal follower | Medium-High | Entry/exit signals, price targets, trade history |
| 6 | **Pro Retail** | Informed individual investor | Medium | Signals with educational context |
| 7 | **Retail** | Casual investor | Low | Plain English, one-screen view, mobile-first |

### 3.2 Build Priority

| Order | Persona | Rationale |
|-------|---------|-----------|
| 1 | **Alpha Gen Pro** | Most visual impact, proves we can build stunning UI |
| 2 | **Hedging Team** | CME's #1 audience, validates institutional features |
| 3 | **Hardcore Quant** | Proves statistical rigor, appeals to technical buyers |
| 4 | **Hedge Fund** | Multi-asset view, builds on Quant components |
| 5 | **Procurement** | Easy variant of Quant (light mode, less density) |
| 6 | **Pro Retail** | Simplified Alpha Gen + education |
| 7 | **Retail** | Most different design, mobile-first |

### 3.3 Persona Feature Matrix

| Feature | Quant | Procurement | Hedging | HF | Alpha Pro | Pro Retail | Retail |
|---------|-------|-------------|---------|-----|-----------|------------|--------|
| All 30+ metrics | ✓ | | | ✓ | | | |
| Signal + confidence | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Price targets | ✓ | | ✓ | ✓ | ✓ | ✓ | |
| Trade history | ✓ | | | ✓ | ✓ | ✓ | |
| Position tracking | | | ✓ | ✓ | | | |
| Correlation matrix | ✓ | | ✓ | ✓ | | | |
| Regime indicators | ✓ | | | ✓ | | | |
| Educational tooltips | | | | | | ✓ | ✓ |
| Plain English | | ✓ | | | | ✓ | ✓ |
| Mobile layout | | | | | | ✓ | ✓ |
| Audit trail | ✓ | ✓ | ✓ | | | | |
| Export to PDF | ✓ | ✓ | ✓ | ✓ | | | |
| Raw API access | ✓ | | | ✓ | | | |
| Dark mode | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ |
| Light mode | ✓ | ✓ | | | | | |

---

## Part 4: Tech Stack

### 4.1 Frontend

```
Next.js 14 (App Router)
├── TypeScript (strict mode)
├── Tailwind CSS (dark mode support)
├── Shadcn/ui (component library base)
├── Apache ECharts (primary charting)
├── Lightweight Charts (price charts)
├── TanStack Query (data fetching/caching)
├── Zustand (state management)
└── WebSocket (future: real-time updates)
```

### 4.2 Backend

```
Python (existing)
├── Flask API (existing, port 5000)
├── FastAPI (recommended upgrade — async, better docs)
├── quantum_ml integration (Ale's library)
├── PostgreSQL or SQLite (signal/trade history)
└── Redis (optional: caching layer)
```

### 4.3 URL Structure

```
/dashboard/[persona]/[asset]

Examples:
/dashboard/quant/crude-oil
/dashboard/hedging/gold
/dashboard/retail/bitcoin
```

### 4.4 API Endpoints Needed

| Endpoint | Purpose |
|----------|---------|
| `GET /api/assets` | List all assets with current signal |
| `GET /api/assets/{id}/signal` | Current signal, confidence, targets |
| `GET /api/assets/{id}/forecasts` | Forecast data for charts |
| `GET /api/assets/{id}/metrics` | All computed metrics |
| `GET /api/assets/{id}/trades` | Trade history |
| `GET /api/assets/{id}/models` | Model-level data (Quant only) |
| `WS /api/stream` | Real-time signal updates (future) |

---

## Part 5: Roles & Responsibilities

### AmiraB (Backend + Integration)
- Fix all 6 critical bugs
- Code consolidation (signals.py, trades.py, assets.json)
- Integrate quantum_ml for accuracy/feature importance
- Build/extend API endpoints
- Validation and testing
- Frontend-backend integration
- Backend API documentation

### Artemis (Design + Frontend Code)
- Design system (colors, typography, components)
- Persona dashboard designs (wireframes → high-fidelity)
- Chart component styling
- **Frontend implementation (Next.js)** — Artemis codes, not just designs
- Responsive/mobile layouts
- Animations and interactions
- OAuth integration (already built by Farzaneh)

### Bill (Stakeholder)
- Plan approval
- Priority decisions
- CME relationship management
- Final sign-off on designs

---

## Part 6: Timeline

### Week 1: Backend Fixes
```
Mon: Bugs #1-3 (data leakage, Sharpe, std)
Tue: Bugs #4-6 (T1 target, temporal, sign-flip)
Wed: Code consolidation
Thu: Validation on Crude Oil
Fri: Roll out to all 15 assets
```

### Week 2-3: Alpha Gen Pro Dashboard
```
Week 2: Design + chart components
Week 3: Full dashboard implementation
```

### Week 3-4: Hedging Team Dashboard
```
Builds on Alpha Gen Pro foundation
Add position tracking, correlations, scenarios
```

### Week 4-5: Quant + Hedge Fund Dashboards
```
Dense metric views
Model-level analysis
Multi-asset overview
```

### Week 5-6: Retail Personas + Polish
```
Mobile-first designs
Educational features
Final polish, performance, export
```

---

## Part 7: Success Criteria

### Backend Fixes
- [ ] All 6 critical bugs resolved
- [ ] Crude Oil metrics validated (before/after comparison shows improvement)
- [ ] No data leakage in walk-forward (verified with held-out test)
- [ ] Statistical tests pass (Sharpe calculations match industry standard)

### Visualizations
- [ ] Charts look "Bloomberg-grade" (Bill approval)
- [ ] All charts interactive (zoom, pan, hover, drill-down)
- [ ] Consistent visual style across all components
- [ ] Mobile layout works for Retail persona

### Personas
- [ ] Alpha Gen Pro fully functional
- [ ] Hedging Team dashboard ready for CME
- [ ] All 7 personas complete

---

## Part 8: AI Integration (Future Feature)

**Vision:** An AI assistant embedded in Nexus that helps users understand and act on forecasts.

### Core Capabilities

| Capability | Description | Example |
|------------|-------------|---------|
| **Forecast Explainer** | Explain what the signals mean in plain language | "The model sees bullish momentum because 8 of 10 horizons point up, with strongest agreement in the D+3 to D+7 range." |
| **Position Advisor** | Help users size and time their positions | "Given your risk tolerance and the current signal strength, consider a position of X contracts with a stop at Y." |
| **What-If Scenarios** | Answer hypothetical questions | "If crude drops 5% tomorrow, how would that affect my hedge?" |
| **Navigation Help** | Guide users through the platform | "To see the correlation matrix, click on the Hedging persona and select 'Cross-Asset Analysis'." |
| **Product Q&A** | Answer questions about methodology | "How is the confidence score calculated?" → explains the ensemble voting process |
| **Market Context** | Connect forecasts to real-world events | "The bullish signal aligns with OPEC's recent production cut announcement." |
| **Alert Configuration** | Natural language alert setup | "Notify me when crude oil turns bearish or confidence drops below 60%." |
| **Report Generation** | Create custom reports on demand | "Generate a weekly summary of all energy sector signals for my team." |

### Persona-Specific AI Behaviors

| Persona | AI Personality | Focus |
|---------|---------------|-------|
| **Quant** | Technical, precise, cites formulas | Statistical methodology, model details |
| **Hedging** | Risk-focused, compliance-aware | Hedge ratios, basis risk, exposure |
| **Hedge Fund** | Alpha-focused, portfolio-aware | Cross-asset opportunities, factor exposure |
| **Alpha Pro** | Action-oriented, trade-focused | Entry/exit timing, target levels |
| **Retail** | Educational, jargon-free | Simple explanations, learning resources |

### Technical Implementation Ideas

| Approach | Pros | Cons |
|----------|------|------|
| **Chat widget** | Familiar UX, always accessible | Can feel separate from dashboard |
| **Contextual prompts** | AI appears where relevant (hover, click) | More integrated, harder to build |
| **Voice interface** | Hands-free, futuristic | Accessibility concerns, noisier |
| **Command palette** | Power-user friendly (Cmd+K) | Learning curve |

### Data the AI Would Access

- Current signals and confidence for all assets
- User's position (if entered)
- Historical accuracy and trade history
- Model metadata and methodology docs
- User's persona and preferences
- Real-time market context (news, events)

### Future Possibilities

- **Proactive insights**: AI notices something and alerts user ("Your crude hedge is now 40% underwater — want to review?")
- **Learning mode**: AI teaches concepts as user explores
- **Multi-turn research**: "Compare crude and Brent forecasts for the next month"
- **Team collaboration**: AI summarizes discussions, tracks decisions
- **Integration with external tools**: Push signals to trading platforms, spreadsheets

---

## Part 9: Decisions Made

| Question | Decision | Notes |
|----------|----------|-------|
| Artemis scope | **Codes + designs** | Full-stack frontend work |
| Real-time | **No WebSocket needed** | Daily data, polling is fine |
| Branding | **QDT branding** | Focus on getting it working, CME white-label later |
| Authentication | **OAuth (already built)** | Farzaneh has this ready |

## Part 9: Open Questions

1. **Historical depth** — How far back should trade history go? (Currently 369 days)

2. **Alerts** — Email/SMS alerts for signals? What triggers?

3. **Hosting** — Where does the frontend deploy? (Vercel? Self-hosted?)

---

## Appendix A: File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `golden_engine.py` | 64 | Horizon data prep |
| `run_dynamic_quantile.py` | 267 | Walk-forward ensemble |
| `find_diverse_combo.py` | 310 | Horizon optimization |
| `precalculate_metrics.py` | 298 | Trading metrics |
| `build_qdt_dashboard.py` | 9,359 | Dashboard generator |
| `matrix_drift_analysis.py` | 96 | Signal generation |
| `api_server.py` | 314 | REST API |

---

## Appendix B: QuantumCloud API Integration

From Ale's email (2026-02-03):

**Available now (via `quantum_ml` package):**
- `batch_compute_accuracy_metrics()` → directional accuracy
- `compute_feature_importance()` → top drivers per prediction
- `dataset.model_name`, `n_predict`, `n_train` → model metadata

**Integration path:** Import `quantum_ml` into Nexus, call `run_backtest(mdl_table, strategy)` directly.

**Pending:** Confirming if `quantum_ml` is pip-installable or needs repo access.

---

*Document created by AmiraB. Last updated: 2026-02-03 09:38 EST*
