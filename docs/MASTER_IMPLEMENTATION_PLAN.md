# Nexus Master Implementation Plan

**Version:** 2.0  
**Date:** 2026-02-03  
**Authors:** AmiraB + Artemis  
**Stakeholder:** Bill Dennis  
**Status:** COMPREHENSIVE BLUEPRINT

---

## Executive Vision

We're building the most visually stunning, AI-powered trading intelligence platform in the industry. When CME's hedging desks see this, their jaws will drop.

**What makes Nexus special:**
1. **Bloomberg-grade aesthetics** — Every pixel earns its place
2. **AI Co-Pilot** — An embedded assistant that explains, advises, and guides
3. **Radical Transparency** — Every number is traceable to its source
4. **Persona Adaptation** — Same data, 7 optimized presentations
5. **Historical Time Travel** — See how metrics evolved over any period
6. **Model Disagreement as a Feature** — When models disagree, we show it

---

## Part 1: Current State Assessment

### 1.1 What We Have (Data Assets)

**From QuantumCloud API (Ale's emails):**

| Capability | Endpoint/Method | Status |
|------------|-----------------|--------|
| Model predictions | `/get_qml_models/<project_id>` | ✅ Available |
| Historical accuracy | `batch_compute_accuracy_metrics()` | ✅ Via quantum_ml |
| Feature importance | `compute_feature_importance()` | ✅ Via quantum_ml |
| Model metadata | `dataset.model_name`, `n_predict`, etc. | ✅ Via quantum_ml |
| Backtest metrics | `run_backtest(mdl_table, strategy)` | ✅ Via quantum_ml |
| Historical rewind | Truncate mdl_table to any date | ✅ Via quantum_ml |
| Confidence intervals | Models can produce them | ⚠️ Not wired up yet |

**Key Insight from Ale:** The mdl_table is **append-only by design**. Every day adds one row per model. This means we can:
```python
# "Time travel" to any historical date
rewind_date = '2025-06-15'
mdl_table_historical = mdl_table[mdl_table['time'] <= rewind_date]
historical_metrics = run_backtest(mdl_table_historical, strategy)
```

This is GOLD for CME — we can show the full trajectory of model reliability over time.

**Asset Coverage:**

| Asset | Models | Horizons | Best For |
|-------|--------|----------|----------|
| Crude Oil | 10,179 | D+1 to D+200 | Primary test asset, CME priority |
| Bitcoin | 4,962 | D+1 to D+100 | Volatility testing |
| S&P 500 | 3,968 | D+1 to D+60 | Index baseline |
| Gold | 2,819 | D+1 to D+60 | Commodity pair |
| NASDAQ | 2,659 | D+1 to D+60 | Tech correlation |
| Brent Oil | 624 | D+1 to D+30 | Crude Oil pair |
| Others | 9-500 | Varies | Secondary priority |

**Data Limitations:**
- 369 days of history (2025-01-01 to 2026-01-04)
- 3,000 CSV files (I/O bottleneck — migrate to HDF5)
- quantum_ml not pip-installable (need GitHub repo access)
- Opaque model IDs until we integrate quantum_ml
- No real-time data (daily updates only)

### 1.2 Current Codebase Issues

**AmiraB's 6 Backend Bugs:**

| # | Bug | File:Line | Impact | Fix |
|---|-----|-----------|--------|-----|
| 1 | Data leakage | `run_dynamic_quantile.py:185-207` | Overfitting | Split lookback 70/30 |
| 2 | Wrong Sharpe | `find_diverse_combo.py:~180` | Inflated metrics | Use `252/avg_hold_days` |
| 3 | Population std | `precalculate_metrics.py:139` | Biased Sharpe | Add `ddof=1` |
| 4 | Fixed T1 target | `build_qdt_dashboard.py:5834` | Meaningless accuracy | Scale by ATR |
| 5 | Temporal misalignment | `precalculate_metrics.py:180-188` | Wrong direction | Compare i to i+1 |
| 6 | Sign-flipping | `build_qdt_dashboard.py:8475-8482` | Hides disagreement | Remove, add warning |

**Artemis's 7 Security Issues:**

| # | Issue | Location | Risk | Fix |
|---|-------|----------|------|-----|
| 1 | SSH key exposed | Git history | Server compromise | BFG + rotate |
| 2 | Hardcoded API key | `api_server.py:63-70` | API abuse | Delete + hash |
| 3 | Plaintext keys | `api_keys.json` | Credential theft | PBKDF2 + salt |
| 4 | CORS wildcard | `api_server.py:54` | XSS attacks | Restrict domains |
| 5 | No rate limiting | Defined but off | DoS risk | Flask-Limiter |
| 6 | No validation | All endpoints | Injection | Whitelist inputs |
| 7 | Stack traces | Error handlers | Info leak | Generic errors |

**Artemis's Additional Bugs:**

| # | Bug | Location | Impact |
|---|-----|----------|--------|
| 8 | Division by zero | `precalculate_metrics.py:195` | Crashes |
| 9 | Empty DataFrame | `run_dynamic_quantile.py:202-203` | IndexError |
| 10 | Race condition | Checkpoint resume | Corruption |
| 11 | Memory leak | gc.collect() symptom | Performance |
| 12 | Silent exceptions | `api_server.py:825` | Hidden errors |
| 13 | Global mutation | `config_sandbox.py` | Thread unsafe |

**Architecture Issues:**
- `build_qdt_dashboard.py` = 7,500 lines, 408KB monolith
- Signal calculation duplicated in 4+ files
- 3,000 CSV files (should be 15 HDF5)
- Zero unit tests
- No logging infrastructure
- No database (JSON files only)

---

## Part 2: The Vision — What We're Building

### 2.1 The Ultimate Trading Intelligence Platform

**Tagline:** *"See what the models see. Decide with confidence."*

**Core Principles:**
1. **Trust through Transparency** — Every metric has provenance
2. **Beauty is Functional** — Stunning visuals that communicate instantly
3. **AI Augmentation** — Intelligence amplified, not replaced
4. **Persona-Appropriate** — Right information, right density, right language
5. **Time Travel** — History builds confidence

### 2.2 The 7 Personas (Refined)

| Persona | User | Goal | Info Density | Key Features |
|---------|------|------|--------------|--------------|
| **Hardcore Quant** | Data scientist | Validate models | Maximum | All 30+ metrics, statistical tests, raw API |
| **Procurement** | Corporate buyer | Due diligence | Low-Medium | Methodology docs, audit trails, compliance |
| **Hedging Team** | Treasury, hedger | Manage exposure | High | Position tracking, Greeks, correlations |
| **Hedge Fund** | Portfolio manager | Generate alpha | High | Multi-asset, factors, regimes |
| **Alpha Gen Pro** | Active trader | Get signals | Medium-High | Entry/exit, targets, stops |
| **Pro Retail** | Informed investor | Understand signals | Medium | Educational context, "why" explanations |
| **Retail** | Casual investor | Quick glance | Low | Plain English, mobile-first, one screen |

### 2.3 AI Agent Integration — The Differentiator

This is what will blow people away. An AI assistant embedded in every dashboard.

**8 Core Capabilities:**

| Capability | What It Does | Example |
|------------|--------------|---------|
| **Forecast Explainer** | Explains signals in plain language | "8 of 10 horizons agree, strongest in D+3 to D+7 range" |
| **Position Advisor** | Helps size trades | "Given 72% confidence, consider 2% position with 1.5 ATR stop" |
| **What-If Scenarios** | Runs hypotheticals | "If crude drops 5%, your hedge would offset $12K" |
| **Navigation Help** | Guides the platform | "To see correlations, open Hedging persona" |
| **Historical Analysis** | Time travel queries | "Model was 78% accurate during OPEC meetings" |
| **Alert Configuration** | Natural language alerts | "Notify me when gold turns bearish" |
| **Report Generation** | Custom reports | "Weekly energy summary, PDF, to my team" |
| **Learning Mode** | Educational content | "Sharpe ratio measures risk-adjusted return..." |

**Persona-Specific AI Behavior:**

| Persona | AI Personality | Communication Style |
|---------|---------------|---------------------|
| Quant | Technical, precise | Cites formulas, confidence intervals |
| Hedging | Risk-focused | Mentions hedge ratios, basis risk |
| Hedge Fund | Alpha-oriented | Factor exposure, portfolio context |
| Alpha Pro | Action-focused | Entry timing, target levels |
| Retail | Educational | Jargon-free, analogies |

**UI Implementation:**

| Component | Priority | Description |
|-----------|----------|-------------|
| **Command Palette (Cmd+K)** | P0 | Primary interaction, instant access |
| **Contextual Hints** | P1 | Inline suggestions based on what user views |
| **Chat Panel** | P2 | Optional sidebar for extended conversations |
| **Voice Input** | P3 | Future capability |

---

## Part 3: Visualization Strategy

### 3.1 Design Philosophy

**"Every pixel should earn its place."**

| Principle | Implementation |
|-----------|----------------|
| Institutional aesthetic | Bloomberg/Reuters DNA |
| Interactive everything | Zoom, pan, hover, drill-down on ALL charts |
| Trust through transparency | Click any number → see how it was calculated |
| Persona-appropriate density | Quant sees 30 metrics; Retail sees 3 |
| Publication quality | Screenshot-ready for reports |
| Accessible | Color-blind friendly, WCAG 2.1 AA |

### 3.2 Color System

**Dark Mode (Primary):**
```css
--bg-primary:       #0f172a;  /* Deep navy */
--bg-secondary:     #1e293b;  /* Card backgrounds */
--bg-tertiary:      #334155;  /* Elevated surfaces */
--text-primary:     #f8fafc;  /* High contrast */
--text-secondary:   #94a3b8;  /* Muted */
--text-tertiary:    #64748b;  /* Disabled */
--accent-bullish:   #22c55e;  /* Green */
--accent-bearish:   #ef4444;  /* Red */
--accent-neutral:   #f59e0b;  /* Amber */
--accent-info:      #3b82f6;  /* Blue */
```

**Persona Accents:**
```css
--accent-quant:      #8b5cf6;  /* Purple — data science */
--accent-hedgefund:  #eab308;  /* Gold — premium */
--accent-retail:     #14b8a6;  /* Teal — friendly */
```

**Light Mode (Procurement):**
```css
--bg-primary:       #ffffff;
--bg-secondary:     #f8fafc;
--text-primary:     #0f172a;
/* Accents unchanged */
```

### 3.3 Chart Library Stack

| Library | Use Case | Why |
|---------|----------|-----|
| **Apache ECharts** | Primary charts | Stunning visuals, animations, mobile |
| **Lightweight Charts** | Price charts | TradingView DNA, trading-specific |
| **D3.js** | Custom viz | Heatmaps, dendrograms, networks |

### 3.4 Key Visualizations

**1. Main Price Chart (All Personas)**
- Candlesticks or line (toggle)
- Volume bars (bottom)
- Signal markers (▲ buy, ▼ sell) with animations
- Confidence bands (shaded fan)
- Price targets (T1, T2, T3 horizontal lines)
- Stop-loss level (dashed red)
- Hover: individual model forecasts

**2. Forecast Fan Chart (Quant/HF)**
- Multiple horizon forecast lines
- Actual price overlay
- 5th-95th percentile confidence shading
- **Time Machine Slider** — drag to any historical date
- Click any point → see model breakdown

**3. Signal Strength Gauge (All)**
- Circular confidence meter
- Animated needle movement
- Color gradient (red → amber → green)
- Historical percentile rank ring
- Pulse effect on signal change

**4. Model Agreement Heatmap (Quant/HF)**
- Horizon × Date matrix
- Color intensity = signal strength
- Click cell → drill into contributing models
- Animated transitions on time change

**5. Equity Curve (Quant/Alpha/HF)**
- Strategy performance line
- Benchmark overlay
- Drawdown fill (shaded below)
- Walk-forward period markers
- Rolling Sharpe secondary axis

**6. Correlation Matrix (Hedging/HF)**
- Cross-asset heatmap
- Hierarchical clustering dendrogram
- Click → time series comparison
- Regime overlay (colors for different market states)

**7. Distribution Analysis (Quant)**
- Return histogram
- Fitted normal curve
- QQ plot
- Fat tail indicators
- Skewness/kurtosis annotations

**8. Trade History Table (Alpha/Pro Retail)**
- Sortable, filterable
- Entry/exit/P&L/targets hit
- Inline sparklines
- Click row → chart annotation

### 3.5 Non-Chart Elements

All must be equally polished:

| Element | Current | Target |
|---------|---------|--------|
| Metric cards | Plain text | Animated counters, sparklines, trend arrows |
| Signal badges | Text label | Large colored circles, confidence rings, pulse |
| Navigation | Basic tabs | Sleek sidebar, persona avatar, asset tree |
| Tables | Basic HTML | Alternating rows, hover glow, inline charts |
| Loading | Spinner | Skeleton screens, shimmer effect |
| Alerts | Browser alert | Styled toasts, icons, stacking |
| Tooltips | Title attr | Rich content, mini-charts, click-to-pin |
| Buttons | Default | Custom styles, hover states, loading states |
| Inputs | Default | Custom validation, inline feedback |

---

## Part 4: Technical Architecture

### 4.1 Frontend Stack

```
Next.js 14 (App Router)
├── TypeScript (strict mode)
├── Tailwind CSS + CSS Variables
├── Shadcn/ui (component foundation)
├── Apache ECharts
├── Lightweight Charts (TradingView)
├── D3.js (custom viz)
├── TanStack Query (data fetching/caching)
├── Zustand (state management)
├── Framer Motion (animations)
├── next-themes (dark/light mode)
├── cmdk (command palette)
└── AI SDK (Vercel AI or custom)
```

### 4.2 Backend Stack

```
Python/FastAPI
├── Async endpoints
├── Pydantic v2 (validation)
├── Redis (caching)
├── quantum_ml (direct import)
├── h5py (HDF5 data)
├── JWT + OAuth (auth)
├── slowapi (rate limiting)
├── structlog (logging)
└── pytest (testing)
```

### 4.3 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  QuantumCloud API                                               │
│        ↓                                                        │
│  quantum_ml (run_backtest, batch_compute_accuracy_metrics)      │
│        ↓                                                        │
│  HDF5 Storage (15 files, one per asset)                         │
│        ↓                                                        │
│  FastAPI Endpoints                                              │
│        ↓                                                        │
│  Redis Cache (OHLCV: 5min, forecasts: 1hr, metrics: 1hr)       │
│        ↓                                                        │
│  Next.js (TanStack Query with SWR)                              │
│        ↓                                                        │
│  User Interface                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        AI PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query (text/voice)                                        │
│        ↓                                                        │
│  Context Builder                                                │
│    - Current signals for viewed asset                           │
│    - User's position (if entered)                               │
│    - Recent chat history                                        │
│    - Persona context (terminology level)                        │
│        ↓                                                        │
│  Prompt Template (persona-specific)                             │
│        ↓                                                        │
│  LLM (Claude/GPT-4)                                             │
│        ↓                                                        │
│  Response Streaming                                             │
│        ↓                                                        │
│  UI Rendering (markdown, charts, actions)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 API Endpoints

**Assets:**
```
GET  /api/assets                    # List all assets with current signal
GET  /api/assets/{id}               # Asset details
GET  /api/assets/{id}/signal        # Current signal + confidence + targets
GET  /api/assets/{id}/forecasts     # Forecast data for charts
GET  /api/assets/{id}/metrics       # All computed metrics
GET  /api/assets/{id}/trades        # Trade history
GET  /api/assets/{id}/history       # Historical metrics (time machine)
GET  /api/assets/{id}/models        # Model-level data (Quant only)
```

**AI:**
```
POST /api/ai/chat                   # Conversational AI
POST /api/ai/explain                # Explain a specific metric/signal
POST /api/ai/scenario               # Run what-if scenario
POST /api/ai/report                 # Generate report
```

**User:**
```
GET  /api/user/profile              # User preferences
PUT  /api/user/preferences          # Update preferences
GET  /api/user/alerts               # User's alerts
POST /api/user/alerts               # Create alert
GET  /api/user/positions            # User's positions
POST /api/user/positions            # Add position
```

### 4.5 URL Structure

```
/                                   # Landing → redirect to default persona
/dashboard/[persona]/[asset]        # Main dashboard view

Examples:
/dashboard/quant/crude-oil
/dashboard/hedging/gold
/dashboard/alpha-pro/bitcoin
/dashboard/retail/sp500

/settings                           # User settings
/alerts                             # Alert management
/reports                            # Report history
```

---

## Part 5: Implementation Phases

### Phase 0: Emergency Security (24-48 Hours)
**Owner: Artemis**
**Priority: CRITICAL — Do before anything else**

| Task | Status | Notes |
|------|--------|-------|
| Remove SSH key from git history | ⬜ | Use BFG Repo-Cleaner |
| Rotate exposed SSH key | ⬜ | Generate new, update servers |
| Delete hardcoded API key | ⬜ | `api_server.py:63-70` |
| Implement API key hashing | ⬜ | PBKDF2 + salt |
| Lock down CORS | ⬜ | Whitelist specific domains |
| Enable rate limiting | ⬜ | Flask-Limiter + Redis |
| Remove stack trace exposure | ⬜ | Generic error responses |
| Add security headers | ⬜ | CSP, HSTS, X-Frame-Options |

**Deliverable:** Secure codebase, all secrets rotated

---

### Phase 1: Foundation (Week 1)
**Goal:** Fix critical bugs, establish project structure

**AmiraB Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Fix bug #1: Data leakage | P0 | 4h |
| Fix bug #2: Sharpe annualization | P0 | 2h |
| Fix bug #3: Population std | P0 | 1h |
| Fix bug #4: ATR-scaled T1 | P0 | 3h |
| Fix bug #5: Temporal alignment | P0 | 2h |
| Fix bug #6: Remove sign-flipping | P0 | 2h |
| Create shared `signals.py` | P1 | 4h |
| Create shared `trades.py` | P1 | 3h |
| Centralize `assets.json` | P1 | 2h |
| Validate on Crude Oil | P0 | 4h |
| Document all changes | P1 | 2h |

**Artemis Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Fix bug #8: Division by zero | P0 | 2h |
| Fix bug #9: Empty DataFrame | P0 | 1h |
| Fix bug #10: Race condition | P1 | 3h |
| Set up Next.js 14 project | P0 | 2h |
| Configure Tailwind + CSS vars | P0 | 2h |
| Install Shadcn/ui components | P1 | 2h |
| Set up OAuth authentication | P0 | 4h |
| Create project structure | P1 | 2h |
| Set up ESLint + Prettier | P2 | 1h |

**Shared:**
| Task | Owner | Notes |
|------|-------|-------|
| Get quantum_ml repo access | Both | Email sent to Ale |
| Define API contracts | Both | Document in shared doc |
| Set up GitHub branch protection | Artemis | Require PR reviews |

**Deliverable:** Bug-free backend (Crude Oil), Next.js scaffold, auth working

---

### Phase 2: Data & Infrastructure (Week 2)
**Goal:** Build the data layer and infrastructure

**AmiraB Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Migrate CSVs to HDF5 | P0 | 8h |
| Integrate quantum_ml | P0 | 8h |
| Build `/api/assets` endpoint | P0 | 2h |
| Build `/api/assets/{id}/signal` | P0 | 3h |
| Build `/api/assets/{id}/forecasts` | P0 | 3h |
| Build `/api/assets/{id}/metrics` | P0 | 4h |
| Build `/api/assets/{id}/history` | P1 | 4h |
| Set up Redis caching | P1 | 4h |
| Write API documentation | P1 | 3h |

**Artemis Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Break up 408KB monolith | P0 | 12h |
| Create chart theme (ECharts) | P0 | 4h |
| Integrate Lightweight Charts | P0 | 3h |
| Build base chart components | P0 | 8h |
| Build loading states/skeletons | P1 | 3h |
| Build metric card component | P1 | 3h |
| Build signal badge component | P1 | 2h |
| Set up Storybook | P2 | 3h |

**Deliverable:** API serving real data, chart components ready, monolith split

---

### Phase 3: Alpha Gen Pro Dashboard (Week 3-4)
**Goal:** First complete persona — prove the pattern

**Artemis Tasks (Week 3):**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Dashboard layout | P0 | 4h |
| Main price chart | P0 | 12h |
| Signal panel | P0 | 6h |
| Technical filters panel | P1 | 4h |
| Trade history table | P1 | 4h |
| Responsive design | P0 | 4h |

**Artemis Tasks (Week 4):**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Alert configuration UI | P1 | 6h |
| Dark mode polish | P0 | 4h |
| Animations/transitions | P1 | 4h |
| Historical accuracy panel | P1 | 4h |
| Position size calculator | P2 | 3h |
| Mobile optimization | P1 | 4h |

**AmiraB Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Signal calculation pipeline | P0 | 6h |
| Price target calculation | P0 | 4h |
| Historical accuracy calc | P0 | 4h |
| Trade simulation engine | P0 | 6h |
| Roll bug fixes to all 15 assets | P1 | 8h |
| Performance optimization | P1 | 4h |

**Deliverable:** Alpha Gen Pro fully functional for all 15 assets

---

### Phase 4: AI Agent MVP (Week 5)
**Goal:** Embedded AI assistant with core features

**AmiraB Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| AI context builder | P0 | 8h |
| Prompt templates (per persona) | P0 | 6h |
| LLM integration (Claude API) | P0 | 4h |
| Response streaming | P0 | 4h |
| Forecast Explainer feature | P0 | 6h |
| Position Advisor feature | P1 | 6h |
| What-If Scenario engine | P1 | 8h |
| Conversation memory | P2 | 4h |

**Artemis Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Command palette (Cmd+K) | P0 | 8h |
| AI response rendering | P0 | 6h |
| Markdown + chart rendering | P1 | 4h |
| Contextual hints system | P1 | 6h |
| Chat panel (sidebar) | P2 | 6h |
| AI loading states | P1 | 2h |
| Feedback mechanism | P2 | 3h |

**Deliverable:** AI assistant operational in Alpha Gen Pro

---

### Phase 5: Hedging Team Dashboard (Week 6)
**Goal:** CME's priority audience

**Artemis Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Dashboard layout | P0 | 4h |
| Position tracking UI | P0 | 8h |
| Hedge recommendation panel | P0 | 6h |
| Correlation matrix viz | P0 | 8h |
| Scenario calculator | P1 | 6h |
| Roll calendar | P1 | 4h |
| Basis risk indicators | P1 | 4h |
| Greeks display | P2 | 4h |

**AmiraB Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Position tracking API | P0 | 6h |
| Correlation calculation | P0 | 4h |
| Scenario engine API | P1 | 6h |
| Hedge ratio optimization | P1 | 6h |
| Basis calculation | P1 | 3h |

**Deliverable:** Hedging Team dashboard complete

---

### Phase 6: Quant & Hedge Fund Dashboards (Week 7)
**Goal:** Power user dashboards

**Artemis Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Quant: Model performance heatmap | P0 | 8h |
| Quant: Forecast fan chart | P0 | 8h |
| Quant: Distribution analysis | P1 | 6h |
| Quant: Walk-forward view | P1 | 6h |
| Quant: Time machine slider | P0 | 6h |
| HF: Multi-asset signal grid | P0 | 8h |
| HF: Factor attribution | P1 | 6h |
| HF: Regime indicator | P1 | 4h |

**AmiraB Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| All 30+ metrics API | P0 | 8h |
| Walk-forward analysis data | P0 | 6h |
| Factor decomposition | P1 | 8h |
| Regime detection | P1 | 8h |
| Model-level data API | P1 | 4h |

**Deliverable:** Quant and Hedge Fund dashboards complete

---

### Phase 7: Retail Personas (Week 8)
**Goal:** Accessible dashboards for broader audience

**Artemis Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Pro Retail: Dashboard layout | P0 | 4h |
| Pro Retail: Educational tooltips | P0 | 6h |
| Pro Retail: "Why this signal?" | P0 | 4h |
| Pro Retail: Watchlist | P1 | 4h |
| Retail: Mobile-first layout | P0 | 8h |
| Retail: Plain English signals | P0 | 4h |
| Retail: Card-based UI | P0 | 4h |
| Retail: Swipe navigation | P1 | 3h |

**AmiraB Tasks:**

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Simplified API responses | P0 | 4h |
| Plain English generation | P0 | 6h |
| Educational content | P1 | 4h |
| Retail-specific AI prompts | P1 | 3h |

**Deliverable:** All 7 personas complete

---

### Phase 8: Polish & Launch (Week 9+)
**Goal:** Production-ready

**Shared Tasks:**

| Task | Priority | Owner |
|------|----------|-------|
| Performance audit | P0 | Both |
| Cross-browser testing | P0 | Artemis |
| Accessibility audit | P0 | Artemis |
| Security audit | P0 | Artemis |
| API documentation | P0 | AmiraB |
| User documentation | P1 | Both |
| CI/CD pipeline | P0 | Artemis |
| Docker deployment | P0 | Artemis |
| Monitoring setup | P1 | AmiraB |
| Final CME review | P0 | Both |

**Deliverable:** Production deployment, CME-ready

---

## Part 6: Division of Labor

### Clear Ownership

| Domain | Primary Owner | Secondary |
|--------|---------------|-----------|
| **Security** | Artemis | — |
| **Frontend Architecture** | Artemis | — |
| **UI Components** | Artemis | — |
| **Charts & Viz** | Artemis | — |
| **Animations** | Artemis | — |
| **Backend Bugs** | AmiraB | — |
| **Data Pipeline** | AmiraB | — |
| **API Design** | AmiraB | Artemis reviews |
| **AI Engine** | AmiraB | Artemis reviews |
| **AI UI** | Artemis | AmiraB reviews |
| **Testing (E2E)** | Artemis | — |
| **Testing (Unit/API)** | AmiraB | — |
| **DevOps** | Artemis | — |
| **Documentation** | Both | — |

### Collaboration Protocol

| Activity | How |
|----------|-----|
| **Daily sync** | Check Google Doc for updates |
| **Code reviews** | All PRs require approval from other |
| **Blockers** | Flag immediately in doc |
| **API changes** | Discuss first in doc, then implement |
| **Design decisions** | Propose in doc, get thumbs up |
| **Weekly milestone** | Demo to Bill at end of each phase |

### Communication Channels

| Channel | Use For |
|---------|---------|
| **Google Doc** | Planning, async discussion, decisions |
| **GitHub PRs** | Code reviews, technical discussion |
| **Email** | Formal communication, external parties |
| **Direct message** | Urgent blockers only |

---

## Part 7: Testing Strategy

### Unit Tests (Target: 60% coverage on critical paths)

| Area | Coverage Target | Owner |
|------|-----------------|-------|
| Signal calculation | 90% | AmiraB |
| Metrics calculation | 90% | AmiraB |
| API authentication | 80% | Artemis |
| API endpoints | 70% | AmiraB |
| Data validation | 80% | AmiraB |

### Integration Tests

| Area | Scope | Owner |
|------|-------|-------|
| API → Database | All endpoints | AmiraB |
| API → Redis | Caching | AmiraB |
| API → quantum_ml | Integration | AmiraB |

### E2E Tests (Playwright)

| Flow | Priority | Owner |
|------|----------|-------|
| Dashboard navigation | P0 | Artemis |
| Chart interactions | P0 | Artemis |
| AI conversation | P1 | Artemis |
| Alert creation | P1 | Artemis |
| Mobile responsive | P1 | Artemis |

### Visual Regression (Chromatic or Percy)

| Component | Owner |
|-----------|-------|
| All chart types | Artemis |
| Metric cards | Artemis |
| Signal badges | Artemis |
| Dark/Light themes | Artemis |

### Performance Benchmarks

| Metric | Target | Test Method |
|--------|--------|-------------|
| API response time | <200ms | k6 load test |
| Chart render time | <500ms | Lighthouse |
| Time to interactive | <3s | Lighthouse |
| Memory usage | <100MB | Chrome DevTools |

---

## Part 8: Success Metrics

### Technical Success

| Metric | Target | How to Measure |
|--------|--------|----------------|
| All bugs fixed | 13/13 | Issue tracker |
| API response time | <200ms | Monitoring |
| Test coverage | >60% | Coverage report |
| Security vulnerabilities | 0 | Security scan |
| Lighthouse score | >90 | Lighthouse audit |

### User Experience Success

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Bill's approval | "Spectacular" | Demo feedback |
| CME demo | "Jaws drop" | Meeting feedback |
| All personas working | 7/7 | Manual testing |
| AI assistant functional | Core features | Manual testing |
| Mobile responsive | Retail works | Device testing |

### Business Success

| Metric | Target | How to Measure |
|--------|--------|----------------|
| CME partnership | Secured | Contract signed |
| Time to market | 8 weeks | Calendar |
| Technical debt | Minimal | Code review |

---

## Part 9: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| quantum_ml access delayed | Medium | High | Work on what we can without it; Ale is responsive |
| Security incident | Low | Critical | Phase 0 first; rotate all keys TODAY |
| Scope creep | High | Medium | Strict phase gates; Bill approval at milestones |
| Performance issues | Medium | Medium | Early benchmarking; Redis caching |
| AI costs | Medium | Low | Rate limiting; caching; usage tracking |
| Integration bugs | Medium | Medium | Extensive testing; staged rollout |

---

## Part 10: Open Questions

| Question | Owner to Decide | Notes |
|----------|-----------------|-------|
| LLM provider | Bill | Claude vs GPT-4 vs both |
| Hosting | Bill | Vercel? AWS? Self-hosted? |
| Domain | Bill | qdtnexus.ai? nexus.qdt.ai? |
| AI usage limits | Bill | Tokens per user per day? |
| Historical data expansion | Ale | Data before 2025-01-01? |
| Real-time capability | Future | WebSocket for live updates? |

---

## Appendix A: File Reference

| File | Lines | Owner | Phase |
|------|-------|-------|-------|
| `run_dynamic_quantile.py` | 267 | AmiraB | 1 |
| `find_diverse_combo.py` | 310 | AmiraB | 1 |
| `precalculate_metrics.py` | 298 | AmiraB | 1 |
| `build_qdt_dashboard.py` | 9,359 | Artemis | 2 |
| `api_server.py` | 314 | Artemis (security) | 0 |
| `golden_engine.py` | 64 | AmiraB | 1 |
| `config_sandbox.py` | 52 | AmiraB | 1 |

---

## Appendix B: quantum_ml Integration Guide

Based on Ale's emails, here's how to use quantum_ml:

```python
# Get metrics for any historical date
def get_historical_metrics(mdl_table, rewind_date, strategy=11):
    """
    Args:
        mdl_table: Full model table from /get_qml_models
        rewind_date: Date string 'YYYY-MM-DD'
        strategy: 9=long-only, 10=short-only, 11=long/short
    Returns:
        dict: All backtest metrics as of that date
    """
    historical = mdl_table[mdl_table['time'] <= rewind_date]
    return run_backtest(historical, strategy)

# Get accuracy metrics
def get_accuracy_metrics(mdl_table):
    """Returns DataFrame with acc_predict, acc_predict_up, acc_predict_down"""
    return batch_compute_accuracy_metrics(mdl_table)

# Get feature importance
def get_feature_importance(model, algorithm_name, X, y, dataset):
    """Returns feature importance dict for a specific model"""
    return compute_feature_importance(model, algorithm_name, X, y, dataset, {})
```

---

## Appendix C: Design Tokens

```typescript
// theme.ts
export const theme = {
  colors: {
    bg: {
      primary: '#0f172a',
      secondary: '#1e293b',
      tertiary: '#334155',
    },
    text: {
      primary: '#f8fafc',
      secondary: '#94a3b8',
      tertiary: '#64748b',
    },
    accent: {
      bullish: '#22c55e',
      bearish: '#ef4444',
      neutral: '#f59e0b',
      info: '#3b82f6',
    },
    persona: {
      quant: '#8b5cf6',
      hedgefund: '#eab308',
      retail: '#14b8a6',
    },
  },
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
  },
  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '12px',
    full: '9999px',
  },
  fonts: {
    sans: 'Inter, system-ui, sans-serif',
    mono: 'JetBrains Mono, Consolas, monospace',
  },
};
```

---

*This document is the single source of truth for Nexus implementation. Update as decisions are made.*

**Created by:** AmiraB  
**Last updated:** 2026-02-03 14:13 EST
