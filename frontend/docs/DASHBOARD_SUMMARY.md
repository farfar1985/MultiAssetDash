# QDT Nexus Dashboard Summary

This document provides an overview of all persona-specific dashboards built for the QDT Nexus trading platform.

## Dashboard Overview

| Persona | Route | Theme | Target User |
|---------|-------|-------|-------------|
| Alpha Gen Pro | `/dashboard/alphapro` | Purple/Blue | Professional traders |
| Hedging Team | `/dashboard/hedging` | Emerald/Teal | Risk managers |
| Quant | `/dashboard/quant` | Cyan/Blue | Quantitative analysts |
| Retail Trader | `/dashboard/retail` | Orange/Amber | Individual investors |
| Executive | `/dashboard/executive` | Slate/Neutral | C-suite executives |

---

## Alpha Gen Pro Dashboard

**Route:** `/dashboard/alphapro`
**Theme:** Purple gradient with blue accents
**File:** `components/dashboard/AlphaProDashboard.tsx`

### Key Features
- **Alpha Score** - Composite score combining Sharpe, accuracy, and model consensus
- **Performance Stats** - Average Sharpe, accuracy, expected P&L
- **Top Alpha Signals** - Ranked opportunities with detailed metrics
- **Model Consensus Visualization** - Shows agreement across ensemble models
- **Live Signal Cards** - Real-time API integration for popular assets

### Metrics Displayed
- Sharpe Ratio
- Directional Accuracy
- Confidence Level
- Model Agreement (X/10,179 models)
- Expected Move %

---

## Hedging Dashboard

**Route:** `/dashboard/hedging`
**Theme:** Emerald/Teal gradient
**File:** `components/dashboard/HedgingDashboard.tsx`

### Key Features
- **$/Trade Focus** - Primary metric for hedging decisions
- **Win Rate Analysis** - Directional accuracy for risk management
- **Horizon Coverage** - Signal consistency across D+1, D+5, D+10
- **Hedge Type Classification** - Protective, Speculative, or Basis trades
- **Urgency Indicators** - ACT NOW, MONITOR, or WATCH labels
- **Actionable Summary Panel** - Quick lists of urgent actions

### Metrics Displayed
- $/Trade (expected value per contract)
- Win Rate %
- Horizon Consistency %
- Hedge Score (composite)
- Confidence Level

### Hedge Types
| Type | Description | Color |
|------|-------------|-------|
| Protective | Bearish signals with high confidence | Emerald |
| Speculative | High Sharpe ratio opportunities | Purple |
| Basis | Standard directional trades | Cyan |

---

## Quant Dashboard

**Route:** `/dashboard/quant`
**Theme:** Cyan/Blue gradient
**File:** `components/dashboard/QuantDashboard.tsx`

### Key Features
- **Statistical Rigor** - Full statistical distributions and confidence intervals
- **Model Performance Matrix** - Grid view of all models with accuracy breakdowns
- **Advanced Metrics Panel** - Sortino, Calmar, Information Ratio
- **Distribution Visualizations** - Signal direction distributions
- **Historical Accuracy Trends** - Time-series performance data

### Metrics Displayed
- Sharpe Ratio (with std dev)
- Sortino Ratio
- Calmar Ratio
- Information Ratio
- Maximum Drawdown
- Directional Accuracy (up/down breakdown)
- Model Count & Agreement

### Special Components
- **AdvancedMetricsPanel** - Deep-dive statistics
- **ModelPerformanceMatrix** - Per-model accuracy breakdown
- **DistributionChart** - Visual signal distributions

---

## Retail Dashboard

**Route:** `/dashboard/retail`
**Theme:** Orange/Amber gradient
**File:** `components/dashboard/RetailDashboard.tsx`

### Key Features
- **Big Direction Arrows** - Animated UP/DOWN arrows with clear labels
- **Confidence Gauge** - Visual semi-circle gauge (0-100%)
- **Plain English Explanations** - Human-readable signal descriptions
- **BUY/SELL/HOLD Recommendations** - Simple action guidance
- **How to Read Signals Panel** - Educational component
- **Today's Top Picks** - Quick action list

### Metrics Displayed (Simplified)
- Confidence % (as visual gauge)
- Signal Direction (BUY/SELL/HOLD)
- Time Horizon (Tomorrow/This week/Next 2 weeks)
- Model Agreement (simplified)

### Plain English Examples
> "Our AI thinks Gold will go UP. This is a strong signal to consider buying."

> "Bitcoin doesn't have a clear direction right now. It might be best to wait for a stronger signal."

### Design Principles
- No jargon (no Sharpe ratios, no directional accuracy)
- Visual-first (gauges, arrows, colors)
- Action-oriented (what should I do?)
- Educational (help panel explains signals)

---

## Executive Dashboard

**Route:** `/dashboard/executive`
**Theme:** Slate/Neutral with accent colors
**File:** `components/dashboard/ExecutiveDashboard.tsx`

### Key Features
- **Minimal Design** - Clean, uncluttered interface
- **Headline Numbers** - 4 key metrics at a glance
- **Market Outlook** - Single paragraph AI-generated summary
- **Portfolio Health** - Simple status indicators
- **Top Opportunities** - Brief list without technical details

### Metrics Displayed
- Portfolio Value
- Daily P&L
- Win Rate
- Active Signals Count

### Design Principles
- Minimal cognitive load
- Decision-support focused
- Mobile-friendly
- Printable for board meetings

---

## Shared Components

All dashboards use these shared components:

| Component | Description |
|-----------|-------------|
| `LiveSignalCard` | Real-time API-connected signal display |
| `ApiHealthIndicator` | Shows API connection status |
| `Card`, `Badge` | UI primitives from shadcn/ui |
| `Separator` | Visual section dividers |

---

## Color Themes Reference

| Dashboard | Primary | Secondary | Accent |
|-----------|---------|-----------|--------|
| Alpha Pro | Purple-900 | Blue-900 | Purple-500 |
| Hedging | Emerald-900 | Teal-900 | Emerald-500 |
| Quant | Cyan-900 | Blue-900 | Cyan-500 |
| Retail | Orange-900 | Amber-900 | Orange-500 |
| Executive | Slate-900 | Neutral-900 | Blue-500 |

---

## Data Sources

All dashboards consume data from:
- `lib/mock-data.ts` - Mock signal and asset data
- `/api/ensemble` - Live ensemble predictions (when API enabled)
- `types/index.ts` - TypeScript type definitions

---

## Adding New Dashboards

1. Create component in `components/dashboard/[Name]Dashboard.tsx`
2. Add persona to `types/index.ts` PERSONAS object
3. Add conditional render in `app/dashboard/[persona]/page.tsx`
4. Follow existing patterns for consistency
