# Nexus Design Brief

## Core Concept
One platform, thousands of ML models, many user types. Same underlying data — different context, language, and detail level per persona.

## Key Design Challenges

### 1. Persona Context Switching
The same prediction (e.g., "AAPL 82% bullish") needs to feel different for each user:
- **Retail**: "Apple looks strong 📈 — 82% of our models agree"
- **Pro**: "AAPL: 82% ensemble consensus | 1,247/1,521 models bullish | Sharpe: 1.4 | Vol: 23%"
- **Hedge Fund**: "AAPL alpha signal: +2.3σ | Factor decomposition: momentum 40%, sentiment 35%, flow 25%"
- **Wealth Manager**: "Apple Inc. — Strong Buy consensus across our model suite. Suitable for growth-oriented portfolios."
- **Casual**: "Most of our AI models think Apple stock will go up. Here's why in plain English..."

### 2. Information Density
- Retail/casual: Less is more. Cards, simple charts, green/red signals
- Pro/HF: Dense data tables, multi-chart layouts, real-time tickers
- Wealth manager: Clean, printable, client-ready reports

### 3. Trust & Transparency  
- Every prediction should show WHERE it came from (which models, what data)
- Confidence intervals, not just point estimates
- Historical accuracy per model and per ensemble

### 4. Model Ensemble Visualization
- How do 1000+ models get aggregated?
- Consensus view, disagreement view, outlier detection
- Which models are "hot" (recently accurate) vs "cold"

## Design Questions for Artemis
- What's the navigation paradigm? Sidebar persona switch? Separate dashboards? Adaptive?
- Color system — do personas get their own color themes?
- Mobile-first or desktop-first?
- Dark mode standard or optional?
- How do we handle model disagreement visually?

## Inspiration
- Bloomberg Terminal (information density)
- Robinhood (simplicity for retail)
- Figma (persona switching UX)
- Observable (data transparency)

---

*This brief will evolve through collaboration with Artemis.*
