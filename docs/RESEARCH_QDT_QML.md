# QDT & QML Research Notes

*Source: qdt.ai, Feb 2026*

## QDT — Quantum Data Technologies
- Founded 2017, self-funded, Vancouver BC
- Serves Fortune 500 (Mars is a named client)
- Co-founder/CEO: **Rajiv Chandrasekaran, PhD, Machine Learning**
- Co-founder/Chairman: **Bill Dennis**

## QML — Quantum Machine Learning Platform

### What It Is
- **Automated Machine Learning (AutoML)** platform for time-series forecasting
- No-code, point-and-click interface — non-technical users (procurement, traders, analysts) can build models
- Forecasts ANY time-series: equities, indices, commodities, currencies, revenue, sales, business KPIs
- Timeframes from 1-day to 5 years
- Cloud (managed) or on-premise deployment

### How It Works
1. **Data Lake**: 200,000+ global data sources (financial, macro, commodities, climate, shipping)
2. **AutoML Engine**: Automates feature selection, model training, hyperparameter tuning
3. **Adaptive AI**: Continuous retraining, challenger/champion ensemble approach
4. **Signals**: Models generate signals → transformed into strategies → backtested → evaluated
5. **Dashboard**: Results presented through charting and analytics for decision-making

### Key Technical Architecture
- Data Lake (Quantum DL) — pre-processed, validated, structured
- Automated ML Engine (Quantum ML) — the model factory
- UI/UX Platform — dashboards, visualizations, strategy simulator
- Kubernetes + NGINX containerization
- Multi-region support, parallel processing
- Supports proprietary + third-party data integration

### Killer Features for Nexus
- **Challenger/Champion framework**: Thousands of models compete, best ones selected for production
- **Adaptive algorithms**: Identify shifts in key drivers, adjust forecasts in real-time
- **Strategy simulator**: Test and refine models with real-time feedback
- **Transparent backtesting**: Performance metrics for direct model comparison

## CME Partnership

### Existing Relationship
- QDT launched **CME DataMine Machine Learning Service**
- "Machine learning-as-a-service" for traders and quants at major investment banks and commodity trading houses
- Cloud-based subscription service deployed via CME
- Integrates CME's unique historical data (DataMine)

### CME DataMine API
- Provides historical market data: Top-of-Book, Block Trades, EOD, Volume/OI, Market Depth, Time & Sales, MBO
- Exchanges: CME, CBOT, NYMEX, COMEX
- Data access via REST API, SFTP, S3 transfer
- Also includes cryptocurrency data (CF Bitcoin)
- Products identified by exchange code + product code + date

### What This Means for Nexus
Nexus is the **next evolution** of the dashboard layer. The CME ML Service already exists — Nexus is the sophisticated front-end that:
- Presents model outputs to different user personas
- Adds ensemble aggregation across thousands of QML models
- Provides institutional-grade drill-down and transparency
- Serves CME's customer base (banks, commodity houses, hedge funds)

## QDT Case Studies (Proof Points for CME Conversations)

| Client | Use Case | Result |
|--------|----------|--------|
| Leading steel manufacturer | Coking coal & steel price forecast | **9% savings ($42M)** in sourcing |
| Leading chocolate company | Cocoa price forecasts | **5-7% savings** in procurement |
| Multiple customers | Commodity forecasting (proteins, agri, hard/soft) | **7-9% avg procurement savings** |
| Leading CPG | Category & brand forecast | **95% accuracy** |
| Leading CPG | Media optimization | **15% uplift in sales** |
| Leading CPG | Pricing optimization | **5% sales increase** |
| Leading QSR | Marketing personalization | **$70M incremental sales (+8%)** |
| CME Group | DataMine ML Service | Deployed to investment banks & commodity houses |

## Implications for Nexus Design

1. **QML already handles the hard ML part** — Nexus focuses purely on presentation, aggregation, and user experience
2. **200K+ data sources** means we need excellent data provenance display
3. **Challenger/champion model framework** = natural visual story (model tournament, performance leaderboard)
4. **CME customer base is already established** — Nexus serves an existing audience, not cold leads
5. **Procurement/hedging is the money use case** — $42M savings at one steel company. Lead with this.
6. **Non-technical users are a key audience** — "procurement officers, buyers, traders, analysts" using no-code tools
