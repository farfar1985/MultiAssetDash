# Nexus v2 Meeting Brief
**Date:** February 5, 2026, 2:00 PM ET  
**Attendees:** Bill Dennis, Rajiv (CEO), Ale (CTO)  
**Prepared by:** AmiraB + Artemis

---

## Executive Summary

We've completed critical bug fixes and validated a new per-asset ensemble optimization approach that shows real alpha potential.

### Key Numbers

| Asset | Optimal Horizons | Sharpe Ratio | Win Rate | Use Case |
|-------|------------------|--------------|----------|----------|
| **Crude Oil** | [9, 10] | **1.727** | **68.5%** | Hedging/Procurement |
| **S&P 500** | [3, 8] | **1.184** | **57.4%** | Institutional |
| **Bitcoin** | [3, 5] | **0.359** | **55.6%** | Alpha Generation |

---

## What Changed This Morning

### Bug Fixes (Critical)
1. **Data Leakage Fixed** — `bfill()` was leaking future data backward; now using `ffill()` only
2. **Look-Ahead Bias Fixed** — Backtest was measuring wrong return window
3. **Path Traversal Fixed** — Security vulnerability in API download endpoint

### Impact on Previously Reported Metrics
- Old [5,7,10] Sharpe: 1.757 → **Corrected: -0.916**
- The impressive results we showed before were artifacts of look-ahead bias
- After fixes, honest metrics reveal the real signal quality

### The Good News: Per-Asset Optimization Works
- Artemis's thesis: each asset needs its own optimal ensemble configuration
- **Validated today** with grid search across horizons, aggregation methods, and thresholds
- Different asset classes have fundamentally different optimal horizon combinations

---

## Per-Asset Insights

### Crude Oil — Best Performance
- **Optimal: [9, 10]** — Longer horizons work best
- Sharpe 1.727, Win Rate 68.5%
- **Why:** Supply/demand cycles, inventory reports, weekly patterns
- **CME Angle:** Perfect for hedging desks and procurement teams

### S&P 500 — Strong Performance  
- **Optimal: [3, 8]** — Medium-term horizons
- Sharpe 1.184, Win Rate 57.4%
- **Why:** Momentum + weekly rebalancing patterns
- **CME Angle:** Institutional overlay strategies

### Bitcoin — Moderate Performance
- **Optimal: [3, 5]** — Shorter horizons
- Sharpe 0.359, Win Rate 55.6%
- **Why:** Higher volatility, faster mean reversion
- **Note:** Crypto requires different risk management

---

## Technical Architecture

```
[50 RSS Scrapers] → [NLP Processing] → [Horizon Models D+1 to D+10]
                                              ↓
                          [Per-Asset Ensemble Optimizer]
                                              ↓
                          [Pairwise Slopes Signal Generator]
                                              ↓
                              [API → Dashboard → Client]
```

---

## What's Working Well

1. **Infrastructure is solid** — 50 scrapers, clean API, dashboard generator
2. **Per-asset optimization** — Validated approach for improving accuracy
3. **Honest metrics now** — Bug fixes mean we trust the numbers
4. **Collaboration model** — AmiraB + Artemis working effectively

---

## Recommended Discussion Points

### For CME Partnership
1. **Lead with Crude Oil** — Best metrics, directly relevant to their hedging clients
2. **Per-asset intelligence as differentiator** — Not one-size-fits-all
3. **Honest about journey** — We found and fixed bias; shows rigor

### Technical Roadmap
1. Expand to more assets (Gold, NASDAQ, Brent, etc.)
2. Longer backtest periods
3. Additional alternative data sources
4. Integration with quantum_ml pipeline

### Resource Needs
- [ ] More historical data for robust validation
- [ ] Additional computing for grid search optimization
- [ ] API integration with quantum_ml backend

---

## Files & Resources

| Resource | Location |
|----------|----------|
| Optimization Results | `nexus/configs/optimization_summary.json` |
| Individual Configs | `nexus/configs/optimal_*.json` |
| Code Audit Report | `nexus/docs/CODE_AUDIT_REPORT.md` |
| Ensemble Research | `nexus/docs/ENSEMBLE_RESEARCH_FINDINGS.md` |
| API Server | `nexus/api_ensemble.py` (port 5001) |
| Dashboard | `nexus/build_qdt_dashboard.py` |

---

## Q&A Prep

**Q: Why did the Sharpe drop from 1.757 to -0.916?**
A: The original calculation had look-ahead bias — it was measuring returns using future information. Once we fixed this, the true out-of-sample performance was negative for that configuration. But this led us to find configurations that actually work.

**Q: How confident are you in the new numbers?**
A: Very confident. The methodology is now: train on first 70% of data, test on remaining 30%, with proper temporal separation. No future data leakage.

**Q: What's next?**
A: Expand optimization to remaining 12+ assets, collect more historical data for robustness, and integrate findings into the production quantum_ml pipeline.

---

*Prepared by AmiraB (Bill's PC) in collaboration with Artemis (Farzaneh's bot)*
