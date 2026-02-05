# Quantum Regime Detector - Real Data Integration Plan

**Created:** 2026-02-05
**Author:** AmiraB
**Status:** Research Phase

## Executive Summary

Currently the 6-qubit enhanced quantum regime detector uses **price-derived proxies** for external data. Bill has directed research into adding **real data sources** to improve regime detection accuracy.

## Current State (Proxies)

| Qubit | Current Proxy | Limitation |
|-------|---------------|------------|
| Q2: Implied Vol | Rolling realized vol + trend | No forward-looking info |
| Q3: Positioning | Momentum as sentiment proxy | Doesn't capture COT data |
| Q4: Macro | Rolling correlation proxy | Not actual economic indicators |
| Q5: Contagion | Random placeholder | Should be real cross-asset correlation |

## Available Real Data Sources in quantum_ml

### Priority 1: Implied Volatility (Q2)

**Source:** `ingest_cme_cvol.py`

| Symbol | Asset | Update Frequency |
|--------|-------|------------------|
| CLVL | WTI Crude Oil CVOL | Real-time |
| NGVL | Natural Gas CVOL | Real-time |
| GCVL | Gold CVOL | Real-time |
| SIVL | Silver CVOL | Real-time |
| EVL | Energy Volatility Index | Real-time |

**Also available:** `ingest_cboe_indices.py`
- VIX (equity vol)
- VVIX (vol of vol)
- Sector-specific VIX

**Integration:** Replace `calculate_implied_vol_proxy()` with real CVOL lookup.

### Priority 2: Trader Positioning (Q3)

**Source:** `ingest_cftc_cot.py`

Weekly Commitments of Traders data with:
- Producer/Merchant positions (commercials)
- Asset Manager positions (institutional)
- Leveraged Funds positions (hedge funds/CTAs)
- Net positioning changes

**Key Metrics:**
- Net speculative positioning
- Commercial vs Non-commercial ratio
- Week-over-week position changes
- Extreme positioning alerts (>2σ from mean)

**Coverage:** 60+ contracts including:
- COT_CL (Crude Oil WTI)
- COT_NG (Natural Gas)
- COT_GC (Gold)
- COT_ES (E-mini S&P 500)
- COT_BTC (Bitcoin CME)

**Integration:** Replace `calculate_positioning_proxy()` with COT net positioning.

### Priority 3: Macro Indicators (Q4)

**Source:** `ingest_fred.py`

Federal Reserve Economic Data:
- GDP growth rate
- Unemployment rate
- Inflation (CPI, PCE)
- Interest rates (Fed Funds, 10Y yield)
- Credit spreads (BAA-AAA)
- M2 money supply

**Recommended Indicators for Regime Detection:**
| Indicator | FRED Symbol | Update | Use Case |
|-----------|-------------|--------|----------|
| 10Y-2Y Spread | T10Y2Y | Daily | Recession signal |
| High Yield Spread | BAMLH0A0HYM2 | Daily | Risk appetite |
| VIX | VIXCLS | Daily | Fear gauge |
| Dollar Index | DTWEXBGS | Daily | USD strength |

**Integration:** Replace `calculate_macro_proxy()` with composite FRED indicator.

### Priority 4: News Sentiment (Q3/Q4)

**Source:** `ingest_gdelt.py`

Global Database of Events, Language, and Tone:
- Event counts by category
- Average tone (sentiment) per asset
- Conflict/cooperation scores
- Global stability metrics

**Challenges:**
- High volume, noisy data
- Requires NLP preprocessing
- Asset-specific filtering needed

**Integration:** Blend with positioning in Q3 or add to macro in Q4.

### Priority 5: Energy-Specific (Q5 for Energy Assets)

**Source:** `ingest_eia.py`, `ingest_eia_wpsr.py`

Weekly Petroleum Status Report:
- Crude oil inventory levels
- Refinery utilization
- Cushing, OK storage levels
- Import/export flows

**Integration:** Enhance contagion qubit for energy assets with inventory surprise factor.

## Implementation Phases

### Phase 1: CVOL Integration (1-2 days)
1. Create `RealDataManager` class extending `DataSourceManager`
2. Add Cassandra connection to fetch CVOL time series
3. Replace `calculate_implied_vol_proxy()` with CVOL lookup
4. Backtest: Does real implied vol improve regime detection?

### Phase 2: COT Integration (2-3 days)
1. Parse COT data structure (weekly to daily interpolation)
2. Calculate net positioning metrics
3. Replace `calculate_positioning_proxy()` with COT data
4. Add extreme positioning alerts

### Phase 3: FRED Integration (1-2 days)
1. Select 3-5 key macro indicators
2. Create composite macro score
3. Replace `calculate_macro_proxy()` with FRED composite
4. Test on different market regimes

### Phase 4: Cross-Asset Contagion (2-3 days)
1. Calculate rolling correlations between assets
2. Use Artemis's dashboards to visualize contagion
3. Replace placeholder contagion with real correlations
4. Test on 2008, 2020 crisis periods (if historical data available)

## Per-Asset Data Mapping

| Asset | CVOL Symbol | COT Symbol | Special Data |
|-------|-------------|------------|--------------|
| Crude Oil | CLVL | COT_CL | EIA inventory |
| Natural Gas | NGVL | COT_NG | EIA storage |
| Gold | GCVL | COT_GC | USD index |
| Bitcoin | (Deribit DVOL) | COT_BTC | On-chain metrics |
| S&P 500 | VIX | COT_ES | Put/Call ratio |
| NASDAQ | VXN | (via ES) | Tech sector VIX |

## Expected Improvements

| Metric | Current (Proxy) | Expected (Real Data) |
|--------|-----------------|---------------------|
| Regime Accuracy | ~70% | 80-85% |
| Early Warning Lead | 1-2 days | 3-5 days |
| False Crisis Alerts | ~15% | <8% |
| Entropy Stability | Variable | More consistent |

## Open Questions for Bill

1. **Data access:** Do we have Cassandra read access to these ingestion tables?
2. **Historical depth:** How far back does CVOL/COT data go in Cassandra?
3. **Real-time vs EOD:** Do we need streaming data or is daily sufficient?
4. **API architecture:** Should real data come from Nexus API or direct Cassandra?
5. **Priority order:** Which data source first - CVOL or COT?

## Next Steps

1. Confirm Cassandra access to ingest_* tables
2. Start with CVOL integration (most impactful for volatility regimes)
3. Build evaluation framework to measure improvement
4. Coordinate with Artemis on dashboard updates for new data

---

*This plan will evolve as we prototype each integration.*
