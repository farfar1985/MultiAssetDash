# QDT Data Lake Integration for Nexus

**Date:** 2026-02-07  
**Source:** QDT Retail Data Dictionary (83,254 fields)  
**Status:** Planning

---

## Executive Summary

The QDT Data Lake contains **83,254 data fields** across 33 providers and 24 categories. This document maps high-value fields to Nexus trading signals to enhance prediction accuracy.

---

## Current State

### Already Integrated (via QDL API)

| Provider | Assets | Status |
|----------|--------|--------|
| **DTNIQ** | 12 futures contracts | ✅ Live |

**Connected symbols:**
- `@ES#C` (S&P 500), `@NQ#C` (NASDAQ), `@YM#C` (Dow), `@RTY#C` (Russell)
- `QCL#` (WTI Crude), `QBZ#` (Brent), `QGC#` (Gold), `QHG#` (Copper), `QNG#` (NatGas)
- `@BTC#C` (Bitcoin), `@DX#C` (USD Index), `@NKD#C` (Nikkei)

### Not Yet Integrated

| Provider | Fields | Value Add |
|----------|--------|-----------|
| **EIA** | 6,531 | Oil inventories, production, refinery data |
| **COINMETRICS** | 2,613 | On-chain metrics, whale tracking |
| **FRED** | 30,729 | Macro indicators, rates, employment |
| **CBOE** | ~1,500 | VIX, options flow |
| **CME** | ~1,000 | COT positioning data |
| **PORTS** | 6,280 | Shipping, supply chain |

---

## Priority Integration Plan

### Phase 1: Energy Fundamentals (Week 1)

**Goal:** Add EIA data to Crude Oil predictions

| Symbol | Name | Frequency | Signal Use |
|--------|------|-----------|------------|
| `EIA_PET_WCRSTUS1_W` | Weekly crude stocks | Weekly | Inventory surprise |
| `EIA_PET_WCRFPUS2_W` | Refinery utilization | Weekly | Demand signal |
| `EIA_STEO_PAPR_WORLD` | World petroleum production | Monthly | Supply trend |
| `EIA_STEO_PAPC_OPEC` | OPEC production | Monthly | Supply constraint |
| `EIA_PET_RWTC_D` | WTI spot price | Daily | Price confirmation |

**Implementation:**
```python
# Add to qdl_client.py
EIA_SYMBOLS = [
    ("EIA_PET_WCRSTUS1_W", "EIA"),
    ("EIA_PET_WCRFPUS2_W", "EIA"),
    ("EIA_STEO_PAPR_WORLD", "EIA"),
]
```

### Phase 2: Crypto On-Chain (Week 2)

**Goal:** Add COINMETRICS to Bitcoin predictions

| Symbol | Name | Signal Use |
|--------|------|------------|
| `BTC_ADRACTCNT` | Active addresses | Network activity |
| `BTC_TXTFRVALADJUSD` | Transfer value (adjusted) | On-chain volume |
| `BTC_NVTADJ` | NVT ratio | Valuation signal |
| `BTC_HASHRATE` | Hash rate | Network security |
| `BTC_SPLYADRTOP100` | Supply in top 100 wallets | Whale concentration |
| `BTC_CAPMVRVCUR` | MVRV ratio | Market cycle indicator |
| `BTC_VTYDAYRET30D` | 30-day volatility | Risk signal |

**High-Value Insight:** MVRV > 3.5 historically signals market tops.

### Phase 3: Volatility Signals (Week 3)

**Goal:** Add CBOE volatility to equity predictions

| Symbol | Name | Signal Use |
|--------|------|------------|
| `CBOE_VIX` | VIX index | Fear gauge |
| `CBOE_VIX3M` | 3-month VIX | Term structure |
| `CBOE_VIX9D` | 9-day VIX | Short-term fear |
| `CBOE_VVIX` | VIX of VIX | Volatility regime |
| `CBOE_SKEW` | SKEW index | Tail risk |
| `CBOE_PUT_CALL` | Put/call ratio | Sentiment |

**Signal Logic:**
- VIX3M/VIX > 1.0 → Contango (bullish)
- VIX3M/VIX < 0.9 → Backwardation (bearish)

### Phase 4: COT Positioning (Week 4)

**Goal:** Add CME Commitment of Traders data

| Symbol | Name | Signal Use |
|--------|------|------------|
| `CME_COT_CL_COMM_LONG` | Crude commercials long | Smart money |
| `CME_COT_CL_SPEC_NET` | Crude speculators net | Crowding risk |
| `CME_COT_GC_COMM_LONG` | Gold commercials long | Hedger positioning |
| `CME_COT_ES_SPEC_NET` | S&P speculators net | Equity sentiment |

**Signal Logic:**
- Extreme speculator positioning → Mean reversion signal
- Commercial hedger extremes → Trend continuation

---

## Data Quality Considerations

### Frequency Alignment

| Nexus Frequency | Data Sources |
|-----------------|--------------|
| **Daily** | DTNIQ, COINMETRICS, CBOE |
| **Weekly** | EIA inventories, COT |
| **Monthly** | EIA production, FRED macro |

**Handling:** Forward-fill lower-frequency data for daily predictions.

### Lag Considerations

| Data Type | Typical Lag | Handling |
|-----------|-------------|----------|
| Futures prices | Real-time | None needed |
| EIA inventories | Wed 10:30 AM | Use T-1 for predictions |
| COT data | Tuesday (released Friday) | 3-day lag |
| On-chain metrics | ~24 hours | Use T-1 |

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NEXUS DATA PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    DTNIQ     │  │     EIA      │  │ COINMETRICS  │          │
│  │  (Futures)   │  │   (Energy)   │  │  (On-chain)  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────────┬────┴────────────────┘                   │
│                      ▼                                          │
│         ┌────────────────────────┐                              │
│         │     QDL API Client     │                              │
│         │    (qdl_client.py)     │                              │
│         └────────────┬───────────┘                              │
│                      ▼                                          │
│         ┌────────────────────────┐                              │
│         │   Feature Engineering  │                              │
│         │  - Normalize           │                              │
│         │  - Lag alignment       │                              │
│         │  - Rolling stats       │                              │
│         └────────────┬───────────┘                              │
│                      ▼                                          │
│         ┌────────────────────────┐                              │
│         │   Pairwise Slopes      │                              │
│         │   Ensemble Engine      │                              │
│         └────────────┬───────────┘                              │
│                      ▼                                          │
│         ┌────────────────────────┐                              │
│         │   Regime Detection     │                              │
│         │  (quantum_regime_qdl)  │                              │
│         └────────────────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints to Add

```python
# Extended QDL client methods

class QDLClient:
    def get_eia_inventories(self, weeks: int = 52) -> pd.DataFrame:
        """Fetch EIA weekly petroleum inventories"""
        
    def get_coinmetrics(self, asset: str, metrics: list) -> pd.DataFrame:
        """Fetch on-chain metrics for crypto asset"""
        
    def get_cboe_volatility(self) -> pd.DataFrame:
        """Fetch VIX term structure"""
        
    def get_cot_positioning(self, asset: str) -> pd.DataFrame:
        """Fetch COT data for futures asset"""
```

---

## Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Crude Oil Sharpe | 1.76 | 2.0+ |
| Bitcoin Win Rate | 67% | 72%+ |
| S&P 500 Accuracy | 65% | 70%+ |
| Data freshness | Daily | Intraday for prices |

---

## Files Created

| File | Location | Purpose |
|------|----------|---------|
| Data dictionary | `clawd/data/qdt_retail_data_dictionary.csv` | Full 83K fields |
| Stats JSON | `clawd/data/qdt_data_stats.json` | Aggregated stats |
| Nexus crossref | `clawd/data/nexus_data_crossref.json` | Asset mapping |
| Provider details | `clawd/data/qdt_providers_detail.json` | Provider breakdown |
| Category details | `clawd/data/qdt_categories_detail.json` | Category breakdown |
| Data catalog | `clawd/docs/QDT_RETAIL_DATA_CATALOG.md` | Human-readable catalog |
| This doc | `nexus/docs/QDT_DATA_INTEGRATION.md` | Integration plan |

---

## Next Steps

1. [ ] Verify QDL API access for EIA, COINMETRICS providers
2. [ ] Build `qdl_extended_client.py` with new data sources
3. [ ] Add EIA inventories to Crude Oil feature set
4. [ ] Add on-chain metrics to Bitcoin predictions
5. [ ] Backtest impact on Sharpe ratio
6. [ ] Update dashboard with new data sources

---

*Generated from QDT Retail Data Dictionary analysis.*
