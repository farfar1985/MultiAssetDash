# QDT.Ensemble - Multi-Asset Forecasting Dashboard

<div align="center">

![QDT Logo](QDT%20LOGO/QDT%20logo.jpg)

**Advanced Multi-Asset Forecasting & Trading Intelligence Platform**

*Quantum Data Technologies*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()
[![Assets](https://img.shields.io/badge/Assets-15+-green.svg)]()

</div>

---

## 🎯 Project Objective

QDT.Ensemble is a **professional-grade trading intelligence platform** that combines machine learning forecasts from multiple time horizons (D+1 to D+10) into a unified ensemble signal. The system provides:

1. **Actionable Trading Signals** - BULLISH, BEARISH, or NEUTRAL signals with confidence levels
2. **Dynamic Accuracy Tracking** - Real-time historical accuracy that updates with filter changes
3. **Professional Risk Management** - ATR-based stop-losses and multiple exit strategies
4. **Comprehensive Quant Analytics** - 30+ professional trading statistics
5. **Marketing Automation** - Social media content and email alert generation

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git (to clone the repository)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/farfar1985/MultiAssetDash.git
cd MultiAssetDash
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Dashboard

**Option 1: Use the Pre-built Dashboard (Quick Start) - NO SETUP NEEDED!**
```bash
# Simply open the HTML file in your browser
# The dashboard file is: QDT_Ensemble_Dashboard.html
```
Open `QDT_Ensemble_Dashboard.html` in any modern web browser. **All data is embedded in the file**, so no server or API is needed. This works immediately after cloning the repo!

**Option 2: Rebuild the Dashboard (After Data Updates)**
```bash
# First, set up API credentials (if you have access):
# Create a .env file in the parent directory with:
# QML_API_KEY=your_api_key_here

# Then update forecasts for all assets and rebuild dashboard:
python run_complete_pipeline.py

# Or update a specific asset:
python run_complete_pipeline.py --asset Crude_Oil
```

**Note:** To rebuild the dashboard, you need:
- API access (QML_API_KEY in `.env` file in parent directory)
- Or use the pre-built `QDT_Ensemble_Dashboard.html` which already contains all current data

The pipeline will:
1. Fetch latest forecast data from the API
2. Calculate signals and metrics
3. Generate `QDT_Ensemble_Dashboard.html`

### Running the API Server

```bash
# Start the API server
python api_server.py

# Server runs on http://localhost:5000 by default
# See API_README.md for full API documentation
```

### Managing API Keys

```bash
# Add a new API key
python manage_api_keys.py --add --user-id "user123" --assets Crude_Oil Bitcoin

# List all API keys
python manage_api_keys.py --list

# Remove an API key
python manage_api_keys.py --remove <api_key>
```

---

## 📊 Dashboard Overview

The dashboard consists of **two main tabs**:

| Tab | Purpose |
|-----|---------|
| **📈 Dashboard** | Live signals, charts, analytics, trade history |
| **📣 Marketing Hub** | Social media content, email templates, audience management |

---

# 📈 TAB 1: MAIN DASHBOARD

## 1. Header Section

### Overall Health Score (0-100)
A composite score indicating the quality of the ensemble for the selected asset:
- **80+** = Excellent ensemble health
- **60-79** = Good, minor improvements possible
- **40-59** = Fair, review recommendations
- **Below 40** = Poor, needs attention

**Components:**
- Number of active timeframes
- Model coverage per horizon
- Historical accuracy
- Signal edge (accuracy - 50%)

### Optimal Configuration Panel
Shows the **recommended horizon selection** based on backtesting:
- Best horizons to enable (e.g., D+1, D+4, D+5, D+8, D+10)
- Average accuracy for this configuration
- Score out of 100

### Auto Optimize Equity Button
One-click optimization that:
- Analyzes all possible horizon combinations
- Selects the configuration with highest edge
- Updates all statistics dynamically

---

## 2. Recommendations Panel

### Timeframe Coverage
- ✅ **Great Coverage** - All 10 forecast periods covered
- ⚠️ **Partial Coverage** - Some horizons missing

### Model Count by Horizon
Interactive horizon toggles showing:
- **D+1, D+2, ... D+10** - Click to enable/disable
- Model count per horizon (✓ = 10+ models, ~ = 2-9 models, ! = 1 model)
- Weak horizons auto-disabled by default

---

## 3. Live Signal Panel

### Current Signal Display
Large visual indicator showing:
- **↑ BULLISH** (Green) - Model predicts price increase
- **↓ BEARISH** (Red) - Model predicts price decrease  
- **↔ NEUTRAL** (Gray) - No clear direction

### Confidence Tier
- **HIGH CONFIDENCE** - Strong signal (>50% net probability)
- **MEDIUM CONFIDENCE** - Moderate signal (20-50% net probability)
- **LOW CONFIDENCE** - Weak signal (<20% net probability)

### Dynamic Accuracy Display
**Now updates in real-time** based on:
- RSI filter status
- EMA filter status
- Ichimoku filter status
- Enabled horizons

Shows: `XX.X% T1 HIT RATE (LAST 20 TRADES)`

### Signal Details
- **Directional Consensus** - % of horizons agreeing
- **Horizons Agreeing** - X/Y horizons
- **Based On** - Number of similar historical signals

---

## 4. Technical Indicator Filters

### RSI Filter (Toggle)
- **ON** - Filters out overbought bullish signals and oversold bearish signals
- Uses asset-specific overbought/oversold levels
- Improves signal accuracy by removing exhaustion trades

### EMA Filter (Dropdown: OFF, 9, 20, 50, 100, 200)
- **ON** - Only allows bullish signals when price > EMA
- Only allows bearish signals when price < EMA
- Adds trend confirmation to signals

### Ichimoku Cloud Filter (Toggle)
- **ON** - Displays Tenkan, Kijun, and Senkou spans
- Filters signals: Bullish only in green cloud, bearish only in red cloud
- Professional-grade trend confirmation

**All filters dynamically update:**
- Snake chart colors
- Accuracy percentage
- Trade History results

---

## 5. Snake Chart

### Visual Price Chart with Ensemble Signals
- **Green segments** - Bullish signal periods
- **Red segments** - Bearish signal periods
- **Gray segments** - Neutral or filtered-out signals
- **Dashed line** - Future forecast projections

### Chart Features
- Interactive Plotly chart
- Zoom, pan, hover for details
- Price overlays (EMA, Ichimoku when enabled)
- Volume information

---

## 6. Price Targets Section

### Dynamic Price Targets
Three target levels based on active horizon forecasts:

| Target | Description | Typical Distance |
|--------|-------------|------------------|
| **T1** | Conservative | ~0.5% from entry |
| **T2** | Base | ~1.0% from entry |
| **T3** | Extended | ~1.5% from entry |

**Features:**
- Prices calculated from ML forecasts
- Filtered by active horizons only
- Updates when horizons toggled
- Shows % move from current price

---

## 7. Trade History Modal

Click **"Trade History"** button to open comprehensive analysis:

### 7.1 Total P&L Summary
- **Total Return %** - Sum of all trade P&L
- **Avg per trade** - Average return per signal
- **$10K Account** - What $10K would become
- **Monthly/Annual** - Projected returns

### 7.2 Target Hit Rates
Shows historical performance of targets:
- **T1 Hit Rate** - % of trades hitting ±0.5%
- **T2 Hit Rate** - % of trades hitting ±1.0%
- **T3 Hit Rate** - % of trades hitting ±1.5%
- **SL Rate** - % stopped out
- **Avg days** to hit each target

### 7.3 Exit Strategy Recommendation
Analyzes 4 exit strategies and recommends the best:

| Strategy | Description | Best When |
|----------|-------------|-----------|
| **Scale Out (⅓ each)** | Exit ⅓ at T1, T2, T3 | Balanced approach |
| **Conservative (All T1)** | Exit 100% at T1 | High T1 hit rate |
| **Aggressive (All T3)** | Hold 100% until T3 | Strong T3 hit rate |
| **Hybrid (50/50)** | Exit 50% at T1, 50% at T3 | Mixed conditions |

**Clickable strategy cards** - Click any to see projections!

### 7.4 ATR-Based Stop-Loss Analysis
Professional volatility-adjusted stop-losses:

| Level | Formula | Risk Profile |
|-------|---------|--------------|
| **S1 Tight** | 1× ATR | Aggressive |
| **S2 Medium** | 1.5× ATR | Balanced |
| **S3 Wide** | 2× ATR | Conservative |

Shows for each:
- Actual SL percentage (dynamic per asset)
- Win rate with this SL
- Stop-out count
- Total P&L
- **Highlights OPTIMAL** stop-loss

### 7.5 Trade History Table
Detailed list of last 20 trades:
- Entry date
- Signal direction
- Entry price
- Targets hit (T1 ✓, T2 ✓, T3 ✓, SL ✗)
- Best target achieved
- P&L per trade

---

## 8. Equity Curve

### Visual Performance Chart
- **Blue line** - QDT Ensemble equity
- **Gray line** - Buy & Hold baseline
- **Dashed line** - Starting capital ($100)

### Outperformance Display
Shows: `+XX.X%` vs Buy & Hold strategy

---

## 9. Trading Performance Section

### Simple Mode (Default)
Key metrics at a glance:
- Total Return, Annualized
- Sharpe Ratio, Profit Factor
- Win Rate, Max Drawdown
- Avg Win/Loss, Best/Worst Day
- vs Buy & Hold comparison

### 🔬 Quant Mode (Toggle ON)
**30+ Professional Statistics** organized into 6 categories:

#### Category 1: Risk-Adjusted Returns
| Metric | Description | Good Value |
|--------|-------------|------------|
| Sharpe Ratio | Return per unit of risk | >1.0 |
| Sortino Ratio | Return per unit of downside risk | >1.5 |
| Calmar Ratio | Annual return / Max drawdown | >1.0 |
| Omega Ratio | Probability-weighted gains/losses | >1.5 |

#### Category 2: Drawdown Analysis
| Metric | Description | Good Value |
|--------|-------------|------------|
| Max Drawdown | Worst peak-to-trough decline | <20% |
| Avg Drawdown | Mean of all drawdowns | <10% |
| Max DD Duration | Days to recover | <60 days |
| Ulcer Index | Pain index (depth × duration) | <5 |

#### Category 3: Trade-Level Metrics
| Metric | Description | Good Value |
|--------|-------------|------------|
| Win Rate | % of winning trades | >50% |
| Profit Factor | Gross wins / Gross losses | >1.5 |
| Payoff Ratio | Avg win / Avg loss | >1.0 |
| Expectancy | Expected $ per trade | >0 |
| Avg Win/Loss | Size of wins vs losses | - |
| Total Trades | Number of signals | - |

#### Category 4: Distribution & Tail Risk
| Metric | Description | Good Value |
|--------|-------------|------------|
| Skewness | Return asymmetry | >0 (positive) |
| Excess Kurtosis | Fat tails | <3 |
| VaR (95%) | Max loss at 95% confidence | - |
| CVaR / ES | Average loss beyond VaR | - |
| Tail Ratio | 95th percentile / 5th percentile | >1 |
| Daily Volatility | Standard deviation of returns | - |

#### Category 5: Consistency Metrics
| Metric | Description | Good Value |
|--------|-------------|------------|
| % Positive Days | Win consistency | >50% |
| Gain-to-Pain Ratio | Sum returns / Sum losses | >1 |
| Recovery Factor | Net profit / Max DD | >2 |
| Kelly % | Optimal position size | 10-25% |
| Max Win/Loss Streak | Consecutive wins/losses | - |
| Sterling Ratio | CAGR / Avg drawdown | >1 |
| Common Sense Ratio | Tail ratio × Profit factor | >2 |

#### Category 6: Performance Summary
- Total Return, Annualized Return
- vs Buy & Hold comparison
- Trading period duration

---

# 📣 TAB 2: MARKETING HUB

## Purpose
Automated content generation for social media and email marketing.

## Features

### 1. Social Media Content Generator
- **Twitter/X Posts** - Character-limited signal summaries
- **LinkedIn Posts** - Professional market analysis
- **Instagram Captions** - Visual-friendly summaries

### 2. Email Templates
- **Daily Signal Alert** - Morning signal notification
- **Weekly Roundup** - Performance summary
- **New Signal Alert** - Real-time notifications

### 3. Content Preview
Live preview of generated content with copy buttons.

### 4. Asset Selection
Generate content for any of the 15+ supported assets.

---

# 🛠️ Technical Architecture

## Data Flow
```
API (Investing.com) → run_sandbox_pipeline.py → JSON Cache
                                ↓
                    grid_search_tuning.py (Optimization)
                                ↓
                    build_qdt_dashboard.py → HTML Dashboard
```

## Key Files

| File | Purpose |
|------|---------|
| `build_qdt_dashboard.py` | Main dashboard generator (7500+ lines) |
| `run_sandbox_pipeline.py` | Fetches forecasts from API |
| `grid_search_tuning.py` | Finds optimal horizon configurations |
| `run_complete_update.py` | Daily update automation |
| `golden_engine.py` | Core ensemble signal logic |

## Data Structure (per asset)
```
data/
├── {asset_id}_{asset_name}/
│   ├── price_history_cache.json    # Historical prices
│   ├── forecast_cache.json         # ML forecasts per horizon
│   ├── optimized_forecast.json     # Optimal configuration
│   ├── confidence_stats.json       # Historical accuracy stats
│   └── live_forecast.json          # Latest forecast
```

---

# 📊 Supported Assets (15+)

| Asset | Category | Project ID |
|-------|----------|------------|
| Crude Oil (WTI) | Commodities | 1866 |
| Brent Oil | Commodities | 1859 |
| Gold | Precious Metals | 477 |
| MCX Copper | Metals | 1435 |
| Bitcoin | Crypto | 1860 |
| S&P 500 | US Indices | 1625 |
| NASDAQ | US Indices | 269 |
| Dow Jones | US Indices | 336 |
| Russell 2000 | US Indices | 1518 |
| Nikkei 225 | Asian Indices | 358 |
| Nifty 50 | Asian Indices | 1398 |
| Nifty Bank | Asian Indices | 1387 |
| US Dollar Index | Forex | 655 |
| USD/INR | Forex | 256 |
| SPDR China ETF | ETFs | 291 |

---

# 🚀 Deployment

## VPS Setup (Vultr)
```bash
# Upload dashboard
scp QDT_Ensemble_Dashboard.html root@YOUR_IP:/var/www/html/QDT.Ensemble/

# Nginx configuration with no-cache headers
# Cloudflare cache purging after updates
```

## Daily Update Automation

### Setup Cron Job

1. **Make the cron script executable:**
```bash
chmod +x q_ensemble_sandbox/daily_update_cron.sh
```

2. **Edit crontab:**
```bash
crontab -e
```

3. **Add this line (runs daily at 6 AM):**
```bash
# QDT Ensemble Dashboard - Daily update at 6 AM
0 6 * * * cd /path/to/q_ensemble_sandbox && ./daily_update_cron.sh
```

**Or if using full path:**
```bash
0 6 * * * /path/to/q_ensemble_sandbox/daily_update_cron.sh
```

4. **Verify cron job:**
```bash
crontab -l
```

5. **Test the script manually:**
```bash
cd /path/to/q_ensemble_sandbox
./daily_update_cron.sh
```

6. **Check logs:**
```bash
ls -la q_ensemble_sandbox/logs/
cat q_ensemble_sandbox/logs/daily_update_*.log
```

### What the Cron Script Does

1. Runs `run_complete_pipeline.py` (updates all 15 assets)
2. Rebuilds `QDT_Ensemble_Dashboard.html`
3. Copies dashboard to web server directory (if configured)
4. Logs everything to `logs/daily_update_YYYYMMDD_HHMMSS.log`

### Server Configuration

Update the `WEB_DIR` variable in `daily_update_cron.sh` to match your server's web directory:
```bash
WEB_DIR="/var/www/html"  # Change to your web server path
```

---

# 📈 Signal Calculation Logic

## Ensemble Signal Generation
1. Fetch forecasts for all 10 horizons (D+1 to D+10)
2. Calculate net probability: `mean(bullish) - mean(bearish)`
3. Apply threshold: `|net_prob| > 0.02` for signal
4. Direction: `net_prob > 0` = BULLISH, `net_prob < 0` = BEARISH

## Dynamic Accuracy
Calculated from filtered signals using last 20 trades:
```javascript
accuracy = (T1_hits / total_trades) × 100
```

Where T1 hit = price moved 0.5% in signal direction within 15 days.

---

# 🔧 Configuration

## Asset-Specific Parameters
```python
ASSETS = {
    "Crude_Oil": {
        "id": "1866",
        "threshold": 0.25,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
    },
}
```

## Indicator Settings
- **RSI Period**: 14
- **EMA Options**: 9, 20, 50, 100, 200
- **Ichimoku**: Tenkan(9), Kijun(26), Senkou(52)
- **ATR Period**: 14

---

# 📝 Recent Updates (January 2026)

- ✅ Dynamic accuracy that updates with filters
- ✅ ATR-based stop-loss analysis (professional)
- ✅ 4 exit strategy recommendations per asset
- ✅ Clickable strategy comparison
- ✅ 30+ Quant Mode statistics
- ✅ Price targets from ML forecasts
- ✅ Trade history with target tracking
- ✅ Ichimoku cloud filtering

---

# 📄 License

Proprietary - Quantum Data Technologies

---

<div align="center">

**Built with ❤️ by QDT**

*Empowering traders with AI-driven market intelligence*

</div>
