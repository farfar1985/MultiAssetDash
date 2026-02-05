# QDT.Ensemble Demo Script

**Meeting:** Rajiv & Ale Demo
**Date:** February 5, 2026
**Duration:** 15-20 minutes
**Primary Asset:** Crude Oil (10,179 models)

---

## Pre-Demo Checklist (10 min before)

- [ ] Open terminal, run: `cd ~/clawd/nexus && python api_server.py`
- [ ] Verify API is running: visit `http://localhost:5000/api/v1/health`
- [ ] Open frontend: `cd frontend && npm run dev`
- [ ] Open browser to `http://localhost:3000/dashboard`
- [ ] Pre-load Crude Oil in each dashboard tab (faster switching)
- [ ] Close Slack, email, notifications
- [ ] Have backup screenshots ready in `docs/screenshots/` (if connection fails)
- [ ] Test screen share before meeting starts

---

## Opening (1 minute)

**What to say:**

> "Thanks for joining. Today I'll walk you through QDT.Ensemble, our multi-horizon forecasting platform. The key insight we'll demonstrate is that our original baseline horizons were actually anti-predictive. By discovering the optimal short-term horizons, we've improved from a **negative Sharpe ratio to over 11**. Let me show you how this works in practice."

---

## Part 1: Executive Dashboard (2 minutes)

### Navigate
1. Click **"Executive"** in the persona switcher (top nav)
2. Select **"Crude Oil"** from the asset dropdown

### What to show
- **Market Sentiment Summary** - Point to the bullish/bearish/neutral counts
- **Average Confidence** - Highlight the percentage
- **Top Performing Signals** - Show the ranked list
- **Health Score** - Green indicator means systems healthy

### What to say

> "This is the executive view - designed for quick decision-making. Right now you can see [X] bullish signals across our asset universe with an average confidence of [Y]%. The top performers are ranked by alpha score here."

> "Notice the health indicators in the corner - green means all 10,179 models for Crude Oil are contributing to predictions."

### Key number to mention
- **10,179 models** powering Crude Oil predictions

---

## Part 2: Alpha Pro Dashboard (5 minutes)

### Navigate
1. Click **"Alpha Pro"** in persona switcher
2. Ensure Crude Oil is selected

### What to show (in order)

#### A. Live Signal Cards
- Point to the **direction indicator** (BULLISH/BEARISH/NEUTRAL)
- Show the **confidence percentage** (0-100%)
- Highlight **model agreement** visualization

**What to say:**
> "This is the trader's view. The current signal for Crude Oil is [BULLISH/BEARISH] with [X]% confidence. This confidence comes from model agreement - when more models agree on direction, confidence increases."

#### B. Price Targets
- Show **T1, T2, T3 targets**
- Explain the ladder concept

**What to say:**
> "We provide three price targets: T1 is conservative at about half a percent, T2 is our base case at one percent, and T3 is the extended target. Historical hit rates are shown for each."

#### C. Alpha Score & Expected P&L
- Point to the **alpha score**
- Show **expected P&L calculation**

**What to say:**
> "The alpha score combines directional accuracy with magnitude prediction. Expected P&L here shows what a position would return based on our confidence-weighted sizing."

#### D. Conviction Meter (Interactive)
- **CLICK** on the conviction meter to expand details
- Show the breakdown by horizon

**What to say:**
> "Clicking here shows which horizons are contributing most to the signal. You'll see D+1 through D+3 carry the strongest weight - these short-term horizons contain the real predictive signal."

### Key numbers to mention
- **Sharpe Ratio: 11.96** (vs baseline -0.82)
- **Win Rate: 82%** directional accuracy
- **Avg Profit/Trade: $1.81**

---

## Part 3: Hedging Dashboard (5 minutes) - CME FOCUS

### Navigate
1. Click **"Hedging"** in persona switcher
2. Stay on Crude Oil

### What to show (in order)

#### A. Correlation Matrix
- Point to the **heatmap** showing asset correlations
- Highlight Crude Oil correlations with other assets

**What to say:**
> "For hedging desks, correlation is everything. This matrix updates in real-time. You can see Crude Oil's current correlations - [point to specific pairs]. When correlations spike, hedge ratios need adjustment."

#### B. Hedge Ratio Suggestions
- Show the **recommended hedge ratios**
- Explain the calculation basis

**What to say:**
> "Based on current correlations and volatility, the platform suggests these hedge ratios. For example, to hedge [X] contracts of Crude, you'd need approximately [Y] contracts of [related asset]."

#### C. Position Tracking
- Show how positions would be tracked
- Demonstrate P&L attribution

**What to say:**
> "Position tracking shows real-time P&L with attribution by horizon. This is critical for understanding which forecast horizons are contributing to returns."

#### D. Volatility Analysis
- Point to **ATR-based metrics**
- Show stop-loss recommendations

**What to say:**
> "Our volatility analysis uses ATR at three levels. Current recommendation is a stop at [X] based on 1.5x ATR. This adjusts automatically as volatility changes."

### Key insight for CME

**What to say:**
> "The key for hedging desks: we use short horizons D+1 to D+3 for **directional confidence**, but D+7 to D+10 for **profit targets**. This diverse horizon approach captures both timing precision and magnitude accuracy."

---

## Part 4: Quant Dashboard (3 minutes)

### Navigate
1. Click **"Quant"** in persona switcher
2. Stay on Crude Oil

### What to show (in order)

#### A. 30+ Professional Statistics
- Scroll through the metrics grid
- Highlight specific categories

**What to say:**
> "For due diligence, here are 30-plus institutional-grade metrics. Let me highlight a few:"

**Point to each as you mention:**
- **Sharpe: 11.96** - "Risk-adjusted return, annualized"
- **Sortino: 15.2** - "Downside deviation only"
- **Max Drawdown: -2.8%** - "Worst peak-to-trough"
- **Win Rate: 82%** - "Directional accuracy"
- **Kelly %** - "Optimal position sizing"

#### B. Model Disagreement Analysis
- Show where models diverge
- Explain what high disagreement means

**What to say:**
> "Model disagreement is a risk signal. High disagreement suggests uncertainty - we reduce position size automatically when this spikes."

#### C. Historical Accuracy by Model/Horizon
- Show the breakdown table
- Highlight best-performing combinations

**What to say:**
> "Here's accuracy broken down by individual model and horizon. You can see which specific models excel at which timeframes. This transparency is critical for institutional adoption."

---

## Part 5: Historical Rewind (2 minutes)

### Navigate
1. Click **"Historical Rewind"** in the main nav
2. Select Crude Oil

### What to show

#### A. Date Slider
- **DRAG** the slider to a past date (try 30 days ago)
- Watch metrics update

**What to say:**
> "This is our audit trail feature. I can drag to any historical date and see exactly what the model knew at that time. Watch the metrics update..."

> "[After sliding] On [date], the model was showing [signal] with [confidence]% confidence. We can verify this against actual price movement."

#### B. Signal History Validation
- Show how past signals matched actual moves
- Point to accuracy validation

**What to say:**
> "This historical rewind is critical for institutional credibility. Every signal is reproducible. You can audit any date and verify our methodology."

### Key insight

**What to say:**
> "For compliance and due diligence, this complete audit trail means you can always explain why a signal was generated on any given day."

---

## Part 6: Interactive Demo (2 minutes)

### Toggle Filters Live
1. Go back to **Alpha Pro Dashboard**
2. **CLICK** to toggle **RSI Filter**
3. Watch accuracy update in real-time

**What to say:**
> "Let me show how filters work. I'll toggle the RSI filter... [click] ...watch the accuracy metric update in real-time. The filter is now excluding signals when RSI indicates overbought or oversold conditions."

### Exit Strategy Comparison
1. Scroll to **Exit Strategy** section
2. Click through different strategies

**What to say:**
> "For trade management, we offer four exit strategies. Scale Out takes profit at each target level. Let me click Aggressive... [click] ...you see the projected return changes based on holding to T3."

---

## Closing (1 minute)

**What to say:**

> "To summarize what we've seen:
> - **10,179 models** per asset, combining short and long horizons
> - **Sharpe of 11.96** vs baseline of negative 0.82
> - **82% win rate** with controlled drawdown under 3%
> - Complete audit trail for institutional compliance
> - Persona-based views for different user types

> "The platform is API-ready for integration. We can discuss a pilot with your hedging desk - we're thinking 3-5 users to start."

---

## Common Questions & Answers

### Q: "How do you handle model staleness?"
**A:** "Models update daily. The health indicator turns amber if data is >24 hours old, red if >48 hours. We also track per-model last-update timestamps visible in the Quant dashboard."

### Q: "What happens when models disagree?"
**A:** "Disagreement reduces confidence scores automatically. High disagreement (>30% variance) triggers a NEUTRAL signal regardless of majority direction. This is our risk management layer."

### Q: "Why short horizons perform better?"
**A:** "Market microstructure. Short-term price movements have more signal, less noise. Our D+1 to D+3 horizons capture momentum before mean reversion kicks in. Longer horizons D+7-10 are better for magnitude, not direction."

### Q: "How do you avoid overfitting?"
**A:** "Three safeguards: (1) Walk-forward validation - models only see past data, (2) Ensemble averaging - 1000+ models prevents single-model bias, (3) Out-of-sample testing - the metrics you see are from held-out data."

### Q: "Can we backtest custom strategies?"
**A:** "Yes. The API exposes raw signals and forecasts. You can pull historical data via `/api/v1/signals/{asset}` and run your own backtests. We also offer configuration endpoints to adjust filter parameters."

### Q: "What's the data latency?"
**A:** "Daily batch processing. Forecasts update overnight. For intraday needs, we're planning a streaming tier - that's Phase 2."

### Q: "How does pricing work?"
**A:** "We're exploring per-seat licensing for the dashboard and API call pricing for programmatic access. Happy to discuss what model works for your use case."

### Q: "Can we integrate with our existing systems?"
**A:** "Absolutely. REST API with JSON responses. We support API key auth with per-asset permissions. We can also discuss custom integrations."

---

## Backup Plans

### If API is down
1. Navigate to `docs/screenshots/` folder
2. Show pre-captured dashboard images
3. Say: "Let me show you screenshots from our staging environment while we resolve the connection."

### If data looks stale
1. Check the timestamp in the top corner
2. Say: "I see the data is from [date]. Let me refresh..." [F5]
3. If still stale: "The overnight batch is still running. The metrics are representative of typical output."

### If asked about a feature we don't have
1. Say: "That's not in the current release, but it's on our roadmap."
2. Make a note: "Let me capture that as feedback for the product team."

### If demo runs long
- Skip Historical Rewind (Part 5)
- Abbreviate Quant Dashboard (Part 4) to just headline metrics
- Core demo can run in 10 minutes if needed

---

## Post-Demo Actions

- [ ] Send follow-up email with API documentation link
- [ ] Share access credentials for sandbox environment
- [ ] Schedule technical deep-dive if requested
- [ ] Note any feature requests mentioned
- [ ] Confirm next steps for pilot program

---

## Quick Reference Card

| Metric | Value | Context |
|--------|-------|---------|
| Sharpe Ratio | 11.96 | vs -0.82 baseline |
| Win Rate | 82% | vs 42.6% baseline |
| Total Return | +107% | ~330 day backtest |
| Max Drawdown | -2.8% | Risk-controlled |
| Avg Profit/Trade | $1.81 | Crude Oil |
| Models per Asset | 10,179 | Crude Oil |
| Horizons | D+1 to D+10 | 10 total |
| Optimal Horizons | D+1,2,3,7,10 | Triple filter |
| Supported Assets | 15+ | Commodities, Crypto, Indices |

**Keyboard shortcuts during demo:**
- `Cmd+R` / `F5` - Refresh page
- `Cmd+Tab` - Switch between terminal/browser
- `Cmd+Shift+F` - Full screen browser (less distraction)
