# q_ensemble_sandbox/plot_persona_dashboard.py
# SANDBOXED - Full Snake Chart for Bitcoin
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config_sandbox as cfg

def generate_persona_dashboard():
    print(f"Generating Interactive Persona Dashboard for {cfg.PROJECT_NAME}...")
    
    # 1. Load Data
    metrics_path = os.path.join(cfg.DATA_DIR, 'matrix_drift_signals.csv')
    if not os.path.exists(metrics_path): 
        print("ERROR: matrix_drift_signals.csv not found. Run matrix_drift_analysis.py first.")
        return
    df = pd.read_csv(metrics_path, parse_dates=['date']).set_index('date')
    
    if not os.path.exists(cfg.PRICE_CACHE_PATH):
        print("ERROR: price_history_cache.json not found.")
        return
    prices = pd.read_json(cfg.PRICE_CACHE_PATH, convert_dates=['date']).set_index('date')['close']
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    
    common = df.index.intersection(prices.index).sort_values()
    df = df.loc[common]
    prices = prices.loc[common]
    
    # Calculate RSI (7)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    dates = common
    price_vals = prices.values
    probs = df['net_prob'].values
    rsi_vals = rsi.values
    
    # 2. Identify Signals (Raw vs Filtered)
    buy_raw_x, buy_raw_y = [], []
    buy_raw_cont_x, buy_raw_cont_y = [], []
    sell_raw_x, sell_raw_y = [], []
    sell_raw_cont_x, sell_raw_cont_y = [], []
    
    is_bull = False
    is_bear = False
    
    for i in range(1, len(probs)):
        curr = probs[i]
        date = dates[i]
        price = price_vals[i]
        
        if curr > 0.25:
            if not is_bull:
                buy_raw_x.append(date); buy_raw_y.append(price)
                is_bull = True
            else:
                buy_raw_cont_x.append(date); buy_raw_cont_y.append(price)
            is_bear = False
        elif curr < -0.25:
            if not is_bear:
                sell_raw_x.append(date); sell_raw_y.append(price)
                is_bear = True
            else:
                sell_raw_cont_x.append(date); sell_raw_cont_y.append(price)
            is_bull = False
        else:
            is_bull = False; is_bear = False

    # Filtered Signals (RSI 7 Filter)
    buy_filt_x, buy_filt_y = [], []
    buy_filt_cont_x, buy_filt_cont_y = [], []
    sell_filt_x, sell_filt_y = [], []
    sell_filt_cont_x, sell_filt_cont_y = [], []
    
    is_bull = False
    is_bear = False
    
    for i in range(1, len(probs)):
        curr = probs[i]
        date = dates[i]
        price = price_vals[i]
        rsi_val = rsi_vals[i]
        
        if curr > 0.25:
            if rsi_val > 70: 
                pass
            elif not is_bull:
                buy_filt_x.append(date); buy_filt_y.append(price)
                is_bull = True
            else:
                buy_filt_cont_x.append(date); buy_filt_cont_y.append(price)
            is_bear = False
        elif curr < -0.25:
            if rsi_val < 30: 
                pass
            elif not is_bear:
                sell_filt_x.append(date); sell_filt_y.append(price)
                is_bear = True
            else:
                sell_filt_cont_x.append(date); sell_filt_cont_y.append(price)
            is_bull = False
        else:
            is_bull = False; is_bear = False

    # 3. Calculate Stats
    future_mean = prices.rolling(5).mean().shift(-5)
    future_ret = prices.pct_change(5).shift(-5).fillna(0)
    
    # Buyer Stats
    bull_mask = probs > 0.25 
    buy_dates_all = dates[bull_mask]
    
    buy_wins_dir = 0
    buy_wins_cost = 0
    buy_cost_savings = 0
    buy_valid = 0
    
    for d in buy_dates_all:
        if d in future_mean.index and d in future_ret.index:
            avg_price = future_mean.loc[d]
            end_ret = future_ret.loc[d]
            entry_price = prices.loc[d]
            
            if not np.isnan(avg_price):
                buy_valid += 1
                savings = (avg_price - entry_price) / entry_price
                buy_cost_savings += savings
                if savings > 0: buy_wins_cost += 1
                if end_ret > 0: buy_wins_dir += 1

    buy_acc_dir = buy_wins_dir / buy_valid if buy_valid else 0
    buy_acc_cost = buy_wins_cost / buy_valid if buy_valid else 0
    buy_avg_saving = buy_cost_savings / buy_valid if buy_valid else 0
    
    # Seller Stats
    bear_mask = probs < -0.25
    sell_dates_all = dates[bear_mask]
    
    sell_wins_dir = 0
    sell_wins_cost = 0
    sell_cost_savings = 0
    sell_valid = 0
    
    for d in sell_dates_all:
        if d in future_mean.index:
            avg_price = future_mean.loc[d]
            end_ret = future_ret.loc[d]
            entry_price = prices.loc[d]
            
            if not np.isnan(avg_price):
                sell_valid += 1
                savings = (entry_price - avg_price) / entry_price
                sell_cost_savings += savings
                if savings > 0: sell_wins_cost += 1
                if end_ret < 0: sell_wins_dir += 1

    sell_acc_dir = sell_wins_dir / sell_valid if sell_valid else 0
    sell_acc_cost = sell_wins_cost / sell_valid if sell_valid else 0
    sell_avg_saving = sell_cost_savings / sell_valid if sell_valid else 0
    
    # Pro Stats
    pro_valid = buy_valid + sell_valid
    pro_wins = buy_wins_dir + sell_wins_dir
    pro_acc = pro_wins / pro_valid if pro_valid else 0
    pro_pnl = buy_avg_saving + sell_avg_saving

    # Annotations Text
    buyer_text = f"<b>BUYER</b><br>✅ Cost Adv Win%: {buy_acc_cost:.1%}<br>🎯 Directional Acc: {buy_acc_dir:.1%}<br>📊 Signals: {buy_valid}<br>💰 Avg Cost Saving: {buy_avg_saving:.1%}"
    seller_text = f"<b>SELLER</b><br>✅ Revenue Win%: {sell_acc_cost:.1%}<br>🎯 Directional Acc: {sell_acc_dir:.1%}<br>📊 Signals: {sell_valid}<br>💰 Avg Revenue Gain: {sell_avg_saving:.1%}"
    pro_text = f"<b>PRO TRADER</b><br>🎯 Accuracy: {pro_acc:.1%}<br>📊 Signals: {pro_valid}<br>💰 Edge: {pro_pnl:.1%}"

    # 4. Build Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])
    
    # Snake Segments
    snake_traces = []
    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i+1]
        p0, p1 = price_vals[i], price_vals[i+1]
        score = probs[i]
        width = 2 + (abs(score) * 12)
        if score > 0.25: c = f'rgba(0, 255, 0, {0.4 + abs(score)*0.6})'
        elif score < -0.25: c = f'rgba(255, 0, 0, {0.4 + abs(score)*0.6})'
        else: c = 'rgba(100, 100, 100, 0.4)'
        
        snake_traces.append(go.Scatter(x=[d0, d1], y=[p0, p1], mode='lines', line=dict(color=c, width=width), showlegend=False, hoverinfo='skip'))
    
    for t in snake_traces: fig.add_trace(t, row=1, col=1)
    num_snake_traces = len(snake_traces)
    
    # Signal Traces
    fig.add_trace(go.Scatter(x=buy_raw_x, y=buy_raw_y, mode='markers', name='BUY Entry', 
                             marker=dict(symbol='triangle-up', size=14, color='#00ff00', line=dict(width=1, color='white'))), row=1, col=1)
    fig.add_trace(go.Scatter(x=buy_raw_cont_x, y=buy_raw_cont_y, mode='markers', name='BUY Hold', 
                             marker=dict(symbol='triangle-up', size=8, color='white', line=dict(width=1, color='#00ff00'), opacity=0.9)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_raw_x, y=sell_raw_y, mode='markers', name='SELL Entry', 
                             marker=dict(symbol='triangle-down', size=14, color='#ff0000', line=dict(width=1, color='white'))), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_raw_cont_x, y=sell_raw_cont_y, mode='markers', name='SELL Hold', 
                             marker=dict(symbol='triangle-down', size=8, color='white', line=dict(width=1, color='#ff0000'), opacity=0.9)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=buy_filt_x, y=buy_filt_y, mode='markers', name='Pro BUY', 
                             marker=dict(symbol='triangle-up', size=14, color='#00ff00', line=dict(width=1, color='gold'))), row=1, col=1)
    fig.add_trace(go.Scatter(x=buy_filt_cont_x, y=buy_filt_cont_y, mode='markers', name='Pro Hold', 
                             marker=dict(symbol='triangle-up', size=8, color='white', line=dict(width=1, color='gold'), opacity=0.9)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_filt_x, y=sell_filt_y, mode='markers', name='Pro SELL', 
                             marker=dict(symbol='triangle-down', size=14, color='#ff0000', line=dict(width=1, color='gold'))), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_filt_cont_x, y=sell_filt_cont_y, mode='markers', name='Pro Hold', 
                             marker=dict(symbol='triangle-down', size=8, color='white', line=dict(width=1, color='gold'), opacity=0.9)), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=dates, y=rsi_vals, mode='lines', name='RSI (7)', line=dict(color='white', width=1)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # --- Live Forecast ---
    live_path = os.path.join(cfg.DATA_DIR, 'live_forecast.json')
    if os.path.exists(live_path):
        with open(live_path, 'r') as f:
            live_data = json.load(f)
        preds = live_data.get('predictions', [])
        if preds:
            last_date = prices.index[-1]
            last_price = prices.iloc[-1]
            
            future_dates = [last_date]
            future_prices = [last_price]
            
            VOLATILITY_MULTIPLIER = 3.0
            
            for p in preds:
                h = p['horizon_days']
                val = p.get('predicted_price')
                if val is None or np.isnan(val): continue
                
                diff = val - last_price
                val_amplified = last_price + (diff * VOLATILITY_MULTIPLIER)
                
                future_date = last_date + pd.tseries.offsets.BusinessDay(n=h)
                future_dates.append(future_date)
                future_prices.append(val_amplified)
            
            # Dynamic Color for Live Line
            live_prices_arr = np.array(future_prices[1:])
            n_points = len(live_prices_arr)
            if n_points >= 2:
                slopes = []
                for i in range(n_points):
                    for j in range(i + 1, n_points):
                        slopes.append(live_prices_arr[j] - live_prices_arr[i])
                
                slopes = np.array(slopes)
                if len(slopes) > 0:
                    bullish = (slopes > 0).sum()
                    bearish = (slopes < 0).sum()
                    total = len(slopes)
                    live_net_prob = (bullish - bearish) / total
                    
                    live_width = 4 + (abs(live_net_prob) * 8)
                    if live_net_prob > 0.25: live_color = '#00ff00'
                    elif live_net_prob < -0.25: live_color = '#ff0000'
                    else: live_color = 'rgba(100, 100, 100, 0.4)'
                else:
                    live_color = 'gold'; live_width = 4
            else:
                live_color = 'gold'; live_width = 4

            fig.add_trace(go.Scatter(
                x=future_dates, y=future_prices,
                mode='lines+markers', name='Live Forecast',
                line=dict(color=live_color, width=live_width),
                marker=dict(size=6, color=live_color)
            ), row=1, col=1)
            
            fig.add_vline(x=last_date, line_width=1, line_dash="dash", line_color="gray", row=1, col=1)

    # Visibility Arrays
    vis_snake = [True] * num_snake_traces
    vis_buyer = vis_snake + [True, True, False, False, False, False, False, False, True, True]
    vis_seller = vis_snake + [False, False, True, True, False, False, False, False, True, True]
    vis_pro = vis_snake + [False, False, False, False, True, True, True, True, True, True]
    
    # Annotations
    annot_base = dict(xref="paper", yref="paper", x=0.02, y=1.12, showarrow=False, align="left", borderwidth=1, font=dict(size=12, color="white"))
    annot_buyer = annot_base.copy(); annot_buyer['text'] = buyer_text; annot_buyer['bgcolor'] = "rgba(0,50,0,0.8)"; annot_buyer['bordercolor'] = "#00ff00"
    annot_seller = annot_base.copy(); annot_seller['text'] = seller_text; annot_seller['bgcolor'] = "rgba(50,0,0,0.8)"; annot_seller['bordercolor'] = "#ff0000"
    annot_pro = annot_base.copy(); annot_pro['text'] = pro_text; annot_pro['bgcolor'] = "rgba(0,0,50,0.8)"; annot_pro['bordercolor'] = "#aaaaff"

    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            bgcolor="#cccccc",
            active=2, 
            bordercolor="#444",
            font=dict(color="black"),
            buttons=list([
                dict(args=[{"visible": vis_buyer}, {"annotations": [annot_buyer]}], label="Buyer", method="update"),
                dict(args=[{"visible": vis_seller}, {"annotations": [annot_seller]}], label="Seller", method="update"),
                dict(args=[{"visible": vis_pro}, {"annotations": [annot_pro]}], label="Pro Trader", method="update")
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.5, xanchor="center",
            y=1.12, yanchor="top"
        ),
    ]
    
    # Initial Range
    last_dt = prices.index[-1]
    start_view = last_dt - timedelta(days=60)
    end_view = last_dt + timedelta(days=15)
    
    visible_prices = prices[prices.index >= start_view]
    if not visible_prices.empty:
        min_y = visible_prices.min()
        max_y = visible_prices.max()
        padding = (max_y - min_y) * 0.1
        y_range = [min_y - padding, max_y + padding]
    else:
        y_range = None
    
    fig.update_layout(
        updatemenus=updatemenus,
        annotations=[annot_pro],
        title={'text': f'{cfg.PROJECT_NAME} Q-Ensemble Dashboard', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 24}},
        template='plotly_dark',
        height=900, 
        margin=dict(t=160, l=40, r=40),
        xaxis=dict(range=[start_view, end_view], autorange=False, showspikes=True, spikemode='across', spikesnap='cursor', showline=True, showgrid=True), 
        yaxis=dict(range=y_range, autorange=False, fixedrange=False, showspikes=True, spikemode='across', spikesnap='cursor', showline=True, showgrid=True),
        hovermode='x unified'
    )
    
    out_path = os.path.join(cfg.DATA_DIR, 'persona_dashboard.html')
    fig.write_html(out_path, config={'scrollZoom': True})
    print(f"Dashboard saved to {out_path}")
    
    return fig

if __name__ == "__main__":
    generate_persona_dashboard()

