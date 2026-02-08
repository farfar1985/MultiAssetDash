# q_ensemble_sandbox/matrix_drift_analysis.py
# SANDBOXED - Signal calculation for Bitcoin
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config_sandbox as cfg

def calculate_matrix_drift():
    print(f"--- Running Full Drift Matrix Analysis for {cfg.PROJECT_NAME} ---")
    
    # DYNAMIC HORIZON DISCOVERY - find all forecast_d*.csv files
    import glob
    import re
    
    forecast_pattern = os.path.join(cfg.DATA_DIR, 'forecast_d*.csv')
    all_forecast_files = glob.glob(forecast_pattern)
    
    horizons = {}
    for path in all_forecast_files:
        filename = os.path.basename(path)
        match = re.search(r'forecast_d(\d+)\.csv', filename)
        if match:
            h = int(match.group(1))
            horizons[h] = pd.read_csv(path, parse_dates=['date']).set_index('date')['prediction']
    
    if not horizons:
        print("No forecast data found.")
        return

    available_horizons = sorted(horizons.keys())
    print(f"[INFO] Discovered {len(available_horizons)} horizons: {available_horizons}")
    
    df = pd.DataFrame(horizons)
    df = df.ffill().fillna(0)  # Forward-fill only, no bfill (leaks future data)
    
    # Load Actuals for Charting (from sandbox price cache)
    prices = None
    if os.path.exists(cfg.PRICE_CACHE_PATH):
        prices = pd.read_json(cfg.PRICE_CACHE_PATH, convert_dates=['date']).set_index('date')
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices['close']
    
    # 2. Calculate Drift Matrix Metrics per Day (using ONLY available horizons)
    results = []
    
    for date in df.index:
        row = df.loc[date]
        
        slopes = []
        # Only compare available horizons
        for i_idx, i in enumerate(available_horizons):
            for j in available_horizons[i_idx + 1:]:
                drift = row[j] - row[i]
                slopes.append(drift)
        
        slopes = np.array(slopes)
        
        total_pairs = len(slopes)
        bullish_pairs = (slopes > 0).sum()
        bearish_pairs = (slopes < 0).sum()
        
        bullish_prob = bullish_pairs / total_pairs
        bearish_prob = bearish_pairs / total_pairs
        
        mean_drift = slopes.mean()
        
        results.append({
            'date': date,
            'bullish_prob': bullish_prob,
            'bearish_prob': bearish_prob,
            'net_prob': bullish_prob - bearish_prob,
            'mean_drift': mean_drift
        })
        
    res_df = pd.DataFrame(results).set_index('date')
    
    # 3. Generate Color-Coded Plot
    if prices is not None:
        common = res_df.index.intersection(prices.index)
        plot_df = pd.DataFrame({'price': prices.loc[common], 'score': res_df.loc[common]['net_prob']})
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=plot_df.index, 
            y=plot_df['price'],
            mode='markers+lines',
            name='Trend Strength',
            line=dict(color='gray', width=1),
            marker=dict(
                size=6,
                color=plot_df['score'],
                colorscale='RdYlGn',
                cmin=-1, cmax=1,
                colorbar=dict(title="Trend Prob")
            )
        ))
        
        fig.update_layout(
            title=f'{cfg.PROJECT_NAME} Price Colored by Matrix Drift Probability',
            template='plotly_dark',
            height=600
        )
        
        out_html = os.path.join(cfg.DATA_DIR, 'matrix_drift_chart.html')
        fig.write_html(out_html)
        print(f"Chart saved to {out_html}")
    
    # Save metrics
    res_df.to_csv(os.path.join(cfg.DATA_DIR, 'matrix_drift_signals.csv'))
    
    print("\n--- Recent Matrix Signals ---")
    print(res_df.tail(10)[['bullish_prob', 'net_prob']])

if __name__ == "__main__":
    calculate_matrix_drift()

