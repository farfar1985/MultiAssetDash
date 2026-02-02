"""
Calculate the LAST 30 trades (most recent) for Crude Oil OPTIMIZED ensemble
Using the SAME method as precalculate_metrics.py to verify 84.6%
"""
import pandas as pd
import numpy as np
import os
import json

# Load Crude Oil data
script_dir = os.path.dirname(os.path.abspath(__file__))
asset_dir = os.path.join(script_dir, 'data/1866_Crude_Oil')

# Load forecasts for optimal horizons [1, 3, 5, 8] - SAME as precalculate_metrics.py
optimal_horizons = [1, 3, 5, 8]
threshold = 0.1

# Build forecast matrix using RAW PREDICTIONS (same as precalculate_metrics.py)
horizons_data = {}
for h in optimal_horizons:
    fp = os.path.join(asset_dir, f'forecast_d{h}.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp, parse_dates=['date'])
        horizons_data[h] = df[['date', 'prediction', 'actual']].copy()

# Merge all horizons
base_df = None
for h, df in horizons_data.items():
    df = df.rename(columns={'prediction': f'pred_{h}'})
    if base_df is None:
        base_df = df[['date', f'pred_{h}', 'actual']].copy()
    else:
        base_df = base_df.merge(df[['date', f'pred_{h}']], on='date', how='outer')

base_df = base_df.sort_values('date').reset_index(drop=True)
base_df = base_df.dropna(subset=['actual'])

# Create forecast matrix (indexed by date)
forecast_matrix = pd.DataFrame(index=pd.to_datetime(base_df['date']))
for h in optimal_horizons:
    forecast_matrix[h] = base_df[f'pred_{h}'].values

prices = base_df.set_index('date')['actual']
prices.index = pd.to_datetime(prices.index)

print(f'Data period: {forecast_matrix.index[0].date()} to {forecast_matrix.index[-1].date()}')
print(f'Total days: {len(forecast_matrix)}')

# Calculate pairwise slopes signals (EXACT SAME as precalculate_metrics.py)
def calculate_signals_pairwise_slopes(forecast_df, horizons, threshold):
    signals = []
    net_probs = []
    
    for date in forecast_df.index:
        row = forecast_df.loc[date]
        slopes = []
        
        for i_idx, h1 in enumerate(horizons):
            for h2 in horizons[i_idx + 1:]:
                if h1 in row.index and h2 in row.index:
                    if pd.notna(row[h1]) and pd.notna(row[h2]):
                        drift = row[h2] - row[h1]
                        slopes.append(drift)
        
        if len(slopes) == 0:
            net_prob = 0.0
        else:
            bullish = sum(1 for s in slopes if s > 0)
            bearish = sum(1 for s in slopes if s < 0)
            net_prob = (bullish - bearish) / len(slopes)
        
        net_probs.append(net_prob)
        
        if net_prob > threshold:
            signals.append('BULLISH')
        elif net_prob < -threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')
    
    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)

signals, net_probs = calculate_signals_pairwise_slopes(forecast_matrix, optimal_horizons, threshold)

# Calculate ALL trades (EXACT SAME as precalculate_metrics.py)
trades = []
position = None
entry_price = None
entry_date = None
entry_signal = None

for i in range(len(signals)):
    current_signal = signals.iloc[i]
    current_price = prices.iloc[i]
    current_date = signals.index[i]
    
    if position is None:
        if current_signal in ['BULLISH', 'BEARISH']:
            position = current_signal
            entry_price = current_price
            entry_date = current_date
            entry_signal = current_signal
    else:
        if current_signal != position:
            # Exit
            if entry_signal == 'BULLISH':
                pnl = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl = ((entry_price - current_price) / entry_price) * 100
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': current_date,
                'signal': entry_signal,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'days': (current_date - entry_date).days
            })
            
            # Re-enter if not neutral
            if current_signal in ['BULLISH', 'BEARISH']:
                position = current_signal
                entry_price = current_price
                entry_date = current_date
                entry_signal = current_signal
            else:
                position = None

print(f'Total trades: {len(trades)}')

# Calculate equity curve (EXACT SAME as precalculate_metrics.py)
equity = 100
for t in trades:
    equity *= (1 + t['pnl'] / 100)

total_return = equity - 100
all_wins = len([t for t in trades if t['pnl'] > 0])
all_losses = len([t for t in trades if t['pnl'] <= 0])

print()
print(f'ALL TRADES ({len(trades)} trades) - Using precalculate method:')
print(f'  Final Equity: {equity:.2f}')
print(f'  Total Return: {total_return:.2f}%  <-- Should match config 84.6%')
print(f'  Win Rate: {all_wins}/{len(trades)} = {all_wins/len(trades)*100:.1f}%')

# Get LAST 30 trades (most recent)
last_30 = trades[-30:] if len(trades) >= 30 else trades

print()
print('=' * 60)
print('LAST 30 TRADES (MOST RECENT) - WHAT SHOULD BE SHOWN')
print('=' * 60)
print(f'Period: {last_30[0]["entry_date"].date()} to {last_30[-1]["exit_date"].date()}')

total_pnl = sum(t['pnl'] for t in last_30)
wins = len([t for t in last_30 if t['pnl'] > 0])
losses = len([t for t in last_30 if t['pnl'] <= 0])
gross_profit = sum(t['pnl'] for t in last_30 if t['pnl'] > 0)
gross_loss = sum(abs(t['pnl']) for t in last_30 if t['pnl'] < 0)
profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

# Equity for last 30
equity_last30 = 100
for t in last_30:
    equity_last30 *= (1 + t['pnl'] / 100)

print()
print('SUMMARY:')
print(f'  Return (sum): {total_pnl:.2f}%')
print(f'  Return (compound): {equity_last30 - 100:.2f}%')
print(f'  Wins/Losses: {wins}/{losses} ({wins/len(last_30)*100:.1f}% win rate)')
print(f'  Profit Factor: {profit_factor:.2f}x')
print(f'  Avg P&L/Trade: {total_pnl/len(last_30):.2f}%')

print()
print('TRADE LIST (newest first):')
print(f'  #   ENTRY        SIGNAL    ENTRY$    EXIT$  DAYS      P&L')
print('-' * 65)

for i, t in enumerate(reversed(last_30)):
    pnl_str = f'+{t["pnl"]:.2f}%' if t['pnl'] > 0 else f'{t["pnl"]:.2f}%'
    print(f' {30-i:>2}  {str(t["entry_date"].date()):>10}  {t["signal"]:>8}  {t["entry_price"]:>7.2f}  {t["exit_price"]:>7.2f}  {t["days"]:>4}  {pnl_str:>8}')

# Also show FIRST 30 for comparison
first_30 = trades[:30]
print()
print('=' * 60)
print('FIRST 30 TRADES (OLDEST) - WHAT DASHBOARD CURRENTLY SHOWS')
print('=' * 60)
print(f'Period: {first_30[0]["entry_date"].date()} to {first_30[-1]["exit_date"].date()}')

# Equity for first 30
equity_first30 = 100
for t in first_30:
    equity_first30 *= (1 + t['pnl'] / 100)

total_pnl_first = sum(t['pnl'] for t in first_30)
wins_first = len([t for t in first_30 if t['pnl'] > 0])
losses_first = len([t for t in first_30 if t['pnl'] <= 0])
gross_profit_first = sum(t['pnl'] for t in first_30 if t['pnl'] > 0)
gross_loss_first = sum(abs(t['pnl']) for t in first_30 if t['pnl'] < 0)
profit_factor_first = gross_profit_first / gross_loss_first if gross_loss_first > 0 else 0

print()
print('SUMMARY:')
print(f'  Return (sum): {total_pnl_first:.2f}%')
print(f'  Return (compound): {equity_first30 - 100:.2f}%')
print(f'  Wins/Losses: {wins_first}/{losses_first} ({wins_first/len(first_30)*100:.1f}% win rate)')
print(f'  Profit Factor: {profit_factor_first:.2f}x')
