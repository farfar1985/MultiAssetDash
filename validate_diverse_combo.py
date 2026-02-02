"""
Validate [1, 3, 5, 8] Combo - Is 100% Win Rate Real?
====================================================
Let's look at the actual trades and test on longer periods.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(SCRIPT_DIR, 'data', '1866_Crude_Oil')

COMBO = [1, 3, 5, 8]
THRESHOLD = 0.1

print("=" * 80)
print("  VALIDATING [1, 3, 5, 8] COMBO - Is 100% Win Rate Real?")
print("=" * 80)


def load_forecast_data():
    horizons_data = {}
    for h in range(1, 11):
        forecast_file = os.path.join(ASSET_DIR, f'forecast_d{h}.csv')
        if os.path.exists(forecast_file):
            df = pd.read_csv(forecast_file, parse_dates=['date'])
            if 'prediction' in df.columns and 'actual' in df.columns:
                horizons_data[h] = df[['date', 'prediction', 'actual']].copy()
    
    available_horizons = sorted(horizons_data.keys())
    
    base_df = None
    for h, df in horizons_data.items():
        df = df.rename(columns={'prediction': f'pred_{h}', 'actual': 'actual'})
        if base_df is None:
            base_df = df
        else:
            base_df = base_df.merge(df[['date', f'pred_{h}']], on='date', how='outer')
    
    base_df = base_df.sort_values('date').reset_index(drop=True)
    base_df = base_df.dropna(subset=['actual'])
    
    forecast_matrix = pd.DataFrame(index=base_df['date'])
    for h in available_horizons:
        forecast_matrix[h] = base_df[f'pred_{h}'].values
    
    prices = base_df.set_index('date')['actual']
    
    return forecast_matrix, prices


def calculate_signals_pairwise_slopes(forecast_df, horizons, threshold):
    signals = []
    net_probs = []
    
    for date in forecast_df.index:
        row = forecast_df.loc[date]
        slopes = []
        
        for i_idx, i in enumerate(horizons):
            for j in horizons[i_idx + 1:]:
                if i in row.index and j in row.index:
                    if pd.notna(row[i]) and pd.notna(row[j]):
                        drift = row[j] - row[i]
                        slopes.append(drift)
        
        if len(slopes) == 0:
            net_prob = 0.0
        else:
            bullish = sum(1 for s in slopes if s > 0)
            bearish = sum(1 for s in slopes if s < 0)
            total = len(slopes)
            net_prob = (bullish - bearish) / total
        
        net_probs.append(net_prob)
        
        if net_prob > threshold:
            signals.append('BULLISH')
        elif net_prob < -threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')
    
    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def get_trades(signals, prices):
    """Get detailed trades."""
    trades = []
    position = None
    entry_price = None
    entry_date = None
    
    for i in range(len(signals)):
        current_signal = signals.iloc[i]
        current_price = prices.iloc[i]
        current_date = signals.index[i]
        
        if position is None:
            if current_signal in ['BULLISH', 'BEARISH']:
                position = current_signal
                entry_price = current_price
                entry_date = current_date
        else:
            should_exit = False
            if position == 'BULLISH' and current_signal != 'BULLISH':
                should_exit = True
            elif position == 'BEARISH' and current_signal != 'BEARISH':
                should_exit = True
            
            if should_exit:
                if position == 'BULLISH':
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'days_held': (current_date - entry_date).days
                })
                
                if current_signal in ['BULLISH', 'BEARISH']:
                    position = current_signal
                    entry_price = current_price
                    entry_date = current_date
                else:
                    position = None
    
    return pd.DataFrame(trades) if trades else None


# Load data
print("\n📊 Loading data...")
forecast_df, prices = load_forecast_data()
forecast_df.index = pd.to_datetime(forecast_df.index)
prices.index = pd.to_datetime(prices.index)

print(f"✅ Total data: {len(forecast_df)} days")
print(f"✅ Date range: {forecast_df.index.min().date()} to {forecast_df.index.max().date()}")

# Calculate signals for ALL data
signals, net_probs = calculate_signals_pairwise_slopes(forecast_df, COMBO, THRESHOLD)

# Get all trades
all_trades = get_trades(signals, prices)

if all_trades is not None and len(all_trades) > 0:
    print(f"\n" + "=" * 80)
    print(f"  ALL TRADES with [1, 3, 5, 8] (Total: {len(all_trades)})")
    print("=" * 80)
    
    for i, trade in all_trades.iterrows():
        emoji = "✅" if trade['pnl_pct'] > 0 else "❌"
        print(f"\n  Trade {i+1}: {trade['position']}")
        print(f"    Entry: {trade['entry_date'].date()} @ ${trade['entry_price']:.2f}")
        print(f"    Exit:  {trade['exit_date'].date()} @ ${trade['exit_price']:.2f}")
        print(f"    P&L:   {emoji} {trade['pnl_pct']:+.2f}% ({trade['days_held']} days)")
    
    # Summary stats
    wins = all_trades[all_trades['pnl_pct'] > 0]
    losses = all_trades[all_trades['pnl_pct'] <= 0]
    
    print(f"\n" + "=" * 80)
    print(f"  OVERALL STATISTICS")
    print("=" * 80)
    print(f"\n  Total Trades:    {len(all_trades)}")
    print(f"  Winners:         {len(wins)} ({len(wins)/len(all_trades)*100:.1f}%)")
    print(f"  Losers:          {len(losses)} ({len(losses)/len(all_trades)*100:.1f}%)")
    print(f"  Total Return:    {all_trades['pnl_pct'].sum():+.1f}%")
    print(f"  Avg Win:         {wins['pnl_pct'].mean():+.2f}%" if len(wins) > 0 else "  Avg Win:         N/A")
    print(f"  Avg Loss:        {losses['pnl_pct'].mean():.2f}%" if len(losses) > 0 else "  Avg Loss:        N/A")
    print(f"  Best Trade:      {all_trades['pnl_pct'].max():+.2f}%")
    print(f"  Worst Trade:     {all_trades['pnl_pct'].min():+.2f}%")
    
    # Test on different periods
    print(f"\n" + "=" * 80)
    print(f"  PERFORMANCE BY PERIOD")
    print("=" * 80)
    
    periods = [
        ("Last 30 days", 30),
        ("Last 60 days", 60),
        ("Last 90 days", 90),
        ("Last 180 days", 180),
        ("All Time", len(forecast_df)),
    ]
    
    for period_name, days in periods:
        period_df = forecast_df.tail(days)
        period_prices = prices.loc[period_df.index]
        
        period_signals, _ = calculate_signals_pairwise_slopes(period_df, COMBO, THRESHOLD)
        period_trades = get_trades(period_signals, period_prices)
        
        if period_trades is not None and len(period_trades) > 0:
            wins = period_trades[period_trades['pnl_pct'] > 0]
            win_rate = len(wins) / len(period_trades) * 100
            total_ret = period_trades['pnl_pct'].sum()
            print(f"\n  {period_name}:")
            print(f"    Trades: {len(period_trades)}, Win Rate: {win_rate:.1f}%, Return: {total_ret:+.1f}%")
        else:
            print(f"\n  {period_name}:")
            print(f"    Not enough trades")

    # Compare with [1,2,3] and ALL horizons
    print(f"\n" + "=" * 80)
    print(f"  COMPARISON: [1,3,5,8] vs [1,2,3] vs ALL HORIZONS (All Time)")
    print("=" * 80)
    
    # Test [1,2,3]
    signals_123, _ = calculate_signals_pairwise_slopes(forecast_df, [1,2,3], THRESHOLD)
    trades_123 = get_trades(signals_123, prices)
    
    # Test ALL horizons [1,2,3,4,5,6,7,8,9,10] with threshold 0.3 (needed for 10 horizons)
    all_horizons = [1,2,3,4,5,6,7,8,9,10]
    signals_all, _ = calculate_signals_pairwise_slopes(forecast_df, all_horizons, 0.3)
    trades_all = get_trades(signals_all, prices)
    
    if trades_123 is not None and trades_all is not None:
        wins_1358 = len(all_trades[all_trades['pnl_pct'] > 0])
        wins_123 = len(trades_123[trades_123['pnl_pct'] > 0])
        wins_all = len(trades_all[trades_all['pnl_pct'] > 0])
        
        # Calculate avg win/loss for each
        avg_win_1358 = all_trades[all_trades['pnl_pct'] > 0]['pnl_pct'].mean() if wins_1358 > 0 else 0
        avg_loss_1358 = abs(all_trades[all_trades['pnl_pct'] <= 0]['pnl_pct'].mean()) if len(all_trades) - wins_1358 > 0 else 0
        
        avg_win_123 = trades_123[trades_123['pnl_pct'] > 0]['pnl_pct'].mean() if wins_123 > 0 else 0
        avg_loss_123 = abs(trades_123[trades_123['pnl_pct'] <= 0]['pnl_pct'].mean()) if len(trades_123) - wins_123 > 0 else 0
        
        avg_win_all = trades_all[trades_all['pnl_pct'] > 0]['pnl_pct'].mean() if wins_all > 0 else 0
        avg_loss_all = abs(trades_all[trades_all['pnl_pct'] <= 0]['pnl_pct'].mean()) if len(trades_all) - wins_all > 0 else 0
        
        print(f"\n  {'Metric':<20} {'[1,3,5,8]':<18} {'[1,2,3]':<18} {'ALL [1-10]':<18}")
        print(f"  {'-'*75}")
        print(f"  {'Horizons':<20} {'4 (diverse)':<18} {'3 (short)':<18} {'10 (all)':<18}")
        print(f"  {'Pairwise Slopes':<20} {'6':<18} {'3':<18} {'45':<18}")
        print(f"  {'Total Trades':<20} {len(all_trades):<18} {len(trades_123):<18} {len(trades_all):<18}")
        print(f"  {'Win Rate':<20} {wins_1358/len(all_trades)*100:.1f}%{'':<13} {wins_123/len(trades_123)*100:.1f}%{'':<13} {wins_all/len(trades_all)*100:.1f}%")
        print(f"  {'Total Return':<20} {all_trades['pnl_pct'].sum():+.1f}%{'':<13} {trades_123['pnl_pct'].sum():+.1f}%{'':<12} {trades_all['pnl_pct'].sum():+.1f}%")
        print(f"  {'Avg Win':<20} {avg_win_1358:+.2f}%{'':<13} {avg_win_123:+.2f}%{'':<13} {avg_win_all:+.2f}%")
        print(f"  {'Avg Loss':<20} {-avg_loss_1358:.2f}%{'':<13} {-avg_loss_123:.2f}%{'':<13} {-avg_loss_all:.2f}%")
        print(f"  {'Best Trade':<20} {all_trades['pnl_pct'].max():+.2f}%{'':<13} {trades_123['pnl_pct'].max():+.2f}%{'':<13} {trades_all['pnl_pct'].max():+.2f}%")
        print(f"  {'Worst Trade':<20} {all_trades['pnl_pct'].min():+.2f}%{'':<13} {trades_123['pnl_pct'].min():+.2f}%{'':<13} {trades_all['pnl_pct'].min():+.2f}%")
        
        # Winner
        returns = {
            '[1,3,5,8]': all_trades['pnl_pct'].sum(),
            '[1,2,3]': trades_123['pnl_pct'].sum(),
            'ALL [1-10]': trades_all['pnl_pct'].sum()
        }
        winner = max(returns, key=returns.get)
        
        print(f"\n  🏆 WINNER BY TOTAL RETURN: {winner} ({returns[winner]:+.1f}%)")
        
        # Risk-adjusted comparison
        print(f"\n  📊 RISK-ADJUSTED ANALYSIS:")
        for name, trades_df in [('[1,3,5,8]', all_trades), ('[1,2,3]', trades_123), ('ALL [1-10]', trades_all)]:
            if len(trades_df) > 0:
                ret = trades_df['pnl_pct'].sum()
                wins = len(trades_df[trades_df['pnl_pct'] > 0])
                wr = wins / len(trades_df) * 100
                avg_w = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
                avg_l = abs(trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean()) if len(trades_df) - wins > 0 else 0.001
                profit_factor = (wins * avg_w) / ((len(trades_df) - wins) * avg_l) if (len(trades_df) - wins) > 0 and avg_l > 0 else float('inf')
                print(f"    {name:<15} Profit Factor: {profit_factor:.2f}")

else:
    print("  No trades generated!")

print("\n" + "=" * 80)

