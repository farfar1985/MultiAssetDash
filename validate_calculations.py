"""
Validation script to cross-check dashboard calculations
"""
import json
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'configs')

def load_and_validate(asset_name, data_folder, config_name):
    asset_dir = os.path.join(DATA_DIR, data_folder)
    config_path = os.path.join(CONFIG_DIR, config_name)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    viable = config.get('viable_horizons', [])
    threshold = config.get('threshold', 0.3)
    
    # Load horizon data
    horizons_data = {}
    for h in range(1, 11):
        fp = os.path.join(asset_dir, f'forecast_d{h}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp, parse_dates=['date'])
            horizons_data[h] = df
    
    available = sorted(horizons_data.keys())
    
    if not horizons_data:
        print(f'{asset_name}: No data found')
        return
    
    # Merge all horizons
    base_df = None
    for h, df in horizons_data.items():
        df = df[['date', 'prediction', 'actual']].copy()
        df = df.rename(columns={'prediction': f'pred_{h}', 'actual': 'actual'})
        if base_df is None:
            base_df = df
        else:
            base_df = base_df.merge(df[['date', f'pred_{h}']], on='date', how='outer')
    
    base_df = base_df.sort_values('date').dropna(subset=['actual'])
    
    # Calculate signals
    def calc_signals(df, horizons, thresh):
        signals = []
        net_probs = []
        for idx, row in df.iterrows():
            slopes = []
            for i, h1 in enumerate(horizons):
                for h2 in horizons[i+1:]:
                    p1 = row.get(f'pred_{h1}')
                    p2 = row.get(f'pred_{h2}')
                    if pd.notna(p1) and pd.notna(p2):
                        slopes.append(p2 - p1)
            
            if slopes:
                bullish = sum(1 for s in slopes if s > 0)
                bearish = sum(1 for s in slopes if s < 0)
                net_prob = (bullish - bearish) / len(slopes)
            else:
                net_prob = 0
            
            net_probs.append(net_prob)
            
            if net_prob > thresh:
                signals.append('BULLISH')
            elif net_prob < -thresh:
                signals.append('BEARISH')
            else:
                signals.append('NEUTRAL')
        
        return signals, net_probs
    
    # Calculate for RAW
    raw_signals, raw_net_probs = calc_signals(base_df, available, threshold)
    base_df['raw_signal'] = raw_signals
    base_df['raw_net_prob'] = raw_net_probs
    
    # Calculate for OPTIMIZED
    opt_horizons = [h for h in viable if h in available]
    if opt_horizons and len(opt_horizons) >= 2:
        opt_signals, opt_net_probs = calc_signals(base_df, opt_horizons, threshold)
        base_df['opt_signal'] = opt_signals
    else:
        base_df['opt_signal'] = raw_signals
        opt_horizons = available
    
    # Calculate signal-following trades
    def calc_trades(df, signal_col):
        trades = []
        current_pos = None
        prices = df['actual'].values
        signals = df[signal_col].values
        dates = df['date'].values
        
        for i in range(1, len(signals)):
            prev_sig = signals[i-1]
            curr_sig = signals[i]
            
            if prev_sig != curr_sig:
                # Close existing position
                if current_pos:
                    exit_price = prices[i]
                    if current_pos['signal'] == 'BULLISH':
                        pnl = ((exit_price - current_pos['entry_price']) / current_pos['entry_price']) * 100
                    else:
                        pnl = ((current_pos['entry_price'] - exit_price) / current_pos['entry_price']) * 100
                    trades.append({
                        'entry_date': current_pos['entry_date'],
                        'exit_date': dates[i],
                        'signal': current_pos['signal'],
                        'pnl': pnl
                    })
                    current_pos = None
                
                # Open new position
                if curr_sig in ['BULLISH', 'BEARISH']:
                    current_pos = {
                        'signal': curr_sig,
                        'entry_date': dates[i],
                        'entry_price': prices[i]
                    }
        
        return trades
    
    raw_trades = calc_trades(base_df, 'raw_signal')
    opt_trades = calc_trades(base_df, 'opt_signal')
    
    # Calculate metrics
    def calc_metrics(trades):
        if not trades:
            return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_return': 0, 'avg_pnl': 0}
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        total_pnl = sum(t['pnl'] for t in trades)
        
        return {
            'total': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'total_return': total_pnl,
            'avg_pnl': total_pnl / len(trades) if trades else 0
        }
    
    raw_metrics = calc_metrics(raw_trades)
    opt_metrics = calc_metrics(opt_trades)
    
    # Signal distribution
    raw_bull = raw_signals.count('BULLISH')
    raw_bear = raw_signals.count('BEARISH')
    raw_neut = raw_signals.count('NEUTRAL')
    
    opt_signals_list = base_df['opt_signal'].tolist()
    opt_bull = opt_signals_list.count('BULLISH')
    opt_bear = opt_signals_list.count('BEARISH')
    opt_neut = opt_signals_list.count('NEUTRAL')
    
    print(f'\n{"="*60}')
    print(f'=== {asset_name} ===')
    print(f'{"="*60}')
    print(f'Available horizons: {available}')
    print(f'Optimal horizons:   {opt_horizons}')
    print(f'Threshold:          {threshold}')
    print(f'Data points:        {len(base_df)}')
    print()
    print(f'--- Signal Distribution ---')
    print(f'RAW:  BULL={raw_bull:3d}, BEAR={raw_bear:3d}, NEUT={raw_neut:3d}')
    print(f'OPT:  BULL={opt_bull:3d}, BEAR={opt_bear:3d}, NEUT={opt_neut:3d}')
    print()
    print(f'--- Trade Metrics (Signal-Following) ---')
    print(f'RAW:  {raw_metrics["total"]:2d} trades, {raw_metrics["wins"]}W/{raw_metrics["losses"]}L, WinRate={raw_metrics["win_rate"]:.1f}%, Return={raw_metrics["total_return"]:.1f}%')
    print(f'OPT:  {opt_metrics["total"]:2d} trades, {opt_metrics["wins"]}W/{opt_metrics["losses"]}L, WinRate={opt_metrics["win_rate"]:.1f}%, Return={opt_metrics["total_return"]:.1f}%')
    print()
    print(f'--- Config Values ---')
    print(f'avg_da_raw:       {config.get("avg_da_raw", "N/A")}')
    print(f'avg_da_optimized: {config.get("avg_da_optimized", "N/A")}')
    
    # Show last 5 trades for verification
    print()
    print(f'--- Last 5 OPT Trades ---')
    for t in opt_trades[-5:]:
        print(f'  {t["entry_date"]} -> {t["signal"]}: {t["pnl"]:+.2f}%')
    
    return {
        'asset': asset_name,
        'raw': raw_metrics,
        'opt': opt_metrics
    }


if __name__ == '__main__':
    # Test key assets
    assets = [
        ('Crude_Oil', '1866_Crude_Oil', 'optimal_crude_oil.json'),
        ('SP500', '1625_SP500', 'optimal_sp500.json'),
        ('Bitcoin', '1860_Bitcoin', 'optimal_bitcoin.json'),
    ]
    
    results = []
    for name, folder, cfg in assets:
        try:
            r = load_and_validate(name, folder, cfg)
            if r:
                results.append(r)
        except Exception as e:
            print(f'{name}: ERROR - {e}')
            import traceback
            traceback.print_exc()
    
    print('\n' + '='*60)
    print('SUMMARY: Which version is better?')
    print('='*60)
    for r in results:
        raw_ret = r['raw']['total_return']
        opt_ret = r['opt']['total_return']
        better = 'OPT' if opt_ret > raw_ret else 'RAW'
        print(f"{r['asset']:15s}: RAW={raw_ret:+6.1f}%  OPT={opt_ret:+6.1f}%  -> {better} is better")

