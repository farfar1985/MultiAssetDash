"""
Pre-calculate RAW and OPTIMIZED metrics for all assets.
This runs in the pipeline and saves everything to config files.
Dashboard just reads these values - no on-the-fly calculation!
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(SCRIPT_DIR, 'configs')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# Asset mapping
ASSETS = {
    'Crude_Oil': {'id': '1866', 'name': 'Crude Oil'},
    'Bitcoin': {'id': '1860', 'name': 'Bitcoin'},
    'SP500': {'id': '1625', 'name': 'S&P 500'},
    'S&P_500': {'id': '1625', 'name': 'S&P 500'},  # Alias for compatibility
    'NASDAQ': {'id': '269', 'name': 'NASDAQ'},
    'RUSSEL': {'id': '1518', 'name': 'RUSSEL'},
    'DOW_JONES_Mini': {'id': '336', 'name': 'DOW JONES Mini'},
    'GOLD': {'id': '477', 'name': 'Gold'},
    'US_DOLLAR_Index': {'id': '655', 'name': 'US Dollar Index'},
    'SPDR_China_ETF': {'id': '291', 'name': 'SPDR China ETF'},
    'Nikkei_225': {'id': '358', 'name': 'Nikkei 225'},
    'Nifty_50': {'id': '1398', 'name': 'Nifty 50'},
    'Nifty_Bank': {'id': '1387', 'name': 'Nifty Bank'},
    'MCX_Copper': {'id': '1435', 'name': 'MCX Copper'},
    'USD_INR': {'id': '256', 'name': 'USD INR'},
    'Brent_Oil': {'id': '1859', 'name': 'Brent Oil'},
    'Gold': {'id': '1861', 'name': 'Gold'},  # Alias
    'Natural_Gas': {'id': '1862', 'name': 'Natural Gas'},
}


def load_forecast_data(asset_dir):
    """Load all horizon forecasts and prices."""
    horizons_data = {}
    # Check up to D+200 to handle all possible horizons
    for h in range(1, 201):
        forecast_file = os.path.join(asset_dir, f'forecast_d{h}.csv')
        if os.path.exists(forecast_file):
            df = pd.read_csv(forecast_file, parse_dates=['date'])
            if 'prediction' in df.columns and 'actual' in df.columns:
                horizons_data[h] = df[['date', 'prediction', 'actual']].copy()
    
    if not horizons_data:
        return None, None, []
    
    available_horizons = sorted(horizons_data.keys())
    
    # Build forecast matrix
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
    for h in available_horizons:
        forecast_matrix[h] = base_df[f'pred_{h}'].values
    
    prices = base_df.set_index('date')['actual']
    prices.index = pd.to_datetime(prices.index)
    
    return forecast_matrix, prices, available_horizons


def calculate_signals_pairwise_slopes(forecast_df, horizons, threshold):
    """Calculate signals using pairwise slopes (matrix drift) method."""
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


def calculate_trading_performance(signals, prices):
    """
    Signal-following strategy:
    - Enter on signal change (BULLISH/BEARISH)
    - Exit when signal flips or turns NEUTRAL
    """
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
            # No position - enter on BULLISH or BEARISH
            if current_signal in ['BULLISH', 'BEARISH']:
                position = current_signal
                entry_price = current_price
                entry_date = current_date
                entry_signal = current_signal
        else:
            # Have position - exit on signal change
            if current_signal != position:
                # Calculate P&L
                if entry_signal == 'BULLISH':
                    pnl = ((current_price - entry_price) / entry_price) * 100
                else:  # BEARISH
                    pnl = ((entry_price - current_price) / entry_price) * 100
                
                # Determine exit reason
                exit_reason = 'Signal → Neutral' if current_signal == 'NEUTRAL' else 'Signal Flip'
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'signal': entry_signal,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'holding_days': (current_date - entry_date).days,
                    'exit_reason': exit_reason
                })
                
                # Re-enter if new signal is not NEUTRAL
                if current_signal in ['BULLISH', 'BEARISH']:
                    position = current_signal
                    entry_price = current_price
                    entry_date = current_date
                    entry_signal = current_signal
                else:
                    position = None
                    entry_price = None
                    entry_date = None
                    entry_signal = None
    
    return trades


def calculate_metrics(trades, prices):
    """Calculate comprehensive performance metrics from trades."""
    if len(trades) < 3:
        return None
    
    # Basic metrics
    total_return = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    # Average win/loss
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0
    
    # Profit factor
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = sum(abs(t['pnl']) for t in losses) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
    
    # Sharpe ratio (annualized)
    trade_returns = [t['pnl'] / 100 for t in trades]
    if len(trade_returns) > 1:
        avg_return = np.mean(trade_returns)
        std_return = np.std(trade_returns)
        # Estimate trades per year (assuming ~252 trading days)
        avg_hold_days = np.mean([t['holding_days'] for t in trades])
        trades_per_year = 252 / avg_hold_days if avg_hold_days > 0 else 50
        sharpe = (avg_return / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0
    else:
        sharpe = 0
    
    # Max drawdown (from equity curve)
    equity = 100
    equity_curve = [equity]
    peak = equity
    max_dd = 0
    
    for trade in trades:
        equity *= (1 + trade['pnl'] / 100)
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        if drawdown > max_dd:
            max_dd = drawdown
    
    # Best/worst trade
    best_trade = max(t['pnl'] for t in trades) if trades else 0
    worst_trade = min(t['pnl'] for t in trades) if trades else 0
    
    return {
        'total_return': round(total_return, 2),
        'sharpe': round(sharpe, 2),
        'win_rate': round(win_rate, 1),
        'total_trades': len(trades),
        'profit_factor': round(profit_factor, 2),
        'max_drawdown': round(-max_dd, 2),  # Negative for consistency
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'best_trade': round(best_trade, 2),
        'worst_trade': round(worst_trade, 2),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2)
    }


def calculate_directional_accuracy(forecast_df, prices, horizons, threshold):
    """Calculate directional accuracy for given horizons."""
    signals, net_probs = calculate_signals_pairwise_slopes(forecast_df, horizons, threshold)
    
    correct = 0
    total = 0
    
    for i in range(1, len(signals)):
        if signals.iloc[i] == 'NEUTRAL':
            continue
        
        pred_direction = 1 if signals.iloc[i] == 'BULLISH' else -1
        actual_direction = 1 if prices.iloc[i] > prices.iloc[i-1] else -1
        
        if pred_direction == actual_direction:
            correct += 1
        total += 1
    
    return (correct / total * 100) if total > 0 else 0


def precalculate_asset_metrics(asset_name):
    """Pre-calculate RAW and OPTIMIZED metrics for one asset."""
    print(f"\n{'='*80}")
    print(f"  PRECALCULATING METRICS: {asset_name}")
    print(f"{'='*80}")
    
    asset_info = ASSETS.get(asset_name)
    if not asset_info:
        print(f"  [FAIL] Unknown asset: {asset_name}")
        return False
    
    asset_id = asset_info['id']
    asset_dir = os.path.join(DATA_DIR, f'{asset_id}_{asset_name}')
    config_file = os.path.join(CONFIGS_DIR, f'optimal_{asset_name.lower()}.json')
    
    if not os.path.exists(asset_dir):
        print(f"  [FAIL] Data directory not found: {asset_dir}")
        return False
    
    if not os.path.exists(config_file):
        print(f"  [FAIL] Config file not found: {config_file}")
        return False
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load forecast data
    forecast_df, prices, available_horizons = load_forecast_data(asset_dir)
    if forecast_df is None:
        print(f"  [FAIL] Could not load forecast data")
        return False
    
    print(f"  [OK] Loaded {len(forecast_df)} days of data")
    print(f"  [OK] Available horizons: {available_horizons}")
    
    # Get threshold
    threshold = config.get('threshold', 0.1)
    print(f"  [OK] Using threshold: {threshold}")
    
    # ==================== CALCULATE RAW METRICS (ALL HORIZONS) ====================
    print(f"\n  [INFO] Calculating RAW metrics (all {len(available_horizons)} horizons)...")
    raw_signals, raw_net_probs = calculate_signals_pairwise_slopes(forecast_df, available_horizons, threshold)
    raw_trades = calculate_trading_performance(raw_signals, prices)
    raw_metrics = calculate_metrics(raw_trades, prices)
    raw_da = calculate_directional_accuracy(forecast_df, prices, available_horizons, threshold)
    
    if raw_metrics:
        print(f"    [OK] Total Return: {raw_metrics['total_return']:+.1f}%")
        print(f"    [OK] Sharpe Ratio: {raw_metrics['sharpe']:.2f}")
        print(f"    [OK] Win Rate: {raw_metrics['win_rate']:.1f}%")
        print(f"    [OK] Total Trades: {raw_metrics['total_trades']}")
        print(f"    [OK] Directional Accuracy: {raw_da:.1f}%")
    else:
        print(f"    [FAIL] Not enough trades for RAW metrics")
        raw_metrics = {}
    
    # ==================== CALCULATE OPTIMIZED METRICS ====================
    optimal_horizons = config.get('viable_horizons', [])
    if not optimal_horizons:
        print(f"\n  [WARN] No optimal horizons in config - skipping optimized metrics")
        opt_metrics = None
        opt_da = None
    else:
        # Filter to only available horizons
        optimal_horizons = [h for h in optimal_horizons if h in available_horizons]
        if not optimal_horizons:
            print(f"\n  [WARN] Optimal horizons not available - skipping optimized metrics")
            opt_metrics = None
            opt_da = None
        else:
            print(f"\n  [INFO] Calculating OPTIMIZED metrics (horizons {optimal_horizons})...")
            opt_signals, opt_net_probs = calculate_signals_pairwise_slopes(forecast_df, optimal_horizons, threshold)
            opt_trades = calculate_trading_performance(opt_signals, prices)
            opt_metrics = calculate_metrics(opt_trades, prices)
            opt_da = calculate_directional_accuracy(forecast_df, prices, optimal_horizons, threshold)
            
            if opt_metrics:
                print(f"    [OK] Total Return: {opt_metrics['total_return']:+.1f}%")
                print(f"    [OK] Sharpe Ratio: {opt_metrics['sharpe']:.2f}")
                print(f"    [OK] Win Rate: {opt_metrics['win_rate']:.1f}%")
                print(f"    [OK] Total Trades: {opt_metrics['total_trades']}")
                print(f"    [OK] Directional Accuracy: {opt_da:.1f}%")
            else:
                print(f"    [FAIL] Not enough trades for OPTIMIZED metrics")
                opt_metrics = None
                opt_da = None
    
    # ==================== UPDATE CONFIG FILE ====================
    print(f"\n  [INFO] Updating config file...")
    
    # Update RAW metrics
    if raw_metrics:
        config['baseline_equity'] = raw_metrics['total_return']
        config['baseline_sharpe'] = raw_metrics['sharpe']
        config['baseline_win_rate'] = raw_metrics['win_rate']
        config['baseline_total_trades'] = raw_metrics['total_trades']
        config['baseline_profit_factor'] = raw_metrics['profit_factor']
        config['baseline_max_drawdown'] = raw_metrics['max_drawdown']
        config['avg_da_raw'] = round(raw_da, 1)
    
    # Update OPTIMIZED metrics
    if opt_metrics:
        config['optimized_equity'] = opt_metrics['total_return']
        config['sharpe_optimized'] = opt_metrics['sharpe']
        config['win_rate'] = opt_metrics['win_rate']
        config['total_trades'] = opt_metrics['total_trades']
        config['profit_factor'] = opt_metrics['profit_factor']
        config['max_drawdown_optimized'] = opt_metrics['max_drawdown']
        config['avg_da_optimized'] = round(opt_da, 1)
    
    # ==================== SAVE LAST 30 TRADES ====================
    def format_trade_for_json(trade):
        """Convert trade dict to JSON-serializable format."""
        return {
            'entry_date': trade['entry_date'].strftime('%Y-%m-%d'),
            'exit_date': trade['exit_date'].strftime('%Y-%m-%d'),
            'signal': trade['signal'],
            'entry_price': round(trade['entry_price'], 2),
            'exit_price': round(trade['exit_price'], 2),
            'pnl': round(trade['pnl'], 2),
            'holding_days': trade['holding_days'],
            'exit_reason': trade.get('exit_reason', 'Signal Flip')
        }
    
    # Save last 30 RAW trades
    if raw_trades and len(raw_trades) > 0:
        last_30_raw = raw_trades[-30:] if len(raw_trades) >= 30 else raw_trades
        config['last_30_trades_raw'] = [format_trade_for_json(t) for t in last_30_raw]
        print(f"  [OK] Saved {len(last_30_raw)} RAW trades (last 30)")
    
    # Save last 30 OPTIMIZED trades
    if opt_trades and len(opt_trades) > 0:
        last_30_opt = opt_trades[-30:] if len(opt_trades) >= 30 else opt_trades
        config['last_30_trades_optimized'] = [format_trade_for_json(t) for t in last_30_opt]
        print(f"  [OK] Saved {len(last_30_opt)} OPTIMIZED trades (last 30)")
    
    # Add timestamp
    config['metrics_calculated_at'] = datetime.now().isoformat()
    
    # Save config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"  [OK] Config file updated: {config_file}")
    
    # Summary
    print(f"\n  [INFO] SUMMARY:")
    if raw_metrics and opt_metrics:
        print(f"    RAW:      {raw_metrics['total_return']:+.1f}% return, {raw_metrics['sharpe']:.2f} Sharpe, {raw_da:.1f}% DA")
        print(f"    OPTIMIZED: {opt_metrics['total_return']:+.1f}% return, {opt_metrics['sharpe']:.2f} Sharpe, {opt_da:.1f}% DA")
        if opt_metrics['total_return'] > raw_metrics['total_return']:
            print(f"    [OK] OPTIMIZED is better by {opt_metrics['total_return'] - raw_metrics['total_return']:.1f}%")
        else:
            print(f"    [WARN] RAW is better by {raw_metrics['total_return'] - opt_metrics['total_return']:.1f}%")
    
    return True


def main():
    """Pre-calculate metrics for all assets."""
    import sys
    
    if len(sys.argv) > 1:
        # Calculate for specific asset
        asset_name = sys.argv[1]
        precalculate_asset_metrics(asset_name)
    else:
        # Calculate for all assets
        print("=" * 80)
        print("  PRECALCULATE METRICS FOR ALL ASSETS")
        print("=" * 80)
        
        for asset_name in ASSETS.keys():
            precalculate_asset_metrics(asset_name)
        
        print("\n" + "=" * 80)
        print("  [OK] ALL METRICS PRECALCULATED")
        print("=" * 80)


if __name__ == '__main__':
    main()

