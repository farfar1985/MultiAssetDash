"""
Multi-Asset Ensemble Experiments: Test best configurations across all assets.
"""

import pandas as pd
import numpy as np
import json
import os
from itertools import combinations
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# All available assets
ASSETS = {
    'Crude_Oil': '1866',
    'Bitcoin': '1860',
    'SP500': '1625',
    'NASDAQ': '269',
    'RUSSEL': '1518',
    'Nifty_50': '1398',
    'Nifty_Bank': '1387',
    'MCX_Copper': '1435',
    'SPDR_China_ETF': '291',
    'Nikkei_225': '358',
    'DOW_JONES_Mini': '336',
}


def load_forecast_data(asset_dir):
    """Load all horizon forecasts and prices."""
    horizons_data = {}
    for h in range(1, 201):
        forecast_file = os.path.join(asset_dir, f'forecast_d{h}.csv')
        if os.path.exists(forecast_file):
            df = pd.read_csv(forecast_file, parse_dates=['date'])
            if 'prediction' in df.columns and 'actual' in df.columns:
                horizons_data[h] = df[['date', 'prediction', 'actual']].copy()

    if not horizons_data:
        return None, None, []

    available_horizons = sorted(horizons_data.keys())

    base_df = None
    for h, df in horizons_data.items():
        df = df.rename(columns={'prediction': f'pred_{h}'})
        if base_df is None:
            base_df = df[['date', f'pred_{h}', 'actual']].copy()
        else:
            base_df = base_df.merge(df[['date', f'pred_{h}']], on='date', how='outer')

    base_df = base_df.sort_values('date').reset_index(drop=True)
    base_df = base_df.dropna(subset=['actual'])

    forecast_matrix = pd.DataFrame(index=pd.to_datetime(base_df['date']))
    for h in available_horizons:
        forecast_matrix[h] = base_df[f'pred_{h}'].values

    prices = base_df.set_index('date')['actual']
    prices.index = pd.to_datetime(prices.index)

    return forecast_matrix, prices, available_horizons


def calculate_signals_equal_weight(forecast_df, horizons, threshold):
    """Equal-weight pairwise slopes."""
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

    return pd.Series(signals, index=forecast_df.index)


def calculate_signals_inverse_spread_weighted(forecast_df, horizons, threshold):
    """Weight by inverse of horizon spread (shorter spreads = more weight)."""
    signals = []
    net_probs = []

    for date in forecast_df.index:
        row = forecast_df.loc[date]
        weighted_sum = 0.0
        total_weight = 0.0

        for i_idx, h1 in enumerate(horizons):
            for h2 in horizons[i_idx + 1:]:
                if h1 in row.index and h2 in row.index:
                    if pd.notna(row[h1]) and pd.notna(row[h2]):
                        drift = row[h2] - row[h1]
                        spread = h2 - h1
                        weight = 1.0 / spread

                        if drift > 0:
                            weighted_sum += weight
                        elif drift < 0:
                            weighted_sum -= weight
                        total_weight += weight

        if total_weight == 0:
            net_prob = 0.0
        else:
            net_prob = weighted_sum / total_weight

        net_probs.append(net_prob)

        if net_prob > threshold:
            signals.append('BULLISH')
        elif net_prob < -threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')

    return pd.Series(signals, index=forecast_df.index)


def calculate_signals_normalized_drift(forecast_df, horizons, threshold):
    """Normalize drift by horizon spread before voting."""
    signals = []
    net_probs = []

    for date in forecast_df.index:
        row = forecast_df.loc[date]
        normalized_drifts = []

        for i_idx, h1 in enumerate(horizons):
            for h2 in horizons[i_idx + 1:]:
                if h1 in row.index and h2 in row.index:
                    if pd.notna(row[h1]) and pd.notna(row[h2]):
                        drift = row[h2] - row[h1]
                        spread = h2 - h1
                        normalized = drift / spread
                        normalized_drifts.append(normalized)

        if len(normalized_drifts) == 0:
            net_prob = 0.0
        else:
            mean_drift = np.mean(normalized_drifts)
            net_prob = np.tanh(mean_drift * 10)

        net_probs.append(net_prob)

        if net_prob > threshold:
            signals.append('BULLISH')
        elif net_prob < -threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')

    return pd.Series(signals, index=forecast_df.index)


def calculate_trading_performance(signals, prices):
    """Signal-following strategy."""
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
                    'holding_days': (current_date - entry_date).days,
                })

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
    """Calculate comprehensive performance metrics."""
    if len(trades) < 3:
        return None

    total_return = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = sum(abs(t['pnl']) for t in losses) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)

    trade_returns = [t['pnl'] / 100 for t in trades]
    if len(trade_returns) > 1:
        avg_return = np.mean(trade_returns)
        std_return = np.std(trade_returns)
        avg_hold_days = np.mean([t['holding_days'] for t in trades])
        trades_per_year = 252 / avg_hold_days if avg_hold_days > 0 else 50
        sharpe = (avg_return / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0
    else:
        sharpe = 0

    equity = 100
    peak = equity
    max_dd = 0

    for trade in trades:
        equity *= (1 + trade['pnl'] / 100)
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        if drawdown > max_dd:
            max_dd = drawdown

    return {
        'total_return': round(total_return, 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(win_rate, 1),
        'total_trades': len(trades),
        'profit_factor': round(profit_factor, 2),
        'max_drawdown': round(-max_dd, 2),
    }


def run_backtest(forecast_df, prices, horizons, signal_func, threshold=0.1):
    """Run a single backtest."""
    signals = signal_func(forecast_df, horizons, threshold)
    trades = calculate_trading_performance(signals, prices)
    metrics = calculate_metrics(trades, prices)
    return metrics


def main():
    print("=" * 90)
    print("  MULTI-ASSET ENSEMBLE EXPERIMENTS")
    print("=" * 90)

    # Best configurations from Crude Oil experiments
    configs = [
        {'horizons': [1, 2, 3], 'method': 'inverse_spread_weighted', 'threshold': 0.2, 'name': 'Best Overall'},
        {'horizons': [1, 2, 3, 4], 'method': 'equal_weight', 'threshold': 0.0, 'name': 'Best Equal Weight'},
        {'horizons': [1, 3], 'method': 'normalized_drift', 'threshold': 0.2, 'name': 'Simple Pair'},
        {'horizons': [5, 7, 10], 'method': 'equal_weight', 'threshold': 0.1, 'name': 'User Baseline'},
    ]

    signal_funcs = {
        'equal_weight': calculate_signals_equal_weight,
        'inverse_spread_weighted': calculate_signals_inverse_spread_weighted,
        'normalized_drift': calculate_signals_normalized_drift,
    }

    all_results = []

    for asset_name, asset_id in ASSETS.items():
        asset_dir = os.path.join(DATA_DIR, f'{asset_id}_{asset_name}')
        if not os.path.exists(asset_dir):
            continue

        forecast_df, prices, available_horizons = load_forecast_data(asset_dir)
        if forecast_df is None or len(forecast_df) < 30:
            continue

        print(f"\n{'='*90}")
        print(f"  {asset_name} ({len(forecast_df)} days, horizons: {available_horizons})")
        print(f"{'='*90}")

        for config in configs:
            # Filter to available horizons
            horizons = [h for h in config['horizons'] if h in available_horizons]
            if len(horizons) < 2:
                print(f"  {config['name']}: SKIPPED (insufficient horizons)")
                continue

            signal_func = signal_funcs[config['method']]
            metrics = run_backtest(forecast_df, prices, horizons, signal_func, config['threshold'])

            if metrics:
                print(f"  {config['name']} {horizons}: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:+.1f}%, WR={metrics['win_rate']:.0f}%, DD={metrics['max_drawdown']:.1f}%")
                all_results.append({
                    'asset': asset_name,
                    'config': config['name'],
                    'horizons': horizons,
                    'method': config['method'],
                    'threshold': config['threshold'],
                    **metrics
                })
            else:
                print(f"  {config['name']}: Not enough trades")

    # Summary
    print("\n" + "=" * 90)
    print("  CROSS-ASSET SUMMARY")
    print("=" * 90)

    df = pd.DataFrame(all_results)
    if len(df) > 0:
        summary = df.groupby('config').agg({
            'sharpe': ['mean', 'std', 'min', 'max'],
            'total_return': 'mean',
            'win_rate': 'mean',
            'max_drawdown': 'mean',
        }).round(2)
        print("\n  Average Performance by Configuration:")
        print(summary.to_string())

        # Save results
        output_file = os.path.join(SCRIPT_DIR, 'multi_asset_results.json')
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': all_results,
            }, f, indent=2)
        print(f"\n  Results saved to {output_file}")


if __name__ == '__main__':
    main()
