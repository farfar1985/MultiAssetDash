"""
Diverse Horizon Experiments: Testing combinations that REQUIRE at least one long horizon (>=D+7).

KEY INSIGHT FROM BILL:
- Short horizons (D+1 to D+3) only capture tiny price moves (~$0.50 for crude)
- Hedging desks need to know where price is going over 5-10 days
- We need DIVERSE horizons: short for direction confidence, long for profit target

CONSTRAINT: Every combination must include at least one horizon >= 7
"""

import pandas as pd
import numpy as np
import json
import os
from itertools import combinations
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
ASSET_DIR = os.path.join(DATA_DIR, '1866_Crude_Oil')

# Minimum long horizon required
MIN_LONG_HORIZON = 7


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


def calculate_signals_inverse_spread_weighted(forecast_df, horizons, threshold):
    """Inverse spread weighted: shorter spreads get more weight."""
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

        net_prob = weighted_sum / total_weight if total_weight > 0 else 0
        net_probs.append(net_prob)

        if net_prob > threshold:
            signals.append('BULLISH')
        elif net_prob < -threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def calculate_signals_equal_weight(forecast_df, horizons, threshold):
    """Equal weight pairwise slopes."""
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
    """Calculate trading performance from signals."""
    trades = []
    position = None
    entry_price = None
    entry_date = None

    for i, (date, signal) in enumerate(signals.items()):
        price = prices.loc[date] if date in prices.index else None
        if price is None:
            continue

        if position is None:
            if signal == 'BULLISH':
                position = 'LONG'
                entry_price = price
                entry_date = date
            elif signal == 'BEARISH':
                position = 'SHORT'
                entry_price = price
                entry_date = date
        else:
            exit_trade = False
            if position == 'LONG' and signal in ['BEARISH', 'NEUTRAL']:
                exit_trade = True
            elif position == 'SHORT' and signal in ['BULLISH', 'NEUTRAL']:
                exit_trade = True

            if exit_trade:
                if position == 'LONG':
                    pnl = price - entry_price
                else:
                    pnl = entry_price - price

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl': pnl,
                    'pnl_pct': (pnl / entry_price) * 100
                })

                if signal == 'BULLISH':
                    position = 'LONG'
                    entry_price = price
                    entry_date = date
                elif signal == 'BEARISH':
                    position = 'SHORT'
                    entry_price = price
                    entry_date = date
                else:
                    position = None
                    entry_price = None
                    entry_date = None

    return trades


def calculate_metrics(trades, prices):
    """Calculate performance metrics."""
    if not trades:
        return None

    pnls = [t['pnl_pct'] for t in trades]
    total_return = sum(pnls)
    avg_return = np.mean(pnls)
    std_return = np.std(pnls, ddof=1) if len(pnls) > 1 else 0.001

    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    win_rate = (wins / len(pnls)) * 100 if pnls else 0

    sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_drawdown = min(drawdown) if len(drawdown) > 0 else 0

    return {
        'total_trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_return': avg_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
    }


def run_backtest(forecast_df, prices, horizons, signal_func, threshold=0.1):
    """Run a single backtest."""
    signals, net_probs = signal_func(forecast_df, horizons, threshold)
    trades = calculate_trading_performance(signals, prices)
    metrics = calculate_metrics(trades, prices)
    return metrics, trades


def generate_diverse_combos(available_horizons, min_long=7, min_size=2, max_size=5):
    """Generate combinations that include at least one horizon >= min_long."""
    combos = []
    for size in range(min_size, min(max_size + 1, len(available_horizons) + 1)):
        for combo in combinations(available_horizons, size):
            # Must include at least one long horizon
            if any(h >= min_long for h in combo):
                combos.append(list(combo))
    return combos


def main():
    print("=" * 80)
    print("  DIVERSE HORIZON EXPERIMENTS")
    print("  Constraint: Must include at least one horizon >= D+7")
    print("=" * 80)

    # Load data
    print("\n[1] Loading forecast data...")
    forecast_df, prices, available_horizons = load_forecast_data(ASSET_DIR)

    if forecast_df is None:
        print("  ERROR: Could not load forecast data")
        return

    print(f"  Loaded {len(forecast_df)} days of data")
    print(f"  Available horizons: {available_horizons}")

    # Signal functions
    signal_functions = {
        'equal_weight': calculate_signals_equal_weight,
        'inverse_spread_weighted': calculate_signals_inverse_spread_weighted,
    }

    thresholds = [0.1, 0.15, 0.2, 0.25]

    # Generate diverse combos
    diverse_combos = generate_diverse_combos(available_horizons, min_long=MIN_LONG_HORIZON, min_size=2, max_size=4)
    print(f"\n[2] Generated {len(diverse_combos)} diverse combinations (require horizon >= {MIN_LONG_HORIZON})")

    # Run experiments
    print("\n[3] Running backtests...")
    results = []

    for combo in diverse_combos:
        for method_name, signal_func in signal_functions.items():
            for threshold in thresholds:
                metrics, _ = run_backtest(forecast_df, prices, combo, signal_func, threshold)
                if metrics and metrics['total_trades'] >= 10:  # Min trades filter
                    results.append({
                        'horizons': combo,
                        'method': method_name,
                        'threshold': threshold,
                        'has_short': any(h <= 3 for h in combo),
                        'has_long': any(h >= 7 for h in combo),
                        'max_horizon': max(combo),
                        **metrics
                    })

    print(f"  Completed {len(results)} valid experiments")

    # Sort by Sharpe
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    # Show top results
    print("\n" + "=" * 80)
    print("  TOP 20 DIVERSE HORIZON COMBINATIONS")
    print("  (All include at least one horizon >= D+7 for meaningful profit targets)")
    print("=" * 80)
    print(f"\n  {'Rank':<5} {'Horizons':<20} {'Method':<25} {'Th':<5} {'Sharpe':<8} {'Return':<10} {'WinRate':<8} {'Trades':<7}")
    print("  " + "-" * 95)

    for i, r in enumerate(results[:20]):
        print(f"  {i+1:<5} {str(r['horizons']):<20} {r['method']:<25} {r['threshold']:<5} {r['sharpe']:<8.2f} {r['total_return']:>+8.1f}% {r['win_rate']:<8.1f} {r['total_trades']:<7}")

    # Filter for combos with BOTH short and long
    mixed_results = [r for r in results if r['has_short'] and r['has_long']]
    mixed_results.sort(key=lambda x: x['sharpe'], reverse=True)

    print("\n" + "=" * 80)
    print("  TOP 20 MIXED HORIZON COMBINATIONS")
    print("  (Short horizons D+1-3 for direction + Long horizons D+7+ for profit)")
    print("=" * 80)
    print(f"\n  {'Rank':<5} {'Horizons':<20} {'Method':<25} {'Th':<5} {'Sharpe':<8} {'Return':<10} {'WinRate':<8} {'Trades':<7}")
    print("  " + "-" * 95)

    for i, r in enumerate(mixed_results[:20]):
        print(f"  {i+1:<5} {str(r['horizons']):<20} {r['method']:<25} {r['threshold']:<5} {r['sharpe']:<8.2f} {r['total_return']:>+8.1f}% {r['win_rate']:<8.1f} {r['total_trades']:<7}")

    # Save results
    output_file = os.path.join(SCRIPT_DIR, 'diverse_horizon_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    if mixed_results:
        best = mixed_results[0]
        print(f"\n  BEST DIVERSE CONFIG (short + long horizons):")
        print(f"    Horizons: {best['horizons']}")
        print(f"    Method: {best['method']}")
        print(f"    Threshold: {best['threshold']}")
        print(f"    Sharpe: {best['sharpe']:.2f}")
        print(f"    Return: {best['total_return']:+.1f}%")
        print(f"    Win Rate: {best['win_rate']:.1f}%")
        print(f"    Trades: {best['total_trades']}")


if __name__ == '__main__':
    main()
