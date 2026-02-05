"""
Ensemble Experiments: Testing horizon combinations and weighted pairwise slopes.

Context:
- Baseline: D+5+D+7+D+10 pairwise slopes = Sharpe 1.757
- Single-horizon ensembles are anti-predictive
- 10,179 models, 369 days data
"""

import pandas as pd
import numpy as np
import json
import os
from itertools import combinations
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'experiment_results.json')

# Primary test asset
ASSET_DIR = os.path.join(DATA_DIR, '1866_Crude_Oil')


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


def calculate_signals_equal_weight(forecast_df, horizons, threshold):
    """Original: equal-weight pairwise slopes."""
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


def calculate_signals_spread_weighted(forecast_df, horizons, threshold):
    """Weight slopes by horizon spread (longer spreads = more weight)."""
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
                        weight = spread  # Weight by horizon spread

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

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def calculate_signals_magnitude_weighted(forecast_df, horizons, threshold):
    """Weight slopes by drift magnitude (stronger signals = more weight)."""
    signals = []
    net_probs = []

    for date in forecast_df.index:
        row = forecast_df.loc[date]
        weighted_sum = 0.0
        total_magnitude = 0.0

        for i_idx, h1 in enumerate(horizons):
            for h2 in horizons[i_idx + 1:]:
                if h1 in row.index and h2 in row.index:
                    if pd.notna(row[h1]) and pd.notna(row[h2]):
                        drift = row[h2] - row[h1]
                        magnitude = abs(drift)

                        if drift > 0:
                            weighted_sum += magnitude
                        elif drift < 0:
                            weighted_sum -= magnitude
                        total_magnitude += magnitude

        if total_magnitude == 0:
            net_prob = 0.0
        else:
            net_prob = weighted_sum / total_magnitude

        net_probs.append(net_prob)

        if net_prob > threshold:
            signals.append('BULLISH')
        elif net_prob < -threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def calculate_signals_sqrt_spread_weighted(forecast_df, horizons, threshold):
    """Weight slopes by sqrt of horizon spread (diminishing returns for longer spreads)."""
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
                        weight = np.sqrt(spread)

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

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def calculate_signals_inverse_spread_weighted(forecast_df, horizons, threshold):
    """Weight slopes by inverse of horizon spread (shorter spreads = more weight)."""
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
                        weight = 1.0 / spread  # Inverse spread

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

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


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
                        normalized = drift / spread  # Per-day drift
                        normalized_drifts.append(normalized)

        if len(normalized_drifts) == 0:
            net_prob = 0.0
        else:
            # Use mean of normalized drifts
            mean_drift = np.mean(normalized_drifts)
            # Scale to [-1, 1] range (approx)
            net_prob = np.tanh(mean_drift * 10)  # Scale factor for sensitivity

        net_probs.append(net_prob)

        if net_prob > threshold:
            signals.append('BULLISH')
        elif net_prob < -threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


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

    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0

    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = sum(abs(t['pnl']) for t in losses) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)

    # Sharpe ratio (annualized)
    trade_returns = [t['pnl'] / 100 for t in trades]
    if len(trade_returns) > 1:
        avg_return = np.mean(trade_returns)
        std_return = np.std(trade_returns)
        avg_hold_days = np.mean([t['holding_days'] for t in trades])
        trades_per_year = 252 / avg_hold_days if avg_hold_days > 0 else 50
        sharpe = (avg_return / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
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
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'avg_hold_days': round(np.mean([t['holding_days'] for t in trades]), 1),
    }


def run_backtest(forecast_df, prices, horizons, signal_func, threshold=0.1):
    """Run a single backtest with given parameters."""
    signals, net_probs = signal_func(forecast_df, horizons, threshold)
    trades = calculate_trading_performance(signals, prices)
    metrics = calculate_metrics(trades, prices)
    return metrics, trades


def generate_horizon_combos(available_horizons, min_size=2, max_size=5):
    """Generate all horizon combinations."""
    combos = []
    for size in range(min_size, min(max_size + 1, len(available_horizons) + 1)):
        for combo in combinations(available_horizons, size):
            combos.append(list(combo))
    return combos


def main():
    print("=" * 80)
    print("  ENSEMBLE EXPERIMENTS: Horizon Combinations & Weighted Slopes")
    print("=" * 80)

    # Load data
    print("\n[1] Loading forecast data...")
    forecast_df, prices, available_horizons = load_forecast_data(ASSET_DIR)

    if forecast_df is None:
        print("  ERROR: Could not load forecast data")
        return

    print(f"  Loaded {len(forecast_df)} days of data")
    print(f"  Available horizons: {available_horizons}")
    print(f"  Date range: {forecast_df.index.min()} to {forecast_df.index.max()}")

    # Define signal functions to test
    signal_functions = {
        'equal_weight': calculate_signals_equal_weight,
        'spread_weighted': calculate_signals_spread_weighted,
        'sqrt_spread_weighted': calculate_signals_sqrt_spread_weighted,
        'inverse_spread_weighted': calculate_signals_inverse_spread_weighted,
        'magnitude_weighted': calculate_signals_magnitude_weighted,
        'normalized_drift': calculate_signals_normalized_drift,
    }

    # Define threshold values to test
    thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # Define specific horizon combos to test
    # Based on context: D+5, D+7, D+10 is baseline
    specific_combos = [
        [5, 7, 10],          # User's baseline
        [5, 10],             # Pair
        [7, 10],             # Pair
        [5, 7],              # Pair
        [3, 5, 7, 10],       # Extended
        [1, 5, 10],          # Short + medium + long
        [1, 3, 5, 7, 10],    # Full spread
        [3, 7, 10],          # Skip 5
        [5, 8, 10],          # Different medium
        [4, 7, 10],          # Different combination
        [6, 8, 10],          # Higher values
        [2, 5, 8],           # Different spread
        [3, 6, 9],           # Even spacing
        [4, 8],              # Large gap pair
        [3, 10],             # Largest gap pair
        [1, 2, 3, 4, 8],     # Current Crude Oil optimal
        available_horizons,  # All horizons
    ]

    # Filter to available horizons only
    specific_combos = [[h for h in combo if h in available_horizons] for combo in specific_combos]
    specific_combos = [combo for combo in specific_combos if len(combo) >= 2]
    specific_combos = list(set(tuple(sorted(combo)) for combo in specific_combos))  # Unique combos
    specific_combos = [list(combo) for combo in specific_combos]

    results = []

    # ==================== PHASE 1: Test User's Baseline ====================
    print("\n" + "=" * 80)
    print("  PHASE 1: Verifying Baseline (D+5, D+7, D+10)")
    print("=" * 80)

    baseline_horizons = [h for h in [5, 7, 10] if h in available_horizons]
    if len(baseline_horizons) == 3:
        for threshold in thresholds:
            metrics, _ = run_backtest(forecast_df, prices, baseline_horizons,
                                      calculate_signals_equal_weight, threshold)
            if metrics:
                results.append({
                    'phase': 'baseline_verification',
                    'horizons': baseline_horizons,
                    'signal_method': 'equal_weight',
                    'threshold': threshold,
                    **metrics
                })
                if threshold == 0.1:
                    print(f"  [D+5, D+7, D+10] @ threshold=0.1:")
                    print(f"    Sharpe: {metrics['sharpe']:.3f}")
                    print(f"    Return: {metrics['total_return']:+.2f}%")
                    print(f"    Win Rate: {metrics['win_rate']:.1f}%")
                    print(f"    Trades: {metrics['total_trades']}")
                    print(f"    Max DD: {metrics['max_drawdown']:.2f}%")
    else:
        print(f"  WARNING: Baseline horizons not all available. Have: {available_horizons}")

    # ==================== PHASE 2: Test All Horizon Combinations ====================
    print("\n" + "=" * 80)
    print("  PHASE 2: Testing Horizon Combinations")
    print("=" * 80)

    best_sharpe = -999
    best_combo = None

    for combo in specific_combos:
        for threshold in [0.1]:  # Test main threshold first
            metrics, _ = run_backtest(forecast_df, prices, combo,
                                      calculate_signals_equal_weight, threshold)
            if metrics:
                results.append({
                    'phase': 'horizon_combos',
                    'horizons': combo,
                    'signal_method': 'equal_weight',
                    'threshold': threshold,
                    **metrics
                })
                if metrics['sharpe'] > best_sharpe:
                    best_sharpe = metrics['sharpe']
                    best_combo = combo

    print(f"\n  Tested {len(specific_combos)} combinations")
    print(f"  Best Sharpe: {best_sharpe:.3f} with horizons {best_combo}")

    # ==================== PHASE 3: Test Weighted Pairwise Slopes ====================
    print("\n" + "=" * 80)
    print("  PHASE 3: Testing Weighted Pairwise Slopes")
    print("=" * 80)

    # Test on baseline and best combo
    test_combos = [baseline_horizons, best_combo] if best_combo != baseline_horizons else [baseline_horizons]
    test_combos = [c for c in test_combos if c]

    for combo in test_combos:
        print(f"\n  Testing horizons {combo}:")
        for method_name, signal_func in signal_functions.items():
            for threshold in thresholds:
                metrics, _ = run_backtest(forecast_df, prices, combo, signal_func, threshold)
                if metrics:
                    results.append({
                        'phase': 'weighted_slopes',
                        'horizons': combo,
                        'signal_method': method_name,
                        'threshold': threshold,
                        **metrics
                    })
                    if threshold == 0.1:
                        print(f"    {method_name}: Sharpe={metrics['sharpe']:.3f}, Return={metrics['total_return']:+.1f}%, WR={metrics['win_rate']:.1f}%")

    # ==================== PHASE 4: Generate All Combinations ====================
    print("\n" + "=" * 80)
    print("  PHASE 4: Exhaustive Combination Search")
    print("=" * 80)

    all_combos = generate_horizon_combos(available_horizons, min_size=2, max_size=5)
    print(f"  Testing {len(all_combos)} combinations...")

    combo_results = []
    for combo in all_combos:
        metrics, _ = run_backtest(forecast_df, prices, combo,
                                  calculate_signals_equal_weight, 0.1)
        if metrics:
            combo_results.append({
                'horizons': combo,
                **metrics
            })

    # Sort by Sharpe
    combo_results.sort(key=lambda x: x['sharpe'], reverse=True)

    print("\n  TOP 10 COMBINATIONS BY SHARPE:")
    print("  " + "-" * 70)
    for i, r in enumerate(combo_results[:10]):
        print(f"  {i+1}. {r['horizons']}: Sharpe={r['sharpe']:.3f}, Return={r['total_return']:+.1f}%, WR={r['win_rate']:.1f}%, Trades={r['total_trades']}")

    # Add to results
    for r in combo_results:
        results.append({
            'phase': 'exhaustive_search',
            'signal_method': 'equal_weight',
            'threshold': 0.1,
            **r
        })

    # ==================== PHASE 5: Best Combo with All Weighting Methods ====================
    print("\n" + "=" * 80)
    print("  PHASE 5: Best Combos with All Weighting Methods")
    print("=" * 80)

    top_combos = [r['horizons'] for r in combo_results[:5]]

    best_overall = None
    best_overall_sharpe = -999

    for combo in top_combos:
        print(f"\n  Horizons {combo}:")
        for method_name, signal_func in signal_functions.items():
            best_threshold_metrics = None
            best_threshold = None
            for threshold in thresholds:
                metrics, _ = run_backtest(forecast_df, prices, combo, signal_func, threshold)
                if metrics and (best_threshold_metrics is None or metrics['sharpe'] > best_threshold_metrics['sharpe']):
                    best_threshold_metrics = metrics
                    best_threshold = threshold

            if best_threshold_metrics:
                results.append({
                    'phase': 'top_combos_all_methods',
                    'horizons': combo,
                    'signal_method': method_name,
                    'threshold': best_threshold,
                    **best_threshold_metrics
                })
                print(f"    {method_name}: Sharpe={best_threshold_metrics['sharpe']:.3f} @ t={best_threshold}")

                if best_threshold_metrics['sharpe'] > best_overall_sharpe:
                    best_overall_sharpe = best_threshold_metrics['sharpe']
                    best_overall = {
                        'horizons': combo,
                        'method': method_name,
                        'threshold': best_threshold,
                        **best_threshold_metrics
                    }

    # ==================== SUMMARY ====================
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    if best_overall:
        print(f"\n  BEST CONFIGURATION FOUND:")
        print(f"    Horizons: {best_overall['horizons']}")
        print(f"    Method: {best_overall['method']}")
        print(f"    Threshold: {best_overall['threshold']}")
        print(f"    Sharpe: {best_overall['sharpe']:.3f}")
        print(f"    Return: {best_overall['total_return']:+.2f}%")
        print(f"    Win Rate: {best_overall['win_rate']:.1f}%")
        print(f"    Max Drawdown: {best_overall['max_drawdown']:.2f}%")
        print(f"    Total Trades: {best_overall['total_trades']}")
        print(f"    Profit Factor: {best_overall['profit_factor']:.2f}")

    # Save results
    print(f"\n  Saving {len(results)} results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'asset': 'Crude_Oil',
            'data_days': len(forecast_df),
            'available_horizons': available_horizons,
            'best_overall': best_overall,
            'results': results
        }, f, indent=2)

    print("  Done!")
    return results, best_overall


if __name__ == '__main__':
    main()
