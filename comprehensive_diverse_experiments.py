"""
Comprehensive Diverse Horizon Experiments

BILL'S KEY INSIGHT:
- Short horizon combos [1,2,3] have great Sharpe but USELESS for hedging
- Why? A 1-3 day price move for crude oil is only ~$0.50 - meaningless for hedging desks
- Hedging desks need to know where price is going over 5-10 days
- We need DIVERSE HORIZONS: short (D+1-3) for direction confidence + long (D+7-10) for profit target

THIS ANALYSIS TESTS:
1. All combinations that include at least one horizon >= D+7
2. Different weighting methods (equal, inverse_spread, normalized_drift, spread, sqrt_spread, magnitude)
3. Different thresholds (0.1, 0.15, 0.2, 0.25, 0.3)
4. AVERAGE PROFIT PER TRADE IN DOLLARS (critical for hedging)

NOTE: Uses standardized Sharpe calculation from utils.metrics
"""

import pandas as pd
import numpy as np
import json
import os
from itertools import combinations
from datetime import datetime

# Use standardized metrics
from utils.metrics import calculate_sharpe_ratio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
ASSET_DIR = os.path.join(DATA_DIR, '1866_Crude_Oil')

# Key configurations
MIN_LONG_HORIZON = 7  # Must include at least one horizon >= 7
THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3]
MIN_TRADES = 10  # Minimum trades for statistical validity

# Priority combinations per Bill's request
PRIORITY_COMBOS = [
    [1, 3, 7],
    [1, 5, 10],
    [2, 5, 10],
    [1, 3, 5, 10],
    [1, 3, 7, 10],
    [1, 2, 7],
    [1, 2, 10],
    [2, 3, 7],
    [2, 3, 10],
    [1, 7],
    [1, 10],
    [2, 7],
    [2, 10],
    [3, 7],
    [3, 10],
    [1, 2, 3, 7],
    [1, 2, 3, 10],
    [1, 2, 3, 7, 10],
    [1, 5, 7],
    [2, 5, 7],
    [3, 5, 7],
    [1, 3, 5, 7],
    [2, 4, 7, 10],
    [1, 4, 7, 10],
]


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


# ============== SIGNAL CALCULATION METHODS ==============

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


def calculate_signals_spread_weighted(forecast_df, horizons, threshold):
    """Spread weighted: longer spreads get more weight (favors long-horizon signals)."""
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
                        weight = spread  # Longer spread = more weight

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


def calculate_signals_sqrt_spread_weighted(forecast_df, horizons, threshold):
    """Sqrt spread weighted: diminishing returns for longer spreads."""
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

        net_prob = weighted_sum / total_weight if total_weight > 0 else 0
        net_probs.append(net_prob)

        if net_prob > threshold:
            signals.append('BULLISH')
        elif net_prob < -threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def calculate_signals_magnitude_weighted(forecast_df, horizons, threshold):
    """Magnitude weighted: larger drifts get more weight."""
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

        net_prob = weighted_sum / total_magnitude if total_magnitude > 0 else 0
        net_probs.append(net_prob)

        if net_prob > threshold:
            signals.append('BULLISH')
        elif net_prob < -threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def calculate_signals_normalized_drift(forecast_df, horizons, threshold):
    """Normalized drift: per-day drift rate."""
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

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def calculate_signals_long_anchor(forecast_df, horizons, threshold):
    """Long anchor: use longest horizon as anchor, measure slopes from shorter horizons to it."""
    signals = []
    net_probs = []

    longest_h = max(horizons)
    shorter_horizons = [h for h in horizons if h < longest_h]

    for date in forecast_df.index:
        row = forecast_df.loc[date]
        slopes = []

        if longest_h in row.index and pd.notna(row[longest_h]):
            for h in shorter_horizons:
                if h in row.index and pd.notna(row[h]):
                    drift = row[longest_h] - row[h]
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


# ============== TRADING PERFORMANCE ==============

def calculate_trading_performance(signals, prices):
    """Calculate trading performance with detailed trade info."""
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
                    pnl_dollars = price - entry_price
                else:
                    pnl_dollars = entry_price - price

                pnl_pct = (pnl_dollars / entry_price) * 100
                holding_days = (date - entry_date).days

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl_dollars': pnl_dollars,
                    'pnl_pct': pnl_pct,
                    'holding_days': holding_days
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
    """Calculate comprehensive performance metrics including hedging-relevant metrics."""
    if not trades or len(trades) < MIN_TRADES:
        return None

    pnls_pct = [t['pnl_pct'] for t in trades]
    pnls_dollars = [t['pnl_dollars'] for t in trades]
    holding_days = [t['holding_days'] for t in trades]

    total_return = sum(pnls_pct)
    avg_return_pct = np.mean(pnls_pct)
    std_return = np.std(pnls_pct, ddof=1) if len(pnls_pct) > 1 else 0.001

    # Dollar metrics (CRITICAL FOR HEDGING)
    avg_profit_dollars = np.mean(pnls_dollars)
    avg_abs_profit_dollars = np.mean([abs(p) for p in pnls_dollars])

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    win_rate = (len(wins) / len(trades)) * 100 if trades else 0

    avg_win_dollars = np.mean([t['pnl_dollars'] for t in wins]) if wins else 0
    avg_loss_dollars = np.mean([abs(t['pnl_dollars']) for t in losses]) if losses else 0
    avg_win_pct = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss_pct = np.mean([abs(t['pnl_pct']) for t in losses]) if losses else 0

    # Sharpe ratio - use standardized calculation
    sharpe = calculate_sharpe_ratio(
        pnls_pct,
        holding_days=holding_days if holding_days else None,
        annualize=True
    )

    # Max drawdown
    cumulative = np.cumsum(pnls_pct)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_drawdown = min(drawdown) if len(drawdown) > 0 else 0

    # Profit factor
    gross_profit = sum(t['pnl_pct'] for t in wins) if wins else 0
    gross_loss = sum(abs(t['pnl_pct']) for t in losses) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(win_rate, 1),
        'total_return': round(total_return, 2),
        'avg_return_pct': round(avg_return_pct, 3),
        'sharpe': round(sharpe, 3),
        'max_drawdown': round(max_drawdown, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_hold_days': round(avg_hold_days, 1),
        # Hedging-critical metrics
        'avg_profit_dollars': round(avg_profit_dollars, 2),
        'avg_abs_profit_dollars': round(avg_abs_profit_dollars, 2),
        'avg_win_dollars': round(avg_win_dollars, 2),
        'avg_loss_dollars': round(avg_loss_dollars, 2),
        'avg_win_pct': round(avg_win_pct, 2),
        'avg_loss_pct': round(avg_loss_pct, 2),
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
            if any(h >= min_long for h in combo):
                combos.append(list(combo))
    return combos


def main():
    print("=" * 100)
    print("  COMPREHENSIVE DIVERSE HORIZON EXPERIMENTS")
    print("  Bill's Constraint: Must include horizon >= D+7 for meaningful hedging profit targets")
    print("=" * 100)

    # Load data
    print("\n[1] Loading forecast data...")
    forecast_df, prices, available_horizons = load_forecast_data(ASSET_DIR)

    if forecast_df is None:
        print("  ERROR: Could not load forecast data")
        return

    print(f"  Loaded {len(forecast_df)} days of data")
    print(f"  Available horizons: {available_horizons}")
    print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")

    # Define all signal methods
    signal_functions = {
        'equal_weight': calculate_signals_equal_weight,
        'inverse_spread': calculate_signals_inverse_spread_weighted,
        'spread_weighted': calculate_signals_spread_weighted,
        'sqrt_spread': calculate_signals_sqrt_spread_weighted,
        'magnitude_weighted': calculate_signals_magnitude_weighted,
        'normalized_drift': calculate_signals_normalized_drift,
        'long_anchor': calculate_signals_long_anchor,
    }

    # Generate all diverse combos
    all_diverse_combos = generate_diverse_combos(available_horizons, min_long=MIN_LONG_HORIZON, min_size=2, max_size=5)

    # Add priority combos (filter to available)
    priority_filtered = [[h for h in combo if h in available_horizons] for combo in PRIORITY_COMBOS]
    priority_filtered = [c for c in priority_filtered if len(c) >= 2 and any(h >= MIN_LONG_HORIZON for h in c)]

    # Combine and deduplicate
    all_combos = list(set(tuple(sorted(c)) for c in all_diverse_combos + priority_filtered))
    all_combos = [list(c) for c in all_combos]

    print(f"\n[2] Testing {len(all_combos)} diverse combinations with {len(signal_functions)} methods and {len(THRESHOLDS)} thresholds")
    print(f"  Total experiments: {len(all_combos) * len(signal_functions) * len(THRESHOLDS)}")

    # Run experiments
    print("\n[3] Running backtests...")
    results = []
    experiment_count = 0
    total_experiments = len(all_combos) * len(signal_functions) * len(THRESHOLDS)

    for combo in all_combos:
        for method_name, signal_func in signal_functions.items():
            for threshold in THRESHOLDS:
                metrics, _ = run_backtest(forecast_df, prices, combo, signal_func, threshold)
                experiment_count += 1

                if experiment_count % 500 == 0:
                    print(f"  Progress: {experiment_count}/{total_experiments} ({100*experiment_count/total_experiments:.1f}%)")

                if metrics:
                    results.append({
                        'horizons': combo,
                        'horizons_str': str(combo),
                        'method': method_name,
                        'threshold': threshold,
                        'has_short': any(h <= 3 for h in combo),
                        'has_medium': any(4 <= h <= 6 for h in combo),
                        'has_long': any(h >= 7 for h in combo),
                        'min_horizon': min(combo),
                        'max_horizon': max(combo),
                        'horizon_span': max(combo) - min(combo),
                        'num_horizons': len(combo),
                        **metrics
                    })

    print(f"\n  Completed {len(results)} valid experiments (min {MIN_TRADES} trades required)")

    # ============== ANALYSIS ==============

    # Sort by different criteria
    results_by_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    results_by_profit = sorted(results, key=lambda x: x['avg_profit_dollars'], reverse=True)
    results_by_return = sorted(results, key=lambda x: x['total_return'], reverse=True)

    # Filter for mixed combos (short + long)
    mixed_results = [r for r in results if r['has_short'] and r['has_long']]
    mixed_by_sharpe = sorted(mixed_results, key=lambda x: x['sharpe'], reverse=True)
    mixed_by_profit = sorted(mixed_results, key=lambda x: x['avg_profit_dollars'], reverse=True)

    # ============== DISPLAY RESULTS ==============

    print("\n" + "=" * 100)
    print("  SECTION A: TOP 25 BY SHARPE RATIO (All Diverse Combos)")
    print("=" * 100)
    print(f"\n  {'Rank':<5} {'Horizons':<18} {'Method':<18} {'Th':<5} {'Sharpe':<8} {'Return':<10} {'$/Trade':<10} {'HoldDays':<10} {'WinRate':<8} {'Trades':<7}")
    print("  " + "-" * 105)

    for i, r in enumerate(results_by_sharpe[:25]):
        print(f"  {i+1:<5} {r['horizons_str']:<18} {r['method']:<18} {r['threshold']:<5} {r['sharpe']:<8.2f} {r['total_return']:>+8.1f}% ${r['avg_profit_dollars']:<9.2f} {r['avg_hold_days']:<10.1f} {r['win_rate']:<8.1f} {r['total_trades']:<7}")

    print("\n" + "=" * 100)
    print("  SECTION B: TOP 25 BY AVERAGE PROFIT PER TRADE ($/trade) - CRITICAL FOR HEDGING")
    print("=" * 100)
    print(f"\n  {'Rank':<5} {'Horizons':<18} {'Method':<18} {'Th':<5} {'$/Trade':<10} {'Sharpe':<8} {'Return':<10} {'HoldDays':<10} {'WinRate':<8} {'Trades':<7}")
    print("  " + "-" * 105)

    for i, r in enumerate(results_by_profit[:25]):
        print(f"  {i+1:<5} {r['horizons_str']:<18} {r['method']:<18} {r['threshold']:<5} ${r['avg_profit_dollars']:<9.2f} {r['sharpe']:<8.2f} {r['total_return']:>+8.1f}% {r['avg_hold_days']:<10.1f} {r['win_rate']:<8.1f} {r['total_trades']:<7}")

    print("\n" + "=" * 100)
    print("  SECTION C: TOP 25 MIXED COMBOS (Short D+1-3 + Long D+7-10) BY SHARPE")
    print("  These combine direction confidence (short) with profit targets (long)")
    print("=" * 100)
    print(f"\n  {'Rank':<5} {'Horizons':<18} {'Method':<18} {'Th':<5} {'Sharpe':<8} {'Return':<10} {'$/Trade':<10} {'HoldDays':<10} {'WinRate':<8} {'Trades':<7}")
    print("  " + "-" * 105)

    for i, r in enumerate(mixed_by_sharpe[:25]):
        print(f"  {i+1:<5} {r['horizons_str']:<18} {r['method']:<18} {r['threshold']:<5} {r['sharpe']:<8.2f} {r['total_return']:>+8.1f}% ${r['avg_profit_dollars']:<9.2f} {r['avg_hold_days']:<10.1f} {r['win_rate']:<8.1f} {r['total_trades']:<7}")

    print("\n" + "=" * 100)
    print("  SECTION D: TOP 25 MIXED COMBOS BY AVERAGE PROFIT PER TRADE")
    print("=" * 100)
    print(f"\n  {'Rank':<5} {'Horizons':<18} {'Method':<18} {'Th':<5} {'$/Trade':<10} {'Sharpe':<8} {'Return':<10} {'HoldDays':<10} {'WinRate':<8} {'Trades':<7}")
    print("  " + "-" * 105)

    for i, r in enumerate(mixed_by_profit[:25]):
        print(f"  {i+1:<5} {r['horizons_str']:<18} {r['method']:<18} {r['threshold']:<5} ${r['avg_profit_dollars']:<9.2f} {r['sharpe']:<8.2f} {r['total_return']:>+8.1f}% {r['avg_hold_days']:<10.1f} {r['win_rate']:<8.1f} {r['total_trades']:<7}")

    # ============== PRIORITY COMBO ANALYSIS ==============

    print("\n" + "=" * 100)
    print("  SECTION E: PRIORITY COMBOS ANALYSIS (Bill's Specific Requests)")
    print("=" * 100)

    priority_combo_strs = [str(c) for c in priority_filtered]
    priority_results = [r for r in results if r['horizons_str'] in priority_combo_strs]

    # Best config for each priority combo
    best_per_combo = {}
    for r in priority_results:
        combo_str = r['horizons_str']
        if combo_str not in best_per_combo or r['sharpe'] > best_per_combo[combo_str]['sharpe']:
            best_per_combo[combo_str] = r

    print(f"\n  {'Combo':<20} {'Best Method':<18} {'Th':<5} {'Sharpe':<8} {'$/Trade':<10} {'Return':<10} {'HoldDays':<10} {'WinRate':<8}")
    print("  " + "-" * 100)

    for combo_str in sorted(best_per_combo.keys()):
        r = best_per_combo[combo_str]
        print(f"  {combo_str:<20} {r['method']:<18} {r['threshold']:<5} {r['sharpe']:<8.2f} ${r['avg_profit_dollars']:<9.2f} {r['total_return']:>+8.1f}% {r['avg_hold_days']:<10.1f} {r['win_rate']:<8.1f}")

    # ============== METHOD COMPARISON ==============

    print("\n" + "=" * 100)
    print("  SECTION F: METHOD COMPARISON (Average across all diverse combos)")
    print("=" * 100)

    method_stats = {}
    for method in signal_functions.keys():
        method_results = [r for r in results if r['method'] == method]
        if method_results:
            method_stats[method] = {
                'count': len(method_results),
                'avg_sharpe': np.mean([r['sharpe'] for r in method_results]),
                'avg_profit': np.mean([r['avg_profit_dollars'] for r in method_results]),
                'avg_return': np.mean([r['total_return'] for r in method_results]),
                'avg_win_rate': np.mean([r['win_rate'] for r in method_results]),
                'max_sharpe': max([r['sharpe'] for r in method_results]),
                'max_profit': max([r['avg_profit_dollars'] for r in method_results]),
            }

    print(f"\n  {'Method':<20} {'Experiments':<12} {'Avg Sharpe':<12} {'Max Sharpe':<12} {'Avg $/Trade':<12} {'Max $/Trade':<12} {'Avg Return':<12}")
    print("  " + "-" * 95)

    for method, stats in sorted(method_stats.items(), key=lambda x: x[1]['avg_sharpe'], reverse=True):
        print(f"  {method:<20} {stats['count']:<12} {stats['avg_sharpe']:<12.3f} {stats['max_sharpe']:<12.3f} ${stats['avg_profit']:<11.2f} ${stats['max_profit']:<11.2f} {stats['avg_return']:>+10.1f}%")

    # ============== THRESHOLD COMPARISON ==============

    print("\n" + "=" * 100)
    print("  SECTION G: THRESHOLD COMPARISON (Average across all diverse combos)")
    print("=" * 100)

    threshold_stats = {}
    for threshold in THRESHOLDS:
        th_results = [r for r in results if r['threshold'] == threshold]
        if th_results:
            threshold_stats[threshold] = {
                'count': len(th_results),
                'avg_sharpe': np.mean([r['sharpe'] for r in th_results]),
                'avg_profit': np.mean([r['avg_profit_dollars'] for r in th_results]),
                'avg_return': np.mean([r['total_return'] for r in th_results]),
                'avg_trades': np.mean([r['total_trades'] for r in th_results]),
            }

    print(f"\n  {'Threshold':<12} {'Experiments':<12} {'Avg Sharpe':<12} {'Avg $/Trade':<12} {'Avg Return':<12} {'Avg Trades':<12}")
    print("  " + "-" * 70)

    for th, stats in sorted(threshold_stats.items()):
        print(f"  {th:<12} {stats['count']:<12} {stats['avg_sharpe']:<12.3f} ${stats['avg_profit']:<11.2f} {stats['avg_return']:>+10.1f}% {stats['avg_trades']:<12.1f}")

    # ============== HOLDING PERIOD ANALYSIS ==============

    print("\n" + "=" * 100)
    print("  SECTION H: HOLDING PERIOD ANALYSIS (Critical for Hedging Utility)")
    print("=" * 100)

    # Group by holding days
    holding_buckets = {
        '1-2 days': [r for r in results if r['avg_hold_days'] <= 2],
        '3-4 days': [r for r in results if 2 < r['avg_hold_days'] <= 4],
        '5-7 days': [r for r in results if 4 < r['avg_hold_days'] <= 7],
        '8-10 days': [r for r in results if 7 < r['avg_hold_days'] <= 10],
        '10+ days': [r for r in results if r['avg_hold_days'] > 10],
    }

    print(f"\n  {'Holding Period':<15} {'Count':<10} {'Avg Sharpe':<12} {'Avg $/Trade':<12} {'Avg Return':<12}")
    print("  " + "-" * 60)

    for bucket, bucket_results in holding_buckets.items():
        if bucket_results:
            print(f"  {bucket:<15} {len(bucket_results):<10} {np.mean([r['sharpe'] for r in bucket_results]):<12.3f} ${np.mean([r['avg_profit_dollars'] for r in bucket_results]):<11.2f} {np.mean([r['total_return'] for r in bucket_results]):>+10.1f}%")

    # ============== SUMMARY ==============

    print("\n" + "=" * 100)
    print("  SUMMARY: BEST CONFIGURATIONS")
    print("=" * 100)

    if mixed_by_sharpe:
        best = mixed_by_sharpe[0]
        print(f"\n  BEST MIXED COMBO BY SHARPE:")
        print(f"    Horizons: {best['horizons']}")
        print(f"    Method: {best['method']}")
        print(f"    Threshold: {best['threshold']}")
        print(f"    Sharpe: {best['sharpe']:.3f}")
        print(f"    Return: {best['total_return']:+.2f}%")
        print(f"    Avg Profit/Trade: ${best['avg_profit_dollars']:.2f}")
        print(f"    Avg Holding: {best['avg_hold_days']:.1f} days")
        print(f"    Win Rate: {best['win_rate']:.1f}%")
        print(f"    Trades: {best['total_trades']}")

    if mixed_by_profit:
        best_profit = mixed_by_profit[0]
        print(f"\n  BEST MIXED COMBO BY $/TRADE (HEDGING UTILITY):")
        print(f"    Horizons: {best_profit['horizons']}")
        print(f"    Method: {best_profit['method']}")
        print(f"    Threshold: {best_profit['threshold']}")
        print(f"    Avg Profit/Trade: ${best_profit['avg_profit_dollars']:.2f}")
        print(f"    Sharpe: {best_profit['sharpe']:.3f}")
        print(f"    Return: {best_profit['total_return']:+.2f}%")
        print(f"    Avg Holding: {best_profit['avg_hold_days']:.1f} days")
        print(f"    Win Rate: {best_profit['win_rate']:.1f}%")
        print(f"    Trades: {best_profit['total_trades']}")

    # Save detailed results
    output_file = os.path.join(SCRIPT_DIR, 'comprehensive_diverse_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'asset': 'Crude_Oil',
            'data_days': len(forecast_df),
            'available_horizons': available_horizons,
            'min_long_horizon': MIN_LONG_HORIZON,
            'thresholds_tested': THRESHOLDS,
            'methods_tested': list(signal_functions.keys()),
            'total_experiments': len(results),
            'best_mixed_sharpe': mixed_by_sharpe[0] if mixed_by_sharpe else None,
            'best_mixed_profit': mixed_by_profit[0] if mixed_by_profit else None,
            'all_results': results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_file}")
    print("  Done!")

    return results


if __name__ == '__main__':
    main()
