"""
Gap Analysis Experiments: Testing Untested Ensemble Approaches

Based on review of ENSEMBLE_METHODS_RESEARCH.md, these promising approaches
have NOT been tested yet:

1. Ensemble of Ensembles - Combine top methods (inverse_spread + normalized_drift)
2. Confirmation-Based Entry - Require multiple signal methods to agree
3. Volatility Filtering - Filter signals during extreme volatility periods
4. Consecutive Signal Requirement - Require N consecutive signals before entry
5. Dynamic Threshold - Adjust threshold based on recent volatility
6. Hierarchical Voting - Vote within short horizons, then long, then combine
7. Price Target Exit - Use long horizon forecast as target, exit when reached
8. Momentum Alignment - Only trade when aligned with simple trend (price > SMA)
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

# Best configs from previous experiments
BEST_SHORT = [1, 2, 3]
BEST_MIXED = [1, 2, 3, 7, 10]
BEST_THRESHOLD = 0.25
MIN_TRADES = 10


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


# ============== BASE SIGNAL METHODS (from previous experiments) ==============

def calculate_signals_inverse_spread(forecast_df, horizons, threshold):
    """Inverse spread weighted - BEST METHOD from previous experiments."""
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


def calculate_signals_normalized_drift(forecast_df, horizons, threshold):
    """Normalized drift - SECOND BEST METHOD."""
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


# ============== NEW GAP-FILLING METHODS ==============

def calculate_signals_ensemble_of_ensembles(forecast_df, horizons, threshold):
    """
    GAP #1: Ensemble of Ensembles
    Combine inverse_spread and normalized_drift signals.
    Only signal when BOTH methods agree.
    """
    signals_inv, probs_inv = calculate_signals_inverse_spread(forecast_df, horizons, threshold)
    signals_norm, probs_norm = calculate_signals_normalized_drift(forecast_df, horizons, threshold)

    combined_signals = []
    combined_probs = []

    for i in range(len(signals_inv)):
        sig_inv = signals_inv.iloc[i]
        sig_norm = signals_norm.iloc[i]
        prob_inv = probs_inv.iloc[i]
        prob_norm = probs_norm.iloc[i]

        # Average the probabilities
        avg_prob = (prob_inv + prob_norm) / 2
        combined_probs.append(avg_prob)

        # Only signal when both agree
        if sig_inv == sig_norm and sig_inv != 'NEUTRAL':
            combined_signals.append(sig_inv)
        else:
            combined_signals.append('NEUTRAL')

    return pd.Series(combined_signals, index=forecast_df.index), pd.Series(combined_probs, index=forecast_df.index)


def calculate_signals_confirmation_entry(forecast_df, horizons, threshold):
    """
    GAP #2: Confirmation-Based Entry
    Use short horizons [1,2,3] for direction, require long horizon agreement.
    """
    short_horizons = [h for h in horizons if h <= 3]
    long_horizons = [h for h in horizons if h >= 7]

    if len(short_horizons) < 2 or len(long_horizons) < 1:
        # Fall back to regular method if horizons not available
        return calculate_signals_inverse_spread(forecast_df, horizons, threshold)

    signals = []
    net_probs = []

    for date in forecast_df.index:
        row = forecast_df.loc[date]

        # Get short-term direction
        short_slopes = []
        for i, h1 in enumerate(short_horizons):
            for h2 in short_horizons[i+1:]:
                if pd.notna(row.get(h1)) and pd.notna(row.get(h2)):
                    short_slopes.append(row[h2] - row[h1])

        short_bullish = sum(1 for s in short_slopes if s > 0)
        short_bearish = sum(1 for s in short_slopes if s < 0)
        short_direction = 'BULLISH' if short_bullish > short_bearish else ('BEARISH' if short_bearish > short_bullish else 'NEUTRAL')

        # Get long-term confirmation (from shortest to longest horizon)
        if len(long_horizons) >= 1 and len(short_horizons) >= 1:
            short_val = row.get(max(short_horizons))
            long_val = row.get(min(long_horizons))
            if pd.notna(short_val) and pd.notna(long_val):
                long_drift = long_val - short_val
                long_confirms = (long_drift > 0 and short_direction == 'BULLISH') or \
                               (long_drift < 0 and short_direction == 'BEARISH')
            else:
                long_confirms = False
        else:
            long_confirms = False

        # Calculate probability
        total_slopes = len(short_slopes)
        if total_slopes > 0:
            net_prob = (short_bullish - short_bearish) / total_slopes
        else:
            net_prob = 0

        net_probs.append(net_prob)

        # Only signal if confirmed
        if long_confirms and abs(net_prob) > threshold:
            signals.append(short_direction)
        else:
            signals.append('NEUTRAL')

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def calculate_signals_with_volatility_filter(forecast_df, horizons, threshold, prices, vol_window=20, vol_percentile=80):
    """
    GAP #3: Volatility Filtering
    Only trade when volatility is below a percentile threshold.
    High volatility = more noise, less reliable signals.
    """
    # Calculate rolling volatility
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=vol_window).std()
    vol_threshold = rolling_vol.quantile(vol_percentile / 100)

    # Get base signals
    base_signals, net_probs = calculate_signals_inverse_spread(forecast_df, horizons, threshold)

    # Filter by volatility
    filtered_signals = []
    for date in forecast_df.index:
        if date in rolling_vol.index:
            current_vol = rolling_vol.loc[date]
            if pd.notna(current_vol) and current_vol > vol_threshold:
                filtered_signals.append('NEUTRAL')  # Too volatile, skip
            else:
                filtered_signals.append(base_signals.loc[date])
        else:
            filtered_signals.append(base_signals.loc[date])

    return pd.Series(filtered_signals, index=forecast_df.index), net_probs


def calculate_signals_consecutive(forecast_df, horizons, threshold, consecutive_days=2):
    """
    GAP #4: Consecutive Signal Requirement
    Require N consecutive days of the same signal before entry.
    Reduces whipsaw trades.
    """
    base_signals, net_probs = calculate_signals_inverse_spread(forecast_df, horizons, threshold)

    filtered_signals = ['NEUTRAL'] * len(base_signals)

    for i in range(consecutive_days, len(base_signals)):
        current_signal = base_signals.iloc[i]

        if current_signal != 'NEUTRAL':
            # Check if previous N signals match
            all_match = True
            for j in range(1, consecutive_days + 1):
                if base_signals.iloc[i - j] != current_signal:
                    all_match = False
                    break

            if all_match:
                filtered_signals[i] = current_signal

    return pd.Series(filtered_signals, index=forecast_df.index), net_probs


def calculate_signals_dynamic_threshold(forecast_df, horizons, base_threshold, prices, vol_window=20):
    """
    GAP #5: Dynamic Threshold
    Adjust threshold based on recent volatility.
    Higher vol = higher threshold (more conservative)
    """
    # Calculate rolling volatility
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=vol_window).std()
    median_vol = rolling_vol.median()

    signals = []
    net_probs = []

    for date in forecast_df.index:
        row = forecast_df.loc[date]

        # Calculate net_prob using inverse spread
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

        # Dynamic threshold based on volatility
        if date in rolling_vol.index and pd.notna(rolling_vol.loc[date]) and median_vol > 0:
            vol_ratio = rolling_vol.loc[date] / median_vol
            dynamic_threshold = base_threshold * vol_ratio
            dynamic_threshold = max(0.1, min(0.5, dynamic_threshold))  # Clamp
        else:
            dynamic_threshold = base_threshold

        if net_prob > dynamic_threshold:
            signals.append('BULLISH')
        elif net_prob < -dynamic_threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def calculate_signals_hierarchical_voting(forecast_df, horizons, threshold):
    """
    GAP #6: Hierarchical Voting
    First vote among short horizons (1-4), then among long (7-10),
    then combine the votes.
    """
    short_horizons = [h for h in horizons if h <= 4]
    long_horizons = [h for h in horizons if h >= 7]

    if len(short_horizons) < 2 and len(long_horizons) < 2:
        return calculate_signals_inverse_spread(forecast_df, horizons, threshold)

    signals = []
    net_probs = []

    for date in forecast_df.index:
        row = forecast_df.loc[date]

        # Short horizon vote
        short_vote = 0
        short_count = 0
        for i, h1 in enumerate(short_horizons):
            for h2 in short_horizons[i+1:]:
                if pd.notna(row.get(h1)) and pd.notna(row.get(h2)):
                    drift = row[h2] - row[h1]
                    short_vote += 1 if drift > 0 else (-1 if drift < 0 else 0)
                    short_count += 1

        short_score = short_vote / short_count if short_count > 0 else 0

        # Long horizon vote
        long_vote = 0
        long_count = 0
        for i, h1 in enumerate(long_horizons):
            for h2 in long_horizons[i+1:]:
                if pd.notna(row.get(h1)) and pd.notna(row.get(h2)):
                    drift = row[h2] - row[h1]
                    long_vote += 1 if drift > 0 else (-1 if drift < 0 else 0)
                    long_count += 1

        long_score = long_vote / long_count if long_count > 0 else 0

        # Combined vote (weighted toward short for direction, long for confirmation)
        combined = 0.6 * short_score + 0.4 * long_score
        net_probs.append(combined)

        if combined > threshold:
            signals.append('BULLISH')
        elif combined < -threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')

    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def calculate_signals_momentum_alignment(forecast_df, horizons, threshold, prices, sma_window=10):
    """
    GAP #7: Momentum Alignment
    Only trade when signal aligns with simple trend (price vs SMA).
    """
    sma = prices.rolling(window=sma_window).mean()

    base_signals, net_probs = calculate_signals_inverse_spread(forecast_df, horizons, threshold)

    filtered_signals = []
    for date in forecast_df.index:
        signal = base_signals.loc[date]

        if date in sma.index and date in prices.index:
            price = prices.loc[date]
            ma = sma.loc[date]

            if pd.notna(price) and pd.notna(ma):
                # Check trend alignment
                trend_bullish = price > ma

                if signal == 'BULLISH' and trend_bullish:
                    filtered_signals.append('BULLISH')
                elif signal == 'BEARISH' and not trend_bullish:
                    filtered_signals.append('BEARISH')
                else:
                    filtered_signals.append('NEUTRAL')  # Misaligned
            else:
                filtered_signals.append(signal)
        else:
            filtered_signals.append(signal)

    return pd.Series(filtered_signals, index=forecast_df.index), net_probs


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
    """Calculate comprehensive performance metrics."""
    if not trades or len(trades) < MIN_TRADES:
        return None

    pnls_pct = [t['pnl_pct'] for t in trades]
    pnls_dollars = [t['pnl_dollars'] for t in trades]
    holding_days = [t['holding_days'] for t in trades]

    total_return = sum(pnls_pct)
    avg_return_pct = np.mean(pnls_pct)
    std_return = np.std(pnls_pct, ddof=1) if len(pnls_pct) > 1 else 0.001

    avg_profit_dollars = np.mean(pnls_dollars)

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    win_rate = (len(wins) / len(trades)) * 100 if trades else 0

    avg_win_pct = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss_pct = np.mean([abs(t['pnl_pct']) for t in losses]) if losses else 0

    avg_hold_days = np.mean(holding_days) if holding_days else 1
    trades_per_year = 252 / avg_hold_days if avg_hold_days > 0 else 50
    sharpe = (avg_return_pct / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0

    cumulative = np.cumsum(pnls_pct)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_drawdown = min(drawdown) if len(drawdown) > 0 else 0

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
        'avg_profit_dollars': round(avg_profit_dollars, 2),
        'avg_win_pct': round(avg_win_pct, 2),
        'avg_loss_pct': round(avg_loss_pct, 2),
    }


def run_backtest(forecast_df, prices, horizons, signal_func, threshold=0.2, **kwargs):
    """Run a single backtest."""
    if 'prices' in signal_func.__code__.co_varnames:
        signals, net_probs = signal_func(forecast_df, horizons, threshold, prices, **kwargs)
    else:
        signals, net_probs = signal_func(forecast_df, horizons, threshold, **kwargs)
    trades = calculate_trading_performance(signals, prices)
    metrics = calculate_metrics(trades, prices)
    return metrics, trades


def main():
    print("=" * 100)
    print("  GAP ANALYSIS EXPERIMENTS: Testing Untested Ensemble Approaches")
    print("=" * 100)

    # Load data
    print("\n[1] Loading forecast data...")
    forecast_df, prices, available_horizons = load_forecast_data(ASSET_DIR)

    if forecast_df is None:
        print("  ERROR: Could not load forecast data")
        return

    print(f"  Loaded {len(forecast_df)} days of data")
    print(f"  Available horizons: {available_horizons}")

    # Define test configurations
    horizon_configs = [
        {'name': 'short', 'horizons': [1, 2, 3]},
        {'name': 'mixed_7', 'horizons': [1, 2, 3, 7]},
        {'name': 'mixed_10', 'horizons': [1, 2, 3, 10]},
        {'name': 'mixed_full', 'horizons': [1, 2, 3, 7, 10]},
        {'name': 'diverse', 'horizons': [1, 3, 5, 8, 10]},
    ]

    thresholds = [0.15, 0.2, 0.25, 0.3]

    results = []

    # ============== GAP #1: Ensemble of Ensembles ==============
    print("\n" + "=" * 100)
    print("  GAP #1: Ensemble of Ensembles (inverse_spread + normalized_drift)")
    print("=" * 100)

    for config in horizon_configs:
        for threshold in thresholds:
            metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                      calculate_signals_ensemble_of_ensembles, threshold)
            if metrics:
                results.append({
                    'method': 'ensemble_of_ensembles',
                    'horizons': config['horizons'],
                    'threshold': threshold,
                    **metrics
                })
                if threshold == 0.2:
                    print(f"  {config['name']} @ {threshold}: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:+.1f}%, WR={metrics['win_rate']:.1f}%, $/trade=${metrics['avg_profit_dollars']:.2f}")

    # ============== GAP #2: Confirmation-Based Entry ==============
    print("\n" + "=" * 100)
    print("  GAP #2: Confirmation-Based Entry (short direction + long confirmation)")
    print("=" * 100)

    confirmation_configs = [
        {'name': 'mixed_7', 'horizons': [1, 2, 3, 7]},
        {'name': 'mixed_10', 'horizons': [1, 2, 3, 10]},
        {'name': 'mixed_full', 'horizons': [1, 2, 3, 7, 10]},
    ]

    for config in confirmation_configs:
        for threshold in thresholds:
            metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                      calculate_signals_confirmation_entry, threshold)
            if metrics:
                results.append({
                    'method': 'confirmation_entry',
                    'horizons': config['horizons'],
                    'threshold': threshold,
                    **metrics
                })
                if threshold == 0.2:
                    print(f"  {config['name']} @ {threshold}: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:+.1f}%, WR={metrics['win_rate']:.1f}%, $/trade=${metrics['avg_profit_dollars']:.2f}")

    # ============== GAP #3: Volatility Filtering ==============
    print("\n" + "=" * 100)
    print("  GAP #3: Volatility Filtering (skip high-vol periods)")
    print("=" * 100)

    vol_percentiles = [70, 80, 90]

    for config in horizon_configs:
        for vol_pct in vol_percentiles:
            metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                      calculate_signals_with_volatility_filter, 0.2,
                                      vol_window=20, vol_percentile=vol_pct)
            if metrics:
                results.append({
                    'method': f'volatility_filter_{vol_pct}',
                    'horizons': config['horizons'],
                    'threshold': 0.2,
                    **metrics
                })
                print(f"  {config['name']} vol<{vol_pct}%: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:+.1f}%, WR={metrics['win_rate']:.1f}%, Trades={metrics['total_trades']}")

    # ============== GAP #4: Consecutive Signal Requirement ==============
    print("\n" + "=" * 100)
    print("  GAP #4: Consecutive Signal Requirement (reduce whipsaw)")
    print("=" * 100)

    consecutive_days_list = [2, 3]

    for config in horizon_configs:
        for consec in consecutive_days_list:
            metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                      calculate_signals_consecutive, 0.2,
                                      consecutive_days=consec)
            if metrics:
                results.append({
                    'method': f'consecutive_{consec}',
                    'horizons': config['horizons'],
                    'threshold': 0.2,
                    **metrics
                })
                print(f"  {config['name']} {consec}-day: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:+.1f}%, WR={metrics['win_rate']:.1f}%, Trades={metrics['total_trades']}")

    # ============== GAP #5: Dynamic Threshold ==============
    print("\n" + "=" * 100)
    print("  GAP #5: Dynamic Threshold (vol-adjusted)")
    print("=" * 100)

    base_thresholds = [0.15, 0.2, 0.25]

    for config in horizon_configs:
        for base_th in base_thresholds:
            metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                      calculate_signals_dynamic_threshold, base_th,
                                      vol_window=20)
            if metrics:
                results.append({
                    'method': f'dynamic_threshold',
                    'horizons': config['horizons'],
                    'threshold': base_th,
                    **metrics
                })
                if base_th == 0.2:
                    print(f"  {config['name']} @ base={base_th}: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:+.1f}%, WR={metrics['win_rate']:.1f}%")

    # ============== GAP #6: Hierarchical Voting ==============
    print("\n" + "=" * 100)
    print("  GAP #6: Hierarchical Voting (short vote + long vote)")
    print("=" * 100)

    hierarchical_configs = [
        {'name': 'mixed_7', 'horizons': [1, 2, 3, 4, 7, 8]},
        {'name': 'mixed_10', 'horizons': [1, 2, 3, 4, 7, 8, 9, 10]},
        {'name': 'wide', 'horizons': [1, 2, 3, 4, 8, 9, 10]},
    ]

    for config in hierarchical_configs:
        for threshold in thresholds:
            metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                      calculate_signals_hierarchical_voting, threshold)
            if metrics:
                results.append({
                    'method': 'hierarchical_voting',
                    'horizons': config['horizons'],
                    'threshold': threshold,
                    **metrics
                })
                if threshold == 0.2:
                    print(f"  {config['name']} @ {threshold}: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:+.1f}%, WR={metrics['win_rate']:.1f}%, $/trade=${metrics['avg_profit_dollars']:.2f}")

    # ============== GAP #7: Momentum Alignment ==============
    print("\n" + "=" * 100)
    print("  GAP #7: Momentum Alignment (signal + trend alignment)")
    print("=" * 100)

    sma_windows = [5, 10, 20]

    for config in horizon_configs:
        for sma in sma_windows:
            metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                      calculate_signals_momentum_alignment, 0.2,
                                      sma_window=sma)
            if metrics:
                results.append({
                    'method': f'momentum_alignment_sma{sma}',
                    'horizons': config['horizons'],
                    'threshold': 0.2,
                    **metrics
                })
                print(f"  {config['name']} SMA{sma}: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:+.1f}%, WR={metrics['win_rate']:.1f}%, Trades={metrics['total_trades']}")

    # ============== BASELINE COMPARISON ==============
    print("\n" + "=" * 100)
    print("  BASELINE COMPARISON (inverse_spread from previous experiments)")
    print("=" * 100)

    for config in horizon_configs:
        metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                  calculate_signals_inverse_spread, 0.2)
        if metrics:
            results.append({
                'method': 'baseline_inverse_spread',
                'horizons': config['horizons'],
                'threshold': 0.2,
                **metrics
            })
            print(f"  {config['name']}: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:+.1f}%, WR={metrics['win_rate']:.1f}%, $/trade=${metrics['avg_profit_dollars']:.2f}")

    # ============== SUMMARY ==============
    print("\n" + "=" * 100)
    print("  SUMMARY: TOP RESULTS BY SHARPE")
    print("=" * 100)

    sorted_results = sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True)

    print(f"\n  {'Rank':<5} {'Method':<30} {'Horizons':<20} {'Sharpe':<8} {'Return':<10} {'$/Trade':<10} {'WinRate':<8}")
    print("  " + "-" * 100)

    for i, r in enumerate(sorted_results[:20]):
        print(f"  {i+1:<5} {r['method']:<30} {str(r['horizons']):<20} {r['sharpe']:<8.2f} {r['total_return']:>+8.1f}% ${r['avg_profit_dollars']:<9.2f} {r['win_rate']:<8.1f}")

    print("\n" + "=" * 100)
    print("  TOP RESULTS BY $/TRADE (HEDGING UTILITY)")
    print("=" * 100)

    sorted_by_dollar = sorted(results, key=lambda x: x.get('avg_profit_dollars', 0), reverse=True)

    print(f"\n  {'Rank':<5} {'Method':<30} {'Horizons':<20} {'$/Trade':<10} {'Sharpe':<8} {'Return':<10} {'WinRate':<8}")
    print("  " + "-" * 100)

    for i, r in enumerate(sorted_by_dollar[:20]):
        print(f"  {i+1:<5} {r['method']:<30} {str(r['horizons']):<20} ${r['avg_profit_dollars']:<9.2f} {r['sharpe']:<8.2f} {r['total_return']:>+8.1f}% {r['win_rate']:<8.1f}")

    # Save results
    output_file = os.path.join(SCRIPT_DIR, 'gap_analysis_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'asset': 'Crude_Oil',
            'data_days': len(forecast_df),
            'gap_methods_tested': [
                'ensemble_of_ensembles',
                'confirmation_entry',
                'volatility_filter',
                'consecutive_signal',
                'dynamic_threshold',
                'hierarchical_voting',
                'momentum_alignment'
            ],
            'total_experiments': len(results),
            'results': results
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_file}")
    print("  Done!")


if __name__ == '__main__':
    main()
