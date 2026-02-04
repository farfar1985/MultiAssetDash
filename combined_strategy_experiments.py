"""
Combined Strategy Experiments: Seeking High Sharpe AND High $/Trade

Previous Gap Analysis Findings:
- Volatility Filter 70% (horizons [1,2,3]): Sharpe 8.64, $/trade $0.61
- Consecutive 3-day (horizons [1,3,5,8,10]): Sharpe 5.89, $/trade $1.84

Goal: Combine these approaches to achieve BOTH high Sharpe AND high $/trade
Also test across multiple assets: Crude Oil, Gold, Bitcoin, S&P500
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# Assets to test
ASSETS = {
    'Crude_Oil': '1866_Crude_Oil',
    'Gold': '477_GOLD',
    'Bitcoin': '1860_Bitcoin',
    'SP500': '1625_SP500',
}

MIN_TRADES = 8


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


# ============== BASE SIGNAL METHODS ==============

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


# ============== COMBINED STRATEGIES ==============

def calculate_signals_volatility_consecutive(forecast_df, horizons, threshold, prices,
                                              vol_window=20, vol_percentile=70, consecutive_days=2):
    """
    COMBINED STRATEGY 1: Volatility Filter + Consecutive Signal
    - First apply volatility filter (skip high-vol periods)
    - Then require consecutive days of same signal
    """
    # Calculate rolling volatility
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=vol_window).std()
    vol_threshold = rolling_vol.quantile(vol_percentile / 100)

    # Get base signals
    base_signals, net_probs = calculate_signals_inverse_spread(forecast_df, horizons, threshold)

    # First pass: Apply volatility filter
    vol_filtered = []
    for date in forecast_df.index:
        if date in rolling_vol.index:
            current_vol = rolling_vol.loc[date]
            if pd.notna(current_vol) and current_vol > vol_threshold:
                vol_filtered.append('NEUTRAL')
            else:
                vol_filtered.append(base_signals.loc[date])
        else:
            vol_filtered.append(base_signals.loc[date])

    vol_filtered = pd.Series(vol_filtered, index=forecast_df.index)

    # Second pass: Apply consecutive requirement
    final_signals = ['NEUTRAL'] * len(vol_filtered)

    for i in range(consecutive_days, len(vol_filtered)):
        current_signal = vol_filtered.iloc[i]

        if current_signal != 'NEUTRAL':
            all_match = True
            for j in range(1, consecutive_days + 1):
                if vol_filtered.iloc[i - j] != current_signal:
                    all_match = False
                    break

            if all_match:
                final_signals[i] = current_signal

    return pd.Series(final_signals, index=forecast_df.index), net_probs


def calculate_signals_strong_momentum(forecast_df, horizons, threshold, prices,
                                       vol_window=20, vol_percentile=80, min_prob=0.3):
    """
    COMBINED STRATEGY 2: Strong Momentum Filter
    - Low volatility environment
    - High confidence signal (prob > min_prob)
    """
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=vol_window).std()
    vol_threshold = rolling_vol.quantile(vol_percentile / 100)

    base_signals, net_probs = calculate_signals_inverse_spread(forecast_df, horizons, threshold)

    filtered_signals = []
    for i, date in enumerate(forecast_df.index):
        signal = base_signals.iloc[i]
        prob = abs(net_probs.iloc[i])

        # Check volatility
        low_vol = True
        if date in rolling_vol.index:
            current_vol = rolling_vol.loc[date]
            if pd.notna(current_vol) and current_vol > vol_threshold:
                low_vol = False

        # Check probability strength
        strong_signal = prob >= min_prob

        if low_vol and strong_signal and signal != 'NEUTRAL':
            filtered_signals.append(signal)
        else:
            filtered_signals.append('NEUTRAL')

    return pd.Series(filtered_signals, index=forecast_df.index), net_probs


def calculate_signals_triple_filter(forecast_df, horizons, threshold, prices,
                                     vol_window=20, vol_percentile=75, consecutive_days=2, min_prob=0.25):
    """
    COMBINED STRATEGY 3: Triple Filter (Vol + Consecutive + Probability)
    Most restrictive - should have highest quality trades
    """
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=vol_window).std()
    vol_threshold = rolling_vol.quantile(vol_percentile / 100)

    base_signals, net_probs = calculate_signals_inverse_spread(forecast_df, horizons, threshold)

    # Pass 1: Volatility filter
    pass1 = []
    for i, date in enumerate(forecast_df.index):
        signal = base_signals.iloc[i]
        if date in rolling_vol.index:
            current_vol = rolling_vol.loc[date]
            if pd.notna(current_vol) and current_vol > vol_threshold:
                pass1.append('NEUTRAL')
            else:
                pass1.append(signal)
        else:
            pass1.append(signal)

    pass1 = pd.Series(pass1, index=forecast_df.index)

    # Pass 2: Consecutive filter
    pass2 = ['NEUTRAL'] * len(pass1)
    for i in range(consecutive_days, len(pass1)):
        current_signal = pass1.iloc[i]
        if current_signal != 'NEUTRAL':
            all_match = True
            for j in range(1, consecutive_days + 1):
                if pass1.iloc[i - j] != current_signal:
                    all_match = False
                    break
            if all_match:
                pass2[i] = current_signal

    # Pass 3: Probability filter
    final_signals = []
    for i in range(len(pass2)):
        if pass2[i] != 'NEUTRAL' and abs(net_probs.iloc[i]) >= min_prob:
            final_signals.append(pass2[i])
        else:
            final_signals.append('NEUTRAL')

    return pd.Series(final_signals, index=forecast_df.index), net_probs


def calculate_signals_adaptive_consecutive(forecast_df, horizons, threshold, prices,
                                            vol_window=20, base_consecutive=2):
    """
    COMBINED STRATEGY 4: Adaptive Consecutive Days
    Low vol -> 2 consecutive, High vol -> 3 consecutive
    """
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=vol_window).std()
    median_vol = rolling_vol.median()

    base_signals, net_probs = calculate_signals_inverse_spread(forecast_df, horizons, threshold)

    final_signals = ['NEUTRAL'] * len(base_signals)

    for i in range(3, len(base_signals)):  # Need at least 3 for max consecutive
        current_signal = base_signals.iloc[i]
        date = forecast_df.index[i]

        if current_signal == 'NEUTRAL':
            continue

        # Determine required consecutive days based on volatility
        if date in rolling_vol.index and pd.notna(rolling_vol.loc[date]):
            vol_ratio = rolling_vol.loc[date] / median_vol if median_vol > 0 else 1
            required_consecutive = base_consecutive if vol_ratio < 1.2 else base_consecutive + 1
        else:
            required_consecutive = base_consecutive

        # Check consecutive requirement
        all_match = True
        for j in range(1, required_consecutive + 1):
            if i - j < 0 or base_signals.iloc[i - j] != current_signal:
                all_match = False
                break

        if all_match:
            final_signals[i] = current_signal

    return pd.Series(final_signals, index=forecast_df.index), net_probs


def calculate_signals_volatility_only(forecast_df, horizons, threshold, prices,
                                       vol_window=20, vol_percentile=70):
    """Baseline: Volatility filter only (for comparison)."""
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=vol_window).std()
    vol_threshold = rolling_vol.quantile(vol_percentile / 100)

    base_signals, net_probs = calculate_signals_inverse_spread(forecast_df, horizons, threshold)

    filtered_signals = []
    for date in forecast_df.index:
        if date in rolling_vol.index:
            current_vol = rolling_vol.loc[date]
            if pd.notna(current_vol) and current_vol > vol_threshold:
                filtered_signals.append('NEUTRAL')
            else:
                filtered_signals.append(base_signals.loc[date])
        else:
            filtered_signals.append(base_signals.loc[date])

    return pd.Series(filtered_signals, index=forecast_df.index), net_probs


def calculate_signals_consecutive_only(forecast_df, horizons, threshold, consecutive_days=2):
    """Baseline: Consecutive filter only (for comparison)."""
    base_signals, net_probs = calculate_signals_inverse_spread(forecast_df, horizons, threshold)

    filtered_signals = ['NEUTRAL'] * len(base_signals)

    for i in range(consecutive_days, len(base_signals)):
        current_signal = base_signals.iloc[i]

        if current_signal != 'NEUTRAL':
            all_match = True
            for j in range(1, consecutive_days + 1):
                if base_signals.iloc[i - j] != current_signal:
                    all_match = False
                    break

            if all_match:
                filtered_signals[i] = current_signal

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
    print("=" * 110)
    print("  COMBINED STRATEGY EXPERIMENTS: Seeking High Sharpe AND High $/Trade")
    print("=" * 110)

    all_results = []

    for asset_name, asset_folder in ASSETS.items():
        asset_dir = os.path.join(DATA_DIR, asset_folder)
        if not os.path.exists(asset_dir):
            print(f"\n  WARNING: {asset_name} data not found at {asset_dir}")
            continue

        print(f"\n{'='*110}")
        print(f"  ASSET: {asset_name}")
        print(f"{'='*110}")

        forecast_df, prices, available_horizons = load_forecast_data(asset_dir)

        if forecast_df is None:
            print("  ERROR: Could not load forecast data")
            continue

        print(f"  Loaded {len(forecast_df)} days of data")
        print(f"  Available horizons: {available_horizons[:20]}{'...' if len(available_horizons) > 20 else ''}")

        # Test configurations
        horizon_configs = [
            {'name': 'short', 'horizons': [1, 2, 3]},
            {'name': 'diverse', 'horizons': [1, 3, 5, 8, 10]},
            {'name': 'mixed', 'horizons': [1, 2, 3, 7, 10]},
        ]

        # Filter to available horizons
        for config in horizon_configs:
            config['horizons'] = [h for h in config['horizons'] if h in available_horizons]

        # ============== BASELINE METHODS ==============
        print("\n  --- BASELINE METHODS ---")

        for config in horizon_configs:
            if len(config['horizons']) < 2:
                continue

            # Base inverse spread
            metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                      calculate_signals_inverse_spread, 0.2)
            if metrics:
                print(f"  Base {config['name']}: Sharpe={metrics['sharpe']:.2f}, $/trade=${metrics['avg_profit_dollars']:.2f}, Trades={metrics['total_trades']}, WR={metrics['win_rate']:.1f}%")
                all_results.append({
                    'asset': asset_name,
                    'method': 'baseline',
                    'horizons': config['horizons'],
                    'config_name': config['name'],
                    **metrics
                })

        # ============== VOLATILITY ONLY ==============
        print("\n  --- VOLATILITY FILTER ONLY ---")

        for vol_pct in [70, 75, 80]:
            for config in horizon_configs:
                if len(config['horizons']) < 2:
                    continue

                metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                          calculate_signals_volatility_only, 0.2,
                                          vol_percentile=vol_pct)
                if metrics:
                    print(f"  Vol{vol_pct} {config['name']}: Sharpe={metrics['sharpe']:.2f}, $/trade=${metrics['avg_profit_dollars']:.2f}, Trades={metrics['total_trades']}")
                    all_results.append({
                        'asset': asset_name,
                        'method': f'vol_only_{vol_pct}',
                        'horizons': config['horizons'],
                        'config_name': config['name'],
                        **metrics
                    })

        # ============== CONSECUTIVE ONLY ==============
        print("\n  --- CONSECUTIVE FILTER ONLY ---")

        for consec in [2, 3]:
            for config in horizon_configs:
                if len(config['horizons']) < 2:
                    continue

                metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                          calculate_signals_consecutive_only, 0.2,
                                          consecutive_days=consec)
                if metrics:
                    print(f"  Consec{consec} {config['name']}: Sharpe={metrics['sharpe']:.2f}, $/trade=${metrics['avg_profit_dollars']:.2f}, Trades={metrics['total_trades']}, WR={metrics['win_rate']:.1f}%")
                    all_results.append({
                        'asset': asset_name,
                        'method': f'consec_only_{consec}',
                        'horizons': config['horizons'],
                        'config_name': config['name'],
                        **metrics
                    })

        # ============== COMBINED: VOL + CONSECUTIVE ==============
        print("\n  --- COMBINED: VOLATILITY + CONSECUTIVE ---")

        for vol_pct in [70, 75, 80]:
            for consec in [2, 3]:
                for config in horizon_configs:
                    if len(config['horizons']) < 2:
                        continue

                    metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                              calculate_signals_volatility_consecutive, 0.2,
                                              vol_percentile=vol_pct, consecutive_days=consec)
                    if metrics:
                        label = f"Vol{vol_pct}+Consec{consec}"
                        print(f"  {label} {config['name']}: Sharpe={metrics['sharpe']:.2f}, $/trade=${metrics['avg_profit_dollars']:.2f}, Trades={metrics['total_trades']}, WR={metrics['win_rate']:.1f}%")
                        all_results.append({
                            'asset': asset_name,
                            'method': f'vol{vol_pct}_consec{consec}',
                            'horizons': config['horizons'],
                            'config_name': config['name'],
                            **metrics
                        })

        # ============== STRONG MOMENTUM ==============
        print("\n  --- STRONG MOMENTUM FILTER ---")

        for vol_pct in [75, 80]:
            for min_prob in [0.3, 0.4]:
                for config in horizon_configs:
                    if len(config['horizons']) < 2:
                        continue

                    metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                              calculate_signals_strong_momentum, 0.2,
                                              vol_percentile=vol_pct, min_prob=min_prob)
                    if metrics:
                        label = f"Strong Vol{vol_pct} Prob{min_prob}"
                        print(f"  {label} {config['name']}: Sharpe={metrics['sharpe']:.2f}, $/trade=${metrics['avg_profit_dollars']:.2f}, Trades={metrics['total_trades']}")
                        all_results.append({
                            'asset': asset_name,
                            'method': f'strong_vol{vol_pct}_prob{min_prob}',
                            'horizons': config['horizons'],
                            'config_name': config['name'],
                            **metrics
                        })

        # ============== TRIPLE FILTER ==============
        print("\n  --- TRIPLE FILTER (Vol + Consec + Prob) ---")

        for config in horizon_configs:
            if len(config['horizons']) < 2:
                continue

            for vol_pct in [70, 75]:
                for consec in [2]:
                    for min_prob in [0.2, 0.25, 0.3]:
                        metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                                  calculate_signals_triple_filter, 0.2,
                                                  vol_percentile=vol_pct, consecutive_days=consec, min_prob=min_prob)
                        if metrics:
                            label = f"Triple V{vol_pct}C{consec}P{min_prob}"
                            print(f"  {label} {config['name']}: Sharpe={metrics['sharpe']:.2f}, $/trade=${metrics['avg_profit_dollars']:.2f}, Trades={metrics['total_trades']}, WR={metrics['win_rate']:.1f}%")
                            all_results.append({
                                'asset': asset_name,
                                'method': f'triple_v{vol_pct}c{consec}p{min_prob}',
                                'horizons': config['horizons'],
                                'config_name': config['name'],
                                **metrics
                            })

        # ============== ADAPTIVE CONSECUTIVE ==============
        print("\n  --- ADAPTIVE CONSECUTIVE ---")

        for config in horizon_configs:
            if len(config['horizons']) < 2:
                continue

            metrics, _ = run_backtest(forecast_df, prices, config['horizons'],
                                      calculate_signals_adaptive_consecutive, 0.2,
                                      base_consecutive=2)
            if metrics:
                print(f"  Adaptive {config['name']}: Sharpe={metrics['sharpe']:.2f}, $/trade=${metrics['avg_profit_dollars']:.2f}, Trades={metrics['total_trades']}, WR={metrics['win_rate']:.1f}%")
                all_results.append({
                    'asset': asset_name,
                    'method': 'adaptive_consec',
                    'horizons': config['horizons'],
                    'config_name': config['name'],
                    **metrics
                })

    # ============== CROSS-ASSET SUMMARY ==============
    print("\n" + "=" * 110)
    print("  CROSS-ASSET SUMMARY: TOP PERFORMERS")
    print("=" * 110)

    if all_results:
        df = pd.DataFrame(all_results)

        # Best by Sharpe
        print("\n  TOP 15 BY SHARPE:")
        print(f"  {'Asset':<12} {'Method':<25} {'Horizons':<18} {'Sharpe':<8} {'$/Trade':<10} {'Trades':<8} {'WR%':<8}")
        print("  " + "-" * 95)

        top_sharpe = df.nlargest(15, 'sharpe')
        for _, r in top_sharpe.iterrows():
            print(f"  {r['asset']:<12} {r['method']:<25} {str(r['horizons']):<18} {r['sharpe']:<8.2f} ${r['avg_profit_dollars']:<9.2f} {r['total_trades']:<8} {r['win_rate']:<8.1f}")

        # Best by $/Trade
        print("\n  TOP 15 BY $/TRADE:")
        print(f"  {'Asset':<12} {'Method':<25} {'Horizons':<18} {'$/Trade':<10} {'Sharpe':<8} {'Trades':<8} {'WR%':<8}")
        print("  " + "-" * 95)

        top_dollar = df.nlargest(15, 'avg_profit_dollars')
        for _, r in top_dollar.iterrows():
            print(f"  {r['asset']:<12} {r['method']:<25} {str(r['horizons']):<18} ${r['avg_profit_dollars']:<9.2f} {r['sharpe']:<8.2f} {r['total_trades']:<8} {r['win_rate']:<8.1f}")

        # Best COMBINED (high Sharpe AND high $/trade)
        print("\n  TOP 15 BALANCED (Sharpe * $/Trade score):")
        df['combined_score'] = df['sharpe'] * df['avg_profit_dollars']
        print(f"  {'Asset':<12} {'Method':<25} {'Horizons':<18} {'Score':<8} {'Sharpe':<8} {'$/Trade':<10} {'WR%':<8}")
        print("  " + "-" * 100)

        top_combined = df.nlargest(15, 'combined_score')
        for _, r in top_combined.iterrows():
            print(f"  {r['asset']:<12} {r['method']:<25} {str(r['horizons']):<18} {r['combined_score']:<8.2f} {r['sharpe']:<8.2f} ${r['avg_profit_dollars']:<9.2f} {r['win_rate']:<8.1f}")

        # Per-asset best
        print("\n  BEST PER ASSET (by combined score):")
        for asset in df['asset'].unique():
            asset_df = df[df['asset'] == asset]
            if len(asset_df) > 0:
                best = asset_df.loc[asset_df['combined_score'].idxmax()]
                print(f"  {asset}: {best['method']} {best['horizons']} -> Sharpe={best['sharpe']:.2f}, $/trade=${best['avg_profit_dollars']:.2f}, WR={best['win_rate']:.1f}%")

        # Save results
        output_file = os.path.join(SCRIPT_DIR, 'combined_strategy_results.json')
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'goal': 'Find strategies with BOTH high Sharpe AND high $/trade',
                'assets_tested': list(ASSETS.keys()),
                'total_experiments': len(all_results),
                'results': all_results
            }, f, indent=2, default=str)

        print(f"\n  Results saved to: {output_file}")

    print("\n  Done!")


if __name__ == '__main__':
    main()
