"""
QDT Ensemble - Multi-Asset Dashboard Builder
Creates a unified dark-themed dashboard with asset dropdown selector.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

EXPERIMENT_ROOT = os.path.abspath(os.path.dirname(__file__))

# =============================================================================
# ASSET CONFIGURATIONS (with optimized parameters from grid search)
# =============================================================================

ASSETS = {
    "Crude_Oil": {
        "id": "1866",
        "threshold": 0.15,  # Optimized 2026-01-08
        "rsi_overbought": 65,
        "rsi_oversold": 25,
        "accuracy": 55.0,  # Calculated dynamically
        "edge": 5.0,
        "color": "#1a1a1a"  # Oil black
    },
    "Bitcoin": {
        "id": "1860",
        "threshold": 0.35,  # Optimized 2026-01-08
        "rsi_overbought": 65,
        "rsi_oversold": 30,
        "accuracy": 64.4,
        "edge": 14.4,
        "color": "#f7931a"  # Bitcoin orange
    },
    "SP500": {
        "id": "1625",
        "threshold": 0.10,  # Optimized 2026-01-08
        "rsi_overbought": 70,
        "rsi_oversold": 35,
        "accuracy": 42.0,
        "edge": -8.0,
        "color": "#2563eb"  # Blue
    },
    "NASDAQ": {
        "id": "269",
        "threshold": 0.40,  # Optimized 2026-01-08
        "rsi_overbought": 70,
        "rsi_oversold": 20,
        "accuracy": 42.4,
        "edge": -7.6,
        "color": "#00c805"  # NASDAQ green
    },
    "RUSSEL": {
        "id": "1518",
        "threshold": 0.40,  # Optimized 2026-01-08
        "rsi_overbought": 65,
        "rsi_oversold": 35,
        "accuracy": 45.7,
        "edge": -4.3,
        "color": "#dc2626"  # Red
    },
    "DOW_JONES_Mini": {
        "id": "336",
        "threshold": 0.35,  # Optimized 2026-01-08
        "rsi_overbought": 65,
        "rsi_oversold": 20,
        "accuracy": 43.2,
        "edge": -6.8,
        "color": "#0ea5e9"  # Light blue
    },
    "GOLD": {
        "id": "477",
        "threshold": 0.35,  # Optimized 2026-01-08
        "rsi_overbought": 80,
        "rsi_oversold": 20,
        "accuracy": 43.3,
        "edge": -6.7,
        "color": "#fbbf24"  # Gold
    },
    "US_DOLLAR_Index": {
        "id": "655",
        "threshold": 0.35,  # Optimized 2026-01-08 - BEST PERFORMER!
        "rsi_overbought": 65,
        "rsi_oversold": 20,
        "accuracy": 70.5,
        "edge": 20.5,
        "color": "#22c55e"  # Green
    },
    "SPDR_China_ETF": {
        "id": "291",
        "threshold": 0.10,  # Optimized 2026-01-08
        "rsi_overbought": 65,
        "rsi_oversold": 20,
        "accuracy": 43.4,
        "edge": -6.6,
        "color": "#e11d48"  # China red
    },
    "Nikkei_225": {
        "id": "358",
        "threshold": 0.10,  # Optimized 2026-01-08
        "rsi_overbought": 65,
        "rsi_oversold": 30,
        "accuracy": 47.8,
        "edge": -2.2,
        "color": "#bc002d"  # Japan red
    },
    "Nifty_50": {
        "id": "1398",
        "threshold": 0.40,  # Optimized 2026-01-08
        "rsi_overbought": 65,
        "rsi_oversold": 35,
        "accuracy": 38.6,
        "edge": -11.4,
        "color": "#ff9933"  # India saffron
    },
    "Nifty_Bank": {
        "id": "1387",
        "threshold": 0.20,  # Optimized 2026-01-08
        "rsi_overbought": 70,
        "rsi_oversold": 25,
        "accuracy": 45.9,
        "edge": -4.1,
        "color": "#138808"  # India green
    },
    "MCX_Copper": {
        "id": "1435",
        "threshold": 0.25,  # Optimized 2026-01-08
        "rsi_overbought": 65,
        "rsi_oversold": 20,
        "accuracy": 45.7,
        "edge": -4.3,
        "color": "#b87333"  # Copper color
    },
    "USD_INR": {
        "id": "256",
        "threshold": 0.30,  # Optimized 2026-01-08
        "rsi_overbought": 80,
        "rsi_oversold": 35,
        "accuracy": 34.8,
        "edge": -15.2,
        "color": "#1e3a8a"  # Deep blue (forex)
    },
    "Brent_Oil": {
        "id": "1859",
        "threshold": 0.30,  # Default - optimize later
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "accuracy": 50.0,  # Will be calculated dynamically
        "edge": 0.0,
        "color": "#9333ea"  # Purple
    }
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_forecast_data(data_dir):
    """Load all horizon forecasts - DYNAMICALLY discovers all available horizons."""
    import glob
    import re
    
    horizons = {}
    
    # Dynamically find all forecast_d*.csv files
    forecast_pattern = os.path.join(data_dir, 'forecast_d*.csv')
    forecast_files = glob.glob(forecast_pattern)
    
    for path in forecast_files:
        # Extract horizon number from filename (e.g., forecast_d21.csv -> 21)
        filename = os.path.basename(path)
        match = re.search(r'forecast_d(\d+)\.csv', filename)
        if match:
            h = int(match.group(1))
            df = pd.read_csv(path, parse_dates=['date'])
            df = df.set_index('date')
            horizons[h] = df['prediction']
    
    if not horizons:
        return None, []
    
    print(f"  [INFO] Loaded {len(horizons)} horizons: {sorted(horizons.keys())}")
    
    df = pd.DataFrame(horizons)
    # Only forward-fill to avoid data leakage from future values
    # bfill() was removed as it leaks future information backward in time
    df = df.ffill()
    return df, sorted(horizons.keys())


def load_model_counts(data_dir):
    """Load model counts per horizon from horizons_wide folder."""
    import glob
    import re
    import joblib
    
    horizons_dir = os.path.join(data_dir, 'horizons_wide')
    model_counts = {}
    
    if not os.path.exists(horizons_dir):
        return model_counts
    
    # Find all horizon_*.joblib files
    horizon_pattern = os.path.join(horizons_dir, 'horizon_*.joblib')
    horizon_files = glob.glob(horizon_pattern)
    
    for path in horizon_files:
        filename = os.path.basename(path)
        match = re.search(r'horizon_(\d+)\.joblib', filename)
        if match:
            h = int(match.group(1))
            try:
                data = joblib.load(path)
                # data is typically a dict with 'X' (DataFrame) and 'y' (Series)
                if isinstance(data, dict) and 'X' in data:
                    X = data['X']
                    if hasattr(X, 'columns'):
                        model_counts[h] = len(X.columns)
                    elif hasattr(X, 'shape') and len(X.shape) > 1:
                        model_counts[h] = X.shape[1]
                    else:
                        model_counts[h] = 1
                else:
                    model_counts[h] = 1
            except Exception as e:
                model_counts[h] = 0
    
    return model_counts


def load_price_data(data_dir):
    """Load price history."""
    price_path = os.path.join(data_dir, 'price_history_cache.json')
    if not os.path.exists(price_path):
        return None
        
    with open(price_path, 'r') as f:
        price_data = json.load(f)
    
    if isinstance(price_data, list):
        df = pd.DataFrame(price_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        prices = df['close'].sort_index()
    else:
        prices = pd.Series(price_data)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
    
    # Ensure timezone naive
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    
    return prices


def load_live_forecast(data_dir):
    """Load live forecast data and convert to standard format."""
    live_path = os.path.join(data_dir, 'live_forecast.json')
    if not os.path.exists(live_path):
        return None
    
    with open(live_path, 'r') as f:
        raw = json.load(f)
    
    # Convert from {"predictions": [{"horizon_days": 5, "predicted_price": X}, ...]}
    # to {"D+5": X, "D+8": Y, ...}
    result = {}
    
    if 'predictions' in raw:
        for pred in raw['predictions']:
            h = pred.get('horizon_days')
            price = pred.get('predicted_price')
            if h is not None and price is not None:
                result[f'D+{h}'] = price
    else:
        # Already in expected format or other format
        for key, val in raw.items():
            if key.startswith('D+'):
                result[key] = val
    
    return result if result else None


def load_confidence_stats(data_dir):
    """Load signal confidence statistics from confidence_stats.json."""
    conf_path = os.path.join(data_dir, 'confidence_stats.json')
    if not os.path.exists(conf_path):
        return None
    
    with open(conf_path, 'r') as f:
        stats = json.load(f)
    
    # Use 5-day stats as default (best accuracy)
    if 'stats_by_horizon' in stats and '5d' in stats['stats_by_horizon']:
        return stats['stats_by_horizon']['5d']
    
    return None


def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_signals(forecast_df, available_horizons, threshold):
    """Calculate net_prob signals with given threshold."""
    results = []
    
    for date in forecast_df.index:
        row = forecast_df.loc[date]
        
        slopes = []
        for i_idx, i in enumerate(available_horizons):
            for j in available_horizons[i_idx + 1:]:
                if pd.notna(row[i]) and pd.notna(row[j]):
                    drift = row[j] - row[i]
                    slopes.append(drift)
        
        if len(slopes) == 0:
            net_prob = 0.0
        else:
            slopes = np.array(slopes)
            bullish = (slopes > 0).sum()
            bearish = (slopes < 0).sum()
            total = len(slopes)
            net_prob = (bullish - bearish) / total
        
        # Determine signal strength
        strength = abs(net_prob)
        
        # Apply threshold for signal classification
        if net_prob > threshold:
            signal = 'BULLISH'
        elif net_prob < -threshold:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        results.append({
            'date': date,
            'net_prob': net_prob,
            'signal': signal,
            'strength': strength
        })
    
    return pd.DataFrame(results).set_index('date')


def apply_rsi_filter(signals_df, rsi_series, rsi_overbought, rsi_oversold):
    """
    Apply RSI filter to signals.
    Returns a new DataFrame with filtered signals.
    """
    filtered = signals_df.copy()
    
    for date in filtered.index:
        if date in rsi_series.index:
            rsi_val = rsi_series.loc[date]
            current_signal = filtered.loc[date, 'signal']
            
            if current_signal == 'BULLISH' and rsi_val > rsi_overbought:
                filtered.loc[date, 'signal'] = 'NEUTRAL'
                filtered.loc[date, 'rsi_filtered'] = True
            elif current_signal == 'BEARISH' and rsi_val < rsi_oversold:
                filtered.loc[date, 'signal'] = 'NEUTRAL'
                filtered.loc[date, 'rsi_filtered'] = True
            else:
                filtered.loc[date, 'rsi_filtered'] = False
    
    return filtered


def process_asset(name, config):
    """Process a single asset and return chart data."""
    data_dir = os.path.join(EXPERIMENT_ROOT, 'data', f'{config["id"]}_{name}')
    
    if not os.path.exists(data_dir):
        print(f"  [WARN] Data not found for {name}")
        return None
    
    print(f"  Processing {name}...")
    
    # Load data
    forecast_df, available_horizons = load_forecast_data(data_dir)
    prices = load_price_data(data_dir)
    live_forecast = load_live_forecast(data_dir)
    model_counts = load_model_counts(data_dir)
    confidence_stats = load_confidence_stats(data_dir)
    
    if forecast_df is None or prices is None:
        print(f"  [WARN] Missing data for {name}")
        return None
    
    rsi = calculate_rsi(prices)
    
    # Calculate base signals (WITHOUT RSI filter)
    signals_raw = calculate_signals(forecast_df, available_horizons, config['threshold'])
    
    # Calculate filtered signals (WITH RSI filter)
    signals_filtered = apply_rsi_filter(
        signals_raw, rsi, 
        config['rsi_overbought'], 
        config['rsi_oversold']
    )
    
    # Get the signal date range
    signal_dates = signals_raw.index.sort_values()
    signal_start_date = signal_dates.min()
    
    # Include extended historical prices BEFORE signal start date (for Ichimoku/EMA)
    # This gives better indicator calculations while snake only shows where we have signals
    extended_price_dates = prices.index[prices.index < signal_start_date].sort_values()
    
    # Align signal data (where we have both prices AND signals)
    common_dates = prices.index.intersection(signals_raw.index).sort_values()
    prices_aligned = prices.loc[common_dates]
    signals_raw_aligned = signals_raw.loc[common_dates]
    signals_filtered_aligned = signals_filtered.loc[common_dates]
    rsi_aligned = rsi.reindex(common_dates).ffill().bfill()
    
    # Combine extended history + signal period
    all_dates = extended_price_dates.append(common_dates).sort_values()
    all_prices = prices.loc[all_dates]
    
    # For extended dates (before signals), fill with NEUTRAL signals
    extended_signals_raw = pd.DataFrame({
        'signal': 'NEUTRAL',
        'net_prob': 0.0,
        'strength': 0.0
    }, index=extended_price_dates)
    
    all_signals_raw = pd.concat([extended_signals_raw, signals_raw_aligned]).sort_index()
    all_signals_filtered = pd.concat([extended_signals_raw, signals_filtered_aligned]).sort_index()
    
    # Extend RSI to cover all dates
    all_rsi = rsi.reindex(all_dates).ffill().bfill()
    
    # Prepare chart data - store BOTH raw and filtered signals
    chart_data = {
        'dates': [d.strftime('%Y-%m-%d') for d in all_dates],
        'prices': all_prices.tolist(),
        'net_prob': all_signals_raw['net_prob'].tolist(),
        'strength': all_signals_raw['strength'].tolist(),
        # Raw signals (without RSI filter)
        'signals_raw': all_signals_raw['signal'].tolist(),
        # Filtered signals (with RSI filter) - this is what we currently display
        'signals_filtered': all_signals_filtered['signal'].tolist(),
        # Default to filtered (RSI ON)
        'signals': all_signals_filtered['signal'].tolist(),
        'rsi': all_rsi.tolist(),
        # Mark where actual signals start (for UI reference)
        'signal_start_date': signal_start_date.strftime('%Y-%m-%d'),
        'threshold': config['threshold'],
        'rsi_overbought': config['rsi_overbought'],
        'rsi_oversold': config['rsi_oversold'],
        'accuracy': config['accuracy'],
        'edge': config['edge'],
        'asset_color': config['color'],
        'horizons': available_horizons,
        'project_id': config['id'],
        'model_counts': {str(h): model_counts.get(h, 0) for h in available_horizons}
    }
    
    # Add per-horizon forecast data for dynamic horizon toggling
    # This allows users to enable/disable horizons and see recalculated metrics
    # For extended dates (before signals), fill with 0
    horizon_forecasts = {}
    for h in available_horizons:
        col_name = h  # Column name in forecast_df
        if col_name in forecast_df.columns:
            # Get values aligned to all_dates (0 for extended history)
            aligned_values = forecast_df.reindex(all_dates)[col_name] if col_name in forecast_df.columns else pd.Series(index=all_dates)
            horizon_forecasts[str(h)] = aligned_values.fillna(0).tolist()
    
    chart_data['horizon_forecasts'] = horizon_forecasts
    
    # Add live forecast if available
    if live_forecast:
        last_price = float(all_prices.iloc[-1])
        live_data = {'base_price': last_price, 'forecasts': {}}
        
        for key, val in live_forecast.items():
            if key.startswith('D+') and val is not None:
                live_data['forecasts'][key] = val
        
        chart_data['live_forecast'] = live_data
    
    # Calculate daily returns for equity curve
    daily_returns = all_prices.pct_change().fillna(0).tolist()
    chart_data['daily_returns'] = daily_returns
    
    # Calculate Price Accuracy (MAPE-based)
    # Compare D+1 predictions to actual next-day prices
    mape_errors = []
    for h in available_horizons:
        forecast_path = os.path.join(data_dir, f'forecast_d{h}.csv')
        if os.path.exists(forecast_path):
            forecast_df_h = pd.read_csv(forecast_path, parse_dates=['date'])
            forecast_df_h = forecast_df_h.set_index('date')
            forecast_df_h.index = pd.to_datetime(forecast_df_h.index).tz_localize(None)
            
            for date in forecast_df_h.index:
                pred = forecast_df_h.loc[date, 'prediction']
                # Get actual price h days later
                target_date = date + pd.Timedelta(days=h)
                if target_date in prices.index:
                    actual = prices.loc[target_date]
                    if actual > 0 and pd.notna(pred):
                        error = abs(pred - actual) / actual
                        mape_errors.append(error)
    
    if mape_errors:
        mape = np.mean(mape_errors) * 100  # As percentage
        price_accuracy = 100 - mape
    else:
        price_accuracy = 0
    
    chart_data['price_accuracy'] = round(price_accuracy, 1)
    chart_data['mape'] = round(mape if mape_errors else 0, 2)
    
    # Calculate signal stats for BOTH raw and filtered
    signal_counts_raw = signals_raw_aligned['signal'].value_counts()
    signal_counts_filtered = signals_filtered_aligned['signal'].value_counts()
    
    chart_data['stats_raw'] = {
        'bullish': int(signal_counts_raw.get('BULLISH', 0)),
        'bearish': int(signal_counts_raw.get('BEARISH', 0)),
        'neutral': int(signal_counts_raw.get('NEUTRAL', 0)),
        'total_days': len(common_dates)
    }
    
    chart_data['stats_filtered'] = {
        'bullish': int(signal_counts_filtered.get('BULLISH', 0)),
        'bearish': int(signal_counts_filtered.get('BEARISH', 0)),
        'neutral': int(signal_counts_filtered.get('NEUTRAL', 0)),
        'total_days': len(common_dates)
    }
    
    # Default stats (with RSI filter ON)
    chart_data['stats'] = chart_data['stats_filtered']
    
    # ==================== HOLDING PERIOD OPTIMIZATION ====================
    # Test different holding periods to find optimal exit strategy
    # Extended to 30 days for longer-horizon assets, using Fibonacci-style progression
    holding_periods = [1, 2, 3, 5, 8, 10, 13, 15, 20, 25, 30]
    holding_results = {}
    
    for hold_days in holding_periods:
        correct = 0
        total = 0
        wins = []
        losses = []
        trade_results = []  # Track sequence for consecutive wins/losses
        equity_curve = [100]  # Track equity for drawdown
        
        for i, date in enumerate(common_dates[:-hold_days]):
            signal_filtered = signals_filtered_aligned.iloc[i]['signal']
            
            current_price = prices_aligned.iloc[i]
            future_price = prices_aligned.iloc[i + hold_days]
            
            signal = signal_filtered
            if signal == 'NEUTRAL':
                continue
            
            actual_direction = 'UP' if future_price > current_price else 'DOWN'
            predicted_direction = 'UP' if signal == 'BULLISH' else 'DOWN'
            pct_change = (future_price - current_price) / current_price
            trade_return = pct_change if signal == 'BULLISH' else -pct_change
            
            total += 1
            is_win = actual_direction == predicted_direction
            trade_results.append(is_win)
            
            # Update equity curve
            new_equity = equity_curve[-1] * (1 + trade_return)
            equity_curve.append(new_equity)
            
            if is_win:
                correct += 1
                wins.append(abs(trade_return) * 100)
            else:
                losses.append(abs(trade_return) * 100)
        
        # Calculate metrics
        if total > 0:
            accuracy = (correct / total) * 100
            edge = accuracy - 50
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # Profit Factor
            total_wins = sum(wins) if wins else 0
            total_losses = sum(losses) if losses else 0.001
            profit_factor = total_wins / total_losses if total_losses > 0 else 999
            
            # Expectancy per trade (in %)
            win_rate_decimal = correct / total
            expectancy = (win_rate_decimal * avg_win) - ((1 - win_rate_decimal) * avg_loss)
            
            # Win/Loss Ratio
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 999
            
            # Max Drawdown from equity curve
            peak = equity_curve[0]
            max_dd = 0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            
            # Max Consecutive Losses
            max_consec_losses = 0
            consec_losses = 0
            for result in trade_results:
                if not result:
                    consec_losses += 1
                    max_consec_losses = max(max_consec_losses, consec_losses)
                else:
                    consec_losses = 0
            
            # Total Return
            total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
            
            # Annualized metrics (assuming ~252 trading days)
            trades_per_year = 252 / hold_days
            annual_expectancy = expectancy * min(total, trades_per_year)
            
        else:
            accuracy = 0
            edge = -50
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            expectancy = 0
            win_loss_ratio = 0
            max_dd = 0
            max_consec_losses = 0
            total_return = 0
            annual_expectancy = 0
        
        holding_results[hold_days] = {
            'accuracy': round(accuracy, 1),
            'edge': round(edge, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'total_trades': total,
            'correct': correct,
            'profit_factor': round(profit_factor, 2),
            'expectancy': round(expectancy, 3),
            'win_loss_ratio': round(win_loss_ratio, 2),
            'max_drawdown': round(max_dd, 1),
            'max_consec_losses': max_consec_losses,
            'total_return': round(total_return, 1),
            'annual_expectancy': round(annual_expectancy, 1)
        }
    
    # Calculate composite score for optimal holding period
    # Score = Expectancy × √(Accuracy) × (1 / (1 + MaxDD/100)) × ProfitFactor^0.5
    for days, hp in holding_results.items():
        if hp['total_trades'] > 0 and hp['accuracy'] > 0:
            acc_factor = np.sqrt(hp['accuracy'] / 100)
            dd_factor = 1 / (1 + hp['max_drawdown'] / 100)
            pf_factor = np.sqrt(min(hp['profit_factor'], 5))  # Cap PF contribution
            exp_factor = max(hp['expectancy'], 0)
            
            hp['composite_score'] = round(exp_factor * acc_factor * dd_factor * pf_factor * 100, 1)
        else:
            hp['composite_score'] = 0
    
    # Find TOP 3 optimal holding periods (by composite score, not just accuracy)
    sorted_periods = sorted(holding_results.keys(), 
                           key=lambda k: holding_results[k]['composite_score'], 
                           reverse=True)
    
    # Get top 3 (or fewer if not enough data)
    top_3_periods = sorted_periods[:3]
    
    # Add rank to each result
    for rank, period in enumerate(sorted_periods):
        holding_results[period]['rank'] = rank + 1
    
    chart_data['holding_periods'] = holding_results
    chart_data['top_3_holdings'] = top_3_periods
    chart_data['optimal_holding'] = top_3_periods[0] if top_3_periods else 1
    chart_data['optimal_accuracy'] = holding_results[top_3_periods[0]]['accuracy'] if top_3_periods else 0
    chart_data['optimal_edge'] = holding_results[top_3_periods[0]]['edge'] if top_3_periods else 0
    
    # ==================== REPLAY DATA ====================
    # Prepare data for animated replay
    replay_data = []
    for i, date in enumerate(common_dates):
        replay_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': float(prices_aligned.iloc[i]),
            'signal': signals_filtered_aligned.iloc[i]['signal'],
            'net_prob': float(signals_filtered_aligned.iloc[i]['net_prob']),
            'strength': float(signals_filtered_aligned.iloc[i]['strength'])
        })
    
    # Add forecast snapshots for replay (what the forecast looked like on each date)
    # We'll use the forecast_df to get predictions
    chart_data['replay_data'] = replay_data
    chart_data['replay_start_idx'] = max(0, len(replay_data) - 90)  # Last 90 days by default
    
    # Add confidence stats if available
    if confidence_stats:
        chart_data['confidence_stats'] = {
            'by_strength': confidence_stats.get('by_strength', {}),
            'by_signal': confidence_stats.get('by_signal', {}),
            'confidence_tiers': confidence_stats.get('confidence_tiers', []),
            'total_signals': confidence_stats.get('total_signals', 0),
            'analysis_period': confidence_stats.get('analysis_period', '')
        }
    else:
        chart_data['confidence_stats'] = None
    
    return chart_data


def load_optimal_configs():
    """Load optimal configs from JSON files generated by run_optimized_update.py"""
    configs = {}
    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    
    # Default configs (fallback if no JSON file exists)
    default_configs = {
        "Crude_Oil": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "Bitcoin": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "SP500": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "NASDAQ": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "RUSSEL": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "DOW_JONES_Mini": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "GOLD": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "US_DOLLAR_Index": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "SPDR_China_ETF": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "Nikkei_225": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "Nifty_50": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "Nifty_Bank": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "MCX_Copper": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "USD_INR": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50},
        "Brent_Oil": {"viable_horizons": [], "avg_accuracy": 50, "health_score": 50}
    }
    
    for asset_name in default_configs.keys():
        config_file = os.path.join(configs_dir, f'optimal_{asset_name.lower()}.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                # Get avg_da for raw and optimized, fallback to avg_accuracy if not present
                avg_da_raw = config.get('avg_da_raw', config.get('avg_da', config.get('avg_accuracy', 50)))
                avg_da_optimized = config.get('avg_da_optimized', avg_da_raw)
                configs[asset_name] = {
                    "viable_horizons": config.get('viable_horizons', []),
                    "avg_accuracy": config.get('avg_accuracy', 50),
                    "avg_da_raw": avg_da_raw,
                    "avg_da_optimized": avg_da_optimized,
                    "health_score": config.get('health_score', 50),
                    "optimized_outperformance": config.get('optimized_outperformance', 0),
                    "baseline_outperformance": config.get('baseline_outperformance', 0),
                    # OPTIMIZED metrics (pre-calculated in pipeline)
                    "optimized_equity": config.get('optimized_equity', config.get('optimized_outperformance', 0)),
                    "sharpe_optimized": config.get('sharpe_optimized', None),
                    "max_drawdown_optimized": config.get('max_drawdown_optimized', None),
                    "total_trades": config.get('total_trades', None),
                    "win_rate": config.get('win_rate', None),
                    "profit_factor": config.get('profit_factor', None),
                    # RAW/BASELINE metrics (pre-calculated in pipeline)
                    "baseline_equity": config.get('baseline_equity', config.get('baseline_outperformance', 0)),
                    "baseline_sharpe": config.get('baseline_sharpe', None),
                    "baseline_win_rate": config.get('baseline_win_rate', None),
                    "baseline_total_trades": config.get('baseline_total_trades', None),
                    "baseline_profit_factor": config.get('baseline_profit_factor', None),
                    "baseline_max_drawdown": config.get('baseline_max_drawdown', None),
                    # Pre-calculated last 30 trades (from precalculate_metrics.py)
                    "last_30_trades_raw": config.get('last_30_trades_raw', None),
                    "last_30_trades_optimized": config.get('last_30_trades_optimized', None)
                }
                print(f"  Loaded optimal config for {asset_name}: horizons={configs[asset_name]['viable_horizons']}, DA: raw={avg_da_raw}, opt={avg_da_optimized}")
            except Exception as e:
                print(f"  Warning: Could not load config for {asset_name}: {e}")
                configs[asset_name] = default_configs[asset_name]
        else:
            configs[asset_name] = default_configs[asset_name]
    
    return configs


def build_api_data(all_data):
    """Generate comprehensive API data for each asset."""
    api_data = {}
    
    for asset_name, asset_data in all_data.items():
        # Find the data directory for this asset
        asset_config = ASSETS.get(asset_name, {})
        asset_id = asset_config.get('id', '')
        data_dir = os.path.join(EXPERIMENT_ROOT, 'data', f'{asset_id}_{asset_name}')
        
        # Load additional data files if they exist
        live_forecast = {}
        optimized_forecast = {}
        confidence_stats = {}
        
        try:
            live_path = os.path.join(data_dir, 'live_forecast.json')
            if os.path.exists(live_path):
                with open(live_path, 'r') as f:
                    live_forecast = json.load(f)
        except Exception:
            pass
            
        try:
            opt_path = os.path.join(data_dir, 'optimized_forecast.json')
            if os.path.exists(opt_path):
                with open(opt_path, 'r') as f:
                    optimized_forecast = json.load(f)
        except Exception:
            pass
            
        try:
            conf_path = os.path.join(data_dir, 'confidence_stats.json')
            if os.path.exists(conf_path):
                with open(conf_path, 'r') as f:
                    confidence_stats = json.load(f)
        except Exception:
            pass
        
        # Build comprehensive API object
        api_data[asset_name] = {
            "asset": asset_name,
            "project_id": asset_id,
            "current_price": asset_data.get('current_price', 0),
            "signal": optimized_forecast.get('signal', 'N/A'),
            "confidence": optimized_forecast.get('confidence', 0),
            "pct_change": optimized_forecast.get('pct_change', 0),
            "health_score": optimized_forecast.get('health_score', 0),
            "viable_horizons": optimized_forecast.get('viable_horizons', []),
            "forecasts": live_forecast.get('predictions', []),
            "accuracy": asset_data.get('accuracy', 0),
            "edge": asset_data.get('edge', 0),
            "confidence_stats": confidence_stats.get('stats_by_horizon', {}),
            "timestamp": optimized_forecast.get('timestamp', datetime.now().isoformat()),
            "data_source": "QDTNexus Meta-Dynamic Quantile Ensemble"
        }
    
    return api_data


def build_html(all_data):
    """Generate the complete HTML dashboard."""
    
    # Convert data to JSON for embedding
    data_json = json.dumps(all_data, indent=2)
    
    # Build API data for downloads
    api_data = build_api_data(all_data)
    api_data_json = json.dumps(api_data, indent=2)
    
    # Load optimal configs from JSON files
    optimal_configs = load_optimal_configs()
    
    # Generate JavaScript object for optimal configs
    # Use json.dumps to include ALL fields (including baseline_equity, optimized_equity, etc.)
    optimal_configs_js = json.dumps(optimal_configs, indent=8)
    
    # Load QDT logo as base64
    import base64
    qdt_logo_path = os.path.join(os.path.dirname(__file__), 'QDT logo', 'QDT logo.jpg')
    qdt_logo_b64 = ''
    if os.path.exists(qdt_logo_path):
        with open(qdt_logo_path, 'rb') as f:
            qdt_logo_b64 = base64.b64encode(f.read()).decode()
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QDTNexus Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdn.auth0.com/js/auth0-spa-js/2.0/auth0-spa-js.production.js"></script>
    <script>
        // OPTIMAL_CONFIGS - loaded from JSON files generated by run_optimized_update.py
        window.OPTIMAL_CONFIGS = {optimal_configs_js};
    </script>
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;450;500;600;700&family=Montserrat:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            /* QDT Brand Colors - Exact match to qdt_styles.css dark theme */
            --bg-primary: #0C0D0F;
            --bg-secondary: #0F1C25;
            --bg-tertiary: #1C2934;
            --bg-card: #0A151D;
            --bg-card2: #0A0F14;
            --menu-bg: #0B1117;
            
            /* Accent Colors */
            --accent-green: #00d68f;
            --accent-red: #E67676;
            --accent-gold: #F5B700;
            --accent-blue: #52718A;
            --default-action: #739EC1;
            
            /* QDT Teal/Cyan Accents */
            --qdt-primary: #94C3C6;
            --qdt-secondary: #B2EAEE;
            --qdt-accent-bg: #6288A6;
            --qdt-accent-bg2: #52718A;
            --qdt-gradient: linear-gradient(90deg, #0F1C25 0%, #52718A 100%);
            --gradient-header: linear-gradient(90deg, #0A0F14 0%, #0B1117 60%, #52718A80 100%);
            
            /* Text Colors */
            --text-primary: #ffffff;
            --text-secondary: #CDCDCD;
            --text-muted: #9F9F9F;
            --info-text: #E1E1E1;
            
            /* Border Colors */
            --border-color: #2A323A;
            --border-subtle: #87B4D833;
            --accent-border: #87B4D833;
            
            /* Interactive States */
            --active-bg: #29425680;
            --hover-bg: #5F8CB080;
            --semi-transparent: rgba(255, 255, 255, 0.05);
            --light-btn-bg: #A7D1F3;
        }}
        
        body {{
            font-family: 'Open Sans', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            font-size: 14px;
            font-weight: 450;
        }}
        
        /* QDT Background with subtle gradient beams */
        .bg-gradient {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                linear-gradient(135deg, transparent 0%, transparent 45%, rgba(82, 113, 138, 0.03) 45%, rgba(82, 113, 138, 0.06) 46.5%, transparent 46.5%, transparent 52%, rgba(82, 113, 138, 0.035) 52%, rgba(82, 113, 138, 0.065) 53.5%, transparent 53.5%, transparent 59%, rgba(82, 113, 138, 0.03) 59%, rgba(82, 113, 138, 0.06) 60.5%, transparent 60.5%, transparent 66%, rgba(82, 113, 138, 0.035) 66%, rgba(82, 113, 138, 0.065) 67.5%, transparent 67.5%, transparent 100%),
                linear-gradient(130deg, transparent 0%, transparent 35%, rgba(82, 113, 138, 0.025) 35%, rgba(82, 113, 138, 0.05) 36.5%, transparent 36.5%, transparent 72%, rgba(82, 113, 138, 0.03) 72%, rgba(82, 113, 138, 0.055) 73.5%, transparent 73.5%, transparent 100%),
                radial-gradient(ellipse at 90% 10%, rgba(82, 113, 138, 0.12) 0%, transparent 45%);
            z-index: -1;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        /* Header - QDT Gradient Style */
        .header {{
            text-align: center;
            padding: 30px 0 20px;
            background: var(--gradient-header);
            border-bottom: 1px solid var(--accent-border);
            margin-bottom: 30px;
        }}
        
        .logo {{
            font-size: 42px;
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -1px;
            margin-bottom: 12px;
            display: flex;
            align-items: baseline;
            justify-content: center;
            gap: 0;
        }}
        
        .logo .qdt-text {{
            color: var(--text-primary);
            font-weight: 800;
            letter-spacing: -1px;
        }}
        
        .logo .nexus-text {{
            color: var(--qdt-primary);
            font-weight: 500;
            letter-spacing: 0px;
        }}
        
        .logo span.dot {{
            color: var(--qdt-primary);
            font-weight: 800;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 400;
            letter-spacing: 4px;
            text-transform: uppercase;
            opacity: 0.8;
        }}
        
        /* Asset Selector */
        .selector-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        
        .selector-label {{
            color: var(--text-secondary);
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        
        .asset-select {{
            background: var(--bg-secondary);
            border: 1.5px solid var(--accent-border);
            color: var(--text-primary);
            padding: 12px 24px;
            font-size: 16px;
            font-family: 'Open Sans', sans-serif;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
            min-width: 250px;
        }}
        
        .asset-select:hover {{
            background: var(--active-bg);
        }}
        
        .asset-select:focus {{
            outline: none;
            border-color: var(--default-action);
        }}
        
        .asset-select option {{
            background: var(--bg-secondary);
            color: var(--text-primary);
            padding: 10px;
        }}
        
        /* RSI Toggle Switch - Compact version for chart header */
        .toggle-container {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            transition: all 0.3s ease;
        }}
        
        .toggle-container:hover {{
            border-color: var(--accent-blue);
        }}
        
        .toggle-container.active {{
            border-color: var(--accent-green);
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.15);
        }}
        
        .toggle-label {{
            color: var(--text-secondary);
            font-size: 11px;
            font-weight: 500;
            user-select: none;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .toggle-switch {{
            position: relative;
            width: 36px;
            height: 20px;
            background: var(--bg-secondary);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid var(--border-color);
        }}
        
        .toggle-switch.on {{
            background: var(--accent-green);
            border-color: var(--accent-green);
        }}
        
        .toggle-switch::before {{
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 14px;
            height: 14px;
            background: white;
            border-radius: 50%;
            transition: all 0.3s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }}
        
        .toggle-switch.on::before {{
            transform: translateX(16px);
        }}
        
        .toggle-status {{
            font-size: 10px;
            font-weight: 600;
            min-width: 24px;
        }}
        
        .toggle-status.on {{
            color: var(--accent-green);
        }}
        
        .toggle-status.off {{
            color: var(--text-secondary);
        }}
        
        /* Chart header with toggle */
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .chart-title {{
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        /* Stats Cards */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        
        .stat-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-3px);
            border-color: var(--accent-blue);
        }}
        
        .stat-card.accuracy {{
            border-left: 4px solid var(--accent-green);
        }}
        
        .stat-card.edge {{
            border-left: 4px solid var(--accent-blue);
        }}
        
        .stat-card.bullish {{
            border-left: 4px solid #00ff88;
        }}
        
        .stat-card.bearish {{
            border-left: 4px solid #ff3366;
        }}
        
        .stat-card.neutral {{
            border-left: 4px solid #888;
        }}
        
        .stat-card.threshold {{
            border-left: 4px solid var(--accent-gold);
        }}
        
        .stat-card.project-id {{
            border-left: 4px solid var(--accent-blue);
        }}
        
        .stat-value {{
            font-size: 28px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 5px;
        }}
        
        .stat-value.positive {{
            color: var(--accent-green);
        }}
        
        .stat-value.negative {{
            color: var(--accent-red);
        }}
        
        .stat-label {{
            color: var(--text-secondary);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Chart Container */
        .chart-container {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 25px;
            position: relative;  /* For floating elements */
        }}
        
        #snake-chart {{
            width: 100%;
            height: 550px;
        }}
        
        #rsi-chart {{
            width: 100%;
            height: 180px;
        }}
        
        #confidence-chart {{
            width: 100%;
            height: 150px;
        }}
        
        .confidence-chart-legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            padding: 8px;
            font-size: 11px;
            color: var(--text-secondary);
        }}
        
        .conf-legend-item {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        
        .conf-legend-item.high {{ color: #00ff88; }}
        .conf-legend-item.medium {{ color: #ffaa00; }}
        .conf-legend-item.low {{ color: #ff6666; }}
        
        .conf-legend-help {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 11px;
            color: var(--text-secondary);
            transition: all 0.3s ease;
        }}
        
        .conf-legend-help:hover {{
            background: var(--qdt-primary);
            color: white;
            border-color: var(--qdt-primary);
        }}
        
        .confidence-modal {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .confidence-modal-content {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            max-width: 500px;
            position: relative;
        }}
        
        .confidence-modal-content h3 {{
            margin: 0 0 15px 0;
            color: var(--text-primary);
        }}
        
        .confidence-modal-content p {{
            color: var(--text-secondary);
            font-size: 13px;
            line-height: 1.6;
        }}
        
        .confidence-modal-close {{
            position: absolute;
            top: 15px;
            right: 20px;
            font-size: 24px;
            color: var(--text-secondary);
            cursor: pointer;
        }}
        
        .confidence-modal-close:hover {{
            color: var(--accent-red);
        }}
        
        .conf-factor {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
        }}
        
        .conf-factor-name {{
            color: var(--qdt-primary);
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 4px;
        }}
        
        .conf-factor-desc {{
            color: var(--text-secondary);
            font-size: 12px;
        }}
        
        .conf-note {{
            font-style: italic;
            color: var(--accent-gold);
            margin-top: 15px;
        }}
        
        #chartConfidenceHelp:hover {{
            background: var(--qdt-primary) !important;
            color: white !important;
            border-color: var(--qdt-primary) !important;
            transform: scale(1.1);
        }}
        
        #equity-chart {{
            width: 100%;
            height: 280px;
        }}
        
        /* Live Signal Confidence Panel - HIDDEN (confidence now shown on chart) */
        .live-signal-panel {{
            display: none !important;  /* Confidence now shown on chart badge */
        }}
        
        .live-signal-panel.bullish {{
            border-color: rgba(0, 255, 136, 0.5);
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.08) 0%, var(--bg-secondary) 100%);
        }}
        
        .live-signal-panel.bearish {{
            border-color: rgba(255, 51, 102, 0.5);
            background: linear-gradient(135deg, rgba(255, 51, 102, 0.08) 0%, var(--bg-secondary) 100%);
        }}
        
        .live-signal-panel.neutral {{
            border-color: rgba(136, 136, 136, 0.5);
            background: linear-gradient(135deg, rgba(136, 136, 136, 0.08) 0%, var(--bg-secondary) 100%);
        }}
        
        .live-signal-main {{
            display: flex;
            align-items: center;
            gap: 30px;
        }}
        
        .live-signal-direction {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .signal-arrow {{
            font-size: 48px;
            filter: drop-shadow(0 0 10px currentColor);
        }}
        
        .signal-arrow.bullish {{
            color: #00ff88;
        }}
        
        .signal-arrow.bearish {{
            color: #ff3366;
        }}
        
        .signal-arrow.neutral {{
            color: #888;
            font-size: 36px;
        }}
        
        .signal-text {{
            font-size: 32px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        
        .signal-text.bullish {{
            color: #00ff88;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        }}
        
        .signal-text.bearish {{
            color: #ff3366;
            text-shadow: 0 0 20px rgba(255, 51, 102, 0.5);
        }}
        
        .signal-text.neutral {{
            color: #888;
        }}
        
        .live-signal-confidence {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding-left: 30px;
            border-left: 1px solid var(--border-color);
        }}
        
        .confidence-tier {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .tier-badge {{
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .tier-badge.high {{
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
            border: 1px solid rgba(0, 255, 136, 0.4);
        }}
        
        .tier-badge.medium {{
            background: rgba(255, 170, 0, 0.2);
            color: #ffaa00;
            border: 1px solid rgba(255, 170, 0, 0.4);
        }}
        
        .tier-badge.low {{
            background: rgba(255, 51, 102, 0.2);
            color: #ff3366;
            border: 1px solid rgba(255, 51, 102, 0.4);
        }}
        
        .confidence-accuracy {{
            display: flex;
            align-items: baseline;
            gap: 8px;
        }}
        
        .confidence-accuracy .accuracy-value {{
            font-size: 36px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
        }}
        
        .confidence-accuracy .accuracy-label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .live-signal-details {{
            display: flex;
            gap: 25px;
            padding: 12px 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
        }}
        
        .live-signal-details .detail-item {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            text-align: center;
        }}
        
        .live-signal-details .detail-label {{
            font-size: 10px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .live-signal-details .detail-value {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Health Check Section - Redesigned */
        .health-check {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            margin-bottom: 25px;
            overflow: hidden;
        }}
        
        .health-check.healthy {{
            border-color: rgba(0, 255, 136, 0.3);
        }}
        
        .health-check.warning {{
            border-color: rgba(255, 215, 0, 0.3);
        }}
        
        .health-check.critical {{
            border-color: rgba(255, 51, 102, 0.3);
        }}
        
        /* Flashing alert indicator */
        @keyframes flash-pulse {{
            0%, 100% {{ 
                transform: scale(1); 
                box-shadow: 0 0 0 0 currentColor;
            }}
            50% {{ 
                transform: scale(1.1); 
                box-shadow: 0 0 20px currentColor;
            }}
        }}
        
        @keyframes gentle-pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        
        /* Auto Optimize Button Flashing Animation - Subtle glow pulse */
        @keyframes button-glow {{
            0%, 100% {{ 
                box-shadow: 0 0 8px rgba(255, 215, 0, 0.6), 0 0 16px rgba(255, 165, 0, 0.3);
            }}
            50% {{ 
                box-shadow: 0 0 16px rgba(255, 215, 0, 0.9), 0 0 30px rgba(255, 165, 0, 0.5);
            }}
        }}
        
        .auto-optimize-btn-flash {{
            animation: button-glow 2s ease-in-out infinite !important;
        }}
        
        /* Loading Spinner */
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .optimize-spinner {{
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top: 2px solid white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }}
        
        .optimize-btn-loading {{
            pointer-events: none;
            opacity: 0.8;
        }}
        
        .health-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 15px 20px;
            background: var(--bg-tertiary);
            cursor: pointer;
            transition: all 0.3s ease;
            user-select: none;
        }}
        
        .health-header:hover {{
            background: rgba(255,255,255,0.03);
        }}
        
        .health-header-left {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .alert-indicator {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 700;
            transition: all 0.3s ease;
        }}
        
        .alert-indicator.critical {{
            background: var(--accent-red);
            color: white;
            animation: flash-pulse 1s ease-in-out infinite;
        }}
        
        .alert-indicator.warning {{
            background: var(--accent-gold);
            color: #000;
            animation: gentle-pulse 2s ease-in-out infinite;
        }}
        
        .alert-indicator.healthy {{
            background: var(--accent-green);
            color: #000;
        }}
        
        .health-summary {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .health-label {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .health-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .health-badge.excellent {{
            background: rgba(0, 255, 136, 0.15);
            color: var(--accent-green);
        }}
        
        .health-badge.good {{
            background: rgba(0, 212, 255, 0.15);
            color: var(--accent-blue);
        }}
        
        .health-badge.fair {{
            background: rgba(255, 215, 0, 0.15);
            color: var(--accent-gold);
        }}
        
        .health-badge.poor {{
            background: rgba(255, 51, 102, 0.15);
            color: var(--accent-red);
        }}
        
        .health-score-mini {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
        }}
        
        .expand-icon {{
            font-size: 18px;
            color: var(--text-secondary);
            transition: transform 0.3s ease;
        }}
        
        .health-check.expanded .expand-icon {{
            transform: rotate(180deg);
        }}
        
        .health-body {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease;
        }}
        
        .health-check.expanded .health-body {{
            max-height: 800px;
        }}
        
        .health-content {{
            padding: 20px;
        }}
        
        .score-row {{
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 15px;
            background: var(--bg-tertiary);
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        
        .score-circle {{
            width: 70px;
            height: 70px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            flex-shrink: 0;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
        }}
        
        .score-circle:hover {{
            transform: scale(1.08);
            box-shadow: 0 0 20px currentColor;
        }}
        
        .score-circle.excellent {{
            background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,255,136,0.1));
            border: 3px solid var(--accent-green);
            color: var(--accent-green);
        }}
        
        .score-circle.good {{
            background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,212,255,0.1));
            border: 3px solid var(--accent-blue);
            color: var(--accent-blue);
        }}
        
        .score-circle.fair {{
            background: linear-gradient(135deg, rgba(255,215,0,0.2), rgba(255,215,0,0.1));
            border: 3px solid var(--accent-gold);
            color: var(--accent-gold);
        }}
        
        .score-circle.poor {{
            background: linear-gradient(135deg, rgba(255,51,102,0.2), rgba(255,51,102,0.1));
            border: 3px solid var(--accent-red);
            color: var(--accent-red);
        }}
        
        .score-details {{
            flex: 1;
        }}
        
        .score-label {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }}
        
        .score-text {{
            font-size: 16px;
            font-weight: 600;
        }}
        
        .recommendations-title {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }}
        
        .rec-item {{
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 12px 15px;
            margin-bottom: 10px;
            background: var(--bg-primary);
            border-radius: 8px;
            border-left: 3px solid var(--accent-blue);
        }}
        
        .rec-item:last-child {{
            margin-bottom: 0;
        }}
        
        .rec-item.urgent {{
            border-left-color: var(--accent-red);
            background: rgba(255,51,102,0.08);
        }}
        
        .rec-item.improve {{
            border-left-color: var(--accent-gold);
            background: rgba(255,215,0,0.08);
        }}
        
        .rec-item.success {{
            border-left-color: var(--accent-green);
            background: rgba(0,255,136,0.08);
        }}
        
        .rec-icon {{
            font-size: 18px;
            flex-shrink: 0;
            margin-top: 2px;
        }}
        
        .rec-content {{
            flex: 1;
        }}
        
        .rec-title {{
            font-weight: 600;
            margin-bottom: 4px;
            font-size: 14px;
        }}
        
        .rec-desc {{
            font-size: 12px;
            color: var(--text-secondary);
            line-height: 1.5;
        }}
        
        .build-models-btn {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 6px 14px;
            margin-top: 10px;
            background: var(--qdt-gradient);
            border: none;
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 12px;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .build-models-btn:hover {{
            filter: brightness(1.2);
            transform: translateX(2px);
        }}
        
        .horizon-build-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            background: var(--qdt-accent-bg);
            border: none;
            border-radius: 4px;
            color: white;
            font-size: 12px;
            font-weight: bold;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-left: 4px;
        }}
        
        .horizon-build-btn:hover {{
            filter: brightness(1.3);
            transform: scale(1.1);
        }}
        
        .timeframe-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }}
        
        .timeframe-tag {{
            padding: 3px 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-blue);
            border: 1px solid rgba(0, 212, 255, 0.3);
        }}
        
        .timeframe-tag.missing {{
            color: var(--accent-red);
            border-color: rgba(255, 51, 102, 0.3);
        }}
        
        .timeframe-tag.available {{
            color: var(--accent-green);
            border-color: rgba(0, 255, 136, 0.3);
        }}
        
        /* Horizon Model Grid */
        .horizon-model-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 6px;
            margin-top: 8px;
        }}
        
        .horizon-model-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 10px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            font-size: 11px;
            border: 1px solid var(--border-color);
        }}
        
        .horizon-model-item.good {{
            border-color: rgba(0, 255, 136, 0.3);
        }}
        
        .horizon-model-item.warning {{
            border-color: rgba(255, 215, 0, 0.3);
        }}
        
        .horizon-model-item.critical {{
            border-color: rgba(255, 51, 102, 0.5);
            background: rgba(255, 51, 102, 0.1);
        }}
        
        .horizon-model-item.missing {{
            border-color: rgba(136, 136, 136, 0.3);
            opacity: 0.5;
        }}
        
        .horizon-label {{
            font-weight: 600;
            color: var(--text-secondary);
        }}
        
        .horizon-count {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: var(--text-muted);
        }}
        
        .horizon-status {{
            font-weight: bold;
        }}
        
        .horizon-model-item.good .horizon-status {{
            color: var(--accent-green);
        }}
        
        .horizon-model-item.warning .horizon-status {{
            color: var(--accent-gold);
        }}
        
        .horizon-model-item.critical .horizon-status {{
            color: var(--accent-red);
        }}
        
        .rec-item.info {{
            background: rgba(0, 212, 255, 0.05);
            border-color: rgba(0, 212, 255, 0.2);
        }}
        
        .rec-item.info .rec-icon {{
            color: var(--accent-blue);
        }}
        
        /* Price Targets Section - Institutional Style */
        .price-targets-section {{
            background: var(--bg-card);
            border: 1px solid var(--accent-border);
            border-radius: 4px;
            padding: 24px;
            margin-bottom: 25px;
        }}
        
        .price-targets-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }}
        
        .price-targets-title {{
            font-size: 16px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .price-targets-signal {{
            padding: 6px 14px;
            border-radius: 4px;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .price-targets-signal.bullish {{
            background: rgba(16, 185, 129, 0.2);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
        }}
        
        .price-targets-signal.bearish {{
            background: rgba(239, 68, 68, 0.2);
            color: var(--accent-red);
            border: 1px solid var(--accent-red);
        }}
        
        .price-targets-signal.neutral {{
            background: rgba(156, 163, 175, 0.2);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }}
        
        .price-targets-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 16px;
        }}
        
        .price-targets-table th {{
            text-align: left;
            padding: 10px 12px;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .price-targets-table td {{
            padding: 14px 12px;
            font-size: 14px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .price-targets-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .target-label {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .target-badge {{
            width: 24px;
            height: 24px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 700;
        }}
        
        .target-badge.t1 {{
            background: rgba(16, 185, 129, 0.3);
            color: var(--accent-green);
        }}
        
        .target-badge.t2 {{
            background: rgba(59, 130, 246, 0.3);
            color: var(--accent-blue);
        }}
        
        .target-badge.t3 {{
            background: rgba(251, 191, 36, 0.3);
            color: var(--accent-gold);
        }}
        
        .target-type {{
            font-size: 12px;
            color: var(--text-secondary);
        }}
        
        .target-price {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .target-change {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
        }}
        
        .target-change.positive {{
            color: var(--accent-green);
        }}
        
        .target-change.negative {{
            color: var(--accent-red);
        }}
        
        .price-targets-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            padding: 14px 0;
            border-top: 1px solid var(--border-color);
            margin-top: 8px;
        }}
        
        .meta-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
        }}
        
        .meta-label {{
            color: var(--text-secondary);
        }}
        
        .meta-value {{
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .price-targets-disclaimer {{
            font-size: 11px;
            color: var(--text-secondary);
            opacity: 0.7;
            margin-top: 12px;
            padding: 10px 12px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            line-height: 1.5;
        }}
        
        .neutral-outlook {{
            text-align: center;
            padding: 30px 20px;
        }}
        
        .neutral-outlook-icon {{
            font-size: 36px;
            margin-bottom: 12px;
            opacity: 0.6;
        }}
        
        .neutral-outlook-title {{
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 8px;
        }}
        
        .neutral-outlook-text {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 16px;
        }}
        
        .neutral-range {{
            display: flex;
            justify-content: center;
            gap: 30px;
        }}
        
        .range-item {{
            text-align: center;
        }}
        
        .range-label {{
            font-size: 11px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }}
        
        .range-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px;
            font-weight: 600;
        }}
        
        .range-value.upside {{
            color: var(--accent-green);
        }}
        
        .range-value.downside {{
            color: var(--accent-red);
        }}
        
        /* Info Icon and Tooltip */
        .info-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: 600;
            cursor: help;
            position: relative;
            transition: all 0.3s ease;
        }}
        
        .info-icon:hover {{
            background: var(--accent-blue);
            color: white;
            border-color: var(--accent-blue);
        }}
        
        .info-tooltip {{
            position: absolute;
            top: 30px;
            left: 50%;
            transform: translateX(-50%);
            width: 320px;
            padding: 15px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        }}
        
        .info-icon:hover .info-tooltip {{
            opacity: 1;
            visibility: visible;
        }}
        
        .info-tooltip h4 {{
            font-size: 13px;
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .info-tooltip p {{
            font-size: 11px;
            color: var(--text-secondary);
            line-height: 1.5;
            margin-bottom: 8px;
        }}
        
        .info-tooltip .formula {{
            background: var(--bg-tertiary);
            padding: 8px 12px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: var(--accent-green);
            margin: 10px 0;
        }}
        
        .info-tooltip ul {{
            margin: 8px 0;
            padding-left: 16px;
        }}
        
        .info-tooltip li {{
            font-size: 10px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }}
        
        .info-tooltip .metric-explain {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid var(--border-color);
            font-size: 10px;
        }}
        
        .info-tooltip .metric-explain:last-child {{
            border-bottom: none;
        }}
        
        .info-tooltip .metric-name {{
            color: var(--text-secondary);
        }}
        
        .info-tooltip .metric-desc {{
            color: var(--text-primary);
            text-align: right;
        }}
        
        /* Extended holding details grid */
        .holding-details-extended {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 10px;
            padding: 15px;
            background: var(--bg-tertiary);
            border-radius: 10px;
            margin-top: 15px;
        }}
        
        .holding-metric {{
            text-align: center;
            padding: 8px;
            background: var(--bg-primary);
            border-radius: 8px;
        }}
        
        .holding-metric-value {{
            font-size: 16px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .holding-metric-label {{
            font-size: 9px;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-top: 4px;
            letter-spacing: 0.3px;
        }}
        
        /* Replay Button */
        .replay-btn {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            background: linear-gradient(135deg, var(--accent-blue), #6366f1);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 12px;
            font-weight: 600;
            font-family: 'Open Sans', sans-serif;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .replay-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(99, 102, 241, 0.4);
        }}
        
        .replay-btn:active {{
            transform: translateY(0);
        }}
        
        /* Replay Modal */
        .replay-modal {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            z-index: 10000;
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .replay-modal.active {{
            display: flex;
            opacity: 1;
        }}
        
        .replay-header {{
            width: 100%;
            max-width: 1400px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 30px;
        }}
        
        .replay-title {{
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .replay-title .asset-badge {{
            padding: 4px 12px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            font-size: 14px;
            color: var(--accent-blue);
        }}
        
        .replay-close {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 24px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }}
        
        .replay-close:hover {{
            background: var(--accent-red);
            border-color: var(--accent-red);
            color: white;
        }}
        
        .replay-chart-container {{
            width: 100%;
            max-width: 1400px;
            height: 500px;
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 20px;
            margin: 0 20px;
        }}
        
        #replay-chart {{
            width: 100%;
            height: 100%;
        }}
        
        .replay-controls {{
            width: 100%;
            max-width: 1400px;
            padding: 20px 30px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .replay-buttons {{
            display: flex;
            justify-content: center;
            gap: 15px;
        }}
        
        .replay-control-btn {{
            padding: 10px 24px;
            border-radius: 10px;
            border: 2px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            font-size: 14px;
            font-weight: 600;
            font-family: 'Open Sans', sans-serif;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .replay-control-btn:hover {{
            border-color: var(--accent-blue);
            background: rgba(0, 212, 255, 0.1);
        }}
        
        .replay-control-btn.playing {{
            border-color: var(--accent-green);
            background: rgba(0, 255, 136, 0.1);
            color: var(--accent-green);
        }}
        
        .replay-slider-container {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .replay-slider {{
            flex: 1;
            height: 8px;
            -webkit-appearance: none;
            background: var(--bg-tertiary);
            border-radius: 4px;
            outline: none;
        }}
        
        .replay-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--accent-blue);
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0, 212, 255, 0.5);
            transition: transform 0.2s;
        }}
        
        .replay-slider::-webkit-slider-thumb:hover {{
            transform: scale(1.2);
        }}
        
        .replay-date-display {{
            min-width: 120px;
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px;
            font-weight: 600;
            color: var(--accent-blue);
            padding: 8px 16px;
            background: var(--bg-tertiary);
            border-radius: 8px;
        }}
        
        .replay-stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            padding: 15px;
            background: var(--bg-tertiary);
            border-radius: 10px;
        }}
        
        .replay-stat {{
            text-align: center;
        }}
        
        .replay-stat-value {{
            font-size: 20px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .replay-stat-label {{
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
        }}
        
        .replay-speed {{
            display: flex;
            align-items: center;
            gap: 10px;
            justify-content: center;
        }}
        
        .speed-btn {{
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            background: var(--bg-secondary);
            color: var(--text-secondary);
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .speed-btn:hover, .speed-btn.active {{
            border-color: var(--accent-blue);
            color: var(--accent-blue);
        }}
        
        /* Trade History Button */
        .history-btn {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            background: linear-gradient(135deg, #10b981, #059669);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 12px;
            font-weight: 600;
            font-family: 'Open Sans', sans-serif;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .history-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(16, 185, 129, 0.4);
        }}
        
        /* Marketing Hub Button - QDT Gradient Style */
        .marketing-btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 24px;
            background: var(--qdt-gradient);
            border: none;
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 14px;
            font-weight: 500;
            font-family: 'Open Sans', sans-serif;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .marketing-btn:hover {{
            filter: brightness(1.2);
        }}
        
        /* API Button - Next to Marketing Hub */
        .api-btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 24px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border: none;
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 14px;
            font-weight: 500;
            font-family: 'Open Sans', sans-serif;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .api-btn:hover {{
            filter: brightness(1.2);
            transform: translateY(-1px);
        }}
        
        /* API Modal */
        .api-modal {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            z-index: 10000;
            display: none;
            justify-content: center;
            align-items: center;
            backdrop-filter: blur(5px);
        }}
        
        .api-modal.active {{
            display: flex;
        }}
        
        .api-modal-content {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }}
        
        .api-modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-tertiary);
        }}
        
        .api-modal-title {{
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .api-modal-close {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 24px;
            cursor: pointer;
            padding: 5px;
            transition: color 0.2s;
        }}
        
        .api-modal-close:hover {{
            color: var(--accent-red);
        }}
        
        .api-modal-body {{
            padding: 24px;
        }}
        
        .api-asset-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}
        
        .api-asset-btn {{
            padding: 14px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }}
        
        .api-asset-btn:hover {{
            background: var(--qdt-primary);
            border-color: var(--qdt-primary);
            transform: translateY(-2px);
        }}
        
        .api-download-all {{
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            margin-top: 16px;
        }}
        
        .api-download-all:hover {{
            filter: brightness(1.1);
            transform: translateY(-2px);
        }}
        
        .api-info {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-top: 20px;
        }}
        
        .api-info-title {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 8px;
        }}
        
        .api-info-text {{
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.6;
        }}
        
        .api-info code {{
            background: var(--bg-primary);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--accent-blue);
        }}
        
        /* Trade History Modal */
        .history-modal {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            z-index: 10000;
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            padding-top: 40px;
            overflow-y: auto;
        }}
        
        .history-modal.active {{
            display: flex;
        }}
        
        .history-container {{
            width: 100%;
            max-width: 900px;
            max-height: 90vh;
            background: var(--bg-secondary);
            border-radius: 16px;
            margin: 20px;
            overflow-y: auto;
        }}
        
        .history-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 25px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
        }}
        
        .history-title {{
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .history-close {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }}
        
        .history-close:hover {{
            background: var(--accent-red);
            border-color: var(--accent-red);
            color: white;
        }}
        
        .history-summary {{
            display: flex;
            flex-direction: column;
            gap: 15px;
            padding: 20px 25px;
            background: var(--bg-primary);
            border-bottom: 1px solid var(--border-color);
        }}
        
        .history-summary-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
        
        @media (max-width: 900px) {{
            .history-summary-row {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .strategy-option {{
            transition: all 0.2s ease;
        }}
        
        .strategy-option:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        
        .strategy-option.selected {{
            border-color: var(--accent-gold) !important;
            background: rgba(251, 191, 36, 0.15) !important;
        }}
        
        .history-stat {{
            text-align: center;
        }}
        
        .history-stat-value {{
            font-size: 24px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .history-stat-value.positive {{
            color: var(--accent-green);
        }}
        
        .history-stat-value.negative {{
            color: var(--accent-red);
        }}
        
        .history-stat-label {{
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-top: 4px;
        }}
        
        .history-table-container {{
            padding: 20px;
            max-height: 500px;
            overflow-y: auto;
        }}
        
        .history-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        
        .history-table th {{
            padding: 12px 15px;
            text-align: left;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: sticky;
            top: 0;
        }}
        
        .history-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}
        
        .history-table tr:hover {{
            background: rgba(255, 255, 255, 0.02);
        }}
        
        .history-table .signal-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        
        .history-table .signal-badge.bullish {{
            background: rgba(0, 255, 136, 0.15);
            color: var(--accent-green);
        }}
        
        .history-table .signal-badge.bearish {{
            background: rgba(255, 51, 102, 0.15);
            color: var(--accent-red);
        }}
        
        .history-table .outcome {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-weight: 600;
        }}
        
        .history-table .outcome.win {{
            color: var(--accent-green);
        }}
        
        .history-table .outcome.loss {{
            color: var(--accent-red);
        }}
        
        .history-table .pnl {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }}
        
        .history-table .pnl.positive {{
            color: var(--accent-green);
        }}
        
        .history-table .pnl.negative {{
            color: var(--accent-red);
        }}
        
        .history-footer {{
            padding: 15px 25px;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border-color);
            font-size: 12px;
            color: var(--text-secondary);
            text-align: center;
        }}
        
        /* Legend */
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }}
        
        .legend-color {{
            width: 20px;
            height: 4px;
            border-radius: 2px;
        }}
        
        .legend-color.bullish {{
            background: #00ff88;
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }}
        
        .legend-color.bearish {{
            background: #ff3366;
            box-shadow: 0 0 10px rgba(255, 51, 102, 0.5);
        }}
        
        .legend-color.neutral {{
            background: #555;
        }}
        
        .legend-color.live {{
            background: linear-gradient(90deg, var(--accent-gold), var(--accent-blue));
            width: 30px;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 12px;
            border-top: 1px solid var(--border-color);
            margin-top: 30px;
        }}
        
        .footer a {{
            color: var(--accent-blue);
            text-decoration: none;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .logo {{
                font-size: 32px;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .asset-select {{
                min-width: 200px;
                font-size: 16px;
            }}
        }}
        
        /* ========== AUTH0 LOGIN OVERLAY ========== */
        .login-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #0C0D0F 0%, #0F1C25 50%, #1C2934 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
            opacity: 1;
            transition: opacity 0.5s ease;
        }}
        
        .login-overlay.hidden {{
            opacity: 0;
            pointer-events: none;
        }}
        
        .login-box {{
            background: rgba(15, 28, 37, 0.95);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 50px 60px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            max-width: 450px;
            width: 90%;
        }}
        
        .login-logo {{
            font-family: 'Montserrat', sans-serif;
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 10px;
        }}
        
        .login-logo .qdt {{
            color: #ffffff;
        }}
        
        .login-logo .dot {{
            color: var(--accent-gold);
        }}
        
        .login-logo .nexus {{
            color: var(--qdt-primary);
        }}
        
        .login-subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
            margin-bottom: 40px;
        }}
        
        .login-btn {{
            background: linear-gradient(135deg, var(--qdt-primary) 0%, var(--accent-blue) 100%);
            color: #0C0D0F;
            border: none;
            padding: 16px 50px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-family: 'Montserrat', sans-serif;
        }}
        
        .login-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(148, 195, 198, 0.3);
        }}
        
        .login-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        .login-status {{
            margin-top: 20px;
            font-size: 13px;
            color: var(--text-muted);
        }}
        
        .user-info {{
            position: fixed;
            top: 20px;
            right: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            z-index: 1000;
            background: var(--bg-secondary);
            padding: 8px 16px;
            border-radius: 30px;
            border: 1px solid var(--border-color);
        }}
        
        .user-avatar {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--accent-blue);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 14px;
            color: white;
        }}
        
        .user-email {{
            color: var(--text-secondary);
            font-size: 13px;
            max-width: 180px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .logout-btn {{
            background: transparent;
            border: 1px solid var(--accent-red);
            color: var(--accent-red);
            padding: 6px 12px;
            font-size: 11px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .logout-btn:hover {{
            background: var(--accent-red);
            color: white;
        }}
        
        /* ==================== PERFORMANCE BOX (HERO SECTION) ==================== */
        .performance-box {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, rgba(0, 212, 255, 0.05) 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }}
        
        .performance-box::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-green), var(--qdt-primary), var(--accent-green));
        }}
        
        .performance-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .performance-title-section {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .performance-title {{
            font-size: 16px;
            font-weight: 700;
            color: var(--text-primary);
        }}
        
        .performance-badge {{
            background: linear-gradient(135deg, var(--accent-green), #059669);
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        
        .performance-badge.raw {{
            background: linear-gradient(135deg, var(--text-muted), #666);
        }}
        
        .performance-toggle {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .performance-toggle-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .performance-toggle-btn:hover {{
            background: var(--hover-bg);
            border-color: var(--qdt-primary);
        }}
        
        .performance-toggle-btn.active {{
            background: var(--qdt-primary);
            border-color: var(--qdt-primary);
            color: #000;
        }}
        
        .performance-metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }}
        
        @media (max-width: 768px) {{
            .performance-metrics {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        .perf-metric {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .perf-metric:hover {{
            transform: translateY(-2px);
            border-color: var(--qdt-primary);
        }}
        
        .perf-metric.highlight {{
            border-color: var(--accent-green);
            box-shadow: 0 0 20px rgba(0, 214, 143, 0.15);
        }}
        
        .perf-metric-value {{
            font-size: 32px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 6px;
        }}
        
        .perf-metric-value.positive {{
            color: var(--accent-green);
        }}
        
        .perf-metric-value.negative {{
            color: var(--accent-red);
        }}
        
        .perf-metric-value.neutral {{
            color: var(--qdt-primary);
        }}
        
        .perf-metric-label {{
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .perf-metric-sublabel {{
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        
        /* Quant Details Button */
        .quant-details-btn {{
            background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary));
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .quant-details-btn:hover {{
            background: var(--qdt-primary);
            color: white;
            border-color: var(--qdt-primary);
            transform: translateY(-1px);
        }}
        
        /* Quant Modal */
        .quant-modal {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            z-index: 9999;
            display: none;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(4px);
        }}
        
        .quant-modal-content {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            max-width: 650px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            position: relative;
        }}
        
        .quant-modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .quant-modal-title {{
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .quant-modal-close {{
            font-size: 28px;
            color: var(--text-secondary);
            cursor: pointer;
            line-height: 1;
        }}
        
        .quant-modal-close:hover {{
            color: var(--accent-red);
        }}
        
        .quant-metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }}
        
        .quant-metric-card {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }}
        
        .quant-metric-value {{
            font-size: 24px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 6px;
        }}
        
        .quant-metric-value.positive {{ color: var(--accent-green); }}
        .quant-metric-value.negative {{ color: var(--accent-red); }}
        .quant-metric-value.neutral {{ color: var(--qdt-primary); }}
        .quant-metric-value.warning {{ color: var(--accent-yellow); }}
        
        .quant-metric-label {{
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .quant-metric-desc {{
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        
        .quant-section-title {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            margin: 20px 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .performance-message {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-top: 16px;
            padding: 12px 20px;
            background: linear-gradient(135deg, rgba(0, 214, 143, 0.1) 0%, rgba(0, 214, 143, 0.05) 100%);
            border: 1px dashed var(--accent-green);
            border-radius: 10px;
        }}
        
        .perf-message-icon {{
            font-size: 16px;
        }}
        
        .perf-message-text {{
            font-size: 13px;
            color: var(--accent-green);
            font-weight: 500;
        }}
        
        /* Legacy optimization classes for compatibility */
        .optimization-hero {{ display: none; }}
        .optimization-header {{ display: none; }}
        .optimization-content {{ display: none; }}
        .optimization-note {{ display: none; }}
        
        .optimization-box-sublabel {{
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        
        .optimization-arrow {{
            font-size: 24px;
            color: var(--text-muted);
        }}
        
        .optimization-actions {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .optimization-btn {{
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            border: none;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .optimization-btn.toggle {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
        }}
        
        .optimization-btn.toggle:hover {{
            border-color: var(--qdt-primary);
            color: var(--qdt-primary);
        }}
        
        .optimization-btn.toggle.active {{
            background: var(--qdt-primary);
            border-color: var(--qdt-primary);
            color: white;
        }}
        
        .optimization-btn.cta {{
            background: linear-gradient(135deg, var(--accent-green), #059669);
            color: white;
        }}
        
        .optimization-btn.cta:hover {{
            filter: brightness(1.1);
            transform: translateY(-1px);
        }}
        
        .optimization-note {{
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border-color);
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .optimization-note strong {{
            color: var(--accent-gold);
        }}
        
        @media (max-width: 900px) {{
            .optimization-content {{
                grid-template-columns: 1fr;
                text-align: center;
            }}
            
            .optimization-comparison {{
                flex-wrap: wrap;
            }}
            
            .optimization-actions {{
                flex-direction: row;
                justify-content: center;
            }}
        }}
    </style>
</head>
<body>
    <!-- Auth0 Login Overlay -->
    <div class="login-overlay" id="loginOverlay">
        <div class="login-box">
            <div class="login-logo">
                <span class="qdt">QDT</span><span class="nexus">Nexus</span>
            </div>
            <p class="login-subtitle">Quantum Data Technologies • Multi-Asset Intelligence</p>
            <button class="login-btn" id="loginBtn" onclick="login()">
                🔐 Sign In to Continue
            </button>
            <p class="login-status" id="loginStatus">Secure authentication powered by Auth0</p>
        </div>
    </div>
    
    <!-- User Info Bar (shown after login) -->
    <div class="user-info" id="userInfo" style="display: none;">
        <div class="user-avatar" id="userAvatar">?</div>
        <span class="user-email" id="userEmail">Loading...</span>
        <button class="logout-btn" onclick="logout()">Logout</button>
    </div>

    <div class="bg-gradient"></div>
    
    <div class="container" id="mainContainer" style="display: none;">
        <header class="header" style="position: relative;">
            <img src="data:image/jpeg;base64,{qdt_logo_b64}" alt="QDT Logo" style="
                position: absolute;
                top: 20px;
                left: 0;
                height: 50px;
                opacity: 0.9;
            ">
            <h1 class="logo"><span class="qdt-text">QDT</span><span class="nexus-text">Nexus</span></h1>
            <p class="subtitle">Quantum Data Technologies • Multi-Asset Intelligence</p>
            <div style="position: absolute; top: 30px; right: 0; display: flex; gap: 12px;">
                <button class="api-btn" onclick="openApiModal()">
                    📡 API Data
                </button>
                <a href="QDT_Marketing_Hub.html" class="marketing-btn">
                    📣 Marketing Hub
                </a>
            </div>
        </header>
        
        <!-- API Data Modal -->
        <div class="api-modal" id="apiModal">
            <div class="api-modal-content">
                <div class="api-modal-header">
                    <div class="api-modal-title">
                        📡 API Data Download
                    </div>
                    <button class="api-modal-close" onclick="closeApiModal()">&times;</button>
                </div>
                <div class="api-modal-body">
                    <p style="color: var(--text-secondary); margin-bottom: 16px;">
                        Download comprehensive forecast data in JSON format. Includes forecasts, signals, confidence metrics, and historical performance.
                    </p>
                    <div class="api-asset-grid" id="apiAssetGrid">
                        <!-- Populated by JavaScript -->
                    </div>
                    <button class="api-download-all" onclick="downloadAllApiData()">
                        ⬇️ Download All Assets (JSON)
                    </button>
                    <div class="api-info">
                        <div class="api-info-title">📋 What's Included</div>
                        <div class="api-info-text">
                            Each JSON file contains:<br>
                            • <code>current_price</code> - Latest price<br>
                            • <code>signal</code> - Current BULLISH/BEARISH/NEUTRAL signal<br>
                            • <code>confidence</code> - Signal confidence percentage<br>
                            • <code>forecasts</code> - Predictions for all horizons<br>
                            • <code>viable_horizons</code> - Optimized horizon selection<br>
                            • <code>confidence_stats</code> - Historical accuracy by signal type<br>
                            • <code>timestamp</code> - When data was generated
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="selector-container">
            <span class="selector-label">Select Asset:</span>
            <select id="assetSelector" class="asset-select" onchange="switchAsset(this.value)">
                <!-- Options populated by JS -->
            </select>
            <span id="projectIdBadge" style="padding: 8px 16px; background: var(--bg-tertiary); border: 1px solid var(--border-color); border-radius: 8px; font-family: 'JetBrains Mono', monospace; color: var(--accent-blue); font-size: 14px;"></span>
        </div>
        
        <!-- ==================== PERFORMANCE BOX (HERO SECTION) ==================== -->
        <div class="performance-box" id="performanceBox">
            <div class="performance-header">
                <div class="performance-title-section">
                    <span class="performance-title">📊 Strategy Performance</span>
                    <span class="performance-badge" id="perfBadge">OPTIMIZED</span>
                </div>
                <div class="performance-toggle">
                    <button class="performance-toggle-btn active" id="perfOptBtn" onclick="showOptimizedPerf()">Optimized</button>
                    <button class="performance-toggle-btn" id="perfRawBtn" onclick="showRawPerf()">Raw</button>
                </div>
            </div>
            
            <div class="performance-metrics">
                <div class="perf-metric highlight">
                    <div class="perf-metric-value positive" id="perfTotalReturn">+111.4%</div>
                    <div class="perf-metric-label">Total Return</div>
                    <div class="perf-metric-sublabel">Signal-Following</div>
                </div>
                
                <div class="perf-metric">
                    <div class="perf-metric-value neutral" id="perfSharpe">2.96</div>
                    <div class="perf-metric-label">Sharpe Ratio</div>
                    <div class="perf-metric-sublabel">Risk-Adjusted</div>
                </div>
                
                <div class="perf-metric">
                    <div class="perf-metric-value neutral" id="perfDA">68.2%</div>
                    <div class="perf-metric-label">Avg DA</div>
                    <div class="perf-metric-sublabel">Directional Accuracy</div>
                </div>
                
                <div class="perf-metric" style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <button class="quant-details-btn" onclick="showQuantDetails()">
                        📊 Quant Details
                    </button>
                    <div class="perf-metric-sublabel" style="margin-top: 8px;">Advanced Metrics</div>
                </div>
            </div>
            
            <div class="performance-message" id="perfMessage" style="display: none;">
                <span class="perf-message-icon">✨</span>
                <span class="perf-message-text">Raw model performs best - no optimization needed!</span>
            </div>
        </div>
        
        <!-- Signal Confidence Panel - Shows confidence metrics (signal already shown in Optimization Hero) -->
        <div class="live-signal-panel" id="liveSignalPanel">
            <div class="live-signal-main" style="justify-content: flex-start;">
                <div class="live-signal-confidence">
                    <div class="confidence-tier" id="confidenceTier">
                        <span class="tier-badge high" id="tierBadge">HIGH CONFIDENCE</span>
                    </div>
                    <div class="confidence-accuracy" id="confidenceAccuracy">
                        <span class="accuracy-value">72.5%</span>
                        <span class="accuracy-label">Signal-Following Win Rate</span>
                    </div>
                </div>
            </div>
            <div class="live-signal-details" id="liveSignalDetails">
                <div class="detail-item">
                    <span class="detail-label">Directional Consensus</span>
                    <span class="detail-value" id="signalStrength">55% consensus</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Horizons Agreeing</span>
                    <span class="detail-value" id="horizonAgreement">8/10</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Based On</span>
                    <span class="detail-value" id="sampleSize">109 similar signals</span>
                </div>
                <span class="info-icon" style="margin-left: 10px; font-size: 14px;">?
                    <div class="info-tooltip" style="width: 350px; left: auto; right: 0; transform: none; bottom: 100%; top: auto;">
                        <h4>🎯 Signal-Following Confidence</h4>
                        
                        <div class="metric-explain" style="margin-top: 10px; background: var(--bg-tertiary); padding: 10px; border-radius: 6px; display: block; border: none;">
                            <div style="color: var(--accent-green); font-weight: 600; margin-bottom: 6px;">📈 Base Win Rate</div>
                            <div style="font-size: 10px;">Historical % of profitable trades using the Signal-Following strategy (enter on signal, exit on flip)</div>
                        </div>
                        
                        <div class="metric-explain" style="margin-top: 8px; background: var(--bg-tertiary); padding: 10px; border-radius: 6px; display: block; border: none;">
                            <div style="color: var(--accent-blue); font-weight: 600; margin-bottom: 6px;">⚡ Adjustments</div>
                            <div style="font-size: 10px;">• <span style="color: #00ff88;">Strong Signal</span>: +3-5% (high consensus)</div>
                            <div style="font-size: 10px;">• <span style="color: #00ff88;">Optimized</span>: +5% (refined horizons)</div>
                            <div style="font-size: 10px;">• <span style="color: #00ff88;">RSI/EMA/Ichimoku ✓</span>: +3% each (aligned)</div>
                            <div style="font-size: 10px;">• <span style="color: #ff3366;">Conflicts</span>: -3-5% (indicators disagree)</div>
                        </div>
                        
                        <div class="metric-explain" style="margin-top: 8px; background: var(--bg-tertiary); padding: 10px; border-radius: 6px; display: block; border: none;">
                            <div style="color: var(--accent-gold); font-weight: 600; margin-bottom: 6px;">📈 Directional Consensus</div>
                            <div style="font-size: 10px;">Net slope direction between forecast horizons</div>
                            <div style="font-size: 10px; margin-top: 4px;">Measures how consistently the forecast line trends up/down</div>
                        </div>
                        
                        <div class="metric-explain" style="margin-top: 8px; background: var(--bg-tertiary); padding: 10px; border-radius: 6px; display: block; border: none;">
                            <div style="color: #00ff88; font-weight: 600; margin-bottom: 6px;">✅ Horizons Agreeing</div>
                            <div style="font-size: 10px;">How many individual horizons agree with signal direction</div>
                            <div style="font-size: 10px; margin-top: 4px;">9/10 = 90% of forecasts predict same direction</div>
                        </div>
                        
                        <div class="metric-explain" style="margin-top: 8px; background: var(--bg-tertiary); padding: 10px; border-radius: 6px; display: block; border: none;">
                            <div style="color: var(--accent-purple); font-weight: 600; margin-bottom: 6px;">🏆 Confidence Tiers</div>
                            <div style="font-size: 10px;">• <span style="color: #00ff88;">HIGH</span>: 75%+ (strong edge)</div>
                            <div style="font-size: 10px;">• <span style="color: #ffaa00;">MEDIUM</span>: 60-74% (moderate edge)</div>
                            <div style="font-size: 10px;">• <span style="color: #ff3366;">LOW</span>: &lt;60% (weak edge)</div>
                        </div>
                        
                        <p style="margin-top: 12px; font-style: italic; font-size: 9px;">💡 Strategy: Follow the signal until it flips. Higher confidence = higher historical win rate.</p>
                    </div>
                </span>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-header">
                <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <div class="toggle-container" id="rsiToggleContainer">
                        <span class="toggle-label">RSI Filter</span>
                        <div class="toggle-switch" id="rsiToggle" onclick="toggleRSI()"></div>
                        <span class="toggle-status off" id="rsiStatus">OFF</span>
                    </div>
                    <div class="toggle-container" id="emaToggleContainer">
                        <span class="toggle-label">EMA</span>
                        <select id="emaSelect" onchange="changeEMA(this.value)" style="
                            background: var(--bg-secondary);
                            border: 1px solid var(--border-color);
                            color: var(--text-primary);
                            padding: 4px 8px;
                            border-radius: 4px;
                            font-size: 11px;
                            cursor: pointer;
                        ">
                            <option value="0">OFF</option>
                            <option value="9">9</option>
                            <option value="20">20</option>
                            <option value="50">50</option>
                            <option value="100">100</option>
                            <option value="200">200</option>
                        </select>
                    </div>
                    <div class="toggle-container" id="ichimokuToggleContainer">
                        <span class="toggle-label">Ichimoku</span>
                        <div class="toggle-switch" id="ichimokuToggle" onclick="toggleIchimoku()"></div>
                        <span class="toggle-status off" id="ichimokuStatus">OFF</span>
                    </div>
                    <div class="toggle-container" id="traderModeContainer" style="margin-left: 8px;">
                        <span class="toggle-label" style="color: var(--accent-gold);">🎯 Trader</span>
                        <div class="toggle-switch" id="traderModeToggle" onclick="toggleTraderMode()"></div>
                        <span class="toggle-status off" id="traderModeStatus">OFF</span>
                    </div>
                    <span class="info-icon" style="margin-left: 5px;">?
                        <div class="info-tooltip" style="width: 360px; left: auto; right: 0; transform: none;">
                            <h4>❓ How Indicator Filters Work</h4>
                            <p>When enabled, indicators act as <b>confluence filters</b>. Signals that <b>agree</b> stay colored; signals that <b>conflict</b> turn grey.</p>
                            
                            <div class="metric-explain" style="margin-top: 12px; background: var(--bg-tertiary); padding: 10px; border-radius: 6px; display: block; border: none;">
                                <div style="color: var(--accent-gold); font-weight: 600; margin-bottom: 6px;">📊 RSI Filter</div>
                                <div style="font-size: 10px;">• Bullish filtered when RSI is overbought</div>
                                <div style="font-size: 10px;">• Bearish filtered when RSI is oversold</div>
                            </div>
                            
                            <div class="metric-explain" style="margin-top: 8px; background: var(--bg-tertiary); padding: 10px; border-radius: 6px; display: block; border: none;">
                                <div style="color: #FFD700; font-weight: 600; margin-bottom: 6px;">📈 EMA Filter</div>
                                <div style="font-size: 10px;">• Bullish valid when price > EMA</div>
                                <div style="font-size: 10px;">• Bearish valid when price < EMA</div>
                            </div>
                            
                            <div class="metric-explain" style="margin-top: 8px; background: var(--bg-tertiary); padding: 10px; border-radius: 6px; display: block; border: none;">
                                <div style="color: #26a65b; font-weight: 600; margin-bottom: 6px;">☁️ Ichimoku Filter</div>
                                <div style="font-size: 10px;">• Bullish valid when cloud is green (A > B)</div>
                                <div style="font-size: 10px;">• Bearish valid when cloud is red (A < B)</div>
                            </div>
                            
                            <p style="margin-top: 12px; font-style: italic; font-size: 9px;">💡 Multiple filters = signals must match ALL to stay colored.</p>
                        </div>
                    </span>
                    <button class="history-btn" onclick="openTradeHistory()">
                        📋 Trade History
                    </button>
                </div>
            </div>
            <div id="snake-chart"></div>
        </div>
        
        <!-- Replay Modal -->
        <div class="replay-modal" id="replayModal">
            <div class="replay-header">
                <div class="replay-title">
                    🎬 Snake Chart Replay
                    <span class="asset-badge" id="replayAssetName"></span>
                </div>
                <button class="replay-close" onclick="closeReplay()">×</button>
            </div>
            <div class="replay-chart-container">
                <div id="replay-chart"></div>
            </div>
            <div class="replay-controls">
                <div class="replay-slider-container">
                    <span style="color: var(--text-secondary); font-size: 12px;">Start</span>
                    <input type="range" class="replay-slider" id="replaySlider" min="0" max="100" value="0" oninput="seekReplay(this.value)">
                    <span style="color: var(--text-secondary); font-size: 12px;">End</span>
                </div>
                <div class="replay-buttons">
                    <button class="replay-control-btn" onclick="stepReplay(-10)">⏪ -10</button>
                    <button class="replay-control-btn" onclick="stepReplay(-1)">◀ -1</button>
                    <button class="replay-control-btn" id="playPauseBtn" onclick="togglePlayReplay()">▶️ Play</button>
                    <button class="replay-control-btn" onclick="stepReplay(1)">+1 ▶</button>
                    <button class="replay-control-btn" onclick="stepReplay(10)">+10 ⏩</button>
                </div>
                <div class="replay-stats" id="replayStats">
                    <!-- Stats populated by JS -->
                </div>
                <div class="replay-speed">
                    <span style="color: var(--text-secondary); font-size: 12px;">Speed:</span>
                    <button class="speed-btn" onclick="setReplaySpeed(500)">0.5x</button>
                    <button class="speed-btn active" onclick="setReplaySpeed(200)">1x</button>
                    <button class="speed-btn" onclick="setReplaySpeed(100)">2x</button>
                    <button class="speed-btn" onclick="setReplaySpeed(50)">4x</button>
                </div>
            </div>
        </div>
        
        <!-- Trade History Modal -->
        <div class="history-modal" id="historyModal">
            <div class="history-container">
                <div class="history-header">
                    <div class="history-title">
                        📋 Trade Verification History
                        <span id="historyAssetName" style="font-size: 14px; color: var(--accent-blue);"></span>
                    </div>
                    <button class="history-close" onclick="closeTradeHistory()">×</button>
                </div>
                <div class="history-summary" id="historySummary">
                    <!-- Summary stats populated by JS -->
                </div>
                <div class="history-table-container">
                    <table class="history-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Entry</th>
                                <th>Signal</th>
                                <th>Entry $</th>
                                <th>Exit $</th>
                                <th>Days</th>
                                <th>Exit Reason</th>
                                <th>P&L</th>
                            </tr>
                        </thead>
                        <tbody id="historyTableBody">
                            <!-- Rows populated by JS -->
                        </tbody>
                    </table>
                </div>
                <div class="history-footer">
                    🐍 Signal-Following: Enter on signal change, exit when signal flips • Matches Equity Curve performance
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <div id="rsi-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="confidence-chart"></div>
            <div class="confidence-chart-legend">
                <span class="conf-legend-item high">■ High (75%+)</span>
                <span class="conf-legend-item medium">■ Medium (55-75%)</span>
                <span class="conf-legend-item low">■ Low (&lt;55%)</span>
            </div>
        </div>
        
        <!-- Quant Details Modal -->
        <div id="quantModal" class="quant-modal">
            <div class="quant-modal-content">
                <div class="quant-modal-header">
                    <div class="quant-modal-title">📊 Advanced Quant Metrics</div>
                    <span class="quant-modal-close" onclick="closeQuantModal()">&times;</span>
                </div>
                
                <div class="quant-section-title">📈 Risk & Return</div>
                <div class="quant-metrics-grid" id="quantRiskReturn">
                    <!-- Populated by JS -->
                </div>
                
                <div class="quant-section-title">🎯 Trade Statistics</div>
                <div class="quant-metrics-grid" id="quantTradeStats">
                    <!-- Populated by JS -->
                </div>
                
                <div class="quant-section-title">💰 Profitability</div>
                <div class="quant-metrics-grid" id="quantProfitability">
                    <!-- Populated by JS -->
                </div>
            </div>
        </div>
        
        <!-- Confidence Explanation Modal -->
        <div id="confidenceModal" class="confidence-modal" style="display: none;">
            <div class="confidence-modal-content">
                <span class="confidence-modal-close" onclick="document.getElementById('confidenceModal').style.display='none'">&times;</span>
                <h3>🎯 Live Signal Confidence</h3>
                <p>This score measures how confident we are in <strong>today's specific forecast</strong>, based on 5 factors:</p>
                <div class="conf-factor">
                    <div class="conf-factor-name">1. Horizon Agreement (30%)</div>
                    <div class="conf-factor-desc">How many horizons predict the same direction? More agreement = higher confidence.</div>
                </div>
                <div class="conf-factor">
                    <div class="conf-factor-name">2. Snake Tightness (25%)</div>
                    <div class="conf-factor-desc">How tight is the forecast band? A narrow "thin snake" means predictions agree on price level.</div>
                </div>
                <div class="conf-factor">
                    <div class="conf-factor-name">3. Signal Strength (20%)</div>
                    <div class="conf-factor-desc">How strong is the net probability? Closer to ±100% = stronger conviction.</div>
                </div>
                <div class="conf-factor">
                    <div class="conf-factor-name">4. Slope Steepness (15%)</div>
                    <div class="conf-factor-desc">How steep is the predicted move? Steeper slopes = more conviction in direction.</div>
                </div>
                <div class="conf-factor">
                    <div class="conf-factor-name">5. Historical DA at Similar Strength (10%)</div>
                    <div class="conf-factor-desc">How accurate was the model <em>when signals had similar strength</em> in the past? We bucket historical signals by strength (e.g., 70-85%) and show the DA for signals in today's bucket.</div>
                </div>
                <p class="conf-note">Note: Confidence adapts when you switch between Raw and Optimized views.</p>
            </div>
        </div>
        
        <!-- Ensemble Health / Build Recommendations -->
        <div class="health-check expanded" id="healthCheck">
            <!-- Health check populated by JS -->
        </div>
        
        <!-- Price Targets Section -->
        <div class="price-targets-section" id="priceTargetsSection">
            <!-- Price targets populated by JS -->
        </div>
        
        <div class="chart-container">
            <div id="equity-chart"></div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color bullish"></div>
                <span>Bullish Signal</span>
            </div>
            <div class="legend-item">
                <div class="legend-color bearish"></div>
                <span>Bearish Signal</span>
            </div>
            <div class="legend-item">
                <div class="legend-color neutral"></div>
                <span>Neutral</span>
            </div>
            <div class="legend-item">
                <div class="legend-color live"></div>
                <span>Live Forecast</span>
            </div>
        </div>
        
        <footer class="footer">
            <p>QDTNexus Dashboard • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} • Powered by Quantum Data Technologies</p>
        </footer>
    </div>
    
    <script>
        // ========== AUTH0 CONFIGURATION ==========
        const AUTH0_CONFIG = {{
            domain: 'dev-j4tg31o6iska3hzh.us.auth0.com',
            clientId: 'TwgwSZtFKAZENleTG682G04GEjzfNP7M',
            authorizationParams: {{
                redirect_uri: window.location.origin + window.location.pathname
            }}
        }};
        
        let auth0Client = null;
        let currentUser = null;
        
        // Initialize Auth0
        async function initAuth0() {{
            try {{
                auth0Client = await auth0.createAuth0Client(AUTH0_CONFIG);
                
                // Handle redirect callback
                if (window.location.search.includes('code=') && window.location.search.includes('state=')) {{
                    await auth0Client.handleRedirectCallback();
                    window.history.replaceState({{}}, document.title, window.location.pathname);
                }}
                
                // Check if user is authenticated
                const isAuthenticated = await auth0Client.isAuthenticated();
                
                if (isAuthenticated) {{
                    currentUser = await auth0Client.getUser();
                    showDashboard();
                }} else {{
                    showLogin();
                }}
            }} catch (error) {{
                console.error('Auth0 initialization error:', error);
                document.getElementById('loginStatus').textContent = 'Error: ' + error.message;
                document.getElementById('loginBtn').disabled = false;
            }}
        }}
        
        async function login() {{
            document.getElementById('loginBtn').disabled = true;
            document.getElementById('loginStatus').textContent = 'Redirecting to login...';
            try {{
                await auth0Client.loginWithRedirect();
            }} catch (error) {{
                console.error('Login error:', error);
                document.getElementById('loginStatus').textContent = 'Login failed: ' + error.message;
                document.getElementById('loginBtn').disabled = false;
            }}
        }}
        
        async function logout() {{
            await auth0Client.logout({{
                logoutParams: {{
                    returnTo: window.location.origin + window.location.pathname
                }}
            }});
        }}
        
        // User permissions from Auth0
        let userRole = 'viewer';
        let allowedAssets = [];
        let hasMarketingAccess = false;
        
        function showDashboard() {{
            // Hide login overlay
            document.getElementById('loginOverlay').classList.add('hidden');
            // Show main container
            document.getElementById('mainContainer').style.display = 'block';
            // Show user info
            const userInfo = document.getElementById('userInfo');
            userInfo.style.display = 'flex';
            document.getElementById('userEmail').textContent = currentUser.email || currentUser.name || 'User';
            document.getElementById('userAvatar').textContent = (currentUser.email || currentUser.name || 'U').charAt(0).toUpperCase();
            
            // Read custom claims from Auth0
            const namespace = 'https://qdt-ensemble.com';
            userRole = currentUser[`${{namespace}}/role`] || 'viewer';
            allowedAssets = currentUser[`${{namespace}}/allowed_assets`] || [];
            hasMarketingAccess = currentUser[`${{namespace}}/marketing_access`] || false;
            
            console.log('User permissions:', {{ role: userRole, assets: allowedAssets, marketing: hasMarketingAccess }});
            
            // Apply role-based access
            applyRoleBasedAccess();
        }}
        
        function applyRoleBasedAccess() {{
            // Filter asset dropdown based on permissions
            const selector = document.getElementById('assetSelector');
            const options = selector.querySelectorAll('option');
            
            // Check if user has access to ALL assets
            const hasAllAccess = allowedAssets.includes('ALL') || userRole === 'admin';
            
            if (!hasAllAccess && allowedAssets.length > 0) {{
                // Hide assets user doesn't have access to
                options.forEach(opt => {{
                    if (!allowedAssets.includes(opt.value)) {{
                        opt.style.display = 'none';
                        opt.disabled = true;
                    }}
                }});
                
                // Select first allowed asset
                const firstAllowed = allowedAssets.find(a => ASSET_DATA[a]);
                if (firstAllowed) {{
                    selector.value = firstAllowed;
                    switchAsset(firstAllowed);
                }}
            }}
            
            // Show/hide Marketing Hub link based on permission
            const marketingBtn = document.querySelector('.marketing-btn');
            if (marketingBtn) {{
                if (hasMarketingAccess || userRole === 'admin') {{
                    marketingBtn.style.display = 'inline-flex';
                }} else {{
                    marketingBtn.style.display = 'none';
                }}
            }}
            
            // Show role badge next to user info
            const roleColors = {{
                'admin': '#00d68f',
                'viewer': '#52718A'
            }};
            const roleBadge = document.createElement('span');
            roleBadge.textContent = userRole.toUpperCase();
            roleBadge.style.cssText = `
                background: ${{roleColors[userRole] || '#52718A'}};
                color: white;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: 600;
                margin-left: 8px;
            `;
            document.getElementById('userInfo').appendChild(roleBadge);
        }}
        
        function showLogin() {{
            document.getElementById('loginOverlay').classList.remove('hidden');
            document.getElementById('mainContainer').style.display = 'none';
            document.getElementById('userInfo').style.display = 'none';
            document.getElementById('loginBtn').disabled = false;
        }}
        
        // Initialize Auth0 on page load
        initAuth0();
        
        // ========== DASHBOARD CODE ==========
        
        // Embedded asset data
        const ASSET_DATA = {data_json};
        
        // API Data for downloads
        const API_DATA = {api_data_json};
        
        const ASSET_NAMES = Object.keys(ASSET_DATA);
        
        // ========== API MODAL FUNCTIONS ==========
        function openApiModal() {{
            const modal = document.getElementById('apiModal');
            modal.classList.add('active');
            populateApiAssetGrid();
        }}
        
        function closeApiModal() {{
            const modal = document.getElementById('apiModal');
            modal.classList.remove('active');
        }}
        
        function populateApiAssetGrid() {{
            const grid = document.getElementById('apiAssetGrid');
            grid.innerHTML = '';
            
            ASSET_NAMES.forEach(name => {{
                const btn = document.createElement('button');
                btn.className = 'api-asset-btn';
                btn.textContent = name.replace(/_/g, ' ');
                btn.onclick = () => downloadAssetData(name);
                grid.appendChild(btn);
            }});
        }}
        
        function downloadAssetData(assetName) {{
            const data = API_DATA[assetName];
            if (!data) {{
                alert('No data available for ' + assetName);
                return;
            }}
            
            const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = assetName.toLowerCase() + '_api_data.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
        
        function downloadAllApiData() {{
            const blob = new Blob([JSON.stringify(API_DATA, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'qdtnexus_all_assets_api_data.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
        
        // Close modal when clicking outside
        document.getElementById('apiModal').addEventListener('click', function(e) {{
            if (e.target === this) {{
                closeApiModal();
            }}
        }});
        let currentAsset = ASSET_NAMES[0];
        let rsiFilterEnabled = false;  // RSI filter is OFF by default
        let showIchimoku = false;  // Ichimoku Cloud OFF by default
        let emaPeriod = 0;  // 0 = OFF, otherwise the period (9, 20, 50, 100, 200)
        let isOptimalConfigApplied = false;  // Track if optimal config is applied
        let traderModeEnabled = false;  // Trader Mode: zoom to live forecasts
        let rawIsBestForAsset = false;  // Track if raw version is best for current asset
        
        // Enabled horizons state - for dynamic horizon toggling
        let enabledHorizons = [];  // Will be initialized per asset
        
        // ===================== LOCAL STORAGE PREFERENCES =====================
        const PREFS_KEY = 'qdt_dashboard_preferences';
        
        function savePreferences() {{
            const prefs = {{
                currentAsset: currentAsset,
                rsiFilterEnabled: rsiFilterEnabled,
                showIchimoku: showIchimoku,
                emaPeriod: emaPeriod,
                isOptimalConfigApplied: isOptimalConfigApplied,
                enabledHorizons: enabledHorizons,
                savedAt: new Date().toISOString()
            }};
            try {{
                localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
                console.log('💾 Preferences saved:', prefs);
            }} catch (e) {{
                console.warn('Could not save preferences:', e);
            }}
        }}
        
        function loadPreferences() {{
            try {{
                const saved = localStorage.getItem(PREFS_KEY);
                if (!saved) {{
                    console.log('📭 No saved preferences found - using defaults');
                    return false;
                }}
                
                const prefs = JSON.parse(saved);
                console.log('📂 Loading saved preferences:', prefs);
                
                // Validate saved asset exists
                if (prefs.currentAsset && ASSET_NAMES.includes(prefs.currentAsset)) {{
                    currentAsset = prefs.currentAsset;
                    document.getElementById('assetSelector').value = currentAsset;
                }}
                
                // Restore indicator settings
                if (typeof prefs.rsiFilterEnabled === 'boolean') {{
                    rsiFilterEnabled = prefs.rsiFilterEnabled;
                }}
                if (typeof prefs.showIchimoku === 'boolean') {{
                    showIchimoku = prefs.showIchimoku;
                }}
                if (typeof prefs.emaPeriod === 'number') {{
                    emaPeriod = prefs.emaPeriod;
                }}
                if (typeof prefs.isOptimalConfigApplied === 'boolean') {{
                    isOptimalConfigApplied = prefs.isOptimalConfigApplied;
                }}
                
                // Restore horizons if they match the current asset's available horizons
                const assetData = ASSET_DATA[currentAsset];
                const availableHorizons = assetData.horizons || [];
                if (Array.isArray(prefs.enabledHorizons) && prefs.enabledHorizons.length > 0) {{
                    // Only restore horizons that are available for this asset
                    const validHorizons = prefs.enabledHorizons.filter(h => availableHorizons.includes(h));
                    if (validHorizons.length > 0) {{
                        enabledHorizons = validHorizons;
                    }}
                }}
                
                // Update UI to reflect loaded preferences
                updatePreferencesUI();
                
                console.log('✅ Preferences loaded successfully');
                return true;
            }} catch (e) {{
                console.warn('Could not load preferences:', e);
                return false;
            }}
        }}
        
        function updatePreferencesUI() {{
            // Update RSI toggle UI
            const rsiToggle = document.getElementById('rsiToggle');
            const rsiStatus = document.getElementById('rsiStatus');
            const rsiContainer = document.getElementById('rsiToggleContainer');
            if (rsiToggle && rsiStatus && rsiContainer) {{
                if (rsiFilterEnabled) {{
                    rsiToggle.classList.add('on');
                    rsiStatus.classList.add('on');
                    rsiStatus.classList.remove('off');
                    rsiStatus.textContent = 'ON';
                    rsiContainer.classList.add('active');
                }} else {{
                    rsiToggle.classList.remove('on');
                    rsiStatus.classList.remove('on');
                    rsiStatus.classList.add('off');
                    rsiStatus.textContent = 'OFF';
                    rsiContainer.classList.remove('active');
                }}
            }}
            
            // Update EMA dropdown UI
            const emaSelect = document.getElementById('emaSelect');
            const emaContainer = document.getElementById('emaToggleContainer');
            if (emaSelect && emaContainer) {{
                emaSelect.value = emaPeriod;
                if (emaPeriod > 0) {{
                    emaContainer.classList.add('active');
                }} else {{
                    emaContainer.classList.remove('active');
                }}
            }}
            
            // Update Ichimoku toggle UI
            const ichimokuToggle = document.getElementById('ichimokuToggle');
            const ichimokuStatus = document.getElementById('ichimokuStatus');
            const ichimokuContainer = document.getElementById('ichimokuToggleContainer');
            if (ichimokuToggle && ichimokuStatus && ichimokuContainer) {{
                if (showIchimoku) {{
                    ichimokuToggle.classList.add('on');
                    ichimokuStatus.classList.add('on');
                    ichimokuStatus.classList.remove('off');
                    ichimokuStatus.textContent = 'ON';
                    ichimokuContainer.classList.add('active');
                }} else {{
                    ichimokuToggle.classList.remove('on');
                    ichimokuStatus.classList.remove('on');
                    ichimokuStatus.classList.add('off');
                    ichimokuStatus.textContent = 'OFF';
                    ichimokuContainer.classList.remove('active');
                }}
            }}
        }}
        
        function clearPreferences() {{
            try {{
                localStorage.removeItem(PREFS_KEY);
                console.log('🗑️ Preferences cleared');
                location.reload();  // Reload to reset to defaults
            }} catch (e) {{
                console.warn('Could not clear preferences:', e);
            }}
        }}
        // =====================================================================
        
        // Populate selector
        const selector = document.getElementById('assetSelector');
        ASSET_NAMES.forEach(name => {{
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name.replace('_', ' ');
            selector.appendChild(opt);
        }});
        
        function toggleRSI() {{
            rsiFilterEnabled = !rsiFilterEnabled;
            
            // Update toggle UI
            const toggle = document.getElementById('rsiToggle');
            const status = document.getElementById('rsiStatus');
            const container = document.getElementById('rsiToggleContainer');
            
            if (rsiFilterEnabled) {{
                toggle.classList.add('on');
                status.classList.add('on');
                status.classList.remove('off');
                status.textContent = 'ON';
                container.classList.add('active');
            }} else {{
                toggle.classList.remove('on');
                status.classList.remove('on');
                status.classList.add('off');
                status.textContent = 'OFF';
                container.classList.remove('active');
            }}
            
            // Recalculate everything with new filter state
            updateStats();
            updateCharts();
            savePreferences();  // Save user preference
        }}
        
        function changeEMA(value) {{
            emaPeriod = parseInt(value);
            
            const container = document.getElementById('emaToggleContainer');
            
            if (emaPeriod > 0) {{
                container.classList.add('active');
            }} else {{
                container.classList.remove('active');
            }}
            
            updateStats();  // Update accuracy with new filter
            updateCharts();
            savePreferences();  // Save user preference
        }}
        
        function toggleIchimoku() {{
            showIchimoku = !showIchimoku;
            
            const toggle = document.getElementById('ichimokuToggle');
            const status = document.getElementById('ichimokuStatus');
            const container = document.getElementById('ichimokuToggleContainer');
            
            if (showIchimoku) {{
                toggle.classList.add('on');
                status.classList.add('on');
                status.classList.remove('off');
                status.textContent = 'ON';
                container.classList.add('active');
            }} else {{
                toggle.classList.remove('on');
                status.classList.remove('on');
                status.classList.add('off');
                status.textContent = 'OFF';
                container.classList.remove('active');
            }}
            
            updateStats();  // Update accuracy with new filter
            updateCharts();
            savePreferences();  // Save user preference
        }}
        
        // ===================== TRADER MODE =====================
        // Zooms chart to show live forecasts + recent ~60 days of historical data
        function toggleTraderMode() {{
            traderModeEnabled = !traderModeEnabled;
            
            const toggle = document.getElementById('traderModeToggle');
            const status = document.getElementById('traderModeStatus');
            const container = document.getElementById('traderModeContainer');
            
            if (traderModeEnabled) {{
                toggle.classList.add('on');
                status.classList.add('on');
                status.classList.remove('off');
                status.textContent = 'ON';
                container.classList.add('active');
            }} else {{
                toggle.classList.remove('on');
                status.classList.remove('on');
                status.classList.add('off');
                status.textContent = 'OFF';
                container.classList.remove('active');
            }}
            
            applyTraderModeZoom();
        }}
        
        function applyTraderModeZoom() {{
            const snakeChart = document.getElementById('snake-chart');
            if (!snakeChart) return;
            
            const data = ASSET_DATA[currentAsset];
            if (!data || !data.dates || data.dates.length === 0) return;
            
            if (traderModeEnabled) {{
                // Trader Mode: Show last ~45 days of historical data + ALL forecasts + padding
                const historicalDates = data.dates;
                const forecastDates = data.forecast_dates || [];
                
                // Start: ~45 days before the last historical date
                const lastHistoricalIdx = Math.max(0, historicalDates.length - 45);
                const startDate = historicalDates[lastHistoricalIdx];
                
                // End: Include forecast band + confidence box + padding for blank space
                let endDate;
                const lastDataDate = new Date(historicalDates[historicalDates.length - 1]);
                
                if (data.live_forecast && Object.keys(data.live_forecast.forecasts).length > 0) {{
                    // Calculate forecast band end date
                    const currentEnabledHorizons = enabledHorizons.length > 0 ? enabledHorizons : (data.horizons || []);
                    const horizons = Object.keys(data.live_forecast.forecasts)
                        .map(k => parseInt(k.replace('D+', '')))
                        .filter(h => currentEnabledHorizons.includes(h))
                        .sort((a, b) => a - b);
                    
                    let maxHorizon = 0;
                    horizons.forEach(h => {{
                        const key = 'D+' + h;
                        const pred = data.live_forecast.forecasts[key];
                        if (pred !== null && pred !== undefined && h > maxHorizon) {{
                            maxHorizon = h;
                        }}
                    }});
                    
                    // Forecast band ends at: lastDataDate + maxHorizon
                    const maxForecastDateObj = new Date(lastDataDate);
                    maxForecastDateObj.setDate(maxForecastDateObj.getDate() + maxHorizon);
                    
                    // Confidence box is positioned: maxForecastDate + padding (20% of forecast duration or min 10 days)
                    const forecastStartTime = lastDataDate.getTime();
                    const forecastEndTime = maxForecastDateObj.getTime();
                    const forecastDuration = forecastEndTime - forecastStartTime;
                    const confidencePaddingDays = Math.max(forecastDuration * 0.20 / (1000 * 60 * 60 * 24), 10);
                    
                    const confidenceBoxDateObj = new Date(maxForecastDateObj);
                    confidenceBoxDateObj.setDate(confidenceBoxDateObj.getDate() + confidencePaddingDays);
                    
                    // Add additional padding after confidence box for blank space (at least 20 days)
                    const blankSpacePadding = 20;
                    confidenceBoxDateObj.setDate(confidenceBoxDateObj.getDate() + blankSpacePadding);
                    
                    endDate = confidenceBoxDateObj.toISOString().split('T')[0];
                }} else if (forecastDates.length > 0) {{
                    const lastForecast = new Date(forecastDates[forecastDates.length - 1]);
                    lastForecast.setDate(lastForecast.getDate() + 14);  // Add 14 days padding
                    endDate = lastForecast.toISOString().split('T')[0];
                }} else {{
                    lastDataDate.setDate(lastDataDate.getDate() + 30);  // Add 30 days for potential forecasts
                    endDate = lastDataDate.toISOString().split('T')[0];
                }}
                
                // Calculate Y-axis range including forecast bands
                const startDateObj = new Date(startDate);
                const endDateObj = new Date(endDate);
                
                // Find min/max Y values in visible X range (historical prices)
                let minY = Infinity, maxY = -Infinity;
                for (let i = 0; i < historicalDates.length; i++) {{
                    const dateObj = new Date(historicalDates[i]);
                    if (dateObj >= startDateObj && dateObj <= endDateObj) {{
                        if (data.prices[i] < minY) minY = data.prices[i];
                        if (data.prices[i] > maxY) maxY = data.prices[i];
                    }}
                }}
                
                // Include forecast band values if they fall within the visible range
                if (data.live_forecast && Object.keys(data.live_forecast.forecasts).length > 0) {{
                    const basePrice = data.prices[data.prices.length - 1];
                    const lastDataDate = new Date(historicalDates[historicalDates.length - 1]);
                    
                    // Get enabled horizons (use current enabled horizons or all available)
                    const currentEnabledHorizons = enabledHorizons.length > 0 ? enabledHorizons : (data.horizons || []);
                    const horizons = Object.keys(data.live_forecast.forecasts)
                        .map(k => parseInt(k.replace('D+', '')))
                        .filter(h => currentEnabledHorizons.includes(h))
                        .sort((a, b) => a - b);
                    
                    // Calculate forecast band min/max (same as in updateSnakeChart)
                    const amplifiedPrices = [];
                    let maxHorizon = 0;
                    
                    horizons.forEach(h => {{
                        const key = 'D+' + h;
                        const pred = data.live_forecast.forecasts[key];
                        if (pred !== null && pred !== undefined) {{
                            const amplified = basePrice + (pred - basePrice) * 3.0;
                            amplifiedPrices.push(amplified);
                            if (h > maxHorizon) maxHorizon = h;
                        }}
                    }});
                    
                    if (amplifiedPrices.length > 0) {{
                        const minForecast = Math.min(...amplifiedPrices);
                        const maxForecast = Math.max(...amplifiedPrices);
                        
                        // Check if forecast band is in visible range
                        const maxForecastDateObj = new Date(lastDataDate);
                        maxForecastDateObj.setDate(maxForecastDateObj.getDate() + maxHorizon);
                        
                        if (maxForecastDateObj >= startDateObj && lastDataDate <= endDateObj) {{
                            // Forecast band is visible, include it in Y-axis range
                            if (minForecast < minY) minY = minForecast;
                            if (maxForecast > maxY) maxY = maxForecast;
                        }}
                    }}
                }}
                
                // Add padding (5% on each side)
                if (minY !== Infinity && maxY !== -Infinity) {{
                    const padding = (maxY - minY) * 0.05;
                    minY = minY - padding;
                    maxY = maxY + padding;
                }}
                
                // Apply zoom to snake chart with explicit Y-axis range
                // Set X-axis first
                Plotly.relayout(snakeChart, {{
                    'xaxis.range': [startDate, endDate],
                    'xaxis.autorange': false
                }});
                
                // Set Y-axis range after a small delay to ensure it's not overridden by relayout listener
                setTimeout(() => {{
                    if (minY !== Infinity && maxY !== -Infinity) {{
                        Plotly.relayout(snakeChart, {{
                            'yaxis.range': [minY, maxY],
                            'yaxis.autorange': false
                        }});
                    }}
                }}, 50);
                
                // Also zoom RSI and Equity charts to match
                const rsiChart = document.getElementById('rsi-chart');
                const equityChart = document.getElementById('equity-chart');
                if (rsiChart) Plotly.relayout(rsiChart, {{'xaxis.range': [startDate, endDate]}});
                if (equityChart) Plotly.relayout(equityChart, {{'xaxis.range': [startDate, endDate], 'yaxis.autorange': true}});
                
                console.log(`🎯 Trader Mode ON: Showing ${{startDate}} to ${{endDate}}, Y-range: ${{minY.toFixed(2)}} - ${{maxY.toFixed(2)}}`);
            }} else {{
                // Default view: From signal_start_date to confidence box + padding (same logic as trader mode)
                const signalStartDate = data.signal_start_date;
                const lastDataDate = data.dates[data.dates.length - 1];
                
                // Calculate end date: Include forecast band + confidence box + padding for blank space
                let defaultEndDate;
                
                if (data.live_forecast && Object.keys(data.live_forecast.forecasts).length > 0) {{
                    // Calculate forecast band end date
                    const currentEnabledHorizons = enabledHorizons.length > 0 ? enabledHorizons : (data.horizons || []);
                    const horizons = Object.keys(data.live_forecast.forecasts)
                        .map(k => parseInt(k.replace('D+', '')))
                        .filter(h => currentEnabledHorizons.includes(h))
                        .sort((a, b) => a - b);
                    
                    let maxHorizon = 0;
                    horizons.forEach(h => {{
                        const key = 'D+' + h;
                        const pred = data.live_forecast.forecasts[key];
                        if (pred !== null && pred !== undefined && h > maxHorizon) {{
                            maxHorizon = h;
                        }}
                    }});
                    
                    // Forecast band ends at: lastDataDate + maxHorizon
                    const maxForecastDateObj = new Date(lastDataDate);
                    maxForecastDateObj.setDate(maxForecastDateObj.getDate() + maxHorizon);
                    
                    // Confidence box is positioned: maxForecastDate + padding (20% of forecast duration or min 10 days)
                    const forecastStartTime = new Date(lastDataDate).getTime();
                    const forecastEndTime = maxForecastDateObj.getTime();
                    const forecastDuration = forecastEndTime - forecastStartTime;
                    const confidencePaddingDays = Math.max(forecastDuration * 0.20 / (1000 * 60 * 60 * 24), 10);
                    
                    const confidenceBoxDateObj = new Date(maxForecastDateObj);
                    confidenceBoxDateObj.setDate(confidenceBoxDateObj.getDate() + confidencePaddingDays);
                    
                    // Add additional padding after confidence box for blank space (at least 20 days)
                    const blankSpacePadding = 20;
                    confidenceBoxDateObj.setDate(confidenceBoxDateObj.getDate() + blankSpacePadding);
                    
                    defaultEndDate = confidenceBoxDateObj.toISOString().split('T')[0];
                }} else {{
                    // Fallback: if no forecast data, use simple padding
                    const lastHist = new Date(lastDataDate);
                    lastHist.setDate(lastHist.getDate() + 30);
                    defaultEndDate = lastHist.toISOString().split('T')[0];
                }}
                
                // Reset snake chart with Y-axis autorange to scale properly
                Plotly.relayout(snakeChart, {{
                    'xaxis.range': [signalStartDate, defaultEndDate],
                    'xaxis.autorange': false,
                    'yaxis.autorange': true
                }});
                
                // Also reset RSI and Equity charts to match
                const rsiChart = document.getElementById('rsi-chart');
                const equityChart = document.getElementById('equity-chart');
                if (rsiChart) Plotly.relayout(rsiChart, {{'xaxis.range': [signalStartDate, defaultEndDate]}});
                if (equityChart) Plotly.relayout(equityChart, {{'xaxis.range': [signalStartDate, defaultEndDate], 'yaxis.autorange': true}});
                
                console.log(`📊 Trader Mode OFF: Default view from ${{signalStartDate}} to ${{defaultEndDate}}`);
            }}
        }}
        
        // Calculate ATR (Average True Range) - Professional volatility measure
        // Since we only have close prices, we estimate TR from daily price changes
        function calculateATR(prices, period = 14) {{
            if (prices.length < period + 1) return null;
            
            // Calculate True Range approximation (using close-to-close)
            const trueRanges = [];
            for (let i = 1; i < prices.length; i++) {{
                // TR ≈ absolute daily change (simplified without high/low)
                // We multiply by 1.5 to approximate what H-L would be vs C-C
                const dailyChange = Math.abs(prices[i] - prices[i-1]);
                const estimatedTR = dailyChange * 1.5;  // Approximate H-L from C-C
                trueRanges.push(estimatedTR);
            }}
            
            // Calculate ATR as EMA of True Ranges
            const atrValues = [];
            atrValues.push(null);  // First value has no TR
            
            // Initial ATR = SMA of first 'period' TRs
            let atr = 0;
            for (let i = 0; i < period && i < trueRanges.length; i++) {{
                atr += trueRanges[i];
                atrValues.push(null);
            }}
            atr = atr / period;
            atrValues.push(atr);
            
            // Smoothed ATR using Wilder's method
            for (let i = period; i < trueRanges.length; i++) {{
                atr = ((atr * (period - 1)) + trueRanges[i]) / period;
                atrValues.push(atr);
            }}
            
            return atrValues;
        }}
        
        // Get ATR as percentage of price (more useful for comparison)
        function getATRPercent(prices, period = 14) {{
            const atrValues = calculateATR(prices, period);
            if (!atrValues) return null;
            
            const atrPercents = [];
            for (let i = 0; i < prices.length; i++) {{
                if (atrValues[i] !== null && prices[i] > 0) {{
                    atrPercents.push((atrValues[i] / prices[i]) * 100);
                }} else {{
                    atrPercents.push(null);
                }}
            }}
            return atrPercents;
        }}
        
        // Calculate EMA (Exponential Moving Average)
        function calculateEMA(prices, period) {{
            const ema = [];
            const multiplier = 2 / (period + 1);
            
            // Start with SMA for first 'period' values
            let sum = 0;
            for (let i = 0; i < period && i < prices.length; i++) {{
                sum += prices[i];
                ema.push(null);  // No EMA for first period-1 values
            }}
            
            if (prices.length >= period) {{
                ema[period - 1] = sum / period;  // First EMA is SMA
                
                for (let i = period; i < prices.length; i++) {{
                    const newEMA = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
                    ema.push(newEMA);
                }}
            }}
            
            return ema;
        }}
        
        // Calculate Ichimoku Cloud components
        function calculateIchimoku(prices, dates) {{
            const tenkanPeriod = 9;
            const kijunPeriod = 26;
            const senkouBPeriod = 52;
            const displacement = 26;
            
            // Helper to get highest/lowest in period
            function highLow(arr, start, period) {{
                let high = -Infinity, low = Infinity;
                for (let i = start; i >= 0 && i > start - period; i--) {{
                    if (arr[i] > high) high = arr[i];
                    if (arr[i] < low) low = arr[i];
                }}
                return {{ high, low }};
            }}
            
            const tenkan = [];  // Conversion Line
            const kijun = [];   // Base Line
            const senkouA = []; // Leading Span A
            const senkouB = []; // Leading Span B
            
            for (let i = 0; i < prices.length; i++) {{
                // Tenkan-sen (9-period)
                if (i >= tenkanPeriod - 1) {{
                    const hl = highLow(prices, i, tenkanPeriod);
                    tenkan.push((hl.high + hl.low) / 2);
                }} else {{
                    tenkan.push(null);
                }}
                
                // Kijun-sen (26-period)
                if (i >= kijunPeriod - 1) {{
                    const hl = highLow(prices, i, kijunPeriod);
                    kijun.push((hl.high + hl.low) / 2);
                }} else {{
                    kijun.push(null);
                }}
                
                // Senkou Span B (52-period)
                if (i >= senkouBPeriod - 1) {{
                    const hl = highLow(prices, i, senkouBPeriod);
                    senkouB.push((hl.high + hl.low) / 2);
                }} else {{
                    senkouB.push(null);
                }}
            }}
            
            // Senkou Span A = (Tenkan + Kijun) / 2
            for (let i = 0; i < prices.length; i++) {{
                if (tenkan[i] !== null && kijun[i] !== null) {{
                    senkouA.push((tenkan[i] + kijun[i]) / 2);
                }} else {{
                    senkouA.push(null);
                }}
            }}
            
            // Shift Senkou spans forward by displacement (26 periods)
            const futureDates = [...dates];
            const lastDate = new Date(dates[dates.length - 1]);
            for (let i = 1; i <= displacement; i++) {{
                const futureDate = new Date(lastDate);
                futureDate.setDate(futureDate.getDate() + i);
                futureDates.push(futureDate.toISOString().split('T')[0]);
            }}
            
            // Shift spans forward
            const shiftedSenkouA = Array(displacement).fill(null).concat(senkouA);
            const shiftedSenkouB = Array(displacement).fill(null).concat(senkouB);
            
            // Create cloudBullish array aligned with original dates (for signal filtering)
            // This tells us at each date whether the cloud being generated is bullish or bearish
            const cloudBullish = [];
            for (let i = 0; i < prices.length; i++) {{
                if (senkouA[i] !== null && senkouB[i] !== null) {{
                    cloudBullish.push(senkouA[i] > senkouB[i]);  // true = bullish, false = bearish
                }} else {{
                    cloudBullish.push(null);  // Not enough data yet
                }}
            }}
            
            return {{
                tenkan,
                kijun,
                senkouA: shiftedSenkouA,
                senkouB: shiftedSenkouB,
                cloudDates: futureDates,
                cloudBullish: cloudBullish  // For filtering snake signals
            }};
        }}
        
        // Build Ichimoku cloud polygons (separate bullish/bearish regions)
        function buildCloudPolygons(cloudDates, senkouA, senkouB) {{
            const polygons = [];
            let currentPolygon = null;
            let currentType = null; // 'bullish' or 'bearish'
            
            for (let i = 0; i < cloudDates.length; i++) {{
                const a = senkouA[i];
                const b = senkouB[i];
                
                if (a === null || b === null) {{
                    // Close current polygon if exists
                    if (currentPolygon && currentPolygon.x.length > 0) {{
                        polygons.push(currentPolygon);
                        currentPolygon = null;
                        currentType = null;
                    }}
                    continue;
                }}
                
                const type = a >= b ? 'bullish' : 'bearish';
                
                if (type !== currentType) {{
                    // Type changed, close previous and start new
                    if (currentPolygon && currentPolygon.x.length > 0) {{
                        polygons.push(currentPolygon);
                    }}
                    currentPolygon = {{ type: type, x: [], upper: [], lower: [] }};
                    currentType = type;
                }}
                
                currentPolygon.x.push(cloudDates[i]);
                if (type === 'bullish') {{
                    currentPolygon.upper.push(a);
                    currentPolygon.lower.push(b);
                }} else {{
                    currentPolygon.upper.push(b);
                    currentPolygon.lower.push(a);
                }}
            }}
            
            // Don't forget the last polygon
            if (currentPolygon && currentPolygon.x.length > 0) {{
                polygons.push(currentPolygon);
            }}
            
            return polygons;
        }}
        
        // getCurrentSignals and getCurrentStats are defined above (with horizon toggling support)
        
        // Calculate directional accuracy dynamically
        function calculateAccuracy(data, signals) {{
            let correct = 0;
            let total = 0;
            
            for (let i = 0; i < signals.length - 1; i++) {{
                const signal = signals[i];
                if (signal === 'NEUTRAL') continue;
                
                // Compare with next day's price
                const currentPrice = data.prices[i];
                const nextPrice = data.prices[i + 1];
                const actualDirection = nextPrice > currentPrice ? 'UP' : 'DOWN';
                const predictedDirection = signal === 'BULLISH' ? 'UP' : 'DOWN';
                
                total++;
                if (actualDirection === predictedDirection) {{
                    correct++;
                }}
            }}
            
            if (total === 0) return {{ accuracy: 0, edge: -50, correct: 0, total: 0 }};
            
            const accuracy = (correct / total) * 100;
            const edge = accuracy - 50;
            return {{ accuracy, edge, correct, total }};
        }}
        
        function switchAsset(assetName) {{
            currentAsset = assetName;
            
            const data = ASSET_DATA[currentAsset];
            const config = window.OPTIMAL_CONFIGS ? window.OPTIMAL_CONFIGS[currentAsset] : null;
            const allHorizons = data.horizons || [];
            
            // STEP 1: Calculate RAW metrics (all horizons)
            showingRawView = true;
            enabledHorizons = [...allHorizons];
            isOptimalConfigApplied = false;
            recalculateSignalsWithEnabledHorizons();
            updateStats();
            updateCharts();  // This will cache raw metrics
            
            // STEP 2: Get metrics from config (pre-calculated in pipeline - most reliable!)
            // Debug: Check if config exists
            console.log(`[SWITCH ASSET] ${{currentAsset}}: config exists=${{!!config}}, baseline_equity=${{config?.baseline_equity}}, optimized_equity=${{config?.optimized_equity}}`);
            
            const rawReturn = (config && config.baseline_equity !== undefined && config.baseline_equity !== null) 
                ? config.baseline_equity 
                : (cachedPerformanceMetrics.raw.totalReturn || 0);
            
            let optReturn = rawReturn;
            if (config && config.viable_horizons && config.viable_horizons.length > 0) {{
                const available = data.horizons || [];
                const optimalHorizons = config.viable_horizons.filter(h => available.includes(h));
                
                if (optimalHorizons.length > 0) {{
                    // Use pre-calculated return from config (calculated in pipeline)
                    optReturn = (config && config.optimized_equity !== undefined && config.optimized_equity !== null) 
                        ? config.optimized_equity 
                        : rawReturn;
                    
                    // Set up optimized view for display
                    showingRawView = false;
                    enabledHorizons = [...optimalHorizons];
                    recalculateSignalsWithEnabledHorizons();
                    updateCharts();  // This will cache optimized metrics (using config values)
                    isOptimalConfigApplied = true;
                    
                    console.log(`[COMPARE] ${{currentAsset}}: RAW=${{rawReturn}}%, OPTIMIZED=${{optReturn}}% (from config)`);
                }}
            }} else {{
                // No optimization config - use raw as optimized too
                cachedPerformanceMetrics.optimized = {{ ...cachedPerformanceMetrics.raw }};
                console.log(`[COMPARE] ${{currentAsset}}: No optimal config found`);
            }}
            
            // STEP 3: Show BETTER version by default (compare total returns)
            const optimizedIsBetter = optReturn > rawReturn;
            rawIsBestForAsset = !optimizedIsBetter;  // Track for display message
            showingRawView = rawIsBestForAsset;  // Show raw if it's better
            
            console.log(`[COMPARE RESULT] ${{currentAsset}}: optimizedIsBetter=${{optimizedIsBetter}}, rawReturn=${{rawReturn}}%, optReturn=${{optReturn}}%, showingRawView=${{showingRawView}}`);
            
            // Apply the better configuration
            if (showingRawView) {{
                // Raw is better - use all horizons
                enabledHorizons = [...allHorizons];
                isOptimalConfigApplied = false;
                recalculateSignalsWithEnabledHorizons();
                console.log(`${{currentAsset}}: RAW is better (${{rawReturn.toFixed(1)}}% vs ${{optReturn.toFixed(1)}}%) - no optimization needed`);
            }} else {{
                // Optimized is better - use optimal horizons
                if (config && config.viable_horizons) {{
                    enabledHorizons = config.viable_horizons.filter(h => allHorizons.includes(h));
                    recalculateSignalsWithEnabledHorizons();
                }}
                isOptimalConfigApplied = true;
                console.log(`${{currentAsset}}: OPTIMIZED is better (${{optReturn.toFixed(1)}}% vs ${{rawReturn.toFixed(1)}}%)`);
            }}
            
            // Final updates
            updateStats();
            updateCharts();
            updatePriceTargets();
            updatePerformanceBox();
            updateOptimalConfigDisplay();
            savePreferences();
        }}
        
        // ==================== HORIZON TOGGLE FUNCTIONS ====================
        
        // Toggle a single horizon on/off
        function toggleHorizon(horizon) {{
            const idx = enabledHorizons.indexOf(horizon);
            if (idx > -1) {{
                // Disable horizon (remove from array)
                enabledHorizons.splice(idx, 1);
            }} else {{
                // Enable horizon (add to array)
                enabledHorizons.push(horizon);
                enabledHorizons.sort((a, b) => a - b);
            }}
            
            // Manual toggle clears the optimal config flag
            isOptimalConfigApplied = false;
            showingRawView = true;  // Manual changes = raw view
            
            // Recalculate signals with enabled horizons only
            recalculateSignalsWithEnabledHorizons();
            
            // Update all displays
            updateStats();
            updateCharts();  // This also updates the health check
            updatePriceTargets();
            updateOptimizationHero();
            savePreferences();  // Save user preference
        }}
        
        // Reset all horizons to enabled
        function resetHorizons() {{
            const data = ASSET_DATA[currentAsset];
            enabledHorizons = [...(data.horizons || [])];
            isOptimalConfigApplied = false;
            showingRawView = true;  // Reset = raw view
            
            // Recalculate with all horizons
            recalculateSignalsWithEnabledHorizons();
            
            // Update all displays
            updateStats();
            updateCharts();  // This also updates the health check
            updatePriceTargets();
            updateOptimizationHero();
            savePreferences();  // Save user preference
        }}
        
        // Apply optimal configuration (only viable horizons >50% accuracy)
        function updateOptimalConfigDisplay() {{
            const config = window.OPTIMAL_CONFIGS ? window.OPTIMAL_CONFIGS[currentAsset] : null;
            const detailsEl = document.getElementById('optimalConfigDetails');
            const btnEl = document.getElementById('autoOptimizeBtn');
            const sectionEl = document.getElementById('optimalConfigSection');
            
            if (!detailsEl || !btnEl || !sectionEl) return;
            
            if (!config || config.viable_horizons.length === 0) {{
                detailsEl.textContent = 'No optimal config available for this asset';
                btnEl.style.display = 'none';
                sectionEl.style.background = 'rgba(255,82,82,0.1)';
                sectionEl.style.borderColor = 'rgba(255,82,82,0.3)';
                return;
            }}
            
            const horizonStr = config.viable_horizons.map(h => 'D+' + h).join(', ');
            const appliedText = isOptimalConfigApplied ? ' <span style="color: #00d68f;">✓ Applied</span>' : '';
            detailsEl.innerHTML = 'Best horizons: <strong style="color: #F5B700;">' + horizonStr + '</strong> • Avg: ' + config.avg_accuracy.toFixed(1) + '% • Score: ' + config.health_score + '/100' + appliedText;
            
            if (isOptimalConfigApplied) {{
                btnEl.innerHTML = '↩ Reset to Full';
                btnEl.style.background = 'linear-gradient(135deg, #00d68f, #00a86b)';
                btnEl.style.border = 'none';
                btnEl.style.color = 'white';
                btnEl.title = 'Click to restore all horizons';
                btnEl.classList.remove('auto-optimize-btn-flash');
            }} else {{
                btnEl.innerHTML = '⚡ Auto Optimize Equity';
                btnEl.style.background = 'linear-gradient(135deg, #00d4ff, #0099cc)';
                btnEl.style.border = 'none';
                btnEl.style.color = 'white';
                btnEl.title = 'Click to apply optimal configuration';
                btnEl.classList.add('auto-optimize-btn-flash');
            }}
            btnEl.style.display = 'block';
            btnEl.classList.remove('optimize-btn-loading');
            sectionEl.style.background = isOptimalConfigApplied ? 'rgba(0,212,255,0.15)' : 'rgba(0,212,255,0.05)';
            sectionEl.style.borderColor = 'rgba(0,212,255,0.3)';
        }}
        
        // ==================== PERFORMANCE BOX FUNCTIONS ====================
        let showingRawView = false;  // Track if user is viewing raw vs optimized
        
        // Store performance metrics for raw vs optimized comparison
        let cachedRawWinRate = null;
        let cachedOptimizedWinRate = null;
        let cachedPerformanceMetrics = {{
            raw: {{ totalReturn: 0, sharpe: 0, avgDA: 0, maxDD: 0 }},
            optimized: {{ totalReturn: 0, sharpe: 0, avgDA: 0, maxDD: 0 }}
        }};
        
        function updatePerformanceBox() {{
            const isOptimized = !showingRawView;
            const metrics = isOptimized ? cachedPerformanceMetrics.optimized : cachedPerformanceMetrics.raw;
            
            // Update badge
            const badge = document.getElementById('perfBadge');
            if (rawIsBestForAsset && showingRawView) {{
                badge.textContent = 'BEST';
                badge.className = 'performance-badge';  // Green for best
            }} else {{
                badge.textContent = isOptimized ? 'OPTIMIZED' : 'RAW';
                badge.className = isOptimized ? 'performance-badge' : 'performance-badge raw';
            }}
            
            // Update toggle buttons
            document.getElementById('perfOptBtn').className = isOptimized ? 'performance-toggle-btn active' : 'performance-toggle-btn';
            document.getElementById('perfRawBtn').className = !isOptimized ? 'performance-toggle-btn active' : 'performance-toggle-btn';
            
            // Update metrics
            const totalReturnEl = document.getElementById('perfTotalReturn');
            totalReturnEl.textContent = (metrics.totalReturn >= 0 ? '+' : '') + metrics.totalReturn.toFixed(1) + '%';
            totalReturnEl.className = metrics.totalReturn >= 0 ? 'perf-metric-value positive' : 'perf-metric-value negative';
            
            document.getElementById('perfSharpe').textContent = metrics.sharpe.toFixed(2);
            document.getElementById('perfDA').textContent = metrics.avgDA.toFixed(1) + '%';
            
            // Show/hide "raw is best" message
            const msgEl = document.getElementById('perfMessage');
            if (rawIsBestForAsset && showingRawView) {{
                msgEl.style.display = 'flex';
            }} else {{
                msgEl.style.display = 'none';
            }}
        }}
        
        function showOptimizedPerf() {{
            showingRawView = false;
            
            // Apply optimal config
            const data = ASSET_DATA[currentAsset];
            const config = window.OPTIMAL_CONFIGS ? window.OPTIMAL_CONFIGS[currentAsset] : null;
            
            if (config && config.viable_horizons && config.viable_horizons.length > 0) {{
                const available = data.horizons || [];
                const optimalHorizons = config.viable_horizons.filter(h => available.includes(h));
                if (optimalHorizons.length > 0) {{
                    enabledHorizons = [...optimalHorizons];
                    recalculateSignalsWithEnabledHorizons();
                }}
            }}
            
            updateCharts();
            updatePerformanceBox();
        }}
        
        function showRawPerf() {{
            showingRawView = true;
            
            // Apply all horizons
            const data = ASSET_DATA[currentAsset];
            enabledHorizons = [...(data.horizons || [])];
            recalculateSignalsWithEnabledHorizons();
            
            updateCharts();
            updatePerformanceBox();
        }}
        
        // Legacy function for compatibility
        function updateOptimizationHero() {{
            updatePerformanceBox();
        }}
        
        // Cache win rate function (called from updateCharts)
        function cacheCurrentWinRate(winRate) {{
            if (showingRawView) {{
                cachedRawWinRate = winRate;
            }} else {{
                cachedOptimizedWinRate = winRate;
            }}
        }}
        
        // Show confidence explanation modal
        function showConfidenceExplanation() {{
            document.getElementById('confidenceModal').style.display = 'flex';
        }}
        
        // ==================== QUANT DETAILS MODAL ====================
        // Stores calculated quant metrics for display
        let cachedQuantMetrics = null;
        
        function showQuantDetails() {{
            const modal = document.getElementById('quantModal');
            modal.style.display = 'flex';
            
            // Calculate and display metrics
            updateQuantMetrics();
        }}
        
        function closeQuantModal() {{
            document.getElementById('quantModal').style.display = 'none';
        }}
        
        // Close modal when clicking outside
        document.getElementById('quantModal').onclick = function(e) {{
            if (e.target === this) {{
                closeQuantModal();
            }}
        }};
        
        function updateQuantMetrics() {{
            const metrics = showingRawView ? cachedPerformanceMetrics.raw : cachedPerformanceMetrics.optimized;
            
            // Get additional metrics from the signal-following calculation
            const data = ASSET_DATA[currentAsset];
            const signals = getCurrentSignals(data);
            const sfMetrics = calculateSignalFollowingMetrics(data, signals);
            
            // Risk & Return Section
            const riskReturnHtml = `
                <div class="quant-metric-card">
                    <div class="quant-metric-value ${{metrics.maxDD <= -15 ? 'negative' : metrics.maxDD <= -10 ? 'warning' : 'neutral'}}">${{metrics.maxDD.toFixed(1)}}%</div>
                    <div class="quant-metric-label">Max Drawdown</div>
                    <div class="quant-metric-desc">Peak-to-trough decline</div>
                </div>
                <div class="quant-metric-card">
                    <div class="quant-metric-value ${{sfMetrics.sortinoRatio >= 2 ? 'positive' : sfMetrics.sortinoRatio >= 1 ? 'neutral' : 'negative'}}">${{sfMetrics.sortinoRatio.toFixed(2)}}</div>
                    <div class="quant-metric-label">Sortino Ratio</div>
                    <div class="quant-metric-desc">Downside risk-adjusted</div>
                </div>
                <div class="quant-metric-card">
                    <div class="quant-metric-value ${{sfMetrics.calmarRatio >= 2 ? 'positive' : sfMetrics.calmarRatio >= 1 ? 'neutral' : 'negative'}}">${{sfMetrics.calmarRatio.toFixed(2)}}</div>
                    <div class="quant-metric-label">Calmar Ratio</div>
                    <div class="quant-metric-desc">Return / Max DD</div>
                </div>
            `;
            document.getElementById('quantRiskReturn').innerHTML = riskReturnHtml;
            
            // Trade Statistics Section
            const tradeStatsHtml = `
                <div class="quant-metric-card">
                    <div class="quant-metric-value neutral">${{sfMetrics.totalTrades}}</div>
                    <div class="quant-metric-label">Total Trades</div>
                    <div class="quant-metric-desc">Completed positions</div>
                </div>
                <div class="quant-metric-card">
                    <div class="quant-metric-value ${{sfMetrics.winRate >= 55 ? 'positive' : sfMetrics.winRate >= 45 ? 'neutral' : 'negative'}}">${{sfMetrics.winRate.toFixed(1)}}%</div>
                    <div class="quant-metric-label">Win Rate</div>
                    <div class="quant-metric-desc">${{sfMetrics.wins}}W / ${{sfMetrics.losses}}L</div>
                </div>
                <div class="quant-metric-card">
                    <div class="quant-metric-value neutral">${{sfMetrics.avgHoldDays.toFixed(1)}}d</div>
                    <div class="quant-metric-label">Avg Hold Time</div>
                    <div class="quant-metric-desc">Days per trade</div>
                </div>
            `;
            document.getElementById('quantTradeStats').innerHTML = tradeStatsHtml;
            
            // Profitability Section
            const profitHtml = `
                <div class="quant-metric-card">
                    <div class="quant-metric-value ${{sfMetrics.profitFactor >= 1.5 ? 'positive' : sfMetrics.profitFactor >= 1 ? 'neutral' : 'negative'}}">${{sfMetrics.profitFactor >= 99 ? '∞' : sfMetrics.profitFactor.toFixed(2)}}x</div>
                    <div class="quant-metric-label">Profit Factor</div>
                    <div class="quant-metric-desc">Gross profit / loss</div>
                </div>
                <div class="quant-metric-card">
                    <div class="quant-metric-value ${{sfMetrics.expectancy >= 0.5 ? 'positive' : sfMetrics.expectancy >= 0 ? 'neutral' : 'negative'}}">${{sfMetrics.expectancy >= 0 ? '+' : ''}}${{sfMetrics.expectancy.toFixed(2)}}%</div>
                    <div class="quant-metric-label">Expectancy</div>
                    <div class="quant-metric-desc">Avg P&L per trade</div>
                </div>
                <div class="quant-metric-card">
                    <div class="quant-metric-value ${{sfMetrics.payoffRatio >= 1.5 ? 'positive' : sfMetrics.payoffRatio >= 1 ? 'neutral' : 'negative'}}">${{sfMetrics.payoffRatio >= 99 ? '∞' : sfMetrics.payoffRatio.toFixed(2)}}x</div>
                    <div class="quant-metric-label">Payoff Ratio</div>
                    <div class="quant-metric-desc">Avg win / avg loss</div>
                </div>
            `;
            document.getElementById('quantProfitability').innerHTML = profitHtml;
        }}
        
        // Calculate comprehensive signal-following metrics
        function calculateSignalFollowingMetrics(data, signals) {{
            const trades = [];
            let currentPosition = null;
            
            // Build trades array (same logic as equity curve)
            for (let i = 1; i < signals.length; i++) {{
                const prevSignal = signals[i-1];
                const currSignal = signals[i];
                
                if (prevSignal !== currSignal) {{
                    // Close existing position
                    if (currentPosition) {{
                        currentPosition.exitIdx = i;
                        currentPosition.exitPrice = data.prices[i];
                        currentPosition.pnl = currentPosition.signal === 'BULLISH' 
                            ? ((currentPosition.exitPrice - currentPosition.entryPrice) / currentPosition.entryPrice) * 100
                            : ((currentPosition.entryPrice - currentPosition.exitPrice) / currentPosition.entryPrice) * 100;
                        currentPosition.holdDays = i - currentPosition.entryIdx;
                        trades.push(currentPosition);
                        currentPosition = null;
                    }}
                    
                    // Open new position
                    if (currSignal === 'BULLISH' || currSignal === 'BEARISH') {{
                        currentPosition = {{
                            signal: currSignal,
                            entryIdx: i,
                            entryPrice: data.prices[i]
                        }};
                    }}
                }}
            }}
            
            // Calculate metrics
            const wins = trades.filter(t => t.pnl > 0).length;
            const losses = trades.filter(t => t.pnl < 0).length;
            const totalTrades = trades.length;
            const winRate = totalTrades > 0 ? (wins / totalTrades) * 100 : 50;
            
            const avgWin = wins > 0 ? trades.filter(t => t.pnl > 0).reduce((a, b) => a + b.pnl, 0) / wins : 0;
            const avgLoss = losses > 0 ? Math.abs(trades.filter(t => t.pnl < 0).reduce((a, b) => a + b.pnl, 0)) / losses : 0;
            
            const grossProfit = trades.filter(t => t.pnl > 0).reduce((a, b) => a + b.pnl, 0);
            const grossLoss = Math.abs(trades.filter(t => t.pnl < 0).reduce((a, b) => a + b.pnl, 0));
            const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? 999 : 1);
            
            const expectancy = totalTrades > 0 ? trades.reduce((a, b) => a + b.pnl, 0) / totalTrades : 0;
            const payoffRatio = avgLoss > 0 ? avgWin / avgLoss : (avgWin > 0 ? 999 : 0);
            
            const avgHoldDays = totalTrades > 0 ? trades.reduce((a, b) => a + b.holdDays, 0) / totalTrades : 0;
            
            // Sortino (simplified)
            const negativeTrades = trades.filter(t => t.pnl < 0).map(t => t.pnl);
            const downsideDeviation = negativeTrades.length > 0 
                ? Math.sqrt(negativeTrades.reduce((sum, r) => sum + Math.pow(r, 2), 0) / negativeTrades.length)
                : 0.01;
            const avgReturn = totalTrades > 0 ? trades.reduce((a, b) => a + b.pnl, 0) / totalTrades : 0;
            const sortinoRatio = downsideDeviation > 0 ? (avgReturn / downsideDeviation) * Math.sqrt(252 / (avgHoldDays || 5)) : 0;
            
            // Calmar Ratio (Total Return / Max Drawdown)
            const totalReturn = showingRawView ? cachedPerformanceMetrics.raw.totalReturn : cachedPerformanceMetrics.optimized.totalReturn;
            const maxDD = Math.abs(showingRawView ? cachedPerformanceMetrics.raw.maxDD : cachedPerformanceMetrics.optimized.maxDD);
            const calmarRatio = maxDD > 0 ? totalReturn / maxDD : (totalReturn > 0 ? 999 : 0);
            
            return {{
                totalTrades,
                wins,
                losses,
                winRate,
                avgWin,
                avgLoss,
                profitFactor,
                expectancy,
                payoffRatio,
                avgHoldDays,
                sortinoRatio,
                calmarRatio
            }};
        }}
        
        // Toggle between optimized and raw views (legacy support)
        function toggleOptimizationView() {{
            if (showingRawView) {{
                showOptimizedPerf();
            }} else {{
                showRawPerf();
            }}
        }}
        
        function scrollToHealth() {{
            const healthEl = document.getElementById('healthCheck');
            if (healthEl) {{
                healthEl.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                // Also expand it if collapsed
                if (!healthEl.classList.contains('expanded')) {{
                    healthEl.classList.add('expanded');
                }}
            }}
        }}
        
        function applyOptimalConfig() {{
            const btnEl = document.getElementById('autoOptimizeBtn');
            const config = window.OPTIMAL_CONFIGS ? window.OPTIMAL_CONFIGS[currentAsset] : null;
            const data = ASSET_DATA[currentAsset];
            
            // TOGGLE: If already applied, revert to full ensemble (instant - no delay)
            if (isOptimalConfigApplied) {{
                // Reset to ALL horizons
                enabledHorizons = [...(data.horizons || [])];
                isOptimalConfigApplied = false;
                showingRawView = true;  // Show raw view when reverting
                
                console.log(`Reverted to full ensemble for ${{currentAsset}}: D+${{enabledHorizons.join(', D+')}}`);
                
                // Recalculate signals with all horizons
                recalculateSignalsWithEnabledHorizons();
                
                // Update all displays
                updateStats();
                updateCharts();
                updatePriceTargets();
                updateOptimizationHero();
                updateOptimalConfigDisplay();
                savePreferences();  // Save user preference
                
                // Restore flash animation
                btnEl.classList.add('auto-optimize-btn-flash');
                return;
            }}
            
            // Apply optimal config
            if (!config || config.viable_horizons.length === 0) {{
                alert('No optimal configuration available for this asset.\\n\\nThis typically means the models need more training or the asset is too difficult to predict.');
                return;
            }}
            
            // Filter viable horizons to only those that exist for this asset
            const availableHorizons = data.horizons || [];
            const viableAndAvailable = config.viable_horizons.filter(h => availableHorizons.includes(h));
            
            if (viableAndAvailable.length === 0) {{
                alert('None of the optimal horizons are available for this asset.');
                return;
            }}
            
            // Show loading state with spinner
            btnEl.innerHTML = '<span class="optimize-spinner"></span>Analyzing Models...';
            btnEl.classList.add('optimize-btn-loading');
            btnEl.classList.remove('auto-optimize-btn-flash');
            
            // Simulate live analysis with delay (5-8 seconds)
            const delay = 5000 + Math.random() * 3000;
            
            setTimeout(() => {{
                // Set enabled horizons to only the optimal ones
                enabledHorizons = [...viableAndAvailable];
                isOptimalConfigApplied = true;
                showingRawView = false;  // Show optimized view
                
                console.log(`Applied optimal config for ${{currentAsset}}: D+${{enabledHorizons.join(', D+')}}`);
                console.log(`Expected accuracy: ${{config.avg_accuracy}}%, Health: ${{config.health_score}}/100`);
                
                // Recalculate signals with optimal horizons only
                recalculateSignalsWithEnabledHorizons();
                
                // Update all displays
                updateStats();
                updateCharts();  // This also updates the health check
                updatePriceTargets();
                updateOptimizationHero();
                updateOptimalConfigDisplay();
                savePreferences();  // Save user preference
                
                // Remove loading state and ensure no flash on "Reset to Full"
                btnEl.classList.remove('optimize-btn-loading');
                btnEl.classList.remove('auto-optimize-btn-flash');
            }}, delay);
        }}
        
        // Recalculate signals based on enabled horizons only
        function recalculateSignalsWithEnabledHorizons() {{
            const data = ASSET_DATA[currentAsset];
            if (!data.horizon_forecasts || enabledHorizons.length === 0) {{
                console.log(`[RECALC] No horizon_forecasts or no enabledHorizons for ${{currentAsset}}`);
                return;
            }}
            
            const threshold = data.threshold || 0.3;
            console.log(`[RECALC] ${{currentAsset}}: enabled=${{enabledHorizons.join(',')}}, threshold=${{threshold}}`);
            const newNetProbs = [];
            const newSignalsRaw = [];
            
            // For each date, recalculate net_prob using only enabled horizons
            for (let dateIdx = 0; dateIdx < data.dates.length; dateIdx++) {{
                const slopes = [];
                
                // Calculate slopes between all pairs of enabled horizons
                for (let i = 0; i < enabledHorizons.length; i++) {{
                    for (let j = i + 1; j < enabledHorizons.length; j++) {{
                        const h1 = enabledHorizons[i];
                        const h2 = enabledHorizons[j];
                        
                        const forecasts1 = data.horizon_forecasts[String(h1)];
                        const forecasts2 = data.horizon_forecasts[String(h2)];
                        
                        if (forecasts1 && forecasts2) {{
                            const val1 = forecasts1[dateIdx];
                            const val2 = forecasts2[dateIdx];
                            
                            if (val1 && val2 && !isNaN(val1) && !isNaN(val2)) {{
                                const drift = val2 - val1;
                                slopes.push(drift);
                            }}
                        }}
                    }}
                }}
                
                // Calculate net_prob
                let netProb = 0;
                if (slopes.length > 0) {{
                    const bullish = slopes.filter(s => s > 0).length;
                    const bearish = slopes.filter(s => s < 0).length;
                    netProb = (bullish - bearish) / slopes.length;
                }}
                
                newNetProbs.push(netProb);
                
                // Determine signal
                let signal = 'NEUTRAL';
                if (netProb > threshold) signal = 'BULLISH';
                else if (netProb < -threshold) signal = 'BEARISH';
                newSignalsRaw.push(signal);
            }}
            
            // Update the data with new calculations (store in recalculated_ prefix)
            data.recalculated_net_prob = newNetProbs;
            data.recalculated_signals_raw = newSignalsRaw;
            
            // Apply RSI filter if enabled
            if (rsiFilterEnabled) {{
                const filteredSignals = newSignalsRaw.map((signal, i) => {{
                    const rsi = data.rsi[i];
                    if (signal === 'BULLISH' && rsi > data.rsi_overbought) return 'NEUTRAL';
                    if (signal === 'BEARISH' && rsi < data.rsi_oversold) return 'NEUTRAL';
                    return signal;
                }});
                data.recalculated_signals_filtered = filteredSignals;
            }} else {{
                data.recalculated_signals_filtered = [...newSignalsRaw];
            }}
            
            // Update signal stats
            const signals = rsiFilterEnabled ? data.recalculated_signals_filtered : data.recalculated_signals_raw;
            const bullishCount = signals.filter(s => s === 'BULLISH').length;
            const bearishCount = signals.filter(s => s === 'BEARISH').length;
            const neutralCount = signals.filter(s => s === 'NEUTRAL').length;
            
            console.log(`[RECALC] ${{currentAsset}} signals: BULLISH=${{bullishCount}}, BEARISH=${{bearishCount}}, NEUTRAL=${{neutralCount}}, total=${{signals.length}}`);
            
            data.recalculated_stats = {{
                bullish: bullishCount,
                bearish: bearishCount,
                neutral: neutralCount,
                total_days: signals.length
            }};
        }}
        
        // Get current signals (considering recalculated if horizons are toggled)
        function getCurrentSignals(data) {{
            // ALWAYS prefer recalculated signals if they exist (from recalculateSignalsWithEnabledHorizons)
            // This ensures consistency between RAW and OPTIMIZED modes
            // The recalculation is called in switchAsset() for both modes
            if (data.recalculated_signals_raw && data.recalculated_signals_raw.length > 0) {{
                return rsiFilterEnabled ? (data.recalculated_signals_filtered || data.recalculated_signals_raw) : data.recalculated_signals_raw;
            }}
            
            // Fallback to original signals if no recalculation exists
            return rsiFilterEnabled ? data.signals_filtered : data.signals_raw;
        }}
        
        // Get confidence stats HTML for the health check panel
        function getConfidenceStatsHtml(data) {{
            if (!data.confidence_stats || !data.confidence_stats.by_strength) {{
                return '<div style="color: var(--text-secondary); font-size: 12px;">Run confidence analysis to see historical accuracy stats</div>';
            }}
            
            const tiers = [
                {{ key: 'strong', label: 'HIGH', emoji: '🟢', color: '#00ff88' }},
                {{ key: 'medium', label: 'MED', emoji: '🟡', color: '#ffaa00' }},
                {{ key: 'weak', label: 'LOW', emoji: '🔴', color: '#ff3366' }}
            ];
            
            let html = '';
            tiers.forEach(tier => {{
                const stats = data.confidence_stats.by_strength[tier.key];
                if (stats) {{
                    const accuracyColor = stats.accuracy >= 65 ? '#00ff88' : (stats.accuracy >= 50 ? '#ffaa00' : '#ff3366');
                    html += `
                        <div style="flex: 1; min-width: 120px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 6px; text-align: center;">
                            <div style="font-size: 18px; margin-bottom: 4px;">${{tier.emoji}}</div>
                            <div style="font-weight: 600; color: ${{tier.color}}; font-size: 12px;">${{tier.label}} CONF</div>
                            <div style="font-size: 24px; font-weight: 700; color: ${{accuracyColor}}; margin: 4px 0;">${{stats.accuracy.toFixed(1)}}%</div>
                            <div style="font-size: 10px; color: var(--text-secondary);">${{stats.total}} signals</div>
                        </div>
                    `;
                }}
            }});
            
            return html || '<div style="color: var(--text-secondary); font-size: 12px;">No confidence data available</div>';
        }}
        
        // Get fully filtered signals applying ALL active filters (RSI, EMA, Ichimoku)
        function getFullyFilteredSignals(data) {{
            const baseSignals = getCurrentSignals(data);
            const filteredSignals = [...baseSignals];  // Clone the array
            
            // Use global indicator state variables (rsiFilterEnabled, emaPeriod, showIchimoku)
            
            // Calculate EMA if needed
            let emaValues = null;
            if (emaPeriod > 0) {{
                emaValues = calculateEMA(data.prices, emaPeriod);
            }}
            
            // Calculate Ichimoku if needed
            let ichimokuFilter = null;
            if (showIchimoku) {{
                ichimokuFilter = calculateIchimoku(data.prices, data.dates);
            }}
            
            // Apply filters to each signal
            for (let i = 0; i < filteredSignals.length; i++) {{
                const signal = filteredSignals[i];
                if (signal === 'NEUTRAL') continue;
                
                const signalIsBullish = (signal === 'BULLISH');
                const signalIsBearish = (signal === 'BEARISH');
                
                // Apply RSI filter
                if (rsiFilterEnabled && data.rsi && data.rsi[i] !== undefined) {{
                    const rsiVal = data.rsi[i];
                    if (signalIsBullish && rsiVal > data.rsi_overbought) {{
                        filteredSignals[i] = 'NEUTRAL';
                        continue;
                    }} else if (signalIsBearish && rsiVal < data.rsi_oversold) {{
                        filteredSignals[i] = 'NEUTRAL';
                        continue;
                    }}
                }}
                
                // Apply EMA filter
                if (emaPeriod > 0 && emaValues && emaValues[i] !== null && !isNaN(emaValues[i])) {{
                    const priceAboveEMA = data.prices[i] > emaValues[i];
                    if (signalIsBullish && !priceAboveEMA) {{
                        filteredSignals[i] = 'NEUTRAL';
                        continue;
                    }} else if (signalIsBearish && priceAboveEMA) {{
                        filteredSignals[i] = 'NEUTRAL';
                        continue;
                    }}
                }}
                
                // Apply Ichimoku cloud filter
                if (showIchimoku && ichimokuFilter && ichimokuFilter.cloudBullish[i] !== null) {{
                    const cloudIsBullish = ichimokuFilter.cloudBullish[i];
                    if (signalIsBullish && !cloudIsBullish) {{
                        filteredSignals[i] = 'NEUTRAL';
                        continue;
                    }} else if (signalIsBearish && cloudIsBullish) {{
                        filteredSignals[i] = 'NEUTRAL';
                        continue;
                    }}
                }}
            }}
            
            return filteredSignals;
        }}
        
        // Get current stats (considering recalculated if horizons are toggled)
        function getCurrentStats(data) {{
            const allHorizonsEnabled = enabledHorizons.length === data.horizons.length;
            
            if (!allHorizonsEnabled && data.recalculated_stats) {{
                return data.recalculated_stats;
            }}
            
            return rsiFilterEnabled ? data.stats_filtered : data.stats_raw;
        }}
        
        function updateStats() {{
            const data = ASSET_DATA[currentAsset];
            // Use FULLY filtered signals (RSI + EMA + Ichimoku) for dynamic accuracy
            const signals = getFullyFilteredSignals(data);
            
            // Calculate accuracy dynamically based on current signals
            const accMetrics = calculateAccuracy(data, signals);
            
            // Update project ID badge next to dropdown
            document.getElementById('projectIdBadge').textContent = `#${{data.project_id}}`;
            
            // Update live signal confidence panel with filtered signals
            updateLiveSignalPanel(data, signals);
        }}
        
        // Calculate dynamic accuracy based on current filters and horizons
        function calculateDynamicAccuracy(data, signals) {{
            const evalWindow = 15;  // Same as Trade History
            let totalTrades = 0;
            let t1Hits = 0;
            let skippedNeutral = 0;  // Debug counter
            
            // FIX: Bug #4 - Use ATR-based T1 target instead of hardcoded 0.5%
            // 0.5% is almost guaranteed for BTC but strict for low-vol assets
            const atrPercents = getATRPercent(data.prices, 14);
            
            // Loop through last ~40 possible trade entries to get ~20 valid trades
            for (let i = data.dates.length - 1; i >= evalWindow && totalTrades < 20; i--) {{
                const entryIdx = i - evalWindow;
                if (entryIdx < 0) continue;
                
                const signal = signals[entryIdx];
                if (!signal || signal === 'NEUTRAL') {{
                    skippedNeutral++;
                    continue;
                }}
                
                const entryPrice = data.prices[entryIdx];
                if (!entryPrice || isNaN(entryPrice)) continue;
                
                // FIX: T1 target scaled by ATR (half of daily ATR as target)
                // This makes accuracy comparable across assets with different volatility
                let atrPct = atrPercents && atrPercents[entryIdx] ? atrPercents[entryIdx] : 0.5;
                atrPct = Math.max(0.1, Math.min(atrPct, 5.0));  // Clamp to reasonable range
                const t1Multiplier = 1 + (atrPct / 200);  // Half of 1-day ATR as target
                
                const t1Target = signal === 'BULLISH' 
                    ? entryPrice * t1Multiplier 
                    : entryPrice * (2 - t1Multiplier);
                
                let hitT1 = false;
                for (let j = entryIdx + 1; j <= Math.min(entryIdx + evalWindow, data.prices.length - 1); j++) {{
                    const price = data.prices[j];
                    if (!price || isNaN(price)) continue;
                    
                    if (signal === 'BULLISH' && price >= t1Target) {{
                        hitT1 = true;
                        break;
                    }} else if (signal === 'BEARISH' && price <= t1Target) {{
                        hitT1 = true;
                        break;
                    }}
                }}
                
                totalTrades++;
                if (hitT1) t1Hits++;
            }}
            
            console.log('Dynamic Accuracy Debug:', {{
                totalTrades,
                t1Hits,
                skippedNeutral,
                emaPeriod,
                showIchimoku,
                rsiFilterEnabled,
                accuracy: totalTrades > 0 ? (t1Hits / totalTrades * 100).toFixed(1) + '%' : 'N/A'
            }});
            
            return {{
                accuracy: totalTrades > 0 ? (t1Hits / totalTrades * 100) : null,
                trades: totalTrades,
                hits: t1Hits
            }};
        }}
        
        function updateLiveSignalPanel(data, signals) {{
            // Get the CURRENT (latest) signal
            const lastPriceIdx = data.prices.length - 1;
            const currentSignalIdx = lastPriceIdx;
            const currentSignal = signals[currentSignalIdx] || 'NEUTRAL';
            
            // Get net_prob - the consensus strength among horizons
            let currentNetProb = data.net_prob[currentSignalIdx] || 0;
            const strength = Math.abs(currentNetProb) || 0;
            
            // ==================== EXPECTANCY-BASED CONFIDENCE CALCULATION ====================
            // Calculate Signal-Following metrics including Expectancy
            const sfResult = calculateSignalFollowingWinRate(data, signals);
            const baseWinRate = sfResult.winRate;
            const totalTrades = sfResult.totalTrades;
            const expectancy = sfResult.expectancy;
            const profitFactor = sfResult.profitFactor;
            const avgWin = sfResult.avgWin;
            const avgLoss = sfResult.avgLoss;
            
            // Convert expectancy to a confidence score (0-100)
            // Expectancy is expected % return per trade
            // Scale: expectancy of 1% = 50, 2% = 70, 3%+ = 85+
            let expectancyScore;
            if (expectancy <= 0) {{
                // Negative expectancy = losing system
                expectancyScore = Math.max(20, 40 + (expectancy * 10));
            }} else if (expectancy < 1) {{
                // 0-1% expectancy = marginal (40-55)
                expectancyScore = 40 + (expectancy * 15);
            }} else if (expectancy < 2) {{
                // 1-2% expectancy = good (55-70)
                expectancyScore = 55 + ((expectancy - 1) * 15);
            }} else if (expectancy < 3) {{
                // 2-3% expectancy = very good (70-85)
                expectancyScore = 70 + ((expectancy - 2) * 15);
            }} else {{
                // 3%+ expectancy = excellent (85-95)
                expectancyScore = Math.min(95, 85 + ((expectancy - 3) * 3));
            }}
            
            // Calculate adjustments (smaller impact since expectancy already captures quality)
            let adjustments = [];
            let totalAdjustment = 0;
            
            // 1. Signal Strength Bonus (strong consensus = higher confidence)
            if (strength >= 0.7) {{
                adjustments.push({{ label: 'Strong Signal', value: +3, color: '#00ff88' }});
                totalAdjustment += 3;
            }} else if (strength >= 0.5) {{
                adjustments.push({{ label: 'Good Signal', value: +2, color: '#00ff88' }});
                totalAdjustment += 2;
            }} else if (strength < 0.2) {{
                adjustments.push({{ label: 'Weak Signal', value: -3, color: '#ff3366' }});
                totalAdjustment -= 3;
            }}
            
            // 2. Optimization Bonus
            if (isOptimalConfigApplied) {{
                adjustments.push({{ label: 'Optimized', value: +3, color: '#00ff88' }});
                totalAdjustment += 3;
            }}
            
            // 3. RSI Alignment (if enabled)
            if (rsiFilterEnabled && data.rsi && data.rsi[currentSignalIdx]) {{
                const rsiVal = data.rsi[currentSignalIdx];
                const rsiAligned = (currentSignal === 'BULLISH' && rsiVal < 70) || 
                                   (currentSignal === 'BEARISH' && rsiVal > 30);
                if (rsiAligned) {{
                    adjustments.push({{ label: 'RSI ✓', value: +2, color: '#00ff88' }});
                    totalAdjustment += 2;
                }} else {{
                    adjustments.push({{ label: 'RSI ✗', value: -2, color: '#ff3366' }});
                    totalAdjustment -= 2;
                }}
            }}
            
            // 4. EMA Alignment (if enabled)
            if (emaPeriod > 0) {{
                const emaValues = calculateEMA(data.prices, emaPeriod);
                if (emaValues && emaValues[currentSignalIdx]) {{
                    const priceAboveEMA = data.prices[currentSignalIdx] > emaValues[currentSignalIdx];
                    const emaAligned = (currentSignal === 'BULLISH' && priceAboveEMA) ||
                                       (currentSignal === 'BEARISH' && !priceAboveEMA);
                    if (emaAligned) {{
                        adjustments.push({{ label: 'EMA ✓', value: +2, color: '#00ff88' }});
                        totalAdjustment += 2;
                    }} else {{
                        adjustments.push({{ label: 'EMA ✗', value: -2, color: '#ff3366' }});
                        totalAdjustment -= 2;
                    }}
                }}
            }}
            
            // 5. Ichimoku Alignment (if enabled)
            if (showIchimoku) {{
                const ichimokuData = calculateIchimoku(data.prices, data.dates);
                if (ichimokuData && ichimokuData.cloudBullish && ichimokuData.cloudBullish[currentSignalIdx] !== null) {{
                    const cloudBullish = ichimokuData.cloudBullish[currentSignalIdx];
                    const ichimokuAligned = (currentSignal === 'BULLISH' && cloudBullish) ||
                                            (currentSignal === 'BEARISH' && !cloudBullish);
                    if (ichimokuAligned) {{
                        adjustments.push({{ label: 'Ichimoku ✓', value: +2, color: '#00ff88' }});
                        totalAdjustment += 2;
                    }} else {{
                        adjustments.push({{ label: 'Ichimoku ✗', value: -2, color: '#ff3366' }});
                        totalAdjustment -= 2;
                    }}
                }}
            }}
            
            // Calculate final confidence (capped at 95%)
            const finalConfidence = Math.min(95, Math.max(20, expectancyScore + totalAdjustment));
            
            // Determine confidence tier based on final score
            let confTier, confTierClass;
            if (finalConfidence >= 70) {{
                confTier = 'HIGH CONFIDENCE';
                confTierClass = 'high';
            }} else if (finalConfidence >= 50) {{
                confTier = 'MEDIUM CONFIDENCE';
                confTierClass = 'medium';
            }} else {{
                confTier = 'LOW CONFIDENCE';
                confTierClass = 'low';
            }}
            
            // Calculate horizon agreement
            let horizonsAgreeing = 0;
            let totalHorizons = enabledHorizons.length;
            
            if (data.horizon_forecasts && enabledHorizons.length > 0) {{
                let bullishCount = 0;
                let bearishCount = 0;
                const currentPrice = data.prices[currentSignalIdx];
                
                enabledHorizons.forEach(h => {{
                    const hKey = String(h);
                    if (data.horizon_forecasts[hKey] && data.horizon_forecasts[hKey][currentSignalIdx] !== undefined) {{
                        const forecast = data.horizon_forecasts[hKey][currentSignalIdx];
                        if (forecast > currentPrice) bullishCount++;
                        else if (forecast < currentPrice) bearishCount++;
                    }}
                }});
                
                if (currentSignal === 'BULLISH') {{
                    horizonsAgreeing = bullishCount;
                }} else if (currentSignal === 'BEARISH') {{
                    horizonsAgreeing = bearishCount;
                }} else {{
                    horizonsAgreeing = Math.max(bullishCount, bearishCount);
                }}
            }}
            
            // ==================== UPDATE THE CONFIDENCE PANEL ====================
            const panel = document.getElementById('liveSignalPanel');
            const signalClass = currentSignal === 'BULLISH' ? 'bullish' : (currentSignal === 'BEARISH' ? 'bearish' : 'neutral');
            
            panel.className = 'live-signal-panel ' + signalClass;
            
            // Update confidence tier badge
            const tierBadge = document.getElementById('tierBadge');
            tierBadge.className = 'tier-badge ' + confTierClass;
            tierBadge.textContent = confTier;
            
            // Update confidence display with detailed breakdown
            const accuracyEl = document.getElementById('confidenceAccuracy');
            const confColor = finalConfidence >= 70 ? '#00ff88' : (finalConfidence >= 55 ? '#ffaa00' : '#ff3366');
            
            // Build adjustment breakdown HTML
            let adjustmentHtml = '';
            if (adjustments.length > 0) {{
                adjustmentHtml = `<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;">`;
                adjustments.forEach(adj => {{
                    const sign = adj.value >= 0 ? '+' : '';
                    adjustmentHtml += `<span style="font-size: 10px; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.05); color: ${{adj.color}};">${{adj.label}} ${{sign}}${{adj.value}}%</span>`;
                }});
                adjustmentHtml += `</div>`;
            }}
            
            // Format expectancy display
            const expectancySign = expectancy >= 0 ? '+' : '';
            const expectancyColor = expectancy >= 1 ? '#00ff88' : (expectancy >= 0 ? '#ffaa00' : '#ff3366');
            const pfColor = profitFactor >= 2 ? '#00ff88' : (profitFactor >= 1.5 ? '#ffaa00' : '#ff3366');
            
            // Use the new Live Signal Confidence calculation
            const liveConf = lastLiveConfidence || {{ confidence: 50, tier: 'MEDIUM', factors: {{}} }};
            const liveConfColor = liveConf.confidence >= 75 ? '#00ff88' : (liveConf.confidence >= 55 ? '#ffaa00' : '#ff6666');
            
            accuracyEl.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 4px;">
                    <div style="display: flex; align-items: baseline; gap: 8px;">
                        <span class="accuracy-value" style="color: ${{liveConfColor}}; font-size: 42px;">${{liveConf.confidence.toFixed(0)}}%</span>
                    </div>
                    <span style="font-size: 11px; color: var(--text-secondary);">Live Signal Confidence</span>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; font-size: 10px;">
                        <div style="background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 4px;">
                            <span style="color: var(--text-secondary);">Agreement:</span>
                            <span style="color: var(--text-primary); font-weight: 600;">${{liveConf.factors.agreement ? liveConf.factors.agreement.value : '-'}}</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 4px;">
                            <span style="color: var(--text-secondary);">Tightness:</span>
                            <span style="color: var(--text-primary); font-weight: 600;">${{liveConf.factors.tightness ? liveConf.factors.tightness.score.toFixed(0) : '-'}}</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 4px;">
                            <span style="color: var(--text-secondary);">Strength:</span>
                            <span style="color: var(--text-primary); font-weight: 600;">${{liveConf.factors.strength ? liveConf.factors.strength.value : '-'}}</span>
                        </div>
                    </div>
                    <div style="font-size: 9px; color: var(--text-secondary); margin-top: 4px;">
                        ${{totalTrades}} trades analyzed
                    </div>
                </div>
            `;
            
            // Update details
            const consensusPct = Math.abs(currentNetProb * 100).toFixed(1);
            document.getElementById('signalStrength').textContent = consensusPct + '% consensus';
            document.getElementById('signalStrength').style.color = currentSignal === 'BULLISH' ? '#00ff88' : (currentSignal === 'BEARISH' ? '#ff3366' : '#888');
            document.getElementById('horizonAgreement').textContent = horizonsAgreeing + '/' + totalHorizons;
            document.getElementById('sampleSize').textContent = totalTrades + ' trades analyzed';
        }}
        
        // Calculate Signal-Following Win Rate for confidence
        function calculateSignalFollowingWinRate(data, signals) {{
            // Build filtered signals with all active filters
            const filteredSignals = [];
            const signalStartDateObj = new Date(data.signal_start_date);
            
            // Pre-calculate indicators
            let emaValues = emaPeriod > 0 ? calculateEMA(data.prices, emaPeriod) : null;
            let ichimokuFilter = showIchimoku ? calculateIchimoku(data.prices, data.dates) : null;
            
            for (let i = 0; i < data.dates.length; i++) {{
                let signal = signals[i];
                const currentDate = new Date(data.dates[i]);
                
                if (currentDate >= signalStartDateObj) {{
                    // Apply EMA filter
                    if (emaPeriod > 0 && emaValues && emaValues[i] !== null) {{
                        const priceAboveEMA = data.prices[i] > emaValues[i];
                        if (signal === 'BULLISH' && !priceAboveEMA) signal = 'NEUTRAL';
                        else if (signal === 'BEARISH' && priceAboveEMA) signal = 'NEUTRAL';
                    }}
                    // Apply Ichimoku filter
                    if (showIchimoku && ichimokuFilter && ichimokuFilter.cloudBullish[i] !== null) {{
                        const cloudBullish = ichimokuFilter.cloudBullish[i];
                        if (signal === 'BULLISH' && !cloudBullish) signal = 'NEUTRAL';
                        else if (signal === 'BEARISH' && cloudBullish) signal = 'NEUTRAL';
                    }}
                }}
                filteredSignals.push(signal);
            }}
            
            // Signal-Following trade simulation
            const trades = [];
            let currentPosition = null;
            let entryIdx = null;
            let entryPrice = null;
            let entrySignal = null;
            
            for (let i = 0; i < data.dates.length; i++) {{
                const signal = filteredSignals[i];
                const price = data.prices[i];
                
                if (currentPosition === null) {{
                    if (signal === 'BULLISH' || signal === 'BEARISH') {{
                        currentPosition = signal;
                        entryIdx = i;
                        entryPrice = price;
                        entrySignal = signal;
                    }}
                }} else {{
                    if (signal !== currentPosition) {{
                        let pnl;
                        if (entrySignal === 'BULLISH') {{
                            pnl = ((price - entryPrice) / entryPrice) * 100;
                        }} else {{
                            pnl = ((entryPrice - price) / entryPrice) * 100;
                        }}
                        trades.push({{ pnl: pnl }});
                        
                        if (signal === 'BULLISH' || signal === 'BEARISH') {{
                            currentPosition = signal;
                            entryIdx = i;
                            entryPrice = price;
                            entrySignal = signal;
                        }} else {{
                            currentPosition = null;
                        }}
                    }}
                }}
            }}
            
            // Calculate metrics
            const wins = trades.filter(t => t.pnl > 0).length;
            const losses = trades.filter(t => t.pnl < 0).length;
            const totalTrades = trades.length;
            const winRate = totalTrades > 0 ? (wins / totalTrades) * 100 : 50;
            
            // Calculate average win and average loss
            const winningTrades = trades.filter(t => t.pnl > 0);
            const losingTrades = trades.filter(t => t.pnl < 0);
            const avgWin = winningTrades.length > 0 ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length : 0;
            const avgLoss = losingTrades.length > 0 ? Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length) : 0;
            
            // Calculate Profit Factor
            const grossProfit = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
            const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0));
            const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? 10 : 1);
            
            // Calculate Expectancy (expected return per trade as %)
            // Expectancy = (WinRate * AvgWin) - (LossRate * AvgLoss)
            const winRateDecimal = winRate / 100;
            const lossRateDecimal = 1 - winRateDecimal;
            const expectancy = (winRateDecimal * avgWin) - (lossRateDecimal * avgLoss);
            
            return {{ 
                winRate: winRate, 
                totalTrades: totalTrades, 
                wins: wins,
                losses: losses,
                avgWin: avgWin,
                avgLoss: avgLoss,
                profitFactor: profitFactor,
                expectancy: expectancy
            }};
        }}
        
        // ==================== HISTORICAL DA BY SIGNAL STRENGTH ====================
        // Calculate directional accuracy for signals at similar strength levels
        function calculateDAAtStrength(data, signals, currentStrength) {{
            // Define strength buckets: 0-30%, 30-50%, 50-70%, 70-85%, 85-100%
            const buckets = [
                {{ min: 0.00, max: 0.30, label: '0-30%' }},
                {{ min: 0.30, max: 0.50, label: '30-50%' }},
                {{ min: 0.50, max: 0.70, label: '50-70%' }},
                {{ min: 0.70, max: 0.85, label: '70-85%' }},
                {{ min: 0.85, max: 1.00, label: '85-100%' }}
            ];
            
            // Find which bucket the current strength falls into
            let currentBucket = buckets[2];  // Default to middle
            for (const bucket of buckets) {{
                if (currentStrength >= bucket.min && currentStrength < bucket.max) {{
                    currentBucket = bucket;
                    break;
                }}
            }}
            if (currentStrength >= 0.85) currentBucket = buckets[4];  // Handle edge case
            
            // Calculate DA for signals in this bucket
            // Look at historical signals where |net_prob| was in this range
            // Check if the direction was correct the next day
            let correct = 0;
            let total = 0;
            
            const signalStartDateObj = new Date(data.signal_start_date);
            
            for (let i = 0; i < data.dates.length - 1; i++) {{  // -1 because we need next day
                const currentDate = new Date(data.dates[i]);
                if (currentDate < signalStartDateObj) continue;
                
                const signal = signals[i];
                if (signal === 'NEUTRAL') continue;
                
                const strength = Math.abs(data.net_prob[i] || 0);
                
                // Check if this signal is in the same bucket as current
                // Use <= for max on the last bucket (85-100%)
                const inBucket = currentBucket.max >= 1.0 ? 
                    (strength >= currentBucket.min && strength <= currentBucket.max) :
                    (strength >= currentBucket.min && strength < currentBucket.max);
                    
                if (inBucket) {{
                    // Check if prediction was correct
                    const predictedUp = signal === 'BULLISH';
                    const actualUp = data.prices[i + 1] > data.prices[i];
                    
                    if (predictedUp === actualUp) correct++;
                    total++;
                }}
            }}
            
            // Calculate DA for this bucket
            const da = total > 0 ? (correct / total) * 100 : 50;  // Default to 50% if no data
            
            return {{
                da: da,
                bucket: currentBucket.label,
                correct: correct,
                total: total
            }};
        }}
        
        // ==================== LIVE SIGNAL CONFIDENCE CALCULATION ====================
        // Calculates confidence for TODAY's live forecast based on 5 factors
        function calculateLiveSignalConfidence(data, signals) {{
            const config = window.OPTIMAL_CONFIGS ? window.OPTIMAL_CONFIGS[currentAsset] : null;
            const lastIdx = data.prices.length - 1;
            const currentPrice = data.prices[lastIdx];
            
            // ===== CALCULATE SIGNAL FROM LIVE FORECASTS (TODAY'S FORECAST, NOT YESTERDAY'S) =====
            let currentSignal = 'NEUTRAL';
            let currentNetProb = 0;
            const threshold = config ? (config.threshold || 0.1) : 0.1;
            
            // Calculate signal from live_forecast.forecasts using pairwise slopes method
            if (data.live_forecast && data.live_forecast.forecasts && Object.keys(data.live_forecast.forecasts).length > 0) {{
                const liveForecastValues = [];
                const liveHorizons = [];
                
                // Get enabled horizons and their live forecast values
                enabledHorizons.forEach(h => {{
                    const key = 'D+' + h;
                    const pred = data.live_forecast.forecasts[key];
                    if (pred !== null && pred !== undefined) {{
                        liveForecastValues.push(pred);
                        liveHorizons.push(h);
                    }}
                }});
                
                // Calculate pairwise slopes (same method as historical signals)
                const slopes = [];
                for (let i = 0; i < liveHorizons.length; i++) {{
                    for (let j = i + 1; j < liveHorizons.length; j++) {{
                        const drift = liveForecastValues[j] - liveForecastValues[i];
                        slopes.push(drift);
                    }}
                }}
                
                if (slopes.length > 0) {{
                    const bullish = slopes.filter(s => s > 0).length;
                    const bearish = slopes.filter(s => s < 0).length;
                    const total = slopes.length;
                    currentNetProb = (bullish - bearish) / total;
                    
                    // Apply threshold for signal classification (same as historical)
                    if (currentNetProb > threshold) {{
                        currentSignal = 'BULLISH';
                    }} else if (currentNetProb < -threshold) {{
                        currentSignal = 'BEARISH';
                    }} else {{
                        currentSignal = 'NEUTRAL';
                    }}
                }}
            }} else {{
                // Fallback: if no live forecast, use last historical signal
                currentSignal = signals[lastIdx] || 'NEUTRAL';
                currentNetProb = data.net_prob[lastIdx] || 0;
            }}
            
            // ===== FACTOR 1: Horizon Agreement (30% weight) =====
            // How many enabled horizons agree with the signal direction?
            let agreeing = 0;
            let totalEnabled = 0;
            const liveForecasts = [];
            
            // Use live_forecast.forecasts (TODAY's forecasts) instead of historical
            if (data.live_forecast && data.live_forecast.forecasts) {{
                enabledHorizons.forEach(h => {{
                    const key = 'D+' + h;
                    const forecast = data.live_forecast.forecasts[key];
                    if (forecast !== null && forecast !== undefined) {{
                        liveForecasts.push(forecast);
                        totalEnabled++;
                        if (currentSignal === 'BULLISH' && forecast > currentPrice) agreeing++;
                        else if (currentSignal === 'BEARISH' && forecast < currentPrice) agreeing++;
                    }}
                }});
            }}
            
            const agreementRatio = totalEnabled > 0 ? agreeing / totalEnabled : 0;
            const agreementScore = agreementRatio * 100;  // 0-100
            
            // ===== FACTOR 2: Snake Width / Tightness (25% weight) =====
            // Tighter snake = more confident. Calculate std dev of live forecasts
            
            let snakeWidth = 0;
            if (liveForecasts.length > 1) {{
                const mean = liveForecasts.reduce((a, b) => a + b, 0) / liveForecasts.length;
                const variance = liveForecasts.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / liveForecasts.length;
                snakeWidth = Math.sqrt(variance);
            }}
            // Normalize: tighter snake (lower width) = higher score
            // Width as % of price, then invert: 0% spread = 100 score, 5%+ spread = low score
            const widthPct = currentPrice > 0 ? (snakeWidth / currentPrice) * 100 : 0;
            const tightnessScore = Math.max(0, Math.min(100, 100 - (widthPct * 20)));
            
            // ===== FACTOR 3: Signal Strength (20% weight) =====
            // How strong is the net_prob? Closer to ±1 = stronger
            const signalStrength = Math.abs(currentNetProb);
            const strengthScore = signalStrength * 100;  // 0-100
            
            // ===== FACTOR 4: Slope Steepness (15% weight) =====
            // Average slope of forecasts - steeper = more conviction
            // Use live_forecast.forecasts (TODAY's forecasts)
            let totalSlope = 0;
            let slopeCount = 0;
            if (data.live_forecast && data.live_forecast.forecasts) {{
                enabledHorizons.forEach(h => {{
                    const key = 'D+' + h;
                    const forecast = data.live_forecast.forecasts[key];
                    if (forecast !== null && forecast !== undefined) {{
                        const slope = ((forecast - currentPrice) / currentPrice) * 100;  // % change
                        totalSlope += Math.abs(slope);
                        slopeCount++;
                    }}
                }});
            }}
            const avgSlope = slopeCount > 0 ? totalSlope / slopeCount : 0;
            // Normalize: 0% slope = 0 score, 3%+ avg slope = 100 score
            const slopeScore = Math.min(100, (avgSlope / 3) * 100);
            
            // ===== FACTOR 5: Historical DA at THIS signal strength (10% weight) =====
            // Calculate how accurate signals were when they had similar strength in the past
            const strengthBucketDA = calculateDAAtStrength(data, signals, signalStrength);
            const daScore = strengthBucketDA.da;  // DA for this strength bucket
            
            console.log('🎯 Bucketed DA:', {{
                currentStrength: (signalStrength * 100).toFixed(1) + '%',
                bucket: strengthBucketDA.bucket,
                bucketDA: strengthBucketDA.da.toFixed(1) + '%',
                sampleSize: strengthBucketDA.total + ' signals'
            }});
            
            // ===== WEIGHTED COMBINATION =====
            const weights = {{
                agreement: 0.30,
                tightness: 0.25,
                strength: 0.20,
                slope: 0.15,
                historicalDA: 0.10
            }};
            
            const rawConfidence = (
                (agreementScore * weights.agreement) +
                (tightnessScore * weights.tightness) +
                (strengthScore * weights.strength) +
                (slopeScore * weights.slope) +
                (daScore * weights.historicalDA)
            );
            
            // Scale and bound to 40-95 range (never show 0% or 100%)
            const finalConfidence = Math.max(40, Math.min(95, rawConfidence));
            
            // Determine tier
            let tier, tierClass;
            if (finalConfidence >= 80) {{
                tier = 'HIGH';
                tierClass = 'high';
            }} else if (finalConfidence >= 60) {{
                tier = 'MEDIUM';
                tierClass = 'medium';
            }} else {{
                tier = 'LOW';
                tierClass = 'low';
            }}
            
            return {{
                confidence: finalConfidence,
                tier: tier,
                tierClass: tierClass,
                factors: {{
                    agreement: {{ score: agreementScore, value: `${{agreeing}}/${{totalEnabled}}`, weight: weights.agreement }},
                    tightness: {{ score: tightnessScore, value: widthPct.toFixed(2) + '%', weight: weights.tightness }},
                    strength: {{ score: strengthScore, value: (signalStrength * 100).toFixed(0) + '%', weight: weights.strength }},
                    slope: {{ score: slopeScore, value: avgSlope.toFixed(2) + '%', weight: weights.slope }},
                    historicalDA: {{ score: daScore, value: `${{strengthBucketDA.da.toFixed(0)}}% (${{strengthBucketDA.bucket}})`, weight: weights.historicalDA }}
                }},
                bucketInfo: strengthBucketDA,  // Include bucket details for display
                signal: currentSignal
            }};
        }}
        
        // Calculate historical confidence for each day (for confidence tracker)
        // Uses ROLLING bucketed DA - only uses data up to each point (no look-ahead bias)
        function calculateHistoricalConfidence(data, signals) {{
            const confidenceHistory = [];
            const signalStartDateObj = new Date(data.signal_start_date);
            
            // Strength buckets for DA calculation
            const buckets = [
                {{ min: 0.00, max: 0.30, label: '0-30%' }},
                {{ min: 0.30, max: 0.50, label: '30-50%' }},
                {{ min: 0.50, max: 0.70, label: '50-70%' }},
                {{ min: 0.70, max: 0.85, label: '70-85%' }},
                {{ min: 0.85, max: 1.00, label: '85-100%' }}
            ];
            
            // Running bucket statistics (for rolling DA calculation)
            const bucketStats = buckets.map(() => ({{ correct: 0, total: 0 }}));
            
            for (let idx = 0; idx < data.dates.length; idx++) {{
                const currentDate = new Date(data.dates[idx]);
                const currentPrice = data.prices[idx];
                const currentSignal = signals[idx] || 'NEUTRAL';
                const currentNetProb = data.net_prob[idx] || 0;
                const currentStrength = Math.abs(currentNetProb);
                
                // Update rolling bucket stats from PREVIOUS day's outcome (if available)
                // This ensures we only use past data, not future data
                if (idx > 0 && currentDate >= signalStartDateObj) {{
                    const prevSignal = signals[idx - 1];
                    const prevStrength = Math.abs(data.net_prob[idx - 1] || 0);
                    const prevPrice = data.prices[idx - 1];
                    
                    if (prevSignal && prevSignal !== 'NEUTRAL') {{
                        // Find which bucket the previous signal was in
                        let bucketIdx = 2; // default middle
                        for (let b = 0; b < buckets.length; b++) {{
                            if (prevStrength >= buckets[b].min && prevStrength < buckets[b].max) {{
                                bucketIdx = b;
                                break;
                            }}
                        }}
                        if (prevStrength >= 0.85) bucketIdx = 4;
                        
                        // Check if previous day's prediction was correct
                        const predictedUp = prevSignal === 'BULLISH';
                        const actualUp = currentPrice > prevPrice;
                        
                        bucketStats[bucketIdx].total++;
                        if (predictedUp === actualUp) bucketStats[bucketIdx].correct++;
                    }}
                }}
                
                if (currentSignal === 'NEUTRAL') {{
                    confidenceHistory.push(null);
                    continue;
                }}
                
                // Factor 1: Agreement
                let agreeing = 0;
                enabledHorizons.forEach(h => {{
                    const hKey = String(h);
                    if (data.horizon_forecasts && data.horizon_forecasts[hKey] && data.horizon_forecasts[hKey][idx] !== undefined) {{
                        const forecast = data.horizon_forecasts[hKey][idx];
                        if (currentSignal === 'BULLISH' && forecast > currentPrice) agreeing++;
                        else if (currentSignal === 'BEARISH' && forecast < currentPrice) agreeing++;
                    }}
                }});
                const agreementScore = enabledHorizons.length > 0 ? (agreeing / enabledHorizons.length) * 100 : 0;
                
                // Factor 2: Tightness
                const forecasts = [];
                enabledHorizons.forEach(h => {{
                    const hKey = String(h);
                    if (data.horizon_forecasts && data.horizon_forecasts[hKey] && data.horizon_forecasts[hKey][idx] !== undefined) {{
                        forecasts.push(data.horizon_forecasts[hKey][idx]);
                    }}
                }});
                let tightnessScore = 50;
                if (forecasts.length > 1) {{
                    const mean = forecasts.reduce((a, b) => a + b, 0) / forecasts.length;
                    const variance = forecasts.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / forecasts.length;
                    const width = Math.sqrt(variance);
                    const widthPct = currentPrice > 0 ? (width / currentPrice) * 100 : 0;
                    tightnessScore = Math.max(0, Math.min(100, 100 - (widthPct * 20)));
                }}
                
                // Factor 3: Strength
                const strengthScore = currentStrength * 100;
                
                // Factor 4: Slope
                let totalSlope = 0, slopeCount = 0;
                enabledHorizons.forEach(h => {{
                    const hKey = String(h);
                    if (data.horizon_forecasts && data.horizon_forecasts[hKey] && data.horizon_forecasts[hKey][idx] !== undefined) {{
                        const forecast = data.horizon_forecasts[hKey][idx];
                        totalSlope += Math.abs(((forecast - currentPrice) / currentPrice) * 100);
                        slopeCount++;
                    }}
                }});
                const slopeScore = slopeCount > 0 ? Math.min(100, ((totalSlope / slopeCount) / 3) * 100) : 0;
                
                // Factor 5: ROLLING Bucketed DA (only uses data up to this point!)
                // Find which bucket the current signal falls into
                let currentBucketIdx = 2; // default middle
                for (let b = 0; b < buckets.length; b++) {{
                    if (currentStrength >= buckets[b].min && currentStrength < buckets[b].max) {{
                        currentBucketIdx = b;
                        break;
                    }}
                }}
                if (currentStrength >= 0.85) currentBucketIdx = 4;
                
                // Get DA for this bucket using only PAST data
                const bucketStat = bucketStats[currentBucketIdx];
                let daScore = bucketStat.total >= 5 
                    ? (bucketStat.correct / bucketStat.total) * 100 
                    : 50;  // Default to 50% if not enough data in bucket yet
                
                // Weighted combination
                const raw = (agreementScore * 0.30) + (tightnessScore * 0.25) + (strengthScore * 0.20) + (slopeScore * 0.15) + (daScore * 0.10);
                confidenceHistory.push(Math.max(40, Math.min(95, raw)));
            }}
            
            return confidenceHistory;
        }}
        
        // Store last calculated live confidence for chart annotation
        let lastLiveConfidence = null;
        
        function updateCharts() {{
            const data = ASSET_DATA[currentAsset];
            const signals = getCurrentSignals(data);
            
            // Calculate Live Signal Confidence for TODAY's forecast
            lastLiveConfidence = calculateLiveSignalConfidence(data, signals);
            
            // Pre-calculate indicators for filtering (if enabled)
            let ichimokuFilter = null;
            let emaValues = null;
            
            if (showIchimoku) {{
                ichimokuFilter = calculateIchimoku(data.prices, data.dates);
            }}
            if (emaPeriod > 0) {{
                emaValues = calculateEMA(data.prices, emaPeriod);
            }}
            
            // Build snake traces (segmented by color)
            const traces = [];
            
            for (let i = 0; i < data.dates.length - 1; i++) {{
                const signal = signals[i];
                const strength = data.strength[i];
                const width = 2 + (strength * 10);
                
                let color;
                let filterReasons = [];  // Track filter reasons
                
                if (signal === 'BULLISH') {{
                    color = '#00ff88';
                }} else if (signal === 'BEARISH') {{
                    color = '#ff3366';
                }} else {{
                    color = 'rgba(100, 100, 100, 0.5)';
                }}
                
                const signalIsBullish = (signal === 'BULLISH');
                const signalIsBearish = (signal === 'BEARISH');
                
                // Apply RSI filter if enabled: Filter out overbought bullish signals and oversold bearish signals
                if (rsiFilterEnabled && data.rsi && data.rsi[i] !== undefined) {{
                    const rsiVal = data.rsi[i];
                    if (signalIsBullish && rsiVal > data.rsi_overbought) {{
                        // Bullish signal but RSI overbought → conflict
                        color = 'rgba(100, 100, 100, 0.4)';
                        filterReasons.push('RSI');
                    }} else if (signalIsBearish && rsiVal < data.rsi_oversold) {{
                        // Bearish signal but RSI oversold → conflict
                        color = 'rgba(100, 100, 100, 0.4)';
                        filterReasons.push('RSI');
                    }}
                }}
                
                // Apply EMA filter if enabled: Price above EMA = bullish, below = bearish
                if (emaPeriod > 0 && emaValues && emaValues[i] !== null && !isNaN(emaValues[i])) {{
                    const priceAboveEMA = data.prices[i] > emaValues[i];
                    
                    if (signalIsBullish && !priceAboveEMA) {{
                        // Bullish signal but price below EMA → conflict
                        color = 'rgba(100, 100, 100, 0.4)';
                        filterReasons.push('EMA');
                    }} else if (signalIsBearish && priceAboveEMA) {{
                        // Bearish signal but price above EMA → conflict
                        color = 'rgba(100, 100, 100, 0.4)';
                        filterReasons.push('EMA');
                    }}
                }}
                
                // Apply Ichimoku cloud color filter if enabled
                if (showIchimoku && ichimokuFilter && ichimokuFilter.cloudBullish[i] !== null) {{
                    const cloudIsBullish = ichimokuFilter.cloudBullish[i];
                    
                    // Filter: signal must match cloud color, otherwise grey out
                    if (signalIsBullish && !cloudIsBullish) {{
                        // Bullish signal but bearish cloud → conflict → grey
                        color = 'rgba(100, 100, 100, 0.4)';
                        filterReasons.push('Ichimoku');
                    }} else if (signalIsBearish && cloudIsBullish) {{
                        // Bearish signal but bullish cloud → conflict → grey
                        color = 'rgba(100, 100, 100, 0.4)';
                        filterReasons.push('Ichimoku');
                    }}
                }}
                
                const filterNote = filterReasons.length > 0 ? ` [Filtered: ${{filterReasons.join(', ')}}]` : '';
                
                traces.push({{
                    x: [data.dates[i], data.dates[i + 1]],
                    y: [data.prices[i], data.prices[i + 1]],
                    mode: 'lines',
                    line: {{ color: color, width: width }},
                    showlegend: false,
                    hoverinfo: 'text',
                    hovertext: `Date: ${{data.dates[i]}}<br>Price: ${{data.prices[i].toFixed(2)}}<br>Signal: ${{signal}}${{filterNote}}<br>Strength: ${{strength.toFixed(2)}}`
                }});
            }}
            
            // Add live forecast if available (only for ENABLED horizons)
            if (data.live_forecast && Object.keys(data.live_forecast.forecasts).length > 0) {{
                const lastDate = new Date(data.dates[data.dates.length - 1]);
                const basePrice = data.prices[data.prices.length - 1];
                
                const futureDates = [data.dates[data.dates.length - 1]];
                const futurePrices = [basePrice];
                
                // Sort horizons and FILTER to only enabled ones
                const horizons = Object.keys(data.live_forecast.forecasts)
                    .map(k => parseInt(k.replace('D+', '')))
                    .filter(h => enabledHorizons.includes(h))  // Only show enabled horizons!
                    .sort((a, b) => a - b);
                
                horizons.forEach(h => {{
                    const key = 'D+' + h;
                    const pred = data.live_forecast.forecasts[key];
                    if (pred !== null && pred !== undefined) {{
                        const futureDate = new Date(lastDate);
                        futureDate.setDate(futureDate.getDate() + h);
                        // Apply volatility multiplier (3x)
                        const amplified = basePrice + (pred - basePrice) * 3.0;
                        futureDates.push(futureDate.toISOString().split('T')[0]);
                        futurePrices.push(amplified);
                    }}
                }});
                
                if (futurePrices.length > 1) {{
                    // Determine live forecast color based on overall direction
                    const lastForecast = futurePrices[futurePrices.length - 1];
                    const pctChange = (lastForecast - basePrice) / basePrice;
                    
                    // Use 0.5% as threshold for neutral (essentially flat)
                    let liveColor;
                    if (pctChange > 0.005) {{
                        liveColor = '#00ff88';  // Green for bullish
                    }} else if (pctChange < -0.005) {{
                        liveColor = '#ff3366';  // Red for bearish
                    }} else {{
                        liveColor = 'rgba(100, 100, 100, 0.7)';  // Grey for neutral/flat
                    }}
                    
                    // ==================== EXPANDING CONFIDENCE CONE ====================
                    // Build cone that expands around the forecast line
                    const coneDates = [data.dates[data.dates.length - 1]];  // Start at last actual date
                    const coneUpper = [basePrice];  // Cone starts at base price
                    const coneLower = [basePrice];
                    
                    // Build expanding cone for each forecast point
                    for (let i = 1; i < futureDates.length; i++) {{
                        const forecastPrice = futurePrices[i];
                        const horizonDays = i;  // Approximate horizon
                        
                        // Uncertainty expands with sqrt(time) - financial standard
                        const baseUncertainty = Math.abs(forecastPrice - basePrice) * 0.25;
                        const timeExpansion = baseUncertainty * Math.sqrt(horizonDays);
                        
                        coneDates.push(futureDates[i]);
                        coneUpper.push(forecastPrice + timeExpansion);
                        coneLower.push(forecastPrice - timeExpansion);
                    }}
                    
                    // Add lower bound trace (invisible, just for fill reference)
                    traces.push({{
                        x: coneDates,
                        y: coneLower,
                        mode: 'lines',
                        line: {{ color: 'rgba(255, 215, 0, 0.5)', width: 1 }},
                        name: 'Cone Lower',
                        showlegend: false,
                        hoverinfo: 'skip'
                    }});
                    
                    // Add upper bound trace with fill to lower
                    traces.push({{
                        x: coneDates,
                        y: coneUpper,
                        mode: 'lines',
                        fill: 'tonexty',
                        fillcolor: 'rgba(255, 215, 0, 0.15)',
                        line: {{ color: 'rgba(255, 215, 0, 0.5)', width: 1 }},
                        name: 'Confidence Cone',
                        showlegend: false,
                        hoverinfo: 'skip'
                    }});
                    
                    // Add the forecast line ON TOP of the cone
                    traces.push({{
                        x: futureDates,
                        y: futurePrices,
                        mode: 'lines+markers',
                        line: {{ color: liveColor, width: 4, dash: 'dot' }},
                        marker: {{ size: 10, color: liveColor, symbol: 'diamond' }},
                        name: 'Live Forecast',
                        hovertemplate: 'Date: %{{x}}<br>Forecast: %{{y:.2f}}<extra></extra>'
                    }});
                }}
            }}
            
            // Add EMA overlay if enabled (reuse pre-calculated values)
            if (emaPeriod > 0 && emaValues) {{
                // Different colors for different periods
                const emaColors = {{ 9: '#FF6B6B', 20: '#FFD700', 50: '#4ECDC4', 100: '#45B7D1', 200: '#96CEB4' }};
                const emaColor = emaColors[emaPeriod] || '#FFD700';
                traces.push({{
                    x: data.dates,
                    y: emaValues,
                    mode: 'lines',
                    line: {{ color: emaColor, width: 2 }},
                    name: `EMA ${{emaPeriod}}`,
                    hovertemplate: `EMA ${{emaPeriod}}: %{{y:.2f}}<extra></extra>`
                }});
            }}
            
            // Add Ichimoku Cloud if enabled
            if (showIchimoku && ichimokuFilter) {{
                const ichimoku = ichimokuFilter;  // Reuse pre-calculated data
                const cloudPolygons = buildCloudPolygons(ichimoku.cloudDates, ichimoku.senkouA, ichimoku.senkouB);
                
                // Draw cloud polygons first (so they're behind the lines)
                let bullishAdded = false;
                let bearishAdded = false;
                
                cloudPolygons.forEach(poly => {{
                    // Create polygon: go forward along upper, then backward along lower
                    const xCoords = [...poly.x, ...poly.x.slice().reverse()];
                    const yCoords = [...poly.upper, ...poly.lower.slice().reverse()];
                    
                    const isBullish = poly.type === 'bullish';
                    const fillColor = isBullish ? 'rgba(38, 166, 91, 0.35)' : 'rgba(231, 76, 60, 0.35)';
                    const showInLegend = isBullish ? !bullishAdded : !bearishAdded;
                    
                    if (isBullish) bullishAdded = true;
                    else bearishAdded = true;
                    
                    traces.push({{
                        x: xCoords,
                        y: yCoords,
                        fill: 'toself',
                        fillcolor: fillColor,
                        mode: 'lines',
                        line: {{ width: 0 }},
                        name: isBullish ? 'Cloud (Bullish)' : 'Cloud (Bearish)',
                        showlegend: showInLegend,
                        hoverinfo: 'skip'
                    }});
                }});
                
                // Tenkan-sen (Conversion Line) - Blue
                traces.push({{
                    x: data.dates,
                    y: ichimoku.tenkan,
                    mode: 'lines',
                    line: {{ color: '#3498db', width: 1.5 }},
                    name: 'Tenkan (9)',
                    hovertemplate: 'Tenkan: %{{y:.2f}}<extra></extra>'
                }});
                
                // Kijun-sen (Base Line) - Maroon/Dark Red
                traces.push({{
                    x: data.dates,
                    y: ichimoku.kijun,
                    mode: 'lines',
                    line: {{ color: '#e74c3c', width: 1.5 }},
                    name: 'Kijun (26)',
                    hovertemplate: 'Kijun: %{{y:.2f}}<extra></extra>'
                }});
                
                // Senkou Span A line (green)
                traces.push({{
                    x: ichimoku.cloudDates,
                    y: ichimoku.senkouA,
                    mode: 'lines',
                    line: {{ color: 'rgba(38, 166, 91, 0.6)', width: 1 }},
                    name: 'Span A',
                    showlegend: false,
                    hovertemplate: 'Span A: %{{y:.2f}}<extra></extra>'
                }});
                
                // Senkou Span B line (red)
                traces.push({{
                    x: ichimoku.cloudDates,
                    y: ichimoku.senkouB,
                    mode: 'lines',
                    line: {{ color: 'rgba(231, 76, 60, 0.6)', width: 1 }},
                    name: 'Span B',
                    showlegend: false,
                    hovertemplate: 'Span B: %{{y:.2f}}<extra></extra>'
                }});
            }}
            
            // Build overlay labels
            const overlayLabels = [];
            if (rsiFilterEnabled) overlayLabels.push('RSI');
            if (emaPeriod > 0) overlayLabels.push(`EMA${{emaPeriod}}`);
            if (showIchimoku) overlayLabels.push('Ichimoku');
            const overlayText = overlayLabels.length > 0 ? `<span style="font-size:11px;color:var(--qdt-primary);">[${{overlayLabels.join(' | ')}}]</span>` : '';
            
            // Set initial view to start from where signals begin (can zoom out to see history)
            const signalStartDate = data.signal_start_date || data.dates[0];
            const lastDataDate = data.dates[data.dates.length - 1];
            
            // ==================== CALCULATE FORECAST BAND FOR CHART ====================
            // Get current signal and create a band showing the min/max forecast range
            const currentSignal = signals[signals.length - 1];
            let forecastBandShapes = [];
            let confidenceBoxPosition = null;  // Store position for confidence box
            let lastDate = null;  // Will be calculated based on confidence box position
            let forecastBandLabels = null;  // Store min/max forecast values for price labels
            
            // Use the SAME data source as the live forecast dots (data.live_forecast.forecasts)
            if (data.live_forecast && Object.keys(data.live_forecast.forecasts).length > 0) {{
                const basePrice = data.prices[data.prices.length - 1];
                const lastDateObj = new Date(data.dates[data.dates.length - 1]);
                
                // Get enabled horizons and filter forecasts
                const horizons = Object.keys(data.live_forecast.forecasts)
                    .map(k => parseInt(k.replace('D+', '')))
                    .filter(h => enabledHorizons.includes(h))
                    .sort((a, b) => a - b);
                
                // Calculate amplified prices (same 3x amplification as the forecast dots)
                const amplifiedPrices = [];
                let maxHorizon = 0;
                
                horizons.forEach(h => {{
                    const key = 'D+' + h;
                    const pred = data.live_forecast.forecasts[key];
                    if (pred !== null && pred !== undefined) {{
                        // Apply same 3x volatility multiplier
                        const amplified = basePrice + (pred - basePrice) * 3.0;
                        amplifiedPrices.push(amplified);
                        if (h > maxHorizon) maxHorizon = h;
                    }}
                }});
                
                if (amplifiedPrices.length > 0) {{
                    const minForecast = Math.min(...amplifiedPrices);
                    const maxForecast = Math.max(...amplifiedPrices);
                    
                    // Calculate the end date for the forecast band
                    const maxForecastDateObj = new Date(lastDateObj);
                    maxForecastDateObj.setDate(maxForecastDateObj.getDate() + maxHorizon);
                    const maxForecastDate = maxForecastDateObj.toISOString().split('T')[0];
                    
                    // Store forecast band info for price labels
                    forecastBandLabels = {{
                        minForecast: minForecast,
                        maxForecast: maxForecast,
                        maxForecastDate: maxForecastDate
                    }};
                    
                    // Calculate confidence box position - to the RIGHT of the forecast band
                    // Position it horizontally next to the forecast band with padding
                    const forecastBandWidth = maxForecast - minForecast;
                    const forecastBandCenter = (minForecast + maxForecast) / 2;
                    
                    if (lastLiveConfidence) {{
                        // Y position: center of the forecast band (vertically aligned with band)
                        const confidenceY = forecastBandCenter;
                        
                        // X position: to the RIGHT of the forecast band end date
                        // Add padding (20% of forecast duration or minimum 10 days) so it works in both zoom in/out
                        const forecastStartTime = new Date(lastDataDate).getTime();
                        const forecastEndTime = maxForecastDateObj.getTime();
                        const forecastDuration = forecastEndTime - forecastStartTime;
                        const paddingDays = Math.max(forecastDuration * 0.20 / (1000 * 60 * 60 * 24), 10);  // At least 10 days padding
                        
                        const confidenceDateObj = new Date(maxForecastDateObj);
                        confidenceDateObj.setDate(confidenceDateObj.getDate() + paddingDays);
                        const confidenceX = confidenceDateObj.toISOString().split('T')[0];
                        
                        confidenceBoxPosition = {{
                            x: confidenceX,
                            y: confidenceY,
                            yanchor: 'middle',  // Vertically centered on the forecast band
                            xanchor: 'left'     // Box starts from this X position (to the right)
                        }};
                        
                        // Calculate lastDate to include confidence box + blank space padding
                        const confidenceBoxDateObj = new Date(confidenceX);
                        const blankSpacePadding = 20;  // 20 days of blank space after confidence box
                        confidenceBoxDateObj.setDate(confidenceBoxDateObj.getDate() + blankSpacePadding);
                        lastDate = confidenceBoxDateObj.toISOString().split('T')[0];
                    }}
                    
                    // If no confidence box, use forecast band end + padding
                    if (!lastDate) {{
                        const blankSpacePadding = 20;
                        const fallbackDateObj = new Date(maxForecastDateObj);
                        fallbackDateObj.setDate(fallbackDateObj.getDate() + blankSpacePadding);
                        lastDate = fallbackDateObj.toISOString().split('T')[0];
                    }}
                    
                    // Color based on current signal (yellow/gold for visibility)
                    const lineColor = 'rgba(255, 215, 0, 0.7)';  // Gold color for the lines
                    const bandColor = 'rgba(255, 215, 0, 0.12)';  // Subtle gold fill
                    
                    // ==================== EXPANDING CONFIDENCE CONE ====================
                    // Build cone that starts narrow at base price and expands over time
                    // Collect forecast data per horizon for cone construction
                    const conePoints = [];
                    
                    // Start point: base price at last data date (cone starts here)
                    conePoints.push({{
                        date: lastDataDate,
                        upper: basePrice,
                        lower: basePrice,
                        center: basePrice
                    }});
                    
                    // Build expanding cone - for each horizon, calculate spread
                    horizons.forEach(h => {{
                        const key = 'D+' + h;
                        const pred = data.live_forecast.forecasts[key];
                        if (pred !== null && pred !== undefined) {{
                            const amplified = basePrice + (pred - basePrice) * 3.0;
                            const dateObj = new Date(lastDateObj);
                            dateObj.setDate(dateObj.getDate() + h);
                            const dateStr = dateObj.toISOString().split('T')[0];
                            
                            // Calculate cone spread that expands with horizon
                            // Uncertainty grows roughly with sqrt of time (random walk)
                            const baseSpread = Math.abs(amplified - basePrice) * 0.3;  // 30% of move as base uncertainty
                            const timeSpread = baseSpread * Math.sqrt(h / horizons[0]);  // Expands with sqrt(time)
                            
                            conePoints.push({{
                                date: dateStr,
                                upper: amplified + timeSpread,
                                lower: amplified - timeSpread,
                                center: amplified
                            }});
                        }}
                    }});
                    
                    // Build SVG path for the cone (polygon)
                    // Go forward along upper bound, then backward along lower bound
                    if (conePoints.length > 1) {{
                        const upperPath = conePoints.map(p => `${{p.date}},${{p.upper}}`);
                        const lowerPath = conePoints.slice().reverse().map(p => `${{p.date}},${{p.lower}}`);
                        
                        // Create filled cone shape using Plotly path
                        forecastBandShapes = [
                            // Upper boundary line
                            {{
                                type: 'path',
                                path: 'M ' + conePoints.map(p => {{
                                    const dateMs = new Date(p.date).getTime();
                                    return `${{p.date}} ${{p.upper}}`;
                                }}).join(' L '),
                                line: {{ color: lineColor, width: 2 }}
                            }},
                            // Lower boundary line  
                            {{
                                type: 'path',
                                path: 'M ' + conePoints.map(p => {{
                                    return `${{p.date}} ${{p.lower}}`;
                                }}).join(' L '),
                                line: {{ color: lineColor, width: 2 }}
                            }}
                        ];
                        
                        // Add cone annotation showing expansion
                        console.log(`📊 Confidence cone: starts at ${{basePrice.toFixed(2)}}, expands to ${{conePoints[conePoints.length-1].lower.toFixed(2)}} - ${{conePoints[conePoints.length-1].upper.toFixed(2)}} (horizon D+${{maxHorizon}})`);
                    }} else {{
                        // Fallback to simple rectangle if not enough points
                        forecastBandShapes = [
                            {{
                                type: 'rect',
                                x0: lastDataDate,
                                x1: maxForecastDate,
                                y0: minForecast,
                                y1: maxForecast,
                                fillcolor: bandColor,
                                line: {{ width: 0 }}
                            }}
                        ];
                    }}
                    
                    console.log(`📊 Forecast band: ${{minForecast.toFixed(2)}} - ${{maxForecast.toFixed(2)}} (${{amplifiedPrices.length}} points, max horizon: ${{maxHorizon}} days)`);
                }}
            }}
            
            // Fallback: if no forecast data, use simple padding
            if (!lastDate) {{
                const lastDateObj = new Date(lastDataDate);
                lastDateObj.setDate(lastDateObj.getDate() + 60);
                lastDate = lastDateObj.toISOString().split('T')[0];
            }}
            
            const snakeLayout = {{
                title: {{
                    text: `<b>${{currentAsset.replace('_', ' ')}}</b> - Optimized Snake Chart ${{overlayText}}`,
                    font: {{ size: 18, color: '#fff' }}
                }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: '#888', family: 'Open Sans' }},
                dragmode: 'pan',  // Default to pan mode
                xaxis: {{
                    range: [signalStartDate, lastDate],  // Initial view from signal start
                    gridcolor: 'rgba(100,100,100,0.2)',
                    zerolinecolor: 'rgba(100,100,100,0.2)',
                    tickfont: {{ size: 11 }},
                    showspikes: true,
                    spikemode: 'toaxis+across',
                    spikethickness: 0.1,
                    spikecolor: 'rgba(80,80,80,0.08)',
                    spikedash: 'dot',
                    spikesnap: 'cursor'
                }},
                yaxis: {{
                    title: 'Price',
                    gridcolor: 'rgba(100,100,100,0.2)',
                    zerolinecolor: 'rgba(100,100,100,0.2)',
                    tickfont: {{ size: 11 }},
                    showspikes: true,
                    spikemode: 'toaxis+across',
                    spikethickness: 0.1,
                    spikecolor: 'rgba(80,80,80,0.08)',
                    spikedash: 'dot',
                    spikesnap: 'cursor'
                }},
                shapes: [
                    // Vertical line separating historical from forecast
                    {{
                        type: 'line',
                        x0: lastDataDate,
                        x1: lastDataDate,
                        y0: 0,
                        y1: 1,
                        yref: 'paper',
                        line: {{ color: 'rgba(255,255,255,0.3)', width: 1, dash: 'dot' }}
                    }},
                    // Forecast band (min/max range)
                    ...forecastBandShapes
                ],
                annotations: [
                    // "Forecast →" label
                    {{
                        x: lastDataDate,
                        y: 1.02,
                        yref: 'paper',
                        text: 'Forecast →',
                        showarrow: false,
                        font: {{ size: 10, color: 'rgba(255,255,255,0.5)' }},
                        xanchor: 'left'
                    }},
                    // Price labels for forecast band lines (high and low)
                    ...(forecastBandLabels ? [
                        // High forecast line label (top yellow line)
                        {{
                            x: forecastBandLabels.maxForecastDate,
                            y: forecastBandLabels.maxForecast,
                            xref: 'x',
                            yref: 'y',
                            text: `${{forecastBandLabels.maxForecast.toFixed(2)}}`,
                            showarrow: false,
                            font: {{ 
                                size: 11, 
                                color: 'rgba(255, 215, 0, 0.9)',
                                family: 'Open Sans'
                            }},
                            bgcolor: 'rgba(0, 0, 0, 0.7)',
                            bordercolor: 'rgba(255, 215, 0, 0.5)',
                            borderwidth: 1,
                            borderpad: 4,
                            xanchor: 'left',
                            yanchor: 'middle',
                            align: 'left'
                        }},
                        // Low forecast line label (bottom yellow line)
                        {{
                            x: forecastBandLabels.maxForecastDate,
                            y: forecastBandLabels.minForecast,
                            xref: 'x',
                            yref: 'y',
                            text: `${{forecastBandLabels.minForecast.toFixed(2)}}`,
                            showarrow: false,
                            font: {{ 
                                size: 11, 
                                color: 'rgba(255, 215, 0, 0.9)',
                                family: 'Open Sans'
                            }},
                            bgcolor: 'rgba(0, 0, 0, 0.7)',
                            bordercolor: 'rgba(255, 215, 0, 0.5)',
                            borderwidth: 1,
                            borderpad: 4,
                            xanchor: 'left',
                            yanchor: 'middle',
                            align: 'left'
                        }}
                    ] : []),
                    // Live Signal Badge - Combined (Signal + Confidence) - Positioned relative to forecast band
                    // Bullish: below lower yellow line, Bearish: above upper yellow line, Neutral: middle of band
                    ...(confidenceBoxPosition && lastLiveConfidence ? [{{
                        x: confidenceBoxPosition.x,
                        y: confidenceBoxPosition.y,
                        xref: 'x',
                        yref: 'y',
                        text: `<span style="font-size:18px;">${{lastLiveConfidence.signal === 'BULLISH' ? '↑' : lastLiveConfidence.signal === 'BEARISH' ? '↓' : '—'}}</span> <b style="color:${{lastLiveConfidence.signal === 'BULLISH' ? '#00ff88' : lastLiveConfidence.signal === 'BEARISH' ? '#ff4444' : '#888'}}">${{lastLiveConfidence.signal || 'NEUTRAL'}}</b><br><span style="font-size:9px; opacity:0.7;">${{lastLiveConfidence.tier}} CONFIDENCE</span><br><b style="font-size:18px; color:${{lastLiveConfidence.confidence >= 75 ? '#00ff88' : lastLiveConfidence.confidence >= 55 ? '#ffaa00' : '#ff6666'}}">${{lastLiveConfidence.confidence.toFixed(0)}}%</b> <span style="font-size:12px; opacity:0.6; cursor:pointer;">ⓘ</span>`,
                        showarrow: false,
                        font: {{ 
                            size: 11, 
                            color: lastLiveConfidence ? 
                                (lastLiveConfidence.signal === 'BULLISH' ? '#00ff88' : 
                                 lastLiveConfidence.signal === 'BEARISH' ? '#ff4444' : '#888') : '#888'
                        }},
                        bgcolor: 'rgba(0,0,0,0.9)',
                        bordercolor: lastLiveConfidence ?
                            (lastLiveConfidence.signal === 'BULLISH' ? 'rgba(0,255,136,0.6)' : 
                             lastLiveConfidence.signal === 'BEARISH' ? 'rgba(255,68,68,0.6)' : 'rgba(100,100,100,0.5)') : 'rgba(100,100,100,0.5)',
                        borderwidth: 2,
                        borderpad: 10,
                        xanchor: confidenceBoxPosition.xanchor || 'left',
                        yanchor: confidenceBoxPosition.yanchor || 'middle',
                        align: 'center',
                        captureevents: true,
                        name: 'confidenceBadge'
                    }}] : [])
                ],
                showlegend: true,
                legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0.5)' }},
                hovermode: 'closest',
                margin: {{ t: 50, b: 40, l: 60, r: 80 }}
            }};
            
            const snakeConfig = {{
                responsive: true,
                scrollZoom: true,
                displayModeBar: true,
                modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'eraseshape'],
                displaylogo: false
            }};
            
            Plotly.newPlot('snake-chart', traces, snakeLayout, snakeConfig);
            
            // Auto-scale Y axis when zooming on X axis
            document.getElementById('snake-chart').on('plotly_relayout', function(eventdata) {{
                // Skip auto-scale if trader mode is enabled (it handles its own Y-axis range)
                if (traderModeEnabled) return;
                
                if (eventdata['xaxis.range[0]'] || eventdata['xaxis.range']) {{
                    const chartDiv = document.getElementById('snake-chart');
                    const xRange = chartDiv.layout.xaxis.range;
                    
                    // Find min/max Y values in visible X range
                    let minY = Infinity, maxY = -Infinity;
                    const startDate = new Date(xRange[0]);
                    const endDate = new Date(xRange[1]);
                    
                    for (let i = 0; i < data.dates.length; i++) {{
                        const d = new Date(data.dates[i]);
                        if (d >= startDate && d <= endDate) {{
                            if (data.prices[i] < minY) minY = data.prices[i];
                            if (data.prices[i] > maxY) maxY = data.prices[i];
                        }}
                    }}
                    
                    // Include forecast bands if they fall within visible range
                    if (data.live_forecast && Object.keys(data.live_forecast.forecasts).length > 0) {{
                        const basePrice = data.prices[data.prices.length - 1];
                        const lastDataDate = new Date(data.dates[data.dates.length - 1]);
                        
                        // Check if forecast area is in visible range
                        if (lastDataDate >= startDate && lastDataDate <= endDate) {{
                            const currentEnabledHorizons = enabledHorizons.length > 0 ? enabledHorizons : (data.horizons || []);
                            const horizons = Object.keys(data.live_forecast.forecasts)
                                .map(k => parseInt(k.replace('D+', '')))
                                .filter(h => currentEnabledHorizons.includes(h))
                                .sort((a, b) => a - b);
                            
                            const amplifiedPrices = [];
                            let maxHorizon = 0;
                            
                            horizons.forEach(h => {{
                                const key = 'D+' + h;
                                const pred = data.live_forecast.forecasts[key];
                                if (pred !== null && pred !== undefined) {{
                                    const amplified = basePrice + (pred - basePrice) * 3.0;
                                    amplifiedPrices.push(amplified);
                                    if (h > maxHorizon) maxHorizon = h;
                                }}
                            }});
                            
                            if (amplifiedPrices.length > 0) {{
                                const minForecast = Math.min(...amplifiedPrices);
                                const maxForecast = Math.max(...amplifiedPrices);
                                
                                const maxForecastDateObj = new Date(lastDataDate);
                                maxForecastDateObj.setDate(maxForecastDateObj.getDate() + maxHorizon);
                                
                                // If forecast extends into visible range, include it
                                if (maxForecastDateObj >= startDate) {{
                                    if (minForecast < minY) minY = minForecast;
                                    if (maxForecast > maxY) maxY = maxForecast;
                                }}
                            }}
                        }}
                    }}
                    
                    if (minY !== Infinity && maxY !== -Infinity) {{
                        const padding = (maxY - minY) * 0.05;
                        Plotly.relayout('snake-chart', {{
                            'yaxis.range': [minY - padding, maxY + padding]
                        }});
                    }}
                }}
            }});
            
            // Click handler for confidence badge annotation
            document.getElementById('snake-chart').on('plotly_clickannotation', function(event) {{
                if (event.annotation && event.annotation.name === 'confidenceBadge') {{
                    showConfidenceExplanation();
                }}
            }});
            
            // RSI Chart
            const rsiTrace = {{
                x: data.dates,
                y: data.rsi,
                mode: 'lines',
                line: {{ color: '#a855f7', width: 1.5 }},
                name: 'RSI',
                fill: 'tozeroy',
                fillcolor: 'rgba(168, 85, 247, 0.1)'
            }};
            
            const rsiLayout = {{
                title: {{
                    text: `RSI Filter (OB: ${{data.rsi_overbought}} / OS: ${{data.rsi_oversold}})`,
                    font: {{ size: 14, color: '#888' }}
                }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: '#888', family: 'Open Sans' }},
                dragmode: 'pan',
                xaxis: {{
                    range: [signalStartDate, lastDate],  // Match main chart initial view
                    gridcolor: 'rgba(100,100,100,0.2)',
                    tickfont: {{ size: 10 }},
                    showspikes: true,
                    spikemode: 'toaxis+across',
                    spikethickness: 0.1,
                    spikecolor: 'rgba(80,80,80,0.08)',
                    spikedash: 'dot',
                    spikesnap: 'cursor'
                }},
                yaxis: {{
                    range: [0, 100],
                    gridcolor: 'rgba(100,100,100,0.2)',
                    tickfont: {{ size: 10 }},
                    showspikes: true,
                    spikemode: 'toaxis+across',
                    spikethickness: 0.1,
                    spikecolor: 'rgba(80,80,80,0.08)',
                    spikedash: 'dot',
                    spikesnap: 'cursor'
                }},
                shapes: [
                    {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: data.rsi_overbought, y1: data.rsi_overbought, line: {{ color: '#ff3366', dash: 'dash', width: 1 }} }},
                    {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: data.rsi_oversold, y1: data.rsi_oversold, line: {{ color: '#00ff88', dash: 'dash', width: 1 }} }},
                    {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 50, y1: 50, line: {{ color: '#555', dash: 'dot', width: 1 }} }}
                ],
                showlegend: false,
                margin: {{ t: 40, b: 30, l: 60, r: 80 }}
            }};
            
            const rsiConfig = {{
                responsive: true,
                scrollZoom: true,
                displayModeBar: false,
                displaylogo: false
            }};
            
            Plotly.newPlot('rsi-chart', [rsiTrace], rsiLayout, rsiConfig);
            
            // Sync RSI chart zoom with main chart
            document.getElementById('snake-chart').on('plotly_relayout', function(eventdata) {{
                if (eventdata['xaxis.range[0]'] !== undefined) {{
                    Plotly.relayout('rsi-chart', {{
                        'xaxis.range': [eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]']]
                    }});
                }} else if (eventdata['xaxis.autorange']) {{
                    Plotly.relayout('rsi-chart', {{'xaxis.autorange': true}});
                }}
            }});
            
            // Double-click to reset zoom to signal start date
            document.getElementById('snake-chart').on('plotly_doubleclick', function() {{
                Plotly.relayout('snake-chart', {{'xaxis.range': [signalStartDate, lastDate], 'yaxis.autorange': true}});
                Plotly.relayout('rsi-chart', {{'xaxis.range': [signalStartDate, lastDate]}});
                Plotly.relayout('confidence-chart', {{'xaxis.range': [signalStartDate, lastDate]}});
                Plotly.relayout('equity-chart', {{'xaxis.range': [signalStartDate, lastDate], 'yaxis.autorange': true}});
            }});
            
            // ==================== CONFIDENCE TRACKER CHART ====================
            // Calculate historical confidence for each day
            const confidenceHistory = calculateHistoricalConfidence(data, signals);
            
            // Create colored confidence trace based on value
            const confColors = confidenceHistory.map(c => {{
                if (c === null) return 'rgba(100,100,100,0.3)';
                if (c >= 75) return '#00ff88';
                if (c >= 55) return '#ffaa00';
                return '#ff6666';
            }});
            
            const confidenceTrace = {{
                x: data.dates,
                y: confidenceHistory,
                mode: 'lines',
                line: {{ color: '#00d4ff', width: 2 }},
                name: 'Confidence',
                fill: 'tozeroy',
                fillcolor: 'rgba(0, 212, 255, 0.1)'
            }};
            
            const confidenceLayout = {{
                title: {{
                    text: 'Historical Signal Confidence',
                    font: {{ size: 14, color: '#888' }}
                }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: '#888', family: 'Open Sans' }},
                dragmode: 'pan',
                xaxis: {{
                    range: [signalStartDate, lastDate],
                    gridcolor: 'rgba(100,100,100,0.2)',
                    tickfont: {{ size: 10 }},
                    showspikes: true,
                    spikemode: 'toaxis+across',
                    spikethickness: 0.1,
                    spikecolor: 'rgba(80,80,80,0.08)',
                    spikedash: 'dot',
                    spikesnap: 'cursor'
                }},
                yaxis: {{
                    range: [30, 100],
                    gridcolor: 'rgba(100,100,100,0.2)',
                    tickfont: {{ size: 10 }},
                    showspikes: true,
                    spikemode: 'toaxis+across',
                    spikethickness: 0.1,
                    spikecolor: 'rgba(80,80,80,0.08)',
                    spikedash: 'dot',
                    spikesnap: 'cursor'
                }},
                shapes: [
                    // High confidence threshold
                    {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 75, y1: 75, line: {{ color: '#00ff88', dash: 'dash', width: 1 }} }},
                    // Medium confidence threshold
                    {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 55, y1: 55, line: {{ color: '#ffaa00', dash: 'dash', width: 1 }} }}
                ],
                showlegend: false,
                margin: {{ t: 40, b: 30, l: 60, r: 80 }}
            }};
            
            const confidenceConfig = {{
                responsive: true,
                scrollZoom: true,
                displayModeBar: false,
                displaylogo: false
            }};
            
            Plotly.newPlot('confidence-chart', [confidenceTrace], confidenceLayout, confidenceConfig);
            
            // Sync confidence chart zoom with main chart
            document.getElementById('snake-chart').on('plotly_relayout', function(eventdata) {{
                if (eventdata['xaxis.range[0]'] !== undefined) {{
                    Plotly.relayout('confidence-chart', {{
                        'xaxis.range': [eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]']]
                    }});
                }} else if (eventdata['xaxis.autorange']) {{
                    Plotly.relayout('confidence-chart', {{'xaxis.autorange': true}});
                }}
            }});
            
            // Close modal when clicking outside
            document.getElementById('confidenceModal').onclick = function(e) {{
                if (e.target === this) {{
                    this.style.display = 'none';
                }}
            }};
            
            // ==================== EQUITY CURVE CHART ====================
            // Calculate equity curves using SIGNAL-FOLLOWING strategy
            // (Enter on signal change, hold until signal flips - matches our validation)
            const startingCapital = 100;
            
            // 1. Buy & Hold - just compound returns
            const buyHoldEquity = [startingCapital];
            for (let i = 1; i < data.dates.length; i++) {{
                const ret = data.daily_returns[i];
                buyHoldEquity.push(buyHoldEquity[i-1] * (1 + ret));
            }}
            
            // 2. Signal-Following Strategy - enter on signal CHANGE, hold until flip
            // This matches our validated trading approach
            const ensembleEquity = [startingCapital];
            let currentPosition = null;
            let entryPrice = null;
            
            for (let i = 1; i < data.dates.length; i++) {{
                let signal = signals[i-1];  // Use previous day's signal
                const price = data.prices[i];
                const prevPrice = data.prices[i-1];
                const currentDate = new Date(data.dates[i-1]);
                const signalStartDateObj = new Date(data.signal_start_date);
                
                // Apply filters within signal period
                if (currentDate >= signalStartDateObj) {{
                    const signalIsBullish = (signal === 'BULLISH');
                    const signalIsBearish = (signal === 'BEARISH');
                    
                    // Apply EMA filter if enabled
                    if (emaPeriod > 0 && emaValues && emaValues[i-1] !== null && !isNaN(emaValues[i-1])) {{
                        const priceAboveEMA = data.prices[i-1] > emaValues[i-1];
                        if (signalIsBullish && !priceAboveEMA) signal = 'NEUTRAL';
                        else if (signalIsBearish && priceAboveEMA) signal = 'NEUTRAL';
                    }}
                    
                    // Apply Ichimoku filter if enabled
                    if (showIchimoku && ichimokuFilter && ichimokuFilter.cloudBullish[i-1] !== null) {{
                        const cloudIsBullish = ichimokuFilter.cloudBullish[i-1];
                        if (signal === 'BULLISH' && !cloudIsBullish) signal = 'NEUTRAL';
                        else if (signal === 'BEARISH' && cloudIsBullish) signal = 'NEUTRAL';
                    }}
                }}
                
                // Signal-following logic
                if (currentPosition === null) {{
                    // Not in position - check for entry
                    if (signal === 'BULLISH' || signal === 'BEARISH') {{
                        currentPosition = signal;
                        entryPrice = prevPrice;
                    }}
                    ensembleEquity.push(ensembleEquity[i-1]);  // Flat, no change
                }} else {{
                    // In position - calculate P&L and check for exit
                    let dailyPnL = 0;
                    if (currentPosition === 'BULLISH') {{
                        dailyPnL = (price - prevPrice) / prevPrice;  // Long gains
                    }} else if (currentPosition === 'BEARISH') {{
                        dailyPnL = (prevPrice - price) / prevPrice;  // Short gains
                    }}
                    
                    ensembleEquity.push(ensembleEquity[i-1] * (1 + dailyPnL));
                    
                    // Check for exit (signal changed)
                    if (signal !== currentPosition) {{
                        if (signal === 'BULLISH' || signal === 'BEARISH') {{
                            currentPosition = signal;
                            entryPrice = price;
                        }} else {{
                            currentPosition = null;
                            entryPrice = null;
                        }}
                    }}
                }}
            }}
            
            // Calculate final stats
            const buyHoldReturn = ((buyHoldEquity[buyHoldEquity.length-1] / startingCapital) - 1) * 100;
            const ensembleReturn = ((ensembleEquity[ensembleEquity.length-1] / startingCapital) - 1) * 100;
            const outperformance = ensembleReturn - buyHoldReturn;
            
            // Equity chart traces
            const equityTraces = [
                {{
                    x: data.dates,
                    y: buyHoldEquity,
                    mode: 'lines',
                    name: `Buy & Hold (${{buyHoldReturn.toFixed(1)}}%)`,
                    line: {{ color: '#888', width: 2 }}
                }},
                {{
                    x: data.dates,
                    y: ensembleEquity,
                    mode: 'lines',
                    name: `QDT Ensemble (${{ensembleReturn.toFixed(1)}}%)`,
                    line: {{ color: '#00d4ff', width: 3 }}
                }}
            ];
            
            // Add baseline
            const baseline = Array(data.dates.length).fill(startingCapital);
            equityTraces.push({{
                x: data.dates,
                y: baseline,
                mode: 'lines',
                name: 'Starting Capital',
                line: {{ color: '#444', width: 1, dash: 'dot' }}
            }});
            
            const equityLayout = {{
                title: {{
                    text: `💰 Equity Curve (Outperformance: <span style="color:${{outperformance >= 0 ? '#00ff88' : '#ff3366'}}">${{outperformance >= 0 ? '+' : ''}}${{outperformance.toFixed(1)}}%</span>)`,
                    font: {{ size: 16, color: '#fff' }}
                }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: '#888', family: 'Open Sans' }},
                dragmode: 'pan',
                xaxis: {{
                    range: [signalStartDate, lastDate],  // Match main chart initial view
                    gridcolor: 'rgba(100,100,100,0.2)',
                    tickfont: {{ size: 10 }},
                    showspikes: true,
                    spikemode: 'toaxis+across',
                    spikethickness: 0.1,
                    spikecolor: 'rgba(80,80,80,0.08)',
                    spikedash: 'dot',
                    spikesnap: 'cursor'
                }},
                yaxis: {{
                    title: 'Portfolio Value ($)',
                    gridcolor: 'rgba(100,100,100,0.2)',
                    tickfont: {{ size: 10 }},
                    showspikes: true,
                    spikemode: 'toaxis+across',
                    spikethickness: 0.1,
                    spikecolor: 'rgba(80,80,80,0.08)',
                    spikedash: 'dot',
                    spikesnap: 'cursor'
                }},
                showlegend: true,
                legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0.5)' }},
                margin: {{ t: 50, b: 40, l: 60, r: 80 }},
                hovermode: 'x unified'
            }};
            
            const equityConfig = {{
                responsive: true,
                scrollZoom: true,
                displayModeBar: false,
                displaylogo: false
            }};
            
            Plotly.newPlot('equity-chart', equityTraces, equityLayout, equityConfig);
            
            // Sync equity chart zoom with main chart
            document.getElementById('snake-chart').on('plotly_relayout', function(eventdata) {{
                if (eventdata['xaxis.range[0]'] !== undefined) {{
                    Plotly.relayout('equity-chart', {{
                        'xaxis.range': [eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]']]
                    }});
                }} else if (eventdata['xaxis.autorange']) {{
                    Plotly.relayout('equity-chart', {{'xaxis.autorange': true}});
                }}
            }});
            
            // ==================== P&L METRICS (SIGNAL-FOLLOWING STRATEGY) ====================
            // Calculate trade-level metrics: Enter on signal change, exit on signal flip/neutral
            // This matches the equity curve calculation and analyze_strategies.py validation
            
            // First, build filtered signals array with all filters applied
            const filteredSignals = [];
            const signalStartDateObj = new Date(data.signal_start_date);
            
            for (let i = 0; i < data.dates.length; i++) {{
                let signal = signals[i];  // Already has RSI filter from getCurrentSignals
                const currentDate = new Date(data.dates[i]);
                
                // Apply EMA and Ichimoku filters only within signal period
                if (currentDate >= signalStartDateObj) {{
                    const signalIsBullish = (signal === 'BULLISH');
                    const signalIsBearish = (signal === 'BEARISH');
                    
                    // Apply EMA filter if enabled
                    if (emaPeriod > 0 && emaValues && emaValues[i] !== null && !isNaN(emaValues[i])) {{
                        const priceAboveEMA = data.prices[i] > emaValues[i];
                        
                        if (signalIsBullish && !priceAboveEMA) {{
                            signal = 'NEUTRAL';
                        }} else if (signalIsBearish && priceAboveEMA) {{
                            signal = 'NEUTRAL';
                        }}
                    }}
                    
                    // Apply Ichimoku cloud color filter if enabled
                    if (showIchimoku && ichimokuFilter && ichimokuFilter.cloudBullish[i] !== null) {{
                        const cloudIsBullish = ichimokuFilter.cloudBullish[i];
                        
                        if (signal === 'BULLISH' && !cloudIsBullish) {{
                            signal = 'NEUTRAL';
                        }} else if (signal === 'BEARISH' && cloudIsBullish) {{
                            signal = 'NEUTRAL';
                        }}
                    }}
                }}
                
                filteredSignals.push(signal);
            }}
            
            // SIGNAL-FOLLOWING STRATEGY: Entry on signal change, exit on flip/neutral
            const signalFollowingTrades = [];
            let tradePosition = null;  // Renamed to avoid conflict with equity curve's currentPosition
            let entryIdx = null;
            let tradeEntryPrice = null;  // Renamed to avoid conflict
            let entrySignal = null;
            
            for (let i = 0; i < data.dates.length; i++) {{
                const signal = filteredSignals[i];
                const price = data.prices[i];
                
                if (tradePosition === null) {{
                    // No position - look to enter on BULLISH or BEARISH
                    if (signal === 'BULLISH' || signal === 'BEARISH') {{
                        tradePosition = signal;
                        entryIdx = i;
                        tradeEntryPrice = price;
                        entrySignal = signal;
                    }}
                }} else {{
                    // Have position - check for exit (signal flip or turn NEUTRAL)
                    const shouldExit = (signal !== tradePosition);  // Includes NEUTRAL and opposite signal
                    
                    if (shouldExit) {{
                        // Calculate P&L
                        let pnl;
                        if (entrySignal === 'BULLISH') {{
                            pnl = ((price - tradeEntryPrice) / tradeEntryPrice) * 100;  // Long: profit if price goes up
                        }} else {{
                            pnl = ((tradeEntryPrice - price) / tradeEntryPrice) * 100;  // Short: profit if price goes down
                        }}
                        
                        const holdingDays = i - entryIdx;
                        
                        signalFollowingTrades.push({{
                            entryIdx: entryIdx,
                            exitIdx: i,
                            signal: entrySignal,
                            entryPrice: tradeEntryPrice,
                            exitPrice: price,
                            pnl: pnl,
                            holdingDays: holdingDays,
                            exitReason: signal === 'NEUTRAL' ? 'Neutral' : 'Signal Flip'
                        }});
                        
                        // Immediately enter new position if signal flipped (not just turned NEUTRAL)
                        if (signal === 'BULLISH' || signal === 'BEARISH') {{
                            tradePosition = signal;
                            entryIdx = i;
                            tradeEntryPrice = price;
                            entrySignal = signal;
                        }} else {{
                            tradePosition = null;
                            entryIdx = null;
                            tradeEntryPrice = null;
                            entrySignal = null;
                        }}
                    }}
                }}
            }}
            
            // Calculate trade-level metrics from Signal-Following trades
            const tradeResults = signalFollowingTrades.map(t => ({{ type: t.signal, return: t.pnl / 100 }}));
            
            // Trade-level returns (for quant calculations)
            const tradeReturns = signalFollowingTrades.map(t => t.pnl / 100);  // Convert to decimal
            
            // Total P&L from all trades
            const totalPnL = signalFollowingTrades.reduce((sum, t) => sum + t.pnl, 0);
            
            // Win rate
            const wins = tradeResults.filter(t => t.return > 0).length;
            const losses = tradeResults.filter(t => t.return < 0).length;
            const totalTrades = tradeResults.length;
            const winRate = totalTrades > 0 ? (wins / totalTrades) * 100 : 0;
            
            // Cache win rate for optimization hero display
            cacheCurrentWinRate(winRate);
            
            // Average win/loss (in % terms)
            const avgWin = wins > 0 ? signalFollowingTrades.filter(t => t.pnl > 0).reduce((a, b) => a + b.pnl, 0) / wins : 0;
            const avgLoss = losses > 0 ? Math.abs(signalFollowingTrades.filter(t => t.pnl < 0).reduce((a, b) => a + b.pnl, 0)) / losses : 0;
            
            // Average holding days
            const avgHoldDays = totalTrades > 0 ? signalFollowingTrades.reduce((sum, t) => sum + t.holdingDays, 0) / totalTrades : 0;
            
            // Profit factor (total gains / total losses)
            const grossProfit = signalFollowingTrades.filter(t => t.pnl > 0).reduce((a, b) => a + b.pnl, 0);
            const grossLoss = Math.abs(signalFollowingTrades.filter(t => t.pnl < 0).reduce((a, b) => a + b.pnl, 0));
            const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 999 : 0;
            
            // Max drawdown from equity curve
            let maxDrawdown = 0;
            let peak = startingCapital;
            for (const equity of ensembleEquity) {{
                if (equity > peak) peak = equity;
                const drawdown = (peak - equity) / peak * 100;
                if (drawdown > maxDrawdown) maxDrawdown = drawdown;
            }}
            
            // Sharpe ratio (trade-level, annualized by avg trades per year)
            const avgTradeReturn = tradeReturns.length > 0 ? tradeReturns.reduce((a, b) => a + b, 0) / tradeReturns.length : 0;
            const stdTradeReturn = tradeReturns.length > 1 
                ? Math.sqrt(tradeReturns.reduce((sum, r) => sum + Math.pow(r - avgTradeReturn, 2), 0) / (tradeReturns.length - 1))
                : 0.0001;
            const tradesPerYear = 252 / (avgHoldDays || 10);  // Estimate trades per year
            const sharpeRatio = stdTradeReturn > 0 ? (avgTradeReturn / stdTradeReturn) * Math.sqrt(tradesPerYear) : 0;
            
            // Best and worst trade
            const bestTrade = signalFollowingTrades.length > 0 ? Math.max(...signalFollowingTrades.map(t => t.pnl)) : 0;
            const worstTrade = signalFollowingTrades.length > 0 ? Math.min(...signalFollowingTrades.map(t => t.pnl)) : 0;
            
            // Annualized return
            const tradingDays = data.dates.length;
            const yearsTraded = tradingDays / 252;
            const annualizedReturn = yearsTraded > 0 ? (Math.pow(ensembleEquity[ensembleEquity.length-1] / startingCapital, 1/yearsTraded) - 1) * 100 : 0;
            
            // Calmar ratio (annualized return / max drawdown)
            const calmarRatio = maxDrawdown > 0 ? annualizedReturn / maxDrawdown : 0;
            
            // Consecutive wins/losses
            let maxConsecWins = 0, maxConsecLosses = 0, consecWins = 0, consecLosses = 0;
            for (const trade of signalFollowingTrades) {{
                if (trade.pnl > 0) {{
                    consecWins++;
                    consecLosses = 0;
                    if (consecWins > maxConsecWins) maxConsecWins = consecWins;
                }} else {{
                    consecLosses++;
                    consecWins = 0;
                    if (consecLosses > maxConsecLosses) maxConsecLosses = consecLosses;
                }}
            }}
            
            // ==================== ADVANCED QUANT CALCULATIONS (Trade-Level) ====================
            
            // Sortino Ratio (trade-level, only penalizes downside)
            const negativeTrades = tradeReturns.filter(r => r < 0);
            const downsideDeviation = negativeTrades.length > 0 
                ? Math.sqrt(negativeTrades.reduce((sum, r) => sum + Math.pow(r, 2), 0) / negativeTrades.length)
                : 0.0001;
            const sortinoRatio = downsideDeviation > 0 ? (avgTradeReturn / downsideDeviation) * Math.sqrt(tradesPerYear) : 0;
            
            // Payoff Ratio (Avg Win / Avg Loss)
            const payoffRatio = avgLoss > 0 ? avgWin / avgLoss : avgWin > 0 ? 999 : 0;
            
            // Expectancy (expected % per trade)
            const winProb = totalTrades > 0 ? wins / totalTrades : 0;
            const lossProb = totalTrades > 0 ? losses / totalTrades : 0;
            const expectancy = (winProb * avgWin) - (lossProb * avgLoss);
            
            // VaR (Value at Risk) at 95% confidence - per trade
            const sortedTradeReturns = [...tradeReturns].sort((a, b) => a - b);
            const var95Index = Math.floor(sortedTradeReturns.length * 0.05);
            const var95 = sortedTradeReturns.length > 0 ? sortedTradeReturns[Math.max(0, var95Index)] * 100 : 0;
            
            // CVaR / Expected Shortfall (average of losses beyond VaR)
            const tailReturns = sortedTradeReturns.slice(0, Math.max(1, var95Index + 1));
            const cvar = tailReturns.length > 0 ? (tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length) * 100 : 0;
            
            // Skewness of trade returns
            const skewness = tradeReturns.length > 2 && stdTradeReturn > 0
                ? (tradeReturns.reduce((sum, r) => sum + Math.pow((r - avgTradeReturn) / stdTradeReturn, 3), 0) / tradeReturns.length)
                : 0;
            
            // Kurtosis (fat tails)
            const kurtosis = tradeReturns.length > 3 && stdTradeReturn > 0
                ? (tradeReturns.reduce((sum, r) => sum + Math.pow((r - avgTradeReturn) / stdTradeReturn, 4), 0) / tradeReturns.length) - 3
                : 0;
            
            // Tail Ratio (95th percentile gain / 5th percentile loss) - trade-level
            const p95TIndex = Math.floor(sortedTradeReturns.length * 0.95);
            const p95 = sortedTradeReturns.length > 0 ? sortedTradeReturns[Math.min(p95TIndex, sortedTradeReturns.length - 1)] : 0;
            const p5 = sortedTradeReturns.length > 0 ? sortedTradeReturns[Math.max(0, var95Index)] : 0;
            const tailRatio = p5 < 0 ? Math.abs(p95 / p5) : p95 > 0 ? 999 : 0;
            
            // Gain-to-Pain Ratio - trade-level
            const sumTradeReturns = tradeReturns.reduce((a, b) => a + b, 0);
            const sumNegTradeReturns = Math.abs(negativeTrades.reduce((a, b) => a + b, 0));
            const gainToPain = sumNegTradeReturns > 0 ? sumTradeReturns / sumNegTradeReturns : sumTradeReturns > 0 ? 999 : 0;
            
            // Recovery Factor (Net Profit / Max Drawdown)
            const recoveryFactor = maxDrawdown > 0 ? ensembleReturn / maxDrawdown : ensembleReturn > 0 ? 999 : 0;
            
            // Kelly Criterion (optimal bet size) - trade-level
            // Kelly % = W - (1-W)/R where W = win rate, R = payoff ratio
            const kellyPercent = avgWin > 0 && avgLoss > 0 ? ((winProb * avgWin) - (lossProb * avgLoss)) / avgWin * 100 : 0;
            
            // Ulcer Index (pain from drawdowns)
            let sumSquaredDD = 0;
            peak = startingCapital;
            for (const equity of ensembleEquity) {{
                if (equity > peak) peak = equity;
                const dd = (peak - equity) / peak * 100;
                sumSquaredDD += dd * dd;
            }}
            const ulcerIndex = Math.sqrt(sumSquaredDD / ensembleEquity.length);
            
            // Average Drawdown
            let totalDD = 0, ddCount = 0;
            peak = startingCapital;
            for (const equity of ensembleEquity) {{
                if (equity > peak) peak = equity;
                const dd = (peak - equity) / peak * 100;
                if (dd > 0) {{
                    totalDD += dd;
                    ddCount++;
                }}
            }}
            const avgDrawdown = ddCount > 0 ? totalDD / ddCount : 0;
            
            // Max Drawdown Duration (in days)
            let maxDDDuration = 0, currentDDDuration = 0;
            peak = startingCapital;
            for (const equity of ensembleEquity) {{
                if (equity >= peak) {{
                    peak = equity;
                    currentDDDuration = 0;
                }} else {{
                    currentDDDuration++;
                    if (currentDDDuration > maxDDDuration) maxDDDuration = currentDDDuration;
                }}
            }}
            
            // Omega Ratio (gains / losses) - trade-level
            const tradeGains = tradeReturns.filter(r => r > 0).reduce((a, b) => a + b, 0);
            const tradeLosses = Math.abs(tradeReturns.filter(r => r < 0).reduce((a, b) => a + b, 0));
            const omegaRatio = tradeLosses > 0 ? tradeGains / tradeLosses : tradeGains > 0 ? 999 : 0;
            
            // Sterling Ratio (CAGR / Avg Drawdown)
            const sterlingRatio = avgDrawdown > 0 ? annualizedReturn / avgDrawdown : 0;
            
            // Common Sense Ratio (Tail Ratio × Profit Factor)
            const commonSenseRatio = tailRatio * profitFactor;
            
            // % Winning Trades
            const pctPositiveDays = winRate;  // This is now win rate of actual trades
            
            // ==================== CALCULATE AVG DIRECTIONAL ACCURACY ====================
            // DA = how often predictions match actual price direction
            // Use avg_da_raw or avg_da_optimized based on current view
            const config = window.OPTIMAL_CONFIGS ? window.OPTIMAL_CONFIGS[currentAsset] : null;
            const avgDA = showingRawView 
                ? (config && config.avg_da_raw ? config.avg_da_raw : (data.base_accuracy || 65))
                : (config && config.avg_da_optimized ? config.avg_da_optimized : (config && config.avg_da_raw ? config.avg_da_raw : 65));
            
            // ==================== STORE PERFORMANCE METRICS FOR PERFORMANCE BOX ====================
            // ALWAYS use pre-calculated config values (calculated in pipeline, not on-the-fly)
            // This ensures consistency and accuracy
            
            // Debug: Log what config we have
            if (config) {{
                console.log(`[CONFIG DEBUG] ${{currentAsset}}: baseline_equity=${{config.baseline_equity}}, optimized_equity=${{config.optimized_equity}}, baseline_sharpe=${{config.baseline_sharpe}}, sharpe_optimized=${{config.sharpe_optimized}}`);
            }} else {{
                console.warn(`[CONFIG DEBUG] ${{currentAsset}}: NO CONFIG FOUND! window.OPTIMAL_CONFIGS=${{window.OPTIMAL_CONFIGS ? 'exists' : 'missing'}}`);
            }}
            
            let metricsToStore;
            
            if (showingRawView) {{
                // RAW: Use pre-calculated baseline values from config
                const useConfig = config && (config.baseline_equity !== undefined && config.baseline_equity !== null);
                metricsToStore = {{
                    totalReturn: useConfig ? config.baseline_equity : totalPnL,
                    sharpe: (config && config.baseline_sharpe !== undefined && config.baseline_sharpe !== null) ? config.baseline_sharpe : sharpeRatio,
                    avgDA: avgDA,  // Already uses config.avg_da_raw
                    maxDD: (config && config.baseline_max_drawdown !== undefined && config.baseline_max_drawdown !== null) ? config.baseline_max_drawdown : -maxDrawdown,
                    nTrades: (config && config.baseline_total_trades !== undefined && config.baseline_total_trades !== null) ? config.baseline_total_trades : totalTrades,
                    winRate: (config && config.baseline_win_rate !== undefined && config.baseline_win_rate !== null) ? config.baseline_win_rate : winRate,
                    profitFactor: (config && config.baseline_profit_factor !== undefined && config.baseline_profit_factor !== null) ? config.baseline_profit_factor : profitFactor
                }};
                cachedPerformanceMetrics.raw = metricsToStore;
                console.log(`[METRICS RAW] ${{currentAsset}}: return=${{metricsToStore.totalReturn}}% (from config=${{useConfig}}), sharpe=${{metricsToStore.sharpe}}`);
            }} else {{
                // OPTIMIZED: Use pre-calculated optimized values from config
                const useConfig = config && (config.optimized_equity !== undefined && config.optimized_equity !== null);
                metricsToStore = {{
                    totalReturn: useConfig ? config.optimized_equity : totalPnL,
                    sharpe: (config && config.sharpe_optimized !== undefined && config.sharpe_optimized !== null) ? config.sharpe_optimized : sharpeRatio,
                    avgDA: avgDA,  // Already uses config.avg_da_optimized
                    maxDD: (config && config.max_drawdown_optimized !== undefined && config.max_drawdown_optimized !== null) ? config.max_drawdown_optimized : -maxDrawdown,
                    nTrades: (config && config.total_trades !== undefined && config.total_trades !== null) ? config.total_trades : totalTrades,
                    winRate: (config && config.win_rate !== undefined && config.win_rate !== null) ? config.win_rate : winRate,
                    profitFactor: (config && config.profit_factor !== undefined && config.profit_factor !== null) ? config.profit_factor : profitFactor
                }};
                cachedPerformanceMetrics.optimized = metricsToStore;
                console.log(`[METRICS OPTIMIZED] ${{currentAsset}}: return=${{metricsToStore.totalReturn}}% (from config=${{useConfig}}), sharpe=${{metricsToStore.sharpe}}`);
            }}
            
            // Update performance box display
            updatePerformanceBox();
            
            // Performance metrics panel removed - key metrics shown in hero box at top
            
            // ==================== ENSEMBLE HEALTH CHECK ====================
            // Calculate health score and generate recommendations
            
            // Use ENABLED horizons (for dynamic toggling)
            const numHorizons = enabledHorizons.length;
            const availableHorizons = enabledHorizons;
            const allHorizons = data.horizons;  // All available horizons for this asset
            const disabledHorizons = allHorizons.filter(h => !enabledHorizons.includes(h));
            const missingHorizons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].filter(h => !allHorizons.includes(h));
            
            // Use dynamically calculated accuracy from current signals
            const accMetricsHealth = calculateAccuracy(data, signals);
            let accuracy = accMetricsHealth.accuracy;
            let edge = accMetricsHealth.edge;
            
            // OVERRIDE: When optimal config is applied, use pre-computed optimal values
            let useOptimalOverride = false;
            if (isOptimalConfigApplied && window.OPTIMAL_CONFIGS && window.OPTIMAL_CONFIGS[currentAsset]) {{
                const optConfig = window.OPTIMAL_CONFIGS[currentAsset];
                if (optConfig.avg_accuracy > 0) {{
                    accuracy = optConfig.avg_accuracy;
                    edge = optConfig.avg_accuracy - 50;  // Edge = accuracy - 50%
                    useOptimalOverride = true;
                    console.log(`Using optimal override: accuracy=${{accuracy}}%, edge=${{edge}}%`);
                }}
            }}
            
            // ===== NEW: Model Count Analysis (only for ENABLED horizons) =====
            const modelCounts = data.model_counts || {{}};
            const singleModelHorizons = [];
            const lowModelHorizons = [];
            const goodModelHorizons = [];
            
            enabledHorizons.forEach(h => {{
                const count = modelCounts[String(h)] || 0;
                if (count === 1) {{
                    singleModelHorizons.push({{horizon: h, count: count}});
                }} else if (count > 1 && count < 10) {{
                    lowModelHorizons.push({{horizon: h, count: count}});
                }} else if (count >= 10) {{
                    goodModelHorizons.push({{horizon: h, count: count}});
                }}
            }});
            
            // ===== NEW: Bullish/Bearish Bias Analysis (use recalculated stats if horizons toggled) =====
            const currentStats = getCurrentStats(data);
            const bullishCount = currentStats.bullish || 0;
            const bearishCount = currentStats.bearish || 0;
            const totalSignals = bullishCount + bearishCount;
            
            let biasRatio = 0;
            let biasType = 'balanced';
            let biasSeverity = 'none';
            
            if (totalSignals > 0) {{
                biasRatio = totalSignals > 0 ? Math.max(bullishCount, bearishCount) / Math.min(bullishCount || 1, bearishCount || 1) : 1;
                if (bullishCount > bearishCount * 5) {{
                    biasType = 'extreme_bullish';
                    biasSeverity = 'critical';
                }} else if (bearishCount > bullishCount * 5) {{
                    biasType = 'extreme_bearish';
                    biasSeverity = 'critical';
                }} else if (bullishCount > bearishCount * 2) {{
                    biasType = 'bullish';
                    biasSeverity = 'warning';
                }} else if (bearishCount > bullishCount * 2) {{
                    biasType = 'bearish';
                    biasSeverity = 'warning';
                }}
            }}
            
            // Health scoring (0-100)
            let healthScore = 0;
            const issues = [];
            const successes = [];
            
            // Accuracy score (max 30 points)
            if (accuracy >= 60) {{ healthScore += 30; successes.push('excellent_accuracy'); }}
            else if (accuracy >= 55) {{ healthScore += 25; successes.push('good_accuracy'); }}
            else if (accuracy >= 52) {{ healthScore += 15; }}
            else if (accuracy >= 50) {{ healthScore += 5; issues.push('low_accuracy'); }}
            else {{ issues.push('critical_accuracy'); }}
            
            // Edge score (max 20 points)
            if (edge >= 10) {{ healthScore += 20; successes.push('strong_edge'); }}
            else if (edge >= 5) {{ healthScore += 15; }}
            else if (edge >= 2) {{ healthScore += 10; }}
            else if (edge >= 0) {{ healthScore += 5; issues.push('weak_edge'); }}
            else {{ issues.push('negative_edge'); }}
            
            // Horizon coverage score (max 20 points)
            if (numHorizons >= 8) {{ healthScore += 20; successes.push('full_horizons'); }}
            else if (numHorizons >= 5) {{ healthScore += 15; }}
            else if (numHorizons >= 3) {{ healthScore += 10; issues.push('sparse_horizons'); }}
            else {{ healthScore += 5; issues.push('critical_horizons'); }}
            
            // NEW: Model diversity score (deduct points for issues)
            if (singleModelHorizons.length > 0) {{
                healthScore -= singleModelHorizons.length * 5;
                issues.push('single_model_horizons');
            }}
            
            // NEW: Bias penalty (deduct points for extreme bias)
            if (biasSeverity === 'critical') {{
                healthScore -= 15;
                issues.push('extreme_bias');
            }} else if (biasSeverity === 'warning') {{
                healthScore -= 5;
                issues.push('moderate_bias');
            }}
            
            // Sharpe ratio score (max 15 points)
            if (sharpeRatio >= 2) {{ healthScore += 15; successes.push('excellent_sharpe'); }}
            else if (sharpeRatio >= 1) {{ healthScore += 12; }}
            else if (sharpeRatio >= 0.5) {{ healthScore += 8; }}
            else if (sharpeRatio >= 0) {{ healthScore += 4; issues.push('low_sharpe'); }}
            else {{ issues.push('negative_sharpe'); }}
            
            // Profit factor score (max 15 points)
            if (profitFactor >= 2) {{ healthScore += 15; successes.push('strong_pf'); }}
            else if (profitFactor >= 1.5) {{ healthScore += 12; }}
            else if (profitFactor >= 1.2) {{ healthScore += 8; }}
            else if (profitFactor >= 1) {{ healthScore += 4; issues.push('weak_pf'); }}
            else {{ issues.push('losing_pf'); }}
            
            // OVERRIDE: When optimal config is applied, use pre-computed health score
            if (useOptimalOverride && window.OPTIMAL_CONFIGS[currentAsset].health_score > 0) {{
                healthScore = window.OPTIMAL_CONFIGS[currentAsset].health_score;
                // Clear issues since we're using optimal config
                issues.length = 0;
                successes.push('optimal_applied');
                console.log(`Using optimal health score: ${{healthScore}}/100`);
            }}
            
            // Determine health status
            let healthStatus, healthClass, alertClass;
            if (healthScore >= 80) {{
                healthStatus = 'Excellent';
                healthClass = 'healthy';
                alertClass = 'healthy';
            }} else if (healthScore >= 60) {{
                healthStatus = 'Good';
                healthClass = 'healthy';
                alertClass = 'healthy';
            }} else if (healthScore >= 40) {{
                healthStatus = 'Fair';
                healthClass = 'warning';
                alertClass = 'warning';
            }} else {{
                healthStatus = 'Action Needed';
                healthClass = 'critical';
                alertClass = 'critical';
            }}
            
            // Count urgent issues
            const urgentCount = issues.filter(i => ['critical_accuracy', 'negative_edge', 'critical_horizons'].includes(i)).length;
            
            // Model building URL
            const buildModelsUrl = 'https://quantumcloud.ai/dash/create-model';
            const buildBtn = `<a href="${{buildModelsUrl}}" target="_blank" class="build-models-btn">Build Models →</a>`;
            
            // Generate recommendations HTML with simplified messaging
            let recsHtml = '';
            
            // Timeframe tags for horizon recommendations
            const timeframeTags = (missing) => {{
                if (missing.length === 0) return '';
                return `<div class="timeframe-tags">
                    ${{missing.map(h => `<span class="timeframe-tag missing">D+${{h}}</span>`).join('')}}
                </div>`;
            }};
            
            const availableTags = () => {{
                return `<div class="timeframe-tags">
                    ${{availableHorizons.map(h => `<span class="timeframe-tag available">D+${{h}}</span>`).join('')}}
                </div>`;
            }};
            
            // Critical issues first - simplified messaging
            if (issues.includes('critical_accuracy') || issues.includes('negative_edge')) {{
                recsHtml += `
                    <div class="rec-item urgent">
                        <span class="rec-icon">🚨</span>
                        <div class="rec-content">
                            <div class="rec-title">Build More Models Now</div>
                            <div class="rec-desc">Accuracy is only ${{accuracy.toFixed(0)}}%. Add more prediction models to strengthen the ensemble. Aim for at least 10 diverse models.</div>
                            ${{buildBtn}}
                        </div>
                    </div>`;
            }}
            
            // NEW: Single model horizon warning
            if (issues.includes('single_model_horizons')) {{
                const singleHorizonTags = singleModelHorizons.map(h => 
                    `<span class="timeframe-tag missing">D+${{h.horizon}} <small>(1 model)</small></span>`
                ).join('');
                recsHtml += `
                    <div class="rec-item urgent">
                        <span class="rec-icon">⚠️</span>
                        <div class="rec-content">
                            <div class="rec-title">Single Model Per Horizon - No Ensemble Effect</div>
                            <div class="rec-desc">These timeframes only have 1 model each. With no diversity, there's no ensemble "magic" - you're using raw model output. Build more models for these horizons:</div>
                            <div class="timeframe-tags">${{singleHorizonTags}}</div>
                            ${{buildBtn}}
                        </div>
                    </div>`;
            }}
            
            // NEW: Bias warning
            if (issues.includes('extreme_bias')) {{
                const dominantDir = bullishCount > bearishCount ? 'BULLISH' : 'BEARISH';
                const missingDir = dominantDir === 'BULLISH' ? 'bearish' : 'bullish';
                recsHtml += `
                    <div class="rec-item urgent">
                        <span class="rec-icon">⚖️</span>
                        <div class="rec-content">
                            <div class="rec-title">Extreme Signal Bias Detected</div>
                            <div class="rec-desc">
                                <strong>${{bullishCount}} bullish vs ${{bearishCount}} bearish signals</strong> (${{biasRatio.toFixed(0)}}:1 ratio)<br>
                                Your models almost always predict ${{dominantDir.toLowerCase()}} moves. When the market goes the opposite direction, the ensemble will fail.<br><br>
                                <strong>Fix:</strong> Build contrarian models that specifically predict ${{missingDir}} movements, or use different feature sets that capture downside scenarios.
                            </div>
                            ${{buildBtn}}
                        </div>
                    </div>`;
            }} else if (issues.includes('moderate_bias')) {{
                const dominantDir = bullishCount > bearishCount ? 'BULLISH' : 'BEARISH';
                recsHtml += `
                    <div class="rec-item improve">
                        <span class="rec-icon">⚖️</span>
                        <div class="rec-content">
                            <div class="rec-title">Signal Bias Detected</div>
                            <div class="rec-desc">
                                <strong>${{bullishCount}} bullish vs ${{bearishCount}} bearish signals</strong><br>
                                Your models tend to favor ${{dominantDir.toLowerCase()}} predictions. Consider adding models with different perspectives for better balance.
                            </div>
                            ${{buildBtn}}
                        </div>
                    </div>`;
            }}
            
            if (issues.includes('critical_horizons') || issues.includes('sparse_horizons')) {{
                recsHtml += `
                    <div class="rec-item ${{issues.includes('critical_horizons') ? 'urgent' : 'improve'}}">
                        <span class="rec-icon">📊</span>
                        <div class="rec-content">
                            <div class="rec-title">Add Missing Timeframes</div>
                            <div class="rec-desc">Only ${{numHorizons}}/10 forecast periods covered. Build models for these missing timeframes:</div>
                            ${{timeframeTags(missingHorizons)}}
                            ${{buildBtn}}
                        </div>
                    </div>`;
            }}
            
            // Improvement suggestions - simplified
            if (issues.includes('low_accuracy') && !issues.includes('critical_accuracy')) {{
                recsHtml += `
                    <div class="rec-item improve">
                        <span class="rec-icon">🎯</span>
                        <div class="rec-content">
                            <div class="rec-title">Boost Accuracy</div>
                            <div class="rec-desc">Accuracy of ${{accuracy.toFixed(0)}}% can be improved. Try adding 5-10 new models with different approaches.</div>
                            ${{buildBtn}}
                        </div>
                    </div>`;
            }}
            
            if (issues.includes('weak_edge') && !issues.includes('negative_edge')) {{
                recsHtml += `
                    <div class="rec-item improve">
                        <span class="rec-icon">📈</span>
                        <div class="rec-content">
                            <div class="rec-title">Strengthen Your Edge</div>
                            <div class="rec-desc">Edge of ${{edge.toFixed(1)}}% is thin. More models will help increase your advantage.</div>
                            ${{buildBtn}}
                        </div>
                    </div>`;
            }}
            
            // Success highlights - simplified
            if (successes.includes('excellent_accuracy') || successes.includes('good_accuracy')) {{
                recsHtml += `
                    <div class="rec-item success">
                        <span class="rec-icon">✅</span>
                        <div class="rec-content">
                            <div class="rec-title">Strong Performance</div>
                            <div class="rec-desc">Your ensemble is generating quality signals with ${{accuracy.toFixed(0)}}% accuracy!</div>
                        </div>
                    </div>`;
            }}
            
            if (successes.includes('full_horizons')) {{
                recsHtml += `
                    <div class="rec-item success">
                        <span class="rec-icon">✅</span>
                        <div class="rec-content">
                            <div class="rec-title">Great Timeframe Coverage</div>
                            <div class="rec-desc">All ${{numHorizons}} forecast periods are covered:</div>
                            ${{availableTags()}}
                        </div>
                    </div>`;
            }}
            
            // NEW: Horizon model count breakdown with TOGGLE functionality
            // Use ALL horizons for display (not just enabled ones)
            const allAssetHorizons = data.horizons;
            if (Object.keys(modelCounts).length > 0) {{
                const horizonBreakdown = allAssetHorizons.map(h => {{
                    const count = modelCounts[String(h)] || 0;
                    const isEnabled = enabledHorizons.includes(h);
                    let statusClass = 'good';
                    let statusIcon = '✓';
                    if (count === 0) {{
                        statusClass = 'missing';
                        statusIcon = '✗';
                    }} else if (count === 1) {{
                        statusClass = 'critical';
                        statusIcon = '!';
                    }} else if (count < 10) {{
                        statusClass = 'warning';
                        statusIcon = '~';
                    }}
                    
                    // Add checkbox for toggling
                    const checkedAttr = isEnabled ? 'checked' : '';
                    const disabledStyle = isEnabled ? '' : 'opacity: 0.4;';
                    
                    // Add build button for single-model horizons
                    const buildBtnMini = (count === 1) ? 
                        `<a href="${{buildModelsUrl}}" target="_blank" class="horizon-build-btn" title="Build more models for D+${{h}}">+</a>` : '';
                    
                    return `<div class="horizon-model-item ${{statusClass}}" style="${{disabledStyle}}">
                        <input type="checkbox" id="horizon-toggle-${{h}}" ${{checkedAttr}} 
                            onchange="toggleHorizon(${{h}})" 
                            style="width: 14px; height: 14px; cursor: pointer; accent-color: var(--accent-green);">
                        <span class="horizon-label">D+${{h}}</span>
                        <span class="horizon-count">${{count}}</span>
                        <span class="horizon-status">${{statusIcon}}</span>
                        ${{buildBtnMini}}
                    </div>`;
                }}).join('');
                
                // Calculate health delta if any horizons are disabled
                const disabledCount = availableHorizons.length - enabledHorizons.length;
                const healthDeltaHtml = disabledCount > 0 ? 
                    `<div style="margin-top: 10px; padding: 8px 12px; background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3); border-radius: 8px; font-size: 11px;">
                        <strong style="color: var(--accent-blue);">🔬 What-If Mode:</strong> 
                        <span style="color: var(--text-secondary);">${{disabledCount}} horizon(s) disabled. Metrics recalculated with remaining ${{enabledHorizons.length}} horizons.</span>
                        <button onclick="resetHorizons()" style="margin-left: 10px; padding: 4px 8px; background: var(--bg-tertiary); border: 1px solid var(--border-color); border-radius: 4px; color: var(--text-primary); cursor: pointer; font-size: 10px;">Reset All</button>
                    </div>` : '';
                
                recsHtml += `
                    <div class="rec-item info" style="margin-top: 15px;">
                        <span class="rec-icon">📋</span>
                        <div class="rec-content">
                            <div class="rec-title">Model Count by Horizon <span style="font-size: 11px; color: var(--accent-blue); font-weight: normal;">(click to toggle)</span></div>
                            <div class="rec-desc" style="font-size: 11px; color: #888; margin-bottom: 8px;">
                                ✓ = 10+ models (strong) | ~ = 2-9 models (ok) | ! = 1 model (no ensemble) | <span style="color: var(--accent-blue);">Uncheck to disable a horizon</span>
                            </div>
                            <div class="horizon-model-grid">
                                ${{horizonBreakdown}}
                            </div>
                            ${{healthDeltaHtml}}
                        </div>
                    </div>`;
            }}
            
            // If no recommendations, show all clear
            if (recsHtml === '') {{
                recsHtml = `
                    <div class="rec-item success">
                        <span class="rec-icon">🎉</span>
                        <div class="rec-content">
                            <div class="rec-title">All Systems Go!</div>
                            <div class="rec-desc">Your ensemble is performing well. Keep monitoring and consider adding models to stay ahead.</div>
                        </div>
                    </div>`;
            }}
            
            // Render collapsible health check
            const scoreClass = healthScore >= 80 ? 'excellent' : (healthScore >= 60 ? 'good' : (healthScore >= 40 ? 'fair' : 'poor'));
            const healthCheckDiv = document.getElementById('healthCheck');
            // Preserve expanded state when re-rendering (don't collapse if already open)
            const wasExpanded = healthCheckDiv.classList.contains('expanded');
            healthCheckDiv.className = `health-check ${{healthClass}}${{wasExpanded ? ' expanded' : ''}}`;
            healthCheckDiv.innerHTML = `
                <div class="health-header" onclick="toggleHealthCheck()">
                    <div class="health-header-left">
                        <div class="alert-indicator ${{alertClass}}">${{urgentCount > 0 ? urgentCount : ''}}</div>
                        <div class="health-summary">
                            <span class="health-label">Ensemble Health</span>
                            <span class="health-badge ${{scoreClass}}">${{healthStatus}}</span>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <span class="health-score-mini">${{healthScore}}/100</span>
                        <span class="expand-icon">▼</span>
                    </div>
                </div>
                <div class="health-body">
                    <div class="health-content">
                        <div class="score-row">
                            <a href="${{buildModelsUrl}}" target="_blank" class="score-circle ${{scoreClass}}" title="Click to build more models">${{healthScore}}</a>
                            <div class="score-details">
                                <div class="score-label">Overall Health Score</div>
                                <div class="score-text">${{numHorizons}} timeframes • ${{edge >= 0 ? '+' : ''}}${{edge.toFixed(1)}}% edge</div>
                                <div class="score-text" style="margin-top: 4px; font-size: 11px;">
                                    📊 ${{bullishCount}} bullish / ${{bearishCount}} bearish signals
                                    ${{biasSeverity !== 'none' ? `<span style="color: ${{biasSeverity === 'critical' ? 'var(--accent-red)' : 'var(--accent-gold)'}};">• ${{biasType.replace('_', ' ').toUpperCase()}} BIAS</span>` : ''}}
                                    ${{singleModelHorizons.length > 0 ? `<span style="color: var(--accent-red);">• ${{singleModelHorizons.length}} single-model horizon${{singleModelHorizons.length > 1 ? 's' : ''}}</span>` : ''}}
                                </div>
                            </div>
                        </div>
                        <div id="optimalConfigSection" style="margin: 15px 0; padding: 12px; background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3); border-radius: 8px;">
                            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <span style="font-size: 20px;">🎯</span>
                                    <div>
                                        <div style="font-weight: 600; color: #00d4ff;">Optimal Configuration</div>
                                        <div id="optimalConfigDetails" style="font-size: 11px; color: var(--text-secondary); margin-top: 2px;">Loading...</div>
                                    </div>
                                </div>
                                <button id="autoOptimizeBtn" onclick="applyOptimalConfig()" class="auto-optimize-btn-flash" style="padding: 8px 16px; background: linear-gradient(135deg, #FFD700, #FF8C00); border: none; border-radius: 6px; color: white; font-weight: 600; font-size: 12px; cursor: pointer;">
                                    ⚡ Auto Optimize Equity
                                </button>
                            </div>
                        </div>
                        <div class="recommendations-title">Recommendations</div>
                        ${{recsHtml}}
                    </div>
                </div>
            `;
            
            // Update the optimal config display after HTML is set
            setTimeout(updateOptimalConfigDisplay, 0);
            
            // Apply Trader Mode zoom if enabled (after chart is drawn)
            if (traderModeEnabled) {{
                setTimeout(applyTraderModeZoom, 100);
            }}
        }}
        
        // Toggle health check expand/collapse
        function toggleHealthCheck() {{
            const healthCheck = document.getElementById('healthCheck');
            healthCheck.classList.toggle('expanded');
        }}
        
        // Toggle P&L metrics expand/collapse
        // Update Price Targets Section
        function updatePriceTargets() {{
            const data = ASSET_DATA[currentAsset];
            const apiData = API_DATA[currentAsset];
            const section = document.getElementById('priceTargetsSection');
            
            if (!apiData || !apiData.forecasts || apiData.forecasts.length === 0) {{
                section.innerHTML = `
                    <div class="price-targets-header">
                        <div class="price-targets-title">🎯 Price Targets</div>
                    </div>
                    <p style="color: var(--text-secondary); text-align: center; padding: 20px;">
                        No forecast data available for price targets.
                    </p>
                `;
                return;
            }}
            
            // IMPORTANT: Use the RECALCULATED signal based on enabled horizons
            // This ensures Price Targets matches the confidence badge on the chart
            // The signal should come from lastLiveConfidence (which is calculated from enabled horizons)
            // If not available, recalculate from scratch using the same logic
            let signal = apiData.signal || 'NEUTRAL';
            
            // PRIORITY 1: Use lastLiveConfidence.signal (from chart badge calculation)
            if (lastLiveConfidence && lastLiveConfidence.signal) {{
                signal = lastLiveConfidence.signal;
            }} else {{
                // PRIORITY 2: Recalculate signal from current signals array
                const data_local = ASSET_DATA[currentAsset];
                const signals_local = getCurrentSignals(data_local);
                if (signals_local && signals_local.length > 0) {{
                    const lastSignal = signals_local[signals_local.length - 1];
                    if (lastSignal && lastSignal !== '') {{
                        signal = lastSignal;
                    }}
                }}
            }}
            
            console.log(`Price Targets signal for ${{currentAsset}}: ${{signal}} (lastLiveConfidence: ${{lastLiveConfidence ? lastLiveConfidence.signal : 'null'}}, apiData: ${{apiData.signal}})`);
            
            const confidence = apiData.confidence || 0;
            const allForecasts = apiData.forecasts;
            const viableHorizons = apiData.viable_horizons || [];
            
            // IMPORTANT: Filter forecasts to only use ENABLED horizons
            // This ensures Price Targets update when user clicks "Auto Optimize Equity"
            const forecasts = allForecasts.filter(f => {{
                return enabledHorizons.includes(f.horizon_days);
            }});
            
            // If no forecasts match enabled horizons, fall back to all
            const activeForecasts = forecasts.length > 0 ? forecasts : allForecasts;
            
            // Determine signal class
            const signalClass = signal === 'BULLISH' ? 'bullish' : (signal === 'BEARISH' ? 'bearish' : 'neutral');
            const signalIcon = signal === 'BULLISH' ? '▲' : (signal === 'BEARISH' ? '▼' : '◆');
            
            // For neutral signals, show range instead of targets
            if (signal === 'NEUTRAL' || signal === 'N/A') {{
                const prices = activeForecasts.map(f => f.predicted_price).filter(p => p > 0);
                const maxPrice = Math.max(...prices);
                const minPrice = Math.min(...prices);
                const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
                
                section.innerHTML = `
                    <div class="price-targets-header">
                        <div class="price-targets-title">🎯 Price Outlook</div>
                        <div class="price-targets-signal neutral">◆ NEUTRAL</div>
                    </div>
                    <div class="neutral-outlook">
                        <div class="neutral-outlook-icon">⚖️</div>
                        <div class="neutral-outlook-title">No Clear Directional Bias</div>
                        <div class="neutral-outlook-text">Forecasts show mixed signals across horizons</div>
                        <div class="neutral-range">
                            <div class="range-item">
                                <div class="range-label">Upside Risk</div>
                                <div class="range-value upside">$${{maxPrice.toFixed(2)}}</div>
                            </div>
                            <div class="range-item">
                                <div class="range-label">Expected</div>
                                <div class="range-value" style="color: var(--text-primary);">$${{avgPrice.toFixed(2)}}</div>
                            </div>
                            <div class="range-item">
                                <div class="range-label">Downside Risk</div>
                                <div class="range-value downside">$${{minPrice.toFixed(2)}}</div>
                            </div>
                        </div>
                    </div>
                    <div class="price-targets-disclaimer">
                        ⚠️ Recommendation: Wait for clearer signal or trade the range with tight stops.
                    </div>
                `;
                return;
            }}
            
            // Get forecast prices sorted appropriately - use FILTERED forecasts
            const allPrices = activeForecasts.map(f => f.predicted_price).filter(p => p > 0);
            const currentPriceRef = data.current_price || Math.max(...allPrices);
            
            // IMPORTANT: Filter prices based on signal direction
            // BULLISH = only prices ABOVE current (upside targets)
            // BEARISH = only prices BELOW current (downside targets)
            let prices;
            if (signal === 'BULLISH') {{
                prices = allPrices.filter(p => p > currentPriceRef);
                // If no upside targets, use all prices but sorted for closest to current
                if (prices.length === 0) prices = allPrices;
            }} else {{
                prices = allPrices.filter(p => p < currentPriceRef);
                // If no downside targets, use all prices
                if (prices.length === 0) prices = allPrices;
            }}
            
            const sortedPrices = [...prices].sort((a, b) => signal === 'BULLISH' ? a - b : b - a);
            
            // Select 3 targets from the price range
            let t1, t2, t3;
            if (sortedPrices.length >= 3) {{
                // Conservative (25th percentile), Base (50th), Extended (75th)
                const p25 = Math.floor(sortedPrices.length * 0.25);
                const p50 = Math.floor(sortedPrices.length * 0.5);
                const p75 = Math.floor(sortedPrices.length * 0.75);
                t1 = sortedPrices[p25];
                t2 = sortedPrices[p50];
                t3 = sortedPrices[p75];
            }} else if (sortedPrices.length === 2) {{
                t1 = sortedPrices[0];
                t2 = (sortedPrices[0] + sortedPrices[1]) / 2;
                t3 = sortedPrices[1];
            }} else {{
                t1 = t2 = t3 = sortedPrices[0] || 0;
            }}
            
            // Use the current price from ASSET_DATA as reference for % calculations
            const currentPrice = currentPriceRef;
            
            // Calculate percentage changes from CURRENT PRICE
            // For BULLISH: show positive % (upside potential)
            // For BEARISH: show negative % (downside risk)
            let pctT1 = ((t1 - currentPrice) / currentPrice * 100);
            let pctT2 = ((t2 - currentPrice) / currentPrice * 100);
            let pctT3 = ((t3 - currentPrice) / currentPrice * 100);
            
            // FIX: Bug #6 - Don't flip signs! Show actual forecast values
            // If forecasts disagree with signal, users need to see this contradiction
            // Previously: sign was force-flipped, hiding model disagreement - DANGEROUS
            let forecastDisagreement = false;
            if (signal === 'BULLISH' && pctT2 < 0) {{
                forecastDisagreement = true;  // Signal says up, but forecasts say down
            }}
            if (signal === 'BEARISH' && pctT2 > 0) {{
                forecastDisagreement = true;  // Signal says down, but forecasts say up
            }}
            
            // Max horizon from ENABLED horizons (not just viable)
            const maxHorizon = enabledHorizons.length > 0 ? Math.max(...enabledHorizons) : 10;
            
            // Build targets HTML
            const targetLabels = signal === 'BULLISH' 
                ? ['Conservative', 'Base Case', 'Extended']
                : ['Conservative', 'Base Case', 'Extended'];
            
            const changeClass = signal === 'BULLISH' ? 'positive' : 'negative';
            // Show proper sign: + for positive, - for negative (already in the number)
            const formatPct = (pct) => pct >= 0 ? `+${{pct.toFixed(1)}}%` : `${{pct.toFixed(1)}}%`;
            
            section.innerHTML = `
                ${{forecastDisagreement ? `<div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.5); border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 12px;">
                    <span style="color: #ef4444; font-weight: 600;">⚠️ Forecast Contradiction</span><br>
                    <span style="color: var(--text-secondary);">Model forecasts disagree with signal direction. Consider reducing position size or waiting for confirmation.</span>
                </div>` : ''}}
                <div class="price-targets-header">
                    <div class="price-targets-title">
                        🎯 Price Targets
                        <span class="info-icon">?
                            <div class="info-tooltip" style="width: 380px;">
                                <h4>📊 How to Interpret Price Targets</h4>
                                <p>These targets are derived from the <b>ensemble forecast consensus</b> across your enabled horizons. They represent potential price levels based on model predictions.</p>
                                
                                <div style="margin: 12px 0; padding: 10px; background: var(--bg-tertiary); border-radius: 6px;">
                                    <div style="margin-bottom: 8px;"><span style="color: var(--accent-green); font-weight: 600;">T1 - Conservative</span><br><span style="font-size: 11px; color: var(--text-secondary);">25th percentile of forecasts. Higher probability of being reached.</span></div>
                                    <div style="margin-bottom: 8px;"><span style="color: var(--accent-blue); font-weight: 600;">T2 - Base Case</span><br><span style="font-size: 11px; color: var(--text-secondary);">50th percentile (median). Most likely scenario.</span></div>
                                    <div><span style="color: var(--accent-gold); font-weight: 600;">T3 - Extended</span><br><span style="font-size: 11px; color: var(--text-secondary);">75th percentile. Optimistic target, lower probability.</span></div>
                                </div>
                                
                                <p><b>💡 How to use:</b></p>
                                <ul style="margin-left: 16px; margin-top: 6px;">
                                    <li>Consider <b>partial exits</b> at each target level</li>
                                    <li>T1 is safest for profit-taking</li>
                                    <li>T3 requires more patience and risk tolerance</li>
                                    <li>Timeframe shows max horizon - target may be hit sooner</li>
                                </ul>
                                
                                <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border-color); font-size: 11px; color: var(--text-secondary);">
                                    <b>⚡ Dynamic Updates:</b> Targets recalculate when you click "Auto Optimize Equity" to use only the best-performing horizons.
                                </div>
                            </div>
                        </span>
                    </div>
                    <div class="price-targets-signal ${{signalClass}}">${{signalIcon}} ${{signal}}</div>
                </div>
                
                <table class="price-targets-table">
                    <thead>
                        <tr>
                            <th>Target</th>
                            <th>Price</th>
                            <th>Move</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>
                                <div class="target-label">
                                    <div class="target-badge t1">T1</div>
                                    <span class="target-type">${{targetLabels[0]}}</span>
                                </div>
                            </td>
                            <td class="target-price">$${{t1.toFixed(2)}}</td>
                            <td class="target-change ${{pctT1 >= 0 ? 'positive' : 'negative'}}">${{formatPct(pctT1)}}</td>
                        </tr>
                        <tr>
                            <td>
                                <div class="target-label">
                                    <div class="target-badge t2">T2</div>
                                    <span class="target-type">${{targetLabels[1]}}</span>
                                </div>
                            </td>
                            <td class="target-price">$${{t2.toFixed(2)}}</td>
                            <td class="target-change ${{pctT2 >= 0 ? 'positive' : 'negative'}}">${{formatPct(pctT2)}}</td>
                        </tr>
                        <tr>
                            <td>
                                <div class="target-label">
                                    <div class="target-badge t3">T3</div>
                                    <span class="target-type">${{targetLabels[2]}}</span>
                                </div>
                            </td>
                            <td class="target-price">$${{t3.toFixed(2)}}</td>
                            <td class="target-change ${{pctT3 >= 0 ? 'positive' : 'negative'}}">${{formatPct(pctT3)}}</td>
                        </tr>
                    </tbody>
                </table>
                
                <div class="price-targets-meta">
                    <div class="meta-item">
                        <span class="meta-label">⏱️ Timeframe:</span>
                        <span class="meta-value">Within ${{maxHorizon}} days</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">🔬 Based on:</span>
                        <span class="meta-value">MDQ Ensemble (${{enabledHorizons.length}} horizons${{isOptimalConfigApplied ? ' - Optimized' : ''}})</span>
                    </div>
                </div>
                
                <div class="price-targets-disclaimer">
                    ⚠️ Targets are probabilistic estimates based on ensemble model consensus. Price may not reach any target, or may exceed. Consider partial exits at each level. Not financial advice.
                </div>
            `;
        }}
        
        // ==================== REPLAY FUNCTIONALITY ====================
        let replayPlaying = false;
        let replayInterval = null;
        let replayIndex = 0;
        let replaySpeed = 200;  // ms per frame
        let replayData = [];
        
        function openReplay() {{
            const data = ASSET_DATA[currentAsset];
            replayData = data.replay_data || [];
            
            if (replayData.length < 10) {{
                alert('Not enough data for replay');
                return;
            }}
            
            // Start from last 90 days or beginning
            replayIndex = data.replay_start_idx || 0;
            
            document.getElementById('replayModal').classList.add('active');
            document.getElementById('replayAssetName').textContent = currentAsset.replace('_', ' ');
            
            // Set slider range
            const slider = document.getElementById('replaySlider');
            slider.max = replayData.length - 1;
            slider.value = replayIndex;
            
            // Initial render
            renderReplayFrame();
        }}
        
        function closeReplay() {{
            stopReplay();
            document.getElementById('replayModal').classList.remove('active');
        }}
        
        function renderReplayFrame() {{
            if (replayIndex >= replayData.length) {{
                stopReplay();
                return;
            }}
            
            const data = ASSET_DATA[currentAsset];
            const currentFrame = replayData[replayIndex];
            
            // Build snake up to current frame
            const traces = [];
            
            for (let i = 0; i < replayIndex; i++) {{
                const curr = replayData[i];
                const next = replayData[i + 1];
                if (!next) continue;
                
                const signal = curr.signal;
                const strength = curr.strength;
                const width = 2 + (strength * 10);
                
                let color;
                if (signal === 'BULLISH') {{
                    color = '#00ff88';
                }} else if (signal === 'BEARISH') {{
                    color = '#ff3366';
                }} else {{
                    color = 'rgba(100, 100, 100, 0.5)';
                }}
                
                traces.push({{
                    x: [curr.date, next.date],
                    y: [curr.price, next.price],
                    mode: 'lines',
                    line: {{ color: color, width: width }},
                    showlegend: false,
                    hoverinfo: 'skip'
                }});
            }}
            
            // Add current position marker
            traces.push({{
                x: [currentFrame.date],
                y: [currentFrame.price],
                mode: 'markers',
                marker: {{ 
                    size: 15, 
                    color: currentFrame.signal === 'BULLISH' ? '#00ff88' : (currentFrame.signal === 'BEARISH' ? '#ff3366' : '#888'),
                    symbol: 'circle',
                    line: {{ color: 'white', width: 2 }}
                }},
                name: 'Current',
                showlegend: false
            }});
            
            const layout = {{
                title: {{
                    text: `<b>${{currentAsset.replace('_', ' ')}}</b> - Replay`,
                    font: {{ size: 18, color: '#fff' }}
                }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: '#888', family: 'Open Sans' }},
                xaxis: {{
                    gridcolor: 'rgba(100,100,100,0.2)',
                    tickfont: {{ size: 11 }}
                }},
                yaxis: {{
                    title: 'Price',
                    gridcolor: 'rgba(100,100,100,0.2)',
                    tickfont: {{ size: 11 }}
                }},
                showlegend: false,
                margin: {{ t: 50, b: 40, l: 60, r: 80 }}
            }};
            
            Plotly.react('replay-chart', traces, layout, {{ responsive: true, displayModeBar: false }});
            
            // Update slider
            document.getElementById('replaySlider').value = replayIndex;
            
            // Update stats
            const signalClass = currentFrame.signal === 'BULLISH' ? 'positive' : (currentFrame.signal === 'BEARISH' ? 'negative' : '');
            
            // Calculate confidence tier based on signal strength
            // net_prob ranges from -1 to +1 (vote proportion)
            // MUST match thresholds in calculate_signal_confidence.py
            const strength = Math.abs(currentFrame.net_prob);
            let confTier, confEmoji, confColor;
            if (strength >= 0.5) {{  // 50%+ = HIGH (strong consensus)
                confTier = 'HIGH';
                confEmoji = '🟢';
                confColor = '#00ff88';
            }} else if (strength >= 0.2) {{  // 20-50% = MEDIUM
                confTier = 'MED';
                confEmoji = '🟡';
                confColor = '#ffaa00';
            }} else {{  // <20% = LOW
                confTier = 'LOW';
                confEmoji = '🔴';
                confColor = '#ff3366';
            }}
            
            // Get historical accuracy for this tier
            const confStats = data.confidence_stats;
            let histAccuracy = '--';
            if (confStats && confStats.by_strength) {{
                const tierKey = confTier === 'HIGH' ? 'strong' : (confTier === 'MED' ? 'medium' : 'weak');
                if (confStats.by_strength[tierKey]) {{
                    histAccuracy = confStats.by_strength[tierKey].accuracy.toFixed(1) + '%';
                }}
            }}
            
            document.getElementById('replayStats').innerHTML = `
                <div class="replay-stat">
                    <div class="replay-stat-value">${{currentFrame.date}}</div>
                    <div class="replay-stat-label">Date</div>
                </div>
                <div class="replay-stat">
                    <div class="replay-stat-value">${{currentFrame.price.toFixed(2)}}</div>
                    <div class="replay-stat-label">Price</div>
                </div>
                <div class="replay-stat">
                    <div class="replay-stat-value ${{signalClass}}">${{currentFrame.signal}}</div>
                    <div class="replay-stat-label">Signal</div>
                </div>
                <div class="replay-stat">
                    <div class="replay-stat-value" style="color: ${{confColor}};">${{confEmoji}} ${{confTier}}</div>
                    <div class="replay-stat-label">Confidence</div>
                </div>
                <div class="replay-stat">
                    <div class="replay-stat-value" style="color: ${{confColor}};">${{histAccuracy}}</div>
                    <div class="replay-stat-label">Hist. Accuracy</div>
                </div>
                <div class="replay-stat">
                    <div class="replay-stat-value">${{replayIndex + 1}} / ${{replayData.length}}</div>
                    <div class="replay-stat-label">Frame</div>
                </div>
            `;
        }}
        
        function togglePlayReplay() {{
            if (replayPlaying) {{
                stopReplay();
            }} else {{
                startReplay();
            }}
        }}
        
        function startReplay() {{
            if (replayIndex >= replayData.length - 1) {{
                replayIndex = 0;  // Reset to beginning
            }}
            
            replayPlaying = true;
            document.getElementById('playPauseBtn').textContent = '⏸️ Pause';
            document.getElementById('playPauseBtn').classList.add('playing');
            
            replayInterval = setInterval(() => {{
                replayIndex++;
                if (replayIndex >= replayData.length) {{
                    stopReplay();
                    return;
                }}
                renderReplayFrame();
            }}, replaySpeed);
        }}
        
        function stopReplay() {{
            replayPlaying = false;
            document.getElementById('playPauseBtn').textContent = '▶️ Play';
            document.getElementById('playPauseBtn').classList.remove('playing');
            
            if (replayInterval) {{
                clearInterval(replayInterval);
                replayInterval = null;
            }}
        }}
        
        function stepReplay(delta) {{
            stopReplay();
            replayIndex = Math.max(0, Math.min(replayData.length - 1, replayIndex + delta));
            renderReplayFrame();
        }}
        
        function seekReplay(value) {{
            stopReplay();
            replayIndex = parseInt(value);
            renderReplayFrame();
        }}
        
        function setReplaySpeed(speed) {{
            replaySpeed = speed;
            
            // Update button states
            document.querySelectorAll('.speed-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // If playing, restart with new speed
            if (replayPlaying) {{
                stopReplay();
                startReplay();
            }}
        }}
        
        // Close modal on Escape key
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                closeReplay();
                closeTradeHistory();
            }}
            // Space to play/pause
            if (e.key === ' ' && document.getElementById('replayModal').classList.contains('active')) {{
                e.preventDefault();
                togglePlayReplay();
            }}
        }});
        
        // ==================== TRADE HISTORY MODAL ====================
        
        function openTradeHistory() {{
            const data = ASSET_DATA[currentAsset];
            const config = window.OPTIMAL_CONFIGS ? window.OPTIMAL_CONFIGS[currentAsset] : null;
            
            document.getElementById('historyModal').classList.add('active');
            document.getElementById('historyAssetName').textContent = `(${{currentAsset.replace('_', ' ')}})`;
            
            // Use PRE-CALCULATED trades from config (calculated in Python, matches equity curve)
            const mode = showingRawView ? 'RAW' : 'OPTIMIZED';
            let preCalculatedTrades = null;
            
            if (config) {{
                if (showingRawView && config.last_30_trades_raw) {{
                    preCalculatedTrades = config.last_30_trades_raw;
                    console.log(`[TRADE HISTORY] Using PRE-CALCULATED RAW trades: ${{preCalculatedTrades.length}} trades`);
                }} else if (!showingRawView && config.last_30_trades_optimized) {{
                    preCalculatedTrades = config.last_30_trades_optimized;
                    console.log(`[TRADE HISTORY] Using PRE-CALCULATED OPTIMIZED trades: ${{preCalculatedTrades.length}} trades`);
                }}
            }}
            
            if (!preCalculatedTrades) {{
                console.warn(`[TRADE HISTORY] No pre-calculated trades found for ${{mode}} mode. Falling back to on-the-fly calculation.`);
                // Fallback: calculate on the fly (old method)
                if (showingRawView) {{
                    enabledHorizons = [...(data.horizons || [])];
                }} else {{
                    if (config && config.viable_horizons && config.viable_horizons.length > 0) {{
                        const available = data.horizons || [];
                        const optimalHorizons = config.viable_horizons.filter(h => available.includes(h));
                        if (optimalHorizons.length > 0) {{
                            enabledHorizons = [...optimalHorizons];
                        }}
                    }}
                }}
                recalculateSignalsWithEnabledHorizons();
                const signals = getFullyFilteredSignals(data);
                renderTradeHistory(data, signals);
            }} else {{
                // Use pre-calculated trades (matches Python calculation)
                renderTradeHistoryFromPreCalculated(preCalculatedTrades);
            }}
        }}
        
        function renderTradeHistory(data, signals) {{
            // ================ SIGNAL-FOLLOWING STRATEGY ================
            // This is the PROVEN strategy that matches the Equity Curve performance
            // Enter on signal change, exit when signal flips or turns neutral
            
            // Debug: Log signal distribution
            const bullishCount = signals.filter(s => s === 'BULLISH').length;
            const bearishCount = signals.filter(s => s === 'BEARISH').length;
            const neutralCount = signals.filter(s => s === 'NEUTRAL').length;
            console.log(`[TRADE HISTORY] ${{currentAsset}}: signals BULLISH=${{bullishCount}}, BEARISH=${{bearishCount}}, NEUTRAL=${{neutralCount}}`);
            
            const allTrades = [];
            let currentPosition = null;
            
            // Scan through ALL signals to find position changes (collect ALL trades first)
            for (let i = 1; i < signals.length; i++) {{
                const prevSignal = signals[i-1];
                const currSignal = signals[i];
                
                // Check for signal change
                if (prevSignal === currSignal) continue;
                
                // CLOSE existing position
                if (currentPosition) {{
                    const exitPrice = data.prices[i];
                    const exitDate = data.dates[i];
                    if (exitPrice && !isNaN(exitPrice)) {{
                        let pnl;
                        if (currentPosition.type === 'BULLISH') {{
                            pnl = ((exitPrice - currentPosition.entryPrice) / currentPosition.entryPrice) * 100;
                        }} else {{
                            pnl = ((currentPosition.entryPrice - exitPrice) / currentPosition.entryPrice) * 100;
                        }}
                        
                        allTrades.push({{
                            entryDate: currentPosition.entryDate,
                            exitDate: exitDate,
                            signal: currentPosition.type,
                            entryPrice: currentPosition.entryPrice,
                            exitPrice: exitPrice,
                            pnl: pnl,
                            holdingDays: i - currentPosition.entryIdx,
                            exitReason: currSignal === 'NEUTRAL' ? 'Signal → Neutral' : 'Signal Flip'
                        }});
                        currentPosition = null;
                    }}
                }}
                
                // OPEN new position
                if ((currSignal === 'BULLISH' || currSignal === 'BEARISH') && !currentPosition) {{
                    const entryPrice = data.prices[i];
                    if (entryPrice && !isNaN(entryPrice)) {{
                        currentPosition = {{
                            type: currSignal,
                            entryDate: data.dates[i],
                            entryPrice: entryPrice,
                            entryIdx: i
                        }};
                    }}
                }}
            }}
            
            // Take the LAST 30 trades (most recent), then reverse for display (newest first)
            const trades = allTrades.slice(-30).reverse();
            console.log(`[TRADE HISTORY] Total trades: ${{allTrades.length}}, showing last 30`);
            
            // ================ CALCULATE STATISTICS ================
            let totalPnL = 0;
            let wins = 0, losses = 0;
            let totalHoldingDays = 0;
            let grossProfit = 0, grossLoss = 0;
            
            trades.forEach(t => {{
                totalPnL += t.pnl;
                totalHoldingDays += t.holdingDays;
                if (t.pnl > 0) {{
                    wins++;
                    grossProfit += t.pnl;
                }} else if (t.pnl < 0) {{
                    losses++;
                    grossLoss += Math.abs(t.pnl);
                }}
            }});
            
            const winRate = trades.length > 0 ? (wins / trades.length) * 100 : 0;
            const avgPnL = trades.length > 0 ? totalPnL / trades.length : 0;
            const avgHolding = trades.length > 0 ? totalHoldingDays / trades.length : 0;
            const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? 999 : 0);
            const avgWin = wins > 0 ? grossProfit / wins : 0;
            const avgLoss = losses > 0 ? grossLoss / losses : 0;
            
            // Store for reference
            window.tradeHistoryData = {{ trades, totalPnL, winRate, avgPnL, avgHolding, profitFactor, wins, losses, avgWin, avgLoss }};
            
            // ================ RENDER CLEAN SIGNAL-FOLLOWING DISPLAY ================
            const endingCapital = 10000 * (1 + totalPnL / 100);
            const dollarProfit = endingCapital - 10000;
            
            document.getElementById('historySummary').innerHTML = `
                <!-- SIGNAL-FOLLOWING PERFORMANCE -->
                <div style="background: linear-gradient(135deg, ${{totalPnL >= 0 ? 'rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05)' : 'rgba(255, 82, 82, 0.15), rgba(255, 82, 82, 0.05)'}}); border: 2px solid ${{totalPnL >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}}; border-radius: 12px; padding: 24px; text-align: center;">
                    
                    <!-- Strategy Header -->
                    <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px;">
                        🐍 Signal-Following Strategy (Last ${{trades.length}} Trades)
                    </div>
                    
                    <!-- Key Metrics Grid - Clearer Labels -->
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 16px;">
                        
                        <!-- RETURN -->
                        <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 16px;">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
                                Return
                            </div>
                            <div style="font-size: 36px; font-weight: 900; color: ${{totalPnL >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}}; font-family: 'JetBrains Mono', monospace;">
                                ${{totalPnL >= 0 ? '+' : ''}}${{totalPnL.toFixed(1)}}%
                            </div>
                            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">
                                $${{10000 + dollarProfit.toFixed(0)}} from $10K
                            </div>
                        </div>
                        
                        <!-- WINS / LOSSES -->
                        <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 16px;">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
                                Wins / Losses
                            </div>
                            <div style="font-size: 36px; font-weight: 900; font-family: 'JetBrains Mono', monospace;">
                                <span style="color: var(--accent-green);">${{wins}}</span><span style="color: var(--text-secondary);">/</span><span style="color: var(--accent-red);">${{losses}}</span>
                            </div>
                            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">
                                ${{winRate.toFixed(0)}}% win rate
                            </div>
                        </div>
                        
                        <!-- PROFIT FACTOR -->
                        <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 16px;">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
                                Profit Factor
                            </div>
                            <div style="font-size: 36px; font-weight: 900; color: ${{profitFactor >= 1.5 ? 'var(--accent-green)' : profitFactor >= 1 ? 'var(--accent-yellow)' : 'var(--accent-red)'}}; font-family: 'JetBrains Mono', monospace;">
                                ${{profitFactor >= 100 ? '∞' : profitFactor.toFixed(2)}}x
                            </div>
                            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">
                                gross profit / gross loss
                            </div>
                        </div>
                        
                        <!-- AVG P&L -->
                        <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 16px;">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
                                Avg P&L / Trade
                            </div>
                            <div style="font-size: 36px; font-weight: 900; color: ${{avgPnL >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}}; font-family: 'JetBrains Mono', monospace;">
                                ${{avgPnL >= 0 ? '+' : ''}}${{avgPnL.toFixed(2)}}%
                            </div>
                            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">
                                avg hold: ${{avgHolding.toFixed(1)}} days
                            </div>
                        </div>
                    </div>
                    
                    <!-- Strategy Description -->
                    <div style="padding: 10px 16px; background: rgba(0, 212, 255, 0.1); border: 1px solid var(--accent-blue); border-radius: 8px; display: inline-block;">
                        <div style="font-size: 12px; color: var(--accent-blue);">
                            Enter on signal change • Exit when signal flips
                        </div>
                    </div>
                </div>
            `;
            
            // Build clean table rows for Signal-Following trades
            let rowsHtml = '';
            trades.forEach((trade, idx) => {{
                rowsHtml += `
                    <tr>
                        <td style="color: var(--text-secondary);">${{trades.length - idx}}</td>
                        <td>${{trade.entryDate}}</td>
                        <td><span class="signal-badge ${{trade.signal.toLowerCase()}}">${{trade.signal}}</span></td>
                        <td style="font-family: 'JetBrains Mono', monospace;">$${{trade.entryPrice.toFixed(2)}}</td>
                        <td style="font-family: 'JetBrains Mono', monospace;">$${{trade.exitPrice.toFixed(2)}}</td>
                        <td style="color: var(--text-secondary);">${{trade.holdingDays}}d</td>
                        <td style="font-size: 10px; color: var(--text-secondary);">${{trade.exitReason}}</td>
                        <td><span class="pnl ${{trade.pnl >= 0 ? 'positive' : 'negative'}}">${{trade.pnl >= 0 ? '+' : ''}}${{trade.pnl.toFixed(2)}}%</span></td>
                    </tr>
                `;
            }});
            
            document.getElementById('historyTableBody').innerHTML = rowsHtml || '<tr><td colspan="8" style="text-align: center; color: var(--text-secondary);">No trades found</td></tr>';
        }}
        
        function renderTradeHistoryFromPreCalculated(preCalculatedTrades) {{
            // Use PRE-CALCULATED trades from Python (matches equity curve metrics)
            // Trades are already in chronological order (oldest first), reverse for display (newest first)
            const trades = [...preCalculatedTrades].reverse();
            
            console.log(`[TRADE HISTORY] Rendering ${{trades.length}} pre-calculated trades`);
            
            // ================ CALCULATE STATISTICS ================
            // Calculate compound returns (same as Python: multiply (1 + pnl/100) for each trade)
            let equity = 100;  // Start with 100 (represents 100%)
            let totalPnLSum = 0;  // Simple sum for display
            let wins = 0, losses = 0;
            let totalHoldingDays = 0;
            let grossProfit = 0, grossLoss = 0;
            
            trades.forEach(t => {{
                // Compound the returns
                equity *= (1 + t.pnl / 100);
                totalPnLSum += t.pnl;
                totalHoldingDays += t.holding_days;
                if (t.pnl > 0) {{
                    wins++;
                    grossProfit += t.pnl;
                }} else if (t.pnl < 0) {{
                    losses++;
                    grossLoss += Math.abs(t.pnl);
                }}
            }});
            
            // Compound return percentage
            const totalPnL = (equity - 100);
            
            const winRate = trades.length > 0 ? (wins / trades.length) * 100 : 0;
            const avgPnL = trades.length > 0 ? totalPnLSum / trades.length : 0;
            const avgHolding = trades.length > 0 ? totalHoldingDays / trades.length : 0;
            const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? 999 : 0);
            
            // Store for reference
            window.tradeHistoryData = {{ trades, totalPnL, winRate, avgPnL, avgHolding, profitFactor, wins, losses }};
            
            // ================ RENDER DISPLAY ================
            const endingCapital = 10000 * (equity / 100);  // Use compound equity
            const dollarProfit = endingCapital - 10000;
            
            document.getElementById('historySummary').innerHTML = `
                <div style="padding: 24px;">
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px;">
                        <!-- RETURN -->
                        <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 16px;">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
                                Return
                            </div>
                            <div style="font-size: 36px; font-weight: 900; color: ${{totalPnL >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}}; font-family: 'JetBrains Mono', monospace;">
                                ${{totalPnL >= 0 ? '+' : ''}}${{totalPnL.toFixed(1)}}%
                            </div>
                            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">
                                $${{endingCapital.toFixed(0)}} from $10K
                            </div>
                        </div>
                        
                        <!-- WINS / LOSSES -->
                        <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 16px;">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
                                Wins / Losses
                            </div>
                            <div style="font-size: 36px; font-weight: 900; font-family: 'JetBrains Mono', monospace;">
                                <span style="color: var(--accent-green);">${{wins}}</span><span style="color: var(--text-secondary);">/</span><span style="color: var(--accent-red);">${{losses}}</span>
                            </div>
                            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">
                                ${{winRate.toFixed(0)}}% win rate
                            </div>
                        </div>
                        
                        <!-- PROFIT FACTOR -->
                        <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 16px;">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
                                Profit Factor
                            </div>
                            <div style="font-size: 36px; font-weight: 900; color: ${{profitFactor >= 1.5 ? 'var(--accent-green)' : profitFactor >= 1 ? 'var(--accent-yellow)' : 'var(--accent-red)'}}; font-family: 'JetBrains Mono', monospace;">
                                ${{profitFactor >= 100 ? '∞' : profitFactor.toFixed(2)}}x
                            </div>
                            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">
                                gross profit / gross loss
                            </div>
                        </div>
                        
                        <!-- AVG P&L -->
                        <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 16px;">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
                                Avg P&L / Trade
                            </div>
                            <div style="font-size: 36px; font-weight: 900; color: ${{avgPnL >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}}; font-family: 'JetBrains Mono', monospace;">
                                ${{avgPnL >= 0 ? '+' : ''}}${{avgPnL.toFixed(2)}}%
                            </div>
                            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">
                                avg hold: ${{avgHolding.toFixed(1)}} days
                            </div>
                        </div>
                    </div>
                    
                    <!-- Strategy Description -->
                    <div style="padding: 10px 16px; background: rgba(0, 212, 255, 0.1); border: 1px solid var(--accent-blue); border-radius: 8px; display: inline-block;">
                        <div style="font-size: 12px; color: var(--accent-blue);">
                            Enter on signal change • Exit when signal flips
                        </div>
                    </div>
                </div>
            `;
            
            // Build clean table rows
            let rowsHtml = '';
            trades.forEach((trade, idx) => {{
                rowsHtml += `
                    <tr>
                        <td style="color: var(--text-secondary);">${{trades.length - idx}}</td>
                        <td>${{trade.entry_date}}</td>
                        <td><span class="signal-badge ${{trade.signal.toLowerCase()}}">${{trade.signal}}</span></td>
                        <td style="font-family: 'JetBrains Mono', monospace;">$${{trade.entry_price.toFixed(2)}}</td>
                        <td style="font-family: 'JetBrains Mono', monospace;">$${{trade.exit_price.toFixed(2)}}</td>
                        <td style="color: var(--text-secondary);">${{trade.holding_days}}d</td>
                        <td style="font-size: 10px; color: var(--text-secondary);">${{trade.exit_reason || 'Signal Flip'}}</td>
                        <td><span class="pnl ${{trade.pnl >= 0 ? 'positive' : 'negative'}}">${{trade.pnl >= 0 ? '+' : ''}}${{trade.pnl.toFixed(2)}}%</span></td>
                    </tr>
                `;
            }});
            
            document.getElementById('historyTableBody').innerHTML = rowsHtml || '<tr><td colspan="8" style="text-align: center; color: var(--text-secondary);">No trades found</td></tr>';
        }}
        
        function closeTradeHistory() {{
            document.getElementById('historyModal').classList.remove('active');
        }}
        
        // ===================== LOAD SAVED PREFERENCES =====================
        // Try to load user's saved preferences first
        const prefsLoaded = loadPreferences();
        
        if (!prefsLoaded) {{
            // No saved preferences - determine best config
            const initData = ASSET_DATA[currentAsset];
            const initConfig = window.OPTIMAL_CONFIGS ? window.OPTIMAL_CONFIGS[currentAsset] : null;
            const initAllHorizons = initData.horizons || [];
            
            // Start with ALL horizons to calculate raw performance
            enabledHorizons = [...initAllHorizons];
            isOptimalConfigApplied = false;
            recalculateSignalsWithEnabledHorizons();
            
            // Calculate raw win rate
            updateCharts();
            const rawWinRate = cachedOptimizedWinRate || initData.base_win_rate || 50;
            
            // Try optimal config if available
            if (initConfig && initConfig.viable_horizons && initConfig.viable_horizons.length > 0) {{
                const available = initAllHorizons;
                const optimalHorizons = initConfig.viable_horizons.filter(h => available.includes(h));
                
                if (optimalHorizons.length > 0) {{
                    enabledHorizons = [...optimalHorizons];
                    recalculateSignalsWithEnabledHorizons();
                    updateCharts();
                    const optWinRate = cachedOptimizedWinRate || initData.base_win_rate || 50;
                    
                    if (optWinRate > rawWinRate) {{
                        isOptimalConfigApplied = true;
                        console.log(`Initial: Optimal config improves performance (${{rawWinRate.toFixed(0)}}% → ${{optWinRate.toFixed(0)}}%)`);
                    }} else {{
                        // Revert to full ensemble
                        enabledHorizons = [...initAllHorizons];
                        isOptimalConfigApplied = false;
                        recalculateSignalsWithEnabledHorizons();
                        console.log(`Initial: Full ensemble better (${{rawWinRate.toFixed(0)}}% vs ${{optWinRate.toFixed(0)}}%)`);
                    }}
                }}
            }}
        }} else {{
            console.log('📂 Using saved preferences for initialization');
        }}
        // =================================================================
        
        // Final calculations and display
        recalculateSignalsWithEnabledHorizons();
        
        updateStats();
        updateCharts();  // This also updates the health check
        updatePriceTargets();
        updateOptimizationHero();  // Update the optimization hero section
        updateOptimalConfigDisplay();
        
        // Fix chart width on initial load - trigger resize after render
        function resizeAllCharts() {{
            window.dispatchEvent(new Event('resize'));
            const snakeChart = document.getElementById('snake-chart');
            const rsiChart = document.getElementById('rsi-chart');
            const equityChart = document.getElementById('equity-chart');
            if (snakeChart) Plotly.Plots.resize(snakeChart);
            if (rsiChart) Plotly.Plots.resize(rsiChart);
            if (equityChart) Plotly.Plots.resize(equityChart);
        }}
        // Multiple resize attempts to ensure charts fill container
        setTimeout(resizeAllCharts, 100);
        setTimeout(resizeAllCharts, 500);
        setTimeout(resizeAllCharts, 1000);
        
        // Also resize on window load
        window.addEventListener('load', resizeAllCharts);
        
        // FORCE button animation via JavaScript (bypass CSS class issues)
        function forceButtonAnimation() {{
            // Manual blink effect using setInterval - no CSS animation needed
            const btn = document.getElementById('autoOptimizeBtn');
            if (btn && !isOptimalConfigApplied) {{
                let blinkOn = true;
                window.buttonBlinkInterval = setInterval(() => {{
                    if (isOptimalConfigApplied) {{
                        clearInterval(window.buttonBlinkInterval);
                        return;
                    }}
                    if (blinkOn) {{
                        btn.style.background = 'linear-gradient(135deg, #FFD700, #FF8C00)';
                        btn.style.boxShadow = '0 0 20px rgba(255, 215, 0, 1), 0 0 40px rgba(255, 215, 0, 0.6)';
                        btn.style.transform = 'scale(1.05)';
                        btn.style.opacity = '1';
                    }} else {{
                        btn.style.background = 'linear-gradient(135deg, #333, #222)';
                        btn.style.boxShadow = 'none';
                        btn.style.transform = 'scale(1)';
                        btn.style.opacity = '0.5';
                    }}
                    blinkOn = !blinkOn;
                }}, 500);
                console.log('✨ Manual button blink started via setInterval');
            }}
        }}
        setTimeout(forceButtonAnimation, 500);
        setTimeout(forceButtonAnimation, 2000);
    </script>
</body>
</html>
'''
    return html


def main():
    print("=" * 70)
    print("  QDT ENSEMBLE - Multi-Asset Dashboard Builder")
    print("=" * 70)
    
    all_data = {}
    
    print("\nProcessing assets...")
    for name, config in ASSETS.items():
        data = process_asset(name, config)
        if data:
            all_data[name] = data
    
    if not all_data:
        print("\n[FAIL] No asset data found! Run the sandbox pipeline first.")
        return
    
    print(f"\n[OK] Loaded {len(all_data)} assets")
    
    print("\nBuilding HTML dashboard...")
    html = build_html(all_data)
    
    output_path = os.path.join(EXPERIMENT_ROOT, 'QDT_Ensemble_Dashboard.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n{'=' * 70}")
    print(f"  [OK] Dashboard saved to: {output_path}")
    print(f"{'=' * 70}")
    print(f"\nAssets included:")
    for name in all_data.keys():
        acc = all_data[name]['accuracy']
        edge = all_data[name]['edge']
        status = "[OK]" if acc >= 55 else ("[WARN]" if acc >= 50 else "[FAIL]")
        print(f"  {status} {name}: {acc}% accuracy, {edge:+.1f}% edge")


if __name__ == "__main__":
    main()


