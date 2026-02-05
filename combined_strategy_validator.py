"""
Combined Strategy Validator
Validates Artemis's breakthrough: vol70_consec3
Target: Sharpe 8.18+, Win Rate 85%, $1.69/trade

Methodology:
1. Calculate inverse_spread signal daily
2. Filter out days where 20-day volatility > 70th percentile
3. Only enter after 3 consecutive days of same signal
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_forecasts(asset_path: Path) -> pd.DataFrame:
    """Load all forecast CSVs for an asset."""
    forecasts = {}
    for f in asset_path.glob("forecast_d*.csv"):
        horizon = f.stem.replace("forecast_d", "")
        df = pd.read_csv(f)
        df['date'] = pd.to_datetime(df['date'])
        forecasts[f'd{horizon}'] = df.set_index('date')['prediction']
    
    if not forecasts:
        return pd.DataFrame()
    
    return pd.DataFrame(forecasts).sort_index()

def load_prices(asset_path: Path) -> pd.Series:
    """Load price history."""
    cache_file = asset_path / "price_history_cache.json"
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        # Data is a list of {date, open, high, low, close, volume}
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # Remove timezone
        prices = df.set_index('date')['close']
        return prices.sort_index()
    return pd.Series()

def calculate_inverse_spread_signal(forecasts: pd.DataFrame, horizons: list) -> pd.Series:
    """Calculate inverse spread weighted signal."""
    signals = []
    
    for date in forecasts.index:
        weighted_sum = 0
        total_weight = 0
        
        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                col1, col2 = f'd{h1}', f'd{h2}'
                if col1 in forecasts.columns and col2 in forecasts.columns:
                    p1 = forecasts.loc[date, col1]
                    p2 = forecasts.loc[date, col2]
                    if pd.notna(p1) and pd.notna(p2):
                        drift = p2 - p1
                        weight = 1.0 / (h2 - h1)
                        if drift > 0:
                            weighted_sum += weight
                        elif drift < 0:
                            weighted_sum -= weight
                        total_weight += weight
        
        if total_weight > 0:
            signals.append(weighted_sum / total_weight)
        else:
            signals.append(0)
    
    return pd.Series(signals, index=forecasts.index)

def apply_volatility_filter(prices: pd.Series, percentile: int = 70) -> pd.Series:
    """Return True for low volatility days (below percentile)."""
    returns = prices.pct_change()
    vol_20d = returns.rolling(20).std()
    threshold = vol_20d.quantile(percentile / 100)
    return vol_20d <= threshold

def apply_consecutive_filter(signals: pd.Series, threshold: float, consecutive: int = 3) -> pd.Series:
    """Only signal after N consecutive days in same direction."""
    direction = pd.Series(0, index=signals.index)
    direction[signals > threshold] = 1
    direction[signals < -threshold] = -1
    
    filtered = pd.Series(0, index=signals.index)
    
    for i in range(consecutive, len(direction)):
        window = direction.iloc[i-consecutive:i]
        if (window == 1).all():
            filtered.iloc[i] = 1
        elif (window == -1).all():
            filtered.iloc[i] = -1
    
    return filtered

def backtest(signals: pd.Series, prices: pd.Series) -> dict:
    """Run backtest and calculate metrics."""
    # Align data
    common_dates = signals.index.intersection(prices.index)
    signals = signals.loc[common_dates]
    prices = prices.loc[common_dates]
    
    # Calculate returns
    price_returns = prices.pct_change().shift(-1)  # Next day return
    
    # Strategy returns (only when we have a signal)
    strategy_returns = signals * price_returns
    strategy_returns = strategy_returns.dropna()
    
    # Filter to actual trades
    trades = strategy_returns[signals != 0]
    
    if len(trades) == 0:
        return {"error": "No trades"}
    
    # Metrics
    total_return = (1 + strategy_returns).prod() - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    win_rate = (trades > 0).sum() / len(trades) * 100
    avg_trade = trades.mean() * prices.mean()  # Approximate $/trade
    
    return {
        "total_trades": len(trades),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 2),
        "win_rate_pct": round(win_rate, 1),
        "avg_trade_dollar": round(avg_trade, 2)
    }

def run_combined_strategy(asset_path: Path, horizons: list, vol_pct: int = 70, consec: int = 3, threshold: float = 0.25):
    """Run the full combined strategy."""
    print(f"\n{'='*60}")
    print(f"Asset: {asset_path.name}")
    print(f"Config: horizons={horizons}, vol_filter={vol_pct}th pct, consecutive={consec}")
    print('='*60)
    
    # Load data
    forecasts = load_forecasts(asset_path)
    prices = load_prices(asset_path)
    
    if forecasts.empty or prices.empty:
        print("ERROR: Could not load data")
        return None
    
    print(f"Data: {len(forecasts)} forecast days, {len(prices)} price days")
    
    # Step 1: Calculate base signal
    raw_signal = calculate_inverse_spread_signal(forecasts, horizons)
    print(f"Raw signals: {(raw_signal.abs() > threshold).sum()} days with signal")
    
    # Step 2: Apply volatility filter
    vol_mask = apply_volatility_filter(prices, vol_pct)
    vol_mask = vol_mask.reindex(raw_signal.index, fill_value=False)
    print(f"Vol filter: {vol_mask.sum()} low-vol days out of {len(vol_mask)}")
    
    # Step 3: Apply consecutive filter
    filtered_signal = apply_consecutive_filter(raw_signal, threshold, consec)
    print(f"Consecutive filter: {(filtered_signal != 0).sum()} signals after {consec}-day requirement")
    
    # Combine filters
    final_signal = filtered_signal.copy()
    final_signal[~vol_mask] = 0
    
    print(f"After vol filter + consecutive: {(final_signal != 0).sum()} trade signals")
    
    # Backtest
    results = backtest(final_signal, prices)
    
    print(f"\nRESULTS:")
    print(f"  Trades: {results.get('total_trades', 0)}")
    print(f"  Sharpe: {results.get('sharpe', 'N/A')}")
    print(f"  Win Rate: {results.get('win_rate_pct', 'N/A')}%")
    print(f"  Avg $/Trade: ${results.get('avg_trade_dollar', 'N/A')}")
    print(f"  Total Return: {results.get('total_return_pct', 'N/A')}%")
    
    return results

if __name__ == "__main__":
    base_path = Path("C:/Users/William Dennis/projects/nexus/data")
    
    # Test configurations - start with less restrictive, then Artemis's breakthrough
    configs = [
        # Less restrictive first (no consecutive filter)
        ("1866_Crude_Oil", [1, 3, 5, 8, 10], 70, 1),  # vol70 only
        ("1866_Crude_Oil", [1, 3, 5, 8, 10], 70, 2),  # vol70_consec2
        ("1866_Crude_Oil", [1, 3, 5, 8, 10], 70, 3),  # vol70_consec3 (Artemis's config)
    ]
    
    all_results = {}
    
    for asset_dir, horizons, vol_pct, consec in configs:
        asset_path = base_path / asset_dir
        if asset_path.exists():
            results = run_combined_strategy(asset_path, horizons, vol_pct, consec)
            if results:
                all_results[asset_dir] = results
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for asset, r in all_results.items():
        print(f"{asset}: Sharpe={r.get('sharpe')}, WinRate={r.get('win_rate_pct')}%, Trades={r.get('total_trades')}")
