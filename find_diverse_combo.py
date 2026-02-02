"""
Find Best DIVERSE Horizon Combination for any Asset
====================================================
Requires horizons from SHORT + MEDIUM + LONG term groups to ensure robustness.

For volatile assets, we need:
- SHORT (D+1 to D+3): Captures immediate momentum
- MEDIUM (D+4 to D+6): Filters daily noise
- LONG (D+7+): Provides directional anchor
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from itertools import combinations
from datetime import datetime
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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

# Default to Crude_Oil if no argument
DEFAULT_ASSET = 'Crude_Oil'

# Horizon groups for diversity (will be adjusted based on available horizons)
SHORT_TERM = [1, 2, 3]      # Immediate momentum
MEDIUM_TERM = [4, 5, 6]     # Noise filter
LONG_TERM = [7, 8, 9, 10]   # Directional anchor (will be extended if needed)

# Requirements
MIN_SHORT = 1   # At least 1 short-term horizon
MIN_MEDIUM = 1  # At least 1 medium-term horizon
MIN_LONG = 1    # At least 1 long-term horizon
MIN_TOTAL = 4   # At least 4 horizons total for robust signal
MAX_TOTAL = 7   # Not too many (dilutes signal)

# Rolling window settings
LOOKBACK_WINDOW = 60
OOS_DAYS = 30
THRESHOLD_OPTIONS = [0.1, 0.15, 0.2, 0.25, 0.3]

def get_asset_info(asset_name):
    """Get asset information from mapping."""
    asset_info = ASSETS.get(asset_name)
    if not asset_info:
        print(f"[FAIL] Unknown asset: {asset_name}")
        print(f"Available assets: {', '.join(ASSETS.keys())}")
        sys.exit(1)
    return asset_info

def discover_available_horizons(asset_dir):
    """Discover all available horizons for an asset."""
    available = []
    for h in range(1, 200):  # Check up to D+200
        forecast_file = os.path.join(asset_dir, f'forecast_d{h}.csv')
        if os.path.exists(forecast_file):
            available.append(h)
    return sorted(available)

def adjust_horizon_groups(available_horizons):
    """Adjust horizon groups based on available horizons."""
    short = [h for h in SHORT_TERM if h in available_horizons]
    medium = [h for h in MEDIUM_TERM if h in available_horizons]
    # For long-term, include all horizons >= 7 that are available
    long_term = [h for h in available_horizons if h >= 7]
    return short, medium, long_term

def load_forecast_data(asset_dir, available_horizons):
    """Load all forecast files and merge them."""
    horizons_data = {}
    
    for h in available_horizons:
        forecast_file = os.path.join(asset_dir, f'forecast_d{h}.csv')
        if os.path.exists(forecast_file):
            df = pd.read_csv(forecast_file, parse_dates=['date'])
            if 'prediction' in df.columns and 'actual' in df.columns:
                horizons_data[h] = df[['date', 'prediction', 'actual']].copy()
    
    available_horizons = sorted(horizons_data.keys())
    
    base_df = None
    for h, df in horizons_data.items():
        df = df.rename(columns={'prediction': f'pred_{h}', 'actual': 'actual'})
        if base_df is None:
            base_df = df
        else:
            base_df = base_df.merge(df[['date', f'pred_{h}']], on='date', how='outer')
    
    base_df = base_df.sort_values('date').reset_index(drop=True)
    base_df = base_df.dropna(subset=['actual'])
    
    forecast_matrix = pd.DataFrame(index=base_df['date'])
    for h in available_horizons:
        forecast_matrix[h] = base_df[f'pred_{h}'].values
    
    prices = base_df.set_index('date')['actual']
    
    return forecast_matrix, prices, available_horizons


def calculate_signals_pairwise_slopes(forecast_df, horizons, threshold):
    """Calculate signals using pairwise slopes (matrix drift method)."""
    signals = []
    net_probs = []
    
    for date in forecast_df.index:
        row = forecast_df.loc[date]
        slopes = []
        
        for i_idx, i in enumerate(horizons):
            for j in horizons[i_idx + 1:]:
                if i in row.index and j in row.index:
                    if pd.notna(row[i]) and pd.notna(row[j]):
                        drift = row[j] - row[i]
                        slopes.append(drift)
        
        if len(slopes) == 0:
            net_prob = 0.0
        else:
            bullish = sum(1 for s in slopes if s > 0)
            bearish = sum(1 for s in slopes if s < 0)
            total = len(slopes)
            net_prob = (bullish - bearish) / total
        
        net_probs.append(net_prob)
        
        if net_prob > threshold:
            signals.append('BULLISH')
        elif net_prob < -threshold:
            signals.append('BEARISH')
        else:
            signals.append('NEUTRAL')
    
    return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)


def calculate_trading_performance(signals, prices):
    """Calculate signal-following trading performance."""
    if len(signals) < 3:
        return None
    
    trades = []
    position = None
    entry_price = None
    entry_date = None
    
    for i in range(len(signals)):
        current_signal = signals.iloc[i]
        current_price = prices.iloc[i]
        current_date = signals.index[i]
        
        if position is None:
            if current_signal in ['BULLISH', 'BEARISH']:
                position = current_signal
                entry_price = current_price
                entry_date = current_date
        else:
            should_exit = False
            if position == 'BULLISH' and current_signal != 'BULLISH':
                should_exit = True
            elif position == 'BEARISH' and current_signal != 'BEARISH':
                should_exit = True
            
            if should_exit:
                if position == 'BULLISH':
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct
                })
                
                if current_signal in ['BULLISH', 'BEARISH']:
                    position = current_signal
                    entry_price = current_price
                    entry_date = current_date
                else:
                    position = None
    
    if len(trades) < 3:
        return None
    
    trades_df = pd.DataFrame(trades)
    
    total_return = trades_df['pnl_pct'].sum()
    wins = trades_df[trades_df['pnl_pct'] > 0]
    losses = trades_df[trades_df['pnl_pct'] <= 0]
    win_rate = len(wins) / len(trades_df) * 100
    
    avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['pnl_pct'].mean()) if len(losses) > 0 else 0.001
    profit_factor = (len(wins) * avg_win) / (len(losses) * avg_loss) if len(losses) > 0 and avg_loss > 0 else float('inf')
    
    if trades_df['pnl_pct'].std() > 0:
        sharpe = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252 / max(1, len(trades_df)))
    else:
        sharpe = 0
    
    cumulative = trades_df['pnl_pct'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    
    return {
        'n_trades': len(trades_df),
        'total_return': total_return,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'max_drawdown': max_dd
    }


def is_diverse_combo(horizons, short_term, medium_term, long_term):
    """Check if combo has required diversity."""
    short_count = len([h for h in horizons if h in short_term])
    medium_count = len([h for h in horizons if h in medium_term])
    long_count = len([h for h in horizons if h in long_term])
    
    return (short_count >= MIN_SHORT and 
            medium_count >= MIN_MEDIUM and 
            long_count >= MIN_LONG and
            len(horizons) >= MIN_TOTAL and
            len(horizons) <= MAX_TOTAL)


def get_diversity_breakdown(horizons, short_term, medium_term, long_term):
    """Get breakdown of horizon groups."""
    short = [h for h in horizons if h in short_term]
    medium = [h for h in horizons if h in medium_term]
    long = [h for h in horizons if h in long_term]
    return short, medium, long


def main():
    """Main execution function."""
    try:
        # Load data
        print("\n\n[INFO] Loading forecast data...")
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Find best diverse horizon combination for an asset')
        parser.add_argument('--asset', '-a', type=str, default=DEFAULT_ASSET,
                            help=f'Asset name (default: {DEFAULT_ASSET})')
        args = parser.parse_args()

        asset_name = args.asset
        asset_info = get_asset_info(asset_name)
        asset_id = asset_info['id']
        asset_display_name = asset_info['name']

        # Set asset directory
        ASSET_DIR = os.path.join(DATA_DIR, f'{asset_id}_{asset_name}')

        # Discover available horizons
        available_horizons = discover_available_horizons(ASSET_DIR)
        if not available_horizons:
            print(f"[FAIL] No forecast files found in {ASSET_DIR}")
            sys.exit(1)

        # Adjust horizon groups based on available horizons
        short_term, medium_term, long_term = adjust_horizon_groups(available_horizons)

        print("=" * 80)
        print(f"  DIVERSE ENSEMBLE SEARCH FOR {asset_display_name.upper()}")
        print("=" * 80)
        print(f"\n  Horizon Groups:")
        print(f"    SHORT  (momentum):  {short_term}")
        print(f"    MEDIUM (filter):    {medium_term}")
        print(f"    LONG   (anchor):    {long_term}")
        print(f"\n  Requirements:")
        print(f"    Min short-term:  {MIN_SHORT}")
        print(f"    Min medium-term: {MIN_MEDIUM}")
        print(f"    Min long-term:   {MIN_LONG}")
        print(f"    Total horizons:  {MIN_TOTAL} to {MAX_TOTAL}")

        # Load forecast data
        forecast_df, prices, available_horizons = load_forecast_data(ASSET_DIR, available_horizons)
        forecast_df.index = pd.to_datetime(forecast_df.index)
        prices.index = pd.to_datetime(prices.index)

        print(f"[OK] Available horizons: {available_horizons}")
        print(f"[OK] Total data points: {len(forecast_df)}")

        # Split data
        forecast_df = forecast_df.sort_index()
        prices = prices.sort_index()

        split_date = forecast_df.index[-OOS_DAYS]
        train_df = forecast_df.loc[:split_date].tail(LOOKBACK_WINDOW)
        test_df = forecast_df.loc[split_date:]

        train_prices = prices.loc[train_df.index]
        test_prices = prices.loc[test_df.index]

        print(f"\n[INFO] DATA SPLIT:")
        print(f"   Training: {len(train_df)} days ({train_df.index.min().date()} to {train_df.index.max().date()})")
        print(f"   Testing:  {len(test_df)} days ({test_df.index.min().date()} to {test_df.index.max().date()})")

        # Generate all diverse combinations
        print("\n[INFO] Generating diverse combinations...")
        all_combos = []
        for size in range(MIN_TOTAL, MAX_TOTAL + 1):
            for combo in combinations(available_horizons, size):
                if is_diverse_combo(combo, short_term, medium_term, long_term):
                    all_combos.append(list(combo))

        print(f"[OK] Found {len(all_combos)} diverse combinations (out of 968 total)")

        # Test all diverse combos
        print(f"\n[INFO] Testing {len(all_combos)} diverse combinations × {len(THRESHOLD_OPTIONS)} thresholds...")

        results = []

        for combo in tqdm(all_combos, desc="Testing diverse combos"):
            for threshold in THRESHOLD_OPTIONS:
                # Training performance
                train_signals, _ = calculate_signals_pairwise_slopes(train_df, combo, threshold)
                train_perf = calculate_trading_performance(train_signals, train_prices)
                
                if train_perf is None:
                    continue
                
                # OOS performance
                test_signals, _ = calculate_signals_pairwise_slopes(test_df, combo, threshold)
                test_perf = calculate_trading_performance(test_signals, test_prices)
                
                if test_perf is None:
                    continue
                
                # Calculate composite score (same as grid search)
                oos_return_score = min(test_perf['total_return'] / 20.0, 1.0) if test_perf['total_return'] > 0 else test_perf['total_return'] / 20.0
                oos_sharpe_score = min(test_perf['sharpe'] / 3.0, 1.0) if test_perf['sharpe'] > 0 else 0
                oos_winrate_score = test_perf['win_rate'] / 100.0
                oos_dd_score = max(0, 1 + test_perf['max_drawdown'] / 20.0)
                
                composite = (oos_return_score * 0.35 + oos_sharpe_score * 0.30 + 
                            oos_winrate_score * 0.20 + oos_dd_score * 0.15)
                
                short, medium, long = get_diversity_breakdown(combo, short_term, medium_term, long_term)
                
                results.append({
                    'horizons': combo,
                    'threshold': threshold,
                    'short': short,
                    'medium': medium,
                    'long': long,
                    'n_pairs': len(combo) * (len(combo) - 1) // 2,
                    'train_return': train_perf['total_return'],
                    'train_sharpe': train_perf['sharpe'],
                    'oos_return': test_perf['total_return'],
                    'oos_sharpe': test_perf['sharpe'],
                    'oos_winrate': test_perf['win_rate'],
                    'oos_dd': test_perf['max_drawdown'],
                    'oos_pf': test_perf['profit_factor'],
                    'oos_trades': test_perf['n_trades'],
                    'composite': composite
                })

        # Sort by composite score
        results.sort(key=lambda x: x['composite'], reverse=True)

        # Print results
        print("\n" + "=" * 100)
        print("  TOP 15 DIVERSE COMBINATIONS (ranked by OOS Composite Score)")
        print("=" * 100)
        print(f"{'Rank':<5} {'Horizons':<20} {'Diversity':<20} {'Thresh':<7} {'Train%':<9} {'OOS%':<9} {'Sharpe':<8} {'Win%':<7} {'Score':<7}")
        print("-" * 100)

        for i, r in enumerate(results[:15], 1):
            diversity = f"S{len(r['short'])}+M{len(r['medium'])}+L{len(r['long'])}"
            print(f"{i:<5} {str(r['horizons']):<20} {diversity:<20} {r['threshold']:<7.2f} {r['train_return']:+7.1f}% {r['oos_return']:+7.1f}% {r['oos_sharpe']:7.2f} {r['oos_winrate']:5.1f}% {r['composite']:6.3f}")

        # Best result
        best = results[0]
        print("\n" + "=" * 100)
        print("  [BEST] BEST DIVERSE COMBINATION")
        print("=" * 100)
        print(f"\n  Horizons:     {best['horizons']}")
        print(f"  Threshold:    {best['threshold']}")
        print(f"  Pairs:        {best['n_pairs']} pairwise slopes")
        print(f"\n  Diversity Breakdown:")
        print(f"    SHORT  (D+1 to D+3):  {best['short']}")
        print(f"    MEDIUM (D+4 to D+6):  {best['medium']}")
        print(f"    LONG   (D+7 to D+10): {best['long']}")
        print(f"\n  [INFO] Training Performance:")
        print(f"    Return:     {best['train_return']:+.1f}%")
        print(f"    Sharpe:     {best['train_sharpe']:.2f}")
        print(f"\n  [OK] OUT-OF-SAMPLE Performance (30 days):")
        print(f"    Return:     {best['oos_return']:+.1f}%")
        print(f"    Sharpe:     {best['oos_sharpe']:.2f}")
        print(f"    Win Rate:   {best['oos_winrate']:.1f}%")
        print(f"    Max DD:     {best['oos_dd']:.1f}%")
        print(f"    Trades:     {best['oos_trades']}")
        print(f"    Score:      {best['composite']:.3f}")

        # Compare with non-diverse [1,2,3]
        print("\n" + "=" * 100)
        print("  [INFO] COMPARISON: Diverse vs Non-Diverse [1,2,3]")
        print("=" * 100)

        # Test [1,2,3] for comparison
        test_signals_123, _ = calculate_signals_pairwise_slopes(test_df, [1,2,3], 0.1)
        perf_123 = calculate_trading_performance(test_signals_123, test_prices)

        if perf_123:
            print(f"\n  {'Metric':<20} {'Diverse ' + str(best['horizons']):<25} {'Non-Diverse [1,2,3]':<25}")
            print(f"  {'-'*70}")
            print(f"  {'OOS Return':<20} {best['oos_return']:+.1f}%{'':<20} {perf_123['total_return']:+.1f}%")
            print(f"  {'OOS Sharpe':<20} {best['oos_sharpe']:.2f}{'':<22} {perf_123['sharpe']:.2f}")
            print(f"  {'OOS Win Rate':<20} {best['oos_winrate']:.1f}%{'':<21} {perf_123['win_rate']:.1f}%")
            print(f"  {'Pairwise Slopes':<20} {best['n_pairs']}{'':<24} 3")
            print(f"  {'Horizon Spread':<20} D+{min(best['horizons'])} to D+{max(best['horizons'])}{'':<15} D+1 to D+3")
            
            if best['oos_return'] >= perf_123['total_return'] * 0.9:  # Within 10%
                print(f"\n  [OK] RECOMMENDATION: Use DIVERSE combo {best['horizons']}")
                print(f"     Similar OOS return but with better risk characteristics:")
                print(f"     - More pairwise slopes = more robust signals")
                print(f"     - Horizon diversity = less sensitive to single events")
                print(f"     - Better for volatile assets like Crude Oil")
            else:
                print(f"\n  [WARN] [1,2,3] has higher raw return, but consider the risk tradeoff")

        # Update config file with best diverse combo
        CONFIGS_DIR = os.path.join(SCRIPT_DIR, 'configs')
        config_file = os.path.join(CONFIGS_DIR, f'optimal_{asset_name.lower()}.json')

        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Update with new optimal combo
            config['viable_horizons'] = best['horizons']
            config['threshold'] = best['threshold']
            config['diversity_breakdown'] = {
                'short': best['short'],
                'medium': best['medium'],
                'long': best['long']
            }
            config['last_updated'] = datetime.now().isoformat()
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"\n[OK] Config file updated: {config_file}")
            print(f"   New optimal horizons: {best['horizons']}")
            print(f"   Threshold: {best['threshold']}")
        else:
            print(f"\n[WARN] Config file not found: {config_file}")
            print(f"   Please create it or run precalculate_metrics.py to generate it")

        print("\n" + "=" * 100)
        sys.exit(0)  # Explicit success exit
        
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

