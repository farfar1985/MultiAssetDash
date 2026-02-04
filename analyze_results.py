import pandas as pd

df = pd.read_csv(r'C:\Users\William Dennis\projects\nexus\results\exhaustive_results_20260203_183524.csv')

print('='*70)
print('  TOP 20 CONFIGURATIONS BY SHARPE')
print('='*70)
top = df.nlargest(20, 'sharpe')[['asset', 'horizon', 'method', 'params', 'lookback', 'sharpe', 'da', 'total_return']]
for i, row in top.iterrows():
    print(f"{row['asset']:12} D+{row['horizon']:<3} | {row['method']:18} | lb={row['lookback']:3} | Sharpe: {row['sharpe']:6.2f} | DA: {row['da']:5.1f}% | Ret: {row['total_return']:+6.1f}%")

print()
print('='*70)
print('  METHOD PERFORMANCE SUMMARY (avg Sharpe by method)')
print('='*70)
method_avg = df.groupby('method')['sharpe'].agg(['mean', 'std', 'max', 'count']).sort_values('mean', ascending=False)
print(method_avg.round(2))

print()
print('='*70)
print('  BEST BY ASSET + HORIZON')
print('='*70)
best_by_asset_horizon = df.loc[df.groupby(['asset', 'horizon'])['sharpe'].idxmax()][
    ['asset', 'horizon', 'method', 'params', 'lookback', 'sharpe', 'da', 'total_return']
].sort_values(['asset', 'horizon'])

for asset in ['Crude_Oil', 'SP500', 'Bitcoin']:
    print(f"\n{asset}:")
    asset_df = best_by_asset_horizon[best_by_asset_horizon['asset'] == asset]
    for i, row in asset_df.iterrows():
        print(f"  D+{row['horizon']:<3} | {row['method']:18} {row['params']:25} | Sharpe: {row['sharpe']:6.2f} | DA: {row['da']:5.1f}% | Ret: {row['total_return']:+6.1f}%")

print()
print('='*70)
print('  PATTERNS')
print('='*70)
print(f"\nBest lookback overall: {df.loc[df['sharpe'].idxmax()]['lookback']}")
print(f"Best method overall: {df.loc[df['sharpe'].idxmax()]['method']}")
print(f"Positive Sharpe configs: {(df['sharpe'] > 0).sum()} / {len(df)} ({100*(df['sharpe'] > 0).mean():.1f}%)")
