"""Quick meta-ensemble test"""
import numpy as np
import joblib
from pathlib import Path

# Load all horizon data
data_dir = Path('data/1866_Crude_Oil/horizons_wide')
horizons = {}
for h in [1,2,3,5,7,10]:
    f = data_dir / f'horizon_{h}.joblib'
    if f.exists():
        data = joblib.load(f)
        X = data['X'].values if hasattr(data['X'], 'values') else np.array(data['X'])
        y = data['y'].values if hasattr(data['y'], 'values') else np.array(data['y'])
        horizons[h] = {'X': X, 'y': y}

y = horizons[5]['y']

# Handle NaN - forward fill
y_clean = np.copy(y)
for i in range(len(y_clean)):
    if np.isnan(y_clean[i]):
        y_clean[i] = y_clean[i-1] if i > 0 else 70.0

actual_returns = np.diff(y_clean)
print(f'Clean data: {len(y_clean)} days, returns: {actual_returns.min():.2f} to {actual_returns.max():.2f}')

def pairwise_slopes_signal(horizons, h_list, threshold):
    signals = []
    for i in range(len(horizons[h_list[0]]['y'])):
        slopes = []
        for j in range(len(h_list)):
            for k in range(j+1, len(h_list)):
                h1, h2 = h_list[j], h_list[k]
                p1 = horizons[h1]['X'][i].mean()
                p2 = horizons[h2]['X'][i].mean()
                slope = (p2 - p1) / (h2 - h1)
                slopes.append(slope)
        avg_slope = np.mean(slopes)
        if avg_slope > threshold:
            signals.append(1)
        elif avg_slope < -threshold:
            signals.append(-1)
        else:
            signals.append(0)
    return np.array(signals)

def calc_sharpe(signals, returns):
    mask = signals != 0
    if mask.sum() < 5:
        return 0, 0, 0, 0
    trade_returns = signals[mask] * returns[mask]
    sharpe = trade_returns.mean() / (trade_returns.std() + 1e-8) * np.sqrt(252)
    total = trade_returns.sum()
    win = (trade_returns > 0).mean()
    return sharpe, total, win, mask.sum()

print('\n' + '='*70)
print('PAIRWISE SLOPES GRID SEARCH')
print('='*70)

configs = []
for h_combo in [[5,7,10], [1,2,3], [1,5,10], [3,5,7], [1,3,7,10], [5,10], [7,10], [1,2], [2,3]]:
    for thresh in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]:
        try:
            sig = pairwise_slopes_signal(horizons, h_combo, thresh)[:-1]
            s, r, w, t = calc_sharpe(sig, actual_returns)
            if not np.isnan(s) and t > 10:
                configs.append((str(h_combo), thresh, s, r, w, t, sig))
        except:
            pass

# Sort by Sharpe
configs_sorted = sorted(configs, key=lambda x: -x[2])[:20]

print(f"\n{'Config':<25} {'Thresh':<8} {'Sharpe':>8} {'Return':>8} {'Win%':>7} {'Trades':>7}")
print('-'*70)
for c in configs_sorted:
    print(f"{c[0]:<25} {c[1]:<8.3f} {c[2]:>8.2f} {c[3]:>8.1f} {c[4]*100:>6.1f}% {c[5]:>7}")

# =============================================================
# META-ENSEMBLE: Stack top configs
# =============================================================
print('\n' + '='*70)
print('META-ENSEMBLE: Stacking Top Configs')
print('='*70)

# Get top 5 diverse configs
top5 = configs_sorted[:5]
sigs = [c[6] for c in top5]

print('\nTop 5 methods:')
for c in top5:
    print(f"  {c[0]} t={c[1]} -> Sharpe {c[2]:.2f}")

# Majority vote (3/5)
print('\n--- Meta-Ensemble Results ---')
meta_vote = np.sign(sum(sigs))
s, r, w, t = calc_sharpe(meta_vote, actual_returns)
print(f"Majority Vote (any):  Sharpe={s:6.2f}, Return={r:6.1f}, Win={w*100:5.1f}%, Trades={t}")

# 3/5 agreement
meta3 = np.zeros(len(sigs[0]))
for i in range(len(sigs[0])):
    votes = [sigs[j][i] for j in range(5)]
    bull = sum(v == 1 for v in votes)
    bear = sum(v == -1 for v in votes)
    if bull >= 3:
        meta3[i] = 1
    elif bear >= 3:
        meta3[i] = -1

s, r, w, t = calc_sharpe(meta3, actual_returns)
print(f"Agreement 3/5:        Sharpe={s:6.2f}, Return={r:6.1f}, Win={w*100:5.1f}%, Trades={t}")

# 4/5 agreement
meta4 = np.zeros(len(sigs[0]))
for i in range(len(sigs[0])):
    votes = [sigs[j][i] for j in range(5)]
    bull = sum(v == 1 for v in votes)
    bear = sum(v == -1 for v in votes)
    if bull >= 4:
        meta4[i] = 1
    elif bear >= 4:
        meta4[i] = -1

s, r, w, t = calc_sharpe(meta4, actual_returns)
print(f"Agreement 4/5:        Sharpe={s:6.2f}, Return={r:6.1f}, Win={w*100:5.1f}%, Trades={t}")

# 5/5 full agreement
meta5 = np.zeros(len(sigs[0]))
for i in range(len(sigs[0])):
    votes = [sigs[j][i] for j in range(5)]
    if all(v == 1 for v in votes):
        meta5[i] = 1
    elif all(v == -1 for v in votes):
        meta5[i] = -1

s, r, w, t = calc_sharpe(meta5, actual_returns)
print(f"Full Agreement 5/5:   Sharpe={s:6.2f}, Return={r:6.1f}, Win={w*100:5.1f}%, Trades={t}")

# Weighted by Sharpe
weights = np.array([c[2] for c in top5])
weights = weights / weights.sum()
meta_wt = np.zeros(len(sigs[0]))
for i in range(len(sigs[0])):
    weighted_vote = sum(sigs[j][i] * weights[j] for j in range(5))
    if weighted_vote > 0.3:
        meta_wt[i] = 1
    elif weighted_vote < -0.3:
        meta_wt[i] = -1

s, r, w, t = calc_sharpe(meta_wt, actual_returns)
print(f"Sharpe-Weighted:      Sharpe={s:6.2f}, Return={r:6.1f}, Win={w*100:5.1f}%, Trades={t}")

print('\n' + '='*70)
print('WINNER')
print('='*70)
best = configs_sorted[0]
print(f"Best single: {best[0]} t={best[1]}")
print(f"  Sharpe: {best[2]:.2f}")
print(f"  Return: {best[3]:.1f}")
print(f"  Win Rate: {best[4]*100:.1f}%")
print(f"  Trades: {best[5]}")
