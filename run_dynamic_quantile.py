# q_ensemble_sandbox/run_dynamic_quantile.py
# SANDBOXED - Core Q Ensemble logic for Bitcoin
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import gc
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config_sandbox as cfg

# --- Meta-Dynamic Configuration ---
LOOKBACK_WINDOW = 60
QUANTILE_CANDIDATES = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
AGGREGATORS = ['mean', 'median']
SCORING_STRATEGIES = {
    'Balanced': {'mse': 0.25, 'acc': 0.25, 'pnl': 0.25, 'shp': 0.25},
    'Profit_Heavy': {'mse': 0.0, 'acc': 0.0, 'pnl': 0.7, 'shp': 0.3},
    'Accuracy_Heavy': {'mse': 0.7, 'acc': 0.3, 'pnl': 0.0, 'shp': 0.0},
    'Risk_Heavy': {'mse': 0.1, 'acc': 0.1, 'pnl': 0.1, 'shp': 0.7},
    'MSE_Only': {'mse': 1.0, 'acc': 0.0, 'pnl': 0.0, 'shp': 0.0}
}

def calculate_metrics(predictions, actuals):
    errors = predictions.sub(actuals, axis=0)
    mse = (errors ** 2).mean()
    
    prev_actuals = actuals.shift(1)
    market_ret = (actuals - prev_actuals) / prev_actuals
    
    actual_dir = np.sign(market_ret)
    pred_dir = np.sign(predictions.sub(prev_actuals, axis=0))
    matches = pred_dir.eq(actual_dir, axis=0)
    dir_acc = matches.mean()
    
    signals = pred_dir
    strat_rets = signals.mul(market_ret, axis=0)
    total_pnl = strat_rets.sum()
    
    mean_ret = strat_rets.mean()
    std_ret = strat_rets.std()
    sharpe = (mean_ret / std_ret.replace(0, np.nan)) * np.sqrt(252)
    sharpe = sharpe.fillna(0)
    
    cum_ret = (1 + strat_rets).cumprod()
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()
    
    metrics = pd.DataFrame({
        'mse': mse,
        'dir_acc': dir_acc,
        'pnl': total_pnl,
        'sharpe': sharpe,
        'max_dd': max_dd
    })
    return metrics

def score_models(metrics, weights):
    r_mse = metrics['mse'].rank(ascending=True, pct=True)
    r_acc = metrics['dir_acc'].rank(ascending=False, pct=True)
    r_pnl = metrics['pnl'].rank(ascending=False, pct=True)
    r_shp = metrics['sharpe'].rank(ascending=False, pct=True)
    
    metrics['score'] = (
        (weights['mse'] * r_mse) + 
        (weights['acc'] * r_acc) + 
        (weights['pnl'] * r_pnl) + 
        (weights['shp'] * r_shp)
    )
    return metrics.sort_values('score')

def evaluate_ensemble_quality(ens_preds, actuals):
    mse = ((ens_preds - actuals)**2).mean()
    prev = actuals.shift(1)
    mkt = (actuals - prev) / prev
    sig = np.sign(ens_preds - prev)
    hits = (sig == np.sign(mkt)).mean()
    rets = sig * mkt
    pnl = rets.sum()
    cum = (1 + rets).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min()
    
    score = (pnl * 10) + (hits * 10) + (dd * 5) - (mse * 1)
    return score

def run_single_model_horizon(h, X, y):
    """Handle horizons with only 1 model - no ensemble needed."""
    common_dates = X.index.intersection(y.index).sort_values()
    
    # Get the single column name
    if isinstance(X, pd.Series):
        predictions = X.loc[common_dates]
        model_name = X.name
    else:
        col = X.columns[0]
        predictions = X.loc[common_dates, col]
        model_name = col
    
    actuals = y.loc[common_dates]
    live_target_date = X.index[-1]
    
    out_file = os.path.join(cfg.DATA_DIR, f'forecast_d{h}.csv')
    
    # Create output dataframe
    results = []
    for date in common_dates:
        results.append({
            'date': date,
            'prediction': predictions.loc[date],
            'actual': actuals.loc[date],
            'best_strat': 'single_model',
            'best_agg': 'none',
            'best_q': 1.0,
            'best_child_prediction': predictions.loc[date]
        })
    
    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"  -> Saved {len(df)} records using single model: {model_name}")
    
    # Generate live forecast
    live_pred = predictions.iloc[-1] if live_target_date in predictions.index else predictions.iloc[-1]
    
    return {
        'horizon_days': h,
        'target_date': str(live_target_date.date()),
        'predicted_price': float(live_pred)
    }

def run_horizon(h):
    print(f"\n--- Running Meta-Dynamic Quantile for Horizon D+{h} ---")
    path = os.path.join(cfg.DATA_DIR, 'horizons_wide', f'horizon_{h}.joblib')
    if not os.path.exists(path): 
        print(f"  -> File not found: {path}")
        return None

    data = joblib.load(path)
    X = data['X']
    y = data['y']
    
    # Handle single-model case - no ensemble needed, just use the single model
    n_models = X.shape[1] if len(X.shape) > 1 else 1
    if n_models == 1:
        print(f"  [WARN] Only 1 model found for D+{h} - using single model predictions directly")
        return run_single_model_horizon(h, X, y)
    
    common_dates = X.index.intersection(y.index).sort_values()
    
    X_train_all = X.loc[common_dates]
    y_train_all = y.loc[common_dates]
    live_target_date = X.index[-1]
    
    out_file = os.path.join(cfg.DATA_DIR, f'forecast_d{h}.csv')
    
    # Checkpoint Logic
    start_idx = LOOKBACK_WINDOW
    processed_dates = set()
    
    if os.path.exists(out_file):
        try:
            existing_df = pd.read_csv(out_file, parse_dates=['date'])
            if not existing_df.empty:
                processed_dates = set(existing_df['date'])
                print(f"  Resuming from {len(processed_dates)} existing records...")
        except:
            print("  Error reading existing file, starting over.")
    else:
        pd.DataFrame(columns=['date', 'prediction', 'actual', 'best_strat', 'best_agg', 'best_q', 'best_child_prediction']).to_csv(out_file, index=False)

    # Walk-Forward Loop
    for i in tqdm(range(start_idx, len(common_dates)), desc=f"D+{h}"):
        today = common_dates[i]
        
        if today in processed_dates:
            continue
            
        window_start = common_dates[i - LOOKBACK_WINDOW]
        window_end = common_dates[i-1]
        
        X_tr = X_train_all.loc[window_start:window_end]
        y_tr = y_train_all.loc[window_start:window_end]
        
        # FIX: Split into train/validate to prevent data leakage (Bug #1)
        # Previously: models were scored AND validated on the same data → overfitting
        # Now: score models on fit portion (70%), validate ensemble on holdout (30%)
        split_point = int(len(X_tr) * 0.7)
        X_fit, X_val = X_tr.iloc[:split_point], X_tr.iloc[split_point:]
        y_fit, y_val = y_tr.iloc[:split_point], y_tr.iloc[split_point:]
        
        # Fall back to full window if splits are too small
        if len(X_fit) < 10 or len(X_val) < 5:
            X_fit, X_val = X_tr, X_tr
            y_fit, y_val = y_tr, y_tr
        
        best_combo = None
        best_combo_score = -float('inf')
        
        for strat_name, weights in SCORING_STRATEGIES.items():
            metrics = calculate_metrics(X_fit, y_fit)  # Score on fit portion
            ranked = score_models(metrics, weights)
            
            for agg in AGGREGATORS:
                for q in QUANTILE_CANDIDATES:
                    top_n = max(1, int(len(ranked) * q))
                    top_cols = ranked.head(top_n).index
                    
                    if agg == 'mean': ens_hist = X_val[top_cols].mean(axis=1)  # Ensemble on validate
                    else: ens_hist = X_val[top_cols].median(axis=1)
                    
                    score = evaluate_ensemble_quality(ens_hist, y_val)  # Evaluate on validate
                    
                    if score > best_combo_score:
                        best_combo_score = score
                        best_combo = {'strat': strat_name, 'weights': weights, 'agg': agg, 'q': q, 'top': top_cols}
        
        # Safety check: if no valid combination found, skip this date
        if best_combo is None:
            print(f"  [WARN] No valid ensemble found for {today}, skipping...")
            continue
            
        top_cols = best_combo['top']
        today_preds = X_train_all.loc[today, top_cols]
        val = today_preds.mean() if best_combo['agg'] == 'mean' else today_preds.median()
        
        best_child_id = ranked.index[0]
        best_child_val = X_train_all.loc[today, best_child_id]
        
        row = {
            'date': today,
            'prediction': val,
            'actual': y_train_all.loc[today],
            'best_strat': best_combo['strat'],
            'best_agg': best_combo['agg'],
            'best_q': best_combo['q'],
            'best_child_prediction': best_child_val
        }
        
        pd.DataFrame([row]).to_csv(out_file, mode='a', header=False, index=False)
        
        if i % 10 == 0: gc.collect()

    # Live Prediction Step
    X_live_tr = X.iloc[:-1].tail(LOOKBACK_WINDOW)
    y_live_tr = y.iloc[:-1].tail(LOOKBACK_WINDOW)
    
    best_combo = None
    best_score = -float('inf')
    
    for strat_name, weights in SCORING_STRATEGIES.items():
        metrics = calculate_metrics(X_live_tr, y_live_tr)
        ranked = score_models(metrics, weights)
        for agg in AGGREGATORS:
            for q in QUANTILE_CANDIDATES:
                top_n = max(1, int(len(ranked) * q))
                top = ranked.head(top_n).index
                if agg == 'mean': ens = X_live_tr[top].mean(axis=1)
                else: ens = X_live_tr[top].median(axis=1)
                score = evaluate_ensemble_quality(ens, y_live_tr)
                if score > best_score:
                    best_score = score
                    best_combo = {'agg': agg, 'q': q, 'top': top}
    
    # Safety check: if no valid live combo found, skip this horizon
    if best_combo is None:
        print(f"  [WARN] D+{h} has insufficient data for live prediction, skipping...")
        return None
                    
    live_preds = X.loc[live_target_date, best_combo['top']]
    if best_combo['agg'] == 'mean': live_val = live_preds.mean()
    else: live_val = live_preds.median()
    
    print(f"D+{h} LIVE: {live_val:.2f} (Using {best_combo['q']*100}% {best_combo['agg']})")
    
    return {'horizon_days': h, 'predicted_price': live_val, 'target_date': str(live_target_date.date())}

def discover_available_horizons():
    """Dynamically discover available horizons from horizons_wide directory."""
    import glob
    import re
    
    horizons_dir = os.path.join(cfg.DATA_DIR, 'horizons_wide')
    horizon_files = glob.glob(os.path.join(horizons_dir, 'horizon_*.joblib'))
    
    horizons = []
    for path in horizon_files:
        filename = os.path.basename(path)
        match = re.search(r'horizon_(\d+)\.joblib', filename)
        if match:
            horizons.append(int(match.group(1)))
    
    return sorted(horizons)

def run_all():
    print(f"\n{'='*50}")
    print(f"Q ENSEMBLE - {cfg.PROJECT_NAME} (Project {cfg.PROJECT_ID})")
    print(f"{'='*50}\n")
    
    # DYNAMIC HORIZON DISCOVERY
    available_horizons = discover_available_horizons()
    print(f"[INFO] Discovered {len(available_horizons)} horizons: {available_horizons}\n")
    
    live_forecasts = []
    for h in available_horizons:
        res = run_horizon(h)
        if res: live_forecasts.append(res)
    
    if live_forecasts:
        out_path = os.path.join(cfg.DATA_DIR, 'live_forecast.json')
        with open(out_path, 'w') as f:
            json.dump({'predictions': live_forecasts}, f, indent=2)
        print(f"\nLive forecasts saved to {out_path}")

if __name__ == "__main__":
    run_all()

