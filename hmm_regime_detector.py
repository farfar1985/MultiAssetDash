"""
HMM REGIME DETECTOR FOR NEXUS V2
=================================
Proper Hidden Markov Model implementation for regime detection.
Built for collaboration with Artemis on frontend visualization.

Features:
- Per-asset HMM training with configurable states (2-5)
- Model selection via AIC/BIC
- Regime persistence analysis
- Early warning signals for regime transitions
- API-ready JSON output matching Artemis's RegimeData interface

Created: 2026-02-06
Author: AmiraB
Collaboration: Artemis (frontend/viz)
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy import stats
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
from datetime import datetime
import warnings
import joblib
warnings.filterwarnings('ignore')


class HMMRegimeDetector:
    """
    Hidden Markov Model based regime detector.
    Detects: bull, bear, sideways, high-volatility, low-volatility regimes.
    """
    
    REGIME_NAMES = {
        2: ['risk-on', 'risk-off'],
        3: ['bull', 'bear', 'sideways'],
        4: ['bull', 'bear', 'high-volatility', 'low-volatility'],
        5: ['bull', 'bear', 'sideways', 'high-volatility', 'low-volatility']
    }
    
    def __init__(self, n_states: int = 3, lookback: int = 252):
        """
        Initialize HMM Regime Detector.
        
        Args:
            n_states: Number of hidden states (2-5)
            lookback: Days of history for feature calculation
        """
        self.n_states = n_states
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
        self.regime_labels = self.REGIME_NAMES.get(n_states, [f'regime_{i}' for i in range(n_states)])
        self.feature_names = []
        self.training_info = {}
        
    def extract_features(self, prices: pd.Series) -> pd.DataFrame:
        """
        Extract features for HMM training/prediction.
        
        Features:
        - returns: Daily log returns
        - volatility: Rolling 20-day realized volatility
        - vol_of_vol: Volatility of volatility
        - trend_strength: Price vs 50-day MA
        - autocorr: Return autocorrelation (mean reversion indicator)
        - skew: Rolling return skewness
        """
        df = pd.DataFrame(index=prices.index)
        
        # Log returns
        df['returns'] = np.log(prices / prices.shift(1))
        
        # Realized volatility (20-day)
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Volatility of volatility
        df['vol_of_vol'] = df['volatility'].rolling(20).std()
        
        # Trend strength (price vs 50-day MA)
        ma50 = prices.rolling(50).mean()
        df['trend_strength'] = (prices - ma50) / ma50
        
        # Autocorrelation (5-day rolling)
        df['autocorr'] = df['returns'].rolling(20).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0,
            raw=False
        )
        
        # Rolling skewness
        df['skew'] = df['returns'].rolling(20).skew()
        
        # Fill NaN with 0 for autocorr/skew
        df = df.fillna(0)
        
        self.feature_names = list(df.columns)
        return df
    
    def fit(self, prices: pd.Series, verbose: bool = True) -> dict:
        """
        Train HMM on price data.
        
        Returns:
            Training info including AIC, BIC, log-likelihood
        """
        # Extract features
        features_df = self.extract_features(prices)
        features = features_df.dropna().values
        
        if len(features) < 100:
            raise ValueError(f"Need at least 100 data points, got {len(features)}")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=200,
            random_state=42,
            verbose=False
        )
        
        self.model.fit(features_scaled)
        
        # Calculate model selection criteria
        log_likelihood = self.model.score(features_scaled)
        n_params = self.n_states * (self.n_states - 1) + \
                   self.n_states * len(self.feature_names) + \
                   self.n_states * len(self.feature_names) * (len(self.feature_names) + 1) // 2
        n_samples = len(features_scaled)
        
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        # Label regimes based on characteristics
        self._label_regimes(features_df.dropna(), features_scaled)
        
        self.training_info = {
            'n_samples': n_samples,
            'n_states': self.n_states,
            'n_features': len(self.feature_names),
            'log_likelihood': float(log_likelihood),
            'aic': float(aic),
            'bic': float(bic),
            'converged': self.model.monitor_.converged,
            'n_iter': self.model.monitor_.iter,
            'trained_at': datetime.now().isoformat()
        }
        
        if verbose:
            print(f"HMM Training Complete:")
            print(f"  States: {self.n_states}")
            print(f"  Samples: {n_samples}")
            print(f"  Log-likelihood: {log_likelihood:.2f}")
            print(f"  AIC: {aic:.2f}")
            print(f"  BIC: {bic:.2f}")
            print(f"  Converged: {self.model.monitor_.converged}")
        
        return self.training_info
    
    def _label_regimes(self, features_df: pd.DataFrame, features_scaled: np.ndarray):
        """
        Assign meaningful labels to HMM states based on their characteristics.
        """
        # Get state sequence
        states = self.model.predict(features_scaled)
        
        # Calculate mean characteristics per state
        state_chars = {}
        for state in range(self.n_states):
            mask = states == state
            if mask.sum() > 0:
                state_chars[state] = {
                    'returns': features_df['returns'].values[mask].mean(),
                    'volatility': features_df['volatility'].values[mask].mean(),
                    'trend': features_df['trend_strength'].values[mask].mean(),
                    'count': int(mask.sum())
                }
        
        # Sort states by characteristics
        sorted_by_returns = sorted(state_chars.keys(), key=lambda s: state_chars[s]['returns'], reverse=True)
        sorted_by_vol = sorted(state_chars.keys(), key=lambda s: state_chars[s]['volatility'], reverse=True)
        
        # Assign labels based on n_states
        self.state_to_label = {}
        
        if self.n_states == 2:
            # Simple: high return = risk-on, low return = risk-off
            self.state_to_label[sorted_by_returns[0]] = 'risk-on'
            self.state_to_label[sorted_by_returns[1]] = 'risk-off'
            
        elif self.n_states == 3:
            # Bull (high returns), Bear (low returns), Sideways (middle)
            self.state_to_label[sorted_by_returns[0]] = 'bull'
            self.state_to_label[sorted_by_returns[-1]] = 'bear'
            for s in range(self.n_states):
                if s not in self.state_to_label:
                    self.state_to_label[s] = 'sideways'
                    
        elif self.n_states == 4:
            # Bull, Bear, High-Vol, Low-Vol
            self.state_to_label[sorted_by_returns[0]] = 'bull'
            self.state_to_label[sorted_by_returns[-1]] = 'bear'
            remaining = [s for s in range(self.n_states) if s not in self.state_to_label]
            if len(remaining) >= 2:
                high_vol = max(remaining, key=lambda s: state_chars[s]['volatility'])
                self.state_to_label[high_vol] = 'high-volatility'
                for s in remaining:
                    if s not in self.state_to_label:
                        self.state_to_label[s] = 'low-volatility'
                        
        elif self.n_states == 5:
            # Bull, Bear, Sideways, High-Vol, Low-Vol
            self.state_to_label[sorted_by_returns[0]] = 'bull'
            self.state_to_label[sorted_by_returns[-1]] = 'bear'
            self.state_to_label[sorted_by_vol[0]] = 'high-volatility'
            self.state_to_label[sorted_by_vol[-1]] = 'low-volatility'
            for s in range(self.n_states):
                if s not in self.state_to_label:
                    self.state_to_label[s] = 'sideways'
        
        # Update regime_labels list in order
        self.regime_labels = [self.state_to_label.get(i, f'regime_{i}') for i in range(self.n_states)]
        self.state_characteristics = state_chars
        
    def predict(self, prices: pd.Series) -> dict:
        """
        Predict current regime and return API-compatible output.
        
        Returns:
            RegimeData dict matching Artemis's TypeScript interface
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Extract features
        features_df = self.extract_features(prices)
        features = features_df.dropna().values
        features_scaled = self.scaler.transform(features)
        
        # Get state probabilities for most recent observation
        log_prob, state_sequence = self.model.decode(features_scaled, algorithm='viterbi')
        state_posteriors = self.model.predict_proba(features_scaled)
        
        current_state = state_sequence[-1]
        current_probs = state_posteriors[-1]
        
        # Calculate regime duration (consecutive days in current regime)
        regime_duration = 1
        for i in range(len(state_sequence) - 2, -1, -1):
            if state_sequence[i] == current_state:
                regime_duration += 1
            else:
                break
        
        # Calculate trend strength from features
        trend_strength = features_df['trend_strength'].iloc[-1]
        
        # Build output matching Artemis's RegimeData interface
        result = {
            'currentRegime': self.state_to_label.get(current_state, f'regime_{current_state}'),
            'confidence': float(current_probs[current_state]),
            'probabilities': [
                {
                    'regime': self.state_to_label.get(i, f'regime_{i}'),
                    'probability': float(current_probs[i])
                }
                for i in range(self.n_states)
            ],
            'trendStrength': float(trend_strength),
            'regimeDuration': int(regime_duration),
            # Extra fields for research
            'stateSequence': state_sequence[-20:].tolist(),  # Last 20 days
            'transitionMatrix': self.model.transmat_.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def get_regime_history(self, prices: pd.Series) -> pd.DataFrame:
        """
        Get full regime history for visualization.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        features_df = self.extract_features(prices)
        features = features_df.dropna().values
        features_scaled = self.scaler.transform(features)
        
        state_sequence = self.model.predict(features_scaled)
        state_posteriors = self.model.predict_proba(features_scaled)
        
        # Build history dataframe
        valid_index = features_df.dropna().index
        history = pd.DataFrame(index=valid_index)
        history['state'] = state_sequence
        history['regime'] = [self.state_to_label.get(s, f'regime_{s}') for s in state_sequence]
        history['confidence'] = [state_posteriors[i, state_sequence[i]] for i in range(len(state_sequence))]
        
        # Add all state probabilities
        for i in range(self.n_states):
            label = self.state_to_label.get(i, f'regime_{i}')
            history[f'prob_{label}'] = state_posteriors[:, i]
        
        return history
    
    def save(self, filepath: str):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'n_states': self.n_states,
            'lookback': self.lookback,
            'regime_labels': self.regime_labels,
            'state_to_label': self.state_to_label,
            'feature_names': self.feature_names,
            'training_info': self.training_info,
            'state_characteristics': self.state_characteristics
        }
        joblib.dump(save_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'HMMRegimeDetector':
        """Load trained model from file."""
        save_data = joblib.load(filepath)
        
        detector = cls(
            n_states=save_data['n_states'],
            lookback=save_data['lookback']
        )
        detector.model = save_data['model']
        detector.scaler = save_data['scaler']
        detector.regime_labels = save_data['regime_labels']
        detector.state_to_label = save_data['state_to_label']
        detector.feature_names = save_data['feature_names']
        detector.training_info = save_data['training_info']
        detector.state_characteristics = save_data.get('state_characteristics', {})
        
        return detector


def calibration_study(prices: pd.Series, asset_name: str, state_range: range = range(2, 6)) -> dict:
    """
    Run calibration study to find optimal number of states for an asset.
    
    Returns:
        Dict with results for each n_states and recommendation
    """
    results = {}
    
    for n_states in state_range:
        try:
            detector = HMMRegimeDetector(n_states=n_states)
            info = detector.fit(prices, verbose=False)
            
            # Get regime history for persistence analysis
            history = detector.get_regime_history(prices)
            
            # Calculate regime persistence (average days per regime)
            regime_changes = (history['state'].diff() != 0).sum()
            avg_persistence = len(history) / max(regime_changes, 1)
            
            results[n_states] = {
                'aic': info['aic'],
                'bic': info['bic'],
                'log_likelihood': info['log_likelihood'],
                'converged': info['converged'],
                'avg_regime_persistence': avg_persistence,
                'regime_labels': detector.regime_labels,
                'state_characteristics': detector.state_characteristics
            }
            
            print(f"  {n_states} states: AIC={info['aic']:.1f}, BIC={info['bic']:.1f}, "
                  f"Persistence={avg_persistence:.1f} days")
            
        except Exception as e:
            print(f"  {n_states} states: FAILED - {e}")
            results[n_states] = {'error': str(e)}
    
    # Find optimal by BIC (penalizes complexity more)
    valid_results = {k: v for k, v in results.items() if 'bic' in v}
    if valid_results:
        optimal_states = min(valid_results.keys(), key=lambda k: valid_results[k]['bic'])
        results['optimal_states'] = optimal_states
        results['recommendation'] = f"Use {optimal_states} states (lowest BIC)"
    
    return results


def process_all_assets(data_dir: str = "data", output_dir: str = "regime_models") -> dict:
    """
    Process all Nexus assets and train HMM models.
    """
    from pathlib import Path
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = {}
    
    # Find all asset directories
    asset_dirs = [d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(asset_dirs)} assets to process\n")
    
    for asset_dir in sorted(asset_dirs):
        asset_name = asset_dir.name
        print(f"\n{'='*50}")
        print(f"Processing: {asset_name}")
        print('='*50)
        
        # Load price data
        try:
            # Look for price CSV
            price_files = list(asset_dir.glob("*price*.csv")) + list(asset_dir.glob("*.csv"))
            if not price_files:
                print(f"  No CSV files found, skipping")
                continue
            
            df = pd.read_csv(price_files[0], parse_dates=['date'] if 'date' in pd.read_csv(price_files[0], nrows=1).columns else [0])
            
            # Find price column
            price_col = None
            for col in ['close', 'Close', 'price', 'Price', 'adj_close']:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col is None:
                # Use last numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    price_col = numeric_cols[-1]
                else:
                    print(f"  No price column found, skipping")
                    continue
            
            # Get date column
            date_col = df.columns[0] if df.columns[0] not in df.select_dtypes(include=[np.number]).columns else 'date'
            
            prices = pd.Series(df[price_col].values, index=pd.to_datetime(df[date_col]))
            prices = prices.dropna().sort_index()
            
            print(f"  Loaded {len(prices)} price points from {prices.index[0].date()} to {prices.index[-1].date()}")
            
            if len(prices) < 200:
                print(f"  Insufficient data (<200 points), skipping")
                continue
            
            # Run calibration study
            print(f"\n  Calibration Study:")
            calib_results = calibration_study(prices, asset_name)
            
            # Train optimal model
            optimal_states = calib_results.get('optimal_states', 3)
            print(f"\n  Training optimal model ({optimal_states} states)...")
            
            detector = HMMRegimeDetector(n_states=optimal_states)
            detector.fit(prices, verbose=False)
            
            # Save model
            model_path = output_path / f"{asset_name}_hmm.joblib"
            detector.save(str(model_path))
            
            # Get current regime
            current_regime = detector.predict(prices)
            print(f"\n  Current Regime: {current_regime['currentRegime']} "
                  f"(confidence: {current_regime['confidence']:.1%})")
            print(f"  Regime Duration: {current_regime['regimeDuration']} days")
            
            # Save regime history
            history = detector.get_regime_history(prices)
            history_path = output_path / f"{asset_name}_regime_history.csv"
            history.to_csv(history_path)
            
            all_results[asset_name] = {
                'calibration': calib_results,
                'optimal_states': optimal_states,
                'current_regime': current_regime,
                'model_path': str(model_path),
                'history_path': str(history_path)
            }
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[asset_name] = {'error': str(e)}
    
    # Save summary
    summary_path = output_path / "regime_summary.json"
    with open(summary_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        json.dump(all_results, f, indent=2, default=convert)
    
    print(f"\n\n{'='*50}")
    print(f"SUMMARY")
    print('='*50)
    print(f"Processed: {len(all_results)} assets")
    print(f"Models saved to: {output_path}")
    print(f"Summary: {summary_path}")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Process all assets
        results = process_all_assets()
    else:
        # Demo with sample data
        print("HMM Regime Detector Demo")
        print("="*50)
        print("\nUsage:")
        print("  python hmm_regime_detector.py --all    # Process all Nexus assets")
        print("\nOr import and use programmatically:")
        print("  from hmm_regime_detector import HMMRegimeDetector")
        print("  detector = HMMRegimeDetector(n_states=3)")
        print("  detector.fit(prices)")
        print("  regime = detector.predict(prices)")
