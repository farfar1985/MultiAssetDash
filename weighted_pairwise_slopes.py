# weighted_pairwise_slopes.py
# Implement weighted pairwise slope calculation based on historical pair accuracy
# Created: 2026-02-03

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

# Use standardized metrics
from utils.metrics import calculate_sharpe_ratio_daily


class WeightedPairwiseSlopes:
    """
    Weighted pairwise slope signal generation.
    
    Instead of equal-weighting all horizon pairs, weight by:
    1. Historical accuracy of each pair
    2. Horizon separation (longer spans = more informative)
    3. Recent performance (exponential decay)
    """
    
    def __init__(self, decay: float = 0.95):
        self.decay = decay
        self.pair_accuracy_cache = {}
    
    def calculate_pair_accuracy(self, forecast_df: pd.DataFrame, prices: pd.Series,
                                 h1: int, h2: int, lookback: int = 60) -> float:
        """
        Calculate historical accuracy of a horizon pair's directional signal.
        
        Args:
            forecast_df: DataFrame with forecast columns for each horizon
            prices: Series of actual prices
            h1, h2: Horizon pair (h2 > h1)
            lookback: Number of days to evaluate
        
        Returns:
            Accuracy as float (0-1)
        """
        col1 = f'd{h1}' if f'd{h1}' in forecast_df.columns else str(h1)
        col2 = f'd{h2}' if f'd{h2}' in forecast_df.columns else str(h2)
        
        if col1 not in forecast_df.columns or col2 not in forecast_df.columns:
            return 0.5  # Default neutral weight
        
        common_idx = forecast_df.index.intersection(prices.index)
        if len(common_idx) < lookback:
            return 0.5
        
        # Use recent data
        recent_idx = common_idx[-lookback:]
        
        correct = 0
        total = 0
        
        for i in range(1, len(recent_idx)):
            date = recent_idx[i]
            prev_date = recent_idx[i-1]
            
            # Get pair drift (forecast difference)
            f1 = forecast_df.loc[date, col1]
            f2 = forecast_df.loc[date, col2]
            
            if pd.isna(f1) or pd.isna(f2):
                continue
            
            drift = f2 - f1
            pair_signal = 1 if drift > 0 else -1  # Bullish or Bearish
            
            # Actual direction (next day's move)
            if date in prices.index and prev_date in prices.index:
                actual_move = prices.loc[date] - prices.loc[prev_date]
                actual_dir = 1 if actual_move > 0 else -1
                
                if pair_signal == actual_dir:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.5
    
    def calculate_all_pair_weights(self, forecast_df: pd.DataFrame, prices: pd.Series,
                                    horizons: List[int], lookback: int = 60) -> Dict[Tuple[int, int], float]:
        """
        Calculate weights for all horizon pairs.
        
        Returns:
            Dict mapping (h1, h2) -> weight
        """
        weights = {}
        
        for i, h1 in enumerate(horizons):
            for h2 in horizons[i+1:]:
                # Component 1: Historical accuracy
                accuracy = self.calculate_pair_accuracy(forecast_df, prices, h1, h2, lookback)
                
                # Component 2: Horizon separation (log scale)
                separation_weight = np.log1p(h2 - h1) / np.log1p(max(horizons))
                
                # Combine: accuracy * separation
                # Higher accuracy + larger separation = higher weight
                weight = accuracy * (1 + separation_weight)
                
                weights[(h1, h2)] = weight
                self.pair_accuracy_cache[(h1, h2)] = accuracy
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def calculate_weighted_signal(self, forecast_df: pd.DataFrame, prices: pd.Series,
                                   horizons: List[int], threshold: float = 0.3,
                                   lookback: int = 60) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate weighted pairwise slope signals.
        
        Returns:
            signals: Series of 'BULLISH', 'BEARISH', 'NEUTRAL'
            net_probs: Series of net probability values
        """
        # Calculate pair weights
        weights = self.calculate_all_pair_weights(forecast_df, prices, horizons, lookback)
        
        signals = []
        net_probs = []
        dates = []
        
        for date in forecast_df.index:
            weighted_bull = 0.0
            weighted_bear = 0.0
            
            for (h1, h2), weight in weights.items():
                col1 = f'd{h1}' if f'd{h1}' in forecast_df.columns else str(h1)
                col2 = f'd{h2}' if f'd{h2}' in forecast_df.columns else str(h2)
                
                if col1 not in forecast_df.columns or col2 not in forecast_df.columns:
                    continue
                
                f1 = forecast_df.loc[date, col1]
                f2 = forecast_df.loc[date, col2]
                
                if pd.isna(f1) or pd.isna(f2):
                    continue
                
                drift = f2 - f1
                
                if drift > 0:
                    weighted_bull += weight
                else:
                    weighted_bear += weight
            
            total_weight = weighted_bull + weighted_bear
            if total_weight > 0:
                net_prob = (weighted_bull - weighted_bear) / total_weight
            else:
                net_prob = 0.0
            
            # Convert to signal
            if net_prob > threshold:
                signal = 'BULLISH'
            elif net_prob < -threshold:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'
            
            signals.append(signal)
            net_probs.append(net_prob)
            dates.append(date)
        
        return pd.Series(signals, index=dates), pd.Series(net_probs, index=dates)
    
    def compare_weighted_vs_unweighted(self, forecast_df: pd.DataFrame, prices: pd.Series,
                                        horizons: List[int], threshold: float = 0.3) -> Dict:
        """
        Compare weighted vs unweighted pairwise slope performance.
        """
        from ensemble_methods import EnsembleMethods
        
        # Weighted signals
        weighted_signals, weighted_probs = self.calculate_weighted_signal(
            forecast_df, prices, horizons, threshold
        )
        
        # Unweighted signals (original method)
        unweighted_signals, unweighted_probs = self._calculate_unweighted_signal(
            forecast_df, horizons, threshold
        )
        
        # Calculate performance for both
        weighted_perf = self._evaluate_signals(weighted_signals, prices)
        unweighted_perf = self._evaluate_signals(unweighted_signals, prices)
        
        return {
            'weighted': weighted_perf,
            'unweighted': unweighted_perf,
            'improvement': weighted_perf.get('sharpe', 0) - unweighted_perf.get('sharpe', 0),
            'pair_accuracies': self.pair_accuracy_cache
        }
    
    def _calculate_unweighted_signal(self, forecast_df: pd.DataFrame, 
                                      horizons: List[int], threshold: float) -> Tuple[pd.Series, pd.Series]:
        """Original equal-weighted pairwise slope method."""
        signals = []
        net_probs = []
        
        for date in forecast_df.index:
            bullish = 0
            bearish = 0
            total = 0
            
            for i, h1 in enumerate(horizons):
                for h2 in horizons[i+1:]:
                    col1 = f'd{h1}' if f'd{h1}' in forecast_df.columns else str(h1)
                    col2 = f'd{h2}' if f'd{h2}' in forecast_df.columns else str(h2)
                    
                    if col1 not in forecast_df.columns or col2 not in forecast_df.columns:
                        continue
                    
                    f1 = forecast_df.loc[date, col1]
                    f2 = forecast_df.loc[date, col2]
                    
                    if pd.isna(f1) or pd.isna(f2):
                        continue
                    
                    drift = f2 - f1
                    if drift > 0:
                        bullish += 1
                    else:
                        bearish += 1
                    total += 1
            
            net_prob = (bullish - bearish) / total if total > 0 else 0
            
            if net_prob > threshold:
                signal = 'BULLISH'
            elif net_prob < -threshold:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'
            
            signals.append(signal)
            net_probs.append(net_prob)
        
        return pd.Series(signals, index=forecast_df.index), pd.Series(net_probs, index=forecast_df.index)
    
    def _evaluate_signals(self, signals: pd.Series, prices: pd.Series) -> Dict:
        """Evaluate signal performance."""
        common_idx = signals.index.intersection(prices.index)
        sigs = signals.loc[common_idx]
        pxs = prices.loc[common_idx]
        
        returns = pxs.pct_change()
        
        # Position based on signal
        position = sigs.map({'BULLISH': 1, 'BEARISH': -1, 'NEUTRAL': 0})
        
        # Strategy returns (signal from previous day applied to today's return)
        strategy_returns = (position.shift(1) * returns).dropna()
        
        if len(strategy_returns) < 2 or strategy_returns.std() == 0:
            return {'sharpe': 0, 'total_return': 0, 'win_rate': 0}

        # Use standardized Sharpe calculation
        sharpe = calculate_sharpe_ratio_daily(strategy_returns.values)
        total_return = strategy_returns.sum() * 100
        win_rate = (strategy_returns > 0).mean() * 100
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'win_rate': win_rate,
            'n_trades': (position.diff() != 0).sum()
        }


if __name__ == "__main__":
    print("Weighted Pairwise Slopes - Cross-Horizon Weighting")
    print("=" * 50)
    print("\nThis module weights horizon pairs by:")
    print("  1. Historical accuracy")
    print("  2. Horizon separation")
    print("\nUse compare_weighted_vs_unweighted() to test improvement.")
