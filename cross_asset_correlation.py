"""
Cross-Asset Regime Correlation Analysis
========================================
Analyze which assets tend to shift regimes together (contagion effects).

Key questions:
1. When asset A shifts regime, which other assets follow within N days?
2. Which assets are "leaders" (shift first) vs "followers"?
3. Are there regime correlation clusters?

Author: AmiraB
Date: 2026-02-06
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')


class CrossAssetCorrelation:
    """Analyze regime change correlations across assets."""
    
    def __init__(self, models_dir: str = "regime_models"):
        self.models_dir = Path(models_dir)
        self.regime_histories: Dict[str, pd.DataFrame] = {}
        self.summary: Dict = {}
        self._load_data()
    
    def _load_data(self):
        """Load regime summary and history files."""
        # Load summary
        summary_path = self.models_dir / "regime_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
        
        # Load regime histories
        for history_file in self.models_dir.glob("*_regime_history.csv"):
            asset_key = history_file.stem.replace("_regime_history", "")
            try:
                df = pd.read_csv(history_file)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                self.regime_histories[asset_key] = df
            except Exception as e:
                print(f"Failed to load {asset_key}: {e}")
        
        print(f"Loaded regime histories for {len(self.regime_histories)} assets")
    
    def detect_regime_changes(self, asset_key: str) -> pd.DataFrame:
        """
        Detect regime change points for an asset.
        
        Returns DataFrame with change dates and transition details.
        """
        if asset_key not in self.regime_histories:
            return pd.DataFrame()
        
        df = self.regime_histories[asset_key].copy()
        
        if 'regime' not in df.columns:
            return pd.DataFrame()
        
        # Find where regime changes
        df['prev_regime'] = df['regime'].shift(1)
        df['changed'] = df['regime'] != df['prev_regime']
        
        changes = df[df['changed'] & df['prev_regime'].notna()].copy()
        changes['from_regime'] = changes['prev_regime']
        changes['to_regime'] = changes['regime']
        
        return changes[['from_regime', 'to_regime']]
    
    def calculate_lead_lag(self, asset_a: str, asset_b: str, 
                           max_lag_days: int = 10) -> Dict:
        """
        Calculate lead-lag relationship between two assets.
        
        Returns which asset tends to lead regime changes and by how many days.
        """
        changes_a = self.detect_regime_changes(asset_a)
        changes_b = self.detect_regime_changes(asset_b)
        
        if changes_a.empty or changes_b.empty:
            return {'error': 'Insufficient data'}
        
        dates_a = changes_a.index.tolist()
        dates_b = changes_b.index.tolist()
        
        # For each change in A, find nearest change in B
        a_leads = []  # A changes before B
        b_leads = []  # B changes before A
        
        for date_a in dates_a:
            # Find closest B change
            min_diff = None
            for date_b in dates_b:
                diff = (date_b - date_a).days
                if abs(diff) <= max_lag_days:
                    if min_diff is None or abs(diff) < abs(min_diff):
                        min_diff = diff
            
            if min_diff is not None:
                if min_diff > 0:
                    a_leads.append(min_diff)
                elif min_diff < 0:
                    b_leads.append(abs(min_diff))
        
        total_related = len(a_leads) + len(b_leads)
        
        if total_related == 0:
            return {
                'asset_a': asset_a,
                'asset_b': asset_b,
                'relationship': 'independent',
                'correlation_strength': 0
            }
        
        a_leads_pct = len(a_leads) / total_related
        b_leads_pct = len(b_leads) / total_related
        
        if a_leads_pct > 0.6:
            leader = asset_a
            avg_lead = np.mean(a_leads) if a_leads else 0
        elif b_leads_pct > 0.6:
            leader = asset_b
            avg_lead = np.mean(b_leads) if b_leads else 0
        else:
            leader = 'concurrent'
            avg_lead = 0
        
        return {
            'asset_a': asset_a,
            'asset_b': asset_b,
            'leader': leader,
            'a_leads_count': len(a_leads),
            'b_leads_count': len(b_leads),
            'a_leads_pct': round(a_leads_pct, 2),
            'b_leads_pct': round(b_leads_pct, 2),
            'avg_lead_days': round(avg_lead, 1),
            'correlation_strength': round(total_related / max(len(dates_a), len(dates_b)), 2)
        }
    
    def build_correlation_matrix(self, max_lag_days: int = 10) -> pd.DataFrame:
        """
        Build a correlation matrix showing regime change synchronization.
        
        Entry (i,j) = correlation strength between assets i and j
        """
        assets = list(self.regime_histories.keys())
        n = len(assets)
        
        matrix = np.zeros((n, n))
        
        for i, asset_a in enumerate(assets):
            for j, asset_b in enumerate(assets):
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j:
                    result = self.calculate_lead_lag(asset_a, asset_b, max_lag_days)
                    strength = result.get('correlation_strength', 0)
                    matrix[i, j] = strength
                    matrix[j, i] = strength
        
        return pd.DataFrame(matrix, index=assets, columns=assets)
    
    def find_leaders_and_followers(self) -> Dict:
        """
        Identify which assets tend to lead regime changes vs follow.
        
        Returns ranking of assets by leadership score.
        """
        assets = list(self.regime_histories.keys())
        leadership_scores = defaultdict(lambda: {'leads': 0, 'follows': 0, 'total': 0})
        
        for i, asset_a in enumerate(assets):
            for j, asset_b in enumerate(assets):
                if i >= j:
                    continue
                
                result = self.calculate_lead_lag(asset_a, asset_b)
                if 'leader' not in result:
                    continue
                
                leader = result['leader']
                if leader == asset_a:
                    leadership_scores[asset_a]['leads'] += 1
                    leadership_scores[asset_b]['follows'] += 1
                elif leader == asset_b:
                    leadership_scores[asset_b]['leads'] += 1
                    leadership_scores[asset_a]['follows'] += 1
                
                leadership_scores[asset_a]['total'] += 1
                leadership_scores[asset_b]['total'] += 1
        
        # Calculate leadership ratio
        rankings = []
        for asset, scores in leadership_scores.items():
            if scores['total'] > 0:
                ratio = scores['leads'] / scores['total']
            else:
                ratio = 0.5
            
            rankings.append({
                'asset': asset,
                'leads': scores['leads'],
                'follows': scores['follows'],
                'total_pairs': scores['total'],
                'leadership_ratio': round(ratio, 2),
                'role': 'LEADER' if ratio > 0.6 else 'FOLLOWER' if ratio < 0.4 else 'NEUTRAL'
            })
        
        rankings.sort(key=lambda x: x['leadership_ratio'], reverse=True)
        return {
            'rankings': rankings,
            'leaders': [r for r in rankings if r['role'] == 'LEADER'],
            'followers': [r for r in rankings if r['role'] == 'FOLLOWER']
        }
    
    def find_clusters(self, threshold: float = 0.3) -> List[List[str]]:
        """
        Find clusters of assets that tend to shift regimes together.
        
        Uses hierarchical clustering on the correlation matrix.
        """
        corr_matrix = self.build_correlation_matrix()
        
        if corr_matrix.empty:
            return []
        
        # Convert correlation to distance
        dist_matrix = 1 - corr_matrix.values
        np.fill_diagonal(dist_matrix, 0)
        
        # Hierarchical clustering
        try:
            condensed = squareform(dist_matrix)
            linkage = hierarchy.linkage(condensed, method='average')
            clusters = hierarchy.fcluster(linkage, t=1-threshold, criterion='distance')
        except Exception as e:
            print(f"Clustering failed: {e}")
            return []
        
        # Group assets by cluster
        cluster_groups = defaultdict(list)
        for asset, cluster_id in zip(corr_matrix.index, clusters):
            cluster_groups[cluster_id].append(asset)
        
        return list(cluster_groups.values())
    
    def analyze_contagion_paths(self, source_asset: str, 
                                max_depth: int = 3) -> Dict:
        """
        Trace potential contagion paths from a source asset.
        
        Shows which assets might be affected and in what order.
        """
        if source_asset not in self.regime_histories:
            return {'error': f'Asset {source_asset} not found'}
        
        assets = list(self.regime_histories.keys())
        
        # Build adjacency with lead-lag info
        edges = []
        for asset in assets:
            if asset == source_asset:
                continue
            
            result = self.calculate_lead_lag(source_asset, asset)
            if result.get('leader') == source_asset and result.get('correlation_strength', 0) > 0.2:
                edges.append({
                    'to': asset,
                    'avg_lag': result.get('avg_lead_days', 0),
                    'strength': result.get('correlation_strength', 0)
                })
        
        # Sort by lag time (first affected first)
        edges.sort(key=lambda x: x['avg_lag'])
        
        return {
            'source': source_asset,
            'direct_followers': edges,
            'total_potentially_affected': len(edges),
            'avg_propagation_time': round(np.mean([e['avg_lag'] for e in edges]), 1) if edges else 0
        }
    
    def generate_report(self) -> Dict:
        """Generate comprehensive cross-asset correlation report."""
        print("Analyzing regime change correlations...")
        
        # Get leaders and followers
        leadership = self.find_leaders_and_followers()
        
        # Get clusters
        clusters = self.find_clusters()
        
        # Get correlation matrix
        corr_matrix = self.build_correlation_matrix()
        
        # Find strongest correlations
        strongest = []
        assets = list(corr_matrix.index)
        for i, a in enumerate(assets):
            for j, b in enumerate(assets):
                if i < j:
                    strength = corr_matrix.loc[a, b]
                    if strength > 0.3:
                        strongest.append({
                            'pair': f"{a} <-> {b}",
                            'correlation': round(strength, 2)
                        })
        
        strongest.sort(key=lambda x: x['correlation'], reverse=True)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_assets': len(self.regime_histories),
            'leadership': leadership,
            'clusters': clusters,
            'strongest_correlations': strongest[:10],
            'correlation_matrix': corr_matrix.to_dict()
        }
    
    def print_report(self):
        """Print correlation report to console."""
        report = self.generate_report()
        
        print("=" * 60)
        print("CROSS-ASSET REGIME CORRELATION ANALYSIS")
        print("=" * 60)
        print(f"\nTotal assets analyzed: {report['total_assets']}")
        
        print("\n" + "-" * 40)
        print("LEADERSHIP RANKINGS")
        print("-" * 40)
        print("(Which assets lead regime changes)")
        
        for r in report['leadership']['rankings'][:5]:
            arrow = ">>>" if r['role'] == 'LEADER' else "<<<" if r['role'] == 'FOLLOWER' else "---"
            print(f"{arrow} {r['asset']}: {r['leadership_ratio']:.0%} leader ({r['leads']}/{r['total_pairs']})")
        
        print("\n" + "-" * 40)
        print("REGIME CLUSTERS")
        print("-" * 40)
        print("(Assets that shift regimes together)")
        
        for i, cluster in enumerate(report['clusters'], 1):
            if len(cluster) > 1:
                print(f"  Cluster {i}: {', '.join(cluster)}")
        
        print("\n" + "-" * 40)
        print("STRONGEST CORRELATIONS")
        print("-" * 40)
        
        for pair in report['strongest_correlations'][:5]:
            print(f"  {pair['pair']}: {pair['correlation']:.0%}")
        
        return report


def main():
    """Run cross-asset correlation analysis."""
    analyzer = CrossAssetCorrelation("regime_models")
    report = analyzer.print_report()
    
    # Save report
    output_path = Path("regime_models/cross_asset_correlation.json")
    with open(output_path, 'w') as f:
        # Convert non-serializable items
        report_clean = {k: v for k, v in report.items() if k != 'correlation_matrix'}
        json.dump(report_clean, f, indent=2)
    print(f"\nReport saved to {output_path}")
    
    return analyzer


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    analyzer = main()
