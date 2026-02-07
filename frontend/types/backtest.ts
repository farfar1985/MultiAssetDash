/**
 * Walk-Forward Validation Types for Backtest Dashboard
 * =====================================================
 *
 * Types matching the backend walk_forward.py data structures
 */

import type { AssetId } from "./index";

/**
 * Available ensemble methods for walk-forward validation
 */
export type WalkForwardMethod =
  // Tier 1
  | "tier1_accuracy"
  | "tier1_magnitude"
  | "tier1_correlation"
  | "tier1_combined"
  // Tier 2
  | "tier2_bma"
  | "tier2_regime"
  | "tier2_conformal"
  | "tier2_combined"
  // Tier 3
  | "tier3_thompson"
  | "tier3_attention"
  | "tier3_quantile"
  | "tier3_combined";

/**
 * Method tier classification
 */
export type MethodTier = "tier1" | "tier2" | "tier3";

/**
 * Method metadata for UI display
 */
export interface MethodInfo {
  id: WalkForwardMethod;
  name: string;
  tier: MethodTier;
  description: string;
}

export const WALK_FORWARD_METHODS: Record<WalkForwardMethod, MethodInfo> = {
  tier1_accuracy: {
    id: "tier1_accuracy",
    name: "Accuracy Weighted",
    tier: "tier1",
    description: "Weights models by historical directional accuracy",
  },
  tier1_magnitude: {
    id: "tier1_magnitude",
    name: "Magnitude Voting",
    tier: "tier1",
    description: "Weights by signal strength/magnitude",
  },
  tier1_correlation: {
    id: "tier1_correlation",
    name: "Error Correlation",
    tier: "tier1",
    description: "Downweights models with correlated errors",
  },
  tier1_combined: {
    id: "tier1_combined",
    name: "Tier 1 Combined",
    tier: "tier1",
    description: "Meta-ensemble of all Tier 1 methods",
  },
  tier2_bma: {
    id: "tier2_bma",
    name: "Bayesian Model Avg",
    tier: "tier2",
    description: "Bayesian model averaging with uncertainty",
  },
  tier2_regime: {
    id: "tier2_regime",
    name: "Regime Adaptive",
    tier: "tier2",
    description: "Adapts weights based on market regime",
  },
  tier2_conformal: {
    id: "tier2_conformal",
    name: "Conformal Prediction",
    tier: "tier2",
    description: "Provides calibrated prediction intervals",
  },
  tier2_combined: {
    id: "tier2_combined",
    name: "Tier 2 Combined",
    tier: "tier2",
    description: "Meta-ensemble of all Tier 2 methods",
  },
  tier3_thompson: {
    id: "tier3_thompson",
    name: "Thompson Sampling",
    tier: "tier3",
    description: "Adaptive exploration via Thompson sampling",
  },
  tier3_attention: {
    id: "tier3_attention",
    name: "Attention Based",
    tier: "tier3",
    description: "Transformer-style attention weighting",
  },
  tier3_quantile: {
    id: "tier3_quantile",
    name: "Quantile Regression",
    tier: "tier3",
    description: "Quantile regression forest for uncertainty",
  },
  tier3_combined: {
    id: "tier3_combined",
    name: "Tier 3 Combined",
    tier: "tier3",
    description: "Meta-ensemble of all Tier 3 methods",
  },
};

/**
 * Result from a single walk-forward fold
 */
export interface FoldResult {
  fold_id: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  n_train: number;
  n_test: number;

  // Core metrics
  accuracy: number; // Directional accuracy (%)
  sharpe_ratio: number; // Annualized Sharpe
  sortino_ratio: number; // Annualized Sortino
  max_drawdown: number; // Maximum drawdown (%)
  win_rate: number; // Percentage of winning trades
  total_return: number; // Total return (%)

  // Trade statistics
  n_trades: number;
  avg_trade_return: number;
  avg_holding_days: number;

  // Signal breakdown
  n_bullish: number;
  n_bearish: number;
  n_neutral: number;

  // Transaction cost summary
  total_costs: number; // Total costs paid ($)
  avg_cost_per_trade: number; // Average cost per trade ($)
  avg_cost_bps: number; // Average cost in basis points
  cost_drag_pct: number; // Total cost as % of returns

  // Per-regime performance
  regime_performance: Record<string, RegimePerformance>;

  // Raw returns for significance testing
  returns: number[];
}

/**
 * Performance metrics broken down by regime
 */
export interface RegimePerformance {
  regime: string;
  n_samples: number;
  accuracy: number;
  sharpe_ratio: number;
  total_return: number;
  win_rate: number;
}

/**
 * Significance test results comparing two methods
 */
export interface SignificanceTest {
  method_a: string;
  method_b: string;
  metric: string;
  mean_diff: number;
  std_diff: number;
  ci_lower: number;
  ci_upper: number;
  p_value: number;
  is_significant: boolean;
  n_bootstrap: number;
}

/**
 * Summary metrics aggregated across all folds
 */
export interface SummaryMetrics {
  // Mean metrics
  mean_accuracy: number;
  mean_sharpe: number;
  mean_sortino: number;
  mean_max_drawdown: number;
  mean_win_rate: number;
  mean_total_return: number;

  // Std deviation
  std_accuracy: number;
  std_sharpe: number;
  std_total_return: number;

  // Cost metrics
  mean_cost_drag_pct: number;
  total_costs: number;

  // Raw vs cost-adjusted comparison
  raw_total_return: number;
  cost_adjusted_return: number;
}

/**
 * Full method comparison results
 */
export interface MethodComparison {
  asset_id: AssetId;
  asset_name: string;
  n_folds: number;
  timestamp: string;

  // Per-method fold results
  method_results: Record<WalkForwardMethod, FoldResult[]>;

  // Aggregated metrics
  summary_metrics: Record<WalkForwardMethod, SummaryMetrics>;

  // Significance tests
  significance_tests: Record<string, SignificanceTest>;

  // Rankings (1 = best)
  rankings: Record<WalkForwardMethod, number>;
}

/**
 * Equity curve point for charting
 */
export interface EquityPoint {
  date: string;
  equity: number;
  drawdown: number;
  benchmark?: number;
  returns?: number;
}

/**
 * Walk-forward backtest request parameters
 */
export interface WalkForwardRequest {
  asset_id: AssetId;
  methods: WalkForwardMethod[];
  n_folds?: number;
  train_pct?: number;
  transaction_cost_bps?: number;
  include_regime_analysis?: boolean;
}

/**
 * Walk-forward backtest response
 */
export interface WalkForwardResponse {
  success: boolean;
  data: MethodComparison;
  equity_curves: Record<WalkForwardMethod, EquityPoint[]>;
  error?: string;
}

/**
 * Cost comparison data for raw vs adjusted
 */
export interface CostComparison {
  method: WalkForwardMethod;
  raw_return: number;
  cost_adjusted_return: number;
  cost_impact: number;
  raw_sharpe: number;
  cost_adjusted_sharpe: number;
}
