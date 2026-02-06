"use client";

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import api, {
  backendApi,
  type AssetData,
  type SignalData,
  type PerformanceMetrics,
  type EnsembleResult,
  type ModelAgreement,
  type HistoricalDataPoint,
  type PortfolioSummary,
  type Horizon,
  type EnsembleMethod,
  type BacktestResult,
  type ModelAccuracy,
  type FeatureImportance,
  type HistoricalMetrics,
  type StrategyId,
  type BackendAsset,
  type BackendSignal,
  type BackendMetrics,
  type BackendForecast,
  type EnsembleConfig,
  type HealthStatus,
  type QuantumDashboard,
} from "@/lib/api-client";
import {
  getHMMRegime,
  getAllHMMRegimes,
  getEnsembleConfidence,
  getPairwiseVoting,
  getConfidenceInterval,
  getEnsembleDashboard,
  getTierComparison,
  getTier1Ensemble,
  getTier2Ensemble,
  getTier3Ensemble,
  getAllTiers,
  type RegimeData,
  type EnsembleConfidenceData,
  type PairwiseVotingData,
  type ConfidenceInterval,
  type EnsembleDashboardData,
  type TierComparisonData,
  type Tier1Result,
  type Tier2Result,
  type Tier3Result,
  type Tier1Method,
  type Tier2Method,
  type Tier3Method,
} from "@/lib/api";
import type { AssetId } from "@/types";

// Query keys factory for consistent cache management
export const queryKeys = {
  all: ["qdt"] as const,
  // Legacy keys (mock API)
  assets: () => [...queryKeys.all, "assets"] as const,
  asset: (symbol: string) => [...queryKeys.assets(), symbol] as const,
  signals: () => [...queryKeys.all, "signals"] as const,
  signal: (symbol: string, horizon: Horizon, method?: EnsembleMethod) =>
    [...queryKeys.signals(), symbol, horizon, method] as const,
  metrics: () => [...queryKeys.all, "metrics"] as const,
  metric: (symbol: string, method?: EnsembleMethod) =>
    [...queryKeys.metrics(), symbol, method] as const,
  ensemble: () => [...queryKeys.all, "ensemble"] as const,
  ensembleResult: (symbol: string, method: EnsembleMethod) =>
    [...queryKeys.ensemble(), symbol, method] as const,
  modelAgreement: (symbol: string) =>
    [...queryKeys.all, "modelAgreement", symbol] as const,
  historical: (symbol: string, startDate: string, endDate: string) =>
    [...queryKeys.all, "historical", symbol, startDate, endDate] as const,
  portfolio: () => [...queryKeys.all, "portfolio"] as const,
  portfolioSummary: (method?: EnsembleMethod) =>
    [...queryKeys.portfolio(), "summary", method] as const,
  // Quantum ML keys
  quantumMl: () => [...queryKeys.all, "quantumMl"] as const,
  backtest: (symbol: string, strategy: StrategyId) =>
    [...queryKeys.quantumMl(), "backtest", symbol, strategy] as const,
  modelAccuracy: (symbol: string) =>
    [...queryKeys.quantumMl(), "accuracy", symbol] as const,
  historicalMetrics: (symbol: string, startDate: string, endDate: string) =>
    [...queryKeys.quantumMl(), "historicalMetrics", symbol, startDate, endDate] as const,
  featureImportance: (symbol: string) =>
    [...queryKeys.quantumMl(), "features", symbol] as const,
  // Backend API keys (real Amira API)
  backend: () => [...queryKeys.all, "backend"] as const,
  backendHealth: () => [...queryKeys.backend(), "health"] as const,
  backendAssets: () => [...queryKeys.backend(), "assets"] as const,
  backendSignal: (asset: string, horizons?: string) =>
    [...queryKeys.backend(), "signal", asset, horizons] as const,
  backendForecast: (asset: string) =>
    [...queryKeys.backend(), "forecast", asset] as const,
  backendMetrics: (asset: string) =>
    [...queryKeys.backend(), "metrics", asset] as const,
  backendOHLCV: (asset: string, startDate?: string, endDate?: string) =>
    [...queryKeys.backend(), "ohlcv", asset, startDate, endDate] as const,
  backendEquity: (asset: string) =>
    [...queryKeys.backend(), "equity", asset] as const,
  backendEnsembleConfig: (asset: string) =>
    [...queryKeys.backend(), "ensembleConfig", asset] as const,
  backendConfig: (asset: string, strategy: string) =>
    [...queryKeys.backend(), "config", asset, strategy] as const,
  backendConfigs: () => [...queryKeys.backend(), "configs"] as const,
  // Quantum API keys
  quantumDashboard: () => [...queryKeys.backend(), "quantum", "dashboard"] as const,
  // HMM Regime Detection keys
  hmmRegime: (assetId: string) => [...queryKeys.backend(), "hmm", "regime", assetId] as const,
  hmmRegimeAll: () => [...queryKeys.backend(), "hmm", "regimes"] as const,
  // Ensemble component keys
  ensembleConfidence: (assetId: string) => [...queryKeys.backend(), "ensemble", "confidence", assetId] as const,
  pairwiseVoting: (assetId: string) => [...queryKeys.backend(), "ensemble", "pairwise", assetId] as const,
  confidenceInterval: (assetId: string, horizon: number) => [...queryKeys.backend(), "ensemble", "interval", assetId, horizon] as const,
  ensembleDashboard: (assetId: string) => [...queryKeys.backend(), "ensemble", "dashboard", assetId] as const,
  tierComparison: (assetId: string) => [...queryKeys.backend(), "ensemble", "tiers", assetId] as const,
  // Individual tier ensemble keys
  tier1Ensemble: (assetId: string, method: string) => [...queryKeys.backend(), "ensemble", "tier1", assetId, method] as const,
  tier2Ensemble: (assetId: string, method: string) => [...queryKeys.backend(), "ensemble", "tier2", assetId, method] as const,
  tier3Ensemble: (assetId: string, method: string) => [...queryKeys.backend(), "ensemble", "tier3", assetId, method] as const,
  allTiers: (assetId: string) => [...queryKeys.backend(), "ensemble", "allTiers", assetId] as const,
};

// ============================================================================
// Backend API Hooks (Real Amira API)
// ============================================================================

/**
 * Health check hook - check connection to backend API
 */
export function useBackendHealth(
  options?: Omit<UseQueryOptions<HealthStatus, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.backendHealth(),
    queryFn: () => backendApi.health(),
    retry: 1,
    refetchInterval: 30000, // Check every 30 seconds
    ...options,
  });
}

/**
 * Fetch assets from backend API
 */
export function useBackendAssets(
  options?: Omit<UseQueryOptions<BackendAsset[], Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.backendAssets(),
    queryFn: () => backendApi.getAssets(),
    ...options,
  });
}

/**
 * Fetch signal data from backend API
 * @param asset - Asset name (e.g., "crude-oil")
 * @param horizons - Optional comma-separated horizons (e.g., "D+5,D+7,D+10")
 */
export function useBackendSignal(
  asset: string,
  horizons?: string,
  options?: Omit<
    UseQueryOptions<{ data: BackendSignal[]; count: number }, Error>,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: queryKeys.backendSignal(asset, horizons),
    queryFn: () => backendApi.getSignal(asset, horizons),
    enabled: !!asset,
    ...options,
  });
}

/**
 * Fetch live forecast from backend API
 */
export function useBackendForecast(
  asset: string,
  options?: Omit<UseQueryOptions<BackendForecast, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.backendForecast(asset),
    queryFn: () => backendApi.getForecast(asset),
    enabled: !!asset,
    ...options,
  });
}

/**
 * Fetch metrics from backend API
 */
export function useBackendMetrics(
  asset: string,
  options?: Omit<UseQueryOptions<BackendMetrics, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.backendMetrics(asset),
    queryFn: () => backendApi.getMetrics(asset),
    enabled: !!asset,
    ...options,
  });
}

/**
 * Fetch OHLCV data from backend API
 */
export function useBackendOHLCV(
  asset: string,
  startDate?: string,
  endDate?: string,
  options?: Omit<
    UseQueryOptions<HistoricalDataPoint[], Error>,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: queryKeys.backendOHLCV(asset, startDate, endDate),
    queryFn: () => backendApi.getOHLCV(asset, startDate, endDate),
    enabled: !!asset,
    ...options,
  });
}

/**
 * Fetch equity curve from backend API
 */
export function useBackendEquity(
  asset: string,
  options?: Omit<
    UseQueryOptions<
      { equity_curve: Array<{ date: string; equity: number }>; final_equity: number; total_return: number },
      Error
    >,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: queryKeys.backendEquity(asset),
    queryFn: () => backendApi.getEquity(asset),
    enabled: !!asset,
    ...options,
  });
}

/**
 * Fetch ensemble configuration from backend API
 */
export function useBackendEnsembleConfig(
  asset: string,
  options?: Omit<UseQueryOptions<EnsembleConfig, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.backendEnsembleConfig(asset),
    queryFn: () => backendApi.getEnsembleConfig(asset),
    enabled: !!asset,
    ...options,
  });
}

/**
 * Fetch strategy configuration from backend API
 * @param asset - Asset name
 * @param strategy - Strategy name (default: "optimal")
 */
export function useBackendConfig(
  asset: string,
  strategy: string = "optimal",
  options?: Omit<
    UseQueryOptions<{ asset: string; strategy: string; config: Record<string, unknown> }, Error>,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: queryKeys.backendConfig(asset, strategy),
    queryFn: () => backendApi.getConfig(asset, strategy),
    enabled: !!asset,
    ...options,
  });
}

/**
 * List all available strategy configurations
 */
export function useBackendConfigs(
  options?: Omit<
    UseQueryOptions<
      {
        configs: Array<{
          filename: string;
          strategy: string;
          asset: string;
          viable_horizons: number[];
          sharpe_ratio: number;
          win_rate: number;
        }>;
        count: number;
      },
      Error
    >,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: queryKeys.backendConfigs(),
    queryFn: () => backendApi.listConfigs(),
    ...options,
  });
}

/**
 * Fetch quantum dashboard with regime status and contagion analysis
 */
export function useQuantumDashboard(
  options?: Omit<UseQueryOptions<QuantumDashboard, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.quantumDashboard(),
    queryFn: () => backendApi.getQuantumDashboard(),
    refetchInterval: 60000, // Refresh every minute
    retry: 2,
    ...options,
  });
}

// ============================================================================
// Legacy API Hooks (Mock endpoints for backward compatibility)
// ============================================================================

/**
 * Fetch all available assets
 */
export function useAssets(
  options?: Omit<UseQueryOptions<AssetData[], Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.assets(),
    queryFn: () => api.getAssets(),
    ...options,
  });
}

/**
 * Fetch a single asset by symbol
 */
export function useAsset(
  symbol: string,
  options?: Omit<UseQueryOptions<AssetData, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.asset(symbol),
    queryFn: () => api.getAsset(symbol),
    enabled: !!symbol,
    ...options,
  });
}

/**
 * Fetch signal for an asset at a specific horizon with optional ensemble method
 */
export function useSignal(
  symbol: string,
  horizon: Horizon = "D+1",
  method?: EnsembleMethod,
  options?: Omit<UseQueryOptions<SignalData, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.signal(symbol, horizon, method),
    queryFn: () => api.getSignal(symbol, horizon, method),
    enabled: !!symbol,
    ...options,
  });
}

/**
 * Fetch performance metrics for an asset with optional ensemble method
 */
export function useMetrics(
  symbol: string,
  method?: EnsembleMethod,
  options?: Omit<
    UseQueryOptions<PerformanceMetrics, Error>,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: queryKeys.metric(symbol, method),
    queryFn: () => api.getMetrics(symbol, method),
    enabled: !!symbol,
    ...options,
  });
}

/**
 * Fetch ensemble results for an asset with a specific method
 */
export function useEnsemble(
  symbol: string,
  method: EnsembleMethod = "accuracy_weighted",
  options?: Omit<UseQueryOptions<EnsembleResult, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.ensembleResult(symbol, method),
    queryFn: () => api.getEnsembleResults(symbol, method),
    enabled: !!symbol,
    ...options,
  });
}

/**
 * Fetch model agreement breakdown for an asset
 */
export function useModelAgreement(
  symbol: string,
  options?: Omit<UseQueryOptions<ModelAgreement, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.modelAgreement(symbol),
    queryFn: () => api.getModelAgreement(symbol),
    enabled: !!symbol,
    ...options,
  });
}

/**
 * Fetch historical data for an asset
 */
export function useHistorical(
  symbol: string,
  startDate: string,
  endDate: string,
  options?: Omit<
    UseQueryOptions<HistoricalDataPoint[], Error>,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: queryKeys.historical(symbol, startDate, endDate),
    queryFn: () => api.getHistorical(symbol, startDate, endDate),
    enabled: !!symbol && !!startDate && !!endDate,
    ...options,
  });
}

/**
 * Fetch portfolio summary across all assets
 */
export function usePortfolioSummary(
  method?: EnsembleMethod,
  options?: Omit<
    UseQueryOptions<PortfolioSummary, Error>,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: queryKeys.portfolioSummary(method),
    queryFn: () => api.getPortfolioSummary(method),
    ...options,
  });
}

// ============================================================================
// Quantum ML Hooks
// ============================================================================

/**
 * Fetch backtest results for a symbol and strategy
 */
export function useBacktest(
  symbol: string,
  strategy: StrategyId = "ensemble_weighted",
  options?: Omit<UseQueryOptions<BacktestResult, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.backtest(symbol, strategy),
    queryFn: () => api.getBacktest(symbol, strategy),
    enabled: !!symbol,
    ...options,
  });
}

/**
 * Fetch model accuracy metrics for a symbol
 */
export function useModelAccuracy(
  symbol: string,
  options?: Omit<UseQueryOptions<ModelAccuracy[], Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.modelAccuracy(symbol),
    queryFn: () => api.getModelAccuracy(symbol),
    enabled: !!symbol,
    ...options,
  });
}

/**
 * Fetch historical metrics for time-travel analysis
 */
export function useHistoricalMetrics(
  symbol: string,
  startDate: string,
  endDate: string,
  options?: Omit<
    UseQueryOptions<HistoricalMetrics[], Error>,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: queryKeys.historicalMetrics(symbol, startDate, endDate),
    queryFn: () => api.getHistoricalMetrics(symbol, startDate, endDate),
    enabled: !!symbol && !!startDate && !!endDate,
    ...options,
  });
}

/**
 * Fetch feature importance rankings for a symbol
 */
export function useFeatureImportance(
  symbol: string,
  options?: Omit<
    UseQueryOptions<FeatureImportance[], Error>,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: queryKeys.featureImportance(symbol),
    queryFn: () => api.getFeatureImportance(symbol),
    enabled: !!symbol,
    ...options,
  });
}

// ============================================================================
// HMM Regime Detection Hooks
// ============================================================================

/**
 * Fetch HMM-detected market regime for an asset
 * @param assetId - Asset identifier (e.g., "crude-oil", "gold", "bitcoin", "sp500")
 */
export function useHMMRegime(
  assetId: AssetId,
  options?: Omit<UseQueryOptions<RegimeData, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.hmmRegime(assetId),
    queryFn: () => getHMMRegime(assetId),
    enabled: !!assetId,
    staleTime: 30000, // Consider data fresh for 30 seconds
    refetchInterval: 60000, // Refresh every minute
    ...options,
  });
}

/**
 * Fetch HMM regime data for all configured assets
 */
export function useAllHMMRegimes(
  options?: Omit<UseQueryOptions<Partial<Record<AssetId, RegimeData>>, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.hmmRegimeAll(),
    queryFn: () => getAllHMMRegimes(),
    staleTime: 30000,
    refetchInterval: 60000,
    ...options,
  });
}

// ============================================================================
// Ensemble Component Hooks
// ============================================================================

/**
 * Fetch ensemble confidence data for an asset
 */
export function useEnsembleConfidence(
  assetId: AssetId,
  options?: Omit<UseQueryOptions<EnsembleConfidenceData, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.ensembleConfidence(assetId),
    queryFn: () => getEnsembleConfidence(assetId),
    enabled: !!assetId,
    staleTime: 30000,
    refetchInterval: 60000,
    ...options,
  });
}

/**
 * Fetch pairwise voting data for an asset
 */
export function usePairwiseVoting(
  assetId: AssetId,
  options?: Omit<UseQueryOptions<PairwiseVotingData, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.pairwiseVoting(assetId),
    queryFn: () => getPairwiseVoting(assetId),
    enabled: !!assetId,
    staleTime: 30000,
    refetchInterval: 60000,
    ...options,
  });
}

/**
 * Fetch confidence interval data for an asset
 */
export function useConfidenceInterval(
  assetId: AssetId,
  horizon: number = 5,
  options?: Omit<UseQueryOptions<ConfidenceInterval, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.confidenceInterval(assetId, horizon),
    queryFn: () => getConfidenceInterval(assetId, horizon),
    enabled: !!assetId,
    staleTime: 30000,
    refetchInterval: 60000,
    ...options,
  });
}

/**
 * Fetch all ensemble data for an asset in a single call
 */
export function useEnsembleDashboard(
  assetId: AssetId,
  options?: Omit<UseQueryOptions<EnsembleDashboardData, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.ensembleDashboard(assetId),
    queryFn: () => getEnsembleDashboard(assetId),
    enabled: !!assetId,
    staleTime: 30000,
    refetchInterval: 60000,
    ...options,
  });
}

/**
 * Fetch tier comparison data for all three ensemble tiers
 */
export function useTierComparison(
  assetId: AssetId,
  options?: Omit<UseQueryOptions<TierComparisonData, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.tierComparison(assetId),
    queryFn: () => getTierComparison(assetId),
    enabled: !!assetId,
    staleTime: 30000,
    refetchInterval: 60000,
    ...options,
  });
}

// ============================================================================
// Individual Tier Ensemble Hooks
// ============================================================================

/**
 * Fetch Tier 1 ensemble prediction for an asset
 * Tier 1 methods: accuracy-weighted, magnitude-weighted, error correlation
 * @param assetId - Asset identifier
 * @param method - Ensemble method (default: "combined")
 */
export function useTier1Ensemble(
  assetId: AssetId,
  method: Tier1Method = "combined",
  options?: Omit<UseQueryOptions<Tier1Result, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.tier1Ensemble(assetId, method),
    queryFn: () => getTier1Ensemble(assetId, method),
    enabled: !!assetId,
    staleTime: 30000,
    refetchInterval: 60000,
    ...options,
  });
}

/**
 * Fetch Tier 2 ensemble prediction for an asset
 * Tier 2 methods: Bayesian model averaging, regime-adaptive, conformal prediction
 * @param assetId - Asset identifier
 * @param method - Ensemble method (default: "combined")
 */
export function useTier2Ensemble(
  assetId: AssetId,
  method: Tier2Method = "combined",
  options?: Omit<UseQueryOptions<Tier2Result, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.tier2Ensemble(assetId, method),
    queryFn: () => getTier2Ensemble(assetId, method),
    enabled: !!assetId,
    staleTime: 30000,
    refetchInterval: 60000,
    ...options,
  });
}

/**
 * Fetch Tier 3 ensemble prediction for an asset
 * Tier 3 methods: Thompson sampling, attention-based, quantile regression forest
 * @param assetId - Asset identifier
 * @param method - Ensemble method (default: "combined")
 */
export function useTier3Ensemble(
  assetId: AssetId,
  method: Tier3Method = "combined",
  options?: Omit<UseQueryOptions<Tier3Result, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.tier3Ensemble(assetId, method),
    queryFn: () => getTier3Ensemble(assetId, method),
    enabled: !!assetId,
    staleTime: 30000,
    refetchInterval: 60000,
    ...options,
  });
}

/**
 * Fetch all tier predictions for an asset (convenience alias)
 * @param assetId - Asset identifier
 */
export function useAllTiers(
  assetId: AssetId,
  options?: Omit<UseQueryOptions<TierComparisonData, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.allTiers(assetId),
    queryFn: () => getAllTiers(assetId),
    enabled: !!assetId,
    staleTime: 30000,
    refetchInterval: 60000,
    ...options,
  });
}

// ============================================================================
// Walk-Forward Validation Hooks
// ============================================================================

import { getWalkForwardResults, getCostComparison } from "@/lib/api";
import type {
  WalkForwardMethod,
  WalkForwardResponse,
  CostComparison,
} from "@/types/backtest";

// Add walk-forward keys to queryKeys
export const walkForwardKeys = {
  all: [...queryKeys.all, "walkForward"] as const,
  results: (assetId: string, methods: string[], nFolds: number) =>
    [...walkForwardKeys.all, "results", assetId, methods.join(","), nFolds] as const,
  costComparison: (assetId: string, methods: string[]) =>
    [...walkForwardKeys.all, "costs", assetId, methods.join(",")] as const,
};

/**
 * Fetch walk-forward validation results
 * @param assetId - Asset to backtest
 * @param methods - Ensemble methods to compare
 * @param nFolds - Number of walk-forward folds
 */
export function useWalkForwardResults(
  assetId: AssetId,
  methods: WalkForwardMethod[],
  nFolds: number = 5,
  options?: Omit<UseQueryOptions<WalkForwardResponse, Error>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: walkForwardKeys.results(assetId, methods, nFolds),
    queryFn: () => getWalkForwardResults(assetId, methods, nFolds),
    enabled: !!assetId && methods.length > 0,
    staleTime: 60000, // Results don't change often
    ...options,
  });
}

/**
 * Get cost comparison data derived from walk-forward results
 */
export function useCostComparison(
  walkForwardData: WalkForwardResponse | undefined
): CostComparison[] {
  if (!walkForwardData?.data?.summary_metrics) {
    return [];
  }
  return getCostComparison(walkForwardData.data.summary_metrics);
}
