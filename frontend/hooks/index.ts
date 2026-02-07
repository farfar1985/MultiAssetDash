export {
  // Legacy API hooks (mock endpoints)
  useAssets,
  useAsset,
  useSignal,
  useMetrics,
  useEnsemble,
  useModelAgreement,
  useHistorical,
  usePortfolioSummary,
  // Quantum ML hooks
  useBacktest,
  useModelAccuracy,
  useHistoricalMetrics,
  useFeatureImportance,
  // Backend API hooks (real Amira API)
  useBackendHealth,
  useBackendAssets,
  useBackendSignal,
  useBackendForecast,
  useBackendMetrics,
  useBackendOHLCV,
  useBackendEquity,
  useBackendEnsembleConfig,
  // HMM Regime Detection hooks
  useHMMRegime,
  useAllHMMRegimes,
  // Ensemble Component hooks
  useEnsembleConfidence,
  usePairwiseVoting,
  useConfidenceInterval,
  useEnsembleDashboard,
  // Query keys
  queryKeys,
} from "./useApi";

export { useCommandPalette } from "./useCommandPalette";
