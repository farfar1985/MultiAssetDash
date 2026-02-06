// Ensemble Visualization Components
// Based on ENSEMBLE_METHODS_PLAN.md research

export { EnsembleConfidenceCard } from "./EnsembleConfidenceCard";
export type {
  EnsembleConfidenceData,
  EnsembleConfidenceCardProps,
  ConfidenceWeight,
} from "./EnsembleConfidenceCard";

export { PairwiseVotingChart } from "./PairwiseVotingChart";
export type {
  PairwiseVotingData,
  PairwiseVotingChartProps,
  HorizonPairVote,
} from "./PairwiseVotingChart";

export { RegimeIndicator } from "./RegimeIndicator";
export type {
  RegimeData,
  RegimeIndicatorProps,
  MarketRegime,
} from "./RegimeIndicator";

export { HMMRegimeIndicator } from "./HMMRegimeIndicator";
export type { HMMRegimeIndicatorProps } from "./HMMRegimeIndicator";

export { ConfidenceIntervalBar } from "./ConfidenceIntervalBar";
export type {
  ConfidenceInterval,
  ConfidenceIntervalBarProps,
} from "./ConfidenceIntervalBar";

// API-connected wrapper components
export { APIEnsembleConfidenceCard } from "./APIEnsembleConfidenceCard";
export type { APIEnsembleConfidenceCardProps } from "./APIEnsembleConfidenceCard";

export { APIPairwiseVotingChart } from "./APIPairwiseVotingChart";
export type { APIPairwiseVotingChartProps } from "./APIPairwiseVotingChart";

export { APIConfidenceIntervalBar } from "./APIConfidenceIntervalBar";
export type { APIConfidenceIntervalBarProps } from "./APIConfidenceIntervalBar";
