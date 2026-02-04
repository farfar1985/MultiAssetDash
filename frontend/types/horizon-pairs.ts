/**
 * Horizon Pair Analysis Types
 *
 * Types for analyzing accuracy correlations between prediction horizons.
 * Key insight: Certain horizon pairs (e.g., D+1 vs D+3) show predictive power
 * that can be exploited for alpha generation.
 */

import type { AssetId } from "./index";

/**
 * Supported prediction horizons
 */
export type Horizon = "D+1" | "D+2" | "D+3" | "D+5" | "D+7" | "D+10";

export const HORIZONS: Horizon[] = ["D+1", "D+2", "D+3", "D+5", "D+7", "D+10"];

/**
 * A single horizon pair with its accuracy metrics
 */
export interface HorizonPair {
  /** First horizon (row in matrix) */
  h1: Horizon;
  /** Second horizon (column in matrix) */
  h2: Horizon;
  /** Directional accuracy when both horizons agree (0-100) */
  accuracy: number;
  /** Number of samples used to calculate accuracy */
  sampleSize: number;
  /** Statistical significance (p-value) */
  pValue?: number;
  /** Confidence interval [lower, upper] */
  confidenceInterval?: [number, number];
}

/**
 * Complete horizon pair matrix for an asset
 */
export interface HorizonPairMatrix {
  /** Asset identifier */
  asset: AssetId;
  /** Asset display name */
  assetName: string;
  /** All horizon pairs with accuracy data */
  pairs: HorizonPair[];
  /** Timestamp when data was calculated */
  calculatedAt: string;
  /** Evaluation period start */
  periodStart: string;
  /** Evaluation period end */
  periodEnd: string;
}

/**
 * Aggregated insights for horizon pair analysis
 */
export interface HorizonPairInsight {
  /** The best performing pair */
  bestPair: HorizonPair;
  /** Average accuracy across all pairs */
  averageAccuracy: number;
  /** Number of pairs with accuracy > 60% */
  alphaPairCount: number;
  /** Recommended weight multiplier for best pair */
  recommendedWeight: number;
  /** Textual recommendation */
  recommendation: string;
  /** Confidence level in the insight */
  confidence: "high" | "medium" | "low";
}

/**
 * Accuracy thresholds for visualization
 */
export const ACCURACY_THRESHOLDS = {
  /** Below this is considered poor (red) */
  poor: 50,
  /** Below this is considered marginal (yellow) */
  marginal: 60,
  /** Above this is considered alpha source (green) */
  good: 60,
  /** Above this is highlighted as exceptional */
  exceptional: 65,
} as const;

/**
 * Get color for accuracy value based on thresholds
 */
export function getAccuracyColor(accuracy: number): string {
  if (accuracy < ACCURACY_THRESHOLDS.poor) return "#ef4444"; // red-500
  if (accuracy < ACCURACY_THRESHOLDS.marginal) return "#eab308"; // yellow-500
  if (accuracy >= ACCURACY_THRESHOLDS.exceptional) return "#22c55e"; // green-500
  return "#4ade80"; // green-400
}

/**
 * Get background color for accuracy value (with opacity)
 */
export function getAccuracyBgColor(accuracy: number): string {
  if (accuracy < ACCURACY_THRESHOLDS.poor) return "rgba(239, 68, 68, 0.15)";
  if (accuracy < ACCURACY_THRESHOLDS.marginal) return "rgba(234, 179, 8, 0.15)";
  if (accuracy >= ACCURACY_THRESHOLDS.exceptional)
    return "rgba(34, 197, 94, 0.25)";
  return "rgba(74, 222, 128, 0.15)";
}

/**
 * Determine if a pair should be highlighted as alpha source
 */
export function isAlphaSource(accuracy: number): boolean {
  return accuracy >= ACCURACY_THRESHOLDS.exceptional;
}
