/**
 * Practical Metrics Types for CME Hedging Desks
 *
 * Focus on ACTIONABLE trading metrics, not backtest vanity metrics.
 * A forecast needs to be actionable — big enough moves, long enough horizons.
 */

import type { AssetId } from "./index";
import type { Horizon } from "./horizon-pairs";

/**
 * Asset-specific minimum move thresholds for actionability
 * These represent the minimum predicted move that justifies action
 */
export const ASSET_MOVE_THRESHOLDS: Partial<Record<AssetId, number>> = {
  "crude-oil": 1.0, // $1.00 minimum for CL
  bitcoin: 500, // $500 minimum for BTC
  gold: 10, // $10 minimum for GC
  silver: 0.25, // $0.25 minimum for SI
  "natural-gas": 0.1, // $0.10 minimum for NG
  copper: 0.05, // $0.05 minimum for HG
  wheat: 5, // 5 cents minimum for ZW
  corn: 3, // 3 cents minimum for ZC
  soybean: 8, // 8 cents minimum for ZS
  platinum: 15, // $15 minimum for PL
};

/**
 * Forecast magnitude — how big is the predicted move?
 */
export interface ForecastMagnitude {
  /** Raw predicted dollar/point move */
  predictedMove: number;
  /** Absolute value of predicted move */
  absoluteMove: number;
  /** Direction: positive = up, negative = down */
  direction: "up" | "down";
  /** Is this move above the asset's actionability threshold? */
  isActionable: boolean;
  /** Move as percentage of current price */
  movePercent: number;
  /** Asset-specific threshold for comparison */
  threshold: number;
  /** How many multiples of threshold is this move? */
  thresholdMultiple: number;
}

/**
 * Horizon coverage — which prediction horizons are represented?
 */
export interface HorizonCoverage {
  /** Which horizons have predictions */
  coveredHorizons: Horizon[];
  /** Which horizons are missing */
  missingHorizons: Horizon[];
  /** Coverage as percentage (0-100) */
  coveragePercent: number;
  /** Does this cover short-term (D+1, D+2)? */
  hasShortTerm: boolean;
  /** Does this cover medium-term (D+3, D+5)? */
  hasMediumTerm: boolean;
  /** Does this cover long-term (D+7, D+10)? */
  hasLongTerm: boolean;
  /** Best horizon for this signal based on confidence */
  optimalHorizon: Horizon | null;
}

/**
 * Actionability level for traffic light display
 */
export type ActionabilityLevel = "high" | "medium" | "low";

/**
 * Practical score — composite score weighing utility factors
 */
export interface PracticalScore {
  /** Overall practical utility score (0-100) */
  score: number;
  /** Traffic light actionability */
  actionability: ActionabilityLevel;
  /** Component scores */
  components: {
    /** Score from forecast magnitude (0-100) */
    magnitudeScore: number;
    /** Score from horizon diversity (0-100) */
    horizonScore: number;
    /** Score from confidence level (0-100) */
    confidenceScore: number;
    /** Score from win rate on big moves (0-100) */
    bigMoveAccuracyScore: number;
  };
  /** Weight applied to each component */
  weights: {
    magnitude: number;
    horizon: number;
    confidence: number;
    bigMoveAccuracy: number;
  };
}

/**
 * Signal that meets minimum thresholds for action
 */
export interface ActionableSignal {
  /** Is this signal actionable? */
  isActionable: boolean;
  /** Why is this actionable (or not)? */
  reasons: string[];
  /** Forecast magnitude details */
  magnitude: ForecastMagnitude;
  /** Horizon coverage details */
  horizonCoverage: HorizonCoverage;
  /** Practical utility score */
  practicalScore: PracticalScore;
  /** Raw confidence from models (0-100) */
  confidence: number;
  /** Win rate specifically on moves > threshold */
  bigMoveWinRate: number;
  /** Recommended position sizing (0-100% of max exposure) */
  recommendedPositionSize: number;
  /** When to act */
  timeToAction: TimeToAction;
}

/**
 * When the signal suggests acting
 */
export interface TimeToAction {
  /** Urgency level */
  urgency: "immediate" | "today" | "this_week" | "monitor";
  /** Days until optimal entry */
  daysToOptimalEntry: number;
  /** Days until signal expires */
  daysUntilExpiry: number;
  /** Reason for the timing recommendation */
  reason: string;
}

/**
 * Complete practical metrics for display
 */
export interface PracticalMetricsData {
  /** Asset being analyzed */
  asset: AssetId;
  /** Current price for context */
  currentPrice: number;
  /** The actionable signal assessment */
  signal: ActionableSignal;
  /** Traditional Sharpe for comparison */
  traditionalSharpe: number;
  /** Overall win rate for comparison */
  overallWinRate: number;
  /** Timestamp of analysis */
  analyzedAt: string;
}

/**
 * Plain English insight for traders
 */
export interface PracticalInsight {
  /** Main headline insight */
  headline: string;
  /** Detailed explanation */
  explanation: string;
  /** Recommendation text */
  recommendation: string;
  /** Why this is/isn't actionable */
  actionabilityReason: string;
  /** Risk considerations */
  riskNote: string;
  /** Confidence in this insight */
  confidence: "high" | "medium" | "low";
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get actionability color (traffic light system)
 */
export function getActionabilityColor(level: ActionabilityLevel): string {
  switch (level) {
    case "high":
      return "#22c55e"; // green-500
    case "medium":
      return "#eab308"; // yellow-500
    case "low":
      return "#ef4444"; // red-500
  }
}

/**
 * Get actionability background color with opacity
 */
export function getActionabilityBgColor(level: ActionabilityLevel): string {
  switch (level) {
    case "high":
      return "rgba(34, 197, 94, 0.15)";
    case "medium":
      return "rgba(234, 179, 8, 0.15)";
    case "low":
      return "rgba(239, 68, 68, 0.15)";
  }
}

/**
 * Get Tailwind classes for actionability level
 */
export function getActionabilityClasses(level: ActionabilityLevel): {
  text: string;
  bg: string;
  border: string;
  dot: string;
} {
  switch (level) {
    case "high":
      return {
        text: "text-green-500",
        bg: "bg-green-500/10",
        border: "border-green-500/30",
        dot: "bg-green-500",
      };
    case "medium":
      return {
        text: "text-yellow-500",
        bg: "bg-yellow-500/10",
        border: "border-yellow-500/30",
        dot: "bg-yellow-500",
      };
    case "low":
      return {
        text: "text-red-500",
        bg: "bg-red-500/10",
        border: "border-red-500/30",
        dot: "bg-red-500",
      };
  }
}

/**
 * Get urgency display text
 */
export function getUrgencyText(urgency: TimeToAction["urgency"]): string {
  switch (urgency) {
    case "immediate":
      return "Act Now";
    case "today":
      return "Today";
    case "this_week":
      return "This Week";
    case "monitor":
      return "Monitor";
  }
}

/**
 * Get urgency color classes
 */
export function getUrgencyClasses(urgency: TimeToAction["urgency"]): string {
  switch (urgency) {
    case "immediate":
      return "text-green-500 bg-green-500/10 border-green-500/30";
    case "today":
      return "text-blue-500 bg-blue-500/10 border-blue-500/30";
    case "this_week":
      return "text-yellow-500 bg-yellow-500/10 border-yellow-500/30";
    case "monitor":
      return "text-neutral-400 bg-neutral-500/10 border-neutral-500/30";
  }
}

/**
 * Calculate practical score from components
 */
export function calculatePracticalScore(
  magnitudeScore: number,
  horizonScore: number,
  confidenceScore: number,
  bigMoveAccuracyScore: number
): PracticalScore {
  // Weights emphasize practical utility
  const weights = {
    magnitude: 0.35, // Biggest weight - move size matters most
    horizon: 0.2, // Diversity of horizons
    confidence: 0.2, // Model confidence
    bigMoveAccuracy: 0.25, // Accuracy on the moves that matter
  };

  const score =
    magnitudeScore * weights.magnitude +
    horizonScore * weights.horizon +
    confidenceScore * weights.confidence +
    bigMoveAccuracyScore * weights.bigMoveAccuracy;

  let actionability: ActionabilityLevel;
  if (score >= 70) {
    actionability = "high";
  } else if (score >= 45) {
    actionability = "medium";
  } else {
    actionability = "low";
  }

  return {
    score,
    actionability,
    components: {
      magnitudeScore,
      horizonScore,
      confidenceScore,
      bigMoveAccuracyScore,
    },
    weights,
  };
}

/**
 * Calculate recommended position size based on signal strength
 */
export function calculatePositionSize(
  practicalScore: number,
  confidence: number,
  bigMoveWinRate: number
): number {
  // Base position from practical score
  let position = practicalScore * 0.6;

  // Adjust for confidence
  if (confidence >= 70) {
    position *= 1.2;
  } else if (confidence < 50) {
    position *= 0.7;
  }

  // Adjust for big move accuracy
  if (bigMoveWinRate >= 65) {
    position *= 1.15;
  } else if (bigMoveWinRate < 50) {
    position *= 0.6;
  }

  // Cap at 100%
  return Math.min(100, Math.max(0, Math.round(position)));
}

/**
 * Format move size for display
 */
export function formatMoveSize(move: number, asset: AssetId): string {
  const absMove = Math.abs(move);
  const sign = move >= 0 ? "+" : "-";

  // Different formatting based on asset
  if (asset === "bitcoin") {
    return `${sign}$${absMove.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  }
  if (asset === "crude-oil" || asset === "natural-gas") {
    return `${sign}$${absMove.toFixed(2)}`;
  }
  if (asset === "gold" || asset === "platinum") {
    return `${sign}$${absMove.toFixed(2)}`;
  }
  // Agricultural - cents
  if (["wheat", "corn", "soybean"].includes(asset)) {
    return `${sign}${absMove.toFixed(1)}¢`;
  }
  // Default
  return `${sign}$${absMove.toFixed(2)}`;
}
