// Target Ladder Signal Types for Amira's target_ladder_signal.py output

/**
 * A single price target level in the ladder
 * Represents one of the n1, n3, n5, n8, n10 targets
 */
export interface TargetLevel {
  /** Target identifier (n1, n3, n5, n8, n10) */
  level: string;
  /** Target price */
  price: number;
  /** Delta from current price (positive = above, negative = below) */
  delta: number;
  /** Percentage delta from current price */
  deltaPercent: number;
  /** Whether this target agrees with the overall direction */
  agreesWithDirection: boolean;
}

/**
 * Direction of the target ladder signal
 */
export type TargetDirection = "bullish" | "bearish";

/**
 * Conviction level based on target agreement
 */
export type ConvictionLevel = "HIGH" | "MEDIUM" | "LOW";

/**
 * Complete target ladder signal from Amira
 * Contains all price targets and consensus metrics
 */
export interface TargetLadderSignal {
  /** Asset symbol (e.g., "CL", "GC", "BTC") */
  symbol: string;
  /** Asset name */
  assetName: string;
  /** Current price of the asset */
  current: number;
  /** Overall signal direction based on target consensus */
  direction: TargetDirection;
  /** Consensus percentage (0-100) - how much targets agree */
  consensus: number;
  /** Number of targets agreeing with direction */
  targetsAgreeing: number;
  /** Total number of targets analyzed */
  targetsTotal: number;
  /** Array of target levels sorted by distance from current */
  targets: TargetLevel[];
  /** Overall conviction level */
  conviction: ConvictionLevel;
  /** Expected move magnitude in points */
  magnitude: number;
  /** Expected move magnitude as percentage */
  magnitudePercent: number;
  /** Whether the signal is actionable (magnitude above threshold) */
  actionable: boolean;
  /** Action threshold used for comparison */
  actionThreshold: number;
  /** Timestamp of signal generation */
  generatedAt: string;
}

/**
 * Summary for displaying multiple target ladder signals
 */
export interface TargetLadderSummary {
  /** Total signals analyzed */
  totalSignals: number;
  /** Number of actionable signals */
  actionableSignals: number;
  /** Number of bullish signals */
  bullishCount: number;
  /** Number of bearish signals */
  bearishCount: number;
  /** Highest conviction signal */
  topSignal: TargetLadderSignal | null;
}

/**
 * Helper to determine conviction level from agreement ratio
 */
export function getConvictionLevel(agreeing: number, total: number): ConvictionLevel {
  const ratio = agreeing / total;
  if (ratio >= 0.8) return "HIGH";
  if (ratio >= 0.6) return "MEDIUM";
  return "LOW";
}

/**
 * Helper to determine if a signal is actionable
 */
export function isSignalActionable(magnitudePercent: number, threshold: number = 0.5): boolean {
  return magnitudePercent >= threshold;
}
