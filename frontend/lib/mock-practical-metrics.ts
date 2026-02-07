/**
 * Mock Practical Metrics Data
 *
 * Generates realistic practical metrics for CME hedging desk utility.
 * Includes actionability scores, forecast magnitudes, and position sizing.
 */

import type { AssetId } from "@/types";
import type { Horizon } from "@/types/horizon-pairs";
import type {
  PracticalMetricsData,
  ActionableSignal,
  ForecastMagnitude,
  HorizonCoverage,
  PracticalScore,
  TimeToAction,
  ActionabilityLevel,
} from "@/types/practical-metrics";
import { MOCK_ASSETS, MOCK_SIGNALS } from "./mock-data";

const ALL_HORIZONS: Horizon[] = ["D+1", "D+2", "D+3", "D+5", "D+7", "D+10"];

// Seeded random for reproducibility
function seededRandom(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

/**
 * Asset-specific move thresholds
 */
const ASSET_THRESHOLDS: Partial<Record<AssetId, number>> = {
  "crude-oil": 1.0,
  bitcoin: 500,
  gold: 10,
  silver: 0.25,
  "natural-gas": 0.1,
  copper: 0.05,
  wheat: 5,
  corn: 3,
  soybean: 8,
  platinum: 15,
};

/**
 * Generate forecast magnitude for an asset
 */
function generateForecastMagnitude(
  assetId: AssetId,
  currentPrice: number,
  seed: number
): ForecastMagnitude {
  const threshold = ASSET_THRESHOLDS[assetId] ?? 1;
  const random = seededRandom(seed);

  // Generate a move that could be above or below threshold
  // 60% chance of being actionable
  const isActionable = random > 0.4;
  const multiplier = isActionable ? 1.2 + random * 2.5 : 0.3 + random * 0.6;
  const absoluteMove = threshold * multiplier;

  // Direction based on signal
  const signal = MOCK_SIGNALS[assetId]?.["D+1"];
  const direction =
    signal?.direction === "bearish" ? "down" : signal?.direction === "bullish" ? "up" : "up";
  const predictedMove = direction === "down" ? -absoluteMove : absoluteMove;

  const movePercent = (absoluteMove / currentPrice) * 100;
  const thresholdMultiple = absoluteMove / threshold;

  return {
    predictedMove,
    absoluteMove,
    direction,
    isActionable: absoluteMove >= threshold,
    movePercent,
    threshold,
    thresholdMultiple,
  };
}

/**
 * Generate horizon coverage
 */
function generateHorizonCoverage(assetId: AssetId, seed: number): HorizonCoverage {
  const random = seededRandom(seed);
  const coverageLevel = random > 0.6 ? "high" : random > 0.3 ? "medium" : "low";

  let coveredHorizons: Horizon[];
  if (coverageLevel === "high") {
    coveredHorizons = ["D+1", "D+2", "D+3", "D+5", "D+7", "D+10"];
  } else if (coverageLevel === "medium") {
    coveredHorizons = ["D+1", "D+3", "D+5", "D+7"];
  } else {
    coveredHorizons = ["D+1", "D+5"];
  }

  const missingHorizons = ALL_HORIZONS.filter((h) => !coveredHorizons.includes(h));
  const coveragePercent = (coveredHorizons.length / ALL_HORIZONS.length) * 100;

  const hasShortTerm = coveredHorizons.some((h) => h === "D+1" || h === "D+2");
  const hasMediumTerm = coveredHorizons.some((h) => h === "D+3" || h === "D+5");
  const hasLongTerm = coveredHorizons.some((h) => h === "D+7" || h === "D+10");

  // Optimal horizon based on asset
  const optimalHorizons: Partial<Record<AssetId, Horizon>> = {
    "crude-oil": "D+3",
    bitcoin: "D+5",
    gold: "D+3",
    silver: "D+5",
    "natural-gas": "D+1",
    copper: "D+5",
    wheat: "D+3",
    corn: "D+5",
    soybean: "D+3",
    platinum: "D+5",
  };

  return {
    coveredHorizons,
    missingHorizons,
    coveragePercent,
    hasShortTerm,
    hasMediumTerm,
    hasLongTerm,
    optimalHorizon: optimalHorizons[assetId] ?? null,
  };
}

/**
 * Generate time to action
 */
function generateTimeToAction(
  actionability: ActionabilityLevel,
  seed: number
): TimeToAction {
  const random = seededRandom(seed);

  if (actionability === "high") {
    return {
      urgency: random > 0.5 ? "immediate" : "today",
      daysToOptimalEntry: 0,
      daysUntilExpiry: Math.floor(random * 3) + 2,
      reason: "Strong signal with favorable conditions. Act within trading session.",
    };
  } else if (actionability === "medium") {
    return {
      urgency: "this_week",
      daysToOptimalEntry: Math.floor(random * 2) + 1,
      daysUntilExpiry: Math.floor(random * 5) + 3,
      reason: "Moderate signal. Wait for confirmation or better entry point.",
    };
  } else {
    return {
      urgency: "monitor",
      daysToOptimalEntry: Math.floor(random * 5) + 3,
      daysUntilExpiry: Math.floor(random * 7) + 5,
      reason: "Weak signal. Monitor for strengthening before action.",
    };
  }
}

/**
 * Generate actionable signal reasons
 */
function generateReasons(
  magnitude: ForecastMagnitude,
  coverage: HorizonCoverage,
  confidence: number,
  bigMoveWinRate: number
): string[] {
  const reasons: string[] = [];

  if (magnitude.isActionable) {
    reasons.push(
      `Forecast of ${magnitude.thresholdMultiple.toFixed(1)}x the minimum threshold`
    );
  } else {
    reasons.push(`Forecast below ${magnitude.threshold} minimum threshold`);
  }

  if (coverage.coveragePercent >= 67) {
    reasons.push("Strong horizon diversity across short, medium, and long-term");
  } else if (coverage.coveragePercent >= 50) {
    reasons.push("Moderate horizon coverage");
  } else {
    reasons.push("Limited horizon diversity");
  }

  if (confidence >= 70) {
    reasons.push(`High model confidence (${confidence.toFixed(0)}%)`);
  } else if (confidence >= 50) {
    reasons.push(`Moderate model confidence (${confidence.toFixed(0)}%)`);
  } else {
    reasons.push(`Low model confidence (${confidence.toFixed(0)}%)`);
  }

  if (bigMoveWinRate >= 60) {
    reasons.push(`Strong big-move accuracy (${bigMoveWinRate.toFixed(0)}%)`);
  } else if (bigMoveWinRate >= 50) {
    reasons.push(`Average big-move accuracy (${bigMoveWinRate.toFixed(0)}%)`);
  } else {
    reasons.push(`Below-average big-move accuracy (${bigMoveWinRate.toFixed(0)}%)`);
  }

  return reasons;
}

/**
 * Generate practical metrics for an asset
 */
export function generatePracticalMetrics(assetId: AssetId): PracticalMetricsData {
  const asset = MOCK_ASSETS[assetId];
  const signal = MOCK_SIGNALS[assetId]?.["D+1"];

  if (!asset || !signal) {
    throw new Error(`No data for asset: ${assetId}`);
  }

  const seed = assetId.length * 1000;
  const currentPrice = asset.currentPrice;

  // Generate components
  const magnitude = generateForecastMagnitude(assetId, currentPrice, seed);
  const horizonCoverage = generateHorizonCoverage(assetId, seed + 1);

  // Use signal confidence
  const confidence = signal.confidence;

  // Big move win rate (slightly higher than overall)
  const bigMoveWinRate = signal.directionalAccuracy + seededRandom(seed + 2) * 8;

  // Calculate component scores
  const magnitudeScore = magnitude.isActionable
    ? 60 + Math.min(40, magnitude.thresholdMultiple * 15)
    : magnitude.thresholdMultiple * 50;

  const horizonScore = horizonCoverage.coveragePercent;

  const confidenceScore =
    confidence >= 80
      ? 90 + (confidence - 80) * 0.5
      : confidence >= 60
        ? 60 + (confidence - 60) * 1.5
        : confidence * 1;

  const bigMoveAccuracyScore =
    bigMoveWinRate >= 65
      ? 80 + (bigMoveWinRate - 65) * 1.3
      : bigMoveWinRate >= 55
        ? 50 + (bigMoveWinRate - 55) * 3
        : bigMoveWinRate * 0.9;

  // Calculate practical score
  const practicalScore: PracticalScore = {
    score:
      magnitudeScore * 0.35 +
      horizonScore * 0.2 +
      confidenceScore * 0.2 +
      bigMoveAccuracyScore * 0.25,
    actionability:
      magnitudeScore * 0.35 +
        horizonScore * 0.2 +
        confidenceScore * 0.2 +
        bigMoveAccuracyScore * 0.25 >=
      70
        ? "high"
        : magnitudeScore * 0.35 +
              horizonScore * 0.2 +
              confidenceScore * 0.2 +
              bigMoveAccuracyScore * 0.25 >=
            45
          ? "medium"
          : "low",
    components: {
      magnitudeScore: Math.round(magnitudeScore),
      horizonScore: Math.round(horizonScore),
      confidenceScore: Math.round(confidenceScore),
      bigMoveAccuracyScore: Math.round(bigMoveAccuracyScore),
    },
    weights: {
      magnitude: 0.35,
      horizon: 0.2,
      confidence: 0.2,
      bigMoveAccuracy: 0.25,
    },
  };

  // Calculate position size
  const recommendedPositionSize = Math.round(
    Math.min(
      100,
      practicalScore.score * 0.6 * (confidence >= 70 ? 1.2 : confidence < 50 ? 0.7 : 1)
    )
  );

  // Generate reasons
  const reasons = generateReasons(magnitude, horizonCoverage, confidence, bigMoveWinRate);

  // Time to action
  const timeToAction = generateTimeToAction(practicalScore.actionability, seed + 3);

  // Build actionable signal
  const actionableSignal: ActionableSignal = {
    isActionable: practicalScore.actionability === "high",
    reasons,
    magnitude,
    horizonCoverage,
    practicalScore,
    confidence,
    bigMoveWinRate: Math.round(bigMoveWinRate * 10) / 10,
    recommendedPositionSize,
    timeToAction,
  };

  return {
    asset: assetId,
    currentPrice,
    signal: actionableSignal,
    traditionalSharpe: signal.sharpeRatio,
    overallWinRate: signal.directionalAccuracy,
    analyzedAt: new Date().toISOString(),
  };
}

/**
 * Pre-generated practical metrics for all assets
 */
export const MOCK_PRACTICAL_METRICS: Partial<Record<AssetId, PracticalMetricsData>> = {
  "crude-oil": generatePracticalMetrics("crude-oil"),
  bitcoin: generatePracticalMetrics("bitcoin"),
  gold: generatePracticalMetrics("gold"),
  silver: generatePracticalMetrics("silver"),
  "natural-gas": generatePracticalMetrics("natural-gas"),
  copper: generatePracticalMetrics("copper"),
  wheat: generatePracticalMetrics("wheat"),
  corn: generatePracticalMetrics("corn"),
  soybean: generatePracticalMetrics("soybean"),
  platinum: generatePracticalMetrics("platinum"),
};

/**
 * Get practical metrics for an asset
 */
export function getPracticalMetrics(assetId: AssetId): PracticalMetricsData {
  return MOCK_PRACTICAL_METRICS[assetId] ?? generatePracticalMetrics(assetId);
}
