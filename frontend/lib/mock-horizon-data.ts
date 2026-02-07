/**
 * Mock Horizon Pair Data
 *
 * Based on Amira's findings from the quantum_ml pipeline analysis.
 * Key insight: Crude Oil D+1 vs D+3 shows 71.2% accuracy.
 */

import type {
  HorizonPairMatrix,
  HorizonPair,
  Horizon,
} from "@/types/horizon-pairs";
import type { AssetId } from "@/types";

const ALL_HORIZONS: Horizon[] = ["D+1", "D+2", "D+3", "D+5", "D+7", "D+10"];

/**
 * Generate all horizon pairs for a matrix
 */
function generatePairs(
  accuracyMap: Record<string, number>,
  baseSampleSize: number = 500
): HorizonPair[] {
  const pairs: HorizonPair[] = [];

  for (const h1 of ALL_HORIZONS) {
    for (const h2 of ALL_HORIZONS) {
      if (h1 === h2) continue; // Skip diagonal

      const key = `${h1}_${h2}`;
      const reverseKey = `${h2}_${h1}`;

      // Use provided accuracy or generate reasonable default
      let accuracy = accuracyMap[key] ?? accuracyMap[reverseKey];
      if (accuracy === undefined) {
        // Generate plausible accuracy based on horizon distance
        const h1Days = parseInt(h1.replace("D+", ""));
        const h2Days = parseInt(h2.replace("D+", ""));
        const distance = Math.abs(h1Days - h2Days);

        // Closer horizons tend to correlate better
        accuracy = 52 + Math.random() * 6 - distance * 0.5;
        accuracy = Math.max(48, Math.min(58, accuracy));
      }

      // Vary sample size slightly
      const sampleSize =
        baseSampleSize + Math.floor(Math.random() * 100) - 50;

      pairs.push({
        h1,
        h2,
        accuracy: Math.round(accuracy * 10) / 10,
        sampleSize,
        pValue: accuracy > 55 ? 0.01 : accuracy > 52 ? 0.05 : 0.1,
        confidenceInterval: [accuracy - 2.5, accuracy + 2.5],
      });
    }
  }

  return pairs;
}

/**
 * Crude Oil - Best performer based on Amira's findings
 * D+1 vs D+3 shows exceptional 71.2% accuracy
 */
export const CRUDE_OIL_HORIZON_DATA: HorizonPairMatrix = {
  asset: "crude-oil",
  assetName: "Crude Oil",
  pairs: generatePairs({
    // Amira's key finding
    "D+1_D+3": 71.2,
    "D+3_D+1": 71.2,
    // Other notable pairs
    "D+1_D+5": 63.8,
    "D+5_D+1": 63.8,
    "D+2_D+3": 58.4,
    "D+3_D+2": 58.4,
    "D+1_D+2": 56.7,
    "D+2_D+1": 56.7,
    "D+3_D+5": 61.2,
    "D+5_D+3": 61.2,
    "D+5_D+7": 55.3,
    "D+7_D+5": 55.3,
    "D+7_D+10": 54.1,
    "D+10_D+7": 54.1,
  }),
  calculatedAt: new Date().toISOString(),
  periodStart: "2023-01-01",
  periodEnd: "2024-12-31",
};

/**
 * Bitcoin - Random walk characteristics, weak pair correlations
 */
export const BITCOIN_HORIZON_DATA: HorizonPairMatrix = {
  asset: "bitcoin",
  assetName: "Bitcoin",
  pairs: generatePairs({
    // All pairs cluster around 50-54% (near random)
    "D+1_D+2": 52.3,
    "D+2_D+1": 52.3,
    "D+1_D+3": 51.8,
    "D+3_D+1": 51.8,
    "D+2_D+3": 53.1,
    "D+3_D+2": 53.1,
    "D+1_D+5": 50.4,
    "D+5_D+1": 50.4,
    "D+3_D+5": 52.7,
    "D+5_D+3": 52.7,
    "D+5_D+7": 51.2,
    "D+7_D+5": 51.2,
    "D+7_D+10": 50.9,
    "D+10_D+7": 50.9,
  }),
  calculatedAt: new Date().toISOString(),
  periodStart: "2023-01-01",
  periodEnd: "2024-12-31",
};

/**
 * S&P 500 - Moderate predictability, best pair at 55.2%
 */
export const SP500_HORIZON_DATA: HorizonPairMatrix = {
  asset: "gold" as AssetId, // Using gold as proxy for S&P 500
  assetName: "S&P 500",
  pairs: generatePairs({
    "D+1_D+2": 55.2,
    "D+2_D+1": 55.2,
    "D+1_D+3": 54.8,
    "D+3_D+1": 54.8,
    "D+2_D+3": 53.9,
    "D+3_D+2": 53.9,
    "D+1_D+5": 52.1,
    "D+5_D+1": 52.1,
    "D+3_D+5": 54.3,
    "D+5_D+3": 54.3,
    "D+5_D+7": 51.7,
    "D+7_D+5": 51.7,
    "D+7_D+10": 50.8,
    "D+10_D+7": 50.8,
  }),
  calculatedAt: new Date().toISOString(),
  periodStart: "2023-01-01",
  periodEnd: "2024-12-31",
};

/**
 * Gold - Moderate correlations with some alpha
 */
export const GOLD_HORIZON_DATA: HorizonPairMatrix = {
  asset: "gold",
  assetName: "Gold",
  pairs: generatePairs({
    "D+1_D+3": 58.7,
    "D+3_D+1": 58.7,
    "D+1_D+2": 56.2,
    "D+2_D+1": 56.2,
    "D+2_D+5": 55.4,
    "D+5_D+2": 55.4,
    "D+3_D+5": 57.1,
    "D+5_D+3": 57.1,
    "D+5_D+7": 53.8,
    "D+7_D+5": 53.8,
    "D+7_D+10": 52.4,
    "D+10_D+7": 52.4,
  }),
  calculatedAt: new Date().toISOString(),
  periodStart: "2023-01-01",
  periodEnd: "2024-12-31",
};

/**
 * Natural Gas - High volatility, some exploitable patterns
 */
export const NATURAL_GAS_HORIZON_DATA: HorizonPairMatrix = {
  asset: "natural-gas",
  assetName: "Natural Gas",
  pairs: generatePairs({
    "D+1_D+3": 62.4,
    "D+3_D+1": 62.4,
    "D+1_D+5": 59.8,
    "D+5_D+1": 59.8,
    "D+2_D+5": 57.3,
    "D+5_D+2": 57.3,
    "D+3_D+7": 56.1,
    "D+7_D+3": 56.1,
    "D+5_D+10": 54.2,
    "D+10_D+5": 54.2,
  }),
  calculatedAt: new Date().toISOString(),
  periodStart: "2023-01-01",
  periodEnd: "2024-12-31",
};

/**
 * All mock data by asset
 */
export const MOCK_HORIZON_DATA: Partial<Record<AssetId, HorizonPairMatrix>> = {
  "crude-oil": CRUDE_OIL_HORIZON_DATA,
  bitcoin: BITCOIN_HORIZON_DATA,
  gold: GOLD_HORIZON_DATA,
  "natural-gas": NATURAL_GAS_HORIZON_DATA,
  // Placeholder for other assets - generate with defaults
  silver: {
    ...GOLD_HORIZON_DATA,
    asset: "silver",
    assetName: "Silver",
    pairs: generatePairs({
      "D+1_D+3": 56.3,
      "D+3_D+1": 56.3,
    }),
  },
  copper: {
    ...GOLD_HORIZON_DATA,
    asset: "copper",
    assetName: "Copper",
    pairs: generatePairs({
      "D+1_D+3": 57.8,
      "D+3_D+1": 57.8,
    }),
  },
  wheat: {
    ...GOLD_HORIZON_DATA,
    asset: "wheat",
    assetName: "Wheat",
    pairs: generatePairs({
      "D+1_D+5": 59.2,
      "D+5_D+1": 59.2,
    }),
  },
  corn: {
    ...GOLD_HORIZON_DATA,
    asset: "corn",
    assetName: "Corn",
    pairs: generatePairs({
      "D+2_D+5": 58.1,
      "D+5_D+2": 58.1,
    }),
  },
  soybean: {
    ...GOLD_HORIZON_DATA,
    asset: "soybean",
    assetName: "Soybean",
    pairs: generatePairs({
      "D+1_D+3": 57.4,
      "D+3_D+1": 57.4,
    }),
  },
  platinum: {
    ...GOLD_HORIZON_DATA,
    asset: "platinum",
    assetName: "Platinum",
    pairs: generatePairs({
      "D+1_D+3": 55.9,
      "D+3_D+1": 55.9,
    }),
  },
};

/**
 * Get horizon pair data for an asset
 */
export function getHorizonPairData(assetId: AssetId): HorizonPairMatrix {
  return MOCK_HORIZON_DATA[assetId] ?? CRUDE_OIL_HORIZON_DATA;
}

/**
 * Get all assets with horizon data
 */
export function getAssetsWithHorizonData(): AssetId[] {
  return Object.keys(MOCK_HORIZON_DATA) as AssetId[];
}

/**
 * Find best pair for an asset
 */
export function findBestPair(matrix: HorizonPairMatrix): HorizonPair {
  return matrix.pairs.reduce((best, current) =>
    current.accuracy > best.accuracy ? current : best
  );
}

/**
 * Calculate average accuracy for a matrix
 */
export function calculateAverageAccuracy(matrix: HorizonPairMatrix): number {
  const sum = matrix.pairs.reduce((acc, pair) => acc + pair.accuracy, 0);
  return Math.round((sum / matrix.pairs.length) * 10) / 10;
}

/**
 * Count alpha source pairs (accuracy >= 65%)
 */
export function countAlphaPairs(matrix: HorizonPairMatrix): number {
  return matrix.pairs.filter((p) => p.accuracy >= 65).length;
}
