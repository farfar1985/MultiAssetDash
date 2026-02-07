import type { AssetId, SignalDirection } from "@/types";
import type { OHLCData } from "@/components/charts/PriceChart";
import type { SignalHistoryData } from "@/components/charts/SignalChart";
import type { AccuracyData } from "@/components/charts/AccuracyChart";
import { MOCK_ASSETS } from "./mock-data";

// Seeded random for reproducibility
function seededRandom(seed: number) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

// Generate dates for the past N days
function generateDates(days: number): string[] {
  const dates: string[] = [];
  const today = new Date();
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    dates.push(date.toISOString().split("T")[0]);
  }
  return dates;
}

// Generate OHLC data with realistic patterns
function generateOHLCData(
  assetId: AssetId,
  days: number,
  basePrice: number
): OHLCData[] {
  const dates = generateDates(days);
  const data: OHLCData[] = [];

  let currentPrice = basePrice * (0.85 + seededRandom(assetId.length) * 0.3);
  const volatility = assetId === "bitcoin" ? 0.04 : assetId === "natural-gas" ? 0.035 : 0.02;

  // Create trend periods
  let trend = seededRandom(assetId.length * 2) > 0.5 ? 1 : -1;
  let trendDuration = 0;
  const maxTrendDuration = 15 + Math.floor(seededRandom(assetId.length * 3) * 20);

  for (let i = 0; i < days; i++) {
    const seed = assetId.length * 1000 + i;
    const random = seededRandom(seed);

    // Change trend periodically
    trendDuration++;
    if (trendDuration > maxTrendDuration && seededRandom(seed * 2) > 0.7) {
      trend = -trend;
      trendDuration = 0;
    }

    // Calculate price movement
    const trendBias = trend * 0.002;
    const change = (random - 0.5) * volatility * currentPrice + trendBias * currentPrice;

    const open = currentPrice;
    const close = currentPrice + change;

    // High and low based on intraday volatility
    const intradayVol = volatility * 0.7;
    const high = Math.max(open, close) * (1 + seededRandom(seed * 3) * intradayVol);
    const low = Math.min(open, close) * (1 - seededRandom(seed * 4) * intradayVol);

    // Volume with some randomness and trend correlation
    const baseVolume = 100000 + seededRandom(seed * 5) * 50000;
    const volumeMultiplier = 1 + Math.abs(change / currentPrice) * 5;
    const volume = Math.round(baseVolume * volumeMultiplier);

    data.push({
      date: dates[i],
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume,
    });

    currentPrice = close;
  }

  return data;
}

// Generate signal history with realistic patterns
function generateSignalHistory(assetId: AssetId, days: number): SignalHistoryData[] {
  const dates = generateDates(days);
  const data: SignalHistoryData[] = [];

  let momentum = (seededRandom(assetId.length * 10) - 0.5) * 60;

  for (let i = 0; i < days; i++) {
    const seed = assetId.length * 2000 + i;

    // Momentum with mean reversion
    const meanReversion = -momentum * 0.05;
    const noise = (seededRandom(seed) - 0.5) * 20;
    momentum = Math.max(-100, Math.min(100, momentum + meanReversion + noise));

    const strength = Math.round(momentum);
    const direction: SignalDirection =
      strength > 30 ? "bullish" : strength < -30 ? "bearish" : "neutral";

    // Confidence correlates with strength magnitude
    const baseConfidence = 50 + Math.abs(strength) * 0.4;
    const confidenceNoise = (seededRandom(seed * 2) - 0.5) * 15;
    const confidence = Math.round(
      Math.max(40, Math.min(95, baseConfidence + confidenceNoise))
    );

    data.push({
      date: dates[i],
      direction,
      confidence,
      strength,
    });
  }

  return data;
}

// Generate accuracy history with realistic patterns
function generateAccuracyHistory(assetId: AssetId, days: number): AccuracyData[] {
  const dates = generateDates(days);
  const data: AccuracyData[] = [];

  // Base accuracy varies by asset
  const baseAccuracy = 52 + seededRandom(assetId.length * 20) * 10;

  let accuracy30d = baseAccuracy + (seededRandom(assetId.length * 21) - 0.5) * 8;
  let accuracy60d = baseAccuracy + (seededRandom(assetId.length * 22) - 0.5) * 6;
  let accuracy90d = baseAccuracy + (seededRandom(assetId.length * 23) - 0.5) * 4;

  for (let i = 0; i < days; i++) {
    const seed = assetId.length * 3000 + i;

    // Random walk with mean reversion
    const revert30 = (baseAccuracy - accuracy30d) * 0.1;
    const revert60 = (baseAccuracy - accuracy60d) * 0.08;
    const revert90 = (baseAccuracy - accuracy90d) * 0.06;

    accuracy30d = Math.max(45, Math.min(68, accuracy30d + revert30 + (seededRandom(seed) - 0.5) * 3));
    accuracy60d = Math.max(47, Math.min(65, accuracy60d + revert60 + (seededRandom(seed * 2) - 0.5) * 2));
    accuracy90d = Math.max(48, Math.min(63, accuracy90d + revert90 + (seededRandom(seed * 3) - 0.5) * 1.5));

    data.push({
      date: dates[i],
      accuracy30d: Number(accuracy30d.toFixed(1)),
      accuracy60d: Number(accuracy60d.toFixed(1)),
      accuracy90d: Number(accuracy90d.toFixed(1)),
    });
  }

  return data;
}

// Generate model agreement breakdown
export interface ModelAgreement {
  bullishCount: number;
  bearishCount: number;
  neutralCount: number;
  totalModels: number;
  overallDirection: SignalDirection;
}

function generateModelAgreement(assetId: AssetId): ModelAgreement {
  const totalModels = 10179;
  const seed = assetId.length * 4000;

  // Get overall direction from mock signals
  const directions: SignalDirection[] = ["bullish", "bearish", "neutral"];
  const directionIndex = Math.floor(seededRandom(seed) * 3);
  const overallDirection = directions[directionIndex];

  // Generate counts based on overall direction
  let bullishPct: number, bearishPct: number;

  if (overallDirection === "bullish") {
    bullishPct = 0.5 + seededRandom(seed * 2) * 0.35;
    bearishPct = (1 - bullishPct) * (0.3 + seededRandom(seed * 3) * 0.4);
  } else if (overallDirection === "bearish") {
    bearishPct = 0.5 + seededRandom(seed * 2) * 0.35;
    bullishPct = (1 - bearishPct) * (0.3 + seededRandom(seed * 3) * 0.4);
  } else {
    const split = 0.25 + seededRandom(seed * 2) * 0.15;
    bullishPct = split;
    bearishPct = split + (seededRandom(seed * 3) - 0.5) * 0.1;
  }

  const bullishCount = Math.round(totalModels * bullishPct);
  const bearishCount = Math.round(totalModels * bearishPct);
  const neutralCount = totalModels - bullishCount - bearishCount;

  return {
    bullishCount,
    bearishCount,
    neutralCount,
    totalModels,
    overallDirection,
  };
}

// Pre-generated data for all assets (90 days)
const DAYS = 90;

export const MOCK_OHLC_DATA: Partial<Record<AssetId, OHLCData[]> = {
  "crude-oil": generateOHLCData("crude-oil", DAYS, MOCK_ASSETS["crude-oil"].currentPrice),
  bitcoin: generateOHLCData("bitcoin", DAYS, MOCK_ASSETS.bitcoin.currentPrice),
  gold: generateOHLCData("gold", DAYS, MOCK_ASSETS.gold.currentPrice),
  silver: generateOHLCData("silver", DAYS, MOCK_ASSETS.silver.currentPrice),
  "natural-gas": generateOHLCData("natural-gas", DAYS, MOCK_ASSETS["natural-gas"].currentPrice),
  copper: generateOHLCData("copper", DAYS, MOCK_ASSETS.copper.currentPrice),
  wheat: generateOHLCData("wheat", DAYS, MOCK_ASSETS.wheat.currentPrice),
  corn: generateOHLCData("corn", DAYS, MOCK_ASSETS.corn.currentPrice),
  soybean: generateOHLCData("soybean", DAYS, MOCK_ASSETS.soybean.currentPrice),
  platinum: generateOHLCData("platinum", DAYS, MOCK_ASSETS.platinum.currentPrice),
};

export const MOCK_SIGNAL_HISTORY: Partial<Record<AssetId, SignalHistoryData[]> = {
  "crude-oil": generateSignalHistory("crude-oil", DAYS),
  bitcoin: generateSignalHistory("bitcoin", DAYS),
  gold: generateSignalHistory("gold", DAYS),
  silver: generateSignalHistory("silver", DAYS),
  "natural-gas": generateSignalHistory("natural-gas", DAYS),
  copper: generateSignalHistory("copper", DAYS),
  wheat: generateSignalHistory("wheat", DAYS),
  corn: generateSignalHistory("corn", DAYS),
  soybean: generateSignalHistory("soybean", DAYS),
  platinum: generateSignalHistory("platinum", DAYS),
};

export const MOCK_ACCURACY_HISTORY: Partial<Record<AssetId, AccuracyData[]> = {
  "crude-oil": generateAccuracyHistory("crude-oil", DAYS),
  bitcoin: generateAccuracyHistory("bitcoin", DAYS),
  gold: generateAccuracyHistory("gold", DAYS),
  silver: generateAccuracyHistory("silver", DAYS),
  "natural-gas": generateAccuracyHistory("natural-gas", DAYS),
  copper: generateAccuracyHistory("copper", DAYS),
  wheat: generateAccuracyHistory("wheat", DAYS),
  corn: generateAccuracyHistory("corn", DAYS),
  soybean: generateAccuracyHistory("soybean", DAYS),
  platinum: generateAccuracyHistory("platinum", DAYS),
};

export const MOCK_MODEL_AGREEMENT: Partial<Record<AssetId, ModelAgreement> = {
  "crude-oil": generateModelAgreement("crude-oil"),
  bitcoin: generateModelAgreement("bitcoin"),
  gold: generateModelAgreement("gold"),
  silver: generateModelAgreement("silver"),
  "natural-gas": generateModelAgreement("natural-gas"),
  copper: generateModelAgreement("copper"),
  wheat: generateModelAgreement("wheat"),
  corn: generateModelAgreement("corn"),
  soybean: generateModelAgreement("soybean"),
  platinum: generateModelAgreement("platinum"),
};

// Sparkline data for metrics cards (last 14 days)
export function generateSparklineData(trend: "up" | "down" | "flat"): number[] {
  const data: number[] = [];
  let value = 50;
  const trendBias = trend === "up" ? 0.8 : trend === "down" ? -0.8 : 0;

  for (let i = 0; i < 14; i++) {
    value += (Math.random() - 0.5 + trendBias) * 5;
    value = Math.max(20, Math.min(80, value));
    data.push(Number(value.toFixed(1)));
  }

  return data;
}

// Helper to get chart data for an asset
export function getChartData(assetId: AssetId) {
  return {
    ohlc: MOCK_OHLC_DATA[assetId],
    signalHistory: MOCK_SIGNAL_HISTORY[assetId],
    accuracyHistory: MOCK_ACCURACY_HISTORY[assetId],
    modelAgreement: MOCK_MODEL_AGREEMENT[assetId],
  };
}
