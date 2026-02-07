import type { AssetId, SignalDirection } from "@/types";

// Horizon types
export type Horizon = "D+1" | "D+5" | "D+10";

// Asset with price data
export interface AssetData {
  id: AssetId;
  name: string;
  symbol: string;
  currentPrice: number;
  change24h: number;
  changePercent24h: number;
}

// Signal data with model agreement
export interface SignalData {
  assetId: AssetId;
  direction: SignalDirection;
  confidence: number;
  horizon: Horizon;
  modelsAgreeing: number;
  modelsTotal: number;
  sharpeRatio: number;
  directionalAccuracy: number;
  totalReturn: number;
}

// Performance metrics
export interface PerformanceMetrics {
  sharpeRatio: number;
  directionalAccuracy: number;
  totalReturn: number;
  maxDrawdown: number;
  winRate: number;
  modelCount: number;
}

// Mock asset data (partial - not all assets have mock data)
export const MOCK_ASSETS: Partial<Record<AssetId, AssetData>> = {
  "crude-oil": {
    id: "crude-oil",
    name: "Crude Oil",
    symbol: "CL",
    currentPrice: 78.42,
    change24h: 1.23,
    changePercent24h: 1.59,
  },
  bitcoin: {
    id: "bitcoin",
    name: "Bitcoin",
    symbol: "BTC",
    currentPrice: 97432.15,
    change24h: -1254.32,
    changePercent24h: -1.27,
  },
  gold: {
    id: "gold",
    name: "Gold",
    symbol: "GC",
    currentPrice: 2891.50,
    change24h: 12.80,
    changePercent24h: 0.44,
  },
  silver: {
    id: "silver",
    name: "Silver",
    symbol: "SI",
    currentPrice: 32.15,
    change24h: -0.28,
    changePercent24h: -0.86,
  },
  "natural-gas": {
    id: "natural-gas",
    name: "Natural Gas",
    symbol: "NG",
    currentPrice: 3.42,
    change24h: 0.15,
    changePercent24h: 4.59,
  },
  copper: {
    id: "copper",
    name: "Copper",
    symbol: "HG",
    currentPrice: 4.28,
    change24h: 0.03,
    changePercent24h: 0.71,
  },
  wheat: {
    id: "wheat",
    name: "Wheat",
    symbol: "ZW",
    currentPrice: 542.25,
    change24h: -8.50,
    changePercent24h: -1.54,
  },
  corn: {
    id: "corn",
    name: "Corn",
    symbol: "ZC",
    currentPrice: 445.75,
    change24h: 3.25,
    changePercent24h: 0.73,
  },
  soybean: {
    id: "soybean",
    name: "Soybean",
    symbol: "ZS",
    currentPrice: 1028.50,
    change24h: -12.75,
    changePercent24h: -1.22,
  },
  platinum: {
    id: "platinum",
    name: "Platinum",
    symbol: "PL",
    currentPrice: 1042.30,
    change24h: 8.20,
    changePercent24h: 0.79,
  },
};

// Mock signals by asset and horizon
export const MOCK_SIGNALS: Partial<Record<AssetId, Record<Horizon, SignalData>> = {
  "crude-oil": {
    "D+1": {
      assetId: "crude-oil",
      direction: "bullish",
      confidence: 78,
      horizon: "D+1",
      modelsAgreeing: 8432,
      modelsTotal: 10179,
      sharpeRatio: 2.34,
      directionalAccuracy: 58.2,
      totalReturn: 45.8,
    },
    "D+5": {
      assetId: "crude-oil",
      direction: "bullish",
      confidence: 72,
      horizon: "D+5",
      modelsAgreeing: 7891,
      modelsTotal: 10179,
      sharpeRatio: 2.18,
      directionalAccuracy: 56.4,
      totalReturn: 42.1,
    },
    "D+10": {
      assetId: "crude-oil",
      direction: "neutral",
      confidence: 54,
      horizon: "D+10",
      modelsAgreeing: 5512,
      modelsTotal: 10179,
      sharpeRatio: 1.92,
      directionalAccuracy: 53.8,
      totalReturn: 38.2,
    },
  },
  bitcoin: {
    "D+1": {
      assetId: "bitcoin",
      direction: "bearish",
      confidence: 65,
      horizon: "D+1",
      modelsAgreeing: 6621,
      modelsTotal: 10179,
      sharpeRatio: 1.87,
      directionalAccuracy: 54.1,
      totalReturn: 67.3,
    },
    "D+5": {
      assetId: "bitcoin",
      direction: "bullish",
      confidence: 71,
      horizon: "D+5",
      modelsAgreeing: 7234,
      modelsTotal: 10179,
      sharpeRatio: 2.12,
      directionalAccuracy: 55.8,
      totalReturn: 72.4,
    },
    "D+10": {
      assetId: "bitcoin",
      direction: "bullish",
      confidence: 82,
      horizon: "D+10",
      modelsAgreeing: 8351,
      modelsTotal: 10179,
      sharpeRatio: 2.56,
      directionalAccuracy: 59.2,
      totalReturn: 89.1,
    },
  },
  gold: {
    "D+1": {
      assetId: "gold",
      direction: "bullish",
      confidence: 84,
      horizon: "D+1",
      modelsAgreeing: 8556,
      modelsTotal: 10179,
      sharpeRatio: 2.78,
      directionalAccuracy: 61.4,
      totalReturn: 32.5,
    },
    "D+5": {
      assetId: "gold",
      direction: "bullish",
      confidence: 79,
      horizon: "D+5",
      modelsAgreeing: 8043,
      modelsTotal: 10179,
      sharpeRatio: 2.45,
      directionalAccuracy: 58.9,
      totalReturn: 29.8,
    },
    "D+10": {
      assetId: "gold",
      direction: "bullish",
      confidence: 76,
      horizon: "D+10",
      modelsAgreeing: 7738,
      modelsTotal: 10179,
      sharpeRatio: 2.31,
      directionalAccuracy: 57.2,
      totalReturn: 28.1,
    },
  },
  silver: {
    "D+1": {
      assetId: "silver",
      direction: "bearish",
      confidence: 62,
      horizon: "D+1",
      modelsAgreeing: 6315,
      modelsTotal: 10179,
      sharpeRatio: 1.68,
      directionalAccuracy: 52.8,
      totalReturn: 24.3,
    },
    "D+5": {
      assetId: "silver",
      direction: "neutral",
      confidence: 51,
      horizon: "D+5",
      modelsAgreeing: 5192,
      modelsTotal: 10179,
      sharpeRatio: 1.54,
      directionalAccuracy: 51.2,
      totalReturn: 21.8,
    },
    "D+10": {
      assetId: "silver",
      direction: "bullish",
      confidence: 68,
      horizon: "D+10",
      modelsAgreeing: 6923,
      modelsTotal: 10179,
      sharpeRatio: 1.89,
      directionalAccuracy: 54.5,
      totalReturn: 26.7,
    },
  },
  "natural-gas": {
    "D+1": {
      assetId: "natural-gas",
      direction: "bullish",
      confidence: 91,
      horizon: "D+1",
      modelsAgreeing: 9263,
      modelsTotal: 10179,
      sharpeRatio: 3.12,
      directionalAccuracy: 64.8,
      totalReturn: 98.4,
    },
    "D+5": {
      assetId: "natural-gas",
      direction: "bullish",
      confidence: 85,
      horizon: "D+5",
      modelsAgreeing: 8652,
      modelsTotal: 10179,
      sharpeRatio: 2.89,
      directionalAccuracy: 62.1,
      totalReturn: 87.2,
    },
    "D+10": {
      assetId: "natural-gas",
      direction: "neutral",
      confidence: 58,
      horizon: "D+10",
      modelsAgreeing: 5904,
      modelsTotal: 10179,
      sharpeRatio: 2.14,
      directionalAccuracy: 55.3,
      totalReturn: 64.8,
    },
  },
  copper: {
    "D+1": {
      assetId: "copper",
      direction: "neutral",
      confidence: 52,
      horizon: "D+1",
      modelsAgreeing: 5293,
      modelsTotal: 10179,
      sharpeRatio: 1.72,
      directionalAccuracy: 52.4,
      totalReturn: 28.9,
    },
    "D+5": {
      assetId: "copper",
      direction: "bullish",
      confidence: 67,
      horizon: "D+5",
      modelsAgreeing: 6821,
      modelsTotal: 10179,
      sharpeRatio: 1.95,
      directionalAccuracy: 54.8,
      totalReturn: 33.4,
    },
    "D+10": {
      assetId: "copper",
      direction: "bullish",
      confidence: 74,
      horizon: "D+10",
      modelsAgreeing: 7534,
      modelsTotal: 10179,
      sharpeRatio: 2.21,
      directionalAccuracy: 57.1,
      totalReturn: 38.7,
    },
  },
  wheat: {
    "D+1": {
      assetId: "wheat",
      direction: "bearish",
      confidence: 73,
      horizon: "D+1",
      modelsAgreeing: 7431,
      modelsTotal: 10179,
      sharpeRatio: 2.08,
      directionalAccuracy: 56.2,
      totalReturn: 31.2,
    },
    "D+5": {
      assetId: "wheat",
      direction: "bearish",
      confidence: 68,
      horizon: "D+5",
      modelsAgreeing: 6923,
      modelsTotal: 10179,
      sharpeRatio: 1.92,
      directionalAccuracy: 54.7,
      totalReturn: 28.4,
    },
    "D+10": {
      assetId: "wheat",
      direction: "neutral",
      confidence: 55,
      horizon: "D+10",
      modelsAgreeing: 5600,
      modelsTotal: 10179,
      sharpeRatio: 1.67,
      directionalAccuracy: 52.9,
      totalReturn: 24.1,
    },
  },
  corn: {
    "D+1": {
      assetId: "corn",
      direction: "bullish",
      confidence: 69,
      horizon: "D+1",
      modelsAgreeing: 7025,
      modelsTotal: 10179,
      sharpeRatio: 1.98,
      directionalAccuracy: 55.3,
      totalReturn: 26.8,
    },
    "D+5": {
      assetId: "corn",
      direction: "neutral",
      confidence: 53,
      horizon: "D+5",
      modelsAgreeing: 5395,
      modelsTotal: 10179,
      sharpeRatio: 1.62,
      directionalAccuracy: 52.1,
      totalReturn: 22.3,
    },
    "D+10": {
      assetId: "corn",
      direction: "bullish",
      confidence: 71,
      horizon: "D+10",
      modelsAgreeing: 7227,
      modelsTotal: 10179,
      sharpeRatio: 2.05,
      directionalAccuracy: 55.8,
      totalReturn: 29.4,
    },
  },
  soybean: {
    "D+1": {
      assetId: "soybean",
      direction: "bearish",
      confidence: 76,
      horizon: "D+1",
      modelsAgreeing: 7738,
      modelsTotal: 10179,
      sharpeRatio: 2.24,
      directionalAccuracy: 57.4,
      totalReturn: 34.8,
    },
    "D+5": {
      assetId: "soybean",
      direction: "bearish",
      confidence: 72,
      horizon: "D+5",
      modelsAgreeing: 7329,
      modelsTotal: 10179,
      sharpeRatio: 2.11,
      directionalAccuracy: 56.1,
      totalReturn: 32.1,
    },
    "D+10": {
      assetId: "soybean",
      direction: "neutral",
      confidence: 56,
      horizon: "D+10",
      modelsAgreeing: 5700,
      modelsTotal: 10179,
      sharpeRatio: 1.78,
      directionalAccuracy: 53.4,
      totalReturn: 27.5,
    },
  },
  platinum: {
    "D+1": {
      assetId: "platinum",
      direction: "bullish",
      confidence: 67,
      horizon: "D+1",
      modelsAgreeing: 6821,
      modelsTotal: 10179,
      sharpeRatio: 1.89,
      directionalAccuracy: 54.6,
      totalReturn: 29.3,
    },
    "D+5": {
      assetId: "platinum",
      direction: "bullish",
      confidence: 74,
      horizon: "D+5",
      modelsAgreeing: 7534,
      modelsTotal: 10179,
      sharpeRatio: 2.15,
      directionalAccuracy: 56.8,
      totalReturn: 33.7,
    },
    "D+10": {
      assetId: "platinum",
      direction: "bullish",
      confidence: 81,
      horizon: "D+10",
      modelsAgreeing: 8245,
      modelsTotal: 10179,
      sharpeRatio: 2.42,
      directionalAccuracy: 58.9,
      totalReturn: 38.2,
    },
  },
};

// Overall performance metrics
export const MOCK_METRICS: PerformanceMetrics = {
  sharpeRatio: 2.34,
  directionalAccuracy: 58.4,
  totalReturn: 47.2,
  maxDrawdown: -12.8,
  winRate: 61.3,
  modelCount: 10179,
};

// Top 5 assets for the main dashboard (ordered by confidence)
export const TOP_ASSETS: AssetId[] = [
  "natural-gas",
  "gold",
  "crude-oil",
  "bitcoin",
  "silver",
];

// Helper function to get signal data
export function getSignal(assetId: AssetId, horizon: Horizon = "D+1"): SignalData | undefined {
  return MOCK_SIGNALS[assetId]?.[horizon];
}

// Helper function to get asset data
export function getAsset(assetId: AssetId): AssetData | undefined {
  return MOCK_ASSETS[assetId];
}

// Helper function to format model agreement
export function formatModelAgreement(agreeing: number, total: number): string {
  return `${agreeing.toLocaleString()} / ${total.toLocaleString()} models agree`;
}

// Helper function to format price
export function formatPrice(price: number, symbol: string): string {
  if (symbol === "BTC") {
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  if (price >= 100) {
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  return `$${price.toFixed(2)}`;
}
