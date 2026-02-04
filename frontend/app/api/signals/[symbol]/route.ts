import { NextResponse } from "next/server";
import type { SignalData, Horizon, ApiResponse } from "@/lib/api-client";
import type { AssetId } from "@/types";

type SignalRecord = Record<string, Record<Horizon, Omit<SignalData, "assetId" | "horizon" | "generatedAt">>>;

// Mock signal data by asset ID and horizon
const MOCK_SIGNALS: SignalRecord = {
  "crude-oil": {
    "D+1": {
      direction: "bullish",
      confidence: 78,
      modelsAgreeing: 8432,
      modelsTotal: 10179,
      sharpeRatio: 2.34,
      directionalAccuracy: 58.2,
      totalReturn: 45.8,
    },
    "D+5": {
      direction: "bullish",
      confidence: 72,
      modelsAgreeing: 7891,
      modelsTotal: 10179,
      sharpeRatio: 2.18,
      directionalAccuracy: 56.4,
      totalReturn: 42.1,
    },
    "D+10": {
      direction: "neutral",
      confidence: 54,
      modelsAgreeing: 5512,
      modelsTotal: 10179,
      sharpeRatio: 1.92,
      directionalAccuracy: 53.8,
      totalReturn: 38.2,
    },
  },
  bitcoin: {
    "D+1": {
      direction: "bearish",
      confidence: 65,
      modelsAgreeing: 6621,
      modelsTotal: 10179,
      sharpeRatio: 1.87,
      directionalAccuracy: 54.1,
      totalReturn: 67.3,
    },
    "D+5": {
      direction: "bullish",
      confidence: 71,
      modelsAgreeing: 7234,
      modelsTotal: 10179,
      sharpeRatio: 2.12,
      directionalAccuracy: 55.8,
      totalReturn: 72.4,
    },
    "D+10": {
      direction: "bullish",
      confidence: 82,
      modelsAgreeing: 8351,
      modelsTotal: 10179,
      sharpeRatio: 2.56,
      directionalAccuracy: 59.2,
      totalReturn: 89.1,
    },
  },
  gold: {
    "D+1": {
      direction: "bullish",
      confidence: 84,
      modelsAgreeing: 8556,
      modelsTotal: 10179,
      sharpeRatio: 2.78,
      directionalAccuracy: 61.4,
      totalReturn: 32.5,
    },
    "D+5": {
      direction: "bullish",
      confidence: 79,
      modelsAgreeing: 8043,
      modelsTotal: 10179,
      sharpeRatio: 2.45,
      directionalAccuracy: 58.9,
      totalReturn: 29.8,
    },
    "D+10": {
      direction: "bullish",
      confidence: 76,
      modelsAgreeing: 7738,
      modelsTotal: 10179,
      sharpeRatio: 2.31,
      directionalAccuracy: 57.2,
      totalReturn: 28.1,
    },
  },
  silver: {
    "D+1": {
      direction: "bearish",
      confidence: 62,
      modelsAgreeing: 6315,
      modelsTotal: 10179,
      sharpeRatio: 1.68,
      directionalAccuracy: 52.8,
      totalReturn: 24.3,
    },
    "D+5": {
      direction: "neutral",
      confidence: 51,
      modelsAgreeing: 5192,
      modelsTotal: 10179,
      sharpeRatio: 1.54,
      directionalAccuracy: 51.2,
      totalReturn: 21.8,
    },
    "D+10": {
      direction: "bullish",
      confidence: 68,
      modelsAgreeing: 6923,
      modelsTotal: 10179,
      sharpeRatio: 1.89,
      directionalAccuracy: 54.5,
      totalReturn: 26.7,
    },
  },
  "natural-gas": {
    "D+1": {
      direction: "bullish",
      confidence: 91,
      modelsAgreeing: 9263,
      modelsTotal: 10179,
      sharpeRatio: 3.12,
      directionalAccuracy: 64.8,
      totalReturn: 98.4,
    },
    "D+5": {
      direction: "bullish",
      confidence: 85,
      modelsAgreeing: 8652,
      modelsTotal: 10179,
      sharpeRatio: 2.89,
      directionalAccuracy: 62.1,
      totalReturn: 87.2,
    },
    "D+10": {
      direction: "neutral",
      confidence: 58,
      modelsAgreeing: 5904,
      modelsTotal: 10179,
      sharpeRatio: 2.14,
      directionalAccuracy: 55.3,
      totalReturn: 64.8,
    },
  },
  copper: {
    "D+1": {
      direction: "neutral",
      confidence: 52,
      modelsAgreeing: 5293,
      modelsTotal: 10179,
      sharpeRatio: 1.72,
      directionalAccuracy: 52.4,
      totalReturn: 28.9,
    },
    "D+5": {
      direction: "bullish",
      confidence: 67,
      modelsAgreeing: 6821,
      modelsTotal: 10179,
      sharpeRatio: 1.95,
      directionalAccuracy: 54.8,
      totalReturn: 33.4,
    },
    "D+10": {
      direction: "bullish",
      confidence: 74,
      modelsAgreeing: 7534,
      modelsTotal: 10179,
      sharpeRatio: 2.21,
      directionalAccuracy: 57.1,
      totalReturn: 38.7,
    },
  },
  wheat: {
    "D+1": {
      direction: "bearish",
      confidence: 73,
      modelsAgreeing: 7431,
      modelsTotal: 10179,
      sharpeRatio: 2.08,
      directionalAccuracy: 56.2,
      totalReturn: 31.2,
    },
    "D+5": {
      direction: "bearish",
      confidence: 68,
      modelsAgreeing: 6923,
      modelsTotal: 10179,
      sharpeRatio: 1.92,
      directionalAccuracy: 54.7,
      totalReturn: 28.4,
    },
    "D+10": {
      direction: "neutral",
      confidence: 55,
      modelsAgreeing: 5600,
      modelsTotal: 10179,
      sharpeRatio: 1.67,
      directionalAccuracy: 52.9,
      totalReturn: 24.1,
    },
  },
  corn: {
    "D+1": {
      direction: "bullish",
      confidence: 69,
      modelsAgreeing: 7025,
      modelsTotal: 10179,
      sharpeRatio: 1.98,
      directionalAccuracy: 55.3,
      totalReturn: 26.8,
    },
    "D+5": {
      direction: "neutral",
      confidence: 53,
      modelsAgreeing: 5395,
      modelsTotal: 10179,
      sharpeRatio: 1.62,
      directionalAccuracy: 52.1,
      totalReturn: 22.3,
    },
    "D+10": {
      direction: "bullish",
      confidence: 71,
      modelsAgreeing: 7227,
      modelsTotal: 10179,
      sharpeRatio: 2.05,
      directionalAccuracy: 55.8,
      totalReturn: 29.4,
    },
  },
  soybean: {
    "D+1": {
      direction: "bearish",
      confidence: 76,
      modelsAgreeing: 7738,
      modelsTotal: 10179,
      sharpeRatio: 2.24,
      directionalAccuracy: 57.4,
      totalReturn: 34.8,
    },
    "D+5": {
      direction: "bearish",
      confidence: 72,
      modelsAgreeing: 7329,
      modelsTotal: 10179,
      sharpeRatio: 2.11,
      directionalAccuracy: 56.1,
      totalReturn: 32.1,
    },
    "D+10": {
      direction: "neutral",
      confidence: 56,
      modelsAgreeing: 5700,
      modelsTotal: 10179,
      sharpeRatio: 1.78,
      directionalAccuracy: 53.4,
      totalReturn: 27.5,
    },
  },
  platinum: {
    "D+1": {
      direction: "bullish",
      confidence: 67,
      modelsAgreeing: 6821,
      modelsTotal: 10179,
      sharpeRatio: 1.89,
      directionalAccuracy: 54.6,
      totalReturn: 29.3,
    },
    "D+5": {
      direction: "bullish",
      confidence: 74,
      modelsAgreeing: 7534,
      modelsTotal: 10179,
      sharpeRatio: 2.15,
      directionalAccuracy: 56.8,
      totalReturn: 33.7,
    },
    "D+10": {
      direction: "bullish",
      confidence: 81,
      modelsAgreeing: 8245,
      modelsTotal: 10179,
      sharpeRatio: 2.42,
      directionalAccuracy: 58.9,
      totalReturn: 38.2,
    },
  },
};

// Map symbols to asset IDs
const SYMBOL_TO_ASSET: Record<string, string> = {
  CL: "crude-oil",
  BTC: "bitcoin",
  GC: "gold",
  SI: "silver",
  NG: "natural-gas",
  HG: "copper",
  ZW: "wheat",
  ZC: "corn",
  ZS: "soybean",
  PL: "platinum",
};

export async function GET(
  request: Request,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;
  const { searchParams } = new URL(request.url);
  const horizon = (searchParams.get("horizon") || "D+1") as Horizon;

  // Resolve symbol to asset ID
  const assetId = SYMBOL_TO_ASSET[symbol.toUpperCase()] || symbol.toLowerCase();
  const signals = MOCK_SIGNALS[assetId];

  if (!signals) {
    const response: ApiResponse<null> = {
      data: null,
      success: false,
      error: `Signals not found for asset: ${symbol}`,
      timestamp: new Date().toISOString(),
    };
    return NextResponse.json(response, { status: 404 });
  }

  const signalData = signals[horizon];
  if (!signalData) {
    const response: ApiResponse<null> = {
      data: null,
      success: false,
      error: `Signal not found for horizon: ${horizon}`,
      timestamp: new Date().toISOString(),
    };
    return NextResponse.json(response, { status: 404 });
  }

  const fullSignal: SignalData = {
    assetId: assetId as AssetId,
    horizon,
    generatedAt: new Date().toISOString(),
    ...signalData,
  };

  const response: ApiResponse<SignalData> = {
    data: fullSignal,
    success: true,
    timestamp: new Date().toISOString(),
  };

  return NextResponse.json(response);
}
