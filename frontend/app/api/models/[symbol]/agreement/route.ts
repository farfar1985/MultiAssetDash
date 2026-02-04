import { NextResponse } from "next/server";
import type { ModelAgreement, ApiResponse } from "@/lib/api-client";

// Mock model agreement data by asset
const MOCK_MODEL_AGREEMENT: Record<string, ModelAgreement> = {
  "crude-oil": {
    bullishCount: 6245,
    bearishCount: 2512,
    neutralCount: 1422,
    totalModels: 10179,
    overallDirection: "bullish",
    topModels: [
      { name: "LSTM Volatility", direction: "bullish", confidence: 82, weight: 0.18 },
      { name: "Transformer Trend", direction: "bullish", confidence: 79, weight: 0.22 },
      { name: "GradientBoost Macro", direction: "bullish", confidence: 75, weight: 0.15 },
      { name: "XGBoost Technical", direction: "neutral", confidence: 54, weight: 0.16 },
      { name: "Neural Ensemble", direction: "bullish", confidence: 81, weight: 0.17 },
    ],
  },
  bitcoin: {
    bullishCount: 3842,
    bearishCount: 4921,
    neutralCount: 1416,
    totalModels: 10179,
    overallDirection: "bearish",
    topModels: [
      { name: "LSTM Volatility", direction: "bearish", confidence: 68, weight: 0.18 },
      { name: "Transformer Trend", direction: "bearish", confidence: 71, weight: 0.22 },
      { name: "GradientBoost Macro", direction: "neutral", confidence: 52, weight: 0.15 },
      { name: "XGBoost Technical", direction: "bearish", confidence: 65, weight: 0.16 },
      { name: "Neural Ensemble", direction: "bearish", confidence: 63, weight: 0.17 },
    ],
  },
  gold: {
    bullishCount: 7124,
    bearishCount: 1854,
    neutralCount: 1201,
    totalModels: 10179,
    overallDirection: "bullish",
    topModels: [
      { name: "LSTM Volatility", direction: "bullish", confidence: 86, weight: 0.18 },
      { name: "Transformer Trend", direction: "bullish", confidence: 88, weight: 0.22 },
      { name: "GradientBoost Macro", direction: "bullish", confidence: 82, weight: 0.15 },
      { name: "XGBoost Technical", direction: "bullish", confidence: 79, weight: 0.16 },
      { name: "Neural Ensemble", direction: "bullish", confidence: 84, weight: 0.17 },
    ],
  },
  silver: {
    bullishCount: 3124,
    bearishCount: 4856,
    neutralCount: 2199,
    totalModels: 10179,
    overallDirection: "bearish",
    topModels: [
      { name: "LSTM Volatility", direction: "bearish", confidence: 64, weight: 0.18 },
      { name: "Transformer Trend", direction: "bearish", confidence: 61, weight: 0.22 },
      { name: "GradientBoost Macro", direction: "neutral", confidence: 51, weight: 0.15 },
      { name: "XGBoost Technical", direction: "bearish", confidence: 58, weight: 0.16 },
      { name: "Neural Ensemble", direction: "bearish", confidence: 62, weight: 0.17 },
    ],
  },
  "natural-gas": {
    bullishCount: 8542,
    bearishCount: 924,
    neutralCount: 713,
    totalModels: 10179,
    overallDirection: "bullish",
    topModels: [
      { name: "LSTM Volatility", direction: "bullish", confidence: 94, weight: 0.18 },
      { name: "Transformer Trend", direction: "bullish", confidence: 92, weight: 0.22 },
      { name: "GradientBoost Macro", direction: "bullish", confidence: 89, weight: 0.15 },
      { name: "XGBoost Technical", direction: "bullish", confidence: 87, weight: 0.16 },
      { name: "Neural Ensemble", direction: "bullish", confidence: 91, weight: 0.17 },
    ],
  },
  copper: {
    bullishCount: 3654,
    bearishCount: 3421,
    neutralCount: 3104,
    totalModels: 10179,
    overallDirection: "neutral",
    topModels: [
      { name: "LSTM Volatility", direction: "neutral", confidence: 53, weight: 0.18 },
      { name: "Transformer Trend", direction: "bullish", confidence: 56, weight: 0.22 },
      { name: "GradientBoost Macro", direction: "bearish", confidence: 51, weight: 0.15 },
      { name: "XGBoost Technical", direction: "neutral", confidence: 50, weight: 0.16 },
      { name: "Neural Ensemble", direction: "bullish", confidence: 54, weight: 0.17 },
    ],
  },
  wheat: {
    bullishCount: 2345,
    bearishCount: 5687,
    neutralCount: 2147,
    totalModels: 10179,
    overallDirection: "bearish",
    topModels: [
      { name: "LSTM Volatility", direction: "bearish", confidence: 75, weight: 0.18 },
      { name: "Transformer Trend", direction: "bearish", confidence: 72, weight: 0.22 },
      { name: "GradientBoost Macro", direction: "bearish", confidence: 68, weight: 0.15 },
      { name: "XGBoost Technical", direction: "bearish", confidence: 71, weight: 0.16 },
      { name: "Neural Ensemble", direction: "bearish", confidence: 74, weight: 0.17 },
    ],
  },
  corn: {
    bullishCount: 5124,
    bearishCount: 2876,
    neutralCount: 2179,
    totalModels: 10179,
    overallDirection: "bullish",
    topModels: [
      { name: "LSTM Volatility", direction: "bullish", confidence: 71, weight: 0.18 },
      { name: "Transformer Trend", direction: "bullish", confidence: 68, weight: 0.22 },
      { name: "GradientBoost Macro", direction: "neutral", confidence: 54, weight: 0.15 },
      { name: "XGBoost Technical", direction: "bullish", confidence: 66, weight: 0.16 },
      { name: "Neural Ensemble", direction: "bullish", confidence: 70, weight: 0.17 },
    ],
  },
  soybean: {
    bullishCount: 2456,
    bearishCount: 5834,
    neutralCount: 1889,
    totalModels: 10179,
    overallDirection: "bearish",
    topModels: [
      { name: "LSTM Volatility", direction: "bearish", confidence: 78, weight: 0.18 },
      { name: "Transformer Trend", direction: "bearish", confidence: 76, weight: 0.22 },
      { name: "GradientBoost Macro", direction: "bearish", confidence: 72, weight: 0.15 },
      { name: "XGBoost Technical", direction: "bearish", confidence: 74, weight: 0.16 },
      { name: "Neural Ensemble", direction: "bearish", confidence: 77, weight: 0.17 },
    ],
  },
  platinum: {
    bullishCount: 5234,
    bearishCount: 2845,
    neutralCount: 2100,
    totalModels: 10179,
    overallDirection: "bullish",
    topModels: [
      { name: "LSTM Volatility", direction: "bullish", confidence: 69, weight: 0.18 },
      { name: "Transformer Trend", direction: "bullish", confidence: 71, weight: 0.22 },
      { name: "GradientBoost Macro", direction: "bullish", confidence: 64, weight: 0.15 },
      { name: "XGBoost Technical", direction: "neutral", confidence: 55, weight: 0.16 },
      { name: "Neural Ensemble", direction: "bullish", confidence: 68, weight: 0.17 },
    ],
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

  // Resolve symbol to asset ID
  const assetId = SYMBOL_TO_ASSET[symbol.toUpperCase()] || symbol.toLowerCase();
  const agreement = MOCK_MODEL_AGREEMENT[assetId];

  if (!agreement) {
    const response: ApiResponse<null> = {
      data: null,
      success: false,
      error: `Model agreement not found for asset: ${symbol}`,
      timestamp: new Date().toISOString(),
    };
    return NextResponse.json(response, { status: 404 });
  }

  const response: ApiResponse<ModelAgreement> = {
    data: agreement,
    success: true,
    timestamp: new Date().toISOString(),
  };

  return NextResponse.json(response);
}
