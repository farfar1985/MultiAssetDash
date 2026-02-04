import { NextResponse } from "next/server";
import type { EnsembleResult, EnsembleMethod, ApiResponse } from "@/lib/api-client";
import type { SignalDirection, AssetId } from "@/types";

// Base signal data by asset
const BASE_SIGNALS: Record<string, { direction: SignalDirection; baseConfidence: number }> = {
  "crude-oil": { direction: "bullish", baseConfidence: 78 },
  bitcoin: { direction: "bearish", baseConfidence: 65 },
  gold: { direction: "bullish", baseConfidence: 84 },
  silver: { direction: "bearish", baseConfidence: 62 },
  "natural-gas": { direction: "bullish", baseConfidence: 91 },
  copper: { direction: "neutral", baseConfidence: 52 },
  wheat: { direction: "bearish", baseConfidence: 73 },
  corn: { direction: "bullish", baseConfidence: 69 },
  soybean: { direction: "bearish", baseConfidence: 76 },
  platinum: { direction: "bullish", baseConfidence: 67 },
};

// Method-specific adjustments
const METHOD_ADJUSTMENTS: Record<EnsembleMethod, { confidenceMultiplier: number; sharpeBoost: number }> = {
  accuracy_weighted: { confidenceMultiplier: 1.0, sharpeBoost: 0 },
  exponential_decay: { confidenceMultiplier: 0.95, sharpeBoost: 0.1 },
  top_k_sharpe: { confidenceMultiplier: 1.05, sharpeBoost: 0.25 },
  ridge_stacking: { confidenceMultiplier: 1.02, sharpeBoost: 0.15 },
  inverse_variance: { confidenceMultiplier: 0.98, sharpeBoost: 0.08 },
  pairwise_slope: { confidenceMultiplier: 1.03, sharpeBoost: 0.18 },
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

function generateEnsembleResult(assetId: string, method: EnsembleMethod): EnsembleResult {
  const baseSignal = BASE_SIGNALS[assetId];
  const adjustment = METHOD_ADJUSTMENTS[method];

  if (!baseSignal) {
    throw new Error(`Unknown asset: ${assetId}`);
  }

  const confidence = Math.min(
    99,
    Math.round(baseSignal.baseConfidence * adjustment.confidenceMultiplier)
  );
  const modelsTotal = 10179;
  const modelsAgreeing = Math.round(modelsTotal * (confidence / 100) * 0.95);

  return {
    method,
    signal: {
      assetId: assetId as AssetId,
      direction: baseSignal.direction,
      confidence,
      horizon: "D+1",
      modelsAgreeing,
      modelsTotal,
      sharpeRatio: 2.34 + adjustment.sharpeBoost,
      directionalAccuracy: 55 + confidence * 0.08,
      totalReturn: 30 + confidence * 0.3,
      generatedAt: new Date().toISOString(),
    },
    modelWeights: {
      lstm_volatility: 0.18 + (method === "top_k_sharpe" ? 0.05 : 0),
      transformer_trend: 0.22 + (method === "accuracy_weighted" ? 0.03 : 0),
      gradient_boost_macro: 0.15 + (method === "pairwise_slope" ? 0.04 : 0),
      random_forest_seasonal: 0.12 + (method === "exponential_decay" ? 0.04 : 0),
      xgboost_technical: 0.16 + (method === "ridge_stacking" ? 0.02 : 0),
      neural_ensemble: 0.17 + (method === "inverse_variance" ? 0.03 : 0),
    },
    backtestMetrics: {
      sharpeRatio: 2.34 + adjustment.sharpeBoost,
      directionalAccuracy: 55 + confidence * 0.08,
      totalReturn: 30 + confidence * 0.3,
      maxDrawdown: -12 - (100 - confidence) * 0.1,
    },
  };
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;
  const { searchParams } = new URL(request.url);
  const method = (searchParams.get("method") || "accuracy_weighted") as EnsembleMethod;

  // Validate method
  const validMethods: EnsembleMethod[] = [
    "accuracy_weighted",
    "exponential_decay",
    "top_k_sharpe",
    "ridge_stacking",
    "inverse_variance",
    "pairwise_slope",
  ];

  if (!validMethods.includes(method)) {
    const response: ApiResponse<null> = {
      data: null,
      success: false,
      error: `Invalid ensemble method: ${method}. Valid methods are: ${validMethods.join(", ")}`,
      timestamp: new Date().toISOString(),
    };
    return NextResponse.json(response, { status: 400 });
  }

  // Resolve symbol to asset ID
  const assetId = SYMBOL_TO_ASSET[symbol.toUpperCase()] || symbol.toLowerCase();

  if (!BASE_SIGNALS[assetId]) {
    const response: ApiResponse<null> = {
      data: null,
      success: false,
      error: `Asset not found: ${symbol}`,
      timestamp: new Date().toISOString(),
    };
    return NextResponse.json(response, { status: 404 });
  }

  const result = generateEnsembleResult(assetId, method);

  const response: ApiResponse<EnsembleResult> = {
    data: result,
    success: true,
    timestamp: new Date().toISOString(),
  };

  return NextResponse.json(response);
}
