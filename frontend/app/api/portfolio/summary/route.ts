import { NextResponse } from "next/server";
import type { PortfolioSummary, EnsembleMethod, ApiResponse } from "@/lib/api-client";

// Generate portfolio summary based on method
function generatePortfolioSummary(method: EnsembleMethod): PortfolioSummary {
  // Base portfolio data
  const baseAssets = [
    { assetId: "natural-gas" as const, direction: "bullish" as const, confidence: 91 },
    { assetId: "gold" as const, direction: "bullish" as const, confidence: 84 },
    { assetId: "crude-oil" as const, direction: "bullish" as const, confidence: 78 },
    { assetId: "soybean" as const, direction: "bearish" as const, confidence: 76 },
    { assetId: "wheat" as const, direction: "bearish" as const, confidence: 73 },
    { assetId: "corn" as const, direction: "bullish" as const, confidence: 69 },
    { assetId: "platinum" as const, direction: "bullish" as const, confidence: 67 },
    { assetId: "bitcoin" as const, direction: "bearish" as const, confidence: 65 },
    { assetId: "silver" as const, direction: "bearish" as const, confidence: 62 },
    { assetId: "copper" as const, direction: "neutral" as const, confidence: 52 },
  ];

  // Method-specific adjustments
  const methodAdjustments: Record<EnsembleMethod, number> = {
    accuracy_weighted: 0,
    exponential_decay: -2,
    top_k_sharpe: 3,
    ridge_stacking: 1,
    inverse_variance: -1,
    pairwise_slope: 2,
  };

  const adjustment = methodAdjustments[method];

  // Calculate overall signal
  let bullishScore = 0;
  let bearishScore = 0;

  for (const asset of baseAssets) {
    const adjustedConfidence = Math.min(99, Math.max(40, asset.confidence + adjustment));
    if (asset.direction === "bullish") {
      bullishScore += adjustedConfidence;
    } else if (asset.direction === "bearish") {
      bearishScore += adjustedConfidence;
    }
  }

  const overallDirection = bullishScore > bearishScore ? "bullish" : bullishScore < bearishScore ? "bearish" : "neutral";
  const overallConfidence = Math.round(
    Math.abs(bullishScore - bearishScore) / baseAssets.length + 50 + adjustment
  );

  return {
    ensembleMethod: method,
    overallSignal: overallDirection,
    overallConfidence: Math.min(99, Math.max(40, overallConfidence)),
    totalModels: 10179,
    lastUpdated: new Date().toISOString(),
    assetBreakdown: baseAssets.map((asset) => ({
      ...asset,
      confidence: Math.min(99, Math.max(40, asset.confidence + adjustment)),
    })),
  };
}

export async function GET(request: Request) {
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

  const summary = generatePortfolioSummary(method);

  const response: ApiResponse<PortfolioSummary> = {
    data: summary,
    success: true,
    timestamp: new Date().toISOString(),
  };

  return NextResponse.json(response);
}
