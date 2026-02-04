import { NextResponse } from "next/server";
import type { PerformanceMetrics, ApiResponse } from "@/lib/api-client";

// Mock metrics data by asset
const MOCK_METRICS: Record<string, PerformanceMetrics> = {
  "crude-oil": {
    sharpeRatio: 2.34,
    directionalAccuracy: 58.2,
    totalReturn: 45.8,
    maxDrawdown: -12.4,
    winRate: 61.5,
    modelCount: 10179,
    lastUpdated: new Date().toISOString(),
  },
  bitcoin: {
    sharpeRatio: 1.87,
    directionalAccuracy: 54.1,
    totalReturn: 67.3,
    maxDrawdown: -24.8,
    winRate: 56.2,
    modelCount: 10179,
    lastUpdated: new Date().toISOString(),
  },
  gold: {
    sharpeRatio: 2.78,
    directionalAccuracy: 61.4,
    totalReturn: 32.5,
    maxDrawdown: -8.2,
    winRate: 64.3,
    modelCount: 10179,
    lastUpdated: new Date().toISOString(),
  },
  silver: {
    sharpeRatio: 1.68,
    directionalAccuracy: 52.8,
    totalReturn: 24.3,
    maxDrawdown: -15.6,
    winRate: 54.8,
    modelCount: 10179,
    lastUpdated: new Date().toISOString(),
  },
  "natural-gas": {
    sharpeRatio: 3.12,
    directionalAccuracy: 64.8,
    totalReturn: 98.4,
    maxDrawdown: -18.3,
    winRate: 67.2,
    modelCount: 10179,
    lastUpdated: new Date().toISOString(),
  },
  copper: {
    sharpeRatio: 1.72,
    directionalAccuracy: 52.4,
    totalReturn: 28.9,
    maxDrawdown: -11.2,
    winRate: 55.1,
    modelCount: 10179,
    lastUpdated: new Date().toISOString(),
  },
  wheat: {
    sharpeRatio: 2.08,
    directionalAccuracy: 56.2,
    totalReturn: 31.2,
    maxDrawdown: -13.5,
    winRate: 58.9,
    modelCount: 10179,
    lastUpdated: new Date().toISOString(),
  },
  corn: {
    sharpeRatio: 1.98,
    directionalAccuracy: 55.3,
    totalReturn: 26.8,
    maxDrawdown: -10.8,
    winRate: 57.4,
    modelCount: 10179,
    lastUpdated: new Date().toISOString(),
  },
  soybean: {
    sharpeRatio: 2.24,
    directionalAccuracy: 57.4,
    totalReturn: 34.8,
    maxDrawdown: -14.2,
    winRate: 59.6,
    modelCount: 10179,
    lastUpdated: new Date().toISOString(),
  },
  platinum: {
    sharpeRatio: 1.89,
    directionalAccuracy: 54.6,
    totalReturn: 29.3,
    maxDrawdown: -12.1,
    winRate: 56.8,
    modelCount: 10179,
    lastUpdated: new Date().toISOString(),
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
  const metrics = MOCK_METRICS[assetId];

  if (!metrics) {
    const response: ApiResponse<null> = {
      data: null,
      success: false,
      error: `Metrics not found for asset: ${symbol}`,
      timestamp: new Date().toISOString(),
    };
    return NextResponse.json(response, { status: 404 });
  }

  const response: ApiResponse<PerformanceMetrics> = {
    data: {
      ...metrics,
      lastUpdated: new Date().toISOString(),
    },
    success: true,
    timestamp: new Date().toISOString(),
  };

  return NextResponse.json(response);
}
