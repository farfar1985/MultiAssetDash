import { NextResponse } from "next/server";
import type { AssetId } from "@/types";
import type { AssetData, ApiResponse } from "@/lib/api-client";

// Mock asset data with full details
const MOCK_ASSETS: Record<AssetId, AssetData> = {
  "crude-oil": {
    id: "crude-oil",
    name: "Crude Oil",
    symbol: "CL",
    category: "energy",
    currentPrice: 78.42,
    change24h: 1.23,
    changePercent24h: 1.59,
  },
  bitcoin: {
    id: "bitcoin",
    name: "Bitcoin",
    symbol: "BTC",
    category: "crypto",
    currentPrice: 97432.15,
    change24h: -1254.32,
    changePercent24h: -1.27,
  },
  gold: {
    id: "gold",
    name: "Gold",
    symbol: "GC",
    category: "metals",
    currentPrice: 2891.5,
    change24h: 12.8,
    changePercent24h: 0.44,
  },
  silver: {
    id: "silver",
    name: "Silver",
    symbol: "SI",
    category: "metals",
    currentPrice: 32.15,
    change24h: -0.28,
    changePercent24h: -0.86,
  },
  "natural-gas": {
    id: "natural-gas",
    name: "Natural Gas",
    symbol: "NG",
    category: "energy",
    currentPrice: 3.42,
    change24h: 0.15,
    changePercent24h: 4.59,
  },
  copper: {
    id: "copper",
    name: "Copper",
    symbol: "HG",
    category: "metals",
    currentPrice: 4.28,
    change24h: 0.03,
    changePercent24h: 0.71,
  },
  wheat: {
    id: "wheat",
    name: "Wheat",
    symbol: "ZW",
    category: "agriculture",
    currentPrice: 542.25,
    change24h: -8.5,
    changePercent24h: -1.54,
  },
  corn: {
    id: "corn",
    name: "Corn",
    symbol: "ZC",
    category: "agriculture",
    currentPrice: 445.75,
    change24h: 3.25,
    changePercent24h: 0.73,
  },
  soybean: {
    id: "soybean",
    name: "Soybean",
    symbol: "ZS",
    category: "agriculture",
    currentPrice: 1028.5,
    change24h: -12.75,
    changePercent24h: -1.22,
  },
  platinum: {
    id: "platinum",
    name: "Platinum",
    symbol: "PL",
    category: "metals",
    currentPrice: 1042.3,
    change24h: 8.2,
    changePercent24h: 0.79,
  },
};

export async function GET() {
  const assets = Object.values(MOCK_ASSETS);

  const response: ApiResponse<AssetData[]> = {
    data: assets,
    success: true,
    timestamp: new Date().toISOString(),
  };

  return NextResponse.json(response);
}
