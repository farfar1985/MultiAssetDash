import { NextResponse } from "next/server";
import type { HistoricalDataPoint, ApiResponse } from "@/lib/api-client";
import type { SignalDirection } from "@/types";

// Map symbols to asset IDs with base prices
const ASSET_BASE_PRICES: Record<string, number> = {
  "crude-oil": 78.42,
  CL: 78.42,
  bitcoin: 97432.15,
  BTC: 97432.15,
  gold: 2891.5,
  GC: 2891.5,
  silver: 32.15,
  SI: 32.15,
  "natural-gas": 3.42,
  NG: 3.42,
  copper: 4.28,
  HG: 4.28,
  wheat: 542.25,
  ZW: 542.25,
  corn: 445.75,
  ZC: 445.75,
  soybean: 1028.5,
  ZS: 1028.5,
  platinum: 1042.3,
  PL: 1042.3,
};

// Generate realistic OHLCV data
function generateHistoricalData(
  basePrice: number,
  startDate: Date,
  endDate: Date
): HistoricalDataPoint[] {
  const data: HistoricalDataPoint[] = [];
  const dayMs = 24 * 60 * 60 * 1000;
  let currentPrice = basePrice * 0.92; // Start 8% lower than current
  const volatility = basePrice < 10 ? 0.04 : basePrice < 100 ? 0.025 : 0.015;

  for (
    let date = new Date(startDate);
    date <= endDate;
    date = new Date(date.getTime() + dayMs)
  ) {
    // Skip weekends for commodities
    if (date.getDay() === 0 || date.getDay() === 6) continue;

    const dailyReturn = (Math.random() - 0.48) * volatility * 2;
    const open = currentPrice;
    currentPrice = currentPrice * (1 + dailyReturn);

    const highLowRange = currentPrice * volatility * 0.8;
    const high = Math.max(open, currentPrice) + Math.random() * highLowRange;
    const low = Math.min(open, currentPrice) - Math.random() * highLowRange;
    const close = currentPrice;

    // Volume with some variance
    const baseVolume = basePrice < 10 ? 50000 : basePrice < 100 ? 25000 : 15000;
    const volume = Math.round(baseVolume * (0.7 + Math.random() * 0.6));

    // Signal based on price movement
    const signal: SignalDirection =
      dailyReturn > 0.01
        ? "bullish"
        : dailyReturn < -0.01
        ? "bearish"
        : "neutral";

    const confidence = Math.round(50 + Math.abs(dailyReturn) * 1000);

    data.push({
      date: date.toISOString().split("T")[0],
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume,
      signal,
      confidence: Math.min(95, confidence),
    });
  }

  return data;
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;
  const { searchParams } = new URL(request.url);

  const startParam = searchParams.get("start");
  const endParam = searchParams.get("end");

  // Default to last 90 days
  const endDate = endParam ? new Date(endParam) : new Date();
  const startDate = startParam
    ? new Date(startParam)
    : new Date(endDate.getTime() - 90 * 24 * 60 * 60 * 1000);

  // Resolve symbol to base price
  const basePrice =
    ASSET_BASE_PRICES[symbol.toUpperCase()] ||
    ASSET_BASE_PRICES[symbol.toLowerCase()];

  if (!basePrice) {
    const response: ApiResponse<null> = {
      data: null,
      success: false,
      error: `Asset not found: ${symbol}`,
      timestamp: new Date().toISOString(),
    };
    return NextResponse.json(response, { status: 404 });
  }

  const data = generateHistoricalData(basePrice, startDate, endDate);

  const response: ApiResponse<HistoricalDataPoint[]> = {
    data,
    success: true,
    timestamp: new Date().toISOString(),
  };

  return NextResponse.json(response);
}
