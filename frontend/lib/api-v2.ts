/**
 * QDT Nexus API v2 - Real Data from QDL
 * 
 * This module provides typed API functions for fetching real market data
 * from the QDT Data Lake instead of mock/simulated data.
 */

import type { AssetId } from "@/types";

// ============================================================================
// Configuration
// ============================================================================

// Use same base URL as v1 - v2 routes are registered on same server
const API_V2_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

// ============================================================================
// Types
// ============================================================================

export interface AssetDataV2 {
  id: string;
  name: string;
  symbol: string;
  currentPrice: number;
  change24h: number;
  changePercent24h: number;
  high24h: number;
  low24h: number;
  volume24h: number;
  lastUpdated: string;
}

export interface SignalDataV2 {
  assetId: string;
  direction: "bullish" | "bearish" | "neutral";
  confidence: number;
  horizon: string;
  modelsAgreeing: number;
  modelsTotal: number;
  sharpeRatio: number;
  directionalAccuracy: number;
  totalReturn: number;
}

export interface ChartDataPointV2 {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketSummaryV2 {
  assetCount: number;
  topSignals: Array<{
    assetId: string;
    assetName: string;
    price: number;
    change24h: number;
    direction: string;
    confidence: number;
    sharpe: number;
  }>;
  allSignals: Array<{
    assetId: string;
    assetName: string;
    price: number;
    change24h: number;
    direction: string;
    confidence: number;
    sharpe: number;
  }>;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Check API health
 */
export async function checkHealth(): Promise<{ status: string; version: string; dataSource: string }> {
  const res = await fetch(`${API_V2_URL}/api/v2/health`);
  const data = await res.json();
  return {
    status: data.status,
    version: data.version,
    dataSource: data.data_source,
  };
}

/**
 * Get all assets with current prices
 */
export async function getAllAssets(): Promise<AssetDataV2[]> {
  const res = await fetch(`${API_V2_URL}/api/v2/assets`);
  const data = await res.json();
  
  if (!data.success) {
    throw new Error(data.error || "Failed to fetch assets");
  }
  
  return data.assets;
}

/**
 * Get single asset details
 */
export async function getAsset(assetId: AssetId | string): Promise<AssetDataV2> {
  const res = await fetch(`${API_V2_URL}/api/v2/assets/${assetId}`);
  const data = await res.json();
  
  if (!data.success) {
    throw new Error(data.error || `Failed to fetch asset: ${assetId}`);
  }
  
  return data.asset;
}

/**
 * Get trading signals for an asset
 */
export async function getSignals(assetId: AssetId | string): Promise<Record<string, SignalDataV2>> {
  const res = await fetch(`${API_V2_URL}/api/v2/signals/${assetId}`);
  const data = await res.json();
  
  if (!data.success) {
    throw new Error(data.error || `Failed to fetch signals: ${assetId}`);
  }
  
  return data.signals;
}

/**
 * Get OHLCV chart data
 */
export async function getChartData(
  assetId: AssetId | string,
  days: number = 365
): Promise<ChartDataPointV2[]> {
  const res = await fetch(`${API_V2_URL}/api/v2/chart/${assetId}?days=${days}`);
  const data = await res.json();
  
  if (!data.success) {
    throw new Error(data.error || `Failed to fetch chart: ${assetId}`);
  }
  
  return data.data;
}

/**
 * Get market summary with all assets and top signals
 */
export async function getMarketSummary(): Promise<MarketSummaryV2> {
  const res = await fetch(`${API_V2_URL}/api/v2/summary`);
  const data = await res.json();
  
  if (!data.success) {
    throw new Error(data.error || "Failed to fetch market summary");
  }
  
  return {
    assetCount: data.assetCount,
    topSignals: data.topSignals,
    allSignals: data.allSignals,
  };
}

/**
 * Refresh asset data from QDL
 */
export async function refreshAsset(assetId: AssetId | string, days: number = 30): Promise<boolean> {
  const res = await fetch(`${API_V2_URL}/api/v2/refresh/${assetId}?days=${days}`, {
    method: "POST",
  });
  const data = await res.json();
  return data.success;
}

// ============================================================================
// Conversion helpers (v2 types → existing types)
// ============================================================================

import { MOCK_ASSETS, type AssetData, type SignalData } from "./mock-data";

/**
 * Convert v2 asset to existing AssetData format
 */
export function toAssetData(v2: AssetDataV2): AssetData {
  return {
    id: v2.id as AssetId,
    name: v2.name,
    symbol: v2.symbol,
    currentPrice: v2.currentPrice,
    change24h: v2.change24h,
    changePercent24h: v2.changePercent24h,
  };
}

/**
 * Convert v2 signal to existing SignalData format
 */
export function toSignalData(v2: SignalDataV2): SignalData {
  return {
    assetId: v2.assetId as AssetId,
    direction: v2.direction,
    confidence: v2.confidence,
    horizon: v2.horizon as "D+1" | "D+5" | "D+10",
    modelsAgreeing: v2.modelsAgreeing,
    modelsTotal: v2.modelsTotal,
    sharpeRatio: v2.sharpeRatio,
    directionalAccuracy: v2.directionalAccuracy,
    totalReturn: v2.totalReturn,
  };
}

// ============================================================================
// Combined data hooks (fetch real, fallback to mock)
// ============================================================================

/**
 * Get asset data with fallback to mock
 */
export async function getAssetWithFallback(assetId: AssetId): Promise<AssetData | null> {
  try {
    const v2 = await getAsset(assetId);
    return toAssetData(v2);
  } catch (error) {
    console.warn(`V2 API failed for ${assetId}, using mock:`, error);
    return MOCK_ASSETS[assetId] ?? null;
  }
}

/**
 * Get all assets with fallback to mock
 */
export async function getAllAssetsWithFallback(): Promise<AssetData[]> {
  try {
    const v2Assets = await getAllAssets();
    return v2Assets.map(toAssetData);
  } catch (error) {
    console.warn("V2 API failed, using mock assets:", error);
    return Object.values(MOCK_ASSETS).filter((a): a is AssetData => a !== undefined);
  }
}
