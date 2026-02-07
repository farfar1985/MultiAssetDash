/**
 * React hooks for fetching real market data from QDL v2 API
 */

import { useState, useEffect, useCallback } from "react";
import type { AssetId } from "@/types";
import * as apiV2 from "@/lib/api-v2";

// ============================================================================
// useAssets - Fetch all assets with current prices
// ============================================================================

export function useAssets() {
  const [assets, setAssets] = useState<apiV2.AssetDataV2[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiV2.getAllAssets();
      setAssets(data);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { assets, loading, error, refetch };
}

// ============================================================================
// useAsset - Fetch single asset
// ============================================================================

export function useAsset(assetId: AssetId | string) {
  const [asset, setAsset] = useState<apiV2.AssetDataV2 | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    if (!assetId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiV2.getAsset(assetId);
      setAsset(data);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, [assetId]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { asset, loading, error, refetch };
}

// ============================================================================
// useSignals - Fetch trading signals for an asset
// ============================================================================

export function useSignals(assetId: AssetId | string) {
  const [signals, setSignals] = useState<Record<string, apiV2.SignalDataV2>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    if (!assetId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiV2.getSignals(assetId);
      setSignals(data);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, [assetId]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { signals, loading, error, refetch };
}

// ============================================================================
// useChartData - Fetch OHLCV data for charts
// ============================================================================

export function useChartData(assetId: AssetId | string, days: number = 365) {
  const [chartData, setChartData] = useState<apiV2.ChartDataPointV2[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    if (!assetId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiV2.getChartData(assetId, days);
      setChartData(data);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, [assetId, days]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { chartData, loading, error, refetch };
}

// ============================================================================
// useMarketSummary - Fetch overall market summary
// ============================================================================

export function useMarketSummary() {
  const [summary, setSummary] = useState<apiV2.MarketSummaryV2 | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiV2.getMarketSummary();
      setSummary(data);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { summary, loading, error, refetch };
}

// ============================================================================
// useRealTimePrice - Auto-refresh price data
// ============================================================================

export function useRealTimePrice(assetId: AssetId | string, refreshInterval: number = 60000) {
  const { asset, loading, error, refetch } = useAsset(assetId);

  useEffect(() => {
    const interval = setInterval(refetch, refreshInterval);
    return () => clearInterval(interval);
  }, [refetch, refreshInterval]);

  return { price: asset?.currentPrice ?? 0, asset, loading, error };
}
