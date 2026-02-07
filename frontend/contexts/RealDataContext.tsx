"use client";

/**
 * Real Data Context
 * 
 * Provides real market data from QDL v2 API to all dashboard components.
 * Falls back to mock data when API is unavailable.
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import type { AssetId } from "@/types";
import * as apiV2 from "@/lib/api-v2";
import { MOCK_ASSETS, MOCK_SIGNALS, type AssetData, type SignalData } from "@/lib/mock-data";

// ============================================================================
// Types
// ============================================================================

interface RealDataContextType {
  // Data
  assets: Record<string, apiV2.AssetDataV2>;
  signals: Record<string, Record<string, apiV2.SignalDataV2>>;
  
  // Loading states
  loading: boolean;
  assetsLoading: boolean;
  signalsLoading: Record<string, boolean>;
  
  // Errors
  error: Error | null;
  
  // Data source
  dataSource: "real" | "mock" | "mixed";
  
  // Actions
  refreshAssets: () => Promise<void>;
  refreshSignals: (assetId: string) => Promise<void>;
  refreshAll: () => Promise<void>;
  
  // Helpers
  getAsset: (assetId: AssetId) => AssetData | null;
  getSignal: (assetId: AssetId, horizon: string) => SignalData | null;
}

const RealDataContext = createContext<RealDataContextType | null>(null);

// ============================================================================
// Provider
// ============================================================================

export function RealDataProvider({ children }: { children: React.ReactNode }) {
  const [assets, setAssets] = useState<Record<string, apiV2.AssetDataV2>>({});
  const [signals, setSignals] = useState<Record<string, Record<string, apiV2.SignalDataV2>>>({});
  const [loading, setLoading] = useState(true);
  const [assetsLoading, setAssetsLoading] = useState(true);
  const [signalsLoading, setSignalsLoading] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<Error | null>(null);
  const [dataSource, setDataSource] = useState<"real" | "mock" | "mixed">("mock");

  // Fetch all assets
  const refreshAssets = useCallback(async () => {
    setAssetsLoading(true);
    try {
      const data = await apiV2.getAllAssets();
      const assetMap: Record<string, apiV2.AssetDataV2> = {};
      data.forEach((a) => {
        assetMap[a.id] = a;
      });
      setAssets(assetMap);
      setDataSource((prev) => (prev === "mock" ? "real" : prev));
      setError(null);
    } catch (e) {
      console.warn("Failed to fetch assets from v2 API, using mock:", e);
      setDataSource("mock");
    } finally {
      setAssetsLoading(false);
    }
  }, []);

  // Fetch signals for a specific asset
  const refreshSignals = useCallback(async (assetId: string) => {
    setSignalsLoading((prev) => ({ ...prev, [assetId]: true }));
    try {
      const data = await apiV2.getSignals(assetId);
      setSignals((prev) => ({ ...prev, [assetId]: data }));
    } catch (e) {
      console.warn(`Failed to fetch signals for ${assetId}:`, e);
    } finally {
      setSignalsLoading((prev) => ({ ...prev, [assetId]: false }));
    }
  }, []);

  // Refresh everything
  const refreshAll = useCallback(async () => {
    setLoading(true);
    await refreshAssets();
    // Fetch signals for all assets
    const assetIds = Object.keys(assets);
    await Promise.all(assetIds.map((id) => refreshSignals(id)));
    setLoading(false);
  }, [refreshAssets, refreshSignals, assets]);

  // Initial load
  useEffect(() => {
    refreshAssets().then(() => setLoading(false));
  }, [refreshAssets]);

  // Helper: Get asset data (real or mock fallback)
  const getAsset = useCallback(
    (assetId: AssetId): AssetData | null => {
      const real = assets[assetId];
      if (real) {
        return apiV2.toAssetData(real);
      }
      return MOCK_ASSETS[assetId] ?? null;
    },
    [assets]
  );

  // Helper: Get signal data (real or mock fallback)
  const getSignal = useCallback(
    (assetId: AssetId, horizon: string): SignalData | null => {
      const realSignals = signals[assetId];
      if (realSignals && realSignals[horizon]) {
        return apiV2.toSignalData(realSignals[horizon]);
      }
      // Fallback to mock
      const mockSignal = MOCK_SIGNALS[assetId]?.[horizon as keyof typeof MOCK_SIGNALS[typeof assetId]];
      return mockSignal ?? null;
    },
    [signals]
  );

  const value: RealDataContextType = {
    assets,
    signals,
    loading,
    assetsLoading,
    signalsLoading,
    error,
    dataSource,
    refreshAssets,
    refreshSignals,
    refreshAll,
    getAsset,
    getSignal,
  };

  return (
    <RealDataContext.Provider value={value}>
      {children}
    </RealDataContext.Provider>
  );
}

// ============================================================================
// Hook
// ============================================================================

export function useRealData() {
  const context = useContext(RealDataContext);
  if (!context) {
    throw new Error("useRealData must be used within a RealDataProvider");
  }
  return context;
}

// ============================================================================
// Convenience Hooks
// ============================================================================

/**
 * Get a single asset with automatic real/mock fallback
 */
export function useAssetData(assetId: AssetId) {
  const { getAsset, assets, assetsLoading, refreshSignals, signals, signalsLoading } = useRealData();
  
  // Fetch signals for this asset if not already loaded
  useEffect(() => {
    if (!signals[assetId] && !signalsLoading[assetId]) {
      refreshSignals(assetId);
    }
  }, [assetId, signals, signalsLoading, refreshSignals]);
  
  return {
    asset: getAsset(assetId),
    loading: assetsLoading,
    isReal: !!assets[assetId],
  };
}

/**
 * Get all assets as an array
 */
export function useAllAssets() {
  const { assets, assetsLoading, dataSource } = useRealData();
  
  // Convert to array, falling back to mock for missing
  const assetList = Object.values(assets).map(apiV2.toAssetData);
  
  // Add any mock assets not in real data
  if (dataSource === "mock" || dataSource === "mixed") {
    Object.entries(MOCK_ASSETS).forEach(([id, mockAsset]) => {
      if (mockAsset && !assets[id]) {
        assetList.push(mockAsset);
      }
    });
  }
  
  return {
    assets: assetList,
    loading: assetsLoading,
    dataSource,
  };
}
