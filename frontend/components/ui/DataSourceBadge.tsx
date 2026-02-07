"use client";

/**
 * Data Source Badge
 * 
 * Shows whether the dashboard is using real QDL data or mock data.
 */

import { useRealData } from "@/contexts/RealDataContext";

export function DataSourceBadge() {
  const { dataSource, loading, assetsLoading } = useRealData();

  if (loading || assetsLoading) {
    return (
      <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium bg-neutral-800 text-neutral-400">
        <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
        Loading...
      </div>
    );
  }

  if (dataSource === "real") {
    return (
      <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium bg-green-900/50 text-green-400 border border-green-800">
        <span className="w-2 h-2 rounded-full bg-green-500" />
        Live Data (QDL)
      </div>
    );
  }

  if (dataSource === "mixed") {
    return (
      <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium bg-yellow-900/50 text-yellow-400 border border-yellow-800">
        <span className="w-2 h-2 rounded-full bg-yellow-500" />
        Mixed (Real + Mock)
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium bg-neutral-800 text-neutral-400 border border-neutral-700">
      <span className="w-2 h-2 rounded-full bg-neutral-500" />
      Mock Data
    </div>
  );
}
