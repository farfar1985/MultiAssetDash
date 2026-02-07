"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  Wind,
  AlertCircle,
  RefreshCw,
  BarChart3,
  Clock,
  Gauge,
  ChevronRight,
} from "lucide-react";
import type { MarketRegime, RegimeData } from "./RegimeIndicator";

// ============================================================================
// Types
// ============================================================================

export interface AssetRegimeData extends RegimeData {
  asset_id: number;
  name: string;
  display_name: string;
  category: string;
  method: string;
}

export interface MultiAssetRegimeResponse {
  timestamp: string;
  regimes: Record<string, AssetRegimeData>;
  by_category: Record<string, AssetRegimeData[]>;
  regime_distribution: Record<string, number>;
  total_assets: number;
  calibration_date: string;
}

export interface MultiAssetRegimeOverviewProps {
  /** View mode: grid shows cards, table shows rows */
  viewMode?: "grid" | "table";
  /** Filter by category */
  category?: string;
  /** Show only specific regimes */
  regimeFilter?: MarketRegime[];
  /** Compact display mode */
  compact?: boolean;
  /** Custom class name */
  className?: string;
  /** Callback when asset is clicked */
  onAssetClick?: (assetId: number, assetName: string) => void;
}

// ============================================================================
// API Hook
// ============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

async function fetchAllRegimes(): Promise<MultiAssetRegimeResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/hmm/regimes`);
  if (!response.ok) {
    throw new Error(`Failed to fetch regimes: ${response.status}`);
  }
  return response.json();
}

export function useAllRegimes() {
  return useQuery({
    queryKey: ["hmm", "regimes", "all"],
    queryFn: fetchAllRegimes,
    staleTime: 30000,
    refetchInterval: 60000,
  });
}

// ============================================================================
// Helper Functions
// ============================================================================

function getRegimeConfig(regime: MarketRegime) {
  const configs: Record<
    MarketRegime,
    {
      label: string;
      shortLabel: string;
      icon: typeof TrendingUp;
      color: string;
      bgColor: string;
      borderColor: string;
      dotColor: string;
    }
  > = {
    bull: {
      label: "Bull",
      shortLabel: "BULL",
      icon: TrendingUp,
      color: "text-green-400",
      bgColor: "bg-green-500/10",
      borderColor: "border-green-500/30",
      dotColor: "bg-green-500",
    },
    bear: {
      label: "Bear",
      shortLabel: "BEAR",
      icon: TrendingDown,
      color: "text-red-400",
      bgColor: "bg-red-500/10",
      borderColor: "border-red-500/30",
      dotColor: "bg-red-500",
    },
    sideways: {
      label: "Sideways",
      shortLabel: "SIDE",
      icon: Minus,
      color: "text-amber-400",
      bgColor: "bg-amber-500/10",
      borderColor: "border-amber-500/30",
      dotColor: "bg-amber-500",
    },
    "high-volatility": {
      label: "High Vol",
      shortLabel: "HVOL",
      icon: Zap,
      color: "text-orange-400",
      bgColor: "bg-orange-500/10",
      borderColor: "border-orange-500/30",
      dotColor: "bg-orange-500",
    },
    "low-volatility": {
      label: "Low Vol",
      shortLabel: "LVOL",
      icon: Wind,
      color: "text-blue-400",
      bgColor: "bg-blue-500/10",
      borderColor: "border-blue-500/30",
      dotColor: "bg-blue-500",
    },
  };
  return configs[regime] || configs.sideways;
}

function getCategoryIcon(category: string) {
  switch (category) {
    case "Commodities":
      return "barrel";
    case "Crypto":
      return "bitcoin";
    case "Indices":
      return "trending";
    default:
      return "chart";
  }
}

// ============================================================================
// Sub-Components
// ============================================================================

function RegimeSummaryBar({
  distribution,
  total,
}: {
  distribution: Record<string, number>;
  total: number;
}) {
  const regimes: MarketRegime[] = ["bull", "bear", "sideways", "high-volatility", "low-volatility"];

  return (
    <div className="flex items-center gap-4">
      {regimes.map((regime) => {
        const count = distribution[regime] || 0;
        if (count === 0) return null;
        const config = getRegimeConfig(regime);
        const Icon = config.icon;
        return (
          <div key={regime} className="flex items-center gap-1.5">
            <Icon className={cn("w-3.5 h-3.5", config.color)} />
            <span className={cn("text-sm font-mono", config.color)}>{count}</span>
          </div>
        );
      })}
      <span className="text-xs text-neutral-500 ml-2">of {total} assets</span>
    </div>
  );
}

function AssetRegimeCard({
  data,
  onClick,
}: {
  data: AssetRegimeData;
  onClick?: () => void;
}) {
  const config = getRegimeConfig(data.regime);
  const Icon = config.icon;
  const confidencePercent = Math.round(data.confidence * 100);

  return (
    <div
      onClick={onClick}
      className={cn(
        "p-4 rounded-xl border transition-all duration-200",
        config.bgColor,
        config.borderColor,
        onClick && "cursor-pointer hover:scale-[1.02] hover:shadow-lg"
      )}
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <h4 className="font-semibold text-neutral-100">{data.display_name}</h4>
          <span className="text-xs text-neutral-500">{data.category}</span>
        </div>
        <div className={cn("p-2 rounded-lg", config.bgColor)}>
          <Icon className={cn("w-5 h-5", config.color)} />
        </div>
      </div>

      <div className="flex items-center gap-2 mb-3">
        <Badge className={cn("text-xs font-medium", config.bgColor, config.color, config.borderColor)}>
          {config.label}
        </Badge>
        <Badge
          className={cn(
            "text-xs font-mono",
            confidencePercent >= 75
              ? "bg-green-500/10 text-green-400 border-green-500/30"
              : "bg-amber-500/10 text-amber-400 border-amber-500/30"
          )}
        >
          {confidencePercent}%
        </Badge>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <div className="text-sm font-mono text-neutral-200">{data.daysInRegime}d</div>
          <div className="text-[10px] text-neutral-500">Duration</div>
        </div>
        <div>
          <div className="text-sm font-mono text-neutral-200">{data.volatility?.toFixed(1) || "—"}%</div>
          <div className="text-[10px] text-neutral-500">Vol</div>
        </div>
        <div>
          <div
            className={cn(
              "text-sm font-mono",
              (data.trendStrength || 0) > 0.2
                ? "text-green-400"
                : (data.trendStrength || 0) < -0.2
                ? "text-red-400"
                : "text-neutral-400"
            )}
          >
            {(data.trendStrength || 0) >= 0 ? "+" : ""}
            {((data.trendStrength || 0) * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-neutral-500">Trend</div>
        </div>
      </div>

      {onClick && (
        <div className="flex items-center justify-end mt-3 pt-2 border-t border-neutral-700/50">
          <span className="text-xs text-neutral-500 flex items-center gap-1">
            View details <ChevronRight className="w-3 h-3" />
          </span>
        </div>
      )}
    </div>
  );
}

function AssetRegimeRow({
  data,
  onClick,
}: {
  data: AssetRegimeData;
  onClick?: () => void;
}) {
  const config = getRegimeConfig(data.regime);
  const Icon = config.icon;
  const confidencePercent = Math.round(data.confidence * 100);

  return (
    <tr
      onClick={onClick}
      className={cn(
        "border-b border-neutral-800/50 transition-colors",
        onClick && "cursor-pointer hover:bg-neutral-800/30"
      )}
    >
      <td className="py-3 px-4">
        <div className="flex items-center gap-3">
          <div className={cn("w-2 h-2 rounded-full", config.dotColor)} />
          <div>
            <div className="font-medium text-neutral-100">{data.display_name}</div>
            <div className="text-xs text-neutral-500">{data.category}</div>
          </div>
        </div>
      </td>
      <td className="py-3 px-4">
        <div className="flex items-center gap-2">
          <Icon className={cn("w-4 h-4", config.color)} />
          <Badge className={cn("text-xs", config.bgColor, config.color, config.borderColor)}>
            {config.shortLabel}
          </Badge>
        </div>
      </td>
      <td className="py-3 px-4">
        <div className="flex items-center gap-2">
          <div
            className="h-1.5 w-16 bg-neutral-800 rounded-full overflow-hidden"
          >
            <div
              className={cn(
                "h-full rounded-full transition-all",
                confidencePercent >= 75 ? "bg-green-500" : "bg-amber-500"
              )}
              style={{ width: `${confidencePercent}%` }}
            />
          </div>
          <span className="text-sm font-mono text-neutral-300">{confidencePercent}%</span>
        </div>
      </td>
      <td className="py-3 px-4 text-center">
        <span className="text-sm font-mono text-neutral-300">{data.daysInRegime}d</span>
      </td>
      <td className="py-3 px-4 text-center">
        <span className="text-sm font-mono text-neutral-300">
          {data.volatility?.toFixed(1) || "—"}%
        </span>
      </td>
      <td className="py-3 px-4 text-center">
        <span
          className={cn(
            "text-sm font-mono",
            (data.trendStrength || 0) > 0.2
              ? "text-green-400"
              : (data.trendStrength || 0) < -0.2
              ? "text-red-400"
              : "text-neutral-400"
          )}
        >
          {(data.trendStrength || 0) >= 0 ? "+" : ""}
          {((data.trendStrength || 0) * 100).toFixed(0)}%
        </span>
      </td>
      <td className="py-3 px-4 text-center">
        <Badge
          className={cn(
            "text-xs",
            data.method === "GaussianHMM"
              ? "bg-cyan-500/10 text-cyan-400 border-cyan-500/30"
              : "bg-neutral-500/10 text-neutral-400 border-neutral-500/30"
          )}
        >
          {data.method === "GaussianHMM" ? "HMM" : "Mock"}
        </Badge>
      </td>
    </tr>
  );
}

function LoadingSkeleton({ viewMode }: { viewMode: "grid" | "table" }) {
  if (viewMode === "table") {
    return (
      <div className="space-y-2">
        {[...Array(6)].map((_, i) => (
          <Skeleton key={i} className="h-14 w-full" />
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {[...Array(8)].map((_, i) => (
        <Skeleton key={i} className="h-40 w-full rounded-xl" />
      ))}
    </div>
  );
}

function ErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 space-y-4">
      <AlertCircle className="h-12 w-12 text-red-400" />
      <div className="text-center">
        <p className="text-red-400 font-medium">Failed to load regime data</p>
        <p className="text-sm text-neutral-500">Unable to fetch multi-asset regime overview</p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded-lg text-red-400 text-sm transition-colors"
        >
          <RefreshCw className="h-4 w-4" />
          Retry
        </button>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function MultiAssetRegimeOverview({
  viewMode = "grid",
  category,
  regimeFilter,
  compact = false,
  className,
  onAssetClick,
}: MultiAssetRegimeOverviewProps) {
  const { data, isLoading, isError, refetch } = useAllRegimes();

  // Filter and sort assets
  const filteredAssets = useMemo(() => {
    if (!data?.regimes) return [];

    let assets = Object.values(data.regimes);

    // Filter by category
    if (category) {
      assets = assets.filter((a) => a.category === category);
    }

    // Filter by regime
    if (regimeFilter && regimeFilter.length > 0) {
      assets = assets.filter((a) => regimeFilter.includes(a.regime));
    }

    // Sort by category, then by confidence
    return assets.sort((a, b) => {
      if (a.category !== b.category) {
        return a.category.localeCompare(b.category);
      }
      return b.confidence - a.confidence;
    });
  }, [data, category, regimeFilter]);

  if (isLoading) {
    return (
      <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
        <CardHeader>
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Multi-Asset Regime Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingSkeleton viewMode={viewMode} />
        </CardContent>
      </Card>
    );
  }

  if (isError) {
    return (
      <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
        <CardHeader>
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Multi-Asset Regime Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorState onRetry={() => refetch()} />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Multi-Asset Regime Overview
          </CardTitle>
          <div className="flex items-center gap-3">
            <Badge className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30 text-xs">
              {data?.total_assets} Assets
            </Badge>
            {data?.calibration_date && (
              <div className="flex items-center gap-1 text-xs text-neutral-500">
                <Clock className="w-3 h-3" />
                Calibrated: {data.calibration_date}
              </div>
            )}
          </div>
        </div>

        {/* Regime Distribution Summary */}
        {data?.regime_distribution && (
          <div className="mt-4 pt-4 border-t border-neutral-800">
            <RegimeSummaryBar
              distribution={data.regime_distribution}
              total={data.total_assets}
            />
          </div>
        )}
      </CardHeader>

      <CardContent>
        {viewMode === "grid" ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {filteredAssets.map((asset) => (
              <AssetRegimeCard
                key={asset.asset_id}
                data={asset}
                onClick={onAssetClick ? () => onAssetClick(asset.asset_id, asset.name) : undefined}
              />
            ))}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-neutral-700 text-left">
                  <th className="py-3 px-4 text-xs font-medium text-neutral-400 uppercase tracking-wider">
                    Asset
                  </th>
                  <th className="py-3 px-4 text-xs font-medium text-neutral-400 uppercase tracking-wider">
                    Regime
                  </th>
                  <th className="py-3 px-4 text-xs font-medium text-neutral-400 uppercase tracking-wider">
                    Confidence
                  </th>
                  <th className="py-3 px-4 text-xs font-medium text-neutral-400 uppercase tracking-wider text-center">
                    Duration
                  </th>
                  <th className="py-3 px-4 text-xs font-medium text-neutral-400 uppercase tracking-wider text-center">
                    Vol
                  </th>
                  <th className="py-3 px-4 text-xs font-medium text-neutral-400 uppercase tracking-wider text-center">
                    Trend
                  </th>
                  <th className="py-3 px-4 text-xs font-medium text-neutral-400 uppercase tracking-wider text-center">
                    Source
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredAssets.map((asset) => (
                  <AssetRegimeRow
                    key={asset.asset_id}
                    data={asset}
                    onClick={onAssetClick ? () => onAssetClick(asset.asset_id, asset.name) : undefined}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Empty State */}
        {filteredAssets.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-neutral-500">
            <BarChart3 className="w-12 h-12 mb-4 opacity-50" />
            <p>No assets match the current filters</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default MultiAssetRegimeOverview;
