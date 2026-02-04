"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { useMetrics } from "@/hooks/useApi";
import type { PerformanceMetrics } from "@/lib/api-client";
import type { AssetId } from "@/types";

interface MetricCardProps {
  label: string;
  value: string;
  subtext?: string;
  valueColor?: string;
  isLoading?: boolean;
}

function MetricCard({
  label,
  value,
  subtext,
  valueColor = "text-neutral-100",
  isLoading = false,
}: MetricCardProps) {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardContent className="p-4">
        <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
          {label}
        </div>
        {isLoading ? (
          <Skeleton className="h-8 w-20 bg-neutral-800" />
        ) : (
          <div className={cn("text-2xl font-bold font-mono", valueColor)}>
            {value}
          </div>
        )}
        {subtext && (
          <div className="text-xs text-neutral-500 mt-1">
            {subtext}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface MetricsGridProps {
  assetId?: AssetId;
  metrics?: PerformanceMetrics;
}

export function MetricsGrid({ assetId, metrics: propMetrics }: MetricsGridProps) {
  const { data: fetchedMetrics, isLoading, error } = useMetrics(assetId || "crude-oil", undefined, {
    enabled: !propMetrics && !!assetId,
  });

  const metrics = propMetrics || fetchedMetrics;

  if (error) {
    return (
      <div className="bg-neutral-900/50 border border-red-900/50 rounded-lg p-4 text-center">
        <p className="text-red-500 text-sm">Failed to load metrics</p>
        <p className="text-xs text-neutral-500 mt-1">{error.message}</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      <MetricCard
        label="Sharpe Ratio"
        value={metrics?.sharpeRatio.toFixed(2) || "-"}
        valueColor="text-blue-400"
        subtext="Risk-adjusted return"
        isLoading={isLoading}
      />
      <MetricCard
        label="Directional Accuracy"
        value={metrics ? `${metrics.directionalAccuracy.toFixed(1)}%` : "-"}
        valueColor={metrics && metrics.directionalAccuracy >= 55 ? "text-green-500" : "text-neutral-100"}
        subtext="Correct predictions"
        isLoading={isLoading}
      />
      <MetricCard
        label="Total Return"
        value={metrics ? `${metrics.totalReturn >= 0 ? "+" : ""}${metrics.totalReturn.toFixed(1)}%` : "-"}
        valueColor={metrics && metrics.totalReturn >= 0 ? "text-green-500" : "text-red-500"}
        subtext="Cumulative performance"
        isLoading={isLoading}
      />
      <MetricCard
        label="Max Drawdown"
        value={metrics ? `${metrics.maxDrawdown.toFixed(1)}%` : "-"}
        valueColor="text-red-500"
        subtext="Largest peak-to-trough"
        isLoading={isLoading}
      />
      <MetricCard
        label="Win Rate"
        value={metrics ? `${metrics.winRate.toFixed(1)}%` : "-"}
        valueColor={metrics && metrics.winRate >= 55 ? "text-green-500" : "text-neutral-100"}
        subtext="Profitable trades"
        isLoading={isLoading}
      />
      <MetricCard
        label="Model Count"
        value={metrics?.modelCount.toLocaleString() || "-"}
        valueColor="text-blue-400"
        subtext="Active ensemble models"
        isLoading={isLoading}
      />
    </div>
  );
}
