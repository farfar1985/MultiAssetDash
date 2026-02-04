"use client";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { useBackendSignal, useBackendMetrics, useBackendForecast } from "@/hooks/useApi";
import type { SignalDirection } from "@/types";

interface LiveSignalCardProps {
  asset: string;
  displayName?: string;
  onClick?: () => void;
}

type BackendSignalType = "LONG" | "SHORT" | "NEUTRAL";

function mapSignalDirection(signal: BackendSignalType): SignalDirection {
  switch (signal) {
    case "LONG":
      return "bullish";
    case "SHORT":
      return "bearish";
    case "NEUTRAL":
    default:
      return "neutral";
  }
}

function getDirectionColor(direction: SignalDirection): string {
  switch (direction) {
    case "bullish":
      return "text-green-500";
    case "bearish":
      return "text-red-500";
    case "neutral":
      return "text-yellow-500";
  }
}

function getDirectionBgColor(direction: SignalDirection): string {
  switch (direction) {
    case "bullish":
      return "bg-green-500/10 border-green-500/30 text-green-500";
    case "bearish":
      return "bg-red-500/10 border-red-500/30 text-red-500";
    case "neutral":
      return "bg-yellow-500/10 border-yellow-500/30 text-yellow-500";
  }
}

function getDirectionIcon(direction: SignalDirection): string {
  switch (direction) {
    case "bullish":
      return "^";
    case "bearish":
      return "v";
    case "neutral":
      return "-";
  }
}

function getConfidenceBarColor(confidence: number): string {
  if (confidence >= 80) return "bg-green-500";
  if (confidence >= 60) return "bg-blue-500";
  if (confidence >= 40) return "bg-yellow-500";
  return "bg-red-500";
}

function LiveSignalCardSkeleton() {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <Skeleton className="h-5 w-32 bg-neutral-800" />
          <Skeleton className="h-4 w-16 bg-neutral-800" />
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-7 w-24 bg-neutral-800" />
          <Skeleton className="h-4 w-20 bg-neutral-800" />
        </div>
        <div>
          <Skeleton className="h-2 w-full bg-neutral-800 rounded-full" />
        </div>
        <div className="grid grid-cols-3 gap-3 pt-2 border-t border-neutral-800">
          <Skeleton className="h-10 bg-neutral-800" />
          <Skeleton className="h-10 bg-neutral-800" />
          <Skeleton className="h-10 bg-neutral-800" />
        </div>
      </CardContent>
    </Card>
  );
}

function LiveSignalCardError({ message }: { message: string }) {
  return (
    <Card className="bg-neutral-900/50 border-red-900/50">
      <CardContent className="p-4">
        <div className="text-center text-red-500 text-sm py-4">
          <p>Failed to load signal</p>
          <p className="text-xs text-neutral-500 mt-1">{message}</p>
        </div>
      </CardContent>
    </Card>
  );
}

export function LiveSignalCard({
  asset,
  displayName,
  onClick,
}: LiveSignalCardProps) {
  const {
    data: signalData,
    isLoading: isSignalLoading,
    error: signalError,
  } = useBackendSignal(asset);

  const {
    data: metricsData,
    isLoading: isMetricsLoading,
    error: metricsError,
  } = useBackendMetrics(asset);

  const {
    data: forecastData,
    isLoading: isForecastLoading,
  } = useBackendForecast(asset);

  const isLoading = isSignalLoading || isMetricsLoading || isForecastLoading;
  const error = signalError || metricsError;

  if (isLoading) {
    return <LiveSignalCardSkeleton />;
  }

  if (error) {
    return <LiveSignalCardError message={error.message} />;
  }

  // Get the latest signal from the signal data
  const latestSignal = signalData?.data?.[signalData.data.length - 1];
  const direction = latestSignal ? mapSignalDirection(latestSignal.signal as BackendSignalType) : "neutral";
  const confidence = latestSignal ? Math.abs(latestSignal.net_prob * 100) : 0;
  const strength = latestSignal ? latestSignal.strength * 100 : 0;

  // Get metrics
  const optimizedMetrics = metricsData?.optimized_metrics || {};
  const config = metricsData?.configuration || {};

  // Get forecast signal (live)
  const liveSignal = forecastData?.signal || "N/A";
  const viableHorizons = forecastData?.viable_horizons || config.viable_horizons || [];

  const assetName = displayName || asset.replace(/_/g, " ");

  return (
    <Card
      className={cn(
        "bg-neutral-900/50 border-neutral-800 hover:border-neutral-700 transition-all duration-200",
        onClick && "cursor-pointer hover:bg-neutral-900/80"
      )}
      onClick={onClick}
    >
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-neutral-100">
              {assetName}
            </span>
          </div>
          <div className="text-right">
            <span className="font-mono text-xs text-neutral-500">
              {viableHorizons.length > 0 && `D+${viableHorizons.join(",")}`}
            </span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0 space-y-4">
        {/* Signal Direction Badge */}
        <div className="flex items-center justify-between">
          <Badge
            className={cn(
              "text-sm font-semibold px-3 py-1 capitalize",
              getDirectionBgColor(direction)
            )}
          >
            <span className="mr-1">{getDirectionIcon(direction)}</span>
            {direction}
          </Badge>

          {/* Live Signal Indicator */}
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs text-neutral-500">
              {liveSignal}
            </span>
          </div>
        </div>

        {/* Confidence/Strength Bar */}
        <div>
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="text-neutral-500">Confidence</span>
            <span className={cn("font-mono font-medium", getDirectionColor(direction))}>
              {confidence.toFixed(1)}%
            </span>
          </div>
          <div className="h-2 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-300",
                getConfidenceBarColor(confidence)
              )}
              style={{ width: `${Math.min(confidence, 100)}%` }}
            />
          </div>
        </div>

        {/* Signal Strength */}
        <div className="text-xs text-neutral-400 font-mono">
          Signal strength: {strength.toFixed(1)}%
        </div>

        {/* Key Metrics from Backend */}
        <div className="grid grid-cols-3 gap-3 pt-2 border-t border-neutral-800">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-0.5">
              Sharpe
            </div>
            <div className="font-mono text-sm font-medium text-blue-400">
              {(optimizedMetrics.sharpe_ratio || 0).toFixed(2)}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-0.5">
              Win%
            </div>
            <div className="font-mono text-sm font-medium text-neutral-100">
              {(optimizedMetrics.win_rate || 0).toFixed(1)}%
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-0.5">
              Return
            </div>
            <div className={cn(
              "font-mono text-sm font-medium",
              (optimizedMetrics.total_return || 0) >= 0 ? "text-green-500" : "text-red-500"
            )}>
              {(optimizedMetrics.total_return || 0) >= 0 ? "+" : ""}{(optimizedMetrics.total_return || 0).toFixed(1)}%
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
