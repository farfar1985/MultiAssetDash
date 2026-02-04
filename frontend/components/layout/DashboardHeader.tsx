"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { usePortfolioSummary } from "@/hooks/useApi";
import { EnsembleSelector } from "@/components/dashboard/EnsembleSelector";
import type { EnsembleMethod } from "@/lib/api-client";
import type { SignalDirection } from "@/types";
import { Clock, Activity, Layers, TrendingUp, TrendingDown, Minus } from "lucide-react";

function getDirectionIcon(direction: SignalDirection) {
  switch (direction) {
    case "bullish":
      return <TrendingUp className="w-4 h-4" />;
    case "bearish":
      return <TrendingDown className="w-4 h-4" />;
    case "neutral":
      return <Minus className="w-4 h-4" />;
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

function formatTimestamp(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

interface DashboardHeaderProps {
  onMethodChange?: (method: EnsembleMethod) => void;
  initialMethod?: EnsembleMethod;
}

export function DashboardHeader({
  onMethodChange,
  initialMethod = "accuracy_weighted",
}: DashboardHeaderProps) {
  const [ensembleMethod, setEnsembleMethod] = useState<EnsembleMethod>(initialMethod);

  const { data: portfolio, isLoading, error } = usePortfolioSummary(ensembleMethod);

  const handleMethodChange = (method: EnsembleMethod) => {
    setEnsembleMethod(method);
    onMethodChange?.(method);
  };

  return (
    <div className="space-y-4">
      {/* Method Selector */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-neutral-100">
            Ensemble Method
          </h2>
          <p className="text-xs text-neutral-500 mt-0.5">
            Select the model combination strategy
          </p>
        </div>
        <EnsembleSelector
          value={ensembleMethod}
          onChange={handleMethodChange}
          variant="tabs"
        />
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Current Method */}
        <Card className="bg-neutral-900/50 border-neutral-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-neutral-500 mb-2">
              <Layers className="w-4 h-4" />
              <span className="text-[10px] uppercase tracking-wider">
                Active Method
              </span>
            </div>
            {isLoading ? (
              <Skeleton className="h-6 w-24 bg-neutral-800" />
            ) : (
              <div className="text-sm font-semibold text-blue-400 capitalize">
                {portfolio?.ensembleMethod.replace(/_/g, " ") || "-"}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Overall Signal */}
        <Card className="bg-neutral-900/50 border-neutral-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-neutral-500 mb-2">
              <Activity className="w-4 h-4" />
              <span className="text-[10px] uppercase tracking-wider">
                Portfolio Signal
              </span>
            </div>
            {isLoading ? (
              <Skeleton className="h-6 w-24 bg-neutral-800" />
            ) : portfolio ? (
              <div className="flex items-center gap-2">
                <Badge
                  className={cn(
                    "text-xs font-semibold px-2 py-0.5 capitalize",
                    getDirectionBgColor(portfolio.overallSignal)
                  )}
                >
                  <span className="mr-1">
                    {getDirectionIcon(portfolio.overallSignal)}
                  </span>
                  {portfolio.overallSignal}
                </Badge>
                <span
                  className={cn(
                    "font-mono text-sm font-medium",
                    getDirectionColor(portfolio.overallSignal)
                  )}
                >
                  {portfolio.overallConfidence}%
                </span>
              </div>
            ) : (
              <span className="text-neutral-500">-</span>
            )}
          </CardContent>
        </Card>

        {/* Total Models */}
        <Card className="bg-neutral-900/50 border-neutral-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-neutral-500 mb-2">
              <Layers className="w-4 h-4" />
              <span className="text-[10px] uppercase tracking-wider">
                Total Models
              </span>
            </div>
            {isLoading ? (
              <Skeleton className="h-6 w-20 bg-neutral-800" />
            ) : (
              <div className="text-xl font-bold font-mono text-neutral-100">
                {portfolio?.totalModels.toLocaleString() || "-"}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Last Updated */}
        <Card className="bg-neutral-900/50 border-neutral-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-neutral-500 mb-2">
              <Clock className="w-4 h-4" />
              <span className="text-[10px] uppercase tracking-wider">
                Last Updated
              </span>
            </div>
            {isLoading ? (
              <Skeleton className="h-6 w-16 bg-neutral-800" />
            ) : (
              <div className="text-sm font-mono text-neutral-300">
                {portfolio ? formatTimestamp(portfolio.lastUpdated) : "-"}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Error State */}
      {error && (
        <div className="bg-red-900/20 border border-red-900/50 rounded-lg p-3 text-center">
          <p className="text-red-500 text-sm">Failed to load portfolio summary</p>
          <p className="text-xs text-neutral-500 mt-1">{error.message}</p>
        </div>
      )}
    </div>
  );
}
