"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ActionabilityLevel } from "@/types/practical-metrics";
import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Activity,
  Clock,
} from "lucide-react";

interface ActionableSummaryProps {
  assetsNeedingAttention: number;
  actionableSignalsToday: number;
  bullishPercent: number;
  bearishPercent: number;
  neutralPercent: number;
  overallStatus: ActionabilityLevel;
  lastUpdated: string;
}

function getStatusColor(status: ActionabilityLevel): string {
  switch (status) {
    case "high":
      return "bg-green-500";
    case "medium":
      return "bg-yellow-500";
    case "low":
      return "bg-red-500";
  }
}

function getStatusText(status: ActionabilityLevel): string {
  switch (status) {
    case "high":
      return "Strong Signals";
    case "medium":
      return "Mixed Signals";
    case "low":
      return "Weak Signals";
  }
}

export function ActionableSummary({
  assetsNeedingAttention,
  actionableSignalsToday,
  bullishPercent,
  bearishPercent,
  neutralPercent,
  overallStatus,
  lastUpdated,
}: ActionableSummaryProps) {
  // Determine portfolio bias
  const getBias = () => {
    if (bullishPercent > bearishPercent + 15) return { text: "Bullish", color: "text-green-500" };
    if (bearishPercent > bullishPercent + 15) return { text: "Bearish", color: "text-red-500" };
    return { text: "Neutral", color: "text-yellow-500" };
  };

  const bias = getBias();

  return (
    <div className="bg-neutral-900/50 border border-neutral-800 rounded-lg p-4">
      <div className="flex flex-wrap items-center gap-4 justify-between">
        {/* Status Indicator */}
        <div className="flex items-center gap-3">
          <div className={cn("w-3 h-3 rounded-full animate-pulse", getStatusColor(overallStatus))} />
          <span className="text-sm font-medium text-neutral-200">
            {getStatusText(overallStatus)}
          </span>
        </div>

        {/* Attention Badge */}
        {assetsNeedingAttention > 0 && (
          <Badge className="bg-orange-500/10 border-orange-500/30 text-orange-500 gap-1.5">
            <AlertTriangle className="w-3 h-3" />
            {assetsNeedingAttention} asset{assetsNeedingAttention !== 1 ? "s" : ""} need attention
          </Badge>
        )}

        {/* Actionable Signals */}
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-400" />
          <span className="text-sm text-neutral-300">
            <span className="font-mono font-semibold text-blue-400">{actionableSignalsToday}</span>
            {" "}signals actionable today
          </span>
        </div>

        {/* Portfolio Bias */}
        <div className="flex items-center gap-2">
          {bias.text === "Bullish" ? (
            <TrendingUp className="w-4 h-4 text-green-500" />
          ) : bias.text === "Bearish" ? (
            <TrendingDown className="w-4 h-4 text-red-500" />
          ) : (
            <Activity className="w-4 h-4 text-yellow-500" />
          )}
          <span className="text-sm text-neutral-300">
            Portfolio bias:{" "}
            <span className={cn("font-semibold", bias.color)}>
              {bullishPercent.toFixed(0)}% bullish
            </span>
          </span>
        </div>

        {/* Directional Breakdown Mini */}
        <div className="flex items-center gap-3 text-xs">
          <span className="text-green-500 font-mono">{bullishPercent.toFixed(0)}% ↑</span>
          <span className="text-red-500 font-mono">{bearishPercent.toFixed(0)}% ↓</span>
          <span className="text-yellow-500 font-mono">{neutralPercent.toFixed(0)}% →</span>
        </div>

        {/* Last Updated */}
        <div className="flex items-center gap-1.5 text-neutral-500 text-xs">
          <Clock className="w-3 h-3" />
          <span className="font-mono">{lastUpdated}</span>
        </div>
      </div>
    </div>
  );
}
