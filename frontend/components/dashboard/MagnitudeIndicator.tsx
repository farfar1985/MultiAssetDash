"use client";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Zap, XCircle } from "lucide-react";

interface MagnitudeIndicatorProps {
  magnitude: number;
  magnitudePercent: number;
  actionable: boolean;
  threshold: number;
  direction: "bullish" | "bearish";
}

function getMagnitudeLevel(magnitudePercent: number): {
  label: string;
  color: string;
  bgColor: string;
  icon: "high" | "medium" | "low";
} {
  if (magnitudePercent >= 2) {
    return {
      label: "SIGNIFICANT",
      color: "text-green-400",
      bgColor: "bg-green-500/10 border-green-500/30",
      icon: "high",
    };
  }
  if (magnitudePercent >= 1) {
    return {
      label: "MODERATE",
      color: "text-yellow-400",
      bgColor: "bg-yellow-500/10 border-yellow-500/30",
      icon: "medium",
    };
  }
  if (magnitudePercent >= 0.5) {
    return {
      label: "MINOR",
      color: "text-orange-400",
      bgColor: "bg-orange-500/10 border-orange-500/30",
      icon: "low",
    };
  }
  return {
    label: "NEGLIGIBLE",
    color: "text-neutral-400",
    bgColor: "bg-neutral-500/10 border-neutral-500/30",
    icon: "low",
  };
}

export function MagnitudeIndicator({
  magnitude,
  magnitudePercent,
  actionable,
  threshold,
  direction,
}: MagnitudeIndicatorProps) {
  const level = getMagnitudeLevel(magnitudePercent);
  const isBullish = direction === "bullish";

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider text-neutral-500">
          Expected Move
        </span>
        <Badge
          className={cn(
            "text-xs font-semibold",
            actionable
              ? "bg-green-500/10 border-green-500/30 text-green-400"
              : "bg-neutral-500/10 border-neutral-500/30 text-neutral-400"
          )}
        >
          {actionable ? (
            <>
              <Zap className="w-3 h-3 mr-1" />
              ACTIONABLE
            </>
          ) : (
            <>
              <XCircle className="w-3 h-3 mr-1" />
              PASS
            </>
          )}
        </Badge>
      </div>

      {/* Magnitude display */}
      <div className={cn("p-3 rounded-lg border", level.bgColor)}>
        <div className="flex items-center justify-between">
          {/* Move value */}
          <div className="flex items-center gap-2">
            {isBullish ? (
              <TrendingUp className={cn("w-5 h-5", "text-green-500")} />
            ) : (
              <TrendingDown className={cn("w-5 h-5", "text-red-500")} />
            )}
            <div>
              <span className={cn("font-mono text-xl font-bold", level.color)}>
                {magnitude >= 0 ? "+" : ""}{magnitude.toFixed(2)}
              </span>
              <span className="text-neutral-500 text-sm ml-1">pts</span>
            </div>
          </div>

          {/* Percentage */}
          <div className="text-right">
            <span className={cn("font-mono text-lg font-semibold", level.color)}>
              ({magnitudePercent >= 0 ? "+" : ""}{magnitudePercent.toFixed(2)}%)
            </span>
          </div>
        </div>

        {/* Level badge */}
        <div className="mt-2 flex items-center justify-between">
          <span className={cn("text-xs font-semibold uppercase", level.color)}>
            {level.label}
          </span>
          <span className="text-xs text-neutral-500">
            Threshold: {threshold}%
          </span>
        </div>
      </div>

      {/* Visual magnitude bar */}
      <div className="space-y-1">
        <div className="h-2 bg-neutral-800 rounded-full overflow-hidden relative">
          {/* Threshold marker */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-neutral-500"
            style={{ left: `${Math.min(threshold * 20, 100)}%` }}
          />
          {/* Magnitude fill */}
          <div
            className={cn(
              "h-full rounded-full transition-all duration-700 ease-out",
              actionable ? "bg-green-500" : "bg-neutral-600"
            )}
            style={{ width: `${Math.min(magnitudePercent * 20, 100)}%` }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-neutral-600 font-mono">
          <span>0%</span>
          <span>2.5%</span>
          <span>5%+</span>
        </div>
      </div>

      {/* Trading recommendation */}
      <div
        className={cn(
          "text-xs text-center py-2 px-3 rounded-md font-medium",
          actionable
            ? "bg-green-500/10 text-green-400 border border-green-500/20"
            : "bg-neutral-800 text-neutral-500 border border-neutral-700"
        )}
      >
        {actionable
          ? `Move ${magnitudePercent.toFixed(2)}% exceeds ${threshold}% threshold — TRADEABLE`
          : `Move ${magnitudePercent.toFixed(2)}% below ${threshold}% threshold — NOT WORTH TRADING`}
      </div>
    </div>
  );
}
