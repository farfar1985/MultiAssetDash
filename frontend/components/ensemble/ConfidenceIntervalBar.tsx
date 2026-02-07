"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Target,
  BarChart3,
  Shield,
  TrendingUp,
  TrendingDown,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface ConfidenceInterval {
  /** Lower bound of prediction interval (percentage) */
  lower: number;
  /** Point estimate (percentage) */
  point: number;
  /** Upper bound of prediction interval (percentage) */
  upper: number;
  /** Coverage level (e.g., 0.90 for 90% CI) */
  coverage: number;
}

export interface ConfidenceIntervalBarProps {
  data: ConfidenceInterval;
  /** Asset name for context */
  assetName?: string;
  /** Time horizon label */
  horizon?: string;
  /** Show detailed breakdown */
  showDetails?: boolean;
  /** Compact mode */
  compact?: boolean;
  /** Orientation */
  orientation?: "horizontal" | "vertical";
  /** Custom class name */
  className?: string;
}

// ============================================================================
// Helper Functions
// ============================================================================

function formatPercent(value: number, showSign = true): string {
  const sign = showSign && value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function getDirectionFromInterval(interval: ConfidenceInterval): "bullish" | "bearish" | "neutral" {
  // If entire interval is positive, bullish
  if (interval.lower > 0.1) return "bullish";
  // If entire interval is negative, bearish
  if (interval.upper < -0.1) return "bearish";
  // If interval spans zero significantly, neutral
  return "neutral";
}

// ============================================================================
// Component
// ============================================================================

export function ConfidenceIntervalBar({
  data,
  assetName,
  horizon = "D+5",
  showDetails = true,
  compact = false,
  orientation: _orientation = "horizontal",
  className,
}: ConfidenceIntervalBarProps) {
  const direction = useMemo(() => getDirectionFromInterval(data), [data]);
  const coveragePercent = Math.round(data.coverage * 100);

  // Calculate visual positions (normalize to -5% to +5% range for display)
  const visualRange = 5; // +-5% scale
  const normalizePosition = (value: number) => {
    const clamped = Math.max(-visualRange, Math.min(visualRange, value));
    return ((clamped + visualRange) / (visualRange * 2)) * 100;
  };

  const lowerPos = normalizePosition(data.lower);
  const pointPos = normalizePosition(data.point);
  const upperPos = normalizePosition(data.upper);
  const intervalWidth = upperPos - lowerPos;

  const directionConfig = {
    bullish: {
      icon: ArrowUpRight,
      color: "text-green-400",
      bgColor: "bg-green-500/10",
      borderColor: "border-green-500/30",
      barColor: "bg-gradient-to-r from-green-600 to-emerald-500",
    },
    bearish: {
      icon: ArrowDownRight,
      color: "text-red-400",
      bgColor: "bg-red-500/10",
      borderColor: "border-red-500/30",
      barColor: "bg-gradient-to-r from-red-600 to-rose-500",
    },
    neutral: {
      icon: Minus,
      color: "text-amber-400",
      bgColor: "bg-amber-500/10",
      borderColor: "border-amber-500/30",
      barColor: "bg-gradient-to-r from-amber-600 to-yellow-500",
    },
  };

  const config = directionConfig[direction];
  const DirectionIcon = config.icon;

  if (compact) {
    return (
      <div className={cn("p-3 rounded-xl bg-neutral-900/50 border border-neutral-800", className)}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-purple-400" />
            <span className="text-xs text-neutral-400">{coveragePercent}% CI</span>
          </div>
          <Badge className={cn("text-xs", config.color, config.bgColor, config.borderColor)}>
            <DirectionIcon className="w-3 h-3 mr-1" />
            {formatPercent(data.point)}
          </Badge>
        </div>

        {/* Compact interval display */}
        <div className="flex items-center gap-2 text-sm font-mono">
          <span className="text-neutral-500">[</span>
          <span className={data.lower >= 0 ? "text-green-400" : "text-red-400"}>
            {formatPercent(data.lower)}
          </span>
          <span className="text-neutral-600">,</span>
          <span className={data.upper >= 0 ? "text-green-400" : "text-red-400"}>
            {formatPercent(data.upper)}
          </span>
          <span className="text-neutral-500">]</span>
        </div>
      </div>
    );
  }

  return (
    <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Shield className="w-4 h-4 text-purple-400" />
            Prediction Interval
            {horizon && <span className="text-neutral-500 font-normal">({horizon})</span>}
          </CardTitle>
          <Badge className="bg-purple-500/10 text-purple-400 border-purple-500/30 text-xs">
            {coveragePercent}% Coverage
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0 space-y-4">
        {/* Point Estimate Display */}
        <div className={cn("p-4 rounded-xl border", config.bgColor, config.borderColor)}>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-neutral-500 mb-1">Point Estimate</div>
              <div className={cn("text-3xl font-bold font-mono", config.color)}>
                {formatPercent(data.point)}
              </div>
              {assetName && (
                <div className="text-xs text-neutral-500 mt-1">{assetName}</div>
              )}
            </div>
            <div className={cn("p-3 rounded-xl", config.bgColor)}>
              <DirectionIcon className={cn("w-8 h-8", config.color)} />
            </div>
          </div>
        </div>

        {/* Interval Values */}
        <div className="grid grid-cols-3 gap-3">
          <div className="p-3 bg-neutral-800/30 rounded-lg text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <TrendingDown className="w-3.5 h-3.5 text-neutral-400" />
              <span className="text-[10px] text-neutral-500">Lower</span>
            </div>
            <div
              className={cn(
                "text-lg font-bold font-mono",
                data.lower >= 0 ? "text-green-400" : "text-red-400"
              )}
            >
              {formatPercent(data.lower)}
            </div>
          </div>
          <div className="p-3 bg-neutral-800/30 rounded-lg text-center border-2 border-neutral-700">
            <div className="flex items-center justify-center gap-1 mb-1">
              <Target className="w-3.5 h-3.5 text-purple-400" />
              <span className="text-[10px] text-neutral-500">Point</span>
            </div>
            <div className={cn("text-lg font-bold font-mono", config.color)}>
              {formatPercent(data.point)}
            </div>
          </div>
          <div className="p-3 bg-neutral-800/30 rounded-lg text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <TrendingUp className="w-3.5 h-3.5 text-neutral-400" />
              <span className="text-[10px] text-neutral-500">Upper</span>
            </div>
            <div
              className={cn(
                "text-lg font-bold font-mono",
                data.upper >= 0 ? "text-green-400" : "text-red-400"
              )}
            >
              {formatPercent(data.upper)}
            </div>
          </div>
        </div>

        {/* Visual Interval Bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-neutral-500">
            <span>Prediction Range</span>
            <span className="font-mono">
              Width: {(data.upper - data.lower).toFixed(2)}%
            </span>
          </div>

          <div className="relative h-10 bg-neutral-800 rounded-lg overflow-hidden">
            {/* Zero line */}
            <div className="absolute inset-y-0 left-1/2 w-0.5 bg-neutral-600 z-20" />

            {/* Confidence interval band */}
            <div
              className={cn(
                "absolute inset-y-2 rounded transition-all duration-500 opacity-50",
                config.barColor
              )}
              style={{
                left: `${lowerPos}%`,
                width: `${intervalWidth}%`,
              }}
            />

            {/* Lower bound marker */}
            <div
              className="absolute inset-y-0 w-1 bg-neutral-400 z-10 transition-all duration-500"
              style={{ left: `${lowerPos}%` }}
            >
              <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[9px] text-neutral-500 font-mono whitespace-nowrap">
                {formatPercent(data.lower)}
              </div>
            </div>

            {/* Point estimate marker */}
            <div
              className={cn(
                "absolute inset-y-0 w-1.5 z-30 transition-all duration-500 rounded-full",
                data.point >= 0 ? "bg-green-500" : "bg-red-500"
              )}
              style={{ left: `${pointPos}%` }}
            >
              <div
                className={cn(
                  "absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] font-bold font-mono whitespace-nowrap",
                  config.color
                )}
              >
                {formatPercent(data.point)}
              </div>
            </div>

            {/* Upper bound marker */}
            <div
              className="absolute inset-y-0 w-1 bg-neutral-400 z-10 transition-all duration-500"
              style={{ left: `${upperPos}%` }}
            >
              <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[9px] text-neutral-500 font-mono whitespace-nowrap">
                {formatPercent(data.upper)}
              </div>
            </div>

            {/* Scale markers */}
            <div className="absolute bottom-0 left-0 right-0 flex justify-between px-1 text-[8px] text-neutral-600">
              <span>-5%</span>
              <span>-2.5%</span>
              <span>0</span>
              <span>+2.5%</span>
              <span>+5%</span>
            </div>
          </div>

          {/* Extra spacing for bottom labels */}
          <div className="h-4" />
        </div>

        {/* Details */}
        {showDetails && (
          <div className="grid grid-cols-2 gap-3 pt-2">
            <div className="p-3 bg-neutral-800/20 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <BarChart3 className="w-3.5 h-3.5 text-neutral-400" />
                <span className="text-[10px] text-neutral-500">Interval Width</span>
              </div>
              <div className="text-sm font-mono text-neutral-200">
                {(data.upper - data.lower).toFixed(2)}%
              </div>
            </div>
            <div className="p-3 bg-neutral-800/20 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <Shield className="w-3.5 h-3.5 text-purple-400" />
                <span className="text-[10px] text-neutral-500">Coverage</span>
              </div>
              <div className="text-sm font-mono text-purple-400">
                {coveragePercent}% Conformal
              </div>
            </div>
          </div>
        )}

        {/* Interpretation */}
        <div className="p-3 bg-neutral-800/20 rounded-lg border border-neutral-700/50">
          <div className="text-xs text-neutral-400">
            <span className="font-medium text-neutral-300">Interpretation: </span>
            {direction === "bullish" && data.lower > 0 ? (
              <>
                The entire {coveragePercent}% confidence interval is positive, suggesting
                high confidence in upward movement.
              </>
            ) : direction === "bearish" && data.upper < 0 ? (
              <>
                The entire {coveragePercent}% confidence interval is negative, suggesting
                high confidence in downward movement.
              </>
            ) : (
              <>
                The {coveragePercent}% confidence interval spans zero, indicating
                directional uncertainty. Point estimate suggests{" "}
                {data.point >= 0 ? "slight upward" : "slight downward"} bias.
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ConfidenceIntervalBar;
