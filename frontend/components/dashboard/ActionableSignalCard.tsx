"use client";

import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TargetLadder } from "./TargetLadder";
import { ConvictionMeter } from "./ConvictionMeter";
import { MagnitudeIndicator } from "./MagnitudeIndicator";
import type { TargetLadderSignal } from "@/types/target-ladder";
import {
  Target,
  TrendingUp,
  TrendingDown,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
} from "lucide-react";

interface ActionableSignalCardProps {
  signal: TargetLadderSignal;
  showFullLadder?: boolean;
  onClick?: () => void;
}

function formatPrice(price: number): string {
  if (price >= 10000) {
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  if (price >= 100) {
    return `$${price.toFixed(2)}`;
  }
  return `$${price.toFixed(4)}`;
}

function formatTime(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function ActionableSignalCard({
  signal,
  showFullLadder = true,
  onClick,
}: ActionableSignalCardProps) {
  const isBullish = signal.direction === "bullish";
  const directionColor = isBullish ? "text-green-500" : "text-red-500";
  const directionBg = isBullish
    ? "bg-green-500/10 border-green-500/30"
    : "bg-red-500/10 border-red-500/30";

  // Determine overall action
  const shouldAct = signal.actionable && signal.conviction !== "LOW";

  return (
    <Card
      className={cn(
        "bg-neutral-900/50 border-neutral-800 hover:border-neutral-700 transition-all duration-200 overflow-hidden",
        onClick && "cursor-pointer hover:bg-neutral-900/80",
        shouldAct && "ring-1 ring-green-500/20"
      )}
      onClick={onClick}
    >
      {/* Top action banner */}
      <div
        className={cn(
          "px-4 py-2 flex items-center justify-between",
          shouldAct
            ? "bg-green-500/10 border-b border-green-500/20"
            : "bg-neutral-800/50 border-b border-neutral-700"
        )}
      >
        <div className="flex items-center gap-2">
          {shouldAct ? (
            <CheckCircle className="w-4 h-4 text-green-500" />
          ) : signal.actionable ? (
            <AlertTriangle className="w-4 h-4 text-yellow-500" />
          ) : (
            <XCircle className="w-4 h-4 text-neutral-500" />
          )}
          <span
            className={cn(
              "text-xs font-bold uppercase tracking-wider",
              shouldAct
                ? "text-green-400"
                : signal.actionable
                  ? "text-yellow-400"
                  : "text-neutral-500"
            )}
          >
            {shouldAct
              ? "TAKE ACTION"
              : signal.actionable
                ? "REVIEW"
                : "WAIT"}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-neutral-500 text-xs">
          <Clock className="w-3 h-3" />
          <span className="font-mono">{formatTime(signal.generatedAt)}</span>
        </div>
      </div>

      <CardHeader className="p-4 pb-3">
        {/* Asset Name + Price Row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
              {signal.symbol}
            </span>
            <span className="text-sm font-semibold text-neutral-100">
              {signal.assetName}
            </span>
            <Badge className={cn("text-xs px-2 py-0.5", directionBg, directionColor)}>
              {isBullish ? (
                <TrendingUp className="w-3 h-3 mr-1" />
              ) : (
                <TrendingDown className="w-3 h-3 mr-1" />
              )}
              {signal.direction.toUpperCase()}
            </Badge>
          </div>
          <div className="text-right">
            <div className="font-mono text-lg font-semibold text-neutral-100">
              {formatPrice(signal.current)}
            </div>
            <div className={cn("font-mono text-xs", directionColor)}>
              {isBullish ? "↑" : "↓"} Target: {formatPrice(signal.targets[0]?.price || signal.current)}
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0 space-y-4">
        {/* Main content grid */}
        <div className={cn("grid gap-4", showFullLadder ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1")}>
          {/* Left: Target Ladder */}
          {showFullLadder && (
            <div className="p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
              <TargetLadder
                currentPrice={signal.current}
                targets={signal.targets}
                direction={signal.direction}
                symbol={signal.symbol}
              />
            </div>
          )}

          {/* Right: Meters */}
          <div className="space-y-4">
            {/* Conviction Meter */}
            <div className="p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
              <ConvictionMeter
                consensus={signal.consensus}
                targetsAgreeing={signal.targetsAgreeing}
                targetsTotal={signal.targetsTotal}
                conviction={signal.conviction}
                variant="bar"
              />
            </div>

            {/* Magnitude Indicator */}
            <div className="p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
              <MagnitudeIndicator
                magnitude={signal.magnitude}
                magnitudePercent={signal.magnitudePercent}
                actionable={signal.actionable}
                threshold={signal.actionThreshold}
                direction={signal.direction}
              />
            </div>
          </div>
        </div>

        {/* Bottom summary row */}
        <div className="pt-3 border-t border-neutral-800">
          <div className="flex items-center justify-between">
            {/* Quick stats */}
            <div className="flex items-center gap-4 text-xs">
              <div className="flex items-center gap-1.5">
                <Target className="w-3.5 h-3.5 text-neutral-500" />
                <span className="text-neutral-400">
                  <span className="font-mono font-semibold text-neutral-200">
                    {signal.targetsAgreeing}
                  </span>
                  /{signal.targetsTotal} targets
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-neutral-400">
                  Move:{" "}
                  <span className={cn("font-mono font-semibold", directionColor)}>
                    {signal.magnitudePercent >= 0 ? "+" : ""}{signal.magnitudePercent.toFixed(2)}%
                  </span>
                </span>
              </div>
            </div>

            {/* Action button/badge */}
            <Badge
              className={cn(
                "text-sm font-bold px-4 py-1.5",
                shouldAct
                  ? "bg-green-500 text-white hover:bg-green-600"
                  : signal.actionable
                    ? "bg-yellow-500/10 border-yellow-500/30 text-yellow-400"
                    : "bg-neutral-700 text-neutral-400"
              )}
            >
              {shouldAct ? (
                <>
                  <CheckCircle className="w-4 h-4 mr-1.5" />
                  EXECUTE {isBullish ? "LONG" : "SHORT"}
                </>
              ) : signal.actionable ? (
                "LOW CONVICTION - REVIEW"
              ) : (
                "NO ACTION"
              )}
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Compact version for overview grids
export function ActionableSignalCardCompact({
  signal,
  onClick,
}: Omit<ActionableSignalCardProps, "showFullLadder">) {
  const isBullish = signal.direction === "bullish";
  const directionColor = isBullish ? "text-green-500" : "text-red-500";
  const shouldAct = signal.actionable && signal.conviction !== "LOW";

  return (
    <Card
      className={cn(
        "bg-neutral-900/50 border-neutral-800 hover:border-neutral-700 transition-all duration-200",
        onClick && "cursor-pointer hover:bg-neutral-900/80",
        shouldAct && "ring-1 ring-green-500/20"
      )}
      onClick={onClick}
    >
      <CardContent className="p-3">
        <div className="flex items-center justify-between gap-3">
          {/* Asset info */}
          <div className="flex items-center gap-2 min-w-0">
            <span className="font-mono text-xs text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
              {signal.symbol}
            </span>
            <span className={cn("text-lg", directionColor)}>
              {isBullish ? "↑" : "↓"}
            </span>
          </div>

          {/* Price */}
          <div className="font-mono text-sm font-semibold text-neutral-100">
            {formatPrice(signal.current)}
          </div>

          {/* Conviction indicator */}
          <div className="flex items-center gap-1">
            {Array.from({ length: signal.targetsTotal }).map((_, i) => (
              <div
                key={i}
                className={cn(
                  "w-1.5 h-3 rounded-sm",
                  i < signal.targetsAgreeing
                    ? signal.conviction === "HIGH"
                      ? "bg-green-500"
                      : signal.conviction === "MEDIUM"
                        ? "bg-yellow-500"
                        : "bg-red-500"
                    : "bg-neutral-700"
                )}
              />
            ))}
          </div>

          {/* Move */}
          <span className={cn("font-mono text-xs font-semibold", directionColor)}>
            {signal.magnitudePercent >= 0 ? "+" : ""}{signal.magnitudePercent.toFixed(1)}%
          </span>

          {/* Action indicator */}
          <div
            className={cn(
              "w-2 h-2 rounded-full",
              shouldAct ? "bg-green-500 animate-pulse" : "bg-neutral-600"
            )}
          />
        </div>
      </CardContent>
    </Card>
  );
}
