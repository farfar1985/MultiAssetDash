"use client";

import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TargetLadder } from "./TargetLadder";
import { ConvictionMeter } from "./ConvictionMeter";
import type {
  TargetLevel,
  TargetDirection,
  ConvictionLevel,
} from "@/types/target-ladder";
import {
  TrendingUp,
  TrendingDown,
  CheckCircle,
  Clock,
  Activity,
  Target,
} from "lucide-react";

/**
 * Signal data for the summary card
 */
export interface SignalSummaryData {
  /** Asset symbol (e.g., "CL", "GC", "BTC") */
  symbol: string;
  /** Asset display name */
  assetName: string;
  /** Current price */
  currentPrice: number;
  /** Signal direction */
  direction: TargetDirection;
  /** Price targets for ladder */
  targets: TargetLevel[];
  /** Consensus percentage (0-100) */
  consensus: number;
  /** Number agreeing (horizons or targets) */
  agreeing: number;
  /** Total (horizons or targets) */
  total: number;
  /** Conviction level */
  conviction: ConvictionLevel;
  /** Whether signal is actionable */
  actionable: boolean;
  /** Sharpe ratio */
  sharpeRatio: number;
  /** Win rate percentage (0-100) */
  winRate: number;
  /** Signal generation timestamp */
  generatedAt?: string;
}

interface SignalSummaryCardProps {
  signal: SignalSummaryData;
  /** Show full target ladder or compact view */
  showFullLadder?: boolean;
  /** Label type for conviction meter */
  labelType?: "targets" | "horizons";
  /** Click handler */
  onClick?: () => void;
  /** Additional class names */
  className?: string;
}

function formatPrice(price: number): string {
  if (price >= 10000) {
    return `$${price.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
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

export function SignalSummaryCard({
  signal,
  showFullLadder = true,
  labelType = "horizons",
  onClick,
  className,
}: SignalSummaryCardProps) {
  const isBullish = signal.direction === "bullish";
  const directionColor = isBullish ? "text-green-500" : "text-red-500";
  const directionBg = isBullish
    ? "bg-green-500/10 border-green-500/30"
    : "bg-red-500/10 border-red-500/30";

  // Determine action recommendation
  const recommendation = signal.actionable ? "ACTIONABLE" : "WAIT";
  const recommendationColor = signal.actionable
    ? "bg-green-500 text-white"
    : "bg-neutral-700 text-neutral-300";

  return (
    <Card
      className={cn(
        "bg-neutral-900/50 border-neutral-800 hover:border-neutral-700",
        "transition-all duration-200 overflow-hidden",
        onClick && "cursor-pointer hover:bg-neutral-900/80",
        signal.actionable && "ring-1 ring-green-500/20",
        className
      )}
      onClick={onClick}
      data-testid="signal-card"
      data-direction={signal.direction}
    >
      {/* Top recommendation banner */}
      <div
        className={cn(
          "px-4 py-2.5 flex items-center justify-between",
          signal.actionable
            ? "bg-green-500/10 border-b border-green-500/20"
            : "bg-neutral-800/50 border-b border-neutral-700"
        )}
      >
        <div className="flex items-center gap-2">
          {signal.actionable ? (
            <CheckCircle className="w-4 h-4 text-green-500" />
          ) : (
            <Clock className="w-4 h-4 text-neutral-500" />
          )}
          <span
            className={cn(
              "text-sm font-bold uppercase tracking-wider",
              signal.actionable ? "text-green-400" : "text-neutral-400"
            )}
          >
            {recommendation}
          </span>
        </div>
        {signal.generatedAt && (
          <div className="flex items-center gap-1.5 text-neutral-500 text-xs">
            <Clock className="w-3 h-3" />
            <span className="font-mono">{formatTime(signal.generatedAt)}</span>
          </div>
        )}
      </div>

      <CardHeader className="p-4 pb-3">
        {/* Asset Name + Price + Direction Badge Row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Symbol */}
            <span className="font-mono text-xs text-neutral-500 bg-neutral-800 px-2 py-1 rounded">
              {signal.symbol}
            </span>
            {/* Asset Name */}
            <span className="text-base font-semibold text-neutral-100">
              {signal.assetName}
            </span>
            {/* Direction Badge */}
            <Badge
              className={cn(
                "text-xs font-bold px-2.5 py-1 uppercase",
                directionBg,
                directionColor
              )}
            >
              {isBullish ? (
                <TrendingUp className="w-3.5 h-3.5 mr-1" />
              ) : (
                <TrendingDown className="w-3.5 h-3.5 mr-1" />
              )}
              {signal.direction}
            </Badge>
          </div>
          {/* Current Price */}
          <div className="text-right">
            <div className="font-mono text-xl font-semibold text-neutral-100">
              {formatPrice(signal.currentPrice)}
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0 space-y-4">
        {/* Main visualization grid */}
        <div
          className={cn(
            "grid gap-4",
            showFullLadder ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"
          )}
        >
          {/* Left: Target Ladder */}
          {showFullLadder && (
            <div className="p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
              <TargetLadder
                currentPrice={signal.currentPrice}
                targets={signal.targets}
                direction={signal.direction}
                symbol={signal.symbol}
                animated={true}
              />
            </div>
          )}

          {/* Right: Conviction Meter */}
          <div className="p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50 flex items-center justify-center">
            <ConvictionMeter
              consensus={signal.consensus}
              targetsAgreeing={signal.agreeing}
              targetsTotal={signal.total}
              conviction={signal.conviction}
              variant="circular"
              labelType={labelType}
              animated={true}
              size="lg"
            />
          </div>
        </div>

        {/* Metrics row */}
        <div className="grid grid-cols-2 gap-4 pt-3 border-t border-neutral-800">
          {/* Sharpe Ratio */}
          <div className="flex items-center gap-3 p-3 bg-neutral-800/30 rounded-lg">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Activity className="w-4 h-4 text-blue-400" />
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-neutral-500">
                Sharpe Ratio
              </div>
              <div
                className={cn(
                  "font-mono text-lg font-bold",
                  signal.sharpeRatio >= 2
                    ? "text-green-400"
                    : signal.sharpeRatio >= 1
                      ? "text-blue-400"
                      : signal.sharpeRatio >= 0
                        ? "text-yellow-400"
                        : "text-red-400"
                )}
              >
                {signal.sharpeRatio.toFixed(2)}
              </div>
            </div>
          </div>

          {/* Win Rate */}
          <div className="flex items-center gap-3 p-3 bg-neutral-800/30 rounded-lg">
            <div className="p-2 bg-green-500/10 rounded-lg">
              <Target className="w-4 h-4 text-green-400" />
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-neutral-500">
                Win Rate
              </div>
              <div
                className={cn(
                  "font-mono text-lg font-bold",
                  signal.winRate >= 60
                    ? "text-green-400"
                    : signal.winRate >= 50
                      ? "text-yellow-400"
                      : "text-red-400"
                )}
              >
                {signal.winRate.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>

        {/* Action button/badge */}
        <div className="pt-2">
          <Badge
            className={cn(
              "w-full justify-center text-sm font-bold py-2.5 uppercase tracking-wider transition-all",
              recommendationColor
            )}
          >
            {signal.actionable ? (
              <>
                <CheckCircle className="w-4 h-4 mr-2" />
                Execute {isBullish ? "Long" : "Short"} Position
              </>
            ) : (
              <>
                <Clock className="w-4 h-4 mr-2" />
                Wait for Better Setup
              </>
            )}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Compact version for grid displays
 */
export function SignalSummaryCardCompact({
  signal,
  onClick,
  className,
}: Omit<SignalSummaryCardProps, "showFullLadder" | "labelType">) {
  const isBullish = signal.direction === "bullish";
  const directionColor = isBullish ? "text-green-500" : "text-red-500";

  return (
    <Card
      className={cn(
        "bg-neutral-900/50 border-neutral-800 hover:border-neutral-700",
        "transition-all duration-200",
        onClick && "cursor-pointer hover:bg-neutral-900/80",
        signal.actionable && "ring-1 ring-green-500/20",
        className
      )}
      onClick={onClick}
      data-testid="signal-card"
      data-direction={signal.direction}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between gap-4">
          {/* Asset info */}
          <div className="flex items-center gap-3 min-w-0">
            <span className="font-mono text-xs text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
              {signal.symbol}
            </span>
            <span className="text-sm font-semibold text-neutral-100 truncate">
              {signal.assetName}
            </span>
            <span className={cn("text-lg", directionColor)}>
              {isBullish ? "↑" : "↓"}
            </span>
          </div>

          {/* Price */}
          <div className="font-mono text-sm font-semibold text-neutral-100">
            {formatPrice(signal.currentPrice)}
          </div>

          {/* Mini conviction indicator */}
          <div className="flex items-center gap-1">
            {Array.from({ length: signal.total }).map((_, i) => (
              <div
                key={i}
                className={cn(
                  "w-1.5 h-4 rounded-sm transition-all",
                  i < signal.agreeing
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

          {/* Quick metrics */}
          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="text-blue-400">{signal.sharpeRatio.toFixed(2)}</span>
            <span className="text-neutral-500">|</span>
            <span className="text-green-400">{signal.winRate.toFixed(0)}%</span>
          </div>

          {/* Action indicator */}
          <div
            className={cn(
              "w-2.5 h-2.5 rounded-full flex-shrink-0",
              signal.actionable
                ? "bg-green-500 animate-pulse"
                : "bg-neutral-600"
            )}
          />
        </div>
      </CardContent>
    </Card>
  );
}
