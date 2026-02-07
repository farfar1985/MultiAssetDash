"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  Users,
  TrendingUp,
  TrendingDown,
  Minus,
  CheckCircle2,
  XCircle,
  Circle,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface HorizonPairVote {
  /** First horizon (e.g., "D+1") */
  h1: string;
  /** Second horizon (e.g., "D+5") */
  h2: string;
  /** Vote direction */
  vote: "bullish" | "bearish" | "neutral";
  /** Magnitude of the drift (percentage) */
  magnitude: number;
  /** Weight of this pair in ensemble */
  weight?: number;
}

export interface PairwiseVotingData {
  /** All horizon pair votes */
  votes: HorizonPairVote[];
  /** Total bullish votes */
  bullishCount: number;
  /** Total bearish votes */
  bearishCount: number;
  /** Total neutral votes */
  neutralCount: number;
  /** Net probability from voting */
  netProbability: number;
  /** Final signal direction */
  signal: "bullish" | "bearish" | "neutral";
}

export interface PairwiseVotingChartProps {
  data: PairwiseVotingData;
  /** Show vote grid visualization */
  showGrid?: boolean;
  /** Compact display */
  compact?: boolean;
  /** Custom class name */
  className?: string;
}

// ============================================================================
// Component
// ============================================================================

export function PairwiseVotingChart({
  data,
  showGrid = true,
  compact = false,
  className,
}: PairwiseVotingChartProps) {
  const totalVotes = data.bullishCount + data.bearishCount + data.neutralCount;
  const bullishPercent = totalVotes > 0 ? (data.bullishCount / totalVotes) * 100 : 0;
  const bearishPercent = totalVotes > 0 ? (data.bearishCount / totalVotes) * 100 : 0;
  const neutralPercent = totalVotes > 0 ? (data.neutralCount / totalVotes) * 100 : 0;

  // Group votes by significance
  const significantVotes = useMemo(() => {
    return data.votes
      .filter((v) => Math.abs(v.magnitude) > 0.5)
      .sort((a, b) => Math.abs(b.magnitude) - Math.abs(a.magnitude))
      .slice(0, 6);
  }, [data.votes]);

  const signalConfig = {
    bullish: {
      icon: TrendingUp,
      color: "text-green-400",
      bgColor: "bg-green-500/10",
      borderColor: "border-green-500/30",
      label: "Bullish",
    },
    bearish: {
      icon: TrendingDown,
      color: "text-red-400",
      bgColor: "bg-red-500/10",
      borderColor: "border-red-500/30",
      label: "Bearish",
    },
    neutral: {
      icon: Minus,
      color: "text-amber-400",
      bgColor: "bg-amber-500/10",
      borderColor: "border-amber-500/30",
      label: "Neutral",
    },
  };

  const signal = signalConfig[data.signal];
  const SignalIcon = signal.icon;

  if (compact) {
    return (
      <div className={cn("p-3 rounded-xl bg-neutral-900/50 border border-neutral-800", className)}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Users className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-medium text-neutral-200">Model Voting</span>
          </div>
          <Badge className={cn("text-xs", signal.color, signal.bgColor, signal.borderColor)}>
            <SignalIcon className="w-3 h-3 mr-1" />
            {signal.label}
          </Badge>
        </div>

        {/* Compact vote bar */}
        <div className="flex items-center gap-2">
          <div className="flex-1 h-6 bg-neutral-800 rounded-lg overflow-hidden flex">
            {bullishPercent > 0 && (
              <div
                className="h-full bg-green-500 flex items-center justify-center"
                style={{ width: `${bullishPercent}%` }}
              >
                {bullishPercent > 15 && (
                  <span className="text-[10px] font-bold text-white">{data.bullishCount}</span>
                )}
              </div>
            )}
            {neutralPercent > 0 && (
              <div
                className="h-full bg-amber-500 flex items-center justify-center"
                style={{ width: `${neutralPercent}%` }}
              >
                {neutralPercent > 15 && (
                  <span className="text-[10px] font-bold text-neutral-900">{data.neutralCount}</span>
                )}
              </div>
            )}
            {bearishPercent > 0 && (
              <div
                className="h-full bg-red-500 flex items-center justify-center"
                style={{ width: `${bearishPercent}%` }}
              >
                {bearishPercent > 15 && (
                  <span className="text-[10px] font-bold text-white">{data.bearishCount}</span>
                )}
              </div>
            )}
          </div>
          <span className="text-xs font-mono text-neutral-400 shrink-0">
            {data.bullishCount}/{totalVotes}
          </span>
        </div>
      </div>
    );
  }

  return (
    <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Users className="w-4 h-4 text-cyan-400" />
            Pairwise Voting
          </CardTitle>
          <Badge className={cn("px-3 py-1", signal.color, signal.bgColor, signal.borderColor)}>
            <SignalIcon className="w-3.5 h-3.5 mr-1.5" />
            {signal.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0 space-y-4">
        {/* Vote Summary */}
        <div className="grid grid-cols-3 gap-3">
          <div className="p-3 bg-green-500/5 border border-green-500/20 rounded-lg text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <span className="text-2xl font-bold font-mono text-green-400">
                {data.bullishCount}
              </span>
            </div>
            <div className="text-[10px] text-neutral-500">Bullish Votes</div>
            <div className="text-xs font-mono text-green-400/70">
              {bullishPercent.toFixed(0)}%
            </div>
          </div>
          <div className="p-3 bg-amber-500/5 border border-amber-500/20 rounded-lg text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <Minus className="w-4 h-4 text-amber-400" />
              <span className="text-2xl font-bold font-mono text-amber-400">
                {data.neutralCount}
              </span>
            </div>
            <div className="text-[10px] text-neutral-500">Neutral</div>
            <div className="text-xs font-mono text-amber-400/70">
              {neutralPercent.toFixed(0)}%
            </div>
          </div>
          <div className="p-3 bg-red-500/5 border border-red-500/20 rounded-lg text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <TrendingDown className="w-4 h-4 text-red-400" />
              <span className="text-2xl font-bold font-mono text-red-400">
                {data.bearishCount}
              </span>
            </div>
            <div className="text-[10px] text-neutral-500">Bearish Votes</div>
            <div className="text-xs font-mono text-red-400/70">
              {bearishPercent.toFixed(0)}%
            </div>
          </div>
        </div>

        {/* Vote Bar Visualization */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-neutral-500">
            <span>Vote Distribution</span>
            <span className="font-mono">{totalVotes} total pairs</span>
          </div>
          <div className="h-8 bg-neutral-800 rounded-lg overflow-hidden flex">
            {bullishPercent > 0 && (
              <div
                className="h-full bg-gradient-to-r from-green-600 to-green-500 flex items-center justify-center transition-all duration-500"
                style={{ width: `${bullishPercent}%` }}
              >
                <span className="text-xs font-bold text-white drop-shadow">
                  {data.bullishCount}
                </span>
              </div>
            )}
            {neutralPercent > 0 && (
              <div
                className="h-full bg-gradient-to-r from-amber-600 to-amber-500 flex items-center justify-center transition-all duration-500"
                style={{ width: `${neutralPercent}%` }}
              >
                <span className="text-xs font-bold text-neutral-900 drop-shadow">
                  {data.neutralCount}
                </span>
              </div>
            )}
            {bearishPercent > 0 && (
              <div
                className="h-full bg-gradient-to-r from-red-600 to-red-500 flex items-center justify-center transition-all duration-500"
                style={{ width: `${bearishPercent}%` }}
              >
                <span className="text-xs font-bold text-white drop-shadow">
                  {data.bearishCount}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Net Probability Gauge */}
        <div className="p-3 bg-neutral-800/30 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-neutral-500">Net Probability</span>
            <span
              className={cn(
                "text-lg font-bold font-mono",
                data.netProbability > 0
                  ? "text-green-400"
                  : data.netProbability < 0
                  ? "text-red-400"
                  : "text-amber-400"
              )}
            >
              {data.netProbability >= 0 ? "+" : ""}
              {(data.netProbability * 100).toFixed(1)}%
            </span>
          </div>
          <div className="relative h-3 bg-neutral-800 rounded-full overflow-hidden">
            {/* Center line */}
            <div className="absolute inset-y-0 left-1/2 w-0.5 bg-neutral-600 z-10" />
            {/* Probability bar */}
            {data.netProbability !== 0 && (
              <div
                className={cn(
                  "absolute inset-y-0 transition-all duration-500 rounded-full",
                  data.netProbability > 0 ? "bg-green-500" : "bg-red-500"
                )}
                style={{
                  left: data.netProbability > 0 ? "50%" : `${50 + data.netProbability * 50}%`,
                  width: `${Math.abs(data.netProbability) * 50}%`,
                }}
              />
            )}
          </div>
          <div className="flex justify-between text-[9px] text-neutral-600 mt-1">
            <span>-100%</span>
            <span>0</span>
            <span>+100%</span>
          </div>
        </div>

        {/* Vote Grid (if enabled) */}
        {showGrid && significantVotes.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-neutral-500 uppercase tracking-wider">
              Top Horizon Pairs
            </div>
            <div className="grid grid-cols-2 gap-2">
              {significantVotes.map((vote, idx) => (
                <div
                  key={`${vote.h1}-${vote.h2}-${idx}`}
                  className={cn(
                    "flex items-center gap-2 p-2 rounded-lg border",
                    vote.vote === "bullish"
                      ? "bg-green-500/5 border-green-500/20"
                      : vote.vote === "bearish"
                      ? "bg-red-500/5 border-red-500/20"
                      : "bg-amber-500/5 border-amber-500/20"
                  )}
                >
                  {vote.vote === "bullish" ? (
                    <CheckCircle2 className="w-4 h-4 text-green-400 shrink-0" />
                  ) : vote.vote === "bearish" ? (
                    <XCircle className="w-4 h-4 text-red-400 shrink-0" />
                  ) : (
                    <Circle className="w-4 h-4 text-amber-400 shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium text-neutral-300">
                      {vote.h1} → {vote.h2}
                    </div>
                    <div
                      className={cn(
                        "text-[10px] font-mono",
                        vote.magnitude > 0 ? "text-green-400" : "text-red-400"
                      )}
                    >
                      {vote.magnitude >= 0 ? "+" : ""}
                      {vote.magnitude.toFixed(2)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default PairwiseVotingChart;
