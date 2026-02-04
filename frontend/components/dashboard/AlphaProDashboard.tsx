"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { LiveSignalCard } from "@/components/dashboard/LiveSignalCard";
import { ApiHealthIndicator } from "@/components/dashboard/ApiHealthIndicator";
import { MOCK_ASSETS, MOCK_SIGNALS, type Horizon, type SignalData } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import {
  Zap,
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  BarChart3,
  DollarSign,
  Percent,
  Award,
  Flame,
  ArrowUpRight,
  ArrowDownRight,
  AlertCircle,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface OptimizedSignal {
  assetId: AssetId;
  assetName: string;
  symbol: string;
  currentPrice: number;
  signal: SignalData;
  alphaScore: number;
  expectedPnL: number;
  sharpeContribution: number;
  rank: number;
}

// ============================================================================
// Signal Generation and Scoring
// ============================================================================

function calculateAlphaScore(signal: SignalData): number {
  // Alpha score based on Sharpe, directional accuracy, and confidence
  const sharpeWeight = Math.min(signal.sharpeRatio * 15, 40);
  const accuracyWeight = signal.directionalAccuracy * 0.4;
  const confidenceWeight = signal.confidence * 0.2;
  const agreementBonus = (signal.modelsAgreeing / signal.modelsTotal) * 10;

  return sharpeWeight + accuracyWeight + confidenceWeight + agreementBonus;
}

function calculateExpectedPnL(signal: SignalData, currentPrice: number): number {
  // Simulate expected P&L based on directional accuracy and typical move
  const expectedMove = currentPrice * (signal.directionalAccuracy / 100) * 0.02;
  return signal.direction === "bearish" ? -expectedMove : expectedMove;
}

function getOptimizedSignals(): OptimizedSignal[] {
  const signals: OptimizedSignal[] = [];
  const horizons: Horizon[] = ["D+1", "D+5", "D+10"];

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    horizons.forEach((horizon) => {
      const signal = MOCK_SIGNALS[assetId as AssetId]?.[horizon];
      if (signal && signal.sharpeRatio >= 0.8) { // Only high-Sharpe signals
        const alphaScore = calculateAlphaScore(signal);
        signals.push({
          assetId: assetId as AssetId,
          assetName: asset.name,
          symbol: asset.symbol,
          currentPrice: asset.currentPrice,
          signal,
          alphaScore,
          expectedPnL: calculateExpectedPnL(signal, asset.currentPrice),
          sharpeContribution: signal.sharpeRatio * (signal.modelsAgreeing / signal.modelsTotal),
          rank: 0,
        });
      }
    });
  });

  // Sort by alpha score and assign ranks
  return signals
    .sort((a, b) => b.alphaScore - a.alphaScore)
    .map((s, i) => ({ ...s, rank: i + 1 }));
}

// ============================================================================
// Header Component
// ============================================================================

function AlphaProHeader() {
  return (
    <div className="bg-gradient-to-r from-purple-900/30 via-blue-900/20 to-purple-900/30 border border-purple-500/20 rounded-xl p-6">
      <div className="flex items-center gap-4 mb-4">
        <div className="p-3 bg-purple-500/20 rounded-xl border border-purple-500/30">
          <Zap className="w-8 h-8 text-purple-400" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-neutral-100">Alpha Gen Pro</h1>
          <p className="text-sm text-neutral-400">
            Optimized ensemble signals for professional alpha generation
          </p>
        </div>
      </div>
      <div className="flex flex-wrap gap-3">
        <Badge className="bg-purple-500/10 border-purple-500/30 text-purple-300 px-3 py-1.5">
          <Award className="w-3.5 h-3.5 mr-1.5" />
          High-Sharpe Focus
        </Badge>
        <Badge className="bg-blue-500/10 border-blue-500/30 text-blue-300 px-3 py-1.5">
          <Target className="w-3.5 h-3.5 mr-1.5" />
          Ensemble Optimized
        </Badge>
        <Badge className="bg-green-500/10 border-green-500/30 text-green-300 px-3 py-1.5">
          <Activity className="w-3.5 h-3.5 mr-1.5" />
          Real-time Signals
        </Badge>
      </div>
    </div>
  );
}

// ============================================================================
// Performance Stats Component
// ============================================================================

interface PerformanceStatsProps {
  signals: OptimizedSignal[];
}

function PerformanceStats({ signals }: PerformanceStatsProps) {
  const stats = useMemo(() => {
    const avgSharpe = signals.reduce((acc, s) => acc + s.signal.sharpeRatio, 0) / signals.length;
    const avgAccuracy = signals.reduce((acc, s) => acc + s.signal.directionalAccuracy, 0) / signals.length;
    const totalExpectedPnL = signals.slice(0, 5).reduce((acc, s) => acc + s.expectedPnL, 0);
    const topSignal = signals[0];

    return { avgSharpe, avgAccuracy, totalExpectedPnL, topSignal };
  }, [signals]);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Average Sharpe */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <BarChart3 className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wider">Avg Sharpe</span>
          </div>
          <div className="text-2xl font-bold font-mono text-blue-400">
            {stats.avgSharpe.toFixed(2)}
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Portfolio risk-adjusted
          </div>
        </CardContent>
      </Card>

      {/* Win Rate */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <Percent className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wider">Avg Accuracy</span>
          </div>
          <div className="text-2xl font-bold font-mono text-green-400">
            {stats.avgAccuracy.toFixed(1)}%
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Directional accuracy
          </div>
        </CardContent>
      </Card>

      {/* Expected P&L */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <DollarSign className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wider">Expected Move</span>
          </div>
          <div className={cn(
            "text-2xl font-bold font-mono",
            stats.totalExpectedPnL >= 0 ? "text-green-400" : "text-red-400"
          )}>
            {stats.totalExpectedPnL >= 0 ? "+" : ""}{stats.totalExpectedPnL.toFixed(2)}%
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Top 5 signals
          </div>
        </CardContent>
      </Card>

      {/* Top Signal */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <Flame className="w-4 h-4 text-orange-400" />
            <span className="text-xs uppercase tracking-wider">Top Signal</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl font-bold font-mono text-orange-400">
              {stats.topSignal?.symbol}
            </span>
            <Badge className={cn(
              "text-xs",
              stats.topSignal?.signal.direction === "bullish"
                ? "bg-green-500/10 border-green-500/30 text-green-400"
                : "bg-red-500/10 border-red-500/30 text-red-400"
            )}>
              {stats.topSignal?.signal.direction}
            </Badge>
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Alpha: {stats.topSignal?.alphaScore.toFixed(1)}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ============================================================================
// Top Alpha Signal Card
// ============================================================================

interface TopAlphaSignalProps {
  signal: OptimizedSignal;
}

function TopAlphaSignalCard({ signal }: TopAlphaSignalProps) {
  const isBullish = signal.signal.direction === "bullish";
  const directionColor = isBullish ? "text-green-500" : "text-red-500";
  const directionBg = isBullish
    ? "bg-green-500/10 border-green-500/30"
    : "bg-red-500/10 border-red-500/30";

  return (
    <Card className={cn(
      "bg-neutral-900/50 border-neutral-800 hover:border-neutral-700 transition-all",
      signal.rank <= 3 && "ring-1 ring-purple-500/30"
    )}>
      <CardContent className="p-5">
        {/* Rank Badge */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center font-bold text-sm",
              signal.rank === 1 ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30" :
              signal.rank === 2 ? "bg-neutral-500/20 text-neutral-300 border border-neutral-500/30" :
              signal.rank === 3 ? "bg-orange-500/20 text-orange-400 border border-orange-500/30" :
              "bg-neutral-800 text-neutral-500"
            )}>
              #{signal.rank}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-neutral-500 bg-neutral-800 px-2 py-0.5 rounded">
                  {signal.symbol}
                </span>
                <span className="font-semibold text-neutral-100">{signal.assetName}</span>
              </div>
              <span className="text-xs text-neutral-500">{signal.signal.horizon}</span>
            </div>
          </div>
          <Badge className={cn("text-sm gap-1.5 px-3 py-1", directionBg, directionColor)}>
            {isBullish ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
            {signal.signal.direction.toUpperCase()}
          </Badge>
        </div>

        {/* Alpha Score Bar */}
        <div className="mb-4">
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="text-neutral-500 flex items-center gap-1">
              <Zap className="w-3 h-3" />
              Alpha Score
            </span>
            <span className="font-mono font-bold text-purple-400">{signal.alphaScore.toFixed(1)}</span>
          </div>
          <div className="h-2 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full transition-all"
              style={{ width: `${Math.min(signal.alphaScore, 100)}%` }}
            />
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-3 gap-3">
          <div className="p-2 bg-neutral-800/50 rounded text-center">
            <div className="text-xs text-neutral-500 mb-0.5">Sharpe</div>
            <div className="font-mono font-bold text-blue-400">{signal.signal.sharpeRatio.toFixed(2)}</div>
          </div>
          <div className="p-2 bg-neutral-800/50 rounded text-center">
            <div className="text-xs text-neutral-500 mb-0.5">Accuracy</div>
            <div className="font-mono font-bold text-green-400">{signal.signal.directionalAccuracy.toFixed(0)}%</div>
          </div>
          <div className="p-2 bg-neutral-800/50 rounded text-center">
            <div className="text-xs text-neutral-500 mb-0.5">Confidence</div>
            <div className="font-mono font-bold text-cyan-400">{signal.signal.confidence}%</div>
          </div>
        </div>

        {/* Model Agreement */}
        <div className="mt-4 pt-3 border-t border-neutral-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-neutral-500">Model Consensus</span>
            <div className="flex items-center gap-2">
              <div className="flex gap-0.5">
                {Array.from({ length: signal.signal.modelsTotal }).map((_, i) => (
                  <div
                    key={i}
                    className={cn(
                      "w-1.5 h-4 rounded-sm",
                      i < signal.signal.modelsAgreeing
                        ? "bg-purple-500"
                        : "bg-neutral-700"
                    )}
                  />
                ))}
              </div>
              <span className="font-mono text-xs text-neutral-400">
                {signal.signal.modelsAgreeing}/{signal.signal.modelsTotal}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Signal Ranking Table
// ============================================================================

interface SignalRankingTableProps {
  signals: OptimizedSignal[];
}

function SignalRankingTable({ signals }: SignalRankingTableProps) {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Award className="w-5 h-5 text-yellow-400" />
            <span className="text-sm font-semibold text-neutral-100">Signal Ranking</span>
          </div>
          <span className="text-xs text-neutral-500">By Alpha Score</span>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="divide-y divide-neutral-800">
          {signals.slice(0, 10).map((signal) => (
            <div
              key={`${signal.assetId}-${signal.signal.horizon}`}
              className="flex items-center justify-between px-4 py-3 hover:bg-neutral-800/30 transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className={cn(
                  "w-6 text-center font-mono text-sm font-bold",
                  signal.rank <= 3 ? "text-yellow-400" : "text-neutral-500"
                )}>
                  {signal.rank}
                </span>
                <span className="font-mono text-xs text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
                  {signal.symbol}
                </span>
                <span className="text-sm text-neutral-300">{signal.assetName}</span>
                <Badge className="text-[10px] px-1.5 py-0 bg-neutral-800 border-neutral-700 text-neutral-400">
                  {signal.signal.horizon}
                </Badge>
              </div>
              <div className="flex items-center gap-4">
                <div className={cn(
                  "flex items-center gap-1",
                  signal.signal.direction === "bullish" ? "text-green-500" : "text-red-500"
                )}>
                  {signal.signal.direction === "bullish"
                    ? <ArrowUpRight className="w-4 h-4" />
                    : <ArrowDownRight className="w-4 h-4" />
                  }
                </div>
                <span className="font-mono text-xs text-neutral-400 w-12 text-right">
                  SR {signal.signal.sharpeRatio.toFixed(2)}
                </span>
                <div className="w-16 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-purple-500 rounded-full"
                    style={{ width: `${Math.min(signal.alphaScore, 100)}%` }}
                  />
                </div>
                <span className="font-mono text-sm font-bold text-purple-400 w-10 text-right">
                  {signal.alphaScore.toFixed(0)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Alpha Pro Dashboard Component
// ============================================================================

export function AlphaProDashboard() {
  const optimizedSignals = useMemo(() => getOptimizedSignals(), []);
  const topSignals = optimizedSignals.slice(0, 6);

  return (
    <div className="space-y-6">
      {/* Header */}
      <AlphaProHeader />

      {/* Performance Stats */}
      <PerformanceStats signals={optimizedSignals} />

      <Separator className="bg-neutral-800" />

      {/* Live Backend Signals */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
              <Activity className="w-5 h-5 text-green-400" />
              Live Ensemble Signals
            </h2>
            <ApiHealthIndicator />
          </div>
          <span className="text-xs text-neutral-500 font-mono">Real-time API</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <LiveSignalCard asset="Crude_Oil" displayName="Crude Oil" />
          <LiveSignalCard asset="Bitcoin" displayName="Bitcoin" />
          <LiveSignalCard asset="SP500" displayName="S&P 500" />
          <LiveSignalCard asset="GOLD" displayName="Gold" />
        </div>
      </section>

      <Separator className="bg-neutral-800" />

      {/* Top Alpha Signals */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
            <Flame className="w-5 h-5 text-orange-400" />
            Top Alpha Opportunities
          </h2>
          <Badge className="bg-purple-500/10 border-purple-500/30 text-purple-300 px-2 py-1 text-xs">
            Sharpe &gt; 0.8
          </Badge>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {topSignals.map((signal) => (
            <TopAlphaSignalCard
              key={`${signal.assetId}-${signal.signal.horizon}`}
              signal={signal}
            />
          ))}
        </div>
      </section>

      <Separator className="bg-neutral-800" />

      {/* Full Ranking Table */}
      <section>
        <SignalRankingTable signals={optimizedSignals} />
      </section>

      {/* Disclaimer */}
      <div className="p-4 bg-neutral-900/30 border border-neutral-800 rounded-lg flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-neutral-500 flex-shrink-0 mt-0.5" />
        <div className="text-xs text-neutral-500">
          <strong className="text-neutral-400">Alpha Pro Disclaimer:</strong> Signals are generated
          by ensemble ML models optimized for risk-adjusted returns. Past performance does not
          guarantee future results. Always validate with your own analysis before trading.
        </div>
      </div>
    </div>
  );
}
