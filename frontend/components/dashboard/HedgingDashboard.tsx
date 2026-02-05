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
  Shield,
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  DollarSign,
  Percent,
  Clock,
  BarChart2,
  ArrowUpRight,
  ArrowDownRight,
  AlertTriangle,
  CheckCircle2,
  Layers,
  Scale,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface HedgeOpportunity {
  assetId: AssetId;
  assetName: string;
  symbol: string;
  currentPrice: number;
  signal: SignalData;
  dollarPerTrade: number;
  winRate: number;
  horizonCoverage: number;
  hedgeScore: number;
  hedgeType: "protective" | "speculative" | "basis";
  urgency: "high" | "medium" | "low";
  rank: number;
}

// ============================================================================
// Hedge Opportunity Generation
// ============================================================================

function calculateDollarPerTrade(signal: SignalData, currentPrice: number): number {
  // Calculate expected $/trade based on accuracy and typical move size
  const expectedMovePercent = signal.directionalAccuracy > 55 ? 0.025 : 0.015;
  const winProb = signal.directionalAccuracy / 100;
  const avgWin = currentPrice * expectedMovePercent;
  const avgLoss = currentPrice * expectedMovePercent * 0.8; // Tighter stops
  return (winProb * avgWin) - ((1 - winProb) * avgLoss);
}

function calculateHorizonCoverage(assetId: AssetId): number {
  const horizons: Horizon[] = ["D+1", "D+5", "D+10"];
  let consistentDirection = 0;
  const signals = horizons.map(h => MOCK_SIGNALS[assetId]?.[h]).filter(Boolean);

  if (signals.length < 2) return 33;

  const firstDirection = signals[0]?.direction;
  for (const signal of signals) {
    if (signal?.direction === firstDirection && signal.direction !== "neutral") {
      consistentDirection++;
    }
  }

  return Math.round((consistentDirection / signals.length) * 100);
}

function determineHedgeType(signal: SignalData): "protective" | "speculative" | "basis" {
  if (signal.direction === "bearish" && signal.confidence >= 70) return "protective";
  if (signal.sharpeRatio >= 2.0) return "speculative";
  return "basis";
}

function determineUrgency(signal: SignalData): "high" | "medium" | "low" {
  if (signal.confidence >= 80 && signal.sharpeRatio >= 2.5) return "high";
  if (signal.confidence >= 65 || signal.sharpeRatio >= 2.0) return "medium";
  return "low";
}

function calculateHedgeScore(
  dollarPerTrade: number,
  winRate: number,
  horizonCoverage: number,
  signal: SignalData
): number {
  // Weighted scoring for hedging purposes:
  // 30% $/trade, 30% win rate, 20% horizon coverage, 20% confidence
  const normalizedDpt = Math.min(Math.max(dollarPerTrade * 10, 0), 30);
  const winRateScore = (winRate / 100) * 30;
  const horizonScore = (horizonCoverage / 100) * 20;
  const confidenceScore = (signal.confidence / 100) * 20;

  return normalizedDpt + winRateScore + horizonScore + confidenceScore;
}

function getHedgeOpportunities(): HedgeOpportunity[] {
  const opportunities: HedgeOpportunity[] = [];
  const horizons: Horizon[] = ["D+1", "D+5", "D+10"];

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    // Find the best horizon for hedging (prioritize higher win rate and confidence)
    let bestSignal: SignalData | null = null;
    let bestHorizon: Horizon = "D+1";

    for (const horizon of horizons) {
      const signal = MOCK_SIGNALS[assetId as AssetId]?.[horizon];
      if (signal && signal.direction !== "neutral") {
        if (!bestSignal ||
            (signal.directionalAccuracy > bestSignal.directionalAccuracy &&
             signal.confidence >= bestSignal.confidence - 5)) {
          bestSignal = signal;
          bestHorizon = horizon;
        }
      }
    }

    if (bestSignal && bestSignal.directionalAccuracy >= 52) {
      const dollarPerTrade = calculateDollarPerTrade(bestSignal, asset.currentPrice);
      const winRate = bestSignal.directionalAccuracy;
      const horizonCoverage = calculateHorizonCoverage(assetId as AssetId);

      opportunities.push({
        assetId: assetId as AssetId,
        assetName: asset.name,
        symbol: asset.symbol,
        currentPrice: asset.currentPrice,
        signal: { ...bestSignal, horizon: bestHorizon },
        dollarPerTrade,
        winRate,
        horizonCoverage,
        hedgeScore: calculateHedgeScore(dollarPerTrade, winRate, horizonCoverage, bestSignal),
        hedgeType: determineHedgeType(bestSignal),
        urgency: determineUrgency(bestSignal),
        rank: 0,
      });
    }
  });

  // Sort by hedge score and assign ranks
  return opportunities
    .sort((a, b) => b.hedgeScore - a.hedgeScore)
    .map((o, i) => ({ ...o, rank: i + 1 }));
}

// ============================================================================
// Header Component
// ============================================================================

function HedgingHeader() {
  return (
    <div className="bg-gradient-to-r from-emerald-900/30 via-teal-900/20 to-emerald-900/30 border border-emerald-500/20 rounded-xl p-6">
      <div className="flex items-center gap-4 mb-4">
        <div className="p-3 bg-emerald-500/20 rounded-xl border border-emerald-500/30">
          <Shield className="w-8 h-8 text-emerald-400" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-neutral-100">Hedging Dashboard</h1>
          <p className="text-sm text-neutral-400">
            Actionable hedge recommendations with practical metrics
          </p>
        </div>
      </div>
      <div className="flex flex-wrap gap-3">
        <Badge className="bg-emerald-500/10 border-emerald-500/30 text-emerald-300 px-3 py-1.5">
          <DollarSign className="w-3.5 h-3.5 mr-1.5" />
          $/Trade Focus
        </Badge>
        <Badge className="bg-teal-500/10 border-teal-500/30 text-teal-300 px-3 py-1.5">
          <Layers className="w-3.5 h-3.5 mr-1.5" />
          Multi-Horizon
        </Badge>
        <Badge className="bg-cyan-500/10 border-cyan-500/30 text-cyan-300 px-3 py-1.5">
          <Scale className="w-3.5 h-3.5 mr-1.5" />
          Risk Balanced
        </Badge>
      </div>
    </div>
  );
}

// ============================================================================
// Hedging Summary Stats
// ============================================================================

interface HedgingStatsProps {
  opportunities: HedgeOpportunity[];
}

function HedgingStats({ opportunities }: HedgingStatsProps) {
  const stats = useMemo(() => {
    const avgWinRate = opportunities.reduce((acc, o) => acc + o.winRate, 0) / opportunities.length;
    const avgDpt = opportunities.reduce((acc, o) => acc + o.dollarPerTrade, 0) / opportunities.length;
    const avgHorizonCoverage = opportunities.reduce((acc, o) => acc + o.horizonCoverage, 0) / opportunities.length;
    const highUrgency = opportunities.filter(o => o.urgency === "high").length;
    const topOpp = opportunities[0];

    return { avgWinRate, avgDpt, avgHorizonCoverage, highUrgency, topOpp };
  }, [opportunities]);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Average Win Rate */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <Percent className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wider">Avg Win Rate</span>
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-400">
            {stats.avgWinRate.toFixed(1)}%
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Across all hedges
          </div>
        </CardContent>
      </Card>

      {/* Average $/Trade */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <DollarSign className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wider">Avg $/Trade</span>
          </div>
          <div className={cn(
            "text-2xl font-bold font-mono",
            stats.avgDpt >= 0 ? "text-green-400" : "text-red-400"
          )}>
            {stats.avgDpt >= 0 ? "+" : ""}{stats.avgDpt.toFixed(2)}
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Expected per contract
          </div>
        </CardContent>
      </Card>

      {/* Horizon Coverage */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <Layers className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wider">Horizon Diversity</span>
          </div>
          <div className="text-2xl font-bold font-mono text-teal-400">
            {stats.avgHorizonCoverage.toFixed(0)}%
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Signal consistency
          </div>
        </CardContent>
      </Card>

      {/* Urgent Hedges */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span className="text-xs uppercase tracking-wider">Action Required</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold font-mono text-amber-400">
              {stats.highUrgency}
            </span>
            <Badge className="bg-amber-500/10 border-amber-500/30 text-amber-400 text-xs">
              High Priority
            </Badge>
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Hedges need attention
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ============================================================================
// Hedge Recommendation Card
// ============================================================================

interface HedgeRecommendationProps {
  opportunity: HedgeOpportunity;
}

function HedgeRecommendationCard({ opportunity }: HedgeRecommendationProps) {
  const isBullish = opportunity.signal.direction === "bullish";
  const directionColor = isBullish ? "text-green-500" : "text-red-500";

  const urgencyConfig = {
    high: { bg: "bg-amber-500/10", border: "border-amber-500/30", text: "text-amber-400", label: "ACT NOW" },
    medium: { bg: "bg-blue-500/10", border: "border-blue-500/30", text: "text-blue-400", label: "MONITOR" },
    low: { bg: "bg-neutral-500/10", border: "border-neutral-500/30", text: "text-neutral-400", label: "WATCH" },
  };

  const hedgeTypeConfig = {
    protective: { icon: Shield, label: "Protective Hedge", color: "text-emerald-400" },
    speculative: { icon: Target, label: "Speculative", color: "text-purple-400" },
    basis: { icon: Scale, label: "Basis Trade", color: "text-cyan-400" },
  };

  const urgency = urgencyConfig[opportunity.urgency];
  const hedgeType = hedgeTypeConfig[opportunity.hedgeType];
  const HedgeIcon = hedgeType.icon;

  return (
    <Card className={cn(
      "bg-neutral-900/50 border-neutral-800 hover:border-neutral-700 transition-all",
      opportunity.rank <= 3 && "ring-1 ring-emerald-500/30"
    )}>
      <CardContent className="p-5">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center font-bold text-sm",
              opportunity.rank === 1 ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30" :
              opportunity.rank === 2 ? "bg-teal-500/20 text-teal-400 border border-teal-500/30" :
              opportunity.rank === 3 ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30" :
              "bg-neutral-800 text-neutral-500"
            )}>
              #{opportunity.rank}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-neutral-500 bg-neutral-800 px-2 py-0.5 rounded">
                  {opportunity.symbol}
                </span>
                <span className="font-semibold text-neutral-100">{opportunity.assetName}</span>
              </div>
              <div className="flex items-center gap-2 mt-0.5">
                <HedgeIcon className={cn("w-3 h-3", hedgeType.color)} />
                <span className={cn("text-xs", hedgeType.color)}>{hedgeType.label}</span>
              </div>
            </div>
          </div>
          <Badge className={cn("text-xs px-2 py-1", urgency.bg, urgency.border, urgency.text)}>
            {urgency.label}
          </Badge>
        </div>

        {/* Direction & Horizon */}
        <div className="flex items-center justify-between mb-4 p-3 bg-neutral-800/50 rounded-lg">
          <div className="flex items-center gap-2">
            {isBullish ? (
              <TrendingUp className={cn("w-5 h-5", directionColor)} />
            ) : (
              <TrendingDown className={cn("w-5 h-5", directionColor)} />
            )}
            <span className={cn("font-semibold", directionColor)}>
              {opportunity.signal.direction.toUpperCase()}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-neutral-500" />
            <span className="text-sm text-neutral-400">{opportunity.signal.horizon}</span>
          </div>
        </div>

        {/* Key Metrics - Hedging Focus */}
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="p-2 bg-neutral-800/50 rounded text-center">
            <div className="text-[10px] text-neutral-500 uppercase tracking-wider mb-0.5">$/Trade</div>
            <div className={cn(
              "font-mono font-bold text-lg",
              opportunity.dollarPerTrade >= 0 ? "text-green-400" : "text-red-400"
            )}>
              {opportunity.dollarPerTrade >= 0 ? "+" : ""}{opportunity.dollarPerTrade.toFixed(2)}
            </div>
          </div>
          <div className="p-2 bg-neutral-800/50 rounded text-center">
            <div className="text-[10px] text-neutral-500 uppercase tracking-wider mb-0.5">Win Rate</div>
            <div className={cn(
              "font-mono font-bold text-lg",
              opportunity.winRate >= 55 ? "text-emerald-400" :
              opportunity.winRate >= 52 ? "text-yellow-400" : "text-neutral-400"
            )}>
              {opportunity.winRate.toFixed(1)}%
            </div>
          </div>
          <div className="p-2 bg-neutral-800/50 rounded text-center">
            <div className="text-[10px] text-neutral-500 uppercase tracking-wider mb-0.5">Confidence</div>
            <div className={cn(
              "font-mono font-bold text-lg",
              opportunity.signal.confidence >= 75 ? "text-cyan-400" :
              opportunity.signal.confidence >= 60 ? "text-blue-400" : "text-neutral-400"
            )}>
              {opportunity.signal.confidence}%
            </div>
          </div>
        </div>

        {/* Horizon Diversity Bar */}
        <div className="mb-4">
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="text-neutral-500 flex items-center gap-1">
              <Layers className="w-3 h-3" />
              Horizon Consistency
            </span>
            <span className="font-mono text-teal-400">{opportunity.horizonCoverage}%</span>
          </div>
          <div className="h-2 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all",
                opportunity.horizonCoverage >= 66 ? "bg-gradient-to-r from-emerald-500 to-teal-500" :
                opportunity.horizonCoverage >= 33 ? "bg-gradient-to-r from-yellow-500 to-amber-500" :
                "bg-gradient-to-r from-red-500 to-orange-500"
              )}
              style={{ width: `${opportunity.horizonCoverage}%` }}
            />
          </div>
        </div>

        {/* Hedge Score */}
        <div className="pt-3 border-t border-neutral-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-neutral-500">Hedge Score</span>
            <div className="flex items-center gap-2">
              <div className="flex gap-0.5">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div
                    key={i}
                    className={cn(
                      "w-2 h-4 rounded-sm",
                      i <= Math.ceil(opportunity.hedgeScore / 20)
                        ? "bg-emerald-500"
                        : "bg-neutral-700"
                    )}
                  />
                ))}
              </div>
              <span className="font-mono font-bold text-emerald-400">
                {opportunity.hedgeScore.toFixed(0)}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Action Items Panel
// ============================================================================

interface ActionItemsProps {
  opportunities: HedgeOpportunity[];
}

function ActionItemsPanel({ opportunities }: ActionItemsProps) {
  const urgentOpps = opportunities.filter(o => o.urgency === "high");
  const protectiveOpps = opportunities.filter(o => o.hedgeType === "protective");
  const highWinRate = opportunities.filter(o => o.winRate >= 56);

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-emerald-400" />
          <span className="text-sm font-semibold text-neutral-100">Actionable Summary</span>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        {/* Urgent Actions */}
        <div className="p-3 bg-amber-500/5 border border-amber-500/20 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span className="text-sm font-medium text-amber-400">Urgent Actions ({urgentOpps.length})</span>
          </div>
          {urgentOpps.length > 0 ? (
            <ul className="space-y-1">
              {urgentOpps.slice(0, 3).map((opp) => (
                <li key={opp.assetId} className="flex items-center gap-2 text-xs">
                  <span className="text-amber-400">•</span>
                  <span className="text-neutral-300">
                    {opp.signal.direction === "bearish" ? "Short" : "Long"} {opp.symbol} hedge
                    <span className="text-neutral-500"> ({opp.signal.horizon})</span>
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-neutral-500">No urgent hedges required</p>
          )}
        </div>

        {/* Protective Hedges */}
        <div className="p-3 bg-emerald-500/5 border border-emerald-500/20 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Shield className="w-4 h-4 text-emerald-400" />
            <span className="text-sm font-medium text-emerald-400">Protective Opportunities ({protectiveOpps.length})</span>
          </div>
          {protectiveOpps.length > 0 ? (
            <ul className="space-y-1">
              {protectiveOpps.slice(0, 3).map((opp) => (
                <li key={opp.assetId} className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-2">
                    <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                    <span className="text-neutral-300">{opp.assetName}</span>
                  </span>
                  <span className="font-mono text-neutral-500">{opp.winRate.toFixed(1)}% win</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-neutral-500">No protective hedges identified</p>
          )}
        </div>

        {/* High Win Rate */}
        <div className="p-3 bg-cyan-500/5 border border-cyan-500/20 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-medium text-cyan-400">High Win Rate (&gt;56%)</span>
          </div>
          {highWinRate.length > 0 ? (
            <ul className="space-y-1">
              {highWinRate.slice(0, 3).map((opp) => (
                <li key={opp.assetId} className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-2">
                    <span className="text-cyan-400">{opp.symbol}</span>
                    <span className="text-neutral-500">{opp.signal.horizon}</span>
                  </span>
                  <span className="font-mono text-green-400">+{opp.dollarPerTrade.toFixed(2)}/trade</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-neutral-500">No high win rate signals</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Hedge Ranking Table
// ============================================================================

interface HedgeRankingTableProps {
  opportunities: HedgeOpportunity[];
}

function HedgeRankingTable({ opportunities }: HedgeRankingTableProps) {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart2 className="w-5 h-5 text-emerald-400" />
            <span className="text-sm font-semibold text-neutral-100">Hedge Ranking</span>
          </div>
          <span className="text-xs text-neutral-500">Sorted by Hedge Score</span>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {/* Table Header */}
        <div className="grid grid-cols-7 gap-2 px-4 py-2 bg-neutral-800/30 text-[10px] uppercase tracking-wider text-neutral-500">
          <div>#</div>
          <div>Asset</div>
          <div className="text-center">Direction</div>
          <div className="text-right">Win Rate</div>
          <div className="text-right">$/Trade</div>
          <div className="text-right">Horizons</div>
          <div className="text-right">Score</div>
        </div>

        <div className="divide-y divide-neutral-800">
          {opportunities.slice(0, 10).map((opp) => (
            <div
              key={opp.assetId}
              className="grid grid-cols-7 gap-2 items-center px-4 py-3 hover:bg-neutral-800/30 transition-colors"
            >
              <span className={cn(
                "font-mono text-sm font-bold",
                opp.rank <= 3 ? "text-emerald-400" : "text-neutral-500"
              )}>
                {opp.rank}
              </span>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
                  {opp.symbol}
                </span>
                <span className="text-sm text-neutral-300 hidden sm:inline">{opp.assetName}</span>
              </div>
              <div className="flex justify-center">
                {opp.signal.direction === "bullish" ? (
                  <ArrowUpRight className="w-4 h-4 text-green-500" />
                ) : (
                  <ArrowDownRight className="w-4 h-4 text-red-500" />
                )}
              </div>
              <span className={cn(
                "text-right font-mono text-sm",
                opp.winRate >= 56 ? "text-emerald-400" :
                opp.winRate >= 54 ? "text-yellow-400" : "text-neutral-400"
              )}>
                {opp.winRate.toFixed(1)}%
              </span>
              <span className={cn(
                "text-right font-mono text-sm",
                opp.dollarPerTrade >= 0 ? "text-green-400" : "text-red-400"
              )}>
                {opp.dollarPerTrade >= 0 ? "+" : ""}{opp.dollarPerTrade.toFixed(2)}
              </span>
              <span className="text-right font-mono text-sm text-teal-400">
                {opp.horizonCoverage}%
              </span>
              <div className="flex items-center justify-end gap-2">
                <div className="w-12 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full"
                    style={{ width: `${Math.min(opp.hedgeScore, 100)}%` }}
                  />
                </div>
                <span className="font-mono text-sm font-bold text-emerald-400 w-8 text-right">
                  {opp.hedgeScore.toFixed(0)}
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
// Main Hedging Dashboard Component
// ============================================================================

export function HedgingDashboard() {
  const opportunities = useMemo(() => getHedgeOpportunities(), []);
  const topOpportunities = opportunities.slice(0, 6);

  return (
    <div className="space-y-6">
      {/* Header */}
      <HedgingHeader />

      {/* Summary Stats */}
      <HedgingStats opportunities={opportunities} />

      <Separator className="bg-neutral-800" />

      {/* Live CME Signals */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-400" />
              Live Hedging Signals
            </h2>
            <ApiHealthIndicator />
          </div>
          <span className="text-xs text-neutral-500 font-mono">CME-focused</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <LiveSignalCard asset="Crude_Oil" displayName="Crude Oil" />
          <LiveSignalCard asset="GOLD" displayName="Gold" />
          <LiveSignalCard asset="Wheat" displayName="Wheat" />
          <LiveSignalCard asset="Corn" displayName="Corn" />
        </div>
      </section>

      <Separator className="bg-neutral-800" />

      {/* Two-Column Layout: Recommendations + Action Items */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Top Hedge Recommendations */}
        <div className="xl:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
              <Shield className="w-5 h-5 text-emerald-400" />
              Top Hedge Recommendations
            </h2>
            <Badge className="bg-emerald-500/10 border-emerald-500/30 text-emerald-300 px-2 py-1 text-xs">
              Win Rate &gt; 52%
            </Badge>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {topOpportunities.map((opp) => (
              <HedgeRecommendationCard key={opp.assetId} opportunity={opp} />
            ))}
          </div>
        </div>

        {/* Action Items Sidebar */}
        <div className="xl:col-span-1">
          <ActionItemsPanel opportunities={opportunities} />
        </div>
      </div>

      <Separator className="bg-neutral-800" />

      {/* Full Ranking Table */}
      <section>
        <HedgeRankingTable opportunities={opportunities} />
      </section>

      {/* Disclaimer */}
      <div className="p-4 bg-neutral-900/30 border border-neutral-800 rounded-lg flex items-start gap-3">
        <Shield className="w-5 h-5 text-neutral-500 flex-shrink-0 mt-0.5" />
        <div className="text-xs text-neutral-500">
          <strong className="text-neutral-400">Hedging Disclaimer:</strong> These recommendations
          are based on ML ensemble signals optimized for win rate and $/trade metrics. Hedging
          decisions should align with your overall risk management policy. Consult with your
          risk team before executing trades.
        </div>
      </div>
    </div>
  );
}
