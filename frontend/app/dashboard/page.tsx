"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { ActionableSummary } from "@/components/dashboard/ActionableSummary";
import { LiveSignalCard } from "@/components/dashboard/LiveSignalCard";
import { ApiHealthIndicator } from "@/components/dashboard/ApiHealthIndicator";
import { MOCK_ASSETS, MOCK_SIGNALS, type Horizon, type SignalData } from "@/lib/mock-data";
import {
  type ActionabilityLevel,
  getActionabilityClasses,
  ASSET_MOVE_THRESHOLDS,
  formatMoveSize,
} from "@/types/practical-metrics";
import type { AssetId } from "@/types";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  Activity,
  BarChart3,
  Filter,
  Zap,
  AlertTriangle,
  Clock,
  Bell,
  ChevronRight,
  Calendar,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

type ActionabilityFilter = "all" | "high" | "medium" | "low";
type DirectionFilter = "all" | "bullish" | "bearish" | "neutral";
type HorizonFilter = "all" | "short" | "medium" | "long";

interface EnrichedSignal {
  assetId: AssetId;
  assetName: string;
  symbol: string;
  currentPrice: number;
  signal: SignalData;
  actionabilityScore: number;
  actionabilityLevel: ActionabilityLevel;
  forecastMove: number;
  horizonCategory: "short" | "medium" | "long";
}

// ============================================================================
// Mock Data Enrichment (simulates API data)
// ============================================================================

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function calculateActionability(signal: SignalData, _assetId: AssetId): { score: number; level: ActionabilityLevel } {
  // Simulate actionability calculation based on confidence, direction strength, etc.
  const confidenceScore = signal.confidence;
  const modelAgreement = (signal.modelsAgreeing / signal.modelsTotal) * 100;
  const sharpeBonus = Math.min(signal.sharpeRatio * 10, 30);

  const score = (confidenceScore * 0.5) + (modelAgreement * 0.3) + sharpeBonus;

  let level: ActionabilityLevel;
  if (score >= 70) level = "high";
  else if (score >= 50) level = "medium";
  else level = "low";

  return { score, level };
}

function getHorizonCategory(horizon: Horizon): "short" | "medium" | "long" {
  if (horizon === "D+1") return "short";
  if (horizon === "D+5") return "medium";
  return "long";
}

function generateForecastMove(assetId: AssetId, signal: SignalData): number {
  // Simulate forecast move based on threshold and confidence
  const threshold = ASSET_MOVE_THRESHOLDS[assetId];
  const multiplier = 0.8 + (signal.confidence / 100) * 1.5;
  const move = threshold * multiplier;
  return signal.direction === "bearish" ? -move : move;
}

// Get all enriched signals across assets and horizons
function getEnrichedSignals(): EnrichedSignal[] {
  const signals: EnrichedSignal[] = [];
  const horizons: Horizon[] = ["D+1", "D+5", "D+10"];

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    horizons.forEach((horizon) => {
      const signal = MOCK_SIGNALS[assetId as AssetId]?.[horizon];
      if (signal) {
        const { score, level } = calculateActionability(signal, assetId as AssetId);
        signals.push({
          assetId: assetId as AssetId,
          assetName: asset.name,
          symbol: asset.symbol,
          currentPrice: asset.currentPrice,
          signal,
          actionabilityScore: score,
          actionabilityLevel: level,
          forecastMove: generateForecastMove(assetId as AssetId, signal),
          horizonCategory: getHorizonCategory(horizon),
        });
      }
    });
  });

  return signals;
}

// ============================================================================
// Morning Briefing Header
// ============================================================================

interface MorningBriefingProps {
  date: string;
  time: string;
  highActionableCount: number;
  urgentSignals: EnrichedSignal[];
}

function MorningBriefing({ date, time, highActionableCount, urgentSignals }: MorningBriefingProps) {
  const marketStatus = useMemo(() => {
    const hour = new Date().getHours();
    if (hour >= 8 && hour < 17) return { text: "Markets Open", color: "text-green-500", dot: "bg-green-500" };
    if (hour >= 6 && hour < 8) return { text: "Pre-Market", color: "text-yellow-500", dot: "bg-yellow-500" };
    return { text: "After Hours", color: "text-neutral-400", dot: "bg-neutral-500" };
  }, []);

  return (
    <div className="bg-gradient-to-r from-neutral-900/80 via-neutral-900/50 to-neutral-900/80 border border-neutral-800 rounded-xl p-5">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        {/* Left: Title & Date */}
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Zap className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-neutral-100">Morning Briefing</h1>
              <div className="flex items-center gap-3 text-sm text-neutral-400">
                <div className="flex items-center gap-1.5">
                  <Calendar className="w-3.5 h-3.5" />
                  <span>{date}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5" />
                  <span>{time}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className={cn("w-2 h-2 rounded-full animate-pulse", marketStatus.dot)} />
                  <span className={marketStatus.color}>{marketStatus.text}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Quick Stats */}
        <div className="flex items-center gap-4">
          {highActionableCount > 0 && (
            <div className="flex items-center gap-2 px-4 py-2 bg-green-500/10 border border-green-500/30 rounded-lg">
              <Bell className="w-4 h-4 text-green-500" />
              <span className="text-green-500 font-semibold">
                {highActionableCount} High-Priority Signal{highActionableCount !== 1 ? "s" : ""}
              </span>
            </div>
          )}
          <div className="text-xs text-neutral-500 font-mono">
            10,179 models · 10 assets
          </div>
        </div>
      </div>

      {/* Urgent Alerts Banner */}
      {urgentSignals.length > 0 && (
        <div className="mt-4 pt-4 border-t border-neutral-800">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="w-4 h-4 text-orange-500" />
            <span className="text-sm font-semibold text-orange-500">Immediate Attention Required</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {urgentSignals.slice(0, 4).map((signal) => (
              <Badge
                key={`${signal.assetId}-${signal.signal.horizon}`}
                className={cn(
                  "px-3 py-1.5 text-sm gap-2 cursor-pointer hover:opacity-80 transition-opacity",
                  signal.signal.direction === "bullish"
                    ? "bg-green-500/10 border-green-500/30 text-green-500"
                    : signal.signal.direction === "bearish"
                    ? "bg-red-500/10 border-red-500/30 text-red-500"
                    : "bg-yellow-500/10 border-yellow-500/30 text-yellow-500"
                )}
              >
                <span className="font-mono">{signal.symbol}</span>
                <span className="font-semibold capitalize">{signal.signal.direction}</span>
                <span className="text-neutral-400">·</span>
                <span className="font-mono">{signal.signal.horizon}</span>
                <ChevronRight className="w-3 h-3" />
              </Badge>
            ))}
            {urgentSignals.length > 4 && (
              <Badge className="bg-neutral-800 border-neutral-700 text-neutral-400 px-3 py-1.5">
                +{urgentSignals.length - 4} more
              </Badge>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Portfolio Overview
// ============================================================================

interface PortfolioOverviewProps {
  signals: EnrichedSignal[];
  totalAssets: number;
  lastUpdated: string;
}

function PortfolioOverview({ signals, totalAssets, lastUpdated }: PortfolioOverviewProps) {
  const highActionable = signals.filter((s) => s.actionabilityLevel === "high").length;
  const mediumActionable = signals.filter((s) => s.actionabilityLevel === "medium").length;
  const attentionNeeded = signals.filter((s) =>
    s.actionabilityLevel === "high" || (s.actionabilityLevel === "medium" && s.signal.confidence >= 70)
  ).length;

  // Overall status based on distribution
  let overallStatus: ActionabilityLevel = "low";
  if (highActionable >= 5) overallStatus = "high";
  else if (highActionable >= 2 || mediumActionable >= 5) overallStatus = "medium";

  const statusClasses = getActionabilityClasses(overallStatus);

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            <span className="text-sm font-semibold text-neutral-100">Portfolio Overview</span>
          </div>
          <div className="flex items-center gap-2">
            <div className={cn("w-3 h-3 rounded-full animate-pulse", statusClasses.dot)} />
            <span className={cn("text-xs font-medium", statusClasses.text)}>
              {overallStatus === "high" ? "Strong Opportunities" : overallStatus === "medium" ? "Mixed Signals" : "Monitoring Mode"}
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Total Assets */}
          <div className="p-3 bg-neutral-800/50 rounded-lg">
            <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">Assets Monitored</div>
            <div className="text-2xl font-bold font-mono text-neutral-100">{totalAssets}</div>
            <div className="text-[10px] text-neutral-600">across all categories</div>
          </div>

          {/* High Actionable */}
          <div className="p-3 bg-green-500/5 border border-green-500/20 rounded-lg">
            <div className="text-[10px] uppercase tracking-wider text-green-500/70 mb-1">High Actionable</div>
            <div className="text-2xl font-bold font-mono text-green-500">{highActionable}</div>
            <div className="text-[10px] text-neutral-600">signals to act on</div>
          </div>

          {/* Attention Needed */}
          <div className="p-3 bg-orange-500/5 border border-orange-500/20 rounded-lg">
            <div className="text-[10px] uppercase tracking-wider text-orange-500/70 mb-1">Need Attention</div>
            <div className="text-2xl font-bold font-mono text-orange-500">{attentionNeeded}</div>
            <div className="text-[10px] text-neutral-600">review today</div>
          </div>

          {/* Last Update */}
          <div className="p-3 bg-neutral-800/50 rounded-lg">
            <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">Last Update</div>
            <div className="text-lg font-bold font-mono text-neutral-100">{lastUpdated}</div>
            <div className="text-[10px] text-neutral-600">market data</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Signal Grid Card
// ============================================================================

interface SignalGridCardProps {
  signal: EnrichedSignal;
  onClick?: () => void;
}

function SignalGridCard({ signal, onClick }: SignalGridCardProps) {
  const actionClasses = getActionabilityClasses(signal.actionabilityLevel);

  const directionIcon = signal.signal.direction === "bullish"
    ? <TrendingUp className="w-4 h-4" />
    : signal.signal.direction === "bearish"
    ? <TrendingDown className="w-4 h-4" />
    : <Minus className="w-4 h-4" />;

  const directionColor = signal.signal.direction === "bullish"
    ? "text-green-500"
    : signal.signal.direction === "bearish"
    ? "text-red-500"
    : "text-yellow-500";

  const directionBg = signal.signal.direction === "bullish"
    ? "bg-green-500/10 border-green-500/30"
    : signal.signal.direction === "bearish"
    ? "bg-red-500/10 border-red-500/30"
    : "bg-yellow-500/10 border-yellow-500/30";

  return (
    <Card
      className={cn(
        "bg-neutral-900/50 border-neutral-800 hover:border-neutral-700 transition-all duration-200 cursor-pointer",
        "hover:bg-neutral-900/80",
        signal.actionabilityLevel === "high" && "ring-1 ring-green-500/20"
      )}
      onClick={onClick}
      data-testid="signal-card"
      data-direction={signal.signal.direction}
    >
      <CardContent className="p-4">
        {/* Header: Asset + Actionability */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
              {signal.symbol}
            </span>
            <span className="text-sm font-semibold text-neutral-100">{signal.assetName}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className={cn("w-2.5 h-2.5 rounded-full", actionClasses.dot)} title={`${signal.actionabilityLevel} actionability`} />
            {signal.actionabilityLevel === "high" && (
              <span className="text-[10px] font-semibold text-green-500 uppercase">ACT</span>
            )}
          </div>
        </div>

        {/* Direction Badge + Horizon */}
        <div className="flex items-center justify-between mb-3">
          <Badge className={cn("text-xs font-semibold px-2 py-0.5 capitalize gap-1", directionBg, directionColor)}>
            {directionIcon}
            {signal.signal.direction}
          </Badge>
          <span className="font-mono text-xs text-neutral-500">{signal.signal.horizon}</span>
        </div>

        {/* Forecast Size */}
        <div className="mb-3 p-2 bg-neutral-800/50 rounded">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-0.5">Forecast Move</div>
          <div className={cn("text-lg font-bold font-mono", directionColor)}>
            {formatMoveSize(signal.forecastMove, signal.assetId)}
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="mb-3">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-neutral-500">Confidence</span>
            <span className="font-mono text-neutral-300">{signal.signal.confidence}%</span>
          </div>
          <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all",
                signal.signal.confidence >= 75 ? "bg-green-500" :
                signal.signal.confidence >= 60 ? "bg-blue-500" :
                signal.signal.confidence >= 45 ? "bg-yellow-500" : "bg-red-500"
              )}
              style={{ width: `${signal.signal.confidence}%` }}
            />
          </div>
        </div>

        {/* Bottom Stats */}
        <div className="flex items-center justify-between text-xs pt-2 border-t border-neutral-800">
          <div className="flex items-center gap-1 text-neutral-500">
            <Target className="w-3 h-3" />
            <span className="font-mono">{signal.signal.sharpeRatio.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-1 text-neutral-500">
            <BarChart3 className="w-3 h-3" />
            <span className="font-mono">{signal.signal.directionalAccuracy.toFixed(1)}%</span>
          </div>
          <Badge
            className={cn(
              "text-[10px] px-1.5 py-0",
              actionClasses.bg,
              actionClasses.border,
              actionClasses.text
            )}
          >
            {signal.actionabilityScore.toFixed(0)}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Market Context Sidebar
// ============================================================================

interface MarketContextProps {
  signals: EnrichedSignal[];
}

function MarketContext({ signals }: MarketContextProps) {
  const bullish = signals.filter((s) => s.signal.direction === "bullish").length;
  const bearish = signals.filter((s) => s.signal.direction === "bearish").length;
  const neutral = signals.filter((s) => s.signal.direction === "neutral").length;
  const total = signals.length;

  const shortTerm = signals.filter((s) => s.horizonCategory === "short").length;
  const mediumTerm = signals.filter((s) => s.horizonCategory === "medium").length;
  const longTerm = signals.filter((s) => s.horizonCategory === "long").length;

  // Calculate net exposure
  const bullishHigh = signals.filter((s) => s.signal.direction === "bullish" && s.actionabilityLevel === "high").length;
  const bearishHigh = signals.filter((s) => s.signal.direction === "bearish" && s.actionabilityLevel === "high").length;
  const netExposure = bullishHigh - bearishHigh;

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-cyan-400" />
          <span className="text-sm font-semibold text-neutral-100">Market Context</span>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        {/* Direction Distribution */}
        <div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2">Signal Direction</div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-3 h-3 text-green-500" />
              <span className="text-xs text-neutral-400 w-16">Bullish</span>
              <div className="flex-1 h-2 bg-neutral-800 rounded-full overflow-hidden">
                <div className="h-full bg-green-500 rounded-full" style={{ width: `${(bullish / total) * 100}%` }} />
              </div>
              <span className="text-xs font-mono text-neutral-300 w-8">{bullish}</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingDown className="w-3 h-3 text-red-500" />
              <span className="text-xs text-neutral-400 w-16">Bearish</span>
              <div className="flex-1 h-2 bg-neutral-800 rounded-full overflow-hidden">
                <div className="h-full bg-red-500 rounded-full" style={{ width: `${(bearish / total) * 100}%` }} />
              </div>
              <span className="text-xs font-mono text-neutral-300 w-8">{bearish}</span>
            </div>
            <div className="flex items-center gap-2">
              <Minus className="w-3 h-3 text-yellow-500" />
              <span className="text-xs text-neutral-400 w-16">Neutral</span>
              <div className="flex-1 h-2 bg-neutral-800 rounded-full overflow-hidden">
                <div className="h-full bg-yellow-500 rounded-full" style={{ width: `${(neutral / total) * 100}%` }} />
              </div>
              <span className="text-xs font-mono text-neutral-300 w-8">{neutral}</span>
            </div>
          </div>
        </div>

        <Separator className="bg-neutral-800" />

        {/* Horizon Distribution */}
        <div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2">Horizon Clustering</div>
          <div className="grid grid-cols-3 gap-2">
            <div className="p-2 bg-neutral-800/50 rounded text-center">
              <div className="text-lg font-bold font-mono text-blue-400">{shortTerm}</div>
              <div className="text-[10px] text-neutral-500">D+1</div>
            </div>
            <div className="p-2 bg-neutral-800/50 rounded text-center">
              <div className="text-lg font-bold font-mono text-cyan-400">{mediumTerm}</div>
              <div className="text-[10px] text-neutral-500">D+5</div>
            </div>
            <div className="p-2 bg-neutral-800/50 rounded text-center">
              <div className="text-lg font-bold font-mono text-purple-400">{longTerm}</div>
              <div className="text-[10px] text-neutral-500">D+10</div>
            </div>
          </div>
        </div>

        <Separator className="bg-neutral-800" />

        {/* Net Exposure */}
        <div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2">Net Directional Bias</div>
          <div className="p-3 bg-neutral-800/30 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-xs text-neutral-400">High-actionable signals</span>
              <span className={cn(
                "font-mono font-bold",
                netExposure > 0 ? "text-green-500" : netExposure < 0 ? "text-red-500" : "text-yellow-500"
              )}>
                {netExposure > 0 ? "+" : ""}{netExposure} net
              </span>
            </div>
            <div className="text-xs text-neutral-500 mt-1">
              {netExposure > 0 ? `${bullishHigh} bullish vs ${bearishHigh} bearish` :
               netExposure < 0 ? `${bearishHigh} bearish vs ${bullishHigh} bullish` :
               "Balanced exposure"}
            </div>
          </div>
        </div>

        {/* Risk Note */}
        <div className="p-2 bg-neutral-800/20 rounded border border-neutral-800 text-[10px] text-neutral-500">
          <strong className="text-neutral-400">Portfolio Exposure:</strong> Based on high-actionability signals only.
          Consider position sizing based on individual signal scores.
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Filter Bar with Quick Filters
// ============================================================================

interface FilterBarProps {
  actionabilityFilter: ActionabilityFilter;
  directionFilter: DirectionFilter;
  horizonFilter: HorizonFilter;
  onActionabilityChange: (value: ActionabilityFilter) => void;
  onDirectionChange: (value: DirectionFilter) => void;
  onHorizonChange: (value: HorizonFilter) => void;
  filteredCount: number;
  totalCount: number;
}

function FilterBar({
  actionabilityFilter,
  directionFilter,
  horizonFilter,
  onActionabilityChange,
  onDirectionChange,
  onHorizonChange,
  filteredCount,
  totalCount,
}: FilterBarProps) {
  // Quick filter presets
  const applyQuickFilter = (preset: "morning" | "bullish" | "bearish" | "all") => {
    switch (preset) {
      case "morning":
        onActionabilityChange("high");
        onDirectionChange("all");
        onHorizonChange("short");
        break;
      case "bullish":
        onActionabilityChange("all");
        onDirectionChange("bullish");
        onHorizonChange("all");
        break;
      case "bearish":
        onActionabilityChange("all");
        onDirectionChange("bearish");
        onHorizonChange("all");
        break;
      case "all":
        onActionabilityChange("all");
        onDirectionChange("all");
        onHorizonChange("all");
        break;
    }
  };

  const isQuickFilterActive = (preset: "morning" | "bullish" | "bearish" | "all"): boolean => {
    switch (preset) {
      case "morning":
        return actionabilityFilter === "high" && directionFilter === "all" && horizonFilter === "short";
      case "bullish":
        return actionabilityFilter === "all" && directionFilter === "bullish" && horizonFilter === "all";
      case "bearish":
        return actionabilityFilter === "all" && directionFilter === "bearish" && horizonFilter === "all";
      case "all":
        return actionabilityFilter === "all" && directionFilter === "all" && horizonFilter === "all";
    }
  };

  return (
    <div className="space-y-3">
      {/* Quick Filter Buttons */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-neutral-500 mr-1">Quick:</span>
        <Button
          size="sm"
          variant={isQuickFilterActive("morning") ? "default" : "outline"}
          className={cn(
            "h-7 text-xs gap-1.5",
            isQuickFilterActive("morning")
              ? "bg-blue-500/20 border-blue-500/40 text-blue-400 hover:bg-blue-500/30"
              : "bg-neutral-800/50 border-neutral-700 text-neutral-400 hover:bg-neutral-800"
          )}
          onClick={() => applyQuickFilter("morning")}
        >
          <Clock className="w-3 h-3" />
          Morning Focus
        </Button>
        <Button
          size="sm"
          variant={isQuickFilterActive("bullish") ? "default" : "outline"}
          className={cn(
            "h-7 text-xs gap-1.5",
            isQuickFilterActive("bullish")
              ? "bg-green-500/20 border-green-500/40 text-green-400 hover:bg-green-500/30"
              : "bg-neutral-800/50 border-neutral-700 text-neutral-400 hover:bg-neutral-800"
          )}
          onClick={() => applyQuickFilter("bullish")}
        >
          <TrendingUp className="w-3 h-3" />
          Bullish Only
        </Button>
        <Button
          size="sm"
          variant={isQuickFilterActive("bearish") ? "default" : "outline"}
          className={cn(
            "h-7 text-xs gap-1.5",
            isQuickFilterActive("bearish")
              ? "bg-red-500/20 border-red-500/40 text-red-400 hover:bg-red-500/30"
              : "bg-neutral-800/50 border-neutral-700 text-neutral-400 hover:bg-neutral-800"
          )}
          onClick={() => applyQuickFilter("bearish")}
        >
          <TrendingDown className="w-3 h-3" />
          Bearish Only
        </Button>
        <Button
          size="sm"
          variant={isQuickFilterActive("all") ? "default" : "outline"}
          className={cn(
            "h-7 text-xs",
            isQuickFilterActive("all")
              ? "bg-neutral-700 border-neutral-600 text-neutral-200"
              : "bg-neutral-800/50 border-neutral-700 text-neutral-400 hover:bg-neutral-800"
          )}
          onClick={() => applyQuickFilter("all")}
        >
          Show All
        </Button>
      </div>

      {/* Advanced Filters */}
      <div className="flex flex-wrap items-center gap-4 p-3 bg-neutral-900/30 border border-neutral-800 rounded-lg">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-neutral-500" />
          <span className="text-xs text-neutral-500">Filters</span>
        </div>

        {/* Actionability Filter */}
        <Select value={actionabilityFilter} onValueChange={(v) => onActionabilityChange(v as ActionabilityFilter)}>
          <SelectTrigger className="w-[130px] h-8 text-xs bg-neutral-800 border-neutral-700">
            <SelectValue placeholder="Actionability" />
          </SelectTrigger>
          <SelectContent className="bg-neutral-900 border-neutral-700">
            <SelectItem value="all" className="text-xs">All Levels</SelectItem>
            <SelectItem value="high" className="text-xs">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-green-500" />
                High
              </span>
            </SelectItem>
            <SelectItem value="medium" className="text-xs">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-yellow-500" />
                Medium
              </span>
            </SelectItem>
            <SelectItem value="low" className="text-xs">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-red-500" />
                Low
              </span>
            </SelectItem>
          </SelectContent>
        </Select>

        {/* Direction Filter */}
        <Select value={directionFilter} onValueChange={(v) => onDirectionChange(v as DirectionFilter)}>
          <SelectTrigger className="w-[120px] h-8 text-xs bg-neutral-800 border-neutral-700">
            <SelectValue placeholder="Direction" />
          </SelectTrigger>
          <SelectContent className="bg-neutral-900 border-neutral-700">
            <SelectItem value="all" className="text-xs">All Directions</SelectItem>
            <SelectItem value="bullish" className="text-xs">
              <span className="flex items-center gap-1.5 text-green-500">
                <TrendingUp className="w-3 h-3" />
                Bullish
              </span>
            </SelectItem>
            <SelectItem value="bearish" className="text-xs">
              <span className="flex items-center gap-1.5 text-red-500">
                <TrendingDown className="w-3 h-3" />
                Bearish
              </span>
            </SelectItem>
            <SelectItem value="neutral" className="text-xs">
              <span className="flex items-center gap-1.5 text-yellow-500">
                <Minus className="w-3 h-3" />
                Neutral
              </span>
            </SelectItem>
          </SelectContent>
        </Select>

        {/* Horizon Filter */}
        <Select value={horizonFilter} onValueChange={(v) => onHorizonChange(v as HorizonFilter)}>
          <SelectTrigger className="w-[130px] h-8 text-xs bg-neutral-800 border-neutral-700">
            <SelectValue placeholder="Horizon" />
          </SelectTrigger>
          <SelectContent className="bg-neutral-900 border-neutral-700">
            <SelectItem value="all" className="text-xs">All Horizons</SelectItem>
            <SelectItem value="short" className="text-xs">Short (D+1)</SelectItem>
            <SelectItem value="medium" className="text-xs">Medium (D+5)</SelectItem>
            <SelectItem value="long" className="text-xs">Long (D+10)</SelectItem>
          </SelectContent>
        </Select>

        {/* Results count */}
        <div className="ml-auto text-xs text-neutral-500">
          Showing <span className="font-mono text-neutral-300">{filteredCount}</span> of{" "}
          <span className="font-mono">{totalCount}</span> signals
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Main Dashboard Component
// ============================================================================

export default function DashboardPage() {
  const router = useRouter();

  // Filter state
  const [actionabilityFilter, setActionabilityFilter] = useState<ActionabilityFilter>("all");
  const [directionFilter, setDirectionFilter] = useState<DirectionFilter>("all");
  const [horizonFilter, setHorizonFilter] = useState<HorizonFilter>("all");

  // Get all enriched signals
  const allSignals = useMemo(() => getEnrichedSignals(), []);

  // Filter signals
  const filteredSignals = useMemo(() => {
    return allSignals
      .filter((s) => {
        if (actionabilityFilter !== "all" && s.actionabilityLevel !== actionabilityFilter) return false;
        if (directionFilter !== "all" && s.signal.direction !== directionFilter) return false;
        if (horizonFilter !== "all" && s.horizonCategory !== horizonFilter) return false;
        return true;
      })
      .sort((a, b) => b.actionabilityScore - a.actionabilityScore);
  }, [allSignals, actionabilityFilter, directionFilter, horizonFilter]);

  // Signals for main grid (High + Medium by default when no filter)
  const displayedSignals = useMemo(() => {
    if (actionabilityFilter === "all") {
      // Default: show only High and Medium
      return filteredSignals.filter((s) => s.actionabilityLevel !== "low");
    }
    return filteredSignals;
  }, [filteredSignals, actionabilityFilter]);

  // Calculate summary stats
  const bullishCount = allSignals.filter((s) => s.signal.direction === "bullish").length;
  const bearishCount = allSignals.filter((s) => s.signal.direction === "bearish").length;
  const neutralCount = allSignals.filter((s) => s.signal.direction === "neutral").length;
  const total = allSignals.length;

  const assetsNeedingAttention = new Set(
    allSignals
      .filter((s) => s.actionabilityLevel === "high")
      .map((s) => s.assetId)
  ).size;

  const actionableToday = allSignals.filter((s) =>
    (s.actionabilityLevel === "high" || s.actionabilityLevel === "medium") &&
    s.horizonCategory === "short"
  ).length;

  // Urgent signals: High actionability + short horizon
  const urgentSignals = useMemo(() =>
    allSignals
      .filter((s) => s.actionabilityLevel === "high" && s.horizonCategory === "short")
      .sort((a, b) => b.actionabilityScore - a.actionabilityScore),
    [allSignals]
  );

  // Overall status
  const highCount = allSignals.filter((s) => s.actionabilityLevel === "high").length;
  let overallStatus: ActionabilityLevel = "low";
  if (highCount >= 5) overallStatus = "high";
  else if (highCount >= 2) overallStatus = "medium";

  const now = new Date();
  const dateStr = now.toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });
  const timeStr = now.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: true
  });

  const handleSignalClick = (signal: EnrichedSignal) => {
    router.push(`/dashboard/assets/${signal.assetId}`);
  };

  return (
    <div className="space-y-6">
      {/* SECTION 1: Morning Briefing Header */}
      <MorningBriefing
        date={dateStr}
        time={timeStr}
        highActionableCount={highCount}
        urgentSignals={urgentSignals}
      />

      {/* SECTION 2: Live Backend Signals */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
              <Activity className="w-5 h-5 text-green-400" />
              Live Signals
            </h2>
            <ApiHealthIndicator />
          </div>
          <span className="text-xs text-neutral-500 font-mono">Backend API @ localhost:5001</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <LiveSignalCard asset="Crude_Oil" displayName="Crude Oil" />
          <LiveSignalCard asset="Bitcoin" displayName="Bitcoin" />
          <LiveSignalCard asset="SP500" displayName="S&P 500" />
          <LiveSignalCard asset="GOLD" displayName="Gold" />
        </div>
      </section>

      <Separator className="bg-neutral-800" />

      {/* SECTION 3: Portfolio Overview */}
      <section>
        <PortfolioOverview
          signals={allSignals}
          totalAssets={Object.keys(MOCK_ASSETS).length}
          lastUpdated={timeStr}
        />
      </section>

      {/* Actionable Summary Bar */}
      <ActionableSummary
        assetsNeedingAttention={assetsNeedingAttention}
        actionableSignalsToday={actionableToday}
        bullishPercent={(bullishCount / total) * 100}
        bearishPercent={(bearishCount / total) * 100}
        neutralPercent={(neutralCount / total) * 100}
        overallStatus={overallStatus}
        lastUpdated={timeStr}
      />

      <Separator className="bg-neutral-800" />

      {/* SECTION 3: Actionable Signals (Main Area) + SECTION 4: Market Context (Sidebar) */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main Signals Grid */}
        <div className="lg:col-span-3 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-400" />
              Actionable Signals
            </h2>
            <span className="text-xs text-neutral-500">Sorted by actionability score</span>
          </div>

          {/* Filter Bar */}
          <FilterBar
            actionabilityFilter={actionabilityFilter}
            directionFilter={directionFilter}
            horizonFilter={horizonFilter}
            onActionabilityChange={setActionabilityFilter}
            onDirectionChange={setDirectionFilter}
            onHorizonChange={setHorizonFilter}
            filteredCount={displayedSignals.length}
            totalCount={allSignals.length}
          />

          {/* Signal Cards Grid */}
          {displayedSignals.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {displayedSignals.map((signal) => (
                <SignalGridCard
                  key={`${signal.assetId}-${signal.signal.horizon}`}
                  signal={signal}
                  onClick={() => handleSignalClick(signal)}
                />
              ))}
            </div>
          ) : (
            <Card className="bg-neutral-900/50 border-neutral-800">
              <CardContent className="p-8 text-center">
                <Filter className="w-8 h-8 text-neutral-600 mx-auto mb-3" />
                <p className="text-neutral-400">No signals match current filters</p>
                <p className="text-xs text-neutral-600 mt-1">Try adjusting the filter criteria</p>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Market Context Sidebar */}
        <div className="lg:col-span-1">
          <MarketContext signals={allSignals} />
        </div>
      </div>
    </div>
  );
}
