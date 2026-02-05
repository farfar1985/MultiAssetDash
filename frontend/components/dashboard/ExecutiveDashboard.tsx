"use client";

import { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { MOCK_ASSETS, MOCK_SIGNALS, type Horizon, type SignalData } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Briefcase,
  DollarSign,
  Target,
  Activity,
  CheckCircle,
  AlertCircle,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface MarketSummary {
  bullishCount: number;
  bearishCount: number;
  neutralCount: number;
  topOpportunity: { name: string; direction: "bullish" | "bearish" | "neutral"; confidence: number } | null;
  overallSentiment: "bullish" | "bearish" | "mixed";
  avgConfidence: number;
}

// ============================================================================
// Helper Functions
// ============================================================================

function getMarketSummary(): MarketSummary {
  let bullishCount = 0;
  let bearishCount = 0;
  let neutralCount = 0;
  let totalConfidence = 0;
  let signalCount = 0;
  let topSignal: { name: string; direction: SignalData["direction"]; confidence: number } | null = null;

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    const signal = MOCK_SIGNALS[assetId as AssetId]?.["D+1"];
    if (signal) {
      signalCount++;
      totalConfidence += signal.confidence;

      if (signal.direction === "bullish") bullishCount++;
      else if (signal.direction === "bearish") bearishCount++;
      else neutralCount++;

      if (!topSignal || signal.confidence > topSignal.confidence) {
        topSignal = { name: asset.name, direction: signal.direction, confidence: signal.confidence };
      }
    }
  });

  const overallSentiment: MarketSummary["overallSentiment"] =
    bullishCount > bearishCount + 2 ? "bullish" :
    bearishCount > bullishCount + 2 ? "bearish" : "mixed";

  return {
    bullishCount,
    bearishCount,
    neutralCount,
    topOpportunity: topSignal,
    overallSentiment,
    avgConfidence: signalCount > 0 ? totalConfidence / signalCount : 0,
  };
}

function generateOutlookText(summary: MarketSummary): string {
  const { overallSentiment, bullishCount, bearishCount, avgConfidence, topOpportunity } = summary;

  if (overallSentiment === "bullish") {
    return `Markets are showing positive momentum with ${bullishCount} assets signaling upward movement. Our AI models have ${avgConfidence.toFixed(0)}% average confidence in current predictions. ${topOpportunity ? `${topOpportunity.name} presents the strongest opportunity today.` : ""} Consider maintaining or increasing exposure to growth positions.`;
  }

  if (overallSentiment === "bearish") {
    return `Markets are showing cautionary signals with ${bearishCount} assets trending downward. Our AI models have ${avgConfidence.toFixed(0)}% average confidence. ${topOpportunity ? `Watch ${topOpportunity.name} closely for potential movements.` : ""} Consider reviewing risk exposure and hedging strategies.`;
  }

  return `Markets are showing mixed signals with ${bullishCount} bullish and ${bearishCount} bearish indicators. Our AI models have ${avgConfidence.toFixed(0)}% average confidence. ${topOpportunity ? `${topOpportunity.name} shows the clearest direction currently.` : ""} Selective positioning recommended.`;
}

// ============================================================================
// Headline Metric Card
// ============================================================================

interface HeadlineMetricProps {
  label: string;
  value: string;
  subtext?: string;
  trend?: "up" | "down" | "neutral";
  icon: React.ReactNode;
}

function HeadlineMetric({ label, value, subtext, trend, icon }: HeadlineMetricProps) {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardContent className="p-6">
        <div className="flex items-start justify-between mb-3">
          <span className="text-sm text-neutral-400">{label}</span>
          <div className="p-2 bg-neutral-800/50 rounded-lg">
            {icon}
          </div>
        </div>
        <div className="flex items-end gap-2">
          <span className={cn(
            "text-3xl font-bold font-mono",
            trend === "up" ? "text-green-400" :
            trend === "down" ? "text-red-400" : "text-neutral-100"
          )}>
            {value}
          </span>
          {trend && trend !== "neutral" && (
            <span className={cn(
              "flex items-center text-sm mb-1",
              trend === "up" ? "text-green-400" : "text-red-400"
            )}>
              {trend === "up" ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
            </span>
          )}
        </div>
        {subtext && (
          <p className="text-xs text-neutral-500 mt-2">{subtext}</p>
        )}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Market Outlook Card
// ============================================================================

interface MarketOutlookProps {
  summary: MarketSummary;
}

function MarketOutlook({ summary }: MarketOutlookProps) {
  const outlookText = generateOutlookText(summary);

  const sentimentConfig = {
    bullish: { icon: TrendingUp, color: "text-green-400", bg: "bg-green-500/10", border: "border-green-500/20", label: "Positive" },
    bearish: { icon: TrendingDown, color: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/20", label: "Cautious" },
    mixed: { icon: Minus, color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20", label: "Mixed" },
  };

  const config = sentimentConfig[summary.overallSentiment];
  const SentimentIcon = config.icon;

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-neutral-100">Market Outlook</h3>
          <Badge className={cn("px-3 py-1", config.bg, config.border, config.color)}>
            <SentimentIcon className="w-3.5 h-3.5 mr-1.5" />
            {config.label}
          </Badge>
        </div>
        <p className="text-neutral-300 leading-relaxed">
          {outlookText}
        </p>

        <Separator className="bg-neutral-800 my-4" />

        {/* Quick Stats Row */}
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <span className="text-xl font-bold font-mono text-green-400">{summary.bullishCount}</span>
            </div>
            <span className="text-xs text-neutral-500">Bullish</span>
          </div>
          <div className="text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <TrendingDown className="w-4 h-4 text-red-400" />
              <span className="text-xl font-bold font-mono text-red-400">{summary.bearishCount}</span>
            </div>
            <span className="text-xs text-neutral-500">Bearish</span>
          </div>
          <div className="text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <Minus className="w-4 h-4 text-neutral-400" />
              <span className="text-xl font-bold font-mono text-neutral-400">{summary.neutralCount}</span>
            </div>
            <span className="text-xs text-neutral-500">Neutral</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Top Opportunities List
// ============================================================================

function TopOpportunities() {
  const opportunities = useMemo(() => {
    const items: { name: string; symbol: string; direction: string; confidence: number }[] = [];

    Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
      const signal = MOCK_SIGNALS[assetId as AssetId]?.["D+1"];
      if (signal && signal.direction !== "neutral" && signal.confidence >= 70) {
        items.push({
          name: asset.name,
          symbol: asset.symbol,
          direction: signal.direction,
          confidence: signal.confidence,
        });
      }
    });

    return items.sort((a, b) => b.confidence - a.confidence).slice(0, 5);
  }, []);

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardContent className="p-6">
        <h3 className="text-lg font-semibold text-neutral-100 mb-4">Top Opportunities</h3>
        {opportunities.length > 0 ? (
          <div className="space-y-3">
            {opportunities.map((opp) => (
              <div
                key={opp.symbol}
                className="flex items-center justify-between p-3 bg-neutral-800/30 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <div className={cn(
                    "p-1.5 rounded",
                    opp.direction === "bullish" ? "bg-green-500/10" : "bg-red-500/10"
                  )}>
                    {opp.direction === "bullish" ? (
                      <TrendingUp className="w-4 h-4 text-green-400" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-red-400" />
                    )}
                  </div>
                  <div>
                    <span className="font-medium text-neutral-200">{opp.name}</span>
                    <span className="text-xs text-neutral-500 ml-2">{opp.symbol}</span>
                  </div>
                </div>
                <div className="text-right">
                  <span className={cn(
                    "font-mono font-bold",
                    opp.direction === "bullish" ? "text-green-400" : "text-red-400"
                  )}>
                    {opp.confidence}%
                  </span>
                  <div className="text-xs text-neutral-500">confidence</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-neutral-500 text-sm">No high-confidence opportunities at this time.</p>
        )}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// System Status
// ============================================================================

function SystemStatus() {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardContent className="p-6">
        <h3 className="text-lg font-semibold text-neutral-100 mb-4">System Status</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span className="text-sm text-neutral-300">AI Models</span>
            </div>
            <span className="text-xs text-green-400">10,179 active</span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span className="text-sm text-neutral-300">Data Feeds</span>
            </div>
            <span className="text-xs text-green-400">Live</span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span className="text-sm text-neutral-300">Signal Generation</span>
            </div>
            <span className="text-xs text-green-400">Operational</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Executive Dashboard
// ============================================================================

export function ExecutiveDashboard() {
  const summary = useMemo(() => getMarketSummary(), []);

  // Mock portfolio values (would come from real data)
  const portfolioValue = "$12.4M";
  const dailyPnL = "+$284K";
  const winRate = "61.3%";
  const activeSignals = "10";

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-slate-800/50 rounded-xl border border-slate-700/50">
            <Briefcase className="w-8 h-8 text-slate-300" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-neutral-100">Executive Summary</h1>
            <p className="text-sm text-neutral-400">
              {new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}
            </p>
          </div>
        </div>
        <Badge className="bg-blue-500/10 border-blue-500/30 text-blue-300 px-3 py-1.5">
          <Activity className="w-3.5 h-3.5 mr-1.5" />
          Live
        </Badge>
      </div>

      {/* Headline Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <HeadlineMetric
          label="Portfolio Value"
          value={portfolioValue}
          subtext="Total AUM"
          icon={<DollarSign className="w-5 h-5 text-neutral-400" />}
        />
        <HeadlineMetric
          label="Today's P&L"
          value={dailyPnL}
          trend="up"
          subtext="vs. yesterday"
          icon={<TrendingUp className="w-5 h-5 text-green-400" />}
        />
        <HeadlineMetric
          label="Win Rate"
          value={winRate}
          subtext="30-day rolling"
          icon={<Target className="w-5 h-5 text-neutral-400" />}
        />
        <HeadlineMetric
          label="Active Signals"
          value={activeSignals}
          subtext="Across all assets"
          icon={<Activity className="w-5 h-5 text-neutral-400" />}
        />
      </div>

      <Separator className="bg-neutral-800" />

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Market Outlook - Main */}
        <div className="lg:col-span-2">
          <MarketOutlook summary={summary} />
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          <TopOpportunities />
          <SystemStatus />
        </div>
      </div>

      {/* Footer */}
      <div className="text-center text-xs text-neutral-500 pt-4">
        AI-generated insights from 10,179 ensemble models. Updated in real-time.
      </div>
    </div>
  );
}
