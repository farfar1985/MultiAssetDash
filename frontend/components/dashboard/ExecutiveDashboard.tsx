"use client";

import { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { MOCK_ASSETS, MOCK_SIGNALS, type SignalData } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Crown,
  AlertTriangle,
  CheckCircle,
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
  overallSentiment: "bullish" | "bearish" | "mixed";
  avgConfidence: number;
}

interface TopSignal {
  name: string;
  symbol: string;
  direction: "bullish" | "bearish" | "neutral";
  confidence: number;
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

  Object.entries(MOCK_ASSETS).forEach(([assetId]) => {
    const signal = MOCK_SIGNALS[assetId as AssetId]?.["D+1"];
    if (signal) {
      signalCount++;
      totalConfidence += signal.confidence;

      if (signal.direction === "bullish") bullishCount++;
      else if (signal.direction === "bearish") bearishCount++;
      else neutralCount++;
    }
  });

  const overallSentiment: MarketSummary["overallSentiment"] =
    bullishCount > bearishCount + 2 ? "bullish" :
    bearishCount > bullishCount + 2 ? "bearish" : "mixed";

  return {
    bullishCount,
    bearishCount,
    neutralCount,
    overallSentiment,
    avgConfidence: signalCount > 0 ? totalConfidence / signalCount : 0,
  };
}

function getTopSignals(count: number = 3): TopSignal[] {
  const signals: TopSignal[] = [];

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    const signal = MOCK_SIGNALS[assetId as AssetId]?.["D+1"];
    if (signal && signal.direction !== "neutral" && signal.confidence >= 65) {
      signals.push({
        name: asset.name,
        symbol: asset.symbol,
        direction: signal.direction,
        confidence: signal.confidence,
      });
    }
  });

  return signals
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, count);
}

function getRiskLevel(summary: MarketSummary): { level: "low" | "moderate" | "elevated"; score: number } {
  const volatilityProxy = Math.abs(summary.bullishCount - summary.bearishCount);
  const confidenceInverse = 100 - summary.avgConfidence;

  const riskScore = Math.round((volatilityProxy * 5) + (confidenceInverse * 0.3));

  if (riskScore < 25) return { level: "low", score: riskScore };
  if (riskScore < 50) return { level: "moderate", score: riskScore };
  return { level: "elevated", score: Math.min(riskScore, 100) };
}

// ============================================================================
// Summary Card Component
// ============================================================================

interface SummaryCardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

function SummaryCard({ title, children, className }: SummaryCardProps) {
  return (
    <Card className={cn(
      "bg-neutral-950 border-amber-900/30",
      className
    )}>
      <CardContent className="p-6">
        <h3 className="text-xs uppercase tracking-widest text-amber-600/80 mb-4 font-medium">
          {title}
        </h3>
        {children}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Market Outlook Card
// ============================================================================

function MarketOutlookCard({ summary }: { summary: MarketSummary }) {
  const outlookConfig = {
    bullish: {
      label: "Positive",
      icon: TrendingUp,
      color: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/30"
    },
    bearish: {
      label: "Cautious",
      icon: TrendingDown,
      color: "text-red-400",
      bg: "bg-red-500/10",
      border: "border-red-500/30"
    },
    mixed: {
      label: "Mixed",
      icon: Minus,
      color: "text-amber-400",
      bg: "bg-amber-500/10",
      border: "border-amber-500/30"
    },
  };

  const config = outlookConfig[summary.overallSentiment];
  const OutlookIcon = config.icon;

  return (
    <SummaryCard title="Market Outlook">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className={cn(
            "p-3 rounded-xl border",
            config.bg,
            config.border
          )}>
            <OutlookIcon className={cn("w-8 h-8", config.color)} />
          </div>
          <div>
            <div className={cn("text-3xl font-bold", config.color)}>
              {config.label}
            </div>
            <div className="text-sm text-neutral-500 mt-1">
              {summary.bullishCount} up / {summary.bearishCount} down / {summary.neutralCount} neutral
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-mono font-bold text-amber-500">
            {summary.avgConfidence.toFixed(0)}%
          </div>
          <div className="text-xs text-neutral-500">Avg Confidence</div>
        </div>
      </div>
    </SummaryCard>
  );
}

// ============================================================================
// Top 3 Signals Card
// ============================================================================

function TopSignalsCard({ signals }: { signals: TopSignal[] }) {
  return (
    <SummaryCard title="Top 3 Signals">
      <div className="space-y-4">
        {signals.map((signal, idx) => {
          const isBullish = signal.direction === "bullish";
          return (
            <div
              key={signal.symbol}
              className="flex items-center justify-between py-2 border-b border-neutral-800/50 last:border-0"
            >
              <div className="flex items-center gap-4">
                <div className={cn(
                  "w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold",
                  idx === 0 ? "bg-amber-500/20 text-amber-400 border border-amber-500/30" :
                  "bg-neutral-800 text-neutral-500"
                )}>
                  {idx + 1}
                </div>
                <div>
                  <div className="font-medium text-neutral-100">{signal.name}</div>
                  <div className="text-xs text-neutral-500">{signal.symbol}</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Badge className={cn(
                  "text-xs px-2 py-1",
                  isBullish
                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                    : "bg-red-500/10 border-red-500/30 text-red-400"
                )}>
                  {isBullish ? (
                    <ArrowUpRight className="w-3 h-3 mr-1" />
                  ) : (
                    <ArrowDownRight className="w-3 h-3 mr-1" />
                  )}
                  {signal.direction.toUpperCase()}
                </Badge>
                <span className="font-mono font-bold text-lg text-amber-500">
                  {signal.confidence}%
                </span>
              </div>
            </div>
          );
        })}
        {signals.length === 0 && (
          <div className="text-neutral-500 text-sm text-center py-4">
            No high-confidence signals at this time
          </div>
        )}
      </div>
    </SummaryCard>
  );
}

// ============================================================================
// Risk Indicator Card
// ============================================================================

function RiskIndicatorCard({ summary }: { summary: MarketSummary }) {
  const risk = getRiskLevel(summary);

  const riskConfig = {
    low: {
      icon: CheckCircle,
      color: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/30",
      label: "Low Risk",
      description: "Market conditions favorable"
    },
    moderate: {
      icon: AlertTriangle,
      color: "text-amber-400",
      bg: "bg-amber-500/10",
      border: "border-amber-500/30",
      label: "Moderate Risk",
      description: "Monitor positions closely"
    },
    elevated: {
      icon: AlertTriangle,
      color: "text-red-400",
      bg: "bg-red-500/10",
      border: "border-red-500/30",
      label: "Elevated Risk",
      description: "Consider hedging exposure"
    },
  };

  const config = riskConfig[risk.level];
  const RiskIcon = config.icon;

  return (
    <SummaryCard title="Key Risk Indicator">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className={cn(
            "p-3 rounded-xl border",
            config.bg,
            config.border
          )}>
            <RiskIcon className={cn("w-8 h-8", config.color)} />
          </div>
          <div>
            <div className={cn("text-2xl font-bold", config.color)}>
              {config.label}
            </div>
            <div className="text-sm text-neutral-500 mt-1">
              {config.description}
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className={cn("text-3xl font-mono font-bold", config.color)}>
            {risk.score}
          </div>
          <div className="text-xs text-neutral-500">Risk Score</div>
        </div>
      </div>
    </SummaryCard>
  );
}

// ============================================================================
// Main Executive Dashboard
// ============================================================================

export function ExecutiveDashboard() {
  const summary = useMemo(() => getMarketSummary(), []);
  const topSignals = useMemo(() => getTopSignals(3), []);

  return (
    <div className="min-h-screen bg-neutral-950 -m-6 p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between pb-6 border-b border-amber-900/20">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-amber-600/20 to-amber-800/20 rounded-xl border border-amber-600/30">
              <Crown className="w-8 h-8 text-amber-500" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-neutral-100">Executive Brief</h1>
              <p className="text-sm text-neutral-500">
                {new Date().toLocaleDateString("en-US", {
                  weekday: "long",
                  year: "numeric",
                  month: "long",
                  day: "numeric"
                })}
              </p>
            </div>
          </div>
          <Badge className="bg-amber-500/10 border-amber-500/30 text-amber-400 px-4 py-2">
            <span className="relative flex h-2 w-2 mr-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
            </span>
            Live
          </Badge>
        </div>

        {/* Summary Cards */}
        <div className="space-y-6">
          <MarketOutlookCard summary={summary} />
          <TopSignalsCard signals={topSignals} />
          <RiskIndicatorCard summary={summary} />
        </div>

        {/* Footer */}
        <div className="text-center pt-6 border-t border-amber-900/20">
          <p className="text-xs text-neutral-600">
            AI-powered insights from 10,179 ensemble models
          </p>
        </div>
      </div>
    </div>
  );
}
