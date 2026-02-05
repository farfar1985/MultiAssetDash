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
  Sparkles,
  TrendingUp,
  TrendingDown,
  Minus,
  ArrowBigUp,
  ArrowBigDown,
  Activity,
  ThumbsUp,
  ThumbsDown,
  Hand,
  CircleDollarSign,
  Lightbulb,
  Clock,
  ShieldCheck,
  Star,
  CircleHelp,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

type Recommendation = "buy" | "sell" | "hold";

interface RetailSignal {
  assetId: AssetId;
  assetName: string;
  symbol: string;
  currentPrice: number;
  signal: SignalData;
  recommendation: Recommendation;
  confidenceLevel: "high" | "medium" | "low";
  plainEnglish: string;
  whyThis: string;
  rank: number;
}

// ============================================================================
// Helper Functions
// ============================================================================

function getRecommendation(signal: SignalData): Recommendation {
  if (signal.direction === "neutral" || signal.confidence < 60) return "hold";
  if (signal.direction === "bullish") return "buy";
  return "sell";
}

function getConfidenceLevel(confidence: number): "high" | "medium" | "low" {
  if (confidence >= 75) return "high";
  if (confidence >= 60) return "medium";
  return "low";
}

function getPlainEnglish(recommendation: Recommendation, assetName: string, confidence: number): string {
  const confidenceWord = confidence >= 75 ? "strong" : confidence >= 60 ? "moderate" : "weak";

  switch (recommendation) {
    case "buy":
      return `Our AI thinks ${assetName} will go UP. This is a ${confidenceWord} signal to consider buying.`;
    case "sell":
      return `Our AI thinks ${assetName} will go DOWN. This is a ${confidenceWord} signal to consider selling.`;
    default:
      return `${assetName} doesn't have a clear direction right now. It might be best to wait for a stronger signal.`;
  }
}

function getWhyThis(signal: SignalData, recommendation: Recommendation): string {
  const agreementPercent = Math.round((signal.modelsAgreeing / signal.modelsTotal) * 100);

  if (recommendation === "hold") {
    return `Only ${agreementPercent}% of our models agree, so we recommend waiting for more clarity.`;
  }

  return `${agreementPercent}% of our ${signal.modelsTotal.toLocaleString()} AI models agree on this direction.`;
}

function getRetailSignals(): RetailSignal[] {
  const signals: RetailSignal[] = [];
  const horizons: Horizon[] = ["D+1", "D+5", "D+10"];

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    // Find the best signal (highest confidence, non-neutral)
    let bestSignal: SignalData | null = null;

    for (const horizon of horizons) {
      const signal = MOCK_SIGNALS[assetId as AssetId]?.[horizon];
      if (signal) {
        if (!bestSignal ||
            (signal.direction !== "neutral" && signal.confidence > bestSignal.confidence)) {
          bestSignal = signal;
        }
      }
    }

    if (bestSignal && bestSignal.confidence >= 55) {
      const recommendation = getRecommendation(bestSignal);
      signals.push({
        assetId: assetId as AssetId,
        assetName: asset.name,
        symbol: asset.symbol,
        currentPrice: asset.currentPrice,
        signal: bestSignal,
        recommendation,
        confidenceLevel: getConfidenceLevel(bestSignal.confidence),
        plainEnglish: getPlainEnglish(recommendation, asset.name, bestSignal.confidence),
        whyThis: getWhyThis(bestSignal, recommendation),
        rank: 0,
      });
    }
  });

  // Sort by confidence and assign ranks
  return signals
    .sort((a, b) => b.signal.confidence - a.signal.confidence)
    .map((s, i) => ({ ...s, rank: i + 1 }));
}

// ============================================================================
// Header Component
// ============================================================================

function RetailHeader() {
  return (
    <div className="bg-gradient-to-r from-orange-900/30 via-amber-900/20 to-orange-900/30 border border-orange-500/20 rounded-xl p-6">
      <div className="flex items-center gap-4 mb-4">
        <div className="p-3 bg-orange-500/20 rounded-xl border border-orange-500/30">
          <Sparkles className="w-8 h-8 text-orange-400" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-neutral-100">Your Trading Signals</h1>
          <p className="text-sm text-neutral-400">
            Simple, clear recommendations powered by AI
          </p>
        </div>
      </div>
      <div className="flex flex-wrap gap-3">
        <Badge className="bg-orange-500/10 border-orange-500/30 text-orange-300 px-3 py-1.5">
          <ThumbsUp className="w-3.5 h-3.5 mr-1.5" />
          Easy to Understand
        </Badge>
        <Badge className="bg-amber-500/10 border-amber-500/30 text-amber-300 px-3 py-1.5">
          <ShieldCheck className="w-3.5 h-3.5 mr-1.5" />
          AI-Powered
        </Badge>
        <Badge className="bg-yellow-500/10 border-yellow-500/30 text-yellow-300 px-3 py-1.5">
          <Lightbulb className="w-3.5 h-3.5 mr-1.5" />
          Plain English
        </Badge>
      </div>
    </div>
  );
}

// ============================================================================
// Quick Summary Stats
// ============================================================================

interface QuickStatsProps {
  signals: RetailSignal[];
}

function QuickStats({ signals }: QuickStatsProps) {
  const stats = useMemo(() => {
    const buySignals = signals.filter(s => s.recommendation === "buy").length;
    const sellSignals = signals.filter(s => s.recommendation === "sell").length;
    const holdSignals = signals.filter(s => s.recommendation === "hold").length;
    const topPick = signals[0];
    const avgConfidence = signals.reduce((acc, s) => acc + s.signal.confidence, 0) / signals.length;

    return { buySignals, sellSignals, holdSignals, topPick, avgConfidence };
  }, [signals]);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Buy Signals */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <ThumbsUp className="w-4 h-4 text-green-400" />
            <span className="text-xs uppercase tracking-wider">Buy Signals</span>
          </div>
          <div className="text-3xl font-bold font-mono text-green-400">
            {stats.buySignals}
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Assets looking good
          </div>
        </CardContent>
      </Card>

      {/* Sell Signals */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <ThumbsDown className="w-4 h-4 text-red-400" />
            <span className="text-xs uppercase tracking-wider">Sell Signals</span>
          </div>
          <div className="text-3xl font-bold font-mono text-red-400">
            {stats.sellSignals}
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Consider avoiding
          </div>
        </CardContent>
      </Card>

      {/* Hold/Wait Signals */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <Hand className="w-4 h-4 text-amber-400" />
            <span className="text-xs uppercase tracking-wider">Wait & See</span>
          </div>
          <div className="text-3xl font-bold font-mono text-amber-400">
            {stats.holdSignals}
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            No clear signal yet
          </div>
        </CardContent>
      </Card>

      {/* Top Pick */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <Star className="w-4 h-4 text-orange-400" />
            <span className="text-xs uppercase tracking-wider">Top Pick</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl font-bold text-orange-400">
              {stats.topPick?.assetName}
            </span>
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            {stats.topPick?.signal.confidence}% confident
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ============================================================================
// Confidence Gauge Component
// ============================================================================

interface ConfidenceGaugeProps {
  confidence: number;
  size?: "sm" | "md" | "lg";
}

function ConfidenceGauge({ confidence, size = "md" }: ConfidenceGaugeProps) {
  const sizeConfig = {
    sm: { width: 80, height: 50, fontSize: "text-sm", strokeWidth: 6 },
    md: { width: 120, height: 70, fontSize: "text-xl", strokeWidth: 8 },
    lg: { width: 160, height: 90, fontSize: "text-2xl", strokeWidth: 10 },
  };

  const config = sizeConfig[size];
  const radius = 40;
  const circumference = Math.PI * radius;
  const progress = (confidence / 100) * circumference;

  const getColor = () => {
    if (confidence >= 75) return { stroke: "#22c55e", text: "text-green-400", label: "Strong" };
    if (confidence >= 60) return { stroke: "#f59e0b", text: "text-amber-400", label: "Moderate" };
    return { stroke: "#ef4444", text: "text-red-400", label: "Weak" };
  };

  const { stroke, text, label } = getColor();

  return (
    <div className="flex flex-col items-center">
      <svg width={config.width} height={config.height} className="overflow-visible">
        {/* Background arc */}
        <path
          d={`M ${config.width / 2 - radius} ${config.height} A ${radius} ${radius} 0 0 1 ${config.width / 2 + radius} ${config.height}`}
          fill="none"
          stroke="#262626"
          strokeWidth={config.strokeWidth}
          strokeLinecap="round"
        />
        {/* Progress arc */}
        <path
          d={`M ${config.width / 2 - radius} ${config.height} A ${radius} ${radius} 0 0 1 ${config.width / 2 + radius} ${config.height}`}
          fill="none"
          stroke={stroke}
          strokeWidth={config.strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference - progress}
          className="transition-all duration-500"
        />
        {/* Percentage text */}
        <text
          x={config.width / 2}
          y={config.height - 15}
          textAnchor="middle"
          className={cn("font-mono font-bold fill-current", text, config.fontSize)}
        >
          {confidence}%
        </text>
      </svg>
      <span className={cn("text-xs font-medium mt-1", text)}>{label}</span>
    </div>
  );
}

// ============================================================================
// Big Direction Arrow Component
// ============================================================================

interface DirectionArrowProps {
  recommendation: Recommendation;
}

function DirectionArrow({ recommendation }: DirectionArrowProps) {
  if (recommendation === "hold") {
    return (
      <div className="flex flex-col items-center p-4 bg-amber-500/10 rounded-xl border border-amber-500/30">
        <Minus className="w-16 h-16 text-amber-400" />
        <span className="text-lg font-bold text-amber-400 mt-2">HOLD</span>
        <span className="text-xs text-amber-300/70">Wait for clarity</span>
      </div>
    );
  }

  if (recommendation === "buy") {
    return (
      <div className="flex flex-col items-center p-4 bg-green-500/10 rounded-xl border border-green-500/30">
        <ArrowBigUp className="w-16 h-16 text-green-400 animate-bounce" />
        <span className="text-lg font-bold text-green-400 mt-2">BUY</span>
        <span className="text-xs text-green-300/70">Going up</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center p-4 bg-red-500/10 rounded-xl border border-red-500/30">
      <ArrowBigDown className="w-16 h-16 text-red-400 animate-bounce" />
      <span className="text-lg font-bold text-red-400 mt-2">SELL</span>
      <span className="text-xs text-red-300/70">Going down</span>
    </div>
  );
}

// ============================================================================
// Retail Signal Card
// ============================================================================

interface RetailSignalCardProps {
  signal: RetailSignal;
}

function RetailSignalCard({ signal }: RetailSignalCardProps) {
  const isTopPick = signal.rank <= 3;

  return (
    <Card className={cn(
      "bg-neutral-900/50 border-neutral-800 hover:border-neutral-700 transition-all",
      isTopPick && "ring-1 ring-orange-500/30"
    )}>
      <CardContent className="p-5">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            {isTopPick && (
              <div className={cn(
                "w-8 h-8 rounded-lg flex items-center justify-center",
                signal.rank === 1 ? "bg-orange-500/20 text-orange-400 border border-orange-500/30" :
                signal.rank === 2 ? "bg-amber-500/20 text-amber-400 border border-amber-500/30" :
                "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"
              )}>
                <Star className="w-4 h-4" />
              </div>
            )}
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-lg text-neutral-100">{signal.assetName}</span>
              </div>
              <div className="flex items-center gap-2 mt-0.5">
                <CircleDollarSign className="w-3 h-3 text-neutral-500" />
                <span className="text-sm text-neutral-400">
                  ${signal.currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </span>
              </div>
            </div>
          </div>
          <Badge className={cn(
            "text-xs px-2 py-1",
            signal.confidenceLevel === "high"
              ? "bg-green-500/10 border-green-500/30 text-green-400"
              : signal.confidenceLevel === "medium"
              ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
              : "bg-neutral-500/10 border-neutral-500/30 text-neutral-400"
          )}>
            {signal.confidenceLevel === "high" ? "High Confidence" :
             signal.confidenceLevel === "medium" ? "Moderate" : "Low Confidence"}
          </Badge>
        </div>

        {/* Direction Arrow and Gauge */}
        <div className="flex items-center justify-between mb-4 gap-4">
          <DirectionArrow recommendation={signal.recommendation} />
          <ConfidenceGauge confidence={signal.signal.confidence} size="md" />
        </div>

        {/* Plain English Explanation */}
        <div className="p-3 bg-orange-500/5 border border-orange-500/20 rounded-lg mb-4">
          <div className="flex items-start gap-2">
            <Lightbulb className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-neutral-300">{signal.plainEnglish}</p>
          </div>
        </div>

        {/* Why This Signal */}
        <div className="flex items-center gap-2 text-xs text-neutral-500">
          <CircleHelp className="w-3 h-3" />
          <span>{signal.whyThis}</span>
        </div>

        {/* Time Horizon */}
        <div className="mt-3 pt-3 border-t border-neutral-800 flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-xs text-neutral-500">
            <Clock className="w-3 h-3" />
            <span>Best for: {signal.signal.horizon === "D+1" ? "Tomorrow" :
                           signal.signal.horizon === "D+5" ? "This week" : "Next 2 weeks"}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Simple Explanation Panel
// ============================================================================

function HowToReadPanel() {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center gap-2">
          <CircleHelp className="w-5 h-5 text-orange-400" />
          <span className="text-sm font-semibold text-neutral-100">How to Read Signals</span>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-3">
        <div className="flex items-center gap-3 p-2 bg-green-500/5 border border-green-500/20 rounded-lg">
          <ArrowBigUp className="w-8 h-8 text-green-400" />
          <div>
            <div className="font-medium text-green-400">BUY Signal</div>
            <div className="text-xs text-neutral-400">AI predicts price will go UP</div>
          </div>
        </div>

        <div className="flex items-center gap-3 p-2 bg-red-500/5 border border-red-500/20 rounded-lg">
          <ArrowBigDown className="w-8 h-8 text-red-400" />
          <div>
            <div className="font-medium text-red-400">SELL Signal</div>
            <div className="text-xs text-neutral-400">AI predicts price will go DOWN</div>
          </div>
        </div>

        <div className="flex items-center gap-3 p-2 bg-amber-500/5 border border-amber-500/20 rounded-lg">
          <Minus className="w-8 h-8 text-amber-400" />
          <div>
            <div className="font-medium text-amber-400">HOLD Signal</div>
            <div className="text-xs text-neutral-400">No clear direction - wait</div>
          </div>
        </div>

        <Separator className="bg-neutral-800" />

        <div className="text-xs text-neutral-500 space-y-1">
          <p><strong className="text-neutral-400">High Confidence (75%+):</strong> Strong signal</p>
          <p><strong className="text-neutral-400">Moderate (60-74%):</strong> Good signal</p>
          <p><strong className="text-neutral-400">Low (&lt;60%):</strong> Weak signal</p>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Quick Action List
// ============================================================================

interface QuickActionsProps {
  signals: RetailSignal[];
}

function QuickActionsList({ signals }: QuickActionsProps) {
  const buySignals = signals.filter(s => s.recommendation === "buy" && s.confidenceLevel !== "low");
  const sellSignals = signals.filter(s => s.recommendation === "sell" && s.confidenceLevel !== "low");

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-orange-400" />
          <span className="text-sm font-semibold text-neutral-100">Today&apos;s Top Picks</span>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        {/* Buy Recommendations */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <ThumbsUp className="w-4 h-4 text-green-400" />
            <span className="text-sm font-medium text-green-400">Consider Buying</span>
          </div>
          {buySignals.length > 0 ? (
            <div className="space-y-2">
              {buySignals.slice(0, 3).map((s) => (
                <div key={s.assetId} className="flex items-center justify-between p-2 bg-green-500/5 border border-green-500/20 rounded-lg">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-green-400" />
                    <span className="text-sm text-neutral-200">{s.assetName}</span>
                  </div>
                  <span className="text-xs font-mono text-green-400">{s.signal.confidence}%</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-neutral-500 p-2">No strong buy signals right now</p>
          )}
        </div>

        {/* Sell Recommendations */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <ThumbsDown className="w-4 h-4 text-red-400" />
            <span className="text-sm font-medium text-red-400">Consider Avoiding</span>
          </div>
          {sellSignals.length > 0 ? (
            <div className="space-y-2">
              {sellSignals.slice(0, 3).map((s) => (
                <div key={s.assetId} className="flex items-center justify-between p-2 bg-red-500/5 border border-red-500/20 rounded-lg">
                  <div className="flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-red-400" />
                    <span className="text-sm text-neutral-200">{s.assetName}</span>
                  </div>
                  <span className="text-xs font-mono text-red-400">{s.signal.confidence}%</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-neutral-500 p-2">No strong sell signals right now</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Retail Dashboard Component
// ============================================================================

export function RetailDashboard() {
  const retailSignals = useMemo(() => getRetailSignals(), []);
  const topSignals = retailSignals.slice(0, 6);

  return (
    <div className="space-y-6">
      {/* Header */}
      <RetailHeader />

      {/* Quick Stats */}
      <QuickStats signals={retailSignals} />

      <Separator className="bg-neutral-800" />

      {/* Live Signals - Popular Assets */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
              <Activity className="w-5 h-5 text-orange-400" />
              Live Prices
            </h2>
            <ApiHealthIndicator />
          </div>
          <span className="text-xs text-neutral-500">Popular assets</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <LiveSignalCard asset="Bitcoin" displayName="Bitcoin" />
          <LiveSignalCard asset="GOLD" displayName="Gold" />
          <LiveSignalCard asset="Crude_Oil" displayName="Crude Oil" />
          <LiveSignalCard asset="SP500" displayName="S&P 500" />
        </div>
      </section>

      <Separator className="bg-neutral-800" />

      {/* Two-Column Layout: Signals + Help */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Top Signal Cards */}
        <div className="xl:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
              <Star className="w-5 h-5 text-orange-400" />
              Your Best Opportunities
            </h2>
            <Badge className="bg-orange-500/10 border-orange-500/30 text-orange-300 px-2 py-1 text-xs">
              AI-Picked
            </Badge>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {topSignals.map((signal) => (
              <RetailSignalCard key={signal.assetId} signal={signal} />
            ))}
          </div>
        </div>

        {/* Sidebar - Help + Quick Actions */}
        <div className="xl:col-span-1 space-y-4">
          <QuickActionsList signals={retailSignals} />
          <HowToReadPanel />
        </div>
      </div>

      {/* Disclaimer */}
      <div className="p-4 bg-neutral-900/30 border border-neutral-800 rounded-lg flex items-start gap-3">
        <ShieldCheck className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" />
        <div className="text-xs text-neutral-500">
          <strong className="text-neutral-400">Remember:</strong> These are AI predictions, not
          financial advice. Markets can be unpredictable. Only invest what you can afford to lose,
          and consider talking to a financial advisor before making big decisions.
        </div>
      </div>
    </div>
  );
}
