"use client";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { AssetId } from "@/types";
import {
  type PracticalMetricsData,
  type PracticalInsight,
  getActionabilityClasses,
  formatMoveSize,
  ASSET_MOVE_THRESHOLDS,
} from "@/types/practical-metrics";

interface PracticalInsightsProps {
  asset: AssetId;
  assetName: string;
  data?: PracticalMetricsData;
  isLoading?: boolean;
  error?: Error | null;
}

// ============================================================================
// Insight Generation
// ============================================================================

function generateInsight(
  data: PracticalMetricsData,
  assetName: string
): PracticalInsight {
  const { signal, currentPrice, asset } = data;
  const { magnitude, horizonCoverage, practicalScore, confidence, bigMoveWinRate } =
    signal;

  // Build headline
  const moveDirection = magnitude.direction === "up" ? "bullish" : "bearish";
  const moveSize = formatMoveSize(magnitude.absoluteMove, asset);
  const horizon = horizonCoverage.optimalHorizon || "D+5";
  const horizonDays = parseInt(horizon.replace("D+", ""));

  const headline = `${assetName}: ${moveDirection.charAt(0).toUpperCase() + moveDirection.slice(1)} signal predicting ${moveSize} move over ${horizonDays} days`;

  // Build explanation
  const explanationParts: string[] = [];

  explanationParts.push(
    `The ensemble forecasts a ${magnitude.direction === "up" ? "positive" : "negative"} move of ${moveSize} (${magnitude.movePercent.toFixed(2)}% from current ${formatMoveSize(currentPrice, asset)}).`
  );

  if (magnitude.isActionable) {
    explanationParts.push(
      `This exceeds the ${formatMoveSize(ASSET_MOVE_THRESHOLDS[asset] ?? 1, asset)} threshold for ${assetName} by ${magnitude.thresholdMultiple.toFixed(1)}x.`
    );
  } else {
    explanationParts.push(
      `This is below the ${formatMoveSize(ASSET_MOVE_THRESHOLDS[asset] ?? 1, asset)} threshold needed for actionability.`
    );
  }

  explanationParts.push(
    `Model confidence is ${confidence.toFixed(0)}% with a ${bigMoveWinRate.toFixed(0)}% win rate on significant moves.`
  );

  const explanation = explanationParts.join(" ");

  // Build recommendation
  let recommendation: string;
  const positionPct = signal.recommendedPositionSize;

  if (practicalScore.actionability === "high") {
    if (magnitude.direction === "up") {
      recommendation = `Consider hedging ${positionPct}% of short exposure or adding ${positionPct}% long position. Strong signal with favorable risk/reward.`;
    } else {
      recommendation = `Consider hedging ${positionPct}% of long exposure or adding ${positionPct}% short position. Strong bearish signal detected.`;
    }
  } else if (practicalScore.actionability === "medium") {
    recommendation = `Monitor closely. If signal strengthens, consider ${Math.round(positionPct * 0.5)}% position. Current signal is moderate — wait for confirmation.`;
  } else {
    recommendation = `No action recommended. Signal is too weak or move too small for practical hedging. Continue monitoring for stronger signals.`;
  }

  // Build actionability reason
  const reasonParts: string[] = [];

  if (magnitude.isActionable) {
    reasonParts.push("large forecast size");
  } else {
    reasonParts.push("forecast below threshold");
  }

  if (horizonCoverage.coveragePercent >= 67) {
    reasonParts.push("good horizon diversity");
  } else if (horizonCoverage.coveragePercent >= 50) {
    reasonParts.push("moderate horizon coverage");
  } else {
    reasonParts.push("limited horizon coverage");
  }

  if (confidence >= 70) {
    reasonParts.push("high model confidence");
  } else if (confidence >= 50) {
    reasonParts.push("moderate confidence");
  } else {
    reasonParts.push("low confidence");
  }

  if (bigMoveWinRate >= 60) {
    reasonParts.push("strong big-move accuracy");
  } else if (bigMoveWinRate < 50) {
    reasonParts.push("poor big-move accuracy");
  }

  const actionabilityReason = signal.isActionable
    ? `Actionable because: ${reasonParts.slice(0, 3).join(", ")}.`
    : `Not actionable: ${reasonParts.filter((r) => r.includes("below") || r.includes("low") || r.includes("poor") || r.includes("limited")).join(", ")}.`;

  // Build risk note
  let riskNote: string;
  if (practicalScore.actionability === "high") {
    riskNote = `Risk: Even strong signals can be wrong. Size position appropriately. Consider ${Math.round(magnitude.absoluteMove * 0.5)} stop-loss.`;
  } else if (practicalScore.actionability === "medium") {
    riskNote = `Risk: Medium signals have ~${bigMoveWinRate.toFixed(0)}% success rate. Smaller position size recommended until confirmation.`;
  } else {
    riskNote = `Risk: Weak signals are often noise. Taking action now could result in losses. Wait for better entry.`;
  }

  // Determine insight confidence
  let insightConfidence: "high" | "medium" | "low";
  if (confidence >= 70 && bigMoveWinRate >= 60 && magnitude.isActionable) {
    insightConfidence = "high";
  } else if (confidence >= 50 && bigMoveWinRate >= 50) {
    insightConfidence = "medium";
  } else {
    insightConfidence = "low";
  }

  return {
    headline,
    explanation,
    recommendation,
    actionabilityReason,
    riskNote,
    confidence: insightConfidence,
  };
}

// ============================================================================
// Sub-components
// ============================================================================

interface InsightSectionProps {
  title: string;
  children: React.ReactNode;
  icon?: string;
  variant?: "default" | "highlight" | "warning" | "muted";
}

function InsightSection({
  title,
  children,
  icon,
  variant = "default",
}: InsightSectionProps) {
  const variantClasses = {
    default: "bg-neutral-800/30 border-neutral-700",
    highlight: "bg-green-500/5 border-green-500/20",
    warning: "bg-yellow-500/5 border-yellow-500/20",
    muted: "bg-neutral-800/20 border-neutral-800",
  };

  return (
    <div className={cn("p-4 rounded-lg border", variantClasses[variant])}>
      <div className="flex items-center gap-2 mb-2">
        {icon && <span className="text-lg">{icon}</span>}
        <span className="text-[10px] uppercase tracking-wider text-neutral-500 font-semibold">
          {title}
        </span>
      </div>
      <div className="text-sm text-neutral-200 leading-relaxed">{children}</div>
    </div>
  );
}

interface ConfidenceBadgeProps {
  confidence: "high" | "medium" | "low";
}

function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  const classes = {
    high: "bg-green-500/10 border-green-500/30 text-green-500",
    medium: "bg-yellow-500/10 border-yellow-500/30 text-yellow-500",
    low: "bg-neutral-500/10 border-neutral-500/30 text-neutral-400",
  };

  const labels = {
    high: "High Confidence",
    medium: "Medium Confidence",
    low: "Low Confidence",
  };

  return (
    <Badge className={cn("text-xs", classes[confidence])}>
      {labels[confidence]}
    </Badge>
  );
}

interface QuickStatProps {
  label: string;
  value: string;
  color?: string;
}

function QuickStat({ label, value, color = "text-neutral-100" }: QuickStatProps) {
  return (
    <div className="text-center">
      <div className={cn("text-lg font-bold font-mono", color)}>{value}</div>
      <div className="text-[10px] text-neutral-500 uppercase tracking-wider">
        {label}
      </div>
    </div>
  );
}

function PracticalInsightsSkeleton() {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <Skeleton className="h-5 w-48 bg-neutral-800" />
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        <Skeleton className="h-6 w-full bg-neutral-800" />
        <Skeleton className="h-24 w-full bg-neutral-800" />
        <Skeleton className="h-20 w-full bg-neutral-800" />
        <Skeleton className="h-16 w-full bg-neutral-800" />
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function PracticalInsights({
  asset,
  assetName,
  data,
  isLoading,
  error,
}: PracticalInsightsProps) {
  if (isLoading) {
    return <PracticalInsightsSkeleton />;
  }

  if (error) {
    return (
      <Card className="bg-neutral-900/50 border-red-900/50">
        <CardContent className="p-4">
          <div className="text-center text-red-500 text-sm py-4">
            <p>Failed to load insights</p>
            <p className="text-xs text-neutral-500 mt-1">{error.message}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="text-center text-neutral-500 text-sm py-4">
            No insights available
          </div>
        </CardContent>
      </Card>
    );
  }

  const insight = generateInsight(data, assetName);
  const { signal } = data;
  const actionClasses = getActionabilityClasses(signal.practicalScore.actionability);

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-neutral-100">
              Trading Insights
            </span>
            <Badge className="bg-amber-500/10 border-amber-500/30 text-amber-500 text-xs">
              Plain English
            </Badge>
          </div>
          <ConfidenceBadge confidence={insight.confidence} />
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0 space-y-4">
        {/* Headline */}
        <div
          className={cn(
            "p-4 rounded-lg border-l-4",
            actionClasses.bg,
            signal.practicalScore.actionability === "high"
              ? "border-l-green-500"
              : signal.practicalScore.actionability === "medium"
              ? "border-l-yellow-500"
              : "border-l-neutral-600"
          )}
        >
          <p className={cn("text-base font-medium", actionClasses.text)}>
            {insight.headline}
          </p>
        </div>

        {/* Quick Stats Row */}
        <div className="grid grid-cols-4 gap-4 py-3 border-y border-neutral-800">
          <QuickStat
            label="Move"
            value={formatMoveSize(signal.magnitude.predictedMove, asset)}
            color={
              signal.magnitude.direction === "up"
                ? "text-green-500"
                : "text-red-500"
            }
          />
          <QuickStat
            label="Horizon"
            value={signal.horizonCoverage.optimalHorizon || "—"}
            color="text-blue-400"
          />
          <QuickStat
            label="Confidence"
            value={`${signal.confidence.toFixed(0)}%`}
            color={
              signal.confidence >= 70
                ? "text-green-500"
                : signal.confidence >= 50
                ? "text-yellow-500"
                : "text-red-500"
            }
          />
          <QuickStat
            label="Position"
            value={`${signal.recommendedPositionSize}%`}
            color={
              signal.recommendedPositionSize >= 50
                ? "text-green-500"
                : "text-neutral-400"
            }
          />
        </div>

        {/* What the Signal Means */}
        <InsightSection title="What This Means" icon="📊">
          {insight.explanation}
        </InsightSection>

        {/* Recommendation */}
        <InsightSection
          title="Recommendation"
          icon="💡"
          variant={
            signal.practicalScore.actionability === "high"
              ? "highlight"
              : signal.practicalScore.actionability === "medium"
              ? "warning"
              : "muted"
          }
        >
          {insight.recommendation}
        </InsightSection>

        {/* Why Actionable */}
        <InsightSection title="Why This Rating" icon="🎯">
          {insight.actionabilityReason}
        </InsightSection>

        {/* Risk Note */}
        <InsightSection title="Risk Considerations" icon="⚠️" variant="muted">
          {insight.riskNote}
        </InsightSection>

        {/* Comparison with Traditional Metrics */}
        <div className="p-3 bg-neutral-800/20 rounded-lg">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2">
            Traditional vs. Practical Comparison
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-neutral-500 mb-1">
                Traditional Sharpe
              </div>
              <div
                className={cn(
                  "text-lg font-bold font-mono",
                  data.traditionalSharpe >= 1.5
                    ? "text-green-500"
                    : data.traditionalSharpe >= 0.5
                    ? "text-yellow-500"
                    : "text-red-500"
                )}
              >
                {data.traditionalSharpe.toFixed(2)}
              </div>
              <div className="text-[10px] text-neutral-600">
                {data.traditionalSharpe >= 1.5
                  ? "Looks great on paper"
                  : data.traditionalSharpe >= 0.5
                  ? "Acceptable risk-adjusted"
                  : "Poor risk-adjusted"}
              </div>
            </div>
            <div>
              <div className="text-xs text-neutral-500 mb-1">Practical Score</div>
              <div className={cn("text-lg font-bold font-mono", actionClasses.text)}>
                {signal.practicalScore.score.toFixed(0)}
              </div>
              <div className="text-[10px] text-neutral-600">
                {signal.practicalScore.actionability === "high"
                  ? "Actually tradeable"
                  : signal.practicalScore.actionability === "medium"
                  ? "Marginally useful"
                  : "Not worth the effort"}
              </div>
            </div>
          </div>
          {data.traditionalSharpe >= 1.5 &&
            signal.practicalScore.actionability === "low" && (
              <div className="mt-3 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-xs text-yellow-500">
                ⚠️ High Sharpe but low practical score — the returns are
                risk-adjusted well, but the moves are too small to action
                profitably after transaction costs.
              </div>
            )}
          {data.traditionalSharpe < 1.0 &&
            signal.practicalScore.actionability === "high" && (
              <div className="mt-3 p-2 bg-green-500/10 border border-green-500/20 rounded text-xs text-green-500">
                ✓ Moderate Sharpe but high practical score — the signal predicts
                large enough moves to be actionable despite higher volatility.
              </div>
            )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between pt-2 border-t border-neutral-800">
          <span className="text-[10px] text-neutral-600">
            Insights generated for CME hedging desk utility
          </span>
          <span className="text-[10px] text-neutral-600 font-mono">
            {new Date(data.analyzedAt).toLocaleString()}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
