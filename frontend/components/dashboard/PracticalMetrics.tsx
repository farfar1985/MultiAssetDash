"use client";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { AssetId } from "@/types";
import type { Horizon } from "@/types/horizon-pairs";
import {
  type PracticalMetricsData,
  type ActionabilityLevel,
  getActionabilityClasses,
  getUrgencyClasses,
  getUrgencyText,
  formatMoveSize,
  ASSET_MOVE_THRESHOLDS,
} from "@/types/practical-metrics";

interface PracticalMetricsProps {
  asset: AssetId;
  data?: PracticalMetricsData;
  isLoading?: boolean;
  error?: Error | null;
}

const ALL_HORIZONS: Horizon[] = ["D+1", "D+2", "D+3", "D+5", "D+7", "D+10"];

// ============================================================================
// Sub-components
// ============================================================================

interface MetricBoxProps {
  label: string;
  value: string | number;
  subtext?: string;
  valueColor?: string;
  size?: "sm" | "md" | "lg";
}

function MetricBox({
  label,
  value,
  subtext,
  valueColor = "text-neutral-100",
  size = "md",
}: MetricBoxProps) {
  const sizeClasses = {
    sm: "text-base",
    md: "text-lg",
    lg: "text-2xl",
  };

  return (
    <div className="p-3 bg-neutral-800/50 rounded-lg">
      <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
        {label}
      </div>
      <div className={cn("font-bold font-mono", sizeClasses[size], valueColor)}>
        {value}
      </div>
      {subtext && (
        <div className="text-[10px] text-neutral-600 mt-0.5">{subtext}</div>
      )}
    </div>
  );
}

interface ActionabilityIndicatorProps {
  level: ActionabilityLevel;
  score: number;
}

function ActionabilityIndicator({ level, score }: ActionabilityIndicatorProps) {
  const classes = getActionabilityClasses(level);
  const labels: Record<ActionabilityLevel, string> = {
    high: "High — Act Now",
    medium: "Medium — Watch Closely",
    low: "Low — Wait",
  };

  return (
    <div
      className={cn(
        "flex items-center gap-3 p-4 rounded-lg border",
        classes.bg,
        classes.border
      )}
    >
      <div className="flex items-center gap-2">
        <div className={cn("w-4 h-4 rounded-full animate-pulse", classes.dot)} />
        <span className={cn("text-sm font-semibold", classes.text)}>
          {labels[level]}
        </span>
      </div>
      <div className="ml-auto text-right">
        <div className={cn("text-2xl font-bold font-mono", classes.text)}>
          {score.toFixed(0)}
        </div>
        <div className="text-[10px] text-neutral-500 uppercase tracking-wider">
          Practical Score
        </div>
      </div>
    </div>
  );
}

interface HorizonPillsProps {
  coveredHorizons: Horizon[];
  optimalHorizon: Horizon | null;
}

function HorizonPills({ coveredHorizons, optimalHorizon }: HorizonPillsProps) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {ALL_HORIZONS.map((horizon) => {
        const isCovered = coveredHorizons.includes(horizon);
        const isOptimal = horizon === optimalHorizon;

        return (
          <span
            key={horizon}
            className={cn(
              "px-2 py-0.5 text-xs font-mono rounded border transition-all",
              isCovered
                ? isOptimal
                  ? "bg-green-500/20 border-green-500/50 text-green-400"
                  : "bg-blue-500/10 border-blue-500/30 text-blue-400"
                : "bg-neutral-800/30 border-neutral-700 text-neutral-600"
            )}
          >
            {horizon}
            {isOptimal && " ★"}
          </span>
        );
      })}
    </div>
  );
}

interface ScoreBreakdownProps {
  components: {
    magnitudeScore: number;
    horizonScore: number;
    confidenceScore: number;
    bigMoveAccuracyScore: number;
  };
}

function ScoreBreakdown({ components }: ScoreBreakdownProps) {
  const items = [
    { label: "Magnitude", value: components.magnitudeScore, weight: "35%" },
    { label: "Horizons", value: components.horizonScore, weight: "20%" },
    { label: "Confidence", value: components.confidenceScore, weight: "20%" },
    { label: "Big Move Win", value: components.bigMoveAccuracyScore, weight: "25%" },
  ];

  return (
    <div className="space-y-2">
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-2">
          <span className="text-[10px] text-neutral-500 w-20">{item.label}</span>
          <div className="flex-1 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all",
                item.value >= 70
                  ? "bg-green-500"
                  : item.value >= 45
                  ? "bg-yellow-500"
                  : "bg-red-500"
              )}
              style={{ width: `${item.value}%` }}
            />
          </div>
          <span className="text-xs font-mono text-neutral-400 w-8">
            {item.value.toFixed(0)}
          </span>
          <span className="text-[10px] text-neutral-600 w-8">{item.weight}</span>
        </div>
      ))}
    </div>
  );
}

function PracticalMetricsSkeleton() {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <Skeleton className="h-5 w-40 bg-neutral-800" />
          <Skeleton className="h-6 w-24 bg-neutral-800" />
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        <Skeleton className="h-20 w-full bg-neutral-800" />
        <div className="grid grid-cols-3 gap-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="p-3 bg-neutral-800/50 rounded-lg">
              <Skeleton className="h-3 w-16 bg-neutral-700 mb-2" />
              <Skeleton className="h-6 w-12 bg-neutral-700" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function PracticalMetrics({
  asset,
  data,
  isLoading,
  error,
}: PracticalMetricsProps) {
  if (isLoading) {
    return <PracticalMetricsSkeleton />;
  }

  if (error) {
    return (
      <Card className="bg-neutral-900/50 border-red-900/50">
        <CardContent className="p-4">
          <div className="text-center text-red-500 text-sm py-4">
            <p>Failed to load practical metrics</p>
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
            No practical metrics available
          </div>
        </CardContent>
      </Card>
    );
  }

  const { signal } = data;
  const actionClasses = getActionabilityClasses(signal.practicalScore.actionability);

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-neutral-100">
              Practical Metrics
            </span>
            <Badge className="bg-cyan-500/10 border-cyan-500/30 text-cyan-500 text-xs">
              CME Utility
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Badge
              className={cn(
                "text-xs",
                getUrgencyClasses(signal.timeToAction.urgency)
              )}
            >
              {getUrgencyText(signal.timeToAction.urgency)}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0 space-y-4">
        {/* Actionability Indicator - The Main Event */}
        <ActionabilityIndicator
          level={signal.practicalScore.actionability}
          score={signal.practicalScore.score}
        />

        {/* Why Actionable/Not */}
        <div className="p-3 bg-neutral-800/30 rounded-lg">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-2">
            Why {signal.isActionable ? "Actionable" : "Not Actionable"}
          </div>
          <ul className="space-y-1">
            {signal.reasons.map((reason, i) => (
              <li key={i} className="flex items-start gap-2 text-xs">
                <span
                  className={cn(
                    "mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0",
                    signal.isActionable ? "bg-green-500" : "bg-neutral-600"
                  )}
                />
                <span className="text-neutral-300">{reason}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {/* Forecast Size */}
          <MetricBox
            label="Forecast Size"
            value={formatMoveSize(signal.magnitude.predictedMove, asset)}
            subtext={`Target: >${formatMoveSize(ASSET_MOVE_THRESHOLDS[asset], asset)}`}
            valueColor={
              signal.magnitude.isActionable ? "text-green-500" : "text-red-500"
            }
            size="lg"
          />

          {/* Big Move Win Rate */}
          <MetricBox
            label="Big Move Win Rate"
            value={`${signal.bigMoveWinRate.toFixed(1)}%`}
            subtext={`On moves >${formatMoveSize(ASSET_MOVE_THRESHOLDS[asset], asset)}`}
            valueColor={
              signal.bigMoveWinRate >= 60
                ? "text-green-500"
                : signal.bigMoveWinRate >= 50
                ? "text-yellow-500"
                : "text-red-500"
            }
          />

          {/* Position Size */}
          <MetricBox
            label="Suggested Position"
            value={`${signal.recommendedPositionSize}%`}
            subtext="Of max exposure"
            valueColor={
              signal.recommendedPositionSize >= 50
                ? "text-green-500"
                : signal.recommendedPositionSize >= 25
                ? "text-yellow-500"
                : "text-neutral-400"
            }
          />

          {/* Raw Confidence */}
          <MetricBox
            label="Model Confidence"
            value={`${signal.confidence.toFixed(0)}%`}
            subtext="Ensemble agreement"
            valueColor={
              signal.confidence >= 70
                ? "text-green-500"
                : signal.confidence >= 50
                ? "text-yellow-500"
                : "text-red-500"
            }
          />

          {/* Traditional Sharpe (for comparison) */}
          <MetricBox
            label="Sharpe Ratio"
            value={data.traditionalSharpe.toFixed(2)}
            subtext="Traditional metric"
            valueColor={
              data.traditionalSharpe >= 1.5
                ? "text-green-500"
                : data.traditionalSharpe >= 0.5
                ? "text-yellow-500"
                : "text-red-500"
            }
          />

          {/* Overall Win Rate (for comparison) */}
          <MetricBox
            label="Overall Win Rate"
            value={`${data.overallWinRate.toFixed(1)}%`}
            subtext="All moves"
            valueColor="text-neutral-400"
          />
        </div>

        {/* Horizon Coverage */}
        <div className="p-3 bg-neutral-800/30 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <div className="text-[10px] uppercase tracking-wider text-neutral-500">
              Horizon Diversity
            </div>
            <span className="text-xs font-mono text-neutral-400">
              {signal.horizonCoverage.coveragePercent.toFixed(0)}% covered
            </span>
          </div>
          <HorizonPills
            coveredHorizons={signal.horizonCoverage.coveredHorizons}
            optimalHorizon={signal.horizonCoverage.optimalHorizon}
          />
          <div className="flex gap-3 mt-2 text-[10px]">
            <span
              className={cn(
                signal.horizonCoverage.hasShortTerm
                  ? "text-green-500"
                  : "text-neutral-600"
              )}
            >
              {signal.horizonCoverage.hasShortTerm ? "✓" : "✗"} Short-term
            </span>
            <span
              className={cn(
                signal.horizonCoverage.hasMediumTerm
                  ? "text-green-500"
                  : "text-neutral-600"
              )}
            >
              {signal.horizonCoverage.hasMediumTerm ? "✓" : "✗"} Medium-term
            </span>
            <span
              className={cn(
                signal.horizonCoverage.hasLongTerm
                  ? "text-green-500"
                  : "text-neutral-600"
              )}
            >
              {signal.horizonCoverage.hasLongTerm ? "✓" : "✗"} Long-term
            </span>
          </div>
        </div>

        {/* Score Breakdown */}
        <div className="p-3 bg-neutral-800/30 rounded-lg">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-3">
            Score Breakdown
          </div>
          <ScoreBreakdown components={signal.practicalScore.components} />
        </div>

        {/* Time to Action */}
        <div className="flex items-center justify-between p-3 bg-neutral-800/30 rounded-lg">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
              Time to Action
            </div>
            <div className="text-sm text-neutral-300">
              {signal.timeToAction.reason}
            </div>
          </div>
          <div className="text-right">
            {signal.timeToAction.daysToOptimalEntry > 0 && (
              <div className="text-xs text-neutral-400">
                Optimal entry in{" "}
                <span className="font-mono text-blue-400">
                  {signal.timeToAction.daysToOptimalEntry}d
                </span>
              </div>
            )}
            <div className="text-xs text-neutral-500">
              Expires in{" "}
              <span className="font-mono">
                {signal.timeToAction.daysUntilExpiry}d
              </span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between pt-2 border-t border-neutral-800">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <div className={cn("w-2 h-2 rounded-full", actionClasses.dot)} />
              <span className="text-xs text-neutral-500">
                {signal.practicalScore.actionability === "high"
                  ? "Strong practical utility"
                  : signal.practicalScore.actionability === "medium"
                  ? "Moderate practical utility"
                  : "Limited practical utility"}
              </span>
            </div>
          </div>
          <span className="text-[10px] text-neutral-600 font-mono">
            {new Date(data.analyzedAt).toLocaleString()}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
