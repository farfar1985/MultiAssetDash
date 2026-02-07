"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  Target,
  TrendingUp,
  TrendingDown,
  Minus,
  Brain,
  Activity,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface ConfidenceWeight {
  method: string;
  weight: number;
  contribution: number;
  accuracy?: number;
}

export interface EnsembleConfidenceData {
  /** Final weighted confidence (0-100) */
  confidence: number;
  /** Signal direction */
  direction: "bullish" | "bearish" | "neutral";
  /** Individual method contributions */
  weights: ConfidenceWeight[];
  /** How many models agree */
  modelsAgreeing: number;
  /** Total models */
  modelsTotal: number;
  /** Ensemble method used */
  ensembleMethod: "accuracy-weighted" | "magnitude-weighted" | "stacking" | "bma" | "regime-adaptive";
}

export interface EnsembleConfidenceCardProps {
  data: EnsembleConfidenceData;
  /** Show detailed breakdown */
  showBreakdown?: boolean;
  /** Compact mode for smaller spaces */
  compact?: boolean;
  /** Custom class name */
  className?: string;
}

// ============================================================================
// Helper Functions
// ============================================================================

function getConfidenceLevel(confidence: number): {
  label: string;
  color: string;
  bgColor: string;
  borderColor: string;
} {
  if (confidence >= 80) {
    return {
      label: "Very High",
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
      borderColor: "border-emerald-500/30",
    };
  }
  if (confidence >= 65) {
    return {
      label: "High",
      color: "text-green-400",
      bgColor: "bg-green-500/10",
      borderColor: "border-green-500/30",
    };
  }
  if (confidence >= 50) {
    return {
      label: "Moderate",
      color: "text-amber-400",
      bgColor: "bg-amber-500/10",
      borderColor: "border-amber-500/30",
    };
  }
  return {
    label: "Low",
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/30",
  };
}

function getMethodLabel(method: EnsembleConfidenceData["ensembleMethod"]): string {
  const labels: Record<EnsembleConfidenceData["ensembleMethod"], string> = {
    "accuracy-weighted": "Accuracy-Weighted",
    "magnitude-weighted": "Magnitude-Weighted",
    stacking: "Stacking Meta-Learner",
    bma: "Bayesian Model Avg",
    "regime-adaptive": "Regime-Adaptive",
  };
  return labels[method];
}

// ============================================================================
// Component
// ============================================================================

export function EnsembleConfidenceCard({
  data,
  showBreakdown = true,
  compact = false,
  className,
}: EnsembleConfidenceCardProps) {
  const confidenceLevel = useMemo(() => getConfidenceLevel(data.confidence), [data.confidence]);
  const agreementPercent = useMemo(
    () => Math.round((data.modelsAgreeing / data.modelsTotal) * 100),
    [data.modelsAgreeing, data.modelsTotal]
  );

  const directionConfig = {
    bullish: {
      icon: TrendingUp,
      color: "text-green-400",
      label: "Bullish",
    },
    bearish: {
      icon: TrendingDown,
      color: "text-red-400",
      label: "Bearish",
    },
    neutral: {
      icon: Minus,
      color: "text-amber-400",
      label: "Neutral",
    },
  };

  const direction = directionConfig[data.direction];
  const DirectionIcon = direction.icon;

  if (compact) {
    return (
      <div
        className={cn(
          "flex items-center gap-3 p-3 rounded-xl border",
          "bg-neutral-900/50",
          confidenceLevel.borderColor,
          className
        )}
      >
        <div className={cn("p-2 rounded-lg", confidenceLevel.bgColor)}>
          <Target className={cn("w-5 h-5", confidenceLevel.color)} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={cn("text-2xl font-bold font-mono", confidenceLevel.color)}>
              {data.confidence}%
            </span>
            <Badge className={cn("text-[10px]", direction.color, "bg-current/10 border-current/20")}>
              <DirectionIcon className="w-3 h-3 mr-1" />
              {direction.label}
            </Badge>
          </div>
          <div className="text-xs text-neutral-500 truncate">
            {getMethodLabel(data.ensembleMethod)} Confidence
          </div>
        </div>
      </div>
    );
  }

  return (
    <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Brain className="w-4 h-4 text-purple-400" />
            Ensemble Confidence
          </CardTitle>
          <Badge className="bg-purple-500/10 text-purple-400 border-purple-500/30 text-xs">
            {getMethodLabel(data.ensembleMethod)}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0 space-y-4">
        {/* Main Confidence Display */}
        <div className={cn("p-4 rounded-xl border", confidenceLevel.bgColor, confidenceLevel.borderColor)}>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className={cn("p-2.5 rounded-lg", confidenceLevel.bgColor)}>
                <Target className={cn("w-6 h-6", confidenceLevel.color)} />
              </div>
              <div>
                <div className={cn("text-3xl font-bold font-mono", confidenceLevel.color)}>
                  {data.confidence}%
                </div>
                <div className="text-xs text-neutral-500">
                  {confidenceLevel.label} Confidence
                </div>
              </div>
            </div>
            <div className="text-right">
              <Badge
                className={cn(
                  "mb-1 px-3 py-1",
                  direction.color,
                  "bg-current/10 border-current/20"
                )}
              >
                <DirectionIcon className="w-3.5 h-3.5 mr-1.5" />
                {direction.label}
              </Badge>
              <div className="text-[10px] text-neutral-500">Signal Direction</div>
            </div>
          </div>

          {/* Confidence Bar */}
          <div className="relative h-3 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className={cn(
                "absolute inset-y-0 left-0 rounded-full transition-all duration-500",
                data.confidence >= 65 ? "bg-gradient-to-r from-green-500 to-emerald-400" :
                data.confidence >= 50 ? "bg-gradient-to-r from-amber-500 to-yellow-400" :
                "bg-gradient-to-r from-red-500 to-rose-400"
              )}
              style={{ width: `${data.confidence}%` }}
            />
            {/* Threshold markers */}
            <div className="absolute top-0 bottom-0 left-1/2 w-px bg-neutral-600" />
            <div className="absolute top-0 bottom-0 left-[65%] w-px bg-neutral-700" />
            <div className="absolute top-0 bottom-0 left-[80%] w-px bg-neutral-700" />
          </div>
          <div className="flex justify-between text-[9px] text-neutral-600 mt-1">
            <span>0%</span>
            <span>50%</span>
            <span>65%</span>
            <span>80%</span>
            <span>100%</span>
          </div>
        </div>

        {/* Model Agreement */}
        <div className="flex items-center justify-between p-3 bg-neutral-800/30 rounded-lg">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <span className="text-sm text-neutral-400">Model Agreement</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold font-mono text-neutral-100">
              {data.modelsAgreeing.toLocaleString()}/{data.modelsTotal.toLocaleString()}
            </span>
            <Badge
              className={cn(
                "text-xs",
                agreementPercent >= 75
                  ? "bg-green-500/10 text-green-400 border-green-500/30"
                  : agreementPercent >= 60
                  ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                  : "bg-red-500/10 text-red-400 border-red-500/30"
              )}
            >
              {agreementPercent}%
            </Badge>
          </div>
        </div>

        {/* Method Breakdown */}
        {showBreakdown && data.weights.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-neutral-500 uppercase tracking-wider mb-2">
              Weight Breakdown
            </div>
            {data.weights.map((w) => (
              <div
                key={w.method}
                className="flex items-center gap-3 p-2 bg-neutral-800/20 rounded-lg"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-medium text-neutral-300 truncate">
                      {w.method}
                    </span>
                    <span className="text-xs font-mono text-neutral-400">
                      {(w.weight * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-purple-500 rounded-full"
                      style={{ width: `${w.weight * 100}%` }}
                    />
                  </div>
                </div>
                {w.accuracy !== undefined && (
                  <div className="text-right shrink-0">
                    <div className="text-xs font-mono text-green-400">
                      {w.accuracy.toFixed(1)}%
                    </div>
                    <div className="text-[9px] text-neutral-600">acc</div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default EnsembleConfidenceCard;
