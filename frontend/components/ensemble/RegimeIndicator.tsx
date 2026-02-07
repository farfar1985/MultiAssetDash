"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  Wind,
  Sun,
  AlertTriangle,
  Gauge,
  ArrowRight,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export type MarketRegime = "bull" | "bear" | "sideways" | "high-volatility" | "low-volatility";

export interface RegimeData {
  /** Current detected regime */
  regime: MarketRegime;
  /** Confidence in regime detection (0-1) */
  confidence: number;
  /** Regime probability distribution */
  probabilities: {
    bull: number;
    bear: number;
    sideways: number;
  };
  /** Days in current regime */
  daysInRegime: number;
  /** Historical accuracy of this regime */
  historicalAccuracy?: number;
  /** Volatility level (annualized %) */
  volatility: number;
  /** Trend strength indicator (-1 to 1) */
  trendStrength: number;
}

export interface RegimeIndicatorProps {
  data: RegimeData;
  /** Show probability breakdown */
  showProbabilities?: boolean;
  /** Compact mode */
  compact?: boolean;
  /** Size variant */
  size?: "sm" | "md" | "lg";
  /** Custom class name */
  className?: string;
}

// ============================================================================
// Helper Functions
// ============================================================================

interface RegimeShiftWarning {
  isWarning: boolean;
  targetRegime: "bull" | "bear" | "sideways";
  probability: number;
  currentProbability: number;
  urgency: "low" | "medium" | "high";
}

/**
 * Detect potential regime shifts based on probability distribution
 * Warning conditions:
 * - Second highest probability >= 30%
 * - Gap between current and competing regime < 15%
 */
function detectRegimeShiftWarning(
  currentRegime: MarketRegime,
  probabilities: { bull: number; bear: number; sideways: number }
): RegimeShiftWarning {
  // Map current regime to probability key
  const regimeToProb: Record<string, "bull" | "bear" | "sideways"> = {
    bull: "bull",
    bear: "bear",
    sideways: "sideways",
    "high-volatility": "sideways", // Map volatility regimes to sideways for probability comparison
    "low-volatility": "sideways",
  };

  const currentProbKey = regimeToProb[currentRegime] || "sideways";
  const currentProb = probabilities[currentProbKey];

  // Find the highest competing regime probability
  const competitors = Object.entries(probabilities)
    .filter(([key]) => key !== currentProbKey)
    .sort(([, a], [, b]) => b - a);

  const [topCompetitorKey, topCompetitorProb] = competitors[0] as [
    "bull" | "bear" | "sideways",
    number
  ];

  // Calculate gap and determine warning status
  const gap = currentProb - topCompetitorProb;

  // Determine urgency based on gap and competitor probability
  let urgency: "low" | "medium" | "high" = "low";
  let isWarning = false;

  if (topCompetitorProb >= 0.40 && gap < 0.10) {
    urgency = "high";
    isWarning = true;
  } else if (topCompetitorProb >= 0.35 && gap < 0.15) {
    urgency = "medium";
    isWarning = true;
  } else if (topCompetitorProb >= 0.30 && gap < 0.20) {
    urgency = "low";
    isWarning = true;
  }

  return {
    isWarning,
    targetRegime: topCompetitorKey,
    probability: topCompetitorProb,
    currentProbability: currentProb,
    urgency,
  };
}

function getRegimeConfig(regime: MarketRegime) {
  const configs: Record<
    MarketRegime,
    {
      label: string;
      icon: typeof TrendingUp;
      color: string;
      bgColor: string;
      borderColor: string;
      description: string;
    }
  > = {
    bull: {
      label: "Bull Market",
      icon: TrendingUp,
      color: "text-green-400",
      bgColor: "bg-green-500/10",
      borderColor: "border-green-500/30",
      description: "Upward trending, risk-on environment",
    },
    bear: {
      label: "Bear Market",
      icon: TrendingDown,
      color: "text-red-400",
      bgColor: "bg-red-500/10",
      borderColor: "border-red-500/30",
      description: "Downward trending, risk-off environment",
    },
    sideways: {
      label: "Sideways",
      icon: Minus,
      color: "text-amber-400",
      bgColor: "bg-amber-500/10",
      borderColor: "border-amber-500/30",
      description: "Range-bound, low directional bias",
    },
    "high-volatility": {
      label: "High Volatility",
      icon: Zap,
      color: "text-orange-400",
      bgColor: "bg-orange-500/10",
      borderColor: "border-orange-500/30",
      description: "Elevated volatility, uncertain direction",
    },
    "low-volatility": {
      label: "Low Volatility",
      icon: Wind,
      color: "text-blue-400",
      bgColor: "bg-blue-500/10",
      borderColor: "border-blue-500/30",
      description: "Calm markets, compressed ranges",
    },
  };
  return configs[regime];
}

// ============================================================================
// Component
// ============================================================================

export function RegimeIndicator({
  data,
  showProbabilities = true,
  compact = false,
  size = "md",
  className,
}: RegimeIndicatorProps) {
  const config = useMemo(() => getRegimeConfig(data.regime), [data.regime]);
  const Icon = config.icon;

  const confidencePercent = Math.round(data.confidence * 100);
  const isHighConfidence = data.confidence >= 0.75;

  // Detect potential regime shift warnings
  const shiftWarning = useMemo(
    () => detectRegimeShiftWarning(data.regime, data.probabilities),
    [data.regime, data.probabilities]
  );

  const targetConfig = useMemo(
    () =>
      shiftWarning.isWarning
        ? getRegimeConfig(shiftWarning.targetRegime as MarketRegime)
        : null,
    [shiftWarning]
  );

  // Size-based classes
  const sizeClasses = {
    sm: {
      icon: "w-4 h-4",
      iconContainer: "p-1.5",
      title: "text-sm",
      subtitle: "text-[10px]",
    },
    md: {
      icon: "w-5 h-5",
      iconContainer: "p-2",
      title: "text-base",
      subtitle: "text-xs",
    },
    lg: {
      icon: "w-6 h-6",
      iconContainer: "p-2.5",
      title: "text-lg",
      subtitle: "text-sm",
    },
  };

  const sizeClass = sizeClasses[size];

  if (compact) {
    return (
      <div className={cn("space-y-2", className)}>
        <div
          className={cn(
            "flex items-center gap-3 p-3 rounded-xl border",
            config.bgColor,
            config.borderColor
          )}
        >
          <div className={cn("rounded-lg", config.bgColor, sizeClass.iconContainer)}>
            <Icon className={cn(sizeClass.icon, config.color)} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className={cn("font-semibold", config.color, sizeClass.title)}>
                {config.label}
              </span>
              <Badge
                className={cn(
                  "text-[9px] px-1.5",
                  isHighConfidence
                    ? "bg-green-500/10 text-green-400 border-green-500/30"
                    : "bg-amber-500/10 text-amber-400 border-amber-500/30"
                )}
              >
                {confidencePercent}%
              </Badge>
            </div>
            <div className={cn("text-neutral-500", sizeClass.subtitle)}>
              {data.daysInRegime} days
            </div>
          </div>
        </div>

        {/* Early Warning - Compact */}
        {shiftWarning.isWarning && targetConfig && (
          <div
            className={cn(
              "flex items-center gap-2 p-2 rounded-lg border animate-pulse",
              shiftWarning.urgency === "high"
                ? "bg-red-500/10 border-red-500/30"
                : shiftWarning.urgency === "medium"
                ? "bg-orange-500/10 border-orange-500/30"
                : "bg-amber-500/10 border-amber-500/30"
            )}
          >
            <AlertTriangle
              className={cn(
                "w-3.5 h-3.5",
                shiftWarning.urgency === "high"
                  ? "text-red-400"
                  : shiftWarning.urgency === "medium"
                  ? "text-orange-400"
                  : "text-amber-400"
              )}
            />
            <span className="text-[10px] text-neutral-300">
              Shift to{" "}
              <span className={targetConfig.color}>{targetConfig.label}</span>
              {" "}({(shiftWarning.probability * 100).toFixed(0)}%)
            </span>
          </div>
        )}
      </div>
    );
  }

  return (
    <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Market Regime
          </CardTitle>
          <Badge className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30 text-xs">
            HMM Detected
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0 space-y-4">
        {/* Main Regime Display */}
        <div className={cn("p-4 rounded-xl border", config.bgColor, config.borderColor)}>
          <div className="flex items-center gap-4">
            <div className={cn("p-3 rounded-xl", config.bgColor)}>
              <Icon className={cn("w-8 h-8", config.color)} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <h3 className={cn("text-xl font-bold", config.color)}>{config.label}</h3>
                <Badge
                  className={cn(
                    "text-xs",
                    isHighConfidence
                      ? "bg-green-500/10 text-green-400 border-green-500/30"
                      : "bg-amber-500/10 text-amber-400 border-amber-500/30"
                  )}
                >
                  {confidencePercent}% confidence
                </Badge>
              </div>
              <p className="text-sm text-neutral-400">{config.description}</p>
            </div>
          </div>

          {/* Regime Stats */}
          <div className="grid grid-cols-3 gap-3 mt-4 pt-4 border-t border-neutral-700/50">
            <div className="text-center">
              <div className="text-lg font-bold font-mono text-neutral-100">
                {data.daysInRegime}
              </div>
              <div className="text-[10px] text-neutral-500">Days in Regime</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold font-mono text-neutral-100">
                {data.volatility.toFixed(1)}%
              </div>
              <div className="text-[10px] text-neutral-500">Volatility</div>
            </div>
            <div className="text-center">
              <div
                className={cn(
                  "text-lg font-bold font-mono",
                  data.trendStrength > 0.3
                    ? "text-green-400"
                    : data.trendStrength < -0.3
                    ? "text-red-400"
                    : "text-amber-400"
                )}
              >
                {data.trendStrength >= 0 ? "+" : ""}
                {(data.trendStrength * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-neutral-500">Trend Strength</div>
            </div>
          </div>
        </div>

        {/* Early Warning Alert */}
        {shiftWarning.isWarning && targetConfig && (
          <div
            className={cn(
              "p-4 rounded-xl border",
              shiftWarning.urgency === "high"
                ? "bg-red-500/10 border-red-500/30"
                : shiftWarning.urgency === "medium"
                ? "bg-orange-500/10 border-orange-500/30"
                : "bg-amber-500/10 border-amber-500/30"
            )}
          >
            <div className="flex items-start gap-3">
              <div
                className={cn(
                  "p-2 rounded-lg animate-pulse",
                  shiftWarning.urgency === "high"
                    ? "bg-red-500/20"
                    : shiftWarning.urgency === "medium"
                    ? "bg-orange-500/20"
                    : "bg-amber-500/20"
                )}
              >
                <AlertTriangle
                  className={cn(
                    "w-5 h-5",
                    shiftWarning.urgency === "high"
                      ? "text-red-400"
                      : shiftWarning.urgency === "medium"
                      ? "text-orange-400"
                      : "text-amber-400"
                  )}
                />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className={cn(
                      "text-sm font-semibold",
                      shiftWarning.urgency === "high"
                        ? "text-red-400"
                        : shiftWarning.urgency === "medium"
                        ? "text-orange-400"
                        : "text-amber-400"
                    )}
                  >
                    Regime Shift Warning
                  </span>
                  <Badge
                    className={cn(
                      "text-[9px] uppercase",
                      shiftWarning.urgency === "high"
                        ? "bg-red-500/20 text-red-400 border-red-500/30"
                        : shiftWarning.urgency === "medium"
                        ? "bg-orange-500/20 text-orange-400 border-orange-500/30"
                        : "bg-amber-500/20 text-amber-400 border-amber-500/30"
                    )}
                  >
                    {shiftWarning.urgency}
                  </Badge>
                </div>
                <p className="text-xs text-neutral-400 mb-3">
                  Probability of transitioning to{" "}
                  <span className={targetConfig.color}>{targetConfig.label}</span> is rising.
                  Monitor for potential regime change.
                </p>
                <div className="flex items-center gap-4 text-xs">
                  <div className="flex items-center gap-2">
                    <Icon className={cn("w-4 h-4", config.color)} />
                    <span className="text-neutral-400">Current:</span>
                    <span className={cn("font-mono font-medium", config.color)}>
                      {(shiftWarning.currentProbability * 100).toFixed(0)}%
                    </span>
                  </div>
                  <ArrowRight className="w-4 h-4 text-neutral-600" />
                  <div className="flex items-center gap-2">
                    <targetConfig.icon className={cn("w-4 h-4", targetConfig.color)} />
                    <span className="text-neutral-400">Target:</span>
                    <span className={cn("font-mono font-medium", targetConfig.color)}>
                      {(shiftWarning.probability * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Trend Strength Gauge */}
        <div className="p-3 bg-neutral-800/30 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Gauge className="w-4 h-4 text-neutral-400" />
              <span className="text-xs text-neutral-500">Trend Strength</span>
            </div>
            <span
              className={cn(
                "text-sm font-mono font-medium",
                data.trendStrength > 0.3
                  ? "text-green-400"
                  : data.trendStrength < -0.3
                  ? "text-red-400"
                  : "text-amber-400"
              )}
            >
              {Math.abs(data.trendStrength) < 0.3
                ? "Weak"
                : Math.abs(data.trendStrength) < 0.6
                ? "Moderate"
                : "Strong"}
            </span>
          </div>
          <div className="relative h-3 bg-neutral-800 rounded-full overflow-hidden">
            {/* Center marker */}
            <div className="absolute inset-y-0 left-1/2 w-0.5 bg-neutral-600 z-10" />
            {/* Trend indicator */}
            <div
              className={cn(
                "absolute inset-y-0 transition-all duration-500 rounded-full",
                data.trendStrength > 0 ? "bg-green-500" : "bg-red-500"
              )}
              style={{
                left: data.trendStrength > 0 ? "50%" : `${50 + data.trendStrength * 50}%`,
                width: `${Math.abs(data.trendStrength) * 50}%`,
              }}
            />
          </div>
          <div className="flex justify-between text-[9px] text-neutral-600 mt-1">
            <span>Bearish</span>
            <span>Neutral</span>
            <span>Bullish</span>
          </div>
        </div>

        {/* Probability Distribution */}
        {showProbabilities && (
          <div className="space-y-2">
            <div className="text-xs text-neutral-500 uppercase tracking-wider">
              Regime Probabilities
            </div>
            <div className="space-y-2">
              {/* Bull */}
              <div className="flex items-center gap-3">
                <div className="w-16 flex items-center gap-1">
                  <TrendingUp className="w-3.5 h-3.5 text-green-400" />
                  <span className="text-xs text-neutral-400">Bull</span>
                </div>
                <div className="flex-1 h-2 bg-neutral-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green-500 rounded-full transition-all duration-500"
                    style={{ width: `${data.probabilities.bull * 100}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-neutral-300 w-12 text-right">
                  {(data.probabilities.bull * 100).toFixed(0)}%
                </span>
              </div>
              {/* Bear */}
              <div className="flex items-center gap-3">
                <div className="w-16 flex items-center gap-1">
                  <TrendingDown className="w-3.5 h-3.5 text-red-400" />
                  <span className="text-xs text-neutral-400">Bear</span>
                </div>
                <div className="flex-1 h-2 bg-neutral-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-red-500 rounded-full transition-all duration-500"
                    style={{ width: `${data.probabilities.bear * 100}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-neutral-300 w-12 text-right">
                  {(data.probabilities.bear * 100).toFixed(0)}%
                </span>
              </div>
              {/* Sideways */}
              <div className="flex items-center gap-3">
                <div className="w-16 flex items-center gap-1">
                  <Minus className="w-3.5 h-3.5 text-amber-400" />
                  <span className="text-xs text-neutral-400">Side</span>
                </div>
                <div className="flex-1 h-2 bg-neutral-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-amber-500 rounded-full transition-all duration-500"
                    style={{ width: `${data.probabilities.sideways * 100}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-neutral-300 w-12 text-right">
                  {(data.probabilities.sideways * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Historical Accuracy (if available) */}
        {data.historicalAccuracy !== undefined && (
          <div className="flex items-center justify-between p-3 bg-neutral-800/30 rounded-lg">
            <div className="flex items-center gap-2">
              <Sun className="w-4 h-4 text-amber-400" />
              <span className="text-xs text-neutral-400">
                Model accuracy in this regime
              </span>
            </div>
            <Badge
              className={cn(
                "text-xs font-mono",
                data.historicalAccuracy >= 65
                  ? "bg-green-500/10 text-green-400 border-green-500/30"
                  : data.historicalAccuracy >= 55
                  ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                  : "bg-red-500/10 text-red-400 border-red-500/30"
              )}
            >
              {data.historicalAccuracy.toFixed(1)}%
            </Badge>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default RegimeIndicator;
