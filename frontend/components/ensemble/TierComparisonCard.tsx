"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { TierComparisonData, TierPrediction } from "@/lib/api";
import {
  Layers,
  TrendingUp,
  TrendingDown,
  Minus,
  CheckCircle2,
  Target,
  Brain,
  Sparkles,
  AlertTriangle,
  Activity,
  BarChart3,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface TierComparisonCardProps {
  data: TierComparisonData;
  compact?: boolean;
}

export type { TierComparisonData };

// ============================================================================
// Helper Functions
// ============================================================================

function getSignalColor(signal: string): string {
  switch (signal) {
    case "BULLISH":
      return "text-green-500";
    case "BEARISH":
      return "text-red-500";
    default:
      return "text-neutral-400";
  }
}

function getSignalBg(signal: string): string {
  switch (signal) {
    case "BULLISH":
      return "bg-green-500/10 border-green-500/30";
    case "BEARISH":
      return "bg-red-500/10 border-red-500/30";
    default:
      return "bg-neutral-500/10 border-neutral-500/30";
  }
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return "text-emerald-400";
  if (confidence >= 0.65) return "text-green-400";
  if (confidence >= 0.5) return "text-amber-400";
  return "text-red-400";
}

function SignalIcon({ signal }: { signal: string }) {
  switch (signal) {
    case "BULLISH":
      return <TrendingUp className="w-4 h-4" />;
    case "BEARISH":
      return <TrendingDown className="w-4 h-4" />;
    default:
      return <Minus className="w-4 h-4" />;
  }
}

// ============================================================================
// Tier Card Component
// ============================================================================

interface TierCardProps {
  tier: number;
  name: string;
  description: string;
  icon: React.ReactNode;
  prediction: TierPrediction;
  color: string;
  borderColor: string;
}

function TierCard({
  tier,
  name,
  description,
  icon,
  prediction,
  color,
  borderColor,
}: TierCardProps) {
  const confidence = prediction.confidence ?? 0;

  return (
    <div
      className={cn(
        "rounded-lg border p-4 transition-all",
        borderColor,
        "bg-neutral-900/50 hover:bg-neutral-900/70"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={cn("p-1.5 rounded-md", color)}>{icon}</div>
          <div>
            <div className="text-sm font-semibold text-neutral-100">
              Tier {tier}
            </div>
            <div className="text-xs text-neutral-500">{name}</div>
          </div>
        </div>
        <Badge
          className={cn(
            "text-xs gap-1",
            getSignalBg(prediction.signal),
            getSignalColor(prediction.signal)
          )}
        >
          <SignalIcon signal={prediction.signal} />
          {prediction.signal}
        </Badge>
      </div>

      {/* Confidence Bar */}
      <div className="mb-3">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-neutral-500">Confidence</span>
          <span className={cn("font-mono font-bold", getConfidenceColor(confidence))}>
            {(confidence * 100).toFixed(0)}%
          </span>
        </div>
        <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
          <div
            className={cn(
              "h-full rounded-full transition-all",
              confidence >= 0.65
                ? "bg-gradient-to-r from-green-500 to-emerald-500"
                : confidence >= 0.5
                ? "bg-gradient-to-r from-amber-500 to-yellow-500"
                : "bg-gradient-to-r from-red-500 to-orange-500"
            )}
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Description */}
      <p className="text-xs text-neutral-500 mb-3">{description}</p>

      {/* Tier-specific details */}
      {tier === 1 && prediction.weights && (
        <TierOneDetails weights={prediction.weights} />
      )}
      {tier === 2 && <TierTwoDetails prediction={prediction} />}
      {tier === 3 && <TierThreeDetails prediction={prediction} />}
    </div>
  );
}

// ============================================================================
// Tier-Specific Details
// ============================================================================

function TierOneDetails({ weights }: { weights: Record<string, number> }) {
  const entries = Object.entries(weights).slice(0, 3);
  const total = entries.reduce((sum, [, v]) => sum + v, 0);

  return (
    <div className="space-y-1.5">
      <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
        Weight Distribution
      </div>
      {entries.map(([key, value]) => {
        const name = key.replace(/_/g, " ").replace(/weighted/i, "").trim();
        const pct = total > 0 ? (value / total) * 100 : 0;
        return (
          <div key={key} className="flex items-center gap-2">
            <span className="text-[10px] text-neutral-500 w-16 truncate capitalize">
              {name}
            </span>
            <div className="flex-1 h-1 bg-neutral-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500/70 rounded-full"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="text-[10px] font-mono text-neutral-400 w-8 text-right">
              {pct.toFixed(0)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

function TierTwoDetails({ prediction }: { prediction: TierPrediction }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {prediction.regime && (
        <div className="p-2 bg-neutral-800/50 rounded text-center">
          <div className="text-[10px] text-neutral-500 mb-0.5">Regime</div>
          <div className="text-xs font-mono font-bold text-purple-400 capitalize">
            {prediction.regime}
          </div>
        </div>
      )}
      {prediction.uncertainty !== undefined && (
        <div className="p-2 bg-neutral-800/50 rounded text-center">
          <div className="text-[10px] text-neutral-500 mb-0.5">Uncertainty</div>
          <div className="text-xs font-mono font-bold text-amber-400">
            {(prediction.uncertainty * 100).toFixed(0)}%
          </div>
        </div>
      )}
      {prediction.interval && (
        <div className="col-span-2 p-2 bg-neutral-800/50 rounded">
          <div className="text-[10px] text-neutral-500 mb-0.5">Interval (90%)</div>
          <div className="flex justify-between">
            <span className="text-xs font-mono text-red-400">
              {prediction.interval.lower.toFixed(2)}%
            </span>
            <span className="text-xs text-neutral-500">to</span>
            <span className="text-xs font-mono text-green-400">
              +{prediction.interval.upper.toFixed(2)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function TierThreeDetails({ prediction }: { prediction: TierPrediction }) {
  const quantiles = prediction.quantiles;

  return (
    <div className="space-y-2">
      {quantiles && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
            Quantile Forecasts
          </div>
          <div className="flex items-center justify-between gap-1">
            {["0.1", "0.25", "0.5", "0.75", "0.9"].map((q) => {
              const val = quantiles[q];
              if (val === undefined) return null;
              return (
                <div key={q} className="flex-1 text-center">
                  <div className="text-[9px] text-neutral-500">Q{parseFloat(q) * 100}</div>
                  <div
                    className={cn(
                      "text-[10px] font-mono font-bold",
                      val >= 0 ? "text-green-400" : "text-red-400"
                    )}
                  >
                    {val >= 0 ? "+" : ""}
                    {val.toFixed(1)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
      {prediction.explorationBonus !== undefined && (
        <div className="flex items-center justify-between p-2 bg-neutral-800/50 rounded">
          <span className="text-[10px] text-neutral-500">Exploration Bonus</span>
          <span className="text-xs font-mono font-bold text-cyan-400">
            +{(prediction.explorationBonus * 100).toFixed(0)}%
          </span>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Consensus Banner
// ============================================================================

interface ConsensusBannerProps {
  consensus: TierComparisonData["consensus"];
}

function ConsensusBanner({ consensus }: ConsensusBannerProps) {
  const allAgree = consensus.tiersAgreeing === consensus.totalTiers;
  const majority = consensus.tiersAgreeing >= 2;

  return (
    <div
      className={cn(
        "rounded-lg border p-4 mb-4",
        allAgree
          ? "bg-gradient-to-r from-emerald-900/20 via-green-900/20 to-emerald-900/20 border-emerald-500/30"
          : majority
          ? "bg-gradient-to-r from-blue-900/20 via-purple-900/20 to-blue-900/20 border-blue-500/30"
          : "bg-neutral-900/50 border-neutral-800"
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {allAgree ? (
            <div className="p-2 bg-emerald-500/20 rounded-lg border border-emerald-500/30">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            </div>
          ) : majority ? (
            <div className="p-2 bg-blue-500/20 rounded-lg border border-blue-500/30">
              <Activity className="w-5 h-5 text-blue-400" />
            </div>
          ) : (
            <div className="p-2 bg-amber-500/20 rounded-lg border border-amber-500/30">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            </div>
          )}
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-neutral-100">
                {allAgree
                  ? "Full Consensus"
                  : majority
                  ? "Majority Agreement"
                  : "Mixed Signals"}
              </span>
              <Badge
                className={cn(
                  "text-xs gap-1",
                  getSignalBg(consensus.signal),
                  getSignalColor(consensus.signal)
                )}
              >
                <SignalIcon signal={consensus.signal} />
                {consensus.signal}
              </Badge>
            </div>
            <span className="text-xs text-neutral-500">
              {consensus.tiersAgreeing} of {consensus.totalTiers} tiers agree
              {allAgree && " - High conviction signal"}
            </span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold font-mono text-neutral-100">
            {(consensus.agreement * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-neutral-500">Agreement</div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function TierComparisonCard({ data, compact = false }: TierComparisonCardProps) {
  const tierConfigs = useMemo(
    () => [
      {
        tier: 1,
        name: "Production",
        description: "Accuracy & magnitude-weighted voting across horizons",
        icon: <Target className="w-4 h-4 text-blue-400" />,
        color: "bg-blue-500/20",
        borderColor: "border-blue-500/20",
        prediction: data.tier1,
      },
      {
        tier: 2,
        name: "Advanced",
        description: "BMA, regime-adaptive, and conformal prediction",
        icon: <Brain className="w-4 h-4 text-purple-400" />,
        color: "bg-purple-500/20",
        borderColor: "border-purple-500/20",
        prediction: data.tier2,
      },
      {
        tier: 3,
        name: "Research",
        description: "Thompson sampling, attention, and quantile methods",
        icon: <Sparkles className="w-4 h-4 text-cyan-400" />,
        color: "bg-cyan-500/20",
        borderColor: "border-cyan-500/20",
        prediction: data.tier3,
      },
    ],
    [data]
  );

  if (compact) {
    return (
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-purple-400" />
              <span className="text-sm font-semibold text-neutral-100">
                Tier Comparison
              </span>
            </div>
            <Badge
              className={cn(
                "text-xs gap-1",
                getSignalBg(data.consensus.signal),
                getSignalColor(data.consensus.signal)
              )}
            >
              <SignalIcon signal={data.consensus.signal} />
              {data.consensus.signal}
            </Badge>
          </div>
          <div className="flex gap-2">
            {tierConfigs.map(({ tier, prediction, color }) => (
              <div
                key={tier}
                className={cn(
                  "flex-1 p-2 rounded-md text-center",
                  color,
                  "border border-neutral-800"
                )}
              >
                <div className="text-[10px] text-neutral-500 mb-0.5">
                  Tier {tier}
                </div>
                <div
                  className={cn(
                    "text-xs font-bold",
                    getSignalColor(prediction.signal)
                  )}
                >
                  {prediction.signal.charAt(0)}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-purple-500/20 rounded-lg border border-purple-500/30">
              <Layers className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-neutral-100">
                Ensemble Tier Comparison
              </h3>
              <p className="text-xs text-neutral-500">
                {data.asset_name.replace(/_/g, " ")} - All tier predictions
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-neutral-500" />
            <span className="text-xs text-neutral-500 font-mono">
              {new Date(data.timestamp).toLocaleTimeString()}
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        {/* Consensus Banner */}
        <ConsensusBanner consensus={data.consensus} />

        {/* Tier Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {tierConfigs.map((config) => (
            <TierCard key={config.tier} {...config} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
