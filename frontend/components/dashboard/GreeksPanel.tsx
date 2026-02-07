"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Clock,
  Zap,
  Wind,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface GreekData {
  delta: number;      // -1 to 1, directional exposure
  gamma: number;      // Rate of delta change (typically 0-0.1)
  vega: number;       // Volatility sensitivity ($ per 1% vol change)
  theta: number;      // Time decay ($ per day, usually negative)
}

interface GreeksPanelProps {
  data?: GreekData;
  className?: string;
  compact?: boolean;
}

// ============================================================================
// Mock Data
// ============================================================================

const MOCK_GREEKS: GreekData = {
  delta: 0.42,        // Net long exposure
  gamma: 0.028,       // Moderate convexity
  vega: 12450,        // $12,450 per 1% vol move
  theta: -3280,       // Losing $3,280/day to time decay
};

// ============================================================================
// Greek Card Component
// ============================================================================

interface GreekCardProps {
  name: string;
  symbol: string;
  value: number;
  displayValue: string;
  description: string;
  tooltip: string;
  icon: typeof Activity;
  getColor: (value: number) => { text: string; bg: string; border: string };
  getBarWidth: (value: number) => number;
  showNegativeBar?: boolean;
}

function GreekCard({
  name,
  symbol,
  value,
  displayValue,
  description,
  tooltip,
  icon: Icon,
  getColor,
  getBarWidth,
  showNegativeBar = false,
}: GreekCardProps) {
  const colors = getColor(value);
  const barWidth = getBarWidth(value);

  return (
    <div
      className={cn(
        "p-4 bg-neutral-800/50 rounded-lg border transition-all hover:border-neutral-600",
        colors.border
      )}
      title={tooltip}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={cn("p-1.5 rounded-lg", colors.bg)}>
            <Icon className={cn("w-4 h-4", colors.text)} />
          </div>
          <div>
            <span className="text-sm font-semibold text-neutral-200">{name}</span>
            <span className={cn("ml-2 text-lg font-mono font-bold", colors.text)}>
              {symbol}
            </span>
          </div>
        </div>
      </div>

      {/* Value */}
      <div className={cn("text-2xl font-mono font-bold mb-2", colors.text)}>
        {displayValue}
      </div>

      {/* Description */}
      <p className="text-xs text-neutral-500 mb-3">{description}</p>

      {/* Bar Visualization */}
      <div className="relative h-2 bg-neutral-700 rounded-full overflow-hidden">
        {showNegativeBar ? (
          // Centered bar for values that can be positive or negative (like delta)
          <div className="absolute inset-0 flex items-center">
            <div className="w-1/2 flex justify-end">
              {value < 0 && (
                <div
                  className={cn("h-full rounded-l-full", colors.bg.replace("/10", ""))}
                  style={{ width: `${Math.abs(barWidth)}%` }}
                />
              )}
            </div>
            <div className="w-px h-full bg-neutral-500" />
            <div className="w-1/2">
              {value >= 0 && (
                <div
                  className={cn("h-full rounded-r-full", colors.bg.replace("/10", ""))}
                  style={{ width: `${barWidth}%` }}
                />
              )}
            </div>
          </div>
        ) : (
          // Standard left-to-right bar
          <div
            className={cn("h-full rounded-full transition-all", colors.bg.replace("/10", ""))}
            style={{ width: `${barWidth}%` }}
          />
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function GreeksPanel({
  data = MOCK_GREEKS,
  className,
  compact = false,
}: GreeksPanelProps) {
  const greeksConfig = useMemo(() => [
    {
      name: "Delta",
      symbol: "Δ",
      value: data.delta,
      displayValue: data.delta >= 0 ? `+${data.delta.toFixed(2)}` : data.delta.toFixed(2),
      description: data.delta >= 0 ? "Net long directional exposure" : "Net short directional exposure",
      tooltip: "Delta measures directional exposure. +1 = fully long, -1 = fully short, 0 = market neutral. Values closer to 0 indicate hedged positions.",
      icon: data.delta >= 0 ? TrendingUp : TrendingDown,
      getColor: (v: number) => {
        const abs = Math.abs(v);
        if (abs <= 0.2) return { text: "text-green-400", bg: "bg-green-500/10", border: "border-green-500/20" };
        if (abs <= 0.5) return { text: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20" };
        return { text: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/20" };
      },
      getBarWidth: (v: number) => Math.abs(v) * 100,
      showNegativeBar: true,
    },
    {
      name: "Gamma",
      symbol: "Γ",
      value: data.gamma,
      displayValue: data.gamma.toFixed(3),
      description: data.gamma > 0.02 ? "High convexity exposure" : "Stable delta profile",
      tooltip: "Gamma measures how fast delta changes. High gamma = delta shifts quickly with price moves. Good for long options, risky for short options.",
      icon: Zap,
      getColor: (v: number) => {
        if (v <= 0.015) return { text: "text-green-400", bg: "bg-green-500/10", border: "border-green-500/20" };
        if (v <= 0.035) return { text: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20" };
        return { text: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/20" };
      },
      getBarWidth: (v: number) => Math.min(v * 1000, 100), // Scale for visualization
      showNegativeBar: false,
    },
    {
      name: "Vega",
      symbol: "ν",
      value: data.vega,
      displayValue: data.vega >= 0 ? `+$${(data.vega / 1000).toFixed(1)}K` : `-$${(Math.abs(data.vega) / 1000).toFixed(1)}K`,
      description: data.vega >= 0 ? "Long volatility exposure" : "Short volatility exposure",
      tooltip: "Vega measures P&L sensitivity to implied volatility. Positive vega profits from vol spikes. Shows $ change per 1% vol move.",
      icon: Wind,
      getColor: (v: number) => {
        // For hedging, being long vol (positive vega) is generally protective
        if (v > 5000) return { text: "text-green-400", bg: "bg-green-500/10", border: "border-green-500/20" };
        if (v > -5000) return { text: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20" };
        return { text: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/20" };
      },
      getBarWidth: (v: number) => Math.min(Math.abs(v) / 250, 100),
      showNegativeBar: false,
    },
    {
      name: "Theta",
      symbol: "Θ",
      value: data.theta,
      displayValue: `${data.theta >= 0 ? "+" : ""}$${(data.theta / 1000).toFixed(1)}K/day`,
      description: data.theta >= 0 ? "Collecting time premium" : "Paying time decay",
      tooltip: "Theta measures daily time decay. Negative theta = paying premium (long options). Positive theta = collecting premium (short options).",
      icon: Clock,
      getColor: (v: number) => {
        // For hedging with long options, negative theta is expected cost
        if (v >= 0) return { text: "text-green-400", bg: "bg-green-500/10", border: "border-green-500/20" };
        if (v >= -5000) return { text: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20" };
        return { text: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/20" };
      },
      getBarWidth: (v: number) => Math.min(Math.abs(v) / 100, 100),
      showNegativeBar: false,
    },
  ], [data]);

  return (
    <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-400" />
            <CardTitle className="text-sm font-semibold text-neutral-200">
              Portfolio Greeks
            </CardTitle>
          </div>
          <span className="text-[10px] uppercase tracking-wider text-neutral-500">
            Real-time Risk
          </span>
        </div>
      </CardHeader>
      <CardContent className={cn("p-4 pt-0", compact ? "space-y-2" : "")}>
        <div className={cn(
          "grid gap-3",
          compact ? "grid-cols-2 lg:grid-cols-4" : "grid-cols-2"
        )}>
          {greeksConfig.map((greek) => (
            <GreekCard
              key={greek.name}
              name={greek.name}
              symbol={greek.symbol}
              value={greek.value}
              displayValue={greek.displayValue}
              description={greek.description}
              tooltip={greek.tooltip}
              icon={greek.icon}
              getColor={greek.getColor}
              getBarWidth={greek.getBarWidth}
              showNegativeBar={greek.showNegativeBar}
            />
          ))}
        </div>

        {/* Summary Footer */}
        <div className="mt-4 p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
          <div className="flex items-center justify-between text-xs">
            <span className="text-neutral-500">Risk Profile:</span>
            <span className={cn(
              "font-medium",
              Math.abs(data.delta) <= 0.3 && data.gamma <= 0.03
                ? "text-green-400"
                : Math.abs(data.delta) <= 0.5
                ? "text-amber-400"
                : "text-red-400"
            )}>
              {Math.abs(data.delta) <= 0.2
                ? "Well Hedged"
                : Math.abs(data.delta) <= 0.5
                ? "Moderate Exposure"
                : "High Directional Risk"}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default GreeksPanel;
