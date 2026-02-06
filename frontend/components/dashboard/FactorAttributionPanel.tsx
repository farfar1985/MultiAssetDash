"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  PieChart,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  BarChart3,
  Waves,
  Zap,
  Droplets,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface FactorExposure {
  name: string;
  symbol: string;
  contribution: number;      // Percentage of total risk (can be negative for hedges)
  dollarValue: number;       // Dollar risk contribution
  trend: "up" | "down" | "flat";
  trendDelta: number;        // Change from previous period
  icon: typeof Activity;
  description: string;
}

interface FactorAttributionData {
  factors: FactorExposure[];
  totalRisk: number;         // Total portfolio risk in dollars
  diversificationBenefit: number; // Risk reduction from diversification
}

interface FactorAttributionPanelProps {
  data?: FactorAttributionData;
  className?: string;
  compact?: boolean;
}

// ============================================================================
// Mock Data
// ============================================================================

const MOCK_FACTOR_DATA: FactorAttributionData = {
  factors: [
    {
      name: "Market Beta",
      symbol: "β",
      contribution: 42.5,
      dollarValue: 186500,
      trend: "down",
      trendDelta: -3.2,
      icon: BarChart3,
      description: "Directional market exposure",
    },
    {
      name: "Volatility",
      symbol: "σ",
      contribution: 24.8,
      dollarValue: 108900,
      trend: "up",
      trendDelta: 5.1,
      icon: Waves,
      description: "Sensitivity to vol regime changes",
    },
    {
      name: "Momentum",
      symbol: "ρ",
      contribution: 15.2,
      dollarValue: 66700,
      trend: "flat",
      trendDelta: 0.3,
      icon: Zap,
      description: "Trend-following factor exposure",
    },
    {
      name: "Carry",
      symbol: "c",
      contribution: 11.8,
      dollarValue: 51800,
      trend: "down",
      trendDelta: -1.8,
      icon: TrendingUp,
      description: "Roll yield and cost of carry",
    },
    {
      name: "Liquidity",
      symbol: "λ",
      contribution: 5.7,
      dollarValue: 25000,
      trend: "up",
      trendDelta: 2.4,
      icon: Droplets,
      description: "Liquidity premium exposure",
    },
  ],
  totalRisk: 438900,
  diversificationBenefit: 12.3,
};

// ============================================================================
// Helper Functions
// ============================================================================

function getFactorColor(contribution: number): {
  bg: string;
  border: string;
  text: string;
  bar: string;
} {
  // Higher concentration = more concerning (amber/red)
  // Lower/diversifying = better (green/blue)
  if (contribution > 35) {
    return {
      bg: "bg-red-500/10",
      border: "border-red-500/30",
      text: "text-red-400",
      bar: "bg-red-500",
    };
  }
  if (contribution > 25) {
    return {
      bg: "bg-amber-500/10",
      border: "border-amber-500/30",
      text: "text-amber-400",
      bar: "bg-amber-500",
    };
  }
  if (contribution > 15) {
    return {
      bg: "bg-blue-500/10",
      border: "border-blue-500/30",
      text: "text-blue-400",
      bar: "bg-blue-500",
    };
  }
  return {
    bg: "bg-green-500/10",
    border: "border-green-500/30",
    text: "text-green-400",
    bar: "bg-green-500",
  };
}

function formatDollar(value: number): string {
  if (Math.abs(value) >= 1000000) {
    return `$${(value / 1000000).toFixed(2)}M`;
  }
  if (Math.abs(value) >= 1000) {
    return `$${(value / 1000).toFixed(1)}K`;
  }
  return `$${value.toFixed(0)}`;
}

// ============================================================================
// Stacked Bar Component
// ============================================================================

function StackedRiskBar({ factors }: { factors: FactorExposure[] }) {
  const barColors = [
    "bg-red-500",
    "bg-amber-500",
    "bg-blue-500",
    "bg-green-500",
    "bg-purple-500",
  ];

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-neutral-500">
        <span>Risk Contribution by Factor</span>
        <span>100%</span>
      </div>
      <div className="h-6 flex rounded-lg overflow-hidden border border-neutral-700">
        {factors.map((factor, idx) => (
          <div
            key={factor.name}
            className={cn(
              "h-full flex items-center justify-center transition-all hover:opacity-80",
              barColors[idx % barColors.length]
            )}
            style={{ width: `${factor.contribution}%` }}
            title={`${factor.name}: ${factor.contribution.toFixed(1)}%`}
          >
            {factor.contribution > 10 && (
              <span className="text-[10px] font-bold text-white/90">
                {factor.contribution.toFixed(0)}%
              </span>
            )}
          </div>
        ))}
      </div>
      {/* Legend */}
      <div className="flex flex-wrap gap-3 mt-2">
        {factors.map((factor, idx) => (
          <div key={factor.name} className="flex items-center gap-1.5">
            <div
              className={cn(
                "w-2.5 h-2.5 rounded-sm",
                barColors[idx % barColors.length]
              )}
            />
            <span className="text-[10px] text-neutral-400">{factor.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Factor Row Component
// ============================================================================

function FactorRow({ factor }: { factor: FactorExposure }) {
  const colors = getFactorColor(factor.contribution);
  const Icon = factor.icon;
  const TrendIcon = factor.trend === "up" ? TrendingUp : factor.trend === "down" ? TrendingDown : Minus;

  return (
    <div
      className={cn(
        "p-3 rounded-lg border transition-all hover:border-neutral-600",
        colors.bg,
        colors.border
      )}
      title={factor.description}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={cn("p-1.5 rounded-lg", colors.bg)}>
            <Icon className={cn("w-3.5 h-3.5", colors.text)} />
          </div>
          <div>
            <span className="text-sm font-medium text-neutral-200">{factor.name}</span>
            <span className={cn("ml-1.5 text-sm font-mono font-bold", colors.text)}>
              {factor.symbol}
            </span>
          </div>
        </div>

        {/* Trend Indicator */}
        <div className={cn(
          "flex items-center gap-1 px-2 py-0.5 rounded-full text-xs",
          factor.trend === "up"
            ? "bg-amber-500/10 text-amber-400"
            : factor.trend === "down"
            ? "bg-green-500/10 text-green-400"
            : "bg-neutral-700/50 text-neutral-400"
        )}>
          <TrendIcon className="w-3 h-3" />
          <span className="font-mono">
            {factor.trendDelta >= 0 ? "+" : ""}{factor.trendDelta.toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="flex items-end justify-between">
        {/* Contribution */}
        <div>
          <span className="text-[10px] uppercase tracking-wider text-neutral-500">
            Contribution
          </span>
          <div className={cn("text-xl font-mono font-bold", colors.text)}>
            {factor.contribution.toFixed(1)}%
          </div>
        </div>

        {/* Dollar Value */}
        <div className="text-right">
          <span className="text-[10px] uppercase tracking-wider text-neutral-500">
            Risk $
          </span>
          <div className="text-lg font-mono text-neutral-300">
            {formatDollar(factor.dollarValue)}
          </div>
        </div>
      </div>

      {/* Mini bar */}
      <div className="mt-2 h-1.5 bg-neutral-700 rounded-full overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all", colors.bar)}
          style={{ width: `${factor.contribution}%` }}
        />
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function FactorAttributionPanel({
  data = MOCK_FACTOR_DATA,
  className,
  compact = false,
}: FactorAttributionPanelProps) {
  // Calculate concentration metrics
  const metrics = useMemo(() => {
    const sortedFactors = [...data.factors].sort((a, b) => b.contribution - a.contribution);
    const topFactor = sortedFactors[0];
    const herfindahl = data.factors.reduce((sum, f) => sum + (f.contribution / 100) ** 2, 0);
    const concentration = herfindahl * 100; // Herfindahl index as percentage

    return {
      topFactor,
      concentration,
      isConcentrated: topFactor.contribution > 40,
      increasingRiskFactors: data.factors.filter(f => f.trend === "up").length,
    };
  }, [data]);

  return (
    <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <PieChart className="w-5 h-5 text-cyan-400" />
            <CardTitle className="text-sm font-semibold text-neutral-200">
              Factor Attribution
            </CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={cn(
              "text-xs",
              metrics.isConcentrated
                ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                : "bg-green-500/10 border-green-500/30 text-green-400"
            )}>
              {metrics.isConcentrated ? "Concentrated" : "Diversified"}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0 space-y-4">
        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-3">
          <div className="p-2 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
            <span className="text-[10px] uppercase tracking-wider text-neutral-500">
              Total Risk
            </span>
            <div className="text-lg font-mono font-bold text-neutral-200">
              {formatDollar(data.totalRisk)}
            </div>
          </div>
          <div className="p-2 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
            <span className="text-[10px] uppercase tracking-wider text-neutral-500">
              Diversification
            </span>
            <div className="text-lg font-mono font-bold text-green-400">
              -{data.diversificationBenefit.toFixed(1)}%
            </div>
          </div>
          <div className="p-2 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
            <span className="text-[10px] uppercase tracking-wider text-neutral-500">
              Top Factor
            </span>
            <div className="text-lg font-mono font-bold text-amber-400">
              {metrics.topFactor.name.split(" ")[0]}
            </div>
          </div>
        </div>

        {/* Stacked Bar */}
        <StackedRiskBar factors={data.factors} />

        {/* Factor Grid */}
        <div className={cn(
          "grid gap-3",
          compact ? "grid-cols-2 lg:grid-cols-3" : "grid-cols-1 sm:grid-cols-2"
        )}>
          {data.factors.map((factor) => (
            <FactorRow key={factor.name} factor={factor} />
          ))}
        </div>

        {/* Risk Alert */}
        {metrics.increasingRiskFactors > 1 && (
          <div className="p-3 bg-amber-500/5 border border-amber-500/20 rounded-lg">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-amber-400" />
              <span className="text-sm text-amber-400 font-medium">
                {metrics.increasingRiskFactors} factors showing increased exposure
              </span>
            </div>
            <p className="text-xs text-neutral-500 mt-1">
              Consider rebalancing to reduce concentration risk
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default FactorAttributionPanel;
