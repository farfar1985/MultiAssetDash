"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { MOCK_ASSETS, MOCK_SIGNALS, type Horizon } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import { CorrelationMatrix } from "@/components/dashboard/CorrelationMatrix";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  Zap,
  PieChart,
  Shield,
  Layers,
  Brain,
  Gauge,
  Briefcase,
  GitBranch,
  Activity,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  DollarSign,
  Percent,
  Scale,
  LineChart,
  AlertTriangle,
  CheckCircle2,
  Info,
  TrendingUp as ChartUp,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface PortfolioMetrics {
  totalAUM: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  mtdReturn: number;
  ytdReturn: number;
  maxDrawdown: number;
  currentDrawdown: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
}

interface AssetPosition {
  assetId: AssetId;
  name: string;
  symbol: string;
  currentPrice: number;
  weight: number;
  targetWeight: number;
  marketValue: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  mtdReturn: number;
  signal: "long" | "short" | "neutral";
  beta: number;
  sharpe: number;
  contribution: number;
}

interface PerformanceAttribution {
  factor: string;
  contribution: number;
  exposure: number;
  benchmark: number;
  active: number;
}

interface BenchmarkComparison {
  name: string;
  symbol: string;
  portfolioReturn: number;
  benchmarkReturn: number;
  alpha: number;
  trackingError: number;
  informationRatio: number;
}

interface PositionSizing {
  assetId: AssetId;
  symbol: string;
  currentWeight: number;
  recommendedWeight: number;
  action: "increase" | "decrease" | "hold";
  reason: string;
  confidence: number;
}

interface FactorExposure {
  factor: string;
  portfolioExposure: number;
  benchmarkExposure: number;
  activeExposure: number;
  targetRange: [number, number];
  status: "in-range" | "overweight" | "underweight";
}

// ============================================================================
// Mock Data Generators
// ============================================================================

function generatePortfolioMetrics(): PortfolioMetrics {
  return {
    totalAUM: 2847500000,
    dailyPnL: 12450000,
    dailyPnLPercent: 0.44,
    mtdReturn: 2.87,
    ytdReturn: 18.42,
    maxDrawdown: -8.2,
    currentDrawdown: -1.4,
    sharpeRatio: 2.14,
    sortinoRatio: 2.89,
    calmarRatio: 2.24,
  };
}

function generateAssetPositions(): AssetPosition[] {
  const positions: AssetPosition[] = [];
  const baseAUM = 2847500000;

  const weights: Record<string, number> = {
    "crude-oil": 18.5,
    "natural-gas": 8.2,
    gold: 22.4,
    bitcoin: 12.8,
    silver: 6.5,
    copper: 10.2,
    wheat: 7.4,
    corn: 5.8,
    soybean: 4.2,
    platinum: 4.0,
  };

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    const signal = MOCK_SIGNALS[assetId as AssetId]?.["D+5"];
    const weight = weights[assetId] || 5;
    const targetWeight = weight + (Math.random() - 0.5) * 2;
    const dailyPnLPercent = (Math.random() - 0.4) * 4;
    const marketValue = baseAUM * (weight / 100);

    positions.push({
      assetId: assetId as AssetId,
      name: asset.name,
      symbol: asset.symbol,
      currentPrice: asset.currentPrice,
      weight,
      targetWeight,
      marketValue,
      dailyPnL: marketValue * (dailyPnLPercent / 100),
      dailyPnLPercent,
      mtdReturn: (Math.random() - 0.3) * 10,
      signal: signal?.direction === "bullish" ? "long" : signal?.direction === "bearish" ? "short" : "neutral",
      beta: 0.3 + Math.random() * 0.9,
      sharpe: (Math.random() - 0.2) * 3,
      contribution: weight * (dailyPnLPercent / 100) / 0.44 * 100,
    });
  });

  return positions.sort((a, b) => b.weight - a.weight);
}

function generatePerformanceAttribution(): PerformanceAttribution[] {
  return [
    { factor: "Market Beta", contribution: 8.2, exposure: 0.65, benchmark: 1.0, active: -0.35 },
    { factor: "Momentum", contribution: 4.8, exposure: 0.42, benchmark: 0.15, active: 0.27 },
    { factor: "Value", contribution: 2.1, exposure: 0.28, benchmark: 0.20, active: 0.08 },
    { factor: "Quality", contribution: 1.9, exposure: 0.35, benchmark: 0.25, active: 0.10 },
    { factor: "Low Volatility", contribution: 0.8, exposure: 0.18, benchmark: 0.30, active: -0.12 },
    { factor: "Size", contribution: -0.4, exposure: -0.15, benchmark: 0.10, active: -0.25 },
    { factor: "Residual Alpha", contribution: 1.0, exposure: 0, benchmark: 0, active: 0 },
  ];
}

function generateBenchmarkComparisons(): BenchmarkComparison[] {
  return [
    { name: "S&P 500", symbol: "SPY", portfolioReturn: 18.42, benchmarkReturn: 12.15, alpha: 6.27, trackingError: 8.4, informationRatio: 0.75 },
    { name: "Bloomberg Commodity", symbol: "BCOM", portfolioReturn: 18.42, benchmarkReturn: 5.82, alpha: 12.60, trackingError: 12.1, informationRatio: 1.04 },
    { name: "60/40 Portfolio", symbol: "60/40", portfolioReturn: 18.42, benchmarkReturn: 9.24, alpha: 9.18, trackingError: 10.2, informationRatio: 0.90 },
    { name: "Risk Parity", symbol: "RPAR", portfolioReturn: 18.42, benchmarkReturn: 7.45, alpha: 10.97, trackingError: 9.8, informationRatio: 1.12 },
  ];
}

function generatePositionSizing(): PositionSizing[] {
  return [
    { assetId: "gold" as AssetId, symbol: "GC", currentWeight: 22.4, recommendedWeight: 25.0, action: "increase", reason: "Strong momentum, low correlation to risk assets", confidence: 82 },
    { assetId: "crude-oil" as AssetId, symbol: "CL", currentWeight: 18.5, recommendedWeight: 15.0, action: "decrease", reason: "Elevated volatility, approaching resistance", confidence: 74 },
    { assetId: "bitcoin" as AssetId, symbol: "BTC", currentWeight: 12.8, recommendedWeight: 12.8, action: "hold", reason: "Neutral signal, maintain current exposure", confidence: 68 },
    { assetId: "copper" as AssetId, symbol: "HG", currentWeight: 10.2, recommendedWeight: 12.0, action: "increase", reason: "Positive macro outlook, underweight vs target", confidence: 76 },
    { assetId: "natural-gas" as AssetId, symbol: "NG", currentWeight: 8.2, recommendedWeight: 6.0, action: "decrease", reason: "Bearish seasonal pattern, high inventory", confidence: 71 },
  ];
}

function generateFactorExposures(): FactorExposure[] {
  return [
    { factor: "Market Beta", portfolioExposure: 0.65, benchmarkExposure: 1.0, activeExposure: -0.35, targetRange: [0.4, 0.8], status: "in-range" },
    { factor: "Momentum", portfolioExposure: 0.42, benchmarkExposure: 0.15, activeExposure: 0.27, targetRange: [0.2, 0.5], status: "in-range" },
    { factor: "Value", portfolioExposure: 0.28, benchmarkExposure: 0.20, activeExposure: 0.08, targetRange: [0.1, 0.4], status: "in-range" },
    { factor: "Volatility", portfolioExposure: 0.18, benchmarkExposure: 0.30, activeExposure: -0.12, targetRange: [0.1, 0.3], status: "in-range" },
    { factor: "Carry", portfolioExposure: 0.55, benchmarkExposure: 0.25, activeExposure: 0.30, targetRange: [0.2, 0.4], status: "overweight" },
    { factor: "Liquidity", portfolioExposure: -0.08, benchmarkExposure: 0.05, activeExposure: -0.13, targetRange: [-0.1, 0.1], status: "in-range" },
  ];
}

// ============================================================================
// Components
// ============================================================================

function DashboardHeader({ metrics }: { metrics: PortfolioMetrics }) {
  return (
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-4">
        <div className="p-3 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl">
          <Briefcase className="w-7 h-7 text-white" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-neutral-100">Hedge Fund Portfolio</h1>
          <p className="text-sm text-neutral-400">Multi-Asset Institutional Dashboard</p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30 px-3 py-1.5">
          <Activity className="w-3.5 h-3.5 mr-1.5 animate-pulse" />
          Live
        </Badge>
        <div className="text-right">
          <div className="text-2xl font-bold text-neutral-100">
            ${(metrics.totalAUM / 1000000000).toFixed(2)}B
          </div>
          <div className="text-xs text-neutral-500">Total AUM</div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Portfolio Metrics Cards
// ============================================================================

function PortfolioMetricsCards({ metrics }: { metrics: PortfolioMetrics }) {
  const cards = [
    {
      label: "Daily P&L",
      value: `$${(metrics.dailyPnL / 1000000).toFixed(2)}M`,
      change: `${metrics.dailyPnLPercent >= 0 ? "+" : ""}${metrics.dailyPnLPercent.toFixed(2)}%`,
      positive: metrics.dailyPnLPercent >= 0,
      icon: DollarSign,
    },
    {
      label: "MTD Return",
      value: `${metrics.mtdReturn >= 0 ? "+" : ""}${metrics.mtdReturn.toFixed(2)}%`,
      change: "vs +1.8% benchmark",
      positive: metrics.mtdReturn > 1.8,
      icon: TrendingUp,
    },
    {
      label: "YTD Return",
      value: `${metrics.ytdReturn >= 0 ? "+" : ""}${metrics.ytdReturn.toFixed(2)}%`,
      change: "+6.27% alpha",
      positive: true,
      icon: ChartUp,
    },
    {
      label: "Current Drawdown",
      value: `${metrics.currentDrawdown.toFixed(1)}%`,
      change: `Max: ${metrics.maxDrawdown.toFixed(1)}%`,
      positive: Math.abs(metrics.currentDrawdown) < 5,
      icon: ArrowDownRight,
    },
    {
      label: "Sharpe Ratio",
      value: metrics.sharpeRatio.toFixed(2),
      change: "Sortino: " + metrics.sortinoRatio.toFixed(2),
      positive: metrics.sharpeRatio > 1.5,
      icon: Gauge,
    },
    {
      label: "Calmar Ratio",
      value: metrics.calmarRatio.toFixed(2),
      change: "Risk-adjusted",
      positive: metrics.calmarRatio > 1.5,
      icon: Scale,
    },
  ];

  return (
    <div className="grid grid-cols-6 gap-3 mb-6">
      {cards.map((card) => (
        <Card key={card.label} className="bg-neutral-900/60 border-neutral-800">
          <CardContent className="p-3">
            <div className="flex items-center justify-between mb-2">
              <card.icon className={cn("w-4 h-4", card.positive ? "text-emerald-400" : "text-red-400")} />
              {card.positive ? (
                <ArrowUpRight className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <ArrowDownRight className="w-3.5 h-3.5 text-red-400" />
              )}
            </div>
            <div className="text-xl font-bold text-neutral-100 mb-0.5">{card.value}</div>
            <div className="text-[10px] text-neutral-500 mb-1">{card.label}</div>
            <div className={cn("text-[10px]", card.positive ? "text-emerald-400" : "text-neutral-500")}>
              {card.change}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ============================================================================
// Asset Allocation Donut Chart
// ============================================================================

function AssetAllocationChart({ positions }: { positions: AssetPosition[] }) {
  const total = positions.reduce((sum, p) => sum + p.weight, 0);
  let cumulativePercent = 0;

  const colors = [
    "#8b5cf6", "#6366f1", "#3b82f6", "#0ea5e9", "#14b8a6",
    "#22c55e", "#eab308", "#f97316", "#ef4444", "#ec4899"
  ];

  const segments = positions.map((pos, idx) => {
    const percent = pos.weight / total;
    const startPercent = cumulativePercent;
    cumulativePercent += percent;
    return {
      ...pos,
      color: colors[idx % colors.length],
      startPercent,
      percent,
    };
  });

  return (
    <Card className="bg-neutral-900/60 border-neutral-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-neutral-300 flex items-center gap-2">
          <PieChart className="w-4 h-4 text-indigo-400" />
          Asset Allocation
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="flex items-center gap-4">
          {/* SVG Donut Chart */}
          <div className="relative w-32 h-32">
            <svg viewBox="0 0 100 100" className="transform -rotate-90">
              {segments.map((seg, idx) => {
                const circumference = 2 * Math.PI * 40;
                const strokeDasharray = `${seg.percent * circumference} ${circumference}`;
                const strokeDashoffset = -seg.startPercent * circumference;

                return (
                  <circle
                    key={seg.assetId}
                    cx="50"
                    cy="50"
                    r="40"
                    fill="none"
                    stroke={seg.color}
                    strokeWidth="12"
                    strokeDasharray={strokeDasharray}
                    strokeDashoffset={strokeDashoffset}
                    className="transition-all duration-500"
                  />
                );
              })}
            </svg>
            <div className="absolute inset-0 flex items-center justify-center flex-col">
              <div className="text-lg font-bold text-neutral-100">100%</div>
              <div className="text-[9px] text-neutral-500">Allocated</div>
            </div>
          </div>

          {/* Legend */}
          <div className="flex-1 grid grid-cols-2 gap-1">
            {segments.slice(0, 8).map((seg) => (
              <div key={seg.assetId} className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: seg.color }} />
                <span className="text-[10px] text-neutral-400 truncate">{seg.symbol}</span>
                <span className="text-[10px] font-mono text-neutral-300">{seg.weight.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Multi-Asset Portfolio Table
// ============================================================================

function PortfolioTable({ positions }: { positions: AssetPosition[] }) {
  return (
    <Card className="bg-neutral-900/60 border-neutral-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-neutral-300 flex items-center gap-2">
          <Layers className="w-4 h-4 text-indigo-400" />
          Portfolio Positions
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-[10px] text-neutral-500 uppercase tracking-wider border-b border-neutral-800">
                <th className="text-left py-2 px-2">Asset</th>
                <th className="text-right py-2 px-2">Weight</th>
                <th className="text-right py-2 px-2">Value</th>
                <th className="text-right py-2 px-2">Daily P&L</th>
                <th className="text-right py-2 px-2">MTD</th>
                <th className="text-right py-2 px-2">Beta</th>
                <th className="text-right py-2 px-2">Sharpe</th>
                <th className="text-center py-2 px-2">Signal</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/50">
              {positions.map((pos) => (
                <tr key={pos.assetId} className="hover:bg-neutral-800/30 transition-colors">
                  <td className="py-2 px-2">
                    <div className="flex items-center gap-2">
                      <div className={cn(
                        "w-1.5 h-6 rounded-full",
                        pos.signal === "long" ? "bg-emerald-500" :
                        pos.signal === "short" ? "bg-red-500" : "bg-neutral-600"
                      )} />
                      <div>
                        <div className="font-medium text-neutral-200">{pos.symbol}</div>
                        <div className="text-[10px] text-neutral-500">{pos.name}</div>
                      </div>
                    </div>
                  </td>
                  <td className="py-2 px-2 text-right">
                    <div className="font-mono text-neutral-200">{pos.weight.toFixed(1)}%</div>
                    <div className={cn(
                      "text-[10px] font-mono",
                      pos.weight > pos.targetWeight ? "text-emerald-400" : "text-red-400"
                    )}>
                      {pos.weight > pos.targetWeight ? "+" : ""}{(pos.weight - pos.targetWeight).toFixed(1)}
                    </div>
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-neutral-300">
                    ${(pos.marketValue / 1000000).toFixed(1)}M
                  </td>
                  <td className={cn(
                    "py-2 px-2 text-right font-mono",
                    pos.dailyPnLPercent >= 0 ? "text-emerald-400" : "text-red-400"
                  )}>
                    {pos.dailyPnLPercent >= 0 ? "+" : ""}{pos.dailyPnLPercent.toFixed(2)}%
                  </td>
                  <td className={cn(
                    "py-2 px-2 text-right font-mono",
                    pos.mtdReturn >= 0 ? "text-emerald-400" : "text-red-400"
                  )}>
                    {pos.mtdReturn >= 0 ? "+" : ""}{pos.mtdReturn.toFixed(2)}%
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-neutral-400">
                    {pos.beta.toFixed(2)}
                  </td>
                  <td className={cn(
                    "py-2 px-2 text-right font-mono",
                    pos.sharpe > 1 ? "text-emerald-400" : pos.sharpe > 0 ? "text-amber-400" : "text-red-400"
                  )}>
                    {pos.sharpe.toFixed(2)}
                  </td>
                  <td className="py-2 px-2 text-center">
                    <Badge className={cn(
                      "text-[9px]",
                      pos.signal === "long" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" :
                      pos.signal === "short" ? "bg-red-500/20 text-red-400 border-red-500/30" :
                      "bg-neutral-700 text-neutral-400 border-neutral-600"
                    )}>
                      {pos.signal.toUpperCase()}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Performance Attribution Panel
// ============================================================================

function PerformanceAttributionPanel({ data }: { data: PerformanceAttribution[] }) {
  const totalContribution = data.reduce((sum, d) => sum + d.contribution, 0);

  return (
    <Card className="bg-neutral-900/60 border-neutral-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-neutral-300 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-indigo-400" />
          Performance Attribution
          <Badge className="bg-indigo-500/20 text-indigo-400 border-indigo-500/30 text-[9px] ml-auto">
            YTD: +{totalContribution.toFixed(1)}%
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-3">
          {data.map((item) => (
            <div key={item.factor}>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-neutral-400">{item.factor}</span>
                <div className="flex items-center gap-3">
                  <span className={cn(
                    "font-mono",
                    item.contribution >= 0 ? "text-emerald-400" : "text-red-400"
                  )}>
                    {item.contribution >= 0 ? "+" : ""}{item.contribution.toFixed(1)}%
                  </span>
                  <span className="text-neutral-600 font-mono text-[10px]">
                    Active: {item.active >= 0 ? "+" : ""}{item.active.toFixed(2)}
                  </span>
                </div>
              </div>
              <div className="h-2 bg-neutral-800 rounded-full overflow-hidden relative">
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-neutral-600" />
                {item.contribution !== 0 && (
                  <div
                    className={cn(
                      "absolute h-full rounded-full",
                      item.contribution > 0 ? "bg-emerald-500" : "bg-red-500"
                    )}
                    style={{
                      left: item.contribution > 0 ? "50%" : `${50 + (item.contribution / totalContribution) * 50}%`,
                      width: `${Math.abs(item.contribution / totalContribution) * 50}%`,
                    }}
                  />
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Benchmark Comparison Panel
// ============================================================================

function BenchmarkComparisonPanel({ data }: { data: BenchmarkComparison[] }) {
  return (
    <Card className="bg-neutral-900/60 border-neutral-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-neutral-300 flex items-center gap-2">
          <Target className="w-4 h-4 text-indigo-400" />
          Benchmark Comparison
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-3">
          {data.map((bench) => (
            <div key={bench.symbol} className="p-2 bg-neutral-800/30 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <span className="text-sm font-medium text-neutral-200">{bench.name}</span>
                  <span className="text-xs text-neutral-500 ml-2">({bench.symbol})</span>
                </div>
                <Badge className={cn(
                  "text-[9px]",
                  bench.alpha > 5 ? "bg-emerald-500/20 text-emerald-400" : "bg-amber-500/20 text-amber-400"
                )}>
                  α: +{bench.alpha.toFixed(2)}%
                </Badge>
              </div>
              <div className="grid grid-cols-4 gap-2 text-[10px]">
                <div>
                  <div className="text-neutral-500">Portfolio</div>
                  <div className="text-emerald-400 font-mono">+{bench.portfolioReturn.toFixed(2)}%</div>
                </div>
                <div>
                  <div className="text-neutral-500">Benchmark</div>
                  <div className="text-neutral-300 font-mono">+{bench.benchmarkReturn.toFixed(2)}%</div>
                </div>
                <div>
                  <div className="text-neutral-500">Track Err</div>
                  <div className="text-neutral-400 font-mono">{bench.trackingError.toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-neutral-500">Info Ratio</div>
                  <div className={cn(
                    "font-mono",
                    bench.informationRatio > 0.5 ? "text-emerald-400" : "text-amber-400"
                  )}>
                    {bench.informationRatio.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Position Sizing Recommendations
// ============================================================================

function PositionSizingPanel({ data }: { data: PositionSizing[] }) {
  const getActionColor = (action: PositionSizing["action"]) => {
    switch (action) {
      case "increase": return "text-emerald-400 bg-emerald-500/20 border-emerald-500/30";
      case "decrease": return "text-red-400 bg-red-500/20 border-red-500/30";
      default: return "text-neutral-400 bg-neutral-700 border-neutral-600";
    }
  };

  const getActionIcon = (action: PositionSizing["action"]) => {
    switch (action) {
      case "increase": return TrendingUp;
      case "decrease": return TrendingDown;
      default: return Minus;
    }
  };

  return (
    <Card className="bg-neutral-900/60 border-neutral-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-neutral-300 flex items-center gap-2">
          <Scale className="w-4 h-4 text-indigo-400" />
          Position Sizing Recommendations
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-2">
          {data.map((item) => {
            const ActionIcon = getActionIcon(item.action);
            return (
              <div key={item.assetId} className="p-2 bg-neutral-800/30 rounded-lg">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-neutral-200">{item.symbol}</span>
                    <Badge className={cn("text-[9px]", getActionColor(item.action))}>
                      <ActionIcon className="w-3 h-3 mr-1" />
                      {item.action.toUpperCase()}
                    </Badge>
                  </div>
                  <div className="text-xs text-neutral-500">
                    {item.confidence}% confidence
                  </div>
                </div>
                <div className="flex items-center gap-3 text-xs mb-1">
                  <span className="text-neutral-500">Current: <span className="text-neutral-300 font-mono">{item.currentWeight.toFixed(1)}%</span></span>
                  <span className="text-neutral-600">→</span>
                  <span className="text-neutral-500">Target: <span className={cn(
                    "font-mono",
                    item.action === "increase" ? "text-emerald-400" :
                    item.action === "decrease" ? "text-red-400" : "text-neutral-300"
                  )}>{item.recommendedWeight.toFixed(1)}%</span></span>
                </div>
                <p className="text-[10px] text-neutral-500">{item.reason}</p>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Factor Exposure Summary
// ============================================================================

function FactorExposurePanel({ data }: { data: FactorExposure[] }) {
  const getStatusBadge = (status: FactorExposure["status"]) => {
    switch (status) {
      case "in-range": return <CheckCircle2 className="w-3 h-3 text-emerald-400" />;
      case "overweight": return <AlertTriangle className="w-3 h-3 text-amber-400" />;
      case "underweight": return <AlertTriangle className="w-3 h-3 text-blue-400" />;
    }
  };

  return (
    <Card className="bg-neutral-900/60 border-neutral-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-neutral-300 flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-indigo-400" />
          Factor Exposure Summary
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-3">
          {data.map((factor) => (
            <div key={factor.factor}>
              <div className="flex items-center justify-between text-xs mb-1">
                <div className="flex items-center gap-2">
                  {getStatusBadge(factor.status)}
                  <span className="text-neutral-400">{factor.factor}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={cn(
                    "font-mono",
                    factor.activeExposure > 0 ? "text-emerald-400" :
                    factor.activeExposure < 0 ? "text-red-400" : "text-neutral-400"
                  )}>
                    {factor.activeExposure >= 0 ? "+" : ""}{factor.activeExposure.toFixed(2)}
                  </span>
                  <span className="text-neutral-600 text-[10px]">active</span>
                </div>
              </div>
              <div className="relative h-2 bg-neutral-800 rounded-full overflow-hidden">
                {/* Target range */}
                <div
                  className="absolute h-full bg-neutral-700/50"
                  style={{
                    left: `${(factor.targetRange[0] + 1) * 50}%`,
                    width: `${(factor.targetRange[1] - factor.targetRange[0]) * 50}%`,
                  }}
                />
                {/* Portfolio exposure marker */}
                <div
                  className={cn(
                    "absolute top-0 bottom-0 w-1 rounded-full",
                    factor.status === "in-range" ? "bg-emerald-500" : "bg-amber-500"
                  )}
                  style={{ left: `${(factor.portfolioExposure + 1) * 50}%` }}
                />
                {/* Benchmark marker */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-neutral-500"
                  style={{ left: `${(factor.benchmarkExposure + 1) * 50}%` }}
                />
              </div>
              <div className="flex justify-between text-[9px] text-neutral-600 mt-0.5">
                <span>-1</span>
                <span>0</span>
                <span>+1</span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Risk Adjusted Returns Comparison
// ============================================================================

function RiskAdjustedReturnsPanel({ positions }: { positions: AssetPosition[] }) {
  const sortedBySharpe = [...positions].sort((a, b) => b.sharpe - a.sharpe);

  return (
    <Card className="bg-neutral-900/60 border-neutral-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-neutral-300 flex items-center gap-2">
          <Gauge className="w-4 h-4 text-indigo-400" />
          Risk-Adjusted Returns
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-2">
          {sortedBySharpe.slice(0, 6).map((pos, idx) => (
            <div key={pos.assetId} className="flex items-center gap-3">
              <div className={cn(
                "w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold",
                idx === 0 ? "bg-amber-500 text-neutral-900" :
                idx === 1 ? "bg-neutral-400 text-neutral-900" :
                idx === 2 ? "bg-amber-700 text-neutral-100" :
                "bg-neutral-700 text-neutral-400"
              )}>
                {idx + 1}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-neutral-200">{pos.symbol}</span>
                  <span className={cn(
                    "text-sm font-mono font-medium",
                    pos.sharpe > 1 ? "text-emerald-400" : pos.sharpe > 0 ? "text-amber-400" : "text-red-400"
                  )}>
                    {pos.sharpe.toFixed(2)}
                  </span>
                </div>
                <div className="h-1 bg-neutral-800 rounded-full overflow-hidden mt-1">
                  <div
                    className={cn(
                      "h-full rounded-full",
                      pos.sharpe > 1 ? "bg-emerald-500" : pos.sharpe > 0 ? "bg-amber-500" : "bg-red-500"
                    )}
                    style={{ width: `${Math.max(0, Math.min(100, (pos.sharpe + 1) * 33))}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Dashboard
// ============================================================================

export function HedgeFundDashboard() {
  const [activeTab, setActiveTab] = useState("overview");

  const metrics = useMemo(() => generatePortfolioMetrics(), []);
  const positions = useMemo(() => generateAssetPositions(), []);
  const attribution = useMemo(() => generatePerformanceAttribution(), []);
  const benchmarks = useMemo(() => generateBenchmarkComparisons(), []);
  const sizing = useMemo(() => generatePositionSizing(), []);
  const factors = useMemo(() => generateFactorExposures(), []);

  return (
    <div className="space-y-4 pb-8 -m-6 p-6 bg-neutral-950 min-h-screen">
      {/* Header */}
      <DashboardHeader metrics={metrics} />

      {/* Portfolio Metrics */}
      <PortfolioMetricsCards metrics={metrics} />

      {/* Tab Navigation */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="bg-neutral-900 border border-neutral-800 p-0.5">
          <TabsTrigger value="overview" className="text-xs data-[state=active]:bg-indigo-500/20 data-[state=active]:text-indigo-400">
            Overview
          </TabsTrigger>
          <TabsTrigger value="positions" className="text-xs data-[state=active]:bg-indigo-500/20 data-[state=active]:text-indigo-400">
            Positions
          </TabsTrigger>
          <TabsTrigger value="attribution" className="text-xs data-[state=active]:bg-indigo-500/20 data-[state=active]:text-indigo-400">
            Attribution
          </TabsTrigger>
          <TabsTrigger value="risk" className="text-xs data-[state=active]:bg-indigo-500/20 data-[state=active]:text-indigo-400">
            Risk & Factors
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="mt-4 space-y-4">
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-3">
              <AssetAllocationChart positions={positions} />
            </div>
            <div className="col-span-5">
              <PerformanceAttributionPanel data={attribution} />
            </div>
            <div className="col-span-4">
              <BenchmarkComparisonPanel data={benchmarks} />
            </div>
          </div>
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-8">
              <PortfolioTable positions={positions} />
            </div>
            <div className="col-span-4">
              <RiskAdjustedReturnsPanel positions={positions} />
            </div>
          </div>
        </TabsContent>

        {/* Positions Tab */}
        <TabsContent value="positions" className="mt-4 space-y-4">
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-9">
              <PortfolioTable positions={positions} />
            </div>
            <div className="col-span-3">
              <PositionSizingPanel data={sizing} />
            </div>
          </div>
        </TabsContent>

        {/* Attribution Tab */}
        <TabsContent value="attribution" className="mt-4 space-y-4">
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-6">
              <PerformanceAttributionPanel data={attribution} />
            </div>
            <div className="col-span-6">
              <BenchmarkComparisonPanel data={benchmarks} />
            </div>
          </div>
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-4">
              <AssetAllocationChart positions={positions} />
            </div>
            <div className="col-span-8">
              <RiskAdjustedReturnsPanel positions={positions} />
            </div>
          </div>
        </TabsContent>

        {/* Risk & Factors Tab */}
        <TabsContent value="risk" className="mt-4 space-y-4">
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-4">
              <FactorExposurePanel data={factors} />
            </div>
            <div className="col-span-8">
              <CorrelationMatrix size="md" showLabels={true} interactive={true} />
            </div>
          </div>
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-6">
              <PositionSizingPanel data={sizing} />
            </div>
            <div className="col-span-6">
              <RiskAdjustedReturnsPanel positions={positions} />
            </div>
          </div>
        </TabsContent>
      </Tabs>

      {/* Footer */}
      <div className="flex items-center justify-between text-[10px] text-neutral-600 pt-4 border-t border-neutral-800">
        <div className="flex items-center gap-2">
          <Brain className="w-3 h-3" />
          <span>QDT Nexus Institutional • AI-Powered Portfolio Analytics</span>
        </div>
        <span>Last Updated: {new Date().toLocaleTimeString()}</span>
      </div>
    </div>
  );
}

export default HedgeFundDashboard;
