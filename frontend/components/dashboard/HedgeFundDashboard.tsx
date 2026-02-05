"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { MOCK_ASSETS, MOCK_SIGNALS, type Horizon, type SignalData } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  Zap,
  PieChart,
  Shield,
  ChevronRight,
  Layers,
  Brain,
  Gauge,
  Trophy,
  Briefcase,
  GitBranch,
  Crosshair,
  Star,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface AlphaOpportunity {
  assetId: AssetId;
  name: string;
  symbol: string;
  currentPrice: number;
  signal: SignalData;
  alphaMetrics: AlphaMetrics;
  rank: number;
}

interface AlphaMetrics {
  expectedReturn: number; // annualized %
  expectedAlpha: number; // vs benchmark %
  informationRatio: number;
  betaToMarket: number;
  volatility: number;
  convictionScore: number; // 0-100
  factorExposures: FactorExposure[];
  riskContribution: number; // % of portfolio risk
}

interface FactorExposure {
  factor: string;
  exposure: number; // -1 to 1
  contribution: number; // % of return
}

interface PortfolioAllocation {
  assetId: AssetId;
  name: string;
  symbol: string;
  weight: number;
  targetWeight: number;
  drift: number;
  signal: "long" | "short" | "neutral";
  pnl: number;
  pnlPercent: number;
}

interface RiskMetric {
  name: string;
  value: number;
  threshold: number;
  status: "safe" | "warning" | "critical";
  unit: string;
}

// ============================================================================
// Data Generators
// ============================================================================

function generateAlphaMetrics(signal: SignalData): AlphaMetrics {
  const baseReturn = signal.direction === "bullish" ? 15 + Math.random() * 20 :
                     signal.direction === "bearish" ? -10 - Math.random() * 15 : Math.random() * 10 - 5;

  const alpha = baseReturn - 8; // Assuming 8% benchmark return
  const volatility = 15 + Math.random() * 25;
  const informationRatio = alpha / volatility * Math.sqrt(252 / 30); // Annualized

  return {
    expectedReturn: baseReturn,
    expectedAlpha: alpha,
    informationRatio: Math.max(-2, Math.min(3, informationRatio)),
    betaToMarket: 0.3 + Math.random() * 1.2,
    volatility,
    convictionScore: signal.confidence,
    factorExposures: [
      { factor: "Momentum", exposure: signal.direction === "bullish" ? 0.6 : -0.4, contribution: 30 },
      { factor: "Value", exposure: 0.2 + Math.random() * 0.3, contribution: 20 },
      { factor: "Quality", exposure: 0.3 + Math.random() * 0.4, contribution: 25 },
      { factor: "Low Vol", exposure: -0.2 + Math.random() * 0.4, contribution: 15 },
      { factor: "Size", exposure: Math.random() * 0.4 - 0.2, contribution: 10 },
    ],
    riskContribution: 5 + Math.random() * 15,
  };
}

function getAlphaOpportunities(): AlphaOpportunity[] {
  const opportunities: AlphaOpportunity[] = [];

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    // Get best signal across horizons
    const horizons: Horizon[] = ["D+1", "D+5", "D+10"];
    let bestSignal: SignalData | null = null;

    for (const horizon of horizons) {
      const signal = MOCK_SIGNALS[assetId as AssetId]?.[horizon];
      if (signal && (!bestSignal || signal.sharpeRatio > bestSignal.sharpeRatio)) {
        bestSignal = signal;
      }
    }

    if (bestSignal) {
      opportunities.push({
        assetId: assetId as AssetId,
        name: asset.name,
        symbol: asset.symbol,
        currentPrice: asset.currentPrice,
        signal: bestSignal,
        alphaMetrics: generateAlphaMetrics(bestSignal),
        rank: 0,
      });
    }
  });

  // Rank by information ratio
  return opportunities
    .sort((a, b) => b.alphaMetrics.informationRatio - a.alphaMetrics.informationRatio)
    .map((opp, idx) => ({ ...opp, rank: idx + 1 }));
}

function getPortfolioAllocations(): PortfolioAllocation[] {
  const allocations: PortfolioAllocation[] = [];

  const targetWeights: Record<string, number> = {
    "crude-oil": 15,
    "natural-gas": 10,
    gold: 20,
    bitcoin: 8,
    silver: 7,
    copper: 12,
    wheat: 8,
    corn: 10,
    soybean: 5,
    platinum: 5,
  };

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    const signal = MOCK_SIGNALS[assetId as AssetId]?.["D+5"];
    const target = targetWeights[assetId] || 5;
    const drift = (Math.random() - 0.5) * 4;
    const pnlPercent = (Math.random() - 0.3) * 20;

    allocations.push({
      assetId: assetId as AssetId,
      name: asset.name,
      symbol: asset.symbol,
      weight: target + drift,
      targetWeight: target,
      drift,
      signal: signal?.direction === "bullish" ? "long" : signal?.direction === "bearish" ? "short" : "neutral",
      pnl: asset.currentPrice * (target / 100) * 1000000 * (pnlPercent / 100),
      pnlPercent,
    });
  });

  return allocations.sort((a, b) => b.weight - a.weight);
}

function getRiskMetrics(): RiskMetric[] {
  return [
    { name: "Portfolio VaR (95%)", value: 2.8, threshold: 5.0, status: "safe", unit: "%" },
    { name: "Max Drawdown", value: 8.2, threshold: 15.0, status: "safe", unit: "%" },
    { name: "Gross Exposure", value: 145, threshold: 200, status: "warning", unit: "%" },
    { name: "Net Exposure", value: 42, threshold: 80, status: "safe", unit: "%" },
    { name: "Sector Concentration", value: 28, threshold: 35, status: "warning", unit: "%" },
    { name: "Beta to S&P", value: 0.65, threshold: 1.0, status: "safe", unit: "" },
  ];
}

// ============================================================================
// Components
// ============================================================================

function DashboardHeader() {
  return (
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3">
        <div className="p-2.5 bg-gradient-to-br from-amber-500 to-orange-600 rounded-xl">
          <Briefcase className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-neutral-100">Hedge Fund Alpha Console</h1>
          <p className="text-sm text-neutral-400">Alpha generation & portfolio optimization</p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Badge className="bg-amber-500/20 text-amber-400 border-amber-500/30 px-3 py-1">
          <Zap className="w-3.5 h-3.5 mr-1.5" />
          Live Alpha
        </Badge>
        <Badge variant="outline" className="text-neutral-400 border-neutral-600">
          AUM: $2.4B
        </Badge>
      </div>
    </div>
  );
}

// Portfolio Performance Summary
function PortfolioSummary() {
  const metrics = [
    { label: "YTD Return", value: "+18.4%", change: "+2.1%", positive: true, icon: TrendingUp },
    { label: "Alpha (vs SPY)", value: "+6.2%", change: "+0.8%", positive: true, icon: Star },
    { label: "Sharpe Ratio", value: "2.14", change: "+0.12", positive: true, icon: Gauge },
    { label: "Information Ratio", value: "1.87", change: "-0.05", positive: false, icon: Target },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      {metrics.map((metric) => (
        <Card key={metric.label} className="bg-gradient-to-br from-neutral-900 to-neutral-900/50 border-neutral-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <metric.icon className={cn("w-5 h-5", metric.positive ? "text-emerald-400" : "text-amber-400")} />
              <span className={cn(
                "text-xs font-medium",
                metric.positive ? "text-emerald-400" : "text-red-400"
              )}>
                {metric.change}
              </span>
            </div>
            <div className="text-2xl font-bold text-neutral-100 mb-1">{metric.value}</div>
            <div className="text-xs text-neutral-500">{metric.label}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// Alpha Opportunity Card - Premium Design
function AlphaCard({ opportunity }: { opportunity: AlphaOpportunity }) {
  const { alphaMetrics, signal } = opportunity;

  const directionConfig = {
    bullish: {
      icon: TrendingUp,
      color: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/20",
      label: "LONG",
    },
    bearish: {
      icon: TrendingDown,
      color: "text-red-400",
      bg: "bg-red-500/10",
      border: "border-red-500/20",
      label: "SHORT",
    },
    neutral: {
      icon: Minus,
      color: "text-neutral-400",
      bg: "bg-neutral-500/10",
      border: "border-neutral-500/20",
      label: "NEUTRAL",
    },
  };

  const config = directionConfig[signal.direction];
  const DirectionIcon = config.icon;

  const rankColors = {
    1: "from-amber-500 to-yellow-600",
    2: "from-neutral-400 to-neutral-500",
    3: "from-amber-700 to-amber-800",
  };

  return (
    <Card className={cn("border relative overflow-hidden transition-all hover:border-amber-500/30", config.bg, config.border)}>
      {/* Rank Badge */}
      {opportunity.rank <= 3 && (
        <div className={cn(
          "absolute top-3 right-3 w-8 h-8 rounded-full flex items-center justify-center",
          "bg-gradient-to-br text-white font-bold text-sm",
          rankColors[opportunity.rank as 1 | 2 | 3] || "from-neutral-600 to-neutral-700"
        )}>
          {opportunity.rank}
        </div>
      )}

      <CardContent className="p-4">
        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className={cn("p-2 rounded-lg", config.bg)}>
            <DirectionIcon className={cn("w-5 h-5", config.color)} />
          </div>
          <div>
            <h3 className="font-semibold text-neutral-100">{opportunity.name}</h3>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-neutral-400">{opportunity.symbol}</span>
              <Badge className={cn("text-xs", config.bg, config.color)}>{config.label}</Badge>
            </div>
          </div>
        </div>

        {/* Alpha Metrics */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="p-2 bg-neutral-800/50 rounded-lg">
            <div className="text-xs text-neutral-500 mb-1">Expected Alpha</div>
            <div className={cn(
              "text-lg font-bold font-mono",
              alphaMetrics.expectedAlpha > 0 ? "text-emerald-400" : "text-red-400"
            )}>
              {alphaMetrics.expectedAlpha > 0 ? "+" : ""}{alphaMetrics.expectedAlpha.toFixed(1)}%
            </div>
          </div>
          <div className="p-2 bg-neutral-800/50 rounded-lg">
            <div className="text-xs text-neutral-500 mb-1">Info Ratio</div>
            <div className={cn(
              "text-lg font-bold font-mono",
              alphaMetrics.informationRatio > 1 ? "text-emerald-400" :
              alphaMetrics.informationRatio > 0 ? "text-amber-400" : "text-red-400"
            )}>
              {alphaMetrics.informationRatio.toFixed(2)}
            </div>
          </div>
        </div>

        {/* Conviction & Risk */}
        <div className="space-y-2 mb-4">
          <div className="flex items-center justify-between text-sm">
            <span className="text-neutral-500">Conviction</span>
            <span className="text-neutral-200 font-medium">{alphaMetrics.convictionScore}%</span>
          </div>
          <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full",
                alphaMetrics.convictionScore > 75 ? "bg-emerald-500" :
                alphaMetrics.convictionScore > 60 ? "bg-amber-500" : "bg-neutral-500"
              )}
              style={{ width: `${alphaMetrics.convictionScore}%` }}
            />
          </div>
        </div>

        {/* Factor Exposures Mini */}
        <div className="flex items-center gap-2 flex-wrap">
          {alphaMetrics.factorExposures.slice(0, 3).map((factor) => (
            <Badge
              key={factor.factor}
              variant="outline"
              className={cn(
                "text-xs",
                factor.exposure > 0.3 ? "border-emerald-500/30 text-emerald-400" :
                factor.exposure < -0.3 ? "border-red-500/30 text-red-400" :
                "border-neutral-600 text-neutral-400"
              )}
            >
              {factor.factor}: {factor.exposure > 0 ? "+" : ""}{(factor.exposure * 100).toFixed(0)}%
            </Badge>
          ))}
        </div>

        {/* Bottom Stats */}
        <div className="flex items-center justify-between mt-4 pt-3 border-t border-neutral-800">
          <div className="text-xs text-neutral-500">
            <span className="text-neutral-400">β:</span> {alphaMetrics.betaToMarket.toFixed(2)}
          </div>
          <div className="text-xs text-neutral-500">
            <span className="text-neutral-400">Vol:</span> {alphaMetrics.volatility.toFixed(1)}%
          </div>
          <div className="text-xs text-neutral-500">
            <span className="text-neutral-400">Risk:</span> {alphaMetrics.riskContribution.toFixed(1)}%
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Portfolio Allocation Table
function PortfolioAllocations({ allocations }: { allocations: PortfolioAllocation[] }) {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <PieChart className="w-5 h-5 text-amber-400" />
          Portfolio Allocation
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-2">
          {allocations.slice(0, 6).map((alloc) => (
            <div key={alloc.assetId} className="flex items-center gap-3 p-2 hover:bg-neutral-800/50 rounded-lg transition-colors">
              {/* Signal Indicator */}
              <div className={cn(
                "w-1.5 h-8 rounded-full",
                alloc.signal === "long" ? "bg-emerald-500" :
                alloc.signal === "short" ? "bg-red-500" : "bg-neutral-600"
              )} />

              {/* Asset Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-neutral-200">{alloc.symbol}</span>
                  <Badge variant="outline" className="text-xs text-neutral-500 border-neutral-700">
                    {alloc.signal.toUpperCase()}
                  </Badge>
                </div>
                <div className="text-xs text-neutral-500">{alloc.name}</div>
              </div>

              {/* Weight Bar */}
              <div className="w-20">
                <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-amber-500 rounded-full"
                    style={{ width: `${(alloc.weight / 25) * 100}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs mt-1">
                  <span className="text-neutral-400">{alloc.weight.toFixed(1)}%</span>
                  {Math.abs(alloc.drift) > 1 && (
                    <span className={alloc.drift > 0 ? "text-emerald-400" : "text-red-400"}>
                      {alloc.drift > 0 ? "+" : ""}{alloc.drift.toFixed(1)}
                    </span>
                  )}
                </div>
              </div>

              {/* PnL */}
              <div className="text-right w-20">
                <div className={cn(
                  "text-sm font-mono font-medium",
                  alloc.pnlPercent > 0 ? "text-emerald-400" : "text-red-400"
                )}>
                  {alloc.pnlPercent > 0 ? "+" : ""}{alloc.pnlPercent.toFixed(1)}%
                </div>
                <div className="text-xs text-neutral-500">
                  ${(alloc.pnl / 1000).toFixed(0)}k
                </div>
              </div>
            </div>
          ))}
        </div>

        <button className="w-full mt-3 p-2 text-sm text-amber-400 hover:text-amber-300
                           bg-amber-500/5 hover:bg-amber-500/10 rounded-lg transition-all
                           flex items-center justify-center gap-2">
          View All Positions
          <ChevronRight className="w-4 h-4" />
        </button>
      </CardContent>
    </Card>
  );
}

// Risk Monitor Panel
function RiskMonitor({ metrics }: { metrics: RiskMetric[] }) {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Shield className="w-5 h-5 text-amber-400" />
          Risk Monitor
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0 space-y-3">
        {metrics.map((metric) => (
          <div key={metric.name} className="space-y-1.5">
            <div className="flex items-center justify-between text-sm">
              <span className="text-neutral-400">{metric.name}</span>
              <div className="flex items-center gap-2">
                <span className={cn(
                  "font-mono font-medium",
                  metric.status === "safe" ? "text-emerald-400" :
                  metric.status === "warning" ? "text-amber-400" : "text-red-400"
                )}>
                  {metric.value}{metric.unit}
                </span>
                <span className="text-xs text-neutral-600">/ {metric.threshold}{metric.unit}</span>
              </div>
            </div>
            <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all",
                  metric.status === "safe" ? "bg-emerald-500" :
                  metric.status === "warning" ? "bg-amber-500" : "bg-red-500"
                )}
                style={{ width: `${Math.min(100, (metric.value / metric.threshold) * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

// Factor Exposure Chart
function FactorExposures({ opportunities }: { opportunities: AlphaOpportunity[] }) {
  // Aggregate factor exposures across portfolio
  const aggregatedFactors = useMemo(() => {
    const factors: Record<string, { total: number; count: number }> = {};

    opportunities.forEach((opp) => {
      opp.alphaMetrics.factorExposures.forEach((f) => {
        if (!factors[f.factor]) {
          factors[f.factor] = { total: 0, count: 0 };
        }
        factors[f.factor].total += f.exposure;
        factors[f.factor].count += 1;
      });
    });

    return Object.entries(factors).map(([factor, data]) => ({
      factor,
      exposure: data.total / data.count,
    })).sort((a, b) => Math.abs(b.exposure) - Math.abs(a.exposure));
  }, [opportunities]);

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-amber-400" />
          Factor Tilts
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0 space-y-3">
        {aggregatedFactors.map((factor) => (
          <div key={factor.factor} className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="text-neutral-400">{factor.factor}</span>
              <span className={cn(
                "font-mono font-medium",
                factor.exposure > 0.2 ? "text-emerald-400" :
                factor.exposure < -0.2 ? "text-red-400" : "text-neutral-400"
              )}>
                {factor.exposure > 0 ? "+" : ""}{(factor.exposure * 100).toFixed(0)}%
              </span>
            </div>
            <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden relative">
              {/* Center marker */}
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-neutral-600" />
              {/* Exposure bar */}
              <div
                className={cn(
                  "absolute h-full rounded-full",
                  factor.exposure > 0 ? "bg-emerald-500" : "bg-red-500"
                )}
                style={{
                  left: factor.exposure > 0 ? "50%" : `${50 + factor.exposure * 50}%`,
                  width: `${Math.abs(factor.exposure) * 50}%`,
                }}
              />
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

// Signal Strength Filter
function SignalFilter({
  selected,
  onChange,
}: {
  selected: string;
  onChange: (filter: string) => void;
}) {
  const filters = [
    { id: "all", label: "All", icon: Layers },
    { id: "high-alpha", label: "High Alpha", icon: Star },
    { id: "high-ir", label: "High IR", icon: Target },
    { id: "low-vol", label: "Low Vol", icon: Shield },
  ];

  return (
    <div className="flex items-center gap-2 mb-4">
      {filters.map((filter) => (
        <button
          key={filter.id}
          onClick={() => onChange(filter.id)}
          className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
            selected === filter.id
              ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
              : "bg-neutral-800/50 text-neutral-400 hover:text-neutral-200 border border-transparent"
          )}
        >
          <filter.icon className="w-4 h-4" />
          {filter.label}
        </button>
      ))}
    </div>
  );
}

// Top Picks Banner
function TopPicksBanner({ opportunities }: { opportunities: AlphaOpportunity[] }) {
  const topPicks = opportunities.slice(0, 3);

  return (
    <Card className="bg-gradient-to-r from-amber-500/10 via-orange-500/5 to-amber-500/10 border-amber-500/20 mb-6">
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-amber-500/20 rounded-lg">
              <Trophy className="w-5 h-5 text-amber-400" />
            </div>
            <div>
              <h3 className="font-semibold text-neutral-100">Top Alpha Opportunities</h3>
              <p className="text-sm text-neutral-400">Highest risk-adjusted return potential</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {topPicks.map((pick, idx) => (
              <div key={pick.assetId} className="flex items-center gap-2">
                <div className={cn(
                  "w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold",
                  idx === 0 ? "bg-amber-500 text-neutral-900" :
                  idx === 1 ? "bg-neutral-400 text-neutral-900" :
                  "bg-amber-700 text-neutral-100"
                )}>
                  {idx + 1}
                </div>
                <div>
                  <div className="text-sm font-medium text-neutral-200">{pick.symbol}</div>
                  <div className={cn(
                    "text-xs font-mono",
                    pick.alphaMetrics.expectedAlpha > 0 ? "text-emerald-400" : "text-red-400"
                  )}>
                    {pick.alphaMetrics.expectedAlpha > 0 ? "+" : ""}
                    {pick.alphaMetrics.expectedAlpha.toFixed(1)}%α
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Dashboard
// ============================================================================

export function HedgeFundDashboard() {
  const opportunities = useMemo(() => getAlphaOpportunities(), []);
  const allocations = useMemo(() => getPortfolioAllocations(), []);
  const riskMetrics = useMemo(() => getRiskMetrics(), []);
  const [signalFilter, setSignalFilter] = useState("all");

  const filteredOpportunities = useMemo(() => {
    switch (signalFilter) {
      case "high-alpha":
        return opportunities.filter((o) => o.alphaMetrics.expectedAlpha > 5);
      case "high-ir":
        return opportunities.filter((o) => o.alphaMetrics.informationRatio > 1);
      case "low-vol":
        return opportunities.filter((o) => o.alphaMetrics.volatility < 20);
      default:
        return opportunities;
    }
  }, [opportunities, signalFilter]);

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <DashboardHeader />

      {/* Portfolio Summary */}
      <PortfolioSummary />

      {/* Top Picks Banner */}
      <TopPicksBanner opportunities={opportunities} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Alpha Opportunities - Main Content */}
        <div className="lg:col-span-3">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
              <Crosshair className="w-5 h-5 text-amber-400" />
              Alpha Opportunities
              <Badge variant="outline" className="text-neutral-400 border-neutral-700">
                {filteredOpportunities.length} signals
              </Badge>
            </h2>
          </div>

          <SignalFilter selected={signalFilter} onChange={setSignalFilter} />

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {filteredOpportunities.map((opportunity) => (
              <AlphaCard key={opportunity.assetId} opportunity={opportunity} />
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          <RiskMonitor metrics={riskMetrics} />
          <PortfolioAllocations allocations={allocations} />
          <FactorExposures opportunities={opportunities} />

          {/* Performance Note */}
          <Card className="bg-amber-500/5 border-amber-500/20">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <Brain className="w-5 h-5 text-amber-400 shrink-0" />
                <div className="text-xs text-neutral-400">
                  <strong className="text-amber-400">AI-Powered Alpha:</strong> Signals combine 10,000+ model predictions with factor analysis for optimal risk-adjusted returns.
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
