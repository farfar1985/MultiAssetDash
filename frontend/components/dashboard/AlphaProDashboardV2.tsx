"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

// Components
import { HeroSignalPanel, generateMockHeroSignal } from "./HeroSignalPanel";
import { SignalGauge } from "./SignalGauge";
import { TradeHistoryTable, generateMockTrades } from "./TradeHistoryTable";
import { ModelConsensusOrb, generateMockConsensus } from "./ModelConsensusOrb";
import { PositionRiskCalculator } from "./PositionRiskCalculator";
import { LiveSignalCard } from "./LiveSignalCard";
import { ApiHealthIndicator } from "./ApiHealthIndicator";
import { EquityCurve, generateEquityData } from "@/components/charts/EquityCurve";
import { PriceChart } from "@/components/charts/PriceChart";
import { getChartData } from "@/lib/mock-chart-data";
import { MarketTicker } from "./MarketTicker";
import { MarketStatusBar } from "./MarketStatusBar";
import { CorrelationMatrix } from "./CorrelationMatrix";
import { SignalAlertFeed } from "./SignalAlertFeed";
import { VolatilitySurface } from "./VolatilitySurface";
import type { AssetId } from "@/types";

import {
  Zap,
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  History,
  Target,
  Flame,
  Clock,
  ChevronRight,
  Sparkles,
  LineChart,
  Scale,
  Users,
  GitBranch,
} from "lucide-react";

// ============================================================================
// Asset Selector Tabs
// ============================================================================

const DASHBOARD_ASSETS: Array<{ id: AssetId; name: string; symbol: string }> = [
  { id: "crude-oil", name: "Crude Oil", symbol: "CL" },
  { id: "bitcoin", name: "Bitcoin", symbol: "BTC" },
  { id: "gold", name: "Gold", symbol: "GC" },
  { id: "natural-gas", name: "Natural Gas", symbol: "NG" },
];

function AssetTabSelector({
  selectedAsset,
  onSelect,
}: {
  selectedAsset: AssetId;
  onSelect: (asset: AssetId) => void;
}) {
  return (
    <div className="flex items-center gap-2 p-1 bg-neutral-800/50 rounded-xl border border-neutral-700/50">
      {DASHBOARD_ASSETS.map((asset) => {
        const isSelected = asset.id === selectedAsset;
        return (
          <button
            key={asset.id}
            onClick={() => onSelect(asset.id)}
            className={cn(
              "px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
              "flex items-center gap-2",
              isSelected
                ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700/50"
            )}
          >
            <span className="font-mono text-xs">{asset.symbol}</span>
            <span className="hidden sm:inline">{asset.name}</span>
          </button>
        );
      })}
    </div>
  );
}

// ============================================================================
// Quick Stats Bar
// ============================================================================

function QuickStatsBar({ stats }: {
  stats: {
    totalReturn: number;
    sharpeRatio: number;
    winRate: number;
    activeSignals: number;
  };
}) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <div className="flex items-center gap-3 p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
        <div className="p-2 bg-green-500/10 rounded-lg">
          <TrendingUp className="w-4 h-4 text-green-400" />
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">Total Return</div>
          <div className={cn(
            "font-mono text-lg font-bold",
            stats.totalReturn >= 0 ? "text-green-400" : "text-red-400"
          )}>
            {stats.totalReturn >= 0 ? "+" : ""}{stats.totalReturn.toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3 p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
        <div className="p-2 bg-blue-500/10 rounded-lg">
          <BarChart3 className="w-4 h-4 text-blue-400" />
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">Sharpe Ratio</div>
          <div className="font-mono text-lg font-bold text-blue-400">
            {stats.sharpeRatio.toFixed(2)}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3 p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
        <div className="p-2 bg-purple-500/10 rounded-lg">
          <Target className="w-4 h-4 text-purple-400" />
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">Win Rate</div>
          <div className="font-mono text-lg font-bold text-purple-400">
            {stats.winRate.toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3 p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
        <div className="p-2 bg-amber-500/10 rounded-lg">
          <Flame className="w-4 h-4 text-amber-400" />
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">Active Signals</div>
          <div className="font-mono text-lg font-bold text-amber-400">
            {stats.activeSignals}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Multi-Asset Signal Grid
// ============================================================================

function MultiAssetSignalGrid() {
  const signalData = useMemo(() => [
    { asset: "Crude_Oil", name: "Crude Oil", signal: 72, direction: "bullish" as const },
    { asset: "Bitcoin", name: "Bitcoin", signal: 85, direction: "bullish" as const },
    { asset: "SP500", name: "S&P 500", signal: -15, direction: "neutral" as const },
    { asset: "GOLD", name: "Gold", signal: -48, direction: "bearish" as const },
  ], []);

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Multi-Asset Overview
          </CardTitle>
          <ApiHealthIndicator />
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <div className="grid grid-cols-2 gap-4">
          {signalData.map((item) => (
            <div
              key={item.asset}
              className="p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50 hover:border-neutral-600 transition-colors cursor-pointer"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-neutral-200">{item.name}</span>
                <Badge
                  className={cn(
                    "text-xs",
                    item.direction === "bullish"
                      ? "bg-green-500/10 border-green-500/30 text-green-400"
                      : item.direction === "bearish"
                      ? "bg-red-500/10 border-red-500/30 text-red-400"
                      : "bg-amber-500/10 border-amber-500/30 text-amber-400"
                  )}
                >
                  {item.direction === "bullish" ? (
                    <TrendingUp className="w-3 h-3 mr-1" />
                  ) : item.direction === "bearish" ? (
                    <TrendingDown className="w-3 h-3 mr-1" />
                  ) : null}
                  {item.direction}
                </Badge>
              </div>
              <div className="flex items-center justify-center">
                <SignalGauge
                  value={item.signal}
                  confidence={60 + Math.random() * 30}
                  size="sm"
                  animated={true}
                />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Dashboard Component
// ============================================================================

export function AlphaProDashboardV2() {
  const [selectedAsset, setSelectedAsset] = useState<AssetId>("crude-oil");
  const [activeTab, setActiveTab] = useState("overview");

  // Generate data based on selected asset
  const heroSignal = useMemo(() => {
    const assetInfo = DASHBOARD_ASSETS.find((a) => a.id === selectedAsset);
    return generateMockHeroSignal(
      assetInfo?.symbol || "CL",
      assetInfo?.name || "Crude Oil",
      Math.random() > 0.3 ? "bullish" : "bearish"
    );
  }, [selectedAsset]);

  const equityData = useMemo(() => generateEquityData(365, 100000, 0.0008, 0.015), []);
  const trades = useMemo(() => generateMockTrades(25), []);
  const chartData = useMemo(() => getChartData(selectedAsset), [selectedAsset]);
  const consensusData = useMemo(() => generateMockConsensus(
    heroSignal.signalDirection === "neutral" ? "neutral" : heroSignal.signalDirection,
    heroSignal.confidence >= 75 ? "strong" : heroSignal.confidence >= 55 ? "moderate" : "weak"
  ), [heroSignal.signalDirection, heroSignal.confidence]);

  const quickStats = useMemo(() => ({
    totalReturn: 28.4,
    sharpeRatio: 2.34,
    winRate: 62.5,
    activeSignals: 8,
  }), []);

  return (
    <div className="flex flex-col min-h-screen -m-6">
      {/* Bloomberg-style Market Ticker */}
      <MarketTicker speed="normal" showVolume={false} pauseOnHover={true} />

      {/* Market Status Bar */}
      <MarketStatusBar showFullDetails={true} />

      {/* Main Content */}
      <div className="flex-1 p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/30 via-blue-900/20 to-purple-900/30 border border-purple-500/20 rounded-xl p-6">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-purple-500/20 rounded-xl border border-purple-500/30">
              <Zap className="w-8 h-8 text-purple-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-neutral-100">Alpha Gen Pro</h1>
              <p className="text-sm text-neutral-400">
                Professional-grade ensemble signals for alpha generation
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge className="bg-green-500/10 border-green-500/30 text-green-400 px-3 py-1.5">
              <Activity className="w-3.5 h-3.5 mr-1.5 animate-pulse" />
              Live
            </Badge>
            <Badge className="bg-purple-500/10 border-purple-500/30 text-purple-300 px-3 py-1.5">
              <Sparkles className="w-3.5 h-3.5 mr-1.5" />
              10,179 Models
            </Badge>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <QuickStatsBar stats={quickStats} />

      {/* Asset Selector */}
      <div className="flex items-center justify-between">
        <AssetTabSelector selectedAsset={selectedAsset} onSelect={setSelectedAsset} />
        <div className="flex items-center gap-2 text-xs text-neutral-500">
          <Clock className="w-3.5 h-3.5" />
          <span>Updated 2 min ago</span>
        </div>
      </div>

      <Separator className="bg-neutral-800" />

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="bg-neutral-800/50 border border-neutral-700/50 p-1">
          <TabsTrigger
            value="overview"
            className="data-[state=active]:bg-blue-500/20 data-[state=active]:text-blue-400"
          >
            <Target className="w-4 h-4 mr-2" />
            Signal Overview
          </TabsTrigger>
          <TabsTrigger
            value="trade-entry"
            className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400"
          >
            <Scale className="w-4 h-4 mr-2" />
            Trade Entry
          </TabsTrigger>
          <TabsTrigger
            value="performance"
            className="data-[state=active]:bg-green-500/20 data-[state=active]:text-green-400"
          >
            <LineChart className="w-4 h-4 mr-2" />
            Performance
          </TabsTrigger>
          <TabsTrigger
            value="history"
            className="data-[state=active]:bg-amber-500/20 data-[state=active]:text-amber-400"
          >
            <History className="w-4 h-4 mr-2" />
            Trade History
          </TabsTrigger>
          <TabsTrigger
            value="analytics"
            className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400"
          >
            <GitBranch className="w-4 h-4 mr-2" />
            Analytics
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Hero Signal Panel */}
          <HeroSignalPanel data={heroSignal} onTrade={() => console.log("Trade clicked")} />

          {/* Two Column Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Price Chart */}
            <Card className="lg:col-span-2 bg-neutral-900/50 border-neutral-800">
              <CardHeader className="p-4 pb-2">
                <CardTitle className="text-sm font-semibold text-neutral-200">
                  Price Action
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 pt-0">
                <div className="h-80">
                  <PriceChart
                    data={chartData.ohlc}
                    assetName={DASHBOARD_ASSETS.find((a) => a.id === selectedAsset)?.name || ""}
                    showVolume={true}
                    chartType="candlestick"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Multi-Asset Grid */}
            <MultiAssetSignalGrid />
          </div>

          {/* Live Backend Signals */}
          <Card className="bg-neutral-900/50 border-neutral-800">
            <CardHeader className="p-4 pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-green-400" />
                  Live Backend Signals
                </CardTitle>
                <span className="text-xs text-neutral-500">From API</span>
              </div>
            </CardHeader>
            <CardContent className="p-4 pt-0">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <LiveSignalCard asset="Crude_Oil" displayName="Crude Oil" />
                <LiveSignalCard asset="Bitcoin" displayName="Bitcoin" />
                <LiveSignalCard asset="SP500" displayName="S&P 500" />
                <LiveSignalCard asset="GOLD" displayName="Gold" />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Trade Entry Tab */}
        <TabsContent value="trade-entry" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Model Consensus Orb */}
            <Card className="lg:col-span-1 bg-neutral-900/50 border-neutral-800">
              <CardHeader className="p-4 pb-2">
                <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                  <Users className="w-4 h-4 text-cyan-400" />
                  Model Voting
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 pt-2 flex justify-center">
                <ModelConsensusOrb
                  votes={consensusData}
                  size="md"
                  showDetails={true}
                  animated={true}
                />
              </CardContent>
            </Card>

            {/* Position Risk Calculator */}
            <div className="lg:col-span-2">
              <PositionRiskCalculator
                assetSymbol={DASHBOARD_ASSETS.find((a) => a.id === selectedAsset)?.symbol || "CL"}
                assetName={DASHBOARD_ASSETS.find((a) => a.id === selectedAsset)?.name || "Crude Oil"}
                currentPrice={heroSignal.currentPrice}
                direction={heroSignal.signalDirection === "bearish" ? "short" : "long"}
                initialConfig={{
                  accountSize: 100000,
                  riskPercent: 1,
                  entryPrice: heroSignal.entryPrice,
                  stopLoss: heroSignal.stopLoss,
                  takeProfitLevels: heroSignal.targets.map((t) => t.price),
                }}
              />
            </div>
          </div>

          {/* Price Chart with Entry/Exit Levels */}
          <Card className="bg-neutral-900/50 border-neutral-800">
            <CardHeader className="p-4 pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-semibold text-neutral-200">
                  Entry Setup Visualization
                </CardTitle>
                <div className="flex items-center gap-2 text-xs">
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-blue-500" />
                    <span className="text-neutral-500">Entry</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-red-500" />
                    <span className="text-neutral-500">Stop</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-green-500" />
                    <span className="text-neutral-500">Targets</span>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-4 pt-0">
              <div className="h-80">
                <PriceChart
                  data={chartData.ohlc}
                  assetName={DASHBOARD_ASSETS.find((a) => a.id === selectedAsset)?.name || ""}
                  showVolume={true}
                  chartType="candlestick"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-6">
          {/* Equity Curve */}
          <Card className="bg-neutral-900/50 border-neutral-800">
            <CardHeader className="p-4 pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                  <LineChart className="w-4 h-4 text-green-400" />
                  Strategy Performance
                </CardTitle>
                <div className="flex items-center gap-2">
                  <Badge className="bg-green-500/10 border-green-500/30 text-green-400 text-xs">
                    Alpha Strategy
                  </Badge>
                  <Badge className="bg-neutral-700 border-neutral-600 text-neutral-400 text-xs">
                    vs Buy & Hold
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-4 pt-0">
              <div className="h-96">
                <EquityCurve
                  data={equityData}
                  showDrawdown={true}
                  showBenchmark={true}
                  strategyName="Alpha Gen Pro"
                  benchmarkName="Buy & Hold"
                />
              </div>
            </CardContent>
          </Card>

          {/* Performance Metrics Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: "Total Return", value: "+28.4%", color: "text-green-400", icon: TrendingUp },
              { label: "Max Drawdown", value: "-12.3%", color: "text-red-400", icon: TrendingDown },
              { label: "Sharpe Ratio", value: "2.34", color: "text-blue-400", icon: BarChart3 },
              { label: "Profit Factor", value: "2.18", color: "text-purple-400", icon: Target },
            ].map((metric) => (
              <Card key={metric.label} className="bg-neutral-900/50 border-neutral-800">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <metric.icon className={cn("w-4 h-4", metric.color)} />
                    <span className="text-xs uppercase tracking-wider text-neutral-500">
                      {metric.label}
                    </span>
                  </div>
                  <div className={cn("font-mono text-2xl font-bold", metric.color)}>
                    {metric.value}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Trade History Tab */}
        <TabsContent value="history" className="space-y-6">
          <TradeHistoryTable trades={trades} maxRows={15} showFilters={true} />
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Volatility Surface */}
            <div className="lg:col-span-2">
              <VolatilitySurface
                showMiniChart={true}
                compact={false}
              />
            </div>

            {/* Signal Alert Feed */}
            <div className="lg:col-span-1 h-[420px]">
              <SignalAlertFeed
                maxAlerts={8}
                autoPlay={true}
                enableSound={false}
                compact={false}
              />
            </div>
          </div>

          {/* Correlation Matrix */}
          <CorrelationMatrix
            size="md"
            showLabels={true}
            interactive={true}
            title="Cross-Asset Correlations"
          />

          {/* Additional Analytics */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: "Active Positions", value: "5", sublabel: "across 4 assets", color: "text-blue-400", icon: Target },
              { label: "Portfolio Beta", value: "1.24", sublabel: "vs S&P 500", color: "text-purple-400", icon: BarChart3 },
              { label: "Daily VaR (95%)", value: "-$2,340", sublabel: "at risk", color: "text-red-400", icon: Activity },
              { label: "Signal Quality", value: "A+", sublabel: "ensemble rating", color: "text-green-400", icon: Zap },
            ].map((metric) => (
              <Card key={metric.label} className="bg-neutral-900/50 border-neutral-800">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <metric.icon className={cn("w-4 h-4", metric.color)} />
                    <span className="text-xs uppercase tracking-wider text-neutral-500">
                      {metric.label}
                    </span>
                  </div>
                  <div className={cn("font-mono text-2xl font-bold", metric.color)}>
                    {metric.value}
                  </div>
                  <div className="text-xs text-neutral-600 mt-1">
                    {metric.sublabel}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>

      {/* Footer */}
      <div className="flex items-center justify-between py-4 border-t border-neutral-800 text-xs text-neutral-500">
        <div className="flex items-center gap-4">
          <span>Last updated: {new Date().toLocaleString()}</span>
          <Badge className="bg-neutral-800 border-neutral-700 text-neutral-400">
            quantum_ml v2.1
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <span>Powered by QDT Ensemble Pipeline</span>
          <ChevronRight className="w-4 h-4" />
        </div>
      </div>
      </div> {/* Close main content div */}
    </div>
  );
}
