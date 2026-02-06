"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

// Components
import { HeroSignalPanel, type HeroSignalData } from "./HeroSignalPanel";
import { SignalGauge } from "./SignalGauge";
import { TradeHistoryTable, generateMockTrades } from "./TradeHistoryTable";
import { ModelConsensusOrb, generateMockConsensus } from "./ModelConsensusOrb";
import { PositionRiskCalculator } from "./PositionRiskCalculator";
import { LiveSignalCard } from "./LiveSignalCard";
import { ApiHealthIndicator } from "./ApiHealthIndicator";
import { EquityCurve } from "@/components/charts/EquityCurve";
import { PriceChart } from "@/components/charts/PriceChart";
import { MarketTicker } from "./MarketTicker";
import { MarketStatusBar } from "./MarketStatusBar";
import { CorrelationMatrix } from "./CorrelationMatrix";
import { SignalAlertFeed } from "./SignalAlertFeed";
import { VolatilitySurface } from "./VolatilitySurface";
// Quantum Status Widgets:
// - SystemQuantumStatus: Compact, system-wide regime/contagion from dashboard version
// - AssetQuantumStatus: Detailed, asset-specific with entropy gauge from quantum directory
import { QuantumStatusWidget as SystemQuantumStatus } from "./QuantumStatusWidget";
import { QuantumStatusWidget as AssetQuantumStatus } from "@/components/quantum";
import type { AssetId } from "@/types";

// API Hooks
import {
  useBackendMetrics,
  useBackendForecast,
  useBackendSignal,
  useBackendOHLCV,
  useBackendEquity,
} from "@/hooks/useApi";
import type { BackendSignal, BackendForecast, BackendMetrics } from "@/lib/api-client";

// ============================================================================
// Data Transformation Helpers
// ============================================================================

/**
 * Transform backend metrics to QuickStats format
 */
function transformMetricsToQuickStats(
  metrics: BackendMetrics | undefined,
  signalCount: number
): { totalReturn: number; sharpeRatio: number; winRate: number; activeSignals: number } {
  if (!metrics) {
    return { totalReturn: 0, sharpeRatio: 0, winRate: 0, activeSignals: 0 };
  }
  const opt = metrics.optimized_metrics;
  return {
    totalReturn: opt.total_return || 0,
    sharpeRatio: opt.sharpe_ratio || 0,
    winRate: opt.win_rate || 0,
    activeSignals: signalCount,
  };
}

/**
 * Transform backend forecast + signal data to HeroSignalData format
 */
function transformToHeroSignal(
  forecast: BackendForecast | undefined,
  metrics: BackendMetrics | undefined,
  latestSignal: BackendSignal | undefined,
  ohlcv: Array<{ date: string; open: number; high: number; low: number; close: number }> | undefined,
  assetInfo: { symbol: string; name: string }
): HeroSignalData {
  // Get current price from latest OHLCV or use a default
  const latestPrice = ohlcv && ohlcv.length > 0 ? ohlcv[ohlcv.length - 1] : null;
  const previousPrice = ohlcv && ohlcv.length > 1 ? ohlcv[ohlcv.length - 2] : null;

  const currentPrice = latestPrice?.close || 0;
  const prevClose = previousPrice?.close || currentPrice;
  const priceChange24h = currentPrice - prevClose;
  const changePercent24h = prevClose > 0 ? (priceChange24h / prevClose) * 100 : 0;

  // Determine signal direction from latest signal or forecast
  let signalDirection: "bullish" | "bearish" | "neutral" = "neutral";
  let signalStrength = 0;
  let confidence = 50;

  if (latestSignal) {
    if (latestSignal.signal === "LONG") {
      signalDirection = "bullish";
      signalStrength = Math.abs(latestSignal.net_prob * 100);
    } else if (latestSignal.signal === "SHORT") {
      signalDirection = "bearish";
      signalStrength = -Math.abs(latestSignal.net_prob * 100);
    }
    confidence = Math.abs(latestSignal.net_prob * 100);
  } else if (forecast?.signal) {
    if (forecast.signal === "BULLISH" || forecast.signal === "LONG") {
      signalDirection = "bullish";
      signalStrength = forecast.confidence || 50;
    } else if (forecast.signal === "BEARISH" || forecast.signal === "SHORT") {
      signalDirection = "bearish";
      signalStrength = -(forecast.confidence || 50);
    }
    confidence = forecast.confidence || 50;
  }

  // Calculate price targets based on ATR (estimated from recent price action)
  const recentPrices = ohlcv?.slice(-14) || [];
  const avgRange = recentPrices.length > 0
    ? recentPrices.reduce((sum, d) => sum + (d.high - d.low), 0) / recentPrices.length
    : currentPrice * 0.02;

  const atr = avgRange;
  const isBullish = signalDirection === "bullish";

  // Entry, stop loss, and targets based on signal direction
  const entryPrice = currentPrice;
  const stopLoss = isBullish
    ? currentPrice - (atr * 1.5)
    : currentPrice + (atr * 1.5);

  const targets = [
    { level: 1, price: isBullish ? currentPrice + atr : currentPrice - atr, probability: 72 },
    { level: 2, price: isBullish ? currentPrice + (atr * 2) : currentPrice - (atr * 2), probability: 55 },
    { level: 3, price: isBullish ? currentPrice + (atr * 3) : currentPrice - (atr * 3), probability: 35 },
  ];

  // Get metrics
  const opt = metrics?.optimized_metrics;

  return {
    assetName: assetInfo.name,
    symbol: assetInfo.symbol,
    currentPrice,
    priceChange24h,
    changePercent24h,
    signalDirection,
    signalStrength,
    confidence,
    entryPrice,
    stopLoss,
    targets,
    sharpeRatio: opt?.sharpe_ratio || 0,
    winRate: opt?.win_rate || 0,
    avgHoldDays: 5, // Default, could be calculated from trade history
    modelsAgreeing: Math.round((confidence / 100) * 10179), // Estimate based on confidence
    modelsTotal: 10179,
    signalAge: "Live",
    nextUpdate: "Real-time",
  };
}

/**
 * Transform backend equity data to EquityCurve format
 */
function transformEquityData(
  equityData: { equity_curve: Array<{ date: string; equity: number; trade_pnl?: number }>; final_equity: number; total_return: number } | undefined,
  startingCapital: number = 100000
): Array<{ date: string; equity: number; drawdown: number; benchmark?: number }> {
  if (!equityData?.equity_curve || equityData.equity_curve.length === 0) {
    // Return minimal data if no equity data
    return [{ date: new Date().toISOString().split("T")[0], equity: startingCapital, drawdown: 0, benchmark: startingCapital }];
  }

  let peak = startingCapital;
  let benchmarkEquity = startingCapital;
  const dailyBenchmarkReturn = 0.0003; // ~7.5% annual

  return equityData.equity_curve.map((point) => {
    // Scale equity to starting capital (backend uses 100 base)
    const scaledEquity = (point.equity / 100) * startingCapital;

    // Track peak for drawdown
    if (scaledEquity > peak) peak = scaledEquity;
    const drawdown = (scaledEquity - peak) / peak;

    // Simple benchmark simulation
    benchmarkEquity *= (1 + dailyBenchmarkReturn);

    return {
      date: point.date,
      equity: Math.round(scaledEquity),
      drawdown,
      benchmark: Math.round(benchmarkEquity),
    };
  });
}

/**
 * Transform OHLCV data for PriceChart component
 */
function transformOHLCVForChart(
  ohlcv: Array<{ date: string; open: number; high: number; low: number; close: number; volume?: number }> | undefined
): Array<{ date: string; open: number; high: number; low: number; close: number; volume: number }> {
  if (!ohlcv || ohlcv.length === 0) {
    return [];
  }
  return ohlcv.map(d => ({
    date: d.date,
    open: d.open,
    high: d.high,
    low: d.low,
    close: d.close,
    volume: d.volume || 0,
  }));
}

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

function QuickStatsBar({ stats, isLoading }: {
  stats: {
    totalReturn: number;
    sharpeRatio: number;
    winRate: number;
    activeSignals: number;
  };
  isLoading?: boolean;
}) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="flex items-center gap-3 p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
            <Skeleton className="w-10 h-10 rounded-lg bg-neutral-700" />
            <div className="space-y-2">
              <Skeleton className="h-3 w-16 bg-neutral-700" />
              <Skeleton className="h-6 w-12 bg-neutral-700" />
            </div>
          </div>
        ))}
      </div>
    );
  }
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
// Multi-Asset Signal Grid (Real API Data)
// ============================================================================

const MULTI_ASSET_CONFIG = [
  { asset: "Crude_Oil", name: "Crude Oil" },
  { asset: "Bitcoin", name: "Bitcoin" },
  { asset: "GOLD", name: "Gold" },
  { asset: "Natural_Gas", name: "Natural Gas" },
];

function MultiAssetSignalItem({ asset, name }: { asset: string; name: string }) {
  const { data: signalData, isLoading } = useBackendSignal(asset);

  // Get latest signal
  const latestSignal = useMemo(() => {
    if (!signalData?.data || signalData.data.length === 0) return null;
    return signalData.data[signalData.data.length - 1];
  }, [signalData]);

  // Calculate direction and signal value
  const { direction, signalValue, confidence } = useMemo(() => {
    if (!latestSignal) {
      return { direction: "neutral" as const, signalValue: 0, confidence: 50 };
    }

    const netProb = latestSignal.net_prob * 100;
    let dir: "bullish" | "bearish" | "neutral" = "neutral";

    if (latestSignal.signal === "LONG") {
      dir = "bullish";
    } else if (latestSignal.signal === "SHORT") {
      dir = "bearish";
    }

    return {
      direction: dir,
      signalValue: latestSignal.signal === "SHORT" ? -Math.abs(netProb) : Math.abs(netProb),
      confidence: Math.abs(netProb),
    };
  }, [latestSignal]);

  if (isLoading) {
    return (
      <div className="p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium text-neutral-200">{name}</span>
          <Skeleton className="h-5 w-16 bg-neutral-700" />
        </div>
        <div className="flex items-center justify-center h-16">
          <Skeleton className="h-12 w-12 rounded-full bg-neutral-700" />
        </div>
      </div>
    );
  }

  return (
    <div className="p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50 hover:border-neutral-600 transition-colors cursor-pointer">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-neutral-200">{name}</span>
        <Badge
          className={cn(
            "text-xs",
            direction === "bullish"
              ? "bg-green-500/10 border-green-500/30 text-green-400"
              : direction === "bearish"
              ? "bg-red-500/10 border-red-500/30 text-red-400"
              : "bg-amber-500/10 border-amber-500/30 text-amber-400"
          )}
        >
          {direction === "bullish" ? (
            <TrendingUp className="w-3 h-3 mr-1" />
          ) : direction === "bearish" ? (
            <TrendingDown className="w-3 h-3 mr-1" />
          ) : null}
          {direction}
        </Badge>
      </div>
      <div className="flex items-center justify-center">
        <SignalGauge
          value={signalValue}
          confidence={confidence}
          size="sm"
          animated={true}
        />
      </div>
    </div>
  );
}

function MultiAssetSignalGrid() {
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
          {MULTI_ASSET_CONFIG.map((item) => (
            <MultiAssetSignalItem
              key={item.asset}
              asset={item.asset}
              name={item.name}
            />
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

  // Get asset info
  const assetInfo = useMemo(() => {
    const asset = DASHBOARD_ASSETS.find((a) => a.id === selectedAsset);
    return { symbol: asset?.symbol || "CL", name: asset?.name || "Crude Oil" };
  }, [selectedAsset]);

  // Backend API asset name mapping (crude-oil -> Crude_Oil)
  const backendAssetName = useMemo(() => {
    const mapping: Record<string, string> = {
      "crude-oil": "Crude_Oil",
      "bitcoin": "Bitcoin",
      "gold": "GOLD",
      "natural-gas": "Natural_Gas",
    };
    return mapping[selectedAsset] || selectedAsset;
  }, [selectedAsset]);

  // ============================================================================
  // Fetch Real Data from Backend API
  // ============================================================================

  // Metrics for QuickStatsBar
  const { data: metricsData, isLoading: metricsLoading } = useBackendMetrics(backendAssetName);

  // Forecast for HeroSignalPanel
  const { data: forecastData, isLoading: forecastLoading } = useBackendForecast(backendAssetName);

  // Historical signals
  const { data: signalData, isLoading: signalLoading } = useBackendSignal(backendAssetName);

  // OHLCV data for price chart
  const { data: ohlcvData, isLoading: ohlcvLoading } = useBackendOHLCV(backendAssetName);

  // Equity curve data
  const { data: equityApiData, isLoading: equityLoading } = useBackendEquity(backendAssetName);

  // ============================================================================
  // Transform Backend Data for Components
  // ============================================================================

  // Get latest signal
  const latestSignal = useMemo(() => {
    if (!signalData?.data || signalData.data.length === 0) return undefined;
    return signalData.data[signalData.data.length - 1];
  }, [signalData]);

  // Count active signals (non-neutral in last 10)
  const activeSignalCount = useMemo(() => {
    if (!signalData?.data) return 0;
    const recent = signalData.data.slice(-10);
    return recent.filter(s => s.signal !== "NEUTRAL").length;
  }, [signalData]);

  // QuickStats from real metrics
  const quickStats = useMemo(() => {
    return transformMetricsToQuickStats(metricsData, activeSignalCount);
  }, [metricsData, activeSignalCount]);

  // HeroSignal from real forecast + metrics + signal data
  const heroSignal = useMemo(() => {
    return transformToHeroSignal(
      forecastData,
      metricsData,
      latestSignal,
      ohlcvData,
      assetInfo
    );
  }, [forecastData, metricsData, latestSignal, ohlcvData, assetInfo]);

  // Equity curve from real data
  const equityData = useMemo(() => {
    return transformEquityData(equityApiData, 100000);
  }, [equityApiData]);

  // OHLCV for price chart
  const chartData = useMemo(() => ({
    ohlc: transformOHLCVForChart(ohlcvData),
  }), [ohlcvData]);

  // Mock data still used for some components (until backend supports)
  const trades = useMemo(() => generateMockTrades(25), []);
  const consensusData = useMemo(() => generateMockConsensus(
    heroSignal.signalDirection === "neutral" ? "neutral" : heroSignal.signalDirection,
    heroSignal.confidence >= 75 ? "strong" : heroSignal.confidence >= 55 ? "moderate" : "weak"
  ), [heroSignal.signalDirection, heroSignal.confidence]);

  // Loading state
  const isLoading = metricsLoading || forecastLoading || signalLoading || ohlcvLoading;

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
          <div className="flex flex-col items-end gap-2">
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
            {/* System-wide Quantum Status */}
            <SystemQuantumStatus compact />
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <QuickStatsBar stats={quickStats} isLoading={metricsLoading} />

      {/* Asset Selector */}
      <div className="flex items-center justify-between">
        <AssetTabSelector selectedAsset={selectedAsset} onSelect={setSelectedAsset} />
        <div className="flex items-center gap-2 text-xs text-neutral-500">
          <Clock className="w-3.5 h-3.5" />
          <span>{isLoading ? "Loading..." : `Updated ${new Date().toLocaleTimeString()}`}</span>
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
          <HeroSignalPanel data={heroSignal} onTrade={() => {}} />

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
                  {ohlcvLoading ? (
                    <div className="h-full flex items-center justify-center">
                      <div className="text-center">
                        <Skeleton className="h-64 w-full bg-neutral-800 mb-2" />
                        <span className="text-xs text-neutral-500">Loading price data...</span>
                      </div>
                    </div>
                  ) : chartData.ohlc.length > 0 ? (
                    <PriceChart
                      data={chartData.ohlc}
                      assetName={assetInfo.name}
                      showVolume={true}
                      chartType="candlestick"
                    />
                  ) : (
                    <div className="h-full flex items-center justify-center text-neutral-500">
                      No price data available
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Sidebar: Quantum Regime + Multi-Asset Grid */}
            <div className="space-y-6">
              {/* Quantum Regime Status - Asset-specific */}
              <AssetQuantumStatus
                asset={assetInfo.symbol}
                size="md"
                showDetails={true}
              />

              {/* Multi-Asset Grid */}
              <MultiAssetSignalGrid />
            </div>
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
                assetSymbol={assetInfo.symbol}
                assetName={assetInfo.name}
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
                {ohlcvLoading ? (
                  <div className="h-full flex items-center justify-center">
                    <Skeleton className="h-64 w-full bg-neutral-800" />
                  </div>
                ) : chartData.ohlc.length > 0 ? (
                  <PriceChart
                    data={chartData.ohlc}
                    assetName={assetInfo.name}
                    showVolume={true}
                    chartType="candlestick"
                  />
                ) : (
                  <div className="h-full flex items-center justify-center text-neutral-500">
                    No price data available
                  </div>
                )}
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
                {equityLoading ? (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <Skeleton className="h-80 w-full bg-neutral-800 mb-2" />
                      <span className="text-xs text-neutral-500">Loading equity data...</span>
                    </div>
                  </div>
                ) : equityData.length > 1 ? (
                  <EquityCurve
                    data={equityData}
                    showDrawdown={true}
                    showBenchmark={true}
                    strategyName="Alpha Gen Pro"
                    benchmarkName="Buy & Hold"
                  />
                ) : (
                  <div className="h-full flex items-center justify-center text-neutral-500">
                    No equity data available
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Performance Metrics Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              {
                label: "Total Return",
                value: metricsData ? `${metricsData.optimized_metrics.total_return >= 0 ? "+" : ""}${metricsData.optimized_metrics.total_return.toFixed(1)}%` : "--",
                color: metricsData && metricsData.optimized_metrics.total_return >= 0 ? "text-green-400" : "text-red-400",
                icon: TrendingUp
              },
              {
                label: "Max Drawdown",
                value: metricsData ? `${metricsData.optimized_metrics.max_drawdown.toFixed(1)}%` : "--",
                color: "text-red-400",
                icon: TrendingDown
              },
              {
                label: "Sharpe Ratio",
                value: metricsData ? metricsData.optimized_metrics.sharpe_ratio.toFixed(2) : "--",
                color: "text-blue-400",
                icon: BarChart3
              },
              {
                label: "Profit Factor",
                value: metricsData ? metricsData.optimized_metrics.profit_factor.toFixed(2) : "--",
                color: "text-purple-400",
                icon: Target
              },
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
