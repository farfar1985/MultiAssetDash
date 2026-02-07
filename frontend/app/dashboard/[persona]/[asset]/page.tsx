"use client";

import { notFound, useParams } from "next/navigation";
import Link from "next/link";
import { useState, useMemo } from "react";
import { PERSONAS, ASSETS, type PersonaId, type AssetId } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { MOCK_ASSETS, MOCK_SIGNALS, formatPrice } from "@/lib/mock-data";
import { getChartData } from "@/lib/mock-chart-data";
import { getPracticalMetrics } from "@/lib/mock-practical-metrics";
import { cn } from "@/lib/utils";
import {
  PriceChart,
  SignalChart,
  ModelAgreementChart,
} from "@/components/charts";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { EnsembleMethod } from "@/lib/api-client";

// Dashboard components
import { PracticalMetrics } from "@/components/dashboard/PracticalMetrics";
import { PracticalInsights } from "@/components/dashboard/PracticalInsights";
import { BacktestMetrics } from "@/components/dashboard/BacktestMetrics";
import { HorizonPairHeatmap } from "@/components/dashboard/HorizonPairHeatmap";
import { HorizonPairInsights } from "@/components/dashboard/HorizonPairInsights";
import { HistoricalRewind } from "@/components/dashboard/HistoricalRewind";
import {
  EnsembleSelector,
  ENSEMBLE_METHODS,
  getMethodConfig,
} from "@/components/dashboard/EnsembleSelector";

export default function AssetDetailPage() {
  const params = useParams();
  const personaId = params.persona as PersonaId;
  const assetId = params.asset as AssetId;

  const [chartType, setChartType] = useState<"candlestick" | "line">("candlestick");
  const [ensembleMethod, setEnsembleMethod] = useState<EnsembleMethod>("top_k_sharpe");

  const persona = PERSONAS[personaId];
  const asset = ASSETS[assetId];
  const mockAsset = MOCK_ASSETS[assetId];
  const signals = MOCK_SIGNALS[assetId];
  const chartData = getChartData(assetId);
  const practicalMetrics = getPracticalMetrics(assetId);

  if (!persona || !asset) {
    notFound();
  }

  const priceChangeColor =
    (mockAsset?.changePercent24h ?? 0) >= 0 ? "text-green-500" : "text-red-500";

  // Determine signal direction badge
  const currentSignal = signals?.["D+1"];
  const signalDirectionBadge = useMemo(() => {
    if (!currentSignal) return null;
    const directionColors = {
      bullish: "bg-green-500/20 text-green-400 border-green-500/30",
      bearish: "bg-red-500/20 text-red-400 border-red-500/30",
      neutral: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    };
    const directionIcons = {
      bullish: "↑",
      bearish: "↓",
      neutral: "→",
    };
    return {
      className: directionColors[currentSignal.direction],
      icon: directionIcons[currentSignal.direction],
      label: currentSignal.direction.charAt(0).toUpperCase() + currentSignal.direction.slice(1),
    };
  }, [currentSignal]);

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-neutral-400">
        <Link
          href={`/dashboard/${personaId}`}
          className="hover:text-neutral-200 transition-colors"
        >
          {persona.name}
        </Link>
        <span>/</span>
        <span className="text-neutral-200">{asset.name}</span>
      </nav>

      {/* Asset Header */}
      <div className="pb-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-neutral-100">{asset.name}</h1>
              {signalDirectionBadge && (
                <Badge
                  variant="outline"
                  className={cn("text-sm font-semibold", signalDirectionBadge.className)}
                >
                  {signalDirectionBadge.icon} {signalDirectionBadge.label}
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-sm font-mono text-blue-500">{asset.symbol}</span>
              <span className="text-xs px-2 py-0.5 rounded bg-neutral-800 text-neutral-400 capitalize">
                {asset.category}
              </span>
              <Badge
                variant="outline"
                className="text-xs bg-purple-500/10 border-purple-500/30 text-purple-400"
              >
                10,179 models
              </Badge>
            </div>
          </div>
          {mockAsset && (
            <div className="text-right">
              <div className="text-2xl font-bold font-mono text-neutral-100">
                {formatPrice(mockAsset.currentPrice, mockAsset.symbol)}
              </div>
              <div className={cn("text-sm font-mono", priceChangeColor)}>
                {mockAsset.changePercent24h >= 0 ? "+" : ""}
                {mockAsset.changePercent24h.toFixed(2)}% (24h)
              </div>
            </div>
          )}
        </div>
      </div>

      <Separator className="bg-neutral-800" />

      {/* Main 2-Column Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LEFT COLUMN - Main Content (2/3 width) */}
        <div className="lg:col-span-2 space-y-6">
          {/* Price Chart */}
          <Card className="bg-neutral-900/50 border-neutral-800">
            <CardHeader className="p-4 pb-2 flex flex-row items-center justify-between">
              <CardTitle className="text-sm font-semibold text-neutral-200">
                Price History
              </CardTitle>
              <Select
                value={chartType}
                onValueChange={(v) => setChartType(v as "candlestick" | "line")}
              >
                <SelectTrigger className="w-32 h-7 text-xs bg-neutral-800 border-neutral-700">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-neutral-900 border-neutral-700">
                  <SelectItem value="candlestick" className="text-xs">
                    Candlestick
                  </SelectItem>
                  <SelectItem value="line" className="text-xs">
                    Line
                  </SelectItem>
                </SelectContent>
              </Select>
            </CardHeader>
            <CardContent className="p-4 pt-0">
              <div className="h-80">
                <PriceChart
                  data={chartData.ohlc ?? []}
                  timeframe="1D"
                  assetName={asset.name}
                  showVolume={true}
                  chartType={chartType}
                />
              </div>
            </CardContent>
          </Card>

          {/* Signal Strength Chart */}
          <Card className="bg-neutral-900/50 border-neutral-800">
            <CardHeader className="p-4 pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-semibold text-neutral-200">
                  Signal Strength History
                </CardTitle>
                <Badge
                  variant="outline"
                  className="text-xs bg-blue-500/10 border-blue-500/30 text-blue-400"
                >
                  90 days
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="p-4 pt-0">
              <div className="h-56">
                <SignalChart data={chartData.signalHistory ?? []} showConfidenceBand={true} />
              </div>
            </CardContent>
          </Card>

          {/* Practical Insights - Plain English Recommendation */}
          <PracticalInsights
            asset={assetId}
            assetName={asset.name}
            data={practicalMetrics}
          />
        </div>

        {/* RIGHT COLUMN - Sidebar (1/3 width) */}
        <div className="space-y-6">
          {/* Practical Metrics - Actionability */}
          <PracticalMetrics asset={assetId} data={practicalMetrics} />

          {/* Backtest Metrics - Official Performance */}
          <BacktestMetrics symbol={assetId} />

          {/* Model Agreement Chart */}
          <Card className="bg-neutral-900/50 border-neutral-800">
            <CardHeader className="p-4 pb-2">
              <CardTitle className="text-sm font-semibold text-neutral-200">
                Model Consensus
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 pt-0">
              {chartData.modelAgreement ? (
                <>
                  <div className="h-48">
                    <ModelAgreementChart
                      bullishCount={chartData.modelAgreement.bullishCount}
                      bearishCount={chartData.modelAgreement.bearishCount}
                      neutralCount={chartData.modelAgreement.neutralCount}
                      totalModels={chartData.modelAgreement.totalModels}
                      overallDirection={chartData.modelAgreement.overallDirection}
                    />
                  </div>
                  <div className="mt-3 space-y-2 text-xs">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-green-500"></span>
                        <span className="text-neutral-400">Bullish</span>
                      </div>
                      <span className="font-mono text-neutral-200">
                        {chartData.modelAgreement.bullishCount.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-red-500"></span>
                        <span className="text-neutral-400">Bearish</span>
                      </div>
                      <span className="font-mono text-neutral-200">
                        {chartData.modelAgreement.bearishCount.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-neutral-500"></span>
                        <span className="text-neutral-400">Neutral</span>
                      </div>
                      <span className="font-mono text-neutral-200">
                        {chartData.modelAgreement.neutralCount.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="h-48 flex items-center justify-center text-neutral-500">
                  No model agreement data available
                </div>
              )}
            </CardContent>
          </Card>

          {/* Horizon Pair Heatmap */}
          <HorizonPairHeatmap assetId={assetId} height={300} />

          {/* Historical Rewind - Time Travel */}
          <HistoricalRewind symbol={assetId} />
        </div>
      </div>

      {/* Bottom Section - Full Width */}
      <Separator className="bg-neutral-800" />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Horizon Pair Insights */}
        <HorizonPairInsights assetId={assetId} />

        {/* Ensemble Selector Card */}
        <Card className="bg-neutral-900/50 border-neutral-800">
          <CardHeader className="p-4 pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CardTitle className="text-sm font-semibold text-neutral-200">
                  Ensemble Method
                </CardTitle>
                <Badge className="bg-green-500/10 border-green-500/30 text-green-500 text-xs">
                  Active
                </Badge>
              </div>
              <span className="text-xs text-neutral-500 font-mono">
                {getMethodConfig(ensembleMethod).label}
              </span>
            </div>
          </CardHeader>
          <CardContent className="p-4 pt-0 space-y-4">
            <EnsembleSelector
              value={ensembleMethod}
              onChange={setEnsembleMethod}
              variant="tabs"
            />

            {/* Method Description */}
            <div className="p-3 bg-neutral-800/30 rounded-lg">
              <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
                Method Description
              </div>
              <p className="text-sm text-neutral-300">
                {getMethodConfig(ensembleMethod).description}
              </p>
            </div>

            {/* Method Performance Comparison */}
            <div className="space-y-2">
              <div className="text-[10px] uppercase tracking-wider text-neutral-500">
                Method Performance
              </div>
              {ENSEMBLE_METHODS.slice(0, 4).map((method) => {
                const isActive = method.value === ensembleMethod;
                // Mock performance data
                const performance = {
                  accuracy_weighted: { sharpe: 2.12, accuracy: 57.2 },
                  exponential_decay: { sharpe: 2.08, accuracy: 56.8 },
                  top_k_sharpe: { sharpe: 2.45, accuracy: 59.4 },
                  ridge_stacking: { sharpe: 2.28, accuracy: 58.1 },
                  inverse_variance: { sharpe: 2.15, accuracy: 57.6 },
                  pairwise_slope: { sharpe: 2.34, accuracy: 58.8 },
                }[method.value] || { sharpe: 2.0, accuracy: 55.0 };

                return (
                  <div
                    key={method.value}
                    className={cn(
                      "flex items-center justify-between p-2 rounded-lg border transition-all",
                      isActive
                        ? "bg-blue-500/10 border-blue-500/30"
                        : "bg-neutral-800/20 border-neutral-800 hover:border-neutral-700"
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={cn(
                          "text-xs",
                          isActive ? "text-blue-400 font-medium" : "text-neutral-400"
                        )}
                      >
                        {method.shortLabel}
                      </span>
                      {method.badge && (
                        <Badge
                          variant="outline"
                          className="text-[9px] px-1 py-0 h-3.5 bg-green-500/10 border-green-500/30 text-green-500"
                        >
                          {method.badge}
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-mono text-neutral-500">
                        SR {performance.sharpe.toFixed(2)}
                      </span>
                      <span className="text-xs font-mono text-neutral-400">
                        {performance.accuracy.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between pt-2 border-t border-neutral-800">
              <span className="text-[10px] text-neutral-600">
                Switching methods recomputes all signals
              </span>
              <span className="text-[10px] text-neutral-600 font-mono">
                10,179 models active
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Signals by Horizon - Full Width Table */}
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardHeader className="p-4 pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-semibold text-neutral-200">
              Signals Across All Horizons
            </CardTitle>
            <Badge
              variant="outline"
              className="text-xs bg-neutral-800/50 border-neutral-700 text-neutral-300"
            >
              {asset.symbol}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="p-4 pt-0">
          <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
            {signals &&
              (["D+1", "D+5", "D+10"] as const).map((horizon) => {
                const signal = signals[horizon];
                const directionColors = {
                  bullish: "text-green-500",
                  bearish: "text-red-500",
                  neutral: "text-yellow-500",
                };
                const directionBg = {
                  bullish: "bg-green-500/10 border-green-500/20",
                  bearish: "bg-red-500/10 border-red-500/20",
                  neutral: "bg-yellow-500/10 border-yellow-500/20",
                };
                const directionIcons = {
                  bullish: "↑",
                  bearish: "↓",
                  neutral: "→",
                };

                return (
                  <div
                    key={horizon}
                    className={cn(
                      "p-3 rounded-lg border transition-all hover:scale-[1.02]",
                      directionBg[signal.direction]
                    )}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-xs font-semibold text-blue-400">
                        {horizon}
                      </span>
                      <span
                        className={cn(
                          "text-lg font-bold",
                          directionColors[signal.direction]
                        )}
                      >
                        {directionIcons[signal.direction]}
                      </span>
                    </div>
                    <div
                      className={cn(
                        "text-sm font-semibold capitalize mb-1",
                        directionColors[signal.direction]
                      )}
                    >
                      {signal.direction}
                    </div>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-neutral-500">Confidence</span>
                        <span className="font-mono text-neutral-300">
                          {signal.confidence}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-500">Dir. Acc.</span>
                        <span className="font-mono text-neutral-300">
                          {signal.directionalAccuracy.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-500">Sharpe</span>
                        <span className="font-mono text-neutral-300">
                          {signal.sharpeRatio.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-500">Models</span>
                        <span className="font-mono text-neutral-400 text-[10px]">
                          {signal.modelsAgreeing.toLocaleString()}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        </CardContent>
      </Card>

      {/* Footer */}
      <div className="flex items-center justify-between py-4 border-t border-neutral-800">
        <div className="flex items-center gap-4">
          <span className="text-xs text-neutral-500">
            Last updated: {new Date().toLocaleString()}
          </span>
          <Badge
            variant="outline"
            className="text-xs bg-neutral-800/50 border-neutral-700 text-neutral-400"
          >
            quantum_ml v2.1
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-neutral-600">
            Powered by QDT Ensemble Pipeline
          </span>
        </div>
      </div>
    </div>
  );
}
