"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Target,
  DollarSign,
  Layers,
  LineChart,
  Minus,
  ChevronDown,
  Award,
  Percent,
  AlertTriangle,
  RefreshCw,
  Wifi,
  WifiOff,
  Loader2,
} from "lucide-react";

import { useWalkForwardResults, useBacktestMethods } from "@/hooks/useApi";
import type { AssetId } from "@/types";
import {
  type WalkForwardMethod,
  type FoldResult,
  type SummaryMetrics,
  type EquityPoint,
  WALK_FORWARD_METHODS,
} from "@/types/backtest";

// ============================================================================
// Constants
// ============================================================================

const AVAILABLE_ASSETS: { id: AssetId; name: string }[] = [
  { id: "crude-oil", name: "Crude Oil" },
  { id: "bitcoin", name: "Bitcoin" },
  { id: "gold", name: "Gold" },
  { id: "natural-gas", name: "Natural Gas" },
  { id: "silver", name: "Silver" },
  { id: "copper", name: "Copper" },
];

const DEFAULT_METHODS: WalkForwardMethod[] = [
  "tier1_combined",
  "tier2_combined",
  "tier3_combined",
];

const TIER_COLORS = {
  tier1: { bg: "bg-blue-500/10", text: "text-blue-400", border: "border-blue-500/30" },
  tier2: { bg: "bg-purple-500/10", text: "text-purple-400", border: "border-purple-500/30" },
  tier3: { bg: "bg-amber-500/10", text: "text-amber-400", border: "border-amber-500/30" },
};

const METHOD_COLORS: Record<string, string> = {
  tier1_accuracy: "#3b82f6",
  tier1_magnitude: "#60a5fa",
  tier1_correlation: "#93c5fd",
  tier1_combined: "#2563eb",
  tier2_bma: "#a855f7",
  tier2_regime: "#c084fc",
  tier2_conformal: "#d8b4fe",
  tier2_combined: "#9333ea",
  tier3_thompson: "#f59e0b",
  tier3_attention: "#fbbf24",
  tier3_quantile: "#fcd34d",
  tier3_combined: "#d97706",
};

// ============================================================================
// Components
// ============================================================================

interface AssetSelectorProps {
  selected: AssetId;
  onSelect: (asset: AssetId) => void;
}

function AssetSelector({ selected, onSelect }: AssetSelectorProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-4 py-2 bg-neutral-900/50 border border-neutral-800 rounded-lg hover:bg-neutral-800/50 transition-colors"
      >
        <span className="text-neutral-100 font-medium">
          {AVAILABLE_ASSETS.find((a) => a.id === selected)?.name || selected}
        </span>
        <ChevronDown className={cn("w-4 h-4 text-neutral-400 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-1 w-48 bg-neutral-900 border border-neutral-800 rounded-lg shadow-xl z-50">
          {AVAILABLE_ASSETS.map((asset) => (
            <button
              key={asset.id}
              onClick={() => {
                onSelect(asset.id);
                setOpen(false);
              }}
              className={cn(
                "w-full px-4 py-2 text-left hover:bg-neutral-800 transition-colors first:rounded-t-lg last:rounded-b-lg",
                selected === asset.id ? "text-green-400 bg-neutral-800/50" : "text-neutral-300"
              )}
            >
              {asset.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

interface MethodSelectorProps {
  selected: WalkForwardMethod[];
  onToggle: (method: WalkForwardMethod) => void;
}

function MethodSelector({ selected, onToggle }: MethodSelectorProps) {
  const methodsByTier = useMemo(() => {
    const tiers: Record<string, WalkForwardMethod[]> = { tier1: [], tier2: [], tier3: [] };
    Object.keys(WALK_FORWARD_METHODS).forEach((m) => {
      const method = m as WalkForwardMethod;
      const tier = method.split("_")[0] as "tier1" | "tier2" | "tier3";
      tiers[tier].push(method);
    });
    return tiers;
  }, []);

  return (
    <div className="space-y-3">
      {(["tier1", "tier2", "tier3"] as const).map((tier) => (
        <div key={tier}>
          <div className="text-xs uppercase tracking-wider text-neutral-500 mb-2">
            {tier === "tier1" ? "Tier 1 - Basic" : tier === "tier2" ? "Tier 2 - Advanced" : "Tier 3 - Experimental"}
          </div>
          <div className="flex flex-wrap gap-2">
            {methodsByTier[tier].map((method) => {
              const info = WALK_FORWARD_METHODS[method];
              const isSelected = selected.includes(method);
              const colors = TIER_COLORS[tier];
              return (
                <button
                  key={method}
                  onClick={() => onToggle(method)}
                  className={cn(
                    "px-3 py-1.5 text-sm rounded-lg border transition-all",
                    isSelected
                      ? `${colors.bg} ${colors.text} ${colors.border}`
                      : "bg-neutral-900/30 text-neutral-500 border-neutral-800 hover:border-neutral-700"
                  )}
                  title={info.description}
                >
                  {info.name}
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  icon: React.ReactNode;
  trend?: "up" | "down" | "neutral";
  highlight?: boolean;
}

function MetricCard({ label, value, subValue, icon, trend, highlight }: MetricCardProps) {
  return (
    <div
      className={cn(
        "p-4 rounded-lg border",
        highlight
          ? "bg-green-500/5 border-green-500/20"
          : "bg-neutral-900/30 border-neutral-800"
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs uppercase tracking-wider text-neutral-500">{label}</span>
        <span className="text-neutral-500">{icon}</span>
      </div>
      <div className="flex items-baseline gap-2">
        <span
          className={cn(
            "text-2xl font-bold font-mono",
            trend === "up"
              ? "text-green-400"
              : trend === "down"
                ? "text-red-400"
                : "text-neutral-100"
          )}
        >
          {value}
        </span>
        {subValue && <span className="text-sm text-neutral-500">{subValue}</span>}
      </div>
    </div>
  );
}

interface FoldTableProps {
  folds: FoldResult[];
}

function FoldTable({ folds }: FoldTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-neutral-800">
            <th className="px-3 py-2 text-left text-neutral-500 font-medium">Fold</th>
            <th className="px-3 py-2 text-left text-neutral-500 font-medium">Period</th>
            <th className="px-3 py-2 text-right text-neutral-500 font-medium">Accuracy</th>
            <th className="px-3 py-2 text-right text-neutral-500 font-medium">Sharpe</th>
            <th className="px-3 py-2 text-right text-neutral-500 font-medium">Return</th>
            <th className="px-3 py-2 text-right text-neutral-500 font-medium">Max DD</th>
            <th className="px-3 py-2 text-right text-neutral-500 font-medium">Win Rate</th>
            <th className="px-3 py-2 text-right text-neutral-500 font-medium">Cost Drag</th>
          </tr>
        </thead>
        <tbody>
          {folds.map((fold) => (
            <tr key={fold.fold_id} className="border-b border-neutral-800/50 hover:bg-neutral-800/20">
              <td className="px-3 py-2 text-neutral-300 font-mono">#{fold.fold_id}</td>
              <td className="px-3 py-2 text-neutral-400 text-xs">
                {fold.test_start} - {fold.test_end}
              </td>
              <td className="px-3 py-2 text-right font-mono">
                <span className={fold.accuracy >= 55 ? "text-green-400" : fold.accuracy >= 50 ? "text-neutral-300" : "text-red-400"}>
                  {fold.accuracy.toFixed(1)}%
                </span>
              </td>
              <td className="px-3 py-2 text-right font-mono">
                <span className={fold.sharpe_ratio >= 1.0 ? "text-green-400" : fold.sharpe_ratio >= 0.5 ? "text-neutral-300" : "text-red-400"}>
                  {fold.sharpe_ratio.toFixed(2)}
                </span>
              </td>
              <td className="px-3 py-2 text-right font-mono">
                <span className={fold.total_return >= 0 ? "text-green-400" : "text-red-400"}>
                  {fold.total_return >= 0 ? "+" : ""}{fold.total_return.toFixed(1)}%
                </span>
              </td>
              <td className="px-3 py-2 text-right font-mono text-red-400">
                {fold.max_drawdown.toFixed(1)}%
              </td>
              <td className="px-3 py-2 text-right font-mono">
                <span className={fold.win_rate >= 50 ? "text-green-400" : "text-red-400"}>
                  {fold.win_rate.toFixed(1)}%
                </span>
              </td>
              <td className="px-3 py-2 text-right font-mono text-amber-400">
                -{fold.cost_drag_pct.toFixed(2)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface EquityCurveChartProps {
  data: Record<WalkForwardMethod, EquityPoint[]>;
  methods: WalkForwardMethod[];
}

function EquityCurveChart({ data, methods }: EquityCurveChartProps) {
  const option: EChartsOption = useMemo(() => {
    const firstMethod = methods[0];
    const dates = data[firstMethod]?.map((d) => d.date) || [];

    const series = methods.map((method) => ({
      name: WALK_FORWARD_METHODS[method]?.name || method,
      type: "line" as const,
      data: data[method]?.map((d) => d.equity) || [],
      smooth: true,
      symbol: "none",
      lineStyle: {
        color: METHOD_COLORS[method] || "#6b7280",
        width: 2,
      },
    }));

    // Add benchmark
    series.push({
      name: "Buy & Hold",
      type: "line" as const,
      data: data[firstMethod]?.map((d) => d.benchmark ?? 100000) || [],
      smooth: true,
      symbol: "none",
      lineStyle: {
        color: "#6b7280",
        width: 1.5,
        type: "dashed" as unknown as undefined,
      } as { color: string; width: number; type?: undefined },
    });

    return {
      backgroundColor: "transparent",
      animation: true,
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(23, 23, 23, 0.95)",
        borderColor: "#404040",
        textStyle: { color: "#f5f5f5", fontSize: 11 },
      },
      legend: {
        data: [...methods.map((m) => WALK_FORWARD_METHODS[m]?.name || m), "Buy & Hold"],
        top: 0,
        textStyle: { color: "#a3a3a3", fontSize: 10 },
        itemGap: 15,
      },
      grid: { left: "6%", right: "4%", top: "15%", bottom: "12%" },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: { lineStyle: { color: "#404040" } },
        axisLabel: { color: "#a3a3a3", fontSize: 10, interval: Math.floor(dates.length / 8) },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        axisLine: { lineStyle: { color: "#404040" } },
        axisLabel: {
          color: "#a3a3a3",
          fontSize: 10,
          formatter: (v: number) => `$${(v / 1000).toFixed(0)}K`,
        },
        splitLine: { lineStyle: { color: "#262626", type: "dashed" } },
      },
      series,
      dataZoom: [{ type: "inside", start: 0, end: 100 }],
    };
  }, [data, methods]);

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} opts={{ renderer: "canvas" }} />;
}

interface RegimePerformanceProps {
  methodResults: Record<WalkForwardMethod, FoldResult[]>;
  methods: WalkForwardMethod[];
}

function RegimePerformancePanel({ methodResults, methods }: RegimePerformanceProps) {
  const regimeData = useMemo(() => {
    const regimes = ["bull", "bear", "sideways"];
    const data: Record<string, Record<string, { accuracy: number; sharpe: number; return: number; winRate: number }>> = {};

    regimes.forEach((regime) => {
      data[regime] = {};
      methods.forEach((method) => {
        const folds = methodResults[method] || [];
        const regimePerfs = folds
          .map((f) => f.regime_performance[regime])
          .filter(Boolean);

        if (regimePerfs.length > 0) {
          data[regime][method] = {
            accuracy: regimePerfs.reduce((s, r) => s + r.accuracy, 0) / regimePerfs.length,
            sharpe: regimePerfs.reduce((s, r) => s + r.sharpe_ratio, 0) / regimePerfs.length,
            return: regimePerfs.reduce((s, r) => s + r.total_return, 0) / regimePerfs.length,
            winRate: regimePerfs.reduce((s, r) => s + r.win_rate, 0) / regimePerfs.length,
          };
        }
      });
    });

    return data;
  }, [methodResults, methods]);

  const regimeLabels = {
    bull: { name: "Bull Market", icon: TrendingUp, color: "text-green-400" },
    bear: { name: "Bear Market", icon: TrendingDown, color: "text-red-400" },
    sideways: { name: "Sideways", icon: Minus, color: "text-neutral-400" },
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {(["bull", "bear", "sideways"] as const).map((regime) => {
        const label = regimeLabels[regime];
        const Icon = label.icon;
        return (
          <Card key={regime} className="bg-neutral-900/30 border-neutral-800">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Icon className={cn("w-4 h-4", label.color)} />
                <span className="text-neutral-200">{label.name}</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-2">
                {methods.map((method) => {
                  const perf = regimeData[regime]?.[method];
                  if (!perf) return null;
                  const tier = method.split("_")[0] as "tier1" | "tier2" | "tier3";
                  const colors = TIER_COLORS[tier];
                  return (
                    <div key={method} className="flex items-center justify-between py-1 border-b border-neutral-800/50 last:border-0">
                      <span className={cn("text-xs", colors.text)}>
                        {WALK_FORWARD_METHODS[method]?.name}
                      </span>
                      <div className="flex items-center gap-3 text-xs font-mono">
                        <span className={perf.accuracy >= 55 ? "text-green-400" : "text-neutral-400"}>
                          {perf.accuracy.toFixed(1)}%
                        </span>
                        <span className={perf.sharpe >= 1 ? "text-green-400" : "text-neutral-400"}>
                          {perf.sharpe.toFixed(2)}
                        </span>
                        <span className={perf.return >= 0 ? "text-green-400" : "text-red-400"}>
                          {perf.return >= 0 ? "+" : ""}{perf.return.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

interface CostComparisonChartProps {
  summaryMetrics: Record<WalkForwardMethod, SummaryMetrics>;
  methods: WalkForwardMethod[];
}

function CostComparisonChart({ summaryMetrics, methods }: CostComparisonChartProps) {
  const option: EChartsOption = useMemo(() => {
    const methodNames = methods.map((m) => WALK_FORWARD_METHODS[m]?.name || m);
    const rawReturns = methods.map((m) => summaryMetrics[m]?.raw_total_return || 0);
    const costAdjusted = methods.map((m) => summaryMetrics[m]?.cost_adjusted_return || 0);
    const costDrag = methods.map((m) => summaryMetrics[m]?.mean_cost_drag_pct || 0);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(23, 23, 23, 0.95)",
        borderColor: "#404040",
        textStyle: { color: "#f5f5f5", fontSize: 11 },
        axisPointer: { type: "shadow" },
      },
      legend: {
        data: ["Raw Return", "Cost-Adjusted Return", "Cost Drag"],
        top: 0,
        textStyle: { color: "#a3a3a3", fontSize: 10 },
      },
      grid: { left: "3%", right: "4%", top: "15%", bottom: "3%", containLabel: true },
      xAxis: {
        type: "category",
        data: methodNames,
        axisLine: { lineStyle: { color: "#404040" } },
        axisLabel: { color: "#a3a3a3", fontSize: 9, rotate: 30 },
      },
      yAxis: {
        type: "value",
        axisLine: { lineStyle: { color: "#404040" } },
        axisLabel: { color: "#a3a3a3", fontSize: 10, formatter: "{value}%" },
        splitLine: { lineStyle: { color: "#262626", type: "dashed" } },
      },
      series: [
        {
          name: "Raw Return",
          type: "bar",
          data: rawReturns,
          itemStyle: { color: "#22c55e" },
          barGap: "10%",
        },
        {
          name: "Cost-Adjusted Return",
          type: "bar",
          data: costAdjusted,
          itemStyle: { color: "#3b82f6" },
        },
        {
          name: "Cost Drag",
          type: "bar",
          data: costDrag,
          itemStyle: { color: "#ef4444" },
        },
      ],
    };
  }, [summaryMetrics, methods]);

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} opts={{ renderer: "canvas" }} />;
}

interface RankingsTableProps {
  summaryMetrics: Record<WalkForwardMethod, SummaryMetrics>;
  rankings: Record<WalkForwardMethod, number>;
  methods: WalkForwardMethod[];
}

function RankingsTable({ summaryMetrics, rankings, methods }: RankingsTableProps) {
  const sortedMethods = useMemo(
    () => [...methods].sort((a, b) => (rankings[a] || 99) - (rankings[b] || 99)),
    [methods, rankings]
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-neutral-800">
            <th className="px-3 py-2 text-left text-neutral-500 font-medium">Rank</th>
            <th className="px-3 py-2 text-left text-neutral-500 font-medium">Method</th>
            <th className="px-3 py-2 text-right text-neutral-500 font-medium">Sharpe</th>
            <th className="px-3 py-2 text-right text-neutral-500 font-medium">Accuracy</th>
            <th className="px-3 py-2 text-right text-neutral-500 font-medium">Return</th>
            <th className="px-3 py-2 text-right text-neutral-500 font-medium">Max DD</th>
            <th className="px-3 py-2 text-right text-neutral-500 font-medium">Cost Impact</th>
          </tr>
        </thead>
        <tbody>
          {sortedMethods.map((method, idx) => {
            const metrics = summaryMetrics[method];
            const tier = method.split("_")[0] as "tier1" | "tier2" | "tier3";
            const colors = TIER_COLORS[tier];
            const isTop = idx === 0;
            return (
              <tr
                key={method}
                className={cn(
                  "border-b border-neutral-800/50",
                  isTop && "bg-green-500/5"
                )}
              >
                <td className="px-3 py-2">
                  {isTop ? (
                    <Award className="w-5 h-5 text-amber-400" />
                  ) : (
                    <span className="text-neutral-500 font-mono">#{idx + 1}</span>
                  )}
                </td>
                <td className="px-3 py-2">
                  <Badge className={cn("font-normal", colors.bg, colors.text, colors.border)}>
                    {WALK_FORWARD_METHODS[method]?.name}
                  </Badge>
                </td>
                <td className="px-3 py-2 text-right font-mono text-green-400">
                  {metrics?.mean_sharpe.toFixed(2)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-neutral-300">
                  {metrics?.mean_accuracy.toFixed(1)}%
                </td>
                <td className="px-3 py-2 text-right font-mono">
                  <span className={(metrics?.mean_total_return || 0) >= 0 ? "text-green-400" : "text-red-400"}>
                    {(metrics?.mean_total_return || 0) >= 0 ? "+" : ""}
                    {metrics?.mean_total_return.toFixed(1)}%
                  </span>
                </td>
                <td className="px-3 py-2 text-right font-mono text-red-400">
                  {metrics?.mean_max_drawdown.toFixed(1)}%
                </td>
                <td className="px-3 py-2 text-right font-mono text-amber-400">
                  -{metrics?.mean_cost_drag_pct.toFixed(2)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ============================================================================
// Main Dashboard
// ============================================================================

export function BacktestDashboard() {
  const [selectedAsset, setSelectedAsset] = useState<AssetId>("crude-oil");
  const [selectedMethods, setSelectedMethods] = useState<WalkForwardMethod[]>(DEFAULT_METHODS);
  const [nFolds, setNFolds] = useState(5);
  const [activeTab, setActiveTab] = useState("overview");

  // Fetch backtest methods (for validation)
  const { data: _methodsData, isLoading: _methodsLoading } = useBacktestMethods();

  // Fetch walk-forward results
  const {
    data: walkForwardData,
    isLoading,
    isFetching,
    error,
    refetch,
    isError,
  } = useWalkForwardResults(selectedAsset, selectedMethods, nFolds);

  const handleMethodToggle = (method: WalkForwardMethod) => {
    setSelectedMethods((prev) =>
      prev.includes(method)
        ? prev.filter((m) => m !== method)
        : [...prev, method]
    );
  };

  const handleRefresh = () => {
    refetch();
  };

  const bestMethod = useMemo(() => {
    if (!walkForwardData?.data?.rankings) return null;
    const sorted = Object.entries(walkForwardData.data.rankings).sort((a, b) => a[1] - b[1]);
    return sorted[0]?.[0] as WalkForwardMethod | undefined;
  }, [walkForwardData]);

  const bestMetrics = bestMethod ? walkForwardData?.data?.summary_metrics[bestMethod] : null;

  // Determine API connection status
  const isConnected = !isError || walkForwardData !== undefined;

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      {/* Header */}
      <div className="border-b border-neutral-800 bg-neutral-950/80 backdrop-blur-sm sticky top-0 z-40">
        <div className="max-w-[1800px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-6 h-6 text-green-400" />
                <h1 className="text-xl font-bold text-neutral-100">Walk-Forward Backtest</h1>
              </div>
              <Badge className="bg-neutral-800 text-neutral-400 border-neutral-700">
                {nFolds} Folds
              </Badge>
            </div>
            <div className="flex items-center gap-4">
              {/* Connection status indicator */}
              <div className="flex items-center gap-2">
                {isConnected ? (
                  <div className="flex items-center gap-1.5 text-green-400">
                    <Wifi className="w-4 h-4" />
                    <span className="text-xs">API</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-1.5 text-amber-400">
                    <WifiOff className="w-4 h-4" />
                    <span className="text-xs">Mock</span>
                  </div>
                )}
              </div>

              <AssetSelector selected={selectedAsset} onSelect={setSelectedAsset} />

              <select
                value={nFolds}
                onChange={(e) => setNFolds(Number(e.target.value))}
                className="px-3 py-2 bg-neutral-900/50 border border-neutral-800 rounded-lg text-neutral-300 focus:outline-none focus:border-neutral-700"
              >
                <option value={3}>3 Folds</option>
                <option value={5}>5 Folds</option>
                <option value={7}>7 Folds</option>
                <option value={10}>10 Folds</option>
              </select>

              {/* Refresh button */}
              <button
                onClick={handleRefresh}
                disabled={isLoading || isFetching}
                className={cn(
                  "p-2 rounded-lg border border-neutral-800 transition-colors",
                  isLoading || isFetching
                    ? "bg-neutral-800/50 text-neutral-500 cursor-not-allowed"
                    : "bg-neutral-900/50 text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200"
                )}
                title="Refresh results"
              >
                <RefreshCw className={cn("w-4 h-4", (isLoading || isFetching) && "animate-spin")} />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-[1800px] mx-auto px-6 py-6 space-y-6">
        {/* Method Selector */}
        <Card className="bg-neutral-900/30 border-neutral-800">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Layers className="w-4 h-4 text-neutral-400" />
              Select Ensemble Methods to Compare
            </CardTitle>
          </CardHeader>
          <CardContent>
            <MethodSelector selected={selectedMethods} onToggle={handleMethodToggle} />
          </CardContent>
        </Card>

        {/* Loading State */}
        {isLoading && (
          <Card className="bg-neutral-900/30 border-neutral-800">
            <CardContent className="py-16">
              <div className="flex flex-col items-center gap-6">
                <div className="relative">
                  <div className="w-16 h-16 border-4 border-neutral-800 rounded-full" />
                  <div className="absolute inset-0 w-16 h-16 border-4 border-green-400 border-t-transparent rounded-full animate-spin" />
                </div>
                <div className="text-center space-y-2">
                  <h3 className="text-lg font-semibold text-neutral-200">Running Walk-Forward Validation</h3>
                  <p className="text-sm text-neutral-500">
                    Evaluating {selectedMethods.length} method{selectedMethods.length !== 1 ? "s" : ""} across {nFolds} folds...
                  </p>
                </div>
                <div className="flex items-center gap-2 text-xs text-neutral-600">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Fetching from API
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Error State */}
        {error && !isLoading && (
          <Card className="bg-red-500/5 border-red-500/20">
            <CardContent className="py-12">
              <div className="flex flex-col items-center gap-4">
                <div className="p-4 bg-red-500/10 rounded-full">
                  <AlertTriangle className="w-8 h-8 text-red-400" />
                </div>
                <div className="text-center space-y-2">
                  <h3 className="text-lg font-semibold text-red-400">Failed to Load Backtest Results</h3>
                  <p className="text-sm text-neutral-400 max-w-md">
                    {error instanceof Error ? error.message : "Unable to connect to the backtest API. Using cached or mock data."}
                  </p>
                </div>
                <button
                  onClick={handleRefresh}
                  className="flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  Try Again
                </button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Fetching indicator (when refetching in background) */}
        {isFetching && !isLoading && (
          <div className="flex items-center justify-center gap-2 py-2 text-xs text-neutral-500">
            <Loader2 className="w-3 h-3 animate-spin" />
            Updating results...
          </div>
        )}

        {/* Results */}
        {walkForwardData && !isLoading && (
          <>
            {/* Summary Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <MetricCard
                label="Best Method"
                value={bestMethod ? WALK_FORWARD_METHODS[bestMethod]?.name : "-"}
                icon={<Award className="w-4 h-4" />}
                highlight
              />
              <MetricCard
                label="Best Sharpe"
                value={bestMetrics?.mean_sharpe.toFixed(2) || "-"}
                icon={<Target className="w-4 h-4" />}
                trend={bestMetrics && bestMetrics.mean_sharpe >= 1 ? "up" : "neutral"}
              />
              <MetricCard
                label="Best Accuracy"
                value={`${bestMetrics?.mean_accuracy.toFixed(1) || "-"}%`}
                icon={<Percent className="w-4 h-4" />}
                trend={bestMetrics && bestMetrics.mean_accuracy >= 55 ? "up" : "neutral"}
              />
              <MetricCard
                label="Best Return"
                value={`${bestMetrics && bestMetrics.mean_total_return >= 0 ? "+" : ""}${bestMetrics?.mean_total_return.toFixed(1) || "-"}%`}
                icon={<TrendingUp className="w-4 h-4" />}
                trend={bestMetrics && bestMetrics.mean_total_return >= 0 ? "up" : "down"}
              />
              <MetricCard
                label="Max Drawdown"
                value={`${bestMetrics?.mean_max_drawdown.toFixed(1) || "-"}%`}
                icon={<TrendingDown className="w-4 h-4" />}
                trend="down"
              />
              <MetricCard
                label="Avg Cost Drag"
                value={`-${bestMetrics?.mean_cost_drag_pct.toFixed(2) || "-"}%`}
                icon={<DollarSign className="w-4 h-4" />}
              />
            </div>

            {/* Tabs */}
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="bg-neutral-900/50 border border-neutral-800">
                <TabsTrigger value="overview" className="data-[state=active]:bg-neutral-800">
                  Overview
                </TabsTrigger>
                <TabsTrigger value="folds" className="data-[state=active]:bg-neutral-800">
                  Fold Details
                </TabsTrigger>
                <TabsTrigger value="equity" className="data-[state=active]:bg-neutral-800">
                  Equity Curves
                </TabsTrigger>
                <TabsTrigger value="regimes" className="data-[state=active]:bg-neutral-800">
                  Regime Analysis
                </TabsTrigger>
                <TabsTrigger value="costs" className="data-[state=active]:bg-neutral-800">
                  Cost Impact
                </TabsTrigger>
              </TabsList>

              {/* Overview Tab */}
              <TabsContent value="overview" className="mt-4 space-y-4">
                <Card className="bg-neutral-900/30 border-neutral-800">
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Award className="w-4 h-4 text-amber-400" />
                      Method Rankings
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <RankingsTable
                      summaryMetrics={walkForwardData.data.summary_metrics}
                      rankings={walkForwardData.data.rankings}
                      methods={selectedMethods}
                    />
                  </CardContent>
                </Card>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card className="bg-neutral-900/30 border-neutral-800">
                    <CardHeader>
                      <CardTitle className="text-sm">Equity Comparison</CardTitle>
                    </CardHeader>
                    <CardContent className="h-[300px]">
                      <EquityCurveChart data={walkForwardData.equity_curves} methods={selectedMethods} />
                    </CardContent>
                  </Card>

                  <Card className="bg-neutral-900/30 border-neutral-800">
                    <CardHeader>
                      <CardTitle className="text-sm">Cost Impact by Method</CardTitle>
                    </CardHeader>
                    <CardContent className="h-[300px]">
                      <CostComparisonChart
                        summaryMetrics={walkForwardData.data.summary_metrics}
                        methods={selectedMethods}
                      />
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* Fold Details Tab */}
              <TabsContent value="folds" className="mt-4 space-y-4">
                {selectedMethods.map((method) => (
                  <Card key={method} className="bg-neutral-900/30 border-neutral-800">
                    <CardHeader>
                      <CardTitle className="text-sm flex items-center gap-2">
                        <Badge
                          className={cn(
                            "font-normal",
                            TIER_COLORS[method.split("_")[0] as "tier1" | "tier2" | "tier3"].bg,
                            TIER_COLORS[method.split("_")[0] as "tier1" | "tier2" | "tier3"].text,
                            TIER_COLORS[method.split("_")[0] as "tier1" | "tier2" | "tier3"].border
                          )}
                        >
                          {WALK_FORWARD_METHODS[method]?.name}
                        </Badge>
                        <span className="text-neutral-500 font-normal text-xs">
                          {WALK_FORWARD_METHODS[method]?.description}
                        </span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <FoldTable
                        folds={walkForwardData.data.method_results[method] || []}
                      />
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>

              {/* Equity Tab */}
              <TabsContent value="equity" className="mt-4">
                <Card className="bg-neutral-900/30 border-neutral-800">
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <LineChart className="w-4 h-4 text-green-400" />
                      Equity Curve Comparison
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="h-[500px]">
                    <EquityCurveChart data={walkForwardData.equity_curves} methods={selectedMethods} />
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Regimes Tab */}
              <TabsContent value="regimes" className="mt-4">
                <RegimePerformancePanel
                  methodResults={walkForwardData.data.method_results}
                  methods={selectedMethods}
                />
              </TabsContent>

              {/* Costs Tab */}
              <TabsContent value="costs" className="mt-4 space-y-4">
                <Card className="bg-neutral-900/30 border-neutral-800">
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <DollarSign className="w-4 h-4 text-amber-400" />
                      Raw vs Cost-Adjusted Performance
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="h-[400px]">
                    <CostComparisonChart
                      summaryMetrics={walkForwardData.data.summary_metrics}
                      methods={selectedMethods}
                    />
                  </CardContent>
                </Card>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {selectedMethods.map((method) => {
                    const metrics = walkForwardData.data.summary_metrics[method];
                    const tier = method.split("_")[0] as "tier1" | "tier2" | "tier3";
                    const colors = TIER_COLORS[tier];
                    return (
                      <Card key={method} className="bg-neutral-900/30 border-neutral-800">
                        <CardHeader className="pb-2">
                          <Badge className={cn("w-fit font-normal", colors.bg, colors.text, colors.border)}>
                            {WALK_FORWARD_METHODS[method]?.name}
                          </Badge>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          <div className="flex justify-between">
                            <span className="text-neutral-500 text-sm">Raw Return</span>
                            <span className="font-mono text-green-400">
                              +{metrics?.raw_total_return.toFixed(2)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-neutral-500 text-sm">Cost-Adjusted</span>
                            <span className="font-mono text-blue-400">
                              +{metrics?.cost_adjusted_return.toFixed(2)}%
                            </span>
                          </div>
                          <div className="flex justify-between border-t border-neutral-800 pt-2">
                            <span className="text-neutral-500 text-sm">Cost Impact</span>
                            <span className="font-mono text-red-400">
                              -{metrics?.mean_cost_drag_pct.toFixed(2)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-neutral-500 text-sm">Total Costs</span>
                            <span className="font-mono text-amber-400">
                              ${metrics?.total_costs.toFixed(0)}
                            </span>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </TabsContent>
            </Tabs>
          </>
        )}
      </div>
    </div>
  );
}

export default BacktestDashboard;
