"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { useAllRegimes } from "@/components/ensemble";
import type { MarketRegime } from "@/components/ensemble";
import {
  Activity,
  BarChart3,
  Calendar,
  Clock,
  GitBranch,
  Grid3X3,
  LineChart,
  TrendingDown,
  TrendingUp,
  Minus,
  Zap,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface RegimeTransition {
  date: string;
  assetId: string;
  assetName: string;
  fromRegime: MarketRegime;
  toRegime: MarketRegime;
}

interface AssetRegimeHistory {
  assetId: string;
  assetName: string;
  history: Array<{
    startDate: string;
    endDate: string;
    regime: MarketRegime;
    duration: number;
  }>;
}

interface RegimePerformance {
  assetId: string;
  assetName: string;
  regime: MarketRegime;
  avgReturn: number;
  avgVolatility: number;
  avgDuration: number;
  occurrences: number;
}

// ============================================================================
// Constants
// ============================================================================

const ALL_ASSETS = [
  { id: "crude-oil", name: "Crude Oil", symbol: "CL" },
  { id: "gold", name: "Gold", symbol: "GC" },
  { id: "silver", name: "Silver", symbol: "SI" },
  { id: "copper", name: "Copper", symbol: "HG" },
  { id: "natural-gas", name: "Natural Gas", symbol: "NG" },
  { id: "wheat", name: "Wheat", symbol: "ZW" },
  { id: "corn", name: "Corn", symbol: "ZC" },
  { id: "soybean", name: "Soybean", symbol: "ZS" },
  { id: "platinum", name: "Platinum", symbol: "PL" },
  { id: "bitcoin", name: "Bitcoin", symbol: "BTC" },
  { id: "ethereum", name: "Ethereum", symbol: "ETH" },
  { id: "sp500", name: "S&P 500", symbol: "SPX" },
  { id: "nasdaq", name: "NASDAQ", symbol: "NDX" },
];

const REGIME_COLORS: Record<MarketRegime, { bg: string; text: string; border: string }> = {
  bull: { bg: "bg-green-500", text: "text-green-400", border: "border-green-500/30" },
  bear: { bg: "bg-red-500", text: "text-red-400", border: "border-red-500/30" },
  sideways: { bg: "bg-amber-500", text: "text-amber-400", border: "border-amber-500/30" },
  "high-volatility": { bg: "bg-orange-500", text: "text-orange-400", border: "border-orange-500/30" },
  "low-volatility": { bg: "bg-blue-500", text: "text-blue-400", border: "border-blue-500/30" },
};

// ============================================================================
// Mock Historical Data Generator
// ============================================================================

function generateMockRegimeHistory(): AssetRegimeHistory[] {
  const regimes: MarketRegime[] = ["bull", "bear", "sideways"];
  const now = new Date();

  return ALL_ASSETS.map((asset) => {
    const history: AssetRegimeHistory["history"] = [];
    const currentDate = new Date(now);
    currentDate.setDate(currentDate.getDate() - 180); // 6 months of history

    // Seed based on asset name for consistent mock data
    let seed = asset.id.charCodeAt(0) + asset.id.charCodeAt(asset.id.length - 1);

    while (currentDate < now) {
      const regime = regimes[seed % 3];
      const duration = 5 + (seed % 25); // 5-30 days
      seed = (seed * 17 + 13) % 100;

      const startDate = new Date(currentDate);
      currentDate.setDate(currentDate.getDate() + duration);
      const endDate = new Date(Math.min(currentDate.getTime(), now.getTime()));

      history.push({
        startDate: startDate.toISOString().split("T")[0],
        endDate: endDate.toISOString().split("T")[0],
        regime,
        duration: Math.round((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)),
      });
    }

    return {
      assetId: asset.id,
      assetName: asset.name,
      history,
    };
  });
}

function generateMockPerformanceData(): RegimePerformance[] {
  const performances: RegimePerformance[] = [];
  const regimes: MarketRegime[] = ["bull", "bear", "sideways"];

  ALL_ASSETS.forEach((asset) => {
    let seed = asset.id.charCodeAt(0);

    regimes.forEach((regime) => {
      seed = (seed * 23 + 7) % 100;
      const baseReturn =
        regime === "bull" ? 15 : regime === "bear" ? -12 : 2;
      const baseVol =
        regime === "bull" ? 18 : regime === "bear" ? 28 : 15;

      performances.push({
        assetId: asset.id,
        assetName: asset.name,
        regime,
        avgReturn: baseReturn + (seed % 10) - 5,
        avgVolatility: baseVol + (seed % 8),
        avgDuration: 8 + (seed % 15),
        occurrences: 3 + (seed % 5),
      });
    });
  });

  return performances;
}

function calculateRegimeCorrelation(
  histories: AssetRegimeHistory[]
): { asset1: string; asset2: string; correlation: number }[] {
  const correlations: { asset1: string; asset2: string; correlation: number }[] = [];

  // Generate all pairs
  for (let i = 0; i < histories.length; i++) {
    for (let j = i + 1; j < histories.length; j++) {
      const h1 = histories[i];
      const h2 = histories[j];

      // Simple correlation: percentage of time in same regime
      let sameRegimeDays = 0;
      let totalDays = 0;

      // Compare regime overlap for each date
      h1.history.forEach((period1) => {
        h2.history.forEach((period2) => {
          // Check overlap
          const start = Math.max(
            new Date(period1.startDate).getTime(),
            new Date(period2.startDate).getTime()
          );
          const end = Math.min(
            new Date(period1.endDate).getTime(),
            new Date(period2.endDate).getTime()
          );

          if (end > start) {
            const overlapDays = (end - start) / (1000 * 60 * 60 * 24);
            totalDays += overlapDays;
            if (period1.regime === period2.regime) {
              sameRegimeDays += overlapDays;
            }
          }
        });
      });

      const correlation = totalDays > 0 ? sameRegimeDays / totalDays : 0;
      correlations.push({
        asset1: h1.assetName,
        asset2: h2.assetName,
        correlation,
      });
    }
  }

  return correlations;
}

// ============================================================================
// Components
// ============================================================================

function RegimeIcon({ regime, size = "sm" }: { regime: MarketRegime; size?: "sm" | "md" }) {
  const iconClass = size === "sm" ? "w-3.5 h-3.5" : "w-4 h-4";
  const colors = REGIME_COLORS[regime];

  switch (regime) {
    case "bull":
      return <TrendingUp className={cn(iconClass, colors.text)} />;
    case "bear":
      return <TrendingDown className={cn(iconClass, colors.text)} />;
    case "sideways":
      return <Minus className={cn(iconClass, colors.text)} />;
    case "high-volatility":
      return <Zap className={cn(iconClass, colors.text)} />;
    default:
      return <Activity className={cn(iconClass, colors.text)} />;
  }
}

function RegimeTimelineChart({ histories }: { histories: AssetRegimeHistory[] }) {
  // Find date range
  const allDates = histories.flatMap((h) =>
    h.history.flatMap((p) => [new Date(p.startDate), new Date(p.endDate)])
  );
  const minDate = new Date(Math.min(...allDates.map((d) => d.getTime())));
  const maxDate = new Date(Math.max(...allDates.map((d) => d.getTime())));
  const totalDays = (maxDate.getTime() - minDate.getTime()) / (1000 * 60 * 60 * 24);

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Calendar className="w-4 h-4 text-cyan-400" />
            Regime Transition Timeline
          </CardTitle>
          <div className="flex items-center gap-4 text-xs">
            {(["bull", "bear", "sideways"] as MarketRegime[]).map((regime) => (
              <div key={regime} className="flex items-center gap-1.5">
                <div className={cn("w-3 h-3 rounded", REGIME_COLORS[regime].bg)} />
                <span className="text-neutral-400 capitalize">{regime}</span>
              </div>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-2">
          {histories.map((asset) => (
            <div key={asset.assetId} className="flex items-center gap-3">
              <div className="w-24 text-xs text-neutral-400 truncate">{asset.assetName}</div>
              <div className="flex-1 h-6 bg-neutral-800/50 rounded-sm relative overflow-hidden">
                {asset.history.map((period, idx) => {
                  const startOffset =
                    ((new Date(period.startDate).getTime() - minDate.getTime()) /
                      (1000 * 60 * 60 * 24) /
                      totalDays) *
                    100;
                  const width = (period.duration / totalDays) * 100;

                  return (
                    <div
                      key={idx}
                      className={cn(
                        "absolute inset-y-0 transition-opacity hover:opacity-80",
                        REGIME_COLORS[period.regime].bg
                      )}
                      style={{
                        left: `${startOffset}%`,
                        width: `${Math.max(width, 0.5)}%`,
                      }}
                      title={`${period.regime}: ${period.startDate} to ${period.endDate} (${period.duration}d)`}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Timeline axis */}
        <div className="flex items-center gap-3 mt-4 pt-2 border-t border-neutral-800">
          <div className="w-24" />
          <div className="flex-1 flex justify-between text-[10px] text-neutral-500">
            <span>{minDate.toLocaleDateString()}</span>
            <span>{maxDate.toLocaleDateString()}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function RegimeCorrelationHeatmap({
  correlations,
  assets,
}: {
  correlations: { asset1: string; asset2: string; correlation: number }[];
  assets: string[];
}) {
  // Build correlation matrix
  const matrix: Record<string, Record<string, number>> = {};
  assets.forEach((a) => {
    matrix[a] = {};
    assets.forEach((b) => {
      matrix[a][b] = a === b ? 1 : 0;
    });
  });

  correlations.forEach(({ asset1, asset2, correlation }) => {
    matrix[asset1][asset2] = correlation;
    matrix[asset2][asset1] = correlation;
  });

  const getColor = (value: number) => {
    if (value >= 0.8) return "bg-green-500";
    if (value >= 0.6) return "bg-green-500/70";
    if (value >= 0.4) return "bg-amber-500/70";
    if (value >= 0.2) return "bg-orange-500/70";
    return "bg-red-500/50";
  };

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Grid3X3 className="w-4 h-4 text-purple-400" />
            Regime Correlation Heatmap
          </CardTitle>
          <div className="flex items-center gap-2 text-[10px]">
            <span className="text-neutral-500">Low</span>
            <div className="flex gap-0.5">
              <div className="w-4 h-3 bg-red-500/50 rounded-sm" />
              <div className="w-4 h-3 bg-orange-500/70 rounded-sm" />
              <div className="w-4 h-3 bg-amber-500/70 rounded-sm" />
              <div className="w-4 h-3 bg-green-500/70 rounded-sm" />
              <div className="w-4 h-3 bg-green-500 rounded-sm" />
            </div>
            <span className="text-neutral-500">High</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0 overflow-x-auto">
        <div className="min-w-max">
          {/* Header row */}
          <div className="flex">
            <div className="w-20" />
            {assets.map((asset) => (
              <div
                key={asset}
                className="w-8 text-[8px] text-neutral-500 text-center truncate transform -rotate-45 origin-left translate-y-3 h-16"
              >
                {asset.slice(0, 8)}
              </div>
            ))}
          </div>

          {/* Matrix rows */}
          {assets.map((row) => (
            <div key={row} className="flex items-center">
              <div className="w-20 text-[10px] text-neutral-400 truncate pr-2">{row}</div>
              {assets.map((col) => {
                const value = matrix[row][col];
                return (
                  <div
                    key={col}
                    className={cn(
                      "w-8 h-8 flex items-center justify-center text-[9px] font-mono",
                      row === col ? "bg-neutral-700" : getColor(value),
                      "border border-neutral-800"
                    )}
                    title={`${row} vs ${col}: ${(value * 100).toFixed(0)}%`}
                  >
                    {row === col ? "-" : (value * 100).toFixed(0)}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function PerformanceByRegimeTable({ performances }: { performances: RegimePerformance[] }) {
  const [selectedRegime, setSelectedRegime] = useState<MarketRegime | "all">("all");

  const filteredPerf = useMemo(() => {
    if (selectedRegime === "all") return performances;
    return performances.filter((p) => p.regime === selectedRegime);
  }, [performances, selectedRegime]);

  // Group by asset for "all" view, or show flat list for specific regime
  const groupedByAsset = useMemo(() => {
    const grouped: Record<string, RegimePerformance[]> = {};
    filteredPerf.forEach((p) => {
      if (!grouped[p.assetName]) grouped[p.assetName] = [];
      grouped[p.assetName].push(p);
    });
    return grouped;
  }, [filteredPerf]);

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-amber-400" />
            Performance by Regime
          </CardTitle>
          <div className="flex items-center gap-1 bg-neutral-800/50 p-1 rounded-lg">
            {(["all", "bull", "bear", "sideways"] as const).map((regime) => (
              <button
                key={regime}
                onClick={() => setSelectedRegime(regime)}
                className={cn(
                  "px-3 py-1 text-xs rounded-md transition-colors capitalize",
                  selectedRegime === regime
                    ? regime === "all"
                      ? "bg-neutral-700 text-white"
                      : cn("bg-opacity-20", REGIME_COLORS[regime as MarketRegime].bg, REGIME_COLORS[regime as MarketRegime].text)
                    : "text-neutral-400 hover:text-neutral-200"
                )}
              >
                {regime}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-neutral-700 text-left">
                <th className="py-2 px-3 text-xs font-medium text-neutral-400">Asset</th>
                <th className="py-2 px-3 text-xs font-medium text-neutral-400">Regime</th>
                <th className="py-2 px-3 text-xs font-medium text-neutral-400 text-right">Avg Return</th>
                <th className="py-2 px-3 text-xs font-medium text-neutral-400 text-right">Volatility</th>
                <th className="py-2 px-3 text-xs font-medium text-neutral-400 text-right">Avg Duration</th>
                <th className="py-2 px-3 text-xs font-medium text-neutral-400 text-right">Occurrences</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(groupedByAsset).map(([assetName, perfs]) =>
                perfs.map((perf, idx) => (
                  <tr
                    key={`${assetName}-${perf.regime}`}
                    className={cn(
                      "border-b border-neutral-800/50",
                      idx === 0 && "bg-neutral-800/20"
                    )}
                  >
                    <td className="py-2 px-3 text-sm text-neutral-200">
                      {idx === 0 ? assetName : ""}
                    </td>
                    <td className="py-2 px-3">
                      <div className="flex items-center gap-2">
                        <RegimeIcon regime={perf.regime} />
                        <span className={cn("text-xs capitalize", REGIME_COLORS[perf.regime].text)}>
                          {perf.regime}
                        </span>
                      </div>
                    </td>
                    <td className="py-2 px-3 text-right">
                      <span
                        className={cn(
                          "text-sm font-mono",
                          perf.avgReturn >= 0 ? "text-green-400" : "text-red-400"
                        )}
                      >
                        {perf.avgReturn >= 0 ? "+" : ""}
                        {perf.avgReturn.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2 px-3 text-right">
                      <span className="text-sm font-mono text-neutral-300">
                        {perf.avgVolatility.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2 px-3 text-right">
                      <span className="text-sm font-mono text-neutral-300">
                        {perf.avgDuration.toFixed(0)}d
                      </span>
                    </td>
                    <td className="py-2 px-3 text-right">
                      <span className="text-sm font-mono text-neutral-400">{perf.occurrences}</span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function CurrentRegimeSummary({ regimeData }: { regimeData?: { regimes?: unknown; regime_distribution?: Record<string, number>; total_assets?: number } }) {
  if (!regimeData?.regimes) return null;

  const distribution = regimeData.regime_distribution || {};
  const total = regimeData.total_assets || 0;

  const summaryStats = [
    {
      label: "Bull Markets",
      count: distribution.bull || 0,
      icon: TrendingUp,
      color: "text-green-400",
      bg: "bg-green-500/10",
    },
    {
      label: "Bear Markets",
      count: distribution.bear || 0,
      icon: TrendingDown,
      color: "text-red-400",
      bg: "bg-red-500/10",
    },
    {
      label: "Sideways",
      count: distribution.sideways || 0,
      icon: Minus,
      color: "text-amber-400",
      bg: "bg-amber-500/10",
    },
  ];

  return (
    <div className="grid grid-cols-3 gap-4">
      {summaryStats.map((stat) => (
        <Card key={stat.label} className={cn("border-neutral-800", stat.bg)}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-3xl font-bold font-mono text-neutral-100">{stat.count}</div>
                <div className="text-xs text-neutral-400">{stat.label}</div>
              </div>
              <stat.icon className={cn("w-8 h-8", stat.color)} />
            </div>
            <div className="mt-2 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
              <div
                className={cn("h-full rounded-full", stat.bg.replace("/10", ""))}
                style={{ width: `${total > 0 ? (stat.count / total) * 100 : 0}%` }}
              />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function RecentTransitions({ histories }: { histories: AssetRegimeHistory[] }) {
  // Extract transitions from histories
  const transitions: RegimeTransition[] = [];

  histories.forEach((asset) => {
    for (let i = 1; i < asset.history.length; i++) {
      if (asset.history[i].regime !== asset.history[i - 1].regime) {
        transitions.push({
          date: asset.history[i].startDate,
          assetId: asset.assetId,
          assetName: asset.assetName,
          fromRegime: asset.history[i - 1].regime,
          toRegime: asset.history[i].regime,
        });
      }
    }
  });

  // Sort by date descending and take latest 10
  const recentTransitions = transitions
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    .slice(0, 10);

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-cyan-400" />
          Recent Regime Transitions
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0 space-y-2">
        {recentTransitions.map((t, idx) => (
          <div
            key={idx}
            className="flex items-center justify-between p-2 bg-neutral-800/30 rounded-lg"
          >
            <div className="flex items-center gap-3">
              <span className="text-xs text-neutral-500 w-20">{t.date}</span>
              <span className="text-sm text-neutral-200">{t.assetName}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1">
                <RegimeIcon regime={t.fromRegime} />
                <span className={cn("text-xs", REGIME_COLORS[t.fromRegime].text)}>
                  {t.fromRegime}
                </span>
              </div>
              <span className="text-neutral-600">→</span>
              <div className="flex items-center gap-1">
                <RegimeIcon regime={t.toRegime} />
                <span className={cn("text-xs", REGIME_COLORS[t.toRegime].text)}>{t.toRegime}</span>
              </div>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Page Component
// ============================================================================

export default function AnalyticsDashboardPage() {
  const { data: regimeData, isLoading } = useAllRegimes();

  // Generate mock historical data
  const histories = useMemo(() => generateMockRegimeHistory(), []);
  const performances = useMemo(() => generateMockPerformanceData(), []);
  const correlations = useMemo(() => calculateRegimeCorrelation(histories), [histories]);
  const assetNames = useMemo(() => ALL_ASSETS.map((a) => a.name), []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-950 via-neutral-900 to-neutral-950 p-6">
      <div className="max-w-[1800px] mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-purple-500/20 to-cyan-500/20 rounded-lg border border-purple-500/30">
                <LineChart className="w-6 h-6 text-purple-400" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Cross-Asset Regime Analytics</h1>
                <p className="text-sm text-neutral-400">
                  Historical transitions, correlations, and performance by market regime
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <Badge className="bg-purple-500/10 text-purple-400 border-purple-500/30 text-xs">
              <Activity className="w-3 h-3 mr-1" />
              13 Assets
            </Badge>
            <Badge className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30 text-xs">
              <Clock className="w-3 h-3 mr-1" />
              6 Month History
            </Badge>
          </div>
        </div>

        {/* Current Regime Summary */}
        {isLoading ? (
          <div className="grid grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-24 bg-neutral-800" />
            ))}
          </div>
        ) : (
          <CurrentRegimeSummary regimeData={regimeData} />
        )}

        {/* Main Content Tabs */}
        <Tabs defaultValue="timeline" className="space-y-6">
          <TabsList className="bg-neutral-800/50 border border-neutral-700/50 p-1">
            <TabsTrigger
              value="timeline"
              className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400"
            >
              <Calendar className="w-4 h-4 mr-2" />
              Timeline
            </TabsTrigger>
            <TabsTrigger
              value="correlation"
              className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400"
            >
              <Grid3X3 className="w-4 h-4 mr-2" />
              Correlation
            </TabsTrigger>
            <TabsTrigger
              value="performance"
              className="data-[state=active]:bg-amber-500/20 data-[state=active]:text-amber-400"
            >
              <BarChart3 className="w-4 h-4 mr-2" />
              Performance
            </TabsTrigger>
          </TabsList>

          {/* Timeline Tab */}
          <TabsContent value="timeline" className="space-y-6">
            <RegimeTimelineChart histories={histories} />
            <RecentTransitions histories={histories} />
          </TabsContent>

          {/* Correlation Tab */}
          <TabsContent value="correlation" className="space-y-6">
            <RegimeCorrelationHeatmap correlations={correlations} assets={assetNames} />

            {/* High correlation pairs */}
            <Card className="bg-neutral-900/50 border-neutral-800">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                  <GitBranch className="w-4 h-4 text-green-400" />
                  Highly Correlated Pairs (Regime Alignment &gt; 60%)
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {correlations
                    .filter((c) => c.correlation >= 0.6)
                    .sort((a, b) => b.correlation - a.correlation)
                    .slice(0, 12)
                    .map((c, idx) => (
                      <div
                        key={idx}
                        className="p-3 bg-green-500/5 border border-green-500/20 rounded-lg"
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-neutral-400">{c.asset1}</span>
                          <span className="text-xs text-green-400 font-mono">
                            {(c.correlation * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="text-xs text-neutral-400">↔ {c.asset2}</div>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-6">
            <PerformanceByRegimeTable performances={performances} />

            {/* Regime performance summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {(["bull", "bear", "sideways"] as MarketRegime[]).map((regime) => {
                const regimePerfs = performances.filter((p) => p.regime === regime);
                const avgReturn =
                  regimePerfs.reduce((sum, p) => sum + p.avgReturn, 0) / regimePerfs.length;
                const avgVol =
                  regimePerfs.reduce((sum, p) => sum + p.avgVolatility, 0) / regimePerfs.length;

                return (
                  <Card
                    key={regime}
                    className={cn("border-neutral-800", `bg-${REGIME_COLORS[regime].bg.split("-")[1]}-500/5`)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-center gap-2 mb-3">
                        <RegimeIcon regime={regime} size="md" />
                        <span className={cn("text-sm font-semibold capitalize", REGIME_COLORS[regime].text)}>
                          {regime} Markets
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <div className="text-xs text-neutral-500">Avg Return</div>
                          <div
                            className={cn(
                              "text-lg font-mono font-bold",
                              avgReturn >= 0 ? "text-green-400" : "text-red-400"
                            )}
                          >
                            {avgReturn >= 0 ? "+" : ""}
                            {avgReturn.toFixed(1)}%
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-neutral-500">Avg Volatility</div>
                          <div className="text-lg font-mono font-bold text-neutral-200">
                            {avgVol.toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <div className="flex items-center justify-between py-4 border-t border-neutral-800 text-xs text-neutral-500">
          <div className="flex items-center gap-4">
            <span>QDT Nexus Regime Analytics</span>
            <Badge className="bg-neutral-800 border-neutral-700 text-neutral-400">v1.0.0</Badge>
          </div>
          <div className="flex items-center gap-2">
            <span>Data: 6-month rolling window | Updated daily</span>
          </div>
        </div>
      </div>
    </div>
  );
}
