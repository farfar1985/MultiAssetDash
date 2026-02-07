"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import {
  MultiAssetRegimeOverview,
  APIEnsembleConfidenceCard,
  HMMRegimeIndicator,
  useAllRegimes,
} from "@/components/ensemble";
import type { AssetId } from "@/types";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  Clock,
  Globe,
  LayoutGrid,
  Radio,
  Sparkles,
  Table,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";

// ============================================================================
// Constants
// ============================================================================

const ALL_ASSETS: Array<{ id: AssetId; name: string; symbol: string; category: string }> = [
  { id: "crude-oil", name: "Crude Oil", symbol: "CL", category: "Commodities" },
  { id: "gold", name: "Gold", symbol: "GC", category: "Commodities" },
  { id: "silver", name: "Silver", symbol: "SI", category: "Commodities" },
  { id: "copper", name: "Copper", symbol: "HG", category: "Commodities" },
  { id: "natural-gas", name: "Natural Gas", symbol: "NG", category: "Commodities" },
  { id: "wheat", name: "Wheat", symbol: "ZW", category: "Commodities" },
  { id: "corn", name: "Corn", symbol: "ZC", category: "Commodities" },
  { id: "soybean", name: "Soybean", symbol: "ZS", category: "Commodities" },
  { id: "platinum", name: "Platinum", symbol: "PL", category: "Commodities" },
  { id: "bitcoin", name: "Bitcoin", symbol: "BTC", category: "Crypto" },
  { id: "ethereum", name: "Ethereum", symbol: "ETH", category: "Crypto" },
  { id: "sp500", name: "S&P 500", symbol: "SPX", category: "Indices" },
  { id: "nasdaq", name: "NASDAQ", symbol: "NDX", category: "Indices" },
];

// ============================================================================
// Early Warning Detection
// ============================================================================

interface EarlyWarning {
  assetId: AssetId;
  assetName: string;
  currentRegime: string;
  targetRegime: string;
  probability: number;
  urgency: "low" | "medium" | "high";
}

function detectWarningsFromRegimes(
  regimes: Record<string, { probabilities?: Record<string, number>; regime?: string; volatility?: number; confidence?: number; display_name?: string }>
): EarlyWarning[] {
  const warnings: EarlyWarning[] = [];

  Object.entries(regimes).forEach(([key, data]) => {
    if (!data?.probabilities) return;

    const probs = data.probabilities;
    const currentRegime = data.regime;

    // Map regime to probability key
    const regimeToProb: Record<string, string> = {
      bull: "bull",
      bear: "bear",
      sideways: "sideways",
    };

    const currentProbKey = (currentRegime && regimeToProb[currentRegime]) || "sideways";
    const currentProb = probs[currentProbKey] || 0;

    // Find competing regime
    const competitors = Object.entries(probs)
      .filter(([k]) => k !== currentProbKey)
      .sort(([, a], [, b]) => (b as number) - (a as number));

    if (competitors.length === 0) return;

    const [topCompetitorKey, topCompetitorProb] = competitors[0] as [string, number];
    const gap = currentProb - topCompetitorProb;

    // Determine if warning should be shown
    let urgency: "low" | "medium" | "high" | null = null;

    if (topCompetitorProb >= 0.40 && gap < 0.10) {
      urgency = "high";
    } else if (topCompetitorProb >= 0.35 && gap < 0.15) {
      urgency = "medium";
    } else if (topCompetitorProb >= 0.30 && gap < 0.20) {
      urgency = "low";
    }

    if (urgency) {
      // Find asset info
      const asset = ALL_ASSETS.find(
        (a) => a.name.toLowerCase().replace(/\s+/g, "_") === key.toLowerCase()
      );

      warnings.push({
        assetId: asset?.id || (key as AssetId),
        assetName: data.display_name || key,
        currentRegime: currentRegime || "unknown",
        targetRegime: topCompetitorKey,
        probability: topCompetitorProb,
        urgency,
      });
    }
  });

  // Sort by urgency (high first) then probability
  return warnings.sort((a, b) => {
    const urgencyOrder = { high: 0, medium: 1, low: 2 };
    if (urgencyOrder[a.urgency] !== urgencyOrder[b.urgency]) {
      return urgencyOrder[a.urgency] - urgencyOrder[b.urgency];
    }
    return b.probability - a.probability;
  });
}

// ============================================================================
// Components
// ============================================================================

function EarlyWarningsPanel({ warnings }: { warnings: EarlyWarning[] }) {
  if (warnings.length === 0) {
    return (
      <Card className="bg-green-500/5 border-green-500/20">
        <CardContent className="py-6">
          <div className="flex items-center justify-center gap-3">
            <div className="p-2 bg-green-500/10 rounded-lg">
              <Activity className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-green-400">All Clear</p>
              <p className="text-xs text-neutral-500">No regime shift warnings detected</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            Early Warning Signals
          </CardTitle>
          <Badge className="bg-amber-500/10 text-amber-400 border-amber-500/30 text-xs animate-pulse">
            {warnings.length} Active
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0 space-y-3">
        {warnings.map((warning) => (
          <div
            key={warning.assetId}
            className={cn(
              "p-4 rounded-xl border",
              warning.urgency === "high"
                ? "bg-red-500/10 border-red-500/30"
                : warning.urgency === "medium"
                ? "bg-orange-500/10 border-orange-500/30"
                : "bg-amber-500/10 border-amber-500/30"
            )}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-neutral-100">{warning.assetName}</span>
                <Badge
                  className={cn(
                    "text-[9px] uppercase",
                    warning.urgency === "high"
                      ? "bg-red-500/20 text-red-400 border-red-500/30"
                      : warning.urgency === "medium"
                      ? "bg-orange-500/20 text-orange-400 border-orange-500/30"
                      : "bg-amber-500/20 text-amber-400 border-amber-500/30"
                  )}
                >
                  {warning.urgency}
                </Badge>
              </div>
              <span className="text-xs text-neutral-500">
                {(warning.probability * 100).toFixed(0)}% probability
              </span>
            </div>
            <p className="text-xs text-neutral-400">
              Potential shift from{" "}
              <span
                className={cn(
                  "font-medium",
                  warning.currentRegime === "bull"
                    ? "text-green-400"
                    : warning.currentRegime === "bear"
                    ? "text-red-400"
                    : "text-amber-400"
                )}
              >
                {warning.currentRegime}
              </span>{" "}
              to{" "}
              <span
                className={cn(
                  "font-medium",
                  warning.targetRegime === "bull"
                    ? "text-green-400"
                    : warning.targetRegime === "bear"
                    ? "text-red-400"
                    : "text-amber-400"
                )}
              >
                {warning.targetRegime}
              </span>
            </p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function EnsembleConfidenceGrid() {
  // Show confidence for key assets
  const keyAssets: AssetId[] = ["crude-oil", "gold", "bitcoin", "sp500", "natural-gas", "ethereum"];

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Brain className="w-4 h-4 text-purple-400" />
            Ensemble Confidence Scores
          </CardTitle>
          <Badge className="bg-purple-500/10 text-purple-400 border-purple-500/30 text-xs">
            Live API
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {keyAssets.map((assetId) => (
            <APIEnsembleConfidenceCard
              key={assetId}
              assetId={assetId}
              showBreakdown={false}
              compact={true}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function QuickRegimeGrid() {
  // Show regime for key assets in compact form
  const keyAssets: AssetId[] = [
    "crude-oil",
    "gold",
    "bitcoin",
    "sp500",
    "natural-gas",
    "silver",
    "ethereum",
    "copper",
  ];

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Quick Regime Status
          </CardTitle>
          <Badge className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30 text-xs">
            HMM Detected
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {keyAssets.map((assetId) => {
            const asset = ALL_ASSETS.find((a) => a.id === assetId);
            return (
              <div key={assetId} className="space-y-1">
                <div className="text-xs text-neutral-500">{asset?.name || assetId}</div>
                <HMMRegimeIndicator
                  assetId={assetId}
                  showProbabilities={false}
                  compact={true}
                  size="sm"
                />
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

function MarketSummaryStats({ regimeData }: { regimeData?: { regime_distribution?: Record<string, number>; total_assets?: number } }) {
  const distribution = regimeData?.regime_distribution || {};
  const _total = regimeData?.total_assets || 0;

  const stats = [
    {
      label: "Bull Markets",
      value: distribution.bull || 0,
      icon: TrendingUp,
      color: "text-green-400",
      bg: "bg-green-500/10",
    },
    {
      label: "Bear Markets",
      value: distribution.bear || 0,
      icon: TrendingDown,
      color: "text-red-400",
      bg: "bg-red-500/10",
    },
    {
      label: "Sideways",
      value: distribution.sideways || 0,
      icon: BarChart3,
      color: "text-amber-400",
      bg: "bg-amber-500/10",
    },
    {
      label: "High Volatility",
      value: distribution["high-volatility"] || 0,
      icon: Zap,
      color: "text-orange-400",
      bg: "bg-orange-500/10",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {stats.map((stat) => (
        <Card key={stat.label} className="bg-neutral-900/50 border-neutral-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className={cn("p-2 rounded-lg", stat.bg)}>
                <stat.icon className={cn("w-5 h-5", stat.color)} />
              </div>
              <div>
                <div className="text-2xl font-bold text-neutral-100 font-mono">
                  {stat.value}
                </div>
                <div className="text-xs text-neutral-500">{stat.label}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ============================================================================
// Main Page Component
// ============================================================================

export default function SignalsDashboardPage() {
  const [viewMode, setViewMode] = useState<"grid" | "table">("grid");
  const { data: regimeData, isLoading } = useAllRegimes();

  // Detect early warnings from regime data
  const earlyWarnings = useMemo(() => {
    if (!regimeData?.regimes) return [];
    return detectWarningsFromRegimes(regimeData.regimes);
  }, [regimeData]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-950 via-neutral-900 to-neutral-950 p-6">
      <div className="max-w-[1800px] mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-cyan-500/20 to-purple-500/20 rounded-lg border border-cyan-500/30">
                <Radio className="w-6 h-6 text-cyan-400" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Unified Signals Dashboard</h1>
                <p className="text-sm text-neutral-400">
                  Live regime status, ensemble confidence, and early warnings across all 13 assets
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Live indicator */}
            <Badge className="bg-green-500/10 text-green-400 border-green-500/30 text-xs">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse" />
              Live
            </Badge>

            {/* Model badges */}
            <Badge className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30 text-xs">
              <Activity className="w-3 h-3 mr-1" />
              HMM Regimes
            </Badge>
            <Badge className="bg-purple-500/10 text-purple-400 border-purple-500/30 text-xs">
              <Brain className="w-3 h-3 mr-1" />
              10,179 Models
            </Badge>

            {/* Auto-refresh */}
            <div className="flex items-center gap-1 text-xs text-neutral-500">
              <Clock className="w-3 h-3" />
              <span>Auto-refresh: 60s</span>
            </div>
          </div>
        </div>

        {/* Market Summary Stats */}
        {isLoading ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-24 bg-neutral-800" />
            ))}
          </div>
        ) : (
          <MarketSummaryStats regimeData={regimeData} />
        )}

        {/* Main Content Tabs */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="bg-neutral-800/50 border border-neutral-700/50 p-1">
            <TabsTrigger
              value="overview"
              className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400"
            >
              <Globe className="w-4 h-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger
              value="regimes"
              className="data-[state=active]:bg-indigo-500/20 data-[state=active]:text-indigo-400"
            >
              <Activity className="w-4 h-4 mr-2" />
              Full Regime Status
            </TabsTrigger>
            <TabsTrigger
              value="ensemble"
              className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400"
            >
              <Brain className="w-4 h-4 mr-2" />
              Ensemble Details
            </TabsTrigger>
            <TabsTrigger
              value="warnings"
              className="data-[state=active]:bg-amber-500/20 data-[state=active]:text-amber-400"
            >
              <AlertTriangle className="w-4 h-4 mr-2" />
              Warnings
              {earlyWarnings.length > 0 && (
                <Badge className="ml-2 bg-amber-500/20 text-amber-400 border-amber-500/30 text-[9px]">
                  {earlyWarnings.length}
                </Badge>
              )}
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Early Warnings (if any) */}
            <EarlyWarningsPanel warnings={earlyWarnings} />

            {/* Quick Regime Grid */}
            <QuickRegimeGrid />

            {/* Ensemble Confidence Grid */}
            <EnsembleConfidenceGrid />
          </TabsContent>

          {/* Full Regime Status Tab */}
          <TabsContent value="regimes" className="space-y-4">
            <div className="flex items-center justify-end gap-2">
              <div className="flex items-center gap-1 bg-neutral-800/50 p-1 rounded-lg border border-neutral-700/50">
                <button
                  onClick={() => setViewMode("grid")}
                  className={cn(
                    "p-2 rounded-md transition-colors",
                    viewMode === "grid"
                      ? "bg-indigo-500/20 text-indigo-400"
                      : "text-neutral-400 hover:text-neutral-200"
                  )}
                >
                  <LayoutGrid className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewMode("table")}
                  className={cn(
                    "p-2 rounded-md transition-colors",
                    viewMode === "table"
                      ? "bg-indigo-500/20 text-indigo-400"
                      : "text-neutral-400 hover:text-neutral-200"
                  )}
                >
                  <Table className="w-4 h-4" />
                </button>
              </div>
            </div>

            <MultiAssetRegimeOverview viewMode={viewMode} />
          </TabsContent>

          {/* Ensemble Details Tab */}
          <TabsContent value="ensemble" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {ALL_ASSETS.map((asset) => (
                <div key={asset.id} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-neutral-200">{asset.name}</span>
                    <Badge className="text-[9px] bg-neutral-800 text-neutral-400 border-neutral-700">
                      {asset.symbol}
                    </Badge>
                  </div>
                  <APIEnsembleConfidenceCard
                    assetId={asset.id}
                    showBreakdown={true}
                    compact={false}
                  />
                </div>
              ))}
            </div>
          </TabsContent>

          {/* Warnings Tab */}
          <TabsContent value="warnings" className="space-y-6">
            <EarlyWarningsPanel warnings={earlyWarnings} />

            {/* Detailed Regime Indicators with Warning Display */}
            <Card className="bg-neutral-900/50 border-neutral-800">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  Detailed Regime Indicators
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {ALL_ASSETS.slice(0, 6).map((asset) => (
                    <HMMRegimeIndicator
                      key={asset.id}
                      assetId={asset.id}
                      showProbabilities={true}
                      compact={false}
                      size="md"
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <div className="flex items-center justify-between py-4 border-t border-neutral-800 text-xs text-neutral-500">
          <div className="flex items-center gap-4">
            <span>QDT Nexus Unified Signals</span>
            <Badge className="bg-neutral-800 border-neutral-700 text-neutral-400">
              v1.0.0
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Sparkles className="w-3 h-3" />
            <span>Powered by HMM + Ensemble Intelligence</span>
          </div>
        </div>
      </div>
    </div>
  );
}
