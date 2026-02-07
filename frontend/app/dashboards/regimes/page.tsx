"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { MultiAssetRegimeOverview } from "@/components/ensemble";
import {
  Activity,
  Globe,
  LayoutGrid,
  Table,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

type ViewMode = "grid" | "table";
type CategoryFilter = "all" | "Commodities" | "Crypto" | "Indices";

// ============================================================================
// Page Component
// ============================================================================

export default function RegimesDashboardPage() {
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [categoryFilter, setCategoryFilter] = useState<CategoryFilter>("all");

  const handleAssetClick = (assetId: number, assetName: string) => {
    console.log(`Navigate to detailed view for ${assetName} (ID: ${assetId})`);
    // Future: router.push(`/dashboards/asset/${assetId}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-950 via-neutral-900 to-neutral-950 p-6">
      <div className="max-w-[1800px] mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-500/10 rounded-lg border border-indigo-500/30">
                <Globe className="w-6 h-6 text-indigo-400" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Multi-Asset Regime Monitor</h1>
                <p className="text-sm text-neutral-400">
                  HMM-based market regime detection across 13 assets
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* View Mode Toggle */}
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

            {/* Model Info Badge */}
            <Badge className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30 text-xs">
              <Activity className="w-3 h-3 mr-1" />
              GaussianHMM (3-State)
            </Badge>

            {/* Last Updated */}
            <div className="flex items-center gap-1 text-xs text-neutral-500">
              <Clock className="w-3 h-3" />
              <span>Auto-refresh: 60s</span>
            </div>
          </div>
        </div>

        {/* Category Filter Tabs */}
        <Tabs value={categoryFilter} onValueChange={(v) => setCategoryFilter(v as CategoryFilter)}>
          <TabsList className="bg-neutral-800/50 border border-neutral-700/50 p-1">
            <TabsTrigger
              value="all"
              className="data-[state=active]:bg-white/10 data-[state=active]:text-white text-neutral-400"
            >
              All Assets
            </TabsTrigger>
            <TabsTrigger
              value="Commodities"
              className="data-[state=active]:bg-amber-500/20 data-[state=active]:text-amber-400 text-neutral-400"
            >
              Commodities
            </TabsTrigger>
            <TabsTrigger
              value="Crypto"
              className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400 text-neutral-400"
            >
              Crypto
            </TabsTrigger>
            <TabsTrigger
              value="Indices"
              className="data-[state=active]:bg-blue-500/20 data-[state=active]:text-blue-400 text-neutral-400"
            >
              Indices
            </TabsTrigger>
          </TabsList>

          <TabsContent value="all" className="mt-6">
            <MultiAssetRegimeOverview
              viewMode={viewMode}
              onAssetClick={handleAssetClick}
            />
          </TabsContent>

          <TabsContent value="Commodities" className="mt-6">
            <MultiAssetRegimeOverview
              viewMode={viewMode}
              category="Commodities"
              onAssetClick={handleAssetClick}
            />
          </TabsContent>

          <TabsContent value="Crypto" className="mt-6">
            <MultiAssetRegimeOverview
              viewMode={viewMode}
              category="Crypto"
              onAssetClick={handleAssetClick}
            />
          </TabsContent>

          <TabsContent value="Indices" className="mt-6">
            <MultiAssetRegimeOverview
              viewMode={viewMode}
              category="Indices"
              onAssetClick={handleAssetClick}
            />
          </TabsContent>
        </Tabs>

        {/* Legend */}
        <Card className="bg-neutral-900/30 border-neutral-800">
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-6">
                <span className="text-xs text-neutral-500 uppercase tracking-wider">Regime Legend:</span>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-green-400" />
                    <span className="text-xs text-neutral-300">Bull</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-red-400" />
                    <span className="text-xs text-neutral-300">Bear</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Minus className="w-4 h-4 text-amber-400" />
                    <span className="text-xs text-neutral-300">Sideways</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-orange-400" />
                    <span className="text-xs text-neutral-300">High Volatility</span>
                  </div>
                </div>
              </div>
              <div className="text-xs text-neutral-500">
                Models calibrated: 2026-02-06 by Amira
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <div className="flex items-center justify-between py-4 border-t border-neutral-800 text-xs text-neutral-500">
          <div className="flex items-center gap-4">
            <span>QDT Nexus Regime Detection</span>
            <Badge className="bg-neutral-800 border-neutral-700 text-neutral-400">
              v1.0.0
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <span>Powered by Hidden Markov Models</span>
          </div>
        </div>
      </div>
    </div>
  );
}
