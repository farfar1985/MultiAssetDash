"use client";

import { TierComparisonCard, type TierComparisonData } from "./TierComparisonCard";
import { useTierComparison } from "@/hooks/useApi";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import type { AssetId } from "@/types";
import { RefreshCw, AlertCircle } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface APITierComparisonCardProps {
  assetId: AssetId;
  compact?: boolean;
  className?: string;
}

// ============================================================================
// Loading Skeleton
// ============================================================================

function TierComparisonSkeleton({ compact }: { compact?: boolean }) {
  if (compact) {
    return (
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-neutral-800 rounded animate-pulse" />
              <div className="w-28 h-4 bg-neutral-800 rounded animate-pulse" />
            </div>
            <div className="w-20 h-5 bg-neutral-800 rounded animate-pulse" />
          </div>
          <div className="flex gap-2">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="flex-1 p-2 rounded-md bg-neutral-800/50 border border-neutral-800"
              >
                <div className="w-12 h-3 bg-neutral-700 rounded animate-pulse mx-auto mb-1" />
                <div className="w-6 h-4 bg-neutral-700 rounded animate-pulse mx-auto" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 bg-neutral-800 rounded-lg animate-pulse" />
            <div>
              <div className="w-40 h-4 bg-neutral-800 rounded animate-pulse mb-1" />
              <div className="w-32 h-3 bg-neutral-800 rounded animate-pulse" />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        {/* Consensus skeleton */}
        <div className="rounded-lg border border-neutral-800 p-4 bg-neutral-900/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 bg-neutral-800 rounded-lg animate-pulse" />
              <div>
                <div className="w-32 h-4 bg-neutral-800 rounded animate-pulse mb-1" />
                <div className="w-24 h-3 bg-neutral-800 rounded animate-pulse" />
              </div>
            </div>
            <div className="w-16 h-8 bg-neutral-800 rounded animate-pulse" />
          </div>
        </div>

        {/* Tier cards skeleton */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="rounded-lg border border-neutral-800 p-4 bg-neutral-900/50"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 bg-neutral-800 rounded-md animate-pulse" />
                  <div>
                    <div className="w-16 h-4 bg-neutral-800 rounded animate-pulse mb-1" />
                    <div className="w-20 h-3 bg-neutral-800 rounded animate-pulse" />
                  </div>
                </div>
                <div className="w-20 h-5 bg-neutral-800 rounded animate-pulse" />
              </div>
              <div className="mb-3">
                <div className="flex justify-between mb-1">
                  <div className="w-16 h-3 bg-neutral-800 rounded animate-pulse" />
                  <div className="w-10 h-3 bg-neutral-800 rounded animate-pulse" />
                </div>
                <div className="h-1.5 bg-neutral-800 rounded-full" />
              </div>
              <div className="w-full h-3 bg-neutral-800 rounded animate-pulse mb-3" />
              <div className="space-y-2">
                {[1, 2].map((j) => (
                  <div key={j} className="h-8 bg-neutral-800/50 rounded animate-pulse" />
                ))}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Error State
// ============================================================================

interface ErrorStateProps {
  onRetry: () => void;
  compact?: boolean;
}

function TierComparisonError({ onRetry, compact }: ErrorStateProps) {
  if (compact) {
    return (
      <Card className="bg-neutral-900/50 border-red-500/30">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-red-400">
              <AlertCircle className="w-4 h-4" />
              <span className="text-xs">Failed to load</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onRetry}
              className="h-6 px-2 text-xs"
            >
              <RefreshCw className="w-3 h-3 mr-1" />
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-neutral-900/50 border-red-500/30">
      <CardContent className="p-8 text-center">
        <div className="flex flex-col items-center gap-4">
          <div className="p-3 bg-red-500/10 rounded-full">
            <AlertCircle className="w-8 h-8 text-red-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-neutral-100 mb-1">
              Failed to Load Tier Comparison
            </h3>
            <p className="text-xs text-neutral-500">
              Unable to fetch ensemble predictions from all tiers.
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onRetry}
            className="border-neutral-700 hover:border-neutral-600"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Try Again
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Default/Fallback Data
// ============================================================================

const DEFAULT_TIER_DATA: TierComparisonData = {
  asset_id: 0,
  asset_name: "Unknown",
  timestamp: new Date().toISOString(),
  tier1: {
    signal: "NEUTRAL",
    confidence: 0.5,
    netProbability: 0,
    weights: { accuracy: 0.33, magnitude: 0.33, correlation: 0.34 },
  },
  tier2: {
    signal: "NEUTRAL",
    confidence: 0.5,
    netProbability: 0,
    uncertainty: 0.25,
    interval: { lower: -2, upper: 2 },
    regime: "sideways",
  },
  tier3: {
    signal: "NEUTRAL",
    confidence: 0.5,
    netProbability: 0,
    quantiles: { "0.1": -2, "0.25": -1, "0.5": 0, "0.75": 1, "0.9": 2 },
    explorationBonus: 0,
  },
  consensus: {
    signal: "NEUTRAL",
    agreement: 1,
    tiersAgreeing: 3,
    totalTiers: 3,
  },
};

// ============================================================================
// Main Component
// ============================================================================

export function APITierComparisonCard({
  assetId,
  compact = false,
  className,
}: APITierComparisonCardProps) {
  const { data, isLoading, isError, refetch } = useTierComparison(assetId);

  if (isLoading) {
    return (
      <div className={className}>
        <TierComparisonSkeleton compact={compact} />
      </div>
    );
  }

  if (isError) {
    return (
      <div className={className}>
        <TierComparisonError onRetry={refetch} compact={compact} />
      </div>
    );
  }

  return (
    <div className={className}>
      <TierComparisonCard data={data || DEFAULT_TIER_DATA} compact={compact} />
    </div>
  );
}
