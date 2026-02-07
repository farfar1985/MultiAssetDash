"use client";

import { PairwiseVotingChart, type PairwiseVotingChartProps, type PairwiseVotingData } from "./PairwiseVotingChart";
import { usePairwiseVoting } from "@/hooks";
import type { AssetId } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, RefreshCw } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface APIPairwiseVotingChartProps extends Omit<PairwiseVotingChartProps, "data"> {
  assetId: AssetId;
  fallbackData?: PairwiseVotingData;
  showLoadingSkeleton?: boolean;
  showError?: boolean;
}

// ============================================================================
// Loading Skeleton
// ============================================================================

function VotingLoadingSkeleton({ compact }: { compact?: boolean }) {
  if (compact) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-xl border border-neutral-800 bg-neutral-900/50">
        <Skeleton className="h-8 w-8 rounded-lg" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-3 w-16" />
        </div>
      </div>
    );
  }

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardContent className="pt-6 space-y-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-6 w-36" />
          <Skeleton className="h-5 w-20" />
        </div>
        <div className="flex gap-2">
          <Skeleton className="h-8 flex-1" />
          <Skeleton className="h-8 flex-1" />
          <Skeleton className="h-8 flex-1" />
        </div>
        <div className="grid grid-cols-3 gap-2">
          {[...Array(6)].map((_, i) => (
            <Skeleton key={i} className="h-10 w-full" />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Error State
// ============================================================================

function VotingErrorState({ compact, onRetry }: { compact?: boolean; onRetry?: () => void }) {
  if (compact) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-xl border border-red-500/30 bg-red-500/10">
        <AlertCircle className="h-5 w-5 text-red-400" />
        <span className="text-sm text-red-400">Failed to load voting</span>
        {onRetry && (
          <button onClick={onRetry} className="ml-auto p-1 hover:bg-red-500/20 rounded">
            <RefreshCw className="h-4 w-4 text-red-400" />
          </button>
        )}
      </div>
    );
  }

  return (
    <Card className="bg-neutral-900/50 border-red-500/30">
      <CardContent className="pt-6 flex flex-col items-center justify-center py-8 space-y-4">
        <AlertCircle className="h-12 w-12 text-red-400" />
        <div className="text-center">
          <p className="text-red-400 font-medium">Failed to load voting data</p>
          <p className="text-sm text-neutral-500">Unable to fetch pairwise voting</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded-lg text-red-400 text-sm transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            Retry
          </button>
        )}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Default Fallback
// ============================================================================

const DEFAULT_FALLBACK: PairwiseVotingData = {
  votes: [],
  bullishCount: 0,
  bearishCount: 0,
  neutralCount: 0,
  netProbability: 0,
  signal: "neutral",
};

// ============================================================================
// Component
// ============================================================================

export function APIPairwiseVotingChart({
  assetId,
  fallbackData = DEFAULT_FALLBACK,
  showLoadingSkeleton = true,
  showError = true,
  compact = false,
  ...props
}: APIPairwiseVotingChartProps) {
  const { data, isLoading, isError, refetch } = usePairwiseVoting(assetId);

  if (isLoading && showLoadingSkeleton) {
    return <VotingLoadingSkeleton compact={compact} />;
  }

  if (isError && showError) {
    return <VotingErrorState compact={compact} onRetry={() => refetch()} />;
  }

  return (
    <PairwiseVotingChart
      data={data || fallbackData}
      compact={compact}
      {...props}
    />
  );
}

export default APIPairwiseVotingChart;
