"use client";

import { ConfidenceIntervalBar, type ConfidenceIntervalBarProps, type ConfidenceInterval } from "./ConfidenceIntervalBar";
import { useConfidenceInterval } from "@/hooks";
import type { AssetId } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, RefreshCw } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface APIConfidenceIntervalBarProps extends Omit<ConfidenceIntervalBarProps, "data"> {
  assetId: AssetId;
  /** Forecast horizon in days (e.g., 5 for D+5) */
  horizonDays?: number;
  fallbackData?: ConfidenceInterval;
  showLoadingSkeleton?: boolean;
  showError?: boolean;
}

// ============================================================================
// Loading Skeleton
// ============================================================================

function IntervalLoadingSkeleton({ compact }: { compact?: boolean }) {
  if (compact) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-xl border border-neutral-800 bg-neutral-900/50">
        <Skeleton className="h-6 w-full" />
      </div>
    );
  }

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardContent className="pt-6 space-y-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-5 w-32" />
          <Skeleton className="h-5 w-16" />
        </div>
        <Skeleton className="h-12 w-full" />
        <div className="flex justify-between">
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-16" />
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Error State
// ============================================================================

function IntervalErrorState({ compact, onRetry }: { compact?: boolean; onRetry?: () => void }) {
  if (compact) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-xl border border-red-500/30 bg-red-500/10">
        <AlertCircle className="h-5 w-5 text-red-400" />
        <span className="text-sm text-red-400">Failed to load interval</span>
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
          <p className="text-red-400 font-medium">Failed to load interval data</p>
          <p className="text-sm text-neutral-500">Unable to fetch confidence interval</p>
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

const DEFAULT_FALLBACK: ConfidenceInterval = {
  lower: 0,
  point: 0,
  upper: 0,
  coverage: 0.90,
};

// ============================================================================
// Component
// ============================================================================

export function APIConfidenceIntervalBar({
  assetId,
  horizonDays = 5,
  fallbackData = DEFAULT_FALLBACK,
  showLoadingSkeleton = true,
  showError = true,
  compact = false,
  horizon,
  ...props
}: APIConfidenceIntervalBarProps) {
  const { data, isLoading, isError, refetch } = useConfidenceInterval(assetId, horizonDays);

  if (isLoading && showLoadingSkeleton) {
    return <IntervalLoadingSkeleton compact={compact} />;
  }

  if (isError && showError) {
    return <IntervalErrorState compact={compact} onRetry={() => refetch()} />;
  }

  return (
    <ConfidenceIntervalBar
      data={data || fallbackData}
      compact={compact}
      horizon={horizon || `D+${horizonDays}`}
      {...props}
    />
  );
}

export default APIConfidenceIntervalBar;
