"use client";

import { RegimeIndicator, type RegimeIndicatorProps, type RegimeData } from "./RegimeIndicator";
import { useHMMRegime } from "@/hooks";
import type { AssetId } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, RefreshCw } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface HMMRegimeIndicatorProps extends Omit<RegimeIndicatorProps, "data"> {
  /** Asset to fetch regime for */
  assetId: AssetId;
  /** Optional fallback data to use while loading or on error */
  fallbackData?: RegimeData;
  /** Show loading skeleton */
  showLoadingSkeleton?: boolean;
  /** Show error state */
  showError?: boolean;
}

// ============================================================================
// Loading Skeleton
// ============================================================================

function RegimeLoadingSkeleton({ compact, size: _size }: { compact?: boolean; size?: "sm" | "md" | "lg" }) {
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
        <div className="flex items-center gap-4">
          <Skeleton className="h-14 w-14 rounded-xl" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-6 w-32" />
            <Skeleton className="h-4 w-48" />
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
        </div>
        <Skeleton className="h-16 w-full" />
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Error State
// ============================================================================

function RegimeErrorState({ compact, onRetry }: { compact?: boolean; onRetry?: () => void }) {
  if (compact) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-xl border border-red-500/30 bg-red-500/10">
        <AlertCircle className="h-5 w-5 text-red-400" />
        <span className="text-sm text-red-400">Failed to load regime</span>
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
          <p className="text-red-400 font-medium">Failed to load regime data</p>
          <p className="text-sm text-neutral-500">Unable to fetch HMM regime detection</p>
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
// Default Fallback Data
// ============================================================================

const DEFAULT_FALLBACK: RegimeData = {
  regime: "sideways",
  confidence: 0.5,
  probabilities: { bull: 0.33, bear: 0.33, sideways: 0.34 },
  daysInRegime: 1,
  volatility: 20.0,
  trendStrength: 0.0,
  historicalAccuracy: 50.0,
};

// ============================================================================
// Component
// ============================================================================

/**
 * HMM Regime Indicator that fetches data from the API
 *
 * This wrapper component uses the useHMMRegime hook to fetch real
 * regime data from the backend HMM endpoint and renders the
 * RegimeIndicator component with the fetched data.
 *
 * Features:
 * - Automatic data fetching with react-query
 * - Loading skeleton state
 * - Error state with retry
 * - Fallback data support
 * - Auto-refresh every 60 seconds
 */
export function HMMRegimeIndicator({
  assetId,
  fallbackData = DEFAULT_FALLBACK,
  showLoadingSkeleton = true,
  showError = true,
  compact = false,
  size = "md",
  ...props
}: HMMRegimeIndicatorProps) {
  const { data, isLoading, isError, refetch } = useHMMRegime(assetId);

  // Show loading skeleton
  if (isLoading && showLoadingSkeleton) {
    return <RegimeLoadingSkeleton compact={compact} size={size} />;
  }

  // Show error state
  if (isError && showError) {
    return <RegimeErrorState compact={compact} onRetry={() => refetch()} />;
  }

  // Use fetched data or fallback
  const regimeData = data || fallbackData;

  return (
    <RegimeIndicator
      data={regimeData}
      compact={compact}
      size={size}
      {...props}
    />
  );
}

export default HMMRegimeIndicator;
