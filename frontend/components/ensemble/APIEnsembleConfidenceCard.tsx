"use client";

import { EnsembleConfidenceCard, type EnsembleConfidenceCardProps, type EnsembleConfidenceData } from "./EnsembleConfidenceCard";
import { useEnsembleConfidence } from "@/hooks";
import type { AssetId } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, RefreshCw } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface APIEnsembleConfidenceCardProps extends Omit<EnsembleConfidenceCardProps, "data"> {
  assetId: AssetId;
  fallbackData?: EnsembleConfidenceData;
  showLoadingSkeleton?: boolean;
  showError?: boolean;
}

// ============================================================================
// Loading Skeleton
// ============================================================================

function ConfidenceLoadingSkeleton({ compact }: { compact?: boolean }) {
  if (compact) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-xl border border-neutral-800 bg-neutral-900/50">
        <Skeleton className="h-10 w-10 rounded-lg" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-5 w-20" />
          <Skeleton className="h-3 w-32" />
        </div>
      </div>
    );
  }

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardContent className="pt-6 space-y-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-6 w-32" />
          <Skeleton className="h-5 w-24" />
        </div>
        <Skeleton className="h-16 w-full" />
        <div className="space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Error State
// ============================================================================

function ConfidenceErrorState({ compact, onRetry }: { compact?: boolean; onRetry?: () => void }) {
  if (compact) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-xl border border-red-500/30 bg-red-500/10">
        <AlertCircle className="h-5 w-5 text-red-400" />
        <span className="text-sm text-red-400">Failed to load confidence</span>
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
          <p className="text-red-400 font-medium">Failed to load confidence data</p>
          <p className="text-sm text-neutral-500">Unable to fetch ensemble confidence</p>
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

const DEFAULT_FALLBACK: EnsembleConfidenceData = {
  confidence: 50,
  direction: "neutral",
  weights: [
    { method: "Default", weight: 1.0, contribution: 0.5, accuracy: 50.0 }
  ],
  modelsAgreeing: 0,
  modelsTotal: 0,
  ensembleMethod: "accuracy-weighted",
};

// ============================================================================
// Component
// ============================================================================

export function APIEnsembleConfidenceCard({
  assetId,
  fallbackData = DEFAULT_FALLBACK,
  showLoadingSkeleton = true,
  showError = true,
  compact = false,
  ...props
}: APIEnsembleConfidenceCardProps) {
  const { data, isLoading, isError, refetch } = useEnsembleConfidence(assetId);

  if (isLoading && showLoadingSkeleton) {
    return <ConfidenceLoadingSkeleton compact={compact} />;
  }

  if (isError && showError) {
    return <ConfidenceErrorState compact={compact} onRetry={() => refetch()} />;
  }

  return (
    <EnsembleConfidenceCard
      data={data || fallbackData}
      compact={compact}
      {...props}
    />
  );
}

export default APIEnsembleConfidenceCard;
