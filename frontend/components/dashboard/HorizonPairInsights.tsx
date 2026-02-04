"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { AssetId } from "@/types";
import type { HorizonPairInsight } from "@/types/horizon-pairs";
import { ACCURACY_THRESHOLDS, getAccuracyColor } from "@/types/horizon-pairs";
import {
  getHorizonPairData,
  findBestPair,
  calculateAverageAccuracy,
  countAlphaPairs,
} from "@/lib/mock-horizon-data";

interface HorizonPairInsightsProps {
  assetId: AssetId;
  className?: string;
}

/**
 * Generate insights from horizon pair data
 */
function generateInsights(assetId: AssetId): HorizonPairInsight {
  const data = getHorizonPairData(assetId);
  const bestPair = findBestPair(data);
  const averageAccuracy = calculateAverageAccuracy(data);
  const alphaPairCount = countAlphaPairs(data);

  // Calculate recommended weight multiplier
  // Higher accuracy = higher weight, capped at 3x
  const excessAccuracy = bestPair.accuracy - ACCURACY_THRESHOLDS.good;
  const recommendedWeight = Math.min(3, 1 + excessAccuracy / 10);

  // Determine confidence
  let confidence: "high" | "medium" | "low";
  if (bestPair.sampleSize >= 500 && bestPair.accuracy >= 65) {
    confidence = "high";
  } else if (bestPair.sampleSize >= 300 && bestPair.accuracy >= 58) {
    confidence = "medium";
  } else {
    confidence = "low";
  }

  // Generate recommendation
  let recommendation: string;
  if (bestPair.accuracy >= 70) {
    recommendation = `Strong alpha signal: Weight ${bestPair.h1}/${bestPair.h2} pair ${recommendedWeight.toFixed(1)}x higher in ensemble`;
  } else if (bestPair.accuracy >= 65) {
    recommendation = `Exploitable edge: Consider ${bestPair.h1}/${bestPair.h2} pair as primary signal with ${recommendedWeight.toFixed(1)}x weight`;
  } else if (bestPair.accuracy >= 60) {
    recommendation = `Marginal edge detected. Use ${bestPair.h1}/${bestPair.h2} as confirmation signal only`;
  } else if (bestPair.accuracy >= 55) {
    recommendation = `Weak signal. Equal-weight all horizon pairs or rely on volume-weighted consensus`;
  } else {
    recommendation = `No exploitable edge. This asset shows near-random pair correlations`;
  }

  return {
    bestPair,
    averageAccuracy,
    alphaPairCount,
    recommendedWeight: Math.round(recommendedWeight * 10) / 10,
    recommendation,
    confidence,
  };
}

export function HorizonPairInsights({
  assetId,
  className,
}: HorizonPairInsightsProps) {
  const insights = useMemo(() => generateInsights(assetId), [assetId]);

  const { bestPair, averageAccuracy, alphaPairCount, recommendedWeight, recommendation, confidence } = insights;

  const accuracyDelta = bestPair.accuracy - averageAccuracy;
  const isPositiveDelta = accuracyDelta > 0;

  const confidenceColors = {
    high: "text-green-400 bg-green-500/10 border-green-500/30",
    medium: "text-yellow-400 bg-yellow-500/10 border-yellow-500/30",
    low: "text-red-400 bg-red-500/10 border-red-500/30",
  };

  return (
    <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-neutral-100 text-sm font-medium">
            Horizon Pair Insights
          </CardTitle>
          <Badge
            variant="outline"
            className={cn("text-xs border", confidenceColors[confidence])}
          >
            {confidence.charAt(0).toUpperCase() + confidence.slice(1)} Confidence
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        {/* Best Pair */}
        <div className="bg-neutral-800/50 rounded-lg p-3 border border-neutral-700/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-neutral-500 uppercase tracking-wide">
              Best Performing Pair
            </span>
            {bestPair.accuracy >= ACCURACY_THRESHOLDS.exceptional && (
              <span className="text-xs text-green-400">⚡ Alpha Source</span>
            )}
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-lg font-mono font-bold text-neutral-100">
                {bestPair.h1}
              </span>
              <span className="text-neutral-500">→</span>
              <span className="text-lg font-mono font-bold text-neutral-100">
                {bestPair.h2}
              </span>
            </div>
            <span
              className="text-2xl font-bold font-mono"
              style={{ color: getAccuracyColor(bestPair.accuracy) }}
            >
              {bestPair.accuracy.toFixed(1)}%
            </span>
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            n={bestPair.sampleSize} samples
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-3 gap-3">
          {/* Average Accuracy */}
          <div className="bg-neutral-800/30 rounded-lg p-2.5">
            <div className="text-xs text-neutral-500 mb-1">Avg Accuracy</div>
            <div className="text-lg font-mono font-semibold text-neutral-200">
              {averageAccuracy.toFixed(1)}%
            </div>
          </div>

          {/* Delta from Average */}
          <div className="bg-neutral-800/30 rounded-lg p-2.5">
            <div className="text-xs text-neutral-500 mb-1">vs. Average</div>
            <div
              className={cn(
                "text-lg font-mono font-semibold",
                isPositiveDelta ? "text-green-400" : "text-red-400"
              )}
            >
              {isPositiveDelta ? "+" : ""}
              {accuracyDelta.toFixed(1)}%
            </div>
          </div>

          {/* Alpha Pairs */}
          <div className="bg-neutral-800/30 rounded-lg p-2.5">
            <div className="text-xs text-neutral-500 mb-1">Alpha Pairs</div>
            <div
              className={cn(
                "text-lg font-mono font-semibold",
                alphaPairCount > 0 ? "text-green-400" : "text-neutral-400"
              )}
            >
              {alphaPairCount}
            </div>
          </div>
        </div>

        {/* Recommendation */}
        <div className="border-t border-neutral-800 pt-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs text-neutral-500 uppercase tracking-wide">
              Recommendation
            </span>
            {recommendedWeight > 1 && (
              <Badge
                variant="outline"
                className="bg-blue-500/10 text-blue-400 border-blue-500/30 text-xs"
              >
                {recommendedWeight.toFixed(1)}x Weight
              </Badge>
            )}
          </div>
          <p className="text-sm text-neutral-300 leading-relaxed">
            {recommendation}
          </p>
        </div>

        {/* Action Hint */}
        {bestPair.accuracy >= ACCURACY_THRESHOLDS.exceptional && (
          <div className="flex items-center gap-2 p-2 bg-green-500/5 border border-green-500/20 rounded-lg">
            <svg
              className="w-4 h-4 text-green-400 flex-shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
            <span className="text-xs text-green-400">
              This pair shows statistically significant alpha potential. Consider
              for production signal weighting.
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
