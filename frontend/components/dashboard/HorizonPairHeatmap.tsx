"use client";

import { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { AssetId } from "@/types";
// HorizonPairMatrix and Horizon types available from @/types/horizon-pairs if needed
import {
  HORIZONS,
  getAccuracyColor,
  isAlphaSource,
  ACCURACY_THRESHOLDS,
} from "@/types/horizon-pairs";
import { getHorizonPairData, findBestPair } from "@/lib/mock-horizon-data";

interface HorizonPairHeatmapProps {
  assetId: AssetId;
  className?: string;
  showLegend?: boolean;
  height?: number;
}

function HorizonPairHeatmapSkeleton() {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-2">
        <Skeleton className="h-5 w-48 bg-neutral-800" />
      </CardHeader>
      <CardContent className="p-4">
        <Skeleton className="h-[300px] w-full bg-neutral-800" />
      </CardContent>
    </Card>
  );
}

export function HorizonPairHeatmap({
  assetId,
  className,
  showLegend = true,
  height = 350,
}: HorizonPairHeatmapProps) {
  const [isLoading] = useState(false);

  // Get data for this asset
  const data = useMemo(() => getHorizonPairData(assetId), [assetId]);
  const bestPair = useMemo(() => findBestPair(data), [data]);

  // Transform data for heatmap
  const heatmapData = useMemo(() => {
    const horizons = HORIZONS;
    const result: [number, number, number, number][] = []; // [x, y, accuracy, sampleSize]

    for (let i = 0; i < horizons.length; i++) {
      for (let j = 0; j < horizons.length; j++) {
        if (i === j) {
          // Diagonal - self correlation (100%)
          result.push([j, i, 100, 0]);
        } else {
          const pair = data.pairs.find(
            (p) => p.h1 === horizons[i] && p.h2 === horizons[j]
          );
          if (pair) {
            result.push([j, i, pair.accuracy, pair.sampleSize]);
          }
        }
      }
    }

    return result;
  }, [data]);

  // Chart configuration
  const option: EChartsOption = useMemo(
    () => ({
      backgroundColor: "transparent",
      animation: true,
      tooltip: {
        trigger: "item",
        backgroundColor: "rgba(23, 23, 23, 0.95)",
        borderColor: "#404040",
        borderWidth: 1,
        textStyle: {
          color: "#f5f5f5",
          fontSize: 12,
          fontFamily: "monospace",
        },
        formatter: (params: unknown) => {
          const p = params as {
            data: [number, number, number, number];
          };
          const [x, y, accuracy, sampleSize] = p.data;
          const h1 = HORIZONS[y];
          const h2 = HORIZONS[x];

          if (h1 === h2) {
            return `<div style="font-family: monospace;">
              <div style="color: #a3a3a3;">Self-correlation</div>
            </div>`;
          }

          const isAlpha = isAlphaSource(accuracy);
          const color = getAccuracyColor(accuracy);

          return `<div style="font-family: monospace;">
            <div style="font-weight: bold; margin-bottom: 4px; color: #f5f5f5;">
              ${h1} → ${h2}
            </div>
            <div style="color: ${color}; font-size: 16px; font-weight: bold;">
              ${accuracy.toFixed(1)}% accuracy
            </div>
            <div style="color: #a3a3a3; font-size: 11px; margin-top: 4px;">
              Sample: n=${sampleSize}
            </div>
            ${isAlpha ? '<div style="color: #22c55e; margin-top: 4px; font-size: 11px;">⚡ Alpha Source</div>' : ""}
          </div>`;
        },
      },
      grid: {
        left: "12%",
        right: "8%",
        top: "8%",
        bottom: showLegend ? "20%" : "12%",
      },
      xAxis: {
        type: "category",
        data: HORIZONS,
        position: "top",
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: "#a3a3a3",
          fontSize: 11,
          fontFamily: "monospace",
        },
        splitArea: { show: false },
      },
      yAxis: {
        type: "category",
        data: HORIZONS,
        inverse: true,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: "#a3a3a3",
          fontSize: 11,
          fontFamily: "monospace",
        },
        splitArea: { show: false },
      },
      visualMap: showLegend
        ? {
            min: 45,
            max: 75,
            calculable: false,
            orient: "horizontal",
            left: "center",
            bottom: "2%",
            inRange: {
              color: [
                "#ef4444", // red - poor
                "#f97316", // orange
                "#eab308", // yellow - marginal
                "#84cc16", // lime
                "#22c55e", // green - good
              ],
            },
            textStyle: {
              color: "#a3a3a3",
              fontSize: 10,
            },
            formatter: "{value}%",
          }
        : undefined,
      series: [
        {
          name: "Accuracy",
          type: "heatmap",
          data: heatmapData,
          label: {
            show: true,
            formatter: (params: unknown) => {
              const p = params as { data: [number, number, number, number] };
              const [x, y, accuracy] = p.data;
              if (HORIZONS[x] === HORIZONS[y]) return "-";
              return `${accuracy.toFixed(0)}`;
            },
            color: "#f5f5f5",
            fontSize: 11,
            fontWeight: "bold",
            fontFamily: "monospace",
          },
          itemStyle: {
            borderColor: "#262626",
            borderWidth: 2,
            borderRadius: 4,
          },
          emphasis: {
            itemStyle: {
              borderColor: "#525252",
              borderWidth: 2,
              shadowBlur: 10,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
        },
        // Highlight alpha sources with a border
        {
          name: "Alpha Highlight",
          type: "heatmap",
          data: heatmapData.filter(([, , acc]) => isAlphaSource(acc)),
          label: { show: false },
          itemStyle: {
            borderColor: "#22c55e",
            borderWidth: 3,
            borderRadius: 4,
            color: "transparent",
          },
          silent: true,
        },
      ],
    }),
    [heatmapData, showLegend]
  );

  if (isLoading) {
    return <HorizonPairHeatmapSkeleton />;
  }

  return (
    <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
      <CardHeader className="p-4 pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-neutral-100 text-sm font-medium">
            Horizon Pair Accuracy Matrix
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className="bg-neutral-800/50 border-neutral-700 text-neutral-300 text-xs"
            >
              {data.assetName}
            </Badge>
            {bestPair.accuracy >= ACCURACY_THRESHOLDS.exceptional && (
              <Badge className="bg-green-500/20 text-green-400 border-green-500/30 text-xs">
                ⚡ {bestPair.h1}→{bestPair.h2}: {bestPair.accuracy.toFixed(1)}%
              </Badge>
            )}
          </div>
        </div>
        <p className="text-xs text-neutral-500 mt-1">
          Directional accuracy when horizon pairs agree. Green borders indicate
          alpha sources (&gt;65%).
        </p>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts
          option={option}
          style={{ height: `${height}px`, width: "100%" }}
          opts={{ renderer: "canvas" }}
          notMerge={true}
        />
      </CardContent>
    </Card>
  );
}
