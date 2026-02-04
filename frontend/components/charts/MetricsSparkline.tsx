"use client";

import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";

interface MetricsSparklineProps {
  data: number[];
  trend: "up" | "down" | "flat";
  width?: number;
  height?: number;
}

export function MetricsSparkline({
  data,
  trend,
  width = 60,
  height = 24,
}: MetricsSparklineProps) {
  const color =
    trend === "up" ? "#22c55e" : trend === "down" ? "#ef4444" : "#6b7280";

  const option: EChartsOption = {
    backgroundColor: "transparent",
    animation: false,
    grid: {
      left: 0,
      right: 0,
      top: 2,
      bottom: 2,
    },
    xAxis: {
      type: "category",
      show: false,
      data: data.map((_, i) => i),
    },
    yAxis: {
      type: "value",
      show: false,
      min: Math.min(...data) * 0.95,
      max: Math.max(...data) * 1.05,
    },
    series: [
      {
        type: "line",
        data: data,
        symbol: "none",
        smooth: true,
        lineStyle: {
          color: color,
          width: 1.5,
        },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: `${color}40` },
              { offset: 1, color: `${color}00` },
            ],
          },
        },
      },
    ],
  };

  return (
    <ReactECharts
      option={option}
      style={{ width: `${width}px`, height: `${height}px` }}
      opts={{ renderer: "canvas" }}
      notMerge={true}
    />
  );
}
