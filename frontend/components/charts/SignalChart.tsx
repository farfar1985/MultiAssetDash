"use client";

import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import type { SignalDirection } from "@/types";

export interface SignalHistoryData {
  date: string;
  direction: SignalDirection;
  confidence: number;
  strength: number; // -100 to 100, negative = bearish, positive = bullish
}

interface SignalChartProps {
  data: SignalHistoryData[];
  showConfidenceBand?: boolean;
}

export function SignalChart({ data, showConfidenceBand = true }: SignalChartProps) {
  const dates = data.map((d) => d.date);
  const strengthData = data.map((d) => d.strength);
  const confidenceUpper = data.map((d) => d.confidence);
  const confidenceLower = data.map((d) => -d.confidence);

  const option: EChartsOption = {
    backgroundColor: "transparent",
    animation: true,
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(23, 23, 23, 0.95)",
      borderColor: "#404040",
      borderWidth: 1,
      textStyle: {
        color: "#f5f5f5",
        fontSize: 12,
      },
      formatter: (params: unknown) => {
        const p = params as Array<{
          axisValue: string;
          data: number;
          seriesName: string;
        }>;
        if (!p || p.length === 0) return "";
        const idx = dates.indexOf(p[0].axisValue);
        if (idx === -1) return "";
        const d = data[idx];
        const directionColor =
          d.direction === "bullish"
            ? "#22c55e"
            : d.direction === "bearish"
            ? "#ef4444"
            : "#6b7280";
        return `
          <div style="font-family: monospace;">
            <div style="margin-bottom: 4px; font-weight: bold;">${d.date}</div>
            <div>Direction: <span style="color: ${directionColor}; text-transform: capitalize;">${d.direction}</span></div>
            <div>Strength: <span style="color: ${d.strength >= 0 ? "#22c55e" : "#ef4444"}">${d.strength > 0 ? "+" : ""}${d.strength}</span></div>
            <div>Confidence: <span style="color: #3b82f6">${d.confidence}%</span></div>
          </div>
        `;
      },
    },
    grid: {
      left: "8%",
      right: "4%",
      top: "12%",
      bottom: "15%",
    },
    xAxis: {
      type: "category",
      data: dates,
      boundaryGap: false,
      axisLine: { lineStyle: { color: "#404040" } },
      axisLabel: { color: "#a3a3a3", fontSize: 10 },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      min: -100,
      max: 100,
      axisLine: { lineStyle: { color: "#404040" } },
      axisLabel: { color: "#a3a3a3", fontSize: 10 },
      splitLine: { lineStyle: { color: "#262626", type: "dashed" } },
    },
    visualMap: {
      show: false,
      pieces: [
        { lte: -30, color: "#ef4444" },
        { gt: -30, lte: 30, color: "#6b7280" },
        { gt: 30, color: "#22c55e" },
      ],
      seriesIndex: 0,
    },
    series: [
      {
        name: "Signal Strength",
        type: "line",
        data: strengthData,
        smooth: true,
        symbol: "none",
        lineStyle: {
          width: 2,
        },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(34, 197, 94, 0.4)" },
              { offset: 0.5, color: "rgba(107, 114, 128, 0.1)" },
              { offset: 1, color: "rgba(239, 68, 68, 0.4)" },
            ],
          },
        },
        markLine: {
          silent: true,
          symbol: "none",
          lineStyle: { color: "#404040", type: "dashed" },
          data: [{ yAxis: 0 }],
        },
      },
      ...(showConfidenceBand
        ? [
            {
              name: "Confidence Upper",
              type: "line" as const,
              data: confidenceUpper,
              smooth: true,
              symbol: "none",
              lineStyle: {
                color: "rgba(59, 130, 246, 0.5)",
                width: 1,
                type: "dashed" as const,
              },
            },
            {
              name: "Confidence Lower",
              type: "line" as const,
              data: confidenceLower,
              smooth: true,
              symbol: "none",
              lineStyle: {
                color: "rgba(59, 130, 246, 0.5)",
                width: 1,
                type: "dashed" as const,
              },
            },
          ]
        : []),
    ],
    dataZoom: [
      {
        type: "inside",
        start: 0,
        end: 100,
      },
    ],
  };

  return (
    <ReactECharts
      option={option}
      style={{ height: "100%", width: "100%" }}
      opts={{ renderer: "canvas" }}
      notMerge={true}
    />
  );
}
