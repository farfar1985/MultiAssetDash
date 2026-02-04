"use client";

import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import type { SignalDirection } from "@/types";

interface ModelAgreementChartProps {
  bullishCount: number;
  bearishCount: number;
  neutralCount: number;
  totalModels: number;
  overallDirection: SignalDirection;
}

export function ModelAgreementChart({
  bullishCount,
  bearishCount,
  neutralCount,
  totalModels,
  overallDirection,
}: ModelAgreementChartProps) {
  const directionColor =
    overallDirection === "bullish"
      ? "#22c55e"
      : overallDirection === "bearish"
      ? "#ef4444"
      : "#6b7280";

  const option: EChartsOption = {
    backgroundColor: "transparent",
    animation: true,
    animationDuration: 800,
    animationEasing: "cubicOut",
    tooltip: {
      trigger: "item",
      backgroundColor: "rgba(23, 23, 23, 0.95)",
      borderColor: "#404040",
      borderWidth: 1,
      textStyle: {
        color: "#f5f5f5",
        fontSize: 12,
      },
      formatter: (params: unknown) => {
        const p = params as {
          name: string;
          value: number;
          percent: number;
          color: string;
        };
        return `
          <div style="font-family: monospace;">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${p.color};margin-right:4px;"></span>
            ${p.name}: ${p.value.toLocaleString()} (${p.percent.toFixed(1)}%)
          </div>
        `;
      },
    },
    series: [
      {
        name: "Model Agreement",
        type: "pie",
        radius: ["55%", "80%"],
        center: ["50%", "50%"],
        avoidLabelOverlap: false,
        padAngle: 2,
        itemStyle: {
          borderRadius: 4,
        },
        label: {
          show: false,
        },
        emphasis: {
          scale: true,
          scaleSize: 5,
        },
        data: [
          {
            value: bullishCount,
            name: "Bullish",
            itemStyle: { color: "#22c55e" },
          },
          {
            value: bearishCount,
            name: "Bearish",
            itemStyle: { color: "#ef4444" },
          },
          {
            value: neutralCount,
            name: "Neutral",
            itemStyle: { color: "#6b7280" },
          },
        ],
      },
    ],
    graphic: {
      type: "group",
      left: "center",
      top: "center",
      children: [
        {
          type: "text",
          style: {
            text: overallDirection.toUpperCase(),
            fill: directionColor,
            fontSize: 14,
            fontWeight: "bold" as const,
            fontFamily: "ui-monospace, monospace",
          },
          left: "center",
          top: -10,
        },
        {
          type: "text",
          style: {
            text: `${totalModels.toLocaleString()}`,
            fill: "#a3a3a3",
            fontSize: 10,
            fontFamily: "ui-monospace, monospace",
          },
          left: "center",
          top: 10,
        },
      ],
    } as unknown as EChartsOption["graphic"],
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
