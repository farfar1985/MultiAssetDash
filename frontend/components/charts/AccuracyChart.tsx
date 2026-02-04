"use client";

import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";

export interface AccuracyData {
  date: string;
  accuracy30d: number;
  accuracy60d: number;
  accuracy90d: number;
}

interface AccuracyChartProps {
  data: AccuracyData[];
}

export function AccuracyChart({ data }: AccuracyChartProps) {
  const dates = data.map((d) => d.date);
  const accuracy30d = data.map((d) => d.accuracy30d);
  const accuracy60d = data.map((d) => d.accuracy60d);
  const accuracy90d = data.map((d) => d.accuracy90d);

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
          color: string;
        }>;
        if (!p || p.length === 0) return "";
        let html = `<div style="font-family: monospace;">
          <div style="margin-bottom: 4px; font-weight: bold;">${p[0].axisValue}</div>`;
        p.forEach((item) => {
          html += `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${item.color};margin-right:4px;"></span>${item.seriesName}: <span style="color: ${item.data >= 50 ? "#22c55e" : "#ef4444"}">${item.data.toFixed(1)}%</span></div>`;
        });
        html += "</div>";
        return html;
      },
    },
    legend: {
      data: ["30-Day", "60-Day", "90-Day", "Benchmark"],
      bottom: 0,
      textStyle: { color: "#a3a3a3", fontSize: 10 },
      icon: "roundRect",
      itemWidth: 12,
      itemHeight: 3,
    },
    grid: {
      left: "8%",
      right: "4%",
      top: "8%",
      bottom: "18%",
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
      min: 40,
      max: 70,
      axisLine: { lineStyle: { color: "#404040" } },
      axisLabel: {
        color: "#a3a3a3",
        fontSize: 10,
        formatter: "{value}%",
      },
      splitLine: { lineStyle: { color: "#262626", type: "dashed" } },
    },
    series: [
      {
        name: "30-Day",
        type: "line",
        data: accuracy30d,
        smooth: true,
        symbol: "none",
        lineStyle: {
          color: "#22c55e",
          width: 2,
        },
      },
      {
        name: "60-Day",
        type: "line",
        data: accuracy60d,
        smooth: true,
        symbol: "none",
        lineStyle: {
          color: "#3b82f6",
          width: 2,
        },
      },
      {
        name: "90-Day",
        type: "line",
        data: accuracy90d,
        smooth: true,
        symbol: "none",
        lineStyle: {
          color: "#a855f7",
          width: 2,
        },
      },
      {
        name: "Benchmark",
        type: "line",
        data: dates.map(() => 50),
        symbol: "none",
        lineStyle: {
          color: "#ef4444",
          width: 1,
          type: "dashed",
        },
      },
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
