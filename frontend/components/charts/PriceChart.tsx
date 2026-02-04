"use client";

import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";

export interface OHLCData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface PriceChartProps {
  data: OHLCData[];
  timeframe?: string;
  assetName: string;
  showVolume?: boolean;
  chartType?: "candlestick" | "line";
}

export function PriceChart({
  data,
  assetName,
  showVolume = true,
  chartType = "candlestick",
}: PriceChartProps) {
  const dates = data.map((d) => d.date);
  const ohlcData = data.map((d) => [d.open, d.close, d.low, d.high]);
  const volumeData = data.map((d) => d.volume ?? 0);
  const lineData = data.map((d) => d.close);

  const upColor = "#22c55e";
  const downColor = "#ef4444";

  const option: EChartsOption = {
    backgroundColor: "transparent",
    animation: true,
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "cross",
        crossStyle: {
          color: "#6b7280",
        },
        lineStyle: {
          color: "#6b7280",
          type: "dashed",
        },
      },
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
          data: number | number[];
          seriesName: string;
          color: string;
        }>;
        if (!p || p.length === 0) return "";
        const idx = p[0];
        const dataIdx = dates.indexOf(idx.axisValue);
        if (dataIdx === -1) return "";
        const d = data[dataIdx];
        return `
          <div style="font-family: monospace;">
            <div style="margin-bottom: 4px; font-weight: bold;">${idx.axisValue}</div>
            <div>Open: <span style="color: ${upColor}">$${d.open.toFixed(2)}</span></div>
            <div>High: <span style="color: ${upColor}">$${d.high.toFixed(2)}</span></div>
            <div>Low: <span style="color: ${downColor}">$${d.low.toFixed(2)}</span></div>
            <div>Close: <span style="color: ${d.close >= d.open ? upColor : downColor}">$${d.close.toFixed(2)}</span></div>
            ${d.volume ? `<div>Volume: ${d.volume.toLocaleString()}</div>` : ""}
          </div>
        `;
      },
    },
    axisPointer: {
      link: [{ xAxisIndex: "all" }],
      label: {
        backgroundColor: "#171717",
      },
    },
    grid: showVolume
      ? [
          { left: "8%", right: "4%", top: "8%", height: "60%" },
          { left: "8%", right: "4%", top: "75%", height: "15%" },
        ]
      : [{ left: "8%", right: "4%", top: "8%", bottom: "12%" }],
    xAxis: showVolume
      ? [
          {
            type: "category",
            data: dates,
            boundaryGap: chartType === "candlestick",
            axisLine: { lineStyle: { color: "#404040" } },
            axisLabel: { color: "#a3a3a3", fontSize: 10 },
            splitLine: { show: false },
          },
          {
            type: "category",
            gridIndex: 1,
            data: dates,
            boundaryGap: true,
            axisLine: { lineStyle: { color: "#404040" } },
            axisLabel: { show: false },
            splitLine: { show: false },
          },
        ]
      : [
          {
            type: "category",
            data: dates,
            boundaryGap: chartType === "candlestick",
            axisLine: { lineStyle: { color: "#404040" } },
            axisLabel: { color: "#a3a3a3", fontSize: 10 },
            splitLine: { show: false },
          },
        ],
    yAxis: showVolume
      ? [
          {
            scale: true,
            splitArea: { show: false },
            axisLine: { lineStyle: { color: "#404040" } },
            axisLabel: {
              color: "#a3a3a3",
              fontSize: 10,
              formatter: (value: number) => `$${value.toFixed(0)}`,
            },
            splitLine: { lineStyle: { color: "#262626", type: "dashed" } },
          },
          {
            scale: true,
            gridIndex: 1,
            splitNumber: 2,
            axisLabel: { show: false },
            axisLine: { show: false },
            axisTick: { show: false },
            splitLine: { show: false },
          },
        ]
      : [
          {
            scale: true,
            splitArea: { show: false },
            axisLine: { lineStyle: { color: "#404040" } },
            axisLabel: {
              color: "#a3a3a3",
              fontSize: 10,
              formatter: (value: number) => `$${value.toFixed(0)}`,
            },
            splitLine: { lineStyle: { color: "#262626", type: "dashed" } },
          },
        ],
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: showVolume ? [0, 1] : [0],
        start: 50,
        end: 100,
      },
      {
        show: true,
        xAxisIndex: showVolume ? [0, 1] : [0],
        type: "slider",
        bottom: showVolume ? 0 : 0,
        height: 20,
        start: 50,
        end: 100,
        borderColor: "#404040",
        backgroundColor: "rgba(38, 38, 38, 0.5)",
        fillerColor: "rgba(59, 130, 246, 0.2)",
        handleStyle: { color: "#3b82f6" },
        textStyle: { color: "#a3a3a3", fontSize: 10 },
      },
    ],
    series: showVolume
      ? [
          chartType === "candlestick"
            ? {
                name: assetName,
                type: "candlestick",
                data: ohlcData,
                itemStyle: {
                  color: upColor,
                  color0: downColor,
                  borderColor: upColor,
                  borderColor0: downColor,
                },
              }
            : {
                name: assetName,
                type: "line",
                data: lineData,
                smooth: true,
                symbol: "none",
                lineStyle: {
                  color: "#3b82f6",
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
                      { offset: 0, color: "rgba(59, 130, 246, 0.3)" },
                      { offset: 1, color: "rgba(59, 130, 246, 0)" },
                    ],
                  },
                },
              },
          {
            name: "Volume",
            type: "bar",
            xAxisIndex: 1,
            yAxisIndex: 1,
            data: volumeData,
            itemStyle: {
              color: (params: { dataIndex: number }) => {
                const idx = params.dataIndex;
                return data[idx].close >= data[idx].open ? upColor : downColor;
              },
              opacity: 0.5,
            },
          },
        ]
      : [
          chartType === "candlestick"
            ? {
                name: assetName,
                type: "candlestick",
                data: ohlcData,
                itemStyle: {
                  color: upColor,
                  color0: downColor,
                  borderColor: upColor,
                  borderColor0: downColor,
                },
              }
            : {
                name: assetName,
                type: "line",
                data: lineData,
                smooth: true,
                symbol: "none",
                lineStyle: {
                  color: "#3b82f6",
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
                      { offset: 0, color: "rgba(59, 130, 246, 0.3)" },
                      { offset: 1, color: "rgba(59, 130, 246, 0)" },
                    ],
                  },
                },
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
