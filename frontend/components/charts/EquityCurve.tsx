"use client";

import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import { useMemo } from "react";

interface EquityPoint {
  date: string;
  equity: number;
  drawdown: number;
  benchmark?: number;
}

interface EquityCurveProps {
  data: EquityPoint[];
  showDrawdown?: boolean;
  showBenchmark?: boolean;
  strategyName?: string;
  benchmarkName?: string;
  startingCapital?: number;
}

export function EquityCurve({
  data,
  showDrawdown = true,
  showBenchmark = true,
  strategyName = "Alpha Strategy",
  benchmarkName = "Buy & Hold",
  startingCapital = 100000,
}: EquityCurveProps) {
  const chartData = useMemo(() => {
    const dates = data.map((d) => d.date);
    const equityValues = data.map((d) => d.equity);
    const drawdownValues = data.map((d) => d.drawdown * 100);
    const benchmarkValues = data.map((d) => d.benchmark ?? startingCapital);

    // Calculate performance metrics
    const finalEquity = equityValues[equityValues.length - 1] || startingCapital;
    const totalReturn = ((finalEquity - startingCapital) / startingCapital) * 100;
    const maxDrawdown = Math.min(...drawdownValues);
    const finalBenchmark = benchmarkValues[benchmarkValues.length - 1] || startingCapital;
    const benchmarkReturn = ((finalBenchmark - startingCapital) / startingCapital) * 100;

    return {
      dates,
      equityValues,
      drawdownValues,
      benchmarkValues,
      metrics: {
        totalReturn,
        maxDrawdown,
        benchmarkReturn,
        alpha: totalReturn - benchmarkReturn,
      },
    };
  }, [data, startingCapital]);

  const option: EChartsOption = {
    backgroundColor: "transparent",
    animation: true,
    animationDuration: 1500,
    animationEasing: "cubicOut",
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "cross",
        crossStyle: { color: "#6b7280" },
        lineStyle: { color: "#6b7280", type: "dashed" },
      },
      backgroundColor: "rgba(23, 23, 23, 0.95)",
      borderColor: "#404040",
      borderWidth: 1,
      textStyle: { color: "#f5f5f5", fontSize: 12 },
      formatter: (params: unknown) => {
        const p = params as Array<{
          axisValue: string;
          data: number;
          seriesName: string;
          color: string;
        }>;
        if (!p || p.length === 0) return "";

        let html = `<div style="font-family: monospace; padding: 4px;">`;
        html += `<div style="margin-bottom: 8px; font-weight: bold; border-bottom: 1px solid #404040; padding-bottom: 4px;">${p[0].axisValue}</div>`;

        p.forEach((item) => {
          const value = typeof item.data === "number" ? item.data : 0;
          let displayValue = "";
          let color = item.color;

          if (item.seriesName === "Drawdown") {
            displayValue = `${value.toFixed(2)}%`;
            color = value < 0 ? "#ef4444" : "#22c55e";
          } else {
            displayValue = `$${value.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
          }

          html += `<div style="display: flex; justify-content: space-between; margin: 4px 0;">`;
          html += `<span style="color: ${item.color};">${item.seriesName}:</span>`;
          html += `<span style="color: ${color}; margin-left: 20px;">${displayValue}</span>`;
          html += `</div>`;
        });

        html += `</div>`;
        return html;
      },
    },
    legend: {
      data: [
        strategyName,
        ...(showBenchmark ? [benchmarkName] : []),
        ...(showDrawdown ? ["Drawdown"] : []),
      ],
      top: 0,
      textStyle: { color: "#a3a3a3", fontSize: 11 },
      itemGap: 20,
    },
    grid: showDrawdown
      ? [
          { left: "8%", right: "4%", top: "12%", height: "55%" },
          { left: "8%", right: "4%", top: "75%", height: "18%" },
        ]
      : [{ left: "8%", right: "4%", top: "12%", bottom: "12%" }],
    xAxis: showDrawdown
      ? [
          {
            type: "category",
            data: chartData.dates,
            axisLine: { lineStyle: { color: "#404040" } },
            axisLabel: { color: "#a3a3a3", fontSize: 10 },
            splitLine: { show: false },
          },
          {
            type: "category",
            gridIndex: 1,
            data: chartData.dates,
            axisLine: { lineStyle: { color: "#404040" } },
            axisLabel: { show: false },
            splitLine: { show: false },
          },
        ]
      : [
          {
            type: "category",
            data: chartData.dates,
            axisLine: { lineStyle: { color: "#404040" } },
            axisLabel: { color: "#a3a3a3", fontSize: 10 },
            splitLine: { show: false },
          },
        ],
    yAxis: showDrawdown
      ? [
          {
            type: "value",
            axisLine: { lineStyle: { color: "#404040" } },
            axisLabel: {
              color: "#a3a3a3",
              fontSize: 10,
              formatter: (value: number) => `$${(value / 1000).toFixed(0)}K`,
            },
            splitLine: { lineStyle: { color: "#262626", type: "dashed" } },
          },
          {
            type: "value",
            gridIndex: 1,
            axisLine: { show: false },
            axisLabel: {
              color: "#a3a3a3",
              fontSize: 10,
              formatter: (value: number) => `${value.toFixed(0)}%`,
            },
            splitLine: { lineStyle: { color: "#262626", type: "dashed" } },
            max: 0,
            min: Math.min(...chartData.drawdownValues) * 1.2,
          },
        ]
      : [
          {
            type: "value",
            axisLine: { lineStyle: { color: "#404040" } },
            axisLabel: {
              color: "#a3a3a3",
              fontSize: 10,
              formatter: (value: number) => `$${(value / 1000).toFixed(0)}K`,
            },
            splitLine: { lineStyle: { color: "#262626", type: "dashed" } },
          },
        ],
    series: [
      // Strategy equity curve
      {
        name: strategyName,
        type: "line",
        data: chartData.equityValues,
        smooth: true,
        symbol: "none",
        lineStyle: {
          color: "#22c55e",
          width: 2.5,
        },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(34, 197, 94, 0.25)" },
              { offset: 1, color: "rgba(34, 197, 94, 0)" },
            ],
          },
        },
      },
      // Benchmark (optional)
      ...(showBenchmark
        ? [
            {
              name: benchmarkName,
              type: "line" as const,
              data: chartData.benchmarkValues,
              smooth: true,
              symbol: "none",
              lineStyle: {
                color: "#6b7280",
                width: 1.5,
                type: "dashed" as const,
              },
            },
          ]
        : []),
      // Drawdown (optional)
      ...(showDrawdown
        ? [
            {
              name: "Drawdown",
              type: "line" as const,
              xAxisIndex: 1,
              yAxisIndex: 1,
              data: chartData.drawdownValues,
              smooth: true,
              symbol: "none",
              lineStyle: {
                color: "#ef4444",
                width: 1.5,
              },
              areaStyle: {
                color: {
                  type: "linear" as const,
                  x: 0,
                  y: 0,
                  x2: 0,
                  y2: 1,
                  colorStops: [
                    { offset: 0, color: "rgba(239, 68, 68, 0)" },
                    { offset: 1, color: "rgba(239, 68, 68, 0.4)" },
                  ],
                },
              },
            },
          ]
        : []),
    ],
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: showDrawdown ? [0, 1] : [0],
        start: 0,
        end: 100,
      },
    ],
  };

  return (
    <div className="relative">
      {/* Performance metrics overlay */}
      <div className="absolute top-8 right-4 z-10 flex gap-3">
        <div className="bg-neutral-900/90 backdrop-blur-sm border border-neutral-800 rounded-lg px-3 py-2">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">Return</div>
          <div
            className={`font-mono text-sm font-bold ${
              chartData.metrics.totalReturn >= 0 ? "text-green-400" : "text-red-400"
            }`}
          >
            {chartData.metrics.totalReturn >= 0 ? "+" : ""}
            {chartData.metrics.totalReturn.toFixed(1)}%
          </div>
        </div>
        <div className="bg-neutral-900/90 backdrop-blur-sm border border-neutral-800 rounded-lg px-3 py-2">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">Max DD</div>
          <div className="font-mono text-sm font-bold text-red-400">
            {chartData.metrics.maxDrawdown.toFixed(1)}%
          </div>
        </div>
        <div className="bg-neutral-900/90 backdrop-blur-sm border border-neutral-800 rounded-lg px-3 py-2">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">Alpha</div>
          <div
            className={`font-mono text-sm font-bold ${
              chartData.metrics.alpha >= 0 ? "text-blue-400" : "text-orange-400"
            }`}
          >
            {chartData.metrics.alpha >= 0 ? "+" : ""}
            {chartData.metrics.alpha.toFixed(1)}%
          </div>
        </div>
      </div>

      <ReactECharts
        option={option}
        style={{ height: "100%", width: "100%" }}
        opts={{ renderer: "canvas" }}
        notMerge={true}
      />
    </div>
  );
}

// Generate mock equity curve data
export function generateEquityData(
  days: number = 365,
  startingCapital: number = 100000,
  dailyReturn: number = 0.0008,
  volatility: number = 0.015,
  benchmarkReturn: number = 0.0004
): EquityPoint[] {
  const data: EquityPoint[] = [];
  let equity = startingCapital;
  let benchmark = startingCapital;
  let peak = startingCapital;

  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);

  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);

    // Random walk with drift
    const randomReturn = dailyReturn + (Math.random() - 0.5) * 2 * volatility;
    const benchmarkRandomReturn = benchmarkReturn + (Math.random() - 0.5) * 2 * volatility * 0.8;

    equity *= 1 + randomReturn;
    benchmark *= 1 + benchmarkRandomReturn;

    // Track peak for drawdown
    if (equity > peak) peak = equity;
    const drawdown = (equity - peak) / peak;

    data.push({
      date: date.toISOString().split("T")[0],
      equity: Math.round(equity),
      drawdown,
      benchmark: Math.round(benchmark),
    });
  }

  return data;
}
