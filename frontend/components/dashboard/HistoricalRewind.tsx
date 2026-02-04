"use client";

import { useState, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useHistoricalMetrics } from "@/hooks/useApi";
import type { HistoricalMetrics, BacktestResult } from "@/types";

interface HistoricalRewindProps {
  symbol: string;
  startDate?: string;
  endDate?: string;
  currentMetrics?: BacktestResult;
  onDateChange?: (date: string, metrics: HistoricalMetrics | null) => void;
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function getDefaultDateRange(): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setMonth(start.getMonth() - 6);
  return {
    start: start.toISOString().split("T")[0],
    end: end.toISOString().split("T")[0],
  };
}

function MetricComparison({
  label,
  historical,
  current,
  format = "number",
  higherIsBetter = true,
}: {
  label: string;
  historical: number | undefined;
  current: number | undefined;
  format?: "number" | "percent" | "ratio";
  higherIsBetter?: boolean;
}) {
  const formatValue = (val: number | undefined): string => {
    if (val === undefined) return "-";
    switch (format) {
      case "percent":
        return `${val.toFixed(2)}%`;
      case "ratio":
        return val.toFixed(2);
      default:
        return val.toFixed(2);
    }
  };

  const getDelta = (): { value: string; isPositive: boolean } | null => {
    if (historical === undefined || current === undefined) return null;
    const delta = current - historical;
    const isPositive = higherIsBetter ? delta > 0 : delta < 0;
    const sign = delta >= 0 ? "+" : "";
    return {
      value: `${sign}${formatValue(delta)}`,
      isPositive,
    };
  };

  const delta = getDelta();

  return (
    <div className="flex items-center justify-between py-2 border-b border-neutral-800 last:border-0">
      <span className="text-xs text-neutral-500 uppercase tracking-wider">
        {label}
      </span>
      <div className="flex items-center gap-3">
        <span className="font-mono text-sm text-neutral-400">
          {formatValue(historical)}
        </span>
        {delta && (
          <Badge
            className={cn(
              "text-xs font-mono px-1.5 py-0.5",
              delta.isPositive
                ? "bg-green-500/10 border-green-500/30 text-green-500"
                : "bg-red-500/10 border-red-500/30 text-red-500"
            )}
          >
            {delta.value}
          </Badge>
        )}
        <span className="font-mono text-sm font-medium text-neutral-100">
          {formatValue(current)}
        </span>
      </div>
    </div>
  );
}

function HistoricalRewindSkeleton() {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <Skeleton className="h-5 w-32 bg-neutral-800" />
          <Skeleton className="h-4 w-24 bg-neutral-800" />
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        <Skeleton className="h-8 w-full bg-neutral-800" />
        <div className="space-y-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="flex justify-between">
              <Skeleton className="h-4 w-20 bg-neutral-800" />
              <Skeleton className="h-4 w-32 bg-neutral-800" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function HistoricalRewind({
  symbol,
  startDate,
  endDate,
  currentMetrics,
  onDateChange,
}: HistoricalRewindProps) {
  const defaultRange = useMemo(() => getDefaultDateRange(), []);
  const queryStartDate = startDate || defaultRange.start;
  const queryEndDate = endDate || defaultRange.end;

  const {
    data: historicalData,
    isLoading,
    error,
  } = useHistoricalMetrics(symbol, queryStartDate, queryEndDate);

  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const selectedMetrics = useMemo(() => {
    if (!historicalData || selectedIndex === null) return null;
    return historicalData[selectedIndex] || null;
  }, [historicalData, selectedIndex]);

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const index = parseInt(e.target.value, 10);
      setSelectedIndex(index);
      if (historicalData && onDateChange) {
        const metrics = historicalData[index] || null;
        onDateChange(metrics?.as_of_date || "", metrics);
      }
    },
    [historicalData, onDateChange]
  );

  if (isLoading) {
    return <HistoricalRewindSkeleton />;
  }

  if (error) {
    return (
      <Card className="bg-neutral-900/50 border-red-900/50">
        <CardContent className="p-4">
          <div className="text-center text-red-500 text-sm py-4">
            <p>Failed to load historical data</p>
            <p className="text-xs text-neutral-500 mt-1">{error.message}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!historicalData || historicalData.length === 0) {
    return (
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="text-center text-neutral-500 text-sm py-4">
            No historical data available for this period
          </div>
        </CardContent>
      </Card>
    );
  }

  const maxIndex = historicalData.length - 1;
  const displayIndex = selectedIndex ?? maxIndex;
  const displayMetrics = selectedMetrics || historicalData[maxIndex];

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-neutral-100">
              Historical Rewind
            </span>
            <Badge className="bg-blue-500/10 border-blue-500/30 text-blue-500 text-xs">
              Time Travel
            </Badge>
          </div>
          <span className="text-xs text-neutral-500 font-mono">
            {formatDate(queryStartDate)} - {formatDate(queryEndDate)}
          </span>
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0 space-y-4">
        {/* Date Slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-neutral-500">Viewing as of:</span>
            <span className="font-mono text-sm font-medium text-blue-400">
              {displayMetrics ? formatDate(displayMetrics.as_of_date) : "-"}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={maxIndex}
            value={displayIndex}
            onChange={handleSliderChange}
            className={cn(
              "w-full h-2 rounded-full appearance-none cursor-pointer",
              "bg-neutral-800",
              "[&::-webkit-slider-thumb]:appearance-none",
              "[&::-webkit-slider-thumb]:w-4",
              "[&::-webkit-slider-thumb]:h-4",
              "[&::-webkit-slider-thumb]:rounded-full",
              "[&::-webkit-slider-thumb]:bg-blue-500",
              "[&::-webkit-slider-thumb]:hover:bg-blue-400",
              "[&::-webkit-slider-thumb]:transition-colors",
              "[&::-moz-range-thumb]:w-4",
              "[&::-moz-range-thumb]:h-4",
              "[&::-moz-range-thumb]:rounded-full",
              "[&::-moz-range-thumb]:bg-blue-500",
              "[&::-moz-range-thumb]:border-0",
              "[&::-moz-range-thumb]:hover:bg-blue-400"
            )}
          />
          <div className="flex justify-between text-[10px] text-neutral-600 font-mono">
            <span>{formatDate(historicalData[0].as_of_date)}</span>
            <span>{formatDate(historicalData[maxIndex].as_of_date)}</span>
          </div>
        </div>

        {/* Metrics Comparison */}
        {displayMetrics && (
          <div className="pt-2">
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-neutral-700">
              <span className="text-[10px] uppercase tracking-wider text-neutral-500">
                Metric
              </span>
              <div className="flex items-center gap-3">
                <span className="text-[10px] uppercase tracking-wider text-neutral-500 w-16 text-right">
                  Historical
                </span>
                <span className="text-[10px] uppercase tracking-wider text-neutral-500 w-12 text-center">
                  Delta
                </span>
                <span className="text-[10px] uppercase tracking-wider text-neutral-500 w-16 text-right">
                  Current
                </span>
              </div>
            </div>

            <MetricComparison
              label="Sharpe"
              historical={displayMetrics.metrics.sharpe}
              current={currentMetrics?.sharpe}
              format="ratio"
              higherIsBetter={true}
            />
            <MetricComparison
              label="Sortino"
              historical={displayMetrics.metrics.sortino}
              current={currentMetrics?.sortino}
              format="ratio"
              higherIsBetter={true}
            />
            <MetricComparison
              label="Win Rate"
              historical={displayMetrics.metrics.win_rate}
              current={currentMetrics?.win_rate}
              format="percent"
              higherIsBetter={true}
            />
            <MetricComparison
              label="Max DD"
              historical={displayMetrics.metrics.max_drawdown}
              current={currentMetrics?.max_drawdown}
              format="percent"
              higherIsBetter={false}
            />
            <MetricComparison
              label="ROI 30d"
              historical={displayMetrics.metrics.roi_30}
              current={currentMetrics?.roi_30}
              format="percent"
              higherIsBetter={true}
            />
            <MetricComparison
              label="ROI 90d"
              historical={displayMetrics.metrics.roi_90}
              current={currentMetrics?.roi_90}
              format="percent"
              higherIsBetter={true}
            />
          </div>
        )}

        {/* Snapshot Info */}
        {displayMetrics && (
          <div className="flex items-center justify-between pt-2 border-t border-neutral-800">
            <span className="text-xs text-neutral-500">
              Snapshot: {displayMetrics.snapshot_type}
            </span>
            <span className="text-xs text-neutral-500 font-mono">
              {displayMetrics.model_accuracies.length} models tracked
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
