"use client";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { useBacktest } from "@/hooks/useApi";
import type { BacktestResult, StrategyId } from "@/types";

interface BacktestMetricsProps {
  symbol: string;
  strategy?: StrategyId;
  metrics?: BacktestResult;
}

interface MetricDisplayProps {
  label: string;
  value: string | number;
  subtext?: string;
  valueColor?: string;
  isLoading?: boolean;
}

function MetricDisplay({
  label,
  value,
  subtext,
  valueColor = "text-neutral-100",
  isLoading = false,
}: MetricDisplayProps) {
  return (
    <div className="p-3 bg-neutral-800/50 rounded-lg">
      <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
        {label}
      </div>
      {isLoading ? (
        <Skeleton className="h-6 w-16 bg-neutral-700" />
      ) : (
        <div className={cn("text-lg font-bold font-mono", valueColor)}>
          {typeof value === "number" ? value.toFixed(2) : value}
        </div>
      )}
      {subtext && !isLoading && (
        <div className="text-[10px] text-neutral-600 mt-0.5">{subtext}</div>
      )}
    </div>
  );
}

function getRatioColor(value: number, thresholds: { good: number; bad: number }): string {
  if (value >= thresholds.good) return "text-green-500";
  if (value <= thresholds.bad) return "text-red-500";
  return "text-yellow-500";
}

function getPercentColor(value: number, higherIsBetter: boolean = true): string {
  if (higherIsBetter) {
    if (value >= 60) return "text-green-500";
    if (value <= 40) return "text-red-500";
    return "text-yellow-500";
  } else {
    if (value <= 10) return "text-green-500";
    if (value >= 25) return "text-red-500";
    return "text-yellow-500";
  }
}

function BacktestMetricsSkeleton() {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <Skeleton className="h-5 w-40 bg-neutral-800" />
          <Skeleton className="h-6 w-24 bg-neutral-800" />
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <div className="grid grid-cols-3 gap-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="p-3 bg-neutral-800/50 rounded-lg">
              <Skeleton className="h-3 w-16 bg-neutral-700 mb-2" />
              <Skeleton className="h-6 w-12 bg-neutral-700" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function BacktestMetrics({
  symbol,
  strategy = "ensemble_weighted",
  metrics: propMetrics,
}: BacktestMetricsProps) {
  const {
    data: fetchedMetrics,
    isLoading,
    error,
  } = useBacktest(symbol, strategy, {
    enabled: !propMetrics,
  });

  const metrics = propMetrics || fetchedMetrics;

  if (isLoading && !propMetrics) {
    return <BacktestMetricsSkeleton />;
  }

  if (error) {
    return (
      <Card className="bg-neutral-900/50 border-red-900/50">
        <CardContent className="p-4">
          <div className="text-center text-red-500 text-sm py-4">
            <p>Failed to load backtest metrics</p>
            <p className="text-xs text-neutral-500 mt-1">{error.message}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!metrics) {
    return (
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="text-center text-neutral-500 text-sm py-4">
            No backtest data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-neutral-100">
              Backtest Results
            </span>
            <Badge className="bg-purple-500/10 border-purple-500/30 text-purple-500 text-xs">
              quantum_ml
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-neutral-500 font-mono">
              {metrics.strategy}
            </span>
            <span className="text-[10px] text-neutral-600">|</span>
            <span className="text-xs text-neutral-500 font-mono">
              {metrics.symbol}
            </span>
          </div>
        </div>
        <div className="text-[10px] text-neutral-600 mt-1 font-mono">
          {metrics.start_date} → {metrics.end_date}
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0">
        <Tabs defaultValue="risk" className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-neutral-800/50 mb-4">
            <TabsTrigger
              value="risk"
              className="text-xs data-[state=active]:bg-neutral-700"
            >
              Risk-Adjusted
            </TabsTrigger>
            <TabsTrigger
              value="roi"
              className="text-xs data-[state=active]:bg-neutral-700"
            >
              ROI
            </TabsTrigger>
            <TabsTrigger
              value="trades"
              className="text-xs data-[state=active]:bg-neutral-700"
            >
              Trades
            </TabsTrigger>
            <TabsTrigger
              value="drawdown"
              className="text-xs data-[state=active]:bg-neutral-700"
            >
              Drawdown
            </TabsTrigger>
          </TabsList>

          {/* Risk-Adjusted Returns Tab */}
          <TabsContent value="risk" className="mt-0">
            <div className="grid grid-cols-3 gap-3">
              <MetricDisplay
                label="Sharpe Ratio"
                value={metrics.sharpe}
                subtext="Risk-adj. return"
                valueColor={getRatioColor(metrics.sharpe, { good: 1.5, bad: 0.5 })}
              />
              <MetricDisplay
                label="Sortino Ratio"
                value={metrics.sortino}
                subtext="Downside risk-adj."
                valueColor={getRatioColor(metrics.sortino, { good: 2.0, bad: 0.5 })}
              />
              <MetricDisplay
                label="Information Ratio"
                value={metrics.information_ratio}
                subtext="vs. benchmark"
                valueColor={getRatioColor(metrics.information_ratio, { good: 0.5, bad: 0 })}
              />
              <MetricDisplay
                label="Win Rate"
                value={`${metrics.win_rate.toFixed(1)}%`}
                subtext="Profitable trades"
                valueColor={getPercentColor(metrics.win_rate)}
              />
              <MetricDisplay
                label="Profit Factor"
                value={metrics.profit_factor}
                subtext="Gross profit / loss"
                valueColor={getRatioColor(metrics.profit_factor, { good: 1.5, bad: 1.0 })}
              />
              <MetricDisplay
                label="Win/Loss Ratio"
                value={metrics.win_loss_ratio}
                subtext="Avg win / avg loss"
                valueColor={getRatioColor(metrics.win_loss_ratio, { good: 1.5, bad: 0.8 })}
              />
            </div>
          </TabsContent>

          {/* ROI Tab */}
          <TabsContent value="roi" className="mt-0">
            <div className="grid grid-cols-3 gap-3">
              <MetricDisplay
                label="30-Day ROI"
                value={`${metrics.roi_30 >= 0 ? "+" : ""}${metrics.roi_30.toFixed(2)}%`}
                valueColor={metrics.roi_30 >= 0 ? "text-green-500" : "text-red-500"}
              />
              <MetricDisplay
                label="60-Day ROI"
                value={`${metrics.roi_60 >= 0 ? "+" : ""}${metrics.roi_60.toFixed(2)}%`}
                valueColor={metrics.roi_60 >= 0 ? "text-green-500" : "text-red-500"}
              />
              <MetricDisplay
                label="90-Day ROI"
                value={`${metrics.roi_90 >= 0 ? "+" : ""}${metrics.roi_90.toFixed(2)}%`}
                valueColor={metrics.roi_90 >= 0 ? "text-green-500" : "text-red-500"}
              />
              <MetricDisplay
                label="180-Day ROI"
                value={`${metrics.roi_180 >= 0 ? "+" : ""}${metrics.roi_180.toFixed(2)}%`}
                valueColor={metrics.roi_180 >= 0 ? "text-green-500" : "text-red-500"}
              />
              <MetricDisplay
                label="360-Day ROI"
                value={`${metrics.roi_360 >= 0 ? "+" : ""}${metrics.roi_360.toFixed(2)}%`}
                valueColor={metrics.roi_360 >= 0 ? "text-green-500" : "text-red-500"}
              />
              <MetricDisplay
                label="Annualized Vol"
                value={`${metrics.annualized_volatility.toFixed(2)}%`}
                subtext="Std. deviation"
                valueColor={getPercentColor(metrics.annualized_volatility, false)}
              />
            </div>
          </TabsContent>

          {/* Trades Tab */}
          <TabsContent value="trades" className="mt-0">
            <div className="grid grid-cols-3 gap-3">
              <MetricDisplay
                label="Total Trades"
                value={metrics.total_trades.toLocaleString()}
                valueColor="text-blue-400"
              />
              <MetricDisplay
                label="Avg Win"
                value={`+${metrics.avg_win.toFixed(2)}%`}
                valueColor="text-green-500"
              />
              <MetricDisplay
                label="Avg Loss"
                value={`${metrics.avg_loss.toFixed(2)}%`}
                valueColor="text-red-500"
              />
              <MetricDisplay
                label="Avg Duration"
                value={`${metrics.avg_trade_duration_days.toFixed(1)}d`}
                subtext="Per trade"
                valueColor="text-neutral-100"
              />
              <MetricDisplay
                label="Profit Factor"
                value={metrics.profit_factor}
                subtext="Total win / loss"
                valueColor={getRatioColor(metrics.profit_factor, { good: 1.5, bad: 1.0 })}
              />
              <MetricDisplay
                label="Win Rate"
                value={`${metrics.win_rate.toFixed(1)}%`}
                valueColor={getPercentColor(metrics.win_rate)}
              />
            </div>
          </TabsContent>

          {/* Drawdown Tab */}
          <TabsContent value="drawdown" className="mt-0">
            <div className="grid grid-cols-3 gap-3">
              <MetricDisplay
                label="Max Drawdown"
                value={`${metrics.max_drawdown.toFixed(2)}%`}
                subtext="Largest peak-to-trough"
                valueColor={getPercentColor(metrics.max_drawdown, false)}
              />
              <MetricDisplay
                label="Avg Drawdown"
                value={`${metrics.avg_drawdown.toFixed(2)}%`}
                subtext="Mean drawdown"
                valueColor={getPercentColor(metrics.avg_drawdown, false)}
              />
              <MetricDisplay
                label="Max DD Duration"
                value={`${metrics.max_drawdown_duration_days}d`}
                subtext="Recovery time"
                valueColor={
                  metrics.max_drawdown_duration_days <= 30
                    ? "text-green-500"
                    : metrics.max_drawdown_duration_days >= 90
                    ? "text-red-500"
                    : "text-yellow-500"
                }
              />
              <MetricDisplay
                label="Downside Vol"
                value={`${metrics.downside_volatility.toFixed(2)}%`}
                subtext="Negative returns only"
                valueColor={getPercentColor(metrics.downside_volatility, false)}
              />
              <MetricDisplay
                label="Sortino Ratio"
                value={metrics.sortino}
                subtext="Return / downside vol"
                valueColor={getRatioColor(metrics.sortino, { good: 2.0, bad: 0.5 })}
              />
              <MetricDisplay
                label="Sharpe Ratio"
                value={metrics.sharpe}
                subtext="Return / total vol"
                valueColor={getRatioColor(metrics.sharpe, { good: 1.5, bad: 0.5 })}
              />
            </div>
          </TabsContent>
        </Tabs>

        {/* Summary Bar */}
        <div className="flex items-center justify-between mt-4 pt-3 border-t border-neutral-800">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <div
                className={cn(
                  "w-2 h-2 rounded-full",
                  metrics.sharpe >= 1.5
                    ? "bg-green-500"
                    : metrics.sharpe >= 0.5
                    ? "bg-yellow-500"
                    : "bg-red-500"
                )}
              />
              <span className="text-xs text-neutral-500">
                {metrics.sharpe >= 1.5
                  ? "Strong"
                  : metrics.sharpe >= 0.5
                  ? "Moderate"
                  : "Weak"}{" "}
                risk-adjusted returns
              </span>
            </div>
          </div>
          <span className="text-[10px] text-neutral-600 font-mono">
            {metrics.total_trades} trades analyzed
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
