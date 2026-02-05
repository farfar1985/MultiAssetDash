"use client";

import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { SignalGauge } from "./SignalGauge";
import {
  TrendingUp,
  TrendingDown,
  Target,
  AlertTriangle,
  Zap,
  Clock,
  Activity,
  ArrowRight,
  CheckCircle2,
  XCircle,
} from "lucide-react";

export interface HeroSignalData {
  assetName: string;
  symbol: string;
  currentPrice: number;
  priceChange24h: number;
  changePercent24h: number;
  // Signal data
  signalDirection: "bullish" | "bearish" | "neutral";
  signalStrength: number; // -100 to +100
  confidence: number; // 0-100
  // Targets
  entryPrice: number;
  stopLoss: number;
  targets: Array<{ level: number; price: number; probability: number }>;
  // Metrics
  sharpeRatio: number;
  winRate: number;
  avgHoldDays: number;
  modelsAgreeing: number;
  modelsTotal: number;
  // Timing
  signalAge: string;
  nextUpdate: string;
}

interface HeroSignalPanelProps {
  data: HeroSignalData;
  onTrade?: () => void;
}

function formatPrice(price: number): string {
  if (price >= 10000) {
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  if (price >= 100) {
    return `$${price.toFixed(2)}`;
  }
  return `$${price.toFixed(4)}`;
}

function PriceDisplay({
  currentPrice,
  priceChange24h,
  changePercent24h,
}: {
  currentPrice: number;
  priceChange24h: number;
  changePercent24h: number;
}) {
  const isPositive = changePercent24h >= 0;

  return (
    <div className="text-center lg:text-left">
      <div className="font-mono text-4xl lg:text-5xl font-bold text-neutral-100 tracking-tight">
        {formatPrice(currentPrice)}
      </div>
      <div
        className={cn(
          "flex items-center justify-center lg:justify-start gap-2 mt-2",
          isPositive ? "text-green-400" : "text-red-400"
        )}
      >
        {isPositive ? (
          <TrendingUp className="w-5 h-5" />
        ) : (
          <TrendingDown className="w-5 h-5" />
        )}
        <span className="font-mono text-lg">
          {isPositive ? "+" : ""}
          {formatPrice(priceChange24h)}
        </span>
        <span className="font-mono text-lg">
          ({isPositive ? "+" : ""}
          {changePercent24h.toFixed(2)}%)
        </span>
      </div>
    </div>
  );
}

function TargetLadderDisplay({
  entryPrice,
  stopLoss,
  targets,
  currentPrice,
  direction,
}: {
  entryPrice: number;
  stopLoss: number;
  targets: HeroSignalData["targets"];
  currentPrice: number;
  direction: HeroSignalData["signalDirection"];
}) {
  const isBullish = direction === "bullish";
  const riskReward = targets.length > 0
    ? Math.abs(targets[0].price - entryPrice) / Math.abs(entryPrice - stopLoss)
    : 0;

  return (
    <div className="space-y-3">
      {/* Entry */}
      <div className="flex items-center justify-between p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
        <div className="flex items-center gap-2">
          <ArrowRight className="w-4 h-4 text-blue-400" />
          <span className="text-sm text-blue-400 font-medium">Entry Zone</span>
        </div>
        <span className="font-mono text-sm font-bold text-blue-300">
          {formatPrice(entryPrice)}
        </span>
      </div>

      {/* Targets */}
      <div className="space-y-2">
        {targets.map((target, index) => {
          const isHit = isBullish
            ? currentPrice >= target.price
            : currentPrice <= target.price;
          const distancePercent = ((target.price - currentPrice) / currentPrice) * 100;

          return (
            <div
              key={index}
              className={cn(
                "flex items-center justify-between p-2.5 rounded-lg border transition-all",
                isHit
                  ? "bg-green-500/20 border-green-500/40"
                  : "bg-neutral-800/50 border-neutral-700/50"
              )}
            >
              <div className="flex items-center gap-2">
                {isHit ? (
                  <CheckCircle2 className="w-4 h-4 text-green-400" />
                ) : (
                  <Target className="w-4 h-4 text-neutral-500" />
                )}
                <span className={cn(
                  "text-sm font-medium",
                  isHit ? "text-green-400" : "text-neutral-400"
                )}>
                  T{target.level}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-neutral-500 font-mono">
                  {distancePercent >= 0 ? "+" : ""}{distancePercent.toFixed(1)}%
                </span>
                <span className={cn(
                  "font-mono text-sm font-bold",
                  isHit ? "text-green-300" : "text-neutral-300"
                )}>
                  {formatPrice(target.price)}
                </span>
                <div className="w-12 h-1.5 bg-neutral-700 rounded-full overflow-hidden">
                  <div
                    className={cn(
                      "h-full rounded-full",
                      isHit ? "bg-green-500" : "bg-blue-500"
                    )}
                    style={{ width: `${target.probability}%` }}
                  />
                </div>
                <span className="text-xs text-neutral-500 font-mono w-8">
                  {target.probability}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Stop Loss */}
      <div className="flex items-center justify-between p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
        <div className="flex items-center gap-2">
          <XCircle className="w-4 h-4 text-red-400" />
          <span className="text-sm text-red-400 font-medium">Stop Loss</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-red-400 font-mono">
            {((stopLoss - currentPrice) / currentPrice * 100).toFixed(1)}%
          </span>
          <span className="font-mono text-sm font-bold text-red-300">
            {formatPrice(stopLoss)}
          </span>
        </div>
      </div>

      {/* Risk/Reward */}
      <div className="flex items-center justify-between px-3 py-2 bg-neutral-800/30 rounded-lg">
        <span className="text-xs text-neutral-500">Risk/Reward Ratio</span>
        <span className={cn(
          "font-mono text-sm font-bold",
          riskReward >= 2 ? "text-green-400" :
          riskReward >= 1.5 ? "text-blue-400" :
          riskReward >= 1 ? "text-yellow-400" : "text-red-400"
        )}>
          1:{riskReward.toFixed(1)}
        </span>
      </div>
    </div>
  );
}

function MetricsRow({ data }: { data: HeroSignalData }) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {/* Sharpe Ratio */}
      <div className="p-3 bg-neutral-800/50 rounded-lg border border-neutral-700/50">
        <div className="flex items-center gap-2 mb-1">
          <Activity className="w-3.5 h-3.5 text-blue-400" />
          <span className="text-[10px] uppercase tracking-wider text-neutral-500">
            Sharpe Ratio
          </span>
        </div>
        <div className={cn(
          "font-mono text-xl font-bold",
          data.sharpeRatio >= 2 ? "text-green-400" :
          data.sharpeRatio >= 1 ? "text-blue-400" : "text-yellow-400"
        )}>
          {data.sharpeRatio.toFixed(2)}
        </div>
      </div>

      {/* Win Rate */}
      <div className="p-3 bg-neutral-800/50 rounded-lg border border-neutral-700/50">
        <div className="flex items-center gap-2 mb-1">
          <Target className="w-3.5 h-3.5 text-green-400" />
          <span className="text-[10px] uppercase tracking-wider text-neutral-500">
            Win Rate
          </span>
        </div>
        <div className={cn(
          "font-mono text-xl font-bold",
          data.winRate >= 60 ? "text-green-400" :
          data.winRate >= 50 ? "text-yellow-400" : "text-red-400"
        )}>
          {data.winRate.toFixed(1)}%
        </div>
      </div>

      {/* Avg Hold Time */}
      <div className="p-3 bg-neutral-800/50 rounded-lg border border-neutral-700/50">
        <div className="flex items-center gap-2 mb-1">
          <Clock className="w-3.5 h-3.5 text-purple-400" />
          <span className="text-[10px] uppercase tracking-wider text-neutral-500">
            Avg Hold
          </span>
        </div>
        <div className="font-mono text-xl font-bold text-purple-400">
          {data.avgHoldDays.toFixed(1)}d
        </div>
      </div>

      {/* Model Consensus */}
      <div className="p-3 bg-neutral-800/50 rounded-lg border border-neutral-700/50">
        <div className="flex items-center gap-2 mb-1">
          <Zap className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-[10px] uppercase tracking-wider text-neutral-500">
            Consensus
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xl font-bold text-cyan-400">
            {data.modelsAgreeing.toLocaleString()}
          </span>
          <span className="text-xs text-neutral-500 font-mono">
            /{data.modelsTotal.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
}

export function HeroSignalPanel({ data, onTrade }: HeroSignalPanelProps) {
  const [isAnimating, setIsAnimating] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setIsAnimating(false), 2000);
    return () => clearTimeout(timer);
  }, []);

  const isBullish = data.signalDirection === "bullish";
  const directionGradient = isBullish
    ? "from-green-900/30 via-emerald-900/20 to-green-900/30"
    : data.signalDirection === "bearish"
    ? "from-red-900/30 via-rose-900/20 to-red-900/30"
    : "from-amber-900/30 via-yellow-900/20 to-amber-900/30";

  const directionBorder = isBullish
    ? "border-green-500/30"
    : data.signalDirection === "bearish"
    ? "border-red-500/30"
    : "border-amber-500/30";

  return (
    <Card
      className={cn(
        "bg-gradient-to-r border-2 overflow-hidden transition-all duration-500",
        directionGradient,
        directionBorder,
        isAnimating && "animate-pulse"
      )}
    >
      <CardContent className="p-6">
        {/* Header Row */}
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 mb-6">
          {/* Asset Info */}
          <div className="flex items-center gap-4">
            <div className="p-3 bg-neutral-800/50 rounded-xl border border-neutral-700/50">
              <span className="font-mono text-2xl font-bold text-neutral-100">
                {data.symbol}
              </span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-neutral-100">{data.assetName}</h1>
              <div className="flex items-center gap-2 mt-1">
                <Badge
                  className={cn(
                    "text-sm font-bold px-3 py-1 gap-1.5",
                    isBullish
                      ? "bg-green-500/20 border-green-500/40 text-green-400"
                      : data.signalDirection === "bearish"
                      ? "bg-red-500/20 border-red-500/40 text-red-400"
                      : "bg-amber-500/20 border-amber-500/40 text-amber-400"
                  )}
                >
                  {isBullish ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : data.signalDirection === "bearish" ? (
                    <TrendingDown className="w-4 h-4" />
                  ) : (
                    <AlertTriangle className="w-4 h-4" />
                  )}
                  {data.signalDirection.toUpperCase()}
                </Badge>
                <span className="text-xs text-neutral-500">
                  Signal age: {data.signalAge}
                </span>
              </div>
            </div>
          </div>

          {/* Price Display */}
          <PriceDisplay
            currentPrice={data.currentPrice}
            priceChange24h={data.priceChange24h}
            changePercent24h={data.changePercent24h}
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Signal Gauge */}
          <div className="flex items-center justify-center p-4 bg-neutral-900/50 rounded-xl border border-neutral-800">
            <SignalGauge
              value={data.signalStrength}
              confidence={data.confidence}
              size="lg"
              animated={true}
              showPercentileRing={true}
              percentile={data.confidence}
              label="Ensemble Signal Strength"
            />
          </div>

          {/* Target Ladder */}
          <div className="lg:col-span-2 p-4 bg-neutral-900/50 rounded-xl border border-neutral-800">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                <Target className="w-4 h-4 text-blue-400" />
                Price Targets
              </h3>
              <span className="text-xs text-neutral-500">
                Updated: {data.nextUpdate}
              </span>
            </div>
            <TargetLadderDisplay
              entryPrice={data.entryPrice}
              stopLoss={data.stopLoss}
              targets={data.targets}
              currentPrice={data.currentPrice}
              direction={data.signalDirection}
            />
          </div>
        </div>

        {/* Metrics Row */}
        <div className="mt-6">
          <MetricsRow data={data} />
        </div>

        {/* Action Button */}
        {onTrade && (
          <div className="mt-6 flex justify-center">
            <button
              onClick={onTrade}
              className={cn(
                "px-8 py-3 rounded-xl font-bold text-lg transition-all duration-200",
                "flex items-center gap-2 shadow-lg",
                isBullish
                  ? "bg-green-500 hover:bg-green-400 text-black shadow-green-500/25"
                  : "bg-red-500 hover:bg-red-400 text-white shadow-red-500/25"
              )}
            >
              <Zap className="w-5 h-5" />
              Execute {isBullish ? "Long" : "Short"} Position
            </button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Generate mock hero signal data
export function generateMockHeroSignal(
  symbol: string = "CL",
  assetName: string = "Crude Oil",
  direction: "bullish" | "bearish" = "bullish"
): HeroSignalData {
  const basePrice = symbol === "BTC" ? 45000 : symbol === "GOLD" ? 2050 : 73.50;
  const priceChange = (Math.random() - 0.5) * basePrice * 0.03;

  return {
    assetName,
    symbol,
    currentPrice: basePrice,
    priceChange24h: priceChange,
    changePercent24h: (priceChange / basePrice) * 100,
    signalDirection: direction,
    signalStrength: direction === "bullish"
      ? 40 + Math.random() * 50
      : -40 - Math.random() * 50,
    confidence: 65 + Math.random() * 25,
    entryPrice: basePrice * (direction === "bullish" ? 0.995 : 1.005),
    stopLoss: basePrice * (direction === "bullish" ? 0.975 : 1.025),
    targets: [
      { level: 1, price: basePrice * (direction === "bullish" ? 1.015 : 0.985), probability: 72 },
      { level: 2, price: basePrice * (direction === "bullish" ? 1.030 : 0.970), probability: 58 },
      { level: 3, price: basePrice * (direction === "bullish" ? 1.050 : 0.950), probability: 38 },
    ],
    sharpeRatio: 1.8 + Math.random() * 0.8,
    winRate: 55 + Math.random() * 15,
    avgHoldDays: 3 + Math.random() * 4,
    modelsAgreeing: Math.floor(7500 + Math.random() * 2500),
    modelsTotal: 10179,
    signalAge: "2h 34m",
    nextUpdate: "in 45m",
  };
}
