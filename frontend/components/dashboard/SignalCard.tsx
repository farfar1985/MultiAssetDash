"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { useAsset, useSignal } from "@/hooks/useApi";
import type { Horizon } from "@/lib/api-client";
import type { AssetId, SignalDirection } from "@/types";

interface SignalCardProps {
  assetId: AssetId;
  defaultHorizon?: Horizon;
  onClick?: () => void;
}

function getDirectionColor(direction: SignalDirection): string {
  switch (direction) {
    case "bullish":
      return "text-green-500";
    case "bearish":
      return "text-red-500";
    case "neutral":
      return "text-yellow-500";
  }
}

function getDirectionBgColor(direction: SignalDirection): string {
  switch (direction) {
    case "bullish":
      return "bg-green-500/10 border-green-500/30 text-green-500";
    case "bearish":
      return "bg-red-500/10 border-red-500/30 text-red-500";
    case "neutral":
      return "bg-yellow-500/10 border-yellow-500/30 text-yellow-500";
  }
}

function getDirectionIcon(direction: SignalDirection): string {
  switch (direction) {
    case "bullish":
      return "↑";
    case "bearish":
      return "↓";
    case "neutral":
      return "→";
  }
}

function getConfidenceBarColor(confidence: number): string {
  if (confidence >= 80) return "bg-green-500";
  if (confidence >= 60) return "bg-blue-500";
  if (confidence >= 40) return "bg-yellow-500";
  return "bg-red-500";
}

function formatPrice(price: number, symbol: string): string {
  if (symbol === "BTC") {
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  if (price >= 100) {
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  return `$${price.toFixed(2)}`;
}

function formatModelAgreement(agreeing: number, total: number): string {
  return `${agreeing.toLocaleString()} / ${total.toLocaleString()} models agree`;
}

function SignalCardSkeleton() {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Skeleton className="h-5 w-10 bg-neutral-800" />
            <Skeleton className="h-4 w-24 bg-neutral-800" />
          </div>
          <div className="text-right">
            <Skeleton className="h-4 w-20 bg-neutral-800 mb-1" />
            <Skeleton className="h-3 w-12 bg-neutral-800" />
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-7 w-24 bg-neutral-800" />
          <Skeleton className="h-7 w-20 bg-neutral-800" />
        </div>
        <div>
          <Skeleton className="h-2 w-full bg-neutral-800 rounded-full" />
        </div>
        <Skeleton className="h-4 w-40 bg-neutral-800" />
        <div className="grid grid-cols-3 gap-3 pt-2 border-t border-neutral-800">
          <Skeleton className="h-10 bg-neutral-800" />
          <Skeleton className="h-10 bg-neutral-800" />
          <Skeleton className="h-10 bg-neutral-800" />
        </div>
      </CardContent>
    </Card>
  );
}

function SignalCardError({ message }: { message: string }) {
  return (
    <Card className="bg-neutral-900/50 border-red-900/50">
      <CardContent className="p-4">
        <div className="text-center text-red-500 text-sm py-4">
          <p>Failed to load signal</p>
          <p className="text-xs text-neutral-500 mt-1">{message}</p>
        </div>
      </CardContent>
    </Card>
  );
}

export function SignalCard({
  assetId,
  defaultHorizon = "D+1",
  onClick,
}: SignalCardProps) {
  const [horizon, setHorizon] = useState<Horizon>(defaultHorizon);

  const {
    data: asset,
    isLoading: isAssetLoading,
    error: assetError,
  } = useAsset(assetId);

  const {
    data: signal,
    isLoading: isSignalLoading,
    error: signalError,
  } = useSignal(assetId, horizon);

  const isLoading = isAssetLoading || isSignalLoading;
  const error = assetError || signalError;

  if (isLoading) {
    return <SignalCardSkeleton />;
  }

  if (error) {
    return <SignalCardError message={error.message} />;
  }

  if (!asset || !signal) {
    return null;
  }

  const priceChangeColor = asset.changePercent24h >= 0 ? "text-green-500" : "text-red-500";

  return (
    <Card
      className={cn(
        "bg-neutral-900/50 border-neutral-800 hover:border-neutral-700 transition-all duration-200",
        onClick && "cursor-pointer hover:bg-neutral-900/80"
      )}
      onClick={onClick}
    >
      <CardHeader className="p-4 pb-3">
        {/* Asset Name + Price Row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
              {asset.symbol}
            </span>
            <span className="text-sm font-semibold text-neutral-100">
              {asset.name}
            </span>
          </div>
          <div className="text-right">
            <div className="font-mono text-sm font-medium text-neutral-100">
              {formatPrice(asset.currentPrice, asset.symbol)}
            </div>
            <div className={cn("font-mono text-xs", priceChangeColor)}>
              {asset.changePercent24h >= 0 ? "+" : ""}
              {asset.changePercent24h.toFixed(2)}%
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0 space-y-4">
        {/* Signal Direction Badge */}
        <div className="flex items-center justify-between">
          <Badge
            className={cn(
              "text-sm font-semibold px-3 py-1 capitalize",
              getDirectionBgColor(signal.direction)
            )}
          >
            <span className="mr-1">{getDirectionIcon(signal.direction)}</span>
            {signal.direction}
          </Badge>

          {/* Horizon Selector */}
          <Select value={horizon} onValueChange={(v) => setHorizon(v as Horizon)}>
            <SelectTrigger className="w-20 h-7 text-xs bg-neutral-800 border-neutral-700">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-neutral-900 border-neutral-700">
              <SelectItem value="D+1" className="text-xs">D+1</SelectItem>
              <SelectItem value="D+5" className="text-xs">D+5</SelectItem>
              <SelectItem value="D+10" className="text-xs">D+10</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Confidence Bar */}
        <div>
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="text-neutral-500">Confidence</span>
            <span className={cn("font-mono font-medium", getDirectionColor(signal.direction))}>
              {signal.confidence}%
            </span>
          </div>
          <div className="h-2 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-300",
                getConfidenceBarColor(signal.confidence)
              )}
              style={{ width: `${signal.confidence}%` }}
            />
          </div>
        </div>

        {/* Model Agreement */}
        <div className="text-xs text-neutral-400 font-mono">
          {formatModelAgreement(signal.modelsAgreeing, signal.modelsTotal)}
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-3 gap-3 pt-2 border-t border-neutral-800">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-0.5">
              Sharpe
            </div>
            <div className="font-mono text-sm font-medium text-blue-400">
              {signal.sharpeRatio.toFixed(2)}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-0.5">
              DA%
            </div>
            <div className="font-mono text-sm font-medium text-neutral-100">
              {signal.directionalAccuracy.toFixed(1)}%
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-0.5">
              Return
            </div>
            <div className={cn(
              "font-mono text-sm font-medium",
              signal.totalReturn >= 0 ? "text-green-500" : "text-red-500"
            )}>
              {signal.totalReturn >= 0 ? "+" : ""}{signal.totalReturn.toFixed(1)}%
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
