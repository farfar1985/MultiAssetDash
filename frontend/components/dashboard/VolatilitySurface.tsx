"use client";

import { useState, useMemo, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  Thermometer,
} from "lucide-react";

interface VolatilityData {
  asset: string;
  symbol: string;
  impliedVol: number; // IV percentage
  historicalVol: number; // HV percentage
  ivRank: number; // 0-100 percentile
  ivPercentile: number; // 0-100
  term: string; // e.g., "30D", "60D"
}

interface VolatilitySurfaceProps {
  data?: VolatilityData[];
  selectedAsset?: string;
  showMiniChart?: boolean;
  compact?: boolean;
}

const DEFAULT_VOL_DATA: VolatilityData[] = [
  { asset: "Crude Oil", symbol: "CL", impliedVol: 28.5, historicalVol: 24.2, ivRank: 65, ivPercentile: 72, term: "30D" },
  { asset: "Gold", symbol: "GC", impliedVol: 14.8, historicalVol: 12.1, ivRank: 42, ivPercentile: 38, term: "30D" },
  { asset: "Bitcoin", symbol: "BTC", impliedVol: 58.2, historicalVol: 52.8, ivRank: 78, ivPercentile: 82, term: "30D" },
  { asset: "S&P 500", symbol: "SPY", impliedVol: 15.2, historicalVol: 13.8, ivRank: 35, ivPercentile: 28, term: "30D" },
  { asset: "Natural Gas", symbol: "NG", impliedVol: 82.4, historicalVol: 78.1, ivRank: 88, ivPercentile: 91, term: "30D" },
  { asset: "EUR/USD", symbol: "EUR", impliedVol: 8.2, historicalVol: 7.5, ivRank: 22, ivPercentile: 18, term: "30D" },
];

function getVolColor(value: number): string {
  if (value >= 80) return "text-red-400";
  if (value >= 60) return "text-orange-400";
  if (value >= 40) return "text-amber-400";
  if (value >= 20) return "text-green-400";
  return "text-blue-400";
}

function getVolBgColor(value: number): string {
  if (value >= 80) return "bg-red-500/10 border-red-500/30";
  if (value >= 60) return "bg-orange-500/10 border-orange-500/30";
  if (value >= 40) return "bg-amber-500/10 border-amber-500/30";
  if (value >= 20) return "bg-green-500/10 border-green-500/30";
  return "bg-blue-500/10 border-blue-500/30";
}

function VolatilityBar({
  value,
  maxValue = 100,
  height = 8,
  showLabel = true,
  animated = true,
}: {
  value: number;
  maxValue?: number;
  height?: number;
  showLabel?: boolean;
  animated?: boolean;
}) {
  const [animatedWidth, setAnimatedWidth] = useState(animated ? 0 : value);

  useEffect(() => {
    if (animated) {
      const timer = setTimeout(() => setAnimatedWidth(value), 100);
      return () => clearTimeout(timer);
    }
  }, [value, animated]);

  const percentage = Math.min(100, (animatedWidth / maxValue) * 100);

  // Color segments
  const getSegmentColor = (segment: number): string => {
    if (segment <= 20) return "bg-blue-500";
    if (segment <= 40) return "bg-green-500";
    if (segment <= 60) return "bg-amber-500";
    if (segment <= 80) return "bg-orange-500";
    return "bg-red-500";
  };

  return (
    <div className="flex items-center gap-2">
      <div
        className="flex-1 bg-neutral-800 rounded-full overflow-hidden"
        style={{ height }}
      >
        <div
          className={cn(
            "h-full rounded-full transition-all duration-1000 ease-out",
            getSegmentColor(value)
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && (
        <span className={cn("text-xs font-mono font-bold min-w-[36px]", getVolColor(value))}>
          {value.toFixed(0)}%
        </span>
      )}
    </div>
  );
}

function IVHVComparison({
  iv,
  hv,
}: {
  iv: number;
  hv: number;
}) {
  const spread = iv - hv;
  const isExpensive = spread > 3;
  const isCheap = spread < -3;

  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 space-y-1">
        <div className="flex items-center justify-between text-xs">
          <span className="text-neutral-500">IV</span>
          <span className="font-mono text-purple-400">{iv.toFixed(1)}%</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-neutral-500">HV</span>
          <span className="font-mono text-cyan-400">{hv.toFixed(1)}%</span>
        </div>
      </div>
      <div
        className={cn(
          "px-2 py-1 rounded text-[10px] font-bold",
          isExpensive ? "bg-red-500/20 text-red-400" :
          isCheap ? "bg-green-500/20 text-green-400" :
          "bg-neutral-700/50 text-neutral-400"
        )}
      >
        {isExpensive ? (
          <div className="flex items-center gap-1">
            <TrendingUp className="w-3 h-3" />
            RICH
          </div>
        ) : isCheap ? (
          <div className="flex items-center gap-1">
            <TrendingDown className="w-3 h-3" />
            CHEAP
          </div>
        ) : (
          <div className="flex items-center gap-1">
            <Minus className="w-3 h-3" />
            FAIR
          </div>
        )}
      </div>
    </div>
  );
}

function MiniVolChart({ data }: { data: VolatilityData }) {
  // Simulate historical IV data points
  const chartPoints = useMemo(() => {
    const points = [];
    const baseIV = data.impliedVol;
    for (let i = 0; i < 20; i++) {
      const variation = (Math.random() - 0.5) * baseIV * 0.3;
      points.push(baseIV + variation);
    }
    points.push(baseIV); // Current point
    return points;
  }, [data.impliedVol]);

  const min = Math.min(...chartPoints) * 0.9;
  const max = Math.max(...chartPoints) * 1.1;
  const range = max - min;

  const pathData = chartPoints.map((point, i) => {
    const x = (i / (chartPoints.length - 1)) * 80;
    const y = 30 - ((point - min) / range) * 30;
    return `${i === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");

  return (
    <svg width="80" height="30" className="opacity-60">
      <defs>
        <linearGradient id="volGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path
        d={pathData + ` L 80 30 L 0 30 Z`}
        fill="url(#volGradient)"
      />
      <path
        d={pathData}
        fill="none"
        stroke="#8b5cf6"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Current point */}
      <circle
        cx="80"
        cy={30 - ((chartPoints[chartPoints.length - 1] - min) / range) * 30}
        r="2"
        fill="#8b5cf6"
      />
    </svg>
  );
}

function VolatilityRow({
  data,
  isSelected,
  showChart,
  onSelect,
}: {
  data: VolatilityData;
  isSelected?: boolean;
  showChart?: boolean;
  onSelect?: () => void;
}) {
  return (
    <div
      onClick={onSelect}
      className={cn(
        "grid grid-cols-[80px_1fr_100px_80px] items-center gap-3 p-3 rounded-lg border transition-all",
        "hover:bg-neutral-800/30 cursor-pointer",
        isSelected
          ? "bg-purple-500/10 border-purple-500/30"
          : "bg-neutral-800/20 border-neutral-700/50"
      )}
    >
      {/* Asset */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-mono font-bold text-neutral-200">
          {data.symbol}
        </span>
      </div>

      {/* IV Rank Bar */}
      <div>
        <VolatilityBar value={data.ivRank} height={6} showLabel={false} />
      </div>

      {/* IV/HV Comparison */}
      <IVHVComparison iv={data.impliedVol} hv={data.historicalVol} />

      {/* Mini Chart or IV Rank */}
      <div className="flex items-center justify-end">
        {showChart ? (
          <MiniVolChart data={data} />
        ) : (
          <div className={cn("px-2 py-1 rounded border", getVolBgColor(data.ivRank))}>
            <span className={cn("text-xs font-mono font-bold", getVolColor(data.ivRank))}>
              IVR {data.ivRank}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export function VolatilitySurface({
  data = DEFAULT_VOL_DATA,
  selectedAsset,
  showMiniChart = true,
  compact = false,
}: VolatilitySurfaceProps) {
  const [selected, setSelected] = useState<string | null>(selectedAsset || null);
  const [sortBy, setSortBy] = useState<"ivRank" | "iv" | "symbol">("ivRank");

  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => {
      switch (sortBy) {
        case "ivRank":
          return b.ivRank - a.ivRank;
        case "iv":
          return b.impliedVol - a.impliedVol;
        case "symbol":
          return a.symbol.localeCompare(b.symbol);
        default:
          return 0;
      }
    });
  }, [data, sortBy]);

  // Summary stats
  const avgIVRank = data.reduce((sum, d) => sum + d.ivRank, 0) / data.length;
  const highVolAssets = data.filter(d => d.ivRank >= 70).length;

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-500/10 rounded-lg border border-purple-500/20">
              <Thermometer className="w-4 h-4 text-purple-400" />
            </div>
            <div>
              <CardTitle className="text-sm font-semibold text-neutral-200">
                Volatility Surface
              </CardTitle>
              <span className="text-[10px] text-neutral-500">
                IV Rank & Premium Analysis
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {highVolAssets > 0 && (
              <Badge className="bg-orange-500/10 border-orange-500/30 text-orange-400 text-[10px]">
                <AlertTriangle className="w-3 h-3 mr-1" />
                {highVolAssets} High IV
              </Badge>
            )}
            <Badge className="bg-purple-500/10 border-purple-500/30 text-purple-400 text-xs">
              Avg IVR: {avgIVRank.toFixed(0)}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0 space-y-3">
        {/* Header Row */}
        <div className="grid grid-cols-[80px_1fr_100px_80px] gap-3 px-3 py-1 text-[10px] text-neutral-500 uppercase tracking-wider">
          <button
            onClick={() => setSortBy("symbol")}
            className={cn("text-left", sortBy === "symbol" && "text-purple-400")}
          >
            Asset
          </button>
          <button
            onClick={() => setSortBy("ivRank")}
            className={cn("text-left", sortBy === "ivRank" && "text-purple-400")}
          >
            IV Rank (0-100)
          </button>
          <button
            onClick={() => setSortBy("iv")}
            className={cn("text-center", sortBy === "iv" && "text-purple-400")}
          >
            IV vs HV
          </button>
          <span className="text-right">
            {showMiniChart ? "Trend" : "IVR"}
          </span>
        </div>

        {/* Data Rows */}
        <div className="space-y-2">
          {sortedData.map((item) => (
            <VolatilityRow
              key={item.symbol}
              data={item}
              isSelected={selected === item.symbol}
              showChart={showMiniChart}
              onSelect={() => setSelected(item.symbol === selected ? null : item.symbol)}
            />
          ))}
        </div>

        {/* Legend */}
        {!compact && (
          <div className="flex items-center justify-between pt-3 border-t border-neutral-800 text-[10px]">
            <div className="flex items-center gap-3">
              <span className="text-neutral-500">IV Rank:</span>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-blue-500/50" />
                <span className="text-blue-400">Low (0-20)</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-green-500/50" />
                <span className="text-green-400">Med (20-40)</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-amber-500/50" />
                <span className="text-amber-400">Elevated (40-60)</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-orange-500/50" />
                <span className="text-orange-400">High (60-80)</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-red-500/50" />
                <span className="text-red-400">Extreme (80+)</span>
              </div>
            </div>
            <div className="flex items-center gap-1.5">
              <Activity className="w-3 h-3 text-neutral-500" />
              <span className="text-neutral-500">30-day term structure</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
