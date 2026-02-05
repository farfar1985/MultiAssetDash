"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  TrendingUp,
  TrendingDown,
  CheckCircle2,
  XCircle,
  Clock,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

export interface TradeRecord {
  id: string;
  date: string;
  asset: string;
  symbol: string;
  direction: "long" | "short";
  entryPrice: number;
  exitPrice: number;
  pnlPercent: number;
  pnlAbsolute: number;
  holdDays: number;
  targetsHit: number[];
  totalTargets: number;
  signalStrength: number;
  status: "win" | "loss" | "breakeven";
}

interface TradeHistoryTableProps {
  trades: TradeRecord[];
  maxRows?: number;
  onTradeClick?: (trade: TradeRecord) => void;
  showFilters?: boolean;
}

type SortField = "date" | "pnlPercent" | "holdDays" | "signalStrength";
type SortDirection = "asc" | "desc";

function formatPrice(price: number): string {
  if (price >= 10000) return `$${price.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  if (price >= 100) return `$${price.toFixed(2)}`;
  return `$${price.toFixed(4)}`;
}

function MiniSparkline({ values, width = 60, height = 20 }: {
  values: number[];
  width?: number;
  height?: number;
}) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x},${y}`;
  }).join(" ");

  const isPositive = values[values.length - 1] >= values[0];

  return (
    <svg width={width} height={height} className="inline-block">
      <polyline
        points={points}
        fill="none"
        stroke={isPositive ? "#22c55e" : "#ef4444"}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function TargetsIndicator({ hit, total }: { hit: number[]; total: number }) {
  return (
    <div className="flex items-center gap-0.5">
      {Array.from({ length: total }).map((_, i) => {
        const isHit = hit.includes(i + 1);
        return (
          <div
            key={i}
            className={cn(
              "w-2 h-5 rounded-sm transition-colors",
              isHit ? "bg-green-500" : "bg-neutral-700"
            )}
            title={`T${i + 1}: ${isHit ? "Hit" : "Missed"}`}
          />
        );
      })}
    </div>
  );
}

export function TradeHistoryTable({
  trades,
  maxRows = 10,
  onTradeClick,
  showFilters = true,
}: TradeHistoryTableProps) {
  const [sortField, setSortField] = useState<SortField>("date");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [filterStatus, setFilterStatus] = useState<"all" | "win" | "loss">("all");
  const [expanded, setExpanded] = useState(false);

  const sortedTrades = useMemo(() => {
    let filtered = [...trades];

    // Apply filter
    if (filterStatus !== "all") {
      filtered = filtered.filter((t) => t.status === filterStatus);
    }

    // Apply sort
    filtered.sort((a, b) => {
      let aVal: number, bVal: number;

      switch (sortField) {
        case "date":
          aVal = new Date(a.date).getTime();
          bVal = new Date(b.date).getTime();
          break;
        case "pnlPercent":
          aVal = a.pnlPercent;
          bVal = b.pnlPercent;
          break;
        case "holdDays":
          aVal = a.holdDays;
          bVal = b.holdDays;
          break;
        case "signalStrength":
          aVal = a.signalStrength;
          bVal = b.signalStrength;
          break;
        default:
          return 0;
      }

      return sortDirection === "asc" ? aVal - bVal : bVal - aVal;
    });

    return filtered;
  }, [trades, sortField, sortDirection, filterStatus]);

  const displayedTrades = expanded ? sortedTrades : sortedTrades.slice(0, maxRows);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  // Calculate summary stats
  const stats = useMemo(() => {
    const wins = trades.filter((t) => t.status === "win").length;
    const totalPnl = trades.reduce((acc, t) => acc + t.pnlPercent, 0);
    const avgHold = trades.reduce((acc, t) => acc + t.holdDays, 0) / trades.length;

    return {
      winRate: (wins / trades.length) * 100,
      totalPnl,
      avgHold,
      totalTrades: trades.length,
    };
  }, [trades]);

  const SortButton = ({ field, label }: { field: SortField; label: string }) => (
    <button
      onClick={() => handleSort(field)}
      className={cn(
        "flex items-center gap-1 text-xs uppercase tracking-wider transition-colors",
        sortField === field ? "text-blue-400" : "text-neutral-500 hover:text-neutral-300"
      )}
    >
      {label}
      {sortField === field && (
        sortDirection === "asc" ? (
          <ChevronUp className="w-3 h-3" />
        ) : (
          <ChevronDown className="w-3 h-3" />
        )
      )}
    </button>
  );

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CardTitle className="text-sm font-semibold text-neutral-200">
              Trade History
            </CardTitle>
            <Badge className="bg-neutral-800 border-neutral-700 text-neutral-400 text-xs">
              {trades.length} trades
            </Badge>
          </div>

          {showFilters && (
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1 p-1 bg-neutral-800/50 rounded-lg">
                <button
                  onClick={() => setFilterStatus("all")}
                  className={cn(
                    "px-2 py-1 text-xs rounded transition-colors",
                    filterStatus === "all"
                      ? "bg-neutral-700 text-neutral-100"
                      : "text-neutral-500 hover:text-neutral-300"
                  )}
                >
                  All
                </button>
                <button
                  onClick={() => setFilterStatus("win")}
                  className={cn(
                    "px-2 py-1 text-xs rounded transition-colors",
                    filterStatus === "win"
                      ? "bg-green-500/20 text-green-400"
                      : "text-neutral-500 hover:text-neutral-300"
                  )}
                >
                  Wins
                </button>
                <button
                  onClick={() => setFilterStatus("loss")}
                  className={cn(
                    "px-2 py-1 text-xs rounded transition-colors",
                    filterStatus === "loss"
                      ? "bg-red-500/20 text-red-400"
                      : "text-neutral-500 hover:text-neutral-300"
                  )}
                >
                  Losses
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-4 gap-3 mt-4">
          <div className="p-2 bg-neutral-800/30 rounded-lg text-center">
            <div className="text-[10px] uppercase tracking-wider text-neutral-500">Win Rate</div>
            <div className={cn(
              "font-mono text-sm font-bold",
              stats.winRate >= 60 ? "text-green-400" :
              stats.winRate >= 50 ? "text-yellow-400" : "text-red-400"
            )}>
              {stats.winRate.toFixed(1)}%
            </div>
          </div>
          <div className="p-2 bg-neutral-800/30 rounded-lg text-center">
            <div className="text-[10px] uppercase tracking-wider text-neutral-500">Total P&L</div>
            <div className={cn(
              "font-mono text-sm font-bold",
              stats.totalPnl >= 0 ? "text-green-400" : "text-red-400"
            )}>
              {stats.totalPnl >= 0 ? "+" : ""}{stats.totalPnl.toFixed(1)}%
            </div>
          </div>
          <div className="p-2 bg-neutral-800/30 rounded-lg text-center">
            <div className="text-[10px] uppercase tracking-wider text-neutral-500">Avg Hold</div>
            <div className="font-mono text-sm font-bold text-blue-400">
              {stats.avgHold.toFixed(1)}d
            </div>
          </div>
          <div className="p-2 bg-neutral-800/30 rounded-lg text-center">
            <div className="text-[10px] uppercase tracking-wider text-neutral-500">Trades</div>
            <div className="font-mono text-sm font-bold text-neutral-300">
              {stats.totalTrades}
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        {/* Table Header */}
        <div className="grid grid-cols-[1fr_80px_100px_100px_80px_60px_80px] gap-2 px-4 py-2 border-b border-neutral-800 bg-neutral-800/30">
          <SortButton field="date" label="Date / Asset" />
          <span className="text-xs uppercase tracking-wider text-neutral-500">Direction</span>
          <span className="text-xs uppercase tracking-wider text-neutral-500">Entry/Exit</span>
          <SortButton field="pnlPercent" label="P&L" />
          <SortButton field="holdDays" label="Hold" />
          <span className="text-xs uppercase tracking-wider text-neutral-500">Targets</span>
          <span className="text-xs uppercase tracking-wider text-neutral-500">Trend</span>
        </div>

        {/* Table Body */}
        <div className="divide-y divide-neutral-800/50">
          {displayedTrades.map((trade) => {
            const isWin = trade.status === "win";
            const isLong = trade.direction === "long";

            // Generate fake sparkline data
            const sparklineData = Array.from({ length: 10 }, (_, i) => {
              const base = trade.entryPrice;
              const end = trade.exitPrice;
              const progress = i / 9;
              return base + (end - base) * progress + (Math.random() - 0.5) * (end - base) * 0.2;
            });

            return (
              <div
                key={trade.id}
                onClick={() => onTradeClick?.(trade)}
                className={cn(
                  "grid grid-cols-[1fr_80px_100px_100px_80px_60px_80px] gap-2 px-4 py-3",
                  "hover:bg-neutral-800/30 transition-colors",
                  onTradeClick && "cursor-pointer"
                )}
              >
                {/* Date / Asset */}
                <div className="flex flex-col">
                  <span className="text-xs text-neutral-500 font-mono">{trade.date}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-neutral-200">{trade.asset}</span>
                    <span className="text-xs text-neutral-600 font-mono">{trade.symbol}</span>
                  </div>
                </div>

                {/* Direction */}
                <div className="flex items-center">
                  <Badge
                    className={cn(
                      "text-xs font-medium gap-1",
                      isLong
                        ? "bg-green-500/10 border-green-500/30 text-green-500"
                        : "bg-red-500/10 border-red-500/30 text-red-500"
                    )}
                  >
                    {isLong ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                    {trade.direction.toUpperCase()}
                  </Badge>
                </div>

                {/* Entry/Exit */}
                <div className="flex flex-col text-xs font-mono">
                  <span className="text-neutral-400">{formatPrice(trade.entryPrice)}</span>
                  <span className="text-neutral-300">{formatPrice(trade.exitPrice)}</span>
                </div>

                {/* P&L */}
                <div className="flex items-center gap-2">
                  {isWin ? (
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-500" />
                  )}
                  <div className="flex flex-col">
                    <span className={cn(
                      "text-sm font-mono font-bold",
                      isWin ? "text-green-400" : "text-red-400"
                    )}>
                      {trade.pnlPercent >= 0 ? "+" : ""}{trade.pnlPercent.toFixed(2)}%
                    </span>
                    <span className="text-xs text-neutral-600 font-mono">
                      {trade.pnlAbsolute >= 0 ? "+" : ""}{formatPrice(trade.pnlAbsolute)}
                    </span>
                  </div>
                </div>

                {/* Hold Days */}
                <div className="flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-neutral-600" />
                  <span className="text-sm font-mono text-neutral-400">{trade.holdDays}d</span>
                </div>

                {/* Targets */}
                <TargetsIndicator hit={trade.targetsHit} total={trade.totalTargets} />

                {/* Sparkline */}
                <div className="flex items-center">
                  <MiniSparkline values={sparklineData} />
                </div>
              </div>
            );
          })}
        </div>

        {/* Show More / Less */}
        {sortedTrades.length > maxRows && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full py-3 text-sm text-neutral-500 hover:text-neutral-300 transition-colors border-t border-neutral-800 flex items-center justify-center gap-2"
          >
            {expanded ? (
              <>
                <ChevronUp className="w-4 h-4" />
                Show Less
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4" />
                Show {sortedTrades.length - maxRows} More
              </>
            )}
          </button>
        )}
      </CardContent>
    </Card>
  );
}

// Generate mock trade data
export function generateMockTrades(count: number = 20): TradeRecord[] {
  const assets = [
    { name: "Crude Oil", symbol: "CL" },
    { name: "Gold", symbol: "GC" },
    { name: "Bitcoin", symbol: "BTC" },
    { name: "S&P 500", symbol: "SPY" },
  ];

  return Array.from({ length: count }, (_, i) => {
    const asset = assets[Math.floor(Math.random() * assets.length)];
    const isLong = Math.random() > 0.4;
    const isWin = Math.random() > 0.35;
    const basePrice = asset.symbol === "BTC" ? 45000 : asset.symbol === "GC" ? 2050 : asset.symbol === "SPY" ? 480 : 73;

    const entryPrice = basePrice * (0.95 + Math.random() * 0.1);
    const pnlPercent = isWin
      ? 1 + Math.random() * 5
      : -(0.5 + Math.random() * 2.5);
    const exitPrice = entryPrice * (1 + pnlPercent / 100);

    const date = new Date();
    date.setDate(date.getDate() - i - Math.floor(Math.random() * 5));

    const totalTargets = 3;
    const targetsHit = isWin
      ? Array.from({ length: Math.floor(Math.random() * 3) + 1 }, (_, j) => j + 1)
      : [];

    return {
      id: `trade-${i}`,
      date: date.toISOString().split("T")[0],
      asset: asset.name,
      symbol: asset.symbol,
      direction: isLong ? "long" : "short",
      entryPrice,
      exitPrice,
      pnlPercent,
      pnlAbsolute: (pnlPercent / 100) * 10000,
      holdDays: Math.floor(Math.random() * 10) + 1,
      targetsHit,
      totalTargets,
      signalStrength: 50 + Math.random() * 45,
      status: isWin ? "win" : "loss",
    };
  });
}
