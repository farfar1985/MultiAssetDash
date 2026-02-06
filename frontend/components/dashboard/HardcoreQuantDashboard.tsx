"use client";

import { useState, useMemo, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

// Reuse existing components
import { CorrelationMatrix } from "@/components/dashboard/CorrelationMatrix";
import { GreeksPanel } from "@/components/dashboard/GreeksPanel";
import { QuantumStatusWidget } from "@/components/dashboard/QuantumStatusWidget";
import { SignalGauge } from "@/components/dashboard/SignalGauge";
import { MarketTicker } from "@/components/dashboard/MarketTicker";
import { MarketStatusBar } from "@/components/dashboard/MarketStatusBar";
import { VolatilitySurface } from "@/components/dashboard/VolatilitySurface";
import { FactorAttributionPanel } from "@/components/dashboard/FactorAttributionPanel";
import { ApiHealthIndicator } from "@/components/dashboard/ApiHealthIndicator";

import {
  Activity,
  BarChart3,
  BookOpen,
  Boxes,
  Cpu,
  Layers,
  LineChart,
  Radio,
  Settings,
  TrendingDown,
  TrendingUp,
  Zap,
  Target,
  Gauge,
  Timer,
  Database,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface TickData {
  time: string;
  price: number;
  size: number;
  side: "buy" | "sell";
}

interface OrderBookLevel {
  price: number;
  size: number;
  total: number;
}

interface OrderBookData {
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  spread: number;
  midPrice: number;
}

interface TimeframeData {
  tf: string;
  signal: "long" | "short" | "neutral";
  strength: number;
  trend: "up" | "down" | "flat";
}

interface ExecutionMetrics {
  fillRate: number;
  avgSlippage: number;
  vwapDeviation: number;
  executionSpeed: number;
  ordersToday: number;
  volumeTraded: number;
}

// ============================================================================
// Mock Data Generators
// ============================================================================

function generateTickData(count: number): TickData[] {
  const basePrice = 73.45;
  const ticks: TickData[] = [];
  let price = basePrice;

  for (let i = 0; i < count; i++) {
    const change = (Math.random() - 0.5) * 0.08;
    price = Math.max(72.5, Math.min(74.5, price + change));
    ticks.push({
      time: new Date(Date.now() - (count - i) * 100).toISOString(),
      price: Number(price.toFixed(2)),
      size: Math.floor(Math.random() * 50) + 1,
      side: Math.random() > 0.5 ? "buy" : "sell",
    });
  }
  return ticks;
}

function generateOrderBook(): OrderBookData {
  const midPrice = 73.45;
  const spread = 0.02;

  const bids: OrderBookLevel[] = [];
  const asks: OrderBookLevel[] = [];

  let bidTotal = 0;
  let askTotal = 0;

  for (let i = 0; i < 15; i++) {
    const bidSize = Math.floor(Math.random() * 200) + 20;
    const askSize = Math.floor(Math.random() * 200) + 20;
    bidTotal += bidSize;
    askTotal += askSize;

    bids.push({
      price: Number((midPrice - spread / 2 - i * 0.01).toFixed(2)),
      size: bidSize,
      total: bidTotal,
    });
    asks.push({
      price: Number((midPrice + spread / 2 + i * 0.01).toFixed(2)),
      size: askSize,
      total: askTotal,
    });
  }

  return { bids, asks, spread, midPrice };
}

function generateTimeframeSignals(): TimeframeData[] {
  return [
    { tf: "1m", signal: "long", strength: 62, trend: "up" },
    { tf: "5m", signal: "long", strength: 71, trend: "up" },
    { tf: "15m", signal: "neutral", strength: 48, trend: "flat" },
    { tf: "1h", signal: "long", strength: 78, trend: "up" },
    { tf: "4h", signal: "long", strength: 84, trend: "up" },
    { tf: "1D", signal: "short", strength: 55, trend: "down" },
  ];
}

function generateExecutionMetrics(): ExecutionMetrics {
  return {
    fillRate: 98.7,
    avgSlippage: 0.012,
    vwapDeviation: -0.08,
    executionSpeed: 2.3,
    ordersToday: 847,
    volumeTraded: 12450000,
  };
}

// ============================================================================
// Tick Tape Component
// ============================================================================

function TickTape({ ticks }: { ticks: TickData[] }) {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-2 pb-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Radio className="w-3.5 h-3.5 text-green-400 animate-pulse" />
            <CardTitle className="text-xs font-semibold text-neutral-300">
              TICK STREAM
            </CardTitle>
          </div>
          <span className="text-[9px] text-neutral-500 font-mono">
            {ticks.length} ticks/s
          </span>
        </div>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        <div className="h-24 overflow-hidden font-mono text-[10px] space-y-0.5">
          {ticks.slice(-12).reverse().map((tick, i) => (
            <div
              key={i}
              className={cn(
                "flex items-center justify-between px-1 py-0.5 rounded",
                tick.side === "buy" ? "bg-green-500/5" : "bg-red-500/5"
              )}
            >
              <span className="text-neutral-500">
                {new Date(tick.time).toLocaleTimeString()}.{new Date(tick.time).getMilliseconds().toString().padStart(3, "0")}
              </span>
              <span className={tick.side === "buy" ? "text-green-400" : "text-red-400"}>
                {tick.price.toFixed(2)}
              </span>
              <span className="text-neutral-400">{tick.size}</span>
              <span className={cn(
                "w-8 text-center text-[9px] rounded",
                tick.side === "buy" ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
              )}>
                {tick.side.toUpperCase()}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Order Book Depth Component
// ============================================================================

function OrderBookDepth({ data }: { data: OrderBookData }) {
  const maxTotal = Math.max(
    data.bids[data.bids.length - 1]?.total || 0,
    data.asks[data.asks.length - 1]?.total || 0
  );

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-2 pb-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
            <CardTitle className="text-xs font-semibold text-neutral-300">
              ORDER BOOK
            </CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[9px] text-neutral-500">Spread:</span>
            <span className="text-[10px] font-mono text-amber-400">
              ${data.spread.toFixed(3)}
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        {/* Mid Price */}
        <div className="text-center py-1 mb-2 bg-neutral-800/50 rounded">
          <span className="text-lg font-mono font-bold text-neutral-100">
            ${data.midPrice.toFixed(2)}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-1">
          {/* Bids */}
          <div className="space-y-0.5">
            <div className="grid grid-cols-3 text-[8px] text-neutral-500 uppercase tracking-wider px-1 mb-1">
              <span>Price</span>
              <span className="text-right">Size</span>
              <span className="text-right">Total</span>
            </div>
            {data.bids.slice(0, 10).map((level, i) => (
              <div key={i} className="relative">
                <div
                  className="absolute inset-0 bg-green-500/10 rounded-sm"
                  style={{ width: `${(level.total / maxTotal) * 100}%` }}
                />
                <div className="relative grid grid-cols-3 text-[10px] font-mono px-1 py-0.5">
                  <span className="text-green-400">{level.price.toFixed(2)}</span>
                  <span className="text-right text-neutral-300">{level.size}</span>
                  <span className="text-right text-neutral-500">{level.total}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Asks */}
          <div className="space-y-0.5">
            <div className="grid grid-cols-3 text-[8px] text-neutral-500 uppercase tracking-wider px-1 mb-1">
              <span>Price</span>
              <span className="text-right">Size</span>
              <span className="text-right">Total</span>
            </div>
            {data.asks.slice(0, 10).map((level, i) => (
              <div key={i} className="relative">
                <div
                  className="absolute inset-0 right-0 bg-red-500/10 rounded-sm"
                  style={{ width: `${(level.total / maxTotal) * 100}%`, marginLeft: "auto" }}
                />
                <div className="relative grid grid-cols-3 text-[10px] font-mono px-1 py-0.5">
                  <span className="text-red-400">{level.price.toFixed(2)}</span>
                  <span className="text-right text-neutral-300">{level.size}</span>
                  <span className="text-right text-neutral-500">{level.total}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Multi-Timeframe Signal Panel
// ============================================================================

function TimeframeSignalPanel({ signals }: { signals: TimeframeData[] }) {
  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-2 pb-1">
        <div className="flex items-center gap-2">
          <Layers className="w-3.5 h-3.5 text-purple-400" />
          <CardTitle className="text-xs font-semibold text-neutral-300">
            MTF SIGNALS
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        <div className="grid grid-cols-6 gap-1">
          {signals.map((s) => (
            <div
              key={s.tf}
              className={cn(
                "p-2 rounded-lg border text-center",
                s.signal === "long"
                  ? "bg-green-500/5 border-green-500/20"
                  : s.signal === "short"
                  ? "bg-red-500/5 border-red-500/20"
                  : "bg-neutral-800/50 border-neutral-700/50"
              )}
            >
              <div className="text-[10px] font-mono text-neutral-400 mb-1">
                {s.tf}
              </div>
              <div className={cn(
                "text-lg font-bold",
                s.signal === "long" ? "text-green-400" :
                s.signal === "short" ? "text-red-400" : "text-neutral-400"
              )}>
                {s.signal === "long" ? (
                  <TrendingUp className="w-5 h-5 mx-auto" />
                ) : s.signal === "short" ? (
                  <TrendingDown className="w-5 h-5 mx-auto" />
                ) : (
                  <span>—</span>
                )}
              </div>
              <div className="text-[10px] font-mono mt-1">
                <span className={cn(
                  s.strength >= 70 ? "text-green-400" :
                  s.strength >= 50 ? "text-amber-400" : "text-red-400"
                )}>
                  {s.strength}%
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Confluence Meter */}
        <div className="mt-2 p-2 bg-neutral-800/30 rounded-lg">
          <div className="flex items-center justify-between text-[10px] mb-1">
            <span className="text-neutral-500">CONFLUENCE</span>
            <span className="font-mono text-cyan-400">
              {signals.filter(s => s.signal === "long").length}/
              {signals.length} LONG
            </span>
          </div>
          <div className="h-1.5 bg-neutral-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-green-500 to-emerald-400 rounded-full"
              style={{
                width: `${(signals.filter(s => s.signal === "long").length / signals.length) * 100}%`
              }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Execution Analytics Component
// ============================================================================

function ExecutionAnalytics({ metrics }: { metrics: ExecutionMetrics }) {
  const stats = [
    {
      label: "Fill Rate",
      value: `${metrics.fillRate.toFixed(1)}%`,
      icon: Target,
      color: metrics.fillRate >= 98 ? "text-green-400" : "text-amber-400",
    },
    {
      label: "Avg Slippage",
      value: `${metrics.avgSlippage.toFixed(3)}%`,
      icon: Gauge,
      color: metrics.avgSlippage <= 0.02 ? "text-green-400" : "text-red-400",
    },
    {
      label: "VWAP Dev",
      value: `${metrics.vwapDeviation >= 0 ? "+" : ""}${metrics.vwapDeviation.toFixed(2)}%`,
      icon: BarChart3,
      color: Math.abs(metrics.vwapDeviation) <= 0.1 ? "text-green-400" : "text-amber-400",
    },
    {
      label: "Exec Speed",
      value: `${metrics.executionSpeed.toFixed(1)}ms`,
      icon: Timer,
      color: metrics.executionSpeed <= 5 ? "text-green-400" : "text-amber-400",
    },
    {
      label: "Orders Today",
      value: metrics.ordersToday.toLocaleString(),
      icon: Boxes,
      color: "text-cyan-400",
    },
    {
      label: "Volume",
      value: `$${(metrics.volumeTraded / 1000000).toFixed(1)}M`,
      icon: Database,
      color: "text-purple-400",
    },
  ];

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-2 pb-1">
        <div className="flex items-center gap-2">
          <Cpu className="w-3.5 h-3.5 text-amber-400" />
          <CardTitle className="text-xs font-semibold text-neutral-300">
            EXECUTION ANALYTICS
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        <div className="grid grid-cols-3 gap-2">
          {stats.map((stat) => (
            <div
              key={stat.label}
              className="p-2 bg-neutral-800/30 rounded-lg border border-neutral-700/50"
            >
              <div className="flex items-center gap-1.5 mb-1">
                <stat.icon className={cn("w-3 h-3", stat.color)} />
                <span className="text-[9px] uppercase tracking-wider text-neutral-500">
                  {stat.label}
                </span>
              </div>
              <div className={cn("text-sm font-mono font-bold", stat.color)}>
                {stat.value}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Technical Indicators Panel
// ============================================================================

function TechnicalIndicators() {
  const indicators = [
    { name: "RSI(14)", value: 62.4, signal: "neutral", range: [30, 70] },
    { name: "MACD", value: 0.24, signal: "bullish", range: [-1, 1] },
    { name: "BB %B", value: 0.72, signal: "overbought", range: [0, 1] },
    { name: "ATR(14)", value: 1.82, signal: "normal", range: [0, 5] },
    { name: "ADX(14)", value: 34.2, signal: "trending", range: [0, 100] },
    { name: "Stoch %K", value: 78.5, signal: "overbought", range: [0, 100] },
    { name: "CCI(20)", value: 124, signal: "bullish", range: [-200, 200] },
    { name: "MFI(14)", value: 58.3, signal: "neutral", range: [0, 100] },
  ];

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-2 pb-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <LineChart className="w-3.5 h-3.5 text-blue-400" />
            <CardTitle className="text-xs font-semibold text-neutral-300">
              TECHNICAL INDICATORS
            </CardTitle>
          </div>
          <Badge className="bg-neutral-800 border-neutral-700 text-neutral-400 text-[9px]">
            8 Active
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        <div className="grid grid-cols-4 gap-1.5">
          {indicators.map((ind) => {
            const getColor = () => {
              if (ind.signal === "bullish" || ind.signal === "trending") return "text-green-400";
              if (ind.signal === "bearish" || ind.signal === "overbought") return "text-red-400";
              return "text-neutral-300";
            };

            return (
              <div
                key={ind.name}
                className="p-1.5 bg-neutral-800/30 rounded border border-neutral-700/50"
              >
                <div className="text-[9px] text-neutral-500 mb-0.5">{ind.name}</div>
                <div className={cn("text-sm font-mono font-bold", getColor())}>
                  {typeof ind.value === "number" && ind.value % 1 !== 0
                    ? ind.value.toFixed(2)
                    : ind.value}
                </div>
                <div className="text-[8px] text-neutral-600 uppercase">{ind.signal}</div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Signal Strength Meters
// ============================================================================

function SignalStrengthMeters() {
  const assets = [
    { symbol: "CL", name: "Crude", signal: 72, direction: "long" as const },
    { symbol: "GC", name: "Gold", signal: 84, direction: "long" as const },
    { symbol: "NG", name: "NatGas", signal: -45, direction: "short" as const },
    { symbol: "BTC", name: "Bitcoin", signal: 58, direction: "long" as const },
    { symbol: "SI", name: "Silver", signal: -28, direction: "short" as const },
    { symbol: "ZW", name: "Wheat", signal: 15, direction: "neutral" as const },
  ];

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-2 pb-1">
        <div className="flex items-center gap-2">
          <Zap className="w-3.5 h-3.5 text-amber-400" />
          <CardTitle className="text-xs font-semibold text-neutral-300">
            SIGNAL STRENGTH
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        <div className="grid grid-cols-3 gap-2">
          {assets.map((asset) => (
            <div
              key={asset.symbol}
              className="p-2 bg-neutral-800/30 rounded-lg border border-neutral-700/50 flex flex-col items-center"
            >
              <div className="text-[10px] font-mono text-neutral-400 mb-1">
                {asset.symbol}
              </div>
              <SignalGauge
                value={asset.signal}
                confidence={Math.abs(asset.signal)}
                size="sm"
                animated={false}
              />
              <div className={cn(
                "text-[10px] font-bold mt-1",
                asset.signal > 20 ? "text-green-400" :
                asset.signal < -20 ? "text-red-400" : "text-neutral-400"
              )}>
                {asset.signal > 0 ? "+" : ""}{asset.signal}%
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// System Status Bar
// ============================================================================

function SystemStatusBar() {
  const [latency, setLatency] = useState(2.4);
  const [cpu, setCpu] = useState(34);
  const [memory, setMemory] = useState(68);

  useEffect(() => {
    const interval = setInterval(() => {
      setLatency(prev => Math.max(0.5, Math.min(10, prev + (Math.random() - 0.5) * 0.5)));
      setCpu(prev => Math.max(10, Math.min(90, prev + (Math.random() - 0.5) * 5)));
      setMemory(prev => Math.max(40, Math.min(85, prev + (Math.random() - 0.5) * 2)));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-4 px-4 py-1.5 bg-neutral-900 border-b border-neutral-800">
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
        <span className="text-[10px] text-neutral-400">SYSTEM ONLINE</span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-neutral-500">LATENCY:</span>
        <span className={cn(
          "text-[10px] font-mono",
          latency < 5 ? "text-green-400" : "text-amber-400"
        )}>
          {latency.toFixed(1)}ms
        </span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-neutral-500">CPU:</span>
        <span className={cn(
          "text-[10px] font-mono",
          cpu < 70 ? "text-green-400" : "text-amber-400"
        )}>
          {cpu.toFixed(0)}%
        </span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-neutral-500">MEM:</span>
        <span className={cn(
          "text-[10px] font-mono",
          memory < 80 ? "text-green-400" : "text-amber-400"
        )}>
          {memory.toFixed(0)}%
        </span>
      </div>
      <div className="flex-1" />
      <ApiHealthIndicator />
      <QuantumStatusWidget compact />
    </div>
  );
}

// ============================================================================
// Main Dashboard Component
// ============================================================================

export function HardcoreQuantDashboard() {
  const [activeTab, setActiveTab] = useState("overview");

  // Generate mock data
  const ticks = useMemo(() => generateTickData(100), []);
  const orderBook = useMemo(() => generateOrderBook(), []);
  const timeframeSignals = useMemo(() => generateTimeframeSignals(), []);
  const executionMetrics = useMemo(() => generateExecutionMetrics(), []);

  return (
    <div className="flex flex-col min-h-screen -m-6 bg-neutral-950">
      {/* Market Ticker */}
      <MarketTicker speed="fast" showVolume={true} pauseOnHover={false} />

      {/* System Status */}
      <SystemStatusBar />

      {/* Market Status */}
      <MarketStatusBar showFullDetails={true} />

      {/* Main Content */}
      <div className="flex-1 p-2 space-y-2">
        {/* Header Row */}
        <div className="flex items-center justify-between px-2">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-500/20 rounded-lg border border-purple-500/30">
              <Cpu className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-neutral-100">Hardcore Quant Terminal</h1>
              <p className="text-[10px] text-neutral-500">
                Maximum data density • Real-time analytics • Multi-asset signals
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge className="bg-green-500/10 border-green-500/30 text-green-400 text-[10px]">
              <Activity className="w-3 h-3 mr-1 animate-pulse" />
              LIVE
            </Badge>
            <Badge className="bg-purple-500/10 border-purple-500/30 text-purple-400 text-[10px]">
              <Boxes className="w-3 h-3 mr-1" />
              10,179 Models
            </Badge>
            <button className="p-1.5 rounded hover:bg-neutral-800 transition-colors">
              <Settings className="w-4 h-4 text-neutral-500" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-neutral-900 border border-neutral-800 p-0.5 h-8">
            <TabsTrigger value="overview" className="text-[10px] h-7 px-3 data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400">
              OVERVIEW
            </TabsTrigger>
            <TabsTrigger value="signals" className="text-[10px] h-7 px-3 data-[state=active]:bg-blue-500/20 data-[state=active]:text-blue-400">
              SIGNALS
            </TabsTrigger>
            <TabsTrigger value="risk" className="text-[10px] h-7 px-3 data-[state=active]:bg-amber-500/20 data-[state=active]:text-amber-400">
              RISK
            </TabsTrigger>
            <TabsTrigger value="execution" className="text-[10px] h-7 px-3 data-[state=active]:bg-green-500/20 data-[state=active]:text-green-400">
              EXECUTION
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab - Maximum Density */}
          <TabsContent value="overview" className="mt-2 space-y-2">
            {/* Row 1: Tick Data + Order Book + MTF Signals */}
            <div className="grid grid-cols-12 gap-2">
              <div className="col-span-3">
                <TickTape ticks={ticks} />
              </div>
              <div className="col-span-4">
                <OrderBookDepth data={orderBook} />
              </div>
              <div className="col-span-5">
                <TimeframeSignalPanel signals={timeframeSignals} />
              </div>
            </div>

            {/* Row 2: Technical Indicators + Signal Meters + Execution */}
            <div className="grid grid-cols-12 gap-2">
              <div className="col-span-5">
                <TechnicalIndicators />
              </div>
              <div className="col-span-4">
                <SignalStrengthMeters />
              </div>
              <div className="col-span-3">
                <ExecutionAnalytics metrics={executionMetrics} />
              </div>
            </div>

            {/* Row 3: Greeks + Correlation */}
            <div className="grid grid-cols-12 gap-2">
              <div className="col-span-5">
                <GreeksPanel compact />
              </div>
              <div className="col-span-7">
                <CorrelationMatrix size="sm" showLabels={true} interactive={true} />
              </div>
            </div>
          </TabsContent>

          {/* Signals Tab */}
          <TabsContent value="signals" className="mt-2 space-y-2">
            <div className="grid grid-cols-12 gap-2">
              <div className="col-span-8">
                <TimeframeSignalPanel signals={timeframeSignals} />
              </div>
              <div className="col-span-4">
                <SignalStrengthMeters />
              </div>
            </div>
            <div className="grid grid-cols-12 gap-2">
              <div className="col-span-6">
                <TechnicalIndicators />
              </div>
              <div className="col-span-6">
                <VolatilitySurface showMiniChart={false} compact={true} />
              </div>
            </div>
          </TabsContent>

          {/* Risk Tab */}
          <TabsContent value="risk" className="mt-2 space-y-2">
            <div className="grid grid-cols-12 gap-2">
              <div className="col-span-5">
                <GreeksPanel />
              </div>
              <div className="col-span-7">
                <FactorAttributionPanel compact />
              </div>
            </div>
            <div className="grid grid-cols-12 gap-2">
              <div className="col-span-7">
                <CorrelationMatrix size="md" showLabels={true} interactive={true} />
              </div>
              <div className="col-span-5">
                <VolatilitySurface showMiniChart={true} compact={false} />
              </div>
            </div>
          </TabsContent>

          {/* Execution Tab */}
          <TabsContent value="execution" className="mt-2 space-y-2">
            <div className="grid grid-cols-12 gap-2">
              <div className="col-span-4">
                <TickTape ticks={ticks} />
              </div>
              <div className="col-span-4">
                <OrderBookDepth data={orderBook} />
              </div>
              <div className="col-span-4">
                <ExecutionAnalytics metrics={executionMetrics} />
              </div>
            </div>
            <div className="grid grid-cols-12 gap-2">
              <div className="col-span-12">
                <Card className="bg-neutral-900/50 border-neutral-800">
                  <CardHeader className="p-2">
                    <CardTitle className="text-xs font-semibold text-neutral-300">
                      EXECUTION TIMELINE
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-2 pt-0">
                    <div className="h-32 flex items-center justify-center text-neutral-500 text-xs">
                      [Execution chart visualization - integrate with real order data]
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <div className="flex items-center justify-between px-2 py-1 text-[9px] text-neutral-600">
          <span>QDT Nexus v2.1 • Hardcore Quant Terminal</span>
          <span>Last sync: {new Date().toLocaleTimeString()}</span>
        </div>
      </div>
    </div>
  );
}

export default HardcoreQuantDashboard;
