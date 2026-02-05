"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { LiveSignalCard } from "@/components/dashboard/LiveSignalCard";
import { ApiHealthIndicator } from "@/components/dashboard/ApiHealthIndicator";
import { PositionRiskCalculator } from "@/components/dashboard/PositionRiskCalculator";
import { CorrelationMatrix } from "@/components/dashboard/CorrelationMatrix";
import { MarketTicker } from "@/components/dashboard/MarketTicker";
import { MarketStatusBar } from "@/components/dashboard/MarketStatusBar";
import { VolatilitySurface } from "@/components/dashboard/VolatilitySurface";
import { MOCK_ASSETS, MOCK_SIGNALS, type Horizon, type SignalData } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import {
  Shield,
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  DollarSign,
  BarChart2,
  ArrowUpRight,
  ArrowDownRight,
  AlertTriangle,
  Scale,
  GitBranch,
  Calculator,
  PieChart,
  Briefcase,
  RefreshCw,
  AlertCircle,
  ChevronRight,
  Zap,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface HedgeOpportunity {
  assetId: AssetId;
  assetName: string;
  symbol: string;
  currentPrice: number;
  signal: SignalData;
  dollarPerTrade: number;
  winRate: number;
  horizonCoverage: number;
  hedgeScore: number;
  hedgeType: "protective" | "speculative" | "basis";
  urgency: "high" | "medium" | "low";
  rank: number;
}

interface ActivePosition {
  id: string;
  asset: string;
  symbol: string;
  direction: "long" | "short";
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  hedgeRatio: number;
  daysHeld: number;
  status: "active" | "partial" | "closing";
}

interface BasisRiskData {
  underlying: string;
  hedgeInstrument: string;
  basisSpread: number;
  basisVolatility: number;
  correlation: number;
  convergenceRisk: "low" | "medium" | "high";
}

// ============================================================================
// Mock Data Generators
// ============================================================================

function generateActivePositions(): ActivePosition[] {
  return [
    {
      id: "H001",
      asset: "WTI Crude",
      symbol: "CL",
      direction: "short",
      quantity: 50,
      entryPrice: 74.25,
      currentPrice: 73.48,
      pnl: 3850,
      pnlPercent: 1.04,
      hedgeRatio: 0.85,
      daysHeld: 12,
      status: "active",
    },
    {
      id: "H002",
      asset: "Gold",
      symbol: "GC",
      direction: "long",
      quantity: 25,
      entryPrice: 2048.50,
      currentPrice: 2068.20,
      pnl: 4925,
      pnlPercent: 0.96,
      hedgeRatio: 1.0,
      daysHeld: 8,
      status: "active",
    },
    {
      id: "H003",
      asset: "Natural Gas",
      symbol: "NG",
      direction: "short",
      quantity: 100,
      entryPrice: 2.85,
      currentPrice: 2.92,
      pnl: -7000,
      pnlPercent: -2.46,
      hedgeRatio: 0.70,
      daysHeld: 5,
      status: "partial",
    },
    {
      id: "H004",
      asset: "Wheat",
      symbol: "ZW",
      direction: "long",
      quantity: 40,
      entryPrice: 615.25,
      currentPrice: 608.50,
      pnl: -2700,
      pnlPercent: -1.10,
      hedgeRatio: 0.92,
      daysHeld: 21,
      status: "closing",
    },
    {
      id: "H005",
      asset: "Copper",
      symbol: "HG",
      direction: "short",
      quantity: 30,
      entryPrice: 3.92,
      currentPrice: 3.88,
      pnl: 1200,
      pnlPercent: 1.02,
      hedgeRatio: 0.78,
      daysHeld: 3,
      status: "active",
    },
  ];
}

function generateBasisRiskData(): BasisRiskData[] {
  return [
    {
      underlying: "Physical WTI",
      hedgeInstrument: "CL Futures",
      basisSpread: 0.35,
      basisVolatility: 0.12,
      correlation: 0.97,
      convergenceRisk: "low",
    },
    {
      underlying: "Jet Fuel",
      hedgeInstrument: "HO Futures",
      basisSpread: 1.85,
      basisVolatility: 0.28,
      correlation: 0.89,
      convergenceRisk: "medium",
    },
    {
      underlying: "EU Gas",
      hedgeInstrument: "NG Futures",
      basisSpread: 4.20,
      basisVolatility: 0.45,
      correlation: 0.72,
      convergenceRisk: "high",
    },
    {
      underlying: "Physical Gold",
      hedgeInstrument: "GC Futures",
      basisSpread: 0.08,
      basisVolatility: 0.05,
      correlation: 0.99,
      convergenceRisk: "low",
    },
    {
      underlying: "Corn Basis",
      hedgeInstrument: "ZC Futures",
      basisSpread: 0.42,
      basisVolatility: 0.18,
      correlation: 0.94,
      convergenceRisk: "low",
    },
  ];
}

// ============================================================================
// Hedge Opportunity Generation
// ============================================================================

function calculateDollarPerTrade(signal: SignalData, currentPrice: number): number {
  const expectedMovePercent = signal.directionalAccuracy > 55 ? 0.025 : 0.015;
  const winProb = signal.directionalAccuracy / 100;
  const avgWin = currentPrice * expectedMovePercent;
  const avgLoss = currentPrice * expectedMovePercent * 0.8;
  return (winProb * avgWin) - ((1 - winProb) * avgLoss);
}

function calculateHorizonCoverage(assetId: AssetId): number {
  const horizons: Horizon[] = ["D+1", "D+5", "D+10"];
  let consistentDirection = 0;
  const signals = horizons.map(h => MOCK_SIGNALS[assetId]?.[h]).filter(Boolean);
  if (signals.length < 2) return 33;
  const firstDirection = signals[0]?.direction;
  for (const signal of signals) {
    if (signal?.direction === firstDirection && signal.direction !== "neutral") {
      consistentDirection++;
    }
  }
  return Math.round((consistentDirection / signals.length) * 100);
}

function determineHedgeType(signal: SignalData): "protective" | "speculative" | "basis" {
  if (signal.direction === "bearish" && signal.confidence >= 70) return "protective";
  if (signal.sharpeRatio >= 2.0) return "speculative";
  return "basis";
}

function determineUrgency(signal: SignalData): "high" | "medium" | "low" {
  if (signal.confidence >= 80 && signal.sharpeRatio >= 2.5) return "high";
  if (signal.confidence >= 65 || signal.sharpeRatio >= 2.0) return "medium";
  return "low";
}

function calculateHedgeScore(
  dollarPerTrade: number,
  winRate: number,
  horizonCoverage: number,
  signal: SignalData
): number {
  const normalizedDpt = Math.min(Math.max(dollarPerTrade * 10, 0), 30);
  const winRateScore = (winRate / 100) * 30;
  const horizonScore = (horizonCoverage / 100) * 20;
  const confidenceScore = (signal.confidence / 100) * 20;
  return normalizedDpt + winRateScore + horizonScore + confidenceScore;
}

function getHedgeOpportunities(): HedgeOpportunity[] {
  const opportunities: HedgeOpportunity[] = [];
  const horizons: Horizon[] = ["D+1", "D+5", "D+10"];

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    let bestSignal: SignalData | null = null;
    let bestHorizon: Horizon = "D+1";

    for (const horizon of horizons) {
      const signal = MOCK_SIGNALS[assetId as AssetId]?.[horizon];
      if (signal && signal.direction !== "neutral") {
        if (!bestSignal ||
            (signal.directionalAccuracy > bestSignal.directionalAccuracy &&
             signal.confidence >= bestSignal.confidence - 5)) {
          bestSignal = signal;
          bestHorizon = horizon;
        }
      }
    }

    if (bestSignal && bestSignal.directionalAccuracy >= 52) {
      const dollarPerTrade = calculateDollarPerTrade(bestSignal, asset.currentPrice);
      const winRate = bestSignal.directionalAccuracy;
      const horizonCoverage = calculateHorizonCoverage(assetId as AssetId);

      opportunities.push({
        assetId: assetId as AssetId,
        assetName: asset.name,
        symbol: asset.symbol,
        currentPrice: asset.currentPrice,
        signal: { ...bestSignal, horizon: bestHorizon },
        dollarPerTrade,
        winRate,
        horizonCoverage,
        hedgeScore: calculateHedgeScore(dollarPerTrade, winRate, horizonCoverage, bestSignal),
        hedgeType: determineHedgeType(bestSignal),
        urgency: determineUrgency(bestSignal),
        rank: 0,
      });
    }
  });

  return opportunities
    .sort((a, b) => b.hedgeScore - a.hedgeScore)
    .map((o, i) => ({ ...o, rank: i + 1 }));
}

// ============================================================================
// Header Component
// ============================================================================

function HedgingHeader() {
  return (
    <div className="bg-gradient-to-r from-emerald-900/30 via-teal-900/20 to-emerald-900/30 border border-emerald-500/20 rounded-xl p-6">
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-emerald-500/20 rounded-xl border border-emerald-500/30">
            <Shield className="w-8 h-8 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-neutral-100">Hedging Command Center</h1>
            <p className="text-sm text-neutral-400">
              Institutional-grade risk management and hedge execution
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Badge className="bg-green-500/10 border-green-500/30 text-green-400 px-3 py-1.5">
            <Activity className="w-3.5 h-3.5 mr-1.5 animate-pulse" />
            Live
          </Badge>
          <Badge className="bg-emerald-500/10 border-emerald-500/30 text-emerald-300 px-3 py-1.5">
            <Briefcase className="w-3.5 h-3.5 mr-1.5" />
            5 Active Hedges
          </Badge>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Portfolio Risk Summary
// ============================================================================

function PortfolioRiskSummary({ positions }: { positions: ActivePosition[] }) {
  const stats = useMemo(() => {
    const totalPnL = positions.reduce((acc, p) => acc + p.pnl, 0);
    const totalNotional = positions.reduce((acc, p) => acc + p.quantity * p.currentPrice, 0);
    const avgHedgeRatio = positions.reduce((acc, p) => acc + p.hedgeRatio, 0) / positions.length;
    const activeHedges = positions.filter(p => p.status === "active").length;
    const profitableHedges = positions.filter(p => p.pnl > 0).length;

    return { totalPnL, totalNotional, avgHedgeRatio, activeHedges, profitableHedges };
  }, [positions]);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <DollarSign className="w-4 h-4" />
            <span className="text-[10px] uppercase tracking-wider">Portfolio P&L</span>
          </div>
          <div className={cn(
            "text-xl font-bold font-mono",
            stats.totalPnL >= 0 ? "text-green-400" : "text-red-400"
          )}>
            {stats.totalPnL >= 0 ? "+" : ""}{stats.totalPnL.toLocaleString("en-US", { style: "currency", currency: "USD" })}
          </div>
        </CardContent>
      </Card>

      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <PieChart className="w-4 h-4" />
            <span className="text-[10px] uppercase tracking-wider">Notional Exposure</span>
          </div>
          <div className="text-xl font-bold font-mono text-blue-400">
            ${(stats.totalNotional / 1000000).toFixed(2)}M
          </div>
        </CardContent>
      </Card>

      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <Scale className="w-4 h-4" />
            <span className="text-[10px] uppercase tracking-wider">Avg Hedge Ratio</span>
          </div>
          <div className={cn(
            "text-xl font-bold font-mono",
            stats.avgHedgeRatio >= 0.9 ? "text-emerald-400" :
            stats.avgHedgeRatio >= 0.75 ? "text-yellow-400" : "text-red-400"
          )}>
            {(stats.avgHedgeRatio * 100).toFixed(0)}%
          </div>
        </CardContent>
      </Card>

      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <Activity className="w-4 h-4" />
            <span className="text-[10px] uppercase tracking-wider">Active Hedges</span>
          </div>
          <div className="text-xl font-bold font-mono text-cyan-400">
            {stats.activeHedges} / {positions.length}
          </div>
        </CardContent>
      </Card>

      <Card className="bg-neutral-900/50 border-neutral-800">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-2">
            <Target className="w-4 h-4" />
            <span className="text-[10px] uppercase tracking-wider">Hit Rate</span>
          </div>
          <div className="text-xl font-bold font-mono text-purple-400">
            {((stats.profitableHedges / positions.length) * 100).toFixed(0)}%
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ============================================================================
// Active Positions Tracker
// ============================================================================

function ActivePositionsTracker({ positions }: { positions: ActivePosition[] }) {
  const statusConfig = {
    active: { bg: "bg-green-500/10", border: "border-green-500/30", text: "text-green-400" },
    partial: { bg: "bg-amber-500/10", border: "border-amber-500/30", text: "text-amber-400" },
    closing: { bg: "bg-red-500/10", border: "border-red-500/30", text: "text-red-400" },
  };

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Briefcase className="w-5 h-5 text-emerald-400" />
            <CardTitle className="text-sm font-semibold text-neutral-200">
              Active Hedge Positions
            </CardTitle>
          </div>
          <Badge className="bg-neutral-800 border-neutral-700 text-neutral-400 text-xs">
            {positions.length} Positions
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {/* Table Header */}
        <div className="grid grid-cols-9 gap-2 px-4 py-2 bg-neutral-800/30 text-[10px] uppercase tracking-wider text-neutral-500">
          <div>ID</div>
          <div>Asset</div>
          <div className="text-center">Dir</div>
          <div className="text-right">Qty</div>
          <div className="text-right">Entry</div>
          <div className="text-right">Current</div>
          <div className="text-right">P&L</div>
          <div className="text-right">Hedge %</div>
          <div className="text-center">Status</div>
        </div>

        <div className="divide-y divide-neutral-800">
          {positions.map((pos) => {
            const status = statusConfig[pos.status];
            return (
              <div
                key={pos.id}
                className="grid grid-cols-9 gap-2 items-center px-4 py-3 hover:bg-neutral-800/30 transition-colors"
              >
                <span className="font-mono text-xs text-neutral-500">{pos.id}</span>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs text-neutral-400 bg-neutral-800 px-1.5 py-0.5 rounded">
                    {pos.symbol}
                  </span>
                  <span className="text-sm text-neutral-300 hidden lg:inline">{pos.asset}</span>
                </div>
                <div className="flex justify-center">
                  {pos.direction === "long" ? (
                    <ArrowUpRight className="w-4 h-4 text-green-500" />
                  ) : (
                    <ArrowDownRight className="w-4 h-4 text-red-500" />
                  )}
                </div>
                <span className="text-right font-mono text-sm text-neutral-300">{pos.quantity}</span>
                <span className="text-right font-mono text-sm text-neutral-400">
                  ${pos.entryPrice.toFixed(2)}
                </span>
                <span className="text-right font-mono text-sm text-neutral-200">
                  ${pos.currentPrice.toFixed(2)}
                </span>
                <span className={cn(
                  "text-right font-mono text-sm font-bold",
                  pos.pnl >= 0 ? "text-green-400" : "text-red-400"
                )}>
                  {pos.pnl >= 0 ? "+" : ""}{pos.pnl.toLocaleString("en-US", { style: "currency", currency: "USD" })}
                </span>
                <div className="text-right">
                  <div className="flex items-center justify-end gap-2">
                    <div className="w-12 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                      <div
                        className={cn(
                          "h-full rounded-full",
                          pos.hedgeRatio >= 0.9 ? "bg-emerald-500" :
                          pos.hedgeRatio >= 0.75 ? "bg-yellow-500" : "bg-red-500"
                        )}
                        style={{ width: `${pos.hedgeRatio * 100}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs text-neutral-400 w-8">
                      {(pos.hedgeRatio * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <div className="flex justify-center">
                  <Badge className={cn("text-[10px] px-2 py-0.5", status.bg, status.border, status.text)}>
                    {pos.status.toUpperCase()}
                  </Badge>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Basis Risk Monitor
// ============================================================================

function BasisRiskMonitor({ basisData }: { basisData: BasisRiskData[] }) {
  const riskConfig = {
    low: { bg: "bg-green-500/10", border: "border-green-500/30", text: "text-green-400", label: "LOW" },
    medium: { bg: "bg-amber-500/10", border: "border-amber-500/30", text: "text-amber-400", label: "MED" },
    high: { bg: "bg-red-500/10", border: "border-red-500/30", text: "text-red-400", label: "HIGH" },
  };

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-cyan-400" />
            <CardTitle className="text-sm font-semibold text-neutral-200">
              Basis Risk Monitor
            </CardTitle>
          </div>
          <Badge className="bg-cyan-500/10 border-cyan-500/30 text-cyan-400 text-xs">
            {basisData.filter(b => b.convergenceRisk === "high").length} High Risk
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-3">
        {basisData.map((basis, idx) => {
          const risk = riskConfig[basis.convergenceRisk];
          return (
            <div
              key={idx}
              className="p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50 hover:border-neutral-600 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <div>
                  <span className="text-sm text-neutral-200">{basis.underlying}</span>
                  <span className="text-neutral-600 mx-2">→</span>
                  <span className="text-sm text-neutral-400">{basis.hedgeInstrument}</span>
                </div>
                <Badge className={cn("text-[10px]", risk.bg, risk.border, risk.text)}>
                  {risk.label}
                </Badge>
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <span className="text-neutral-500">Spread:</span>
                  <span className="ml-2 font-mono text-neutral-300">${basis.basisSpread.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-neutral-500">Vol:</span>
                  <span className="ml-2 font-mono text-neutral-300">{(basis.basisVolatility * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <span className="text-neutral-500">Corr:</span>
                  <span className={cn(
                    "ml-2 font-mono",
                    basis.correlation >= 0.95 ? "text-green-400" :
                    basis.correlation >= 0.85 ? "text-yellow-400" : "text-red-400"
                  )}>
                    {basis.correlation.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Hedge Ratio Calculator
// ============================================================================

function HedgeRatioCalculator() {
  const [physicalValue, setPhysicalValue] = useState(10000000);
  const [correlation, setCorrelation] = useState(0.92);
  const [volatilityRatio, setVolatilityRatio] = useState(1.08);

  const optimalHedgeRatio = useMemo(() => {
    return correlation * volatilityRatio;
  }, [correlation, volatilityRatio]);

  const hedgedValue = useMemo(() => {
    return physicalValue * optimalHedgeRatio;
  }, [physicalValue, optimalHedgeRatio]);

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center gap-2">
          <Calculator className="w-5 h-5 text-purple-400" />
          <CardTitle className="text-sm font-semibold text-neutral-200">
            Optimal Hedge Ratio
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-4">
        {/* Physical Exposure Input */}
        <div>
          <label className="flex items-center justify-between text-xs text-neutral-500 mb-2">
            <span>Physical Exposure</span>
            <span className="font-mono text-neutral-300">${(physicalValue / 1000000).toFixed(1)}M</span>
          </label>
          <input
            type="range"
            min={1000000}
            max={50000000}
            step={1000000}
            value={physicalValue}
            onChange={(e) => setPhysicalValue(Number(e.target.value))}
            className="w-full h-2 bg-neutral-800 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Correlation Input */}
        <div>
          <label className="flex items-center justify-between text-xs text-neutral-500 mb-2">
            <span>Correlation (ρ)</span>
            <span className="font-mono text-cyan-400">{correlation.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min={0.5}
            max={1.0}
            step={0.01}
            value={correlation}
            onChange={(e) => setCorrelation(Number(e.target.value))}
            className="w-full h-2 bg-neutral-800 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Volatility Ratio Input */}
        <div>
          <label className="flex items-center justify-between text-xs text-neutral-500 mb-2">
            <span>σ(Spot) / σ(Futures)</span>
            <span className="font-mono text-amber-400">{volatilityRatio.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min={0.5}
            max={1.5}
            step={0.01}
            value={volatilityRatio}
            onChange={(e) => setVolatilityRatio(Number(e.target.value))}
            className="w-full h-2 bg-neutral-800 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <Separator className="bg-neutral-800" />

        {/* Results */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 bg-gradient-to-br from-purple-500/10 to-blue-500/10 rounded-lg border border-purple-500/20">
            <span className="text-[10px] uppercase tracking-wider text-neutral-500">Optimal Ratio</span>
            <div className="text-2xl font-bold font-mono text-purple-400">
              {(optimalHedgeRatio * 100).toFixed(1)}%
            </div>
          </div>
          <div className="p-3 bg-gradient-to-br from-emerald-500/10 to-teal-500/10 rounded-lg border border-emerald-500/20">
            <span className="text-[10px] uppercase tracking-wider text-neutral-500">Hedge Notional</span>
            <div className="text-2xl font-bold font-mono text-emerald-400">
              ${(hedgedValue / 1000000).toFixed(2)}M
            </div>
          </div>
        </div>

        {/* Formula Display */}
        <div className="p-3 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
          <span className="text-[10px] uppercase tracking-wider text-neutral-500">Formula</span>
          <div className="mt-1 font-mono text-xs text-neutral-400">
            h* = ρ × (σ<sub>S</sub> / σ<sub>F</sub>) = {correlation.toFixed(2)} × {volatilityRatio.toFixed(2)} = <span className="text-purple-400 font-bold">{optimalHedgeRatio.toFixed(3)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Hedge Recommendation Card (Simplified)
// ============================================================================

function HedgeRecommendationCard({ opportunity }: { opportunity: HedgeOpportunity }) {
  const isBullish = opportunity.signal.direction === "bullish";
  const directionColor = isBullish ? "text-green-500" : "text-red-500";

  const urgencyConfig = {
    high: { bg: "bg-amber-500/10", border: "border-amber-500/30", text: "text-amber-400", label: "ACT NOW" },
    medium: { bg: "bg-blue-500/10", border: "border-blue-500/30", text: "text-blue-400", label: "MONITOR" },
    low: { bg: "bg-neutral-500/10", border: "border-neutral-500/30", text: "text-neutral-400", label: "WATCH" },
  };

  const urgency = urgencyConfig[opportunity.urgency];

  return (
    <Card className={cn(
      "bg-neutral-900/50 border-neutral-800 hover:border-neutral-700 transition-all",
      opportunity.rank <= 3 && "ring-1 ring-emerald-500/30"
    )}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={cn(
              "w-7 h-7 rounded-lg flex items-center justify-center font-bold text-xs",
              opportunity.rank === 1 ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30" :
              opportunity.rank === 2 ? "bg-teal-500/20 text-teal-400 border border-teal-500/30" :
              "bg-neutral-800 text-neutral-500"
            )}>
              #{opportunity.rank}
            </div>
            <div>
              <span className="font-mono text-xs text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
                {opportunity.symbol}
              </span>
              <span className="ml-2 text-sm text-neutral-200">{opportunity.assetName}</span>
            </div>
          </div>
          <Badge className={cn("text-[10px]", urgency.bg, urgency.border, urgency.text)}>
            {urgency.label}
          </Badge>
        </div>

        <div className="flex items-center justify-between p-2 bg-neutral-800/50 rounded-lg mb-3">
          <div className="flex items-center gap-2">
            {isBullish ? (
              <TrendingUp className={cn("w-4 h-4", directionColor)} />
            ) : (
              <TrendingDown className={cn("w-4 h-4", directionColor)} />
            )}
            <span className={cn("text-sm font-semibold", directionColor)}>
              {opportunity.signal.direction.toUpperCase()}
            </span>
          </div>
          <span className="text-xs text-neutral-500">{opportunity.signal.horizon}</span>
        </div>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="text-[10px] text-neutral-500 mb-0.5">$/Trade</div>
            <div className={cn(
              "font-mono font-bold",
              opportunity.dollarPerTrade >= 0 ? "text-green-400" : "text-red-400"
            )}>
              {opportunity.dollarPerTrade >= 0 ? "+" : ""}{opportunity.dollarPerTrade.toFixed(2)}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-neutral-500 mb-0.5">Win Rate</div>
            <div className="font-mono font-bold text-emerald-400">
              {opportunity.winRate.toFixed(0)}%
            </div>
          </div>
          <div>
            <div className="text-[10px] text-neutral-500 mb-0.5">Score</div>
            <div className="font-mono font-bold text-purple-400">
              {opportunity.hedgeScore.toFixed(0)}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Quick Actions Panel
// ============================================================================

function QuickActionsPanel({ opportunities }: { opportunities: HedgeOpportunity[] }) {
  const urgentCount = opportunities.filter(o => o.urgency === "high").length;
  const protectiveCount = opportunities.filter(o => o.hedgeType === "protective").length;

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-amber-400" />
          <CardTitle className="text-sm font-semibold text-neutral-200">
            Quick Actions
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-3">
        {/* Urgent Alert */}
        {urgentCount > 0 && (
          <div className="p-3 bg-amber-500/5 border border-amber-500/20 rounded-lg">
            <div className="flex items-center gap-2 mb-1">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              <span className="text-sm font-medium text-amber-400">{urgentCount} Urgent Actions</span>
            </div>
            <p className="text-xs text-neutral-500">Hedges requiring immediate attention</p>
          </div>
        )}

        {/* Protective Hedges */}
        <div className="p-3 bg-emerald-500/5 border border-emerald-500/20 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <Shield className="w-4 h-4 text-emerald-400" />
            <span className="text-sm font-medium text-emerald-400">{protectiveCount} Protective</span>
          </div>
          <p className="text-xs text-neutral-500">Downside protection opportunities</p>
        </div>

        {/* Action Buttons */}
        <div className="grid grid-cols-2 gap-2 pt-2">
          <button className="p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400 text-xs font-medium hover:bg-emerald-500/20 transition-colors flex items-center justify-center gap-1">
            <RefreshCw className="w-3 h-3" />
            Rebalance
          </button>
          <button className="p-2 bg-blue-500/10 border border-blue-500/30 rounded-lg text-blue-400 text-xs font-medium hover:bg-blue-500/20 transition-colors flex items-center justify-center gap-1">
            <Calculator className="w-3 h-3" />
            Recalculate
          </button>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Hedging Dashboard Component
// ============================================================================

export function HedgingDashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const opportunities = useMemo(() => getHedgeOpportunities(), []);
  const positions = useMemo(() => generateActivePositions(), []);
  const basisData = useMemo(() => generateBasisRiskData(), []);
  const topOpportunities = opportunities.slice(0, 6);

  return (
    <div className="flex flex-col min-h-screen -m-6">
      {/* Bloomberg-style Market Ticker */}
      <MarketTicker speed="normal" showVolume={false} pauseOnHover={true} />

      {/* Market Status Bar */}
      <MarketStatusBar showFullDetails={true} />

      {/* Main Content */}
      <div className="flex-1 p-6 space-y-6">
        {/* Header */}
        <HedgingHeader />

        {/* Portfolio Risk Summary */}
        <PortfolioRiskSummary positions={positions} />

        <Separator className="bg-neutral-800" />

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-neutral-800/50 border border-neutral-700/50 p-1">
            <TabsTrigger
              value="overview"
              className="data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-400"
            >
              <Target className="w-4 h-4 mr-2" />
              Position Overview
            </TabsTrigger>
            <TabsTrigger
              value="recommendations"
              className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400"
            >
              <Shield className="w-4 h-4 mr-2" />
              Hedge Signals
            </TabsTrigger>
            <TabsTrigger
              value="calculator"
              className="data-[state=active]:bg-blue-500/20 data-[state=active]:text-blue-400"
            >
              <Calculator className="w-4 h-4 mr-2" />
              Risk Calculator
            </TabsTrigger>
            <TabsTrigger
              value="analytics"
              className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400"
            >
              <BarChart2 className="w-4 h-4 mr-2" />
              Analytics
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Active Positions */}
            <ActivePositionsTracker positions={positions} />

            {/* Two Column Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Basis Risk Monitor */}
              <div className="lg:col-span-2">
                <BasisRiskMonitor basisData={basisData} />
              </div>

              {/* Quick Actions */}
              <QuickActionsPanel opportunities={opportunities} />
            </div>

            {/* Live CME Signals */}
            <Card className="bg-neutral-900/50 border-neutral-800">
              <CardHeader className="p-4 pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Activity className="w-5 h-5 text-emerald-400" />
                    <CardTitle className="text-sm font-semibold text-neutral-200">
                      Live CME Signals
                    </CardTitle>
                  </div>
                  <ApiHealthIndicator />
                </div>
              </CardHeader>
              <CardContent className="p-4 pt-0">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <LiveSignalCard asset="Crude_Oil" displayName="Crude Oil" />
                  <LiveSignalCard asset="GOLD" displayName="Gold" />
                  <LiveSignalCard asset="Wheat" displayName="Wheat" />
                  <LiveSignalCard asset="Natural_Gas" displayName="Natural Gas" />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Recommendations Tab */}
          <TabsContent value="recommendations" className="space-y-6">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
                <Shield className="w-5 h-5 text-emerald-400" />
                Top Hedge Recommendations
              </h2>
              <Badge className="bg-emerald-500/10 border-emerald-500/30 text-emerald-300 px-2 py-1 text-xs">
                Win Rate &gt; 52%
              </Badge>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {topOpportunities.map((opp) => (
                <HedgeRecommendationCard key={opp.assetId} opportunity={opp} />
              ))}
            </div>

            {/* Full Ranking Table */}
            <Card className="bg-neutral-900/50 border-neutral-800">
              <CardHeader className="p-4 pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <BarChart2 className="w-5 h-5 text-emerald-400" />
                    <CardTitle className="text-sm font-semibold text-neutral-200">
                      Full Hedge Ranking
                    </CardTitle>
                  </div>
                  <span className="text-xs text-neutral-500">Sorted by Score</span>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <div className="grid grid-cols-7 gap-2 px-4 py-2 bg-neutral-800/30 text-[10px] uppercase tracking-wider text-neutral-500">
                  <div>#</div>
                  <div>Asset</div>
                  <div className="text-center">Dir</div>
                  <div className="text-right">Win Rate</div>
                  <div className="text-right">$/Trade</div>
                  <div className="text-right">Conf</div>
                  <div className="text-right">Score</div>
                </div>
                <div className="divide-y divide-neutral-800 max-h-80 overflow-y-auto">
                  {opportunities.map((opp) => (
                    <div key={opp.assetId} className="grid grid-cols-7 gap-2 items-center px-4 py-3 hover:bg-neutral-800/30">
                      <span className={cn("font-mono text-sm", opp.rank <= 3 ? "text-emerald-400 font-bold" : "text-neutral-500")}>
                        {opp.rank}
                      </span>
                      <span className="font-mono text-xs text-neutral-400">{opp.symbol}</span>
                      <div className="flex justify-center">
                        {opp.signal.direction === "bullish" ? (
                          <ArrowUpRight className="w-4 h-4 text-green-500" />
                        ) : (
                          <ArrowDownRight className="w-4 h-4 text-red-500" />
                        )}
                      </div>
                      <span className="text-right font-mono text-sm text-emerald-400">{opp.winRate.toFixed(1)}%</span>
                      <span className={cn("text-right font-mono text-sm", opp.dollarPerTrade >= 0 ? "text-green-400" : "text-red-400")}>
                        {opp.dollarPerTrade >= 0 ? "+" : ""}{opp.dollarPerTrade.toFixed(2)}
                      </span>
                      <span className="text-right font-mono text-sm text-cyan-400">{opp.signal.confidence}%</span>
                      <span className="text-right font-mono text-sm font-bold text-purple-400">{opp.hedgeScore.toFixed(0)}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Calculator Tab */}
          <TabsContent value="calculator" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Hedge Ratio Calculator */}
              <HedgeRatioCalculator />

              {/* Position Risk Calculator */}
              <PositionRiskCalculator
                assetSymbol="CL"
                assetName="Crude Oil"
                currentPrice={73.48}
                direction="short"
                initialConfig={{
                  accountSize: 500000,
                  riskPercent: 1.5,
                  entryPrice: 74.25,
                  stopLoss: 76.50,
                  takeProfitLevels: [72.00, 70.50, 68.00],
                }}
              />
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Correlation Matrix */}
              <CorrelationMatrix
                size="md"
                showLabels={true}
                interactive={true}
                title="Hedge Instrument Correlations"
              />

              {/* Volatility Surface */}
              <VolatilitySurface
                showMiniChart={true}
                compact={false}
              />
            </div>
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <div className="flex items-center justify-between py-4 border-t border-neutral-800 text-xs text-neutral-500">
          <div className="flex items-center gap-4">
            <span>Last updated: {new Date().toLocaleString()}</span>
            <Badge className="bg-neutral-800 border-neutral-700 text-neutral-400">
              Hedging Suite v2.0
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <span>CME Group Certified</span>
            <ChevronRight className="w-4 h-4" />
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="px-6 pb-6">
        <div className="p-4 bg-neutral-900/30 border border-neutral-800 rounded-lg flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-neutral-500 flex-shrink-0 mt-0.5" />
          <div className="text-xs text-neutral-500">
            <strong className="text-neutral-400">Hedging Disclaimer:</strong> This dashboard provides
            ML-driven hedge recommendations for professional risk management. All hedge ratios and
            recommendations should be validated against your firm&apos;s risk policy. Past performance
            does not guarantee future results. Consult your risk committee before executing.
          </div>
        </div>
      </div>
    </div>
  );
}
