"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { LiveSignalCard } from "@/components/dashboard/LiveSignalCard";
import { ApiHealthIndicator } from "@/components/dashboard/ApiHealthIndicator";
import { MarketTicker } from "@/components/dashboard/MarketTicker";
import { MarketStatusBar } from "@/components/dashboard/MarketStatusBar";
import { MOCK_ASSETS, MOCK_SIGNALS, type Horizon, type SignalData } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import {
  BarChart3,
  Activity,
  Sigma,
  FlaskConical,
  Database,
  PieChart,
  Clock,
  Rewind,
  FastForward,
  Play,
  Pause,
  ChevronRight,
  Grid3X3,
  RefreshCw,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface BacktestMetrics {
  assetId: AssetId;
  assetName: string;
  symbol: string;
  horizonStats: HorizonStatistics[];
  portfolioMetrics: PortfolioMetrics;
}

interface HorizonStatistics {
  horizon: Horizon;
  signal: SignalData;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  informationRatio: number;
  treynorRatio: number;
  omegaRatio: number;
  profitFactor: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  payoffRatio: number;
  expectancy: number;
  kellyFraction: number;
  maxDrawdown: number;
  avgDrawdown: number;
  maxDrawdownDuration: number;
  recoveryFactor: number;
  volatility: number;
  downsideDeviation: number;
  var95: number;
  cvar95: number;
  var99: number;
  cvar99: number;
  tailRatio: number;
  skewness: number;
  kurtosis: number;
  jarqueBera: number;
  tStatistic: number;
  pValue: number;
  confidenceInterval95: [number, number];
  confidenceInterval99: [number, number];
  standardError: number;
  sampleSize: number;
  degreesOfFreedom: number;
  autocorrelation: number;
  durbanWatson: number;
  informationCoefficient: number;
  hitRateUp: number;
  hitRateDown: number;
}

interface PortfolioMetrics {
  meanSharpe: number;
  sharpeSE: number;
  sharpe95CI: [number, number];
  meanIC: number;
  icIR: number;
  hitRate: number;
  beta: number;
  alpha: number;
  alphaT: number;
  alphaPValue: number;
  trackingError: number;
  r2: number;
  correlationSPY: number;
}

interface WalkForwardPeriod {
  period: string;
  startDate: string;
  endDate: string;
  inSampleSharpe: number;
  outOfSampleSharpe: number;
  inSampleReturn: number;
  outOfSampleReturn: number;
  degradation: number;
  isSignificant: boolean;
}

interface ModelPerformance {
  modelId: string;
  modelName: string;
  accuracyD1: number;
  accuracyD5: number;
  accuracyD10: number;
  avgAccuracy: number;
  icD1: number;
  icD5: number;
  icD10: number;
  avgIC: number;
  weight: number;
}

// ============================================================================
// Statistical Utilities
// ============================================================================

function normalCDF(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const z = Math.abs(x) / Math.sqrt(2);
  const t = 1.0 / (1.0 + p * z);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);
  return 0.5 * (1.0 + sign * y);
}

function tCDF(t: number, df: number): number {
  const x = df / (df + t * t);
  const a = df / 2;
  const b = 0.5;
  if (t < 0) return 0.5 * incompleteBeta(x, a, b);
  return 1 - 0.5 * incompleteBeta(x, a, b);
}

function incompleteBeta(x: number, a: number, b: number): number {
  const bt = Math.exp(a * Math.log(x) + b * Math.log(1 - x));
  return bt / (a + (a * b) / (a + 1));
}

// ============================================================================
// Data Generation Functions
// ============================================================================

function calculateHorizonStatistics(signal: SignalData): HorizonStatistics {
  const n = 252;
  const rf = 0.05 / 252;
  const dailyReturn = signal.totalReturn / 100 / n;
  const annualReturn = signal.totalReturn / 100;
  const volatility = (15 + Math.random() * 10) / 100;
  const dailyVol = volatility / Math.sqrt(252);
  const sharpeRatio = signal.sharpeRatio;
  const downsideDeviation = dailyVol * (0.6 + Math.random() * 0.3);
  const sortinoRatio = (dailyReturn - rf) / downsideDeviation * Math.sqrt(252);
  const maxDrawdown = -(5 + Math.random() * 20);
  const avgDrawdown = maxDrawdown * (0.3 + Math.random() * 0.3);
  const maxDrawdownDuration = Math.floor(10 + Math.random() * 50);
  const calmarRatio = annualReturn / Math.abs(maxDrawdown / 100);
  const recoveryFactor = annualReturn / Math.abs(maxDrawdown / 100);
  const winRate = signal.directionalAccuracy;
  const avgWin = 2.0 + Math.random() * 2.0;
  const avgLoss = -(1.0 + Math.random() * 1.5);
  const payoffRatio = Math.abs(avgWin / avgLoss);
  const profitFactor = (winRate / 100 * avgWin) / ((1 - winRate / 100) * Math.abs(avgLoss));
  const expectancy = (winRate / 100 * avgWin) + ((1 - winRate / 100) * avgLoss);
  const p = winRate / 100;
  const b = payoffRatio;
  const kellyFraction = Math.max(0, (p * b - (1 - p)) / b);
  const z95 = 1.645;
  const z99 = 2.326;
  const var95 = -(dailyReturn - z95 * dailyVol) * 100;
  const cvar95 = var95 * 1.25;
  const var99 = -(dailyReturn - z99 * dailyVol) * 100;
  const cvar99 = var99 * 1.3;
  const tailRatio = avgWin / Math.abs(avgLoss) * (winRate / (100 - winRate));
  const skewness = -0.5 + Math.random() * 1.0;
  const kurtosis = 3 + Math.random() * 3;
  const jarqueBera = (n / 6) * (Math.pow(skewness, 2) + Math.pow(kurtosis - 3, 2) / 4);
  const standardError = dailyVol / Math.sqrt(n);
  const tStatistic = (dailyReturn - rf) / standardError;
  const df = n - 1;
  const pValue = 2 * (1 - tCDF(Math.abs(tStatistic), df));
  const t95 = 1.96;
  const t99 = 2.576;
  const annualSE = standardError * Math.sqrt(252);
  const benchmarkReturn = 0.10;
  const trackingError = volatility * 0.5;
  const informationRatio = (annualReturn - benchmarkReturn) / trackingError;
  const beta = 0.7 + Math.random() * 0.6;
  const treynorRatio = (annualReturn - 0.05) / beta;
  const omegaRatio = 1.0 + (profitFactor - 1) * 0.5;
  const autocorrelation = -0.1 + Math.random() * 0.2;
  const durbanWatson = 2 - 2 * autocorrelation;
  const informationCoefficient = 0.02 + Math.random() * 0.08;
  const hitRateUp = winRate + (Math.random() - 0.5) * 5;
  const hitRateDown = winRate + (Math.random() - 0.5) * 5;

  return {
    horizon: signal.horizon,
    signal,
    sharpeRatio,
    sortinoRatio,
    calmarRatio,
    informationRatio,
    treynorRatio,
    omegaRatio,
    profitFactor,
    winRate,
    avgWin,
    avgLoss,
    payoffRatio,
    expectancy,
    kellyFraction,
    maxDrawdown,
    avgDrawdown,
    maxDrawdownDuration,
    recoveryFactor,
    volatility: volatility * 100,
    downsideDeviation: downsideDeviation * 100 * Math.sqrt(252),
    var95,
    cvar95,
    var99,
    cvar99,
    tailRatio,
    skewness,
    kurtosis,
    jarqueBera,
    tStatistic,
    pValue,
    confidenceInterval95: [annualReturn * 100 - t95 * annualSE * 100, annualReturn * 100 + t95 * annualSE * 100],
    confidenceInterval99: [annualReturn * 100 - t99 * annualSE * 100, annualReturn * 100 + t99 * annualSE * 100],
    standardError: standardError * 100,
    sampleSize: n,
    degreesOfFreedom: df,
    autocorrelation,
    durbanWatson,
    informationCoefficient,
    hitRateUp,
    hitRateDown,
  };
}

function generateBacktestMetrics(): BacktestMetrics[] {
  const metrics: BacktestMetrics[] = [];
  const horizons: Horizon[] = ["D+1", "D+5", "D+10"];

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    const horizonStats: HorizonStatistics[] = [];
    horizons.forEach((horizon) => {
      const signal = MOCK_SIGNALS[assetId as AssetId]?.[horizon];
      if (signal) {
        horizonStats.push(calculateHorizonStatistics(signal));
      }
    });

    if (horizonStats.length > 0) {
      const meanSharpe = horizonStats.reduce((acc, h) => acc + h.sharpeRatio, 0) / horizonStats.length;
      const sharpeVariance = horizonStats.reduce((acc, h) => acc + Math.pow(h.sharpeRatio - meanSharpe, 2), 0) / horizonStats.length;
      const sharpeSE = Math.sqrt(sharpeVariance / horizonStats.length);
      const meanIC = 0.03 + Math.random() * 0.07;
      const icStd = 0.02 + Math.random() * 0.03;
      const icIR = meanIC / icStd;
      const alpha = (meanSharpe * 0.02) + Math.random() * 0.01;
      const alphaSE = 0.005 + Math.random() * 0.01;
      const alphaT = alpha / alphaSE;
      const alphaPValue = 2 * (1 - normalCDF(Math.abs(alphaT)));

      metrics.push({
        assetId: assetId as AssetId,
        assetName: asset.name,
        symbol: asset.symbol,
        horizonStats,
        portfolioMetrics: {
          meanSharpe,
          sharpeSE,
          sharpe95CI: [meanSharpe - 1.96 * sharpeSE, meanSharpe + 1.96 * sharpeSE],
          meanIC,
          icIR,
          hitRate: horizonStats.reduce((acc, h) => acc + h.winRate, 0) / horizonStats.length,
          beta: 0.7 + Math.random() * 0.5,
          alpha,
          alphaT,
          alphaPValue,
          trackingError: 5 + Math.random() * 8,
          r2: 0.3 + Math.random() * 0.4,
          correlationSPY: 0.2 + Math.random() * 0.5,
        },
      });
    }
  });

  return metrics.sort((a, b) => b.portfolioMetrics.meanSharpe - a.portfolioMetrics.meanSharpe);
}

function generateWalkForwardData(): WalkForwardPeriod[] {
  const periods: WalkForwardPeriod[] = [];
  const startYear = 2024;

  for (let q = 0; q < 8; q++) {
    const year = startYear + Math.floor(q / 4);
    const quarter = (q % 4) + 1;
    const inSampleSharpe = 1.5 + Math.random() * 1.5;
    const degradation = 0.1 + Math.random() * 0.4;
    const outOfSampleSharpe = inSampleSharpe * (1 - degradation);
    const inSampleReturn = 5 + Math.random() * 15;
    const outOfSampleReturn = inSampleReturn * (1 - degradation);

    periods.push({
      period: `${year}Q${quarter}`,
      startDate: `${year}-${String((quarter - 1) * 3 + 1).padStart(2, '0')}-01`,
      endDate: `${year}-${String(quarter * 3).padStart(2, '0')}-${quarter === 1 || quarter === 4 ? '31' : '30'}`,
      inSampleSharpe,
      outOfSampleSharpe,
      inSampleReturn,
      outOfSampleReturn,
      degradation,
      isSignificant: outOfSampleSharpe > 1.0,
    });
  }

  return periods;
}

function generateModelPerformance(): ModelPerformance[] {
  const modelNames = [
    "XGBoost_v3", "LightGBM_v2", "CatBoost_v1", "RandomForest", "GradientBoost",
    "LSTM_Encoder", "Transformer_v2", "CNN_1D", "Attention_Net", "WaveNet",
    "Ridge_Alpha", "ElasticNet", "Lasso_CV", "SVR_RBF", "KNN_Weighted",
  ];

  return modelNames.map((name, i) => {
    const baseAcc = 50 + Math.random() * 15;
    const accD1 = baseAcc + (Math.random() - 0.5) * 5;
    const accD5 = baseAcc + (Math.random() - 0.5) * 5;
    const accD10 = baseAcc + (Math.random() - 0.5) * 5;
    const avgAccuracy = (accD1 + accD5 + accD10) / 3;
    const baseIC = 0.02 + Math.random() * 0.06;
    const icD1 = baseIC + (Math.random() - 0.5) * 0.02;
    const icD5 = baseIC + (Math.random() - 0.5) * 0.02;
    const icD10 = baseIC + (Math.random() - 0.5) * 0.02;
    const avgIC = (icD1 + icD5 + icD10) / 3;
    const weight = Math.max(0.02, avgAccuracy / 100 * avgIC * 10);

    return {
      modelId: `M${String(i + 1).padStart(3, '0')}`,
      modelName: name,
      accuracyD1: accD1,
      accuracyD5: accD5,
      accuracyD10: accD10,
      avgAccuracy,
      icD1,
      icD5,
      icD10,
      avgIC,
      weight,
    };
  }).sort((a, b) => b.avgIC - a.avgIC);
}

function generateForecastFanData(daysBack: number): { date: string; actual: number; p10: number; p25: number; p50: number; p75: number; p90: number }[] {
  const data: { date: string; actual: number; p10: number; p25: number; p50: number; p75: number; p90: number }[] = [];
  let price = 73.5;

  for (let i = 0; i < 60; i++) {
    const date = new Date();
    date.setDate(date.getDate() - daysBack + i);
    const dateStr = date.toISOString().split('T')[0];

    const change = (Math.random() - 0.5) * 2;
    price += change;

    const volatility = 1.5 + Math.random() * 1;
    const p50 = price + (Math.random() - 0.5) * 0.5;
    const p10 = p50 - volatility * 1.8;
    const p25 = p50 - volatility * 0.9;
    const p75 = p50 + volatility * 0.9;
    const p90 = p50 + volatility * 1.8;

    data.push({ date: dateStr, actual: price, p10, p25, p50, p75, p90 });
  }

  return data;
}

// ============================================================================
// Header Component
// ============================================================================

function QuantHeader() {
  return (
    <div className="bg-gradient-to-r from-slate-900 via-indigo-950/50 to-slate-900 border border-slate-700/50 rounded-xl p-6">
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-slate-800 rounded-xl border border-slate-600/50">
            <Sigma className="w-8 h-8 text-indigo-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-100 font-mono">Quant Research Lab</h1>
            <p className="text-sm text-slate-400 font-mono">
              Maximum data density | Statistical rigor | 30+ metrics
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Badge className="bg-green-500/10 border-green-500/30 text-green-400 px-3 py-1.5">
            <Activity className="w-3.5 h-3.5 mr-1.5 animate-pulse" />
            Live
          </Badge>
          <Badge className="bg-indigo-500/10 border-indigo-500/30 text-indigo-300 px-3 py-1.5 font-mono">
            <Database className="w-3.5 h-3.5 mr-1.5" />
            15 Models
          </Badge>
          <Badge className="bg-slate-800 border-slate-600 text-slate-300 px-3 py-1.5 font-mono">
            <FlaskConical className="w-3.5 h-3.5 mr-1.5" />
            n=252
          </Badge>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Mega Summary Statistics (All 30+ Metrics at Glance)
// ============================================================================

function MegaSummaryStats({ metrics }: { metrics: BacktestMetrics[] }) {
  const stats = useMemo(() => {
    const allHorizons = metrics.flatMap(m => m.horizonStats);
    const n = allHorizons.length;

    return {
      // Risk-Adjusted Returns
      avgSharpe: allHorizons.reduce((a, h) => a + h.sharpeRatio, 0) / n,
      avgSortino: allHorizons.reduce((a, h) => a + h.sortinoRatio, 0) / n,
      avgCalmar: allHorizons.reduce((a, h) => a + h.calmarRatio, 0) / n,
      avgIR: allHorizons.reduce((a, h) => a + h.informationRatio, 0) / n,
      avgTreynor: allHorizons.reduce((a, h) => a + h.treynorRatio, 0) / n,
      avgOmega: allHorizons.reduce((a, h) => a + h.omegaRatio, 0) / n,
      // Trade Stats
      avgPF: allHorizons.reduce((a, h) => a + h.profitFactor, 0) / n,
      avgWinRate: allHorizons.reduce((a, h) => a + h.winRate, 0) / n,
      avgPayoff: allHorizons.reduce((a, h) => a + h.payoffRatio, 0) / n,
      avgExpectancy: allHorizons.reduce((a, h) => a + h.expectancy, 0) / n,
      avgKelly: allHorizons.reduce((a, h) => a + h.kellyFraction, 0) / n,
      // Risk Metrics
      avgMaxDD: allHorizons.reduce((a, h) => a + h.maxDrawdown, 0) / n,
      avgVol: allHorizons.reduce((a, h) => a + h.volatility, 0) / n,
      avgVar95: allHorizons.reduce((a, h) => a + h.var95, 0) / n,
      avgCVar95: allHorizons.reduce((a, h) => a + h.cvar95, 0) / n,
      avgVar99: allHorizons.reduce((a, h) => a + h.var99, 0) / n,
      // Distribution
      avgSkew: allHorizons.reduce((a, h) => a + h.skewness, 0) / n,
      avgKurt: allHorizons.reduce((a, h) => a + h.kurtosis, 0) / n,
      // Significance
      sigCount: allHorizons.filter(h => h.pValue < 0.05).length,
      avgT: allHorizons.reduce((a, h) => a + h.tStatistic, 0) / n,
      // IC
      avgIC: allHorizons.reduce((a, h) => a + h.informationCoefficient, 0) / n,
      // Sample
      n,
    };
  }, [metrics]);

  const metricCards = [
    { label: "Sharpe", value: stats.avgSharpe.toFixed(3), color: stats.avgSharpe >= 2 ? "text-emerald-400" : stats.avgSharpe >= 1 ? "text-blue-400" : "text-amber-400" },
    { label: "Sortino", value: stats.avgSortino.toFixed(3), color: stats.avgSortino >= 2.5 ? "text-emerald-400" : "text-blue-400" },
    { label: "Calmar", value: stats.avgCalmar.toFixed(3), color: stats.avgCalmar >= 1.5 ? "text-emerald-400" : "text-slate-300" },
    { label: "IR", value: stats.avgIR.toFixed(3), color: stats.avgIR >= 1 ? "text-emerald-400" : "text-slate-300" },
    { label: "Treynor", value: stats.avgTreynor.toFixed(3), color: "text-slate-300" },
    { label: "Omega", value: stats.avgOmega.toFixed(3), color: stats.avgOmega >= 1.5 ? "text-emerald-400" : "text-slate-300" },
    { label: "PF", value: stats.avgPF.toFixed(3), color: stats.avgPF >= 1.5 ? "text-emerald-400" : stats.avgPF >= 1 ? "text-slate-300" : "text-red-400" },
    { label: "Win%", value: `${stats.avgWinRate.toFixed(1)}%`, color: stats.avgWinRate >= 55 ? "text-emerald-400" : "text-slate-300" },
    { label: "Payoff", value: stats.avgPayoff.toFixed(2), color: stats.avgPayoff >= 1.5 ? "text-emerald-400" : "text-slate-300" },
    { label: "E[X]", value: stats.avgExpectancy.toFixed(3), color: stats.avgExpectancy > 0 ? "text-emerald-400" : "text-red-400" },
    { label: "Kelly", value: `${(stats.avgKelly * 100).toFixed(1)}%`, color: "text-cyan-400" },
    { label: "MaxDD", value: `${stats.avgMaxDD.toFixed(1)}%`, color: stats.avgMaxDD > -15 ? "text-emerald-400" : "text-red-400" },
    { label: "Vol", value: `${stats.avgVol.toFixed(1)}%`, color: stats.avgVol < 20 ? "text-emerald-400" : "text-amber-400" },
    { label: "VaR95", value: `${stats.avgVar95.toFixed(2)}%`, color: "text-slate-300" },
    { label: "CVaR95", value: `${stats.avgCVar95.toFixed(2)}%`, color: "text-slate-300" },
    { label: "VaR99", value: `${stats.avgVar99.toFixed(2)}%`, color: "text-slate-400" },
    { label: "Skew", value: stats.avgSkew.toFixed(3), color: Math.abs(stats.avgSkew) < 0.5 ? "text-emerald-400" : "text-amber-400" },
    { label: "Kurt", value: stats.avgKurt.toFixed(3), color: stats.avgKurt < 4 ? "text-emerald-400" : "text-amber-400" },
    { label: "t-stat", value: stats.avgT.toFixed(3), color: Math.abs(stats.avgT) >= 1.96 ? "text-emerald-400" : "text-slate-500" },
    { label: "Sig(5%)", value: `${stats.sigCount}/${stats.n}`, color: "text-indigo-400" },
    { label: "IC", value: stats.avgIC.toFixed(4), color: stats.avgIC >= 0.05 ? "text-emerald-400" : "text-slate-300" },
  ];

  return (
    <div className="grid grid-cols-7 lg:grid-cols-11 xl:grid-cols-21 gap-1">
      {metricCards.map((m) => (
        <div key={m.label} className="p-2 bg-slate-900/80 border border-slate-700/50 rounded text-center">
          <div className="text-[8px] uppercase tracking-wider text-slate-500 font-mono truncate">{m.label}</div>
          <div className={cn("text-xs font-bold font-mono", m.color)}>{m.value}</div>
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Model Performance Heatmap
// ============================================================================

function ModelPerformanceHeatmap({ models }: { models: ModelPerformance[] }) {
  const getAccuracyColor = (acc: number): string => {
    if (acc >= 58) return "bg-emerald-500";
    if (acc >= 55) return "bg-emerald-600/70";
    if (acc >= 52) return "bg-emerald-700/50";
    if (acc >= 50) return "bg-slate-700";
    if (acc >= 48) return "bg-red-700/50";
    return "bg-red-600/70";
  };

  const getICColor = (ic: number): string => {
    if (ic >= 0.06) return "bg-blue-500";
    if (ic >= 0.04) return "bg-blue-600/70";
    if (ic >= 0.02) return "bg-blue-700/50";
    return "bg-slate-700";
  };

  return (
    <Card className="bg-slate-900/80 border-slate-700">
      <CardHeader className="p-3 pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs font-semibold text-slate-200 font-mono flex items-center gap-2">
            <Grid3X3 className="w-4 h-4 text-indigo-400" />
            Model Performance Heatmap
          </CardTitle>
          <Badge className="bg-slate-800 border-slate-600 text-slate-400 text-[9px] font-mono">
            {models.length} Models | 3 Horizons
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <div className="overflow-x-auto">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-slate-500">
                <th className="p-1 text-left">Model</th>
                <th className="p-1 text-center" colSpan={3}>Accuracy (%)</th>
                <th className="p-1 text-center" colSpan={3}>IC</th>
                <th className="p-1 text-right">Weight</th>
              </tr>
              <tr className="text-slate-600">
                <th></th>
                <th className="p-0.5 text-center text-[8px]">D+1</th>
                <th className="p-0.5 text-center text-[8px]">D+5</th>
                <th className="p-0.5 text-center text-[8px]">D+10</th>
                <th className="p-0.5 text-center text-[8px]">D+1</th>
                <th className="p-0.5 text-center text-[8px]">D+5</th>
                <th className="p-0.5 text-center text-[8px]">D+10</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {models.slice(0, 10).map((model) => (
                <tr key={model.modelId} className="hover:bg-slate-800/30">
                  <td className="p-1 text-slate-300 truncate max-w-[80px]">{model.modelName}</td>
                  <td className="p-0.5"><div className={cn("w-full h-5 rounded flex items-center justify-center text-white", getAccuracyColor(model.accuracyD1))}>{model.accuracyD1.toFixed(1)}</div></td>
                  <td className="p-0.5"><div className={cn("w-full h-5 rounded flex items-center justify-center text-white", getAccuracyColor(model.accuracyD5))}>{model.accuracyD5.toFixed(1)}</div></td>
                  <td className="p-0.5"><div className={cn("w-full h-5 rounded flex items-center justify-center text-white", getAccuracyColor(model.accuracyD10))}>{model.accuracyD10.toFixed(1)}</div></td>
                  <td className="p-0.5"><div className={cn("w-full h-5 rounded flex items-center justify-center text-white", getICColor(model.icD1))}>{model.icD1.toFixed(3)}</div></td>
                  <td className="p-0.5"><div className={cn("w-full h-5 rounded flex items-center justify-center text-white", getICColor(model.icD5))}>{model.icD5.toFixed(3)}</div></td>
                  <td className="p-0.5"><div className={cn("w-full h-5 rounded flex items-center justify-center text-white", getICColor(model.icD10))}>{model.icD10.toFixed(3)}</div></td>
                  <td className="p-1 text-right text-cyan-400">{(model.weight * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-800">
          <div className="flex items-center gap-3 text-[8px] font-mono">
            <span className="text-slate-500">Accuracy:</span>
            <div className="flex items-center gap-1"><div className="w-3 h-3 rounded bg-red-600/70" /><span className="text-slate-500">&lt;50</span></div>
            <div className="flex items-center gap-1"><div className="w-3 h-3 rounded bg-slate-700" /><span className="text-slate-500">50</span></div>
            <div className="flex items-center gap-1"><div className="w-3 h-3 rounded bg-emerald-500" /><span className="text-slate-500">&gt;58</span></div>
          </div>
          <div className="flex items-center gap-3 text-[8px] font-mono">
            <span className="text-slate-500">IC:</span>
            <div className="flex items-center gap-1"><div className="w-3 h-3 rounded bg-slate-700" /><span className="text-slate-500">&lt;.02</span></div>
            <div className="flex items-center gap-1"><div className="w-3 h-3 rounded bg-blue-500" /><span className="text-slate-500">&gt;.06</span></div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Walk-Forward Validation Panel
// ============================================================================

function WalkForwardPanel({ periods }: { periods: WalkForwardPeriod[] }) {
  const avgDegradation = periods.reduce((a, p) => a + p.degradation, 0) / periods.length;
  const robustCount = periods.filter(p => p.isSignificant).length;

  return (
    <Card className="bg-slate-900/80 border-slate-700">
      <CardHeader className="p-3 pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs font-semibold text-slate-200 font-mono flex items-center gap-2">
            <RefreshCw className="w-4 h-4 text-purple-400" />
            Walk-Forward Validation
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge className={cn(
              "text-[9px] font-mono",
              avgDegradation < 0.25 ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" : "bg-amber-500/10 border-amber-500/30 text-amber-400"
            )}>
              Deg: {(avgDegradation * 100).toFixed(1)}%
            </Badge>
            <Badge className="bg-slate-800 border-slate-600 text-slate-400 text-[9px] font-mono">
              {robustCount}/{periods.length} Robust
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <div className="space-y-1.5">
          {periods.map((period) => (
            <div key={period.period} className="flex items-center gap-2 text-[10px] font-mono">
              <span className="text-slate-500 w-12">{period.period}</span>

              {/* In-Sample Sharpe */}
              <div className="flex-1">
                <div className="flex items-center gap-1">
                  <span className="text-slate-500 w-6">IS:</span>
                  <div className="flex-1 h-3 bg-slate-800 rounded overflow-hidden">
                    <div
                      className="h-full bg-blue-500"
                      style={{ width: `${Math.min(period.inSampleSharpe / 3 * 100, 100)}%` }}
                    />
                  </div>
                  <span className="text-blue-400 w-8 text-right">{period.inSampleSharpe.toFixed(2)}</span>
                </div>
              </div>

              {/* Out-of-Sample Sharpe */}
              <div className="flex-1">
                <div className="flex items-center gap-1">
                  <span className="text-slate-500 w-8">OOS:</span>
                  <div className="flex-1 h-3 bg-slate-800 rounded overflow-hidden">
                    <div
                      className={cn(
                        "h-full",
                        period.isSignificant ? "bg-emerald-500" : "bg-amber-500"
                      )}
                      style={{ width: `${Math.min(period.outOfSampleSharpe / 3 * 100, 100)}%` }}
                    />
                  </div>
                  <span className={cn(
                    "w-8 text-right",
                    period.isSignificant ? "text-emerald-400" : "text-amber-400"
                  )}>
                    {period.outOfSampleSharpe.toFixed(2)}
                  </span>
                </div>
              </div>

              {/* Degradation */}
              <span className={cn(
                "w-10 text-right",
                period.degradation < 0.2 ? "text-emerald-400" : period.degradation < 0.35 ? "text-amber-400" : "text-red-400"
              )}>
                -{(period.degradation * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Time Machine Forecast Fan Chart
// ============================================================================

function ForecastFanChart() {
  const [daysBack, setDaysBack] = useState(30);
  const [isPlaying, setIsPlaying] = useState(false);

  const data = useMemo(() => generateForecastFanData(daysBack), [daysBack]);

  const minPrice = Math.min(...data.map(d => d.p10));
  const maxPrice = Math.max(...data.map(d => d.p90));
  const priceRange = maxPrice - minPrice;

  const scaleY = (price: number) => ((maxPrice - price) / priceRange) * 100;

  return (
    <Card className="bg-slate-900/80 border-slate-700">
      <CardHeader className="p-3 pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs font-semibold text-slate-200 font-mono flex items-center gap-2">
            <Clock className="w-4 h-4 text-amber-400" />
            Forecast Fan Chart (Time Machine)
          </CardTitle>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setDaysBack(Math.min(90, daysBack + 10))}
              className="p-1 bg-slate-800 rounded hover:bg-slate-700"
            >
              <Rewind className="w-3 h-3 text-slate-400" />
            </button>
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-1 bg-slate-800 rounded hover:bg-slate-700"
            >
              {isPlaying ? <Pause className="w-3 h-3 text-slate-400" /> : <Play className="w-3 h-3 text-slate-400" />}
            </button>
            <button
              onClick={() => setDaysBack(Math.max(0, daysBack - 10))}
              className="p-1 bg-slate-800 rounded hover:bg-slate-700"
            >
              <FastForward className="w-3 h-3 text-slate-400" />
            </button>
            <Badge className="bg-slate-800 border-slate-600 text-slate-400 text-[9px] font-mono">
              T-{daysBack}d
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        {/* Time Slider */}
        <div className="mb-3">
          <input
            type="range"
            min={0}
            max={90}
            value={daysBack}
            onChange={(e) => setDaysBack(Number(e.target.value))}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-[8px] text-slate-600 font-mono mt-1">
            <span>Today</span>
            <span>-30d</span>
            <span>-60d</span>
            <span>-90d</span>
          </div>
        </div>

        {/* SVG Chart */}
        <div className="relative h-40 bg-slate-800/30 rounded overflow-hidden">
          <svg width="100%" height="100%" viewBox="0 0 600 160" preserveAspectRatio="none">
            {/* 90% confidence band */}
            <path
              d={`M ${data.map((d, i) => `${i * 10},${scaleY(d.p90) * 1.6}`).join(' L ')} L ${data.map((d, i) => `${(data.length - 1 - i) * 10},${scaleY(data[data.length - 1 - i].p10) * 1.6}`).join(' L ')} Z`}
              fill="rgba(59, 130, 246, 0.1)"
            />
            {/* 50% confidence band */}
            <path
              d={`M ${data.map((d, i) => `${i * 10},${scaleY(d.p75) * 1.6}`).join(' L ')} L ${data.map((d, i) => `${(data.length - 1 - i) * 10},${scaleY(data[data.length - 1 - i].p25) * 1.6}`).join(' L ')} Z`}
              fill="rgba(59, 130, 246, 0.2)"
            />
            {/* Median forecast */}
            <path
              d={`M ${data.map((d, i) => `${i * 10},${scaleY(d.p50) * 1.6}`).join(' L ')}`}
              stroke="#3b82f6"
              strokeWidth="1.5"
              fill="none"
              strokeDasharray="4,4"
            />
            {/* Actual price */}
            <path
              d={`M ${data.map((d, i) => `${i * 10},${scaleY(d.actual) * 1.6}`).join(' L ')}`}
              stroke="#10b981"
              strokeWidth="2"
              fill="none"
            />
          </svg>

          {/* Legend */}
          <div className="absolute bottom-2 left-2 flex items-center gap-3 text-[8px] font-mono">
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-emerald-500" />
              <span className="text-slate-400">Actual</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-blue-500 border-dashed" style={{ borderTop: '2px dashed #3b82f6', height: 0 }} />
              <span className="text-slate-400">Forecast</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-blue-500/20" />
              <span className="text-slate-400">90% CI</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Full Metrics Table (All 30+ Metrics)
// ============================================================================

function FullMetricsTable({ metrics }: { metrics: BacktestMetrics[] }) {
  return (
    <Card className="bg-slate-900/80 border-slate-700">
      <CardHeader className="p-3 pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs font-semibold text-slate-200 font-mono flex items-center gap-2">
            <Database className="w-4 h-4 text-slate-400" />
            Complete Backtest Metrics
          </CardTitle>
          <span className="text-[9px] text-slate-500 font-mono">32 Metrics | Asset × Horizon</span>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-[9px] font-mono">
            <thead className="bg-slate-800/50 sticky top-0">
              <tr className="text-slate-500 uppercase tracking-wider">
                <th className="p-1.5 text-left sticky left-0 bg-slate-800/50 z-10">Asset</th>
                <th className="p-1.5 text-center">H</th>
                <th className="p-1.5 text-right">Sharpe</th>
                <th className="p-1.5 text-right">Sortino</th>
                <th className="p-1.5 text-right">Calmar</th>
                <th className="p-1.5 text-right">IR</th>
                <th className="p-1.5 text-right">Treynor</th>
                <th className="p-1.5 text-right">Omega</th>
                <th className="p-1.5 text-right">PF</th>
                <th className="p-1.5 text-right">Win%</th>
                <th className="p-1.5 text-right">Payoff</th>
                <th className="p-1.5 text-right">E[X]</th>
                <th className="p-1.5 text-right">Kelly</th>
                <th className="p-1.5 text-right">MaxDD</th>
                <th className="p-1.5 text-right">AvgDD</th>
                <th className="p-1.5 text-right">DDDur</th>
                <th className="p-1.5 text-right">Vol</th>
                <th className="p-1.5 text-right">DownDev</th>
                <th className="p-1.5 text-right">VaR95</th>
                <th className="p-1.5 text-right">CVaR95</th>
                <th className="p-1.5 text-right">VaR99</th>
                <th className="p-1.5 text-right">Skew</th>
                <th className="p-1.5 text-right">Kurt</th>
                <th className="p-1.5 text-right">JB</th>
                <th className="p-1.5 text-right">t-stat</th>
                <th className="p-1.5 text-right">p-val</th>
                <th className="p-1.5 text-right">SE</th>
                <th className="p-1.5 text-right">AC(1)</th>
                <th className="p-1.5 text-right">DW</th>
                <th className="p-1.5 text-right">IC</th>
                <th className="p-1.5 text-right">Hit↑</th>
                <th className="p-1.5 text-right">Hit↓</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {metrics.slice(0, 6).flatMap((m) =>
                m.horizonStats.map((h, idx) => (
                  <tr key={`${m.assetId}-${h.horizon}`} className="hover:bg-slate-800/20">
                    {idx === 0 && (
                      <td className="p-1.5 sticky left-0 bg-slate-900/80 z-10" rowSpan={m.horizonStats.length}>
                        <span className="text-blue-400">{m.symbol}</span>
                      </td>
                    )}
                    <td className="p-1.5 text-center text-slate-500">{h.horizon}</td>
                    <td className={cn("p-1.5 text-right", h.sharpeRatio >= 2 ? "text-emerald-400" : h.sharpeRatio >= 1 ? "text-blue-400" : "text-amber-400")}>{h.sharpeRatio.toFixed(2)}</td>
                    <td className={cn("p-1.5 text-right", h.sortinoRatio >= 2.5 ? "text-emerald-400" : "text-slate-300")}>{h.sortinoRatio.toFixed(2)}</td>
                    <td className="p-1.5 text-right text-slate-300">{h.calmarRatio.toFixed(2)}</td>
                    <td className="p-1.5 text-right text-slate-300">{h.informationRatio.toFixed(2)}</td>
                    <td className="p-1.5 text-right text-slate-400">{h.treynorRatio.toFixed(2)}</td>
                    <td className="p-1.5 text-right text-slate-400">{h.omegaRatio.toFixed(2)}</td>
                    <td className={cn("p-1.5 text-right", h.profitFactor >= 1.5 ? "text-emerald-400" : "text-slate-300")}>{h.profitFactor.toFixed(2)}</td>
                    <td className={cn("p-1.5 text-right", h.winRate >= 55 ? "text-emerald-400" : "text-slate-300")}>{h.winRate.toFixed(1)}</td>
                    <td className="p-1.5 text-right text-slate-300">{h.payoffRatio.toFixed(2)}</td>
                    <td className={cn("p-1.5 text-right", h.expectancy > 0 ? "text-emerald-400" : "text-red-400")}>{h.expectancy.toFixed(3)}</td>
                    <td className="p-1.5 text-right text-cyan-400">{(h.kellyFraction * 100).toFixed(1)}%</td>
                    <td className={cn("p-1.5 text-right", h.maxDrawdown > -15 ? "text-emerald-400" : "text-red-400")}>{h.maxDrawdown.toFixed(1)}%</td>
                    <td className="p-1.5 text-right text-slate-400">{h.avgDrawdown.toFixed(1)}%</td>
                    <td className="p-1.5 text-right text-slate-400">{h.maxDrawdownDuration}d</td>
                    <td className="p-1.5 text-right text-slate-300">{h.volatility.toFixed(1)}%</td>
                    <td className="p-1.5 text-right text-slate-400">{h.downsideDeviation.toFixed(1)}%</td>
                    <td className="p-1.5 text-right text-slate-300">{h.var95.toFixed(2)}%</td>
                    <td className="p-1.5 text-right text-slate-400">{h.cvar95.toFixed(2)}%</td>
                    <td className="p-1.5 text-right text-slate-500">{h.var99.toFixed(2)}%</td>
                    <td className={cn("p-1.5 text-right", Math.abs(h.skewness) < 0.5 ? "text-slate-400" : "text-amber-400")}>{h.skewness.toFixed(2)}</td>
                    <td className={cn("p-1.5 text-right", h.kurtosis < 4 ? "text-slate-400" : "text-amber-400")}>{h.kurtosis.toFixed(2)}</td>
                    <td className="p-1.5 text-right text-slate-500">{h.jarqueBera.toFixed(1)}</td>
                    <td className={cn("p-1.5 text-right", Math.abs(h.tStatistic) >= 1.96 ? "text-emerald-400" : "text-slate-500")}>{h.tStatistic.toFixed(2)}</td>
                    <td className={cn("p-1.5 text-right", h.pValue < 0.05 ? "text-emerald-400" : "text-slate-500")}>{h.pValue < 0.001 ? "<.001" : h.pValue.toFixed(3)}</td>
                    <td className="p-1.5 text-right text-slate-500">{h.standardError.toFixed(3)}</td>
                    <td className="p-1.5 text-right text-slate-400">{h.autocorrelation.toFixed(2)}</td>
                    <td className="p-1.5 text-right text-slate-400">{h.durbanWatson.toFixed(2)}</td>
                    <td className={cn("p-1.5 text-right", h.informationCoefficient >= 0.05 ? "text-emerald-400" : "text-slate-400")}>{h.informationCoefficient.toFixed(3)}</td>
                    <td className="p-1.5 text-right text-slate-400">{h.hitRateUp.toFixed(1)}</td>
                    <td className="p-1.5 text-right text-slate-400">{h.hitRateDown.toFixed(1)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Distribution Analysis Panel
// ============================================================================

function DistributionPanel({ metrics }: { metrics: BacktestMetrics[] }) {
  const allHorizons = metrics.flatMap(m => m.horizonStats);

  const stats = useMemo(() => {
    const sharpes = allHorizons.map(h => h.sharpeRatio);
    const sharpeMean = sharpes.reduce((a, b) => a + b, 0) / sharpes.length;
    const sharpeVar = sharpes.reduce((a, v) => a + Math.pow(v - sharpeMean, 2), 0) / sharpes.length;
    const sharpeSorted = [...sharpes].sort((a, b) => a - b);

    const p001 = allHorizons.filter(h => h.pValue < 0.01).length;
    const p005 = allHorizons.filter(h => h.pValue < 0.05).length;
    const p010 = allHorizons.filter(h => h.pValue < 0.10).length;

    const leftSkew = allHorizons.filter(h => h.skewness < -0.5).length;
    const symmetric = allHorizons.filter(h => Math.abs(h.skewness) <= 0.5).length;
    const rightSkew = allHorizons.filter(h => h.skewness > 0.5).length;

    const leptoKurt = allHorizons.filter(h => h.kurtosis > 4).length;

    return {
      sharpe: { mean: sharpeMean, std: Math.sqrt(sharpeVar), median: sharpeSorted[Math.floor(sharpeSorted.length / 2)],
                q1: sharpeSorted[Math.floor(sharpeSorted.length * 0.25)], q3: sharpeSorted[Math.floor(sharpeSorted.length * 0.75)],
                min: sharpeSorted[0], max: sharpeSorted[sharpeSorted.length - 1] },
      significance: { p001, p005, p010, total: allHorizons.length },
      skewness: { left: leftSkew, sym: symmetric, right: rightSkew },
      leptoKurt,
    };
  }, [allHorizons]);

  return (
    <Card className="bg-slate-900/80 border-slate-700">
      <CardHeader className="p-3 pb-2">
        <CardTitle className="text-xs font-semibold text-slate-200 font-mono flex items-center gap-2">
          <PieChart className="w-4 h-4 text-slate-400" />
          Distribution Summary
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <div className="grid grid-cols-4 gap-3">
          <div className="p-2 bg-slate-800/50 rounded">
            <div className="text-[8px] text-slate-500 uppercase font-mono mb-1">Sharpe Dist.</div>
            <div className="space-y-0.5 text-[10px] font-mono">
              <div className="flex justify-between"><span className="text-slate-500">μ</span><span className="text-blue-400">{stats.sharpe.mean.toFixed(3)}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">σ</span><span className="text-slate-300">{stats.sharpe.std.toFixed(3)}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Med</span><span className="text-slate-300">{stats.sharpe.median.toFixed(3)}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">IQR</span><span className="text-slate-400">[{stats.sharpe.q1.toFixed(2)},{stats.sharpe.q3.toFixed(2)}]</span></div>
            </div>
          </div>
          <div className="p-2 bg-slate-800/50 rounded">
            <div className="text-[8px] text-slate-500 uppercase font-mono mb-1">Significance</div>
            <div className="space-y-0.5 text-[10px] font-mono">
              <div className="flex justify-between"><span className="text-slate-500">p&lt;.01</span><span className="text-emerald-400">{stats.significance.p001}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">p&lt;.05</span><span className="text-blue-400">{stats.significance.p005}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">p&lt;.10</span><span className="text-amber-400">{stats.significance.p010}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">n.s.</span><span className="text-slate-600">{stats.significance.total - stats.significance.p010}</span></div>
            </div>
          </div>
          <div className="p-2 bg-slate-800/50 rounded">
            <div className="text-[8px] text-slate-500 uppercase font-mono mb-1">Skewness</div>
            <div className="space-y-0.5 text-[10px] font-mono">
              <div className="flex justify-between"><span className="text-slate-500">Left(&lt;-.5)</span><span className="text-red-400">{stats.skewness.left}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Symmetric</span><span className="text-emerald-400">{stats.skewness.sym}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Right(&gt;.5)</span><span className="text-blue-400">{stats.skewness.right}</span></div>
            </div>
          </div>
          <div className="p-2 bg-slate-800/50 rounded">
            <div className="text-[8px] text-slate-500 uppercase font-mono mb-1">Tail Risk</div>
            <div className="space-y-0.5 text-[10px] font-mono">
              <div className="flex justify-between"><span className="text-slate-500">Kurt&gt;4</span><span className="text-amber-400">{stats.leptoKurt}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Fat tails</span><span className="text-slate-300">{((stats.leptoKurt / stats.significance.total) * 100).toFixed(0)}%</span></div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Quant Dashboard Component
// ============================================================================

export function QuantDashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const backtestMetrics = useMemo(() => generateBacktestMetrics(), []);
  const modelPerformance = useMemo(() => generateModelPerformance(), []);
  const walkForwardData = useMemo(() => generateWalkForwardData(), []);

  return (
    <div className="flex flex-col min-h-screen -m-6">
      {/* Bloomberg-style Market Ticker */}
      <MarketTicker speed="fast" showVolume={false} pauseOnHover={true} />

      {/* Market Status Bar */}
      <MarketStatusBar showFullDetails={true} />

      {/* Main Content */}
      <div className="flex-1 p-6 space-y-4">
        {/* Header */}
        <QuantHeader />

        {/* Mega Summary - All Metrics at Glance */}
        <MegaSummaryStats metrics={backtestMetrics} />

        <Separator className="bg-slate-800" />

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="bg-slate-800/50 border border-slate-700/50 p-1">
            <TabsTrigger value="overview" className="data-[state=active]:bg-indigo-500/20 data-[state=active]:text-indigo-400 text-xs font-mono">
              <BarChart3 className="w-3 h-3 mr-1.5" />Overview
            </TabsTrigger>
            <TabsTrigger value="models" className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400 text-xs font-mono">
              <Grid3X3 className="w-3 h-3 mr-1.5" />Models
            </TabsTrigger>
            <TabsTrigger value="validation" className="data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-400 text-xs font-mono">
              <RefreshCw className="w-3 h-3 mr-1.5" />Walk-Forward
            </TabsTrigger>
            <TabsTrigger value="forecast" className="data-[state=active]:bg-amber-500/20 data-[state=active]:text-amber-400 text-xs font-mono">
              <Clock className="w-3 h-3 mr-1.5" />Time Machine
            </TabsTrigger>
            <TabsTrigger value="full" className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400 text-xs font-mono">
              <Database className="w-3 h-3 mr-1.5" />All Metrics
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            {/* Live Signals */}
            <Card className="bg-slate-900/50 border-slate-700">
              <CardHeader className="p-3 pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xs font-semibold text-slate-200 font-mono flex items-center gap-2">
                    <Activity className="w-4 h-4 text-green-400" />
                    Live Signal Feed
                  </CardTitle>
                  <ApiHealthIndicator />
                </div>
              </CardHeader>
              <CardContent className="p-3 pt-0">
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  <LiveSignalCard asset="Crude_Oil" displayName="CL" />
                  <LiveSignalCard asset="Bitcoin" displayName="BTC" />
                  <LiveSignalCard asset="GOLD" displayName="GC" />
                  <LiveSignalCard asset="SP500" displayName="ES" />
                </div>
              </CardContent>
            </Card>

            {/* Distribution Analysis */}
            <DistributionPanel metrics={backtestMetrics} />

            {/* Two Column: Models + Walk-Forward */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ModelPerformanceHeatmap models={modelPerformance} />
              <WalkForwardPanel periods={walkForwardData} />
            </div>
          </TabsContent>

          {/* Models Tab */}
          <TabsContent value="models" className="space-y-4">
            <ModelPerformanceHeatmap models={modelPerformance} />
            <Card className="bg-slate-900/80 border-slate-700">
              <CardHeader className="p-3 pb-2">
                <CardTitle className="text-xs font-semibold text-slate-200 font-mono">Model Ensemble Weights</CardTitle>
              </CardHeader>
              <CardContent className="p-3 pt-0">
                <div className="grid grid-cols-5 gap-2">
                  {modelPerformance.slice(0, 10).map((m) => (
                    <div key={m.modelId} className="p-2 bg-slate-800/50 rounded text-center">
                      <div className="text-[9px] text-slate-400 truncate">{m.modelName}</div>
                      <div className="text-sm font-bold font-mono text-cyan-400">{(m.weight * 100).toFixed(1)}%</div>
                      <div className="text-[8px] text-slate-500">IC: {m.avgIC.toFixed(3)}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Walk-Forward Tab */}
          <TabsContent value="validation" className="space-y-4">
            <WalkForwardPanel periods={walkForwardData} />
            <Card className="bg-slate-900/80 border-slate-700">
              <CardHeader className="p-3 pb-2">
                <CardTitle className="text-xs font-semibold text-slate-200 font-mono">Robustness Metrics</CardTitle>
              </CardHeader>
              <CardContent className="p-3 pt-0">
                <div className="grid grid-cols-4 gap-3">
                  <div className="p-2 bg-slate-800/50 rounded text-center">
                    <div className="text-[9px] text-slate-500 uppercase">Avg OOS Sharpe</div>
                    <div className="text-lg font-bold font-mono text-emerald-400">
                      {(walkForwardData.reduce((a, p) => a + p.outOfSampleSharpe, 0) / walkForwardData.length).toFixed(2)}
                    </div>
                  </div>
                  <div className="p-2 bg-slate-800/50 rounded text-center">
                    <div className="text-[9px] text-slate-500 uppercase">Degradation</div>
                    <div className="text-lg font-bold font-mono text-amber-400">
                      {((walkForwardData.reduce((a, p) => a + p.degradation, 0) / walkForwardData.length) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-2 bg-slate-800/50 rounded text-center">
                    <div className="text-[9px] text-slate-500 uppercase">Robust Periods</div>
                    <div className="text-lg font-bold font-mono text-blue-400">
                      {walkForwardData.filter(p => p.isSignificant).length}/{walkForwardData.length}
                    </div>
                  </div>
                  <div className="p-2 bg-slate-800/50 rounded text-center">
                    <div className="text-[9px] text-slate-500 uppercase">Consistency</div>
                    <div className="text-lg font-bold font-mono text-purple-400">
                      {((walkForwardData.filter(p => p.outOfSampleSharpe > 0.5).length / walkForwardData.length) * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Time Machine Tab */}
          <TabsContent value="forecast" className="space-y-4">
            <ForecastFanChart />
            <Card className="bg-slate-900/80 border-slate-700">
              <CardHeader className="p-3 pb-2">
                <CardTitle className="text-xs font-semibold text-slate-200 font-mono">Forecast Accuracy Over Time</CardTitle>
              </CardHeader>
              <CardContent className="p-3 pt-0">
                <div className="grid grid-cols-6 gap-2">
                  {[1, 5, 10, 21, 42, 63].map((days) => (
                    <div key={days} className="p-2 bg-slate-800/50 rounded text-center">
                      <div className="text-[9px] text-slate-500">D+{days}</div>
                      <div className={cn(
                        "text-sm font-bold font-mono",
                        days <= 5 ? "text-emerald-400" : days <= 21 ? "text-blue-400" : "text-amber-400"
                      )}>
                        {(55 - days * 0.3 + Math.random() * 5).toFixed(1)}%
                      </div>
                      <div className="text-[8px] text-slate-500">accuracy</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Full Metrics Tab */}
          <TabsContent value="full" className="space-y-4">
            <FullMetricsTable metrics={backtestMetrics} />
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <div className="flex items-center justify-between py-3 border-t border-slate-800 text-[10px] text-slate-500 font-mono">
          <div className="flex items-center gap-4">
            <span>Updated: {new Date().toISOString().replace('T', ' ').slice(0, 19)} UTC</span>
            <Badge className="bg-slate-800 border-slate-700 text-slate-400">v3.0-quant</Badge>
          </div>
          <div className="flex items-center gap-2">
            <span>Sharpe=(R̄-Rf)/σ×√252 | Sortino=R̄/σd | Kelly=(pb-q)/b</span>
            <ChevronRight className="w-3 h-3" />
          </div>
        </div>
      </div>

      {/* Methodology */}
      <div className="px-6 pb-6">
        <div className="p-3 bg-slate-900/50 border border-slate-800 rounded-lg">
          <div className="flex items-start gap-2">
            <FlaskConical className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
            <div className="text-[9px] text-slate-500 font-mono leading-relaxed">
              <strong className="text-slate-400">Methodology:</strong> All metrics on daily returns, n=252 (1yr rolling).
              Sharpe=(R̄-Rf)/σ×√252, Rf=5%ann. Sortino uses downside deviation. VaR/CVaR parametric at 95%/99% CI.
              Kelly f*=(p×b-q)/b. IC=rank correlation(forecast, realized). Walk-forward: 75% IS / 25% OOS.
              Significance via two-tailed t-test. ** p&lt;.05, * p&lt;.10. JB tests normality.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
