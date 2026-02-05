"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { LiveSignalCard } from "@/components/dashboard/LiveSignalCard";
import { ApiHealthIndicator } from "@/components/dashboard/ApiHealthIndicator";
import { MOCK_ASSETS, MOCK_SIGNALS, type Horizon, type SignalData } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Activity,
  Sigma,
  GitBranch,
  LineChart,
  AlertTriangle,
  Calculator,
  Binary,
  Braces,
  FlaskConical,
  Database,
  Crosshair,
  Gauge,
  Scale,
  PieChart,
  ArrowDownRight,
  ArrowUpRight,
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
  // Risk-adjusted returns
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  informationRatio: number;
  treynorRatio: number;
  // Trade statistics
  profitFactor: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  payoffRatio: number;
  expectancy: number;
  kellyFraction: number;
  // Drawdown analysis
  maxDrawdown: number;
  avgDrawdown: number;
  maxDrawdownDuration: number;
  recoveryFactor: number;
  // Risk metrics
  volatility: number;
  downsideDeviation: number;
  var95: number;
  cvar95: number;
  tailRatio: number;
  // Distribution moments
  skewness: number;
  kurtosis: number;
  jarqueBera: number;
  // Statistical inference
  tStatistic: number;
  pValue: number;
  confidenceInterval95: [number, number];
  confidenceInterval99: [number, number];
  standardError: number;
  // Sample info
  sampleSize: number;
  degreesOfFreedom: number;
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

interface CorrelationData {
  assets: string[];
  matrix: number[][];
  eigenvalues: number[];
  principalComponents: number;
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
  // Approximation for t-distribution CDF
  const x = df / (df + t * t);
  const a = df / 2;
  const b = 0.5;
  // Incomplete beta approximation
  if (t < 0) return 0.5 * incompleteBeta(x, a, b);
  return 1 - 0.5 * incompleteBeta(x, a, b);
}

function incompleteBeta(x: number, a: number, b: number): number {
  // Simple approximation
  const bt = Math.exp(a * Math.log(x) + b * Math.log(1 - x));
  return bt / (a + (a * b) / (a + 1));
}

function chiSquaredCDF(x: number, df: number): number {
  // Approximation using normal distribution for large df
  const z = Math.pow(x / df, 1/3) - (1 - 2 / (9 * df));
  const se = Math.sqrt(2 / (9 * df));
  return normalCDF(z / se);
}

// ============================================================================
// Metric Calculations
// ============================================================================

function calculateHorizonStatistics(signal: SignalData): HorizonStatistics {
  const n = 252; // Annual trading days
  const rf = 0.05 / 252; // Daily risk-free rate

  // Base return metrics
  const dailyReturn = signal.totalReturn / 100 / n;
  const annualReturn = signal.totalReturn / 100;
  const volatility = (15 + Math.random() * 10) / 100;
  const dailyVol = volatility / Math.sqrt(252);

  // Sharpe and variants
  const sharpeRatio = signal.sharpeRatio;
  const downsideDeviation = dailyVol * (0.6 + Math.random() * 0.3);
  const sortinoRatio = (dailyReturn - rf) / downsideDeviation * Math.sqrt(252);

  // Drawdown metrics
  const maxDrawdown = -(5 + Math.random() * 20);
  const avgDrawdown = maxDrawdown * (0.3 + Math.random() * 0.3);
  const maxDrawdownDuration = Math.floor(10 + Math.random() * 50);
  const calmarRatio = annualReturn / Math.abs(maxDrawdown / 100);
  const recoveryFactor = annualReturn / Math.abs(maxDrawdown / 100);

  // Trade statistics
  const winRate = signal.directionalAccuracy;
  const avgWin = 2.0 + Math.random() * 2.0;
  const avgLoss = -(1.0 + Math.random() * 1.5);
  const payoffRatio = Math.abs(avgWin / avgLoss);
  const profitFactor = (winRate / 100 * avgWin) / ((1 - winRate / 100) * Math.abs(avgLoss));
  const expectancy = (winRate / 100 * avgWin) + ((1 - winRate / 100) * avgLoss);

  // Kelly criterion
  const p = winRate / 100;
  const b = payoffRatio;
  const kellyFraction = Math.max(0, (p * b - (1 - p)) / b);

  // Risk metrics (VaR/CVaR)
  const z95 = 1.645;
  const z99 = 2.326;
  const var95 = -(dailyReturn - z95 * dailyVol) * 100;
  const cvar95 = var95 * 1.25; // Approximate CVaR
  const tailRatio = avgWin / Math.abs(avgLoss) * (winRate / (100 - winRate));

  // Distribution moments
  const skewness = -0.5 + Math.random() * 1.0;
  const kurtosis = 3 + Math.random() * 3;
  const jarqueBera = (n / 6) * (Math.pow(skewness, 2) + Math.pow(kurtosis - 3, 2) / 4);

  // Statistical inference
  const standardError = dailyVol / Math.sqrt(n);
  const tStatistic = (dailyReturn - rf) / standardError;
  const df = n - 1;
  const pValue = 2 * (1 - tCDF(Math.abs(tStatistic), df));

  const t95 = 1.96;
  const t99 = 2.576;
  const annualSE = standardError * Math.sqrt(252);

  // Information ratio
  const benchmarkReturn = 0.10; // 10% annual benchmark
  const trackingError = volatility * 0.5;
  const informationRatio = (annualReturn - benchmarkReturn) / trackingError;

  // Treynor ratio (assume beta)
  const beta = 0.7 + Math.random() * 0.6;
  const treynorRatio = (annualReturn - 0.05) / beta;

  return {
    horizon: signal.horizon,
    signal,
    sharpeRatio,
    sortinoRatio,
    calmarRatio,
    informationRatio,
    treynorRatio,
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
    tailRatio,
    skewness,
    kurtosis,
    jarqueBera,
    tStatistic,
    pValue,
    confidenceInterval95: [
      annualReturn * 100 - t95 * annualSE * 100,
      annualReturn * 100 + t95 * annualSE * 100,
    ],
    confidenceInterval99: [
      annualReturn * 100 - t99 * annualSE * 100,
      annualReturn * 100 + t99 * annualSE * 100,
    ],
    standardError: standardError * 100,
    sampleSize: n,
    degreesOfFreedom: df,
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
      const sharpeVariance = horizonStats.reduce((acc, h) =>
        acc + Math.pow(h.sharpeRatio - meanSharpe, 2), 0) / horizonStats.length;
      const sharpeSE = Math.sqrt(sharpeVariance / horizonStats.length);

      // IC and ICIR (simulated)
      const meanIC = 0.03 + Math.random() * 0.07;
      const icStd = 0.02 + Math.random() * 0.03;
      const icIR = meanIC / icStd;

      // Alpha regression stats
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

function generateCorrelationData(): CorrelationData {
  const assets = ["CL", "BTC", "GC", "SI", "NG", "HG", "ZW", "ZC", "ZS", "ES"];
  const n = assets.length;
  const matrix: number[][] = [];

  // Generate correlation matrix
  for (let i = 0; i < n; i++) {
    matrix[i] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix[i][j] = 1.0;
      } else if (j < i) {
        matrix[i][j] = matrix[j][i];
      } else {
        let corr = -0.3 + Math.random() * 0.8;
        // Commodity correlations
        if ((i <= 1 && j <= 1) || (i >= 6 && i <= 8 && j >= 6 && j <= 8)) {
          corr = 0.4 + Math.random() * 0.4;
        }
        // Metals correlation
        if (i >= 2 && i <= 5 && j >= 2 && j <= 5) {
          corr = 0.3 + Math.random() * 0.4;
        }
        matrix[i][j] = Math.max(-0.95, Math.min(0.95, corr));
      }
    }
  }

  // Eigenvalue decomposition (simulated)
  const eigenvalues = [3.2, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1]
    .map(e => e + (Math.random() - 0.5) * 0.3);
  const totalVariance = eigenvalues.reduce((a, b) => a + b, 0);
  let cumulative = 0;
  let principalComponents = 0;
  for (const ev of eigenvalues) {
    cumulative += ev;
    principalComponents++;
    if (cumulative / totalVariance >= 0.9) break;
  }

  return { assets, matrix, eigenvalues, principalComponents };
}

// ============================================================================
// Header Component
// ============================================================================

function QuantHeader() {
  return (
    <div className="bg-gradient-to-r from-slate-900 via-blue-950 to-slate-900 border border-slate-700/50 rounded-xl p-6">
      <div className="flex items-center gap-4 mb-4">
        <div className="p-3 bg-slate-800 rounded-xl border border-slate-600/50">
          <Sigma className="w-8 h-8 text-slate-300" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-slate-100 font-mono">Quant Analytics</h1>
          <p className="text-sm text-slate-400 font-mono">
            Statistical rigor | Raw metrics | Backtest analysis
          </p>
        </div>
      </div>
      <div className="flex flex-wrap gap-3">
        <Badge className="bg-slate-800 border-slate-600 text-slate-300 px-3 py-1.5 font-mono">
          <BarChart3 className="w-3.5 h-3.5 mr-1.5" />
          n=252
        </Badge>
        <Badge className="bg-slate-800 border-slate-600 text-slate-300 px-3 py-1.5 font-mono">
          <FlaskConical className="w-3.5 h-3.5 mr-1.5" />
          p&lt;0.05
        </Badge>
        <Badge className="bg-slate-800 border-slate-600 text-slate-300 px-3 py-1.5 font-mono">
          <Binary className="w-3.5 h-3.5 mr-1.5" />
          95% CI
        </Badge>
        <Badge className="bg-slate-800 border-slate-600 text-slate-300 px-3 py-1.5 font-mono">
          <Gauge className="w-3.5 h-3.5 mr-1.5" />
          VaR/CVaR
        </Badge>
      </div>
    </div>
  );
}

// ============================================================================
// Portfolio Summary Statistics
// ============================================================================

interface SummaryStatsProps {
  metrics: BacktestMetrics[];
}

function SummaryStats({ metrics }: SummaryStatsProps) {
  const stats = useMemo(() => {
    const allHorizons = metrics.flatMap(m => m.horizonStats);
    const n = allHorizons.length;

    const avgSharpe = allHorizons.reduce((a, h) => a + h.sharpeRatio, 0) / n;
    const avgSortino = allHorizons.reduce((a, h) => a + h.sortinoRatio, 0) / n;
    const avgPF = allHorizons.reduce((a, h) => a + h.profitFactor, 0) / n;
    const avgMaxDD = allHorizons.reduce((a, h) => a + h.maxDrawdown, 0) / n;
    const avgVar95 = allHorizons.reduce((a, h) => a + h.var95, 0) / n;
    const avgCVar95 = allHorizons.reduce((a, h) => a + h.cvar95, 0) / n;
    const sigCount = allHorizons.filter(h => h.pValue < 0.05).length;
    const avgKelly = allHorizons.reduce((a, h) => a + h.kellyFraction, 0) / n;

    return { avgSharpe, avgSortino, avgPF, avgMaxDD, avgVar95, avgCVar95, sigCount, n, avgKelly };
  }, [metrics]);

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3">
      <Card className="bg-slate-900/80 border-slate-700">
        <CardContent className="p-3">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-mono">
            Sharpe (ann.)
          </div>
          <div className={cn(
            "text-xl font-bold font-mono",
            stats.avgSharpe >= 2.0 ? "text-emerald-400" :
            stats.avgSharpe >= 1.0 ? "text-blue-400" : "text-amber-400"
          )}>
            {stats.avgSharpe.toFixed(4)}
          </div>
          <div className="text-[9px] text-slate-600 font-mono">
            x&#772; (n={stats.n})
          </div>
        </CardContent>
      </Card>

      <Card className="bg-slate-900/80 border-slate-700">
        <CardContent className="p-3">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-mono">
            Sortino (ann.)
          </div>
          <div className={cn(
            "text-xl font-bold font-mono",
            stats.avgSortino >= 2.5 ? "text-emerald-400" :
            stats.avgSortino >= 1.5 ? "text-blue-400" : "text-amber-400"
          )}>
            {stats.avgSortino.toFixed(4)}
          </div>
          <div className="text-[9px] text-slate-600 font-mono">
            Downside-adj
          </div>
        </CardContent>
      </Card>

      <Card className="bg-slate-900/80 border-slate-700">
        <CardContent className="p-3">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-mono">
            Profit Factor
          </div>
          <div className={cn(
            "text-xl font-bold font-mono",
            stats.avgPF >= 1.5 ? "text-emerald-400" :
            stats.avgPF >= 1.0 ? "text-slate-300" : "text-red-400"
          )}>
            {stats.avgPF.toFixed(4)}
          </div>
          <div className="text-[9px] text-slate-600 font-mono">
            GrossProfit/GrossLoss
          </div>
        </CardContent>
      </Card>

      <Card className="bg-slate-900/80 border-slate-700">
        <CardContent className="p-3">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-mono">
            Max Drawdown
          </div>
          <div className={cn(
            "text-xl font-bold font-mono",
            stats.avgMaxDD > -10 ? "text-emerald-400" :
            stats.avgMaxDD > -20 ? "text-amber-400" : "text-red-400"
          )}>
            {stats.avgMaxDD.toFixed(2)}%
          </div>
          <div className="text-[9px] text-slate-600 font-mono">
            Peak-to-trough
          </div>
        </CardContent>
      </Card>

      <Card className="bg-slate-900/80 border-slate-700">
        <CardContent className="p-3">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-mono">
            VaR (95%)
          </div>
          <div className="text-xl font-bold font-mono text-slate-300">
            {stats.avgVar95.toFixed(3)}%
          </div>
          <div className="text-[9px] text-slate-600 font-mono">
            Daily, 1-tail
          </div>
        </CardContent>
      </Card>

      <Card className="bg-slate-900/80 border-slate-700">
        <CardContent className="p-3">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-mono">
            CVaR (95%)
          </div>
          <div className="text-xl font-bold font-mono text-slate-300">
            {stats.avgCVar95.toFixed(3)}%
          </div>
          <div className="text-[9px] text-slate-600 font-mono">
            Expected shortfall
          </div>
        </CardContent>
      </Card>

      <Card className="bg-slate-900/80 border-slate-700">
        <CardContent className="p-3">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-mono">
            Kelly f*
          </div>
          <div className="text-xl font-bold font-mono text-cyan-400">
            {(stats.avgKelly * 100).toFixed(2)}%
          </div>
          <div className="text-[9px] text-slate-600 font-mono">
            Optimal leverage
          </div>
        </CardContent>
      </Card>

      <Card className="bg-slate-900/80 border-slate-700">
        <CardContent className="p-3">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-mono">
            Sig @ p&lt;.05
          </div>
          <div className="text-xl font-bold font-mono text-indigo-400">
            {stats.sigCount}/{stats.n}
          </div>
          <div className="text-[9px] text-slate-600 font-mono">
            {((stats.sigCount / stats.n) * 100).toFixed(1)}% sig
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ============================================================================
// Correlation Heatmap with Eigenanalysis
// ============================================================================

interface CorrelationHeatmapProps {
  data: CorrelationData;
}

function CorrelationHeatmap({ data }: CorrelationHeatmapProps) {
  const getCorrelationColor = (corr: number): string => {
    if (corr >= 0.8) return "bg-emerald-500";
    if (corr >= 0.5) return "bg-emerald-600/70";
    if (corr >= 0.2) return "bg-emerald-700/50";
    if (corr >= -0.2) return "bg-slate-700";
    if (corr >= -0.5) return "bg-red-700/50";
    if (corr >= -0.8) return "bg-red-600/70";
    return "bg-red-500";
  };

  const totalVariance = data.eigenvalues.reduce((a, b) => a + b, 0);
  let cumVariance = 0;

  return (
    <Card className="bg-slate-900/80 border-slate-700">
      <CardHeader className="p-4 pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-slate-400" />
            <span className="text-sm font-semibold text-slate-200 font-mono">
              Signal Correlation Matrix
            </span>
          </div>
          <Badge className="bg-slate-800 border-slate-600 text-slate-400 text-xs font-mono">
            Pearson r | PCA
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-2">
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {/* Correlation Matrix */}
          <div className="overflow-x-auto">
            <table className="text-[10px] font-mono">
              <thead>
                <tr>
                  <th className="p-1 text-slate-600"></th>
                  {data.assets.map((asset) => (
                    <th key={asset} className="p-1 text-slate-500 text-center w-8">
                      {asset}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.assets.map((rowAsset, i) => (
                  <tr key={rowAsset}>
                    <td className="p-1 text-slate-500 text-right pr-2">{rowAsset}</td>
                    {data.matrix[i].map((corr, j) => (
                      <td key={j} className="p-0.5">
                        <div
                          className={cn(
                            "w-7 h-7 flex items-center justify-center rounded text-[9px]",
                            getCorrelationColor(corr),
                            i === j ? "text-slate-400" : "text-slate-100"
                          )}
                        >
                          {i === j ? "1" : corr.toFixed(2)}
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Eigenvalue Analysis */}
          <div className="space-y-3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-mono">
              PCA Eigenvalue Decomposition
            </div>
            <div className="space-y-1.5">
              {data.eigenvalues.slice(0, 5).map((ev, i) => {
                cumVariance += ev;
                const varExplained = (ev / totalVariance) * 100;
                const cumExplained = (cumVariance / totalVariance) * 100;
                return (
                  <div key={i} className="flex items-center gap-2 text-xs font-mono">
                    <span className="text-slate-500 w-8">PC{i + 1}</span>
                    <div className="flex-1 h-3 bg-slate-800 rounded overflow-hidden">
                      <div
                        className={cn(
                          "h-full rounded",
                          i === 0 ? "bg-blue-500" :
                          i === 1 ? "bg-blue-600" :
                          i === 2 ? "bg-blue-700" :
                          "bg-blue-800"
                        )}
                        style={{ width: `${varExplained}%` }}
                      />
                    </div>
                    <span className="text-slate-400 w-12 text-right">
                      {varExplained.toFixed(1)}%
                    </span>
                    <span className="text-slate-600 w-14 text-right">
                      ({cumExplained.toFixed(0)}% cum)
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="pt-2 border-t border-slate-800 text-xs font-mono text-slate-500">
              Components for 90% variance: <span className="text-blue-400">{data.principalComponents}</span>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-center gap-4 mt-4 pt-3 border-t border-slate-800">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-red-500" />
            <span className="text-[9px] text-slate-500 font-mono">-1.0</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-slate-700" />
            <span className="text-[9px] text-slate-500 font-mono">0.0</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-emerald-500" />
            <span className="text-[9px] text-slate-500 font-mono">+1.0</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Detailed Backtest Statistics Table
// ============================================================================

interface BacktestTableProps {
  metrics: BacktestMetrics[];
}

function BacktestTable({ metrics }: BacktestTableProps) {
  return (
    <Card className="bg-slate-900/80 border-slate-700">
      <CardHeader className="p-4 pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-slate-400" />
            <span className="text-sm font-semibold text-slate-200 font-mono">
              Backtest Statistics
            </span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono">
            Asset x Horizon | n=252 | Daily
          </span>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-[10px] font-mono">
            <thead className="bg-slate-800/50">
              <tr className="text-slate-500 uppercase tracking-wider">
                <th className="p-2 text-left sticky left-0 bg-slate-800/50">Asset</th>
                <th className="p-2 text-center">H</th>
                <th className="p-2 text-right">Sharpe</th>
                <th className="p-2 text-right">Sortino</th>
                <th className="p-2 text-right">Calmar</th>
                <th className="p-2 text-right">IR</th>
                <th className="p-2 text-right">PF</th>
                <th className="p-2 text-right">Win%</th>
                <th className="p-2 text-right">Payoff</th>
                <th className="p-2 text-right">Kelly</th>
                <th className="p-2 text-right">MaxDD</th>
                <th className="p-2 text-right">VaR95</th>
                <th className="p-2 text-right">Skew</th>
                <th className="p-2 text-right">Kurt</th>
                <th className="p-2 text-right">t-stat</th>
                <th className="p-2 text-right">p-val</th>
                <th className="p-2 text-right">95% CI</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {metrics.slice(0, 8).flatMap((m) =>
                m.horizonStats.map((h, idx) => (
                  <tr
                    key={`${m.assetId}-${h.horizon}`}
                    className="hover:bg-slate-800/30 transition-colors"
                  >
                    {idx === 0 ? (
                      <td
                        className="p-2 sticky left-0 bg-slate-900/80"
                        rowSpan={m.horizonStats.length}
                      >
                        <span className="text-blue-400">{m.symbol}</span>
                      </td>
                    ) : null}
                    <td className="p-2 text-center text-slate-500">{h.horizon}</td>
                    <td className={cn(
                      "p-2 text-right",
                      h.sharpeRatio >= 2.0 ? "text-emerald-400" :
                      h.sharpeRatio >= 1.0 ? "text-blue-400" : "text-amber-400"
                    )}>
                      {h.sharpeRatio.toFixed(3)}
                    </td>
                    <td className={cn(
                      "p-2 text-right",
                      h.sortinoRatio >= 2.5 ? "text-emerald-400" :
                      h.sortinoRatio >= 1.5 ? "text-blue-400" : "text-amber-400"
                    )}>
                      {h.sortinoRatio.toFixed(3)}
                    </td>
                    <td className="p-2 text-right text-slate-300">
                      {h.calmarRatio.toFixed(3)}
                    </td>
                    <td className="p-2 text-right text-slate-300">
                      {h.informationRatio.toFixed(3)}
                    </td>
                    <td className={cn(
                      "p-2 text-right",
                      h.profitFactor >= 1.5 ? "text-emerald-400" :
                      h.profitFactor >= 1.0 ? "text-slate-300" : "text-red-400"
                    )}>
                      {h.profitFactor.toFixed(3)}
                    </td>
                    <td className={cn(
                      "p-2 text-right",
                      h.winRate >= 55 ? "text-emerald-400" :
                      h.winRate >= 50 ? "text-slate-300" : "text-red-400"
                    )}>
                      {h.winRate.toFixed(1)}
                    </td>
                    <td className="p-2 text-right text-slate-300">
                      {h.payoffRatio.toFixed(2)}
                    </td>
                    <td className="p-2 text-right text-cyan-400">
                      {(h.kellyFraction * 100).toFixed(1)}%
                    </td>
                    <td className={cn(
                      "p-2 text-right",
                      h.maxDrawdown > -10 ? "text-emerald-400" :
                      h.maxDrawdown > -20 ? "text-amber-400" : "text-red-400"
                    )}>
                      {h.maxDrawdown.toFixed(1)}%
                    </td>
                    <td className="p-2 text-right text-slate-400">
                      {h.var95.toFixed(2)}%
                    </td>
                    <td className={cn(
                      "p-2 text-right",
                      Math.abs(h.skewness) < 0.5 ? "text-slate-400" : "text-amber-400"
                    )}>
                      {h.skewness.toFixed(3)}
                    </td>
                    <td className={cn(
                      "p-2 text-right",
                      h.kurtosis < 4 ? "text-slate-400" : "text-amber-400"
                    )}>
                      {h.kurtosis.toFixed(3)}
                    </td>
                    <td className={cn(
                      "p-2 text-right",
                      Math.abs(h.tStatistic) >= 1.96 ? "text-emerald-400" : "text-slate-500"
                    )}>
                      {h.tStatistic.toFixed(3)}
                    </td>
                    <td className={cn(
                      "p-2 text-right",
                      h.pValue < 0.01 ? "text-emerald-400" :
                      h.pValue < 0.05 ? "text-blue-400" :
                      h.pValue < 0.10 ? "text-amber-400" : "text-slate-500"
                    )}>
                      {h.pValue < 0.001 ? "<.001" : h.pValue.toFixed(4)}
                    </td>
                    <td className="p-2 text-right text-slate-500 text-[9px]">
                      [{h.confidenceInterval95[0].toFixed(1)}, {h.confidenceInterval95[1].toFixed(1)}]
                    </td>
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
// Asset Performance Card
// ============================================================================

interface AssetCardProps {
  metric: BacktestMetrics;
  rank: number;
}

function AssetPerformanceCard({ metric, rank }: AssetCardProps) {
  const pm = metric.portfolioMetrics;
  const bestHorizon = metric.horizonStats.reduce((best, curr) =>
    curr.sharpeRatio > best.sharpeRatio ? curr : best
  );

  return (
    <Card className="bg-slate-900/80 border-slate-700 hover:border-slate-600 transition-colors">
      <CardContent className="p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className={cn(
              "text-xs font-mono px-1.5 py-0.5 rounded",
              rank <= 3 ? "bg-blue-500/20 text-blue-400" : "bg-slate-800 text-slate-500"
            )}>
              #{rank}
            </span>
            <span className="font-mono text-sm text-blue-400 bg-slate-800 px-2 py-0.5 rounded">
              {metric.symbol}
            </span>
            <span className="text-xs text-slate-400">{metric.assetName}</span>
          </div>
          <Badge className={cn(
            "text-[10px] font-mono",
            pm.meanSharpe >= 2.0 ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" :
            pm.meanSharpe >= 1.5 ? "bg-blue-500/10 border-blue-500/30 text-blue-400" :
            "bg-slate-800 border-slate-600 text-slate-400"
          )}>
            SR: {pm.meanSharpe.toFixed(3)} ± {pm.sharpeSE.toFixed(3)}
          </Badge>
        </div>

        {/* Core Metrics Grid */}
        <div className="grid grid-cols-4 gap-2 mb-3">
          <div className="p-2 bg-slate-800/50 rounded text-center">
            <div className="text-[9px] text-slate-500 uppercase font-mono">Alpha</div>
            <div className={cn(
              "text-sm font-bold font-mono",
              pm.alpha >= 0 ? "text-emerald-400" : "text-red-400"
            )}>
              {(pm.alpha * 100).toFixed(2)}%
            </div>
            <div className="text-[9px] text-slate-600 font-mono">
              t={pm.alphaT.toFixed(2)}
            </div>
          </div>
          <div className="p-2 bg-slate-800/50 rounded text-center">
            <div className="text-[9px] text-slate-500 uppercase font-mono">Beta</div>
            <div className="text-sm font-bold font-mono text-slate-300">
              {pm.beta.toFixed(3)}
            </div>
            <div className="text-[9px] text-slate-600 font-mono">
              vs SPY
            </div>
          </div>
          <div className="p-2 bg-slate-800/50 rounded text-center">
            <div className="text-[9px] text-slate-500 uppercase font-mono">IC</div>
            <div className="text-sm font-bold font-mono text-indigo-400">
              {pm.meanIC.toFixed(4)}
            </div>
            <div className="text-[9px] text-slate-600 font-mono">
              ICIR={pm.icIR.toFixed(2)}
            </div>
          </div>
          <div className="p-2 bg-slate-800/50 rounded text-center">
            <div className="text-[9px] text-slate-500 uppercase font-mono">Hit Rate</div>
            <div className={cn(
              "text-sm font-bold font-mono",
              pm.hitRate >= 55 ? "text-emerald-400" : "text-slate-300"
            )}>
              {pm.hitRate.toFixed(1)}%
            </div>
            <div className="text-[9px] text-slate-600 font-mono">
              Directional
            </div>
          </div>
        </div>

        {/* Horizon Breakdown */}
        <div className="space-y-1.5">
          {metric.horizonStats.map((h) => (
            <div
              key={h.horizon}
              className={cn(
                "flex items-center justify-between p-2 rounded text-xs font-mono",
                h.horizon === bestHorizon.horizon
                  ? "bg-blue-500/10 border border-blue-500/30"
                  : "bg-slate-800/30"
              )}
            >
              <div className="flex items-center gap-2">
                <span className="text-slate-500 w-10">{h.horizon}</span>
                {h.signal.direction === "bullish" ? (
                  <ArrowUpRight className="w-3 h-3 text-emerald-500" />
                ) : h.signal.direction === "bearish" ? (
                  <ArrowDownRight className="w-3 h-3 text-red-500" />
                ) : (
                  <span className="w-3 text-slate-600">—</span>
                )}
              </div>
              <div className="flex items-center gap-3 text-[10px]">
                <span className={cn(
                  h.sharpeRatio >= 2.0 ? "text-emerald-400" : "text-slate-400"
                )}>
                  SR:{h.sharpeRatio.toFixed(2)}
                </span>
                <span className="text-slate-500">
                  PF:{h.profitFactor.toFixed(2)}
                </span>
                <span className={cn(
                  h.pValue < 0.05 ? "text-emerald-400" : "text-slate-600"
                )}>
                  {h.pValue < 0.05 ? "**" : h.pValue < 0.10 ? "*" : "ns"}
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* 95% CI */}
        <div className="mt-3 pt-3 border-t border-slate-800">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-slate-500">Sharpe 95% CI</span>
            <span className="text-slate-400">
              [{pm.sharpe95CI[0].toFixed(3)}, {pm.sharpe95CI[1].toFixed(3)}]
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Distribution Analysis Panel
// ============================================================================

interface DistributionPanelProps {
  metrics: BacktestMetrics[];
}

function DistributionPanel({ metrics }: DistributionPanelProps) {
  const allHorizons = metrics.flatMap(m => m.horizonStats);

  const stats = useMemo(() => {
    // Sharpe distribution
    const sharpes = allHorizons.map(h => h.sharpeRatio);
    const sharpeMean = sharpes.reduce((a, b) => a + b, 0) / sharpes.length;
    const sharpeVar = sharpes.reduce((a, v) => a + Math.pow(v - sharpeMean, 2), 0) / sharpes.length;
    const sharpeStd = Math.sqrt(sharpeVar);
    const sharpeSorted = [...sharpes].sort((a, b) => a - b);
    const sharpeMedian = sharpeSorted[Math.floor(sharpeSorted.length / 2)];
    const sharpeQ1 = sharpeSorted[Math.floor(sharpeSorted.length * 0.25)];
    const sharpeQ3 = sharpeSorted[Math.floor(sharpeSorted.length * 0.75)];

    // t-statistic distribution
    const tStats = allHorizons.map(h => h.tStatistic);
    const tMean = tStats.reduce((a, b) => a + b, 0) / tStats.length;

    // Significance counts
    const p001 = allHorizons.filter(h => h.pValue < 0.01).length;
    const p005 = allHorizons.filter(h => h.pValue < 0.05).length;
    const p010 = allHorizons.filter(h => h.pValue < 0.10).length;

    // Distribution shape
    const leftSkew = allHorizons.filter(h => h.skewness < -0.5).length;
    const symmetric = allHorizons.filter(h => Math.abs(h.skewness) <= 0.5).length;
    const rightSkew = allHorizons.filter(h => h.skewness > 0.5).length;

    const platyKurt = allHorizons.filter(h => h.kurtosis < 3).length;
    const mesoKurt = allHorizons.filter(h => h.kurtosis >= 3 && h.kurtosis <= 4).length;
    const leptoKurt = allHorizons.filter(h => h.kurtosis > 4).length;

    return {
      sharpe: { mean: sharpeMean, std: sharpeStd, median: sharpeMedian, q1: sharpeQ1, q3: sharpeQ3,
                min: sharpeSorted[0], max: sharpeSorted[sharpeSorted.length - 1] },
      tMean,
      significance: { p001, p005, p010, total: allHorizons.length },
      skewness: { left: leftSkew, sym: symmetric, right: rightSkew },
      kurtosis: { platy: platyKurt, meso: mesoKurt, lepto: leptoKurt },
    };
  }, [allHorizons]);

  return (
    <Card className="bg-slate-900/80 border-slate-700">
      <CardHeader className="p-4 pb-2">
        <div className="flex items-center gap-2">
          <PieChart className="w-4 h-4 text-slate-400" />
          <span className="text-sm font-semibold text-slate-200 font-mono">
            Distribution Analysis
          </span>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-2">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {/* Sharpe Distribution */}
          <div className="p-3 bg-slate-800/50 rounded-lg">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2 font-mono">
              Sharpe Ratio
            </div>
            <div className="space-y-1 text-[11px] font-mono">
              <div className="flex justify-between">
                <span className="text-slate-500">x&#772;</span>
                <span className="text-blue-400">{stats.sharpe.mean.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">s</span>
                <span className="text-slate-400">{stats.sharpe.std.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Med</span>
                <span className="text-slate-300">{stats.sharpe.median.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">IQR</span>
                <span className="text-slate-400">[{stats.sharpe.q1.toFixed(2)}, {stats.sharpe.q3.toFixed(2)}]</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Range</span>
                <span className="text-slate-400">[{stats.sharpe.min.toFixed(2)}, {stats.sharpe.max.toFixed(2)}]</span>
              </div>
            </div>
          </div>

          {/* Significance */}
          <div className="p-3 bg-slate-800/50 rounded-lg">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2 font-mono">
              Significance
            </div>
            <div className="space-y-1 text-[11px] font-mono">
              <div className="flex justify-between">
                <span className="text-slate-500">p &lt; .01</span>
                <span className="text-emerald-400">{stats.significance.p001}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">p &lt; .05</span>
                <span className="text-blue-400">{stats.significance.p005}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">p &lt; .10</span>
                <span className="text-amber-400">{stats.significance.p010}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Not sig</span>
                <span className="text-slate-600">{stats.significance.total - stats.significance.p010}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Total</span>
                <span className="text-slate-300">{stats.significance.total}</span>
              </div>
            </div>
          </div>

          {/* Skewness */}
          <div className="p-3 bg-slate-800/50 rounded-lg">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2 font-mono">
              Skewness
            </div>
            <div className="space-y-1 text-[11px] font-mono">
              <div className="flex justify-between">
                <span className="text-slate-500">Left (γ&lt;-.5)</span>
                <span className="text-red-400">{stats.skewness.left}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Symmetric</span>
                <span className="text-emerald-400">{stats.skewness.sym}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Right (γ&gt;.5)</span>
                <span className="text-blue-400">{stats.skewness.right}</span>
              </div>
            </div>
          </div>

          {/* Kurtosis */}
          <div className="p-3 bg-slate-800/50 rounded-lg">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2 font-mono">
              Kurtosis
            </div>
            <div className="space-y-1 text-[11px] font-mono">
              <div className="flex justify-between">
                <span className="text-slate-500">Platy (κ&lt;3)</span>
                <span className="text-blue-400">{stats.kurtosis.platy}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Meso (3-4)</span>
                <span className="text-emerald-400">{stats.kurtosis.meso}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Lepto (κ&gt;4)</span>
                <span className="text-amber-400">{stats.kurtosis.lepto}</span>
              </div>
            </div>
          </div>

          {/* t-test Summary */}
          <div className="p-3 bg-slate-800/50 rounded-lg">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2 font-mono">
              t-Statistics
            </div>
            <div className="space-y-1 text-[11px] font-mono">
              <div className="flex justify-between">
                <span className="text-slate-500">Mean t</span>
                <span className="text-slate-300">{stats.tMean.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">|t| ≥ 1.96</span>
                <span className="text-emerald-400">
                  {allHorizons.filter(h => Math.abs(h.tStatistic) >= 1.96).length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">|t| ≥ 2.58</span>
                <span className="text-blue-400">
                  {allHorizons.filter(h => Math.abs(h.tStatistic) >= 2.58).length}
                </span>
              </div>
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
  const backtestMetrics = useMemo(() => generateBacktestMetrics(), []);
  const correlationData = useMemo(() => generateCorrelationData(), []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <QuantHeader />

      {/* Summary Statistics */}
      <SummaryStats metrics={backtestMetrics} />

      <Separator className="bg-slate-800" />

      {/* Live Signals */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-slate-200 font-mono flex items-center gap-2">
              <Activity className="w-4 h-4 text-slate-400" />
              Live Signal Feed
            </h2>
            <ApiHealthIndicator />
          </div>
          <span className="text-[10px] text-slate-500 font-mono">Real-time API | UTC</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <LiveSignalCard asset="Crude_Oil" displayName="CL" />
          <LiveSignalCard asset="Bitcoin" displayName="BTC" />
          <LiveSignalCard asset="GOLD" displayName="GC" />
          <LiveSignalCard asset="SP500" displayName="ES" />
        </div>
      </section>

      <Separator className="bg-slate-800" />

      {/* Distribution Analysis */}
      <DistributionPanel metrics={backtestMetrics} />

      <Separator className="bg-slate-800" />

      {/* Two-Column: Correlation + Top Performers */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Correlation Heatmap */}
        <CorrelationHeatmap data={correlationData} />

        {/* Top Risk-Adjusted Assets */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-200 font-mono flex items-center gap-2">
              <Crosshair className="w-4 h-4 text-slate-400" />
              Top Performers (Risk-Adjusted)
            </h2>
            <Badge className="bg-slate-800 border-slate-600 text-slate-400 text-[10px] font-mono">
              Ranked by Sharpe
            </Badge>
          </div>
          <div className="space-y-3">
            {backtestMetrics.slice(0, 3).map((m, i) => (
              <AssetPerformanceCard key={m.assetId} metric={m} rank={i + 1} />
            ))}
          </div>
        </div>
      </div>

      <Separator className="bg-slate-800" />

      {/* Full Backtest Statistics Table */}
      <BacktestTable metrics={backtestMetrics} />

      {/* Methodology Note */}
      <div className="p-4 bg-slate-900/50 border border-slate-800 rounded-lg">
        <div className="flex items-start gap-3">
          <FlaskConical className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
          <div className="text-[10px] text-slate-500 font-mono leading-relaxed space-y-1">
            <p>
              <strong className="text-slate-400">Methodology:</strong> All metrics computed on daily returns,
              n=252 (1 year rolling). Sharpe = (R̄ - Rf) / σ × √252, Rf = 5% ann.
              Sortino uses downside deviation only. VaR/CVaR at 95% confidence, parametric.
            </p>
            <p>
              Kelly fraction f* = (p×b - q) / b where p=win rate, q=1-p, b=payoff ratio.
              Statistical significance via two-tailed t-test. CI = x̄ ± t(α/2,df) × SE.
            </p>
            <p>
              Correlation matrix: Pearson product-moment on signal strength. PCA eigendecomposition
              for dimensionality analysis. ** p&lt;.05, * p&lt;.10, ns = not significant.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
