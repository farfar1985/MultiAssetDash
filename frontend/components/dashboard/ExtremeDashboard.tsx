'use client';

/**
 * EXTREME DASHBOARD — Everything. All of it.
 * ==========================================
 * The master view that combines ALL intelligence modules.
 * For power users who want to see it all at once.
 * 
 * Includes:
 * - Market Pulse (all assets)
 * - Signal Confluence (per asset)
 * - Historical Analogs
 * - Yield Curve + Recession Probability
 * - Correlation Regime
 * - HHT Regime Detection
 * - VIX Intelligence
 * - Smart Money / COT Signals
 * - Credit Spreads
 * - Crowded Trade Detection
 * - Factor Attribution
 * - Hedge Calculator
 * - Position Sizing
 * - Crypto On-Chain
 * - AI Market Summary
 * 
 * Author: AmiraB
 * Created: 2026-02-08
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  Coins,
  FileText,
  Flame,
  Gauge,
  GitBranch,
  Globe,
  Layers,
  LineChart,
  Radio,
  Shield,
  Target,
  TrendingDown,
  TrendingUp,
  Zap,
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

interface MarketPulseAsset {
  asset: string;
  conviction_score: number;
  conviction_label: string;
  confidence: number;
  bullish: number;
  bearish: number;
  headline: string;
  top_driver: string | null;
}

interface YieldCurveData {
  spread_2y10y: number;
  state: string;
  recession_probability: number;
  signal: string;
}

interface CorrelationData {
  regime: string;
  avg_correlation: number;
  diversification_score: number;
  risk_level: string;
}

interface CreditSpreadData {
  spreads: { quality_spread: number; treasury_spread: number };
  regime: string;
  signal: string;
  confidence: number;
  implications: { equity_bias: string; recession_risk_delta: number };
}

interface CrowdingAsset {
  crowding: { level: string; score: number; positioning: string };
  unwind_risk: { level: string; probability_30d: number };
  signal: { recommendation: string };
}

interface VIXData {
  current_level: number;
  percentile: number;
  regime: string;
  term_structure: string;
  signal: string;
}

interface HHTAsset {
  regime: string;
  confidence: number;
  trend_strength: number;
  regime_change_probability: number;
}

interface AISummary {
  headlines: { main: string; sub: string };
  sentiment: { overall: string; risk_level: string; confidence: number };
  themes: string[];
  opportunities: string[];
  risks: string[];
}

// ============================================================================
// API Fetchers
// ============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5002';

async function fetchAPI<T>(endpoint: string): Promise<T | null> {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`);
    if (!res.ok) return null;
    const data = await res.json();
    return data.success !== false ? data : null;
  } catch {
    return null;
  }
}

// ============================================================================
// Sub-Components
// ============================================================================

function ScoreGauge({ score, label, size = 'md' }: { score: number; label: string; size?: 'sm' | 'md' | 'lg' }) {
  const color = score > 30 ? 'text-green-500' : score < -30 ? 'text-red-500' : 'text-yellow-500';
  const bgColor = score > 30 ? 'bg-green-500' : score < -30 ? 'bg-red-500' : 'bg-yellow-500';
  const sizeClass = size === 'lg' ? 'text-4xl' : size === 'md' ? 'text-2xl' : 'text-lg';
  
  return (
    <div className="text-center">
      <div className={`font-bold ${sizeClass} ${color}`}>
        {score > 0 ? '+' : ''}{score.toFixed(0)}
      </div>
      <div className={`h-1 w-full ${bgColor} rounded mt-1`} style={{ opacity: Math.abs(score) / 100 }} />
      <div className="text-xs text-gray-500 mt-1">{label}</div>
    </div>
  );
}

function StatusBadge({ status, variant = 'default' }: { status: string; variant?: 'default' | 'outline' }) {
  const colors: Record<string, string> = {
    BULLISH: 'bg-green-500',
    BEARISH: 'bg-red-500',
    NEUTRAL: 'bg-gray-500',
    BUY: 'bg-green-500',
    SELL: 'bg-red-500',
    HOLD: 'bg-yellow-500',
    STRESS: 'bg-red-500',
    ELEVATED: 'bg-orange-500',
    NORMAL: 'bg-gray-500',
    COMPRESSED: 'bg-green-500',
    EXTREME: 'bg-red-500',
    HIGH: 'bg-orange-500',
    LOW: 'bg-green-500',
    CRISIS: 'bg-red-600',
    RISK_OFF: 'bg-red-500',
    RISK_ON: 'bg-green-500',
    CAUTION: 'bg-orange-500',
    BULL: 'bg-green-500',
    BEAR: 'bg-red-500',
    TRANSITION: 'bg-yellow-500',
  };

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-bold text-white ${colors[status] || 'bg-gray-500'}`}>
      {status}
    </span>
  );
}

function MiniCard({ title, icon: Icon, children, className = '' }: { 
  title: string; 
  icon: React.ElementType; 
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Card className={`${className}`}>
      <CardHeader className="py-2 px-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Icon className="w-4 h-4" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="py-2 px-3">
        {children}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function ExtremeDashboard() {
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  
  // All state
  const [pulse, setPulse] = useState<MarketPulseAsset[]>([]);
  const [yieldCurve, setYieldCurve] = useState<YieldCurveData | null>(null);
  const [correlations, setCorrelations] = useState<CorrelationData | null>(null);
  const [creditSpreads, setCreditSpreads] = useState<CreditSpreadData | null>(null);
  const [crowding, setCrowding] = useState<Record<string, CrowdingAsset>>({});
  const [vix, setVix] = useState<VIXData | null>(null);
  const [hht, setHht] = useState<Record<string, HHTAsset>>({});
  const [aiSummary, setAiSummary] = useState<AISummary | null>(null);

  // Fetch all data
  useEffect(() => {
    async function loadAll() {
      setLoading(true);
      
      const [
        pulseData,
        ycData,
        corrData,
        creditData,
        crowdData,
        vixData,
        summaryData,
      ] = await Promise.all([
        fetchAPI<{ assets: MarketPulseAsset[] }>('/api/v2/market-pulse'),
        fetchAPI<YieldCurveData>('/api/v2/yield-curve'),
        fetchAPI<CorrelationData>('/api/v2/correlations'),
        fetchAPI<CreditSpreadData>('/api/v2/credit-spreads'),
        fetchAPI<{ assets: Record<string, CrowdingAsset> }>('/api/v2/crowding'),
        fetchAPI<VIXData>('/api/v2/vix'),
        fetchAPI<AISummary>('/api/v2/ai-summary'),
      ]);

      if (pulseData?.assets) setPulse(pulseData.assets);
      if (ycData) setYieldCurve(ycData);
      if (corrData) setCorrelations(corrData);
      if (creditData) setCreditSpreads(creditData);
      if (crowdData?.assets) setCrowding(crowdData.assets);
      if (vixData) setVix(vixData);
      if (summaryData) setAiSummary(summaryData);

      // Fetch HHT for each asset
      const assets = ['SP500', 'GOLD', 'CRUDE', 'BITCOIN', 'NASDAQ'];
      const hhtResults: Record<string, HHTAsset> = {};
      for (const asset of assets) {
        const data = await fetchAPI<HHTAsset>(`/api/v2/hht/${asset}`);
        if (data) hhtResults[asset] = data;
      }
      setHht(hhtResults);

      setLoading(false);
      setLastUpdate(new Date());
    }

    loadAll();
    const interval = setInterval(loadAll, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <Zap className="w-16 h-16 mx-auto animate-pulse text-yellow-500" />
          <div className="mt-4 text-xl font-bold">Loading EXTREME Dashboard...</div>
          <div className="text-gray-500">Fetching all intelligence modules</div>
        </div>
      </div>
    );
  }

  // Calculate aggregate scores
  const avgConviction = pulse.length > 0 
    ? pulse.reduce((sum, p) => sum + p.conviction_score, 0) / pulse.length 
    : 0;
  
  const extremeCrowding = Object.entries(crowding).filter(([_, c]) => c.crowding.level === 'EXTREME');
  const highCrowding = Object.entries(crowding).filter(([_, c]) => c.crowding.level === 'HIGH');

  return (
    <div className="min-h-screen bg-black text-white p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Flame className="w-8 h-8 text-orange-500" />
          <div>
            <h1 className="text-2xl font-bold">EXTREME DASHBOARD</h1>
            <p className="text-xs text-gray-500">All intelligence. No filters.</p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-500">Last Update</div>
          <div className="text-sm font-mono">{lastUpdate.toLocaleTimeString()}</div>
        </div>
      </div>

      {/* AI Summary Banner */}
      {aiSummary && (
        <div className="mb-4 p-4 bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-lg border border-purple-500/30">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold">{aiSummary.headlines.main}</h2>
              <p className="text-gray-400">{aiSummary.headlines.sub}</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-center">
                <StatusBadge status={aiSummary.sentiment.overall} />
                <div className="text-xs text-gray-500 mt-1">Sentiment</div>
              </div>
              <div className="text-center">
                <StatusBadge status={aiSummary.sentiment.risk_level} />
                <div className="text-xs text-gray-500 mt-1">Risk</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">{aiSummary.sentiment.confidence}%</div>
                <div className="text-xs text-gray-500">Confidence</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Alert Bar */}
      {(extremeCrowding.length > 0 || (yieldCurve?.state === 'INVERTED')) && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-500/50 rounded-lg flex items-center gap-3">
          <AlertTriangle className="w-6 h-6 text-red-500" />
          <div className="flex-1">
            {extremeCrowding.length > 0 && (
              <span className="mr-4">
                🚨 EXTREME CROWDING: {extremeCrowding.map(([a]) => a).join(', ')}
              </span>
            )}
            {yieldCurve?.state === 'INVERTED' && (
              <span>⚠️ YIELD CURVE INVERTED — Recession risk elevated</span>
            )}
          </div>
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-3">
        
        {/* Row 1: Market Pulse (full width) */}
        <div className="col-span-12">
          <Card className="bg-gray-900/50 border-gray-700">
            <CardHeader className="py-2 px-4">
              <CardTitle className="text-sm flex items-center gap-2">
                <Radio className="w-4 h-4" />
                MARKET PULSE — All Assets
              </CardTitle>
            </CardHeader>
            <CardContent className="py-2 px-4">
              <div className="grid grid-cols-5 gap-4">
                {pulse.map((p) => (
                  <div key={p.asset} className="text-center p-3 bg-gray-800/50 rounded-lg">
                    <div className="font-bold text-lg">{p.asset}</div>
                    <ScoreGauge score={p.conviction_score} label={p.conviction_label} size="md" />
                    <div className="mt-2 flex justify-center gap-2">
                      <span className="text-xs text-green-400">↑{p.bullish}</span>
                      <span className="text-xs text-red-400">↓{p.bearish}</span>
                    </div>
                    {hht[p.asset] && (
                      <div className="mt-1">
                        <StatusBadge status={hht[p.asset].regime} />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Row 2: Macro Panel */}
        <div className="col-span-3">
          <MiniCard title="Yield Curve" icon={LineChart} className="bg-gray-900/50 border-gray-700 h-full">
            {yieldCurve ? (
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-2xl font-bold">{yieldCurve.spread_2y10y}bps</span>
                  <StatusBadge status={yieldCurve.state} />
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Recession Prob:</span>
                  <span className={yieldCurve.recession_probability > 50 ? 'text-red-400' : 'text-green-400'}>
                    {yieldCurve.recession_probability}%
                  </span>
                </div>
                <div className="text-xs text-gray-500">Signal: {yieldCurve.signal}</div>
              </div>
            ) : <div className="text-gray-500">Loading...</div>}
          </MiniCard>
        </div>

        <div className="col-span-3">
          <MiniCard title="Correlations" icon={GitBranch} className="bg-gray-900/50 border-gray-700 h-full">
            {correlations ? (
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-2xl font-bold">{(correlations.avg_correlation * 100).toFixed(0)}%</span>
                  <StatusBadge status={correlations.regime} />
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Diversification:</span>
                  <span className="text-blue-400">{correlations.diversification_score}/100</span>
                </div>
                <div className="text-xs text-gray-500">Risk: {correlations.risk_level}</div>
              </div>
            ) : <div className="text-gray-500">Loading...</div>}
          </MiniCard>
        </div>

        <div className="col-span-3">
          <MiniCard title="Credit Spreads" icon={Layers} className="bg-gray-900/50 border-gray-700 h-full">
            {creditSpreads ? (
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-2xl font-bold">{creditSpreads.spreads.quality_spread.toFixed(2)}%</span>
                  <StatusBadge status={creditSpreads.regime} />
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Signal:</span>
                  <StatusBadge status={creditSpreads.signal} />
                </div>
                <div className="text-xs text-gray-500">
                  Equity: {creditSpreads.implications.equity_bias}
                </div>
              </div>
            ) : <div className="text-gray-500">Loading...</div>}
          </MiniCard>
        </div>

        <div className="col-span-3">
          <MiniCard title="VIX Intelligence" icon={Activity} className="bg-gray-900/50 border-gray-700 h-full">
            {vix ? (
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-2xl font-bold">{vix.current_level.toFixed(1)}</span>
                  <StatusBadge status={vix.regime} />
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Percentile:</span>
                  <span>{vix.percentile}%</span>
                </div>
                <div className="text-xs text-gray-500">
                  Term: {vix.term_structure} | {vix.signal}
                </div>
              </div>
            ) : <div className="text-gray-500">Loading...</div>}
          </MiniCard>
        </div>

        {/* Row 3: Crowding + HHT Regimes */}
        <div className="col-span-6">
          <MiniCard title="Crowded Trade Detection" icon={Target} className="bg-gray-900/50 border-gray-700">
            <div className="grid grid-cols-5 gap-2">
              {Object.entries(crowding).map(([asset, data]) => {
                const levelColors: Record<string, string> = {
                  EXTREME: 'border-red-500 bg-red-900/30',
                  HIGH: 'border-orange-500 bg-orange-900/30',
                  MODERATE: 'border-yellow-500 bg-yellow-900/30',
                  LOW: 'border-green-500 bg-green-900/30',
                  CONTRARIAN: 'border-purple-500 bg-purple-900/30',
                };
                return (
                  <div key={asset} className={`p-2 rounded border ${levelColors[data.crowding.level] || 'border-gray-600'}`}>
                    <div className="font-bold text-sm">{asset}</div>
                    <div className="text-lg font-mono">{data.crowding.score.toFixed(0)}</div>
                    <div className="text-xs text-gray-400">{data.crowding.level}</div>
                    <div className="text-xs mt-1">
                      <StatusBadge status={data.signal.recommendation} />
                    </div>
                  </div>
                );
              })}
            </div>
          </MiniCard>
        </div>

        <div className="col-span-6">
          <MiniCard title="HHT Regime Detection" icon={Brain} className="bg-gray-900/50 border-gray-700">
            <div className="grid grid-cols-5 gap-2">
              {Object.entries(hht).map(([asset, data]) => {
                const regimeColors: Record<string, string> = {
                  BULL: 'border-green-500 bg-green-900/30',
                  BEAR: 'border-red-500 bg-red-900/30',
                  TRANSITION: 'border-yellow-500 bg-yellow-900/30',
                  SIDEWAYS: 'border-gray-500 bg-gray-900/30',
                };
                return (
                  <div key={asset} className={`p-2 rounded border ${regimeColors[data.regime] || 'border-gray-600'}`}>
                    <div className="font-bold text-sm">{asset}</div>
                    <StatusBadge status={data.regime} />
                    <div className="text-xs text-gray-400 mt-1">
                      {data.confidence.toFixed(0)}% conf
                    </div>
                    <div className="text-xs">
                      Δ {(data.regime_change_probability * 100).toFixed(0)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </MiniCard>
        </div>

        {/* Row 4: Opportunities & Risks */}
        {aiSummary && (
          <>
            <div className="col-span-6">
              <MiniCard title="Opportunities" icon={TrendingUp} className="bg-green-900/20 border-green-700">
                <ul className="space-y-1">
                  {aiSummary.opportunities.length > 0 ? (
                    aiSummary.opportunities.map((o, i) => (
                      <li key={i} className="text-sm text-green-400">✅ {o}</li>
                    ))
                  ) : (
                    <li className="text-sm text-gray-500">No high-conviction opportunities</li>
                  )}
                </ul>
              </MiniCard>
            </div>
            <div className="col-span-6">
              <MiniCard title="Risks" icon={TrendingDown} className="bg-red-900/20 border-red-700">
                <ul className="space-y-1">
                  {aiSummary.risks.length > 0 ? (
                    aiSummary.risks.map((r, i) => (
                      <li key={i} className="text-sm text-red-400">⚠️ {r}</li>
                    ))
                  ) : (
                    <li className="text-sm text-gray-500">Standard market risks apply</li>
                  )}
                </ul>
              </MiniCard>
            </div>
          </>
        )}

        {/* Row 5: Key Themes */}
        {aiSummary && aiSummary.themes.length > 0 && (
          <div className="col-span-12">
            <MiniCard title="Key Themes" icon={Globe} className="bg-gray-900/50 border-gray-700">
              <div className="flex flex-wrap gap-2">
                {aiSummary.themes.map((theme, i) => (
                  <span key={i} className="px-3 py-1 bg-blue-900/50 text-blue-300 rounded-full text-sm">
                    {theme}
                  </span>
                ))}
              </div>
            </MiniCard>
          </div>
        )}

        {/* Row 6: Aggregate Score */}
        <div className="col-span-12">
          <Card className="bg-gradient-to-r from-gray-900 to-gray-800 border-gray-600">
            <CardContent className="py-4">
              <div className="flex items-center justify-around">
                <div className="text-center">
                  <div className="text-5xl font-bold">
                    <span className={avgConviction > 20 ? 'text-green-400' : avgConviction < -20 ? 'text-red-400' : 'text-yellow-400'}>
                      {avgConviction > 0 ? '+' : ''}{avgConviction.toFixed(0)}
                    </span>
                  </div>
                  <div className="text-gray-400">Aggregate Conviction</div>
                </div>
                <div className="text-center">
                  <div className="text-5xl font-bold text-blue-400">
                    {pulse.filter(p => p.conviction_score > 20).length}
                  </div>
                  <div className="text-gray-400">Bullish Assets</div>
                </div>
                <div className="text-center">
                  <div className="text-5xl font-bold text-red-400">
                    {pulse.filter(p => p.conviction_score < -20).length}
                  </div>
                  <div className="text-gray-400">Bearish Assets</div>
                </div>
                <div className="text-center">
                  <div className="text-5xl font-bold text-orange-400">
                    {extremeCrowding.length + highCrowding.length}
                  </div>
                  <div className="text-gray-400">Crowded Trades</div>
                </div>
                <div className="text-center">
                  <div className="text-5xl font-bold text-purple-400">
                    {yieldCurve?.recession_probability || 0}%
                  </div>
                  <div className="text-gray-400">Recession Prob</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

      </div>

      {/* Footer */}
      <div className="mt-4 text-center text-xs text-gray-600">
        QDT Intelligence Engine • {Object.keys(crowding).length + Object.keys(hht).length + 4} modules active • 
        Refresh: 60s • Built by AmiraB
      </div>
    </div>
  );
}

export default ExtremeDashboard;
