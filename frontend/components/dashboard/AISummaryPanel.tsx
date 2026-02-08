'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface MarketSummaryData {
  headlines: {
    main: string;
    sub: string;
  };
  sentiment: {
    overall: string;
    risk_level: string;
    confidence: number;
  };
  asset_summaries: Record<string, string>;
  themes: string[];
  opportunities: string[];
  risks: string[];
  actions: {
    hedger: string[];
    investor: string[];
    trader: string[];
  };
  narrative: string;
}

interface AISummaryPanelProps {
  data?: MarketSummaryData;
  loading?: boolean;
  persona?: 'hedger' | 'investor' | 'trader';
}

export function AISummaryPanel({ data, loading, persona = 'investor' }: AISummaryPanelProps) {
  if (loading) {
    return (
      <Card className="animate-pulse">
        <CardHeader>
          <CardTitle>AI Market Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 bg-gray-200 dark:bg-gray-700 rounded"></div>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return null;
  }

  const sentimentColors: Record<string, string> = {
    BULLISH: 'text-green-600 bg-green-50',
    BEARISH: 'text-red-600 bg-red-50',
    NEUTRAL: 'text-gray-600 bg-gray-50',
  };

  const riskColors: Record<string, string> = {
    LOW: 'text-green-600',
    MODERATE: 'text-yellow-600',
    ELEVATED: 'text-orange-600',
    HIGH: 'text-red-600',
  };

  const sentimentEmojis: Record<string, string> = {
    BULLISH: '📈',
    BEARISH: '📉',
    NEUTRAL: '➡️',
  };

  const personaActions = data.actions[persona] || data.actions.investor;

  return (
    <Card className="col-span-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl">AI Market Intelligence</CardTitle>
          <span className={`px-3 py-1 rounded-full font-medium ${sentimentColors[data.sentiment.overall] || 'bg-gray-50'}`}>
            {sentimentEmojis[data.sentiment.overall]} {data.sentiment.overall}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        {/* Headlines */}
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            {data.headlines.main}
          </h2>
          <p className="text-gray-600 dark:text-gray-300">{data.headlines.sub}</p>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="text-center p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
            <div className={`text-3xl font-bold ${sentimentColors[data.sentiment.overall]?.split(' ')[0] || 'text-gray-600'}`}>
              {data.sentiment.overall}
            </div>
            <div className="text-sm text-gray-500">Sentiment</div>
          </div>
          <div className="text-center p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
            <div className={`text-3xl font-bold ${riskColors[data.sentiment.risk_level] || 'text-gray-600'}`}>
              {data.sentiment.risk_level}
            </div>
            <div className="text-sm text-gray-500">Risk Level</div>
          </div>
          <div className="text-center p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
            <div className="text-3xl font-bold text-blue-600">{data.sentiment.confidence}%</div>
            <div className="text-sm text-gray-500">Confidence</div>
          </div>
        </div>

        {/* Key Themes */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3">Key Themes</h3>
          <div className="flex flex-wrap gap-2">
            {data.themes.map((theme, i) => (
              <span key={i} className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full text-sm">
                {theme}
              </span>
            ))}
          </div>
        </div>

        {/* Asset Summaries */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3">Asset Overview</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {Object.entries(data.asset_summaries).map(([asset, summary]) => (
              <div key={asset} className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                <div className="font-medium text-sm">{summary}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Opportunities & Risks */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
            <h4 className="font-semibold text-green-700 dark:text-green-400 mb-2">✅ Opportunities</h4>
            <ul className="space-y-1">
              {data.opportunities.length > 0 ? (
                data.opportunities.map((opp, i) => (
                  <li key={i} className="text-sm text-green-600 dark:text-green-400">• {opp}</li>
                ))
              ) : (
                <li className="text-sm text-gray-500">No high-conviction opportunities currently</li>
              )}
            </ul>
          </div>
          <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
            <h4 className="font-semibold text-red-700 dark:text-red-400 mb-2">⚠️ Risks</h4>
            <ul className="space-y-1">
              {data.risks.length > 0 ? (
                data.risks.map((risk, i) => (
                  <li key={i} className="text-sm text-red-600 dark:text-red-400">• {risk}</li>
                ))
              ) : (
                <li className="text-sm text-gray-500">Standard market risks apply</li>
              )}
            </ul>
          </div>
        </div>

        {/* Persona-Specific Actions */}
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
          <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-2">
            📋 Recommended Actions ({persona.charAt(0).toUpperCase() + persona.slice(1)})
          </h4>
          <ul className="space-y-2">
            {personaActions.map((action, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-purple-700 dark:text-purple-300">
                <span className="mt-1">→</span>
                <span>{action}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Persona Selector */}
        <div className="mt-4 text-center">
          <span className="text-xs text-gray-500">
            View actions for: 
            <button className={`ml-2 px-2 py-0.5 rounded ${persona === 'hedger' ? 'bg-purple-200' : 'hover:bg-gray-100'}`}>
              Hedger
            </button>
            <button className={`ml-1 px-2 py-0.5 rounded ${persona === 'investor' ? 'bg-purple-200' : 'hover:bg-gray-100'}`}>
              Investor
            </button>
            <button className={`ml-1 px-2 py-0.5 rounded ${persona === 'trader' ? 'bg-purple-200' : 'hover:bg-gray-100'}`}>
              Trader
            </button>
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

export default AISummaryPanel;
