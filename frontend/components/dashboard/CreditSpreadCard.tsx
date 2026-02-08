'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface CreditSpreadData {
  spreads: {
    baa_yield: number;
    aaa_yield: number;
    quality_spread: number;
    treasury_spread: number;
  };
  regime: string;
  signal: string;
  confidence: number;
  context: {
    percentile: number;
    z_score: number;
    change_30d: number;
    change_90d: number;
  };
  implications: {
    equity_bias: string;
    credit_bias: string;
    recession_risk_delta: number;
  };
  reasoning: string[];
}

interface CreditSpreadCardProps {
  data?: CreditSpreadData;
  loading?: boolean;
}

export function CreditSpreadCard({ data, loading }: CreditSpreadCardProps) {
  if (loading) {
    return (
      <Card className="animate-pulse">
        <CardHeader>
          <CardTitle>Credit Spreads</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return null;
  }

  const regimeColors: Record<string, string> = {
    STRESS: 'text-red-500 bg-red-50',
    ELEVATED: 'text-orange-500 bg-orange-50',
    NORMAL: 'text-gray-500 bg-gray-50',
    COMPRESSED: 'text-green-500 bg-green-50',
    EUPHORIA: 'text-yellow-500 bg-yellow-50',
  };

  const signalColors: Record<string, string> = {
    RISK_OFF: 'bg-red-500',
    CAUTION: 'bg-orange-500',
    NEUTRAL: 'bg-gray-500',
    RISK_ON: 'bg-green-500',
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Credit Spreads</CardTitle>
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${regimeColors[data.regime] || 'bg-gray-50'}`}>
            {data.regime}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        {/* Current Levels */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="text-center p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
            <div className="text-2xl font-bold">{data.spreads.quality_spread.toFixed(2)}%</div>
            <div className="text-xs text-gray-500">Quality Spread (BAA-AAA)</div>
          </div>
          <div className="text-center p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
            <div className="text-2xl font-bold">{data.spreads.treasury_spread.toFixed(2)}%</div>
            <div className="text-xs text-gray-500">Treasury Spread</div>
          </div>
        </div>

        {/* Signal Bar */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium">Risk Signal</span>
            <span className="text-sm text-gray-500">{data.confidence}% confidence</span>
          </div>
          <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
            <div 
              className={`h-full ${signalColors[data.signal] || 'bg-gray-500'} rounded-full transition-all`}
              style={{ width: `${data.confidence}%` }}
            />
          </div>
          <div className="text-center mt-1 text-sm font-medium">
            {data.signal.replace('_', ' ')}
          </div>
        </div>

        {/* Context Metrics */}
        <div className="grid grid-cols-4 gap-2 mb-4 text-center text-xs">
          <div>
            <div className="font-semibold">{data.context.z_score.toFixed(1)}σ</div>
            <div className="text-gray-500">Z-Score</div>
          </div>
          <div>
            <div className="font-semibold">{data.context.percentile.toFixed(0)}%</div>
            <div className="text-gray-500">Percentile</div>
          </div>
          <div>
            <div className={`font-semibold ${data.context.change_30d > 0 ? 'text-red-500' : 'text-green-500'}`}>
              {data.context.change_30d > 0 ? '+' : ''}{data.context.change_30d.toFixed(2)}%
            </div>
            <div className="text-gray-500">30d Δ</div>
          </div>
          <div>
            <div className={`font-semibold ${data.context.change_90d > 0 ? 'text-red-500' : 'text-green-500'}`}>
              {data.context.change_90d > 0 ? '+' : ''}{data.context.change_90d.toFixed(2)}%
            </div>
            <div className="text-gray-500">90d Δ</div>
          </div>
        </div>

        {/* Implications */}
        <div className="space-y-2 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
          <div className="flex justify-between text-sm">
            <span>Equity Bias:</span>
            <span className="font-medium">{data.implications.equity_bias}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span>Credit Bias:</span>
            <span className="font-medium">{data.implications.credit_bias}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span>Recession Risk Δ:</span>
            <span className={`font-medium ${data.implications.recession_risk_delta > 0 ? 'text-red-500' : 'text-green-500'}`}>
              {data.implications.recession_risk_delta > 0 ? '+' : ''}{data.implications.recession_risk_delta}%
            </span>
          </div>
        </div>

        {/* Reasoning */}
        <div className="mt-4">
          <h4 className="text-xs font-medium text-gray-500 mb-2">Analysis</h4>
          <ul className="text-xs space-y-1">
            {data.reasoning.map((r, i) => (
              <li key={i} className="text-gray-600 dark:text-gray-400">• {r}</li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

export default CreditSpreadCard;
