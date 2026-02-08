'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface FactorExposure {
  name: string;
  beta: number;
  contribution_pct: number;
  t_stat: number;
  significance: string;
  level: string;
}

interface FactorAttributionData {
  asset: string;
  period: string;
  summary: {
    r_squared: number;
    total_return: number;
    factor_return: number;
    alpha: number;
  };
  factors: FactorExposure[];
  risk_decomposition: {
    factor_risk_pct: number;
    specific_risk_pct: number;
  };
  insights: {
    dominant_factor: string;
    risk_warning: string;
  };
}

interface FactorAttributionPanelProps {
  data?: FactorAttributionData;
  loading?: boolean;
  compact?: boolean;
}

export function FactorAttributionPanel({ data, loading, compact: _compact = false }: FactorAttributionPanelProps) {
  if (loading) {
    return (
      <Card className="animate-pulse">
        <CardHeader>
          <CardTitle>Factor Attribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-48 bg-gray-200 dark:bg-gray-700 rounded"></div>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return null;
  }

  const levelColors: Record<string, string> = {
    HIGH: 'text-red-500',
    NORMAL: 'text-gray-600',
    LOW: 'text-green-500',
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{data.asset} Factor Attribution</CardTitle>
          <span className="text-xs text-gray-500">{data.period}</span>
        </div>
      </CardHeader>
      <CardContent>
        {/* Summary Stats */}
        <div className="grid grid-cols-4 gap-3 mb-4">
          <div className="text-center p-2 bg-blue-50 dark:bg-blue-900/30 rounded">
            <div className="text-lg font-bold text-blue-600">{(data.summary.r_squared * 100).toFixed(0)}%</div>
            <div className="text-xs text-gray-500">R²</div>
          </div>
          <div className="text-center p-2 bg-slate-50 dark:bg-slate-800 rounded">
            <div className={`text-lg font-bold ${data.summary.total_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {data.summary.total_return >= 0 ? '+' : ''}{data.summary.total_return.toFixed(1)}%
            </div>
            <div className="text-xs text-gray-500">Total Return</div>
          </div>
          <div className="text-center p-2 bg-slate-50 dark:bg-slate-800 rounded">
            <div className="text-lg font-bold">{data.summary.factor_return.toFixed(1)}%</div>
            <div className="text-xs text-gray-500">Factor Return</div>
          </div>
          <div className="text-center p-2 bg-purple-50 dark:bg-purple-900/30 rounded">
            <div className={`text-lg font-bold ${data.summary.alpha >= 0 ? 'text-purple-600' : 'text-orange-600'}`}>
              {data.summary.alpha >= 0 ? '+' : ''}{data.summary.alpha.toFixed(2)}%
            </div>
            <div className="text-xs text-gray-500">Alpha</div>
          </div>
        </div>

        {/* Factor Exposures Table */}
        <div className="mb-4">
          <h4 className="text-sm font-medium mb-2">Factor Exposures</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2">Factor</th>
                  <th className="text-right py-2">Beta</th>
                  <th className="text-right py-2">Contrib</th>
                  <th className="text-right py-2">t-stat</th>
                  <th className="text-center py-2">Level</th>
                </tr>
              </thead>
              <tbody>
                {data.factors.map((f, i) => (
                  <tr key={i} className="border-b border-gray-100 dark:border-gray-800">
                    <td className="py-2 font-medium">{f.name}</td>
                    <td className={`text-right py-2 ${f.beta >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {f.beta >= 0 ? '+' : ''}{f.beta.toFixed(3)}
                    </td>
                    <td className="text-right py-2">{f.contribution_pct.toFixed(1)}%</td>
                    <td className="text-right py-2">
                      {f.t_stat.toFixed(2)}{f.significance}
                    </td>
                    <td className={`text-center py-2 ${levelColors[f.level] || 'text-gray-600'}`}>
                      {f.level}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Risk Decomposition */}
        <div className="mb-4">
          <h4 className="text-sm font-medium mb-2">Risk Decomposition</h4>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden flex">
              <div 
                className="h-full bg-blue-500"
                style={{ width: `${data.risk_decomposition.factor_risk_pct}%` }}
              />
              <div 
                className="h-full bg-orange-500"
                style={{ width: `${data.risk_decomposition.specific_risk_pct}%` }}
              />
            </div>
          </div>
          <div className="flex justify-between text-xs mt-1">
            <span className="text-blue-600">
              Factor Risk: {data.risk_decomposition.factor_risk_pct.toFixed(0)}%
            </span>
            <span className="text-orange-600">
              Specific Risk: {data.risk_decomposition.specific_risk_pct.toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Insights */}
        <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
          <div className="text-sm">
            <span className="text-gray-500">Dominant Factor: </span>
            <span className="font-medium">{data.insights.dominant_factor}</span>
          </div>
          {data.insights.risk_warning && (
            <div className="text-sm text-orange-600 mt-1">
              ⚠️ {data.insights.risk_warning}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default FactorAttributionPanel;
