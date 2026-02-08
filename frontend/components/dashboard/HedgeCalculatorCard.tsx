'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

interface HedgeRecommendation {
  asset: string;
  position: string;
  notional: number;
  hedge: {
    optimal_ratio: number;
    notional: number;
    instrument: string;
  };
  costs: {
    basis_risk_bps: number;
    roll_cost_pct: number;
    total_annualized_bps: number;
  };
  effectiveness: {
    variance_reduction_pct: number;
    r_squared: number;
    tracking_error_bps: number;
  };
  execution: {
    urgency: string;
    entry: string;
  };
  reasoning: string[];
}

interface HedgeCalculatorCardProps {
  data?: HedgeRecommendation;
  onCalculate?: (asset: string, position: string, notional: number) => void;
  loading?: boolean;
}

export function HedgeCalculatorCard({ data, onCalculate, loading }: HedgeCalculatorCardProps) {
  const [asset, setAsset] = useState('CRUDE');
  const [position, setPosition] = useState('LONG');
  const [notional, setNotional] = useState('1000000');

  const urgencyColors: Record<string, string> = {
    IMMEDIATE: 'bg-red-500 text-white',
    TODAY: 'bg-orange-500 text-white',
    THIS_WEEK: 'bg-yellow-500 text-black',
    MONITOR: 'bg-gray-500 text-white',
  };

  const handleCalculate = () => {
    if (onCalculate) {
      onCalculate(asset, position, parseFloat(notional));
    }
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Optimal Hedge Calculator</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Input Form */}
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div>
            <Label htmlFor="asset" className="text-xs">Asset</Label>
            <select
              id="asset"
              value={asset}
              onChange={(e) => setAsset(e.target.value)}
              className="w-full p-2 text-sm border rounded bg-white dark:bg-gray-800"
            >
              <option value="CRUDE">Crude Oil</option>
              <option value="GOLD">Gold</option>
              <option value="SP500">S&P 500</option>
              <option value="NASDAQ">NASDAQ</option>
              <option value="BITCOIN">Bitcoin</option>
            </select>
          </div>
          <div>
            <Label htmlFor="position" className="text-xs">Position</Label>
            <select
              id="position"
              value={position}
              onChange={(e) => setPosition(e.target.value)}
              className="w-full p-2 text-sm border rounded bg-white dark:bg-gray-800"
            >
              <option value="LONG">Long (Buy Hedge)</option>
              <option value="SHORT">Short (Sell Hedge)</option>
            </select>
          </div>
          <div>
            <Label htmlFor="notional" className="text-xs">Notional ($)</Label>
            <Input
              id="notional"
              type="number"
              value={notional}
              onChange={(e) => setNotional(e.target.value)}
              className="text-sm"
            />
          </div>
        </div>

        <button
          onClick={handleCalculate}
          disabled={loading}
          className="w-full py-2 bg-blue-600 text-white rounded font-medium hover:bg-blue-700 disabled:opacity-50 mb-4"
        >
          {loading ? 'Calculating...' : 'Calculate Optimal Hedge'}
        </button>

        {/* Results */}
        {data && (
          <>
            {/* Hedge Details */}
            <div className="grid grid-cols-3 gap-3 mb-4">
              <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/30 rounded">
                <div className="text-xl font-bold text-blue-600">{data.hedge.optimal_ratio.toFixed(3)}</div>
                <div className="text-xs text-gray-500">Hedge Ratio</div>
              </div>
              <div className="text-center p-3 bg-slate-50 dark:bg-slate-800 rounded">
                <div className="text-xl font-bold">${(data.hedge.notional / 1000).toFixed(0)}K</div>
                <div className="text-xs text-gray-500">Hedge Notional</div>
              </div>
              <div className="text-center p-3 bg-slate-50 dark:bg-slate-800 rounded">
                <div className="text-xl font-bold">{data.hedge.instrument}</div>
                <div className="text-xs text-gray-500">Instrument</div>
              </div>
            </div>

            {/* Costs */}
            <div className="mb-4 p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
              <h4 className="text-sm font-medium mb-2">Estimated Costs (Annual)</h4>
              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <div>
                  <div className="font-semibold">{data.costs.basis_risk_bps.toFixed(0)} bps</div>
                  <div className="text-gray-500">Basis Risk</div>
                </div>
                <div>
                  <div className="font-semibold">{(data.costs.roll_cost_pct * 100).toFixed(1)} bps</div>
                  <div className="text-gray-500">Roll Cost</div>
                </div>
                <div>
                  <div className="font-semibold text-orange-600">{data.costs.total_annualized_bps.toFixed(0)} bps</div>
                  <div className="text-gray-500">Total</div>
                </div>
              </div>
            </div>

            {/* Effectiveness */}
            <div className="mb-4 p-3 bg-green-50 dark:bg-green-900/20 rounded">
              <h4 className="text-sm font-medium mb-2">Hedge Effectiveness</h4>
              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <div>
                  <div className="font-semibold text-green-600">{data.effectiveness.variance_reduction_pct.toFixed(0)}%</div>
                  <div className="text-gray-500">Var Reduction</div>
                </div>
                <div>
                  <div className="font-semibold">{data.effectiveness.r_squared.toFixed(2)}</div>
                  <div className="text-gray-500">R²</div>
                </div>
                <div>
                  <div className="font-semibold">{data.effectiveness.tracking_error_bps.toFixed(0)} bps</div>
                  <div className="text-gray-500">Tracking Error</div>
                </div>
              </div>
            </div>

            {/* Execution */}
            <div className="flex items-center justify-between mb-4">
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${urgencyColors[data.execution.urgency] || 'bg-gray-500 text-white'}`}>
                {data.execution.urgency}
              </span>
              <span className="text-sm">
                Entry: <span className="font-medium">{data.execution.entry}</span>
              </span>
            </div>

            {/* Reasoning */}
            <div>
              <h4 className="text-xs font-medium text-gray-500 mb-2">Analysis</h4>
              <ul className="text-xs space-y-1">
                {data.reasoning.map((r, i) => (
                  <li key={i} className="text-gray-600 dark:text-gray-400">• {r}</li>
                ))}
              </ul>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default HedgeCalculatorCard;
