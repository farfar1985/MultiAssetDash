'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface CrowdedTradeData {
  asset: string;
  crowding: {
    level: string;
    score: number;
    cot_z_score: number;
    positioning: string;
  };
  volume: {
    z_score: number;
    trend: string;
  };
  unwind_risk: {
    level: string;
    probability_30d: number;
  };
  signal: {
    recommendation: string;
    confidence: number;
  };
  history: {
    similar_unwinds: number;
    avg_magnitude_pct: number;
  };
  reasoning: string[];
}

interface AllCrowdingData {
  summary: {
    extreme_crowding: string[];
    high_crowding: string[];
    contrarian_opportunities: string[];
  };
  assets: Record<string, CrowdedTradeData>;
}

interface CrowdedTradeCardProps {
  data?: AllCrowdingData;
  loading?: boolean;
}

const levelEmojis: Record<string, string> = {
  EXTREME: '🔴',
  HIGH: '🟠',
  MODERATE: '🟡',
  LOW: '🟢',
  CONTRARIAN: '💡',
};

const levelColors: Record<string, string> = {
  EXTREME: 'bg-red-100 text-red-700 border-red-300',
  HIGH: 'bg-orange-100 text-orange-700 border-orange-300',
  MODERATE: 'bg-yellow-100 text-yellow-700 border-yellow-300',
  LOW: 'bg-green-100 text-green-700 border-green-300',
  CONTRARIAN: 'bg-purple-100 text-purple-700 border-purple-300',
};

const unwindColors: Record<string, string> = {
  CRITICAL: 'text-red-600',
  ELEVATED: 'text-orange-600',
  NORMAL: 'text-gray-600',
  LOW: 'text-green-600',
};

const recColors: Record<string, string> = {
  FADE: 'bg-red-500 text-white',
  AVOID: 'bg-orange-500 text-white',
  FOLLOW: 'bg-green-500 text-white',
  NEUTRAL: 'bg-gray-500 text-white',
};

export function CrowdedTradeCard({ data, loading }: CrowdedTradeCardProps) {
  if (loading) {
    return (
      <Card className="animate-pulse">
        <CardHeader>
          <CardTitle>Crowded Trade Detection</CardTitle>
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

  const { summary, assets } = data;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Crowded Trade Detection</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Summary Alerts */}
        {summary.extreme_crowding.length > 0 && (
          <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <div className="flex items-center gap-2 text-red-700 dark:text-red-400 font-medium">
              <span>🚨</span>
              <span>EXTREME CROWDING: {summary.extreme_crowding.join(', ')}</span>
            </div>
            <p className="text-xs text-red-600 dark:text-red-400 mt-1">
              High probability of violent unwind. Consider fading or reducing exposure.
            </p>
          </div>
        )}

        {summary.contrarian_opportunities.length > 0 && (
          <div className="mb-4 p-3 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg">
            <div className="flex items-center gap-2 text-purple-700 dark:text-purple-400 font-medium">
              <span>💡</span>
              <span>Contrarian: {summary.contrarian_opportunities.join(', ')}</span>
            </div>
            <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
              Market positioned opposite direction. Potential opportunity.
            </p>
          </div>
        )}

        {/* Asset Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {Object.entries(assets).map(([assetName, assetData]) => (
            <div
              key={assetName}
              className={`p-3 rounded-lg border ${levelColors[assetData.crowding.level] || 'bg-gray-50 border-gray-200'}`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">
                  {levelEmojis[assetData.crowding.level]} {assetName}
                </span>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${recColors[assetData.signal.recommendation] || 'bg-gray-500 text-white'}`}>
                  {assetData.signal.recommendation}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs mb-2">
                <div>
                  <span className="text-gray-500">Score: </span>
                  <span className="font-medium">{assetData.crowding.score.toFixed(0)}/100</span>
                </div>
                <div>
                  <span className="text-gray-500">COT Z: </span>
                  <span className="font-medium">{assetData.crowding.cot_z_score.toFixed(2)}</span>
                </div>
              </div>

              <div className="text-xs">
                <span className="text-gray-500">Unwind Risk: </span>
                <span className={`font-medium ${unwindColors[assetData.unwind_risk.level]}`}>
                  {assetData.unwind_risk.level} ({assetData.unwind_risk.probability_30d}% in 30d)
                </span>
              </div>

              <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
                {assetData.crowding.positioning !== 'BALANCED' && (
                  <span>{assetData.crowding.positioning.replace('_', ' ')}</span>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="mt-4 flex flex-wrap gap-2 text-xs">
          <span className="text-gray-500">Levels:</span>
          {Object.entries(levelEmojis).map(([level, emoji]) => (
            <span key={level} className="flex items-center gap-1">
              {emoji} {level}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default CrowdedTradeCard;
