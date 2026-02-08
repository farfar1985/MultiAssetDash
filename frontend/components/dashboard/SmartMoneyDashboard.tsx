"use client";

import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  Info,
  DollarSign,
  Target,
  Shield,
} from "lucide-react";

interface COTSignal {
  signal: string;
  confidence: number;
  z_score: number;
  percentile: number;
  current_net: number;
  message: string;
  reasoning: string[];
}

interface SmartMoneyData {
  timestamp: string;
  signals: Record<string, COTSignal>;
  overall_sentiment: string;
  risk_on_off: string;
  key_divergences: string[];
  recommendations: string[];
}

interface RegimeData {
  regime: {
    name: string;
    confidence: number;
    vix_level: number;
    vix_regime: string;
    momentum_20d: number;
    risk_appetite: number;
  };
  context: {
    risk_score: number;
    risk_label: string;
    leading_indicator: string;
    correlations: Record<string, number>;
    divergences: string[];
  };
}

interface _PositionSizing {
  position_size_pct: number;
  position_value: number;
  risk_management: {
    stop_loss_pct: number;
    profit_target_pct: number;
    risk_reward: number;
    max_loss: number;
  };
  timing: {
    urgency: string;
    entry_strategy: string;
  };
  reasoning: string[];
  warnings: string[];
}

const SignalBadge = ({ signal }: { signal: string }) => {
  const variants: Record<string, { color: string; icon: React.ReactNode }> = {
    STRONG_BUY: { color: "bg-green-500", icon: <TrendingUp className="h-3 w-3" /> },
    BUY: { color: "bg-green-400", icon: <TrendingUp className="h-3 w-3" /> },
    NEUTRAL: { color: "bg-gray-400", icon: <Minus className="h-3 w-3" /> },
    SELL: { color: "bg-red-400", icon: <TrendingDown className="h-3 w-3" /> },
    STRONG_SELL: { color: "bg-red-500", icon: <TrendingDown className="h-3 w-3" /> },
  };

  const v = variants[signal] || variants.NEUTRAL;

  return (
    <Badge className={`${v.color} text-white flex items-center gap-1`}>
      {v.icon}
      {signal.replace("_", " ")}
    </Badge>
  );
};

const ZScoreGauge = ({ value, label }: { value: number; label: string }) => {
  // Map z-score (-3 to +3) to percentage (0 to 100)
  const pct = Math.min(100, Math.max(0, ((value + 3) / 6) * 100));
  
  const getColor = () => {
    if (value > 2) return "bg-red-500";
    if (value > 1) return "bg-orange-400";
    if (value < -2) return "bg-green-500";
    if (value < -1) return "bg-green-400";
    return "bg-gray-400";
  };

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>Short</span>
        <span>{label}</span>
        <span>Long</span>
      </div>
      <div className="relative h-2 bg-gray-200 dark:bg-gray-700 rounded">
        <div
          className={`absolute h-full ${getColor()} rounded`}
          style={{ width: `${pct}%` }}
        />
        <div
          className="absolute w-0.5 h-4 bg-black dark:bg-white -top-1"
          style={{ left: `${pct}%` }}
        />
      </div>
      <div className="text-center text-sm font-medium">
        z = {value.toFixed(2)}
      </div>
    </div>
  );
};

export function SmartMoneyDashboard() {
  const [cotData, setCotData] = useState<SmartMoneyData | null>(null);
  const [regimeData, setRegimeData] = useState<RegimeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [cotRes, regimeRes] = await Promise.all([
          fetch("/api/v2/cot"),
          fetch("/api/v2/regime"),
        ]);

        if (cotRes.ok) {
          const cot = await cotRes.json();
          if (cot.success !== false) {
            setCotData(cot);
          }
        }

        if (regimeRes.ok) {
          const regime = await regimeRes.json();
          if (regime.success !== false) {
            setRegimeData(regime);
          }
        }

        setLoading(false);
      } catch {
        setError("Failed to load smart money data");
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            <div className="h-4 bg-gray-200 rounded w-1/3" />
            <div className="h-20 bg-gray-200 rounded" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6 text-red-500">{error}</CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Market Regime Overview */}
      {regimeData && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Market Regime
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {regimeData.regime.name}
                </div>
                <div className="text-xs text-muted-foreground">
                  Current Regime
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {regimeData.regime.vix_level.toFixed(1)}
                </div>
                <div className="text-xs text-muted-foreground">
                  VIX ({regimeData.regime.vix_regime})
                </div>
              </div>
              <div className="text-center">
                <div
                  className={`text-2xl font-bold ${
                    regimeData.context.risk_score > 0
                      ? "text-green-500"
                      : regimeData.context.risk_score < 0
                      ? "text-red-500"
                      : ""
                  }`}
                >
                  {regimeData.context.risk_label}
                </div>
                <div className="text-xs text-muted-foreground">
                  Risk Environment
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {regimeData.context.leading_indicator}
                </div>
                <div className="text-xs text-muted-foreground">
                  Leading Asset
                </div>
              </div>
            </div>

            {regimeData.context.divergences.length > 0 && (
              <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded border border-yellow-200 dark:border-yellow-800">
                <div className="flex items-center gap-2 text-yellow-700 dark:text-yellow-400">
                  <AlertTriangle className="h-4 w-4" />
                  <span className="font-medium">Divergences Detected</span>
                </div>
                <ul className="mt-2 text-sm text-yellow-600 dark:text-yellow-300">
                  {regimeData.context.divergences.map((d, i) => (
                    <li key={i}>• {d}</li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* COT Positioning Signals */}
      {cotData && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Smart Money Positioning (COT)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-4 flex items-center gap-4">
              <Badge
                variant={
                  cotData.risk_on_off === "RISK-ON"
                    ? "default"
                    : cotData.risk_on_off === "RISK-OFF"
                    ? "destructive"
                    : "secondary"
                }
              >
                {cotData.risk_on_off}
              </Badge>
              <span className="text-sm text-muted-foreground">
                {cotData.overall_sentiment}
              </span>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
              {Object.entries(cotData.signals).map(([asset, signal]) => (
                <Card key={asset} className="bg-muted/50">
                  <CardContent className="p-4">
                    <div className="flex justify-between items-start mb-3">
                      <span className="font-medium text-lg">{asset}</span>
                      <SignalBadge signal={signal.signal} />
                    </div>

                    <ZScoreGauge
                      value={signal.z_score}
                      label={`${signal.percentile.toFixed(0)}th pctl`}
                    />

                    <div className="mt-3 text-sm text-muted-foreground">
                      {signal.message}
                    </div>

                    <div className="mt-2 flex items-center justify-between text-xs">
                      <span>Confidence</span>
                      <span className="font-medium">
                        {signal.confidence.toFixed(0)}%
                      </span>
                    </div>
                    <Progress value={signal.confidence} className="h-1 mt-1" />

                    {signal.reasoning.length > 0 && (
                      <div className="mt-3 space-y-1">
                        {signal.reasoning.slice(0, 2).map((r, i) => (
                          <div
                            key={i}
                            className="text-xs text-muted-foreground flex items-start gap-1"
                          >
                            <Info className="h-3 w-3 mt-0.5 flex-shrink-0" />
                            {r}
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Recommendations */}
            {cotData.recommendations.length > 0 && (
              <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800">
                <div className="flex items-center gap-2 text-blue-700 dark:text-blue-400">
                  <DollarSign className="h-4 w-4" />
                  <span className="font-medium">Recommendations</span>
                </div>
                <ul className="mt-2 text-sm text-blue-600 dark:text-blue-300">
                  {cotData.recommendations.map((r, i) => (
                    <li key={i}>• {r}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Key Divergences */}
            {cotData.key_divergences.length > 0 && (
              <div className="mt-4 p-3 bg-orange-50 dark:bg-orange-900/20 rounded border border-orange-200 dark:border-orange-800">
                <div className="flex items-center gap-2 text-orange-700 dark:text-orange-400">
                  <AlertTriangle className="h-4 w-4" />
                  <span className="font-medium">Extreme Readings</span>
                </div>
                <ul className="mt-2 text-sm text-orange-600 dark:text-orange-300">
                  {cotData.key_divergences.map((d, i) => (
                    <li key={i}>• {d}</li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default SmartMoneyDashboard;
