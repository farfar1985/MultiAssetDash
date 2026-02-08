"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Target,
  Zap,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";

interface Signal {
  name: string;
  source: string;
  value: number;
  direction: string;
  direction_value: number;
  confidence: number;
  timeframe: string;
  historical_win_rate: number;
  description: string;
}

interface ConfluenceData {
  asset: string;
  timestamp: string;
  conviction_score: number;
  conviction_label: string;
  confidence: number;
  signals: Signal[];
  signal_counts: {
    bullish: number;
    bearish: number;
    neutral: number;
    total: number;
  };
  timeframe_analysis: {
    daily_bias: number;
    weekly_bias: number;
    monthly_bias: number;
    alignment: number;
  };
  historical_context: {
    similar_setups: number;
    avg_return: number;
    win_rate: number;
  };
  risk: {
    score: number;
    suggested_position_pct: number;
  };
  narrative: {
    headline: string;
    key_drivers: string[];
    risks: string[];
  };
}

interface Props {
  data?: ConfluenceData;
  assetId?: string;
}

export function ConfluenceDashboard({ data, assetId }: Props) {
  const [confluenceData, setConfluenceData] = useState<ConfluenceData | null>(data || null);

  useEffect(() => {
    if (!data && assetId) {
      fetch(`/api/v2/confluence/${assetId}`)
        .then((res) => res.json())
        .then((d) => setConfluenceData(d))
        .catch(console.error);
    }
  }, [assetId, data]);

  if (!confluenceData) {
    return (
      <Card className="col-span-2">
        <CardContent className="flex items-center justify-center h-64">
          <div className="animate-pulse text-muted-foreground">Loading intelligence...</div>
        </CardContent>
      </Card>
    );
  }

  const score = confluenceData.conviction_score;
  const isPositive = score > 0;
  const scoreColor = score > 30 ? "text-green-500" : score < -30 ? "text-red-500" : "text-yellow-500";
  const scoreBg = score > 30 ? "bg-green-500/10" : score < -30 ? "bg-red-500/10" : "bg-yellow-500/10";

  const labelColors: Record<string, string> = {
    "STRONG BUY": "bg-green-500 text-white",
    "BUY": "bg-green-400 text-white",
    "NEUTRAL": "bg-yellow-500 text-white",
    "SELL": "bg-red-400 text-white",
    "STRONG SELL": "bg-red-500 text-white",
  };

  return (
    <div className="space-y-4">
      {/* Main Conviction Card */}
      <Card className={`${scoreBg} border-2 ${score > 30 ? "border-green-500/30" : score < -30 ? "border-red-500/30" : "border-yellow-500/30"}`}>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl flex items-center gap-2">
                <Zap className="h-6 w-6" />
                Signal Confluence
              </CardTitle>
              <CardDescription>{confluenceData.asset} Intelligence</CardDescription>
            </div>
            <Badge className={labelColors[confluenceData.conviction_label] || "bg-gray-500"}>
              {confluenceData.conviction_label}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          {/* Conviction Gauge */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Conviction Score</span>
              <span className={`text-3xl font-bold ${scoreColor}`}>
                {score > 0 ? "+" : ""}{score.toFixed(0)}
              </span>
            </div>
            
            {/* Visual gauge */}
            <div className="relative h-8 bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-lg overflow-hidden">
              <div 
                className="absolute top-0 bottom-0 w-1 bg-white shadow-lg z-10"
                style={{ 
                  left: `${((score + 100) / 200) * 100}%`,
                  transition: "left 0.5s ease-out"
                }}
              />
              {/* Scale markers */}
              <div className="absolute inset-0 flex justify-between items-center px-2 text-xs font-bold text-white/80">
                <span>-100</span>
                <span>-50</span>
                <span>0</span>
                <span>+50</span>
                <span>+100</span>
              </div>
            </div>
            
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Strong Sell</span>
              <span>Sell</span>
              <span>Neutral</span>
              <span>Buy</span>
              <span>Strong Buy</span>
            </div>
          </div>

          {/* Headline */}
          <div className="p-4 bg-background/50 rounded-lg mb-4">
            <p className="text-lg font-medium">{confluenceData.narrative.headline}</p>
          </div>

          {/* Signal Counts */}
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="text-center p-3 bg-green-500/10 rounded-lg">
              <div className="flex items-center justify-center gap-1 text-green-500">
                <TrendingUp className="h-4 w-4" />
                <span className="text-2xl font-bold">{confluenceData.signal_counts.bullish}</span>
              </div>
              <div className="text-xs text-muted-foreground">Bullish</div>
            </div>
            <div className="text-center p-3 bg-yellow-500/10 rounded-lg">
              <div className="flex items-center justify-center gap-1 text-yellow-500">
                <Minus className="h-4 w-4" />
                <span className="text-2xl font-bold">{confluenceData.signal_counts.neutral}</span>
              </div>
              <div className="text-xs text-muted-foreground">Neutral</div>
            </div>
            <div className="text-center p-3 bg-red-500/10 rounded-lg">
              <div className="flex items-center justify-center gap-1 text-red-500">
                <TrendingDown className="h-4 w-4" />
                <span className="text-2xl font-bold">{confluenceData.signal_counts.bearish}</span>
              </div>
              <div className="text-xs text-muted-foreground">Bearish</div>
            </div>
          </div>

          {/* Confidence */}
          <div className="mb-4">
            <div className="flex justify-between text-sm mb-1">
              <span>Signal Agreement</span>
              <span className="font-medium">{confluenceData.confidence.toFixed(0)}%</span>
            </div>
            <Progress value={confluenceData.confidence} className="h-2" />
          </div>
        </CardContent>
      </Card>

      {/* Key Drivers & Risks */}
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              Key Drivers
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {confluenceData.narrative.key_drivers.map((driver, i) => (
                <li key={i} className="text-sm flex items-start gap-2">
                  <ArrowUpRight className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                  <span>{driver}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-yellow-500" />
              Risk Factors
            </CardTitle>
          </CardHeader>
          <CardContent>
            {confluenceData.narrative.risks.length > 0 ? (
              <ul className="space-y-2">
                {confluenceData.narrative.risks.map((risk, i) => (
                  <li key={i} className="text-sm flex items-start gap-2">
                    <ArrowDownRight className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                    <span>{risk}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted-foreground">No significant risks identified</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Timeframe Analysis */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Clock className="h-4 w-4" />
            Multi-Timeframe Alignment
          </CardTitle>
          <CardDescription>
            {confluenceData.timeframe_analysis.alignment > 70 
              ? "Strong alignment across timeframes"
              : confluenceData.timeframe_analysis.alignment > 40
              ? "Moderate alignment"
              : "Conflicting timeframes — exercise caution"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[
              { label: "Daily", value: confluenceData.timeframe_analysis.daily_bias },
              { label: "Weekly", value: confluenceData.timeframe_analysis.weekly_bias },
              { label: "Monthly", value: confluenceData.timeframe_analysis.monthly_bias },
            ].map((tf) => (
              <div key={tf.label} className="flex items-center gap-4">
                <span className="text-sm w-16">{tf.label}</span>
                <div className="flex-1 h-6 bg-muted rounded relative">
                  <div 
                    className={`absolute top-0 bottom-0 ${tf.value > 0 ? "bg-green-500" : "bg-red-500"} rounded`}
                    style={{
                      left: tf.value > 0 ? "50%" : `${50 + tf.value / 2}%`,
                      width: `${Math.abs(tf.value) / 2}%`,
                    }}
                  />
                  <div className="absolute top-0 bottom-0 left-1/2 w-px bg-foreground/30" />
                </div>
                <span className={`text-sm font-mono w-12 text-right ${tf.value > 0 ? "text-green-500" : "text-red-500"}`}>
                  {tf.value > 0 ? "+" : ""}{tf.value.toFixed(0)}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-3 border-t">
            <div className="flex justify-between text-sm">
              <span>Alignment Score</span>
              <span className={`font-bold ${confluenceData.timeframe_analysis.alignment > 70 ? "text-green-500" : "text-yellow-500"}`}>
                {confluenceData.timeframe_analysis.alignment.toFixed(0)}%
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Signals */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            All Signals
          </CardTitle>
          <CardDescription>
            {confluenceData.signal_counts.total} signals analyzed with historical win rates
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {confluenceData.signals.map((signal, i) => (
              <div key={i} className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50">
                <div className={`p-1.5 rounded ${
                  signal.direction_value > 0 ? "bg-green-500/20" : 
                  signal.direction_value < 0 ? "bg-red-500/20" : "bg-yellow-500/20"
                }`}>
                  {signal.direction_value > 0 ? (
                    <TrendingUp className="h-4 w-4 text-green-500" />
                  ) : signal.direction_value < 0 ? (
                    <TrendingDown className="h-4 w-4 text-red-500" />
                  ) : (
                    <Minus className="h-4 w-4 text-yellow-500" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{signal.name}</span>
                    <Badge variant="outline" className="text-xs">
                      {signal.timeframe}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground truncate">{signal.description}</p>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium">
                    {(signal.historical_win_rate * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-muted-foreground">win rate</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Position Suggestion */}
      <Card className="bg-gradient-to-r from-blue-500/10 to-purple-500/10">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Target className="h-4 w-4" />
            Suggested Position Size
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-3xl font-bold">
                {confluenceData.risk.suggested_position_pct.toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">of portfolio</div>
            </div>
            <div className="text-right">
              <div className="text-sm">Risk Score</div>
              <div className={`text-lg font-bold ${
                confluenceData.risk.score > 60 ? "text-red-500" : 
                confluenceData.risk.score > 30 ? "text-yellow-500" : "text-green-500"
              }`}>
                {confluenceData.risk.score.toFixed(0)}/100
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default ConfluenceDashboard;
